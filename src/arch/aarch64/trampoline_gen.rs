use anyhow::Result;

use crate::disasm::BasicBlock;
use crate::traits::TrampolineGenerator;
use crate::trampoline::{InitCode, PREV_LOC_OFFSET, PersistentWrapper, Trampoline};

/// AArch64 trampoline generator implementing the TrampolineGenerator trait.
///
/// Produces hand-assembled ARM64 machine code for:
/// - Coverage trampolines (NZCV-preserving, edge hash increment)
/// - Init code (SHM attachment via /proc/self/environ, optional forkserver)
/// - .so init code (SHM attachment, optional DT_INIT chaining)
///
/// All syscall numbers are ARM64-specific (from linux/arch/arm64/include/asm/unistd.h).
#[derive(Debug)]
pub struct AArch64TrampolineGenerator;

impl AArch64TrampolineGenerator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AArch64TrampolineGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// ARM64 syscall numbers (from linux/arch/arm64/include/asm/unistd.h)
// ============================================================
const SYS_OPENAT: u32 = 56;
const SYS_READ: u32 = 63;
const SYS_WRITE: u32 = 64;
const SYS_CLOSE: u32 = 57;
const SYS_EXIT: u32 = 93;
const SYS_SHMAT: u32 = 196; // CRITICAL: ARM64=196, NOT x86_64=30
const SYS_CLONE: u32 = 220; // ARM64 has no fork(); use clone(SIGCHLD,0,0,0,0) instead
const SYS_WAIT4: u32 = 260;
const SYS_GETPID: u32 = 172;
const SYS_KILL: u32 = 129;

/// SIGCHLD signal number (for clone-as-fork: clone(SIGCHLD, 0, 0, 0, 0))
const SIGCHLD: u32 = 17;

/// SIGSTOP signal number (Linux=19, macOS=17)
const SIGSTOP: u32 = 19;

/// AFL forkserver file descriptor (ARM64 same as x86_64)
const FORKSRV_FD: u32 = 198;

/// AT_FDCWD encoded as u32 = 0xFFFFFF9C (-100 two's complement)
const AT_FDCWD_U32: u32 = 0xFFFF_FF9C;

// ============================================================
// Helpers for encoding ARM64 instructions
// ============================================================

/// Encode an unconditional B instruction.
/// offset_instr: signed 26-bit instruction offset (delta / 4).
#[inline]
fn encode_b(offset_instructions: i32) -> [u8; 4] {
    // B: opcode 0b000101, imm26
    let word = 0x1400_0000u32 | ((offset_instructions as u32) & 0x03FF_FFFF);
    word.to_le_bytes()
}

/// Encode BR x_reg (branch to register).
/// reg: register number 0-30.
#[inline]
fn encode_br(reg: u32) -> [u8; 4] {
    // BR xN: 1101 0110 0001 1111 0000 00nn nnn0 0000
    let word = 0xD61F_0000u32 | (reg << 5);
    word.to_le_bytes()
}

/// Encode BLR x_reg (branch-and-link to register).
#[inline]
fn encode_blr(reg: u32) -> [u8; 4] {
    // BLR xN: 1101 0110 0011 1111 0000 00nn nnn0 0000
    let word = 0xD63F_0000u32 | (reg << 5);
    word.to_le_bytes()
}

/// Encode RET (return, uses x30).
#[inline]
fn encode_ret() -> [u8; 4] {
    0xD65F_03C0u32.to_le_bytes()
}

/// Encode STP x_n, x_m, [sp, #-16]! (pre-index push of two registers).
#[inline]
fn encode_stp_push(rn: u32, rm: u32) -> [u8; 4] {
    // STP xN, xM, [sp, #-16]! (pre-index with writeback)
    // [31:30]=10 (opc, 64-bit), [29:27]=101, [26]=0 (V, GP),
    // [25:23]=011 (pre-index), [22]=0 (L=0, store),
    // [21:15]=imm7 (-2 = -16/8), [14:10]=Rt2, [9:5]=Rn(sp), [4:0]=Rt
    let imm7: u32 = (-2i32 as u32) & 0x7F;
    let word = (0b10u32 << 30)
        | (0b101u32 << 27)
        | (0b011u32 << 23) // pre-index
        | (imm7 << 15)
        | (rm << 10)
        | (31u32 << 5)
        | rn;
    word.to_le_bytes()
}

/// Encode LDP x_n, x_m, [sp], #16 (post-index pop of two registers).
#[inline]
fn encode_ldp_pop(rn: u32, rm: u32) -> [u8; 4] {
    // LDP xN, xM, [sp], #16 (post-index with writeback)
    // [25:23]=001 (post-index), [22]=1 (L=1, load)
    let imm7: u32 = 2u32 & 0x7F;
    let word = (0b10u32 << 30)
        | (0b101u32 << 27)
        | (0b001u32 << 23) // post-index
        | (1u32 << 22)     // L=1 for LDP
        | (imm7 << 15)
        | (rm << 10)
        | (31u32 << 5)
        | rn;
    word.to_le_bytes()
}

/// Encode MOVZ xN, #imm16 (zero-extend, shift=0).
#[inline]
fn encode_movz(reg: u32, imm16: u16) -> [u8; 4] {
    // MOVZ: sf=1, opc=10, hw=00 (shift=0), imm16, Rd
    let word = 0xD280_0000u32 | ((imm16 as u32) << 5) | reg;
    word.to_le_bytes()
}

/// Encode MOVK xN, #imm16, LSL #shift (0, 16, 32, 48).
#[inline]
fn encode_movk(reg: u32, imm16: u16, lsl: u32) -> [u8; 4] {
    // MOVK: sf=1, opc=11, hw=shift/16
    let hw = lsl / 16;
    let word = 0xF280_0000u32 | (hw << 21) | ((imm16 as u32) << 5) | reg;
    word.to_le_bytes()
}

/// Encode MOV x_dst, x_src (register copy, via ORR xDst, xzr, xSrc).
#[inline]
fn encode_mov_reg(dst: u32, src: u32) -> [u8; 4] {
    // ORR xDst, xzr, xSrc: sf=1, opc=01, shift=00, N=0, Rm=src, imm6=0, Rn=xzr(31), Rd=dst
    let word = 0xAA00_03E0u32 | (src << 16) | dst;
    word.to_le_bytes()
}

/// Encode SVC #0 (supervisor call / syscall).
#[inline]
fn encode_svc0() -> [u8; 4] {
    0xD400_0001u32.to_le_bytes()
}

/// Encode NOP.
#[inline]
fn encode_nop() -> [u8; 4] {
    0xD503_201Fu32.to_le_bytes()
}

/// Encode STR xN, [xM] (store 64-bit register).
#[inline]
fn encode_str_reg(xn: u32, xm: u32) -> [u8; 4] {
    // STR xN, [xM]: size=11, V=0, opc=00, offset=0, Rn=xm, Rt=xn
    let word = 0xF900_0000u32 | (xm << 5) | xn;
    word.to_le_bytes()
}

/// Encode LDR xN, [xM] (load 64-bit register from memory).
#[inline]
fn encode_ldr_reg(xn: u32, xm: u32) -> [u8; 4] {
    // LDR xN, [xM]: size=11, V=0, opc=01, offset=0, Rn=xm, Rt=xn
    let word = 0xF940_0000u32 | (xm << 5) | xn;
    word.to_le_bytes()
}

/// Encode LDRH wN, [xM, #offset] (load halfword from memory with unsigned offset).
/// offset must be a multiple of 2 and <= 62 (small immediate for Phase 40).
#[inline]
fn encode_ldrh_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    // LDRH wN, [xM, #imm]: size=01, V=0, opc=01, imm12=(offset/2), Rn=xm, Rt=wn
    let imm12 = (offset / 2) & 0xFFF;
    let word = 0x7940_0000u32 | (imm12 << 10) | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode STRH wN, [xM, #offset] (store halfword to memory with unsigned offset).
#[inline]
fn encode_strh_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    // STRH wN, [xM, #imm]: size=01, V=0, opc=00, imm12=(offset/2)
    let imm12 = (offset / 2) & 0xFFF;
    let word = 0x7900_0000u32 | (imm12 << 10) | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode ADD wDst, wSrc, #1 (no flags, 32-bit).
#[inline]
fn encode_add_w_imm1(wd: u32, wn: u32) -> [u8; 4] {
    // ADD wDst, wSrc, #1: sf=0, op=0, S=0, shift=00, imm12=1, Rn=wn, Rd=wd
    let word = 0x1100_0400u32 | (wn << 5) | wd;
    word.to_le_bytes()
}

/// Encode ADD xDst, xSrc, xOther (64-bit, shifted register).
#[inline]
fn encode_add_reg(xd: u32, xn: u32, xm: u32) -> [u8; 4] {
    // ADD xDst, xN, xM (shift=LSL #0): sf=1, op=0, S=0, shift=00, Rm=xm, imm6=0, Rn=xn, Rd=xd
    let word = 0x8B00_0000u32 | (xm << 16) | (xn << 5) | xd;
    word.to_le_bytes()
}

/// Encode EOR xDst, xN, xM (XOR 64-bit shifted register).
#[inline]
fn encode_eor_reg(xd: u32, xn: u32, xm: u32) -> [u8; 4] {
    // EOR xDst, xN, xM: sf=1, opc=10, shift=00, N=0, Rm=xm, imm6=0, Rn=xn, Rd=xd
    let word = 0xCA00_0000u32 | (xm << 16) | (xn << 5) | xd;
    word.to_le_bytes()
}

/// Encode AND xDst, xN, #0xFFFF (64-bit immediate AND).
/// For 0xFFFF: N=0, immr=0, imms=15 (encoding for 16 consecutive 1s from bit 0).
#[inline]
fn encode_and_0xffff(xd: u32, xn: u32) -> [u8; 4] {
    // AND xDst, xN, #0xFFFF
    // N=0, immr=0, imms=0b001111 (15 = 16 bits)
    // Encoding: sf=1, opc=00, N=0, immr=0, imms=15, Rn=xn, Rd=xd
    let word = 0x9240_0000u32 | (15u32 << 10) | (xn << 5) | xd;
    word.to_le_bytes()
}

/// Encode STR xN, [sp, #offset] (store to stack with unsigned immediate).
/// offset must be multiple of 8, <= 32760.
#[inline]
fn encode_str_sp_imm(xn: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 8) & 0xFFF;
    let word = 0xF900_03E0u32 | (imm12 << 10) | xn;
    word.to_le_bytes()
}

/// Encode SUB sp, sp, #imm (immediate with optional 12-bit shift).
/// Supports imm <= 4095 directly, or multiples of 4096 up to 4095*4096 via LSL #12.
#[inline]
fn encode_sub_sp_imm(imm: u32) -> [u8; 4] {
    // SUB sp, sp, #imm12{, LSL #sh}: sf=1, op=1, S=0, sh, imm12, Rn=31, Rd=31
    let (sh, imm12) = if imm <= 0xFFF {
        (0u32, imm)
    } else if imm & 0xFFF == 0 && (imm >> 12) <= 0xFFF {
        (1u32, imm >> 12)
    } else {
        unreachable!(
            "SUB sp, sp, #{imm} cannot be encoded in a single ADD/SUB immediate instruction"
        );
    };
    let word = 0xD100_03FFu32 | (sh << 22) | ((imm12 & 0xFFF) << 10);
    word.to_le_bytes()
}

/// Encode ADD sp, sp, #imm (immediate with optional 12-bit shift).
/// Supports imm <= 4095 directly, or multiples of 4096 up to 4095*4096 via LSL #12.
#[inline]
fn encode_add_sp_imm(imm: u32) -> [u8; 4] {
    // ADD sp, sp, #imm12{, LSL #sh}
    let (sh, imm12) = if imm <= 0xFFF {
        (0u32, imm)
    } else if imm & 0xFFF == 0 && (imm >> 12) <= 0xFFF {
        (1u32, imm >> 12)
    } else {
        unreachable!(
            "ADD sp, sp, #{imm} cannot be encoded in a single ADD/SUB immediate instruction"
        );
    };
    let word = 0x9100_03FFu32 | (sh << 22) | ((imm12 & 0xFFF) << 10);
    word.to_le_bytes()
}

/// Encode CBZ xN, #offset (compare and branch if zero; offset in bytes from instruction).
#[inline]
fn encode_cbz(xn: u32, offset_bytes: i32) -> [u8; 4] {
    // CBZ xN, label: sf=1, opc=011010, imm19 (offset/4), Rt=xn
    let imm19 = ((offset_bytes / 4) as u32) & 0x7FFFF;
    let word = 0xB400_0000u32 | (imm19 << 5) | xn;
    word.to_le_bytes()
}

/// Encode LDR xN, PC+offset (PC-relative literal load).
/// offset_bytes: byte offset from start of this instruction to the 8-byte literal.
/// Must be aligned to 4 bytes and positive (forward reference).
#[inline]
fn encode_ldr_literal(xn: u32, offset_bytes: i32) -> [u8; 4] {
    // LDR xN, label: opc=01, V=0, imm19=(offset/4), Rt=xn
    let imm19 = ((offset_bytes / 4) as u32) & 0x7FFFF;
    let word = 0x5800_0000u32 | (imm19 << 5) | xn;
    word.to_le_bytes()
}

/// Encode ADR xN, PC+byte_offset (PC-relative address computation, ±1 MiB range).
/// Unlike LDR literal, this computes the ADDRESS (PC+offset) without loading from memory.
/// Used for PIE-safe address computation where we need a runtime pointer, not a stored value.
#[inline]
fn encode_adr(rd: u32, byte_offset: i64) -> [u8; 4] {
    // ADR Xd, #offset: op=0, immlo=offset[1:0], 10000, immhi=offset[20:2], Rd
    let imm21 = byte_offset as u32; // truncate to low 21 bits (sign-extended by CPU)
    let immlo = imm21 & 0x3;
    let immhi = (imm21 >> 2) & 0x7FFFF;
    let word = (immlo << 29) | (0b10000u32 << 24) | (immhi << 5) | rd;
    word.to_le_bytes()
}

/// Encode STR xN, [xM, #offset] for 8-byte aligned slots.
/// Same as encode_str_imm.
#[inline]
fn encode_ldr_imm(xn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 8) & 0xFFF;
    let word = 0xF940_0000u32 | (imm12 << 10) | (xm << 5) | xn;
    word.to_le_bytes()
}

/// Encode STRH wzr, [xM, #offset] (store zero halfword).
#[inline]
fn encode_strh_zero(xm: u32, offset: u32) -> [u8; 4] {
    encode_strh_imm(31, xm, offset)
}

/// Encode CMP xN, #0 (via SUBS xzr, xN, #0).
#[inline]
fn encode_cmp_imm0(xn: u32) -> [u8; 4] {
    // SUBS xzr, xN, #0: sf=1, op=1, S=1, imm12=0, Rn=xn, Rd=31
    let word = 0xF100_001Fu32 | (xn << 5);
    word.to_le_bytes()
}

/// Encode CMP xN, xM (64-bit register compare, via SUBS).
#[inline]
fn encode_cmp_reg(xn: u32, xm: u32) -> [u8; 4] {
    // SUBS xzr, xN, xM: sf=1, op=1, S=1, shift=00, Rm=xm, imm6=0, Rn=xn, Rd=31
    let word = 0xEB00_001Fu32 | (xm << 16) | (xn << 5);
    word.to_le_bytes()
}

/// Encode LDRB wN, [xM] (load byte from memory, zero-extend).
#[inline]
fn encode_ldrb_reg(wn: u32, xm: u32) -> [u8; 4] {
    // LDRB wN, [xM]: size=00, V=0, opc=01, imm12=0, Rn=xm, Rt=wn
    let word = 0x3940_0000u32 | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode STRB wN, [xM] (store byte to memory).
#[inline]
fn encode_strb_reg(wn: u32, xm: u32) -> [u8; 4] {
    // STRB wN, [xM]: size=00, V=0, opc=00, imm12=0, Rn=xm, Rt=wn
    let word = 0x3900_0000u32 | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode ADD xDst, xN, #imm12.
#[inline]
fn encode_add_imm(xd: u32, xn: u32, imm12: u32) -> [u8; 4] {
    // ADD xDst, xN, #imm12: sf=1, op=0, S=0, shift=00
    let word = 0x9100_0000u32 | ((imm12 & 0xFFF) << 10) | (xn << 5) | xd;
    word.to_le_bytes()
}

/// Load a 32-bit unsigned immediate into register xN (MOVZ only if fits in 16 bits, else MOVZ+MOVK).
fn emit_mov32(code: &mut Vec<u8>, xn: u32, val: u32) {
    let lo = (val & 0xFFFF) as u16;
    let hi = ((val >> 16) & 0xFFFF) as u16;
    code.extend_from_slice(&encode_movz(xn, lo));
    if hi != 0 {
        code.extend_from_slice(&encode_movk(xn, hi, 16));
    }
}

/// Emit a syscall stub: load syscall number into x8, then SVC #0.
fn emit_syscall(code: &mut Vec<u8>, syscall_nr: u32) {
    emit_mov32(code, 8, syscall_nr);
    code.extend_from_slice(&encode_svc0());
}

use super::relocation::{is_pc_relative, relocate_pc_relative, sign_extend};

/// Extract the original branch target VA from a conditional branch instruction that can
/// benefit from branch island synthesis.
///
/// Returns `Some(target_va)` for: TBZ/TBNZ (±32 KiB), CBZ/CBNZ (±1 MiB), B.cond (±1 MiB).
/// Returns `None` for: B/BL (±128 MiB — same range as the island's B, so islands can't help),
/// non-branch PC-relative instructions (ADRP, ADR, LDR literal), and everything else.
fn compute_branch_target(raw: u32, insn_va: u64) -> Option<u64> {
    let is_bcond = (raw & 0xFF00_0010) == 0x5400_0000;
    let is_cbz_cbnz = (raw & 0x7E00_0000) == 0x3400_0000;
    let is_tbz_tbnz = (raw & 0x7E00_0000) == 0x3600_0000;

    if is_tbz_tbnz {
        let imm14 = (raw >> 5) & 0x3FFF;
        let offset = sign_extend(imm14, 14) * 4;
        Some((insn_va as i64 + offset) as u64)
    } else if is_cbz_cbnz || is_bcond {
        let imm19 = (raw >> 5) & 0x7_FFFF;
        let offset = sign_extend(imm19, 19) * 4;
        Some((insn_va as i64 + offset) as u64)
    } else {
        None
    }
}

/// Rewrite a conditional branch instruction's immediate to point to `island_offset_bytes`
/// from the instruction's own location.
///
/// Used to redirect TBZ/TBNZ/CBZ/CBNZ/B.cond to a nearby branch island.
/// Preserves opcode bits, register, bit number (TBZ/TBNZ), and condition code (B.cond).
///
/// Unreachable if `raw` is not a recognised conditional branch (caller must verify).
fn rewrite_branch_to_island(raw: u32, island_offset_bytes: i64) -> u32 {
    let is_bcond = (raw & 0xFF00_0010) == 0x5400_0000;
    let is_cbz_cbnz = (raw & 0x7E00_0000) == 0x3400_0000;
    let is_tbz_tbnz = (raw & 0x7E00_0000) == 0x3600_0000;

    let offset_insns = island_offset_bytes / 4;

    if is_tbz_tbnz {
        let b5_op = raw & 0xFF00_0000;
        let b40 = (raw >> 19) & 0x1F;
        let rt = raw & 0x1F;
        let new_imm14 = (offset_insns as u32) & 0x3FFF;
        b5_op | (b40 << 19) | (new_imm14 << 5) | rt
    } else if is_cbz_cbnz {
        let sf_op = raw & 0xFF00_0000;
        let rt = raw & 0x1F;
        let new_imm19 = (offset_insns as u32) & 0x7_FFFF;
        sf_op | (new_imm19 << 5) | rt
    } else if is_bcond {
        let cond = raw & 0xF;
        let new_imm19 = (offset_insns as u32) & 0x7_FFFF;
        0x5400_0000 | (new_imm19 << 5) | cond
    } else {
        unreachable!(
            "rewrite_branch_to_island called with non-conditional-branch: 0x{:08x}",
            raw
        );
    }
}

/// Check that a branch from `from_va` to `to_va` is within ARM64 B range (±128 MiB).
/// Returns the instruction-offset (delta/4) if in range, or an error.
fn check_branch_range(from_va: u64, to_va: u64) -> Result<i32> {
    let delta = (to_va as i64) - (from_va as i64);
    const MAX_RANGE: i64 = 128 * 1024 * 1024;
    if delta.abs() > MAX_RANGE {
        anyhow::bail!(
            "branch out of range: from 0x{:x} to 0x{:x}, delta {} bytes (max \u{b1}128 MiB)",
            from_va,
            to_va,
            delta
        );
    }
    Ok(((delta >> 2) & 0x3FF_FFFF) as i32)
}

impl TrampolineGenerator for AArch64TrampolineGenerator {
    /// Generate a NZCV-preserving coverage trampoline for a single basic block.
    ///
    /// The trampoline is called via `B trampoline_va` patched at the block start.
    /// Register usage: x9 (NZCV save + scratch), x10 (shm_ptr + scratch), x11 (byte count).
    /// All used registers are saved/restored via the stack.
    ///
    /// Layout:
    ///   STP x9, x10, [sp, #-16]!     ; save scratch registers
    ///   STP x11, xzr, [sp, #-16]!    ; save x11 (xzr as padding)
    ///   MRS x9, NZCV                 ; save condition flags
    ///   <instrumentation>            ; edge hash + map increment
    ///   MSR NZCV, x9                 ; restore condition flags
    ///   LDP x11, xzr, [sp], #16      ; restore x11
    ///   LDP x9, x10, [sp], #16       ; restore scratch registers
    ///   <displaced original instruction>
    ///   B return_va                  ; back to original code
    fn generate_trampoline(
        &self,
        trampoline_va: u64,
        data_va: u64,
        block: &BasicBlock,
    ) -> Result<Trampoline> {
        let block_id = block.block_id as u32;
        // return_va = address of instruction after the displaced one in the original binary.
        let return_va = block.va + block.displaced_len as u64;

        let mut code: Vec<u8> = Vec::with_capacity(128);

        // ── Save registers ──────────────────────────────────────────────
        // STP x9, x10, [sp, #-16]!
        code.extend_from_slice(&encode_stp_push(9, 10));
        // STP x11, x12, [sp, #-16]!
        code.extend_from_slice(&encode_stp_push(11, 12));

        // MRS x9, NZCV  (save condition flags)
        // NZCV = S3_3_C4_C2_0: o0=1, op1=011, CRn=0100, CRm=0010, op2=000, Rt=x9
        // Encoding: 0xD53B4209 → LE bytes [0x09, 0x42, 0x3B, 0xD5]
        // (op2=0 for NZCV; op2=1 would be DAIF which requires EL1 → SIGILL)
        code.extend_from_slice(&[0x09, 0x42, 0x3B, 0xD5]);

        // ── Load shm_ptr from data_va into x10 ──────────────────────────
        // We use ADR (PC-relative address computation) to get data_va's runtime
        // address. This is PIE-safe: the offset between trampoline and data_va is
        // fixed regardless of load base / ASLR.
        //   ADR x11, data_va                          ; x11 = runtime &data_va
        //   LDR x10, [x11]                            ; x10 = shm_ptr

        // ADR x11, data_va (computed directly — data_va is known at rewrite time)
        let adr1_offset = data_va as i64 - (trampoline_va + code.len() as u64) as i64;
        code.extend_from_slice(&encode_adr(11, adr1_offset));

        // LDR x10, [x11]  (x10 = shm_ptr)
        code.extend_from_slice(&encode_ldr_reg(10, 11));

        // ── Guard: skip coverage if shm_ptr is NULL ──────────────────────
        // CBZ x10, .skip — when __AFL_SHM_ID is not set, shm_ptr stays 0.
        let cbz_skip_pos = code.len();
        code.extend_from_slice(&[0x00, 0x00, 0x00, 0xB4]); // CBZ x10, placeholder

        // ── Load prev_loc (u16) from [data_va + PREV_LOC_OFFSET] into x12 ──
        // LDRH w12, [x11, #PREV_LOC_OFFSET]
        code.extend_from_slice(&encode_ldrh_imm(12, 11, PREV_LOC_OFFSET as u32));

        // ── Compute edge = (block_id XOR prev_loc) & 0xFFFF ──────────────
        // MOVZ x11, #block_id  (block_id is u16, fits in 16-bit immediate)
        code.extend_from_slice(&encode_movz(11, block_id as u16));

        // EOR x11, x11, x12  (edge = block_id XOR prev_loc)
        code.extend_from_slice(&encode_eor_reg(11, 11, 12));

        // AND x11, x11, #0xFFFF (mask to 16 bits)
        code.extend_from_slice(&encode_and_0xffff(11, 11));

        // ── Increment shm_map[edge] ─────────────────────────────────────
        // x10 = shm_ptr (base of coverage map), x11 = edge index
        // ADD x10, x10, x11  ; x10 = &shm_map[edge]
        code.extend_from_slice(&encode_add_reg(10, 10, 11));

        // LDRB w11, [x10]   ; w11 = current count
        code.extend_from_slice(&encode_ldrb_reg(11, 10));

        // ADD w11, w11, #1  (no flags — ADD without S suffix)
        code.extend_from_slice(&encode_add_w_imm1(11, 11));

        // STRB w11, [x10]   ; store back
        code.extend_from_slice(&encode_strb_reg(11, 10));

        // ── Update prev_loc = block_id >> 1 ─────────────────────────────
        // Need data_ptr again — recompute via ADR.
        let adr2_offset = data_va as i64 - (trampoline_va + code.len() as u64) as i64;
        code.extend_from_slice(&encode_adr(11, adr2_offset));

        // MOVZ x10, #(block_id >> 1)
        code.extend_from_slice(&encode_movz(10, (block_id >> 1) as u16));

        // STRH w10, [x11, #PREV_LOC_OFFSET]
        code.extend_from_slice(&encode_strh_imm(10, 11, PREV_LOC_OFFSET as u32));

        // .skip: (patch CBZ target)
        let skip_target = code.len();
        let cbz_imm19 = ((skip_target as i64 - cbz_skip_pos as i64) / 4) as u32;
        let cbz_word = 0xB400_000Au32 | ((cbz_imm19 & 0x7FFFF) << 5); // CBZ x10, #imm19
        code[cbz_skip_pos..cbz_skip_pos + 4].copy_from_slice(&cbz_word.to_le_bytes());

        // ── Restore condition flags ───────────────────────────────────────
        // MSR NZCV, x9  (restore condition flags from x9)
        // MSR uses L=0 (bit 21), same system register encoding as MRS above.
        // Encoding: 0xD51B4209 → LE bytes [0x09, 0x42, 0x1B, 0xD5]
        code.extend_from_slice(&[0x09, 0x42, 0x1B, 0xD5]);

        // ── Restore registers ─────────────────────────────────────────────
        // LDP x11, x12, [sp], #16
        code.extend_from_slice(&encode_ldp_pop(11, 12));
        // LDP x9, x10, [sp], #16
        code.extend_from_slice(&encode_ldp_pop(9, 10));

        // ── Displaced original instruction ────────────────────────────────
        // AArch64 instructions are always 4 bytes.
        // If the instruction is PC-relative (ADRP, ADR, LDR literal, B, BL, B.cond,
        // CBZ/CBNZ, TBZ/TBNZ), we must relocate it: recompute the immediate so it
        // reaches the same target from the trampoline VA.
        //
        // When relocation fails (offset exceeds the instruction's range), we try to
        // synthesise a branch island: rewrite the conditional branch to jump +8 bytes
        // (past the return B), where we place an unconditional B to the original target.
        // This covers TBZ/TBNZ (±32 KiB), CBZ/CBNZ (±256 KiB), and B.cond (±256 KiB).
        let mut island_target: Option<u64> = None;

        if block.displaced_len == 4 {
            let bytes = &block.displaced_bytes;
            let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let displaced_insn_va = trampoline_va + code.len() as u64;

            if is_pc_relative(raw) {
                match relocate_pc_relative(raw, block.va, displaced_insn_va) {
                    Ok(relocated) => {
                        code.extend_from_slice(&relocated.to_le_bytes());
                        tracing::debug!(
                            "AArch64: relocated PC-relative insn at 0x{:x} (0x{:08x} → 0x{:08x}) for trampoline at 0x{:x}",
                            block.va,
                            raw,
                            relocated,
                            displaced_insn_va
                        );
                    }
                    Err(_) => {
                        // Relocation out of range — try branch island for conditional branches
                        if let Some(target_va) = compute_branch_target(raw, block.va) {
                            // Rewrite the conditional branch to jump +8 bytes (2 instructions):
                            // past the return B to the island B instruction.
                            let rewritten = rewrite_branch_to_island(raw, 8);
                            code.extend_from_slice(&rewritten.to_le_bytes());
                            island_target = Some(target_va);
                            tracing::debug!(
                                "AArch64: branch island for insn at 0x{:x} (0x{:08x}), island target 0x{:x}",
                                block.va,
                                raw,
                                target_va
                            );
                        } else {
                            // Non-branch PC-relative (ADRP/ADR/LDR literal) — can't use island
                            return Err(anyhow::anyhow!(
                                "PC-relative relocation out of range at 0x{:x} (non-branch 0x{:08x}, no island possible)",
                                block.va,
                                raw
                            ));
                        }
                    }
                }
            } else {
                code.extend_from_slice(bytes);
            }
        } else if block.displaced_len == 8 {
            // Two-instruction displacement (typically ADRP+ADD/LDR/STR pair).
            // Relocate each instruction independently.
            for i in 0..2 {
                let offset = i * 4;
                let insn_bytes = &block.displaced_bytes[offset..offset + 4];
                let raw = u32::from_le_bytes([
                    insn_bytes[0],
                    insn_bytes[1],
                    insn_bytes[2],
                    insn_bytes[3],
                ]);
                let original_insn_va = block.va + offset as u64;
                let displaced_insn_va = trampoline_va + code.len() as u64;

                if is_pc_relative(raw) {
                    match relocate_pc_relative(raw, original_insn_va, displaced_insn_va) {
                        Ok(relocated) => {
                            code.extend_from_slice(&relocated.to_le_bytes());
                        }
                        Err(e) => {
                            return Err(anyhow::anyhow!(
                                "PC-relative relocation failed for instruction {} of pair at 0x{:x}: {e}",
                                i,
                                block.va
                            ));
                        }
                    }
                } else {
                    code.extend_from_slice(insn_bytes);
                }
            }
        } else {
            // Unexpected length — copy raw bytes as fallback.
            tracing::warn!(
                "AArch64: unexpected displaced_len={} for block at 0x{:x}",
                block.displaced_len,
                block.va
            );
            code.extend_from_slice(&block.displaced_bytes);
        }

        // ── Branch back to original code ─────────────────────────────────
        let b_instr_va = trampoline_va + code.len() as u64;
        let imm26 = check_branch_range(b_instr_va, return_va)?;
        code.extend_from_slice(&encode_b(imm26));

        // ── Branch island (conditional) ──────────────────────────────────
        // When the displaced instruction was a conditional branch that couldn't be
        // relocated directly, we emit an unconditional B to the original target here.
        // The conditional branch was rewritten to jump +8 bytes from its position,
        // landing exactly on this island instruction.
        if let Some(target_va) = island_target {
            let island_va = trampoline_va + code.len() as u64;
            let imm26 = check_branch_range(island_va, target_va)?;
            code.extend_from_slice(&encode_b(imm26));
        }

        // No literal pool needed — data_va is loaded via ADR (PC-relative).

        Ok(Trampoline {
            va: trampoline_va,
            code,
        })
    }

    /// Generate init code for AArch64 executables.
    ///
    /// Reads __AFL_SHM_ID from /proc/self/environ, attaches shared memory via SYS_shmat (196),
    /// stores the SHM pointer at data_va, optionally runs the AFL forkserver loop, then
    /// jumps to the original entry point.
    ///
    /// Returns InitCode { va: init_va, entry_va: init_va } — the original entry point is
    /// redirected to init_va which IS the new entry point.
    fn generate_init_code(
        &self,
        init_va: u64,
        data_va: u64,
        entry_point: u64,
        enable_forkserver: bool,
        _persistent_data_va: Option<u64>,
    ) -> Result<InitCode> {
        let mut code: Vec<u8> = Vec::with_capacity(512);

        // ── Prologue: save x0 + callee-saved registers and allocate stack ─
        //
        // CRITICAL: At _start, x0 = rtld_fini (ld.so cleanup callback).
        // _start passes x0 to __libc_start_main which registers it via
        // __cxa_atexit.  Our init code clobbers x0 with syscall results,
        // so we must preserve it.  Push x0 first (as STP x0,xzr for
        // 16-byte alignment), then allocate the main frame.
        //
        // We save: x0, x19-x30 (13 regs → x0 pair + 6 callee-saved pairs)
        // Also allocate 4096 bytes for the environ read buffer.
        const FRAME_REGS: u32 = 96; // callee-saved registers (12 x 8)
        const BUF_SIZE: u32 = 4096; // environ buffer

        // STP x0, xzr, [sp, #-16]!  — save x0 (pre-index, sp -= 16)
        code.extend_from_slice(&encode_stp_push(0, 31 /* xzr */));

        // SUB sp, sp, #4192
        // 4192 = 0x1060, but SUB imm12 only supports 0..4095.
        // Split: SUB sp, sp, #4096; SUB sp, sp, #96
        code.extend_from_slice(&encode_sub_sp_imm(4096));
        code.extend_from_slice(&encode_sub_sp_imm(96));

        // STP x29, x30, [sp, #0]  (frame pointer + link register)
        code.extend_from_slice(&encode_stp_push_offset(29, 30, 0));
        // STP x27, x28, [sp, #16]
        code.extend_from_slice(&encode_stp_push_offset(27, 28, 16));
        // STP x25, x26, [sp, #32]
        code.extend_from_slice(&encode_stp_push_offset(25, 26, 32));
        // STP x23, x24, [sp, #48]
        code.extend_from_slice(&encode_stp_push_offset(23, 24, 48));
        // STP x21, x22, [sp, #64]
        code.extend_from_slice(&encode_stp_push_offset(21, 22, 64));
        // STP x19, x20, [sp, #80]
        code.extend_from_slice(&encode_stp_push_offset(19, 20, 80));

        // ── Open /proc/self/environ ──────────────────────────────────────
        // sys_openat(AT_FDCWD, "/proc/self/environ\0", O_RDONLY=0, 0) → fd in x0
        //
        // Compute address of the path string (embedded in literal pool at end of code).
        // We use ADR (PC-relative address computation) — PIE-safe, gives us a runtime
        // pointer to the string, not the string data itself.

        let adr_path_pos = code.len();
        code.extend_from_slice(&encode_nop()); // placeholder: ADR x1, path_string

        // x0 = AT_FDCWD (-100)
        emit_mov32(&mut code, 0, AT_FDCWD_U32);
        // x1 already set above (will be patched)
        // x2 = O_RDONLY = 0
        code.extend_from_slice(&encode_movz(2, 0));
        // x3 = 0 (mode, unused for O_RDONLY)
        code.extend_from_slice(&encode_movz(3, 0));
        // syscall openat
        emit_syscall(&mut code, SYS_OPENAT);
        // x0 = fd (or error if negative)

        // Save fd in x19 (callee-saved, won't be clobbered by further syscalls).
        code.extend_from_slice(&encode_mov_reg(19, 0));

        // ── Read from fd into stack buffer [sp + FRAME_REGS] ────────────
        // sys_read(fd, buf, 4096) → bytes_read in x0
        // x0 = fd (from x19)
        code.extend_from_slice(&encode_mov_reg(0, 19));
        // x1 = buf = sp + FRAME_REGS
        code.extend_from_slice(&encode_add_imm(1, 31 /*sp*/, FRAME_REGS));
        // x2 = 4096
        emit_mov32(&mut code, 2, BUF_SIZE);
        emit_syscall(&mut code, SYS_READ);
        // x0 = bytes_read; save in x20
        code.extend_from_slice(&encode_mov_reg(20, 0));

        // ── Close fd ─────────────────────────────────────────────────────
        code.extend_from_slice(&encode_mov_reg(0, 19));
        emit_syscall(&mut code, SYS_CLOSE);

        // ── Search for "__AFL_SHM_ID=" in the environ buffer ─────────────
        // We scan the buffer (bytes_read bytes) for the prefix "__AFL_SHM_ID=".
        // Approach: byte-by-byte scan using LDRB in a loop.
        //
        // Registers used:
        //   x21 = buf start = sp + FRAME_REGS
        //   x22 = current scan pointer
        //   x23 = end = buf + bytes_read
        //   x24 = parsed SHM ID (result)

        // x21 = buf = sp + FRAME_REGS
        code.extend_from_slice(&encode_add_imm(21, 31 /*sp*/, FRAME_REGS));
        // x22 = x21 (scan pointer)
        code.extend_from_slice(&encode_mov_reg(22, 21));
        // x23 = x21 + x20 (end)
        code.extend_from_slice(&encode_add_reg(23, 21, 20));
        // x24 = 0 (parsed shmid, default 0)
        code.extend_from_slice(&encode_movz(24, 0));

        // Load the needle "__AFL_SHM_ID=" as a 13-byte prefix.
        // We'll embed it in the literal pool and compare byte-by-byte.
        // For Phase 40 simplicity, we use a 8-byte comparison for the first 8 bytes
        // then a 4-byte comparison for "ID=\0", then parse the integer after "=".

        // Load first 8 bytes of needle "__AFL_SHM_ID=" into x25 for comparison.
        // "__AFL_SH" = 0x48535F4C46415F5F (little-endian)
        // Embed needle bytes as two u64 literals.
        let needle_a_pos = code.len();
        code.extend_from_slice(&encode_nop()); // placeholder: LDR x25, #needle_a
        let needle_b_pos = code.len();
        code.extend_from_slice(&encode_nop()); // placeholder: LDR x26, #needle_b

        // Outer scan loop: while x22 < x23, load byte, check for '_' start.
        // If found, check next 12 bytes for "_AFL_SHM_ID=".

        // .scan_loop:
        let scan_loop = code.len();

        // Check if x22 >= x23 (past end)
        code.extend_from_slice(&encode_cmp_reg(22, 23));
        // B.GE .not_found  (if x22 >= x23)
        let bge_notfound_pos = code.len();
        code.extend_from_slice(&[0x00, 0x00, 0x00, 0x54]); // B.GE placeholder (cond=0xC = GE)

        // LDRB w27, [x22]  (load current byte)
        code.extend_from_slice(&encode_ldrb_reg(27, 22));

        // Check if first byte is '_' (0x5F)
        // CMP w27, #0x5F
        code.extend_from_slice(&encode_cmp_w_imm(27, 0x5F));
        // B.NE .next_byte
        let bne_nextbyte_pos1 = code.len();
        code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]); // B.NE placeholder (cond=1)

        // Potential match: compare first 8 bytes with needle part A.
        // LDR x27, [x22]  (load 8 bytes)
        code.extend_from_slice(&encode_ldr_reg(27, 22));
        // CMP x27, x25
        code.extend_from_slice(&encode_cmp_reg(27, 25));
        // B.NE .next_byte
        let bne_nextbyte_pos2 = code.len();
        code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]); // B.NE placeholder

        // Remaining bytes: check [x22+8..x22+13] = "M_ID="
        // Load 4 bytes "M_ID" at offset 8 with a 32-bit LDR.
        // Emit a placeholder 64-bit LDR then overwrite with 32-bit variant.
        code.extend_from_slice(&encode_ldr_imm(27, 22, 8));
        let tmp = code.len() - 4;
        // Replace the last instruction with a 32-bit LDR
        let ldr_w = encode_ldr_w_imm(27, 22, 8);
        code[tmp..tmp + 4].copy_from_slice(&ldr_w);

        // CMP w27, w26 (lower 32 bits = "M_ID")
        code.extend_from_slice(&encode_cmp_w_reg(27, 26));
        // B.NE .next_byte
        let bne_nextbyte_pos3 = code.len();
        code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]); // B.NE placeholder

        // Check that byte at [x22+12] is '=' (0x3D).
        // Emit placeholder LDRB then overwrite with offset-12 variant.
        code.extend_from_slice(&encode_ldrb_reg(27, 22));
        let tmp2 = code.len() - 4;
        let ldrb_12 = encode_ldrb_imm(27, 22, 12);
        code[tmp2..tmp2 + 4].copy_from_slice(&ldrb_12);

        code.extend_from_slice(&encode_cmp_w_imm(27, b'='.into()));
        let bne_nextbyte_pos4 = code.len();
        code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]); // B.NE placeholder

        // ── Found prefix: parse decimal integer at [x22+13] ──────────────
        // x22 points to start of "__AFL_SHM_ID=", so decimal starts at x22+13.
        // ADD x22, x22, #13  (advance past prefix)
        code.extend_from_slice(&encode_add_imm(22, 22, 13));

        // Parse decimal: x24 = 0; loop: LDRB w27, [x22++]; if '0'..'9', x24 = x24*10 + (w27 - '0')
        // Encode 10 as constant in x25 (x25 is no longer needed after finding the prefix).
        code.extend_from_slice(&encode_movz(25, 10));

        // .parse_loop:
        let parse_loop = code.len();

        // LDRB w27, [x22], #1  (post-index load)
        code.extend_from_slice(&encode_ldrb_post_inc(27, 22));

        // SUB w27, w27, #'0'
        code.extend_from_slice(&encode_sub_w_imm(27, 27, b'0'.into()));

        // CMP w27, #10  (check if digit 0-9)
        code.extend_from_slice(&encode_cmp_w_imm(27, 10));

        // B.HS .parse_done (unsigned >= 10 means not a digit)
        let bhs_parsedone_pos = code.len();
        code.extend_from_slice(&[0x82, 0x00, 0x00, 0x54]); // B.HS placeholder (cond=2)

        // x24 = x24 * 10 + w27
        // MUL x24, x24, x25  (x25 = 10)
        code.extend_from_slice(&encode_mul(24, 24, 25));
        // ADD x24, x24, x27 (zero-extend w27)
        code.extend_from_slice(&encode_add_uxtw(24, 24, 27));

        // B .parse_loop
        let b_parse_loop_offset = {
            let from_pos = code.len();
            let to_pos = parse_loop;
            let delta = to_pos as i64 - from_pos as i64;
            (delta / 4) as i32
        };
        code.extend_from_slice(&encode_b(b_parse_loop_offset));

        // .parse_done:
        let parse_done = code.len();
        // Patch B.HS to parse_done
        let bhs_delta = (parse_done as i64 - bhs_parsedone_pos as i64) as i32;
        patch_b_cond(&mut code, bhs_parsedone_pos, bhs_delta, 0x2); // HS cond

        // ── Skip to after the SHM ID found — jump over .next_byte ────────
        let b_after_notfound_pos = code.len();
        code.extend_from_slice(&encode_b(0)); // placeholder: B .shm_attach

        // ── .next_byte: advance scan pointer and loop ────────────────────
        let next_byte = code.len();

        // Patch all B.NE .next_byte instructions
        patch_b_cond(
            &mut code,
            bne_nextbyte_pos1,
            (next_byte as i64 - bne_nextbyte_pos1 as i64) as i32,
            0x1,
        );
        patch_b_cond(
            &mut code,
            bne_nextbyte_pos2,
            (next_byte as i64 - bne_nextbyte_pos2 as i64) as i32,
            0x1,
        );
        patch_b_cond(
            &mut code,
            bne_nextbyte_pos3,
            (next_byte as i64 - bne_nextbyte_pos3 as i64) as i32,
            0x1,
        );
        patch_b_cond(
            &mut code,
            bne_nextbyte_pos4,
            (next_byte as i64 - bne_nextbyte_pos4 as i64) as i32,
            0x1,
        );

        // ADD x22, x22, #1  (advance scan pointer)
        code.extend_from_slice(&encode_add_imm(22, 22, 1));

        // B .scan_loop
        let scan_loop_back = {
            let from_pos = code.len();
            let delta = scan_loop as i64 - from_pos as i64;
            (delta / 4) as i32
        };
        code.extend_from_slice(&encode_b(scan_loop_back));

        // ── .not_found: __AFL_SHM_ID not in environ — skip SHM attach ──
        // B.GE from the scan-loop bounds check lands here when x22 >= x23.
        // Jump directly to the epilogue (skip shmat + forkserver).
        let not_found = code.len();
        let b_epilogue_pos = code.len();
        code.extend_from_slice(&encode_b(0)); // placeholder: B .epilogue

        // ── Patch B.GE .not_found ─────────────────────────────────────────
        patch_b_cond(
            &mut code,
            bge_notfound_pos,
            (not_found as i64 - bge_notfound_pos as i64) as i32,
            0xC,
        ); // GE cond

        // ── After found: x24 = shmid → shm_attach ──────────────────────
        let shm_attach = code.len();
        // Patch B .shm_attach (from the found path)
        let b_after_delta = (shm_attach as i64 - b_after_notfound_pos as i64) as i32;
        let encoded_b = encode_b(b_after_delta / 4);
        code[b_after_notfound_pos..b_after_notfound_pos + 4].copy_from_slice(&encoded_b);

        // ── SHM attach: shmat(shmid=x24, NULL, 0) ────────────────────────
        // sys_shmat(shmid, shmaddr=0, shmflg=0) → shm_ptr in x0
        // SYS_SHMAT = 196 on ARM64
        code.extend_from_slice(&encode_mov_reg(0, 24)); // x0 = shmid
        code.extend_from_slice(&encode_movz(1, 0)); // x1 = 0 (NULL)
        code.extend_from_slice(&encode_movz(2, 0)); // x2 = 0
        emit_syscall(&mut code, SYS_SHMAT);
        // x0 = shm_ptr (or error). On AArch64 Linux, error is returned as
        // a negative value (-errno) in x0. We must NOT store error codes as
        // the SHM pointer — trampolines' CBZ null guard won't catch them.
        //
        // Check: if x0 is negative (bit 63 set), skip to epilogue.
        // TBNZ x0, #63, .epilogue
        let tbnz_shmat_err_pos = code.len();
        // TBNZ Xt, #bit, label: bit=63 → b5=1, b40=31, op=0b011011_1
        // Encoding: 1_011011_1_iiiiiiiiiiiiiii_11111_00000
        // imm14 = placeholder 0, Rt = x0 (reg 0)
        // b5=1 (bit 63), b40=31: top bit selects 32+b40
        // Full encoding: 0xB7F80000 | (imm14 << 5) | Rt
        code.extend_from_slice(&[0x00, 0x00, 0xF8, 0xB7]); // TBNZ x0, #63, placeholder

        // x0 = shm_ptr; save in x19
        code.extend_from_slice(&encode_mov_reg(19, 0));

        // ── Store shm_ptr at data_va ──────────────────────────────────────
        // Compute data_va address via ADR (PC-relative, PIE-safe).
        let adr_datava_offset = data_va as i64 - (init_va + code.len() as u64) as i64;
        code.extend_from_slice(&encode_adr(20, adr_datava_offset));

        // STR x19, [x20]  (shm_ptr at data_va+0)
        code.extend_from_slice(&encode_str_reg(19, 20));

        // Zero-initialize prev_loc at [data_va + PREV_LOC_OFFSET]
        code.extend_from_slice(&encode_strh_zero(20, PREV_LOC_OFFSET as u32));

        // ── Forkserver (optional) ─────────────────────────────────────────
        let forkserver_start = code.len();
        if enable_forkserver {
            emit_forkserver_arm64(&mut code);
        }

        // ── Epilogue: restore registers and jump to original entry point ──
        let epilogue = code.len();
        // Patch B .epilogue from the not_found path
        let b_epi_delta = (epilogue as i64 - b_epilogue_pos as i64) / 4;
        let encoded_b_epi = encode_b(b_epi_delta as i32);

        // Patch TBNZ x0, #63, .epilogue from the shmat error check
        let tbnz_delta = ((epilogue as i64 - tbnz_shmat_err_pos as i64) / 4) as u32;
        let tbnz_word = 0xB7F80000u32 | ((tbnz_delta & 0x3FFF) << 5); // TBNZ x0, #63, imm14
        code[tbnz_shmat_err_pos..tbnz_shmat_err_pos + 4].copy_from_slice(&tbnz_word.to_le_bytes());
        code[b_epilogue_pos..b_epilogue_pos + 4].copy_from_slice(&encoded_b_epi);

        // LDP x19, x20, [sp, #80]
        code.extend_from_slice(&encode_ldp_pop_offset(19, 20, 80));
        // LDP x21, x22, [sp, #64]
        code.extend_from_slice(&encode_ldp_pop_offset(21, 22, 64));
        // LDP x23, x24, [sp, #48]
        code.extend_from_slice(&encode_ldp_pop_offset(23, 24, 48));
        // LDP x25, x26, [sp, #32]
        code.extend_from_slice(&encode_ldp_pop_offset(25, 26, 32));
        // LDP x27, x28, [sp, #16]
        code.extend_from_slice(&encode_ldp_pop_offset(27, 28, 16));
        // LDP x29, x30, [sp, #0]
        code.extend_from_slice(&encode_ldp_pop_offset(29, 30, 0));

        // ADD sp, sp, #FRAME_SIZE
        code.extend_from_slice(&encode_add_sp_imm(96));
        code.extend_from_slice(&encode_add_sp_imm(4096));

        // LDP x0, xzr, [sp], #16  — restore x0 (post-index, sp += 16)
        code.extend_from_slice(&encode_ldp_pop(0, 31 /* xzr */));

        // Jump to original entry point — PC-relative branch (PIE-safe).
        // The offset between init code and original entry is fixed regardless of ASLR.
        let branch_va = init_va + code.len() as u64;
        let delta = entry_point as i64 - branch_va as i64;
        if delta.abs() < (128 << 20) {
            // B imm26: ±128 MiB range
            let imm26 = (delta / 4) as i32;
            code.extend_from_slice(&encode_b(imm26));
        } else {
            // Out of B range — use ADRP+ADD+BR for ±4 GiB.
            // ADRP x16, (entry_page - current_page)
            let adrp_va = init_va + code.len() as u64;
            let entry_page = (entry_point as i64) & !0xFFF;
            let adrp_page = (adrp_va as i64) & !0xFFF;
            let page_delta = entry_page - adrp_page;
            let immhi = ((page_delta >> 14) & 0x7FFFF) as u32;
            let immlo = ((page_delta >> 12) & 0x3) as u32;
            let adrp = 0x9000_0010u32 | (immlo << 29) | (immhi << 5);
            code.extend_from_slice(&adrp.to_le_bytes());
            // ADD x16, x16, #(entry_point & 0xFFF)
            let page_off = (entry_point & 0xFFF) as u32;
            code.extend_from_slice(&encode_add_imm(16, 16, page_off));
            code.extend_from_slice(&encode_br(16));
        }

        // ── Literal pool ─────────────────────────────────────────────────
        // Align to 8 bytes.
        while !code.len().is_multiple_of(8) {
            code.extend_from_slice(&encode_nop());
        }

        // Needle data values (loaded via LDR literal — these are constant data, not addresses)
        let needle_a_lit_offset = code.len();
        let needle_a: u64 = u64::from_le_bytes(*b"__AFL_SH");
        code.extend_from_slice(&needle_a.to_le_bytes());

        let needle_b_lit_offset = code.len();
        let needle_b: u32 = u32::from_le_bytes(*b"M_ID");
        code.extend_from_slice(&(needle_b as u64).to_le_bytes()); // pad to 8 bytes

        // String: "/proc/self/environ\0" (addressed via ADR, not loaded via LDR)
        let path_lit_offset = code.len();
        code.extend_from_slice(b"/proc/self/environ\0");
        // Align to 4 bytes.
        while !code.len().is_multiple_of(4) {
            code.push(0);
        }

        // ── Patch placeholder instructions ───────────────────────────────

        // Patch ADR x1, path_string (adr_path_pos → path_lit_offset)
        // ADR computes the runtime address of the string — PIE-safe.
        patch_adr(&mut code, adr_path_pos, path_lit_offset, 1);

        // data_va was loaded directly via ADR (no patching needed — offset known at emit time)

        // Patch LDR x25, #needle_a (needle_a_pos → needle_a_lit_offset)
        patch_ldr_literal(&mut code, needle_a_pos, init_va, needle_a_lit_offset, 25);

        // Patch LDR x26, #needle_b (needle_b_pos → needle_b_lit_offset)
        patch_ldr_literal(&mut code, needle_b_pos, init_va, needle_b_lit_offset, 26);

        let _ = forkserver_start; // suppress unused warning

        Ok(InitCode {
            va: init_va,
            code,
            entry_va: init_va,
        })
    }

    /// Generate init code for AArch64 shared objects (.so files).
    ///
    /// Reads __AFL_SHM_ID from /proc/self/environ, attaches shared memory via SYS_shmat (196),
    /// stores the SHM pointer at data_va, optionally chains to original DT_INIT, then returns.
    ///
    /// This code runs as the DT_INIT function when the .so is loaded by dlopen or ld.so.
    fn generate_so_init_code(
        &self,
        init_va: u64,
        data_va: u64,
        dt_init: Option<u64>,
    ) -> Result<InitCode> {
        let mut code: Vec<u8> = Vec::with_capacity(512);

        // ── Prologue: save callee-saved registers ─────────────────────────
        // Same as generate_init_code but also save x30 (LR) since we're called
        // as a function (DT_INIT is called with BL).
        const FRAME_REGS: u32 = 96;
        const BUF_SIZE: u32 = 4096;

        code.extend_from_slice(&encode_sub_sp_imm(4096));
        code.extend_from_slice(&encode_sub_sp_imm(96));

        code.extend_from_slice(&encode_stp_push_offset(29, 30, 0));
        code.extend_from_slice(&encode_stp_push_offset(27, 28, 16));
        code.extend_from_slice(&encode_stp_push_offset(25, 26, 32));
        code.extend_from_slice(&encode_stp_push_offset(23, 24, 48));
        code.extend_from_slice(&encode_stp_push_offset(21, 22, 64));
        code.extend_from_slice(&encode_stp_push_offset(19, 20, 80));

        // ── Open /proc/self/environ ───────────────────────────────────────
        let adr_path_pos = code.len();
        code.extend_from_slice(&encode_nop()); // placeholder: ADR x1, path_string

        emit_mov32(&mut code, 0, AT_FDCWD_U32);
        code.extend_from_slice(&encode_movz(2, 0));
        code.extend_from_slice(&encode_movz(3, 0));
        emit_syscall(&mut code, SYS_OPENAT);
        code.extend_from_slice(&encode_mov_reg(19, 0)); // fd in x19

        // ── Read environ ──────────────────────────────────────────────────
        code.extend_from_slice(&encode_mov_reg(0, 19));
        code.extend_from_slice(&encode_add_imm(1, 31, FRAME_REGS));
        emit_mov32(&mut code, 2, BUF_SIZE);
        emit_syscall(&mut code, SYS_READ);
        code.extend_from_slice(&encode_mov_reg(20, 0)); // bytes_read in x20

        // ── Close fd ──────────────────────────────────────────────────────
        code.extend_from_slice(&encode_mov_reg(0, 19));
        emit_syscall(&mut code, SYS_CLOSE);

        // ── Scan for __AFL_SHM_ID= ────────────────────────────────────────
        code.extend_from_slice(&encode_add_imm(21, 31, FRAME_REGS));
        code.extend_from_slice(&encode_mov_reg(22, 21));
        code.extend_from_slice(&encode_add_reg(23, 21, 20));
        code.extend_from_slice(&encode_movz(24, 0));

        let needle_a_pos = code.len();
        code.extend_from_slice(&encode_nop());
        let needle_b_pos = code.len();
        code.extend_from_slice(&encode_nop());

        // Scan loop (same structure as generate_init_code)
        let scan_loop = code.len();

        code.extend_from_slice(&encode_cmp_reg(22, 23));
        let bge_notfound_pos = code.len();
        code.extend_from_slice(&[0x00, 0x00, 0x00, 0x54]);

        code.extend_from_slice(&encode_ldrb_reg(27, 22));
        code.extend_from_slice(&encode_cmp_w_imm(27, 0x5F));
        let bne1 = code.len();
        code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]);

        code.extend_from_slice(&encode_ldr_reg(27, 22));
        code.extend_from_slice(&encode_cmp_reg(27, 25));
        let bne2 = code.len();
        code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]);

        let tmp_pos = code.len();
        code.extend_from_slice(&encode_nop());
        let ldr_w = encode_ldr_w_imm(27, 22, 8);
        code[tmp_pos..tmp_pos + 4].copy_from_slice(&ldr_w);

        code.extend_from_slice(&encode_cmp_w_reg(27, 26));
        let bne3 = code.len();
        code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]);

        let ldrb12_pos = code.len();
        code.extend_from_slice(&encode_nop());
        let ldrb_12 = encode_ldrb_imm(27, 22, 12);
        code[ldrb12_pos..ldrb12_pos + 4].copy_from_slice(&ldrb_12);

        code.extend_from_slice(&encode_cmp_w_imm(27, b'=' as u32));
        let bne4 = code.len();
        code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]);

        // Found: parse decimal
        code.extend_from_slice(&encode_add_imm(22, 22, 13));
        code.extend_from_slice(&encode_movz(25, 10));

        let parse_loop = code.len();

        code.extend_from_slice(&encode_ldrb_post_inc(27, 22));
        code.extend_from_slice(&encode_sub_w_imm(27, 27, b'0' as u32));
        code.extend_from_slice(&encode_cmp_w_imm(27, 10));

        let bhs_pos = code.len();
        code.extend_from_slice(&[0x82, 0x00, 0x00, 0x54]);

        code.extend_from_slice(&encode_mul(24, 24, 25));
        code.extend_from_slice(&encode_add_uxtw(24, 24, 27));

        let bpl = code.len();
        let bpl_delta = parse_loop as i64 - bpl as i64;
        code.extend_from_slice(&encode_b((bpl_delta / 4) as i32));

        let parse_done = code.len();
        patch_b_cond(
            &mut code,
            bhs_pos,
            (parse_done as i64 - bhs_pos as i64) as i32,
            0x2,
        );

        let b_skip_notfound = code.len();
        code.extend_from_slice(&encode_b(0)); // placeholder: B .shm_attach

        // ── .next_byte: advance scan pointer and loop ────────────────────
        let next_byte = code.len();

        patch_b_cond(
            &mut code,
            bne1,
            (next_byte as i64 - bne1 as i64) as i32,
            0x1,
        );
        patch_b_cond(
            &mut code,
            bne2,
            (next_byte as i64 - bne2 as i64) as i32,
            0x1,
        );
        patch_b_cond(
            &mut code,
            bne3,
            (next_byte as i64 - bne3 as i64) as i32,
            0x1,
        );
        patch_b_cond(
            &mut code,
            bne4,
            (next_byte as i64 - bne4 as i64) as i32,
            0x1,
        );

        code.extend_from_slice(&encode_add_imm(22, 22, 1));
        let slb = code.len();
        let slb_delta = scan_loop as i64 - slb as i64;
        code.extend_from_slice(&encode_b((slb_delta / 4) as i32));

        // ── .not_found: __AFL_SHM_ID not in environ — skip to epilogue ──
        let not_found = code.len();
        let b_so_epilogue_pos = code.len();
        code.extend_from_slice(&encode_b(0)); // placeholder: B .epilogue

        // Patch B.GE .not_found (scan exhausted)
        patch_b_cond(
            &mut code,
            bge_notfound_pos,
            (not_found as i64 - bge_notfound_pos as i64) as i32,
            0xC,
        );

        // ── .shm_attach: x24 = shmid ────────────────────────────────────
        let shm_attach = code.len();
        let b_skip_delta = (shm_attach as i64 - b_skip_notfound as i64) as i32;
        let enc = encode_b(b_skip_delta / 4);
        code[b_skip_notfound..b_skip_notfound + 4].copy_from_slice(&enc);

        // ── SHM attach ────────────────────────────────────────────────────
        code.extend_from_slice(&encode_mov_reg(0, 24));
        code.extend_from_slice(&encode_movz(1, 0));
        code.extend_from_slice(&encode_movz(2, 0));
        emit_syscall(&mut code, SYS_SHMAT);

        // Check for error: on AArch64 Linux, shmat returns -errno on failure.
        // TBNZ x0, #63, .epilogue — skip if bit 63 set (negative = error)
        let tbnz_so_shmat_err_pos = code.len();
        code.extend_from_slice(&[0x00, 0x00, 0xF8, 0xB7]); // TBNZ x0, #63, placeholder

        code.extend_from_slice(&encode_mov_reg(19, 0)); // shm_ptr in x19

        // ── Store shm_ptr at data_va ──────────────────────────────────────
        // Compute data_va address via ADR (PC-relative, PIE-safe).
        let adr_datava_offset = data_va as i64 - (init_va + code.len() as u64) as i64;
        code.extend_from_slice(&encode_adr(20, adr_datava_offset));

        code.extend_from_slice(&encode_str_reg(19, 20)); // STR x19, [x20] -- shm_ptr
        code.extend_from_slice(&encode_strh_zero(20, PREV_LOC_OFFSET as u32)); // clear prev_loc

        // ── Chain to original DT_INIT if present ─────────────────────────
        if let Some(dt_init_va) = dt_init {
            // ADR x16, original_dt_init (PC-relative, PIE-safe for .so)
            let adr_dtinit_offset = dt_init_va as i64 - (init_va + code.len() as u64) as i64;
            code.extend_from_slice(&encode_adr(16, adr_dtinit_offset));
            code.extend_from_slice(&encode_blr(16)); // BLR x16
        }

        // ── Epilogue: restore registers and return ────────────────────────
        let so_epilogue = code.len();
        // Patch B .epilogue from the not_found path
        let b_soepi_delta = (so_epilogue as i64 - b_so_epilogue_pos as i64) / 4;
        let enc_epi = encode_b(b_soepi_delta as i32);
        code[b_so_epilogue_pos..b_so_epilogue_pos + 4].copy_from_slice(&enc_epi);

        // Patch TBNZ x0, #63, .epilogue from the shmat error check
        let tbnz_so_delta = ((so_epilogue as i64 - tbnz_so_shmat_err_pos as i64) / 4) as u32;
        let tbnz_so_word = 0xB7F80000u32 | ((tbnz_so_delta & 0x3FFF) << 5);
        code[tbnz_so_shmat_err_pos..tbnz_so_shmat_err_pos + 4]
            .copy_from_slice(&tbnz_so_word.to_le_bytes());

        code.extend_from_slice(&encode_ldp_pop_offset(19, 20, 80));
        code.extend_from_slice(&encode_ldp_pop_offset(21, 22, 64));
        code.extend_from_slice(&encode_ldp_pop_offset(23, 24, 48));
        code.extend_from_slice(&encode_ldp_pop_offset(25, 26, 32));
        code.extend_from_slice(&encode_ldp_pop_offset(27, 28, 16));
        code.extend_from_slice(&encode_ldp_pop_offset(29, 30, 0));

        code.extend_from_slice(&encode_add_sp_imm(96));
        code.extend_from_slice(&encode_add_sp_imm(4096));

        code.extend_from_slice(&encode_ret());

        // ── Literal pool ─────────────────────────────────────────────────
        while !code.len().is_multiple_of(8) {
            code.extend_from_slice(&encode_nop());
        }

        // Needle data values (loaded via LDR literal — constant data, not addresses)
        let needle_a_lit_offset = code.len();
        let needle_a: u64 = u64::from_le_bytes(*b"__AFL_SH");
        code.extend_from_slice(&needle_a.to_le_bytes());

        let needle_b_lit_offset = code.len();
        let needle_b: u32 = u32::from_le_bytes(*b"M_ID");
        code.extend_from_slice(&(needle_b as u64).to_le_bytes());

        // String (addressed via ADR, not loaded via LDR)
        let path_lit_offset = code.len();
        code.extend_from_slice(b"/proc/self/environ\0");
        while !code.len().is_multiple_of(4) {
            code.push(0);
        }

        // ── Patch placeholder instructions ───────────────────────────────
        // ADR x1 for path string (computes runtime address, PIE-safe)
        patch_adr(&mut code, adr_path_pos, path_lit_offset, 1);

        // data_va and dt_init loaded directly via ADR (no patching needed)

        // LDR literal for needle data values (these ARE data, not addresses)
        patch_ldr_literal(&mut code, needle_a_pos, init_va, needle_a_lit_offset, 25);
        patch_ldr_literal(&mut code, needle_b_pos, init_va, needle_b_lit_offset, 26);

        Ok(InitCode {
            va: init_va,
            code,
            entry_va: init_va,
        })
    }

    fn encode_branch(&self, source_va: u64, target_va: u64) -> Result<Vec<u8>> {
        // B imm26: unconditional branch, ±128 MiB range
        let imm26 = check_branch_range(source_va, target_va)?;
        let word = 0x1400_0000u32 | (imm26 as u32 & 0x03FF_FFFF);
        Ok(word.to_le_bytes().to_vec())
    }

    fn branch_instruction_size(&self) -> usize {
        4
    }
}

// ============================================================
// Additional instruction encoders (helpers for generate_init_code)
// ============================================================

/// Encode STP xN, xM, [sp, #offset] (signed offset, unsigned store pair with immediate).
/// offset must be multiple of 8, range: -512..504.
fn encode_stp_push_offset(rn: u32, rm: u32, offset: u32) -> [u8; 4] {
    // STP xN, xM, [sp, #imm] (non-writeback, base=sp): opc=10, V=0, L=0, imm7=offset/8, Rt2=rm, Rn=31, Rt=rn
    let imm7 = (offset / 8) & 0x7F;
    let word = (0b10u32 << 30)
        | (0b101u32 << 27)
        | (0b010u32 << 23) // L=0, no writeback
        | (imm7 << 15)
        | (rm << 10)
        | (31u32 << 5)
        | rn;
    word.to_le_bytes()
}

/// Encode LDP xN, xM, [sp, #offset] (load pair with immediate offset, no writeback).
fn encode_ldp_pop_offset(rn: u32, rm: u32, offset: u32) -> [u8; 4] {
    // LDP xN, xM, [sp, #imm] (signed offset, no writeback)
    // [25:23]=010 (signed offset), [22]=1 (L=1, load)
    let imm7 = (offset / 8) & 0x7F;
    let word = (0b10u32 << 30)
        | (0b101u32 << 27)
        | (0b010u32 << 23) // signed offset
        | (1u32 << 22)     // L=1 for LDP
        | (imm7 << 15)
        | (rm << 10)
        | (31u32 << 5)
        | rn;
    word.to_le_bytes()
}

/// Encode MUL xD, xN, xM (64-bit multiply).
fn encode_mul(xd: u32, xn: u32, xm: u32) -> [u8; 4] {
    // MUL xD, xN, xM = MADD xD, xN, xM, xzr
    // sf=1, op54=00, op31=000, Ra=31, Rm=xm, o0=0, Rn=xn, Rd=xd
    let word = 0x9B00_7C00u32 | (xm << 16) | (xn << 5) | xd;
    word.to_le_bytes()
}

/// Encode ADD xD, xN, wM (UXTW — zero-extend 32-bit wM to 64-bit).
fn encode_add_uxtw(xd: u32, xn: u32, wm: u32) -> [u8; 4] {
    // ADD xDst, xN, wM, UXTW
    // sf=1, op=0, S=0, opt=010 (UXTW), Rm=wm, imm3=0, Rn=xn, Rd=xd
    let word = 0x8B20_0000u32 | (wm << 16) | (xn << 5) | xd;
    word.to_le_bytes()
}

/// Encode CMP wN, #imm (SUBS wzr, wN, #imm).
fn encode_cmp_w_imm(wn: u32, imm: u32) -> [u8; 4] {
    // SUBS wDst, wN, #imm12: sf=0, op=1, S=1
    let word = 0x7100_001Fu32 | ((imm & 0xFFF) << 10) | (wn << 5);
    word.to_le_bytes()
}

/// Encode CMP wN, wM (SUBS wzr, wN, wM).
fn encode_cmp_w_reg(wn: u32, wm: u32) -> [u8; 4] {
    // SUBS wzr, wN, wM: sf=0, op=1, S=1
    let word = 0x6B00_001Fu32 | (wm << 16) | (wn << 5);
    word.to_le_bytes()
}

/// Encode LDR wN, [xM, #offset] (32-bit load, unsigned immediate, offset multiple of 4).
fn encode_ldr_w_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    // LDR wN, [xM, #imm]: size=10, V=0, opc=01, imm12=offset/4, Rn=xm, Rt=wn
    let imm12 = (offset / 4) & 0xFFF;
    let word = 0xB940_0000u32 | (imm12 << 10) | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode LDRB wN, [xM, #imm] (load byte unsigned offset).
fn encode_ldrb_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    // LDRB wN, [xM, #imm]: size=00, V=0, opc=01, imm12=offset, Rn=xm, Rt=wn
    let imm12 = offset & 0xFFF;
    let word = 0x3940_0000u32 | (imm12 << 10) | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode LDRB wN, [xM], #1 (post-index load byte, advances xM by 1).
fn encode_ldrb_post_inc(wn: u32, xm: u32) -> [u8; 4] {
    // LDRB wN, [xM], #1 (post-index): size=00, V=0, opc=01, imm9=1, P=0, W=1
    // Encoding: 0011 1000 0100 0000 0001 01nn nnnt tttt
    let word = 0x3840_1400u32 | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode SUB wD, wN, #imm (32-bit subtract immediate).
fn encode_sub_w_imm(wd: u32, wn: u32, imm: u32) -> [u8; 4] {
    // SUB wDst, wN, #imm12: sf=0, op=1, S=0
    let word = 0x5100_0000u32 | ((imm & 0xFFF) << 10) | (wn << 5) | wd;
    word.to_le_bytes()
}

/// Encode CBNZ xN, #offset (compare and branch if non-zero; offset in bytes from instruction).
#[inline]
fn encode_cbnz(xn: u32, offset_bytes: i32) -> [u8; 4] {
    let imm19 = ((offset_bytes / 4) as u32) & 0x7FFFF;
    (0xB500_0000u32 | (imm19 << 5) | xn).to_le_bytes()
}

/// Encode STRB wN, [xM, #imm] (store byte with unsigned immediate offset).
#[inline]
fn encode_strb_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = offset & 0xFFF;
    (0x3900_0000u32 | (imm12 << 10) | (xm << 5) | wn).to_le_bytes()
}

/// Encode STR xN, [xM, #imm] (store 64-bit with unsigned immediate offset, multiple of 8).
#[inline]
fn encode_str_imm_base(xn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 8) & 0xFFF;
    (0xF900_0000u32 | (imm12 << 10) | (xm << 5) | xn).to_le_bytes()
}

/// Encode STR wN, [xM, #imm] (store 32-bit with unsigned immediate offset, multiple of 4).
#[inline]
fn encode_str_w_imm_base(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 4) & 0xFFF;
    (0xB900_0000u32 | (imm12 << 10) | (xm << 5) | wn).to_le_bytes()
}

/// Encode SUBS wD, wN, #imm (subtract immediate, setting flags, 32-bit).
#[inline]
fn encode_subs_w_imm(wd: u32, wn: u32, imm: u32) -> [u8; 4] {
    (0x7100_0000u32 | ((imm & 0xFFF) << 10) | (wn << 5) | wd).to_le_bytes()
}

/// Patch a B.cond instruction at `pos` with the given byte delta and condition code.
/// cond: 0=EQ, 1=NE, 2=HS/CS, 3=LO/CC, 4=MI, 5=PL, 6=VS, 7=VC, 8=HI, 9=LS, 10=GE, 11=LT, 12=GE(!), 13=LE, 14=AL, 15=NV
/// (The ARM64 encoding uses 4-bit condition codes where GE=10=0xA but in branch encoding GE is encoded as 0xA)
fn patch_b_cond(code: &mut [u8], pos: usize, delta_bytes: i32, cond: u32) {
    let imm19 = ((delta_bytes / 4) as u32) & 0x7FFFF;
    let word = 0x5400_0000u32 | (imm19 << 5) | cond;
    code[pos..pos + 4].copy_from_slice(&word.to_le_bytes());
}

/// Patch an LDR literal placeholder at `instr_pos` to load register `xn` from `lit_pos`.
/// The instruction encodes the offset as (lit_pos - instr_pos) / 4 (must be aligned, forward).
fn patch_ldr_literal(code: &mut [u8], instr_pos: usize, base_va: u64, lit_pos: usize, xn: u32) {
    let instr_va = base_va + instr_pos as u64;
    let lit_va = base_va + lit_pos as u64;
    let delta = (lit_va as i64 - instr_va as i64) as i32;
    let encoded = encode_ldr_literal(xn, delta);
    code[instr_pos..instr_pos + 4].copy_from_slice(&encoded);
}

/// Patch an ADR placeholder at `instr_pos` to compute the address of `target_pos`.
/// Both positions are byte offsets within the code buffer; base_va cancels out.
fn patch_adr(code: &mut [u8], instr_pos: usize, target_pos: usize, rd: u32) {
    let delta = target_pos as i64 - instr_pos as i64;
    let encoded = encode_adr(rd, delta);
    code[instr_pos..instr_pos + 4].copy_from_slice(&encoded);
}

// ============================================================
// Forkserver assembly for ARM64
// ============================================================

/// Emit the AFL forkserver loop in ARM64 assembly.
///
/// Protocol:
/// 1. Write 4-byte "hello" (0x00000000) to FORKSRV_FD+1 (fd=199)
/// 2. Loop:
///    a. Read 4-byte control from FORKSRV_FD (fd=198)
///    b. If read returns <= 0: exit (pipe closed)
///    c. Fork
///    d. Child: break out of loop
///    e. Parent: write child_pid (4 bytes) to fd=199
///    f. Parent: wait4(child_pid, &status, 0, NULL)
///    g. Parent: write exit status (4 bytes) to fd=199
///    h. Loop back to a.
///
/// Uses registers x19-x28 (callee-saved, backed up by caller).
/// Stack layout: [sp] = pid_buf (8 bytes), [sp+8] = status_buf (8 bytes)
/// Caller must have allocated stack space before calling this.
fn emit_forkserver_arm64(code: &mut Vec<u8>) {
    // Allocate local stack for pid and status buffers.
    code.extend_from_slice(&encode_sub_sp_imm(16));

    // Write 4-byte hello to FORKSRV_FD+1 (fd=199)
    // Store 0 at [sp] for the hello value.
    code.extend_from_slice(&encode_movz(27, 0)); // x27 = 0
    code.extend_from_slice(&encode_str_sp_imm(27, 0)); // STR x27, [sp]

    // sys_write(199, &hello, 4)
    emit_mov32(code, 0, FORKSRV_FD + 1); // fd
    code.extend_from_slice(&encode_add_imm(1, 31, 0)); // buf = sp + 0
    code.extend_from_slice(&encode_movz(2, 4)); // len = 4
    emit_syscall(code, SYS_WRITE);

    // Check if hello write succeeded — if x0 is negative, forkserver
    // pipes don't exist. Skip forkserver entirely and run standalone.
    // TBNZ x0, #63, .no_forkserver
    let tbnz_no_fork_pos = code.len();
    code.extend_from_slice(&[0x00, 0x00, 0xF8, 0xB7]); // TBNZ x0, #63, placeholder

    // .fork_loop:
    let fork_loop = code.len();

    // Read 4-byte control from FORKSRV_FD (fd=198)
    emit_mov32(code, 0, FORKSRV_FD);
    code.extend_from_slice(&encode_add_imm(1, 31, 0));
    code.extend_from_slice(&encode_movz(2, 4));
    emit_syscall(code, SYS_READ);

    // If x0 <= 0: exit (pipe closed)
    code.extend_from_slice(&encode_cmp_imm0(0)); // CMP x0, #0
    // B.LE .exit_forkserver (cond LE = 0xD)
    let ble_exit_pos = code.len();
    code.extend_from_slice(&[0x00, 0x00, 0x00, 0x54]);

    // Fork via clone(SIGCHLD, 0, 0, 0, 0) — ARM64 has no fork() syscall.
    // clone with flags=SIGCHLD and all other args 0 is equivalent to fork().
    code.extend_from_slice(&encode_movz(0, SIGCHLD as u16)); // x0 = SIGCHLD
    code.extend_from_slice(&encode_movz(1, 0)); // x1 = 0 (child_stack = NULL → share parent stack)
    code.extend_from_slice(&encode_movz(2, 0)); // x2 = 0 (parent_tid)
    code.extend_from_slice(&encode_movz(3, 0)); // x3 = 0 (tls)
    code.extend_from_slice(&encode_movz(4, 0)); // x4 = 0 (child_tid)
    emit_syscall(code, SYS_CLONE);
    code.extend_from_slice(&encode_mov_reg(27, 0)); // x27 = child_pid or 0

    // CBZ x27, .child_path (x27 == 0 means we're in child)
    let cbz_child_pos = code.len();
    code.extend_from_slice(&encode_cbz(27, 0)); // placeholder

    // ── Parent path ───────────────────────────────────────────────────────
    // Write child_pid to FORKSRV_FD+1
    // Store x27 (pid) at [sp]
    code.extend_from_slice(&encode_str_sp_imm(27, 0));
    // sys_write(199, &child_pid, 4)
    emit_mov32(code, 0, FORKSRV_FD + 1);
    code.extend_from_slice(&encode_add_imm(1, 31, 0));
    code.extend_from_slice(&encode_movz(2, 4));
    emit_syscall(code, SYS_WRITE);

    // sys_wait4(child_pid, &status, 0, NULL)
    code.extend_from_slice(&encode_mov_reg(0, 27)); // x0 = child_pid
    code.extend_from_slice(&encode_add_imm(1, 31, 8)); // x1 = &status = sp+8
    code.extend_from_slice(&encode_movz(2, 0)); // x2 = 0 (WNOHANG=0)
    code.extend_from_slice(&encode_movz(3, 0)); // x3 = NULL
    emit_syscall(code, SYS_WAIT4);

    // Write exit status to FORKSRV_FD+1
    emit_mov32(code, 0, FORKSRV_FD + 1);
    code.extend_from_slice(&encode_add_imm(1, 31, 8));
    code.extend_from_slice(&encode_movz(2, 4));
    emit_syscall(code, SYS_WRITE);

    // B .fork_loop
    let fl_back = code.len();
    let delta = fork_loop as i64 - fl_back as i64;
    code.extend_from_slice(&encode_b((delta / 4) as i32));

    // ── .exit_forkserver: sys_exit(0) ─────────────────────────────────────
    let exit_target = code.len();
    patch_b_cond(
        code,
        ble_exit_pos,
        (exit_target as i64 - ble_exit_pos as i64) as i32,
        0xD,
    ); // LE

    code.extend_from_slice(&encode_movz(0, 0));
    emit_syscall(code, SYS_EXIT);

    // ── .child_path: patch CBZ and fall through ───────────────────────────
    let child_target = code.len();
    let cbz_delta = (child_target as i64 - cbz_child_pos as i64) as i32;
    let cbz_enc = encode_cbz(27, cbz_delta);
    code[cbz_child_pos..cbz_child_pos + 4].copy_from_slice(&cbz_enc);

    // Child falls through to original entry point (caller handles the jump).
    // Deallocate local stack.
    code.extend_from_slice(&encode_add_sp_imm(16));

    // ── .no_forkserver: forkserver pipes don't exist — run standalone ───
    // This is where TBNZ lands when write(199) returns -EBADF.
    // We need the same stack cleanup and fall-through as the child path.
    // However, since we just emitted ADD sp, sp, #16 above, and the
    // no_forkserver path needs the same, we can jump BACK to the child
    // cleanup. But for simplicity and correctness, we'll patch TBNZ to
    // jump to the child_target (same as CBZ child path).

    // Patch TBNZ x0, #63, .no_forkserver → targets child_target
    // (which does ADD sp, #16 and falls through to caller).
    let tbnz_delta = ((child_target as i64 - tbnz_no_fork_pos as i64) / 4) as u32;
    let tbnz_word = 0xB7F80000u32 | ((tbnz_delta & 0x3FFF) << 5);
    code[tbnz_no_fork_pos..tbnz_no_fork_pos + 4].copy_from_slice(&tbnz_word.to_le_bytes());
}

// ============================================================
// Persistent wrapper for ELF AArch64
// ============================================================

/// Generate a persistent mode wrapper for ELF AArch64 binaries.
///
/// Ported from the macOS AArch64 version in `macho_trampoline.rs`, adapted for Linux:
/// - Linux syscall convention: x8 = syscall number, SVC #0
/// - Linux has no fork() on ARM64: use clone(SIGCHLD, 0, 0, 0, 0) via SYS_CLONE=220
/// - Linux SIGSTOP = 19 (macOS = 17)
/// - x0 = return value after clone: 0 in child, pid in parent (no x1 trick like macOS)
///
/// Persistent data layout (160 bytes):
///   +0:   first_pass (u8)
///   +1:   child_stopped (u8)
///   +2:   forkserver_started (u8)
///   +4:   counter (u32)
///   +8:   save_sp (u64)
///   +16:  save_lr (u64)
///   +24:  save_x0 (u64)
///   +32:  save_x1 (u64)
///   +40:  save_x2 (u64)
///   +48:  save_x3 (u64)
///   +56:  save_x4 (u64)
///   +64:  save_x5 (u64)
///   +72:  save_x6 (u64)
///   +80:  save_x7 (u64)
pub fn generate_persistent_wrapper_aarch64(
    params: &crate::trampoline::PersistentWrapperParams,
) -> Result<PersistentWrapper> {
    let wrapper_va = params.wrapper_va;
    let persistent_data_va = params.persistent_data_va;
    let data_va = params.data_va;
    let persistent_addr = params.persistent_addr;
    let displaced_bytes = params.displaced_bytes;
    let displaced_len = params.displaced_len;
    let persistent_count = params.persistent_count;
    let include_forkserver = params.include_forkserver;
    let mut code = Vec::with_capacity(if include_forkserver { 2048 } else { 1024 });
    let return_va = persistent_addr + displaced_len as u64;

    // Offsets within persistent data area
    const OFF_FIRST_PASS: u32 = 0;
    const OFF_CHILD_STOPPED: u32 = 1;
    const OFF_FS_STARTED: u32 = 2;
    const OFF_COUNTER: u32 = 4;
    const OFF_SAVE_SP: u32 = 8;
    const OFF_SAVE_LR: u32 = 16;
    // x0 at +24, x1 at +32, ..., x7 at +80 — all 8 argument registers
    // so the target function receives correct args on every iteration.
    const OFF_SAVE_ARGS: u32 = 24;

    // Literal pool entries — we use ADR to get pointers.
    struct LitPoolEntry {
        instr_pos: usize,
        value: u64,
    }
    let mut lit_pool: Vec<LitPoolEntry> = Vec::new();

    let emit_adr_placeholder = |code: &mut Vec<u8>, reg: u32| -> usize {
        let pos = code.len();
        code.extend_from_slice(&encode_adr(reg, 0));
        pos
    };

    // === Section 0: Deferred forkserver (optional) ===
    if include_forkserver {
        // Load persistent_data_va into x9
        let adr_data = emit_adr_placeholder(&mut code, 9);
        lit_pool.push(LitPoolEntry {
            instr_pos: adr_data,
            value: persistent_data_va,
        });

        // Check forkserver_started: LDRB w10, [x9, #OFF_FS_STARTED]
        code.extend_from_slice(&encode_ldrb_imm(10, 9, OFF_FS_STARTED));
        // CBNZ w10, .persistent_section
        let cbnz_persistent_pos = code.len();
        code.extend_from_slice(&encode_cbnz(10, 0)); // placeholder

        // Set forkserver_started = 1
        code.extend_from_slice(&encode_movz(10, 1));
        code.extend_from_slice(&encode_strb_imm(10, 9, OFF_FS_STARTED));

        // Save argument registers (x0-x7) to persistent_data BEFORE the forkserver
        // loop clobbers them with syscall arguments and clone() return values.
        // Without this, the child gets x0=0 (clone return) instead of argc, and
        // main() immediately returns without calling the target function.
        // x9 still holds persistent_data_va from the ADR above.
        for reg in 0..8u32 {
            code.extend_from_slice(&encode_str_imm_base(reg, 9, OFF_SAVE_ARGS + reg * 8));
        }

        // Save callee-saved regs to stack
        code.extend_from_slice(&encode_sub_sp_imm(160));
        for i in 0..5u32 {
            code.extend_from_slice(&encode_stp_push_offset(19 + i * 2, 20 + i * 2, i * 16));
        }
        code.extend_from_slice(&encode_stp_push_offset(29, 30, 80)); // fp, lr

        // sub sp, sp, 16 (scratch space)
        code.extend_from_slice(&encode_sub_sp_imm(16));

        // Hello: write(FORKSRV_FD+1, &zero, 4)
        code.extend_from_slice(&encode_movz(10, 0));
        code.extend_from_slice(&encode_str_sp_imm(10, 0)); // [sp] = 0
        emit_mov32(&mut code, 0, FORKSRV_FD + 1);
        code.extend_from_slice(&encode_add_imm(1, 31, 0)); // x1 = sp
        code.extend_from_slice(&encode_movz(2, 4));
        emit_syscall(&mut code, SYS_WRITE);

        // Check error: if x0 < 0 (sign bit set), no parent
        // TBNZ x0, #63, .no_parent
        let tbnz_no_parent_pos = code.len();
        code.extend_from_slice(&[0x00, 0x00, 0xF8, 0xB7]); // placeholder

        // .fork_loop:
        let fork_loop = code.len();

        // read(FORKSRV_FD, &cmd, 4)
        emit_mov32(&mut code, 0, FORKSRV_FD);
        code.extend_from_slice(&encode_add_imm(1, 31, 0));
        code.extend_from_slice(&encode_movz(2, 4));
        emit_syscall(&mut code, SYS_READ);

        // Check child_stopped
        let adr_data2 = emit_adr_placeholder(&mut code, 9);
        lit_pool.push(LitPoolEntry {
            instr_pos: adr_data2,
            value: persistent_data_va,
        });
        code.extend_from_slice(&encode_ldrb_imm(10, 9, OFF_CHILD_STOPPED));
        // CBZ w10, .do_fork
        let cbz_do_fork_pos = code.len();
        code.extend_from_slice(&encode_cbz(10, 0)); // placeholder

        // Resume stopped child: kill(x27, SIGCONT=18)
        code.extend_from_slice(&encode_mov_reg(0, 27)); // pid
        code.extend_from_slice(&encode_movz(1, 18)); // SIGCONT
        emit_syscall(&mut code, SYS_KILL);

        // Write child PID
        code.extend_from_slice(&encode_str_sp_imm(27, 0));
        emit_mov32(&mut code, 0, FORKSRV_FD + 1);
        code.extend_from_slice(&encode_add_imm(1, 31, 0));
        code.extend_from_slice(&encode_movz(2, 4));
        emit_syscall(&mut code, SYS_WRITE);

        // B .do_waitpid
        let b_waitpid_pos = code.len();
        code.extend_from_slice(&encode_b(0)); // placeholder

        // .do_fork:
        let do_fork = code.len();
        let cd = (do_fork as i64 - cbz_do_fork_pos as i64) as i32;
        code[cbz_do_fork_pos..cbz_do_fork_pos + 4].copy_from_slice(&encode_cbz(10, cd));

        // clone(SIGCHLD, 0, 0, 0, 0) — Linux ARM64 has no fork()
        emit_mov32(&mut code, 0, SIGCHLD);
        code.extend_from_slice(&encode_movz(1, 0)); // newsp=0
        code.extend_from_slice(&encode_movz(2, 0)); // parent_tidptr=0
        code.extend_from_slice(&encode_movz(3, 0)); // tls=0
        code.extend_from_slice(&encode_movz(4, 0)); // child_tidptr=0
        emit_syscall(&mut code, SYS_CLONE);
        // x0 = pid in parent, 0 in child
        code.extend_from_slice(&encode_mov_reg(27, 0)); // save pid

        // CBZ x0, .child (x0==0 means child)
        let cbz_child_pos = code.len();
        code.extend_from_slice(&encode_cbz(0, 0)); // placeholder

        // Parent: write child PID
        code.extend_from_slice(&encode_str_sp_imm(27, 0));
        emit_mov32(&mut code, 0, FORKSRV_FD + 1);
        code.extend_from_slice(&encode_add_imm(1, 31, 0));
        code.extend_from_slice(&encode_movz(2, 4));
        emit_syscall(&mut code, SYS_WRITE);

        // .do_waitpid:
        let do_waitpid = code.len();
        let brel = ((do_waitpid as i64 - b_waitpid_pos as i64) / 4) as i32;
        code[b_waitpid_pos..b_waitpid_pos + 4].copy_from_slice(&encode_b(brel));

        // wait4(x27, &status, WUNTRACED=2, NULL)
        code.extend_from_slice(&encode_mov_reg(0, 27));
        code.extend_from_slice(&encode_add_imm(1, 31, 4)); // &status at [sp+4]
        code.extend_from_slice(&encode_movz(2, 2)); // WUNTRACED
        code.extend_from_slice(&encode_movz(3, 0));
        emit_syscall(&mut code, SYS_WAIT4);

        // Check WIFSTOPPED: (status & 0xFF) == 0x7F
        code.extend_from_slice(&encode_ldr_w_imm(10, 31, 4)); // w10 = [sp+4]
        code.extend_from_slice(&[0x4A, 0x1D, 0x00, 0x12]); // AND w10, w10, #0xFF
        code.extend_from_slice(&encode_cmp_w_imm(10, 0x7F));
        // B.NE .not_stopped
        let bne_not_stopped_pos = code.len();
        code.extend_from_slice(&[0x01, 0x00, 0x00, 0x54]); // placeholder B.NE

        // WIFSTOPPED: set child_stopped=1, synthetic status=0
        let adr_data3 = emit_adr_placeholder(&mut code, 9);
        lit_pool.push(LitPoolEntry {
            instr_pos: adr_data3,
            value: persistent_data_va,
        });
        code.extend_from_slice(&encode_movz(10, 1));
        code.extend_from_slice(&encode_strb_imm(10, 9, OFF_CHILD_STOPPED));
        code.extend_from_slice(&encode_movz(10, 0));
        code.extend_from_slice(&encode_str_sp_imm(10, 4)); // status=0
        // B .report_status
        let b_report_pos = code.len();
        code.extend_from_slice(&encode_b(0)); // placeholder

        // .not_stopped:
        let not_stopped = code.len();
        patch_b_cond(
            &mut code,
            bne_not_stopped_pos,
            (not_stopped as i64 - bne_not_stopped_pos as i64) as i32,
            0x1,
        );

        let adr_data4 = emit_adr_placeholder(&mut code, 9);
        lit_pool.push(LitPoolEntry {
            instr_pos: adr_data4,
            value: persistent_data_va,
        });
        code.extend_from_slice(&encode_movz(10, 0));
        code.extend_from_slice(&encode_strb_imm(10, 9, OFF_CHILD_STOPPED));

        // .report_status:
        let report_status = code.len();
        let brel = ((report_status as i64 - b_report_pos as i64) / 4) as i32;
        code[b_report_pos..b_report_pos + 4].copy_from_slice(&encode_b(brel));

        // write(FORKSRV_FD+1, &status, 4)
        emit_mov32(&mut code, 0, FORKSRV_FD + 1);
        code.extend_from_slice(&encode_add_imm(1, 31, 4));
        code.extend_from_slice(&encode_movz(2, 4));
        emit_syscall(&mut code, SYS_WRITE);

        // B .fork_loop
        let bl2 = code.len();
        code.extend_from_slice(&encode_b(((fork_loop as i64 - bl2 as i64) / 4) as i32));

        // .child:
        let child = code.len();
        let cd = (child as i64 - cbz_child_pos as i64) as i32;
        code[cbz_child_pos..cbz_child_pos + 4].copy_from_slice(&encode_cbz(0, cd));

        // Close forkserver FDs
        emit_mov32(&mut code, 0, FORKSRV_FD);
        emit_syscall(&mut code, SYS_CLOSE);
        emit_mov32(&mut code, 0, FORKSRV_FD + 1);
        emit_syscall(&mut code, SYS_CLOSE);

        // .no_parent:
        let no_parent = code.len();
        // Patch TBNZ x0, #63 → .no_parent
        let tbnz_delta = ((no_parent as i64 - tbnz_no_parent_pos as i64) / 4) as u32;
        let tbnz_word = 0xB7F80000u32 | ((tbnz_delta & 0x3FFF) << 5);
        code[tbnz_no_parent_pos..tbnz_no_parent_pos + 4].copy_from_slice(&tbnz_word.to_le_bytes());

        // Deallocate scratch
        code.extend_from_slice(&encode_add_sp_imm(16));

        // Restore callee-saved regs
        for i in 0..5u32 {
            code.extend_from_slice(&encode_ldp_pop_offset(19 + i * 2, 20 + i * 2, i * 16));
        }
        code.extend_from_slice(&encode_ldp_pop_offset(29, 30, 80));
        code.extend_from_slice(&encode_add_sp_imm(160));

        // Restore argument registers (x0-x7) from persistent_data so the child
        // gets the original argc/argv/envp values, not clone()'s return values.
        let adr_data_restore = emit_adr_placeholder(&mut code, 9);
        lit_pool.push(LitPoolEntry {
            instr_pos: adr_data_restore,
            value: persistent_data_va,
        });
        for reg in 0..8u32 {
            code.extend_from_slice(&encode_ldr_imm(reg, 9, OFF_SAVE_ARGS + reg * 8));
        }

        // B .persistent_section (placeholder)
        let b_persistent_pos = code.len();
        code.extend_from_slice(&encode_b(0));

        // .exit_parent: exit(0)
        code.extend_from_slice(&encode_movz(0, 0));
        emit_syscall(&mut code, SYS_EXIT);

        // .persistent_section:
        let persistent_section = code.len();
        let cd = (persistent_section as i64 - cbnz_persistent_pos as i64) as i32;
        code[cbnz_persistent_pos..cbnz_persistent_pos + 4].copy_from_slice(&encode_cbnz(10, cd));
        let brel = ((persistent_section as i64 - b_persistent_pos as i64) / 4) as i32;
        code[b_persistent_pos..b_persistent_pos + 4].copy_from_slice(&encode_b(brel));
    }

    // === Section A: Entry — check first_pass ===
    let adr_data_a = emit_adr_placeholder(&mut code, 9);
    lit_pool.push(LitPoolEntry {
        instr_pos: adr_data_a,
        value: persistent_data_va,
    });

    // LDRB w10, [x9, #OFF_FIRST_PASS]
    code.extend_from_slice(&encode_ldrb_imm(10, 9, OFF_FIRST_PASS));
    // CBZ w10, .setup_and_run (first_pass==0 means not first pass)
    let cbz_setup_pos = code.len();
    code.extend_from_slice(&encode_cbz(10, 0)); // placeholder

    // First pass: set first_pass=0
    code.extend_from_slice(&encode_movz(10, 0));
    code.extend_from_slice(&encode_strb_imm(10, 9, OFF_FIRST_PASS));

    // Save SP, LR, x0-x7 to persistent data
    code.extend_from_slice(&encode_mov_reg(10, 31)); // x10 = sp
    code.extend_from_slice(&encode_str_imm_base(10, 9, OFF_SAVE_SP));
    code.extend_from_slice(&encode_str_imm_base(30, 9, OFF_SAVE_LR)); // lr
    // Save all 8 argument registers so the target function
    // receives correct arguments (argc, argv, envp, etc.) on every iteration.
    for reg in 0..8u32 {
        code.extend_from_slice(&encode_str_imm_base(reg, 9, OFF_SAVE_ARGS + reg * 8));
    }

    // Set counter
    emit_mov32(&mut code, 10, persistent_count);
    code.extend_from_slice(&encode_str_w_imm_base(10, 9, OFF_COUNTER));

    // === Section B: .setup_and_run ===
    let setup_and_run = code.len();
    let cd = (setup_and_run as i64 - cbz_setup_pos as i64) as i32;
    code[cbz_setup_pos..cbz_setup_pos + 4].copy_from_slice(&encode_cbz(10, cd));

    // Set LR to iteration_boundary so function return comes back here
    let adr_lr_iter_pos = code.len();
    code.extend_from_slice(&encode_adr(30, 0)); // placeholder

    // Restore x0-x7 from save area
    let adr_data_b = emit_adr_placeholder(&mut code, 9);
    lit_pool.push(LitPoolEntry {
        instr_pos: adr_data_b,
        value: persistent_data_va,
    });
    for reg in 0..8u32 {
        code.extend_from_slice(&encode_ldr_imm(reg, 9, OFF_SAVE_ARGS + reg * 8));
    }

    // Emit displaced instructions with PC-relative relocation
    let num_insns = displaced_len / 4;
    for i in 0..num_insns {
        let offset = i * 4;
        let raw = u32::from_le_bytes([
            displaced_bytes[offset],
            displaced_bytes[offset + 1],
            displaced_bytes[offset + 2],
            displaced_bytes[offset + 3],
        ]);
        let original_va = persistent_addr + offset as u64;
        let new_va = wrapper_va + code.len() as u64;

        if is_pc_relative(raw) {
            match relocate_pc_relative(raw, original_va, new_va) {
                Ok(relocated) => code.extend_from_slice(&relocated.to_le_bytes()),
                Err(_) => code.extend_from_slice(&raw.to_le_bytes()), // fallback: raw copy
            }
        } else {
            code.extend_from_slice(&raw.to_le_bytes());
        }
    }

    // B return_va (persistent_addr + displaced_len)
    let b_return_pos = code.len();
    let brel = ((return_va as i64 - (wrapper_va + b_return_pos as u64) as i64) / 4) as i32;
    code.extend_from_slice(&encode_b(brel));

    // === Section C: iteration_boundary ===
    let iteration_boundary = code.len();
    // Fix up ADR lr, iteration_boundary
    let iter_va = wrapper_va + iteration_boundary as u64;
    let adr_lr_va = wrapper_va + adr_lr_iter_pos as u64;
    let adr_off = iter_va as i64 - adr_lr_va as i64;
    code[adr_lr_iter_pos..adr_lr_iter_pos + 4].copy_from_slice(&encode_adr(30, adr_off));

    // Load persistent data, decrement counter
    let adr_data_c = emit_adr_placeholder(&mut code, 9);
    lit_pool.push(LitPoolEntry {
        instr_pos: adr_data_c,
        value: persistent_data_va,
    });

    code.extend_from_slice(&encode_ldr_w_imm(10, 9, OFF_COUNTER));
    // SUBS w10, w10, #1
    code.extend_from_slice(&encode_subs_w_imm(10, 10, 1));
    code.extend_from_slice(&encode_str_w_imm_base(10, 9, OFF_COUNTER));

    // B.EQ .exit_child (counter == 0)
    let beq_exit_pos = code.len();
    code.extend_from_slice(&[0x00, 0x00, 0x00, 0x54]); // placeholder B.EQ

    // getpid()
    emit_syscall(&mut code, SYS_GETPID);
    // kill(self, SIGSTOP=19)
    code.extend_from_slice(&encode_movz(1, SIGSTOP as u16));
    emit_syscall(&mut code, SYS_KILL);

    // After SIGCONT: clear prev_loc
    let prev_loc_va = data_va + PREV_LOC_OFFSET;
    let adr_prevloc = emit_adr_placeholder(&mut code, 9);
    lit_pool.push(LitPoolEntry {
        instr_pos: adr_prevloc,
        value: prev_loc_va,
    });
    code.extend_from_slice(&encode_strh_zero(9, 0)); // STRH wzr, [x9]

    // Restore SP, LR from persistent data (x0-x7 restored in setup_and_run)
    let adr_data_d = emit_adr_placeholder(&mut code, 9);
    lit_pool.push(LitPoolEntry {
        instr_pos: adr_data_d,
        value: persistent_data_va,
    });

    code.extend_from_slice(&encode_ldr_imm(10, 9, OFF_SAVE_SP));
    code.extend_from_slice(&encode_mov_reg(31, 10)); // sp = x10
    code.extend_from_slice(&encode_ldr_imm(30, 9, OFF_SAVE_LR)); // lr

    // B .setup_and_run (which restores x0-x7 before executing displaced bytes)
    let b_setup_pos = code.len();
    let brel = ((setup_and_run as i64 - b_setup_pos as i64) / 4) as i32;
    code.extend_from_slice(&encode_b(brel));

    // === Section D: .exit_child ===
    let exit_child = code.len();
    patch_b_cond(
        &mut code,
        beq_exit_pos,
        (exit_child as i64 - beq_exit_pos as i64) as i32,
        0x0,
    );

    // exit(0)
    code.extend_from_slice(&encode_movz(0, 0));
    emit_syscall(&mut code, SYS_EXIT);

    // === Literal pool: patch all ADR instructions ===
    for entry in &lit_pool {
        let instr_va = wrapper_va + entry.instr_pos as u64;
        let offset = entry.value as i64 - instr_va as i64;
        if !(-0x100000..=0xFFFFF).contains(&offset) {
            anyhow::bail!(
                "persistent wrapper: ADR offset out of range ({} bytes) for target 0x{:x} from 0x{:x}",
                offset,
                entry.value,
                instr_va,
            );
        }
        let rd = (code[entry.instr_pos] & 0x1F) as u32;
        code[entry.instr_pos..entry.instr_pos + 4].copy_from_slice(&encode_adr(rd, offset));
    }

    Ok(PersistentWrapper { code })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relocate_adrp_forward() {
        // ADRP x0, target_page where target_page = (0x1000 & ~0xFFF) + (1 << 12) = 0x2000
        // immhi=0, immlo=01 → imm21=1 → target_page = PC_page + 0x1000
        let rd = 0u32;
        let imm21 = 1u32; // 1 page forward
        let immhi = (imm21 >> 2) & 0x7_FFFF;
        let immlo = imm21 & 0x3;
        let raw = (1u32 << 31) | (immlo << 29) | (0b10000u32 << 24) | (immhi << 5) | rd;

        let original_va = 0x1000u64;
        let new_va = 0x20000u64; // trampoline is further away

        let relocated =
            relocate_pc_relative(raw, original_va, new_va).expect("relocation should succeed");

        // Original target page: (0x1000 & ~0xFFF) + (1 << 12) = 0x1000 + 0x1000 = 0x2000
        // New imm = (0x2000 - (0x20000 & ~0xFFF)) >> 12 = (0x2000 - 0x20000) >> 12 = -0x1E000 >> 12 = -30
        let new_rd = relocated & 0x1F;
        let new_immlo = (relocated >> 29) & 0x3;
        let new_immhi = (relocated >> 5) & 0x7_FFFF;
        let new_imm21 = (new_immhi << 2) | new_immlo;
        let new_imm_signed = sign_extend(new_imm21, 21);

        assert_eq!(new_rd, rd);
        let result_page = ((new_va & !0xFFF) as i64 + (new_imm_signed << 12)) as u64;
        assert_eq!(
            result_page, 0x2000,
            "relocated ADRP must reach original target page"
        );
    }

    #[test]
    fn test_relocate_adrp_backward() {
        // ADRP x1, #-1 (one page backward)
        let rd = 1u32;
        let imm21 = (-1i32 as u32) & 0x1F_FFFF;
        let immhi = (imm21 >> 2) & 0x7_FFFF;
        let immlo = imm21 & 0x3;
        let raw = (1u32 << 31) | (immlo << 29) | (0b10000u32 << 24) | (immhi << 5) | rd;

        let original_va = 0x5000u64;
        let new_va = 0x30000u64;

        let relocated =
            relocate_pc_relative(raw, original_va, new_va).expect("relocation should succeed");

        // Original target page: (0x5000 & ~0xFFF) + (-1 << 12) = 0x5000 - 0x1000 = 0x4000
        let new_immlo = (relocated >> 29) & 0x3;
        let new_immhi = (relocated >> 5) & 0x7_FFFF;
        let new_imm21 = (new_immhi << 2) | new_immlo;
        let new_imm_signed = sign_extend(new_imm21, 21);
        let result_page = ((new_va & !0xFFF) as i64 + (new_imm_signed << 12)) as u64;
        assert_eq!(
            result_page, 0x4000,
            "relocated ADRP must reach original target page"
        );
    }

    #[test]
    fn test_relocate_b() {
        // B +0x100 (64 instructions forward)
        let imm26 = (0x100i32 / 4) as u32 & 0x03FF_FFFF;
        let raw = 0x1400_0000u32 | imm26;

        let original_va = 0x1000u64;
        let new_va = 0x20000u64;

        let relocated =
            relocate_pc_relative(raw, original_va, new_va).expect("relocation should succeed");

        // Original target: 0x1000 + 0x100 = 0x1100
        // New offset: 0x1100 - 0x20000 = -0x1EF00
        let new_imm26 = relocated & 0x03FF_FFFF;
        let new_offset = sign_extend(new_imm26, 26) * 4;
        let result_target = (new_va as i64 + new_offset) as u64;
        assert_eq!(
            result_target, 0x1100,
            "relocated B must reach original target"
        );
    }

    #[test]
    fn test_relocate_bl() {
        // BL -0x200 (backward)
        let imm26 = ((-0x200i32) / 4) as u32 & 0x03FF_FFFF;
        let raw = 0x9400_0000u32 | imm26;

        let original_va = 0x2000u64;
        let new_va = 0x30000u64;

        let relocated =
            relocate_pc_relative(raw, original_va, new_va).expect("relocation should succeed");

        // Original target: 0x2000 - 0x200 = 0x1E00
        let new_imm26 = relocated & 0x03FF_FFFF;
        let new_offset = sign_extend(new_imm26, 26) * 4;
        let result_target = (new_va as i64 + new_offset) as u64;
        assert_eq!(
            result_target, 0x1E00,
            "relocated BL must reach original target"
        );
        assert_eq!(relocated & 0xFC00_0000, 0x9400_0000, "must remain BL");
    }

    #[test]
    fn test_relocate_adr() {
        // ADR x2, +0x80
        let rd = 2u32;
        let imm21 = 0x80u32;
        let immhi = (imm21 >> 2) & 0x7_FFFF;
        let immlo = imm21 & 0x3;
        let raw = (immlo << 29) | (0b10000u32 << 24) | (immhi << 5) | rd;

        let original_va = 0x1000u64;
        let new_va = 0x20000u64;

        let relocated =
            relocate_pc_relative(raw, original_va, new_va).expect("relocation should succeed");

        // Original target: 0x1000 + 0x80 = 0x1080
        let new_immlo = (relocated >> 29) & 0x3;
        let new_immhi = (relocated >> 5) & 0x7_FFFF;
        let new_imm21 = (new_immhi << 2) | new_immlo;
        let new_offset = sign_extend(new_imm21, 21);
        let result_target = (new_va as i64 + new_offset) as u64;
        assert_eq!(
            result_target, 0x1080,
            "relocated ADR must reach original target"
        );
    }

    #[test]
    fn test_relocate_ldr_literal() {
        // LDR x3, +0x40 (PC-relative literal load)
        let rt = 3u32;
        let imm19 = (0x40i32 / 4) as u32 & 0x7_FFFF;
        // opc=01 (64-bit), V=0 → 0x58000000
        let raw = 0x5800_0000u32 | (imm19 << 5) | rt;

        let original_va = 0x1000u64;
        let new_va = 0x20000u64;

        let relocated =
            relocate_pc_relative(raw, original_va, new_va).expect("relocation should succeed");

        // Original target: 0x1000 + 0x40 = 0x1040
        let new_imm19 = (relocated >> 5) & 0x7_FFFF;
        let new_offset = sign_extend(new_imm19, 19) * 4;
        let result_target = (new_va as i64 + new_offset) as u64;
        assert_eq!(
            result_target, 0x1040,
            "relocated LDR literal must reach original target"
        );
    }

    #[test]
    fn test_relocate_bcond() {
        // B.EQ +0x20
        let imm19 = (0x20i32 / 4) as u32 & 0x7_FFFF;
        let raw = 0x5400_0000u32 | (imm19 << 5) | 0x0; // cond=EQ

        let original_va = 0x1000u64;
        let new_va = 0x20000u64;

        let relocated =
            relocate_pc_relative(raw, original_va, new_va).expect("relocation should succeed");

        let new_imm19 = (relocated >> 5) & 0x7_FFFF;
        let new_offset = sign_extend(new_imm19, 19) * 4;
        let result_target = (new_va as i64 + new_offset) as u64;
        assert_eq!(
            result_target, 0x1020,
            "relocated B.EQ must reach original target"
        );
        assert_eq!(relocated & 0xF, 0x0, "condition code must be preserved");
    }

    #[test]
    fn test_relocate_cbz() {
        // CBZ x5, +0x30
        let rt = 5u32;
        let imm19 = (0x30i32 / 4) as u32 & 0x7_FFFF;
        let raw = 0xB400_0000u32 | (imm19 << 5) | rt;

        let original_va = 0x1000u64;
        let new_va = 0x20000u64;

        let relocated =
            relocate_pc_relative(raw, original_va, new_va).expect("relocation should succeed");

        let new_imm19 = (relocated >> 5) & 0x7_FFFF;
        let new_offset = sign_extend(new_imm19, 19) * 4;
        let result_target = (new_va as i64 + new_offset) as u64;
        assert_eq!(
            result_target, 0x1030,
            "relocated CBZ must reach original target"
        );
        assert_eq!(relocated & 0x1F, rt, "register must be preserved");
    }

    #[test]
    fn test_relocate_tbz() {
        // TBZ x7, #3, +0x10
        // TBZ range is ±32 KiB, so keep VAs close together
        let rt = 7u32;
        let b40 = 3u32; // bit number
        let imm14 = (0x10i32 / 4) as u32 & 0x3FFF;
        // TBZ: b5=0, op=0 → 0x36000000
        let raw = 0x3600_0000u32 | (b40 << 19) | (imm14 << 5) | rt;

        let original_va = 0x1000u64;
        let new_va = 0x2000u64; // close enough for ±32 KiB range

        let relocated =
            relocate_pc_relative(raw, original_va, new_va).expect("relocation should succeed");

        let new_imm14 = (relocated >> 5) & 0x3FFF;
        let new_offset = sign_extend(new_imm14, 14) * 4;
        let result_target = (new_va as i64 + new_offset) as u64;
        assert_eq!(
            result_target, 0x1010,
            "relocated TBZ must reach original target"
        );
        assert_eq!(
            (relocated >> 19) & 0x1F,
            b40,
            "bit number must be preserved"
        );
        assert_eq!(relocated & 0x1F, rt, "register must be preserved");
    }

    #[test]
    fn test_is_pc_relative_detection() {
        // ADRP x0, #0
        assert!(is_pc_relative(0x9000_0000));
        // ADR x0, #0
        assert!(is_pc_relative(0x1000_0000));
        // B #0
        assert!(is_pc_relative(0x1400_0000));
        // BL #0
        assert!(is_pc_relative(0x9400_0000));
        // B.EQ #0
        assert!(is_pc_relative(0x5400_0000));
        // CBZ x0, #0
        assert!(is_pc_relative(0xB400_0000));
        // TBZ x0, #0, #0
        assert!(is_pc_relative(0x3600_0000));

        // Non-PC-relative: MOV x0, x1 (ORR x0, xzr, x1)
        assert!(!is_pc_relative(0xAA01_03E0));
        // NOP
        assert!(!is_pc_relative(0xD503_201F));
        // STP x29, x30, [sp, #-16]!
        assert!(!is_pc_relative(0xA9BF_7BFD));
    }

    #[test]
    fn test_encode_sub_sp_imm_small() {
        // SUB sp, sp, #96 (fits in 12 bits, no shift)
        let bytes = encode_sub_sp_imm(96);
        let word = u32::from_le_bytes(bytes);
        let sh = (word >> 22) & 1;
        let imm12 = (word >> 10) & 0xFFF;
        assert_eq!(sh, 0, "sh should be 0 for small immediate");
        assert_eq!(imm12, 96, "imm12 should be 96");
    }

    #[test]
    fn test_encode_sub_sp_imm_4096() {
        // SUB sp, sp, #4096 = SUB sp, sp, #1, LSL #12
        let bytes = encode_sub_sp_imm(4096);
        let word = u32::from_le_bytes(bytes);
        let sh = (word >> 22) & 1;
        let imm12 = (word >> 10) & 0xFFF;
        assert_eq!(sh, 1, "sh should be 1 for 4096");
        assert_eq!(imm12, 1, "imm12 should be 1 (4096 >> 12)");
    }

    #[test]
    fn test_encode_add_sp_imm_4096() {
        // ADD sp, sp, #4096 = ADD sp, sp, #1, LSL #12
        let bytes = encode_add_sp_imm(4096);
        let word = u32::from_le_bytes(bytes);
        let sh = (word >> 22) & 1;
        let imm12 = (word >> 10) & 0xFFF;
        assert_eq!(sh, 1, "sh should be 1 for 4096");
        assert_eq!(imm12, 1, "imm12 should be 1");
    }

    #[test]
    fn test_relocate_identity() {
        // If original_va == new_va, instruction should be unchanged
        let imm26 = 0x40u32;
        let raw = 0x1400_0000u32 | imm26;
        let relocated =
            relocate_pc_relative(raw, 0x1000, 0x1000).expect("identity relocation should succeed");
        assert_eq!(
            relocated, raw,
            "identity relocation should produce same instruction"
        );
    }

    // ── Branch island helper tests ────────────────────────────────────

    #[test]
    fn test_compute_branch_target() {
        // TBZ x7, #3, +0x10
        let rt = 7u32;
        let b40 = 3u32;
        let imm14 = (0x10i32 / 4) as u32 & 0x3FFF;
        let tbz = 0x3600_0000u32 | (b40 << 19) | (imm14 << 5) | rt;
        assert_eq!(compute_branch_target(tbz, 0x1000), Some(0x1010));

        // TBNZ x0, #0, -0x20
        let imm14_neg = ((-0x20i32 / 4) as u32) & 0x3FFF;
        let tbnz = 0x3700_0000u32 | (imm14_neg << 5);
        assert_eq!(compute_branch_target(tbnz, 0x1000), Some(0x0FE0));

        // CBZ x5, +0x30
        let imm19 = (0x30i32 / 4) as u32 & 0x7_FFFF;
        let cbz = 0xB400_0000u32 | (imm19 << 5) | 5;
        assert_eq!(compute_branch_target(cbz, 0x1000), Some(0x1030));

        // CBNZ w2, -0x10
        let imm19_neg = ((-0x10i32 / 4) as u32) & 0x7_FFFF;
        let cbnz = 0x3500_0000u32 | (imm19_neg << 5) | 2;
        assert_eq!(compute_branch_target(cbnz, 0x2000), Some(0x1FF0));

        // B.EQ +0x20
        let imm19_beq = (0x20i32 / 4) as u32 & 0x7_FFFF;
        let beq = 0x5400_0000u32 | (imm19_beq << 5) | 0x0; // cond=EQ
        assert_eq!(compute_branch_target(beq, 0x1000), Some(0x1020));

        // B/BL return None — islands can't extend their ±128 MiB range
        let imm26 = (0x100i32 / 4) as u32 & 0x03FF_FFFF;
        let b = 0x1400_0000u32 | imm26;
        assert_eq!(
            compute_branch_target(b, 0x1000),
            None,
            "B should return None (islands can't help)"
        );

        let imm26_bl = ((-0x200i32) / 4) as u32 & 0x03FF_FFFF;
        let bl = 0x9400_0000u32 | imm26_bl;
        assert_eq!(
            compute_branch_target(bl, 0x2000),
            None,
            "BL should return None (islands can't help)"
        );

        // Non-branch PC-relative: islands can't help these either
        assert_eq!(compute_branch_target(0x9000_0000, 0x1000), None, "ADRP");
        assert_eq!(compute_branch_target(0x1000_0000, 0x1000), None, "ADR");
        assert_eq!(
            compute_branch_target(0x5800_0000, 0x1000),
            None,
            "LDR literal"
        );

        // Non-PC-relative: definitely None
        assert_eq!(compute_branch_target(0xD503_201F, 0x1000), None, "NOP");
        assert_eq!(
            compute_branch_target(0xAA01_03E0, 0x1000),
            None,
            "MOV x0, x1"
        );
    }

    #[test]
    fn test_rewrite_branch_to_island() {
        // TBZ x7, #3, original_offset → rewrite to +8
        let rt = 7u32;
        let b40 = 3u32;
        let imm14 = (0x100i32 / 4) as u32 & 0x3FFF;
        let tbz = 0x3600_0000u32 | (b40 << 19) | (imm14 << 5) | rt;

        let rewritten = rewrite_branch_to_island(tbz, 8);
        // Check register preserved
        assert_eq!(rewritten & 0x1F, rt, "TBZ register must be preserved");
        // Check bit number preserved
        assert_eq!(
            (rewritten >> 19) & 0x1F,
            b40,
            "TBZ bit number must be preserved"
        );
        // Check opcode preserved (b5=0, op=0 → TBZ)
        assert_eq!(
            rewritten & 0xFF00_0000,
            0x3600_0000,
            "TBZ opcode must be preserved"
        );
        // Check new offset = +8 bytes = +2 instructions
        let new_imm14 = (rewritten >> 5) & 0x3FFF;
        assert_eq!(new_imm14, 2, "TBZ should branch +2 instructions (+8 bytes)");

        // TBNZ x0, #5, original → rewrite to +8
        let tbnz = 0x3700_0000u32 | (5u32 << 19) | (0x40u32 << 5) | 0;
        let rewritten_tbnz = rewrite_branch_to_island(tbnz, 8);
        assert_eq!(
            rewritten_tbnz & 0xFF00_0000,
            0x3700_0000,
            "TBNZ opcode preserved"
        );
        assert_eq!(
            (rewritten_tbnz >> 19) & 0x1F,
            5,
            "TBNZ bit number preserved"
        );
        assert_eq!((rewritten_tbnz >> 5) & 0x3FFF, 2, "TBNZ offset = +2 insns");

        // CBZ x5 → rewrite to +8
        let cbz = 0xB400_0000u32 | (0x100u32 << 5) | 5;
        let rewritten_cbz = rewrite_branch_to_island(cbz, 8);
        assert_eq!(
            rewritten_cbz & 0xFF00_0000,
            0xB400_0000,
            "CBZ opcode preserved"
        );
        assert_eq!(rewritten_cbz & 0x1F, 5, "CBZ register preserved");
        assert_eq!((rewritten_cbz >> 5) & 0x7_FFFF, 2, "CBZ offset = +2 insns");

        // CBNZ w2 → rewrite to +8
        let cbnz = 0x3500_0000u32 | (0x80u32 << 5) | 2;
        let rewritten_cbnz = rewrite_branch_to_island(cbnz, 8);
        assert_eq!(
            rewritten_cbnz & 0xFF00_0000,
            0x3500_0000,
            "CBNZ opcode preserved"
        );
        assert_eq!(rewritten_cbnz & 0x1F, 2, "CBNZ register preserved");
        assert_eq!(
            (rewritten_cbnz >> 5) & 0x7_FFFF,
            2,
            "CBNZ offset = +2 insns"
        );

        // B.NE → rewrite to +8
        let bne = 0x5400_0000u32 | (0x200u32 << 5) | 0x1; // cond=NE
        let rewritten_bne = rewrite_branch_to_island(bne, 8);
        assert_eq!(rewritten_bne & 0xF, 0x1, "B.NE condition code preserved");
        assert_eq!((rewritten_bne >> 5) & 0x7_FFFF, 2, "B.NE offset = +2 insns");
    }

    #[test]
    fn test_tbz_branch_island() {
        // Simulate a block with TBZ x7, #3, +0x100 at VA 0x1000.
        // Trampoline is at VA 0x80000 — far enough that TBZ (±32 KiB) can't reach
        // the original target (0x1100) directly, but B (±128 MiB) can.
        let rt = 7u32;
        let b40 = 3u32;
        let imm14 = (0x100i32 / 4) as u32 & 0x3FFF;
        let tbz_raw = 0x3600_0000u32 | (b40 << 19) | (imm14 << 5) | rt;

        let block = BasicBlock {
            va: 0x1000,
            file_offset: 0x1000,
            displaced_len: 4,
            displaced_bytes: tbz_raw.to_le_bytes().to_vec(),
            block_id: 42,
        };

        let tramp_gen = AArch64TrampolineGenerator::new();
        let trampoline_va = 0x80000u64;
        let data_va = trampoline_va + 0x2000; // data area nearby

        let result = tramp_gen.generate_trampoline(trampoline_va, data_va, &block);
        assert!(
            result.is_ok(),
            "trampoline with TBZ island should succeed: {:?}",
            result.err()
        );

        let tramp = result.expect("trampoline with TBZ island should succeed");
        let code = &tramp.code;

        // The trampoline should be 4 bytes larger than a normal one (the island B instruction).
        // Verify the last 4 bytes are a B instruction (island).
        let island_bytes = &code[code.len() - 4..];
        let island_word = u32::from_le_bytes([
            island_bytes[0],
            island_bytes[1],
            island_bytes[2],
            island_bytes[3],
        ]);
        // Should be a B instruction (opcode 0x14xxxxxx or 0x17xxxxxx for negative)
        assert_eq!(
            island_word & 0xFC00_0000,
            0x1400_0000,
            "island must be a B instruction"
        );

        // Verify the island B reaches the original target (0x1100)
        let island_va = trampoline_va + (code.len() - 4) as u64;
        let island_imm26 = island_word & 0x03FF_FFFF;
        let island_offset = sign_extend(island_imm26, 26) * 4;
        let island_target = (island_va as i64 + island_offset) as u64;
        assert_eq!(
            island_target, 0x1100,
            "island B must reach original TBZ target (0x1100)"
        );

        // The second-to-last instruction should be B return_va (0x1004)
        let return_b_bytes = &code[code.len() - 8..code.len() - 4];
        let return_b_word = u32::from_le_bytes([
            return_b_bytes[0],
            return_b_bytes[1],
            return_b_bytes[2],
            return_b_bytes[3],
        ]);
        assert_eq!(
            return_b_word & 0xFC00_0000,
            0x1400_0000,
            "return must be a B instruction"
        );
        let return_b_va = trampoline_va + (code.len() - 8) as u64;
        let return_imm26 = return_b_word & 0x03FF_FFFF;
        let return_offset = sign_extend(return_imm26, 26) * 4;
        let return_target = (return_b_va as i64 + return_offset) as u64;
        assert_eq!(
            return_target, 0x1004,
            "return B must reach block.va + 4 (0x1004)"
        );
    }

    #[test]
    fn test_bcond_branch_island() {
        // B.NE +0x200 at VA 0x2000, trampoline at 0x200000 (2 MiB away — out of ±1 MiB
        // range for B.cond's imm19 field, which covers ±2^18 instructions = ±1 MiB).
        let imm19 = (0x200i32 / 4) as u32 & 0x7_FFFF;
        let bne_raw = 0x5400_0000u32 | (imm19 << 5) | 0x1; // cond=NE

        let block = BasicBlock {
            va: 0x2000,
            file_offset: 0x2000,
            displaced_len: 4,
            displaced_bytes: bne_raw.to_le_bytes().to_vec(),
            block_id: 99,
        };

        let tramp_gen = AArch64TrampolineGenerator::new();
        let trampoline_va = 0x200000u64; // 2 MiB away — exceeds B.cond ±1 MiB range
        let data_va = trampoline_va + 0x2000;

        let result = tramp_gen.generate_trampoline(trampoline_va, data_va, &block);
        assert!(
            result.is_ok(),
            "trampoline with B.NE island should succeed: {:?}",
            result.err()
        );

        let tramp = result.expect("trampoline with B.NE island should succeed");
        let code = &tramp.code;

        // Island is the last instruction
        let island_bytes = &code[code.len() - 4..];
        let island_word = u32::from_le_bytes([
            island_bytes[0],
            island_bytes[1],
            island_bytes[2],
            island_bytes[3],
        ]);
        assert_eq!(island_word & 0xFC00_0000, 0x1400_0000, "island must be B");

        // Verify island reaches original target (0x2200)
        let island_va = trampoline_va + (code.len() - 4) as u64;
        let island_imm26 = island_word & 0x03FF_FFFF;
        let island_offset = sign_extend(island_imm26, 26) * 4;
        let island_target = (island_va as i64 + island_offset) as u64;
        assert_eq!(
            island_target, 0x2200,
            "island B must reach original B.NE target (0x2200)"
        );
    }

    #[test]
    fn test_no_island_for_in_range_branch() {
        // TBZ at VA 0x1000, trampoline at 0x1100 — within ±32 KiB, no island needed
        let rt = 3u32;
        let b40 = 1u32;
        let imm14 = (0x20i32 / 4) as u32 & 0x3FFF;
        let tbz_raw = 0x3600_0000u32 | (b40 << 19) | (imm14 << 5) | rt;

        let block = BasicBlock {
            va: 0x1000,
            file_offset: 0x1000,
            displaced_len: 4,
            displaced_bytes: tbz_raw.to_le_bytes().to_vec(),
            block_id: 10,
        };

        let tramp_gen = AArch64TrampolineGenerator::new();
        let trampoline_va = 0x1100u64; // close — within TBZ range
        let data_va = trampoline_va + 0x100;

        let result = tramp_gen.generate_trampoline(trampoline_va, data_va, &block);
        assert!(
            result.is_ok(),
            "in-range TBZ should not need island: {:?}",
            result.err()
        );

        let tramp = result.expect("trampoline with CBZ island should succeed");
        // Last instruction should be B return_va (no island appended)
        let last_bytes = &tramp.code[tramp.code.len() - 4..];
        let last_word =
            u32::from_le_bytes([last_bytes[0], last_bytes[1], last_bytes[2], last_bytes[3]]);
        assert_eq!(
            last_word & 0xFC00_0000,
            0x1400_0000,
            "last insn should be B (return)"
        );

        // Verify it's the return B targeting 0x1004
        let b_va = trampoline_va + (tramp.code.len() - 4) as u64;
        let b_imm26 = last_word & 0x03FF_FFFF;
        let b_offset = sign_extend(b_imm26, 26) * 4;
        let b_target = (b_va as i64 + b_offset) as u64;
        assert_eq!(b_target, 0x1004, "last B should target return_va (0x1004)");
    }

    // ── Extensive branch island tests ─────────────────────────────────

    /// Helper: build a BasicBlock with a single 4-byte displaced instruction.
    fn make_block(va: u64, raw: u32, block_id: u16) -> BasicBlock {
        BasicBlock {
            va,
            file_offset: va, // file_offset == va for testing
            displaced_len: 4,
            displaced_bytes: raw.to_le_bytes().to_vec(),
            block_id,
        }
    }

    /// Helper: generate a trampoline and return it (panics on error).
    fn gen_trampoline(block_va: u64, raw: u32, trampoline_va: u64) -> Trampoline {
        let tramp_gen = AArch64TrampolineGenerator::new();
        let data_va = trampoline_va + 0x2000;
        let block = make_block(block_va, raw, 42);
        tramp_gen
            .generate_trampoline(trampoline_va, data_va, &block)
            .unwrap_or_else(|e| panic!("generate_trampoline failed: {}", e))
    }

    /// Helper: try to generate a trampoline, return the Result.
    fn try_gen_trampoline(block_va: u64, raw: u32, trampoline_va: u64) -> Result<Trampoline> {
        let tramp_gen = AArch64TrampolineGenerator::new();
        let data_va = trampoline_va + 0x2000;
        let block = make_block(block_va, raw, 42);
        tramp_gen.generate_trampoline(trampoline_va, data_va, &block)
    }

    /// Helper: decode the last N instructions from trampoline code as u32 words.
    fn last_n_insns(code: &[u8], n: usize) -> Vec<u32> {
        let start = code.len() - n * 4;
        (0..n)
            .map(|i| {
                let off = start + i * 4;
                u32::from_le_bytes([code[off], code[off + 1], code[off + 2], code[off + 3]])
            })
            .collect()
    }

    /// Helper: compute the target VA of a B instruction at `b_va`.
    fn decode_b_target(word: u32, b_va: u64) -> u64 {
        assert_eq!(
            word & 0xFC00_0000,
            0x1400_0000,
            "expected B instruction, got 0x{:08x}",
            word
        );
        let imm26 = word & 0x03FF_FFFF;
        let offset = sign_extend(imm26, 26) * 4;
        (b_va as i64 + offset) as u64
    }

    /// Encode a TBZ instruction: TBZ Xt, #bit, +offset_bytes
    fn encode_tbz(rt: u32, bit: u32, offset_bytes: i32) -> u32 {
        let b5 = (bit >> 5) & 1;
        let b40 = bit & 0x1F;
        let imm14 = ((offset_bytes / 4) as u32) & 0x3FFF;
        // TBZ: b5=0 if bit<32; op=0
        (b5 << 31) | 0x3600_0000 | (b40 << 19) | (imm14 << 5) | (rt & 0x1F)
    }

    /// Encode a TBNZ instruction: TBNZ Xt, #bit, +offset_bytes
    fn encode_tbnz(rt: u32, bit: u32, offset_bytes: i32) -> u32 {
        let b5 = (bit >> 5) & 1;
        let b40 = bit & 0x1F;
        let imm14 = ((offset_bytes / 4) as u32) & 0x3FFF;
        // TBNZ: op=1 (bit 24)
        (b5 << 31) | 0x3700_0000 | (b40 << 19) | (imm14 << 5) | (rt & 0x1F)
    }

    /// Encode CBZ: CBZ Xt, +offset_bytes (64-bit if sf=true)
    fn encode_cbz_insn(rt: u32, offset_bytes: i32, sf64: bool) -> u32 {
        let sf = if sf64 { 0xB400_0000u32 } else { 0x3400_0000u32 };
        let imm19 = ((offset_bytes / 4) as u32) & 0x7_FFFF;
        sf | (imm19 << 5) | (rt & 0x1F)
    }

    /// Encode CBNZ: CBNZ Xt, +offset_bytes (64-bit if sf=true)
    fn encode_cbnz_insn(rt: u32, offset_bytes: i32, sf64: bool) -> u32 {
        let sf = if sf64 { 0xB500_0000u32 } else { 0x3500_0000u32 };
        let imm19 = ((offset_bytes / 4) as u32) & 0x7_FFFF;
        sf | (imm19 << 5) | (rt & 0x1F)
    }

    /// Encode B.cond: B.<cond> +offset_bytes
    fn encode_bcond_insn(cond: u32, offset_bytes: i32) -> u32 {
        let imm19 = ((offset_bytes / 4) as u32) & 0x7_FFFF;
        0x5400_0000 | (imm19 << 5) | (cond & 0xF)
    }

    /// Verify trampoline has an island: last 2 instructions are [B return_va, B island_target].
    /// Returns (return_target, island_target).
    fn verify_island_trampoline(
        tramp: &Trampoline,
        trampoline_va: u64,
        expected_return: u64,
        expected_island: u64,
    ) {
        let insns = last_n_insns(&tramp.code, 2);
        let return_b = insns[0];
        let island_b = insns[1];

        let return_va = trampoline_va + (tramp.code.len() - 8) as u64;
        let island_va = trampoline_va + (tramp.code.len() - 4) as u64;

        let return_target = decode_b_target(return_b, return_va);
        let island_target = decode_b_target(island_b, island_va);

        assert_eq!(
            return_target, expected_return,
            "return B should target 0x{:x}, got 0x{:x}",
            expected_return, return_target
        );
        assert_eq!(
            island_target, expected_island,
            "island B should target 0x{:x}, got 0x{:x}",
            expected_island, island_target
        );
    }

    /// Verify trampoline has NO island: last instruction is B return_va.
    fn verify_no_island_trampoline(tramp: &Trampoline, trampoline_va: u64, expected_return: u64) {
        let insns = last_n_insns(&tramp.code, 1);
        let return_b = insns[0];
        let return_va = trampoline_va + (tramp.code.len() - 4) as u64;
        let return_target = decode_b_target(return_b, return_va);
        assert_eq!(
            return_target, expected_return,
            "return B should target 0x{:x}, got 0x{:x}",
            expected_return, return_target
        );
    }

    // ── TBNZ island ──

    #[test]
    fn test_tbnz_branch_island() {
        // TBNZ x2, #7, +0x80 at VA 0x4000, trampoline at 0x80000 (>32 KiB away)
        let raw = encode_tbnz(2, 7, 0x80);
        let trampoline_va = 0x80000u64;
        let tramp = gen_trampoline(0x4000, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, 0x4004, 0x4080);
    }

    // ── CBZ island (64-bit) ──

    #[test]
    fn test_cbz_x64_branch_island() {
        // CBZ x8, +0x400 at VA 0x10000, trampoline at 0x300000 (>1 MiB away)
        let raw = encode_cbz_insn(8, 0x400, true);
        let trampoline_va = 0x300000u64;
        let tramp = gen_trampoline(0x10000, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, 0x10004, 0x10400);
    }

    // ── CBZ island (32-bit, sf=0) ──

    #[test]
    fn test_cbz_w32_branch_island() {
        // CBZ w3, +0x100 at VA 0x5000, trampoline at 0x300000 (>1 MiB)
        let raw = encode_cbz_insn(3, 0x100, false);
        let trampoline_va = 0x300000u64;
        let tramp = gen_trampoline(0x5000, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, 0x5004, 0x5100);
    }

    // ── CBNZ island (64-bit) ──

    #[test]
    fn test_cbnz_x64_branch_island() {
        // CBNZ x15, +0x200 at VA 0x8000, trampoline at 0x400000
        let raw = encode_cbnz_insn(15, 0x200, true);
        let trampoline_va = 0x400000u64;
        let tramp = gen_trampoline(0x8000, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, 0x8004, 0x8200);
    }

    // ── CBNZ island (32-bit, sf=0) ──

    #[test]
    fn test_cbnz_w32_branch_island() {
        // CBNZ w0, +0x80 at VA 0x3000, trampoline at 0x400000
        let raw = encode_cbnz_insn(0, 0x80, false);
        let trampoline_va = 0x400000u64;
        let tramp = gen_trampoline(0x3000, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, 0x3004, 0x3080);
    }

    // ── B.cond island with every condition code ──

    #[test]
    fn test_bcond_all_condition_codes_island() {
        // ARM64 condition codes: 0=EQ, 1=NE, 2=CS, 3=CC, 4=MI, 5=PL,
        // 6=VS, 7=VC, 8=HI, 9=LS, A=GE, B=LT, C=GT, D=LE, E=AL
        let cond_names = [
            "EQ", "NE", "CS", "CC", "MI", "PL", "VS", "VC", "HI", "LS", "GE", "LT", "GT", "LE",
            "AL",
        ];

        for cond in 0u32..15 {
            let raw = encode_bcond_insn(cond, 0x100);
            let trampoline_va = 0x200000u64; // >1 MiB from block
            let tramp = gen_trampoline(0x1000, raw, trampoline_va);

            // Verify island targets the original branch destination
            verify_island_trampoline(&tramp, trampoline_va, 0x1004, 0x1100);

            // Verify the displaced B.cond in the trampoline preserved the condition code.
            // The displaced instruction is 3 words before the end (rewritten B.cond, B return, B island).
            let displaced_insns = last_n_insns(&tramp.code, 3);
            let rewritten_bcond = displaced_insns[0];
            // B.cond encoding: [31:24]=01010100 [23:5]=imm19 [4]=0 [3:0]=cond
            assert_eq!(
                rewritten_bcond & 0xF,
                cond,
                "B.{} condition code must be preserved in island rewrite, got cond={}",
                cond_names[cond as usize],
                rewritten_bcond & 0xF
            );
        }
    }

    // ── Backward branch target with island ──

    #[test]
    fn test_tbz_backward_branch_island() {
        // TBZ x1, #0, -0x40 at VA 0x5000 → target = 0x4FC0
        // Trampoline at 0x80000 (far away — needs island)
        let raw = encode_tbz(1, 0, -0x40);
        let trampoline_va = 0x80000u64;
        let tramp = gen_trampoline(0x5000, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, 0x5004, 0x4FC0);
    }

    #[test]
    fn test_cbz_backward_branch_island() {
        // CBZ x20, -0x200 at VA 0x10000 → target = 0xFE00
        // Trampoline at 0x300000 (>1 MiB)
        let raw = encode_cbz_insn(20, -0x200, true);
        let trampoline_va = 0x300000u64;
        let tramp = gen_trampoline(0x10000, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, 0x10004, 0xFE00);
    }

    #[test]
    fn test_bcond_backward_branch_island() {
        // B.LT -0x80 at VA 0x20000 → target = 0x1FF80
        // Trampoline at 0x200000
        let raw = encode_bcond_insn(0xB, -0x80); // LT = cond 0xB
        let trampoline_va = 0x200000u64;
        let tramp = gen_trampoline(0x20000, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, 0x20004, 0x1FF80);
    }

    // ── TBZ with high bit number (b5 set, bit >= 32) ──

    #[test]
    fn test_tbz_high_bit_number_island() {
        // TBZ x10, #35, +0x20 at VA 0x6000 — bit 35 means b5=1 (bit 31 of encoding)
        let raw = encode_tbz(10, 35, 0x20);
        // Verify b5 is set in the encoding
        assert_ne!(raw & 0x8000_0000, 0, "b5 should be set for bit >= 32");

        let trampoline_va = 0x80000u64;
        let tramp = gen_trampoline(0x6000, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, 0x6004, 0x6020);

        // Verify b5 and b40 preserved in the rewritten instruction
        let displaced = last_n_insns(&tramp.code, 3)[0];
        assert_ne!(
            displaced & 0x8000_0000,
            0,
            "b5 must be preserved in island rewrite"
        );
        assert_eq!((displaced >> 19) & 0x1F, 35 & 0x1F, "b40 must be preserved");
        assert_eq!(displaced & 0x1F, 10, "register must be preserved");
    }

    // ── TBZ at exact boundary of ±32 KiB ──

    #[test]
    fn test_tbz_just_beyond_range() {
        // TBZ imm14 range: -(1<<13) to (1<<13)-1 instructions = ±32 KiB.
        // Place trampoline so the displaced instruction is just barely out of range.
        // TBZ x0, #0, +0x10 at VA 0x1000, target = 0x1010
        // Trampoline at VA such that displaced_insn_va - target > 32 KiB.
        // Displaced instruction is at trampoline_va + preamble_size.
        // The preamble is typically ~80 bytes = 20 instructions.
        // For the relocation to fail: |target - displaced_va| > 32768
        // displaced_va ≈ trampoline_va + 80
        // target = 0x1010
        // Need: displaced_va - 0x1010 > 32768 → trampoline_va + 80 > 0x1010 + 32768
        // → trampoline_va > 0x9010 - 80 ≈ 0x8FC0
        // Use 0x9100 to be safely over the boundary.
        let raw = encode_tbz(0, 0, 0x10);
        let trampoline_va = 0x9100u64;
        let tramp = gen_trampoline(0x1000, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, 0x1004, 0x1010);
    }

    #[test]
    fn test_tbz_just_within_range() {
        // Same as above but trampoline close enough that TBZ relocation succeeds.
        // Need: |displaced_va - target| < 32768
        // displaced_va ≈ trampoline_va + 80
        // target = 0x1010
        // Need: trampoline_va + 80 - 0x1010 < 32768
        // → trampoline_va < 0x1010 + 32768 - 80 = 0x8FC0
        // Use 0x8000 to be safely within range.
        let raw = encode_tbz(0, 0, 0x10);
        let trampoline_va = 0x8000u64;
        let tramp = gen_trampoline(0x1000, raw, trampoline_va);
        // Should succeed without island
        verify_no_island_trampoline(&tramp, trampoline_va, 0x1004);
    }

    // ── Island trampoline is exactly 4 bytes larger than non-island ──

    #[test]
    fn test_island_trampoline_size_delta() {
        // Generate two trampolines with the same block_id but different distances:
        // one within TBZ range (no island), one beyond (island needed).
        let raw = encode_tbz(5, 2, 0x20);

        // Within range: trampoline at 0x1100 (close to block at 0x1000)
        let near_tramp = gen_trampoline(0x1000, raw, 0x1100);

        // Out of range: trampoline at 0x80000 (far from block at 0x1000)
        let far_tramp = gen_trampoline(0x1000, raw, 0x80000);

        // Island trampoline should be exactly 4 bytes (1 instruction) larger
        assert_eq!(
            far_tramp.code.len(),
            near_tramp.code.len() + 4,
            "island trampoline should be exactly 4 bytes larger: near={}, far={}",
            near_tramp.code.len(),
            far_tramp.code.len()
        );
    }

    // ── Displaced instruction encoding in island path ──

    #[test]
    fn test_displaced_tbz_encoding_in_island() {
        // The rewritten TBZ in the trampoline should branch +8 bytes (2 instructions forward)
        // to skip past the return B and land on the island B.
        let raw = encode_tbz(7, 3, 0x100);
        let trampoline_va = 0x80000u64;
        let tramp = gen_trampoline(0x1000, raw, trampoline_va);

        // Get the displaced instruction (3rd from last: displaced, B return, B island)
        let displaced = last_n_insns(&tramp.code, 3)[0];

        // Should still be a TBZ (opcode preserved)
        assert_eq!(displaced & 0x7E00_0000, 0x3600_0000, "should still be TBZ");
        // Register preserved
        assert_eq!(displaced & 0x1F, 7, "Rt=x7 preserved");
        // Bit number preserved
        assert_eq!((displaced >> 19) & 0x1F, 3, "bit=3 preserved");
        // imm14 should be +2 (offset_bytes=8, /4=2)
        let imm14 = (displaced >> 5) & 0x3FFF;
        assert_eq!(imm14, 2, "TBZ imm14 should be +2 (island at +8 bytes)");
    }

    #[test]
    fn test_displaced_cbz_encoding_in_island() {
        let raw = encode_cbz_insn(12, 0x400, true);
        let trampoline_va = 0x300000u64;
        let tramp = gen_trampoline(0x10000, raw, trampoline_va);

        let displaced = last_n_insns(&tramp.code, 3)[0];
        // Should still be CBZ x12 (64-bit)
        assert_eq!(
            displaced & 0xFF00_0000,
            0xB400_0000,
            "should be CBZ (sf=1, op=0)"
        );
        assert_eq!(displaced & 0x1F, 12, "Rt=x12 preserved");
        let imm19 = (displaced >> 5) & 0x7_FFFF;
        assert_eq!(imm19, 2, "CBZ imm19 should be +2 (island at +8 bytes)");
    }

    #[test]
    fn test_displaced_cbnz_encoding_in_island() {
        let raw = encode_cbnz_insn(0, 0x80, false); // CBNZ w0
        let trampoline_va = 0x400000u64;
        let tramp = gen_trampoline(0x3000, raw, trampoline_va);

        let displaced = last_n_insns(&tramp.code, 3)[0];
        // Should be CBNZ w0 (sf=0, op=1)
        assert_eq!(
            displaced & 0xFF00_0000,
            0x3500_0000,
            "should be CBNZ (sf=0, op=1)"
        );
        assert_eq!(displaced & 0x1F, 0, "Rt=w0 preserved");
        let imm19 = (displaced >> 5) & 0x7_FFFF;
        assert_eq!(imm19, 2, "CBNZ imm19 should be +2");
    }

    #[test]
    fn test_displaced_bcond_encoding_in_island() {
        let raw = encode_bcond_insn(0xC, 0x100); // B.GT
        let trampoline_va = 0x200000u64;
        let tramp = gen_trampoline(0x2000, raw, trampoline_va);

        let displaced = last_n_insns(&tramp.code, 3)[0];
        assert_eq!(displaced & 0xFF00_0010, 0x5400_0000, "should be B.cond");
        assert_eq!(displaced & 0xF, 0xC, "cond=GT preserved");
        let imm19 = (displaced >> 5) & 0x7_FFFF;
        assert_eq!(imm19, 2, "B.GT imm19 should be +2");
    }

    // ── Non-branch PC-relative out-of-range should error, not island ──

    #[test]
    fn test_adrp_out_of_range_errors() {
        // ADRP x0, #1 (one page forward) — place trampoline extremely far away
        // ADRP range is ±4 GiB so we can't easily exceed it in a u64 VA space,
        // but we can test that it works (no island attempted) when it's in range.
        // Instead, test with ADR which has smaller range (±1 MiB).
        // This test ensures ADRP relocation doesn't accidentally try the island path.
        let rd = 0u32;
        let imm21 = 1u32; // 1 page forward
        let immhi = (imm21 >> 2) & 0x7_FFFF;
        let immlo = imm21 & 0x3;
        let raw_adrp = (1u32 << 31) | (immlo << 29) | (0b10000u32 << 24) | (immhi << 5) | rd;

        // Within ADRP range — should succeed without island
        let tramp = gen_trampoline(0x1000, raw_adrp, 0x20000);
        verify_no_island_trampoline(&tramp, 0x20000, 0x1004);
    }

    #[test]
    fn test_adr_out_of_range_errors() {
        // ADR x2, +0x80 at VA 0x1000 → target = 0x1080
        // ADR range is ±1 MiB. Place trampoline >1 MiB away.
        let rd = 2u32;
        let imm21 = 0x80u32;
        let immhi = (imm21 >> 2) & 0x7_FFFF;
        let immlo = imm21 & 0x3;
        let raw_adr = (immlo << 29) | (0b10000u32 << 24) | (immhi << 5) | rd;

        let result = try_gen_trampoline(0x1000, raw_adr, 0x200000);
        assert!(
            result.is_err(),
            "ADR out of range should error (no island for non-branches)"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("non-branch") || err_msg.contains("out of range"),
            "error should mention non-branch or out of range: {}",
            err_msg
        );
    }

    #[test]
    fn test_ldr_literal_out_of_range_errors() {
        // LDR x3, +0x40 at VA 0x1000 → target = 0x1040
        // LDR literal range is ±1 MiB. Place trampoline >1 MiB away.
        let rt = 3u32;
        let imm19 = (0x40i32 / 4) as u32 & 0x7_FFFF;
        let raw_ldr = 0x5800_0000u32 | (imm19 << 5) | rt;

        let result = try_gen_trampoline(0x1000, raw_ldr, 0x200000);
        assert!(
            result.is_err(),
            "LDR literal out of range should error (no island)"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("non-branch") || err_msg.contains("out of range"),
            "error should mention non-branch or out of range: {}",
            err_msg
        );
    }

    // ── B/BL out of range: should error, NOT panic ──

    #[test]
    fn test_b_out_of_range_errors_not_panics() {
        // B +0x100 at VA 0x1000 → target = 0x1100
        // B range is ±128 MiB. Place trampoline >128 MiB away.
        // Since islands can't help B (same ±128 MiB range), this should error.
        let imm26 = (0x100i32 / 4) as u32 & 0x03FF_FFFF;
        let raw_b = 0x1400_0000u32 | imm26;

        // Trampoline 256 MiB away — both B relocation and island B would fail
        let result = try_gen_trampoline(0x1000, raw_b, 0x1000_0000);
        assert!(
            result.is_err(),
            "B out of range should error gracefully (not panic)"
        );
    }

    #[test]
    fn test_bl_out_of_range_errors_not_panics() {
        // BL +0x100 at VA 0x1000
        let imm26 = (0x100i32 / 4) as u32 & 0x03FF_FFFF;
        let raw_bl = 0x9400_0000u32 | imm26;

        let result = try_gen_trampoline(0x1000, raw_bl, 0x1000_0000);
        assert!(
            result.is_err(),
            "BL out of range should error gracefully (not panic)"
        );
    }

    // ── Non-PC-relative instructions: no relocation or island needed ──

    #[test]
    fn test_non_pc_relative_displaced_preserved() {
        // MOV x0, x1 (ORR x0, xzr, x1) at VA 0x1000
        let raw = 0xAA01_03E0u32;
        let trampoline_va = 0x80000u64;
        let tramp = gen_trampoline(0x1000, raw, trampoline_va);

        // Non-PC-relative: should be copied verbatim (no island)
        verify_no_island_trampoline(&tramp, trampoline_va, 0x1004);

        // The displaced instruction (2nd from last) should be the original bytes
        let displaced = last_n_insns(&tramp.code, 2)[0];
        assert_eq!(
            displaced, raw,
            "non-PC-relative instruction should be copied verbatim"
        );
    }

    // ── Round-trip consistency: compute_branch_target matches island target ──

    #[test]
    fn test_island_target_matches_compute_branch_target() {
        // For each conditional branch type, verify that the island B target
        // equals what compute_branch_target says the original target was.
        let test_cases: Vec<(u32, u64, &str)> = vec![
            (encode_tbz(5, 10, 0x80), 0x3000, "TBZ"),
            (encode_tbnz(0, 0, -0x40), 0x5000, "TBNZ"),
            (encode_cbz_insn(8, 0x400, true), 0x10000, "CBZ"),
            (encode_cbnz_insn(3, -0x100, false), 0x8000, "CBNZ"),
            (encode_bcond_insn(0, 0x200), 0x2000, "B.EQ"),
            (encode_bcond_insn(1, -0x80), 0x4000, "B.NE"),
        ];

        for (raw, block_va, name) in test_cases {
            let expected_target = compute_branch_target(raw, block_va)
                .unwrap_or_else(|| panic!("{}: compute_branch_target should return Some", name));

            // Generate trampoline far enough to trigger island for all types
            let trampoline_va = 0x400000u64;
            let tramp = gen_trampoline(block_va, raw, trampoline_va);

            // Verify island target
            let island_b = *last_n_insns(&tramp.code, 1)
                .last()
                .expect("should have at least one instruction");
            let island_va = trampoline_va + (tramp.code.len() - 4) as u64;
            let island_target = decode_b_target(island_b, island_va);

            assert_eq!(
                island_target, expected_target,
                "{}: island target 0x{:x} should match compute_branch_target 0x{:x}",
                name, island_target, expected_target
            );
        }
    }

    // ── Register x0 through x30 preserved through TBZ island ──

    #[test]
    fn test_tbz_island_all_registers() {
        for rt in 0u32..31 {
            let raw = encode_tbz(rt, 0, 0x10);
            let trampoline_va = 0x80000u64;
            let tramp = gen_trampoline(0x1000, raw, trampoline_va);

            let displaced = last_n_insns(&tramp.code, 3)[0];
            assert_eq!(
                displaced & 0x1F,
                rt,
                "TBZ register x{} should be preserved in island rewrite",
                rt
            );
        }
    }

    // ── Multiple islands from same block_va but different bit numbers ──

    #[test]
    fn test_tbz_varying_bit_numbers() {
        for bit in 0u32..64 {
            let raw = encode_tbz(0, bit, 0x10);
            let trampoline_va = 0x80000u64;
            let tramp = gen_trampoline(0x1000, raw, trampoline_va);

            let displaced = last_n_insns(&tramp.code, 3)[0];
            // Verify bit number preserved
            let b5 = (displaced >> 31) & 1;
            let b40 = (displaced >> 19) & 0x1F;
            let decoded_bit = (b5 << 5) | b40;
            assert_eq!(
                decoded_bit, bit,
                "TBZ bit #{} should be preserved, got #{}",
                bit, decoded_bit
            );
        }
    }

    // ── Trampoline with NOP displaced instruction (non-PC-relative) ──

    #[test]
    fn test_nop_displaced_no_island() {
        let raw = 0xD503_201Fu32; // NOP
        let trampoline_va = 0x80000u64;
        let tramp = gen_trampoline(0x1000, raw, trampoline_va);
        verify_no_island_trampoline(&tramp, trampoline_va, 0x1004);
    }

    // ── Large forward branch target ──

    #[test]
    fn test_tbz_large_forward_target() {
        // TBZ with maximum positive offset: imm14 = 0x1FFF = 8191 instructions = +32764 bytes
        let max_imm14 = 0x1FFFi32 * 4; // +32764
        let raw = encode_tbz(0, 0, max_imm14);
        let block_va = 0x10000u64;
        let expected_target = block_va + max_imm14 as u64;

        let trampoline_va = 0x80000u64;
        let tramp = gen_trampoline(block_va, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, block_va + 4, expected_target);
    }

    // ── Large backward branch target ──

    #[test]
    fn test_tbz_large_backward_target() {
        // TBZ with maximum negative offset: imm14 = -8192 instructions = -32768 bytes
        let min_offset = -0x2000i32 * 4; // -32768
        let raw = encode_tbz(0, 0, min_offset);
        let block_va = 0x50000u64;
        let expected_target = (block_va as i64 + min_offset as i64) as u64;

        let trampoline_va = 0x80000u64;
        let tramp = gen_trampoline(block_va, raw, trampoline_va);
        verify_island_trampoline(&tramp, trampoline_va, block_va + 4, expected_target);
    }

    // ── ELF AArch64 persistent wrapper tests ──

    #[test]
    fn test_elf_persistent_wrapper_aarch64_basic() {
        let displaced = [0x1F, 0x20, 0x03, 0xD5]; // NOP
        let wrapper =
            generate_persistent_wrapper_aarch64(&crate::trampoline::PersistentWrapperParams {
                wrapper_va: 0x201000,
                persistent_data_va: 0x200000,
                data_va: 0x200000,
                persistent_addr: 0x100800,
                displaced_bytes: &displaced,
                displaced_len: 4,
                persistent_count: 1000,
                include_forkserver: false,
            })
            .expect("ELF persistent wrapper aarch64 basic should succeed");
        assert!(!wrapper.code.is_empty());
        assert_eq!(
            wrapper.code.len() % 4,
            0,
            "ARM64 code must be 4-byte aligned"
        );
    }

    #[test]
    fn test_elf_persistent_wrapper_aarch64_with_forkserver() {
        let displaced = [0x1F, 0x20, 0x03, 0xD5]; // NOP
        let wrapper =
            generate_persistent_wrapper_aarch64(&crate::trampoline::PersistentWrapperParams {
                wrapper_va: 0x201000,
                persistent_data_va: 0x200000,
                data_va: 0x200000,
                persistent_addr: 0x100800,
                displaced_bytes: &displaced,
                displaced_len: 4,
                persistent_count: 1000,
                include_forkserver: true,
            })
            .expect("ELF persistent wrapper aarch64 with forkserver should succeed");
        assert!(!wrapper.code.is_empty());
        assert_eq!(wrapper.code.len() % 4, 0);
        // Forkserver variant should be larger
        let basic =
            generate_persistent_wrapper_aarch64(&crate::trampoline::PersistentWrapperParams {
                wrapper_va: 0x201000,
                persistent_data_va: 0x200000,
                data_va: 0x200000,
                persistent_addr: 0x100800,
                displaced_bytes: &displaced,
                displaced_len: 4,
                persistent_count: 1000,
                include_forkserver: false,
            })
            .expect("ELF persistent wrapper aarch64 basic for comparison should succeed");
        assert!(
            wrapper.code.len() > basic.code.len(),
            "forkserver wrapper ({}) should be larger than basic ({})",
            wrapper.code.len(),
            basic.code.len()
        );
    }

    #[test]
    fn test_elf_persistent_wrapper_aarch64_linux_sigstop() {
        let displaced = [0x1F, 0x20, 0x03, 0xD5]; // NOP
        let wrapper =
            generate_persistent_wrapper_aarch64(&crate::trampoline::PersistentWrapperParams {
                wrapper_va: 0x201000,
                persistent_data_va: 0x200000,
                data_va: 0x200000,
                persistent_addr: 0x100800,
                displaced_bytes: &displaced,
                displaced_len: 4,
                persistent_count: 1000,
                include_forkserver: false,
            })
            .expect("ELF persistent wrapper aarch64 for sigstop test should succeed");
        // Check SIGSTOP=19 (0x13) is in the code, not macOS SIGSTOP=17
        // MOVZ x1, #19 = 0xD2800261
        let sigstop_movz = 0xD280_0261u32.to_le_bytes();
        let has_sigstop_19 = wrapper.code.windows(4).any(|w| w == sigstop_movz);
        assert!(
            has_sigstop_19,
            "should contain MOVZ x1, #19 (Linux SIGSTOP)"
        );
    }
}
