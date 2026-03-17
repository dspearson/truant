//! Shared AArch64 instruction encoding helpers.
//!
//! Provides hand-assembled ARM64 machine code encoding for all instructions
//! used by the coverage trampoline generator and hook trampoline generator.
//! Each function produces a fixed-width 4-byte (or 8-byte for wide pairs)
//! little-endian instruction encoding.
//!
//! Centralised here to eliminate duplication and ensure fixes (e.g. range
//! guards, masking) apply everywhere.

// ============================================================
// Branch / control-flow instructions
// ============================================================

/// Encode an unconditional B instruction.
/// `offset_instructions`: signed 26-bit instruction offset (byte_delta / 4).
#[inline]
pub(crate) fn encode_b(offset_instructions: i32) -> [u8; 4] {
    let word = 0x1400_0000u32 | ((offset_instructions as u32) & 0x03FF_FFFF);
    word.to_le_bytes()
}

/// Encode BR xN (branch to register).
#[inline]
pub(crate) fn encode_br(reg: u32) -> [u8; 4] {
    let word = 0xD61F_0000u32 | (reg << 5);
    word.to_le_bytes()
}

/// Encode BLR xN (branch-and-link to register).
#[inline]
pub(crate) fn encode_blr(reg: u32) -> [u8; 4] {
    let word = 0xD63F_0000u32 | (reg << 5);
    word.to_le_bytes()
}

/// Encode RET (return, uses x30).
#[inline]
pub(crate) fn encode_ret() -> [u8; 4] {
    0xD65F_03C0u32.to_le_bytes()
}

/// Encode CBZ xN, #offset (compare and branch if zero; offset in bytes).
#[inline]
pub(crate) fn encode_cbz(xn: u32, offset_bytes: i32) -> [u8; 4] {
    let imm19 = ((offset_bytes / 4) as u32) & 0x7FFFF;
    let word = 0xB400_0000u32 | (imm19 << 5) | xn;
    word.to_le_bytes()
}

/// Encode CBNZ xN, #offset (compare and branch if non-zero; offset in bytes).
#[inline]
pub(crate) fn encode_cbnz(xn: u32, offset_bytes: i32) -> [u8; 4] {
    let imm19 = ((offset_bytes / 4) as u32) & 0x7FFFF;
    (0xB500_0000u32 | (imm19 << 5) | xn).to_le_bytes()
}

/// Encode B.cond with a 19-bit signed instruction offset.
/// `cond`: 4-bit condition code (0=EQ, 1=NE, ...).
#[inline]
pub(crate) fn encode_bcond(cond: u32, offset_instructions: i32) -> [u8; 4] {
    let imm19 = (offset_instructions as u32) & 0x7_FFFF;
    let word = 0x5400_0000 | (imm19 << 5) | cond;
    word.to_le_bytes()
}

// ============================================================
// Load/store pair instructions
// ============================================================

/// Encode STP xN, xM, [sp, #-16]! (pre-index push of two registers).
#[inline]
pub(crate) fn encode_stp_push(rn: u32, rm: u32) -> [u8; 4] {
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

/// Encode LDP xN, xM, [sp], #16 (post-index pop of two registers).
#[inline]
pub(crate) fn encode_ldp_pop(rn: u32, rm: u32) -> [u8; 4] {
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

/// Encode STP xN, xM, [sp, #offset] (signed offset, no writeback).
/// `offset` is in bytes, must be multiple of 8, range -512..504.
#[inline]
pub(crate) fn encode_stp_offset(rn: u32, rm: u32, offset: i32) -> [u8; 4] {
    let imm7 = ((offset / 8) as u32) & 0x7F;
    let word = (0b10u32 << 30)
        | (0b101u32 << 27)
        | (0b010u32 << 23) // signed offset, no writeback
        | (imm7 << 15)
        | (rm << 10)
        | (31u32 << 5)
        | rn;
    word.to_le_bytes()
}

/// Encode LDP xN, xM, [sp, #offset] (signed offset, no writeback).
/// `offset` is in bytes, must be multiple of 8, range -512..504.
#[inline]
pub(crate) fn encode_ldp_offset(rn: u32, rm: u32, offset: i32) -> [u8; 4] {
    let imm7 = ((offset / 8) as u32) & 0x7F;
    let word = (0b10u32 << 30)
        | (0b101u32 << 27)
        | (0b010u32 << 23) // signed offset
        | (1u32 << 22)     // L=1 (load)
        | (imm7 << 15)
        | (rm << 10)
        | (31u32 << 5)
        | rn;
    word.to_le_bytes()
}

// ============================================================
// Move instructions
// ============================================================

/// Encode MOVZ xN, #imm16 (zero-extend, shift=0).
#[inline]
pub(crate) fn encode_movz(reg: u32, imm16: u16) -> [u8; 4] {
    let word = 0xD280_0000u32 | ((imm16 as u32) << 5) | reg;
    word.to_le_bytes()
}

/// Encode MOVK xN, #imm16, LSL #shift (0, 16, 32, 48).
#[inline]
pub(crate) fn encode_movk(reg: u32, imm16: u16, lsl: u32) -> [u8; 4] {
    let hw = lsl / 16;
    let word = 0xF280_0000u32 | (hw << 21) | ((imm16 as u32) << 5) | reg;
    word.to_le_bytes()
}

/// Encode MOV xDst, xSrc (register copy, via ORR xDst, xzr, xSrc).
#[inline]
pub(crate) fn encode_mov_reg(dst: u32, src: u32) -> [u8; 4] {
    let word = 0xAA00_03E0u32 | (src << 16) | dst;
    word.to_le_bytes()
}

// ============================================================
// System instructions
// ============================================================

/// Encode SVC #0 (supervisor call / syscall).
#[inline]
pub(crate) fn encode_svc0() -> [u8; 4] {
    0xD400_0001u32.to_le_bytes()
}

/// Encode NOP.
#[inline]
pub(crate) fn encode_nop() -> [u8; 4] {
    0xD503_201Fu32.to_le_bytes()
}

// ============================================================
// Load/store (register and immediate offset)
// ============================================================

/// Encode STR xN, [xM] (store 64-bit register, zero offset).
#[inline]
pub(crate) fn encode_str_reg(xn: u32, xm: u32) -> [u8; 4] {
    let word = 0xF900_0000u32 | (xm << 5) | xn;
    word.to_le_bytes()
}

/// Encode LDR xN, [xM] (load 64-bit register, zero offset).
#[inline]
pub(crate) fn encode_ldr_reg(xn: u32, xm: u32) -> [u8; 4] {
    let word = 0xF940_0000u32 | (xm << 5) | xn;
    word.to_le_bytes()
}

/// Encode LDRH wN, [xM, #offset] (load halfword, unsigned offset, multiple of 2).
#[inline]
pub(crate) fn encode_ldrh_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 2) & 0xFFF;
    let word = 0x7940_0000u32 | (imm12 << 10) | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode STRH wN, [xM, #offset] (store halfword, unsigned offset, multiple of 2).
#[inline]
pub(crate) fn encode_strh_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 2) & 0xFFF;
    let word = 0x7900_0000u32 | (imm12 << 10) | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode STRH wzr, [xM, #offset] (store zero halfword).
#[inline]
pub(crate) fn encode_strh_zero(xm: u32, offset: u32) -> [u8; 4] {
    encode_strh_imm(31, xm, offset)
}

/// Encode STR xN, [sp, #offset] (store to stack, unsigned immediate, multiple of 8).
#[inline]
pub(crate) fn encode_str_sp_imm(xn: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 8) & 0xFFF;
    let word = 0xF900_03E0u32 | (imm12 << 10) | xn;
    word.to_le_bytes()
}

/// Encode LDR xN, [sp, #offset] (load from stack, unsigned immediate, multiple of 8).
#[inline]
pub(crate) fn encode_ldr_sp_imm(xn: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 8) & 0xFFF;
    let word = 0xF940_03E0u32 | (imm12 << 10) | xn;
    word.to_le_bytes()
}

/// Encode LDR xN, [xM, #offset] (64-bit load, unsigned immediate, multiple of 8).
#[inline]
pub(crate) fn encode_ldr_imm(xn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 8) & 0xFFF;
    let word = 0xF940_0000u32 | (imm12 << 10) | (xm << 5) | xn;
    word.to_le_bytes()
}

/// Encode STR xN, [xM, #offset] (64-bit store, unsigned immediate, multiple of 8).
#[inline]
pub(crate) fn encode_str_imm_base(xn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 8) & 0xFFF;
    (0xF900_0000u32 | (imm12 << 10) | (xm << 5) | xn).to_le_bytes()
}

/// Encode LDR wN, [xM, #offset] (32-bit load, unsigned immediate, multiple of 4).
#[inline]
pub(crate) fn encode_ldr_w_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 4) & 0xFFF;
    let word = 0xB940_0000u32 | (imm12 << 10) | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode STR wN, [xM, #offset] (32-bit store, unsigned immediate, multiple of 4).
#[inline]
pub(crate) fn encode_str_w_imm_base(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 4) & 0xFFF;
    (0xB900_0000u32 | (imm12 << 10) | (xm << 5) | wn).to_le_bytes()
}

/// Encode LDRB wN, [xM] (load byte, zero offset).
#[inline]
pub(crate) fn encode_ldrb_reg(wn: u32, xm: u32) -> [u8; 4] {
    let word = 0x3940_0000u32 | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode LDRB wN, [xM, #imm] (load byte, unsigned immediate offset).
#[inline]
pub(crate) fn encode_ldrb_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = offset & 0xFFF;
    let word = 0x3940_0000u32 | (imm12 << 10) | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode LDRB wN, [xM], #1 (post-index load byte, advances xM by 1).
#[inline]
pub(crate) fn encode_ldrb_post_inc(wn: u32, xm: u32) -> [u8; 4] {
    let word = 0x3840_1400u32 | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode STRB wN, [xM] (store byte, zero offset).
#[inline]
pub(crate) fn encode_strb_reg(wn: u32, xm: u32) -> [u8; 4] {
    let word = 0x3900_0000u32 | (xm << 5) | wn;
    word.to_le_bytes()
}

/// Encode STRB wN, [xM, #imm] (store byte, unsigned immediate offset).
#[inline]
pub(crate) fn encode_strb_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
    let imm12 = offset & 0xFFF;
    (0x3900_0000u32 | (imm12 << 10) | (xm << 5) | wn).to_le_bytes()
}

/// Encode LDR xN, PC+offset (PC-relative literal load, ±1 MiB).
#[inline]
pub(crate) fn encode_ldr_literal(xn: u32, offset_bytes: i32) -> [u8; 4] {
    let imm19 = ((offset_bytes / 4) as u32) & 0x7FFFF;
    let word = 0x5800_0000u32 | (imm19 << 5) | xn;
    word.to_le_bytes()
}

// ============================================================
// Arithmetic / logic instructions
// ============================================================

/// Encode ADD wDst, wSrc, #1 (32-bit, no flags).
#[inline]
pub(crate) fn encode_add_w_imm1(wd: u32, wn: u32) -> [u8; 4] {
    let word = 0x1100_0400u32 | (wn << 5) | wd;
    word.to_le_bytes()
}

/// Encode ADD xDst, xN, xM (64-bit, shifted register, LSL #0).
#[inline]
pub(crate) fn encode_add_reg(xd: u32, xn: u32, xm: u32) -> [u8; 4] {
    let word = 0x8B00_0000u32 | (xm << 16) | (xn << 5) | xd;
    word.to_le_bytes()
}

/// Encode ADD xD, xN, #imm12 (64-bit immediate, no shift).
#[inline]
pub(crate) fn encode_add_imm(rd: u32, rn: u32, imm12: u32) -> [u8; 4] {
    let word = 0x9100_0000u32 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd;
    word.to_le_bytes()
}

/// Encode ADD sp, sp, #imm (with optional LSL #12 for multiples of 4096).
#[inline]
pub(crate) fn encode_add_sp_imm(imm: u32) -> [u8; 4] {
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

/// Encode ADD xD, sp, #imm (read sp into register with immediate offset).
#[inline]
pub(crate) fn encode_add_from_sp(xd: u32, imm: u32) -> [u8; 4] {
    let word = 0x9100_03E0u32 | ((imm & 0xFFF) << 10) | xd;
    word.to_le_bytes()
}

/// Encode ADD xD, xN, wM, UXTW (zero-extend 32-bit wM to 64-bit).
#[inline]
pub(crate) fn encode_add_uxtw(xd: u32, xn: u32, wm: u32) -> [u8; 4] {
    let word = 0x8B20_0000u32 | (wm << 16) | (xn << 5) | xd;
    word.to_le_bytes()
}

/// Encode SUB sp, sp, #imm (with optional LSL #12 for multiples of 4096).
#[inline]
pub(crate) fn encode_sub_sp_imm(imm: u32) -> [u8; 4] {
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

/// Encode SUB wD, wN, #imm (32-bit subtract immediate).
#[inline]
pub(crate) fn encode_sub_w_imm(wd: u32, wn: u32, imm: u32) -> [u8; 4] {
    let word = 0x5100_0000u32 | ((imm & 0xFFF) << 10) | (wn << 5) | wd;
    word.to_le_bytes()
}

/// Encode EOR xDst, xN, xM (XOR 64-bit shifted register).
#[inline]
pub(crate) fn encode_eor_reg(xd: u32, xn: u32, xm: u32) -> [u8; 4] {
    let word = 0xCA00_0000u32 | (xm << 16) | (xn << 5) | xd;
    word.to_le_bytes()
}

/// Encode AND xDst, xN, #0xFFFF (64-bit bitmask immediate).
#[inline]
pub(crate) fn encode_and_0xffff(xd: u32, xn: u32) -> [u8; 4] {
    let word = 0x9240_0000u32 | (15u32 << 10) | (xn << 5) | xd;
    word.to_le_bytes()
}

/// Encode MUL xD, xN, xM (64-bit multiply, via MADD xD, xN, xM, xzr).
#[inline]
pub(crate) fn encode_mul(xd: u32, xn: u32, xm: u32) -> [u8; 4] {
    let word = 0x9B00_7C00u32 | (xm << 16) | (xn << 5) | xd;
    word.to_le_bytes()
}

// ============================================================
// Compare instructions
// ============================================================

/// Encode CMP xN, #0 (via SUBS xzr, xN, #0).
#[inline]
pub(crate) fn encode_cmp_imm0(xn: u32) -> [u8; 4] {
    let word = 0xF100_001Fu32 | (xn << 5);
    word.to_le_bytes()
}

/// Encode CMP xN, xM (64-bit register compare, via SUBS xzr, xN, xM).
#[inline]
pub(crate) fn encode_cmp_reg(xn: u32, xm: u32) -> [u8; 4] {
    let word = 0xEB00_001Fu32 | (xm << 16) | (xn << 5);
    word.to_le_bytes()
}

/// Encode CMP wN, #imm (32-bit compare immediate, via SUBS wzr, wN, #imm).
#[inline]
pub(crate) fn encode_cmp_w_imm(wn: u32, imm: u32) -> [u8; 4] {
    let word = 0x7100_001Fu32 | ((imm & 0xFFF) << 10) | (wn << 5);
    word.to_le_bytes()
}

/// Encode CMP wN, wM (32-bit register compare, via SUBS wzr, wN, wM).
#[inline]
pub(crate) fn encode_cmp_w_reg(wn: u32, wm: u32) -> [u8; 4] {
    let word = 0x6B00_001Fu32 | (wm << 16) | (wn << 5);
    word.to_le_bytes()
}

/// Encode SUBS wD, wN, #imm (subtract immediate, setting flags, 32-bit).
#[inline]
pub(crate) fn encode_subs_w_imm(wd: u32, wn: u32, imm: u32) -> [u8; 4] {
    (0x7100_0000u32 | ((imm & 0xFFF) << 10) | (wn << 5) | wd).to_le_bytes()
}

// ============================================================
// PC-relative address instructions
// ============================================================

/// Encode ADR xN, PC+byte_offset (±1 MiB range, 21-bit signed).
#[inline]
pub(crate) fn encode_adr(rd: u32, byte_offset: i64) -> [u8; 4] {
    debug_assert!(
        byte_offset >= -(1 << 20) && byte_offset < (1 << 20),
        "ADR offset out of ±1 MiB range: {:#x} — use emit_adr_auto() instead",
        byte_offset
    );
    let imm21 = (byte_offset & 0x1FFFFF) as u32;
    let immlo = imm21 & 0x3;
    let immhi = (imm21 >> 2) & 0x7FFFF;
    let word = (immlo << 29) | (0b10000u32 << 24) | (immhi << 5) | rd;
    word.to_le_bytes()
}

/// Encode ADRP xN, PC+page_offset (±4 GiB range, page-aligned).
#[inline]
pub(crate) fn encode_adrp(rd: u32, byte_offset: i64) -> [u8; 4] {
    let page_offset = byte_offset >> 12;
    let imm21 = (page_offset & 0x1FFFFF) as u32;
    let immlo = imm21 & 0x3;
    let immhi = (imm21 >> 2) & 0x7FFFF;
    let word = (1u32 << 31) | (immlo << 29) | (0b10000u32 << 24) | (immhi << 5) | rd;
    word.to_le_bytes()
}

/// Wide PC-relative address computation: ADRP + ADD (±4 GiB range, 8 bytes).
/// `pc` must be the VA of the ADRP instruction itself.
#[inline]
pub(crate) fn encode_adr_wide(rd: u32, pc: u64, target: u64) -> [u8; 8] {
    let pc_page = pc & !0xFFF;
    let target_page = target & !0xFFF;
    let page_offset = target_page as i64 - pc_page as i64;
    let page_off12 = (target & 0xFFF) as u32;
    let adrp = encode_adrp(rd, page_offset);
    let add = encode_add_imm(rd, rd, page_off12);
    let mut result = [0u8; 8];
    result[..4].copy_from_slice(&adrp);
    result[4..].copy_from_slice(&add);
    result
}

/// Returns true if `byte_offset` fits in ADR's ±1 MiB range.
#[inline]
pub(crate) fn adr_in_range(byte_offset: i64) -> bool {
    byte_offset >= -(1 << 20) && byte_offset < (1 << 20)
}

/// Emit ADR (4 bytes) or ADRP+ADD (8 bytes) depending on range.
pub(crate) fn emit_adr_auto(code: &mut Vec<u8>, rd: u32, pc: u64, target: u64) {
    let offset = target as i64 - pc as i64;
    if adr_in_range(offset) {
        code.extend_from_slice(&encode_adr(rd, offset));
    } else {
        code.extend_from_slice(&encode_adr_wide(rd, pc, target));
    }
}

// ============================================================
// Patch helpers (fix up placeholders in emitted code)
// ============================================================

/// Patch a B.cond instruction at `pos` with the given byte delta and condition code.
pub(crate) fn patch_b_cond(code: &mut [u8], pos: usize, delta_bytes: i32, cond: u32) {
    let imm19 = ((delta_bytes / 4) as u32) & 0x7FFFF;
    let word = 0x5400_0000u32 | (imm19 << 5) | cond;
    code[pos..pos + 4].copy_from_slice(&word.to_le_bytes());
}

/// Patch an LDR literal placeholder at `instr_pos` to load register `xn` from `lit_pos`.
pub(crate) fn patch_ldr_literal(
    code: &mut [u8],
    instr_pos: usize,
    base_va: u64,
    lit_pos: usize,
    xn: u32,
) {
    let instr_va = base_va + instr_pos as u64;
    let lit_va = base_va + lit_pos as u64;
    let delta = (lit_va as i64 - instr_va as i64) as i32;
    let encoded = encode_ldr_literal(xn, delta);
    code[instr_pos..instr_pos + 4].copy_from_slice(&encoded);
}

/// Patch an 8-byte placeholder (two NOPs) with ADR+NOP or ADRP+ADD.
pub(crate) fn patch_adr_auto(code: &mut [u8], pos: usize, pc: u64, target: u64, rd: u32) {
    let offset = target as i64 - pc as i64;
    if adr_in_range(offset) {
        code[pos..pos + 4].copy_from_slice(&encode_adr(rd, offset));
        code[pos + 4..pos + 8].copy_from_slice(&encode_nop());
    } else {
        code[pos..pos + 8].copy_from_slice(&encode_adr_wide(rd, pc, target));
    }
}

/// Patch an ADR placeholder at `instr_pos` to compute the address of `target_pos`.
pub(crate) fn patch_adr(code: &mut [u8], instr_pos: usize, target_pos: usize, rd: u32) {
    let delta = target_pos as i64 - instr_pos as i64;
    let encoded = encode_adr(rd, delta);
    code[instr_pos..instr_pos + 4].copy_from_slice(&encoded);
}

// ============================================================
// Multi-instruction emission helpers
// ============================================================

/// Load a 32-bit unsigned immediate into register xN (MOVZ, optionally + MOVK).
pub(crate) fn emit_mov32(code: &mut Vec<u8>, xn: u32, val: u32) {
    let lo = (val & 0xFFFF) as u16;
    let hi = ((val >> 16) & 0xFFFF) as u16;
    code.extend_from_slice(&encode_movz(xn, lo));
    if hi != 0 {
        code.extend_from_slice(&encode_movk(xn, hi, 16));
    }
}

/// Load a 64-bit immediate into register xN (MOVZ + up to 3 MOVK).
pub(crate) fn emit_mov64(code: &mut Vec<u8>, xn: u32, val: u64) {
    let w0 = (val & 0xFFFF) as u16;
    let w1 = ((val >> 16) & 0xFFFF) as u16;
    let w2 = ((val >> 32) & 0xFFFF) as u16;
    let w3 = ((val >> 48) & 0xFFFF) as u16;

    code.extend_from_slice(&encode_movz(xn, w0));
    if w1 != 0 {
        code.extend_from_slice(&encode_movk(xn, w1, 16));
    }
    if w2 != 0 {
        code.extend_from_slice(&encode_movk(xn, w2, 32));
    }
    if w3 != 0 {
        code.extend_from_slice(&encode_movk(xn, w3, 48));
    }
}

/// Emit a syscall stub: load syscall number into x8, then SVC #0.
pub(crate) fn emit_syscall(code: &mut Vec<u8>, syscall_nr: u32) {
    emit_mov32(code, 8, syscall_nr);
    code.extend_from_slice(&encode_svc0());
}
