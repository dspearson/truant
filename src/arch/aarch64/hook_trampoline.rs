//! AArch64 hook trampoline generation.
//!
//! Generates trampolines for pre-hook, post-hook, and replace-hook modes on ARM64.
//! Each trampoline saves/restores all GPRs (x0-x30) + SP + PC + NZCV via a
//! RegContext struct (272 bytes), calls the hook handler, then transfers control
//! back to the original code.
//!
//! AArch64 differences from x86_64:
//! - No red zone (AAPCS64 does not define one)
//! - 16-byte stack alignment required at all times
//! - Fixed 4-byte instruction width → displaced_len is always 4
//! - STP/LDP paired save/restore (2 registers at a time)
//! - BLR for indirect calls, B for direct branches
//! - NZCV preserved via MRS/MSR (system register S3_3_C4_C2_0)

use anyhow::{Context, Result};

use crate::hooks::{CondOp, HookCondition, HookMode, HookSource, ResolvedHook};
use crate::trampoline::Trampoline;

/// Size of AArch64 RegContext in bytes.
/// x0-x30 (31 regs) + sp + pc + nzcv = 34 × 8 = 272.
const REG_CONTEXT_SIZE: u32 = 272;

/// Generate an AArch64 hook trampoline for the given resolved hook.
pub fn generate_hook_trampoline(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
) -> Result<Trampoline> {
    match hook.mode {
        HookMode::Pre => {
            generate_pre_hook(trampoline_va, hook, hook_data_va, shellcode_va, toggle_va)
        }
        HookMode::Post => {
            generate_post_hook(trampoline_va, hook, hook_data_va, shellcode_va, toggle_va)
        }
        HookMode::Replace => {
            generate_replace_hook(trampoline_va, hook, hook_data_va, shellcode_va, toggle_va)
        }
        HookMode::Return => anyhow::bail!(
            "Return mode hooks must use generate_return_hook_trampolines() (hook at 0x{:x})",
            hook.target_va,
        ),
    }
}

/// Generate AArch64 return hook trampolines: entry + return.
///
/// **Entry trampoline**:
///   1. Save x16 (scratch) onto stack
///   2. ADR x16 to data slot, STR x30 (LR) to slot (save original RA)
///   3. ADR x30 to return trampoline VA (overwrite LR)
///   4. Restore x16
///   5. Relocated displaced instruction
///   6. B return_va (continue function body)
///
/// **Return trampoline**:
///   1. Allocate RegContext + save regs
///   2. [Toggle/condition checks]
///   3. MOV x0, sp; BLR handler
///   4. Restore regs + deallocate
///   5. Load saved original LR from data slot into x16
///   6. BR x16 (return to original caller)
pub fn generate_return_hook_trampolines(
    entry_va: u64,
    ret_tramp_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
    return_slot_va: u64,
) -> Result<(Trampoline, Trampoline)> {
    let return_va = hook.target_va + hook.displaced_len as u64;

    // === Entry trampoline ===
    let mut entry = Vec::with_capacity(64);

    // Save scratch register x16 to stack.
    // NOTE: must NOT use pre-index writeback `STR x16, [sp, #-16]!` — macOS
    // ARM64 runtime exit cleanup (dyld/libsystem) crashes with SIGBUS when
    // a segment without compact unwind info contains SP-writeback stores.
    // Use separate SUB sp + STR instead.
    // SUB sp, sp, #16 = D100_43FF
    entry.extend_from_slice(&0xD100_43FFu32.to_le_bytes());
    // STR x16, [sp] = F900_03F0
    entry.extend_from_slice(&0xF900_03F0u32.to_le_bytes());

    // ADR x16, data_slot — PC-relative address of return address slot
    let adr1_va = entry_va + entry.len() as u64;
    let adr1_offset = return_slot_va as i64 - adr1_va as i64;
    entry.extend_from_slice(&encode_adr(16, adr1_offset));

    // STR x30, [x16] — save original LR to data slot
    // STR x30, [x16, #0] = F900_021E
    entry.extend_from_slice(&0xF900_021Eu32.to_le_bytes());

    // ADR x30, ret_tramp_va — set LR to return trampoline
    let adr2_va = entry_va + entry.len() as u64;
    let adr2_offset = ret_tramp_va as i64 - adr2_va as i64;
    entry.extend_from_slice(&encode_adr(30, adr2_offset));

    // Restore scratch register x16 from stack.
    // LDR x16, [sp] = F940_03F0
    entry.extend_from_slice(&0xF940_03F0u32.to_le_bytes());
    // ADD sp, sp, #16 = 9100_43FF
    entry.extend_from_slice(&0x9100_43FFu32.to_le_bytes());

    // Relocated displaced instruction(s)
    let displaced_va = entry_va + entry.len() as u64;
    let relocated = relocate_displaced_aarch64(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate displaced instructions for return hook entry at 0x{:x}",
                hook.target_va,
            )
        })?;
    entry.extend_from_slice(&relocated);

    // B return_va (continue function body)
    let b_va = entry_va + entry.len() as u64;
    let delta = return_va as i64 - b_va as i64;
    if delta % 4 != 0 || delta.abs() > 128 * 1024 * 1024 {
        anyhow::bail!(
            "return hook entry at 0x{:x}: body branch out of range (delta={} bytes)",
            hook.target_va,
            delta,
        );
    }
    entry.extend_from_slice(&encode_b((delta / 4) as i32));

    let entry_tramp = Trampoline {
        va: entry_va,
        code: entry,
    };

    // === Return trampoline ===
    let mut ret_code = Vec::with_capacity(512);

    // 0. Save FP/SIMD return-value registers d0-d7 (64 bytes).
    // The handler call can clobber these, but the hooked function's return
    // value lives in d0 (float/double results per AAPCS64).
    // SUB sp, sp, #64
    ret_code.extend_from_slice(&encode_sub_sp(64));
    // STP d0,d1,[sp,#0]; STP d2,d3,[sp,#16]; STP d4,d5,[sp,#32]; STP d6,d7,[sp,#48]
    ret_code.extend_from_slice(&0x6D0007E0u32.to_le_bytes()); // STP d0, d1, [sp, #0]
    ret_code.extend_from_slice(&0x6D010FE2u32.to_le_bytes()); // STP d2, d3, [sp, #16]
    ret_code.extend_from_slice(&0x6D0217E4u32.to_le_bytes()); // STP d4, d5, [sp, #32]
    ret_code.extend_from_slice(&0x6D031FE6u32.to_le_bytes()); // STP d6, d7, [sp, #48]

    // 1. Allocate RegContext
    ret_code.extend_from_slice(&encode_sub_sp(REG_CONTEXT_SIZE));

    // 2. Save all registers
    emit_save_regs(&mut ret_code, hook.target_va);

    // 2a. Toggle check (if present).
    let toggle_fixup =
        toggle_va.map(|tva| emit_toggle_check_aarch64(&mut ret_code, ret_tramp_va, tva));

    // 2b. Condition check (if present).
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_aarch64(&mut ret_code, cond)?)
    } else {
        None
    };

    // 3. MOV x0, sp (arg1 = &RegContext)
    ret_code.extend_from_slice(&encode_mov_from_sp(0));

    // 4. Call hook handler
    emit_hook_call(
        &mut ret_code,
        ret_tramp_va,
        &hook.source,
        hook_data_va,
        shellcode_va,
    );

    // 4b. Fixup condition skip target.
    if let Some(fixup_pos) = cond_fixup {
        let target = ret_code.len();
        fixup_condition_skip_aarch64(&mut ret_code, fixup_pos, target);
    }

    // 4c. Fixup toggle skip target.
    if let Some(fixup_pos) = toggle_fixup {
        let target = ret_code.len();
        fixup_toggle_skip_aarch64(&mut ret_code, fixup_pos, target);
    }

    // 5. Restore all registers
    emit_restore_regs(&mut ret_code);

    // 6. Deallocate RegContext
    ret_code.extend_from_slice(&encode_add_sp(REG_CONTEXT_SIZE));

    // 6b. Restore FP/SIMD registers d0-d7.
    // LDP d0,d1,[sp,#0]; LDP d2,d3,[sp,#16]; LDP d4,d5,[sp,#32]; LDP d6,d7,[sp,#48]
    ret_code.extend_from_slice(&0x6D4007E0u32.to_le_bytes()); // LDP d0, d1, [sp, #0]
    ret_code.extend_from_slice(&0x6D410FE2u32.to_le_bytes()); // LDP d2, d3, [sp, #16]
    ret_code.extend_from_slice(&0x6D4217E4u32.to_le_bytes()); // LDP d4, d5, [sp, #32]
    ret_code.extend_from_slice(&0x6D431FE6u32.to_le_bytes()); // LDP d6, d7, [sp, #48]
    // ADD sp, sp, #64
    ret_code.extend_from_slice(&encode_add_sp(64));

    // 7. Load saved original LR from data slot into x16, BR x16
    let adr3_va = ret_tramp_va + ret_code.len() as u64;
    let adr3_offset = return_slot_va as i64 - adr3_va as i64;
    ret_code.extend_from_slice(&encode_adr(16, adr3_offset));
    ret_code.extend_from_slice(&encode_ldr_reg(16, 16)); // LDR x16, [x16]
    // BR x16: 0xD61F_0200
    ret_code.extend_from_slice(&0xD61F_0200u32.to_le_bytes());

    let ret_tramp = Trampoline {
        va: ret_tramp_va,
        code: ret_code,
    };

    Ok((entry_tramp, ret_tramp))
}

/// Generate a chained AArch64 hook trampoline for multiple hooks on the same target VA.
///
/// All hooks must target the same VA and have the same mode (validated by caller).
pub fn generate_chained_hook_trampoline(
    trampoline_va: u64,
    hooks: &[&ResolvedHook],
    hook_data_va: u64,
    shellcode_vas: &[Option<u64>],
    toggle_vas: &[Option<u64>],
) -> Result<Trampoline> {
    let hook0 = hooks[0];
    let mode = hook0.mode;
    let return_va = hook0.target_va + hook0.displaced_len as u64;
    let mut code = Vec::with_capacity(512);

    // For post mode, displaced instructions come first.
    if mode == HookMode::Post {
        let displaced_va = trampoline_va + code.len() as u64;
        let relocated =
            relocate_displaced_aarch64(&hook0.displaced_bytes, hook0.target_va, displaced_va)
                .context("failed to relocate displaced instructions for chained hook")?;
        code.extend_from_slice(&relocated);
    }

    // 1. Allocate RegContext
    code.extend_from_slice(&encode_sub_sp(REG_CONTEXT_SIZE));

    // 2. Save all registers
    emit_save_regs(&mut code, hook0.target_va);

    // For replace mode, ADR x1 to original stub (fixup later).
    let adr_x1_fixup_pos = if mode == HookMode::Replace {
        let pos = code.len();
        code.extend_from_slice(&encode_adr(1, 0)); // placeholder
        Some(pos)
    } else {
        None
    };

    // 3. Call each hook handler in sequence.
    for (i, hook) in hooks.iter().enumerate() {
        // Per-hook toggle check (if present).
        let toggle_fixup =
            toggle_vas[i].map(|tva| emit_toggle_check_aarch64(&mut code, trampoline_va, tva));

        // Per-hook condition check (if present).
        let per_hook_fixup = if let Some(ref cond) = hook.condition {
            Some(emit_condition_check_aarch64(&mut code, cond)?)
        } else {
            None
        };

        // Refresh x0 = sp (arg1 = &RegContext) before each call.
        code.extend_from_slice(&encode_mov_from_sp(0));

        // Call this hook's handler
        emit_hook_call(
            &mut code,
            trampoline_va,
            &hook.source,
            hook_data_va,
            shellcode_vas[i],
        );

        // Fixup per-hook condition skip.
        if let Some(fixup_pos) = per_hook_fixup {
            let target = code.len();
            fixup_condition_skip_aarch64(&mut code, fixup_pos, target);
        }

        // Fixup toggle skip.
        if let Some(fixup_pos) = toggle_fixup {
            let target = code.len();
            fixup_toggle_skip_aarch64(&mut code, fixup_pos, target);
        }
    }

    // 4. Restore all registers
    emit_restore_regs(&mut code);

    // 5. Deallocate RegContext
    code.extend_from_slice(&encode_add_sp(REG_CONTEXT_SIZE));

    match mode {
        HookMode::Pre => {
            // Displaced instructions + B return_va
            let displaced_va = trampoline_va + code.len() as u64;
            let relocated =
                relocate_displaced_aarch64(&hook0.displaced_bytes, hook0.target_va, displaced_va)
                    .context("failed to relocate displaced instructions for chained pre-hook")?;
            code.extend_from_slice(&relocated);

            let b_va = trampoline_va + code.len() as u64;
            let delta = return_va as i64 - b_va as i64;
            if delta % 4 != 0 || delta.abs() > 128 * 1024 * 1024 {
                anyhow::bail!("chained hook: return branch out of range");
            }
            code.extend_from_slice(&encode_b((delta / 4) as i32));
        }
        HookMode::Post => {
            let b_va = trampoline_va + code.len() as u64;
            let delta = return_va as i64 - b_va as i64;
            if delta % 4 != 0 || delta.abs() > 128 * 1024 * 1024 {
                anyhow::bail!("chained hook: return branch out of range");
            }
            code.extend_from_slice(&encode_b((delta / 4) as i32));
        }
        HookMode::Replace => {
            // RET to caller
            code.extend_from_slice(&[0xC0, 0x03, 0x5F, 0xD6]);

            // Original function stub
            let stub_offset = code.len();
            let stub_va = trampoline_va + stub_offset as u64;

            if let Some(fixup_pos) = adr_x1_fixup_pos {
                let adr_va = trampoline_va + fixup_pos as u64;
                let adr_offset = stub_va as i64 - adr_va as i64;
                let adr_word = encode_adr(1, adr_offset);
                code[fixup_pos..fixup_pos + 4].copy_from_slice(&adr_word);
            }

            let displaced_va = trampoline_va + code.len() as u64;
            let relocated =
                relocate_displaced_aarch64(&hook0.displaced_bytes, hook0.target_va, displaced_va)
                    .context("failed to relocate displaced instructions for chained replace hook")?;
            code.extend_from_slice(&relocated);

            let b2_va = trampoline_va + code.len() as u64;
            let delta2 = return_va as i64 - b2_va as i64;
            if delta2 % 4 != 0 || delta2.abs() > 128 * 1024 * 1024 {
                anyhow::bail!("chained hook: stub return branch out of range");
            }
            code.extend_from_slice(&encode_b((delta2 / 4) as i32));
        }
        // Return mode is rejected before this match is reached.
        _ => unreachable!(),
    }

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

// ============================================================
// ARM64 instruction encoding helpers (reused from trampoline_gen.rs patterns)
// ============================================================

/// Encode B (unconditional branch, ±128 MiB).
#[inline]
fn encode_b(offset_instructions: i32) -> [u8; 4] {
    let word = 0x1400_0000u32 | ((offset_instructions as u32) & 0x03FF_FFFF);
    word.to_le_bytes()
}

/// Encode BLR xN (branch-and-link to register).
#[inline]
fn encode_blr(reg: u32) -> [u8; 4] {
    let word = 0xD63F_0000u32 | (reg << 5);
    word.to_le_bytes()
}

/// Encode STP xN, xM, [sp, #offset] (signed offset, no writeback).
#[inline]
fn encode_stp_offset(rn: u32, rm: u32, offset: i32) -> [u8; 4] {
    let imm7 = ((offset / 8) as u32) & 0x7F;
    let word = (0b10u32 << 30)
        | (0b101u32 << 27)
        | (0b010u32 << 23) // signed offset
        | (imm7 << 15)
        | (rm << 10)
        | (31u32 << 5) // sp
        | rn;
    word.to_le_bytes()
}

/// Encode LDP xN, xM, [sp, #offset] (signed offset, no writeback).
#[inline]
fn encode_ldp_offset(rn: u32, rm: u32, offset: i32) -> [u8; 4] {
    let imm7 = ((offset / 8) as u32) & 0x7F;
    let word = (0b10u32 << 30)
        | (0b101u32 << 27)
        | (0b010u32 << 23) // signed offset
        | (1u32 << 22)     // L=1 (load)
        | (imm7 << 15)
        | (rm << 10)
        | (31u32 << 5) // sp
        | rn;
    word.to_le_bytes()
}

/// Encode STR xN, [sp, #offset] (unsigned immediate, 8-byte aligned).
#[inline]
fn encode_str_sp(xn: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 8) & 0xFFF;
    let word = 0xF900_03E0u32 | (imm12 << 10) | xn;
    word.to_le_bytes()
}

/// Encode LDR xN, [sp, #offset] (unsigned immediate, 8-byte aligned).
#[inline]
fn encode_ldr_sp(xn: u32, offset: u32) -> [u8; 4] {
    let imm12 = (offset / 8) & 0xFFF;
    let word = 0xF940_03E0u32 | (imm12 << 10) | xn;
    word.to_le_bytes()
}

/// Encode SUB sp, sp, #imm.
#[inline]
fn encode_sub_sp(imm: u32) -> [u8; 4] {
    let (sh, imm12) = if imm <= 0xFFF {
        (0u32, imm)
    } else if imm & 0xFFF == 0 && (imm >> 12) <= 0xFFF {
        (1u32, imm >> 12)
    } else {
        panic!("SUB sp, sp, #{imm} cannot be encoded");
    };
    let word = 0xD100_03FFu32 | (sh << 22) | ((imm12 & 0xFFF) << 10);
    word.to_le_bytes()
}

/// Encode ADD sp, sp, #imm.
#[inline]
fn encode_add_sp(imm: u32) -> [u8; 4] {
    let (sh, imm12) = if imm <= 0xFFF {
        (0u32, imm)
    } else if imm & 0xFFF == 0 && (imm >> 12) <= 0xFFF {
        (1u32, imm >> 12)
    } else {
        panic!("ADD sp, sp, #{imm} cannot be encoded");
    };
    let word = 0x9100_03FFu32 | (sh << 22) | ((imm12 & 0xFFF) << 10);
    word.to_le_bytes()
}

/// Encode ADD xD, sp, #imm (read sp into register).
#[inline]
fn encode_add_from_sp(xd: u32, imm: u32) -> [u8; 4] {
    // ADD xD, sp, #imm: sf=1, op=0, S=0, Rn=31(sp), Rd=xD
    let word = 0x9100_03E0u32 | ((imm & 0xFFF) << 10) | xd;
    word.to_le_bytes()
}

/// Encode MOV xD, sp (via ADD xD, sp, #0).
#[inline]
fn encode_mov_from_sp(xd: u32) -> [u8; 4] {
    encode_add_from_sp(xd, 0)
}

/// Encode MOVZ xN, #imm16.
#[inline]
fn encode_movz(reg: u32, imm16: u16) -> [u8; 4] {
    let word = 0xD280_0000u32 | ((imm16 as u32) << 5) | reg;
    word.to_le_bytes()
}

/// Encode MOVK xN, #imm16, LSL #shift.
#[inline]
fn encode_movk(reg: u32, imm16: u16, lsl: u32) -> [u8; 4] {
    let hw = lsl / 16;
    let word = 0xF280_0000u32 | (hw << 21) | ((imm16 as u32) << 5) | reg;
    word.to_le_bytes()
}

/// Encode LDR xN, [xM] (load 64-bit).
#[inline]
fn encode_ldr_reg(xn: u32, xm: u32) -> [u8; 4] {
    let word = 0xF940_0000u32 | (xm << 5) | xn;
    word.to_le_bytes()
}

/// Encode ADR xN, PC+offset (±1 MiB).
#[inline]
fn encode_adr(rd: u32, byte_offset: i64) -> [u8; 4] {
    let imm21 = byte_offset as u32;
    let immlo = imm21 & 0x3;
    let immhi = (imm21 >> 2) & 0x7FFFF;
    let word = (immlo << 29) | (0b10000u32 << 24) | (immhi << 5) | rd;
    word.to_le_bytes()
}

/// Load a 64-bit immediate into register xN.
fn emit_mov64(code: &mut Vec<u8>, xn: u32, val: u64) {
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

// ============================================================
// RegContext layout (AArch64, 272 bytes)
// ============================================================
//
// Offset  Field     Notes
// ──────  ──────    ─────────────────────
//   +0    x0
//   +8    x1
//  ...    ...
// +232    x29 (fp)
// +240    x30 (lr)
// +248    sp        original SP before trampoline
// +256    pc        hooked VA (read-only informational)
// +264    nzcv      condition flags

const OFF_SP: u32 = 248;
const OFF_PC: u32 = 256;
const OFF_NZCV: u32 = 264;

/// Emit the save-all-registers block.
///
/// Saves x0-x30, sp (computed original), pc (target_va), nzcv into
/// RegContext at [sp]. Caller must have already allocated REG_CONTEXT_SIZE.
///
/// Uses x16 (IP0) as scratch — ABI-reserved for linker veneers/trampolines,
/// safe to clobber in trampoline context.
fn emit_save_regs(code: &mut Vec<u8>, target_va: u64) {
    // Save x0-x29 as pairs: STP x0,x1,[sp,#0]; STP x2,x3,[sp,#16]; ... STP x28,x29,[sp,#224]
    for i in (0..30).step_by(2) {
        let offset = (i as i32) * 8;
        code.extend_from_slice(&encode_stp_offset(i, i + 1, offset));
    }

    // Save x30 (LR): STR x30, [sp, #240]
    code.extend_from_slice(&encode_str_sp(30, 240));

    // Compute and save original SP: original_sp = current_sp + REG_CONTEXT_SIZE
    // ADD x16, sp, #REG_CONTEXT_SIZE
    code.extend_from_slice(&encode_add_from_sp(16, REG_CONTEXT_SIZE));
    // STR x16, [sp, #248]
    code.extend_from_slice(&encode_str_sp(16, OFF_SP));

    // Store target_va at [sp+256] (pc field, read-only info)
    emit_mov64(code, 16, target_va);
    code.extend_from_slice(&encode_str_sp(16, OFF_PC));

    // Save NZCV: MRS x16, NZCV; STR x16, [sp, #264]
    // MRS x16, NZCV = S3_3_C4_C2_0: [0x10, 0x42, 0x3B, 0xD5]
    code.extend_from_slice(&[0x10, 0x42, 0x3B, 0xD5]);
    code.extend_from_slice(&encode_str_sp(16, OFF_NZCV));
}

/// Emit the restore-all-registers block from RegContext at [sp].
///
/// Restores nzcv, then x0-x30. Does NOT restore SP (the trampoline manages
/// SP explicitly via ADD sp, sp, #frame_size).
fn emit_restore_regs(code: &mut Vec<u8>) {
    // Restore NZCV first: LDR x16, [sp, #264]; MSR NZCV, x16
    code.extend_from_slice(&encode_ldr_sp(16, OFF_NZCV));
    // MSR NZCV, x16: [0x10, 0x42, 0x1B, 0xD5]
    code.extend_from_slice(&[0x10, 0x42, 0x1B, 0xD5]);

    // Restore x0-x29 as pairs
    for i in (0..30).step_by(2) {
        let offset = (i as i32) * 8;
        code.extend_from_slice(&encode_ldp_offset(i, i + 1, offset));
    }

    // Restore x30: LDR x30, [sp, #240]
    code.extend_from_slice(&encode_ldr_sp(30, 240));
}

/// Emit the hook call sequence: load handler address into x16, BLR x16.
///
/// For library symbols: load from data slot (indirect via literal pool or ADR+LDR).
/// For shellcode: ADR to shellcode VA.
fn emit_hook_call(
    code: &mut Vec<u8>,
    trampoline_va: u64,
    source: &HookSource,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
) {
    match source {
        HookSource::LibrarySymbol {
            data_slot_index, ..
        } => {
            // Load handler pointer from data slot.
            // ADR x16, data_slot_va (PC-relative)
            // LDR x16, [x16]       (dereference pointer)
            let slot_va = hook_data_va + (*data_slot_index as u64) * 8;
            let adr_va = trampoline_va + code.len() as u64;
            let adr_offset = slot_va as i64 - adr_va as i64;
            code.extend_from_slice(&encode_adr(16, adr_offset));
            code.extend_from_slice(&encode_ldr_reg(16, 16));
        }
        HookSource::Shellcode(_) => {
            // ADR x16, shellcode_va (PC-relative address of embedded shellcode)
            let sc_va = shellcode_va.expect("shellcode_va must be set for shellcode hooks");
            let adr_va = trampoline_va + code.len() as u64;
            let adr_offset = sc_va as i64 - adr_va as i64;
            code.extend_from_slice(&encode_adr(16, adr_offset));
        }
    }

    // BLR x16
    code.extend_from_slice(&encode_blr(16));
}

/// Relocate displaced AArch64 instructions from their original VA to a new VA.
///
/// Handles PC-relative encodings (ADR, ADRP, LDR literal, B, BL, B.cond, CBZ/CBNZ, TBZ/TBNZ).
fn relocate_displaced_aarch64(bytes: &[u8], orig_va: u64, new_va: u64) -> Result<Vec<u8>> {
    // AArch64 instructions are 4 bytes each.
    if !bytes.len().is_multiple_of(4) {
        anyhow::bail!(
            "displaced bytes length {} is not a multiple of 4",
            bytes.len()
        );
    }

    let mut result = Vec::with_capacity(bytes.len());
    let num_insns = bytes.len() / 4;

    for i in 0..num_insns {
        let off = i * 4;
        let raw = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let insn_orig_va = orig_va + off as u64;
        let insn_new_va = new_va + off as u64;

        // Check if this is a PC-relative instruction.
        if is_pc_relative(raw) {
            let relocated =
                relocate_pc_relative(raw, insn_orig_va, insn_new_va).with_context(|| {
                    format!(
                        "failed to relocate PC-relative instruction 0x{:08x} at VA 0x{:x}",
                        raw, insn_orig_va,
                    )
                })?;
            result.extend_from_slice(&relocated.to_le_bytes());
        } else {
            // Non-PC-relative: copy as-is.
            result.extend_from_slice(&bytes[off..off + 4]);
        }
    }

    Ok(result)
}

/// Return true if the instruction word is PC-relative.
fn is_pc_relative(raw: u32) -> bool {
    let is_adrp = (raw & 0x9F00_0000) == 0x9000_0000;
    let is_adr = (raw & 0x9F00_0000) == 0x1000_0000;
    let is_ldr_lit = (raw & 0x3B00_0000) == 0x1800_0000;
    let is_b = (raw & 0xFC00_0000) == 0x1400_0000;
    let is_bl = (raw & 0xFC00_0000) == 0x9400_0000;
    let is_bcond = (raw & 0xFF00_0010) == 0x5400_0000;
    let is_cbz_cbnz = (raw & 0x7E00_0000) == 0x3400_0000;
    let is_tbz_tbnz = (raw & 0x7E00_0000) == 0x3600_0000;
    is_adrp || is_adr || is_ldr_lit || is_b || is_bl || is_bcond || is_cbz_cbnz || is_tbz_tbnz
}

/// Sign-extend a value from `bits` width to i64.
fn sign_extend(val: u32, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((val as i64) << shift) >> shift
}

/// Relocate a single PC-relative ARM64 instruction.
fn relocate_pc_relative(raw: u32, original_va: u64, new_va: u64) -> Result<u32> {
    let is_adrp = (raw & 0x9F00_0000) == 0x9000_0000;
    let is_adr = (raw & 0x9F00_0000) == 0x1000_0000;
    let is_ldr_lit = (raw & 0x3B00_0000) == 0x1800_0000;
    let is_b = (raw & 0xFC00_0000) == 0x1400_0000;
    let is_bl = (raw & 0xFC00_0000) == 0x9400_0000;
    let is_bcond = (raw & 0xFF00_0010) == 0x5400_0000;
    let is_cbz_cbnz = (raw & 0x7E00_0000) == 0x3400_0000;
    let is_tbz_tbnz = (raw & 0x7E00_0000) == 0x3600_0000;

    if is_adrp {
        let rd = raw & 0x1F;
        let immlo = (raw >> 29) & 0x3;
        let immhi = (raw >> 5) & 0x7_FFFF;
        let imm21 = (immhi << 2) | immlo;
        let imm_signed = sign_extend(imm21, 21);
        let target_page = ((original_va & !0xFFF) as i64 + (imm_signed << 12)) as u64;
        let new_imm = ((target_page as i64) - (new_va as i64 & !0xFFF_i64)) >> 12;
        if !(-(1 << 20)..(1 << 20)).contains(&new_imm) {
            anyhow::bail!("ADRP relocation out of range");
        }
        let ni = (new_imm as u32) & 0x1F_FFFF;
        Ok((1u32 << 31)
            | ((ni & 0x3) << 29)
            | (0b10000u32 << 24)
            | (((ni >> 2) & 0x7_FFFF) << 5)
            | rd)
    } else if is_adr {
        let rd = raw & 0x1F;
        let immlo = (raw >> 29) & 0x3;
        let immhi = (raw >> 5) & 0x7_FFFF;
        let imm21 = (immhi << 2) | immlo;
        let target = (original_va as i64 + sign_extend(imm21, 21)) as u64;
        let new_offset = target as i64 - new_va as i64;
        if !(-(1 << 20)..(1 << 20)).contains(&new_offset) {
            anyhow::bail!("ADR relocation out of range");
        }
        let ni = (new_offset as u32) & 0x1F_FFFF;
        Ok(((ni & 0x3) << 29) | (0b10000u32 << 24) | (((ni >> 2) & 0x7_FFFF) << 5) | rd)
    } else if is_ldr_lit {
        let rt = raw & 0x1F;
        let opc_v = raw & 0xFF00_0000;
        let imm19 = (raw >> 5) & 0x7_FFFF;
        let target = (original_va as i64 + sign_extend(imm19, 19) * 4) as u64;
        let new_offset = target as i64 - new_va as i64;
        if new_offset % 4 != 0 || new_offset / 4 < -(1 << 18) || new_offset / 4 >= (1 << 18) {
            anyhow::bail!("LDR literal relocation out of range");
        }
        Ok(opc_v | ((((new_offset / 4) as u32) & 0x7_FFFF) << 5) | rt)
    } else if is_b || is_bl {
        let opcode_bits = raw & 0xFC00_0000;
        let imm26 = raw & 0x03FF_FFFF;
        let target = (original_va as i64 + sign_extend(imm26, 26) * 4) as u64;
        let new_offset = target as i64 - new_va as i64;
        if new_offset % 4 != 0 || new_offset / 4 < -(1 << 25) || new_offset / 4 >= (1 << 25) {
            anyhow::bail!("B/BL relocation out of range");
        }
        Ok(opcode_bits | (((new_offset / 4) as u32) & 0x03FF_FFFF))
    } else if is_bcond {
        let cond = raw & 0xF;
        let imm19 = (raw >> 5) & 0x7_FFFF;
        let target = (original_va as i64 + sign_extend(imm19, 19) * 4) as u64;
        let new_offset = target as i64 - new_va as i64;
        if new_offset % 4 != 0 || new_offset / 4 < -(1 << 18) || new_offset / 4 >= (1 << 18) {
            anyhow::bail!("B.cond relocation out of range");
        }
        Ok(0x5400_0000 | ((((new_offset / 4) as u32) & 0x7_FFFF) << 5) | cond)
    } else if is_cbz_cbnz {
        let sf_op = raw & 0xFF00_0000;
        let rt = raw & 0x1F;
        let imm19 = (raw >> 5) & 0x7_FFFF;
        let target = (original_va as i64 + sign_extend(imm19, 19) * 4) as u64;
        let new_offset = target as i64 - new_va as i64;
        if new_offset % 4 != 0 || new_offset / 4 < -(1 << 18) || new_offset / 4 >= (1 << 18) {
            anyhow::bail!("CBZ/CBNZ relocation out of range");
        }
        Ok(sf_op | ((((new_offset / 4) as u32) & 0x7_FFFF) << 5) | rt)
    } else if is_tbz_tbnz {
        let b5_op = raw & 0xFF00_0000;
        let b40 = (raw >> 19) & 0x1F;
        let rt = raw & 0x1F;
        let imm14 = (raw >> 5) & 0x3FFF;
        let target = (original_va as i64 + sign_extend(imm14, 14) * 4) as u64;
        let new_offset = target as i64 - new_va as i64;
        if new_offset % 4 != 0 || new_offset / 4 < -(1 << 13) || new_offset / 4 >= (1 << 13) {
            anyhow::bail!("TBZ/TBNZ relocation out of range");
        }
        Ok(b5_op | (b40 << 19) | ((((new_offset / 4) as u32) & 0x3FFF) << 5) | rt)
    } else {
        anyhow::bail!("unrecognised PC-relative instruction 0x{:08x}", raw);
    }
}

// ============================================================
// Condition check (AArch64)
// ============================================================

/// Map an AArch64 register name (x0-x30) to its byte offset in the RegContext.
fn reg_context_offset_aarch64(reg: &str) -> Result<u32> {
    if let Some(rest) = reg.strip_prefix('x') {
        if let Ok(n) = rest.parse::<u32>() {
            if n <= 30 {
                return Ok(n * 8);
            }
        }
    }
    anyhow::bail!("unknown AArch64 register '{}'", reg);
}

/// Encode B.cond with a 19-bit signed instruction offset.
/// `cond` is the 4-bit condition code (0=EQ, 1=NE, 2=CS/HS, 3=CC/LO, etc.).
#[inline]
fn encode_bcond(cond: u32, offset_instructions: i32) -> [u8; 4] {
    let imm19 = (offset_instructions as u32) & 0x7_FFFF;
    let word = 0x5400_0000 | (imm19 << 5) | cond;
    word.to_le_bytes()
}

/// Emit an AArch64 condition check that reads a register from the RegContext
/// on the stack, compares it, and emits a B.cond to skip the hook call
/// when the condition is FALSE.
///
/// Uses x9 and x10 as scratch (these are caller-saved temporaries, safe to
/// use before setting up x0 for the hook call).
///
/// Returns the byte offset in `code` where the B.cond instruction is located
/// (for later fixup once the skip target address is known).
fn emit_condition_check_aarch64(code: &mut Vec<u8>, condition: &HookCondition) -> Result<usize> {
    let reg_offset = reg_context_offset_aarch64(&condition.register)?;

    // LDR x9, [sp, #reg_offset]  — load saved register value
    code.extend_from_slice(&encode_ldr_sp(9, reg_offset));

    if condition.op == CondOp::BitSet || condition.op == CondOp::BitClear {
        // Load mask into x10, then TST x9, x10 (= ANDS xzr, x9, x10)
        emit_mov64(code, 10, condition.value);
        // TST x9, x10 (= ANDS xzr, x9, x10)
        let word: u32 = (1 << 31) | (3 << 29) | (0b01010 << 24) | (10 << 16) | (9 << 5) | 31;
        code.extend_from_slice(&word.to_le_bytes());
    } else {
        // Load value into x10, then CMP x9, x10 (= SUBS xzr, x9, x10)
        emit_mov64(code, 10, condition.value);
        // CMP x9, x10 = SUBS xzr, x9, x10
        // sf=1, op=1, S=1, 01011, shift=00, 0, Rm=x10, imm6=0, Rn=x9, Rd=xzr
        let word: u32 =
            (1 << 31) | (1 << 30) | (1 << 29) | (0b01011 << 24) | (10 << 16) | (9 << 5) | 31;
        code.extend_from_slice(&word.to_le_bytes());
    }

    // B.cond to skip hook when condition is FALSE.
    // Condition codes for B.cond (inverse of user condition):
    let cond_code = match condition.op {
        CondOp::Eq => 1,       // B.NE — skip if not equal
        CondOp::Ne => 0,       // B.EQ — skip if equal
        CondOp::Gt => 9,       // B.LS — skip if lower or same (unsigned <=)
        CondOp::Gte => 3,      // B.CC/LO — skip if carry clear (unsigned <)
        CondOp::Lt => 2,       // B.CS/HS — skip if carry set (unsigned >=)
        CondOp::Lte => 8,      // B.HI — skip if higher (unsigned >)
        CondOp::BitSet => 0,   // B.EQ — skip if zero (bits not set)
        CondOp::BitClear => 1, // B.NE — skip if nonzero (bits are set)
    };

    let fixup_pos = code.len();
    code.extend_from_slice(&encode_bcond(cond_code, 0)); // placeholder offset

    Ok(fixup_pos)
}

/// Fix up a B.cond placeholder to jump to the current code position.
fn fixup_condition_skip_aarch64(code: &mut [u8], fixup_pos: usize, skip_target_offset: usize) {
    let delta = (skip_target_offset as i64) - (fixup_pos as i64);
    assert!(delta % 4 == 0, "B.cond target must be 4-byte aligned");
    let offset_insns = (delta / 4) as i32;
    let new_word = encode_bcond(
        // Preserve the existing condition code from the placeholder
        u32::from_le_bytes(code[fixup_pos..fixup_pos + 4].try_into().unwrap()) & 0xF,
        offset_insns,
    );
    code[fixup_pos..fixup_pos + 4].copy_from_slice(&new_word);
}

// ============================================================
// Toggle check (AArch64)
// ============================================================

/// Emit an AArch64 toggle check: load toggle byte from data segment, if zero
/// branch to skip (B.EQ skip). Uses x9 as scratch.
///
/// Returns the byte offset of the B.EQ instruction for later fixup.
fn emit_toggle_check_aarch64(code: &mut Vec<u8>, trampoline_va: u64, toggle_va: u64) -> usize {
    // ADR x9, toggle_va (PC-relative)
    let adr_va = trampoline_va + code.len() as u64;
    let adr_offset = toggle_va as i64 - adr_va as i64;
    code.extend_from_slice(&encode_adr(9, adr_offset));

    // LDRB w9, [x9]  —  load toggle byte (0 or 1)
    // LDRB w9, [x9] = 0x39400129
    code.extend_from_slice(&0x3940_0129u32.to_le_bytes());

    // CBZ w9, skip  —  skip hook if toggle == 0
    // CBZ w9, +0 (placeholder, 32-bit CBZ: sf=0)
    let fixup_pos = code.len();
    code.extend_from_slice(&0x3400_0009u32.to_le_bytes()); // CBZ w9, +0 (placeholder)

    fixup_pos
}

/// Fix up a CBZ placeholder emitted by emit_toggle_check_aarch64.
fn fixup_toggle_skip_aarch64(code: &mut [u8], fixup_pos: usize, skip_target_offset: usize) {
    let delta = (skip_target_offset as i64) - (fixup_pos as i64);
    assert!(delta % 4 == 0, "CBZ target must be 4-byte aligned");
    let offset_insns = (delta / 4) as i32;
    // CBZ w9: sf=0, op=0, imm19=offset_insns, Rt=9
    let imm19 = (offset_insns as u32) & 0x7_FFFF;
    let word = 0x3400_0000 | (imm19 << 5) | 9;
    code[fixup_pos..fixup_pos + 4].copy_from_slice(&word.to_le_bytes());
}

// ============================================================
// Mixed-mode chain trampoline (AArch64)
// ============================================================

/// Generate a mixed pre+post chained trampoline for AArch64.
///
/// Sequence:
/// 1. Save regs
/// 2. Run pre hooks (with toggle/condition each)
/// 3. Restore regs
/// 4. Displaced instructions
/// 5. Save regs
/// 6. Run post hooks (with toggle/condition each)
/// 7. Restore regs
/// 8. B return_va
pub fn generate_mixed_chain_trampoline_aarch64(
    trampoline_va: u64,
    hooks: &[&ResolvedHook],
    hook_data_va: u64,
    shellcode_vas: &[Option<u64>],
    toggle_vas: &[Option<u64>],
) -> Result<Trampoline> {
    let hook0 = hooks[0];
    let return_va = hook0.target_va + hook0.displaced_len as u64;
    let mut code = Vec::with_capacity(1024);

    // Partition hooks into pre and post.
    let pre_indices: Vec<usize> = hooks
        .iter()
        .enumerate()
        .filter(|(_, h)| h.mode == HookMode::Pre)
        .map(|(i, _)| i)
        .collect();
    let post_indices: Vec<usize> = hooks
        .iter()
        .enumerate()
        .filter(|(_, h)| h.mode == HookMode::Post)
        .map(|(i, _)| i)
        .collect();

    // === Pre hooks ===
    // 1. Allocate RegContext
    code.extend_from_slice(&encode_sub_sp(REG_CONTEXT_SIZE));

    // 2. Save all registers
    emit_save_regs(&mut code, hook0.target_va);

    // 3. Call pre hooks
    for &i in &pre_indices {
        let hook = hooks[i];

        let toggle_fixup =
            toggle_vas[i].map(|tva| emit_toggle_check_aarch64(&mut code, trampoline_va, tva));

        let cond_fixup = if let Some(ref cond) = hook.condition {
            Some(emit_condition_check_aarch64(&mut code, cond)?)
        } else {
            None
        };

        code.extend_from_slice(&encode_mov_from_sp(0));
        emit_hook_call(
            &mut code,
            trampoline_va,
            &hook.source,
            hook_data_va,
            shellcode_vas[i],
        );

        if let Some(fixup_pos) = cond_fixup {
            let target = code.len();
            fixup_condition_skip_aarch64(&mut code, fixup_pos, target);
        }
        if let Some(fixup_pos) = toggle_fixup {
            let target = code.len();
            fixup_toggle_skip_aarch64(&mut code, fixup_pos, target);
        }
    }

    // 4. Restore all registers
    emit_restore_regs(&mut code);

    // 5. Deallocate RegContext
    code.extend_from_slice(&encode_add_sp(REG_CONTEXT_SIZE));

    // === Displaced instructions ===
    let displaced_va = trampoline_va + code.len() as u64;
    let relocated =
        relocate_displaced_aarch64(&hook0.displaced_bytes, hook0.target_va, displaced_va)
            .context("failed to relocate displaced instructions for mixed chain hook")?;
    code.extend_from_slice(&relocated);

    // === Post hooks ===
    // 6. Allocate RegContext
    code.extend_from_slice(&encode_sub_sp(REG_CONTEXT_SIZE));

    // 7. Save all registers
    emit_save_regs(&mut code, hook0.target_va);

    // 8. Call post hooks
    for &i in &post_indices {
        let hook = hooks[i];

        let toggle_fixup =
            toggle_vas[i].map(|tva| emit_toggle_check_aarch64(&mut code, trampoline_va, tva));

        let cond_fixup = if let Some(ref cond) = hook.condition {
            Some(emit_condition_check_aarch64(&mut code, cond)?)
        } else {
            None
        };

        code.extend_from_slice(&encode_mov_from_sp(0));
        emit_hook_call(
            &mut code,
            trampoline_va,
            &hook.source,
            hook_data_va,
            shellcode_vas[i],
        );

        if let Some(fixup_pos) = cond_fixup {
            let target = code.len();
            fixup_condition_skip_aarch64(&mut code, fixup_pos, target);
        }
        if let Some(fixup_pos) = toggle_fixup {
            let target = code.len();
            fixup_toggle_skip_aarch64(&mut code, fixup_pos, target);
        }
    }

    // 9. Restore all registers
    emit_restore_regs(&mut code);

    // 10. Deallocate RegContext
    code.extend_from_slice(&encode_add_sp(REG_CONTEXT_SIZE));

    // 11. B return_va
    let b_va = trampoline_va + code.len() as u64;
    let delta = return_va as i64 - b_va as i64;
    if delta % 4 != 0 || delta.abs() > 128 * 1024 * 1024 {
        anyhow::bail!("mixed chain hook: return branch out of range");
    }
    code.extend_from_slice(&encode_b((delta / 4) as i32));

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

// ============================================================
// Trampoline generators
// ============================================================

/// Generate a pre-hook trampoline (AArch64).
///
/// Sequence:
/// 1. SUB sp, sp, #REG_CONTEXT_SIZE (allocate RegContext)
/// 2. Save all GPRs + sp + pc + nzcv
/// 3. MOV x0, sp (arg1 = &RegContext)
/// 4. Load hook address into x16
/// 5. BLR x16 (call hook)
/// 6. Restore all GPRs + nzcv
/// 7. ADD sp, sp, #REG_CONTEXT_SIZE (deallocate)
/// 8. Relocated displaced instruction
/// 9. B return_va
fn generate_pre_hook(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
) -> Result<Trampoline> {
    let return_va = hook.target_va + hook.displaced_len as u64;
    let mut code = Vec::with_capacity(512);

    // 1. Allocate RegContext
    code.extend_from_slice(&encode_sub_sp(REG_CONTEXT_SIZE));

    // 2. Save all registers
    emit_save_regs(&mut code, hook.target_va);

    // 2a. Toggle check (if present).
    let toggle_fixup =
        toggle_va.map(|tva| emit_toggle_check_aarch64(&mut code, trampoline_va, tva));

    // 2b. Condition check (if present).
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_aarch64(&mut code, cond)?)
    } else {
        None
    };

    // 3. MOV x0, sp (arg1 = &RegContext)
    code.extend_from_slice(&encode_mov_from_sp(0));

    // 4-5. Call hook handler
    emit_hook_call(
        &mut code,
        trampoline_va,
        &hook.source,
        hook_data_va,
        shellcode_va,
    );

    // 5b. Fixup condition skip target.
    if let Some(fixup_pos) = cond_fixup {
        let target = code.len();
        fixup_condition_skip_aarch64(&mut code, fixup_pos, target);
    }

    // 5c. Fixup toggle skip target.
    if let Some(fixup_pos) = toggle_fixup {
        let target = code.len();
        fixup_toggle_skip_aarch64(&mut code, fixup_pos, target);
    }

    // 6. Restore all registers
    emit_restore_regs(&mut code);

    // 7. Deallocate RegContext
    code.extend_from_slice(&encode_add_sp(REG_CONTEXT_SIZE));

    // 8. Relocated displaced instruction(s)
    let displaced_va = trampoline_va + code.len() as u64;
    let relocated = relocate_displaced_aarch64(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate displaced instructions for hook at 0x{:x}",
                hook.target_va,
            )
        })?;
    code.extend_from_slice(&relocated);

    // 9. B return_va
    let b_va = trampoline_va + code.len() as u64;
    let delta = return_va as i64 - b_va as i64;
    if delta % 4 != 0 || delta.abs() > 128 * 1024 * 1024 {
        anyhow::bail!(
            "hook at 0x{:x}: return branch out of range (delta={} bytes)",
            hook.target_va,
            delta,
        );
    }
    code.extend_from_slice(&encode_b((delta / 4) as i32));

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

/// Generate a post-hook trampoline (AArch64).
///
/// Displaced instruction executes FIRST, then hook fires.
fn generate_post_hook(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
) -> Result<Trampoline> {
    let return_va = hook.target_va + hook.displaced_len as u64;
    let mut code = Vec::with_capacity(512);

    // 1. Relocated displaced instruction(s) FIRST
    let displaced_va = trampoline_va + code.len() as u64;
    let relocated = relocate_displaced_aarch64(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate displaced instructions for hook at 0x{:x}",
                hook.target_va,
            )
        })?;
    code.extend_from_slice(&relocated);

    // 2. Allocate RegContext
    code.extend_from_slice(&encode_sub_sp(REG_CONTEXT_SIZE));

    // 3. Save all registers
    emit_save_regs(&mut code, hook.target_va);

    // 3a. Toggle check (if present).
    let toggle_fixup =
        toggle_va.map(|tva| emit_toggle_check_aarch64(&mut code, trampoline_va, tva));

    // 3b. Condition check (if present).
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_aarch64(&mut code, cond)?)
    } else {
        None
    };

    // 4. MOV x0, sp
    code.extend_from_slice(&encode_mov_from_sp(0));

    // 5-6. Call hook handler
    emit_hook_call(
        &mut code,
        trampoline_va,
        &hook.source,
        hook_data_va,
        shellcode_va,
    );

    // 6b. Fixup condition skip target.
    if let Some(fixup_pos) = cond_fixup {
        let target = code.len();
        fixup_condition_skip_aarch64(&mut code, fixup_pos, target);
    }

    // 6c. Fixup toggle skip target.
    if let Some(fixup_pos) = toggle_fixup {
        let target = code.len();
        fixup_toggle_skip_aarch64(&mut code, fixup_pos, target);
    }

    // 7. Restore all registers
    emit_restore_regs(&mut code);

    // 8. Deallocate RegContext
    code.extend_from_slice(&encode_add_sp(REG_CONTEXT_SIZE));

    // 9. B return_va
    let b_va = trampoline_va + code.len() as u64;
    let delta = return_va as i64 - b_va as i64;
    if delta % 4 != 0 || delta.abs() > 128 * 1024 * 1024 {
        anyhow::bail!(
            "hook at 0x{:x}: return branch out of range (delta={} bytes)",
            hook.target_va,
            delta,
        );
    }
    code.extend_from_slice(&encode_b((delta / 4) as i32));

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

/// Generate a replace-hook trampoline (AArch64).
///
/// Hook handler receives:
/// - x0 (arg1) = &RegContext
/// - x1 (arg2) = pointer to original function stub
///
/// After hook returns, restore registers and B to return_va (skip displaced).
/// Original function stub appended after main trampoline.
fn generate_replace_hook(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
) -> Result<Trampoline> {
    let return_va = hook.target_va + hook.displaced_len as u64;
    let mut code = Vec::with_capacity(512);

    // === Main trampoline ===

    // 1. Allocate RegContext
    code.extend_from_slice(&encode_sub_sp(REG_CONTEXT_SIZE));

    // 2. Save all registers
    emit_save_regs(&mut code, hook.target_va);

    // 2a. Toggle check (if present) — skip to fallback path when toggle is 0.
    let toggle_fixup =
        toggle_va.map(|tva| emit_toggle_check_aarch64(&mut code, trampoline_va, tva));

    // 2b. Condition check (if present) — skip to fallback path when false.
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_aarch64(&mut code, cond)?)
    } else {
        None
    };

    // 3. MOV x0, sp (arg1 = &RegContext)
    code.extend_from_slice(&encode_mov_from_sp(0));

    // 4. ADR x1, original_stub (arg2) — placeholder, fix up later
    let adr_x1_fixup_pos = code.len();
    code.extend_from_slice(&encode_adr(1, 0)); // placeholder offset

    // 5. Call hook handler
    emit_hook_call(
        &mut code,
        trampoline_va,
        &hook.source,
        hook_data_va,
        shellcode_va,
    );

    // 6. Restore all registers
    emit_restore_regs(&mut code);

    // 7. Deallocate RegContext
    code.extend_from_slice(&encode_add_sp(REG_CONTEXT_SIZE));

    // 8. RET to caller — the hook has fully replaced the function.
    code.extend_from_slice(&[0xC0, 0x03, 0x5F, 0xD6]); // RET (x30)

    // === Fallback path (toggle/condition was false) ===
    let has_fallback = toggle_fixup.is_some() || cond_fixup.is_some();
    if has_fallback {
        let fallback_target = code.len();
        if let Some(fixup_pos) = toggle_fixup {
            fixup_toggle_skip_aarch64(&mut code, fixup_pos, fallback_target);
        }
        if let Some(fixup_pos) = cond_fixup {
            fixup_condition_skip_aarch64(&mut code, fixup_pos, fallback_target);
        }

        // Restore all registers
        emit_restore_regs(&mut code);

        // Deallocate RegContext
        code.extend_from_slice(&encode_add_sp(REG_CONTEXT_SIZE));

        // Execute displaced instruction(s)
        let displaced_va = trampoline_va + code.len() as u64;
        let relocated = relocate_displaced_aarch64(
            &hook.displaced_bytes,
            hook.target_va,
            displaced_va,
        )
        .with_context(|| {
            format!(
                "failed to relocate displaced instructions for conditional replace hook at 0x{:x}",
                hook.target_va,
            )
        })?;
        code.extend_from_slice(&relocated);

        // B to continue original function
        let b_va = trampoline_va + code.len() as u64;
        let delta = return_va as i64 - b_va as i64;
        if delta % 4 != 0 || delta.abs() > 128 * 1024 * 1024 {
            anyhow::bail!(
                "hook at 0x{:x}: conditional fallback branch out of range (delta={} bytes)",
                hook.target_va,
                delta,
            );
        }
        code.extend_from_slice(&encode_b((delta / 4) as i32));
    }

    // === Original function stub (for when hook fires and wants to call original) ===
    let stub_offset = code.len();
    let stub_va = trampoline_va + stub_offset as u64;

    // Fix up ADR x1 to point to the stub.
    let adr_va = trampoline_va + adr_x1_fixup_pos as u64;
    let adr_offset = stub_va as i64 - adr_va as i64;
    let adr_word = encode_adr(1, adr_offset);
    code[adr_x1_fixup_pos..adr_x1_fixup_pos + 4].copy_from_slice(&adr_word);

    // Relocated displaced instruction(s)
    let displaced_va = trampoline_va + code.len() as u64;
    let relocated = relocate_displaced_aarch64(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate displaced instructions for replace hook at 0x{:x}",
                hook.target_va,
            )
        })?;
    code.extend_from_slice(&relocated);

    // B to continue original function (target_va + displaced_len)
    let b2_va = trampoline_va + code.len() as u64;
    let delta2 = return_va as i64 - b2_va as i64;
    if delta2 % 4 != 0 || delta2.abs() > 128 * 1024 * 1024 {
        anyhow::bail!(
            "hook at 0x{:x}: stub return branch out of range (delta={} bytes)",
            hook.target_va,
            delta2,
        );
    }
    code.extend_from_slice(&encode_b((delta2 / 4) as i32));

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::{HookMode, HookSource, ResolvedHook};

    fn make_test_hook(mode: HookMode) -> ResolvedHook {
        ResolvedHook {
            target_va: 0x100000,
            file_offset: 0x1000,
            // NOP (4 bytes on AArch64)
            displaced_bytes: vec![0x1F, 0x20, 0x03, 0xD5],
            displaced_len: 4,
            mode,
            source: HookSource::Shellcode(vec![
                0xC0, 0x03, 0x5F, 0xD6, // RET
            ]),
            condition: None,
            toggle_index: 0,
            initial_enabled: true,
            return_slot_index: if mode == HookMode::Return {
                Some(0)
            } else {
                None
            },
        }
    }

    #[test]
    fn test_pre_hook_generates() {
        let hook = make_test_hook(HookMode::Pre);
        let result = generate_hook_trampoline(0x200000, &hook, 0x180000, Some(0x1F0000), None);
        assert!(result.is_ok(), "pre-hook failed: {:?}", result.err());
        let tramp = result.unwrap();
        assert_eq!(tramp.va, 0x200000);
        // Must be multiple of 4 (ARM64 instruction alignment)
        assert_eq!(tramp.code.len() % 4, 0);
        // Must have reasonable size
        assert!(
            tramp.code.len() >= 100,
            "trampoline too small: {}",
            tramp.code.len()
        );
    }

    #[test]
    fn test_post_hook_generates() {
        let hook = make_test_hook(HookMode::Post);
        let result = generate_hook_trampoline(0x200000, &hook, 0x180000, Some(0x1F0000), None);
        assert!(result.is_ok(), "post-hook failed: {:?}", result.err());
        let tramp = result.unwrap();
        assert_eq!(tramp.code.len() % 4, 0);
        // Post-hook starts with displaced NOP instruction
        let first_word = u32::from_le_bytes(tramp.code[0..4].try_into().unwrap());
        // The relocated NOP should still be a NOP (not PC-relative)
        assert_eq!(first_word, 0xD503_201F, "first instruction should be NOP");
    }

    #[test]
    fn test_replace_hook_generates() {
        let hook = make_test_hook(HookMode::Replace);
        let result = generate_hook_trampoline(0x200000, &hook, 0x180000, Some(0x1F0000), None);
        assert!(result.is_ok(), "replace-hook failed: {:?}", result.err());
        let tramp = result.unwrap();
        assert_eq!(tramp.code.len() % 4, 0);
        // Replace trampoline is larger (has original function stub appended)
        assert!(
            tramp.code.len() > 100,
            "replace trampoline too small: {}",
            tramp.code.len()
        );
        // Main flow ends with RET, stub ends with B — expect at least 1 B instruction
        let b_count = (0..tramp.code.len())
            .step_by(4)
            .filter(|&i| {
                let w = u32::from_le_bytes(tramp.code[i..i + 4].try_into().unwrap());
                (w & 0xFC00_0000) == 0x1400_0000 // B opcode
            })
            .count();
        assert!(
            b_count >= 1,
            "expected at least 1 B instruction (stub), got {}",
            b_count
        );
        // Should contain a RET instruction (0xC0035FD6)
        let ret_count = (0..tramp.code.len())
            .step_by(4)
            .filter(|&i| tramp.code[i..i + 4] == [0xC0, 0x03, 0x5F, 0xD6])
            .count();
        assert!(ret_count >= 1, "expected RET in replace trampoline");
    }

    #[test]
    fn test_library_symbol_hook() {
        let hook = ResolvedHook {
            target_va: 0x100000,
            file_offset: 0x1000,
            displaced_bytes: vec![0x1F, 0x20, 0x03, 0xD5], // NOP
            displaced_len: 4,
            mode: HookMode::Pre,
            source: HookSource::LibrarySymbol {
                name: "my_hook".to_string(),
                data_slot_index: 0,
            },
            condition: None,
            toggle_index: 0,
            initial_enabled: true,
            return_slot_index: None,
        };
        let result = generate_hook_trampoline(0x200000, &hook, 0x180000, None, None);
        assert!(result.is_ok(), "library hook failed: {:?}", result.err());
        let tramp = result.unwrap();
        // Should contain BLR x16 somewhere: 0xD63F0200
        let has_blr = (0..tramp.code.len()).step_by(4).any(|i| {
            let w = u32::from_le_bytes(tramp.code[i..i + 4].try_into().unwrap());
            w == 0xD63F_0200 // BLR x16
        });
        assert!(has_blr, "expected BLR x16 in trampoline");
    }

    #[test]
    fn test_relocate_nop() {
        // NOP is not PC-relative, should be unchanged
        let nop = vec![0x1F, 0x20, 0x03, 0xD5];
        let result = relocate_displaced_aarch64(&nop, 0x1000, 0x2000).unwrap();
        assert_eq!(result, nop);
    }

    #[test]
    fn test_relocate_b() {
        // B +0x100 at VA 0x1000 → target = 0x1100
        // imm26 = 0x100/4 = 0x40 → 0x14000040
        let b_insn = 0x1400_0040u32.to_le_bytes().to_vec();
        // Relocate to VA 0x2000 → new offset = 0x1100 - 0x2000 = -0xF00
        // imm26 = -0xF00/4 = -0x3C0 → 0x14 | (-0x3C0 & 0x3FFFFFF) = 0x17FFFC40
        let result = relocate_displaced_aarch64(&b_insn, 0x1000, 0x2000).unwrap();
        let relocated = u32::from_le_bytes(result[0..4].try_into().unwrap());
        // Verify it's still a B instruction
        assert_eq!(relocated & 0xFC00_0000, 0x1400_0000);
        // Verify target: new_va(0x2000) + imm26*4 should == 0x1100
        let imm26 = relocated & 0x03FF_FFFF;
        let offset = sign_extend(imm26, 26) * 4;
        let target = (0x2000i64 + offset) as u64;
        assert_eq!(target, 0x1100);
    }

    #[test]
    fn test_return_hook_generates_two_trampolines() {
        let hook = make_test_hook(HookMode::Return);
        let entry_va = 0x200000u64;
        let ret_tramp_va = 0x200100u64;
        let return_slot_va = 0x180010u64;
        let result = generate_return_hook_trampolines(
            entry_va,
            ret_tramp_va,
            &hook,
            0x180000,
            Some(0x1F0000),
            None,
            return_slot_va,
        );
        assert!(
            result.is_ok(),
            "AArch64 return hook failed: {:?}",
            result.err()
        );
        let (entry, ret_tramp) = result.unwrap();
        assert_eq!(entry.va, entry_va);
        assert_eq!(ret_tramp.va, ret_tramp_va);
        // ARM64 instructions must be 4-byte aligned
        assert_eq!(entry.code.len() % 4, 0, "entry code not 4-byte aligned");
        assert_eq!(
            ret_tramp.code.len() % 4,
            0,
            "return code not 4-byte aligned"
        );
        // Both must be non-empty
        assert!(!entry.code.is_empty(), "entry trampoline empty");
        assert!(!ret_tramp.code.is_empty(), "return trampoline empty");
    }

    #[test]
    fn test_return_hook_entry_ends_with_branch() {
        let hook = make_test_hook(HookMode::Return);
        let entry_va = 0x200000u64;
        let ret_tramp_va = 0x200100u64;
        let return_slot_va = 0x180010u64;
        let (entry, _) = generate_return_hook_trampolines(
            entry_va,
            ret_tramp_va,
            &hook,
            0x180000,
            Some(0x1F0000),
            None,
            return_slot_va,
        )
        .unwrap();
        // Last instruction should be B (unconditional branch)
        let last_word = u32::from_le_bytes(entry.code[entry.code.len() - 4..].try_into().unwrap());
        assert_eq!(
            last_word & 0xFC00_0000,
            0x1400_0000,
            "entry should end with B instruction, got 0x{:08x}",
            last_word
        );
    }

    #[test]
    fn test_return_hook_return_tramp_ends_with_br() {
        let hook = make_test_hook(HookMode::Return);
        let entry_va = 0x200000u64;
        let ret_tramp_va = 0x200100u64;
        let return_slot_va = 0x180010u64;
        let (_, ret_tramp) = generate_return_hook_trampolines(
            entry_va,
            ret_tramp_va,
            &hook,
            0x180000,
            Some(0x1F0000),
            None,
            return_slot_va,
        )
        .unwrap();
        // Last instruction should be BR x16 (0xD61F0200)
        let last_word = u32::from_le_bytes(
            ret_tramp.code[ret_tramp.code.len() - 4..]
                .try_into()
                .unwrap(),
        );
        assert_eq!(
            last_word, 0xD61F_0200,
            "return tramp should end with BR x16, got 0x{:08x}",
            last_word
        );
    }
}
