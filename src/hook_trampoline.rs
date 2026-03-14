//! x86_64 and x86 (PE32) hook trampoline generation.
//!
//! Generates trampolines for pre-hook, post-hook, and replace-hook modes.
//! x86_64 trampolines save/restore a RegContext (144 bytes) and manage the
//! red zone, stack alignment, and displaced instruction relocation.
//! PE32 (32-bit x86) trampolines use a smaller RegContext (40 bytes), cdecl
//! calling convention, and absolute addresses instead of RIP-relative.

use anyhow::{Context, Result};

use crate::hooks::{CondOp, HookCondition, HookMode, HookSource, ResolvedHook};
use crate::trampoline::Trampoline;

/// Target architecture and calling convention for hook trampolines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetAbi {
    /// x86_64 System V ABI (ELF Linux/macOS). 128-byte red zone, rdi/rsi args.
    SysV64,
    /// AArch64 (ELF/Mach-O). Fixed 4-byte instructions, STP/LDP register save.
    Aarch64,
    /// Microsoft x64 ABI (PE64 Windows). No red zone, rcx/rdx args, 32-byte shadow space.
    Win64,
    /// x86 cdecl ABI (PE32 Windows). 32-bit registers, stack-based args, no red zone.
    Win32,
}

/// Size of the x86_64 RegContext struct in bytes.
/// 16 GPRs (rax..r15) + rip + rflags = 18 * 8 = 144.
const REG_CONTEXT_SIZE: u64 = 144;

/// Red zone size (x86-64 SysV ABI).
const RED_ZONE: i32 = 128;

/// Windows x64 ABI has no red zone.
const RED_ZONE_NONE: i32 = 0;

/// Windows x64 shadow space: 32 bytes above return address for any CALL.
const SHADOW_SPACE: u8 = 32;

// ---------------------------------------------------------------------------
// ABI-abstraction helpers
// ---------------------------------------------------------------------------

/// Emit red zone skip (lea rsp, [rsp - 128] on SysV, NOP on Windows).
fn emit_red_zone_skip(code: &mut Vec<u8>, windows_abi: bool) {
    if !windows_abi {
        // lea rsp, [rsp - 128]
        code.extend_from_slice(&[0x48, 0x8D, 0xA4, 0x24, 0x80, 0xFF, 0xFF, 0xFF]);
    }
}

/// Emit red zone restore (lea rsp, [rsp + 128] on SysV, NOP on Windows).
fn emit_red_zone_restore(code: &mut Vec<u8>, windows_abi: bool) {
    if !windows_abi {
        // lea rsp, [rsp + 128]
        code.extend_from_slice(&[0x48, 0x8D, 0xA4, 0x24, 0x80, 0x00, 0x00, 0x00]);
    }
}

/// Emit `mov <arg1_reg>, rsp` for handler first argument.
/// SysV: mov rdi, rsp (48 89 E7). Windows: mov rcx, rsp (48 89 E1).
fn emit_arg1_from_rsp(code: &mut Vec<u8>, windows_abi: bool) {
    if windows_abi {
        // mov rcx, rsp
        code.extend_from_slice(&[0x48, 0x89, 0xE1]);
    } else {
        // mov rdi, rsp
        code.extend_from_slice(&[0x48, 0x89, 0xE7]);
    }
}

/// Emit `lea <arg2_reg>, [rip + disp32]` for replace mode (original function ptr).
/// SysV: lea rsi, [rip + disp32]. Windows: lea rdx, [rip + disp32].
/// Returns the byte offset of the LEA instruction start (for disp32 fixup at +3..+7).
fn emit_replace_arg2_lea(code: &mut Vec<u8>, windows_abi: bool) -> usize {
    let pos = code.len();
    if windows_abi {
        // lea rdx, [rip + disp32]  ; 48 8D 15 <disp32>
        code.extend_from_slice(&[0x48, 0x8D, 0x15, 0x00, 0x00, 0x00, 0x00]);
    } else {
        // lea rsi, [rip + disp32]  ; 48 8D 35 <disp32>
        code.extend_from_slice(&[0x48, 0x8D, 0x35, 0x00, 0x00, 0x00, 0x00]);
    }
    pos
}

/// Emit pre-call shadow space allocation for Windows (sub rsp, 32). NOP on SysV.
fn emit_pre_call_shadow(code: &mut Vec<u8>, windows_abi: bool) {
    if windows_abi {
        // sub rsp, 32  ; 48 83 EC 20
        code.extend_from_slice(&[0x48, 0x83, 0xEC, SHADOW_SPACE]);
    }
}

/// Emit post-call shadow space cleanup for Windows (add rsp, 32). NOP on SysV.
fn emit_post_call_shadow(code: &mut Vec<u8>, windows_abi: bool) {
    if windows_abi {
        // add rsp, 32  ; 48 83 C4 20
        code.extend_from_slice(&[0x48, 0x83, 0xC4, SHADOW_SPACE]);
    }
}

/// Return the red zone size for the given ABI.
fn red_zone_size(windows_abi: bool) -> i32 {
    if windows_abi { RED_ZONE_NONE } else { RED_ZONE }
}

/// Relocate displaced instructions from `orig_ip` to `new_ip`.
fn relocate_instructions(bytes: &[u8], orig_ip: u64, new_ip: u64) -> Result<Vec<u8>> {
    use iced_x86::{BlockEncoder, BlockEncoderOptions, Decoder, DecoderOptions, InstructionBlock};

    let mut decoder = Decoder::with_ip(64, bytes, orig_ip, DecoderOptions::NONE);
    let mut instructions = Vec::new();

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.is_invalid() {
            break;
        }
        instructions.push(instr);
    }

    if instructions.is_empty() {
        anyhow::bail!("no valid instructions to relocate");
    }

    let block = InstructionBlock::new(&instructions, new_ip);
    let result = BlockEncoder::encode(64, block, BlockEncoderOptions::NONE)
        .context("iced-x86 BlockEncoder failed")?;

    Ok(result.code_buffer)
}

/// Generate a hook trampoline for the given resolved hook.
///
/// Dispatches to the appropriate generator based on hook mode and target ABI.
///
/// - `trampoline_va`: VA where this trampoline will be placed.
/// - `hook`: the resolved hook definition.
/// - `hook_data_va`: base VA of the hook data area (for library symbol slots).
/// - `shellcode_va`: VA where shellcode blob is placed (for shellcode hooks).
/// - `abi`: target architecture and calling convention.
/// - `toggle_va`: optional VA of the 1-byte toggle flag for this hook.
pub fn generate_hook_trampoline(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    abi: TargetAbi,
    toggle_va: Option<u64>,
) -> Result<Trampoline> {
    if hook.mode == HookMode::Return {
        anyhow::bail!(
            "Return mode hooks must use generate_return_hook_trampolines() (hook at 0x{:x})",
            hook.target_va,
        );
    }
    match abi {
        TargetAbi::Win32 => generate_pe32_hook_trampoline(
            trampoline_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
        ),
        TargetAbi::Aarch64 => {
            #[cfg(feature = "aarch64")]
            {
                crate::arch::aarch64::hook_trampoline::generate_hook_trampoline(
                    trampoline_va,
                    hook,
                    hook_data_va,
                    shellcode_va,
                    toggle_va,
                )
            }
            #[cfg(not(feature = "aarch64"))]
            {
                anyhow::bail!("AArch64 hook trampolines require --features aarch64")
            }
        }
        TargetAbi::SysV64 => generate_x86_64_hook_trampoline(
            trampoline_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
            false,
        ),
        TargetAbi::Win64 => generate_x86_64_hook_trampoline(
            trampoline_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
            true,
        ),
    }
}

/// Generate a pair of trampolines for a Return mode hook.
///
/// Return hooks require two trampolines:
/// 1. **Entry trampoline** (placed at `entry_va`): saves the original return address
///    to a data slot, replaces it with the return trampoline VA, then runs the
///    displaced instructions and jumps into the function body.
/// 2. **Return trampoline** (placed at `ret_tramp_va`): when the function hits RET,
///    saves registers, calls the handler, restores registers, then jumps back to the
///    original caller via the saved return address.
///
/// The `return_slot_va` is the VA of an 8-byte data slot used to store the
/// original return address between entry and return.
#[allow(clippy::too_many_arguments)]
pub fn generate_return_hook_trampolines(
    entry_va: u64,
    ret_tramp_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    abi: TargetAbi,
    toggle_va: Option<u64>,
    return_slot_va: u64,
) -> Result<(Trampoline, Trampoline)> {
    assert_eq!(
        hook.mode,
        HookMode::Return,
        "generate_return_hook_trampolines requires Return mode"
    );
    match abi {
        TargetAbi::Win32 => generate_return_hook_trampolines_pe32(
            entry_va,
            ret_tramp_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
            return_slot_va,
        ),
        TargetAbi::Aarch64 => {
            #[cfg(feature = "aarch64")]
            {
                crate::arch::aarch64::hook_trampoline::generate_return_hook_trampolines(
                    entry_va,
                    ret_tramp_va,
                    hook,
                    hook_data_va,
                    shellcode_va,
                    toggle_va,
                    return_slot_va,
                )
            }
            #[cfg(not(feature = "aarch64"))]
            {
                anyhow::bail!("AArch64 hook trampolines require --features aarch64")
            }
        }
        TargetAbi::SysV64 => generate_return_hook_trampolines_x86_64(
            entry_va,
            ret_tramp_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
            return_slot_va,
            false,
        ),
        TargetAbi::Win64 => generate_return_hook_trampolines_x86_64(
            entry_va,
            ret_tramp_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
            return_slot_va,
            true,
        ),
    }
}

/// Generate a chained hook trampoline for multiple hooks on the same target VA.
///
/// All hooks must target the same VA and have the same mode (or pre+post mix).
/// Handlers are called in order within a single save/restore frame.
///
/// - `trampoline_va`: VA where this trampoline will be placed.
/// - `hooks`: the hooks in the chain (same target VA).
/// - `hook_data_va`: base VA of the hook data area (for library symbol slots).
/// - `shellcode_vas`: shellcode VA for each hook (Some for shellcode, None for library).
/// - `abi`: target architecture and calling convention.
/// - `toggle_vas`: per-hook toggle VAs (Some for hooks with toggles, None otherwise).
pub fn generate_chained_hook_trampoline(
    trampoline_va: u64,
    hooks: &[&ResolvedHook],
    hook_data_va: u64,
    shellcode_vas: &[Option<u64>],
    abi: TargetAbi,
    toggle_vas: &[Option<u64>],
) -> Result<Trampoline> {
    assert!(!hooks.is_empty());
    assert_eq!(hooks.len(), shellcode_vas.len());
    assert_eq!(hooks.len(), toggle_vas.len());

    // Return mode cannot participate in chains.
    let has_return = hooks.iter().any(|h| h.mode == HookMode::Return);
    if has_return {
        anyhow::bail!(
            "chained hooks at 0x{:x}: return mode cannot be mixed with other hooks",
            hooks[0].target_va,
        );
    }

    // Check for replace mixed with pre/post.
    let has_replace = hooks.iter().any(|h| h.mode == HookMode::Replace);
    let has_pre_or_post = hooks
        .iter()
        .any(|h| h.mode == HookMode::Pre || h.mode == HookMode::Post);

    if has_replace && has_pre_or_post {
        anyhow::bail!(
            "chained hooks at 0x{:x}: replace mode cannot be mixed with pre/post",
            hooks[0].target_va,
        );
    }

    // PE32 (32-bit x86) dispatch -- handles all same-mode chains.
    if abi == TargetAbi::Win32 {
        let mode = hooks[0].mode;
        for (i, hook) in hooks.iter().enumerate().skip(1) {
            if hook.mode != mode {
                anyhow::bail!(
                    "chained hooks at 0x{:x}: mixed modes ({:?} vs {:?}) at index {} (PE32 does not support mixed chains)",
                    hooks[0].target_va,
                    mode,
                    hook.mode,
                    i,
                );
            }
        }
        return generate_pe32_chained_hook_trampoline(
            trampoline_va,
            hooks,
            mode,
            hook_data_va,
            shellcode_vas,
            toggle_vas,
        );
    }

    let is_aarch64 = abi == TargetAbi::Aarch64;
    let windows_abi = abi == TargetAbi::Win64;

    // Check for mixed pre+post (new mixed-mode path).
    let has_pre = hooks.iter().any(|h| h.mode == HookMode::Pre);
    let has_post = hooks.iter().any(|h| h.mode == HookMode::Post);
    let is_mixed = has_pre && has_post;

    if is_mixed {
        // Mixed pre+post chaining.
        if is_aarch64 {
            #[cfg(feature = "aarch64")]
            {
                return crate::arch::aarch64::hook_trampoline::generate_mixed_chain_trampoline_aarch64(
                    trampoline_va, hooks, hook_data_va, shellcode_vas, toggle_vas,
                );
            }
            #[cfg(not(feature = "aarch64"))]
            {
                anyhow::bail!("AArch64 hook trampolines require --features aarch64")
            }
        } else {
            return generate_x86_64_mixed_chain_trampoline(
                trampoline_va,
                hooks,
                hook_data_va,
                shellcode_vas,
                toggle_vas,
                windows_abi,
            );
        }
    }

    // All same mode -- existing codegen.
    let mode = hooks[0].mode;
    for (i, hook) in hooks.iter().enumerate().skip(1) {
        if hook.mode != mode {
            anyhow::bail!(
                "chained hooks at 0x{:x}: mixed modes ({:?} vs {:?}) at index {}",
                hooks[0].target_va,
                mode,
                hook.mode,
                i,
            );
        }
    }

    if is_aarch64 {
        #[cfg(feature = "aarch64")]
        {
            crate::arch::aarch64::hook_trampoline::generate_chained_hook_trampoline(
                trampoline_va,
                hooks,
                hook_data_va,
                shellcode_vas,
                toggle_vas,
            )
        }
        #[cfg(not(feature = "aarch64"))]
        {
            anyhow::bail!("AArch64 hook trampolines require --features aarch64")
        }
    } else {
        generate_x86_64_chained_hook_trampoline(
            trampoline_va,
            hooks,
            mode,
            hook_data_va,
            shellcode_vas,
            toggle_vas,
            windows_abi,
        )
    }
}

/// x86_64 chained hook trampoline generation.
fn generate_x86_64_chained_hook_trampoline(
    trampoline_va: u64,
    hooks: &[&ResolvedHook],
    mode: HookMode,
    hook_data_va: u64,
    shellcode_vas: &[Option<u64>],
    toggle_vas: &[Option<u64>],
    windows_abi: bool,
) -> Result<Trampoline> {
    // All hooks share the same target_va, displaced_bytes, displaced_len.
    let hook0 = hooks[0];
    let return_va = hook0.target_va + hook0.displaced_len as u64;
    let mut code = Vec::with_capacity(512);

    // For post mode, displaced instructions come first.
    if mode == HookMode::Post {
        let displaced_va = trampoline_va + code.len() as u64;
        let relocated =
            relocate_instructions(&hook0.displaced_bytes, hook0.target_va, displaced_va)
                .context("failed to relocate displaced instructions for chained hook")?;
        code.extend_from_slice(&relocated);
    }

    // 1. Skip red zone
    emit_red_zone_skip(&mut code, windows_abi);

    // 2. Allocate RegContext
    code.extend_from_slice(&[0x48, 0x81, 0xEC]); // SUB RSP, imm32
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 3. Save all registers
    emit_save_regs(&mut code, hook0.target_va, windows_abi);

    // For replace mode, set arg2 = ptr to original stub (fixup later).
    let lea_arg2_fixup_pos = if mode == HookMode::Replace {
        Some(emit_replace_arg2_lea(&mut code, windows_abi))
    } else {
        None
    };

    // 4. Call each hook handler in sequence.
    for (i, hook) in hooks.iter().enumerate() {
        // Per-hook toggle check (if present).
        let toggle_fixup =
            toggle_vas[i].map(|tva| emit_toggle_check_x86_64(&mut code, trampoline_va, tva));

        // Per-hook condition check (if present).
        let per_hook_fixup = if let Some(ref cond) = hook.condition {
            Some(emit_condition_check_x86_64(&mut code, cond)?)
        } else {
            None
        };

        // Refresh arg1 = rsp (&RegContext) before each call.
        emit_arg1_from_rsp(&mut code, windows_abi);

        // Align stack
        code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x08]); // SUB RSP, 8

        // Shadow space (Windows only)
        emit_pre_call_shadow(&mut code, windows_abi);

        // Call this hook's handler
        emit_hook_call(
            &mut code,
            trampoline_va,
            &hook.source,
            hook_data_va,
            shellcode_vas[i],
        );

        // Shadow space cleanup (Windows only)
        emit_post_call_shadow(&mut code, windows_abi);

        // Remove alignment
        code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x08]); // ADD RSP, 8

        // Fixup per-hook condition skip.
        if let Some(fixup_pos) = per_hook_fixup {
            let target = code.len();
            fixup_condition_skip(&mut code, fixup_pos, target);
        }

        // Fixup toggle skip.
        if let Some(fixup_pos) = toggle_fixup {
            let target = code.len();
            fixup_toggle_skip(&mut code, fixup_pos, target);
        }
    }

    // 5. Restore all registers
    emit_restore_regs(&mut code);

    // 6. Deallocate RegContext
    code.extend_from_slice(&[0x48, 0x81, 0xC4]); // ADD RSP, imm32
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 7. Restore red zone
    emit_red_zone_restore(&mut code, windows_abi);

    match mode {
        HookMode::Pre => {
            // 8. Displaced instructions + JMP return_va
            let displaced_va = trampoline_va + code.len() as u64;
            let relocated =
                relocate_instructions(&hook0.displaced_bytes, hook0.target_va, displaced_va)
                    .context("failed to relocate displaced instructions for chained pre-hook")?;
            code.extend_from_slice(&relocated);

            let jmp_ip = trampoline_va + code.len() as u64 + 5;
            let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
            code.push(0xE9); // JMP rel32
            code.extend_from_slice(&jmp_rel.to_le_bytes());
        }
        HookMode::Post => {
            // Displaced instructions already emitted at start. Just JMP.
            let jmp_ip = trampoline_va + code.len() as u64 + 5;
            let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
            code.push(0xE9); // JMP rel32
            code.extend_from_slice(&jmp_rel.to_le_bytes());
        }
        HookMode::Replace => {
            // RET to caller.
            code.push(0xC3); // RET

            // Append original function stub.
            let stub_offset = code.len();
            let stub_va = trampoline_va + stub_offset as u64;

            // Fix up lea arg2 displacement.
            if let Some(fixup_pos) = lea_arg2_fixup_pos {
                let lea_rip_after = trampoline_va + fixup_pos as u64 + 7;
                let lea_disp = (stub_va as i64 - lea_rip_after as i64) as i32;
                code[fixup_pos + 3..fixup_pos + 7].copy_from_slice(&lea_disp.to_le_bytes());
            }

            let displaced_va = trampoline_va + code.len() as u64;
            let relocated =
                relocate_instructions(&hook0.displaced_bytes, hook0.target_va, displaced_va)
                    .context(
                        "failed to relocate displaced instructions for chained replace hook",
                    )?;
            code.extend_from_slice(&relocated);

            let jmp2_ip = trampoline_va + code.len() as u64 + 5;
            let jmp2_rel = (return_va as i64 - jmp2_ip as i64) as i32;
            code.push(0xE9); // JMP rel32
            code.extend_from_slice(&jmp2_rel.to_le_bytes());
        }
        // Return mode is rejected before this match is reached.
        _ => unreachable!(),
    }

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

/// x86_64 mixed pre+post chained hook trampoline.
///
/// Sequence:
/// 1. Save regs -> run pre hooks (with toggle/condition each) -> restore regs
/// 2. Displaced instructions
/// 3. Save regs -> run post hooks (with toggle/condition each) -> restore regs
/// 4. JMP return_va
fn generate_x86_64_mixed_chain_trampoline(
    trampoline_va: u64,
    hooks: &[&ResolvedHook],
    hook_data_va: u64,
    shellcode_vas: &[Option<u64>],
    toggle_vas: &[Option<u64>],
    windows_abi: bool,
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
    // 1. Skip red zone
    emit_red_zone_skip(&mut code, windows_abi);

    // 2. Allocate RegContext
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 3. Save all registers
    emit_save_regs(&mut code, hook0.target_va, windows_abi);

    // 4. Call pre hooks
    for &i in &pre_indices {
        let hook = hooks[i];

        let toggle_fixup =
            toggle_vas[i].map(|tva| emit_toggle_check_x86_64(&mut code, trampoline_va, tva));

        let cond_fixup = if let Some(ref cond) = hook.condition {
            Some(emit_condition_check_x86_64(&mut code, cond)?)
        } else {
            None
        };

        emit_arg1_from_rsp(&mut code, windows_abi);
        code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x08]); // SUB RSP, 8 (align stack)
        emit_pre_call_shadow(&mut code, windows_abi);
        emit_hook_call(
            &mut code,
            trampoline_va,
            &hook.source,
            hook_data_va,
            shellcode_vas[i],
        );
        emit_post_call_shadow(&mut code, windows_abi);
        code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x08]); // ADD RSP, 8 (remove alignment)

        if let Some(fixup_pos) = cond_fixup {
            let target = code.len();
            fixup_condition_skip(&mut code, fixup_pos, target);
        }
        if let Some(fixup_pos) = toggle_fixup {
            let target = code.len();
            fixup_toggle_skip(&mut code, fixup_pos, target);
        }
    }

    // 5. Restore all registers
    emit_restore_regs(&mut code);

    // 6. Deallocate RegContext
    code.extend_from_slice(&[0x48, 0x81, 0xC4]); // ADD RSP, imm32
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 7. Restore red zone
    emit_red_zone_restore(&mut code, windows_abi);

    // === Displaced instructions ===
    let displaced_va = trampoline_va + code.len() as u64;
    let relocated = relocate_instructions(&hook0.displaced_bytes, hook0.target_va, displaced_va)
        .context("failed to relocate displaced instructions for mixed chain hook")?;
    code.extend_from_slice(&relocated);

    // === Post hooks ===
    // 8. Skip red zone
    emit_red_zone_skip(&mut code, windows_abi);

    // 9. Allocate RegContext
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 10. Save all registers
    emit_save_regs(&mut code, hook0.target_va, windows_abi);

    // 11. Call post hooks
    for &i in &post_indices {
        let hook = hooks[i];

        let toggle_fixup =
            toggle_vas[i].map(|tva| emit_toggle_check_x86_64(&mut code, trampoline_va, tva));

        let cond_fixup = if let Some(ref cond) = hook.condition {
            Some(emit_condition_check_x86_64(&mut code, cond)?)
        } else {
            None
        };

        emit_arg1_from_rsp(&mut code, windows_abi);
        code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x08]); // align stack
        emit_pre_call_shadow(&mut code, windows_abi);
        emit_hook_call(
            &mut code,
            trampoline_va,
            &hook.source,
            hook_data_va,
            shellcode_vas[i],
        );
        emit_post_call_shadow(&mut code, windows_abi);
        code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x08]); // remove alignment

        if let Some(fixup_pos) = cond_fixup {
            let target = code.len();
            fixup_condition_skip(&mut code, fixup_pos, target);
        }
        if let Some(fixup_pos) = toggle_fixup {
            let target = code.len();
            fixup_toggle_skip(&mut code, fixup_pos, target);
        }
    }

    // 12. Restore all registers
    emit_restore_regs(&mut code);

    // 13. Deallocate RegContext
    code.extend_from_slice(&[0x48, 0x81, 0xC4]);
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 14. Restore red zone
    emit_red_zone_restore(&mut code, windows_abi);

    // 15. JMP return_va
    let jmp_ip = trampoline_va + code.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

/// x86_64 hook trampoline dispatch.
fn generate_x86_64_hook_trampoline(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
    windows_abi: bool,
) -> Result<Trampoline> {
    match hook.mode {
        HookMode::Pre => generate_pre_hook_trampoline(
            trampoline_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
            windows_abi,
        ),
        HookMode::Post => generate_post_hook_trampoline(
            trampoline_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
            windows_abi,
        ),
        HookMode::Replace => generate_replace_hook_trampoline(
            trampoline_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
            windows_abi,
        ),
        HookMode::Return => anyhow::bail!(
            "Return mode hooks must use generate_return_hook_trampolines() (hook at 0x{:x})",
            hook.target_va,
        ),
    }
}

/// Emit a toggle check for x86_64: load toggle byte, test, jz to skip.
///
/// Returns the byte offset of the jz rel32 displacement (for fixup),
/// or None if no toggle_va was provided.
fn emit_toggle_check_x86_64(code: &mut Vec<u8>, trampoline_va: u64, toggle_va: u64) -> usize {
    // movzx eax, byte [rip + disp32]   ; 0F B6 05 <disp32>  (7 bytes)
    let rip_after = trampoline_va + code.len() as u64 + 7;
    let disp = (toggle_va as i64 - rip_after as i64) as i32;
    code.extend_from_slice(&[0x0F, 0xB6, 0x05]);
    code.extend_from_slice(&disp.to_le_bytes());

    // test al, al                      ; 84 C0  (2 bytes)
    code.extend_from_slice(&[0x84, 0xC0]);

    // jz rel32                         ; 0F 84 <disp32>  (6 bytes)
    code.extend_from_slice(&[0x0F, 0x84]);
    let fixup_pos = code.len();
    code.extend_from_slice(&0i32.to_le_bytes()); // placeholder

    fixup_pos
}

/// Fix up a jz rel32 placeholder emitted by emit_toggle_check_x86_64.
fn fixup_toggle_skip(code: &mut [u8], fixup_pos: usize, skip_target_offset: usize) {
    let rel32 = (skip_target_offset as i32) - (fixup_pos as i32 + 4);
    code[fixup_pos..fixup_pos + 4].copy_from_slice(&rel32.to_le_bytes());
}

/// Emit the save-all-registers block into `code`.
///
/// Saves all 16 GPRs + rflags into a RegContext at [rsp].
/// Caller must have already allocated REG_CONTEXT_SIZE bytes on the stack.
/// Also stores the target VA (immutable) at [rsp+128] (rip field).
///
/// Returns the number of bytes emitted.
fn emit_save_regs(code: &mut Vec<u8>, target_va: u64, windows_abi: bool) {
    // Store GPRs into RegContext at [rsp].
    // Layout: rax=+0, rbx=+8, rcx=+16, rdx=+24, rsi=+32, rdi=+40,
    //         rbp=+48, rsp=+56, r8=+64, r9=+72, r10=+80, r11=+88,
    //         r12=+96, r13=+104, r14=+112, r15=+120, rip=+128, rflags=+136

    // mov [rsp+0], rax    ; 48 89 04 24
    emit_mov_to_rsp_off(code, 0x48, 0x89, 0x04, 0); // rax -> +0
    emit_mov_to_rsp_off(code, 0x48, 0x89, 0x1C, 8); // rbx -> +8
    emit_mov_to_rsp_off(code, 0x48, 0x89, 0x0C, 16); // rcx -> +16
    emit_mov_to_rsp_off(code, 0x48, 0x89, 0x14, 24); // rdx -> +24
    emit_mov_to_rsp_off(code, 0x48, 0x89, 0x34, 32); // rsi -> +32
    emit_mov_to_rsp_off(code, 0x48, 0x89, 0x3C, 40); // rdi -> +40
    emit_mov_to_rsp_off(code, 0x48, 0x89, 0x2C, 48); // rbp -> +48

    // For RSP, we need to compute the original RSP value:
    // original_rsp = current_rsp + REG_CONTEXT_SIZE + red_zone
    // lea rax, [rsp + delta]
    // mov [rsp+56], rax
    let orig_rsp_delta = (REG_CONTEXT_SIZE as i32) + red_zone_size(windows_abi);
    // lea rax, [rsp + delta]   ; 48 8D 84 24 <disp32>
    code.extend_from_slice(&[0x48, 0x8D, 0x84, 0x24]);
    code.extend_from_slice(&orig_rsp_delta.to_le_bytes());
    // mov [rsp+56], rax        ; 48 89 44 24 38
    code.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 56]);

    // r8..r15
    emit_mov_to_rsp_off(code, 0x4C, 0x89, 0x04, 64); // r8 -> +64  (offset fits in i8)
    emit_mov_to_rsp_off(code, 0x4C, 0x89, 0x0C, 72); // r9 -> +72
    emit_mov_to_rsp_off(code, 0x4C, 0x89, 0x14, 80); // r10 -> +80
    emit_mov_to_rsp_off(code, 0x4C, 0x89, 0x1C, 88); // r11 -> +88
    emit_mov_to_rsp_off(code, 0x4C, 0x89, 0x24, 96); // r12 -> +96
    emit_mov_to_rsp_off(code, 0x4C, 0x89, 0x2C, 104); // r13 -> +104
    emit_mov_to_rsp_off(code, 0x4C, 0x89, 0x34, 112); // r14 -> +112
    emit_mov_to_rsp_off(code, 0x4C, 0x89, 0x3C, 120); // r15 -> +120

    // Store target_va at [rsp+128] (rip field, immutable info for the hook).
    // mov rax, imm64    ; 48 B8 <imm64>
    code.extend_from_slice(&[0x48, 0xB8]);
    code.extend_from_slice(&target_va.to_le_bytes());
    // mov [rsp+128], rax ; 48 89 84 24 80 00 00 00
    code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]);
    code.extend_from_slice(&128i32.to_le_bytes());

    // Save rflags via pushfq/pop rax/mov [rsp+136], rax.
    // pushfq    ; 9C
    code.push(0x9C);
    // pop rax   ; 58
    code.push(0x58);
    // mov [rsp+136], rax  ; 48 89 84 24 88 00 00 00
    code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]);
    code.extend_from_slice(&136i32.to_le_bytes());
}

/// Emit `mov [rsp+offset], reg` using SIB byte addressing.
///
/// For offsets that fit in i8, use disp8 form. Otherwise disp32.
/// `rex`: REX prefix byte (0x48 for rax-rdi/rbp/rsp, 0x4C for r8-r15)
/// `opcode`: 0x89 (mov r/m64, r64)
/// `modrm_reg_nibble`: the reg-field bits of ModR/M (e.g. 0x04=rax, 0x0C=rcx, 0x1C=rbx).
///   Combined with mod and r/m=100 (SIB follows) to form the full ModR/M byte.
fn emit_mov_to_rsp_off(code: &mut Vec<u8>, rex: u8, opcode: u8, modrm_reg_nibble: u8, offset: i32) {
    // Encoding: REX opcode ModR/M SIB [disp]
    // ModR/M = mod | reg | r/m=100 (SIB follows)
    // SIB = 0x24 = 00|100|100 (base=RSP, no index)
    // disp8 form: mod=01, disp32 form: mod=10

    if (-128..=127).contains(&offset) {
        // mod=01 (disp8), r/m=100 (SIB follows)
        // modrm = 01|reg|100
        // The modrm_reg_nibble is actually (reg<<3)|0x04 for the rsp case
        // So modrm = 0x44 + (reg<<3) - 0x04 + 0x04 = 0x44 | (modrm_reg_nibble & 0x38)
        let modrm = 0x44 | (modrm_reg_nibble & 0x38);
        code.extend_from_slice(&[rex, opcode, modrm, 0x24, offset as u8]);
    } else {
        // mod=10 (disp32), r/m=100 (SIB follows)
        let modrm = 0x84 | (modrm_reg_nibble & 0x38);
        code.extend_from_slice(&[rex, opcode, modrm, 0x24]);
        code.extend_from_slice(&offset.to_le_bytes());
    }
}

/// Emit the restore-all-registers block from RegContext at [rsp].
fn emit_restore_regs(code: &mut Vec<u8>) {
    // Restore rflags first (before clobbering rax).
    // mov rax, [rsp+136]  ; 48 8B 84 24 88 00 00 00
    code.extend_from_slice(&[0x48, 0x8B, 0x84, 0x24]);
    code.extend_from_slice(&136i32.to_le_bytes());
    // push rax ; 50
    code.push(0x50);
    // popfq   ; 9D
    code.push(0x9D);

    // r15..r8 (reverse order doesn't matter, just restore all)
    emit_mov_from_rsp_off(code, 0x4C, 0x8B, 0x3C, 120); // r15 <- +120
    emit_mov_from_rsp_off(code, 0x4C, 0x8B, 0x34, 112); // r14 <- +112
    emit_mov_from_rsp_off(code, 0x4C, 0x8B, 0x2C, 104); // r13 <- +104
    emit_mov_from_rsp_off(code, 0x4C, 0x8B, 0x24, 96); // r12 <- +96
    emit_mov_from_rsp_off(code, 0x4C, 0x8B, 0x1C, 88); // r11 <- +88
    emit_mov_from_rsp_off(code, 0x4C, 0x8B, 0x14, 80); // r10 <- +80
    emit_mov_from_rsp_off(code, 0x4C, 0x8B, 0x0C, 72); // r9 <- +72
    emit_mov_from_rsp_off(code, 0x4C, 0x8B, 0x04, 64); // r8 <- +64

    // rbp, rdi, rsi, rdx, rcx, rbx (skip rsp — we'll fix that with add)
    emit_mov_from_rsp_off(code, 0x48, 0x8B, 0x2C, 48); // rbp <- +48
    emit_mov_from_rsp_off(code, 0x48, 0x8B, 0x3C, 40); // rdi <- +40
    emit_mov_from_rsp_off(code, 0x48, 0x8B, 0x34, 32); // rsi <- +32
    emit_mov_from_rsp_off(code, 0x48, 0x8B, 0x14, 24); // rdx <- +24
    emit_mov_from_rsp_off(code, 0x48, 0x8B, 0x0C, 16); // rcx <- +16
    emit_mov_from_rsp_off(code, 0x48, 0x8B, 0x1C, 8); // rbx <- +8
    // rax last (used as scratch above)
    emit_mov_from_rsp_off(code, 0x48, 0x8B, 0x04, 0); // rax <- +0
}

/// Emit `mov reg, [rsp+offset]`.
fn emit_mov_from_rsp_off(
    code: &mut Vec<u8>,
    rex: u8,
    opcode: u8,
    modrm_reg_nibble: u8,
    offset: i32,
) {
    emit_mov_to_rsp_off(code, rex, opcode, modrm_reg_nibble, offset);
}

/// Emit the hook call sequence: load address + call rax.
///
/// For library symbols: `mov rax, [rip + disp_to_data_slot]` (indirect via data slot).
/// For shellcode: `lea rax, [rip + disp_to_shellcode]` (direct to embedded code).
fn emit_hook_call(
    code: &mut Vec<u8>,
    current_va: u64,
    source: &HookSource,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
) {
    match source {
        HookSource::LibrarySymbol {
            data_slot_index, ..
        } => {
            // mov rax, [rip + disp]  ; 48 8B 05 <disp32>  (7 bytes)
            let slot_va = hook_data_va + (*data_slot_index as u64) * 8;
            let rip_after = current_va + code.len() as u64 + 7;
            let disp = (slot_va as i64 - rip_after as i64) as i32;
            code.extend_from_slice(&[0x48, 0x8B, 0x05]);
            code.extend_from_slice(&disp.to_le_bytes());
        }
        HookSource::Shellcode(_) => {
            // lea rax, [rip + disp]  ; 48 8D 05 <disp32>  (7 bytes)
            let sc_va = shellcode_va.expect("shellcode_va must be set for shellcode hooks");
            let rip_after = current_va + code.len() as u64 + 7;
            let disp = (sc_va as i64 - rip_after as i64) as i32;
            code.extend_from_slice(&[0x48, 0x8D, 0x05]);
            code.extend_from_slice(&disp.to_le_bytes());
        }
    }

    // call rax  ; FF D0  (2 bytes)
    code.extend_from_slice(&[0xFF, 0xD0]);
}

/// Map an x86_64 register name to its byte offset in the RegContext struct.
fn reg_context_offset_x86_64(reg: &str) -> Result<i32> {
    match reg {
        "rax" => Ok(0),
        "rbx" => Ok(8),
        "rcx" => Ok(16),
        "rdx" => Ok(24),
        "rsi" => Ok(32),
        "rdi" => Ok(40),
        "rbp" => Ok(48),
        "rsp" => Ok(56),
        "r8" => Ok(64),
        "r9" => Ok(72),
        "r10" => Ok(80),
        "r11" => Ok(88),
        "r12" => Ok(96),
        "r13" => Ok(104),
        "r14" => Ok(112),
        "r15" => Ok(120),
        _ => anyhow::bail!("unknown x86_64 register '{}'", reg),
    }
}

/// Emit an x86_64 condition check that reads a register from the RegContext
/// on the stack, compares it, and emits a Jcc rel32 to skip the hook call
/// when the condition is FALSE.
///
/// Returns the byte offset in `code` where the Jcc rel32 displacement starts
/// (for later fixup once the skip target address is known).
fn emit_condition_check_x86_64(code: &mut Vec<u8>, condition: &HookCondition) -> Result<usize> {
    let reg_offset = reg_context_offset_x86_64(&condition.register)?;

    // Load the register value from RegContext: mov rax, [rsp + reg_offset]
    emit_mov_from_rsp_off(code, 0x48, 0x8B, 0x04, reg_offset);

    if condition.op == CondOp::BitSet || condition.op == CondOp::BitClear {
        // TEST rax, mask
        if condition.value <= u32::MAX as u64 {
            // test rax, imm32 (sign-extended): 48 A9 <imm32>
            code.extend_from_slice(&[0x48, 0xA9]);
            code.extend_from_slice(&(condition.value as u32).to_le_bytes());
        } else {
            // mov rcx, imm64; test rax, rcx
            code.extend_from_slice(&[0x48, 0xB9]);
            code.extend_from_slice(&condition.value.to_le_bytes());
            // test rax, rcx: 48 85 C8
            code.extend_from_slice(&[0x48, 0x85, 0xC8]);
        }
    } else {
        // CMP rax, value
        if condition.value <= i32::MAX as u64 {
            // cmp rax, imm32 (sign-extended): 48 3D <imm32>
            code.extend_from_slice(&[0x48, 0x3D]);
            code.extend_from_slice(&(condition.value as u32).to_le_bytes());
        } else {
            // mov rcx, imm64; cmp rax, rcx
            code.extend_from_slice(&[0x48, 0xB9]);
            code.extend_from_slice(&condition.value.to_le_bytes());
            // cmp rax, rcx: 48 39 C8
            code.extend_from_slice(&[0x48, 0x39, 0xC8]);
        }
    }

    // Jcc rel32 — skip hook when condition is FALSE.
    // The Jcc is the INVERSE of the condition: if cond says "eq", we skip
    // when NOT equal (JNE).
    let jcc_opcode = match condition.op {
        CondOp::Eq => 0x85,       // JNE — skip if not equal
        CondOp::Ne => 0x84,       // JE  — skip if equal
        CondOp::Gt => 0x86,       // JBE — skip if below or equal (unsigned)
        CondOp::Gte => 0x82,      // JB  — skip if below (unsigned)
        CondOp::Lt => 0x83,       // JAE — skip if above or equal (unsigned)
        CondOp::Lte => 0x87,      // JA  — skip if above (unsigned)
        CondOp::BitSet => 0x84,   // JE  — skip if ZF=1 (TEST result zero → bits not set)
        CondOp::BitClear => 0x85, // JNE — skip if ZF=0 (TEST result nonzero → bits are set)
    };

    // 0F 8x <rel32> — 6-byte near conditional jump
    code.extend_from_slice(&[0x0F, jcc_opcode]);
    let fixup_pos = code.len();
    code.extend_from_slice(&0i32.to_le_bytes()); // placeholder displacement

    Ok(fixup_pos)
}

/// Fix up a Jcc rel32 placeholder to jump to the current code position.
///
/// `fixup_pos` is the byte offset in `code` where the rel32 displacement
/// starts (returned by `emit_condition_check_x86_64`).
fn fixup_condition_skip(code: &mut [u8], fixup_pos: usize, skip_target_offset: usize) {
    // Jcc instruction ends at fixup_pos + 4. rel32 is relative to that.
    let rel32 = (skip_target_offset as i32) - (fixup_pos as i32 + 4);
    code[fixup_pos..fixup_pos + 4].copy_from_slice(&rel32.to_le_bytes());
}

/// Generate a pre-hook trampoline.
///
/// Sequence:
/// 1. Skip red zone
/// 2. Allocate RegContext
/// 3. Save all registers + rflags
/// 4. mov rdi, rsp (arg1 = &RegContext)
/// 5. Align stack to 16 bytes
/// 6. Call hook
/// 7. Remove alignment
/// 8. Restore all registers
/// 9. Deallocate RegContext
/// 10. Restore red zone
/// 11. Execute relocated displaced instructions
/// 12. JMP to return_va
fn generate_pre_hook_trampoline(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
    windows_abi: bool,
) -> Result<Trampoline> {
    let return_va = hook.target_va + hook.displaced_len as u64;
    let mut code = Vec::with_capacity(512);

    // 1. Skip red zone
    emit_red_zone_skip(&mut code, windows_abi);

    // 2. Allocate RegContext: sub rsp, 144
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 3. Save all registers
    emit_save_regs(&mut code, hook.target_va, windows_abi);

    // 3a. Toggle check (if present) — skip hook call if toggle is 0.
    let toggle_fixup = toggle_va.map(|tva| emit_toggle_check_x86_64(&mut code, trampoline_va, tva));

    // 3b. Condition check (if present) — skip hook call if predicate is false.
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_x86_64(&mut code, cond)?)
    } else {
        None
    };

    // 4. arg1 = &RegContext (rdi on SysV, rcx on Windows)
    emit_arg1_from_rsp(&mut code, windows_abi);

    // 5. Align stack: sub rsp, 8
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x08]);

    // 5b. Shadow space (Windows only)
    emit_pre_call_shadow(&mut code, windows_abi);

    // 6. Call hook
    emit_hook_call(
        &mut code,
        trampoline_va,
        &hook.source,
        hook_data_va,
        shellcode_va,
    );

    // 7. Shadow space cleanup (Windows only)
    emit_post_call_shadow(&mut code, windows_abi);

    // 7. Remove alignment: add rsp, 8
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x08]);

    // 7b. Fixup condition skip target (jumps here to skip the hook call).
    if let Some(fixup_pos) = cond_fixup {
        let target = code.len();
        fixup_condition_skip(&mut code, fixup_pos, target);
    }

    // 7c. Fixup toggle skip target.
    if let Some(fixup_pos) = toggle_fixup {
        let target = code.len();
        fixup_toggle_skip(&mut code, fixup_pos, target);
    }

    // 8. Restore all registers
    emit_restore_regs(&mut code);

    // 9. Deallocate RegContext: add rsp, 144
    code.extend_from_slice(&[0x48, 0x81, 0xC4]);
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 10. Restore red zone
    emit_red_zone_restore(&mut code, windows_abi);

    // 11. Relocated displaced instructions
    let displaced_va = trampoline_va + code.len() as u64;
    let relocated = relocate_instructions(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate displaced instructions for hook at 0x{:x}",
                hook.target_va,
            )
        })?;
    code.extend_from_slice(&relocated);

    // 12. JMP return_va
    let jmp_ip = trampoline_va + code.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

/// Generate a post-hook trampoline.
///
/// Same as pre-hook but displaced instructions execute BEFORE the hook call.
fn generate_post_hook_trampoline(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
    windows_abi: bool,
) -> Result<Trampoline> {
    let return_va = hook.target_va + hook.displaced_len as u64;
    let mut code = Vec::with_capacity(512);

    // 1. Execute relocated displaced instructions FIRST
    let displaced_va = trampoline_va + code.len() as u64;
    let relocated = relocate_instructions(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate displaced instructions for hook at 0x{:x}",
                hook.target_va,
            )
        })?;
    code.extend_from_slice(&relocated);

    // 2. Skip red zone
    emit_red_zone_skip(&mut code, windows_abi);

    // 3. Allocate RegContext
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 4. Save all registers
    emit_save_regs(&mut code, hook.target_va, windows_abi);

    // 4a. Toggle check (if present).
    let toggle_fixup = toggle_va.map(|tva| emit_toggle_check_x86_64(&mut code, trampoline_va, tva));

    // 4b. Condition check (if present).
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_x86_64(&mut code, cond)?)
    } else {
        None
    };

    // 5. arg1 = &RegContext
    emit_arg1_from_rsp(&mut code, windows_abi);

    // 6. Align stack
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x08]);

    // 6b. Shadow space (Windows only)
    emit_pre_call_shadow(&mut code, windows_abi);

    // 7. Call hook
    emit_hook_call(
        &mut code,
        trampoline_va,
        &hook.source,
        hook_data_va,
        shellcode_va,
    );

    // 7b. Shadow space cleanup (Windows only)
    emit_post_call_shadow(&mut code, windows_abi);

    // 8. Remove alignment
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x08]);

    // 8b. Fixup condition skip target.
    if let Some(fixup_pos) = cond_fixup {
        let target = code.len();
        fixup_condition_skip(&mut code, fixup_pos, target);
    }

    // 8c. Fixup toggle skip target.
    if let Some(fixup_pos) = toggle_fixup {
        let target = code.len();
        fixup_toggle_skip(&mut code, fixup_pos, target);
    }

    // 9. Restore all registers
    emit_restore_regs(&mut code);

    // 10. Deallocate RegContext
    code.extend_from_slice(&[0x48, 0x81, 0xC4]);
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 11. Restore red zone
    emit_red_zone_restore(&mut code, windows_abi);

    // 12. JMP return_va
    let jmp_ip = trampoline_va + code.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

/// Generate a replace-hook trampoline.
///
/// The hook handler receives:
/// - rdi (arg1) = &RegContext
/// - rsi (arg2) = pointer to original function stub
///
/// After the hook returns, we restore registers and jump to return_va
/// (skipping the displaced instructions in the main flow).
///
/// An "original function stub" is appended after the main trampoline:
///   - relocated displaced instructions
///   - JMP (target_va + displaced_len) to continue the original function
fn generate_replace_hook_trampoline(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
    windows_abi: bool,
) -> Result<Trampoline> {
    let return_va = hook.target_va + hook.displaced_len as u64;
    let mut code = Vec::with_capacity(512);

    // === Main trampoline ===

    // 1. Skip red zone
    emit_red_zone_skip(&mut code, windows_abi);

    // 2. Allocate RegContext
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 3. Save all registers
    emit_save_regs(&mut code, hook.target_va, windows_abi);

    // 3a. Toggle check (if present) — skip to fallback path when toggle is 0.
    let toggle_fixup = toggle_va.map(|tva| emit_toggle_check_x86_64(&mut code, trampoline_va, tva));

    // 3b. Condition check (if present) — skip to fallback path when false.
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_x86_64(&mut code, cond)?)
    } else {
        None
    };

    // 4. arg1 = &RegContext (rdi on SysV, rcx on Windows)
    emit_arg1_from_rsp(&mut code, windows_abi);

    // 5. arg2 = original function stub pointer (rsi on SysV, rdx on Windows)
    // We don't know the stub offset yet — emit placeholder, fix up after.
    let lea_arg2_fixup_pos = emit_replace_arg2_lea(&mut code, windows_abi);

    // 6. Align stack
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x08]);

    // 6b. Shadow space (Windows only)
    emit_pre_call_shadow(&mut code, windows_abi);

    // 7. Call hook
    emit_hook_call(
        &mut code,
        trampoline_va,
        &hook.source,
        hook_data_va,
        shellcode_va,
    );

    // 7b. Shadow space cleanup (Windows only)
    emit_post_call_shadow(&mut code, windows_abi);

    // 8. Remove alignment
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x08]);

    // 9. Restore all registers (hook may have modified RegContext)
    emit_restore_regs(&mut code);

    // 10. Deallocate RegContext
    code.extend_from_slice(&[0x48, 0x81, 0xC4]);
    code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 11. Restore red zone
    emit_red_zone_restore(&mut code, windows_abi);

    // 12. RET to caller — the hook has fully replaced the function.
    code.push(0xC3);

    // === Fallback path (toggle/condition was false) ===
    // When a toggle or condition is present and evaluates to false, we skip the
    // hook call and execute the original code instead: restore → displaced → JMP.
    let has_fallback = toggle_fixup.is_some() || cond_fixup.is_some();
    if has_fallback {
        let fallback_target = code.len();
        if let Some(fixup_pos) = toggle_fixup {
            fixup_toggle_skip(&mut code, fixup_pos, fallback_target);
        }
        if let Some(fixup_pos) = cond_fixup {
            fixup_condition_skip(&mut code, fixup_pos, fallback_target);
        }

        // Restore all registers
        emit_restore_regs(&mut code);

        // Deallocate RegContext
        code.extend_from_slice(&[0x48, 0x81, 0xC4]);
        code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

        // Restore red zone
        emit_red_zone_restore(&mut code, windows_abi);

        // Execute displaced instructions
        let displaced_va = trampoline_va + code.len() as u64;
        let relocated = relocate_instructions(
            &hook.displaced_bytes,
            hook.target_va,
            displaced_va,
        ).with_context(|| format!(
            "failed to relocate displaced instructions for conditional replace hook at 0x{:x}",
            hook.target_va,
        ))?;
        code.extend_from_slice(&relocated);

        // JMP to continue original function
        let jmp_ip = trampoline_va + code.len() as u64 + 5;
        let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
        code.push(0xE9);
        code.extend_from_slice(&jmp_rel.to_le_bytes());
    }

    // === Original function stub (for when hook fires and wants to call original) ===
    let stub_offset = code.len();
    let stub_va = trampoline_va + stub_offset as u64;

    // Fix up the lea arg2 displacement.
    let lea_rip_after = trampoline_va + lea_arg2_fixup_pos as u64 + 7;
    let lea_disp = (stub_va as i64 - lea_rip_after as i64) as i32;
    code[lea_arg2_fixup_pos + 3..lea_arg2_fixup_pos + 7].copy_from_slice(&lea_disp.to_le_bytes());

    // Relocated displaced instructions
    let displaced_va = trampoline_va + code.len() as u64;
    let relocated = relocate_instructions(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate displaced instructions for replace hook at 0x{:x}",
                hook.target_va,
            )
        })?;
    code.extend_from_slice(&relocated);

    // JMP to continue original function (target_va + displaced_len)
    let jmp2_ip = trampoline_va + code.len() as u64 + 5;
    let jmp2_rel = (return_va as i64 - jmp2_ip as i64) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp2_rel.to_le_bytes());

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

/// Generate an x86_64 return hook: entry trampoline + return trampoline.
///
/// **Entry trampoline** (placed at the hook site):
///   1. Save rax (scratch)
///   2. Load original return address from [rsp+8] (the RA pushed by the CALL)
///   3. Store it to the return-address data slot (RIP-relative)
///   4. Load return trampoline VA into rax
///   5. Overwrite [rsp+8] with return trampoline VA
///   6. Restore rax
///   7. Execute relocated displaced instructions
///   8. JMP to function body (target_va + displaced_len)
///
/// **Return trampoline** (jumped to when the function returns):
///   1. Skip red zone + allocate RegContext + save regs
///   2. [Toggle check if toggle_va present]
///   3. [Condition check if condition present]
///   4. mov rdi, rsp (arg1 = &RegContext)
///   5. Call handler
///   6. Restore regs + deallocate + restore red zone
///   7. Load saved original RA from data slot
///   8. Push it and RET (return to original caller)
#[allow(clippy::too_many_arguments)]
fn generate_return_hook_trampolines_x86_64(
    entry_va: u64,
    ret_tramp_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
    return_slot_va: u64,
    windows_abi: bool,
) -> Result<(Trampoline, Trampoline)> {
    let return_va = hook.target_va + hook.displaced_len as u64;

    // === Entry trampoline ===
    let mut entry = Vec::with_capacity(128);

    // push rax (save scratch)
    entry.push(0x50);

    // mov rax, [rsp+8] — load original return address
    // The stack at this point: [rax_saved] [original_RA] ...
    // rsp+0 = saved rax, rsp+8 = original RA
    entry.extend_from_slice(&[0x48, 0x8B, 0x44, 0x24, 0x08]);

    // mov [rip+disp], rax — store original RA to data slot
    let rip_after_store = entry_va + entry.len() as u64 + 7;
    let store_disp = (return_slot_va as i64 - rip_after_store as i64) as i32;
    entry.extend_from_slice(&[0x48, 0x89, 0x05]); // mov [rip+disp32], rax
    entry.extend_from_slice(&store_disp.to_le_bytes());

    // lea rax, [rip+disp] — load return trampoline VA
    let rip_after_lea = entry_va + entry.len() as u64 + 7;
    let lea_disp = (ret_tramp_va as i64 - rip_after_lea as i64) as i32;
    entry.extend_from_slice(&[0x48, 0x8D, 0x05]); // lea rax, [rip+disp32]
    entry.extend_from_slice(&lea_disp.to_le_bytes());

    // mov [rsp+8], rax — overwrite return address with return trampoline VA
    entry.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 0x08]);

    // pop rax (restore scratch)
    entry.push(0x58);

    // Relocated displaced instructions
    let displaced_va = entry_va + entry.len() as u64;
    let relocated = relocate_instructions(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate displaced instructions for return hook entry at 0x{:x}",
                hook.target_va,
            )
        })?;
    entry.extend_from_slice(&relocated);

    // JMP return_va (continue function body)
    let jmp_ip = entry_va + entry.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    entry.push(0xE9);
    entry.extend_from_slice(&jmp_rel.to_le_bytes());

    let entry_tramp = Trampoline {
        va: entry_va,
        code: entry,
    };

    // === Return trampoline ===
    let mut ret_code = Vec::with_capacity(512);

    // 1. Skip red zone
    emit_red_zone_skip(&mut ret_code, windows_abi);

    // 2. Allocate RegContext: sub rsp, 144
    ret_code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    ret_code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 3. Save all registers
    emit_save_regs(&mut ret_code, hook.target_va, windows_abi);

    // 3a. Toggle check (if present).
    let toggle_fixup =
        toggle_va.map(|tva| emit_toggle_check_x86_64(&mut ret_code, ret_tramp_va, tva));

    // 3b. Condition check (if present).
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_x86_64(&mut ret_code, cond)?)
    } else {
        None
    };

    // 4. arg1 = &RegContext
    emit_arg1_from_rsp(&mut ret_code, windows_abi);

    // 5. Call hook handler.
    //
    // NOTE: No `sub rsp, 8` alignment here — unlike pre/post trampolines which
    // are entered via JMP from a CALL site (RSP ≡ 8 mod 16), the return
    // trampoline is entered via RET (RSP ≡ 0 mod 16).  After the red zone
    // skip (128 or 0) and RegContext allocation (144), RSP is still ≡ 0 mod 16,
    // which is the correct alignment for CALL.
    emit_pre_call_shadow(&mut ret_code, windows_abi);
    emit_hook_call(
        &mut ret_code,
        ret_tramp_va,
        &hook.source,
        hook_data_va,
        shellcode_va,
    );
    emit_post_call_shadow(&mut ret_code, windows_abi);

    // 7b. Fixup condition skip target.
    if let Some(fixup_pos) = cond_fixup {
        let target = ret_code.len();
        fixup_condition_skip(&mut ret_code, fixup_pos, target);
    }

    // 7c. Fixup toggle skip target.
    if let Some(fixup_pos) = toggle_fixup {
        let target = ret_code.len();
        fixup_toggle_skip(&mut ret_code, fixup_pos, target);
    }

    // 8. Restore all registers
    emit_restore_regs(&mut ret_code);

    // 9. Deallocate RegContext: add rsp, 144
    ret_code.extend_from_slice(&[0x48, 0x81, 0xC4]);
    ret_code.extend_from_slice(&(REG_CONTEXT_SIZE as u32).to_le_bytes());

    // 10. Restore red zone
    emit_red_zone_restore(&mut ret_code, windows_abi);

    // 11. Load saved original RA from data slot into rax, push, restore rax, ret
    // push rax (save rax)
    ret_code.push(0x50);

    // mov rax, [rip+disp] — load saved original RA from data slot
    let rip_after_load = ret_tramp_va + ret_code.len() as u64 + 7;
    let load_disp = (return_slot_va as i64 - rip_after_load as i64) as i32;
    ret_code.extend_from_slice(&[0x48, 0x8B, 0x05]); // mov rax, [rip+disp32]
    ret_code.extend_from_slice(&load_disp.to_le_bytes());

    // xchg rax, [rsp] — push RA, restore rax
    // xchg rax, [rsp]: 48 87 04 24
    ret_code.extend_from_slice(&[0x48, 0x87, 0x04, 0x24]);

    // ret — return to original caller
    ret_code.push(0xC3);

    let ret_tramp = Trampoline {
        va: ret_tramp_va,
        code: ret_code,
    };

    Ok((entry_tramp, ret_tramp))
}

// ===========================================================================
// PE32 (32-bit x86) hook trampoline generation
// ===========================================================================

/// Relocate displaced instructions from `orig_ip` to `new_ip` using 32-bit decoding.
fn relocate_instructions_pe32(bytes: &[u8], orig_ip: u64, new_ip: u64) -> Result<Vec<u8>> {
    use iced_x86::{BlockEncoder, BlockEncoderOptions, Decoder, DecoderOptions, InstructionBlock};

    let mut decoder = Decoder::with_ip(32, bytes, orig_ip, DecoderOptions::NONE);
    let mut instructions = Vec::new();

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.is_invalid() {
            break;
        }
        instructions.push(instr);
    }

    if instructions.is_empty() {
        anyhow::bail!("no valid 32-bit instructions to relocate");
    }

    let block = InstructionBlock::new(&instructions, new_ip);
    let result = BlockEncoder::encode(32, block, BlockEncoderOptions::NONE)
        .context("iced-x86 BlockEncoder failed (32-bit)")?;

    Ok(result.code_buffer)
}

/// Map an x86 (32-bit) register name to its byte offset in the PE32 RegContext struct.
///
/// PE32 RegContext layout (40 bytes):
///   edi=+0, esi=+4, ebp=+8, esp_saved=+12, ebx=+16,
///   edx=+20, ecx=+24, eax=+28, eip=+32, eflags=+36
///
/// This mirrors the pushad order (eax first pushed = highest offset).
fn reg_context_offset_pe32(reg: &str) -> Result<i32> {
    match reg {
        "edi" => Ok(0),
        "esi" => Ok(4),
        "ebp" => Ok(8),
        "esp" => Ok(12),
        "ebx" => Ok(16),
        "edx" => Ok(20),
        "ecx" => Ok(24),
        "eax" => Ok(28),
        "eip" => Ok(32),
        "eflags" => Ok(36),
        _ => anyhow::bail!("unknown x86 (PE32) register '{}' — use eax-edi", reg),
    }
}

/// Emit PE32 save-all-registers: pushfd + pushad + store EIP field.
///
/// After this sequence, ESP points to a 40-byte RegContext with the layout:
///   [esp+0]=edi, [esp+4]=esi, [esp+8]=ebp, [esp+12]=esp_saved,
///   [esp+16]=ebx, [esp+20]=edx, [esp+24]=ecx, [esp+28]=eax,
///   [esp+32]=eip_field, [esp+36]=eflags
///
/// We push eflags and eip first (at highest addresses), then pushad fills the rest.
fn emit_save_regs_pe32(code: &mut Vec<u8>, target_va: u64) {
    // pushfd — push EFLAGS (4 bytes on stack)
    code.push(0x9C);

    // push imm32 (target_va as EIP field) — 5 bytes
    code.push(0x68);
    code.extend_from_slice(&(target_va as u32).to_le_bytes());

    // pushad — pushes eax, ecx, edx, ebx, esp_original, ebp, esi, edi (32 bytes)
    // This gives us the correct layout: edi at lowest address.
    code.push(0x60);
}

/// Emit PE32 restore-all-registers: popad + skip EIP + popfd.
fn emit_restore_regs_pe32(code: &mut Vec<u8>) {
    // popad — restores edi, esi, ebp, (skips esp), ebx, edx, ecx, eax (32 bytes)
    code.push(0x61);

    // add esp, 4 — skip stored EIP field
    code.extend_from_slice(&[0x83, 0xC4, 0x04]);

    // popfd — restore EFLAGS
    code.push(0x9D);
}

/// Emit PE32 hook call: load handler address into eax, push arg, call eax, clean up.
///
/// cdecl convention: push arguments right-to-left, caller cleans stack.
fn emit_hook_call_pe32(
    code: &mut Vec<u8>,
    source: &HookSource,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    extra_args: &[u32],
) {
    // Push extra args in reverse order (right-to-left for cdecl).
    for &arg in extra_args.iter().rev() {
        // push imm32
        code.push(0x68);
        code.extend_from_slice(&arg.to_le_bytes());
    }

    // Push arg1 = address of RegContext. The RegContext is at a fixed offset
    // from current esp (extra_args.len() dwords were pushed above it).
    //
    // lea eax, [esp + offset_to_regctx]
    // push eax
    let offset_to_regctx = (extra_args.len() as u8) * 4;
    if offset_to_regctx == 0 {
        // push esp — RegContext is right at esp
        code.push(0x54);
    } else {
        // lea eax, [esp + offset]
        code.extend_from_slice(&[0x8D, 0x44, 0x24, offset_to_regctx]);
        // push eax
        code.push(0x50);
    }

    // Load handler address into eax.
    match source {
        HookSource::LibrarySymbol {
            data_slot_index, ..
        } => {
            // mov eax, [abs_addr] — load handler pointer from data slot (32-bit absolute)
            let slot_va = hook_data_va + (*data_slot_index as u64) * 4;
            code.push(0xA1); // mov eax, [moffs32]
            code.extend_from_slice(&(slot_va as u32).to_le_bytes());
        }
        HookSource::Shellcode(_) => {
            // mov eax, imm32 — shellcode address
            let sc_va = shellcode_va.expect("shellcode_va must be set for shellcode hooks");
            code.push(0xB8); // mov eax, imm32
            code.extend_from_slice(&(sc_va as u32).to_le_bytes());
        }
    }

    // call eax
    code.extend_from_slice(&[0xFF, 0xD0]);

    // cdecl cleanup: caller pops args (1 + extra_args.len() dwords)
    let cleanup = (1 + extra_args.len() as u8) * 4;
    if cleanup <= 127 {
        // add esp, imm8
        code.extend_from_slice(&[0x83, 0xC4, cleanup]);
    } else {
        // add esp, imm32
        code.extend_from_slice(&[0x81, 0xC4]);
        code.extend_from_slice(&(cleanup as u32).to_le_bytes());
    }
}

/// Emit PE32 toggle check: load byte from absolute address, test, jz to skip.
///
/// Returns the byte offset of the jz rel32 displacement for fixup.
fn emit_toggle_check_pe32(code: &mut Vec<u8>, toggle_va: u64) -> usize {
    // mov al, [abs_addr]  ; A0 <addr32>  (5 bytes)
    code.push(0xA0);
    code.extend_from_slice(&(toggle_va as u32).to_le_bytes());

    // test al, al  ; 84 C0  (2 bytes)
    code.extend_from_slice(&[0x84, 0xC0]);

    // jz rel32  ; 0F 84 <disp32>  (6 bytes)
    code.extend_from_slice(&[0x0F, 0x84]);
    let fixup_pos = code.len();
    code.extend_from_slice(&0i32.to_le_bytes()); // placeholder
    fixup_pos
}

/// Emit PE32 condition check: load register from RegContext, compare, emit Jcc to skip.
///
/// Returns the byte offset for fixup.
fn emit_condition_check_pe32(code: &mut Vec<u8>, condition: &HookCondition) -> Result<usize> {
    let reg_offset = reg_context_offset_pe32(&condition.register)?;

    // mov eax, [esp + reg_offset]  — load from RegContext on stack
    if (-128..=127).contains(&reg_offset) {
        // mov eax, [esp + disp8]  ; 8B 44 24 <disp8>
        code.extend_from_slice(&[0x8B, 0x44, 0x24, reg_offset as u8]);
    } else {
        // mov eax, [esp + disp32]  ; 8B 84 24 <disp32>
        code.extend_from_slice(&[0x8B, 0x84, 0x24]);
        code.extend_from_slice(&reg_offset.to_le_bytes());
    }

    if condition.op == CondOp::BitSet || condition.op == CondOp::BitClear {
        // test eax, imm32  ; A9 <imm32>
        code.push(0xA9);
        code.extend_from_slice(&(condition.value as u32).to_le_bytes());
    } else {
        // cmp eax, imm32  ; 3D <imm32>
        code.push(0x3D);
        code.extend_from_slice(&(condition.value as u32).to_le_bytes());
    }

    // Jcc rel32 — skip hook when condition is FALSE (inverse logic).
    let jcc_opcode = match condition.op {
        CondOp::Eq => 0x85,       // JNE
        CondOp::Ne => 0x84,       // JE
        CondOp::Gt => 0x86,       // JBE
        CondOp::Gte => 0x82,      // JB
        CondOp::Lt => 0x83,       // JAE
        CondOp::Lte => 0x87,      // JA
        CondOp::BitSet => 0x84,   // JE
        CondOp::BitClear => 0x85, // JNE
    };

    code.extend_from_slice(&[0x0F, jcc_opcode]);
    let fixup_pos = code.len();
    code.extend_from_slice(&0i32.to_le_bytes()); // placeholder
    Ok(fixup_pos)
}

/// PE32 hook trampoline dispatch.
fn generate_pe32_hook_trampoline(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
) -> Result<Trampoline> {
    match hook.mode {
        HookMode::Pre => generate_pe32_pre_hook_trampoline(
            trampoline_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
        ),
        HookMode::Post => generate_pe32_post_hook_trampoline(
            trampoline_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
        ),
        HookMode::Replace => generate_pe32_replace_hook_trampoline(
            trampoline_va,
            hook,
            hook_data_va,
            shellcode_va,
            toggle_va,
        ),
        HookMode::Return => anyhow::bail!(
            "Return mode hooks must use generate_return_hook_trampolines() (hook at 0x{:x})",
            hook.target_va,
        ),
    }
}

/// Generate a PE32 pre-hook trampoline.
///
/// Sequence:
/// 1. pushfd + push eip_field + pushad (40 bytes = RegContext)
/// 2. [toggle check]
/// 3. [condition check]
/// 4. push esp (arg1=&RegContext), load handler, call, cleanup
/// 5. popad + skip eip + popfd
/// 6. Relocated displaced instructions
/// 7. JMP return_va
fn generate_pe32_pre_hook_trampoline(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
) -> Result<Trampoline> {
    let return_va = hook.target_va + hook.displaced_len as u64;
    let mut code = Vec::with_capacity(256);

    // 1. Save registers
    emit_save_regs_pe32(&mut code, hook.target_va);

    // 2. Toggle check
    let toggle_fixup = toggle_va.map(|tva| emit_toggle_check_pe32(&mut code, tva));

    // 3. Condition check
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_pe32(&mut code, cond)?)
    } else {
        None
    };

    // 4. Call handler
    emit_hook_call_pe32(&mut code, &hook.source, hook_data_va, shellcode_va, &[]);

    // 5. Fixup skips
    if let Some(fixup_pos) = cond_fixup {
        let target = code.len();
        fixup_condition_skip(&mut code, fixup_pos, target);
    }
    if let Some(fixup_pos) = toggle_fixup {
        let target = code.len();
        fixup_toggle_skip(&mut code, fixup_pos, target);
    }

    // 6. Restore registers
    emit_restore_regs_pe32(&mut code);

    // 7. Displaced instructions
    let displaced_va = trampoline_va + code.len() as u64;
    let relocated = relocate_instructions_pe32(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate 32-bit displaced instructions for hook at 0x{:x}",
                hook.target_va,
            )
        })?;
    code.extend_from_slice(&relocated);

    // 8. JMP return_va
    let jmp_ip = trampoline_va + code.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

/// Generate a PE32 post-hook trampoline.
///
/// Same as pre-hook but displaced instructions execute BEFORE the hook call.
fn generate_pe32_post_hook_trampoline(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
) -> Result<Trampoline> {
    let return_va = hook.target_va + hook.displaced_len as u64;
    let mut code = Vec::with_capacity(256);

    // 1. Displaced instructions first
    let displaced_va = trampoline_va + code.len() as u64;
    let relocated = relocate_instructions_pe32(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate 32-bit displaced instructions for post-hook at 0x{:x}",
                hook.target_va,
            )
        })?;
    code.extend_from_slice(&relocated);

    // 2. Save registers
    emit_save_regs_pe32(&mut code, hook.target_va);

    // 3. Toggle check
    let toggle_fixup = toggle_va.map(|tva| emit_toggle_check_pe32(&mut code, tva));

    // 4. Condition check
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_pe32(&mut code, cond)?)
    } else {
        None
    };

    // 5. Call handler
    emit_hook_call_pe32(&mut code, &hook.source, hook_data_va, shellcode_va, &[]);

    // 6. Fixup skips
    if let Some(fixup_pos) = cond_fixup {
        let target = code.len();
        fixup_condition_skip(&mut code, fixup_pos, target);
    }
    if let Some(fixup_pos) = toggle_fixup {
        let target = code.len();
        fixup_toggle_skip(&mut code, fixup_pos, target);
    }

    // 7. Restore registers
    emit_restore_regs_pe32(&mut code);

    // 8. JMP return_va
    let jmp_ip = trampoline_va + code.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

/// Generate a PE32 replace-hook trampoline.
///
/// The handler receives (RegContext*, original_func_ptr) via cdecl.
/// After the handler returns, we restore registers and RET to caller.
/// An original-function stub is appended for the handler to call.
fn generate_pe32_replace_hook_trampoline(
    trampoline_va: u64,
    hook: &ResolvedHook,
    hook_data_va: u64,
    shellcode_va: Option<u64>,
    toggle_va: Option<u64>,
) -> Result<Trampoline> {
    let return_va = hook.target_va + hook.displaced_len as u64;
    let mut code = Vec::with_capacity(256);

    // 1. Save registers
    emit_save_regs_pe32(&mut code, hook.target_va);

    // 2. Toggle check
    let toggle_fixup = toggle_va.map(|tva| emit_toggle_check_pe32(&mut code, tva));

    // 3. Condition check
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_pe32(&mut code, cond)?)
    } else {
        None
    };

    // 4. Call handler with extra arg: stub_va (placeholder, fixed up later)
    // We need to know the stub address. Emit a placeholder push imm32 that we fix up.
    let stub_fixup_pos = code.len() + 1; // offset of the imm32 in "push imm32"
    code.push(0x68); // push imm32 (stub_va placeholder — arg2)
    code.extend_from_slice(&0u32.to_le_bytes());

    // push esp+4 — arg1 = &RegContext (shifted by 4 because we just pushed stub_va)
    // lea eax, [esp+4] ; push eax
    code.extend_from_slice(&[0x8D, 0x44, 0x24, 0x04]);
    code.push(0x50);

    // Load handler into eax and call
    match &hook.source {
        HookSource::LibrarySymbol {
            data_slot_index, ..
        } => {
            let slot_va = hook_data_va + (*data_slot_index as u64) * 4;
            code.push(0xA1); // mov eax, [moffs32]
            code.extend_from_slice(&(slot_va as u32).to_le_bytes());
        }
        HookSource::Shellcode(_) => {
            let sc_va = shellcode_va.expect("shellcode_va must be set for shellcode hooks");
            code.push(0xB8); // mov eax, imm32
            code.extend_from_slice(&(sc_va as u32).to_le_bytes());
        }
    }
    // call eax
    code.extend_from_slice(&[0xFF, 0xD0]);
    // cdecl cleanup: 2 args = 8 bytes
    code.extend_from_slice(&[0x83, 0xC4, 0x08]);

    // 5. Restore registers
    emit_restore_regs_pe32(&mut code);

    // 6. RET to caller
    code.push(0xC3);

    // === Fallback path (toggle/condition was false) ===
    let has_fallback = toggle_fixup.is_some() || cond_fixup.is_some();
    if has_fallback {
        let fallback_target = code.len();
        if let Some(fixup_pos) = toggle_fixup {
            fixup_toggle_skip(&mut code, fixup_pos, fallback_target);
        }
        if let Some(fixup_pos) = cond_fixup {
            fixup_condition_skip(&mut code, fixup_pos, fallback_target);
        }

        // Restore registers
        emit_restore_regs_pe32(&mut code);

        // Displaced instructions
        let displaced_va = trampoline_va + code.len() as u64;
        let relocated = relocate_instructions_pe32(
            &hook.displaced_bytes, hook.target_va, displaced_va,
        ).with_context(|| format!(
            "failed to relocate 32-bit displaced instructions for conditional replace hook at 0x{:x}",
            hook.target_va,
        ))?;
        code.extend_from_slice(&relocated);

        // JMP to continue original function
        let jmp_ip = trampoline_va + code.len() as u64 + 5;
        let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
        code.push(0xE9);
        code.extend_from_slice(&jmp_rel.to_le_bytes());
    }

    // === Original function stub ===
    let stub_va = trampoline_va + code.len() as u64;

    // Fix up the stub_va placeholder.
    code[stub_fixup_pos..stub_fixup_pos + 4].copy_from_slice(&(stub_va as u32).to_le_bytes());

    // Relocated displaced instructions
    let displaced_va = trampoline_va + code.len() as u64;
    let relocated = relocate_instructions_pe32(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate 32-bit displaced instructions for replace hook stub at 0x{:x}",
                hook.target_va,
            )
        })?;
    code.extend_from_slice(&relocated);

    // JMP to continue original function
    let jmp_ip = trampoline_va + code.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

/// PE32 chained hook trampoline: multiple hooks on the same VA (same mode).
fn generate_pe32_chained_hook_trampoline(
    trampoline_va: u64,
    hooks: &[&ResolvedHook],
    mode: HookMode,
    hook_data_va: u64,
    shellcode_vas: &[Option<u64>],
    toggle_vas: &[Option<u64>],
) -> Result<Trampoline> {
    let hook0 = hooks[0];
    let return_va = hook0.target_va + hook0.displaced_len as u64;
    let mut code = Vec::with_capacity(512);

    // For post mode, displaced instructions come first.
    if mode == HookMode::Post {
        let displaced_va = trampoline_va + code.len() as u64;
        let relocated =
            relocate_instructions_pe32(&hook0.displaced_bytes, hook0.target_va, displaced_va)
                .context("failed to relocate 32-bit displaced instructions for chained hook")?;
        code.extend_from_slice(&relocated);
    }

    // Save registers
    emit_save_regs_pe32(&mut code, hook0.target_va);

    // Call each hook handler
    for (i, hook) in hooks.iter().enumerate() {
        let toggle_fixup = toggle_vas[i].map(|tva| emit_toggle_check_pe32(&mut code, tva));

        let cond_fixup = if let Some(ref cond) = hook.condition {
            Some(emit_condition_check_pe32(&mut code, cond)?)
        } else {
            None
        };

        emit_hook_call_pe32(&mut code, &hook.source, hook_data_va, shellcode_vas[i], &[]);

        if let Some(fixup_pos) = cond_fixup {
            let target = code.len();
            fixup_condition_skip(&mut code, fixup_pos, target);
        }
        if let Some(fixup_pos) = toggle_fixup {
            let target = code.len();
            fixup_toggle_skip(&mut code, fixup_pos, target);
        }
    }

    // Restore registers
    emit_restore_regs_pe32(&mut code);

    match mode {
        HookMode::Pre => {
            let displaced_va = trampoline_va + code.len() as u64;
            let relocated = relocate_instructions_pe32(
                &hook0.displaced_bytes,
                hook0.target_va,
                displaced_va,
            )
            .context("failed to relocate 32-bit displaced instructions for chained pre-hook")?;
            code.extend_from_slice(&relocated);

            let jmp_ip = trampoline_va + code.len() as u64 + 5;
            let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
            code.push(0xE9);
            code.extend_from_slice(&jmp_rel.to_le_bytes());
        }
        HookMode::Post => {
            let jmp_ip = trampoline_va + code.len() as u64 + 5;
            let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
            code.push(0xE9);
            code.extend_from_slice(&jmp_rel.to_le_bytes());
        }
        HookMode::Replace => {
            code.push(0xC3); // RET

            // Original function stub
            let displaced_va = trampoline_va + code.len() as u64;
            let relocated =
                relocate_instructions_pe32(&hook0.displaced_bytes, hook0.target_va, displaced_va)
                    .context(
                    "failed to relocate 32-bit displaced instructions for chained replace hook",
                )?;
            code.extend_from_slice(&relocated);

            let jmp_ip = trampoline_va + code.len() as u64 + 5;
            let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
            code.push(0xE9);
            code.extend_from_slice(&jmp_rel.to_le_bytes());
        }
        _ => unreachable!(),
    }

    Ok(Trampoline {
        va: trampoline_va,
        code,
    })
}

/// Generate PE32 return hook trampolines (entry + return).
///
/// **Entry trampoline**: save eax, load original RA from [esp+4], store to data slot,
/// overwrite [esp+4] with return trampoline VA, restore eax, displaced, JMP into body.
///
/// **Return trampoline**: save regs, call handler, restore, load saved RA, push+ret.
///
/// The `return_slot_va` is the VA of a 4-byte data slot for the original return address.
#[allow(clippy::too_many_arguments)]
fn generate_return_hook_trampolines_pe32(
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

    // push eax (save scratch)
    entry.push(0x50);

    // mov eax, [esp+4] — load original return address
    // Stack: [saved_eax] [original_RA] ...
    entry.extend_from_slice(&[0x8B, 0x44, 0x24, 0x04]);

    // mov [return_slot_va], eax — store original RA to data slot (absolute address)
    entry.push(0xA3); // mov [moffs32], eax
    entry.extend_from_slice(&(return_slot_va as u32).to_le_bytes());

    // mov eax, ret_tramp_va — load return trampoline VA
    entry.push(0xB8); // mov eax, imm32
    entry.extend_from_slice(&(ret_tramp_va as u32).to_le_bytes());

    // mov [esp+4], eax — overwrite return address
    entry.extend_from_slice(&[0x89, 0x44, 0x24, 0x04]);

    // pop eax (restore scratch)
    entry.push(0x58);

    // Relocated displaced instructions
    let displaced_va = entry_va + entry.len() as u64;
    let relocated = relocate_instructions_pe32(&hook.displaced_bytes, hook.target_va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate 32-bit displaced instructions for return hook entry at 0x{:x}",
                hook.target_va,
            )
        })?;
    entry.extend_from_slice(&relocated);

    // JMP return_va (continue function body)
    let jmp_ip = entry_va + entry.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    entry.push(0xE9);
    entry.extend_from_slice(&jmp_rel.to_le_bytes());

    let entry_tramp = Trampoline {
        va: entry_va,
        code: entry,
    };

    // === Return trampoline ===
    let mut ret_code = Vec::with_capacity(256);

    // 1. Save registers
    emit_save_regs_pe32(&mut ret_code, hook.target_va);

    // 2. Toggle check
    let toggle_fixup = toggle_va.map(|tva| emit_toggle_check_pe32(&mut ret_code, tva));

    // 3. Condition check
    let cond_fixup = if let Some(ref cond) = hook.condition {
        Some(emit_condition_check_pe32(&mut ret_code, cond)?)
    } else {
        None
    };

    // 4. Call handler
    emit_hook_call_pe32(&mut ret_code, &hook.source, hook_data_va, shellcode_va, &[]);

    // 5. Fixup skips
    if let Some(fixup_pos) = cond_fixup {
        let target = ret_code.len();
        fixup_condition_skip(&mut ret_code, fixup_pos, target);
    }
    if let Some(fixup_pos) = toggle_fixup {
        let target = ret_code.len();
        fixup_toggle_skip(&mut ret_code, fixup_pos, target);
    }

    // 6. Restore registers
    emit_restore_regs_pe32(&mut ret_code);

    // 7. Load saved original RA from data slot, push, ret
    // push eax (save eax)
    ret_code.push(0x50);

    // mov eax, [return_slot_va] — load saved RA
    ret_code.push(0xA1); // mov eax, [moffs32]
    ret_code.extend_from_slice(&(return_slot_va as u32).to_le_bytes());

    // xchg eax, [esp] — swap: eax gets restored, saved RA goes to [esp] (for RET)
    ret_code.extend_from_slice(&[0x87, 0x04, 0x24]);

    // ret
    ret_code.push(0xC3);

    let ret_tramp = Trampoline {
        va: ret_tramp_va,
        code: ret_code,
    };

    Ok((entry_tramp, ret_tramp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::{HookMode, HookSource, ResolvedHook};

    fn make_test_hook(mode: HookMode) -> ResolvedHook {
        ResolvedHook {
            target_va: 0x401000,
            file_offset: 0x1000,
            // NOP; NOP; NOP; NOP; NOP — 5 bytes of NOPs
            displaced_bytes: vec![0x90, 0x90, 0x90, 0x90, 0x90],
            displaced_len: 5,
            mode,
            source: HookSource::Shellcode(vec![0xC3]), // ret
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
    fn test_pre_hook_trampoline_generates() {
        let hook = make_test_hook(HookMode::Pre);
        let shellcode_va = 0x500000u64;
        let result = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(shellcode_va),
            TargetAbi::SysV64,
            None,
        );
        assert!(result.is_ok());
        let tramp = result.expect("pre-hook trampoline generation should succeed");
        assert_eq!(tramp.va, 0x600000);
        // Should contain the red zone skip pattern
        assert!(tramp.code.windows(4).any(|w| w == [0x80, 0xFF, 0xFF, 0xFF]));
        // Should end with JMP rel32 (E9)
        assert_eq!(tramp.code[tramp.code.len() - 5], 0xE9);
    }

    #[test]
    fn test_post_hook_trampoline_generates() {
        let hook = make_test_hook(HookMode::Post);
        let shellcode_va = 0x500000u64;
        let result = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(shellcode_va),
            TargetAbi::SysV64,
            None,
        );
        assert!(result.is_ok());
        let tramp = result.expect("post-hook trampoline generation should succeed");
        // Post-hook: displaced instructions come first (NOPs), then red zone skip
        // First bytes should be the relocated NOPs
        assert_eq!(&tramp.code[0..5], &[0x90, 0x90, 0x90, 0x90, 0x90]);
    }

    #[test]
    fn test_replace_hook_trampoline_generates() {
        let hook = make_test_hook(HookMode::Replace);
        let shellcode_va = 0x500000u64;
        let result = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(shellcode_va),
            TargetAbi::SysV64,
            None,
        );
        assert!(result.is_ok());
        let tramp = result.expect("replace-hook trampoline generation should succeed");
        // Replace trampoline should be larger (has original function stub appended)
        assert!(tramp.code.len() > 100);
        // Main flow ends with RET (0xC3), stub ends with JMP rel32 (0xE9).
        let jmp_count = tramp
            .code
            .windows(1)
            .enumerate()
            .filter(|(i, w)| w[0] == 0xE9 && *i + 5 <= tramp.code.len())
            .count();
        assert!(
            jmp_count >= 1,
            "expected at least 1 JMP rel32 (stub), got {}",
            jmp_count
        );
        // Should contain a RET in the main flow (before the stub).
        assert!(
            tramp.code.contains(&0xC3),
            "expected RET in replace trampoline"
        );
    }

    #[test]
    fn test_library_symbol_hook() {
        let hook = ResolvedHook {
            target_va: 0x401000,
            file_offset: 0x1000,
            displaced_bytes: vec![0x90, 0x90, 0x90, 0x90, 0x90],
            displaced_len: 5,
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
        let result =
            generate_hook_trampoline(0x600000, &hook, 0x500000, None, TargetAbi::SysV64, None);
        assert!(result.is_ok());
        let tramp = result.expect("library symbol hook generation should succeed");
        // Should contain mov rax, [rip+disp] (48 8B 05) for indirect call
        assert!(tramp.code.windows(3).any(|w| w == [0x48, 0x8B, 0x05]));
    }

    #[test]
    fn test_emit_mov_to_rsp_small_offset() {
        let mut code = Vec::new();
        emit_mov_to_rsp_off(&mut code, 0x48, 0x89, 0x04, 0);
        // Should be 4 bytes: REX MOV ModR/M SIB disp8
        // 48 89 44 24 00
        assert_eq!(code.len(), 5);
        assert_eq!(code[0], 0x48);
        assert_eq!(code[1], 0x89);
    }

    #[test]
    fn test_emit_mov_to_rsp_large_offset() {
        let mut code = Vec::new();
        emit_mov_to_rsp_off(&mut code, 0x48, 0x89, 0x84, 136);
        // Should be 8 bytes: REX MOV ModR/M SIB disp32
        assert_eq!(code.len(), 8);
    }

    #[test]
    fn test_reg_context_offset_x86_64() {
        assert_eq!(
            reg_context_offset_x86_64("rax").expect("rax is a valid x86_64 register"),
            0
        );
        assert_eq!(
            reg_context_offset_x86_64("rdi").expect("rdi is a valid x86_64 register"),
            40
        );
        assert_eq!(
            reg_context_offset_x86_64("r15").expect("r15 is a valid x86_64 register"),
            120
        );
        assert!(reg_context_offset_x86_64("x0").is_err());
        assert!(reg_context_offset_x86_64("eax").is_err());
    }

    #[test]
    fn test_condition_check_emits_jcc() {
        let cond = HookCondition {
            register: "rdi".to_string(),
            op: CondOp::Eq,
            value: 42,
        };
        let mut code = Vec::new();
        let fixup_pos = emit_condition_check_x86_64(&mut code, &cond)
            .expect("condition check emission should succeed");
        // Should contain a Jcc (0F 85 = JNE for Eq condition)
        assert!(
            code.windows(2).any(|w| w == [0x0F, 0x85]),
            "expected JNE (0F 85) for Eq condition"
        );
        // Fixup pos should be within the code
        assert!(fixup_pos + 4 <= code.len());
    }

    #[test]
    fn test_condition_check_large_value() {
        let cond = HookCondition {
            register: "rax".to_string(),
            op: CondOp::Gte,
            value: 0x1_0000_0000, // > i32::MAX
        };
        let mut code = Vec::new();
        let _fixup = emit_condition_check_x86_64(&mut code, &cond)
            .expect("condition check with large value should succeed");
        // Should contain mov rcx, imm64 (48 B9)
        assert!(
            code.windows(2).any(|w| w == [0x48, 0xB9]),
            "expected MOV RCX, imm64 for large value"
        );
    }

    #[test]
    fn test_condition_check_bit_set() {
        let cond = HookCondition {
            register: "rsi".to_string(),
            op: CondOp::BitSet,
            value: 0x80, // bit 7
        };
        let mut code = Vec::new();
        let _fixup = emit_condition_check_x86_64(&mut code, &cond)
            .expect("condition check for BitSet should succeed");
        // Should contain TEST rax, imm32 (48 A9)
        assert!(
            code.windows(2).any(|w| w == [0x48, 0xA9]),
            "expected TEST RAX, imm32 for BitSet"
        );
        // Should have JE (0F 84) — skip if ZF (bits not set)
        assert!(
            code.windows(2).any(|w| w == [0x0F, 0x84]),
            "expected JE (0F 84) for BitSet condition"
        );
    }

    #[test]
    fn test_conditional_pre_hook_generates() {
        let mut hook = make_test_hook(HookMode::Pre);
        hook.condition = Some(HookCondition {
            register: "rdi".to_string(),
            op: CondOp::Eq,
            value: 42,
        });
        let shellcode_va = 0x500000u64;
        let result = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(shellcode_va),
            TargetAbi::SysV64,
            None,
        );
        assert!(
            result.is_ok(),
            "conditional pre-hook failed: {:?}",
            result.err()
        );
        let tramp = result.expect("conditional pre-hook generation should succeed");
        // Should be larger than unconditional (condition check adds bytes)
        let uncond_hook = make_test_hook(HookMode::Pre);
        let uncond = generate_hook_trampoline(
            0x600000,
            &uncond_hook,
            0x500000,
            Some(shellcode_va),
            TargetAbi::SysV64,
            None,
        )
        .expect("unconditional pre-hook generation should succeed");
        assert!(
            tramp.code.len() > uncond.code.len(),
            "conditional trampoline should be larger: {} vs {}",
            tramp.code.len(),
            uncond.code.len()
        );
    }

    #[test]
    fn test_conditional_replace_hook_has_fallback() {
        let mut hook = make_test_hook(HookMode::Replace);
        hook.condition = Some(HookCondition {
            register: "rdi".to_string(),
            op: CondOp::Ne,
            value: 0,
        });
        let shellcode_va = 0x500000u64;
        let result = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(shellcode_va),
            TargetAbi::SysV64,
            None,
        );
        assert!(
            result.is_ok(),
            "conditional replace hook failed: {:?}",
            result.err()
        );
        let tramp = result.expect("conditional replace hook generation should succeed");
        // Conditional replace trampoline should be substantially larger
        // (has a fallback path with restore + displaced + JMP)
        let uncond_hook = make_test_hook(HookMode::Replace);
        let uncond = generate_hook_trampoline(
            0x600000,
            &uncond_hook,
            0x500000,
            Some(shellcode_va),
            TargetAbi::SysV64,
            None,
        )
        .expect("unconditional replace hook generation should succeed");
        assert!(
            tramp.code.len() > uncond.code.len() + 50,
            "conditional replace trampoline should be much larger: {} vs {}",
            tramp.code.len(),
            uncond.code.len()
        );
    }

    #[test]
    fn test_fixup_condition_skip() {
        let mut code = vec![0u8; 20];
        // Simulate Jcc at position 6: [0F 8x] at 4-5, rel32 at 6-9
        // Want to skip to position 16
        fixup_condition_skip(&mut code, 6, 16);
        let rel32 = i32::from_le_bytes(code[6..10].try_into().expect("slice is 4 bytes"));
        // rel32 should be 16 - (6 + 4) = 6
        assert_eq!(rel32, 6);
    }

    #[test]
    fn test_toggle_check_emitted() {
        // A trampoline with a toggle_va should be larger than one without.
        let hook = make_test_hook(HookMode::Pre);
        let no_toggle = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::SysV64,
            None,
        )
        .expect("no-toggle trampoline generation should succeed");
        let with_toggle = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::SysV64,
            Some(0x580000),
        )
        .expect("toggle trampoline generation should succeed");
        assert!(
            with_toggle.code.len() > no_toggle.code.len(),
            "toggle trampoline ({}) should be larger than no-toggle ({})",
            with_toggle.code.len(),
            no_toggle.code.len(),
        );
    }

    #[test]
    fn test_toggle_disabled_initial_value() {
        // Verify that a hook with enabled=false has initial_enabled=false
        // in the ResolvedHook, producing a 0 byte (not testing trampoline
        // code directly, but the data flow that produces the toggle byte).
        let hook = ResolvedHook {
            target_va: 0x401000,
            file_offset: 0x1000,
            displaced_bytes: vec![0x90; 5],
            displaced_len: 5,
            mode: HookMode::Pre,
            source: HookSource::Shellcode(vec![0xC3]),
            condition: None,
            toggle_index: 0,
            initial_enabled: false,
            return_slot_index: None,
        };
        assert!(!hook.initial_enabled);
        assert_eq!(hook.toggle_index, 0);
    }

    #[test]
    fn test_toggle_check_with_condition() {
        // Trampoline with both toggle and condition should be larger than
        // either alone.
        let mut hook = make_test_hook(HookMode::Pre);
        hook.condition = Some(HookCondition {
            register: "rdi".to_string(),
            op: CondOp::Eq,
            value: 42,
        });
        let cond_only = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::SysV64,
            None,
        )
        .expect("condition-only trampoline generation should succeed");

        hook.condition = None;
        let toggle_only = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::SysV64,
            Some(0x580000),
        )
        .expect("toggle-only trampoline generation should succeed");

        hook.condition = Some(HookCondition {
            register: "rdi".to_string(),
            op: CondOp::Eq,
            value: 42,
        });
        let both = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::SysV64,
            Some(0x580000),
        )
        .expect("toggle+condition trampoline generation should succeed");

        assert!(
            both.code.len() > cond_only.code.len(),
            "toggle+condition ({}) should be larger than condition-only ({})",
            both.code.len(),
            cond_only.code.len(),
        );
        assert!(
            both.code.len() > toggle_only.code.len(),
            "toggle+condition ({}) should be larger than toggle-only ({})",
            both.code.len(),
            toggle_only.code.len(),
        );
    }

    #[test]
    fn test_return_hook_generates_two_trampolines() {
        let hook = make_test_hook(HookMode::Return);
        let entry_va = 0x600000u64;
        let ret_tramp_va = 0x600100u64;
        let return_slot_va = 0x580000u64;
        let result = generate_return_hook_trampolines(
            entry_va,
            ret_tramp_va,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::SysV64,
            None,
            return_slot_va,
        );
        assert!(result.is_ok(), "return hook failed: {:?}", result.err());
        let (entry, ret_tramp) = result.expect("return hook trampoline generation should succeed");
        assert_eq!(entry.va, entry_va);
        assert_eq!(ret_tramp.va, ret_tramp_va);
        // Both should have non-empty code
        assert!(
            !entry.code.is_empty(),
            "entry trampoline should not be empty"
        );
        assert!(
            !ret_tramp.code.is_empty(),
            "return trampoline should not be empty"
        );
    }

    #[test]
    fn test_return_hook_entry_trampoline_structure() {
        let hook = make_test_hook(HookMode::Return);
        let entry_va = 0x600000u64;
        let ret_tramp_va = 0x600100u64;
        let return_slot_va = 0x580000u64;
        let (entry, _) = generate_return_hook_trampolines(
            entry_va,
            ret_tramp_va,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::SysV64,
            None,
            return_slot_va,
        )
        .expect("return hook entry trampoline generation should succeed");
        // Entry trampoline should start with push rax (0x50)
        assert_eq!(entry.code[0], 0x50, "entry should start with push rax");
        // Entry trampoline should end with JMP rel32 (E9)
        assert_eq!(
            entry.code[entry.code.len() - 5],
            0xE9,
            "entry should end with JMP rel32"
        );
        // Should contain pop rax (0x58) before the displaced instructions
        assert!(entry.code.contains(&0x58), "entry should contain pop rax");
    }

    #[test]
    fn test_return_hook_return_trampoline_structure() {
        let hook = make_test_hook(HookMode::Return);
        let entry_va = 0x600000u64;
        let ret_tramp_va = 0x600100u64;
        let return_slot_va = 0x580000u64;
        let (_, ret_tramp) = generate_return_hook_trampolines(
            entry_va,
            ret_tramp_va,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::SysV64,
            None,
            return_slot_va,
        )
        .expect("return hook return trampoline generation should succeed");
        // Return trampoline should end with RET (0xC3)
        assert_eq!(
            *ret_tramp
                .code
                .last()
                .expect("return trampoline code should not be empty"),
            0xC3,
            "return trampoline should end with RET"
        );
        // Should contain xchg rax, [rsp] (48 87 04 24)
        assert!(
            ret_tramp
                .code
                .windows(4)
                .any(|w| w == [0x48, 0x87, 0x04, 0x24]),
            "return trampoline should contain xchg rax, [rsp]"
        );
        // Should contain the red zone skip pattern
        assert!(
            ret_tramp
                .code
                .windows(4)
                .any(|w| w == [0x80, 0xFF, 0xFF, 0xFF]),
            "return trampoline should contain red zone skip"
        );
    }

    #[test]
    fn test_return_mode_rejected_in_chain() {
        let hook_pre = make_test_hook(HookMode::Pre);
        let hook_ret = make_test_hook(HookMode::Return);
        let hooks: Vec<&ResolvedHook> = vec![&hook_pre, &hook_ret];
        let sc_vas = vec![Some(0x500000u64); 2];
        let toggle_vas = vec![None; 2];
        let result = generate_chained_hook_trampoline(
            0x600000,
            &hooks,
            0x500000,
            &sc_vas,
            TargetAbi::SysV64,
            &toggle_vas,
        );
        assert!(result.is_err(), "return mode should be rejected in chain");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("return mode"),
            "error should mention return mode: {}",
            err
        );
    }

    #[test]
    fn test_return_mode_rejected_in_single_via_generate_hook_trampoline() {
        let hook = make_test_hook(HookMode::Return);
        let result = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::SysV64,
            None,
        );
        assert!(
            result.is_err(),
            "Return mode should be rejected by generate_hook_trampoline"
        );
    }

    // -----------------------------------------------------------------------
    // PE32 (32-bit x86) hook trampoline tests
    // -----------------------------------------------------------------------

    fn make_test_hook_pe32(mode: HookMode) -> ResolvedHook {
        ResolvedHook {
            target_va: 0x401000,
            file_offset: 0x1000,
            // NOP; NOP; NOP; NOP; NOP — 5 bytes of NOPs (same encoding in 32-bit)
            displaced_bytes: vec![0x90, 0x90, 0x90, 0x90, 0x90],
            displaced_len: 5,
            mode,
            source: HookSource::Shellcode(vec![0xC3]), // ret
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
    fn test_pe32_pre_hook_generates() {
        let hook = make_test_hook_pe32(HookMode::Pre);
        let shellcode_va = 0x500000u64;
        let result = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(shellcode_va),
            TargetAbi::Win32,
            None,
        );
        assert!(result.is_ok(), "PE32 pre-hook failed: {:?}", result.err());
        let tramp = result.expect("PE32 pre-hook trampoline generation should succeed");
        assert_eq!(tramp.va, 0x600000);
        assert!(!tramp.code.is_empty());
        // Should end with JMP rel32 (E9)
        assert_eq!(
            tramp.code[tramp.code.len() - 5],
            0xE9,
            "PE32 pre-hook should end with JMP rel32"
        );
    }

    #[test]
    fn test_pe32_hook_uses_32bit_register_saves() {
        let hook = make_test_hook_pe32(HookMode::Pre);
        let result = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::Win32,
            None,
        );
        let tramp = result.expect("PE32 register save hook generation should succeed");
        // Should contain pushfd (0x9C) — 32-bit flags save
        assert!(
            tramp.code.contains(&0x9C),
            "PE32 hook should contain pushfd (0x9C)"
        );
        // Should contain pushad (0x60) — 32-bit register save
        assert!(
            tramp.code.contains(&0x60),
            "PE32 hook should contain pushad (0x60)"
        );
        // Should contain popad (0x61) — 32-bit register restore
        assert!(
            tramp.code.contains(&0x61),
            "PE32 hook should contain popad (0x61)"
        );
        // Should contain popfd (0x9D) — 32-bit flags restore
        assert!(
            tramp.code.contains(&0x9D),
            "PE32 hook should contain popfd (0x9D)"
        );
        // Should NOT contain any REX prefixes (0x48, 0x4C) used by x86_64
        assert!(
            !tramp.code.windows(2).any(|w| w == [0x48, 0x81]),
            "PE32 hook should not contain x86_64 REX-prefixed instructions"
        );
    }

    #[test]
    fn test_pe32_hook_uses_absolute_address() {
        let hook = ResolvedHook {
            target_va: 0x401000,
            file_offset: 0x1000,
            displaced_bytes: vec![0x90; 5],
            displaced_len: 5,
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
        let result =
            generate_hook_trampoline(0x600000, &hook, 0x500000, None, TargetAbi::Win32, None);
        let tramp = result.expect("PE32 absolute address hook generation should succeed");
        // Should contain mov eax, [moffs32] (0xA1) — absolute address load
        assert!(
            tramp.code.contains(&0xA1),
            "PE32 hook should contain absolute address load (0xA1 = mov eax, [addr])"
        );
        // Should NOT contain RIP-relative load (48 8B 05)
        assert!(
            !tramp.code.windows(3).any(|w| w == [0x48, 0x8B, 0x05]),
            "PE32 hook should NOT contain RIP-relative addressing"
        );
    }

    #[test]
    fn test_pe32_post_hook_generates() {
        let hook = make_test_hook_pe32(HookMode::Post);
        let result = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::Win32,
            None,
        );
        assert!(result.is_ok(), "PE32 post-hook failed: {:?}", result.err());
        let tramp = result.expect("PE32 post-hook trampoline generation should succeed");
        // Post-hook: displaced instructions (NOPs) come first
        assert_eq!(
            &tramp.code[0..5],
            &[0x90, 0x90, 0x90, 0x90, 0x90],
            "PE32 post-hook should start with displaced NOPs"
        );
    }

    #[test]
    fn test_pe32_replace_hook_generates() {
        let hook = make_test_hook_pe32(HookMode::Replace);
        let result = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::Win32,
            None,
        );
        assert!(
            result.is_ok(),
            "PE32 replace-hook failed: {:?}",
            result.err()
        );
        let tramp = result.expect("PE32 replace-hook trampoline generation should succeed");
        // Replace trampoline should contain RET (0xC3)
        assert!(
            tramp.code.contains(&0xC3),
            "PE32 replace-hook should contain RET"
        );
        // Should have at least one JMP rel32 (for the original function stub)
        let has_jmp = tramp
            .code
            .windows(1)
            .enumerate()
            .any(|(i, w)| w[0] == 0xE9 && i + 5 <= tramp.code.len());
        assert!(has_jmp, "PE32 replace-hook should have JMP rel32 in stub");
    }

    #[test]
    fn test_pe32_return_hook_generates() {
        let hook = make_test_hook_pe32(HookMode::Return);
        let result = generate_return_hook_trampolines(
            0x600000,
            0x600100,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::Win32,
            None,
            0x580000,
        );
        assert!(
            result.is_ok(),
            "PE32 return hook failed: {:?}",
            result.err()
        );
        let (entry, ret_tramp) =
            result.expect("PE32 return hook trampoline generation should succeed");
        assert!(!entry.code.is_empty());
        assert!(!ret_tramp.code.is_empty());
        // Entry starts with push eax (0x50)
        assert_eq!(entry.code[0], 0x50, "PE32 entry should start with push eax");
        // Return trampoline ends with RET (0xC3)
        assert_eq!(
            *ret_tramp
                .code
                .last()
                .expect("PE32 return trampoline code should not be empty"),
            0xC3,
            "PE32 return trampoline should end with RET"
        );
    }

    #[test]
    fn test_pe32_reg_context_offsets() {
        assert_eq!(
            reg_context_offset_pe32("eax").expect("eax is a valid PE32 register"),
            28
        );
        assert_eq!(
            reg_context_offset_pe32("edi").expect("edi is a valid PE32 register"),
            0
        );
        assert_eq!(
            reg_context_offset_pe32("eflags").expect("eflags is a valid PE32 register"),
            36
        );
        assert!(reg_context_offset_pe32("rax").is_err());
        assert!(reg_context_offset_pe32("r8").is_err());
    }

    #[test]
    fn test_pe32_toggle_check() {
        let hook = make_test_hook_pe32(HookMode::Pre);
        let no_toggle = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::Win32,
            None,
        )
        .expect("PE32 no-toggle trampoline generation should succeed");
        let with_toggle = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::Win32,
            Some(0x580000),
        )
        .expect("PE32 toggle trampoline generation should succeed");
        assert!(
            with_toggle.code.len() > no_toggle.code.len(),
            "PE32 toggle trampoline ({}) should be larger than no-toggle ({})",
            with_toggle.code.len(),
            no_toggle.code.len()
        );
        // Toggle check should use absolute load: mov al, [addr] (0xA0)
        assert!(
            with_toggle.code.contains(&0xA0),
            "PE32 toggle check should use absolute byte load (0xA0)"
        );
    }

    #[test]
    fn test_pe32_condition_check() {
        let mut hook = make_test_hook_pe32(HookMode::Pre);
        hook.condition = Some(HookCondition {
            register: "eax".to_string(),
            op: CondOp::Eq,
            value: 42,
        });
        let result = generate_hook_trampoline(
            0x600000,
            &hook,
            0x500000,
            Some(0x500000),
            TargetAbi::Win32,
            None,
        );
        assert!(
            result.is_ok(),
            "PE32 conditional hook failed: {:?}",
            result.err()
        );
        let tramp = result.expect("PE32 conditional hook generation should succeed");
        // Should contain JNE (0F 85) for Eq condition
        assert!(
            tramp.code.windows(2).any(|w| w == [0x0F, 0x85]),
            "PE32 conditional hook should contain JNE (0F 85)"
        );
    }

    #[test]
    fn test_pe32_chained_pre_hooks() {
        let hook1 = make_test_hook_pe32(HookMode::Pre);
        let hook2 = make_test_hook_pe32(HookMode::Pre);
        let hooks: Vec<&ResolvedHook> = vec![&hook1, &hook2];
        let sc_vas = vec![Some(0x500000u64); 2];
        let toggle_vas = vec![None; 2];
        let result = generate_chained_hook_trampoline(
            0x600000,
            &hooks,
            0x500000,
            &sc_vas,
            TargetAbi::Win32,
            &toggle_vas,
        );
        assert!(
            result.is_ok(),
            "PE32 chained hooks failed: {:?}",
            result.err()
        );
        let tramp = result.expect("PE32 chained hooks generation should succeed");
        assert!(!tramp.code.is_empty());
        // Should contain two call eax sequences (FF D0)
        let call_count = tramp.code.windows(2).filter(|w| *w == [0xFF, 0xD0]).count();
        assert_eq!(
            call_count, 2,
            "PE32 chained hooks should have 2 call eax sequences"
        );
    }
}
