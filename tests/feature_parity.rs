//! Feature parity tests across all 4 architecture/platform combinations.
//!
//! Exercises the truant code generators for:
//! - x86_64 ELF
//! - x86_64 Mach-O
//! - AArch64 ELF
//! - AArch64 Mach-O
//!
//! These are unit-level parity tests that verify the code generators produce
//! correct output for each platform combo without requiring live binaries.

use std::path::Path;

use truant::disasm::BasicBlock;
use truant::hook_preload;
use truant::hook_trampoline::{
    ReturnHookContext, TargetAbi, generate_chained_hook_trampoline, generate_hook_trampoline,
    generate_return_hook_trampolines,
};
use truant::hooks::{CondOp, HookCondition, HookMode, HookSource, ResolvedHook};
use truant::macho_trampoline;
#[cfg(feature = "coverage")]
use truant::preload;
#[cfg(feature = "coverage")]
use truant::sidecar_preload;
use truant::trampoline::{self, PersistentWrapperParams};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// 5-byte x86_64 NOP (NOP dword [rax+rax*1+0x00]).
const X86_NOP5: [u8; 5] = [0x0F, 0x1F, 0x44, 0x00, 0x00];

/// 4-byte AArch64 NOP.
const AARCH64_NOP: [u8; 4] = [0x1F, 0x20, 0x03, 0xD5];

/// Build a minimal ResolvedHook for testing.
fn make_hook(mode: HookMode, displaced: &[u8]) -> ResolvedHook {
    ResolvedHook {
        target_va: 0x40_1000,
        file_offset: 0x1000,
        displaced_bytes: displaced.to_vec(),
        displaced_len: displaced.len(),
        mode,
        source: HookSource::Shellcode(vec![0xCC]), // INT3 as dummy shellcode
        condition: None,
        toggle_index: 0,
        initial_enabled: true,
        return_slot_index: None,
    }
}

/// Build a ResolvedHook with a condition.
fn make_conditional_hook(mode: HookMode, displaced: &[u8], register: &str) -> ResolvedHook {
    let mut hook = make_hook(mode, displaced);
    hook.condition = Some(HookCondition {
        register: register.to_string(),
        op: CondOp::Gte,
        value: 65536,
    });
    hook
}

/// Build a ResolvedHook for Return mode.
fn make_return_hook(displaced: &[u8]) -> ResolvedHook {
    let mut hook = make_hook(HookMode::Return, displaced);
    hook.return_slot_index = Some(0);
    hook
}

/// Build a ResolvedHook with a library handler (for preload tests).
fn make_library_hook(mode: HookMode, displaced: &[u8], name: &str) -> ResolvedHook {
    ResolvedHook {
        target_va: 0x40_1000,
        file_offset: 0x1000,
        displaced_bytes: displaced.to_vec(),
        displaced_len: displaced.len(),
        mode,
        source: HookSource::LibrarySymbol {
            name: name.to_string(),
            data_slot_index: 0,
        },
        condition: None,
        toggle_index: 0,
        initial_enabled: true,
        return_slot_index: None,
    }
}

/// Build a BasicBlock for coverage trampoline tests.
fn make_basic_block(va: u64, displaced: &[u8], block_id: u16) -> BasicBlock {
    BasicBlock {
        va,
        file_offset: va - 0x400000,
        displaced_len: displaced.len(),
        displaced_bytes: displaced.to_vec(),
        block_id,
    }
}

// ===========================================================================
// 1. Hook trampoline generation parity
// ===========================================================================

#[test]
fn test_pre_hook_trampoline_x86_64() {
    let hook = make_hook(HookMode::Pre, &X86_NOP5);
    let tramp = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::SysV64,
        None,
    )
    .expect("x86_64 pre-hook trampoline generation failed");

    assert!(
        !tramp.code.is_empty(),
        "trampoline code should be non-empty"
    );
    assert!(
        tramp.code.len() > 20,
        "x86_64 pre-hook trampoline should be substantial (got {} bytes)",
        tramp.code.len(),
    );
    // Starts with red zone skip: lea rsp, [rsp - 128] => 0x48 0x8D ...
    assert_eq!(
        tramp.code[0], 0x48,
        "expected REX.W prefix (0x48) for red zone skip, got 0x{:02x}",
        tramp.code[0],
    );
    assert_eq!(
        tramp.code[1], 0x8D,
        "expected LEA opcode (0x8D), got 0x{:02x}",
        tramp.code[1],
    );
}

#[cfg(feature = "aarch64")]
#[test]
fn test_pre_hook_trampoline_aarch64() {
    let hook = make_hook(HookMode::Pre, &AARCH64_NOP);
    let tramp = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::Aarch64,
        None,
    )
    .expect("AArch64 pre-hook trampoline generation failed");

    assert!(
        !tramp.code.is_empty(),
        "trampoline code should be non-empty"
    );
    assert!(
        tramp.code.len() > 16,
        "AArch64 pre-hook trampoline should be substantial (got {} bytes)",
        tramp.code.len(),
    );
    // AArch64 instructions are always 4-byte aligned.
    assert_eq!(
        tramp.code.len() % 4,
        0,
        "AArch64 trampoline size ({}) must be a multiple of 4 bytes",
        tramp.code.len(),
    );
}

#[test]
fn test_post_hook_trampoline_x86_64() {
    let hook = make_hook(HookMode::Post, &X86_NOP5);
    let tramp = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::SysV64,
        None,
    )
    .expect("x86_64 post-hook trampoline generation failed");

    assert!(
        !tramp.code.is_empty(),
        "trampoline code should be non-empty"
    );
    assert!(
        tramp.code.len() > 20,
        "x86_64 post-hook should be substantial (got {} bytes)",
        tramp.code.len(),
    );
}

#[cfg(feature = "aarch64")]
#[test]
fn test_post_hook_trampoline_aarch64() {
    let hook = make_hook(HookMode::Post, &AARCH64_NOP);
    let tramp = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::Aarch64,
        None,
    )
    .expect("AArch64 post-hook trampoline generation failed");

    assert!(!tramp.code.is_empty());
    assert_eq!(tramp.code.len() % 4, 0, "AArch64 must be 4-byte aligned");
}

#[test]
fn test_replace_hook_trampoline_x86_64() {
    let hook = make_hook(HookMode::Replace, &X86_NOP5);
    let tramp = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::SysV64,
        None,
    )
    .expect("x86_64 replace-hook trampoline generation failed");

    assert!(!tramp.code.is_empty());
    assert!(
        tramp.code.len() > 20,
        "x86_64 replace-hook should be substantial (got {} bytes)",
        tramp.code.len(),
    );
}

#[cfg(feature = "aarch64")]
#[test]
fn test_replace_hook_trampoline_aarch64() {
    let hook = make_hook(HookMode::Replace, &AARCH64_NOP);
    let tramp = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::Aarch64,
        None,
    )
    .expect("AArch64 replace-hook trampoline generation failed");

    assert!(!tramp.code.is_empty());
    assert_eq!(tramp.code.len() % 4, 0, "AArch64 must be 4-byte aligned");
}

// ===========================================================================
// 2. Conditional hook parity
// ===========================================================================

#[test]
fn test_conditional_hook_x86_64() {
    let hook = make_conditional_hook(HookMode::Pre, &X86_NOP5, "rdi");
    let tramp = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::SysV64,
        None,
    )
    .expect("x86_64 conditional hook generation failed");

    assert!(!tramp.code.is_empty());
    // The conditional hook should be larger than the unconditional one because
    // it includes the condition check instructions (CMP + Jcc).
    let unconditional = make_hook(HookMode::Pre, &X86_NOP5);
    let uncond_tramp = generate_hook_trampoline(
        0x50_0000,
        &unconditional,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::SysV64,
        None,
    )
    .unwrap();
    assert!(
        tramp.code.len() > uncond_tramp.code.len(),
        "conditional trampoline ({}) should be larger than unconditional ({})",
        tramp.code.len(),
        uncond_tramp.code.len(),
    );
}

#[cfg(feature = "aarch64")]
#[test]
fn test_conditional_hook_aarch64() {
    let hook = make_conditional_hook(HookMode::Pre, &AARCH64_NOP, "x0");
    let tramp = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::Aarch64,
        None,
    )
    .expect("AArch64 conditional hook generation failed");

    assert!(!tramp.code.is_empty());
    assert_eq!(tramp.code.len() % 4, 0);
    // Should be larger than unconditional.
    let unconditional = make_hook(HookMode::Pre, &AARCH64_NOP);
    let uncond_tramp = generate_hook_trampoline(
        0x50_0000,
        &unconditional,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::Aarch64,
        None,
    )
    .unwrap();
    assert!(
        tramp.code.len() > uncond_tramp.code.len(),
        "conditional AArch64 trampoline ({}) should be larger than unconditional ({})",
        tramp.code.len(),
        uncond_tramp.code.len(),
    );
}

// ===========================================================================
// 3. Return hook parity
// ===========================================================================

#[test]
fn test_return_hook_x86_64() {
    let hook = make_return_hook(&X86_NOP5);
    let rctx = ReturnHookContext {
        entry_va: 0x50_0000,
        ret_tramp_va: 0x50_1000,
        hook_data_va: 0x60_0000,
        shellcode_va: Some(0x70_0000),
        toggle_va: None,
        return_slot_va: 0x60_0100,
    };
    let (entry, ret_tramp) = generate_return_hook_trampolines(&rctx, &hook, TargetAbi::SysV64)
        .expect("x86_64 return hook generation failed");

    assert!(
        !entry.code.is_empty(),
        "entry trampoline should be non-empty"
    );
    assert!(
        !ret_tramp.code.is_empty(),
        "return trampoline should be non-empty"
    );
    assert_ne!(
        entry.va, ret_tramp.va,
        "entry and return trampolines must have different VAs",
    );
}

#[cfg(feature = "aarch64")]
#[test]
fn test_return_hook_aarch64() {
    let hook = make_return_hook(&AARCH64_NOP);
    let rctx = ReturnHookContext {
        entry_va: 0x50_0000,
        ret_tramp_va: 0x50_1000,
        hook_data_va: 0x60_0000,
        shellcode_va: Some(0x70_0000),
        toggle_va: None,
        return_slot_va: 0x60_0100,
    };
    let (entry, ret_tramp) = generate_return_hook_trampolines(&rctx, &hook, TargetAbi::Aarch64)
        .expect("AArch64 return hook generation failed");

    assert!(
        !entry.code.is_empty(),
        "entry trampoline should be non-empty"
    );
    assert!(
        !ret_tramp.code.is_empty(),
        "return trampoline should be non-empty"
    );
    assert_eq!(
        entry.code.len() % 4,
        0,
        "AArch64 entry must be 4-byte aligned"
    );
    assert_eq!(
        ret_tramp.code.len() % 4,
        0,
        "AArch64 return must be 4-byte aligned"
    );
    assert_ne!(entry.va, ret_tramp.va);
}

// ===========================================================================
// 4. Hook chaining parity
// ===========================================================================

#[test]
fn test_hook_chaining_x86_64() {
    let hook1 = make_hook(HookMode::Pre, &X86_NOP5);
    let hook2 = make_hook(HookMode::Pre, &X86_NOP5);
    let hooks: Vec<&ResolvedHook> = vec![&hook1, &hook2];

    let tramp = generate_chained_hook_trampoline(
        0x50_0000,
        &hooks,
        0x60_0000,
        &[Some(0x70_0000), Some(0x70_0100)],
        TargetAbi::SysV64,
        &[None, None],
    )
    .expect("x86_64 chained hook generation failed");

    assert!(!tramp.code.is_empty());
    // Chained trampoline should be larger than a single hook because it calls two handlers.
    let single = generate_hook_trampoline(
        0x50_0000,
        &hook1,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::SysV64,
        None,
    )
    .unwrap();
    assert!(
        tramp.code.len() > single.code.len(),
        "chained ({}) should be larger than single ({})",
        tramp.code.len(),
        single.code.len(),
    );
}

#[cfg(feature = "aarch64")]
#[test]
fn test_hook_chaining_aarch64() {
    let hook1 = make_hook(HookMode::Pre, &AARCH64_NOP);
    let hook2 = make_hook(HookMode::Pre, &AARCH64_NOP);
    let hooks: Vec<&ResolvedHook> = vec![&hook1, &hook2];

    let tramp = generate_chained_hook_trampoline(
        0x50_0000,
        &hooks,
        0x60_0000,
        &[Some(0x70_0000), Some(0x70_0100)],
        TargetAbi::Aarch64,
        &[None, None],
    )
    .expect("AArch64 chained hook generation failed");

    assert!(!tramp.code.is_empty());
    assert_eq!(
        tramp.code.len() % 4,
        0,
        "AArch64 chain must be 4-byte aligned"
    );
    // Chained should be larger than single.
    let single = generate_hook_trampoline(
        0x50_0000,
        &hook1,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::Aarch64,
        None,
    )
    .unwrap();
    assert!(
        tramp.code.len() > single.code.len(),
        "AArch64 chained ({}) should be larger than single ({})",
        tramp.code.len(),
        single.code.len(),
    );
}

// ===========================================================================
// 5. Mixed-mode hook parity (pre + post on same target)
// ===========================================================================

#[test]
fn test_mixed_mode_hooks_x86_64() {
    let pre_hook = make_hook(HookMode::Pre, &X86_NOP5);
    let post_hook = make_hook(HookMode::Post, &X86_NOP5);
    let hooks: Vec<&ResolvedHook> = vec![&pre_hook, &post_hook];

    let tramp = generate_chained_hook_trampoline(
        0x50_0000,
        &hooks,
        0x60_0000,
        &[Some(0x70_0000), Some(0x70_0100)],
        TargetAbi::SysV64,
        &[None, None],
    )
    .expect("x86_64 mixed-mode hook generation failed");

    assert!(!tramp.code.is_empty());
    // Mixed-mode has two save/restore frames with displaced instructions sandwiched.
    assert!(
        tramp.code.len() > 100,
        "mixed-mode trampoline should be substantial (got {} bytes)",
        tramp.code.len(),
    );
}

#[cfg(feature = "aarch64")]
#[test]
fn test_mixed_mode_hooks_aarch64() {
    let pre_hook = make_hook(HookMode::Pre, &AARCH64_NOP);
    let post_hook = make_hook(HookMode::Post, &AARCH64_NOP);
    let hooks: Vec<&ResolvedHook> = vec![&pre_hook, &post_hook];

    let tramp = generate_chained_hook_trampoline(
        0x50_0000,
        &hooks,
        0x60_0000,
        &[Some(0x70_0000), Some(0x70_0100)],
        TargetAbi::Aarch64,
        &[None, None],
    )
    .expect("AArch64 mixed-mode hook generation failed");

    assert!(!tramp.code.is_empty());
    assert_eq!(tramp.code.len() % 4, 0, "AArch64 must be 4-byte aligned");
    assert!(
        tramp.code.len() > 64,
        "AArch64 mixed-mode trampoline should be substantial (got {} bytes)",
        tramp.code.len(),
    );
}

// ===========================================================================
// 6. Runtime toggle parity
// ===========================================================================

#[test]
fn test_runtime_toggle_x86_64() {
    let hook = make_hook(HookMode::Pre, &X86_NOP5);
    let tramp_with_toggle = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::SysV64,
        Some(0x60_0200),
    )
    .expect("x86_64 toggle hook generation failed");

    let tramp_no_toggle = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::SysV64,
        None,
    )
    .expect("x86_64 non-toggle hook generation failed");

    assert!(!tramp_with_toggle.code.is_empty());
    // Toggle adds a movzx + test + jz sequence, making it larger.
    assert!(
        tramp_with_toggle.code.len() > tramp_no_toggle.code.len(),
        "toggle trampoline ({}) should be larger than non-toggle ({})",
        tramp_with_toggle.code.len(),
        tramp_no_toggle.code.len(),
    );
}

#[cfg(feature = "aarch64")]
#[test]
fn test_runtime_toggle_aarch64() {
    let hook = make_hook(HookMode::Pre, &AARCH64_NOP);
    let tramp_with_toggle = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::Aarch64,
        Some(0x60_0200),
    )
    .expect("AArch64 toggle hook generation failed");

    let tramp_no_toggle = generate_hook_trampoline(
        0x50_0000,
        &hook,
        0x60_0000,
        Some(0x70_0000),
        TargetAbi::Aarch64,
        None,
    )
    .expect("AArch64 non-toggle hook generation failed");

    assert!(!tramp_with_toggle.code.is_empty());
    assert_eq!(tramp_with_toggle.code.len() % 4, 0);
    // Toggle adds ADR + LDRB + CBZ, making it larger.
    assert!(
        tramp_with_toggle.code.len() > tramp_no_toggle.code.len(),
        "AArch64 toggle trampoline ({}) should be larger than non-toggle ({})",
        tramp_with_toggle.code.len(),
        tramp_no_toggle.code.len(),
    );
}

// ===========================================================================
// 7. Hook preload generation
// ===========================================================================

#[test]
fn test_hook_preload_spec_elf() {
    let hook = make_library_hook(HookMode::Pre, &X86_NOP5, "my_handler");

    let spec = hook_preload::build_preload_spec(
        &[hook],
        Path::new("/tmp/libhooks.so"),
        0x60_0000, // hook_data_va
        0x60_1000, // toggle_data_va
        true,      // is_pie
        false,     // is_macho (ELF)
    );

    let spec = spec.expect("ELF preload spec should be Some for library hooks");
    assert_eq!(spec.handlers.len(), 1);
    assert_eq!(spec.handlers[0].0, "my_handler");
    assert!(!spec.is_macho, "ELF spec should have is_macho=false");
    assert!(spec.is_pie);
}

#[test]
fn test_hook_preload_spec_macho() {
    let hook = make_library_hook(HookMode::Pre, &X86_NOP5, "my_handler");

    let spec = hook_preload::build_preload_spec(
        &[hook],
        Path::new("/tmp/libhooks.dylib"),
        0x60_0000,
        0x60_1000,
        true, // is_pie
        true, // is_macho
    );

    let spec = spec.expect("Mach-O preload spec should be Some for library hooks");
    assert_eq!(spec.handlers.len(), 1);
    assert!(spec.is_macho, "Mach-O spec should have is_macho=true");
}

#[test]
fn test_hook_preload_path_elf() {
    let path = hook_preload::hook_preload_lib_path(Path::new("/tmp/target_rewritten"), false);
    let ext = path.extension().unwrap().to_str().unwrap();
    assert_eq!(ext, "so", "ELF preload lib should have .so extension");
}

#[test]
fn test_hook_preload_path_macho() {
    let path = hook_preload::hook_preload_lib_path(Path::new("/tmp/target_rewritten"), true);
    let ext = path.extension().unwrap().to_str().unwrap();
    assert_eq!(
        ext, "dylib",
        "Mach-O preload lib should have .dylib extension"
    );
}

// ===========================================================================
// 8. Heap sanitiser source generation
// ===========================================================================

#[test]
#[cfg(feature = "coverage")]
fn test_heap_san_path_elf() {
    let path = preload::preload_lib_path(Path::new("/tmp/target_rewritten"), false);
    let ext = path.extension().unwrap().to_str().unwrap();
    assert_eq!(ext, "so", "ELF heap san should produce .so");
    let name = path.file_name().unwrap().to_str().unwrap();
    assert!(
        name.contains("heap_san"),
        "ELF heap san lib name should contain 'heap_san', got: {}",
        name,
    );
}

#[test]
#[cfg(feature = "coverage")]
fn test_heap_san_path_macho() {
    let path = preload::preload_lib_path(Path::new("/tmp/target_rewritten"), true);
    let ext = path.extension().unwrap().to_str().unwrap();
    assert_eq!(ext, "dylib", "Mach-O heap san should produce .dylib");
    let name = path.file_name().unwrap().to_str().unwrap();
    assert!(
        name.contains("heap_san"),
        "Mach-O heap san lib name should contain 'heap_san', got: {}",
        name,
    );
}

// ===========================================================================
// 9. Sidecar sanitiser source generation
// ===========================================================================

#[test]
#[cfg(feature = "coverage")]
fn test_sidecar_san_path_elf() {
    let path = sidecar_preload::sidecar_preload_lib_path(Path::new("/tmp/target_rewritten"), false);
    let ext = path.extension().unwrap().to_str().unwrap();
    assert_eq!(ext, "so", "ELF sidecar san should produce .so");
    let name = path.file_name().unwrap().to_str().unwrap();
    assert!(
        name.contains("sidecar_san"),
        "ELF sidecar san lib name should contain 'sidecar_san', got: {}",
        name,
    );
}

#[test]
#[cfg(feature = "coverage")]
fn test_sidecar_san_path_macho() {
    let path = sidecar_preload::sidecar_preload_lib_path(Path::new("/tmp/target_rewritten"), true);
    let ext = path.extension().unwrap().to_str().unwrap();
    assert_eq!(ext, "dylib", "Mach-O sidecar san should produce .dylib");
    let name = path.file_name().unwrap().to_str().unwrap();
    assert!(
        name.contains("sidecar_san"),
        "Mach-O sidecar san lib name should contain 'sidecar_san', got: {}",
        name,
    );
}

// ===========================================================================
// 10. Init code generation parity
// ===========================================================================

#[test]
fn test_init_code_x86_64_elf() {
    let init = trampoline::generate_init_code(
        0x80_0000, // init_va
        0x90_0000, // data_va
        0x40_1000, // original_entry
        true,      // enable_forkserver
        None,      // persistent_data_va
    )
    .expect("x86_64 ELF init code generation failed");

    assert!(!init.code.is_empty(), "init code should be non-empty");
    assert!(
        init.code.len() > 100,
        "x86_64 ELF init code should be substantial (got {} bytes)",
        init.code.len(),
    );
    // Contains at least one syscall instruction (0x0F 0x05).
    let has_syscall = init.code.windows(2).any(|w| w[0] == 0x0F && w[1] == 0x05);
    assert!(
        has_syscall,
        "x86_64 ELF init code should contain syscall (0F 05)"
    );
}

#[cfg(feature = "aarch64")]
#[test]
fn test_init_code_aarch64_elf() {
    use truant::AArch64TrampolineGenerator;
    use truant::traits::TrampolineGenerator;

    let tramp_gen = AArch64TrampolineGenerator::new();
    let init = tramp_gen
        .generate_init_code(
            0x80_0000, // init_va
            0x90_0000, // data_va
            0x40_1000, // entry_point
            true,      // enable_forkserver
            None,      // persistent_data_va
        )
        .expect("AArch64 ELF init code generation failed");

    assert!(!init.code.is_empty(), "init code should be non-empty");
    assert!(
        init.code.len() > 100,
        "AArch64 ELF init code should be substantial (got {} bytes)",
        init.code.len(),
    );
    // All instructions are 4-byte aligned.
    assert_eq!(
        init.code.len() % 4,
        0,
        "AArch64 init code size ({}) must be a multiple of 4",
        init.code.len(),
    );
    // Contains SVC #0 instruction (0x01 0x00 0x00 0xD4).
    let svc0_bytes: [u8; 4] = 0xD400_0001u32.to_le_bytes();
    let has_svc = init.code.windows(4).any(|w| w == svc0_bytes);
    assert!(has_svc, "AArch64 ELF init code should contain SVC #0");
}

#[test]
fn test_init_code_x86_64_macho() {
    let init = macho_trampoline::generate_macho_exec_init_x86_64(
        0x80_0000, // init_va
        0x90_0000, // data_va
        0x40_1000, // original_entry
        true,      // enable_forkserver
        false,     // use_got_shmat
    )
    .expect("x86_64 Mach-O init code generation failed");

    assert!(!init.code.is_empty(), "macOS init code should be non-empty");
    assert!(
        init.code.len() > 100,
        "x86_64 Mach-O init code should be substantial (got {} bytes)",
        init.code.len(),
    );
    // Contains syscall instruction for macOS BSD syscalls.
    let has_syscall = init.code.windows(2).any(|w| w[0] == 0x0F && w[1] == 0x05);
    assert!(
        has_syscall,
        "x86_64 Mach-O init code should contain syscall (0F 05)"
    );
}

#[test]
fn test_init_code_aarch64_macho() {
    let init = macho_trampoline::generate_macho_exec_init_aarch64(
        0x80_0000, // init_va
        0x90_0000, // data_va
        0x40_1000, // original_entry
        true,      // enable_forkserver
        false,     // use_got_shmat
    )
    .expect("AArch64 Mach-O init code generation failed");

    assert!(
        !init.code.is_empty(),
        "macOS AArch64 init code should be non-empty"
    );
    assert!(
        init.code.len() > 100,
        "AArch64 Mach-O init code should be substantial (got {} bytes)",
        init.code.len(),
    );
    assert_eq!(
        init.code.len() % 4,
        0,
        "AArch64 Mach-O init code size ({}) must be a multiple of 4",
        init.code.len(),
    );
    // Contains SVC #0x80 (macOS supervisor call): 0x01 0x10 0x00 0xD4
    let svc80_bytes: [u8; 4] = 0xD400_1001u32.to_le_bytes();
    let has_svc80 = init.code.windows(4).any(|w| w == svc80_bytes);
    assert!(
        has_svc80,
        "AArch64 Mach-O init code should contain SVC #0x80"
    );
}

// ===========================================================================
// 11. Persistent mode wrapper parity
// ===========================================================================

#[test]
fn test_persistent_wrapper_x86_64_elf() {
    let wrapper = trampoline::generate_persistent_wrapper(&PersistentWrapperParams {
        wrapper_va: 0x80_0000,
        persistent_data_va: 0x90_0000,
        data_va: 0xA0_0000,
        persistent_addr: 0x40_1000,
        displaced_bytes: &X86_NOP5,
        displaced_len: X86_NOP5.len(),
        persistent_count: 1000,
        include_forkserver: false,
    })
    .expect("x86_64 ELF persistent wrapper generation failed");

    assert!(
        !wrapper.code.is_empty(),
        "persistent wrapper should be non-empty"
    );
    assert!(
        wrapper.code.len() > 50,
        "x86_64 persistent wrapper should be substantial (got {} bytes)",
        wrapper.code.len(),
    );
}

#[cfg(feature = "aarch64")]
#[test]
fn test_persistent_wrapper_aarch64_elf() {
    use truant::arch::aarch64::trampoline_gen::generate_persistent_wrapper_aarch64;

    // AArch64 ADR instruction has +/-1 MiB range, so wrapper_va and data areas
    // must be close together. Use nearby VAs to stay within range.
    let wrapper = generate_persistent_wrapper_aarch64(&PersistentWrapperParams {
        wrapper_va: 0x10_0000,
        persistent_data_va: 0x10_2000,
        data_va: 0x10_4000,
        persistent_addr: 0x10_8000,
        displaced_bytes: &AARCH64_NOP,
        displaced_len: AARCH64_NOP.len(),
        persistent_count: 1000,
        include_forkserver: false,
    })
    .expect("AArch64 ELF persistent wrapper generation failed");

    assert!(
        !wrapper.code.is_empty(),
        "persistent wrapper should be non-empty"
    );
    assert_eq!(
        wrapper.code.len() % 4,
        0,
        "AArch64 persistent wrapper size ({}) must be 4-byte aligned",
        wrapper.code.len(),
    );
}

#[test]
fn test_persistent_wrapper_x86_64_macho() {
    let wrapper =
        macho_trampoline::generate_macho_persistent_wrapper_x86_64(&PersistentWrapperParams {
            wrapper_va: 0x80_0000,
            persistent_data_va: 0x90_0000,
            data_va: 0xA0_0000,
            persistent_addr: 0x40_1000,
            displaced_bytes: &X86_NOP5,
            displaced_len: X86_NOP5.len(),
            persistent_count: 1000,
            include_forkserver: false,
        })
        .expect("x86_64 Mach-O persistent wrapper generation failed");

    assert!(!wrapper.code.is_empty());
    assert!(
        wrapper.code.len() > 50,
        "x86_64 Mach-O persistent wrapper should be substantial (got {} bytes)",
        wrapper.code.len(),
    );
}

#[test]
fn test_persistent_wrapper_aarch64_macho() {
    // AArch64 ADR has +/-1 MiB range -- use nearby VAs.
    let wrapper =
        macho_trampoline::generate_macho_persistent_wrapper_aarch64(&PersistentWrapperParams {
            wrapper_va: 0x10_0000,
            persistent_data_va: 0x10_2000,
            data_va: 0x10_4000,
            persistent_addr: 0x10_8000,
            displaced_bytes: &AARCH64_NOP,
            displaced_len: AARCH64_NOP.len(),
            persistent_count: 1000,
            include_forkserver: false,
        })
        .expect("AArch64 Mach-O persistent wrapper generation failed");

    assert!(!wrapper.code.is_empty());
    assert_eq!(
        wrapper.code.len() % 4,
        0,
        "AArch64 Mach-O persistent wrapper size ({}) must be 4-byte aligned",
        wrapper.code.len(),
    );
}

// ===========================================================================
// 12. Coverage trampoline parity
// ===========================================================================

#[test]
fn test_coverage_trampoline_x86_64() {
    let block = make_basic_block(0x40_1000, &X86_NOP5, 0x1234);
    let tramp = trampoline::generate_trampoline(
        0x80_0000, // trampoline_va
        0x90_0000, // data_va
        &block,
    )
    .expect("x86_64 coverage trampoline generation failed");

    assert!(
        !tramp.code.is_empty(),
        "coverage trampoline should be non-empty"
    );
    // Starts with red zone skip: lea rsp, [rsp - 128] => 0x48 0x8D 0xA4 0x24
    assert!(
        tramp.code.len() >= 4,
        "trampoline too short: {} bytes",
        tramp.code.len(),
    );
    assert_eq!(tramp.code[0], 0x48, "expected REX.W prefix");
    assert_eq!(tramp.code[1], 0x8D, "expected LEA opcode");
    assert_eq!(tramp.code[2], 0xA4, "expected modrm for [rsp+disp32]");
    assert_eq!(tramp.code[3], 0x24, "expected SIB for [rsp+...]");

    // Should end with a JMP rel32 back to the original code.
    let last5 = &tramp.code[tramp.code.len() - 5..];
    assert_eq!(
        last5[0], 0xE9,
        "trampoline should end with JMP rel32 (0xE9)"
    );
}

#[cfg(feature = "aarch64")]
#[test]
fn test_coverage_trampoline_aarch64() {
    use truant::AArch64TrampolineGenerator;
    use truant::traits::TrampolineGenerator;

    let block = make_basic_block(0x40_1000, &AARCH64_NOP, 0x1234);
    let tramp_gen = AArch64TrampolineGenerator::new();
    let tramp = tramp_gen
        .generate_trampoline(
            0x80_0000, // trampoline_va
            0x90_0000, // data_va
            &block,
        )
        .expect("AArch64 coverage trampoline generation failed");

    assert!(
        !tramp.code.is_empty(),
        "coverage trampoline should be non-empty"
    );
    assert_eq!(
        tramp.code.len() % 4,
        0,
        "AArch64 coverage trampoline size ({}) must be 4-byte aligned",
        tramp.code.len(),
    );
    assert!(
        tramp.code.len() > 16,
        "AArch64 coverage trampoline should be substantial (got {} bytes)",
        tramp.code.len(),
    );

    // Last instruction should be an unconditional B (branch) back.
    // B encoding: top 6 bits = 000101 => first byte of big-endian word has 0x14 or 0x15 etc.
    // In little-endian, the top byte (byte index 3) has bits [31:24].
    // For B: bit 31=0, bit 30=0, bit 29=0, bit 28=1, bit 27=0, bit 26=1 => byte[3] = 0b000101xx = 0x14..0x17
    let last4 = &tramp.code[tramp.code.len() - 4..];
    let last_word = u32::from_le_bytes([last4[0], last4[1], last4[2], last4[3]]);
    let opcode_top6 = (last_word >> 26) & 0x3F;
    assert_eq!(
        opcode_top6, 0b000101,
        "AArch64 trampoline should end with B instruction (got top6=0b{:06b})",
        opcode_top6,
    );
}

// ===========================================================================
// Cross-platform size consistency checks
// ===========================================================================

#[cfg(feature = "aarch64")]
#[test]
fn test_init_code_elf_both_arches_generate() {
    use truant::AArch64TrampolineGenerator;
    use truant::traits::TrampolineGenerator;

    // Verify both architectures generate init code without errors.
    let x86_init = trampoline::generate_init_code(0x80_0000, 0x90_0000, 0x40_1000, true, None)
        .expect("x86_64 ELF init code");

    let a64_gen = AArch64TrampolineGenerator::new();
    let a64_init = a64_gen
        .generate_init_code(0x80_0000, 0x90_0000, 0x40_1000, true, None)
        .expect("AArch64 ELF init code");

    // Both should produce non-trivial output.
    assert!(x86_init.code.len() > 100);
    assert!(a64_init.code.len() > 100);
}

#[cfg(feature = "aarch64")]
#[test]
fn test_persistent_wrapper_both_arches_elf() {
    use truant::arch::aarch64::trampoline_gen::generate_persistent_wrapper_aarch64;

    let x86_wrapper = trampoline::generate_persistent_wrapper(&PersistentWrapperParams {
        wrapper_va: 0x80_0000,
        persistent_data_va: 0x90_0000,
        data_va: 0xA0_0000,
        persistent_addr: 0x40_1000,
        displaced_bytes: &X86_NOP5,
        displaced_len: X86_NOP5.len(),
        persistent_count: 1000,
        include_forkserver: false,
    })
    .expect("x86_64 persistent wrapper");

    // AArch64 ADR has +/-1 MiB range -- use nearby VAs.
    let a64_wrapper = generate_persistent_wrapper_aarch64(&PersistentWrapperParams {
        wrapper_va: 0x10_0000,
        persistent_data_va: 0x10_2000,
        data_va: 0x10_4000,
        persistent_addr: 0x10_8000,
        displaced_bytes: &AARCH64_NOP,
        displaced_len: AARCH64_NOP.len(),
        persistent_count: 1000,
        include_forkserver: false,
    })
    .expect("AArch64 persistent wrapper");

    assert!(x86_wrapper.code.len() > 50);
    assert!(a64_wrapper.code.len() > 50);
    assert_eq!(a64_wrapper.code.len() % 4, 0);
}

#[test]
fn test_persistent_wrapper_both_arches_macho() {
    let x86_wrapper =
        macho_trampoline::generate_macho_persistent_wrapper_x86_64(&PersistentWrapperParams {
            wrapper_va: 0x80_0000,
            persistent_data_va: 0x90_0000,
            data_va: 0xA0_0000,
            persistent_addr: 0x40_1000,
            displaced_bytes: &X86_NOP5,
            displaced_len: X86_NOP5.len(),
            persistent_count: 1000,
            include_forkserver: false,
        })
        .expect("x86_64 Mach-O persistent wrapper");

    // AArch64 ADR has +/-1 MiB range -- use nearby VAs.
    let a64_wrapper =
        macho_trampoline::generate_macho_persistent_wrapper_aarch64(&PersistentWrapperParams {
            wrapper_va: 0x10_0000,
            persistent_data_va: 0x10_2000,
            data_va: 0x10_4000,
            persistent_addr: 0x10_8000,
            displaced_bytes: &AARCH64_NOP,
            displaced_len: AARCH64_NOP.len(),
            persistent_count: 1000,
            include_forkserver: false,
        })
        .expect("AArch64 Mach-O persistent wrapper");

    assert!(x86_wrapper.code.len() > 50);
    assert!(a64_wrapper.code.len() > 50);
    assert_eq!(a64_wrapper.code.len() % 4, 0);
}

#[test]
fn test_init_code_macho_both_arches_generate() {
    let x86_init = macho_trampoline::generate_macho_exec_init_x86_64(
        0x80_0000, 0x90_0000, 0x40_1000, true, false,
    )
    .expect("x86_64 Mach-O init code");

    let a64_init = macho_trampoline::generate_macho_exec_init_aarch64(
        0x80_0000, 0x90_0000, 0x40_1000, true, false,
    )
    .expect("AArch64 Mach-O init code");

    assert!(x86_init.code.len() > 100);
    assert!(a64_init.code.len() > 100);
    assert_eq!(a64_init.code.len() % 4, 0);
}
