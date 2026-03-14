//! Hook unit-style integration tests (migrated from src/lib.rs).
//!
//! These test individual hook modes using real compiled binaries.
//! Run with: cargo test -p truant --test hook_unit -- --test-threads=1

mod common;
use common::*;

use std::path::PathBuf;

use truant::{RewriteConfig, rewrite};

#[test]
fn test_hook_pre_shellcode_nop() {
    // Pre-hook with NOP shellcode: original function should still run unchanged.
    // The shellcode is a simple RET (0xC3), so the pre-hook returns immediately
    // and the original code executes normally.
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_pre_nop",
        r#"
        int target_func(void) { return 42; }
        int main() { return target_func(); }
    "#,
    );
    let output = td.path().join("unit_pre_nop_out");

    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let toml_content = format!(
        r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]
"#,
        target_va
    );
    let hooks_toml = write_hooks(td.path(), "unit_pre_nop", &toml_content);

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1, "expected 1 hook applied");

    let status = std::process::Command::new(&output)
        .status()
        .expect("failed to run hooked binary");
    // The original function returns 42, and the pre-hook (RET) just returns
    // without modifying registers, so the original code should still run.
    assert_eq!(
        status.code(),
        Some(42),
        "pre-hook NOP should not change exit code, expected 42, got {:?}",
        status
    );
}

#[test]
fn test_hook_pre_shellcode_by_symbol() {
    // Same as above but using symbol name instead of hex VA.
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_pre_sym",
        r#"
        int target_func(void) { return 42; }
        int main() { return target_func(); }
    "#,
    );
    let output = td.path().join("unit_pre_sym_out");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_pre_sym",
        r#"
[[hook]]
target = "target_func"
mode = "pre"
shellcode = [0xC3]
"#,
    );

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1);

    let status = std::process::Command::new(&output)
        .status()
        .expect("failed to run hooked binary");
    assert_eq!(
        status.code(),
        Some(42),
        "pre-hook by symbol should not change exit code, expected 42, got {:?}",
        status
    );
}

#[test]
fn test_hook_replace_shellcode() {
    // Replace mode: shellcode replaces the function entirely.
    // The hook receives rdi = &RegContext. To set the return value,
    // it writes to RegContext.rax (offset 0). The trampoline restores
    // rax from RegContext, then RETs to the caller.
    //
    // Shellcode: mov qword [rdi], 7; ret
    //   48 C7 07 07 00 00 00 = mov qword ptr [rdi], 7
    //   C3                   = ret
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_replace_sc",
        r#"
        int target_func(void) { return 42; }
        int main() { return target_func(); }
    "#,
    );
    let output = td.path().join("unit_replace_sc_out");

    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let toml_content = format!(
        r#"
[[hook]]
target = "0x{:x}"
mode = "replace"
shellcode = [0x48, 0xC7, 0x07, 0x07, 0x00, 0x00, 0x00, 0xC3]
"#,
        target_va
    );
    let hooks_toml = write_hooks(td.path(), "unit_replace_sc", &toml_content);

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1);

    let status = std::process::Command::new(&output)
        .status()
        .expect("failed to run hooked binary");
    // The replace shellcode sets RegContext.rax = 7, so the function returns 7.
    assert_eq!(
        status.code(),
        Some(7),
        "replace hook should change exit code to 7, got {:?}",
        status
    );
}

#[test]
fn test_hook_post_shellcode() {
    // Post-hook with NOP shellcode: original function runs first,
    // then the post-hook (RET) fires. Should not change behaviour.
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_post_sc",
        r#"
        int target_func(void) { return 42; }
        int main() { return target_func(); }
    "#,
    );
    let output = td.path().join("unit_post_sc_out");

    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let toml_content = format!(
        r#"
[[hook]]
target = "0x{:x}"
mode = "post"
shellcode = [0xC3]
"#,
        target_va
    );
    let hooks_toml = write_hooks(td.path(), "unit_post_sc", &toml_content);

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1);

    let status = std::process::Command::new(&output)
        .status()
        .expect("failed to run hooked binary");
    assert_eq!(
        status.code(),
        Some(42),
        "post-hook NOP should not change exit code, expected 42, got {:?}",
        status
    );
}

#[test]
fn test_hook_multiple_targets() {
    // Apply hooks to two different functions in the same binary.
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_multi",
        r#"
        int func_a(void) { return 10; }
        int func_b(void) { return 20; }
        int main() { return func_a() + func_b(); }
    "#,
    );
    let output = td.path().join("unit_multi_out");

    let va_a = find_symbol_va(&input, "func_a").expect("func_a not found");
    let va_b = find_symbol_va(&input, "func_b").expect("func_b not found");

    // Pre-hook both with NOP shellcode — original behaviour preserved.
    let toml_content = format!(
        r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]

[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]
"#,
        va_a, va_b
    );
    let hooks_toml = write_hooks(td.path(), "unit_multi", &toml_content);

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 2, "expected 2 hooks applied");

    let status = std::process::Command::new(&output)
        .status()
        .expect("failed to run multi-hooked binary");
    assert_eq!(
        status.code(),
        Some(30),
        "multi-hook NOP should preserve exit code 30 (10+20), got {:?}",
        status
    );
}

#[test]
fn test_hook_coverage_excluded_at_hook_site() {
    // When a hook is applied, the target block should be excluded from
    // coverage instrumentation (hook trampoline replaces coverage trampoline).
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_cov_excl",
        r#"
        int target_func(void) { return 0; }
        int main() { return target_func(); }
    "#,
    );
    let output_no_hooks = td.path().join("unit_cov_excl_nohook");
    let output_hooks = td.path().join("unit_cov_excl_hooked");

    // Rewrite without hooks.
    let config_no_hooks = RewriteConfig {
        input: input.clone(),
        output: output_no_hooks.clone(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: None,
        no_coverage: false,
    };
    let result_no_hooks = rewrite(&config_no_hooks).expect("rewrite without hooks failed");

    // Rewrite with hook on target_func.
    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");
    let toml_content = format!(
        r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]
"#,
        target_va
    );
    let hooks_toml = write_hooks(td.path(), "unit_cov_excl", &toml_content);

    let result_hooks = rewrite_hooked(&input, &output_hooks, &hooks_toml);

    // The hooked version should have one fewer coverage block (the hook target).
    assert!(
        result_hooks.blocks_instrumented < result_no_hooks.blocks_instrumented,
        "hooked binary should have fewer coverage blocks ({} should be < {})",
        result_hooks.blocks_instrumented,
        result_no_hooks.blocks_instrumented,
    );
    assert_eq!(result_hooks.hooks_applied, 1);
}

#[test]
fn test_hook_error_symbol_not_found() {
    // Hooking a non-existent symbol should produce an error.
    let td = test_dir();
    let input = compile_bin(td.path(), "unit_err_sym", "int main() { return 0; }\n");
    let output = td.path().join("unit_err_sym_out");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_err_sym",
        r#"
[[hook]]
target = "nonexistent_function"
mode = "pre"
shellcode = [0xC3]
"#,
    );

    let config = RewriteConfig {
        input: input.clone(),
        output: output.clone(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: Some(hooks_toml.clone()),
        no_coverage: false,
    };
    let err = rewrite(&config).unwrap_err();
    let msg = format!("{:#}", err);
    assert!(
        msg.contains("not found"),
        "error should mention symbol not found: {}",
        msg,
    );
}

#[test]
fn test_hook_chaining_same_target() {
    // Two pre-hooks on the same target should chain (both execute in order).
    // Each hook is a NOP (RET shellcode) — original behaviour preserved.
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_chain",
        r#"
        int target_func(void) { return 42; }
        int main() { return target_func(); }
    "#,
    );
    let output = td.path().join("unit_chain_out");

    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let toml_content = format!(
        r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]

[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]
"#,
        target_va, target_va
    );
    let hooks_toml = write_hooks(td.path(), "unit_chain", &toml_content);

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 2, "expected 2 chained hooks applied");

    let status = std::process::Command::new(&output)
        .status()
        .expect("failed to run chained-hook binary");
    assert_eq!(
        status.code(),
        Some(42),
        "chained NOP hooks should preserve exit code 42, got {:?}",
        status
    );
}

#[test]
fn test_hook_chaining_mixed_modes_supported() {
    // Mixed modes (pre + post) on the same target should now work
    // via the mixed-mode chain trampoline.
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_chain_mix",
        r#"
        int target_func(void) { return 0; }
        int main() { return target_func(); }
    "#,
    );
    let output = td.path().join("unit_chain_mix_out");

    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let toml_content = format!(
        r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]

[[hook]]
target = "0x{:x}"
mode = "post"
shellcode = [0xC3]
"#,
        target_va, target_va
    );
    let hooks_toml = write_hooks(td.path(), "unit_chain_mix", &toml_content);

    let config = RewriteConfig {
        input: input.clone(),
        output: output.clone(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: Some(hooks_toml.clone()),
        no_coverage: false,
    };
    let result = rewrite(&config).expect("rewrite should succeed with mixed-mode hooks");
    assert_eq!(
        result.hooks_applied, 2,
        "mixed-mode hooks on same target should be applied via mixed chain, got {} applied",
        result.hooks_applied,
    );
}

#[test]
fn test_hook_error_handler_without_library() {
    // Using handler= without [hooks] library should fail at parse time.
    let td = test_dir();
    let hooks_toml = write_hooks(
        td.path(),
        "unit_err_nolib",
        r#"
[[hook]]
target = "main"
mode = "pre"
handler = "my_handler"
"#,
    );

    let config = RewriteConfig {
        input: PathBuf::from("/usr/bin/true"),
        output: td.path().join("unit_err_nolib_out"),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: Some(hooks_toml.clone()),
        no_coverage: false,
    };
    let err = rewrite(&config).unwrap_err();
    let msg = format!("{:#}", err);
    assert!(
        msg.contains("library"),
        "error should mention missing library: {}",
        msg,
    );
}

#[test]
fn test_hook_plt_stub() {
    // Hook a PLT stub (puts@plt) with a pre-hook NOP shellcode.
    // The binary prints via puts() and exits 0 — hooking .plt with a NOP
    // should not affect behaviour (the pre-hook returns, then the real PLT
    // stub executes normally).
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_plt_hook",
        r#"
        #include <stdio.h>
        int main() { puts("hello"); return 0; }
    "#,
    );
    let output = td.path().join("unit_plt_hook_out");

    let plt_va = find_plt_va(&input, "puts")
        .expect("puts@plt not found — binary must be dynamically linked");

    let toml_content = format!(
        r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]
"#,
        plt_va
    );
    let hooks_toml = write_hooks(td.path(), "unit_plt_hook", &toml_content);

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1, "PLT hook should be applied");

    let status = std::process::Command::new(&output)
        .status()
        .expect("failed to run PLT-hooked binary");
    assert_eq!(
        status.code(),
        Some(0),
        "PLT pre-hook NOP should preserve exit code 0, got {:?}",
        status,
    );
}

#[test]
fn test_hook_no_coverage_mode() {
    // --no-coverage: apply hooks without any coverage instrumentation.
    // The binary should run normally with the hook applied but zero coverage blocks.
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_nocov",
        r#"
        int target_func(void) { return 42; }
        int main() { return target_func(); }
    "#,
    );
    let output = td.path().join("unit_nocov_out");

    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let toml_content = format!(
        r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]
"#,
        target_va
    );
    let hooks_toml = write_hooks(td.path(), "unit_nocov", &toml_content);

    let config = RewriteConfig {
        input: input.clone(),
        output: output.clone(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: Some(hooks_toml.clone()),
        no_coverage: true,
    };
    let result = rewrite(&config).expect("no-coverage rewrite failed");
    assert_eq!(result.hooks_applied, 1, "hook should be applied");
    assert_eq!(
        result.blocks_instrumented, 0,
        "no coverage blocks in no-coverage mode"
    );

    let status = std::process::Command::new(&output)
        .status()
        .expect("failed to run no-coverage binary");
    assert_eq!(
        status.code(),
        Some(42),
        "pre-hook NOP should preserve exit code 42, got {:?}",
        status,
    );
}

#[test]
fn test_hook_with_condition() {
    // Conditional hook: pre-hook on main with condition { register = "rdi", op = "eq", value = 0 }.
    // The condition is checked at runtime inside the trampoline. On entry to main,
    // argc (rdi) is typically 1 (not 0), so the hook should NOT fire.
    // The shellcode would `xor eax,eax; ret` (exit 0) if it fired.
    // Since it should not fire, the binary should exit with 42.
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_condhook",
        r#"
        int main() { return 42; }
    "#,
    );
    let output = td.path().join("unit_condhook_out");

    let main_va = find_symbol_va(&input, "main").expect("main not found");

    // Shellcode: xor eax,eax; ret — would make exit code 0 if hook fires
    let toml_content = format!(
        r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0x31, 0xC0, 0xC3]
condition = {{ register = "rdi", op = "eq", value = 0 }}
"#,
        main_va
    );
    let hooks_toml = write_hooks(td.path(), "unit_condhook", &toml_content);

    let config = RewriteConfig {
        input: input.clone(),
        output: output.clone(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: Some(hooks_toml.clone()),
        no_coverage: true,
    };
    let result = rewrite(&config).expect("conditional hook rewrite failed");
    assert_eq!(result.hooks_applied, 1, "hook should be applied");

    let status = std::process::Command::new(&output)
        .status()
        .expect("failed to run conditional hook binary");
    // argc==1 (not 0), so condition is false → hook skipped → exit 42
    assert_eq!(
        status.code(),
        Some(42),
        "conditional hook should NOT fire (rdi!=0), expected exit 42, got {:?}",
        status,
    );
}
