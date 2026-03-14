//! Hook unit-style integration tests (migrated from src/lib.rs).
//!
//! These test individual hook modes using real compiled binaries.
//! Run with: cargo test -p truant --test hook_unit -- --test-threads=1

mod common;
mod shellcode;

use common::*;
use shellcode::*;
use std::path::PathBuf;
use truant::{RewriteConfig, rewrite};

#[test]
fn test_hook_pre_shellcode_nop() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_pre_nop",
        "__attribute__((noinline)) int target_func(void) { return 42; }\nint main() { return target_func(); }\n",
    );
    let output = td.path().join("unit_pre_nop_out");
    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_pre_nop",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            target_va,
            shellcode_toml(NOP),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        42,
        "pre-hook NOP should preserve exit code 42"
    );
}

#[test]
fn test_hook_pre_shellcode_by_symbol() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_pre_sym",
        "__attribute__((noinline)) int target_func(void) { return 42; }\nint main() { return target_func(); }\n",
    );
    let output = td.path().join("unit_pre_sym_out");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_pre_sym",
        &format!(
            "[[hook]]\ntarget = \"target_func\"\nmode = \"pre\"\nshellcode = {}\n",
            shellcode_toml(NOP)
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        42,
        "pre-hook by symbol should preserve exit code 42"
    );
}

#[test]
fn test_hook_replace_shellcode() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_replace_sc",
        "__attribute__((noinline)) int target_func(void) { return 42; }\nint main() { return target_func(); }\n",
    );
    let output = td.path().join("unit_replace_sc_out");
    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_replace_sc",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\n",
            target_va,
            shellcode_toml(RETURN_7),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(run(&output), 7, "replace hook should change exit code to 7");
}

#[test]
fn test_hook_post_shellcode() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_post_sc",
        "__attribute__((noinline)) int target_func(void) { return 42; }\nint main() { return target_func(); }\n",
    );
    let output = td.path().join("unit_post_sc_out");
    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_post_sc",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"post\"\nshellcode = {}\n",
            target_va,
            shellcode_toml(NOP),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        42,
        "post-hook NOP should preserve exit code 42"
    );
}

#[test]
fn test_hook_multiple_targets() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_multi",
        "__attribute__((noinline)) int func_a(void) { return 10; }\n__attribute__((noinline)) int func_b(void) { return 20; }\nint main() { return func_a() + func_b(); }\n",
    );
    let output = td.path().join("unit_multi_out");
    let va_a = find_symbol_va(&input, "func_a").expect("func_a not found");
    let va_b = find_symbol_va(&input, "func_b").expect("func_b not found");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_multi",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n\n\
             [[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            va_a,
            shellcode_toml(NOP),
            va_b,
            shellcode_toml(NOP),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 2);
    assert_eq!(
        run(&output),
        30,
        "multi-hook NOP should preserve exit code 30"
    );
}

#[test]
fn test_hook_coverage_excluded_at_hook_site() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_cov_excl",
        "__attribute__((noinline)) int target_func(void) { return 0; }\nint main() { return target_func(); }\n",
    );
    let output_no_hooks = td.path().join("unit_cov_excl_nohook");
    let output_hooks = td.path().join("unit_cov_excl_hooked");

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

    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");
    let hooks_toml = write_hooks(
        td.path(),
        "unit_cov_excl",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            target_va,
            shellcode_toml(NOP),
        ),
    );
    let result_hooks = rewrite_hooked(&input, &output_hooks, &hooks_toml);

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
    let td = test_dir();
    let input = compile_bin(td.path(), "unit_err_sym", "int main() { return 0; }\n");
    let output = td.path().join("unit_err_sym_out");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_err_sym",
        &format!(
            "[[hook]]\ntarget = \"nonexistent_function\"\nmode = \"pre\"\nshellcode = {}\n",
            shellcode_toml(NOP),
        ),
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
        msg
    );
}

#[test]
fn test_hook_chaining_same_target() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_chain",
        "__attribute__((noinline)) int target_func(void) { return 42; }\nint main() { return target_func(); }\n",
    );
    let output = td.path().join("unit_chain_out");
    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_chain",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n\n\
             [[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            target_va,
            shellcode_toml(NOP),
            target_va,
            shellcode_toml(NOP),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert!(
        result.hooks_applied >= 1,
        "at least one chained hook applied"
    );
    assert_eq!(
        run(&output),
        42,
        "chained NOP hooks should preserve exit code 42"
    );
}

#[test]
fn test_hook_chaining_mixed_modes_supported() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_chain_mix",
        "__attribute__((noinline)) int target_func(void) { return 0; }\nint main() { return target_func(); }\n",
    );
    let output = td.path().join("unit_chain_mix_out");
    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_chain_mix",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n\n\
             [[hook]]\ntarget = \"0x{:x}\"\nmode = \"post\"\nshellcode = {}\n",
            target_va,
            shellcode_toml(NOP),
            target_va,
            shellcode_toml(NOP),
        ),
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
    let result = rewrite(&config).expect("mixed-mode hooks should succeed");
    assert!(
        result.hooks_applied >= 1,
        "mixed-mode hooks should be applied"
    );
}

#[test]
fn test_hook_error_handler_without_library() {
    let td = test_dir();
    let hooks_toml = write_hooks(
        td.path(),
        "unit_err_nolib",
        "[[hook]]\ntarget = \"main\"\nmode = \"pre\"\nhandler = \"my_handler\"\n",
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
        msg
    );
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn test_hook_plt_stub() {
    // PLT hooking is ELF x86_64-specific (uses objdump for PLT VA resolution)
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_plt_hook",
        "#include <stdio.h>\nint main() { puts(\"hello\"); return 0; }\n",
    );
    let output = td.path().join("unit_plt_hook_out");
    let plt_va = find_plt_va(&input, "puts").expect("puts@plt not found");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_plt_hook",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            plt_va,
            shellcode_toml(NOP),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        0,
        "PLT pre-hook NOP should preserve exit code 0"
    );
}

#[test]
fn test_hook_no_coverage_mode() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "unit_nocov",
        "__attribute__((noinline)) int target_func(void) { return 42; }\nint main() { return target_func(); }\n",
    );
    let output = td.path().join("unit_nocov_out");
    let target_va = find_symbol_va(&input, "target_func").expect("target_func not found");

    let hooks_toml = write_hooks(
        td.path(),
        "unit_nocov",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            target_va,
            shellcode_toml(NOP),
        ),
    );

    let result = rewrite_no_cov(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(result.blocks_instrumented, 0);
    assert_eq!(
        run(&output),
        42,
        "pre-hook NOP should preserve exit code 42"
    );
}

#[test]
fn test_hook_with_condition() {
    let td = test_dir();
    let input = compile_bin(td.path(), "unit_condhook", "int main() { return 42; }\n");
    let output = td.path().join("unit_condhook_out");
    let main_va = find_symbol_va(&input, "main").expect("main not found");

    // Condition: first arg register == 0. On entry to main, argc is typically 1,
    // so the hook should NOT fire. If it did fire, it would set return value to 0.
    let hooks_toml = write_hooks(
        td.path(),
        "unit_condhook",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n\
             condition = {{ register = \"{}\", op = \"eq\", value = 0 }}\n",
            main_va,
            shellcode_toml(RETURN_0),
            FIRST_ARG_REG,
        ),
    );

    let result = rewrite_no_cov(&input, &output, &hooks_toml);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        42,
        "conditional hook should NOT fire (argc != 0)"
    );
}
