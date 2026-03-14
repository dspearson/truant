//! End-to-end hook verification tests.
//!
//! Every test compiles a real C binary, rewrites it with hooks via truant,
//! executes the result, and verifies the hook genuinely modified behaviour.
//!
//! Arch-portable shellcode is provided for both x86_64 and AArch64 via
//! the `shellcode` module. Tests that are inherently single-arch (inline
//! asm, PLT) are gated with `#[cfg(target_arch)]`.
//!
//! NOTE: AArch64 ELF hooks currently fail due to a trampoline range
//! limitation (B instruction ±128MiB vs segment placement). Tests that
//! require live hook execution are gated to x86_64 on Linux. On macOS
//! (Mach-O), AArch64 hooks work because the patcher places trampolines
//! within range.
//!
//! Run with: cargo test -p truant --test hook_e2e -- --test-threads=1

mod common;
mod shellcode;

use common::*;
use shellcode::*;

// ---------------------------------------------------------------------------
// 1. Pre-hook modifies function argument
// ---------------------------------------------------------------------------
#[test]
fn pre_hook_modifies_argument() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_prearg",
        r#"
        __attribute__((noinline))
        int add_ten(int x) { return x + 10; }
        int main() { return add_ten(5); }
    "#,
    );
    let output = td.path().join("e2e_prearg_out");
    let va = find_symbol_va(&input, "add_ten").expect("add_ten not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_prearg",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            va,
            shellcode_toml(SET_ARG_20),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        30,
        "pre-hook should change arg: add_ten(20)=30"
    );
}

// ---------------------------------------------------------------------------
// 2. Replace hook reads argument, computes triple, returns it
// ---------------------------------------------------------------------------
#[test]
fn replace_reads_args_computes() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_repcmp",
        r#"
        __attribute__((noinline))
        int target(int x) { return x; }
        int main() { return target(11); }
    "#,
    );
    let output = td.path().join("e2e_repcmp_out");
    let va = find_symbol_va(&input, "target").expect("target not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_repcmp",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\n",
            va,
            shellcode_toml(TRIPLE_ARG_TO_RETVAL),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(run(&output), 33, "replace should triple arg: 11*3=33");
}

// ---------------------------------------------------------------------------
// 3. Replace hook returns a constant
// ---------------------------------------------------------------------------
#[test]
fn replace_constant_return() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_repconst",
        r#"
        __attribute__((noinline))
        int target(void) { return 42; }
        int main() { return target(); }
    "#,
    );
    let output = td.path().join("e2e_repconst_out");
    let va = find_symbol_va(&input, "target").expect("target not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_repconst",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\n",
            va,
            shellcode_toml(RETURN_99),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(run(&output), 99, "replace should return 99, not 42");
}

// ---------------------------------------------------------------------------
// 4. Conditional replace fires when condition is TRUE
// ---------------------------------------------------------------------------
#[test]
fn conditional_fires_when_true() {
    let td = test_dir();
    let input = compile_bin(td.path(), "e2e_condtrue", "int main() { return 42; }\n");
    let output = td.path().join("e2e_condtrue_out");
    let va = find_symbol_va(&input, "main").expect("main not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_condtrue",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\n\
             condition = {{ register = \"{}\", op = \"gte\", value = 1 }}\n",
            va,
            shellcode_toml(RETURN_7),
            FIRST_ARG_REG,
        ),
    );

    let result = rewrite_no_cov(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        7,
        "condition true (argc>=1) -> hook fires -> exit 7"
    );
}

// ---------------------------------------------------------------------------
// 5. Conditional bit_set fires
// ---------------------------------------------------------------------------
#[test]
fn conditional_bit_set() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_bitset",
        r#"
        __attribute__((noinline))
        int add_ten(int x) { return x + 10; }
        int main() { return add_ten(15); }
    "#,
    );
    let output = td.path().join("e2e_bitset_out");
    let va = find_symbol_va(&input, "add_ten").expect("add_ten not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_bitset",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n\
             condition = {{ register = \"{}\", op = \"bit_set\", value = 1 }}\n",
            va,
            shellcode_toml(SET_ARG_100),
            FIRST_ARG_REG,
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        110,
        "bit_set(0) on 0x0F -> fires -> add_ten(100)=110"
    );
}

// ---------------------------------------------------------------------------
// 6. Conditional bit_clear fires
// ---------------------------------------------------------------------------
#[test]
fn conditional_bit_clear() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_bitclr",
        r#"
        __attribute__((noinline))
        int add_ten(int x) { return x + 10; }
        int main() { return add_ten(14); }
    "#,
    );
    let output = td.path().join("e2e_bitclr_out");
    let va = find_symbol_va(&input, "add_ten").expect("add_ten not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_bitclr",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n\
             condition = {{ register = \"{}\", op = \"bit_clear\", value = 1 }}\n",
            va,
            shellcode_toml(SET_ARG_100),
            FIRST_ARG_REG,
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        110,
        "bit_clear(0) on 0x0E -> fires -> add_ten(100)=110"
    );
}

// ---------------------------------------------------------------------------
// 7. Hook fires on every call to the same function
// ---------------------------------------------------------------------------
#[test]
fn hook_fires_every_call() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_every",
        r#"
        __attribute__((noinline))
        int add_one(int x) { return x + 1; }
        int main() { return add_one(0) + add_one(0) + add_one(0); }
    "#,
    );
    let output = td.path().join("e2e_every_out");
    let va = find_symbol_va(&input, "add_one").expect("add_one not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_every",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            va,
            shellcode_toml(ADD_10_TO_ARG),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(run(&output), 33, "hook should fire 3 times: (0+10+1)*3=33");
}

// ---------------------------------------------------------------------------
// 8. Recursive function with pre-hook NOP — no stack corruption
// ---------------------------------------------------------------------------
#[test]
fn recursive_function_stable() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_recurse",
        r#"
        __attribute__((noinline))
        int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }
        int main() { return factorial(5); }
    "#,
    );
    let output = td.path().join("e2e_recurse_out");
    let va = find_symbol_va(&input, "factorial").expect("factorial not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_recurse",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            va,
            shellcode_toml(NOP),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        120,
        "factorial(5)=120 must be preserved through recursive hooks"
    );
}

// ---------------------------------------------------------------------------
// 9. Chained pre-hooks accumulate mutations
// ---------------------------------------------------------------------------
#[test]
fn chained_hooks_accumulate() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_chain",
        r#"
        __attribute__((noinline))
        int identity(int x) { return x; }
        int main() { return identity(0); }
    "#,
    );
    let output = td.path().join("e2e_chain_out");
    let va = find_symbol_va(&input, "identity").expect("identity not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_chain",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n\n\
             [[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            va,
            shellcode_toml(ADD_10_TO_ARG),
            va,
            shellcode_toml(ADD_5_TO_ARG),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert!(
        result.hooks_applied >= 1,
        "at least one hook should be applied"
    );
    assert_eq!(run(&output), 15, "chained hooks: 0 + 10 + 5 = 15");
}

// ---------------------------------------------------------------------------
// 10. Mixed pre + post hooks both work
// ---------------------------------------------------------------------------
#[test]
fn mixed_pre_post_both_work() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_mixed",
        r#"
        __attribute__((noinline))
        int func(int x) { return x + 1; }
        int main() { return func(0); }
    "#,
    );
    let output = td.path().join("e2e_mixed_out");
    let va = find_symbol_va(&input, "func").expect("func not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_mixed",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n\n\
             [[hook]]\ntarget = \"0x{:x}\"\nmode = \"post\"\nshellcode = {}\n",
            va,
            shellcode_toml(SET_ARG_20),
            va,
            shellcode_toml(NOP),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert!(
        result.hooks_applied >= 1,
        "at least one hook should be applied"
    );
    assert_eq!(
        run(&output),
        21,
        "pre sets arg=20 -> func(20)=21, post NOP preserves"
    );
}

// ---------------------------------------------------------------------------
// 11. PLT replace suppresses library call (ELF x86_64 only)
// ---------------------------------------------------------------------------
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn plt_replace_suppresses_output() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_pltrepl",
        r#"
        #include <stdio.h>
        int main() { puts("hello world"); return 0; }
    "#,
    );
    let output = td.path().join("e2e_pltrepl_out");
    let plt_va = find_plt_va(&input, "puts")
        .expect("puts@plt not found — binary must be dynamically linked");

    let hooks = write_hooks(
        td.path(),
        "e2e_pltrepl",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\n",
            plt_va,
            shellcode_toml(RETURN_0),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);

    let stdout = run_stdout(&output);
    assert!(
        !stdout.contains("hello world"),
        "PLT replace should suppress puts output, but got: {:?}",
        stdout
    );
}

// ---------------------------------------------------------------------------
// 12. Library handler hook (.so based)
// ---------------------------------------------------------------------------
#[cfg(target_os = "linux")]
#[test]
fn library_handler_hook() {
    let td = test_dir();
    let handler_so = compile_so(
        td.path(),
        "e2e_handler",
        &format!(
            r#"
        #include <stdint.h>
        void my_handler(void *ctx) {{
            *(uint64_t *)ctx = 77;
        }}
    "#,
        ),
    );

    let input = compile_bin(
        td.path(),
        "e2e_libhook",
        r#"
        __attribute__((noinline))
        int target(void) { return 42; }
        int main() { return target(); }
    "#,
    );
    let output = td.path().join("e2e_libhook_out");

    let hooks = write_hooks(
        td.path(),
        "e2e_libhook",
        &format!(
            "[hooks]\nlibrary = \"{}\"\n\n\
             [[hook]]\ntarget = \"target\"\nmode = \"replace\"\nhandler = \"my_handler\"\n",
            handler_so.display()
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    let preload = result
        .hook_preload_lib_path
        .as_ref()
        .expect("companion preload .so should be generated");
    assert!(
        preload.exists(),
        "companion preload .so must exist at {:?}",
        preload
    );

    let code = run_with_preload(&output, preload);
    assert_eq!(code, 77, "library handler should set rax=77 -> exit 77");
}

// ---------------------------------------------------------------------------
// 13. Return hook modifies return value
// ---------------------------------------------------------------------------
#[test]
fn return_hook_modifies_retval() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_rethook",
        r#"
        __attribute__((noinline))
        int target(void) { return 42; }
        int main() { return target(); }
    "#,
    );
    let output = td.path().join("e2e_rethook_out");
    let va = find_symbol_va(&input, "target").expect("target not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_rethook",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"return\"\nshellcode = {}\n",
            va,
            shellcode_toml(ADD_8_TO_RETVAL),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(run(&output), 50, "return hook should add 8: 42+8=50");
}

// ---------------------------------------------------------------------------
// 14. Toggle disabled — hook skipped
// ---------------------------------------------------------------------------
#[test]
fn toggle_disabled_skips_hook() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_togoff",
        r#"
        __attribute__((noinline))
        int target(void) { return 42; }
        int main() { return target(); }
    "#,
    );
    let output = td.path().join("e2e_togoff_out");
    let va = find_symbol_va(&input, "target").expect("target not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_togoff",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\nenabled = false\n",
            va,
            shellcode_toml(RETURN_99),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        42,
        "enabled=false -> hook skipped -> original exit 42"
    );
}

// ---------------------------------------------------------------------------
// 15. Toggle enabled — hook fires
// ---------------------------------------------------------------------------
#[test]
fn toggle_enabled_fires_hook() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_togon",
        r#"
        __attribute__((noinline))
        int target(void) { return 42; }
        int main() { return target(); }
    "#,
    );
    let output = td.path().join("e2e_togon_out");
    let va = find_symbol_va(&input, "target").expect("target not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_togon",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\nenabled = true\n",
            va,
            shellcode_toml(RETURN_99),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(run(&output), 99, "enabled=true -> hook fires -> exit 99");
}

// ---------------------------------------------------------------------------
// 16. Callee-saved registers preserved across hook (x86_64 only — uses inline asm)
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[test]
fn callee_saved_regs_preserved() {
    let td = test_dir();
    let input = compile_bin_flags(
        td.path(),
        "e2e_callee",
        r#"
        __attribute__((noinline))
        int target(int x) { return x; }
        int main() {
            int r;
            asm volatile(
                "mov $10, %%rbx\n"
                "mov $20, %%r12\n"
                "mov $30, %%r13\n"
                "mov $40, %%r14\n"
                "mov $50, %%r15\n"
                "xor %%edi, %%edi\n"
                "call target\n"
                "add %%ebx, %%eax\n"
                "add %%r12d, %%eax\n"
                "add %%r13d, %%eax\n"
                "add %%r14d, %%eax\n"
                "add %%r15d, %%eax\n"
                : "=a"(r)
                :
                : "rbx", "r12", "r13", "r14", "r15",
                  "rdi", "rcx", "rdx", "rsi",
                  "r8", "r9", "r10", "r11", "memory"
            );
            return r;
        }
    "#,
        &["-O0"],
    );
    let output = td.path().join("e2e_callee_out");
    let va = find_symbol_va(&input, "target").expect("target not found");

    // Shellcode clobbers rbx, r12-r15 (machine regs, NOT RegContext fields).
    // xor ebx,ebx; xor r12d,r12d; xor r13d,r13d; xor r14d,r14d; xor r15d,r15d; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_callee",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\n\
             shellcode = [0x31, 0xDB, 0x45, 0x31, 0xE4, 0x45, 0x31, 0xED, 0x45, 0x31, 0xF6, 0x45, 0x31, 0xFF, 0xC3]\n",
            va
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        150,
        "callee-saved regs must survive hook: 0+10+20+30+40+50=150"
    );
}

// ---------------------------------------------------------------------------
// 17. Stripped binary — hook by hex VA, no symbols
// ---------------------------------------------------------------------------
#[test]
fn stripped_binary_hook_by_va() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_strip",
        r#"
        __attribute__((noinline))
        int target(void) { return 42; }
        int main() { return target(); }
    "#,
    );
    let va = find_symbol_va(&input, "target").expect("target not found");

    let stripped = td.path().join("e2e_strip_stripped");
    std::fs::copy(&input, &stripped).unwrap();
    let mut strip_cmd = std::process::Command::new("strip");
    if cfg!(target_os = "macos") {
        strip_cmd.arg(stripped.to_str().unwrap());
    } else {
        strip_cmd.args(["-s", stripped.to_str().unwrap()]);
    }
    let st = strip_cmd.status().unwrap();
    assert!(st.success(), "strip failed");
    assert!(
        find_symbol_va(&stripped, "target").is_none(),
        "target should not be found after stripping"
    );

    let output = td.path().join("e2e_strip_out");
    let hooks = write_hooks(
        td.path(),
        "e2e_strip",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\n",
            va,
            shellcode_toml(RETURN_99),
        ),
    );

    let result = rewrite_hooked(&stripped, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(run(&output), 99, "hook by VA on stripped binary -> exit 99");
}

// ---------------------------------------------------------------------------
// 18. Selective multi-function hooking
// ---------------------------------------------------------------------------
#[test]
fn selective_multi_function() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_select",
        r#"
        __attribute__((noinline)) int func_a(void) { return 10; }
        __attribute__((noinline)) int func_b(void) { return 20; }
        __attribute__((noinline)) int func_c(void) { return 30; }
        int main() { return func_a() + func_b() + func_c(); }
    "#,
    );
    let output = td.path().join("e2e_select_out");
    let va_b = find_symbol_va(&input, "func_b").expect("func_b not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_select",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\n",
            va_b,
            shellcode_toml(RETURN_100),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(run(&output), 140, "only func_b hooked: 10+100+30=140");
}

// ---------------------------------------------------------------------------
// 19. Hook preserves stdio output
// ---------------------------------------------------------------------------
#[test]
fn hook_preserves_stdio() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_stdio",
        r#"
        #include <stdio.h>
        int main() { printf("hello 42\n"); return 0; }
    "#,
    );
    let output = td.path().join("e2e_stdio_out");
    let va = find_symbol_va(&input, "main").expect("main not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_stdio",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            va,
            shellcode_toml(NOP),
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);

    let stdout = run_stdout(&output);
    assert!(
        stdout.contains("hello 42"),
        "stdio output should be preserved, got: {:?}",
        stdout
    );
}

// ---------------------------------------------------------------------------
// 20. No-coverage mode: hooks only, zero instrumentation
// ---------------------------------------------------------------------------
#[test]
fn no_coverage_hooks_only() {
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_nocov",
        r#"
        __attribute__((noinline))
        int target(void) { return 42; }
        int main() { return target(); }
    "#,
    );
    let output = td.path().join("e2e_nocov_out");
    let va = find_symbol_va(&input, "target").expect("target not found");

    let hooks = write_hooks(
        td.path(),
        "e2e_nocov",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
            va,
            shellcode_toml(NOP),
        ),
    );

    let result = rewrite_no_cov(&input, &output, &hooks);
    assert_eq!(result.blocks_instrumented, 0, "no-coverage: zero blocks");
    assert!(result.hooks_applied >= 1, "hooks should still be applied");
    assert_eq!(run(&output), 42, "binary should run normally");
}
