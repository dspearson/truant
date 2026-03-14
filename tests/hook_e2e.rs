//! End-to-end hook verification tests.
//!
//! Every test compiles a real C binary, rewrites it with hooks via truant,
//! executes the result, and verifies the hook genuinely modified behaviour.
//!
//! Run with: cargo test -p truant --test hook_e2e -- --test-threads=1

mod common;
use common::*;

// ============================================================================
// RegContext layout (x86_64, 18 * 8 = 144 bytes)
// ============================================================================
//
//   +0   rax     +64  r8      +128 rip
//   +8   rbx     +72  r9      +136 rflags
//   +16  rcx     +80  r10
//   +24  rdx     +88  r11
//   +32  rsi     +96  r12
//   +40  rdi     +104 r13
//   +48  rbp     +112 r14
//   +56  rsp     +120 r15
//
// Shellcode is called with: rdi = &RegContext

// ============================================================================
// Tests
// ============================================================================

// ---------------------------------------------------------------------------
// 1. Pre-hook modifies function argument
// ---------------------------------------------------------------------------
#[test]
fn pre_hook_modifies_argument() {
    // add_ten(5) would return 15. Pre-hook writes 20 to RegContext.rdi (+40),
    // so add_ten receives 20 and returns 30.
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

    // mov qword [rdi+0x28], 20; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_prearg",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0x48, 0xC7, 0x47, 0x28, 0x14, 0x00, 0x00, 0x00, 0xC3]
"#,
            va
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
    // target(11) would return 11. Replace reads rdi(+40), triples, stores rax(+0).
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

    // mov rax,[rdi+0x28]; imul rax,rax,3; mov [rdi],rax; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_repcmp",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "replace"
shellcode = [0x48, 0x8B, 0x47, 0x28, 0x48, 0x6B, 0xC0, 0x03, 0x48, 0x89, 0x07, 0xC3]
"#,
            va
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

    // mov qword [rdi], 99; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_repconst",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "replace"
shellcode = [0x48, 0xC7, 0x07, 0x63, 0x00, 0x00, 0x00, 0xC3]
"#,
            va
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
    // On entry to main, rdi = argc = 1. Condition: rdi >= 1 → true → fires.
    let td = test_dir();
    let input = compile_bin(
        td.path(),
        "e2e_condtrue",
        r#"
        int main() { return 42; }
    "#,
    );
    let output = td.path().join("e2e_condtrue_out");
    let va = find_symbol_va(&input, "main").expect("main not found");

    // mov qword [rdi], 7; ret  — sets rax=7 via RegContext
    let hooks = write_hooks(
        td.path(),
        "e2e_condtrue",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "replace"
shellcode = [0x48, 0xC7, 0x07, 0x07, 0x00, 0x00, 0x00, 0xC3]
condition = {{ register = "rdi", op = "gte", value = 1 }}
"#,
            va
        ),
    );

    let result = rewrite_no_cov(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        7,
        "condition true (argc>=1) → hook fires → exit 7"
    );
}

// ---------------------------------------------------------------------------
// 5. Conditional bit_set fires
// ---------------------------------------------------------------------------
#[test]
fn conditional_bit_set() {
    // add_ten(15): rdi=15=0x0F. bit_set with value=1 checks bit 0.
    // 15 & 1 = 1 → condition true → hook fires → changes rdi to 100 → exit 110.
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

    // mov qword [rdi+0x28], 100; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_bitset",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0x48, 0xC7, 0x47, 0x28, 0x64, 0x00, 0x00, 0x00, 0xC3]
condition = {{ register = "rdi", op = "bit_set", value = 1 }}
"#,
            va
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        110,
        "bit_set(0) on 0x0F → fires → add_ten(100)=110"
    );
}

// ---------------------------------------------------------------------------
// 6. Conditional bit_clear fires
// ---------------------------------------------------------------------------
#[test]
fn conditional_bit_clear() {
    // add_ten(14): rdi=14=0x0E. bit_clear with value=1 checks bit 0 is clear.
    // 14 & 1 = 0 → condition true → hook fires → changes rdi to 100 → exit 110.
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

    // mov qword [rdi+0x28], 100; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_bitclr",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0x48, 0xC7, 0x47, 0x28, 0x64, 0x00, 0x00, 0x00, 0xC3]
condition = {{ register = "rdi", op = "bit_clear", value = 1 }}
"#,
            va
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        110,
        "bit_clear(0) on 0x0E → fires → add_ten(100)=110"
    );
}

// ---------------------------------------------------------------------------
// 7. Hook fires on every call to the same function
// ---------------------------------------------------------------------------
#[test]
fn hook_fires_every_call() {
    // add_one(0) called 3 times. Pre-hook adds 10 to rdi each time.
    // Each call: add_one(0+10) = 11. Total: 11+11+11 = 33.
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

    // mov rax,[rdi+0x28]; add rax,10; mov [rdi+0x28],rax; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_every",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0x48, 0x8B, 0x47, 0x28, 0x48, 0x83, 0xC0, 0x0A, 0x48, 0x89, 0x47, 0x28, 0xC3]
"#,
            va
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
    // factorial(5) = 120. Pre-hook NOP must not corrupt the stack.
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
            r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]
"#,
            va
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
    // identity(0): hook A adds 10 to rdi, hook B adds 5. Result: identity(15)=15.
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

    // Hook A: add 10 to rdi(+40)
    // mov rax,[rdi+0x28]; add rax,10; mov [rdi+0x28],rax; ret
    // Hook B: add 5 to rdi(+40)
    // mov rax,[rdi+0x28]; add rax,5; mov [rdi+0x28],rax; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_chain",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0x48, 0x8B, 0x47, 0x28, 0x48, 0x83, 0xC0, 0x0A, 0x48, 0x89, 0x47, 0x28, 0xC3]

[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0x48, 0x8B, 0x47, 0x28, 0x48, 0x83, 0xC0, 0x05, 0x48, 0x89, 0x47, 0x28, 0xC3]
"#,
            va, va
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 2);
    assert_eq!(run(&output), 15, "chained hooks: 0 + 10 + 5 = 15");
}

// ---------------------------------------------------------------------------
// 10. Mixed pre + post hooks both work
// ---------------------------------------------------------------------------
#[test]
fn mixed_pre_post_both_work() {
    // func(0): pre-hook writes 20 to rdi → func(20) → returns 21.
    // Post-hook is NOP, verifies it doesn't break the chain.
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
            r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0x48, 0xC7, 0x47, 0x28, 0x14, 0x00, 0x00, 0x00, 0xC3]

[[hook]]
target = "0x{:x}"
mode = "post"
shellcode = [0xC3]
"#,
            va, va
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 2);
    assert_eq!(
        run(&output),
        21,
        "pre sets arg=20 → func(20)=21, post NOP preserves"
    );
}

// ---------------------------------------------------------------------------
// 11. PLT replace suppresses library call
// ---------------------------------------------------------------------------
#[test]
fn plt_replace_suppresses_output() {
    // Replace puts@plt so it never actually calls puts.
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

    // Replace: set rax=0 (success) and return.
    // mov qword [rdi], 0; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_pltrepl",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "replace"
shellcode = [0x48, 0xC7, 0x07, 0x00, 0x00, 0x00, 0x00, 0xC3]
"#,
            plt_va
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
#[test]
fn library_handler_hook() {
    // Handler .so sets RegContext.rax = 77.
    let td = test_dir();
    let handler_so = compile_so(
        td.path(),
        "e2e_handler",
        r#"
        #include <stdint.h>
        void my_handler(void *ctx) {
            *(uint64_t *)ctx = 77;
        }
    "#,
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
            r#"
[hooks]
library = "{}"

[[hook]]
target = "target"
mode = "replace"
handler = "my_handler"
"#,
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
    assert_eq!(code, 77, "library handler should set rax=77 → exit 77");
}

// ---------------------------------------------------------------------------
// 13. Return hook modifies return value
// ---------------------------------------------------------------------------
#[test]
fn return_hook_modifies_retval() {
    // target() returns 42. Return hook adds 8 to rax → exit 50.
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

    // mov rax,[rdi]; add rax,8; mov [rdi],rax; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_rethook",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "return"
shellcode = [0x48, 0x8B, 0x07, 0x48, 0x83, 0xC0, 0x08, 0x48, 0x89, 0x07, 0xC3]
"#,
            va
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
    // Replace hook with enabled=false → toggle byte=0 → hook skipped → exit 42.
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

    // Would set rax=99 if it fired.
    let hooks = write_hooks(
        td.path(),
        "e2e_togoff",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "replace"
shellcode = [0x48, 0xC7, 0x07, 0x63, 0x00, 0x00, 0x00, 0xC3]
enabled = false
"#,
            va
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(
        run(&output),
        42,
        "enabled=false → hook skipped → original exit 42"
    );
}

// ---------------------------------------------------------------------------
// 15. Toggle enabled — hook fires
// ---------------------------------------------------------------------------
#[test]
fn toggle_enabled_fires_hook() {
    // Same as above but enabled=true → toggle byte=1 → hook fires → exit 99.
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
            r#"
[[hook]]
target = "0x{:x}"
mode = "replace"
shellcode = [0x48, 0xC7, 0x07, 0x63, 0x00, 0x00, 0x00, 0xC3]
enabled = true
"#,
            va
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(run(&output), 99, "enabled=true → hook fires → exit 99");
}

// ---------------------------------------------------------------------------
// 16. Callee-saved registers preserved across hook
// ---------------------------------------------------------------------------
#[test]
fn callee_saved_regs_preserved() {
    // main sets callee-saved regs (rbx, r12-r15) to known values, calls
    // target() through inline asm so they're live across the call, then sums
    // them. A pre-hook that clobbers those machine regs must not affect the
    // result because the trampoline's save/restore restores them from RegContext.
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

    // Shellcode clobbers rbx, r12, r13, r14, r15 (machine regs, NOT RegContext fields).
    // xor ebx,ebx; xor r12d,r12d; xor r13d,r13d; xor r14d,r14d; xor r15d,r15d; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_callee",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0x31, 0xDB, 0x45, 0x31, 0xE4, 0x45, 0x31, 0xED, 0x45, 0x31, 0xF6, 0x45, 0x31, 0xFF, 0xC3]
"#,
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
    // Compile with symbols, find VA, strip, hook by VA.
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

    // Strip symbols.
    let stripped = td.path().join("e2e_strip_stripped");
    std::fs::copy(&input, &stripped).unwrap();
    let st = std::process::Command::new("strip")
        .args(["-s", stripped.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(st.success(), "strip failed");

    // Verify symbol is gone.
    assert!(
        find_symbol_va(&stripped, "target").is_none(),
        "target should not be found after stripping"
    );

    let output = td.path().join("e2e_strip_out");

    // Replace hook by hex VA → exit 99.
    let hooks = write_hooks(
        td.path(),
        "e2e_strip",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "replace"
shellcode = [0x48, 0xC7, 0x07, 0x63, 0x00, 0x00, 0x00, 0xC3]
"#,
            va
        ),
    );

    let result = rewrite_hooked(&stripped, &output, &hooks);
    assert_eq!(result.hooks_applied, 1);
    assert_eq!(run(&output), 99, "hook by VA on stripped binary → exit 99");
}

// ---------------------------------------------------------------------------
// 18. Selective multi-function hooking
// ---------------------------------------------------------------------------
#[test]
fn selective_multi_function() {
    // Three functions: a()=10, b()=20, c()=30. Hook only b() to return 100.
    // Original: 10+20+30=60. With hook: 10+100+30=140.
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

    // Replace func_b: mov qword [rdi], 100; ret
    let hooks = write_hooks(
        td.path(),
        "e2e_select",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "replace"
shellcode = [0x48, 0xC7, 0x07, 0x64, 0x00, 0x00, 0x00, 0xC3]
"#,
            va_b
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
    // Pre-hook NOP on main should not interfere with printf output.
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
            r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]
"#,
            va
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

    // Pre-hook NOP.
    let hooks = write_hooks(
        td.path(),
        "e2e_nocov",
        &format!(
            r#"
[[hook]]
target = "0x{:x}"
mode = "pre"
shellcode = [0xC3]
"#,
            va
        ),
    );

    let result = rewrite_no_cov(&input, &output, &hooks);
    assert_eq!(result.blocks_instrumented, 0, "no-coverage: zero blocks");
    assert!(result.hooks_applied >= 1, "hooks should still be applied");
    assert_eq!(run(&output), 42, "binary should run normally");
}
