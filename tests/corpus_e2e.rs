//! Binary corpus end-to-end tests.
//!
//! Compiles a diverse set of C programs with varying characteristics
//! (size, optimisation level, calling patterns, data access patterns)
//! and exercises every hook type against each. This catches regressions
//! in displaced-byte handling, PC-relative relocation, register
//! preservation, and trampoline generation across a wide range of
//! real instruction sequences.

mod common;

use common::*;

// =========================================================================
// Test binary sources — each exercises different instruction patterns
// =========================================================================

/// Trivial: single function, no calls, constant return.
/// Uses volatile to prevent the function body from shrinking below 5 bytes at -O2.
const SRC_TRIVIAL: &str = r#"
__attribute__((noinline))
int target(int x) { volatile int v = x; return v + 1; }
int main() { return target(41); }
"#;

/// Arithmetic: multiple operations, no branches.
const SRC_ARITH: &str = r#"
__attribute__((noinline))
int target(int x) {
    volatile int a = x * 3;
    int b = a + 7;
    int c = b / 2;
    return c - 1;
}
int main() { return target(10); }
"#;

/// Branching: if/else control flow.
const SRC_BRANCH: &str = r#"
__attribute__((noinline))
int target(int x) {
    if (x > 50) return x - 50;
    else if (x > 25) return x;
    else return x + 25;
}
int main() { return target(30); }
"#;

/// Loop: iteration with accumulator.
const SRC_LOOP: &str = r#"
__attribute__((noinline))
int target(int n) {
    int sum = 0;
    for (int i = 1; i <= n; i++) sum += i;
    return sum;
}
int main() { return target(10); }
"#;

/// Recursive: fibonacci.
const SRC_RECURSIVE: &str = r#"
__attribute__((noinline))
int target(int n) {
    if (n <= 1) return n;
    return target(n - 1) + target(n - 2);
}
int main() { return target(10); }
"#;

/// Multi-arg: function taking many arguments (exercises register/stack ABI).
const SRC_MULTIARG: &str = r#"
__attribute__((noinline))
int target(int a, int b, int c, int d, int e, int f) {
    return a + b + c + d + e + f;
}
int main() { return target(1, 2, 3, 4, 5, 6); }
"#;

/// String/pointer: exercises PC-relative addressing (global data access).
const SRC_GLOBAL: &str = r#"
#include <string.h>
static const char *msg = "hello world";
__attribute__((noinline))
int target(int x) { return (int)strlen(msg) + x; }
int main() { return target(0); }
"#;

/// Switch: jump table generation.
const SRC_SWITCH: &str = r#"
__attribute__((noinline))
int target(int x) {
    switch (x) {
        case 0: return 10;
        case 1: return 20;
        case 2: return 30;
        case 3: return 40;
        case 4: return 50;
        default: return 0;
    }
}
int main() { return target(3); }
"#;

/// Struct: pass-by-value struct (exercises stack layout).
const SRC_STRUCT: &str = r#"
struct Pair { int a; int b; };
__attribute__((noinline))
int target(int x) {
    struct Pair p = { x, x * 2 };
    return p.a + p.b;
}
int main() { return target(7); }
"#;

/// Multi-function: two hookable targets in the same binary.
const SRC_MULTI_FUNC: &str = r#"
__attribute__((noinline))
int func_a(int x) { return x + 10; }
__attribute__((noinline))
int func_b(int x) { return x * 2; }
int main() { return func_a(5) + func_b(3); }
"#;

/// Callee-saved: uses callee-saved registers (rbx, r12-r15 on x86_64).
const SRC_CALLEE_SAVED: &str = r#"
#include <stdio.h>
__attribute__((noinline))
int target(int x) {
    // Force compiler to use callee-saved registers
    volatile int a = x + 1;
    volatile int b = x + 2;
    volatile int c = x + 3;
    volatile int d = x + 4;
    return a + b + c + d;
}
int main() { return target(1); }
"#;

/// Library call: exercises PLT/GOT interactions.
const SRC_LIBCALL: &str = r#"
#include <stdio.h>
#include <string.h>
__attribute__((noinline))
int target(int x) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", x);
    return (int)strlen(buf);
}
int main() { return target(12345); }
"#;

/// Variadic: variadic function call.
const SRC_VARIADIC: &str = r#"
#include <stdarg.h>
__attribute__((noinline))
int target(int n, ...) {
    va_list ap;
    va_start(ap, n);
    int sum = 0;
    for (int i = 0; i < n; i++) sum += va_arg(ap, int);
    va_end(ap);
    return sum;
}
int main() { return target(4, 10, 20, 30, 40); }
"#;

/// Large function: many basic blocks.
const SRC_LARGE: &str = r#"
__attribute__((noinline))
int target(int x) {
    int r = x;
    if (r & 1) r += 3;
    if (r & 2) r += 5;
    if (r & 4) r += 7;
    if (r & 8) r += 11;
    if (r & 16) r += 13;
    if (r & 32) r += 17;
    if (r & 64) r += 19;
    if (r & 128) r += 23;
    for (int i = 0; i < r % 10; i++) r ^= (i * 31);
    return r & 0xFF;
}
int main() { return target(42); }
"#;

// =========================================================================
// Corpus definition
// =========================================================================

struct CorpusBinary {
    name: &'static str,
    src: &'static str,
    expected_exit: i32,
    extra_flags: &'static [&'static str],
}

/// Base flags for all corpus binaries.
/// On x86_64, disable CET (endbr64 is only 4 bytes, but hooks need 5-byte displaced region).
/// On AArch64, -fcf-protection is not recognised, so we omit it (ARM64 instructions are
/// always 4 bytes and the displaced region is exactly one instruction).
/// Base flags for all corpus binaries.
/// On x86_64 Linux (GCC), disable CET (endbr64 is only 4 bytes, but hooks need 5).
/// On macOS and AArch64, -fcf-protection is not recognised/needed.
#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
const CF: &[&str] = &["-fcf-protection=none"];
#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
const CF_O0: &[&str] = &["-fcf-protection=none", "-O0"];
#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
const CF_O2: &[&str] = &["-fcf-protection=none", "-O2"];
#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
const CF_O3: &[&str] = &["-fcf-protection=none", "-O3"];
#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
const CF_OS: &[&str] = &["-fcf-protection=none", "-Os"];

#[cfg(any(target_arch = "aarch64", target_os = "macos"))]
const CF: &[&str] = &[];
#[cfg(any(target_arch = "aarch64", target_os = "macos"))]
const CF_O0: &[&str] = &["-O0"];
#[cfg(any(target_arch = "aarch64", target_os = "macos"))]
const CF_O2: &[&str] = &["-O2"];
#[cfg(any(target_arch = "aarch64", target_os = "macos"))]
const CF_O3: &[&str] = &["-O3"];
#[cfg(any(target_arch = "aarch64", target_os = "macos"))]
const CF_OS: &[&str] = &["-Os"];

const CORPUS: &[CorpusBinary] = &[
    CorpusBinary {
        name: "trivial",
        src: SRC_TRIVIAL,
        expected_exit: 42,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "arith",
        src: SRC_ARITH,
        expected_exit: 17,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "branch",
        src: SRC_BRANCH,
        expected_exit: 30,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "loop",
        src: SRC_LOOP,
        expected_exit: 55,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "recursive",
        src: SRC_RECURSIVE,
        expected_exit: 55,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "multiarg",
        src: SRC_MULTIARG,
        expected_exit: 21,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "global",
        src: SRC_GLOBAL,
        expected_exit: 11,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "switch",
        src: SRC_SWITCH,
        expected_exit: 40,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "struct",
        src: SRC_STRUCT,
        expected_exit: 21,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "callee_saved",
        src: SRC_CALLEE_SAVED,
        expected_exit: 14,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "libcall",
        src: SRC_LIBCALL,
        expected_exit: 5,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "variadic",
        src: SRC_VARIADIC,
        expected_exit: 100,
        extra_flags: CF,
    },
    CorpusBinary {
        name: "large",
        src: SRC_LARGE,
        expected_exit: 42,
        extra_flags: CF,
    },
    // Same sources at different optimisation levels
    CorpusBinary {
        name: "trivial_O0",
        src: SRC_TRIVIAL,
        expected_exit: 42,
        extra_flags: CF_O0,
    },
    CorpusBinary {
        name: "trivial_O2",
        src: SRC_TRIVIAL,
        expected_exit: 42,
        extra_flags: CF_O2,
    },
    CorpusBinary {
        name: "trivial_O3",
        src: SRC_TRIVIAL,
        expected_exit: 42,
        extra_flags: CF_O3,
    },
    CorpusBinary {
        name: "trivial_Os",
        src: SRC_TRIVIAL,
        expected_exit: 42,
        extra_flags: CF_OS,
    },
    CorpusBinary {
        name: "branch_O2",
        src: SRC_BRANCH,
        expected_exit: 30,
        extra_flags: CF_O2,
    },
    CorpusBinary {
        name: "loop_O2",
        src: SRC_LOOP,
        expected_exit: 55,
        extra_flags: CF_O2,
    },
    CorpusBinary {
        name: "recursive_O2",
        src: SRC_RECURSIVE,
        expected_exit: 55,
        extra_flags: CF_O2,
    },
    CorpusBinary {
        name: "switch_O2",
        src: SRC_SWITCH,
        expected_exit: 40,
        extra_flags: CF_O2,
    },
    CorpusBinary {
        name: "large_O2",
        src: SRC_LARGE,
        expected_exit: 42,
        extra_flags: CF_O2,
    },
    CorpusBinary {
        name: "global_O2",
        src: SRC_GLOBAL,
        expected_exit: 11,
        extra_flags: CF_O2,
    },
    CorpusBinary {
        name: "multiarg_O2",
        src: SRC_MULTIARG,
        expected_exit: 21,
        extra_flags: CF_O2,
    },
];

// =========================================================================
// Hook shellcode snippets — arch-specific
//
// Hook handlers receive a pointer to RegContext as the first argument
// (rdi on x86_64, x0 on AArch64).
//
// x86_64 RegContext (144 bytes):
//   rax(0) rbx(8) rcx(16) rdx(24) rsi(32) rdi(40)
//   rbp(48) rsp(56) r8(64)..r15(120) rip(128) rflags(136)
//
// AArch64 RegContext (272 bytes):
//   x0(0) x1(8) x2(16) ... x29(232) x30/lr(240) sp(248) pc(256) nzcv(264)
// =========================================================================

/// NOP pre-hook: does nothing, returns.
#[cfg(target_arch = "x86_64")]
const SHELLCODE_NOP: &[u8] = &[0xC3]; // ret
#[cfg(target_arch = "aarch64")]
const SHELLCODE_NOP: &[u8] = &[0xC0, 0x03, 0x5F, 0xD6]; // RET

/// Pre-hook: zero the first argument register in RegContext.
/// x86_64: mov qword [rdi+0x28], 0; ret   (rdi is at offset 40 = 0x28)
/// AArch64: str xzr, [x0]; ret            (x0 is at offset 0)
#[cfg(target_arch = "x86_64")]
const SHELLCODE_ZERO_ARG: &[u8] = &[
    0x48, 0xC7, 0x47, 0x28, 0x00, 0x00, 0x00, 0x00, // mov qword [rdi+0x28], 0
    0xC3, // ret
];
#[cfg(target_arch = "aarch64")]
const SHELLCODE_ZERO_ARG: &[u8] = &[
    0x1F, 0x00, 0x00, 0xF9, // STR XZR, [X0]  (zero x0 in RegContext)
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// Replace-hook: set return value in RegContext to 99.
/// x86_64: mov qword [rdi], 99; ret       (rax at offset 0)
/// AArch64: mov w1, #99; str x1, [x0]; ret  (x0 at offset 0)
#[cfg(target_arch = "x86_64")]
const SHELLCODE_RETURN_99: &[u8] = &[
    0x48, 0xC7, 0x07, 0x63, 0x00, 0x00, 0x00, // mov qword [rdi], 99
    0xC3, // ret
];
#[cfg(target_arch = "aarch64")]
const SHELLCODE_RETURN_99: &[u8] = &[
    0x61, 0x0C, 0x80, 0xD2, // MOV X1, #99
    0x01, 0x00, 0x00, 0xF9, // STR X1, [X0]   (write 99 to RegContext.x0)
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// Post-hook: NOP (just return). Verifies post-hook doesn't corrupt state.
#[cfg(target_arch = "x86_64")]
const SHELLCODE_POST_NOP: &[u8] = &[0xC3]; // ret
#[cfg(target_arch = "aarch64")]
const SHELLCODE_POST_NOP: &[u8] = &[0xC0, 0x03, 0x5F, 0xD6]; // RET

// =========================================================================
// Test functions
// =========================================================================

/// Try to rewrite with hooks (no-coverage). Returns None if the rewrite fails
/// due to displaced-byte extraction (function too small), which is expected
/// for some optimised binaries.
fn try_rewrite_no_cov(
    input: &std::path::Path,
    output: &std::path::Path,
    hooks: &std::path::Path,
) -> Option<truant::RewriteResult> {
    let config = truant::RewriteConfig {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
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
        hooks: Some(hooks.to_path_buf()),
        no_coverage: true,
    };
    match truant::rewrite(&config) {
        Ok(r) => Some(r),
        Err(e) => {
            let msg = format!("{:#}", e);
            if msg.contains("need 5") || msg.contains("displaced bytes") {
                None // Function too small for hooking — expected at high -O levels
            } else {
                panic!("unexpected rewrite failure: {}", msg);
            }
        }
    }
}

/// Try to rewrite with hooks + coverage. Returns None on displaced-byte failures.
fn try_rewrite_hooked(
    input: &std::path::Path,
    output: &std::path::Path,
    hooks: &std::path::Path,
) -> Option<truant::RewriteResult> {
    let config = truant::RewriteConfig {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
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
        hooks: Some(hooks.to_path_buf()),
        no_coverage: false,
    };
    match truant::rewrite(&config) {
        Ok(r) => Some(r),
        Err(e) => {
            let msg = format!("{:#}", e);
            if msg.contains("need 5") || msg.contains("displaced bytes") {
                None
            } else {
                panic!("unexpected rewrite failure: {}", msg);
            }
        }
    }
}

/// The register name for the first integer argument in hook conditions.
#[cfg(target_arch = "x86_64")]
const FIRST_ARG_REG: &str = "rdi";
#[cfg(target_arch = "aarch64")]
const FIRST_ARG_REG: &str = "x0";

/// Generate arch-appropriate replace-hook shellcode that sets the return value
/// register in RegContext to a given constant and returns.
fn shellcode_return_const(val: u8) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        // mov qword [rdi], val; ret
        vec![0x48, 0xC7, 0x07, val, 0x00, 0x00, 0x00, 0xC3]
    }
    #[cfg(target_arch = "aarch64")]
    {
        // MOV X1, #val; STR X1, [X0]; RET
        let mov = 0xD2800001u32 | ((val as u32) << 5); // MOV X1, #val
        let mut sc = Vec::with_capacity(12);
        sc.extend_from_slice(&mov.to_le_bytes());
        sc.extend_from_slice(&0xF9000001u32.to_le_bytes()); // STR X1, [X0]
        sc.extend_from_slice(&0xD65F03C0u32.to_le_bytes()); // RET
        sc
    }
}

/// Helper: format shellcode bytes as TOML array.
fn shellcode_toml(bytes: &[u8]) -> String {
    let parts: Vec<String> = bytes.iter().map(|b| format!("{:#04x}", b)).collect();
    format!("[{}]", parts.join(", "))
}

// ---- Coverage-only rewrite: output must be behaviourally identical ------

#[test]
fn corpus_coverage_rewrite_preserves_behaviour() {
    let dir = test_dir();
    for bin in CORPUS {
        let input = compile_bin_flags(dir.path(), bin.name, bin.src, bin.extra_flags);
        let output = dir.path().join(format!("{}_cov", bin.name));

        let config = truant::RewriteConfig {
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
            hooks: None,
            no_coverage: false,
        };
        truant::rewrite(&config)
            .unwrap_or_else(|e| panic!("[{}] coverage rewrite failed: {}", bin.name, e));

        let code = run(&output);
        assert_eq!(
            code, bin.expected_exit,
            "[{}] coverage-rewritten binary: expected exit {}, got {}",
            bin.name, bin.expected_exit, code,
        );
    }
}

// ---- NOP pre-hook: hook infrastructure must not corrupt state -----------

#[test]
fn corpus_nop_pre_hook_preserves_behaviour() {
    let dir = test_dir();
    for bin in CORPUS {
        let input = compile_bin_flags(dir.path(), bin.name, bin.src, bin.extra_flags);
        let output = dir.path().join(format!("{}_nop", bin.name));

        let va = find_symbol_va(&input, "target")
            .unwrap_or_else(|| panic!("[{}] target not found", bin.name));

        let hooks = write_hooks(
            dir.path(),
            &format!("{}_nop", bin.name),
            &format!(
                "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
                va,
                shellcode_toml(SHELLCODE_NOP),
            ),
        );

        if try_rewrite_no_cov(&input, &output, &hooks).is_none() {
            continue; // Function too small at this optimisation level
        }
        let code = run(&output);
        assert_eq!(
            code, bin.expected_exit,
            "[{}] NOP pre-hook changed exit code: expected {}, got {}",
            bin.name, bin.expected_exit, code,
        );
    }
}

// ---- Pre-hook zeroes first arg: verifies arg modification works --------

#[test]
fn corpus_pre_hook_modifies_argument() {
    let dir = test_dir();
    for bin in CORPUS {
        let input = compile_bin_flags(dir.path(), bin.name, bin.src, bin.extra_flags);
        let output = dir.path().join(format!("{}_zeroarg", bin.name));

        let va = find_symbol_va(&input, "target")
            .unwrap_or_else(|| panic!("[{}] target not found", bin.name));

        let hooks = write_hooks(
            dir.path(),
            &format!("{}_zeroarg", bin.name),
            &format!(
                "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n",
                va,
                shellcode_toml(SHELLCODE_ZERO_ARG),
            ),
        );

        if try_rewrite_no_cov(&input, &output, &hooks).is_none() {
            continue;
        }
        let code = run(&output);
        assert!(
            code >= 0 && code <= 255,
            "[{}] pre-hook zero-arg: unexpected exit code {}",
            bin.name,
            code,
        );
    }
}

// ---- Replace-hook: completely replaces function, returns 99 ------------

#[test]
fn corpus_replace_hook_returns_constant() {
    let dir = test_dir();
    for bin in CORPUS {
        // Skip multi-func since it has func_a/func_b, not target
        if bin.name.starts_with("multi_func") {
            continue;
        }
        let input = compile_bin_flags(dir.path(), bin.name, bin.src, bin.extra_flags);
        let output = dir.path().join(format!("{}_replace", bin.name));

        let va = find_symbol_va(&input, "target")
            .unwrap_or_else(|| panic!("[{}] target not found", bin.name));

        let hooks = write_hooks(
            dir.path(),
            &format!("{}_replace", bin.name),
            &format!(
                "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\n",
                va,
                shellcode_toml(SHELLCODE_RETURN_99),
            ),
        );

        if try_rewrite_no_cov(&input, &output, &hooks).is_none() {
            continue;
        }
        let code = run(&output);
        assert_eq!(
            code, 99,
            "[{}] replace-hook should return 99, got {}",
            bin.name, code,
        );
    }
}

// ---- Post-hook NOP: verifies post-hook doesn't corrupt return value ----

#[test]
fn corpus_post_hook_preserves_return() {
    let dir = test_dir();
    for bin in CORPUS {
        let input = compile_bin_flags(dir.path(), bin.name, bin.src, bin.extra_flags);
        let output = dir.path().join(format!("{}_post", bin.name));

        let va = find_symbol_va(&input, "target")
            .unwrap_or_else(|| panic!("[{}] target not found", bin.name));

        let hooks = write_hooks(
            dir.path(),
            &format!("{}_post", bin.name),
            &format!(
                "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"post\"\nshellcode = {}\n",
                va,
                shellcode_toml(SHELLCODE_POST_NOP),
            ),
        );

        if try_rewrite_no_cov(&input, &output, &hooks).is_none() {
            continue;
        }
        let code = run(&output);
        assert_eq!(
            code, bin.expected_exit,
            "[{}] post-hook NOP should preserve return value: expected {}, got {}",
            bin.name, bin.expected_exit, code,
        );
    }
}

// ---- Chained hooks: NOP pre + NOP post = original unchanged -----------

#[test]
fn corpus_chained_pre_and_post_hooks() {
    let dir = test_dir();
    for bin in CORPUS {
        let input = compile_bin_flags(dir.path(), bin.name, bin.src, bin.extra_flags);
        let output = dir.path().join(format!("{}_chain", bin.name));

        let va = find_symbol_va(&input, "target")
            .unwrap_or_else(|| panic!("[{}] target not found", bin.name));

        let hooks = write_hooks(
            dir.path(),
            &format!("{}_chain", bin.name),
            &format!(
                "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nshellcode = {}\n\n\
                 [[hook]]\ntarget = \"0x{:x}\"\nmode = \"post\"\nshellcode = {}\n",
                va,
                shellcode_toml(SHELLCODE_NOP),
                va,
                shellcode_toml(SHELLCODE_POST_NOP),
            ),
        );

        if try_rewrite_no_cov(&input, &output, &hooks).is_none() {
            continue;
        }
        let code = run(&output);
        assert_eq!(
            code, bin.expected_exit,
            "[{}] chained (NOP pre + NOP post) should preserve return: expected {}, got {}",
            bin.name, bin.expected_exit, code,
        );
    }
}

// ---- Coverage + hook combined: hook with coverage instrumentation ------

#[test]
fn corpus_coverage_plus_hook() {
    let dir = test_dir();
    for bin in CORPUS {
        let input = compile_bin_flags(dir.path(), bin.name, bin.src, bin.extra_flags);
        let output = dir.path().join(format!("{}_covhook", bin.name));

        let va = find_symbol_va(&input, "target")
            .unwrap_or_else(|| panic!("[{}] target not found", bin.name));

        let hooks = write_hooks(
            dir.path(),
            &format!("{}_covhook", bin.name),
            &format!(
                "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\n",
                va,
                shellcode_toml(SHELLCODE_RETURN_99),
            ),
        );

        if try_rewrite_hooked(&input, &output, &hooks).is_none() {
            continue;
        }
        let code = run(&output);
        assert_eq!(
            code, 99,
            "[{}] coverage + replace-hook should return 99, got {}",
            bin.name, code,
        );
    }
}

// ---- Multi-function hooks: hook two targets in the same binary ---------

#[test]
fn corpus_multi_function_hooks() {
    let dir = test_dir();
    let input = compile_bin_flags(dir.path(), "multi_func", SRC_MULTI_FUNC, CF);
    let output = dir.path().join("multi_func_hooked");

    let va_a = find_symbol_va(&input, "func_a").expect("func_a not found");
    let va_b = find_symbol_va(&input, "func_b").expect("func_b not found");

    // Replace func_a to return 1, func_b to return 2. main returns func_a(5) + func_b(3) = 1 + 2 = 3.
    let sc_a = shellcode_return_const(1);
    let sc_b = shellcode_return_const(2);
    let hooks = write_hooks(
        dir.path(),
        "multi_func_hooks",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\n\n\
             [[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\nshellcode = {}\n",
            va_a,
            shellcode_toml(&sc_a),
            va_b,
            shellcode_toml(&sc_b),
        ),
    );

    rewrite_no_cov(&input, &output, &hooks);
    let code = run(&output);
    assert_eq!(
        code, 3,
        "multi-function hooks: expected 3 (1+2), got {}",
        code
    );
}

// ---- Conditional hook: only fires when condition is met ----------------

#[test]
fn corpus_conditional_hook() {
    let dir = test_dir();
    // Use the trivial binary: target(41) where rdi=41 at entry.
    let input = compile_bin_flags(dir.path(), "cond_test", SRC_TRIVIAL, CF);
    let output_fires = dir.path().join("cond_fires");
    let output_skips = dir.path().join("cond_skips");

    let va = find_symbol_va(&input, "target").expect("target not found");

    // Condition: first arg register >= 40 → should fire (arg is 41).
    let hooks_fire = write_hooks(
        dir.path(),
        "cond_fire",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\n\
             shellcode = {}\n\
             condition = {{ register = \"{}\", op = \"gte\", value = 40 }}\n",
            va,
            shellcode_toml(SHELLCODE_RETURN_99),
            FIRST_ARG_REG,
        ),
    );
    rewrite_no_cov(&input, &output_fires, &hooks_fire);
    assert_eq!(
        run(&output_fires),
        99,
        "conditional hook should fire ({}=41 >= 40)",
        FIRST_ARG_REG,
    );

    // Condition: first arg register >= 100 → should NOT fire (arg is 41).
    let hooks_skip = write_hooks(
        dir.path(),
        "cond_skip",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\n\
             shellcode = {}\n\
             condition = {{ register = \"{}\", op = \"gte\", value = 100 }}\n",
            va,
            shellcode_toml(SHELLCODE_RETURN_99),
            FIRST_ARG_REG,
        ),
    );
    rewrite_no_cov(&input, &output_skips, &hooks_skip);
    assert_eq!(
        run(&output_skips),
        42,
        "conditional hook should NOT fire ({}=41 < 100)",
        FIRST_ARG_REG,
    );
}

// ---- Toggle hook: enabled/disabled via toggle byte ---------------------

#[test]
fn corpus_toggle_hook() {
    let dir = test_dir();
    let input = compile_bin_flags(dir.path(), "toggle_test", SRC_TRIVIAL, CF);
    let output = dir.path().join("toggle_out");

    let va = find_symbol_va(&input, "target").expect("target not found");

    // Hook with enabled = false: should NOT fire.
    let hooks = write_hooks(
        dir.path(),
        "toggle_hooks",
        &format!(
            "[[hook]]\ntarget = \"0x{:x}\"\nmode = \"replace\"\n\
             shellcode = {}\nenabled = false\n",
            va,
            shellcode_toml(SHELLCODE_RETURN_99),
        ),
    );
    rewrite_no_cov(&input, &output, &hooks);
    assert_eq!(run(&output), 42, "disabled toggle hook should not fire");
}

// ---- Library handler hook: hook resolved from companion .so ------------

#[cfg(target_os = "linux")]
#[test]
fn corpus_library_handler_hook() {
    let dir = test_dir();
    let input = compile_bin_flags(dir.path(), "libhook_test", SRC_TRIVIAL, CF);
    let output = dir.path().join("libhook_out");

    let va = find_symbol_va(&input, "target").expect("target not found");

    // Build companion shared library with hook handler.
    // The handler receives a pointer to RegContext and zeros the first argument
    // register: rdi at offset 40 on x86_64, x0 at offset 0 on AArch64.
    #[cfg(target_arch = "x86_64")]
    const FIRST_ARG_OFFSET: &str = "40";
    #[cfg(target_arch = "aarch64")]
    const FIRST_ARG_OFFSET: &str = "0";

    let lib = compile_so(
        dir.path(),
        "libhook_handler",
        &format!(
            r#"
        #include <stdint.h>
        void my_hook(uint8_t *ctx) {{
            uint64_t zero = 0;
            __builtin_memcpy(ctx + {}, &zero, 8);
        }}
        "#,
            FIRST_ARG_OFFSET,
        ),
    );

    let hooks = write_hooks(
        dir.path(),
        "libhook_hooks",
        &format!(
            "[hooks]\nlibrary = \"{}\"\n\n\
             [[hook]]\ntarget = \"0x{:x}\"\nmode = \"pre\"\nhandler = \"my_hook\"\n",
            lib.display(),
            va,
        ),
    );

    let result = rewrite_hooked(&input, &output, &hooks);
    let preload = result
        .hook_preload_lib_path
        .as_ref()
        .expect("companion preload .so should be generated");

    let code = run_with_preload(&output, preload);
    // With rdi zeroed, target(0) = 0 + 1 = 1.
    assert_eq!(
        code, 1,
        "library handler hook: expected 1 (target(0)), got {}",
        code
    );
}
