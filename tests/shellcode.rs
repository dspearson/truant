//! Arch-portable shellcode constants for hook E2E tests.
//!
//! All hook handlers receive a pointer to RegContext as the first argument
//! (rdi on x86_64 SysV, rcx on Windows x64, x0 on AArch64). These shellcode
//! snippets operate on RegContext fields at known offsets.
//!
//! x86_64 RegContext (144 bytes):
//!   rax(0) rbx(8) rcx(16) rdx(24) rsi(32) rdi(40)
//!   rbp(48) rsp(56) r8(64)..r15(120) rip(128) rflags(136)
//!
//! On SysV: first arg in rdi (offset 40), ctx pointer in rdi.
//! On Win64: first arg in rcx (offset 16), ctx pointer in rcx.
//!
//! AArch64 RegContext (272 bytes):
//!   x0(0) x1(8) x2(16) ... x29(232) x30/lr(240) sp(248) pc(256) nzcv(264)

#![allow(dead_code)]

/// RegContext byte offset of the first integer argument register.
/// SysV: rdi at offset 40. Win64: rcx at offset 16. AArch64: x0 at offset 0.
#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
pub const FIRST_ARG_OFFSET: u8 = 0x28; // rdi
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
pub const FIRST_ARG_OFFSET: u8 = 0x10; // rcx
#[cfg(target_arch = "aarch64")]
pub const FIRST_ARG_OFFSET: u8 = 0x00; // x0

/// The register name for the first integer argument in hook conditions.
#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
pub const FIRST_ARG_REG: &str = "rdi";
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
pub const FIRST_ARG_REG: &str = "rcx";
#[cfg(target_arch = "aarch64")]
pub const FIRST_ARG_REG: &str = "x0";

// x86_64 ModRM bytes for ctx pointer register.
// SysV: rdi → [rdi]=0x07, [rdi+d8]=0x47, load [rdi+d8]=0x47, store [rdi+d8]=0x47
// Win64: rcx → [rcx]=0x01, [rcx+d8]=0x41, load [rcx+d8]=0x41, store [rcx+d8]=0x41
#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
const P: u8 = 0x07;   // ModRM for [rdi]
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
const P: u8 = 0x01;   // ModRM for [rcx]
#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
const PD: u8 = 0x47;  // ModRM for [rdi+disp8]
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
const PD: u8 = 0x41;  // ModRM for [rcx+disp8]
/// Offset alias for first arg in RegContext.
#[cfg(target_arch = "x86_64")]
const A: u8 = FIRST_ARG_OFFSET;

/// NOP: just return.
#[cfg(target_arch = "x86_64")]
pub const NOP: &[u8] = &[0xC3];
#[cfg(target_arch = "aarch64")]
pub const NOP: &[u8] = &[0xC0, 0x03, 0x5F, 0xD6];

/// Pre-hook: set first arg to 20 in RegContext.
#[cfg(target_arch = "x86_64")]
pub const SET_ARG_20: &[u8] = &[0x48, 0xC7, PD, A, 0x14, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const SET_ARG_20: &[u8] = &[
    0x81, 0x02, 0x80, 0xD2, 0x01, 0x00, 0x00, 0xF9, 0xC0, 0x03, 0x5F, 0xD6,
];

/// Pre-hook: set first arg to 100 in RegContext.
#[cfg(target_arch = "x86_64")]
pub const SET_ARG_100: &[u8] = &[0x48, 0xC7, PD, A, 0x64, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const SET_ARG_100: &[u8] = &[
    0x81, 0x0C, 0x80, 0xD2, 0x01, 0x00, 0x00, 0xF9, 0xC0, 0x03, 0x5F, 0xD6,
];

/// Replace-hook: set return value (rax at offset 0) to 7.
#[cfg(target_arch = "x86_64")]
pub const RETURN_7: &[u8] = &[0x48, 0xC7, P, 0x07, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const RETURN_7: &[u8] = &[
    0xE1, 0x00, 0x80, 0xD2, 0x01, 0x00, 0x00, 0xF9, 0xC0, 0x03, 0x5F, 0xD6,
];

/// Replace-hook: set return value to 99.
#[cfg(target_arch = "x86_64")]
pub const RETURN_99: &[u8] = &[0x48, 0xC7, P, 0x63, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const RETURN_99: &[u8] = &[
    0x61, 0x0C, 0x80, 0xD2, 0x01, 0x00, 0x00, 0xF9, 0xC0, 0x03, 0x5F, 0xD6,
];

/// Replace-hook: set return value to 100.
#[cfg(target_arch = "x86_64")]
pub const RETURN_100: &[u8] = &[0x48, 0xC7, P, 0x64, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const RETURN_100: &[u8] = &[
    0x81, 0x0C, 0x80, 0xD2, 0x01, 0x00, 0x00, 0xF9, 0xC0, 0x03, 0x5F, 0xD6,
];

/// Replace-hook: set return value to 0.
#[cfg(target_arch = "x86_64")]
pub const RETURN_0: &[u8] = &[0x48, 0xC7, P, 0x00, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const RETURN_0: &[u8] = &[
    0x1F, 0x00, 0x00, 0xF9, 0xC0, 0x03, 0x5F, 0xD6,
];

/// Pre-hook: read first arg from RegContext, add 10, write back.
#[cfg(target_arch = "x86_64")]
pub const ADD_10_TO_ARG: &[u8] = &[
    0x48, 0x8B, PD, A,       // mov rax, [ctx+A]
    0x48, 0x83, 0xC0, 0x0A,  // add rax, 10
    0x48, 0x89, PD, A,       // mov [ctx+A], rax
    0xC3,
];
#[cfg(target_arch = "aarch64")]
pub const ADD_10_TO_ARG: &[u8] = &[
    0x01, 0x00, 0x40, 0xF9, 0x21, 0x28, 0x00, 0x91,
    0x01, 0x00, 0x00, 0xF9, 0xC0, 0x03, 0x5F, 0xD6,
];

/// Pre-hook: read first arg from RegContext, add 5, write back.
#[cfg(target_arch = "x86_64")]
pub const ADD_5_TO_ARG: &[u8] = &[
    0x48, 0x8B, PD, A,       // mov rax, [ctx+A]
    0x48, 0x83, 0xC0, 0x05,  // add rax, 5
    0x48, 0x89, PD, A,       // mov [ctx+A], rax
    0xC3,
];
#[cfg(target_arch = "aarch64")]
pub const ADD_5_TO_ARG: &[u8] = &[
    0x01, 0x00, 0x40, 0xF9, 0x21, 0x14, 0x00, 0x91,
    0x01, 0x00, 0x00, 0xF9, 0xC0, 0x03, 0x5F, 0xD6,
];

/// Replace-hook: read first arg from RegContext, triple it, write to return value.
#[cfg(target_arch = "x86_64")]
pub const TRIPLE_ARG_TO_RETVAL: &[u8] = &[
    0x48, 0x8B, PD, A,       // mov rax, [ctx+A]
    0x48, 0x6B, 0xC0, 0x03,  // imul rax, rax, 3
    0x48, 0x89, P,            // mov [ctx], rax
    0xC3,
];
#[cfg(target_arch = "aarch64")]
pub const TRIPLE_ARG_TO_RETVAL: &[u8] = &[
    0x01, 0x00, 0x40, 0xF9, 0x22, 0x04, 0x01, 0x8B,
    0x02, 0x00, 0x00, 0xF9, 0xC0, 0x03, 0x5F, 0xD6,
];

/// Return hook: read rax from RegContext (offset 0), add 8, write back.
#[cfg(target_arch = "x86_64")]
pub const ADD_8_TO_RETVAL: &[u8] = &[
    0x48, 0x8B, P,            // mov rax, [ctx]
    0x48, 0x83, 0xC0, 0x08,   // add rax, 8
    0x48, 0x89, P,             // mov [ctx], rax
    0xC3,
];
#[cfg(target_arch = "aarch64")]
pub const ADD_8_TO_RETVAL: &[u8] = &[
    0x01, 0x00, 0x40, 0xF9, 0x21, 0x20, 0x00, 0x91,
    0x01, 0x00, 0x00, 0xF9, 0xC0, 0x03, 0x5F, 0xD6,
];

/// Format shellcode bytes as a TOML array string.
pub fn shellcode_toml(bytes: &[u8]) -> String {
    let parts: Vec<String> = bytes.iter().map(|b| format!("{:#04x}", b)).collect();
    format!("[{}]", parts.join(", "))
}
