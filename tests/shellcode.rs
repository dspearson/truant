//! Arch-portable shellcode constants for hook E2E tests.
//!
//! All hook handlers receive a pointer to RegContext as the first argument
//! (rdi on x86_64, x0 on AArch64). These shellcode snippets operate on
//! RegContext fields at known offsets.
//!
//! x86_64 RegContext (144 bytes):
//!   rax(0) rbx(8) rcx(16) rdx(24) rsi(32) rdi(40)
//!   rbp(48) rsp(56) r8(64)..r15(120) rip(128) rflags(136)
//!
//! AArch64 RegContext (272 bytes):
//!   x0(0) x1(8) x2(16) ... x29(232) x30/lr(240) sp(248) pc(256) nzcv(264)

#![allow(dead_code)]

/// NOP: just return, do nothing.
#[cfg(target_arch = "x86_64")]
pub const NOP: &[u8] = &[0xC3];
#[cfg(target_arch = "aarch64")]
pub const NOP: &[u8] = &[0xC0, 0x03, 0x5F, 0xD6];

/// Pre-hook: set first arg register to 20 in RegContext.
/// x86_64: mov qword [rdi+0x28], 20; ret
/// AArch64: mov x1, #20; str x1, [x0]; ret
#[cfg(target_arch = "x86_64")]
pub const SET_ARG_20: &[u8] = &[0x48, 0xC7, 0x47, 0x28, 0x14, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const SET_ARG_20: &[u8] = &[
    0x81, 0x02, 0x80, 0xD2, // MOV X1, #20
    0x01, 0x00, 0x00, 0xF9, // STR X1, [X0]
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// Pre-hook: set first arg register to 100 in RegContext.
/// x86_64: mov qword [rdi+0x28], 100; ret
/// AArch64: mov x1, #100; str x1, [x0]; ret
#[cfg(target_arch = "x86_64")]
pub const SET_ARG_100: &[u8] = &[0x48, 0xC7, 0x47, 0x28, 0x64, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const SET_ARG_100: &[u8] = &[
    0x81, 0x0C, 0x80, 0xD2, // MOV X1, #100
    0x01, 0x00, 0x00, 0xF9, // STR X1, [X0]
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// Replace-hook: set return value (rax/x0 at offset 0) to 7.
/// x86_64: mov qword [rdi], 7; ret
/// AArch64: mov x1, #7; str x1, [x0]; ret
#[cfg(target_arch = "x86_64")]
pub const RETURN_7: &[u8] = &[0x48, 0xC7, 0x07, 0x07, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const RETURN_7: &[u8] = &[
    0xE1, 0x00, 0x80, 0xD2, // MOV X1, #7
    0x01, 0x00, 0x00, 0xF9, // STR X1, [X0]
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// Replace-hook: set return value to 99.
/// x86_64: mov qword [rdi], 99; ret
/// AArch64: mov x1, #99; str x1, [x0]; ret
#[cfg(target_arch = "x86_64")]
pub const RETURN_99: &[u8] = &[0x48, 0xC7, 0x07, 0x63, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const RETURN_99: &[u8] = &[
    0x61, 0x0C, 0x80, 0xD2, // MOV X1, #99
    0x01, 0x00, 0x00, 0xF9, // STR X1, [X0]
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// Replace-hook: set return value to 100.
#[cfg(target_arch = "x86_64")]
pub const RETURN_100: &[u8] = &[0x48, 0xC7, 0x07, 0x64, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const RETURN_100: &[u8] = &[
    0x81, 0x0C, 0x80, 0xD2, // MOV X1, #100
    0x01, 0x00, 0x00, 0xF9, // STR X1, [X0]
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// Replace-hook: set return value to 0.
/// x86_64: mov qword [rdi], 0; ret
/// AArch64: str xzr, [x0]; ret
#[cfg(target_arch = "x86_64")]
pub const RETURN_0: &[u8] = &[0x48, 0xC7, 0x07, 0x00, 0x00, 0x00, 0x00, 0xC3];
#[cfg(target_arch = "aarch64")]
pub const RETURN_0: &[u8] = &[
    0x1F, 0x00, 0x00, 0xF9, // STR XZR, [X0]
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// Pre-hook: read first arg from RegContext, add 10, write back.
/// x86_64: mov rax,[rdi+0x28]; add rax,10; mov [rdi+0x28],rax; ret
/// AArch64: ldr x1,[x0]; add x1,x1,#10; str x1,[x0]; ret
#[cfg(target_arch = "x86_64")]
pub const ADD_10_TO_ARG: &[u8] = &[
    0x48, 0x8B, 0x47, 0x28, // mov rax, [rdi+0x28]
    0x48, 0x83, 0xC0, 0x0A, // add rax, 10
    0x48, 0x89, 0x47, 0x28, // mov [rdi+0x28], rax
    0xC3, // ret
];
#[cfg(target_arch = "aarch64")]
pub const ADD_10_TO_ARG: &[u8] = &[
    0x01, 0x00, 0x40, 0xF9, // LDR X1, [X0]
    0x21, 0x28, 0x00, 0x91, // ADD X1, X1, #10
    0x01, 0x00, 0x00, 0xF9, // STR X1, [X0]
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// Pre-hook: read first arg from RegContext, add 5, write back.
/// x86_64: mov rax,[rdi+0x28]; add rax,5; mov [rdi+0x28],rax; ret
/// AArch64: ldr x1,[x0]; add x1,x1,#5; str x1,[x0]; ret
#[cfg(target_arch = "x86_64")]
pub const ADD_5_TO_ARG: &[u8] = &[
    0x48, 0x8B, 0x47, 0x28, // mov rax, [rdi+0x28]
    0x48, 0x83, 0xC0, 0x05, // add rax, 5
    0x48, 0x89, 0x47, 0x28, // mov [rdi+0x28], rax
    0xC3, // ret
];
#[cfg(target_arch = "aarch64")]
pub const ADD_5_TO_ARG: &[u8] = &[
    0x01, 0x00, 0x40, 0xF9, // LDR X1, [X0]
    0x21, 0x14, 0x00, 0x91, // ADD X1, X1, #5
    0x01, 0x00, 0x00, 0xF9, // STR X1, [X0]
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// Replace-hook: read first arg from RegContext, triple it, write to return value.
/// x86_64: mov rax,[rdi+0x28]; imul rax,rax,3; mov [rdi],rax; ret
/// AArch64: ldr x1,[x0]; add x2,x1,x1,lsl#1; str x2,[x0]; ret
#[cfg(target_arch = "x86_64")]
pub const TRIPLE_ARG_TO_RETVAL: &[u8] = &[
    0x48, 0x8B, 0x47, 0x28, // mov rax, [rdi+0x28]
    0x48, 0x6B, 0xC0, 0x03, // imul rax, rax, 3
    0x48, 0x89, 0x07, // mov [rdi], rax
    0xC3, // ret
];
#[cfg(target_arch = "aarch64")]
pub const TRIPLE_ARG_TO_RETVAL: &[u8] = &[
    0x01, 0x00, 0x40, 0xF9, // LDR X1, [X0]
    0x22, 0x04, 0x01, 0x8B, // ADD X2, X1, X1, LSL #1  (x2 = x1 * 3)
    0x02, 0x00, 0x00, 0xF9, // STR X2, [X0]
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// Return hook: read rax/x0 from RegContext, add 8, write back.
/// x86_64: mov rax,[rdi]; add rax,8; mov [rdi],rax; ret
/// AArch64: ldr x1,[x0]; add x1,x1,#8; str x1,[x0]; ret
#[cfg(target_arch = "x86_64")]
pub const ADD_8_TO_RETVAL: &[u8] = &[
    0x48, 0x8B, 0x07, // mov rax, [rdi]
    0x48, 0x83, 0xC0, 0x08, // add rax, 8
    0x48, 0x89, 0x07, // mov [rdi], rax
    0xC3, // ret
];
#[cfg(target_arch = "aarch64")]
pub const ADD_8_TO_RETVAL: &[u8] = &[
    0x01, 0x00, 0x40, 0xF9, // LDR X1, [X0]
    0x21, 0x20, 0x00, 0x91, // ADD X1, X1, #8
    0x01, 0x00, 0x00, 0xF9, // STR X1, [X0]
    0xC0, 0x03, 0x5F, 0xD6, // RET
];

/// The register name for the first integer argument in hook conditions.
#[cfg(target_arch = "x86_64")]
pub const FIRST_ARG_REG: &str = "rdi";
#[cfg(target_arch = "aarch64")]
pub const FIRST_ARG_REG: &str = "x0";

/// The RegContext byte offset of the first argument register.
/// x86_64: rdi at offset 40. AArch64: x0 at offset 0.
#[cfg(target_arch = "x86_64")]
pub const FIRST_ARG_OFFSET: usize = 40;
#[cfg(target_arch = "aarch64")]
pub const FIRST_ARG_OFFSET: usize = 0;

/// Format shellcode bytes as a TOML array string.
pub fn shellcode_toml(bytes: &[u8]) -> String {
    let parts: Vec<String> = bytes.iter().map(|b| format!("{:#04x}", b)).collect();
    format!("[{}]", parts.join(", "))
}
