/*
 * truant RegContext ABI — stable layout for hook handler shellcode.
 *
 * When a hook fires, the handler receives a pointer to a RegContext struct
 * containing saved register values. The handler can read and modify these
 * values; changes take effect when the hook returns.
 *
 * On x86_64: pointer is in RDI (System V) or RCX (Windows).
 * On AArch64: pointer is in X0.
 * On x86 (PE32): pointer is on top of stack [ESP+4].
 *
 * These layouts are ABI-stable. Do not reorder fields.
 */

#ifndef TRUANT_REGCTX_H
#define TRUANT_REGCTX_H

#include <stdint.h>

/* ================================================================
 * x86_64 RegContext (128 bytes)
 * ================================================================ */

struct truant_regctx_x64 {
    uint64_t rax;   /* offset  0 */
    uint64_t rbx;   /* offset  8 */
    uint64_t rcx;   /* offset 16 */
    uint64_t rdx;   /* offset 24 */
    uint64_t rsi;   /* offset 32 */
    uint64_t rdi;   /* offset 40 */
    uint64_t rbp;   /* offset 48 */
    uint64_t rsp;   /* offset 56 */
    uint64_t r8;    /* offset 64 */
    uint64_t r9;    /* offset 72 */
    uint64_t r10;   /* offset 80 */
    uint64_t r11;   /* offset 88 */
    uint64_t r12;   /* offset 96 */
    uint64_t r13;   /* offset 104 */
    uint64_t r14;   /* offset 112 */
    uint64_t r15;   /* offset 120 */
};

#define TRUANT_X64_RAX   0
#define TRUANT_X64_RBX   8
#define TRUANT_X64_RCX  16
#define TRUANT_X64_RDX  24
#define TRUANT_X64_RSI  32
#define TRUANT_X64_RDI  40
#define TRUANT_X64_RBP  48
#define TRUANT_X64_RSP  56
#define TRUANT_X64_R8   64
#define TRUANT_X64_R9   72
#define TRUANT_X64_R10  80
#define TRUANT_X64_R11  88
#define TRUANT_X64_R12  96
#define TRUANT_X64_R13 104
#define TRUANT_X64_R14 112
#define TRUANT_X64_R15 120
#define TRUANT_X64_SIZE 128

/* ================================================================
 * x86 (PE32) RegContext (40 bytes)
 *
 * Layout matches PUSHAD order + EIP + EFLAGS.
 * ================================================================ */

struct truant_regctx_x86 {
    uint32_t edi;    /* offset  0 */
    uint32_t esi;    /* offset  4 */
    uint32_t ebp;    /* offset  8 */
    uint32_t esp;    /* offset 12 */
    uint32_t ebx;    /* offset 16 */
    uint32_t edx;    /* offset 20 */
    uint32_t ecx;    /* offset 24 */
    uint32_t eax;    /* offset 28 */
    uint32_t eip;    /* offset 32 */
    uint32_t eflags; /* offset 36 */
};

#define TRUANT_X86_EDI     0
#define TRUANT_X86_ESI     4
#define TRUANT_X86_EBP     8
#define TRUANT_X86_ESP    12
#define TRUANT_X86_EBX    16
#define TRUANT_X86_EDX    20
#define TRUANT_X86_ECX    24
#define TRUANT_X86_EAX    28
#define TRUANT_X86_EIP    32
#define TRUANT_X86_EFLAGS 36
#define TRUANT_X86_SIZE   40

/* ================================================================
 * AArch64 RegContext (272 bytes)
 *
 * x0-x30 (31 regs), SP, PC (hooked VA, read-only), NZCV (flags).
 * ================================================================ */

struct truant_regctx_aarch64 {
    uint64_t x[31]; /* x0 at offset 0, x1 at offset 8, ..., x30 at offset 240 */
    uint64_t sp;    /* offset 248 — original stack pointer */
    uint64_t pc;    /* offset 256 — hooked VA (read-only) */
    uint64_t nzcv;  /* offset 264 — condition flags (N, Z, C, V in bits 31-28) */
};

#define TRUANT_A64_X(n) ((n) * 8)
#define TRUANT_A64_SP   248
#define TRUANT_A64_PC   256
#define TRUANT_A64_NZCV 264
#define TRUANT_A64_SIZE 272

#endif /* TRUANT_REGCTX_H */
