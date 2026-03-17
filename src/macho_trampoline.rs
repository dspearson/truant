//! macOS-specific init code generation for SHM attachment and forkserver.
//!
//! Unlike Linux ELF, macOS uses different syscall conventions:
//! - x86_64: rax = 0x2000000 + bsd_number, CF set on error
//! - ARM64:  x16 = bsd_number, svc #0x80, carry set on error
//!
//! Executables with LC_MAIN: dyld calls main(argc, argv, envp, apple)
//! with envp in rdx (x86_64) or x2 (ARM64) — no stack walking needed.
//!
//! Dylibs: __mod_init_func constructors get no arguments. Use sysctl
//! KERN_PROCARGS2 to read process environment.

use crate::trampoline::{InitCode, PersistentWrapper, PersistentWrapperParams};
use anyhow::{Context, Result};

// macOS BSD syscall numbers (from XNU bsd/kern/syscalls.master)
const MACOS_SYS_EXIT: u32 = 1;
const MACOS_SYS_FORK: u32 = 2;
const MACOS_SYS_READ: u32 = 3;
const MACOS_SYS_WRITE: u32 = 4;
const MACOS_SYS_CLOSE: u32 = 6;
const MACOS_SYS_WAIT4: u32 = 7;
const MACOS_SYS_GETPID: u32 = 20;
const MACOS_SYS_KILL: u32 = 37;
const MACOS_SYS_SYSCTL: u32 = 202;
const MACOS_SYS_SHMAT: u32 = 262;

// macOS x86_64 syscall class prefix
const BSD_CLASS: u32 = 0x0200_0000;

const FORKSRV_FD: u32 = 198;

// sysctl MIB values
const CTL_KERN: u32 = 1;
const KERN_PROCARGS2: u32 = 49;

/// Generate macOS x86_64 init code for executables (LC_MAIN entry).
///
/// With LC_MAIN, dyld calls the entry point as a C function:
///   main(rdi=argc, rsi=argv, rdx=envp, rcx=apple)
///
/// We scan envp for __AFL_SHM_ID, attach SHM, optionally start a forkserver,
/// then jump to the original main.
pub fn generate_macho_exec_init_x86_64(
    init_va: u64,
    data_va: u64,
    original_entry: u64,
    enable_forkserver: bool,
    use_got_shmat: bool,
) -> Result<InitCode> {
    let mut code = Vec::with_capacity(512);
    let entry_va = init_va;

    // Save all argument registers (dyld passes argc/argv/envp/apple)
    code.push(0x57); // push rdi (argc)
    code.push(0x56); // push rsi (argv)
    code.push(0x52); // push rdx (envp)
    code.push(0x51); // push rcx (apple)
    code.push(0x55); // push rbp
    code.push(0x53); // push rbx
    code.extend_from_slice(&[0x41, 0x54]); // push r12
    code.extend_from_slice(&[0x41, 0x55]); // push r13
    code.extend_from_slice(&[0x41, 0x56]); // push r14

    // sub rsp, 16 ; scratch space
    // 9 pushes = 72 bytes, + 16 = 88 offset from original rsp
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x10]);

    // lea r12, [rip + delta_to_data_va]
    let lea_r12_pos = code.len();
    code.extend_from_slice(&[0x4C, 0x8D, 0x25, 0x00, 0x00, 0x00, 0x00]);
    let lea_r12_rip = init_va + code.len() as u64;
    let delta = (data_va as i64 - lea_r12_rip as i64) as i32;
    code[lea_r12_pos + 3..lea_r12_pos + 7].copy_from_slice(&delta.to_le_bytes());

    // envp is in rdx (from dyld). Move to rbx for scanning.
    // mov rbx, rdx
    code.extend_from_slice(&[0x48, 0x89, 0xD3]);

    // Load needle "__AFL_SH" into rbp
    let needle_q = u64::from_le_bytes(*b"__AFL_SH");
    code.extend_from_slice(&[0x48, 0xBD]); // mov rbp, imm64
    code.extend_from_slice(&needle_q.to_le_bytes());

    // .envp_loop:
    let envp_loop = code.len();

    // mov rdi, [rbx]  ; envp[i]
    code.extend_from_slice(&[0x48, 0x8B, 0x3B]);
    // test rdi, rdi
    code.extend_from_slice(&[0x48, 0x85, 0xFF]);
    // jz .skip_shm
    let jz_skip_shm_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);

    // cmp [rdi], rbp  ; "__AFL_SH"?
    code.extend_from_slice(&[0x48, 0x39, 0x2F]);
    // jne .next_env
    let jne_next_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]);

    // cmp dword [rdi+8], "M_ID"
    let needle2_dw = u32::from_le_bytes(*b"M_ID");
    code.extend_from_slice(&[0x81, 0x7F, 0x08]);
    code.extend_from_slice(&needle2_dw.to_le_bytes());
    let jne2_next_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]);

    // cmp byte [rdi+12], '='
    code.extend_from_slice(&[0x80, 0x7F, 0x0C, b'=']);
    let jne3_next_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]);

    // Found! Parse decimal at rdi+13.
    code.extend_from_slice(&[0x48, 0x8D, 0x5F, 0x0D]); // lea rbx, [rdi+13]
    code.extend_from_slice(&[0x4D, 0x31, 0xED]); // xor r13, r13

    let parse_loop_start = code.len();
    code.extend_from_slice(&[0x0F, 0xB6, 0x03]); // movzx eax, byte [rbx]
    code.extend_from_slice(&[0x2C, 0x30]); // sub al, '0'
    code.extend_from_slice(&[0x3C, 0x09]); // cmp al, 9
    let ja_parse_done_pos = code.len();
    code.extend_from_slice(&[0x77, 0x00]); // ja .parse_done
    code.extend_from_slice(&[0x4D, 0x6B, 0xED, 0x0A]); // imul r13, r13, 10
    code.extend_from_slice(&[0x0F, 0xB6, 0xC0]); // movzx eax, al
    code.extend_from_slice(&[0x49, 0x01, 0xC5]); // add r13, rax
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]); // inc rbx
    let jmp_parse_rel = (parse_loop_start as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_parse_rel as u8]);

    let parse_done = code.len();
    code[ja_parse_done_pos + 1] = (parse_done - (ja_parse_done_pos + 2)) as u8;

    // shmat(r13, NULL, 0) — GOT indirect or raw syscall
    let skip_store_jmp_pos;
    if use_got_shmat {
        // Load resolved shmat() address from GOT slot at [r12 + 16]
        // mov rax, [r12 + 16]
        code.extend_from_slice(&[0x49, 0x8B, 0x44, 0x24, 0x10]);
        // mov rdi, r13 (shmid)
        code.extend_from_slice(&[0x4C, 0x89, 0xEF]);
        // xor esi, esi (shmaddr = NULL)
        code.extend_from_slice(&[0x31, 0xF6]);
        // xor edx, edx (shmflg = 0)
        code.extend_from_slice(&[0x31, 0xD2]);
        // call rax
        code.extend_from_slice(&[0xFF, 0xD0]);
        // libc returns (void*)-1 on error
        // cmp rax, -1
        code.extend_from_slice(&[0x48, 0x83, 0xF8, 0xFF]);
        // je .skip_store
        skip_store_jmp_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    } else {
        // Raw BSD syscall: mov eax, 0x2000106  (BSD_CLASS + 262)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_SHMAT).to_le_bytes());
        // mov rdi, r13 (shmid)
        code.extend_from_slice(&[0x4C, 0x89, 0xEF]);
        // xor esi, esi (shmaddr = NULL)
        code.extend_from_slice(&[0x31, 0xF6]);
        // xor edx, edx (shmflg = 0)
        code.extend_from_slice(&[0x31, 0xD2]);
        // macOS BSD syscall convention: 4th arg goes in r10 (not rcx).
        // shmat takes 3 args; zero r10 to avoid EINVAL from garbage in rcx.
        code.extend_from_slice(&[0x4D, 0x31, 0xD2]); // xor r10, r10
        // syscall
        code.extend_from_slice(&[0x0F, 0x05]);
        // macOS: CF set on error. Check with jc (jump if carry)
        skip_store_jmp_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x82, 0x00, 0x00, 0x00, 0x00]); // jc rel32
    }

    // Store SHM pointer: mov [r12], rax
    code.extend_from_slice(&[0x49, 0x89, 0x04, 0x24]);

    // jmp to forkserver/epilogue
    let jmp_to_forkserver_pos = code.len();
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);

    // .next_env:
    let next_env = code.len();
    code[jne_next_pos + 1] = (next_env - (jne_next_pos + 2)) as u8;
    code[jne2_next_pos + 1] = (next_env - (jne2_next_pos + 2)) as u8;
    code[jne3_next_pos + 1] = (next_env - (jne3_next_pos + 2)) as u8;

    // add rbx, 8
    code.extend_from_slice(&[0x48, 0x83, 0xC3, 0x08]);
    // jmp .envp_loop
    let jmp_envp_rel = (envp_loop as i32) - (code.len() as i32 + 5);
    code.push(0xE9);
    code.extend_from_slice(&jmp_envp_rel.to_le_bytes());

    // .skip_shm:
    let skip_shm = code.len();
    let skip_shm_offset = |from: usize| -> i32 { (skip_shm as i32) - (from as i32 + 4) };
    let rel = skip_shm_offset(jz_skip_shm_pos + 2);
    code[jz_skip_shm_pos + 2..jz_skip_shm_pos + 6].copy_from_slice(&rel.to_le_bytes());
    let rel = skip_shm_offset(skip_store_jmp_pos + 2);
    code[skip_store_jmp_pos + 2..skip_store_jmp_pos + 6].copy_from_slice(&rel.to_le_bytes());

    if enable_forkserver {
        let forkserver_start = code.len();
        let rel = (forkserver_start as i32) - (jmp_to_forkserver_pos as i32 + 5);
        code[jmp_to_forkserver_pos + 1..jmp_to_forkserver_pos + 5]
            .copy_from_slice(&rel.to_le_bytes());

        // hello message
        code.extend_from_slice(&[0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00]); // mov dword [rsp], 0

        // SYS_write(FORKSRV_FD+1, &hello, 4) — macOS
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WRITE).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes()); // edi = 199
        code.extend_from_slice(&[0x48, 0x89, 0xE6]); // mov rsi, rsp
        code.extend_from_slice(&[0xBA, 0x04, 0x00, 0x00, 0x00]); // mov edx, 4
        code.extend_from_slice(&[0x0F, 0x05]); // syscall
        // macOS sets CF on syscall error (EBADF if fd 199 doesn't exist).
        // If hello write fails, skip the forkserver and run the target directly.
        let jc_skip_forkserver_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x82, 0x00, 0x00, 0x00, 0x00]); // jc .skip_forkserver

        // .fork_loop:
        let fork_loop = code.len();

        // SYS_read(FORKSRV_FD, &cmd, 4)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_READ).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&FORKSRV_FD.to_le_bytes());
        code.extend_from_slice(&[0x48, 0x89, 0xE6]); // mov rsi, rsp
        code.extend_from_slice(&[0xBA, 0x04, 0x00, 0x00, 0x00]); // mov edx, 4
        code.extend_from_slice(&[0x0F, 0x05]); // syscall

        // SYS_fork — macOS
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_FORK).to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]); // syscall
        // macOS fork: parent gets child pid in rax, child gets 0 in rax
        // BUT also: rdx=0 in parent, rdx=1 in child

        // test edx, edx  ; child?
        code.extend_from_slice(&[0x85, 0xD2]);
        // jnz .child (child: rdx=1)
        let jnz_child_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);

        // Parent: store child pid
        code.extend_from_slice(&[0x89, 0x04, 0x24]); // mov [rsp], eax (child pid)

        // SYS_write(FORKSRV_FD+1, &child_pid, 4)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WRITE).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
        code.extend_from_slice(&[0x48, 0x89, 0xE6]); // rsi = rsp
        code.extend_from_slice(&[0xBA, 0x04, 0x00, 0x00, 0x00]); // edx = 4
        code.extend_from_slice(&[0x0F, 0x05]);

        // SYS_wait4(child_pid, &status, 0, NULL)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WAIT4).to_le_bytes());
        code.extend_from_slice(&[0x8B, 0x3C, 0x24]); // mov edi, [rsp] (child pid)
        // lea rsi, [rsp+4] ; &status
        code.extend_from_slice(&[0x48, 0x8D, 0x74, 0x24, 0x04]);
        code.extend_from_slice(&[0x31, 0xD2]); // xor edx, edx (options=0)
        code.extend_from_slice(&[0x4D, 0x31, 0xC0]); // xor r8, r8 (rusage=NULL)
        // macOS x86_64 syscall convention: args in rdi, rsi, rdx, r10, r8, r9.
        // wait4(pid, status, options, rusage): 4th arg (rusage) goes in r10.
        code.extend_from_slice(&[0x4D, 0x31, 0xD2]); // xor r10, r10 (rusage=NULL)
        code.extend_from_slice(&[0x0F, 0x05]); // syscall

        // SYS_write(FORKSRV_FD+1, &status, 4)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WRITE).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x74, 0x24, 0x04]); // lea rsi, [rsp+4]
        code.extend_from_slice(&[0xBA, 0x04, 0x00, 0x00, 0x00]);
        code.extend_from_slice(&[0x0F, 0x05]);

        // jmp .fork_loop
        let jmp_fork_rel = (fork_loop as i32) - (code.len() as i32 + 5);
        code.push(0xE9);
        code.extend_from_slice(&jmp_fork_rel.to_le_bytes());

        // .child:
        let child = code.len();
        let rel = (child as i32) - (jnz_child_pos as i32 + 6);
        code[jnz_child_pos + 2..jnz_child_pos + 6].copy_from_slice(&rel.to_le_bytes());

        // Close forkserver FDs
        // SYS_close(FORKSRV_FD)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_CLOSE).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&FORKSRV_FD.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);
        // SYS_close(FORKSRV_FD+1)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_CLOSE).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // .skip_forkserver: patch jc target → falls through to epilogue
        let skip_forkserver = code.len();
        let rel = (skip_forkserver as i32) - (jc_skip_forkserver_pos as i32 + 6);
        code[jc_skip_forkserver_pos + 2..jc_skip_forkserver_pos + 6]
            .copy_from_slice(&rel.to_le_bytes());
    } else {
        // No forkserver: fix jmp_to_forkserver → epilogue
        let epilogue = code.len();
        let rel = (epilogue as i32) - (jmp_to_forkserver_pos as i32 + 5);
        code[jmp_to_forkserver_pos + 1..jmp_to_forkserver_pos + 5]
            .copy_from_slice(&rel.to_le_bytes());
    }

    // Epilogue: restore registers and jump to original main
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x10]); // add rsp, 16
    code.extend_from_slice(&[0x41, 0x5E]); // pop r14
    code.extend_from_slice(&[0x41, 0x5D]); // pop r13
    code.extend_from_slice(&[0x41, 0x5C]); // pop r12
    code.push(0x5B); // pop rbx
    code.push(0x5D); // pop rbp
    code.push(0x59); // pop rcx
    code.push(0x5A); // pop rdx
    code.push(0x5E); // pop rsi
    code.push(0x5F); // pop rdi

    // jmp original_entry
    let jmp_pos = code.len();
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
    let jmp_rip = init_va + code.len() as u64;
    let rel = (original_entry as i64 - jmp_rip as i64) as i32;
    code[jmp_pos + 1..jmp_pos + 5].copy_from_slice(&rel.to_le_bytes());

    Ok(InitCode {
        va: init_va,
        code,
        entry_va,
    })
}

/// Generate macOS x86_64 init code for dylibs (sysctl KERN_PROCARGS2).
///
/// Dylib constructors (__mod_init_func) receive no arguments. We use sysctl
/// to read the process environment from the kernel.
pub fn generate_macho_dylib_init_x86_64(
    init_va: u64,
    data_va: u64,
    chain_to: Option<u64>,
    use_got_shmat: bool,
) -> Result<InitCode> {
    let mut code = Vec::with_capacity(1024);
    let entry_va = init_va;

    // Save callee-saved registers
    code.push(0x55); // push rbp
    code.push(0x53); // push rbx
    code.extend_from_slice(&[0x41, 0x54]); // push r12
    code.extend_from_slice(&[0x41, 0x55]); // push r13
    code.extend_from_slice(&[0x41, 0x56]); // push r14

    // Allocate stack: 8192 for sysctl buffer + 24 for MIB + 16 for len + scratch
    // Total: 8248, round up to 8256 (divisible by 16)
    let stack_alloc: u32 = 8256;
    let buf_offset: i32 = 64; // sysctl buffer at [rsp+64]
    let mib_offset: i32 = 16; // MIB array at [rsp+16] (3 x u32 = 12 bytes)
    let len_offset: i32 = 32; // oldlenp at [rsp+32] (8 bytes)

    // sub rsp, stack_alloc
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&stack_alloc.to_le_bytes());

    // lea r12, [rip + delta_to_data_va]
    let lea_r12_pos = code.len();
    code.extend_from_slice(&[0x4C, 0x8D, 0x25, 0x00, 0x00, 0x00, 0x00]);
    let lea_r12_rip = init_va + code.len() as u64;
    let delta = (data_va as i64 - lea_r12_rip as i64) as i32;
    code[lea_r12_pos + 3..lea_r12_pos + 7].copy_from_slice(&delta.to_le_bytes());

    // Step 1: getpid
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_GETPID).to_le_bytes());
    code.extend_from_slice(&[0x0F, 0x05]);
    // mov r13d, eax  (pid)
    code.extend_from_slice(&[0x41, 0x89, 0xC5]);

    // Step 2: set up MIB [CTL_KERN, KERN_PROCARGS2, pid]
    // mov dword [rsp+mib_offset], CTL_KERN
    code.extend_from_slice(&[0xC7, 0x44, 0x24, mib_offset as u8]);
    code.extend_from_slice(&CTL_KERN.to_le_bytes());
    // mov dword [rsp+mib_offset+4], KERN_PROCARGS2
    code.extend_from_slice(&[0xC7, 0x44, 0x24, (mib_offset + 4) as u8]);
    code.extend_from_slice(&KERN_PROCARGS2.to_le_bytes());
    // mov dword [rsp+mib_offset+8], r13d (pid)
    code.extend_from_slice(&[0x44, 0x89, 0x6C, 0x24, (mib_offset + 8) as u8]);

    // Step 3: set oldlenp
    let buf_size: u64 = 8192;
    // mov qword [rsp+len_offset], buf_size
    code.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, len_offset as u8]);
    code.extend_from_slice(&(buf_size as u32).to_le_bytes());

    // Step 4: sysctl(mib, 3, buf, &len, NULL, 0) — macOS
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_SYSCTL).to_le_bytes());
    // rdi = &mib
    code.extend_from_slice(&[0x48, 0x8D, 0x7C, 0x24, mib_offset as u8]);
    // esi = 3 (mib length)
    code.extend_from_slice(&[0xBE, 0x03, 0x00, 0x00, 0x00]);
    // rdx = buf (old)
    code.extend_from_slice(&[0x48, 0x8D, 0x94, 0x24]);
    code.extend_from_slice(&(buf_offset as u32).to_le_bytes());
    // r10 = &len (oldlenp) — 4th arg via r10 on macOS x86_64 syscalls
    code.extend_from_slice(&[0x4C, 0x8D, 0x54, 0x24, len_offset as u8]);
    // r8 = NULL (new)
    code.extend_from_slice(&[0x4D, 0x31, 0xC0]);
    // r9 = 0 (newlen)
    code.extend_from_slice(&[0x4D, 0x31, 0xC9]);
    // syscall
    code.extend_from_slice(&[0x0F, 0x05]);

    // Check error (CF set by sysctl)
    let jc_done_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x82, 0x00, 0x00, 0x00, 0x00]); // jc .done

    // Step 5: parse KERN_PROCARGS2 buffer
    // Buffer format: [argc: i32] [exec_path: null-terminated] [NUL padding]
    //                [argv[0]...argv[argc-1]: null-terminated each]
    //                [env[0]...env[N]: null-terminated each]
    //
    // Strategy: skip argc(4) + exec_path + NULs + argc argv strings, then scan env strings.

    // lea rbx, [rsp+buf_offset]   ; buffer start
    code.extend_from_slice(&[0x48, 0x8D, 0x9C, 0x24]);
    code.extend_from_slice(&(buf_offset as u32).to_le_bytes());

    // mov r14, [rsp+len_offset]   ; actual bytes returned
    code.extend_from_slice(&[0x4C, 0x8B, 0x74, 0x24, len_offset as u8]);

    // Compute buf_end = rbx + actual_len
    code.extend_from_slice(&[0x49, 0x89, 0xDD]); // mov r13, rbx
    // mov rax, [rsp+len_offset]
    code.extend_from_slice(&[0x48, 0x8B, 0x44, 0x24, len_offset as u8]);
    // lea r14, [rbx + rax]  ; buf_end
    code.extend_from_slice(&[0x4C, 0x8D, 0x34, 0x03]); // lea r14, [rbx + rax]

    // Skip argc (4 bytes)
    // add rbx, 4
    code.extend_from_slice(&[0x48, 0x83, 0xC3, 0x04]);

    // Skip exec_path: scan for NUL.
    // Pattern: load byte into eax, advance rbx, test byte; loop while non-zero.
    // When al==0 (NUL found), falls through with rbx one past the NUL.
    // NOTE: must NOT use "cmp [rbx],0; inc rbx; jne" — INC clobbers ZF set by CMP.
    //       Use "movzx eax,[rbx]; inc rbx; test al,al; jnz" instead.
    // .skip_exec:
    let skip_exec_loop = code.len();
    code.extend_from_slice(&[0x49, 0x39, 0xDE]); // cmp r14, rbx  (bounds check)
    let jbe_done_pos1 = code.len();
    code.extend_from_slice(&[0x0F, 0x86, 0x00, 0x00, 0x00, 0x00]); // jbe .done
    code.extend_from_slice(&[0x0F, 0xB6, 0x03]); // movzx eax, byte [rbx]  (load byte)
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]); // inc rbx  (advance)
    code.extend_from_slice(&[0x84, 0xC0]); // test al, al  (test loaded byte)
    // jnz .skip_exec  (loop while byte != 0)
    let jnz_rel = (skip_exec_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0x75, jnz_rel as u8]);
    // Falls through when al==0: NUL found, rbx is already one past the NUL.

    // Skip NUL padding: loop while byte == 0.
    // After loop, al != 0, rbx is one past the first non-NUL byte — back up with dec.
    // .skip_nuls:
    let skip_nuls_loop = code.len();
    code.extend_from_slice(&[0x49, 0x39, 0xDE]); // cmp r14, rbx  (bounds check)
    let jbe_done_pos2 = code.len();
    code.extend_from_slice(&[0x0F, 0x86, 0x00, 0x00, 0x00, 0x00]); // jbe .done
    code.extend_from_slice(&[0x0F, 0xB6, 0x03]); // movzx eax, byte [rbx]  (load byte)
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]); // inc rbx  (advance)
    code.extend_from_slice(&[0x84, 0xC0]); // test al, al  (test loaded byte)
    let jz_rel = (skip_nuls_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0x74, jz_rel as u8]); // jz .skip_nuls (keep skipping NULs)
    // After loop: rbx is one past first non-NUL byte. Back up to point at it.
    code.extend_from_slice(&[0x48, 0xFF, 0xCB]); // dec rbx

    // Skip argc argv strings: read original argc from buf, skip that many strings
    // Original argc was at [rsp+buf_offset] (first 4 bytes)
    // mov ecx, [rsp+buf_offset]
    code.extend_from_slice(&[0x8B, 0x8C, 0x24]);
    code.extend_from_slice(&(buf_offset as u32).to_le_bytes());

    // .skip_argv_loop:
    let skip_argv_loop = code.len();
    // test ecx, ecx
    code.extend_from_slice(&[0x85, 0xC9]);
    // jz .scan_env
    let jz_scan_env_pos = code.len();
    code.extend_from_slice(&[0x74, 0x00]);

    // Skip one string: scan for NUL. Same safe pattern: movzx/inc/test/jnz.
    // Falls through with rbx one past the NUL terminator.
    // .skip_one_arg:
    let skip_one_arg = code.len();
    code.extend_from_slice(&[0x49, 0x39, 0xDE]); // cmp r14, rbx  (bounds check)
    let jbe_done_pos3 = code.len();
    code.extend_from_slice(&[0x0F, 0x86, 0x00, 0x00, 0x00, 0x00]); // jbe .done
    code.extend_from_slice(&[0x0F, 0xB6, 0x03]); // movzx eax, byte [rbx]  (load byte)
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]); // inc rbx  (advance)
    code.extend_from_slice(&[0x84, 0xC0]); // test al, al  (test loaded byte)
    let jne_rel2 = (skip_one_arg as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0x75, jne_rel2 as u8]); // jnz .skip_one_arg (loop while non-zero)
    // dec ecx
    code.extend_from_slice(&[0xFF, 0xC9]);
    // jmp .skip_argv_loop
    let jmp_skip_argv_rel = (skip_argv_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_skip_argv_rel as u8]);

    // .scan_env:
    let scan_env = code.len();
    code[jz_scan_env_pos + 1] = (scan_env - (jz_scan_env_pos + 2)) as u8;

    // Now rbx points into env strings. Scan for "__AFL_SHM_ID="
    let needle_q2 = u64::from_le_bytes(*b"__AFL_SH");
    let needle2_dw = u32::from_le_bytes(*b"M_ID");
    code.extend_from_slice(&[0x48, 0xBD]); // mov rbp, imm64
    code.extend_from_slice(&needle_q2.to_le_bytes());

    // .env_scan_loop:
    let env_scan_loop = code.len();
    // Skip leading NULs between env strings
    // Skip leading NULs: loop while byte == 0. Safe movzx/inc/test/jz pattern.
    // After loop: rbx is one past first non-NUL byte — back up with dec.
    // .skip_env_nuls:
    let skip_env_nuls = code.len();
    code.extend_from_slice(&[0x49, 0x39, 0xDE]); // cmp r14, rbx  (bounds check)
    let jbe_done_pos4 = code.len();
    code.extend_from_slice(&[0x0F, 0x86, 0x00, 0x00, 0x00, 0x00]); // jbe .done
    code.extend_from_slice(&[0x0F, 0xB6, 0x03]); // movzx eax, byte [rbx]  (load byte)
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]); // inc rbx  (advance)
    code.extend_from_slice(&[0x84, 0xC0]); // test al, al  (test loaded byte)
    let je_rel2 = (skip_env_nuls as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0x74, je_rel2 as u8]); // jz .skip_env_nuls (loop while zero)
    // Back up to start of string (rbx is one past first non-NUL byte)
    code.extend_from_slice(&[0x48, 0xFF, 0xCB]); // dec rbx

    // Need at least 13 bytes remaining for "__AFL_SHM_ID="
    // lea rax, [rbx + 13]
    code.extend_from_slice(&[0x48, 0x8D, 0x43, 0x0D]);
    // cmp rax, r14
    code.extend_from_slice(&[0x49, 0x39, 0xC6]); // cmp r14, rax
    let jb_next_env_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x82, 0x00, 0x00, 0x00, 0x00]); // jb .done (not enough room)

    // cmp [rbx], rbp  ("__AFL_SH")
    code.extend_from_slice(&[0x48, 0x39, 0x2B]);
    let jne_skip_this_env_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne .skip_this_env

    // cmp dword [rbx+8], "M_ID"
    code.extend_from_slice(&[0x81, 0x7B, 0x08]);
    code.extend_from_slice(&needle2_dw.to_le_bytes());
    let jne_skip_this_env_pos2 = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne .skip_this_env

    // cmp byte [rbx+12], '='
    code.extend_from_slice(&[0x80, 0x7B, 0x0C, b'=']);
    let jne_skip_this_env_pos3 = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne .skip_this_env

    // Found! Parse decimal at rbx+13
    code.extend_from_slice(&[0x48, 0x8D, 0x5B, 0x0D]); // lea rbx, [rbx+13]
    code.extend_from_slice(&[0x4D, 0x31, 0xED]); // xor r13, r13

    let parse2_loop_start = code.len();
    code.extend_from_slice(&[0x0F, 0xB6, 0x03]); // movzx eax, byte [rbx]
    code.extend_from_slice(&[0x2C, 0x30]); // sub al, '0'
    code.extend_from_slice(&[0x3C, 0x09]); // cmp al, 9
    let ja2_parse_done_pos = code.len();
    code.extend_from_slice(&[0x77, 0x00]); // ja .parse2_done
    code.extend_from_slice(&[0x4D, 0x6B, 0xED, 0x0A]); // imul r13, r13, 10
    code.extend_from_slice(&[0x0F, 0xB6, 0xC0]); // movzx eax, al
    code.extend_from_slice(&[0x49, 0x01, 0xC5]); // add r13, rax
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]); // inc rbx
    let jmp_parse2_rel = (parse2_loop_start as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_parse2_rel as u8]);

    let parse2_done = code.len();
    code[ja2_parse_done_pos + 1] = (parse2_done - (ja2_parse_done_pos + 2)) as u8;

    // shmat(r13, NULL, 0) — GOT indirect or raw syscall
    let jc_done_pos2;
    if use_got_shmat {
        // Load resolved shmat() address from GOT slot at [r12 + 16]
        code.extend_from_slice(&[0x49, 0x8B, 0x44, 0x24, 0x10]); // mov rax, [r12 + 16]
        code.extend_from_slice(&[0x4C, 0x89, 0xEF]); // mov rdi, r13
        code.extend_from_slice(&[0x31, 0xF6]); // xor esi, esi
        code.extend_from_slice(&[0x31, 0xD2]); // xor edx, edx
        code.extend_from_slice(&[0xFF, 0xD0]); // call rax
        code.extend_from_slice(&[0x48, 0x83, 0xF8, 0xFF]); // cmp rax, -1
        jc_done_pos2 = code.len();
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // je .done
    } else {
        code.extend_from_slice(&[0xB8]); // mov eax, imm32
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_SHMAT).to_le_bytes());
        code.extend_from_slice(&[0x4C, 0x89, 0xEF]); // mov rdi, r13
        code.extend_from_slice(&[0x31, 0xF6]); // xor esi, esi
        code.extend_from_slice(&[0x31, 0xD2]); // xor edx, edx
        // macOS BSD syscall convention: 4th arg goes in r10 (not rcx).
        // shmat takes 3 args; zero r10 to avoid EINVAL from garbage in rcx.
        code.extend_from_slice(&[0x4D, 0x31, 0xD2]); // xor r10, r10
        code.extend_from_slice(&[0x0F, 0x05]); // syscall
        jc_done_pos2 = code.len();
        code.extend_from_slice(&[0x0F, 0x82, 0x00, 0x00, 0x00, 0x00]); // jc .done
    }

    // Store SHM pointer
    code.extend_from_slice(&[0x49, 0x89, 0x04, 0x24]); // mov [r12], rax

    // jmp .done
    let jmp_done_pos = code.len();
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);

    // .skip_this_env: advance rbx past current string, then loop
    let skip_this_env = code.len();
    code[jne_skip_this_env_pos + 1] = (skip_this_env - (jne_skip_this_env_pos + 2)) as u8;
    code[jne_skip_this_env_pos2 + 1] = (skip_this_env - (jne_skip_this_env_pos2 + 2)) as u8;
    code[jne_skip_this_env_pos3 + 1] = (skip_this_env - (jne_skip_this_env_pos3 + 2)) as u8;

    // Skip to end of current string: scan for NUL. Safe movzx/inc/test/jnz pattern.
    // Falls through with rbx one past the NUL (ready for next env string scan).
    // .skip_str:
    let skip_str = code.len();
    code.extend_from_slice(&[0x49, 0x39, 0xDE]); // cmp r14, rbx  (bounds check)
    let jbe_done_pos5 = code.len();
    code.extend_from_slice(&[0x0F, 0x86, 0x00, 0x00, 0x00, 0x00]); // jbe .done
    code.extend_from_slice(&[0x0F, 0xB6, 0x03]); // movzx eax, byte [rbx]  (load byte)
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]); // inc rbx  (advance)
    code.extend_from_slice(&[0x84, 0xC0]); // test al, al  (test loaded byte)
    let jne_rel3 = (skip_str as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0x75, jne_rel3 as u8]); // jnz .skip_str (loop while non-zero)

    // jmp .env_scan_loop
    let jmp_env_scan_rel = (env_scan_loop as isize - (code.len() as isize + 5)) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp_env_scan_rel.to_le_bytes());

    // .done:
    let done = code.len();
    // Fix up all .done jump placeholders.
    // These are all 6-byte near conditionals: 0F 8x xx xx xx xx
    // RIP after decoding = pos + 6, so displacement = done - (pos + 6).
    let fix_done = |code: &mut Vec<u8>, pos: usize| {
        let rel = (done as i32) - (pos as i32 + 6);
        code[pos + 2..pos + 6].copy_from_slice(&rel.to_le_bytes());
    };
    fix_done(&mut code, jc_done_pos);
    fix_done(&mut code, jbe_done_pos1);
    fix_done(&mut code, jbe_done_pos2);
    fix_done(&mut code, jbe_done_pos3);
    fix_done(&mut code, jbe_done_pos4);
    fix_done(&mut code, jbe_done_pos5);
    fix_done(&mut code, jb_next_env_pos);
    let rel = (done as i32) - (jc_done_pos2 as i32 + 6);
    code[jc_done_pos2 + 2..jc_done_pos2 + 6].copy_from_slice(&rel.to_le_bytes());
    let rel = (done as i32) - (jmp_done_pos as i32 + 5);
    code[jmp_done_pos + 1..jmp_done_pos + 5].copy_from_slice(&rel.to_le_bytes());

    // Epilogue: deallocate stack, restore registers
    code.extend_from_slice(&[0x48, 0x81, 0xC4]); // add rsp, stack_alloc
    code.extend_from_slice(&stack_alloc.to_le_bytes());
    code.extend_from_slice(&[0x41, 0x5E]); // pop r14
    code.extend_from_slice(&[0x41, 0x5D]); // pop r13
    code.extend_from_slice(&[0x41, 0x5C]); // pop r12
    code.push(0x5B); // pop rbx
    code.push(0x5D); // pop rbp

    // Chain to original init function if present
    if let Some(orig_init) = chain_to {
        let jmp_pos = code.len();
        code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
        let jmp_rip = init_va + code.len() as u64;
        let rel = (orig_init as i64 - jmp_rip as i64) as i32;
        code[jmp_pos + 1..jmp_pos + 5].copy_from_slice(&rel.to_le_bytes());
    } else {
        code.push(0xC3); // ret
    }

    Ok(InitCode {
        va: init_va,
        code,
        entry_va,
    })
}

/// Generate macOS x86_64 init code for LC_UNIXTHREAD executables.
///
/// LC_UNIXTHREAD is used by older macOS binaries and kernel extensions.
/// The entry receives a raw stack (like ELF): [argc] [argv...] [NULL] [envp...] [NULL]
/// This is identical to the ELF convention, so we walk the stack.
pub fn generate_macho_unixthread_init_x86_64(
    init_va: u64,
    data_va: u64,
    original_entry: u64,
    enable_forkserver: bool,
    use_got_shmat: bool,
) -> Result<InitCode> {
    // The stack layout is identical to ELF entry, except syscall numbers differ.
    // Reuse the envp-walking approach from generate_macho_exec_init_x86_64
    // but walk the stack like the ELF init code.
    let mut code = Vec::with_capacity(512);
    let entry_va = init_va;

    // Save registers
    code.push(0x55); // push rbp
    code.push(0x53); // push rbx
    code.extend_from_slice(&[0x41, 0x54]); // push r12
    code.extend_from_slice(&[0x41, 0x55]); // push r13
    code.extend_from_slice(&[0x41, 0x56]); // push r14

    // sub rsp, 16
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x10]);
    const STACK_OFFSET: u8 = 56; // 5 pushes (40) + sub 16

    // lea r12, [rip + delta_to_data_va]
    let lea_r12_pos = code.len();
    code.extend_from_slice(&[0x4C, 0x8D, 0x25, 0x00, 0x00, 0x00, 0x00]);
    let lea_r12_rip = init_va + code.len() as u64;
    let delta = (data_va as i64 - lea_r12_rip as i64) as i32;
    code[lea_r12_pos + 3..lea_r12_pos + 7].copy_from_slice(&delta.to_le_bytes());

    // Walk stack: [rsp + STACK_OFFSET] = argc, then argv[], then envp[]
    // mov rax, [rsp + STACK_OFFSET]
    code.extend_from_slice(&[0x48, 0x8B, 0x44, 0x24, STACK_OFFSET]);
    // lea rbx, [rsp + STACK_OFFSET + 8]
    code.extend_from_slice(&[0x48, 0x8D, 0x5C, 0x24, STACK_OFFSET + 8]);
    // lea rbx, [rbx + rax*8 + 8]  ; skip argv + NULL → envp
    code.extend_from_slice(&[0x48, 0x8D, 0x5C, 0xC3, 0x08]);

    // Load needle
    let needle_q = u64::from_le_bytes(*b"__AFL_SH");
    code.extend_from_slice(&[0x48, 0xBD]);
    code.extend_from_slice(&needle_q.to_le_bytes());

    // .envp_loop:
    let envp_loop = code.len();
    code.extend_from_slice(&[0x48, 0x8B, 0x3B]); // mov rdi, [rbx]
    code.extend_from_slice(&[0x48, 0x85, 0xFF]); // test rdi, rdi
    let jz_skip_shm_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);

    code.extend_from_slice(&[0x48, 0x39, 0x2F]); // cmp [rdi], rbp
    let jne_next_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]);

    let needle2_dw = u32::from_le_bytes(*b"M_ID");
    code.extend_from_slice(&[0x81, 0x7F, 0x08]);
    code.extend_from_slice(&needle2_dw.to_le_bytes());
    let jne2_next_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]);

    code.extend_from_slice(&[0x80, 0x7F, 0x0C, b'=']);
    let jne3_next_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]);

    // Parse decimal
    code.extend_from_slice(&[0x48, 0x8D, 0x5F, 0x0D]);
    code.extend_from_slice(&[0x4D, 0x31, 0xED]);

    let parse_loop = code.len();
    code.extend_from_slice(&[0x0F, 0xB6, 0x03]);
    code.extend_from_slice(&[0x2C, 0x30]);
    code.extend_from_slice(&[0x3C, 0x09]);
    let ja_done_pos = code.len();
    code.extend_from_slice(&[0x77, 0x00]);
    code.extend_from_slice(&[0x4D, 0x6B, 0xED, 0x0A]);
    code.extend_from_slice(&[0x0F, 0xB6, 0xC0]);
    code.extend_from_slice(&[0x49, 0x01, 0xC5]);
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]);
    let jmp_parse_rel = (parse_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_parse_rel as u8]);

    let parse_done = code.len();
    code[ja_done_pos + 1] = (parse_done - (ja_done_pos + 2)) as u8;

    // shmat(r13, NULL, 0) — GOT indirect or raw syscall
    let jc_skip_pos;
    if use_got_shmat {
        code.extend_from_slice(&[0x49, 0x8B, 0x44, 0x24, 0x10]); // mov rax, [r12 + 16]
        code.extend_from_slice(&[0x4C, 0x89, 0xEF]); // mov rdi, r13
        code.extend_from_slice(&[0x31, 0xF6]); // xor esi, esi
        code.extend_from_slice(&[0x31, 0xD2]); // xor edx, edx
        code.extend_from_slice(&[0xFF, 0xD0]); // call rax
        code.extend_from_slice(&[0x48, 0x83, 0xF8, 0xFF]); // cmp rax, -1
        jc_skip_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // je .skip_shm
    } else {
        code.extend_from_slice(&[0xB8]); // mov eax, imm32
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_SHMAT).to_le_bytes());
        code.extend_from_slice(&[0x4C, 0x89, 0xEF]); // mov rdi, r13
        code.extend_from_slice(&[0x31, 0xF6]); // xor esi, esi
        code.extend_from_slice(&[0x31, 0xD2]); // xor edx, edx
        // macOS BSD syscall convention: 4th arg goes in r10 (not rcx).
        // shmat takes 3 args; zero r10 to avoid EINVAL from garbage in rcx.
        code.extend_from_slice(&[0x4D, 0x31, 0xD2]); // xor r10, r10
        code.extend_from_slice(&[0x0F, 0x05]); // syscall
        jc_skip_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x82, 0x00, 0x00, 0x00, 0x00]); // jc .skip_shm
    }

    code.extend_from_slice(&[0x49, 0x89, 0x04, 0x24]); // mov [r12], rax

    let jmp_forkserver_pos = code.len();
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);

    // .next_env:
    let next_env = code.len();
    code[jne_next_pos + 1] = (next_env - (jne_next_pos + 2)) as u8;
    code[jne2_next_pos + 1] = (next_env - (jne2_next_pos + 2)) as u8;
    code[jne3_next_pos + 1] = (next_env - (jne3_next_pos + 2)) as u8;

    code.extend_from_slice(&[0x48, 0x83, 0xC3, 0x08]);
    let jmp_envp_rel = (envp_loop as i32) - (code.len() as i32 + 5);
    code.push(0xE9);
    code.extend_from_slice(&jmp_envp_rel.to_le_bytes());

    // .skip_shm:
    let skip_shm = code.len();
    let skip_shm_offset = |from: usize| -> i32 { (skip_shm as i32) - (from as i32 + 4) };
    let rel = skip_shm_offset(jz_skip_shm_pos + 2);
    code[jz_skip_shm_pos + 2..jz_skip_shm_pos + 6].copy_from_slice(&rel.to_le_bytes());
    let rel = skip_shm_offset(jc_skip_pos + 2);
    code[jc_skip_pos + 2..jc_skip_pos + 6].copy_from_slice(&rel.to_le_bytes());

    // Forkserver (simplified: same as LC_MAIN variant)
    if enable_forkserver {
        let forkserver_start = code.len();
        let rel = (forkserver_start as i32) - (jmp_forkserver_pos as i32 + 5);
        code[jmp_forkserver_pos + 1..jmp_forkserver_pos + 5].copy_from_slice(&rel.to_le_bytes());

        emit_macos_forkserver_x86_64(&mut code);
    } else {
        let epilogue = code.len();
        let rel = (epilogue as i32) - (jmp_forkserver_pos as i32 + 5);
        code[jmp_forkserver_pos + 1..jmp_forkserver_pos + 5].copy_from_slice(&rel.to_le_bytes());
    }

    // Epilogue
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x10]); // add rsp, 16
    code.extend_from_slice(&[0x41, 0x5E]); // pop r14
    code.extend_from_slice(&[0x41, 0x5D]); // pop r13
    code.extend_from_slice(&[0x41, 0x5C]); // pop r12
    code.push(0x5B); // pop rbx
    code.push(0x5D); // pop rbp

    // jmp original_entry
    let jmp_pos = code.len();
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);
    let jmp_rip = init_va + code.len() as u64;
    let rel = (original_entry as i64 - jmp_rip as i64) as i32;
    code[jmp_pos + 1..jmp_pos + 5].copy_from_slice(&rel.to_le_bytes());

    Ok(InitCode {
        va: init_va,
        code,
        entry_va,
    })
}

/// Emit forkserver loop using macOS syscalls.
/// Assumes [rsp] is available for scratch (4 bytes for messages).
fn emit_macos_forkserver_x86_64(code: &mut Vec<u8>) {
    // hello message
    code.extend_from_slice(&[0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00]);

    // write(FORKSRV_FD+1, &hello, 4)
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WRITE).to_le_bytes());
    code.extend_from_slice(&[0xBF]);
    code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
    code.extend_from_slice(&[0x48, 0x89, 0xE6]);
    code.extend_from_slice(&[0xBA, 0x04, 0x00, 0x00, 0x00]);
    code.extend_from_slice(&[0x0F, 0x05]);

    let fork_loop = code.len();

    // read(FORKSRV_FD, &cmd, 4)
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_READ).to_le_bytes());
    code.extend_from_slice(&[0xBF]);
    code.extend_from_slice(&FORKSRV_FD.to_le_bytes());
    code.extend_from_slice(&[0x48, 0x89, 0xE6]);
    code.extend_from_slice(&[0xBA, 0x04, 0x00, 0x00, 0x00]);
    code.extend_from_slice(&[0x0F, 0x05]);

    // fork()
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_FORK).to_le_bytes());
    code.extend_from_slice(&[0x0F, 0x05]);

    // macOS fork: rdx=0 parent, rdx=1 child
    code.extend_from_slice(&[0x85, 0xD2]); // test edx, edx
    let jnz_child_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);

    // Parent: report child pid
    code.extend_from_slice(&[0x89, 0x04, 0x24]); // mov [rsp], eax

    // write(FORKSRV_FD+1, &child_pid, 4)
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WRITE).to_le_bytes());
    code.extend_from_slice(&[0xBF]);
    code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
    code.extend_from_slice(&[0x48, 0x89, 0xE6]);
    code.extend_from_slice(&[0xBA, 0x04, 0x00, 0x00, 0x00]);
    code.extend_from_slice(&[0x0F, 0x05]);

    // wait4(child_pid, &status, 0, NULL)
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WAIT4).to_le_bytes());
    code.extend_from_slice(&[0x8B, 0x3C, 0x24]); // mov edi, [rsp]
    code.extend_from_slice(&[0x48, 0x8D, 0x74, 0x24, 0x04]); // lea rsi, [rsp+4]
    code.extend_from_slice(&[0x31, 0xD2]); // xor edx, edx
    code.extend_from_slice(&[0x4D, 0x31, 0xD2]); // xor r10, r10
    code.extend_from_slice(&[0x0F, 0x05]);

    // write(FORKSRV_FD+1, &status, 4)
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WRITE).to_le_bytes());
    code.extend_from_slice(&[0xBF]);
    code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
    code.extend_from_slice(&[0x48, 0x8D, 0x74, 0x24, 0x04]);
    code.extend_from_slice(&[0xBA, 0x04, 0x00, 0x00, 0x00]);
    code.extend_from_slice(&[0x0F, 0x05]);

    // jmp fork_loop
    let jmp_fork_rel = (fork_loop as i32) - (code.len() as i32 + 5);
    code.push(0xE9);
    code.extend_from_slice(&jmp_fork_rel.to_le_bytes());

    // .child:
    let child = code.len();
    let rel = (child as i32) - (jnz_child_pos as i32 + 6);
    code[jnz_child_pos + 2..jnz_child_pos + 6].copy_from_slice(&rel.to_le_bytes());

    // Close forkserver FDs
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_CLOSE).to_le_bytes());
    code.extend_from_slice(&[0xBF]);
    code.extend_from_slice(&FORKSRV_FD.to_le_bytes());
    code.extend_from_slice(&[0x0F, 0x05]);

    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_CLOSE).to_le_bytes());
    code.extend_from_slice(&[0xBF]);
    code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
    code.extend_from_slice(&[0x0F, 0x05]);
}

// ============================================================
// ARM64 macOS encoding helpers
// ============================================================
// Duplicated from arch::aarch64::trampoline_gen (which is behind
// #[cfg(feature = "aarch64")]) to avoid feature-gate coupling.

mod arm64 {
    #[inline]
    pub fn encode_b(off_instr: i32) -> [u8; 4] {
        (0x1400_0000u32 | ((off_instr as u32) & 0x03FF_FFFF)).to_le_bytes()
    }
    #[inline]
    pub fn encode_br(reg: u32) -> [u8; 4] {
        (0xD61F_0000u32 | (reg << 5)).to_le_bytes()
    }
    #[inline]
    pub fn encode_blr(reg: u32) -> [u8; 4] {
        (0xD63F_0000u32 | (reg << 5)).to_le_bytes()
    }
    #[inline]
    pub fn encode_ret() -> [u8; 4] {
        0xD65F_03C0u32.to_le_bytes()
    }
    #[inline]
    pub fn encode_nop() -> [u8; 4] {
        0xD503_201Fu32.to_le_bytes()
    }
    #[inline]
    pub fn encode_svc_0x80() -> [u8; 4] {
        0xD400_1001u32.to_le_bytes()
    }

    #[inline]
    pub fn encode_movz(reg: u32, imm16: u16) -> [u8; 4] {
        (0xD280_0000u32 | ((imm16 as u32) << 5) | reg).to_le_bytes()
    }
    #[inline]
    pub fn encode_movk(reg: u32, imm16: u16, lsl: u32) -> [u8; 4] {
        (0xF280_0000u32 | ((lsl / 16) << 21) | ((imm16 as u32) << 5) | reg).to_le_bytes()
    }
    #[inline]
    pub fn encode_mov_reg(dst: u32, src: u32) -> [u8; 4] {
        (0xAA00_03E0u32 | (src << 16) | dst).to_le_bytes()
    }
    #[inline]
    pub fn encode_add_imm(xd: u32, xn: u32, imm12: u32) -> [u8; 4] {
        (0x9100_0000u32 | ((imm12 & 0xFFF) << 10) | (xn << 5) | xd).to_le_bytes()
    }
    #[inline]
    pub fn encode_add_reg(xd: u32, xn: u32, xm: u32) -> [u8; 4] {
        (0x8B00_0000u32 | (xm << 16) | (xn << 5) | xd).to_le_bytes()
    }
    #[inline]
    pub fn encode_sub_sp_imm(imm: u32) -> [u8; 4] {
        let (sh, imm12) = if imm <= 0xFFF {
            (0u32, imm)
        } else if imm & 0xFFF == 0 && (imm >> 12) <= 0xFFF {
            (1u32, imm >> 12)
        } else {
            unreachable!("SUB sp, sp, #{imm}: immediate too large for single ADD/SUB encoding")
        };
        (0xD100_03FFu32 | (sh << 22) | ((imm12 & 0xFFF) << 10)).to_le_bytes()
    }
    #[inline]
    pub fn encode_add_sp_imm(imm: u32) -> [u8; 4] {
        let (sh, imm12) = if imm <= 0xFFF {
            (0u32, imm)
        } else if imm & 0xFFF == 0 && (imm >> 12) <= 0xFFF {
            (1u32, imm >> 12)
        } else {
            unreachable!("ADD sp, sp, #{imm}: immediate too large for single ADD/SUB encoding")
        };
        (0x9100_03FFu32 | (sh << 22) | ((imm12 & 0xFFF) << 10)).to_le_bytes()
    }
    #[inline]
    pub fn encode_stp_push(rn: u32, rm: u32) -> [u8; 4] {
        let imm7 = (-2i32 as u32) & 0x7F;
        ((0b10u32 << 30)
            | (0b101u32 << 27)
            | (0b011u32 << 23)
            | (imm7 << 15)
            | (rm << 10)
            | (31u32 << 5)
            | rn)
            .to_le_bytes()
    }
    #[inline]
    pub fn encode_ldp_pop(rn: u32, rm: u32) -> [u8; 4] {
        let imm7 = 2u32 & 0x7F;
        ((0b10u32 << 30)
            | (0b101u32 << 27)
            | (0b001u32 << 23)
            | (1u32 << 22)
            | (imm7 << 15)
            | (rm << 10)
            | (31u32 << 5)
            | rn)
            .to_le_bytes()
    }
    #[inline]
    pub fn encode_stp_offset(rn: u32, rm: u32, offset: u32) -> [u8; 4] {
        let imm7 = (offset / 8) & 0x7F;
        ((0b10u32 << 30)
            | (0b101u32 << 27)
            | (0b010u32 << 23)
            | (imm7 << 15)
            | (rm << 10)
            | (31u32 << 5)
            | rn)
            .to_le_bytes()
    }
    #[inline]
    pub fn encode_ldp_offset(rn: u32, rm: u32, offset: u32) -> [u8; 4] {
        let imm7 = (offset / 8) & 0x7F;
        ((0b10u32 << 30)
            | (0b101u32 << 27)
            | (0b010u32 << 23)
            | (1u32 << 22)
            | (imm7 << 15)
            | (rm << 10)
            | (31u32 << 5)
            | rn)
            .to_le_bytes()
    }
    #[inline]
    pub fn encode_str_reg(xn: u32, xm: u32) -> [u8; 4] {
        (0xF900_0000u32 | (xm << 5) | xn).to_le_bytes()
    }
    #[inline]
    pub fn encode_str_sp_imm(xn: u32, offset: u32) -> [u8; 4] {
        (0xF900_03E0u32 | (((offset / 8) & 0xFFF) << 10) | xn).to_le_bytes()
    }
    #[inline]
    pub fn encode_ldr_reg(xn: u32, xm: u32) -> [u8; 4] {
        (0xF940_0000u32 | (xm << 5) | xn).to_le_bytes()
    }
    #[inline]
    pub fn encode_ldr_imm(xn: u32, xm: u32, offset: u32) -> [u8; 4] {
        (0xF940_0000u32 | (((offset / 8) & 0xFFF) << 10) | (xm << 5) | xn).to_le_bytes()
    }
    #[inline]
    pub fn encode_ldr_w_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
        (0xB940_0000u32 | (((offset / 4) & 0xFFF) << 10) | (xm << 5) | wn).to_le_bytes()
    }
    #[inline]
    pub fn encode_ldrb_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
        (0x3940_0000u32 | ((offset & 0xFFF) << 10) | (xm << 5) | wn).to_le_bytes()
    }
    #[inline]
    pub fn encode_ldrb_post_inc(wn: u32, xm: u32) -> [u8; 4] {
        (0x3840_1400u32 | (xm << 5) | wn).to_le_bytes()
    }
    #[inline]
    pub fn encode_strh_zero(xm: u32, offset: u32) -> [u8; 4] {
        let imm12 = (offset / 2) & 0xFFF;
        (0x7900_0000u32 | (imm12 << 10) | (xm << 5) | 31).to_le_bytes()
    }
    #[inline]
    pub fn encode_cmp_reg(xn: u32, xm: u32) -> [u8; 4] {
        (0xEB00_001Fu32 | (xm << 16) | (xn << 5)).to_le_bytes()
    }
    #[inline]
    pub fn encode_cmp_w_imm(wn: u32, imm: u32) -> [u8; 4] {
        (0x7100_001Fu32 | ((imm & 0xFFF) << 10) | (wn << 5)).to_le_bytes()
    }
    #[inline]
    pub fn encode_cmp_w_reg(wn: u32, wm: u32) -> [u8; 4] {
        (0x6B00_001Fu32 | (wm << 16) | (wn << 5)).to_le_bytes()
    }
    #[inline]
    pub fn encode_cmp_imm0(xn: u32) -> [u8; 4] {
        (0xF100_001Fu32 | (xn << 5)).to_le_bytes()
    }
    #[inline]
    pub fn encode_cbz(xn: u32, offset_bytes: i32) -> [u8; 4] {
        let imm19 = ((offset_bytes / 4) as u32) & 0x7FFFF;
        (0xB400_0000u32 | (imm19 << 5) | xn).to_le_bytes()
    }
    #[inline]
    pub fn encode_cbnz(xn: u32, offset_bytes: i32) -> [u8; 4] {
        let imm19 = ((offset_bytes / 4) as u32) & 0x7FFFF;
        (0xB500_0000u32 | (imm19 << 5) | xn).to_le_bytes()
    }
    #[inline]
    pub fn encode_mul(xd: u32, xn: u32, xm: u32) -> [u8; 4] {
        (0x9B00_7C00u32 | (xm << 16) | (xn << 5) | xd).to_le_bytes()
    }
    #[inline]
    pub fn encode_add_uxtw(xd: u32, xn: u32, wm: u32) -> [u8; 4] {
        (0x8B20_0000u32 | (wm << 16) | (xn << 5) | xd).to_le_bytes()
    }
    #[inline]
    pub fn encode_sub_w_imm(wd: u32, wn: u32, imm: u32) -> [u8; 4] {
        (0x5100_0000u32 | ((imm & 0xFFF) << 10) | (wn << 5) | wd).to_le_bytes()
    }
    #[inline]
    pub fn encode_ldr_literal(xn: u32, offset_bytes: i32) -> [u8; 4] {
        let imm19 = ((offset_bytes / 4) as u32) & 0x7FFFF;
        (0x5800_0000u32 | (imm19 << 5) | xn).to_le_bytes()
    }
    #[inline]
    pub fn encode_adr(rd: u32, byte_offset: i64) -> [u8; 4] {
        let imm21 = (byte_offset & 0x1FFFFF) as u32;
        let immlo = imm21 & 0x3;
        let immhi = (imm21 >> 2) & 0x7FFFF;
        ((immlo << 29) | (0b10000u32 << 24) | (immhi << 5) | rd).to_le_bytes()
    }
    #[inline]
    pub fn encode_adrp(rd: u32, byte_offset: i64) -> [u8; 4] {
        let page_offset = byte_offset >> 12;
        let imm21 = (page_offset & 0x1FFFFF) as u32;
        let immlo = imm21 & 0x3;
        let immhi = (imm21 >> 2) & 0x7FFFF;
        ((1u32 << 31) | (immlo << 29) | (0b10000u32 << 24) | (immhi << 5) | rd).to_le_bytes()
    }
    /// Wide PC-relative address: ADRP + ADD (±4 GiB, 8 bytes).
    pub fn encode_adr_wide(rd: u32, pc: u64, target: u64) -> [u8; 8] {
        let pc_page = pc & !0xFFF;
        let target_page = target & !0xFFF;
        let page_offset = target_page as i64 - pc_page as i64;
        let page_off12 = (target & 0xFFF) as u32;
        let adrp = encode_adrp(rd, page_offset);
        let add = encode_add_imm(rd, rd, page_off12);
        let mut result = [0u8; 8];
        result[..4].copy_from_slice(&adrp);
        result[4..].copy_from_slice(&add);
        result
    }
    #[inline]
    pub fn adr_in_range(byte_offset: i64) -> bool {
        byte_offset >= -(1 << 20) && byte_offset < (1 << 20)
    }
    /// Patch an 8-byte placeholder (two NOPs) with ADR+NOP or ADRP+ADD.
    pub fn patch_adr_auto(code: &mut [u8], pos: usize, pc: u64, target: u64, rd: u32) {
        let offset = target as i64 - pc as i64;
        if adr_in_range(offset) {
            code[pos..pos + 4].copy_from_slice(&encode_adr(rd, offset));
            code[pos + 4..pos + 8].copy_from_slice(&encode_nop());
        } else {
            code[pos..pos + 8].copy_from_slice(&encode_adr_wide(rd, pc, target));
        }
    }

    pub fn emit_mov32(code: &mut Vec<u8>, reg: u32, val: u32) {
        code.extend_from_slice(&encode_movz(reg, (val & 0xFFFF) as u16));
        if (val >> 16) != 0 {
            code.extend_from_slice(&encode_movk(reg, ((val >> 16) & 0xFFFF) as u16, 16));
        }
    }
    pub fn emit_macos_syscall(code: &mut Vec<u8>, num: u32) {
        emit_mov32(code, 16, num);
        code.extend_from_slice(&encode_svc_0x80());
    }
    pub fn patch_b_cond(code: &mut [u8], pos: usize, delta_bytes: i32, cond: u32) {
        let imm19 = ((delta_bytes / 4) as u32) & 0x7FFFF;
        let word = 0x5400_0000u32 | (imm19 << 5) | cond;
        code[pos..pos + 4].copy_from_slice(&word.to_le_bytes());
    }
    pub fn patch_ldr_literal(
        code: &mut [u8],
        instr_pos: usize,
        base_va: u64,
        lit_pos: usize,
        xn: u32,
    ) {
        let delta = (base_va + lit_pos as u64) as i64 - (base_va + instr_pos as u64) as i64;
        code[instr_pos..instr_pos + 4].copy_from_slice(&encode_ldr_literal(xn, delta as i32));
    }
    /// STRB wn, [xm, #offset] — byte store with unsigned immediate offset
    #[inline]
    pub fn encode_strb_imm(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
        (0x3900_0000u32 | ((offset & 0xFFF) << 10) | (xm << 5) | wn).to_le_bytes()
    }
    /// STR xn, [xm, #offset] — 64-bit store with unsigned immediate offset
    #[inline]
    pub fn encode_str_imm_base(xn: u32, xm: u32, offset: u32) -> [u8; 4] {
        (0xF900_0000u32 | (((offset / 8) & 0xFFF) << 10) | (xm << 5) | xn).to_le_bytes()
    }
    /// STR wn, [xm, #offset] — 32-bit store with unsigned immediate offset
    #[inline]
    pub fn encode_str_w_imm_base(wn: u32, xm: u32, offset: u32) -> [u8; 4] {
        (0xB900_0000u32 | (((offset / 4) & 0xFFF) << 10) | (xm << 5) | wn).to_le_bytes()
    }
    /// SUBS wd, wn, #imm12 — subtract immediate, setting flags (32-bit)
    #[inline]
    pub fn encode_subs_w_imm(wd: u32, wn: u32, imm: u32) -> [u8; 4] {
        (0x7100_0000u32 | ((imm & 0xFFF) << 10) | (wn << 5) | wd).to_le_bytes()
    }

    // ------------------------------------------------------------------
    // PC-relative instruction relocation
    // ------------------------------------------------------------------
    // Duplicated from arch::aarch64::trampoline_gen (behind feature gate)
    // to keep macho_trampoline self-contained without feature-gate coupling.

    /// Sign-extend a value from `bits` width to i64.
    fn sign_extend(val: u32, bits: u32) -> i64 {
        let shift = 64 - bits;
        ((val as i64) << shift) >> shift
    }

    /// Return true if the instruction word is PC-relative.
    pub fn is_pc_relative(raw: u32) -> bool {
        let is_adrp = (raw & 0x9F00_0000) == 0x9000_0000;
        let is_adr = (raw & 0x9F00_0000) == 0x1000_0000;
        let is_ldr_lit = (raw & 0x3B00_0000) == 0x1800_0000;
        let is_b = (raw & 0xFC00_0000) == 0x1400_0000;
        let is_bl = (raw & 0xFC00_0000) == 0x9400_0000;
        let is_bcond = (raw & 0xFF00_0010) == 0x5400_0000;
        let is_cbz_cbnz = (raw & 0x7E00_0000) == 0x3400_0000;
        let is_tbz_tbnz = (raw & 0x7E00_0000) == 0x3600_0000;
        is_adrp || is_adr || is_ldr_lit || is_b || is_bl || is_bcond || is_cbz_cbnz || is_tbz_tbnz
    }

    /// Relocate a PC-relative AArch64 instruction from `original_va` to `new_va`.
    ///
    /// Decodes the instruction, computes its original target, then re-encodes with an
    /// adjusted immediate so it reaches the same target from the new location.
    /// Returns the relocated instruction word, or an error if the new offset is out of range.
    pub fn relocate_pc_relative(raw: u32, original_va: u64, new_va: u64) -> anyhow::Result<u32> {
        let is_adrp = (raw & 0x9F00_0000) == 0x9000_0000;
        let is_adr = (raw & 0x9F00_0000) == 0x1000_0000;
        let is_ldr_lit = (raw & 0x3B00_0000) == 0x1800_0000;
        let is_b = (raw & 0xFC00_0000) == 0x1400_0000;
        let is_bl = (raw & 0xFC00_0000) == 0x9400_0000;
        let is_bcond = (raw & 0xFF00_0010) == 0x5400_0000;
        let is_cbz_cbnz = (raw & 0x7E00_0000) == 0x3400_0000;
        let is_tbz_tbnz = (raw & 0x7E00_0000) == 0x3600_0000;

        if is_adrp {
            let rd = raw & 0x1F;
            let immlo = (raw >> 29) & 0x3;
            let immhi = (raw >> 5) & 0x7_FFFF;
            let imm21 = (immhi << 2) | immlo;
            let imm_signed = sign_extend(imm21, 21);
            let target_page = ((original_va & !0xFFF) as i64 + (imm_signed << 12)) as u64;

            let new_imm = ((target_page as i64) - (new_va as i64 & !0xFFF_i64)) >> 12;
            if !(-(1 << 20)..(1 << 20)).contains(&new_imm) {
                anyhow::bail!(
                    "ADRP relocation out of range: target page 0x{:x}, new VA 0x{:x}, delta {} pages",
                    target_page,
                    new_va,
                    new_imm
                );
            }
            let new_imm21 = (new_imm as u32) & 0x1F_FFFF;
            let new_immhi = (new_imm21 >> 2) & 0x7_FFFF;
            let new_immlo = new_imm21 & 0x3;
            Ok((1u32 << 31) | (new_immlo << 29) | (0b10000u32 << 24) | (new_immhi << 5) | rd)
        } else if is_adr {
            let rd = raw & 0x1F;
            let immlo = (raw >> 29) & 0x3;
            let immhi = (raw >> 5) & 0x7_FFFF;
            let imm21 = (immhi << 2) | immlo;
            let imm_signed = sign_extend(imm21, 21);
            let target = (original_va as i64 + imm_signed) as u64;

            let new_offset = target as i64 - new_va as i64;
            if !(-(1 << 20)..(1 << 20)).contains(&new_offset) {
                anyhow::bail!(
                    "ADR relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {}",
                    target,
                    new_va,
                    new_offset
                );
            }
            let new_imm21 = (new_offset as u32) & 0x1F_FFFF;
            let new_immhi = (new_imm21 >> 2) & 0x7_FFFF;
            let new_immlo = new_imm21 & 0x3;
            Ok((new_immlo << 29) | (0b10000u32 << 24) | (new_immhi << 5) | rd)
        } else if is_ldr_lit {
            let rt = raw & 0x1F;
            let opc_v = raw & 0xFF00_0000;
            let imm19 = (raw >> 5) & 0x7_FFFF;
            let offset_signed = sign_extend(imm19, 19) * 4;
            let target = (original_va as i64 + offset_signed) as u64;

            let new_offset = target as i64 - new_va as i64;
            if new_offset % 4 != 0 || new_offset / 4 < -(1 << 18) || new_offset / 4 >= (1 << 18) {
                anyhow::bail!(
                    "LDR literal relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {}",
                    target,
                    new_va,
                    new_offset
                );
            }
            let new_imm19 = ((new_offset / 4) as u32) & 0x7_FFFF;
            Ok(opc_v | (new_imm19 << 5) | rt)
        } else if is_b || is_bl {
            let opcode_bits = raw & 0xFC00_0000;
            let imm26 = raw & 0x03FF_FFFF;
            let offset_signed = sign_extend(imm26, 26) * 4;
            let target = (original_va as i64 + offset_signed) as u64;

            let new_offset = target as i64 - new_va as i64;
            if new_offset % 4 != 0 || new_offset / 4 < -(1 << 25) || new_offset / 4 >= (1 << 25) {
                anyhow::bail!(
                    "B/BL relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {}",
                    target,
                    new_va,
                    new_offset
                );
            }
            let new_imm26 = ((new_offset / 4) as u32) & 0x03FF_FFFF;
            Ok(opcode_bits | new_imm26)
        } else if is_bcond {
            let cond = raw & 0xF;
            let imm19 = (raw >> 5) & 0x7_FFFF;
            let offset_signed = sign_extend(imm19, 19) * 4;
            let target = (original_va as i64 + offset_signed) as u64;

            let new_offset = target as i64 - new_va as i64;
            if new_offset % 4 != 0 || new_offset / 4 < -(1 << 18) || new_offset / 4 >= (1 << 18) {
                anyhow::bail!(
                    "B.cond relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {}",
                    target,
                    new_va,
                    new_offset
                );
            }
            let new_imm19 = ((new_offset / 4) as u32) & 0x7_FFFF;
            Ok(0x5400_0000 | (new_imm19 << 5) | cond)
        } else if is_cbz_cbnz {
            let sf_op = raw & 0xFF00_0000;
            let rt = raw & 0x1F;
            let imm19 = (raw >> 5) & 0x7_FFFF;
            let offset_signed = sign_extend(imm19, 19) * 4;
            let target = (original_va as i64 + offset_signed) as u64;

            let new_offset = target as i64 - new_va as i64;
            if new_offset % 4 != 0 || new_offset / 4 < -(1 << 18) || new_offset / 4 >= (1 << 18) {
                anyhow::bail!(
                    "CBZ/CBNZ relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {}",
                    target,
                    new_va,
                    new_offset
                );
            }
            let new_imm19 = ((new_offset / 4) as u32) & 0x7_FFFF;
            Ok(sf_op | (new_imm19 << 5) | rt)
        } else if is_tbz_tbnz {
            let b5_op = raw & 0xFF00_0000;
            let b40 = (raw >> 19) & 0x1F;
            let rt = raw & 0x1F;
            let imm14 = (raw >> 5) & 0x3FFF;
            let offset_signed = sign_extend(imm14, 14) * 4;
            let target = (original_va as i64 + offset_signed) as u64;

            let new_offset = target as i64 - new_va as i64;
            if new_offset % 4 != 0 || new_offset / 4 < -(1 << 13) || new_offset / 4 >= (1 << 13) {
                anyhow::bail!(
                    "TBZ/TBNZ relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {}",
                    target,
                    new_va,
                    new_offset
                );
            }
            let new_imm14 = ((new_offset / 4) as u32) & 0x3FFF;
            Ok(b5_op | (b40 << 19) | (new_imm14 << 5) | rt)
        } else {
            anyhow::bail!(
                "instruction 0x{:08x} is not a recognised PC-relative encoding",
                raw
            );
        }
    }
}

use crate::trampoline::PREV_LOC_OFFSET;

// ============================================================
// ARM64 macOS init code generators
// ============================================================

/// Generate macOS ARM64 init code for executables (LC_MAIN entry).
///
/// With LC_MAIN, dyld calls the entry point as: main(x0=argc, x1=argv, x2=envp, x3=apple).
/// We scan envp for __AFL_SHM_ID=, attach SHM, optionally start a forkserver,
/// then jump to the original entry.
pub fn generate_macho_exec_init_aarch64(
    init_va: u64,
    data_va: u64,
    original_entry: u64,
    enable_forkserver: bool,
    use_got_shmat: bool,
) -> Result<InitCode> {
    use arm64::*;
    let mut code = Vec::with_capacity(1024);
    let entry_va = init_va;

    // ── Prologue: save x0 + callee-saved regs ──
    // x0..x3 = argc/argv/envp/apple from dyld
    code.extend_from_slice(&encode_stp_push(0, 1)); // STP x0, x1, [sp, #-16]!
    code.extend_from_slice(&encode_stp_push(2, 3)); // STP x2, x3, [sp, #-16]!
    code.extend_from_slice(&encode_sub_sp_imm(96)); // 6 pairs of callee-saved
    code.extend_from_slice(&encode_stp_offset(29, 30, 0));
    code.extend_from_slice(&encode_stp_offset(27, 28, 16));
    code.extend_from_slice(&encode_stp_offset(25, 26, 32));
    code.extend_from_slice(&encode_stp_offset(23, 24, 48));
    code.extend_from_slice(&encode_stp_offset(21, 22, 64));
    code.extend_from_slice(&encode_stp_offset(19, 20, 80));

    // x20 = envp (from x2, saved above)
    code.extend_from_slice(&encode_mov_reg(20, 2));

    // x19 = data_va (via ADR or ADRP+ADD)
    let adr_data_pos = code.len();
    code.extend_from_slice(&encode_nop()); // placeholder slot 1
    code.extend_from_slice(&encode_nop()); // placeholder slot 2

    // Load needle bytes for __AFL_SHM_ID= comparison
    let needle_a_pos = code.len();
    code.extend_from_slice(&encode_nop()); // placeholder: LDR x25, needle_a
    let needle_b_pos = code.len();
    code.extend_from_slice(&encode_nop()); // placeholder: LDR x26, needle_b

    // ── Scan envp array ──
    // x20 = pointer to envp[] (array of char*)
    let scan_loop = code.len();

    // LDR x21, [x20], #8  (load next envp pointer, advance)
    // Post-index: LDR x21, [x20], #8 = 0xF8408680 | (Rn<<5) | Rt
    // 0xF8408680 = size=11(64), V=0, opc=01, imm9=8, op2=01(post-index), Rn=0, Rt=0
    code.extend_from_slice(&(0xF840_8680u32 | (20 << 5) | 21).to_le_bytes());

    // CBZ x21, .not_found
    let cbz_notfound_pos = code.len();
    code.extend_from_slice(&encode_cbz(21, 0)); // placeholder

    // LDR x27, [x21]  (first 8 bytes)
    code.extend_from_slice(&encode_ldr_reg(27, 21));
    // CMP x27, x25 ("__AFL_SH")
    code.extend_from_slice(&encode_cmp_reg(27, 25));
    let bne1 = code.len();
    code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]); // B.NE placeholder

    // LDR w27, [x21, #8]  (next 4 bytes: "M_ID")
    code.extend_from_slice(&encode_ldr_w_imm(27, 21, 8));
    code.extend_from_slice(&encode_cmp_w_reg(27, 26));
    let bne2 = code.len();
    code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]); // B.NE placeholder

    // LDRB w27, [x21, #12]  (check '=')
    code.extend_from_slice(&encode_ldrb_imm(27, 21, 12));
    code.extend_from_slice(&encode_cmp_w_imm(27, b'=' as u32));
    let bne3 = code.len();
    code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]); // B.NE placeholder

    // ── Found: parse decimal at x21+13 ──
    code.extend_from_slice(&encode_add_imm(22, 21, 13)); // x22 = &decimal
    code.extend_from_slice(&encode_movz(24, 0)); // x24 = result
    code.extend_from_slice(&encode_movz(25, 10)); // x25 = 10

    let parse_loop = code.len();
    code.extend_from_slice(&encode_ldrb_post_inc(27, 22));
    code.extend_from_slice(&encode_sub_w_imm(27, 27, b'0' as u32));
    code.extend_from_slice(&encode_cmp_w_imm(27, 10));
    let bhs_pos = code.len();
    code.extend_from_slice(&[0x82, 0x00, 0x00, 0x54]); // B.HS placeholder
    code.extend_from_slice(&encode_mul(24, 24, 25));
    code.extend_from_slice(&encode_add_uxtw(24, 24, 27));
    let bl = code.len();
    code.extend_from_slice(&encode_b(((parse_loop as i64 - bl as i64) / 4) as i32));

    let parse_done = code.len();
    patch_b_cond(
        &mut code,
        bhs_pos,
        (parse_done as i64 - bhs_pos as i64) as i32,
        0x2,
    );

    // B .shm_attach
    let b_shm_pos = code.len();
    code.extend_from_slice(&encode_b(0)); // placeholder

    // ── .next_env: loop back ──
    let next_env = code.len();
    patch_b_cond(&mut code, bne1, (next_env as i64 - bne1 as i64) as i32, 0x1);
    patch_b_cond(&mut code, bne2, (next_env as i64 - bne2 as i64) as i32, 0x1);
    patch_b_cond(&mut code, bne3, (next_env as i64 - bne3 as i64) as i32, 0x1);
    let sl = code.len();
    code.extend_from_slice(&encode_b(((scan_loop as i64 - sl as i64) / 4) as i32));

    // ── .not_found ──
    let not_found = code.len();
    let cbz_delta = (not_found as i64 - cbz_notfound_pos as i64) as i32;
    code[cbz_notfound_pos..cbz_notfound_pos + 4].copy_from_slice(&encode_cbz(21, cbz_delta));
    let b_epilogue_pos = code.len();
    code.extend_from_slice(&encode_b(0)); // placeholder: B .epilogue

    // ── .shm_attach: x24 = shmid ──
    let shm_attach = code.len();
    let sd = (shm_attach as i64 - b_shm_pos as i64) / 4;
    code[b_shm_pos..b_shm_pos + 4].copy_from_slice(&encode_b(sd as i32));

    // shmat(x24, 0, 0)
    let b_cs_skip_pos;
    if use_got_shmat {
        // Load shmat from GOT slot at [x19+16]
        code.extend_from_slice(&encode_ldr_imm(9, 19, 16)); // x9 = [data_va+16]
        code.extend_from_slice(&encode_mov_reg(0, 24)); // x0 = shmid
        code.extend_from_slice(&encode_movz(1, 0));
        code.extend_from_slice(&encode_movz(2, 0));
        code.extend_from_slice(&encode_blr(9));
        // Check return: (void*)-1 = error
        code.extend_from_slice(&encode_add_imm(9, 0, 1)); // x9 = x0+1
        code.extend_from_slice(&encode_cmp_imm0(9)); // x0+1 == 0 means x0 was -1
        b_cs_skip_pos = code.len();
        // B.EQ .epilogue (cond=0)
        code.extend_from_slice(&[0x00, 0x00, 0x00, 0x54]); // placeholder
    } else {
        code.extend_from_slice(&encode_mov_reg(0, 24)); // x0 = shmid
        code.extend_from_slice(&encode_movz(1, 0)); // x1 = NULL
        code.extend_from_slice(&encode_movz(2, 0)); // x2 = 0
        emit_macos_syscall(&mut code, MACOS_SYS_SHMAT);
        // B.CS .epilogue (carry set = error, cond=2)
        b_cs_skip_pos = code.len();
        code.extend_from_slice(&[0x02, 0x00, 0x00, 0x54]); // placeholder
    }

    // Store SHM pointer: STR x0, [x19]
    code.extend_from_slice(&encode_str_reg(0, 19));
    // Clear prev_loc
    code.extend_from_slice(&encode_strh_zero(19, PREV_LOC_OFFSET as u32));

    // ── Forkserver (optional) ──
    if enable_forkserver {
        emit_macos_forkserver_aarch64(&mut code);
    }

    // ── Epilogue ──
    let epilogue = code.len();
    // Patch B .epilogue
    let ed = (epilogue as i64 - b_epilogue_pos as i64) / 4;
    code[b_epilogue_pos..b_epilogue_pos + 4].copy_from_slice(&encode_b(ed as i32));
    // Patch B.CS/B.EQ skip
    patch_b_cond(
        &mut code,
        b_cs_skip_pos,
        (epilogue as i64 - b_cs_skip_pos as i64) as i32,
        if use_got_shmat { 0x0 } else { 0x2 },
    );

    // Restore callee-saved
    code.extend_from_slice(&encode_ldp_offset(19, 20, 80));
    code.extend_from_slice(&encode_ldp_offset(21, 22, 64));
    code.extend_from_slice(&encode_ldp_offset(23, 24, 48));
    code.extend_from_slice(&encode_ldp_offset(25, 26, 32));
    code.extend_from_slice(&encode_ldp_offset(27, 28, 16));
    code.extend_from_slice(&encode_ldp_offset(29, 30, 0));
    code.extend_from_slice(&encode_add_sp_imm(96));
    code.extend_from_slice(&encode_ldp_pop(2, 3)); // restore x2, x3
    code.extend_from_slice(&encode_ldp_pop(0, 1)); // restore x0, x1

    // Jump to original entry
    let branch_va = init_va + code.len() as u64;
    let delta = original_entry as i64 - branch_va as i64;
    if delta.abs() < (128 << 20) {
        code.extend_from_slice(&encode_b((delta / 4) as i32));
    } else {
        // ADRP+ADD+BR for long range
        let adrp_va = init_va + code.len() as u64;
        let entry_page = (original_entry as i64) & !0xFFF;
        let adrp_page = (adrp_va as i64) & !0xFFF;
        let page_delta = entry_page - adrp_page;
        let immhi = ((page_delta >> 14) & 0x7FFFF) as u32;
        let immlo = ((page_delta >> 12) & 0x3) as u32;
        let adrp = 0x9000_0010u32 | (immlo << 29) | (immhi << 5);
        code.extend_from_slice(&adrp.to_le_bytes());
        code.extend_from_slice(&encode_add_imm(16, 16, (original_entry & 0xFFF) as u32));
        code.extend_from_slice(&encode_br(16));
    }

    // ── Literal pool ──
    while code.len() % 8 != 0 {
        code.extend_from_slice(&encode_nop());
    }
    let needle_a_lit = code.len();
    code.extend_from_slice(&u64::from_le_bytes(*b"__AFL_SH").to_le_bytes());
    let needle_b_lit = code.len();
    code.extend_from_slice(&(u32::from_le_bytes(*b"M_ID") as u64).to_le_bytes());

    // ── Patch placeholders ──
    let adr_data_pc = init_va + adr_data_pos as u64;
    patch_adr_auto(&mut code, adr_data_pos, adr_data_pc, data_va, 19);
    patch_ldr_literal(&mut code, needle_a_pos, init_va, needle_a_lit, 25);
    patch_ldr_literal(&mut code, needle_b_pos, init_va, needle_b_lit, 26);

    Ok(InitCode {
        va: init_va,
        code,
        entry_va,
    })
}

/// Generate macOS ARM64 init code for dylibs (sysctl KERN_PROCARGS2).
///
/// Dylib constructors receive no arguments. We use sysctl to read the
/// process environment from the kernel, then scan for __AFL_SHM_ID=.
pub fn generate_macho_dylib_init_aarch64(
    init_va: u64,
    data_va: u64,
    chain_to: Option<u64>,
    use_got_shmat: bool,
) -> Result<InitCode> {
    use arm64::*;
    let mut code = Vec::with_capacity(1024);
    let entry_va = init_va;

    // ── Prologue ──
    const BUF_OFF: u32 = 96; // sysctl buffer at sp+96
    // Stack: 96 regs + 8192 buf - 32 padding = 8256
    // Split into three SUBs: 4096 + 4096 + 64 (each must be <= 4095 or 4096-aligned)
    code.extend_from_slice(&encode_sub_sp_imm(4096));
    code.extend_from_slice(&encode_sub_sp_imm(4096));
    code.extend_from_slice(&encode_sub_sp_imm(64));
    code.extend_from_slice(&encode_stp_offset(29, 30, 0));
    code.extend_from_slice(&encode_stp_offset(27, 28, 16));
    code.extend_from_slice(&encode_stp_offset(25, 26, 32));
    code.extend_from_slice(&encode_stp_offset(23, 24, 48));
    code.extend_from_slice(&encode_stp_offset(21, 22, 64));
    code.extend_from_slice(&encode_stp_offset(19, 20, 80));

    // x19 = data_va (via ADR or ADRP+ADD)
    let adr_data_pos = code.len();
    code.extend_from_slice(&encode_nop()); // placeholder slot 1
    code.extend_from_slice(&encode_nop()); // placeholder slot 2

    // Step 1: getpid
    emit_macos_syscall(&mut code, MACOS_SYS_GETPID);
    code.extend_from_slice(&encode_mov_reg(21, 0)); // x21 = pid

    // Step 2: set up MIB [CTL_KERN=1, KERN_PROCARGS2=49, pid] at sp+16
    let mib_off: u32 = 16;
    emit_mov32(&mut code, 27, CTL_KERN);
    code.extend_from_slice(&encode_str_sp_imm(27, mib_off));
    emit_mov32(&mut code, 27, KERN_PROCARGS2);
    code.extend_from_slice(&encode_str_sp_imm(27, mib_off + 8));
    code.extend_from_slice(&encode_str_sp_imm(21, mib_off + 16)); // pid

    // Step 3: set oldlenp at sp+40
    let len_off: u32 = 40;
    let buf_size: u64 = 8192;
    emit_mov32(&mut code, 27, buf_size as u32);
    code.extend_from_slice(&encode_str_sp_imm(27, len_off));

    // Step 4: sysctl(mib, 3, buf, &len, NULL, 0)
    code.extend_from_slice(&encode_add_imm(0, 31, mib_off)); // x0 = &mib
    code.extend_from_slice(&encode_movz(1, 3)); // x1 = 3
    code.extend_from_slice(&encode_add_imm(2, 31, BUF_OFF)); // x2 = buf
    code.extend_from_slice(&encode_add_imm(3, 31, len_off)); // x3 = &len
    code.extend_from_slice(&encode_movz(4, 0)); // x4 = NULL
    code.extend_from_slice(&encode_movz(5, 0)); // x5 = 0
    emit_macos_syscall(&mut code, MACOS_SYS_SYSCTL);
    // B.CS .done
    let bcs_done_pos = code.len();
    code.extend_from_slice(&[0x02, 0x00, 0x00, 0x54]); // placeholder

    // Step 5: parse KERN_PROCARGS2 buffer
    // x22 = buf start, x23 = buf end
    code.extend_from_slice(&encode_add_imm(22, 31, BUF_OFF));
    // Load actual bytes returned
    code.extend_from_slice(&encode_ldr_imm(27, 31, len_off)); // x27 = [sp+len_off] = bytes
    code.extend_from_slice(&encode_add_reg(23, 22, 27)); // x23 = buf + bytes_returned

    // Skip argc (4 bytes)
    code.extend_from_slice(&encode_add_imm(22, 22, 4));

    // Skip exec_path: scan for NUL byte
    let skip_exec_loop = code.len();
    code.extend_from_slice(&encode_cmp_reg(22, 23));
    let bge_done1 = code.len();
    code.extend_from_slice(&[0x0A, 0x00, 0x00, 0x54]); // B.GE .done placeholder
    code.extend_from_slice(&encode_ldrb_post_inc(27, 22));
    code.extend_from_slice(&encode_cmp_w_imm(27, 0));
    let bne_skip_exec = code.len();
    code.extend_from_slice(&[0x01, 0x00, 0x00, 0x54]); // B.NE placeholder
    patch_b_cond(
        &mut code,
        bne_skip_exec,
        (skip_exec_loop as i64 - bne_skip_exec as i64) as i32,
        0x1,
    );

    // Skip NUL padding between exec_path and argv
    let skip_nuls_loop = code.len();
    code.extend_from_slice(&encode_cmp_reg(22, 23));
    let bge_done2 = code.len();
    code.extend_from_slice(&[0x0A, 0x00, 0x00, 0x54]); // B.GE .done
    code.extend_from_slice(&encode_ldrb_post_inc(27, 22));
    code.extend_from_slice(&encode_cmp_w_imm(27, 0));
    let beq_nuls = code.len();
    code.extend_from_slice(&[0x00, 0x00, 0x00, 0x54]); // B.EQ placeholder
    patch_b_cond(
        &mut code,
        beq_nuls,
        (skip_nuls_loop as i64 - beq_nuls as i64) as i32,
        0x0,
    );
    // Back up one byte (post-inc went one past first non-NUL)
    // SUB x22, x22, #1 (64-bit subtract — x22 is a pointer, must use 64-bit form)
    let sub_x = 0xD100_0000u32 | (1 << 10) | (22 << 5) | 22;
    code.extend_from_slice(&sub_x.to_le_bytes());

    // Read argc from start of buffer: LDR w24, [sp+BUF_OFF]
    code.extend_from_slice(&encode_ldr_w_imm(24, 31, BUF_OFF));

    // Skip argc argv strings
    let skip_argv_loop = code.len();
    code.extend_from_slice(&encode_cbz(24, 0)); // CBZ x24, .scan_env (placeholder)
    let cbz_scan_env_pos = code.len() - 4;

    // Skip one arg string
    let skip_one_arg = code.len();
    code.extend_from_slice(&encode_cmp_reg(22, 23));
    let bge_done3 = code.len();
    code.extend_from_slice(&[0x0A, 0x00, 0x00, 0x54]); // B.GE .done
    code.extend_from_slice(&encode_ldrb_post_inc(27, 22));
    code.extend_from_slice(&encode_cmp_w_imm(27, 0));
    let bne_one = code.len();
    code.extend_from_slice(&[0x01, 0x00, 0x00, 0x54]); // B.NE .skip_one_arg
    patch_b_cond(
        &mut code,
        bne_one,
        (skip_one_arg as i64 - bne_one as i64) as i32,
        0x1,
    );
    // SUB x24, x24, #1 (decrement argc)
    let sub_x24 = 0xD100_0400u32 | (24 << 5) | 24; // SUB x24, x24, #1
    code.extend_from_slice(&sub_x24.to_le_bytes());
    let b_argv = code.len();
    code.extend_from_slice(&encode_b(
        ((skip_argv_loop as i64 - b_argv as i64) / 4) as i32,
    ));

    // .scan_env:
    let scan_env = code.len();
    let cbz_delta = (scan_env as i64 - cbz_scan_env_pos as i64) as i32;
    code[cbz_scan_env_pos..cbz_scan_env_pos + 4].copy_from_slice(&encode_cbz(24, cbz_delta));

    // Load needle
    let needle_a_pos_d = code.len();
    code.extend_from_slice(&encode_nop()); // LDR x25, needle_a
    let needle_b_pos_d = code.len();
    code.extend_from_slice(&encode_nop()); // LDR x26, needle_b

    // Env scan loop: same structure as exec init
    // Skip leading NULs between env strings
    let env_scan_loop = code.len();
    let skip_env_nuls = code.len();
    code.extend_from_slice(&encode_cmp_reg(22, 23));
    let bge_done4 = code.len();
    code.extend_from_slice(&[0x0A, 0x00, 0x00, 0x54]); // B.GE .done
    code.extend_from_slice(&encode_ldrb_post_inc(27, 22));
    code.extend_from_slice(&encode_cmp_w_imm(27, 0));
    let beq_nuls2 = code.len();
    code.extend_from_slice(&[0x00, 0x00, 0x00, 0x54]); // B.EQ placeholder
    patch_b_cond(
        &mut code,
        beq_nuls2,
        (skip_env_nuls as i64 - beq_nuls2 as i64) as i32,
        0x0,
    );
    // Back up one byte
    let sub_x22 = 0xD100_0400u32 | (22 << 5) | 22;
    code.extend_from_slice(&sub_x22.to_le_bytes());

    // Need 13 bytes: ADD x27, x22, #13; CMP x27, x23; B.GE .done
    code.extend_from_slice(&encode_add_imm(27, 22, 13));
    code.extend_from_slice(&encode_cmp_reg(27, 23));
    let bge_done5 = code.len();
    code.extend_from_slice(&[0x0A, 0x00, 0x00, 0x54]); // B.GE .done

    // Compare first 8 bytes with needle_a
    code.extend_from_slice(&encode_ldr_reg(27, 22));
    code.extend_from_slice(&encode_cmp_reg(27, 25));
    let bne_skip1 = code.len();
    code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]); // B.NE

    // Compare next 4 bytes "M_ID"
    code.extend_from_slice(&encode_ldr_w_imm(27, 22, 8));
    code.extend_from_slice(&encode_cmp_w_reg(27, 26));
    let bne_skip2 = code.len();
    code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]); // B.NE

    // Check '=' at offset 12
    code.extend_from_slice(&encode_ldrb_imm(27, 22, 12));
    code.extend_from_slice(&encode_cmp_w_imm(27, b'=' as u32));
    let bne_skip3 = code.len();
    code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]); // B.NE

    // Found! Parse decimal at x22+13
    code.extend_from_slice(&encode_add_imm(22, 22, 13));
    code.extend_from_slice(&encode_movz(24, 0));
    code.extend_from_slice(&encode_movz(25, 10));

    let parse2_loop = code.len();
    code.extend_from_slice(&encode_ldrb_post_inc(27, 22));
    code.extend_from_slice(&encode_sub_w_imm(27, 27, b'0' as u32));
    code.extend_from_slice(&encode_cmp_w_imm(27, 10));
    let bhs2 = code.len();
    code.extend_from_slice(&[0x82, 0x00, 0x00, 0x54]); // B.HS
    code.extend_from_slice(&encode_mul(24, 24, 25));
    code.extend_from_slice(&encode_add_uxtw(24, 24, 27));
    let bp2 = code.len();
    code.extend_from_slice(&encode_b(((parse2_loop as i64 - bp2 as i64) / 4) as i32));

    let parse2_done = code.len();
    patch_b_cond(
        &mut code,
        bhs2,
        (parse2_done as i64 - bhs2 as i64) as i32,
        0x2,
    );

    // B .shm_attach
    let b_shm2 = code.len();
    code.extend_from_slice(&encode_b(0)); // placeholder

    // .skip_this_env: skip current string, loop
    let skip_this_env = code.len();
    patch_b_cond(
        &mut code,
        bne_skip1,
        (skip_this_env as i64 - bne_skip1 as i64) as i32,
        0x1,
    );
    patch_b_cond(
        &mut code,
        bne_skip2,
        (skip_this_env as i64 - bne_skip2 as i64) as i32,
        0x1,
    );
    patch_b_cond(
        &mut code,
        bne_skip3,
        (skip_this_env as i64 - bne_skip3 as i64) as i32,
        0x1,
    );
    let skip_str_loop = code.len();
    code.extend_from_slice(&encode_cmp_reg(22, 23));
    let bge_done6 = code.len();
    code.extend_from_slice(&[0x0A, 0x00, 0x00, 0x54]); // B.GE .done
    code.extend_from_slice(&encode_ldrb_post_inc(27, 22));
    code.extend_from_slice(&encode_cmp_w_imm(27, 0));
    let bne_str = code.len();
    code.extend_from_slice(&[0x01, 0x00, 0x00, 0x54]); // B.NE
    patch_b_cond(
        &mut code,
        bne_str,
        (skip_str_loop as i64 - bne_str as i64) as i32,
        0x1,
    );
    let be = code.len();
    code.extend_from_slice(&encode_b(((env_scan_loop as i64 - be as i64) / 4) as i32));

    // .shm_attach: x24 = shmid
    let shm_attach2 = code.len();
    let sd3 = (shm_attach2 as i64 - b_shm2 as i64) / 4;
    code[b_shm2..b_shm2 + 4].copy_from_slice(&encode_b(sd3 as i32));

    let bcs_done2;
    if use_got_shmat {
        code.extend_from_slice(&encode_ldr_imm(9, 19, 16));
        code.extend_from_slice(&encode_mov_reg(0, 24));
        code.extend_from_slice(&encode_movz(1, 0));
        code.extend_from_slice(&encode_movz(2, 0));
        code.extend_from_slice(&encode_blr(9));
        code.extend_from_slice(&encode_add_imm(9, 0, 1));
        code.extend_from_slice(&encode_cmp_imm0(9));
        bcs_done2 = code.len();
        code.extend_from_slice(&[0x00, 0x00, 0x00, 0x54]); // B.EQ placeholder
    } else {
        code.extend_from_slice(&encode_mov_reg(0, 24));
        code.extend_from_slice(&encode_movz(1, 0));
        code.extend_from_slice(&encode_movz(2, 0));
        emit_macos_syscall(&mut code, MACOS_SYS_SHMAT);
        bcs_done2 = code.len();
        code.extend_from_slice(&[0x02, 0x00, 0x00, 0x54]); // B.CS placeholder
    }
    code.extend_from_slice(&encode_str_reg(0, 19));
    code.extend_from_slice(&encode_strh_zero(19, PREV_LOC_OFFSET as u32));

    // .done:
    let done = code.len();

    // Patch all B.GE .done and B.CS .done
    let done_targets = [
        bcs_done_pos,
        bge_done1,
        bge_done2,
        bge_done3,
        bge_done4,
        bge_done5,
        bge_done6,
    ];
    for &pos in &done_targets {
        // Read existing cond from the placeholder
        let existing = u32::from_le_bytes(code[pos..pos + 4].try_into().expect("slice is 4 bytes"));
        let cond = existing & 0xF;
        patch_b_cond(&mut code, pos, (done as i64 - pos as i64) as i32, cond);
    }
    patch_b_cond(
        &mut code,
        bcs_done2,
        (done as i64 - bcs_done2 as i64) as i32,
        if use_got_shmat { 0x0 } else { 0x2 },
    );

    // Epilogue: restore and chain/ret
    code.extend_from_slice(&encode_ldp_offset(19, 20, 80));
    code.extend_from_slice(&encode_ldp_offset(21, 22, 64));
    code.extend_from_slice(&encode_ldp_offset(23, 24, 48));
    code.extend_from_slice(&encode_ldp_offset(25, 26, 32));
    code.extend_from_slice(&encode_ldp_offset(27, 28, 16));
    code.extend_from_slice(&encode_ldp_offset(29, 30, 0));
    code.extend_from_slice(&encode_add_sp_imm(64));
    code.extend_from_slice(&encode_add_sp_imm(4096));
    code.extend_from_slice(&encode_add_sp_imm(4096));

    if let Some(orig_init) = chain_to {
        // Jump to original constructor
        let branch_va2 = init_va + code.len() as u64;
        let delta2 = orig_init as i64 - branch_va2 as i64;
        if delta2.abs() < (128 << 20) {
            code.extend_from_slice(&encode_b((delta2 / 4) as i32));
        } else {
            let adrp_va = init_va + code.len() as u64;
            let ep = (orig_init as i64) & !0xFFF;
            let ap = (adrp_va as i64) & !0xFFF;
            let pd = ep - ap;
            let immhi = ((pd >> 14) & 0x7FFFF) as u32;
            let immlo = ((pd >> 12) & 0x3) as u32;
            code.extend_from_slice(&(0x9000_0010u32 | (immlo << 29) | (immhi << 5)).to_le_bytes());
            code.extend_from_slice(&encode_add_imm(16, 16, (orig_init & 0xFFF) as u32));
            code.extend_from_slice(&encode_br(16));
        }
    } else {
        code.extend_from_slice(&encode_ret());
    }

    // ── Literal pool ──
    while code.len() % 8 != 0 {
        code.extend_from_slice(&encode_nop());
    }
    let needle_a_lit2 = code.len();
    code.extend_from_slice(&u64::from_le_bytes(*b"__AFL_SH").to_le_bytes());
    let needle_b_lit2 = code.len();
    code.extend_from_slice(&(u32::from_le_bytes(*b"M_ID") as u64).to_le_bytes());

    // Patch placeholders
    let adr_data_pc = init_va + adr_data_pos as u64;
    patch_adr_auto(&mut code, adr_data_pos, adr_data_pc, data_va, 19);
    patch_ldr_literal(&mut code, needle_a_pos_d, init_va, needle_a_lit2, 25);
    patch_ldr_literal(&mut code, needle_b_pos_d, init_va, needle_b_lit2, 26);

    Ok(InitCode {
        va: init_va,
        code,
        entry_va,
    })
}

/// Generate macOS ARM64 init code for LC_UNIXTHREAD executables.
///
/// LC_UNIXTHREAD provides raw register state. Entry receives a raw stack:
/// [argc] [argv...] [NULL] [envp...] [NULL] — same as ELF convention.
pub fn generate_macho_unixthread_init_aarch64(
    init_va: u64,
    data_va: u64,
    original_entry: u64,
    enable_forkserver: bool,
    use_got_shmat: bool,
) -> Result<InitCode> {
    use arm64::*;
    let mut code = Vec::with_capacity(1024);
    let entry_va = init_va;

    // ── Prologue ──
    // At entry (LC_UNIXTHREAD), stack has: [argc][argv...][NULL][envp...][NULL]
    // No x0 to save (no rtld_fini on macOS UNIXTHREAD)
    code.extend_from_slice(&encode_sub_sp_imm(96));
    code.extend_from_slice(&encode_stp_offset(29, 30, 0));
    code.extend_from_slice(&encode_stp_offset(27, 28, 16));
    code.extend_from_slice(&encode_stp_offset(25, 26, 32));
    code.extend_from_slice(&encode_stp_offset(23, 24, 48));
    code.extend_from_slice(&encode_stp_offset(21, 22, 64));
    code.extend_from_slice(&encode_stp_offset(19, 20, 80));

    // Walk stack: sp+96 = argc, sp+104 = argv[0], ...
    let frame_size: u32 = 96;
    // LDR x24, [sp, #frame_size] (argc)
    code.extend_from_slice(&encode_ldr_imm(24, 31, frame_size));
    // x20 = &argv[0] = sp + frame_size + 8
    code.extend_from_slice(&encode_add_imm(20, 31, frame_size + 8));
    // x20 = envp = argv + (argc+1)*8
    // ADD x24, x24, #1  (argc+1)
    code.extend_from_slice(&encode_add_imm(24, 24, 1));
    // x24 = (argc+1) << 3 ... need LSL #3
    // Use ADD x20, x20, x24, LSL #3:
    // Encoding: ADD x20, x20, x24, LSL #3 = 0x8B180E94
    let add_lsl3 = 0x8B00_0000u32 | (24 << 16) | (0b011u32 << 10) | (20 << 5) | 20; // shift=LSL, amount=3
    code.extend_from_slice(&add_lsl3.to_le_bytes());

    // x19 = data_va (via ADR or ADRP+ADD)
    let adr_data_pos = code.len();
    code.extend_from_slice(&encode_nop()); // placeholder slot 1
    code.extend_from_slice(&encode_nop()); // placeholder slot 2

    // Now x20 = envp pointer array. Same scan as exec init.
    let needle_a_pos = code.len();
    code.extend_from_slice(&encode_nop());
    let needle_b_pos = code.len();
    code.extend_from_slice(&encode_nop());

    // Scan loop
    let scan_loop = code.len();
    // LDR x21, [x20], #8  (post-index: 0xF8408680 | (Rn<<5) | Rt)
    code.extend_from_slice(&(0xF840_8680u32 | (20 << 5) | 21).to_le_bytes());
    let cbz_nf = code.len();
    code.extend_from_slice(&encode_cbz(21, 0));

    code.extend_from_slice(&encode_ldr_reg(27, 21));
    code.extend_from_slice(&encode_cmp_reg(27, 25));
    let bne1 = code.len();
    code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]);

    code.extend_from_slice(&encode_ldr_w_imm(27, 21, 8));
    code.extend_from_slice(&encode_cmp_w_reg(27, 26));
    let bne2 = code.len();
    code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]);

    code.extend_from_slice(&encode_ldrb_imm(27, 21, 12));
    code.extend_from_slice(&encode_cmp_w_imm(27, b'=' as u32));
    let bne3 = code.len();
    code.extend_from_slice(&[0x61, 0x00, 0x00, 0x54]);

    // Parse decimal
    code.extend_from_slice(&encode_add_imm(22, 21, 13));
    code.extend_from_slice(&encode_movz(24, 0));
    code.extend_from_slice(&encode_movz(25, 10));
    let pl = code.len();
    code.extend_from_slice(&encode_ldrb_post_inc(27, 22));
    code.extend_from_slice(&encode_sub_w_imm(27, 27, b'0' as u32));
    code.extend_from_slice(&encode_cmp_w_imm(27, 10));
    let bhs = code.len();
    code.extend_from_slice(&[0x82, 0x00, 0x00, 0x54]);
    code.extend_from_slice(&encode_mul(24, 24, 25));
    code.extend_from_slice(&encode_add_uxtw(24, 24, 27));
    let bp = code.len();
    code.extend_from_slice(&encode_b(((pl as i64 - bp as i64) / 4) as i32));
    let pd = code.len();
    patch_b_cond(&mut code, bhs, (pd as i64 - bhs as i64) as i32, 0x2);
    let b_shm = code.len();
    code.extend_from_slice(&encode_b(0));

    // .next_env
    let ne = code.len();
    patch_b_cond(&mut code, bne1, (ne as i64 - bne1 as i64) as i32, 0x1);
    patch_b_cond(&mut code, bne2, (ne as i64 - bne2 as i64) as i32, 0x1);
    patch_b_cond(&mut code, bne3, (ne as i64 - bne3 as i64) as i32, 0x1);
    let sl = code.len();
    code.extend_from_slice(&encode_b(((scan_loop as i64 - sl as i64) / 4) as i32));

    // .not_found
    let nf = code.len();
    let cd = (nf as i64 - cbz_nf as i64) as i32;
    code[cbz_nf..cbz_nf + 4].copy_from_slice(&encode_cbz(21, cd));
    let b_epi = code.len();
    code.extend_from_slice(&encode_b(0));

    // .shm_attach
    let sa = code.len();
    let sd = (sa as i64 - b_shm as i64) / 4;
    code[b_shm..b_shm + 4].copy_from_slice(&encode_b(sd as i32));

    let bcs_skip;
    if use_got_shmat {
        code.extend_from_slice(&encode_ldr_imm(9, 19, 16));
        code.extend_from_slice(&encode_mov_reg(0, 24));
        code.extend_from_slice(&encode_movz(1, 0));
        code.extend_from_slice(&encode_movz(2, 0));
        code.extend_from_slice(&encode_blr(9));
        code.extend_from_slice(&encode_add_imm(9, 0, 1));
        code.extend_from_slice(&encode_cmp_imm0(9));
        bcs_skip = code.len();
        code.extend_from_slice(&[0x00, 0x00, 0x00, 0x54]); // B.EQ
    } else {
        code.extend_from_slice(&encode_mov_reg(0, 24));
        code.extend_from_slice(&encode_movz(1, 0));
        code.extend_from_slice(&encode_movz(2, 0));
        emit_macos_syscall(&mut code, MACOS_SYS_SHMAT);
        bcs_skip = code.len();
        code.extend_from_slice(&[0x02, 0x00, 0x00, 0x54]); // B.CS
    }
    code.extend_from_slice(&encode_str_reg(0, 19));
    code.extend_from_slice(&encode_strh_zero(19, PREV_LOC_OFFSET as u32));

    if enable_forkserver {
        emit_macos_forkserver_aarch64(&mut code);
    }

    // Epilogue
    let epi = code.len();
    let ed = (epi as i64 - b_epi as i64) / 4;
    code[b_epi..b_epi + 4].copy_from_slice(&encode_b(ed as i32));
    patch_b_cond(
        &mut code,
        bcs_skip,
        (epi as i64 - bcs_skip as i64) as i32,
        if use_got_shmat { 0x0 } else { 0x2 },
    );

    code.extend_from_slice(&encode_ldp_offset(19, 20, 80));
    code.extend_from_slice(&encode_ldp_offset(21, 22, 64));
    code.extend_from_slice(&encode_ldp_offset(23, 24, 48));
    code.extend_from_slice(&encode_ldp_offset(25, 26, 32));
    code.extend_from_slice(&encode_ldp_offset(27, 28, 16));
    code.extend_from_slice(&encode_ldp_offset(29, 30, 0));
    code.extend_from_slice(&encode_add_sp_imm(96));

    let bv = init_va + code.len() as u64;
    let delta = original_entry as i64 - bv as i64;
    if delta.abs() < (128 << 20) {
        code.extend_from_slice(&encode_b((delta / 4) as i32));
    } else {
        let av = init_va + code.len() as u64;
        let ep = (original_entry as i64) & !0xFFF;
        let ap = (av as i64) & !0xFFF;
        let pd = ep - ap;
        let immhi = ((pd >> 14) & 0x7FFFF) as u32;
        let immlo = ((pd >> 12) & 0x3) as u32;
        code.extend_from_slice(&(0x9000_0010u32 | (immlo << 29) | (immhi << 5)).to_le_bytes());
        code.extend_from_slice(&encode_add_imm(16, 16, (original_entry & 0xFFF) as u32));
        code.extend_from_slice(&encode_br(16));
    }

    // Literal pool
    while code.len() % 8 != 0 {
        code.extend_from_slice(&encode_nop());
    }
    let na_lit = code.len();
    code.extend_from_slice(&u64::from_le_bytes(*b"__AFL_SH").to_le_bytes());
    let nb_lit = code.len();
    code.extend_from_slice(&(u32::from_le_bytes(*b"M_ID") as u64).to_le_bytes());

    let adr_data_pc = init_va + adr_data_pos as u64;
    patch_adr_auto(&mut code, adr_data_pos, adr_data_pc, data_va, 19);
    patch_ldr_literal(&mut code, needle_a_pos, init_va, na_lit, 25);
    patch_ldr_literal(&mut code, needle_b_pos, init_va, nb_lit, 26);

    Ok(InitCode {
        va: init_va,
        code,
        entry_va,
    })
}

/// Emit macOS ARM64 forkserver loop.
///
/// Uses SYS_fork (not clone) and macOS syscall convention (x16 + SVC #0x80).
/// macOS fork returns: x0 = pid, x1 = 0 in parent, x1 = 1 in child.
fn emit_macos_forkserver_aarch64(code: &mut Vec<u8>) {
    use arm64::*;

    code.extend_from_slice(&encode_sub_sp_imm(16));

    // Hello message
    code.extend_from_slice(&encode_movz(27, 0));
    code.extend_from_slice(&encode_str_sp_imm(27, 0));

    // write(FORKSRV_FD+1, &hello, 4)
    emit_mov32(code, 0, FORKSRV_FD + 1);
    code.extend_from_slice(&encode_add_imm(1, 31, 0));
    code.extend_from_slice(&encode_movz(2, 4));
    emit_macos_syscall(code, MACOS_SYS_WRITE);

    // B.CS .no_forkserver (carry set = error → no forkserver FDs)
    let bcs_no_fork = code.len();
    code.extend_from_slice(&[0x02, 0x00, 0x00, 0x54]); // placeholder

    // .fork_loop:
    let fork_loop = code.len();

    // read(FORKSRV_FD, &cmd, 4)
    emit_mov32(code, 0, FORKSRV_FD);
    code.extend_from_slice(&encode_add_imm(1, 31, 0));
    code.extend_from_slice(&encode_movz(2, 4));
    emit_macos_syscall(code, MACOS_SYS_READ);

    // fork()
    emit_macos_syscall(code, MACOS_SYS_FORK);
    // x0 = pid, x1 = 0 in parent, 1 in child
    code.extend_from_slice(&encode_mov_reg(27, 0)); // save pid

    // CBNZ x1, .child
    let cbnz_child = code.len();
    code.extend_from_slice(&encode_cbnz(1, 0)); // placeholder

    // ── Parent ──
    code.extend_from_slice(&encode_str_sp_imm(27, 0));
    emit_mov32(code, 0, FORKSRV_FD + 1);
    code.extend_from_slice(&encode_add_imm(1, 31, 0));
    code.extend_from_slice(&encode_movz(2, 4));
    emit_macos_syscall(code, MACOS_SYS_WRITE);

    // wait4(child_pid, &status, 0, NULL)
    code.extend_from_slice(&encode_mov_reg(0, 27));
    code.extend_from_slice(&encode_add_imm(1, 31, 8));
    code.extend_from_slice(&encode_movz(2, 0));
    code.extend_from_slice(&encode_movz(3, 0));
    emit_macos_syscall(code, MACOS_SYS_WAIT4);

    // write(FORKSRV_FD+1, &status, 4)
    emit_mov32(code, 0, FORKSRV_FD + 1);
    code.extend_from_slice(&encode_add_imm(1, 31, 8));
    code.extend_from_slice(&encode_movz(2, 4));
    emit_macos_syscall(code, MACOS_SYS_WRITE);

    // B .fork_loop
    let bl = code.len();
    code.extend_from_slice(&encode_b(((fork_loop as i64 - bl as i64) / 4) as i32));

    // .child:
    let child = code.len();
    let cd = (child as i64 - cbnz_child as i64) as i32;
    code[cbnz_child..cbnz_child + 4].copy_from_slice(&encode_cbnz(1, cd));

    // Close forkserver FDs
    emit_mov32(code, 0, FORKSRV_FD);
    emit_macos_syscall(code, MACOS_SYS_CLOSE);
    emit_mov32(code, 0, FORKSRV_FD + 1);
    emit_macos_syscall(code, MACOS_SYS_CLOSE);

    // Deallocate forkserver stack
    code.extend_from_slice(&encode_add_sp_imm(16));

    // Skip over the no_forkserver cleanup — child already cleaned up above.
    // Without this branch, the child falls through and executes ADD sp, 16
    // a second time, corrupting the stack frame and causing SIGSEGV when
    // the epilogue restores callee-saved registers from wrong offsets.
    let b_skip_nofork = code.len();
    code.extend_from_slice(&encode_b(0)); // placeholder: B .after_cleanup

    // .no_forkserver: B.CS lands here (forkserver pipe write failed)
    let no_fork = code.len();
    code.extend_from_slice(&encode_add_sp_imm(16));

    // .after_cleanup: child branch target
    let after_cleanup = code.len();
    let skip_delta = ((after_cleanup as i64 - b_skip_nofork as i64) / 4) as i32;
    code[b_skip_nofork..b_skip_nofork + 4].copy_from_slice(&encode_b(skip_delta));

    // Patch B.CS to .no_forkserver
    patch_b_cond(
        code,
        bcs_no_fork,
        (no_fork as i64 - bcs_no_fork as i64) as i32,
        0x2,
    );

    // Both child and no_forkserver paths fall through here.
}

// ============================================================
// Persistent mode wrappers for macOS
// ============================================================

/// Relocate displaced x86_64 instructions from `orig_ip` to `new_ip`.
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

/// Generate macOS x86_64 persistent wrapper.
///
/// Same structure as the ELF variant but uses BSD syscall convention:
/// `mov eax, 0x2000000 | num; syscall` (CF set on error).
/// macOS fork: rdx=0 parent, rdx=1 child.
///
/// Data layout (160 bytes, same as ELF):
///   +0:   first_pass (u8)
///   +1:   child_stopped (u8)
///   +2:   forkserver_started (u8)
///   +3:   padding
///   +4:   counter (u32)
///   +8:   save_regs (16 * 8 = 128 bytes)
///   +136: save_ret_addr (8 bytes)
///   +144: save_rsp (8 bytes)
///   +152: padding (8 bytes)
pub fn generate_macho_persistent_wrapper_x86_64(
    params: &PersistentWrapperParams,
) -> Result<PersistentWrapper> {
    let wrapper_va = params.wrapper_va;
    let persistent_data_va = params.persistent_data_va;
    let data_va = params.data_va;
    let persistent_addr = params.persistent_addr;
    let displaced_bytes = params.displaced_bytes;
    let displaced_len = params.displaced_len;
    let persistent_count = params.persistent_count;
    let include_forkserver = params.include_forkserver;
    let mut code = Vec::with_capacity(if include_forkserver { 1024 } else { 512 });
    let return_va = persistent_addr + displaced_len as u64;

    // Offsets within persistent data area
    const OFF_FIRST_PASS: u64 = 0;
    const OFF_CHILD_STOPPED: u64 = 1;
    const OFF_FS_STARTED: u64 = 2;
    const OFF_COUNTER: u64 = 4;
    const OFF_SAVE_REGS: u64 = 8;
    const OFF_SAVE_RET: u64 = 136;
    const OFF_SAVE_RSP: u64 = 144;

    // RIP-relative displacement to a persistent data field
    let data_disp = |code_pos: u64, field_offset: u64| -> i32 {
        (persistent_data_va as i64 + field_offset as i64 - code_pos as i64) as i32
    };

    // GPR save opcodes: mov [rip+disp32], reg
    let gpr_save_opcodes: &[(u8, u8, u8)] = &[
        (0x48, 0x89, 0x05), // rax
        (0x48, 0x89, 0x0D), // rcx
        (0x48, 0x89, 0x15), // rdx
        (0x48, 0x89, 0x1D), // rbx
        (0x48, 0x89, 0x25), // rsp
        (0x48, 0x89, 0x2D), // rbp
        (0x48, 0x89, 0x35), // rsi
        (0x48, 0x89, 0x3D), // rdi
        (0x4C, 0x89, 0x05), // r8
        (0x4C, 0x89, 0x0D), // r9
        (0x4C, 0x89, 0x15), // r10
        (0x4C, 0x89, 0x1D), // r11
        (0x4C, 0x89, 0x25), // r12
        (0x4C, 0x89, 0x2D), // r13
        (0x4C, 0x89, 0x35), // r14
        (0x4C, 0x89, 0x3D), // r15
    ];

    // GPR load opcodes: mov reg, [rip+disp32]
    let gpr_load_opcodes_all: &[(u8, u8, u8)] = &[
        (0x48, 0x8B, 0x05), // rax
        (0x48, 0x8B, 0x0D), // rcx
        (0x48, 0x8B, 0x15), // rdx
        (0x48, 0x8B, 0x1D), // rbx
        (0x48, 0x8B, 0x25), // rsp
        (0x48, 0x8B, 0x2D), // rbp
        (0x48, 0x8B, 0x35), // rsi
        (0x48, 0x8B, 0x3D), // rdi
        (0x4C, 0x8B, 0x05), // r8
        (0x4C, 0x8B, 0x0D), // r9
        (0x4C, 0x8B, 0x15), // r10
        (0x4C, 0x8B, 0x1D), // r11
        (0x4C, 0x8B, 0x25), // r12
        (0x4C, 0x8B, 0x2D), // r13
        (0x4C, 0x8B, 0x35), // r14
        (0x4C, 0x8B, 0x3D), // r15
    ];

    // === Section 0: Deferred forkserver (optional) ===
    if include_forkserver {
        // Check if forkserver already started
        // cmp byte [rip+forkserver_started], 0
        let cmp_ip = wrapper_va + code.len() as u64 + 7;
        code.extend_from_slice(&[0x80, 0x3D]);
        code.extend_from_slice(&data_disp(cmp_ip, OFF_FS_STARTED).to_le_bytes());
        code.push(0x00);

        // jne .persistent_section
        let jne_persistent_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);

        // Set forkserver_started = 1
        let mov_fs_ip = wrapper_va + code.len() as u64 + 7;
        code.extend_from_slice(&[0xC6, 0x05]);
        code.extend_from_slice(&data_disp(mov_fs_ip, OFF_FS_STARTED).to_le_bytes());
        code.push(0x01);

        // Save all 16 GPRs
        for (i, &(rex, op, modrm)) in gpr_save_opcodes.iter().enumerate() {
            let instr_ip = wrapper_va + code.len() as u64 + 7;
            code.push(rex);
            code.push(op);
            code.push(modrm);
            code.extend_from_slice(
                &data_disp(instr_ip, OFF_SAVE_REGS + i as u64 * 8).to_le_bytes(),
            );
        }

        // sub rsp, 16 (scratch space)
        code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x10]);

        // Hello: write(FORKSRV_FD+1, &zero, 4)
        code.extend_from_slice(&[0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00]); // mov dword [rsp], 0
        code.extend_from_slice(&[0xB8]); // mov eax, SYS_write
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WRITE).to_le_bytes());
        code.extend_from_slice(&[0xBF]); // mov edi, FORKSRV_FD+1
        code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]); // lea rsi, [rsp]
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&4u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]); // syscall

        // jc .no_parent (CF set = error on macOS)
        let jc_no_parent_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x82, 0x00, 0x00, 0x00, 0x00]); // jc rel32

        // cmp eax, 4
        code.extend_from_slice(&[0x83, 0xF8, 0x04]);
        // jne .no_parent
        let jne_no_parent_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);

        // .fork_loop:
        let fork_loop = code.len();

        // read(FORKSRV_FD, [rsp], 4)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_READ).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&FORKSRV_FD.to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]);
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&4u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // cmp eax, 4 / jne .exit_parent
        code.extend_from_slice(&[0x83, 0xF8, 0x04]);
        let jne_exit_parent_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);

        // Check child_stopped
        let cmp_cs_ip = wrapper_va + code.len() as u64 + 7;
        code.extend_from_slice(&[0x80, 0x3D]);
        code.extend_from_slice(&data_disp(cmp_cs_ip, OFF_CHILD_STOPPED).to_le_bytes());
        code.push(0x00);

        // je .do_fork
        let je_do_fork_pos = code.len();
        code.extend_from_slice(&[0x74, 0x00]);

        // Resume stopped child: kill(r13d, SIGCONT=18)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_KILL).to_le_bytes());
        code.extend_from_slice(&[0x44, 0x89, 0xEF]); // mov edi, r13d
        code.extend_from_slice(&[0xBE]);
        code.extend_from_slice(&18u32.to_le_bytes()); // SIGCONT
        code.extend_from_slice(&[0x0F, 0x05]);

        // Write child PID to parent
        code.extend_from_slice(&[0x44, 0x89, 0x2C, 0x24]); // mov [rsp], r13d
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WRITE).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]);
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&4u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // jmp .do_waitpid
        let jmp_waitpid_pos = code.len();
        code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);

        // .do_fork:
        let do_fork = code.len();
        code[je_do_fork_pos + 1] = (do_fork - (je_do_fork_pos + 2)) as u8;

        // fork() — macOS: rdx=0 parent, rdx=1 child; rax=child_pid in parent
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_FORK).to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // jc .exit_parent (CF = fork failed)
        let jc_exit_parent_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x82, 0x00, 0x00, 0x00, 0x00]);

        // macOS: test edx, edx / jnz .child_path (rdx=1 for child)
        code.extend_from_slice(&[0x85, 0xD2]); // test edx, edx
        let jnz_child_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);

        // Parent: save child PID, write to parent
        code.extend_from_slice(&[0x41, 0x89, 0xC5]); // mov r13d, eax
        code.extend_from_slice(&[0x89, 0x04, 0x24]); // mov [rsp], eax
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WRITE).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]);
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&4u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // .do_waitpid:
        let do_waitpid = code.len();
        let rel = (do_waitpid as i32) - (jmp_waitpid_pos as i32 + 5);
        code[jmp_waitpid_pos + 1..jmp_waitpid_pos + 5].copy_from_slice(&rel.to_le_bytes());

        // wait4(r13d, [rsp], WUNTRACED=2, NULL)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WAIT4).to_le_bytes());
        code.extend_from_slice(&[0x44, 0x89, 0xEF]); // mov edi, r13d
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]); // lea rsi, [rsp]
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&2u32.to_le_bytes()); // WUNTRACED
        code.extend_from_slice(&[0x4D, 0x31, 0xD2]); // xor r10, r10
        code.extend_from_slice(&[0x0F, 0x05]);

        // Check WIFSTOPPED: (status & 0xFF) == 0x7F
        code.extend_from_slice(&[0x8B, 0x04, 0x24]); // mov eax, [rsp]
        code.extend_from_slice(&[0x25]);
        code.extend_from_slice(&0xFFu32.to_le_bytes());
        code.extend_from_slice(&[0x3D]);
        code.extend_from_slice(&0x7Fu32.to_le_bytes());
        let jne_not_stopped_pos = code.len();
        code.extend_from_slice(&[0x75, 0x00]);

        // WIFSTOPPED: set child_stopped=1, synthetic status=0
        let mov_cs_ip = wrapper_va + code.len() as u64 + 7;
        code.extend_from_slice(&[0xC6, 0x05]);
        code.extend_from_slice(&data_disp(mov_cs_ip, OFF_CHILD_STOPPED).to_le_bytes());
        code.push(0x01);
        code.extend_from_slice(&[0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00]);
        let jmp_report_pos = code.len();
        code.extend_from_slice(&[0xEB, 0x00]);

        // .not_stopped:
        let not_stopped = code.len();
        code[jne_not_stopped_pos + 1] = (not_stopped - (jne_not_stopped_pos + 2)) as u8;

        let mov_cs2_ip = wrapper_va + code.len() as u64 + 7;
        code.extend_from_slice(&[0xC6, 0x05]);
        code.extend_from_slice(&data_disp(mov_cs2_ip, OFF_CHILD_STOPPED).to_le_bytes());
        code.push(0x00);

        // .report_status:
        let report_status = code.len();
        code[jmp_report_pos + 1] = (report_status - (jmp_report_pos + 2)) as u8;

        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_WRITE).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]);
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&4u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // jmp .fork_loop
        let jmp_forkloop_rel = (fork_loop as i32) - (code.len() as i32 + 5);
        code.push(0xE9);
        code.extend_from_slice(&jmp_forkloop_rel.to_le_bytes());

        // .child_path:
        let child_path = code.len();
        let rel = (child_path as i32) - (jnz_child_pos as i32 + 6);
        code[jnz_child_pos + 2..jnz_child_pos + 6].copy_from_slice(&rel.to_le_bytes());

        // Close forkserver FDs
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_CLOSE).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&FORKSRV_FD.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_CLOSE).to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&(FORKSRV_FD + 1).to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // .no_parent:
        let no_parent = code.len();
        let rel = (no_parent as i32) - (jc_no_parent_pos as i32 + 6);
        code[jc_no_parent_pos + 2..jc_no_parent_pos + 6].copy_from_slice(&rel.to_le_bytes());
        let rel = (no_parent as i32) - (jne_no_parent_pos as i32 + 6);
        code[jne_no_parent_pos + 2..jne_no_parent_pos + 6].copy_from_slice(&rel.to_le_bytes());

        // add rsp, 16
        code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x10]);

        // Restore all GPRs
        for (i, &(rex, op, modrm)) in gpr_load_opcodes_all.iter().enumerate() {
            let instr_ip = wrapper_va + code.len() as u64 + 7;
            code.push(rex);
            code.push(op);
            code.push(modrm);
            code.extend_from_slice(
                &data_disp(instr_ip, OFF_SAVE_REGS + i as u64 * 8).to_le_bytes(),
            );
        }

        // Jump to persistent section
        let jmp_to_persistent_pos = code.len();
        code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);

        // .exit_parent:
        let exit_parent = code.len();
        let rel = (exit_parent as i32) - (jne_exit_parent_pos as i32 + 6);
        code[jne_exit_parent_pos + 2..jne_exit_parent_pos + 6].copy_from_slice(&rel.to_le_bytes());
        let rel = (exit_parent as i32) - (jc_exit_parent_pos as i32 + 6);
        code[jc_exit_parent_pos + 2..jc_exit_parent_pos + 6].copy_from_slice(&rel.to_le_bytes());

        // exit(0) — macOS
        code.extend_from_slice(&[0x31, 0xFF]); // xor edi, edi
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_EXIT).to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // .persistent_section:
        let persistent_section = code.len();
        let rel = (persistent_section as i32) - (jne_persistent_pos as i32 + 6);
        code[jne_persistent_pos + 2..jne_persistent_pos + 6].copy_from_slice(&rel.to_le_bytes());
        let rel = (persistent_section as i32) - (jmp_to_persistent_pos as i32 + 5);
        code[jmp_to_persistent_pos + 1..jmp_to_persistent_pos + 5]
            .copy_from_slice(&rel.to_le_bytes());
    }

    // === Section A: Entry — check first_pass ===
    let cmp_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x80, 0x3D]);
    code.extend_from_slice(&data_disp(cmp_ip, OFF_FIRST_PASS).to_le_bytes());
    code.push(0x00);

    // je .setup_and_run
    let je_setup_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);

    // First pass: set first_pass=0
    let mov_fp_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0xC6, 0x05]);
    code.extend_from_slice(&data_disp(mov_fp_ip, OFF_FIRST_PASS).to_le_bytes());
    code.push(0x00);

    // Save all 16 GPRs
    for (i, &(rex, op, modrm)) in gpr_save_opcodes.iter().enumerate() {
        let instr_ip = wrapper_va + code.len() as u64 + 7;
        code.push(rex);
        code.push(op);
        code.push(modrm);
        code.extend_from_slice(&data_disp(instr_ip, OFF_SAVE_REGS + i as u64 * 8).to_le_bytes());
    }

    // Save return address: mov rax, [rsp]; mov [rip+save_ret], rax
    code.extend_from_slice(&[0x48, 0x8B, 0x04, 0x24]);
    let mov_ret_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x89, 0x05]);
    code.extend_from_slice(&data_disp(mov_ret_ip, OFF_SAVE_RET).to_le_bytes());

    // Save RSP (caller RSP = rsp + 8)
    code.extend_from_slice(&[0x48, 0x8D, 0x44, 0x24, 0x08]); // lea rax, [rsp+8]
    let mov_rsp_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x89, 0x05]);
    code.extend_from_slice(&data_disp(mov_rsp_ip, OFF_SAVE_RSP).to_le_bytes());

    // Set counter
    let mov_ctr_ip = wrapper_va + code.len() as u64 + 10;
    code.extend_from_slice(&[0xC7, 0x05]);
    code.extend_from_slice(&data_disp(mov_ctr_ip, OFF_COUNTER).to_le_bytes());
    code.extend_from_slice(&persistent_count.to_le_bytes());

    // === Section B: .setup_and_run ===
    let setup_and_run = code.len();
    let rel = (setup_and_run as i32) - (je_setup_pos as i32 + 6);
    code[je_setup_pos + 2..je_setup_pos + 6].copy_from_slice(&rel.to_le_bytes());

    // Patch return address: lea rax, [rip+iteration_boundary]; mov [rsp], rax
    let lea_iter_pos = code.len();
    code.extend_from_slice(&[0x48, 0x8D, 0x05, 0x00, 0x00, 0x00, 0x00]);
    code.extend_from_slice(&[0x48, 0x89, 0x04, 0x24]);

    // Restore rax
    let mov_rax_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x8B, 0x05]);
    code.extend_from_slice(&data_disp(mov_rax_ip, OFF_SAVE_REGS).to_le_bytes());

    // Relocated displaced instructions
    let displaced_va = wrapper_va + code.len() as u64;
    let relocated = relocate_instructions(displaced_bytes, persistent_addr, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate instructions for persistent wrapper at 0x{:x}",
                persistent_addr,
            )
        })?;
    code.extend_from_slice(&relocated);

    // jmp return_va
    let jmp_ip = wrapper_va + code.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    // === Section C: iteration_boundary ===
    let iteration_boundary = code.len();

    // Fix up lea rax, [rip+iteration_boundary]
    let lea_rip = wrapper_va + lea_iter_pos as u64 + 7;
    let iter_va = wrapper_va + iteration_boundary as u64;
    let lea_disp = (iter_va as i64 - lea_rip as i64) as i32;
    code[lea_iter_pos + 3..lea_iter_pos + 7].copy_from_slice(&lea_disp.to_le_bytes());

    // dec counter
    let dec_ip = wrapper_va + code.len() as u64 + 6;
    code.extend_from_slice(&[0xFF, 0x0D]);
    code.extend_from_slice(&data_disp(dec_ip, OFF_COUNTER).to_le_bytes());

    // jz .exit_child
    let jz_exit_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);

    // getpid() — macOS
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_GETPID).to_le_bytes());
    code.extend_from_slice(&[0x0F, 0x05]);

    // kill(eax, SIGSTOP=17) — macOS SIGSTOP is 17
    code.extend_from_slice(&[0x89, 0xC7]); // mov edi, eax
    code.extend_from_slice(&[0xBE]);
    code.extend_from_slice(&17u32.to_le_bytes()); // SIGSTOP=17 on macOS (not 19 like Linux)
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_KILL).to_le_bytes());
    code.extend_from_slice(&[0x0F, 0x05]);

    // After SIGCONT: clear prev_loc
    let prev_loc_va = data_va + PREV_LOC_OFFSET;
    let mov_pl_ip = wrapper_va + code.len() as u64 + 9;
    let pl_disp = (prev_loc_va as i64 - mov_pl_ip as i64) as i32;
    code.extend_from_slice(&[0x66, 0xC7, 0x05]);
    code.extend_from_slice(&pl_disp.to_le_bytes());
    code.extend_from_slice(&[0x00, 0x00]);

    // Restore all GPRs except rsp
    let gpr_load_opcodes: &[(u8, u8, u8)] = &[
        (0x48, 0x8B, 0x05), // rax
        (0x48, 0x8B, 0x0D), // rcx
        (0x48, 0x8B, 0x15), // rdx
        (0x48, 0x8B, 0x1D), // rbx
        // skip rsp
        (0x48, 0x8B, 0x2D), // rbp
        (0x48, 0x8B, 0x35), // rsi
        (0x48, 0x8B, 0x3D), // rdi
        (0x4C, 0x8B, 0x05), // r8
        (0x4C, 0x8B, 0x0D), // r9
        (0x4C, 0x8B, 0x15), // r10
        (0x4C, 0x8B, 0x1D), // r11
        (0x4C, 0x8B, 0x25), // r12
        (0x4C, 0x8B, 0x2D), // r13
        (0x4C, 0x8B, 0x35), // r14
        (0x4C, 0x8B, 0x3D), // r15
    ];
    let gpr_indices: &[usize] = &[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    for (&(rex, op, modrm), &save_idx) in gpr_load_opcodes.iter().zip(gpr_indices.iter()) {
        let instr_ip = wrapper_va + code.len() as u64 + 7;
        code.push(rex);
        code.push(op);
        code.push(modrm);
        code.extend_from_slice(
            &data_disp(instr_ip, OFF_SAVE_REGS + save_idx as u64 * 8).to_le_bytes(),
        );
    }

    // Restore RSP
    let mov_rsp_restore_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x8B, 0x25]);
    code.extend_from_slice(&data_disp(mov_rsp_restore_ip, OFF_SAVE_RSP).to_le_bytes());

    // Push saved return address and jump back to setup_and_run
    code.push(0x50); // push rax
    let mov_ret2_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x8B, 0x05]);
    code.extend_from_slice(&data_disp(mov_ret2_ip, OFF_SAVE_RET).to_le_bytes());
    code.extend_from_slice(&[0x48, 0x87, 0x04, 0x24]); // xchg rax, [rsp]

    // jmp .setup_and_run
    let jmp_setup_ip = wrapper_va + code.len() as u64 + 5;
    let setup_va = wrapper_va + setup_and_run as u64;
    let jmp_setup_rel = (setup_va as i64 - jmp_setup_ip as i64) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp_setup_rel.to_le_bytes());

    // === Section D: .exit_child ===
    let exit_child = code.len();
    let rel = (exit_child as i32) - (jz_exit_pos as i32 + 6);
    code[jz_exit_pos + 2..jz_exit_pos + 6].copy_from_slice(&rel.to_le_bytes());

    // exit(0) — macOS
    code.extend_from_slice(&[0x31, 0xFF]); // xor edi, edi
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&(BSD_CLASS | MACOS_SYS_EXIT).to_le_bytes());
    code.extend_from_slice(&[0x0F, 0x05]);

    Ok(PersistentWrapper { code })
}

/// Generate macOS AArch64 persistent wrapper.
///
/// Uses macOS ARM64 syscall convention: x16 = syscall number, SVC #0x80.
/// macOS fork on ARM64: x0 = pid, x1 = 0 in parent, x1 = 1 in child.
///
/// Data layout (160 bytes, same as x86_64):
///   +0:   first_pass (u8)
///   +1:   child_stopped (u8)
///   +2:   forkserver_started (u8)
///   +4:   counter (u32)
///   +8:   (ARM64 uses the stack for register saves via STP/LDP; the persistent
///         data area stores only control fields plus saved SP and LR for loop-back.)
///
/// Data layout (160 bytes):
///   +0:   first_pass (u8)
///   +1:   child_stopped (u8)
///   +2:   forkserver_started (u8)
///   +4:   counter (u32)
///   +8:   save_sp (8 bytes)
///   +16:  save_lr (8 bytes)
///   +24:  save_x0 (8 bytes)  -- we save x0 for restoring after displaced instrs
///   +32:  reserved (128 bytes)
pub fn generate_macho_persistent_wrapper_aarch64(
    params: &PersistentWrapperParams,
) -> Result<PersistentWrapper> {
    use arm64::*;

    let wrapper_va = params.wrapper_va;
    let persistent_data_va = params.persistent_data_va;
    let data_va = params.data_va;
    let persistent_addr = params.persistent_addr;
    let displaced_bytes = params.displaced_bytes;
    let displaced_len = params.displaced_len;
    let persistent_count = params.persistent_count;
    let include_forkserver = params.include_forkserver;
    let mut code = Vec::with_capacity(if include_forkserver { 2048 } else { 1024 });
    let return_va = persistent_addr + displaced_len as u64;

    // Offsets within persistent data area
    const OFF_FIRST_PASS: u32 = 0;
    const OFF_CHILD_STOPPED: u32 = 1;
    const OFF_FS_STARTED: u32 = 2;
    const OFF_COUNTER: u32 = 4;
    const OFF_SAVE_SP: u32 = 8;
    const OFF_SAVE_LR: u32 = 16;
    const OFF_SAVE_X0: u32 = 24;

    // Literal pool entries — we use ADR to get pointers and LDR literal for data values.
    // For ARM64, persistent_data_va and other VAs are loaded via literal pool at the end.
    // We'll collect literal pool entries and patch them at the end.
    struct LitPoolEntry {
        instr_pos: usize, // position of the LDR/ADR instruction
        value: u64,
    }
    let mut lit_pool: Vec<LitPoolEntry> = Vec::new();

    // Helper: emit ADR xN, <label> (placeholder, patched later)
    let emit_adr_placeholder = |code: &mut Vec<u8>, reg: u32| -> usize {
        let pos = code.len();
        code.extend_from_slice(&encode_adr(reg, 0));
        pos
    };

    // === Section 0: Deferred forkserver (optional) ===
    if include_forkserver {
        // Load persistent_data_va into x9
        let adr_data = emit_adr_placeholder(&mut code, 9);
        lit_pool.push(LitPoolEntry {
            instr_pos: adr_data,
            value: persistent_data_va,
        });

        // Check forkserver_started: LDRB w10, [x9, #OFF_FS_STARTED]
        code.extend_from_slice(&encode_ldrb_imm(10, 9, OFF_FS_STARTED));
        // CBNZ w10, .persistent_section
        let cbnz_persistent_pos = code.len();
        code.extend_from_slice(&encode_cbnz(10, 0)); // placeholder

        // Set forkserver_started = 1
        // MOV w10, #1; STRB w10, [x9, #OFF_FS_STARTED]
        code.extend_from_slice(&encode_movz(10, 1));
        code.extend_from_slice(&encode_strb_imm(10, 9, OFF_FS_STARTED));

        // Save all callee-saved regs to stack
        // STP x19, x20, [sp, #-160]! -- save 10 pairs on stack
        code.extend_from_slice(&encode_sub_sp_imm(160));
        for i in 0..5u32 {
            code.extend_from_slice(&encode_stp_offset(19 + i * 2, 20 + i * 2, i * 16));
        }
        code.extend_from_slice(&encode_stp_offset(29, 30, 80)); // fp, lr

        // sub sp, sp, 16 (scratch space)
        code.extend_from_slice(&encode_sub_sp_imm(16));

        // Hello: write(FORKSRV_FD+1, &zero, 4)
        code.extend_from_slice(&encode_movz(10, 0));
        code.extend_from_slice(&encode_str_sp_imm(10, 0)); // [sp] = 0
        emit_mov32(&mut code, 0, FORKSRV_FD + 1);
        code.extend_from_slice(&encode_add_imm(1, 31, 0)); // x1 = sp
        code.extend_from_slice(&encode_movz(2, 4));
        emit_macos_syscall(&mut code, MACOS_SYS_WRITE);

        // B.CS .no_parent
        let bcs_no_parent_pos = code.len();
        code.extend_from_slice(&[0x02, 0x00, 0x00, 0x54]); // placeholder

        // .fork_loop:
        let fork_loop = code.len();

        // read(FORKSRV_FD, &cmd, 4)
        emit_mov32(&mut code, 0, FORKSRV_FD);
        code.extend_from_slice(&encode_add_imm(1, 31, 0));
        code.extend_from_slice(&encode_movz(2, 4));
        emit_macos_syscall(&mut code, MACOS_SYS_READ);

        // Check child_stopped
        // ADR x9, persistent_data (need fresh ADR since we're in a loop)
        let adr_data2 = emit_adr_placeholder(&mut code, 9);
        lit_pool.push(LitPoolEntry {
            instr_pos: adr_data2,
            value: persistent_data_va,
        });
        code.extend_from_slice(&encode_ldrb_imm(10, 9, OFF_CHILD_STOPPED));
        // CBZ w10, .do_fork
        let cbz_do_fork_pos = code.len();
        code.extend_from_slice(&encode_cbz(10, 0)); // placeholder

        // Resume stopped child: kill(x27, SIGCONT=18)
        code.extend_from_slice(&encode_mov_reg(0, 27)); // pid
        code.extend_from_slice(&encode_movz(1, 18)); // SIGCONT
        emit_macos_syscall(&mut code, MACOS_SYS_KILL);

        // Write child PID
        code.extend_from_slice(&encode_str_sp_imm(27, 0));
        emit_mov32(&mut code, 0, FORKSRV_FD + 1);
        code.extend_from_slice(&encode_add_imm(1, 31, 0));
        code.extend_from_slice(&encode_movz(2, 4));
        emit_macos_syscall(&mut code, MACOS_SYS_WRITE);

        // B .do_waitpid
        let b_waitpid_pos = code.len();
        code.extend_from_slice(&encode_b(0)); // placeholder

        // .do_fork:
        let do_fork = code.len();
        let cd = (do_fork as i64 - cbz_do_fork_pos as i64) as i32;
        code[cbz_do_fork_pos..cbz_do_fork_pos + 4].copy_from_slice(&encode_cbz(10, cd));

        // fork()
        emit_macos_syscall(&mut code, MACOS_SYS_FORK);
        // x0 = pid, x1 = 0 parent, 1 child
        code.extend_from_slice(&encode_mov_reg(27, 0)); // save pid

        // CBNZ x1, .child
        let cbnz_child_pos = code.len();
        code.extend_from_slice(&encode_cbnz(1, 0)); // placeholder

        // Parent: write child PID
        code.extend_from_slice(&encode_str_sp_imm(27, 0));
        emit_mov32(&mut code, 0, FORKSRV_FD + 1);
        code.extend_from_slice(&encode_add_imm(1, 31, 0));
        code.extend_from_slice(&encode_movz(2, 4));
        emit_macos_syscall(&mut code, MACOS_SYS_WRITE);

        // .do_waitpid:
        let do_waitpid = code.len();
        let brel = ((do_waitpid as i64 - b_waitpid_pos as i64) / 4) as i32;
        code[b_waitpid_pos..b_waitpid_pos + 4].copy_from_slice(&encode_b(brel));

        // wait4(x27, &status, WUNTRACED=2, NULL)
        code.extend_from_slice(&encode_mov_reg(0, 27));
        code.extend_from_slice(&encode_add_imm(1, 31, 4)); // &status at [sp+4]
        code.extend_from_slice(&encode_movz(2, 2)); // WUNTRACED
        code.extend_from_slice(&encode_movz(3, 0));
        emit_macos_syscall(&mut code, MACOS_SYS_WAIT4);

        // Check WIFSTOPPED: (status & 0xFF) == 0x7F
        code.extend_from_slice(&encode_ldr_w_imm(10, 31, 4)); // w10 = [sp+4]
        code.extend_from_slice(&[0x4A, 0x1D, 0x00, 0x12]); // AND w10, w10, #0xFF
        code.extend_from_slice(&encode_cmp_w_imm(10, 0x7F));
        // B.NE .not_stopped
        let bne_not_stopped_pos = code.len();
        code.extend_from_slice(&[0x01, 0x00, 0x00, 0x54]); // placeholder B.NE

        // WIFSTOPPED: set child_stopped=1, synthetic status=0
        let adr_data3 = emit_adr_placeholder(&mut code, 9);
        lit_pool.push(LitPoolEntry {
            instr_pos: adr_data3,
            value: persistent_data_va,
        });
        code.extend_from_slice(&encode_movz(10, 1));
        code.extend_from_slice(&encode_strb_imm(10, 9, OFF_CHILD_STOPPED));
        code.extend_from_slice(&encode_movz(10, 0));
        code.extend_from_slice(&encode_str_sp_imm(10, 4)); // status=0
        // B .report_status
        let b_report_pos = code.len();
        code.extend_from_slice(&encode_b(0)); // placeholder

        // .not_stopped:
        let not_stopped = code.len();
        patch_b_cond(
            &mut code,
            bne_not_stopped_pos,
            (not_stopped as i64 - bne_not_stopped_pos as i64) as i32,
            0x1,
        );

        let adr_data4 = emit_adr_placeholder(&mut code, 9);
        lit_pool.push(LitPoolEntry {
            instr_pos: adr_data4,
            value: persistent_data_va,
        });
        code.extend_from_slice(&encode_movz(10, 0));
        code.extend_from_slice(&encode_strb_imm(10, 9, OFF_CHILD_STOPPED));

        // .report_status:
        let report_status = code.len();
        let brel = ((report_status as i64 - b_report_pos as i64) / 4) as i32;
        code[b_report_pos..b_report_pos + 4].copy_from_slice(&encode_b(brel));

        // write(FORKSRV_FD+1, &status, 4) -- status is at [sp+4]
        emit_mov32(&mut code, 0, FORKSRV_FD + 1);
        code.extend_from_slice(&encode_add_imm(1, 31, 4));
        code.extend_from_slice(&encode_movz(2, 4));
        emit_macos_syscall(&mut code, MACOS_SYS_WRITE);

        // B .fork_loop
        let bl2 = code.len();
        code.extend_from_slice(&encode_b(((fork_loop as i64 - bl2 as i64) / 4) as i32));

        // .child:
        let child = code.len();
        let cd = (child as i64 - cbnz_child_pos as i64) as i32;
        code[cbnz_child_pos..cbnz_child_pos + 4].copy_from_slice(&encode_cbnz(1, cd));

        // Close forkserver FDs
        emit_mov32(&mut code, 0, FORKSRV_FD);
        emit_macos_syscall(&mut code, MACOS_SYS_CLOSE);
        emit_mov32(&mut code, 0, FORKSRV_FD + 1);
        emit_macos_syscall(&mut code, MACOS_SYS_CLOSE);

        // .no_parent:
        let no_parent = code.len();
        patch_b_cond(
            &mut code,
            bcs_no_parent_pos,
            (no_parent as i64 - bcs_no_parent_pos as i64) as i32,
            0x2,
        );

        // Deallocate scratch
        code.extend_from_slice(&encode_add_sp_imm(16));

        // Restore callee-saved regs
        for i in 0..5u32 {
            code.extend_from_slice(&encode_ldp_offset(19 + i * 2, 20 + i * 2, i * 16));
        }
        code.extend_from_slice(&encode_ldp_offset(29, 30, 80));
        code.extend_from_slice(&encode_add_sp_imm(160));

        // B .persistent_section (placeholder)
        let b_persistent_pos = code.len();
        code.extend_from_slice(&encode_b(0));

        // .exit_parent: exit(0)
        code.extend_from_slice(&encode_movz(0, 0));
        emit_macos_syscall(&mut code, MACOS_SYS_EXIT);

        // .persistent_section:
        let persistent_section = code.len();
        let cd = (persistent_section as i64 - cbnz_persistent_pos as i64) as i32;
        code[cbnz_persistent_pos..cbnz_persistent_pos + 4].copy_from_slice(&encode_cbnz(10, cd));
        let brel = ((persistent_section as i64 - b_persistent_pos as i64) / 4) as i32;
        code[b_persistent_pos..b_persistent_pos + 4].copy_from_slice(&encode_b(brel));
    }

    // === Section A: Entry — check first_pass ===
    // ADR x9, persistent_data
    let adr_data_a = emit_adr_placeholder(&mut code, 9);
    lit_pool.push(LitPoolEntry {
        instr_pos: adr_data_a,
        value: persistent_data_va,
    });

    // LDRB w10, [x9, #OFF_FIRST_PASS]
    code.extend_from_slice(&encode_ldrb_imm(10, 9, OFF_FIRST_PASS));
    // CBZ w10, .setup_and_run (first_pass==0 means not first pass)
    let cbz_setup_pos = code.len();
    code.extend_from_slice(&encode_cbz(10, 0)); // placeholder

    // First pass: set first_pass=0
    code.extend_from_slice(&encode_movz(10, 0));
    code.extend_from_slice(&encode_strb_imm(10, 9, OFF_FIRST_PASS));

    // Save SP, LR, X0 to persistent data
    code.extend_from_slice(&encode_mov_reg(10, 31)); // x10 = sp
    code.extend_from_slice(&encode_str_imm_base(10, 9, OFF_SAVE_SP));
    code.extend_from_slice(&encode_str_imm_base(30, 9, OFF_SAVE_LR)); // lr
    code.extend_from_slice(&encode_str_imm_base(0, 9, OFF_SAVE_X0)); // x0

    // Set counter
    emit_mov32(&mut code, 10, persistent_count);
    code.extend_from_slice(&encode_str_w_imm_base(10, 9, OFF_COUNTER));

    // === Section B: .setup_and_run ===
    let setup_and_run = code.len();
    let cd = (setup_and_run as i64 - cbz_setup_pos as i64) as i32;
    code[cbz_setup_pos..cbz_setup_pos + 4].copy_from_slice(&encode_cbz(10, cd));

    // Set LR to iteration_boundary so function return comes back here
    // ADR lr, iteration_boundary (placeholder, patched later)
    let adr_lr_iter_pos = code.len();
    code.extend_from_slice(&encode_adr(30, 0)); // placeholder

    // Restore x0 from save area
    let adr_data_b = emit_adr_placeholder(&mut code, 9);
    lit_pool.push(LitPoolEntry {
        instr_pos: adr_data_b,
        value: persistent_data_va,
    });
    code.extend_from_slice(&encode_ldr_imm(0, 9, OFF_SAVE_X0));

    // Emit displaced instructions with PC-relative relocation.
    // AArch64 instructions that reference PC (ADRP, ADR, LDR literal, B, BL, B.cond,
    // CBZ/CBNZ, TBZ/TBNZ) need their immediates adjusted when moved to a new VA.
    let num_insns = displaced_len / 4;
    for i in 0..num_insns {
        let offset = i * 4;
        let raw = u32::from_le_bytes([
            displaced_bytes[offset],
            displaced_bytes[offset + 1],
            displaced_bytes[offset + 2],
            displaced_bytes[offset + 3],
        ]);
        let original_insn_va = persistent_addr + offset as u64;
        let new_insn_va = wrapper_va + code.len() as u64;

        if is_pc_relative(raw) {
            match relocate_pc_relative(raw, original_insn_va, new_insn_va) {
                Ok(relocated) => code.extend_from_slice(&relocated.to_le_bytes()),
                Err(_) => code.extend_from_slice(&raw.to_le_bytes()),
            }
        } else {
            code.extend_from_slice(&raw.to_le_bytes());
        }
    }

    // B return_va (persistent_addr + displaced_len)
    let b_return_pos = code.len();
    let brel = ((return_va as i64 - (wrapper_va + b_return_pos as u64) as i64) / 4) as i32;
    code.extend_from_slice(&encode_b(brel));

    // === Section C: iteration_boundary ===
    let iteration_boundary = code.len();
    // Fix up ADR lr, iteration_boundary
    let iter_va = wrapper_va + iteration_boundary as u64;
    let adr_lr_va = wrapper_va + adr_lr_iter_pos as u64;
    let adr_off = iter_va as i64 - adr_lr_va as i64;
    code[adr_lr_iter_pos..adr_lr_iter_pos + 4].copy_from_slice(&encode_adr(30, adr_off));

    // Load persistent data, decrement counter
    let adr_data_c = emit_adr_placeholder(&mut code, 9);
    lit_pool.push(LitPoolEntry {
        instr_pos: adr_data_c,
        value: persistent_data_va,
    });

    code.extend_from_slice(&encode_ldr_w_imm(10, 9, OFF_COUNTER));
    // SUBS w10, w10, #1
    code.extend_from_slice(&encode_subs_w_imm(10, 10, 1));
    code.extend_from_slice(&encode_str_w_imm_base(10, 9, OFF_COUNTER));

    // B.EQ .exit_child (counter == 0)
    let beq_exit_pos = code.len();
    code.extend_from_slice(&[0x00, 0x00, 0x00, 0x54]); // placeholder B.EQ

    // getpid()
    emit_macos_syscall(&mut code, MACOS_SYS_GETPID);
    // kill(self, SIGSTOP=17)
    code.extend_from_slice(&encode_movz(1, 17)); // SIGSTOP=17 on macOS
    emit_macos_syscall(&mut code, MACOS_SYS_KILL);

    // After SIGCONT: clear prev_loc
    // ADR x9, data_va + PREV_LOC_OFFSET
    let prev_loc_va = data_va + PREV_LOC_OFFSET;
    let adr_prevloc = emit_adr_placeholder(&mut code, 9);
    lit_pool.push(LitPoolEntry {
        instr_pos: adr_prevloc,
        value: prev_loc_va,
    });
    code.extend_from_slice(&encode_strh_zero(9, 0)); // STRH wzr, [x9]

    // Restore SP, LR, X0 from persistent data
    let adr_data_d = emit_adr_placeholder(&mut code, 9);
    lit_pool.push(LitPoolEntry {
        instr_pos: adr_data_d,
        value: persistent_data_va,
    });

    code.extend_from_slice(&encode_ldr_imm(10, 9, OFF_SAVE_SP));
    code.extend_from_slice(&encode_mov_reg(31, 10)); // sp = x10
    code.extend_from_slice(&encode_ldr_imm(30, 9, OFF_SAVE_LR)); // lr
    code.extend_from_slice(&encode_ldr_imm(0, 9, OFF_SAVE_X0)); // x0

    // B .setup_and_run
    let b_setup_pos = code.len();
    let brel = ((setup_and_run as i64 - b_setup_pos as i64) / 4) as i32;
    code.extend_from_slice(&encode_b(brel));

    // === Section D: .exit_child ===
    let exit_child = code.len();
    patch_b_cond(
        &mut code,
        beq_exit_pos,
        (exit_child as i64 - beq_exit_pos as i64) as i32,
        0x0,
    );

    // exit(0)
    code.extend_from_slice(&encode_movz(0, 0));
    emit_macos_syscall(&mut code, MACOS_SYS_EXIT);

    // === Literal pool: patch all ADR instructions ===
    // ADR has a ±1MB range (21-bit signed offset). Since all our targets are nearby
    // (persistent_data_va and data_va are in the same segment), we use ADR directly
    // to the absolute VA. But wait — ADR is PC-relative, so we need:
    //   ADR xN, <offset_from_this_instruction_to_target>
    // Since the wrapper is placed in the new segment at wrapper_va, and the data area
    // is also in the new segment, the offset should be within ADR's range.
    for entry in &lit_pool {
        let instr_va = wrapper_va + entry.instr_pos as u64;
        let offset = entry.value as i64 - instr_va as i64;
        // Check ADR range (±1MB = ±0x100000)
        if !(-0x100000..=0xFFFFF).contains(&offset) {
            anyhow::bail!(
                "persistent wrapper: ADR offset out of range ({} bytes) for target 0x{:x} from 0x{:x}",
                offset,
                entry.value,
                instr_va,
            );
        }
        let rd = (code[entry.instr_pos] & 0x1F) as u32;
        code[entry.instr_pos..entry.instr_pos + 4].copy_from_slice(&encode_adr(rd, offset));
    }

    Ok(PersistentWrapper { code })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exec_init_code_size() {
        let code = generate_macho_exec_init_x86_64(0x200000, 0x100000, 0x100800, false, false)
            .expect("exec init x86_64 should succeed");
        assert!(!code.code.is_empty());
        assert!(
            code.code.len() < 1024,
            "init code too large: {} bytes",
            code.code.len()
        );
        assert_eq!(code.entry_va, 0x200000);
    }

    #[test]
    fn test_exec_init_with_forkserver() {
        let code = generate_macho_exec_init_x86_64(0x200000, 0x100000, 0x100800, true, false)
            .expect("exec init x86_64 with forkserver should succeed");
        assert!(!code.code.is_empty());
        // Forkserver adds ~200 bytes
        assert!(code.code.len() > 100);
    }

    #[test]
    fn test_dylib_init_code_size() {
        let code = generate_macho_dylib_init_x86_64(0x200000, 0x100000, None, false)
            .expect("dylib init x86_64 should succeed");
        assert!(!code.code.is_empty());
        assert!(
            code.code.len() < 2048,
            "dylib init code too large: {} bytes",
            code.code.len()
        );
    }

    #[test]
    fn test_dylib_init_with_chain() {
        let code = generate_macho_dylib_init_x86_64(0x200000, 0x100000, Some(0x150000), false)
            .expect("dylib init x86_64 with chain should succeed");
        assert!(!code.code.is_empty());
        // Should end with a JMP to the chain target, not RET
        let last5 = &code.code[code.code.len() - 5..];
        assert_eq!(last5[0], 0xE9, "should end with JMP rel32");
    }

    #[test]
    fn test_unixthread_init_code_size() {
        let code =
            generate_macho_unixthread_init_x86_64(0x200000, 0x100000, 0x100800, false, false)
                .expect("unixthread init x86_64 should succeed");
        assert!(!code.code.is_empty());
        assert!(code.code.len() < 1024);
    }

    #[test]
    fn test_macos_syscall_numbers() {
        assert_eq!(BSD_CLASS | MACOS_SYS_EXIT, 0x2000001);
        assert_eq!(BSD_CLASS | MACOS_SYS_SHMAT, 0x2000106);
        assert_eq!(BSD_CLASS | MACOS_SYS_SYSCTL, 0x20000CA);
        assert_eq!(BSD_CLASS | MACOS_SYS_FORK, 0x2000002);
    }

    #[test]
    fn test_aarch64_exec_init_code_size() {
        let code = generate_macho_exec_init_aarch64(0x200000, 0x100000, 0x100800, false, false)
            .expect("aarch64 exec init should succeed");
        assert!(!code.code.is_empty());
        assert!(
            code.code.len() < 2048,
            "ARM64 exec init too large: {}",
            code.code.len()
        );
        assert_eq!(code.entry_va, 0x200000);
        // All instructions must be 4-byte aligned
        assert_eq!(code.code.len() % 4, 0);
    }

    #[test]
    fn test_aarch64_exec_init_with_forkserver() {
        let code = generate_macho_exec_init_aarch64(0x200000, 0x100000, 0x100800, true, false)
            .expect("aarch64 exec init with forkserver should succeed");
        assert!(!code.code.is_empty());
        assert!(code.code.len() > 200);
        assert_eq!(code.code.len() % 4, 0);
    }

    #[test]
    fn test_aarch64_dylib_init_code_size() {
        let code = generate_macho_dylib_init_aarch64(0x200000, 0x100000, None, false)
            .expect("aarch64 dylib init should succeed");
        assert!(!code.code.is_empty());
        assert!(
            code.code.len() < 4096,
            "ARM64 dylib init too large: {}",
            code.code.len()
        );
        assert_eq!(code.code.len() % 4, 0);
    }

    #[test]
    fn test_aarch64_dylib_init_with_chain() {
        let code = generate_macho_dylib_init_aarch64(0x200000, 0x100000, Some(0x150000), false)
            .expect("aarch64 dylib init with chain should succeed");
        assert!(!code.code.is_empty());
        assert_eq!(code.code.len() % 4, 0);
        // Should contain a B to chain target (before the literal pool at the end)
        // Scan backwards from the literal pool to find the B instruction
        let mut found_b = false;
        for i in (0..code.code.len()).step_by(4).rev() {
            let instr =
                u32::from_le_bytes(code.code[i..i + 4].try_into().expect("slice is 4 bytes"));
            if instr & 0xFC000000 == 0x14000000 && instr != 0x14000000 {
                found_b = true;
                break;
            }
        }
        assert!(found_b, "should contain B to chain target");
    }

    #[test]
    fn test_aarch64_unixthread_init_code_size() {
        let code =
            generate_macho_unixthread_init_aarch64(0x200000, 0x100000, 0x100800, false, false)
                .expect("aarch64 unixthread init should succeed");
        assert!(!code.code.is_empty());
        assert!(code.code.len() < 2048);
        assert_eq!(code.code.len() % 4, 0);
    }

    #[test]
    fn test_persistent_wrapper_x86_64_basic() {
        // 5 NOPs as displaced bytes (typical minimum for JMP rel32)
        let displaced = vec![0x90; 5];
        let wrapper = generate_macho_persistent_wrapper_x86_64(&PersistentWrapperParams {
            wrapper_va: 0x300000,
            persistent_data_va: 0x200000,
            data_va: 0x100000,
            persistent_addr: 0x100800,
            displaced_bytes: &displaced,
            displaced_len: 5,
            persistent_count: 1000,
            include_forkserver: false,
        })
        .expect("persistent wrapper x86_64 basic should succeed");
        assert!(!wrapper.code.is_empty());
        assert!(
            wrapper.code.len() < 512,
            "wrapper too large: {} bytes",
            wrapper.code.len()
        );
        // Should contain macOS SYS_exit (0x2000001)
        let exit_bytes = (BSD_CLASS | MACOS_SYS_EXIT).to_le_bytes();
        assert!(
            wrapper.code.windows(4).any(|w| w == exit_bytes),
            "should contain macOS SYS_exit"
        );
        // Should contain macOS SYS_getpid (0x2000014)
        let getpid_bytes = (BSD_CLASS | MACOS_SYS_GETPID).to_le_bytes();
        assert!(
            wrapper.code.windows(4).any(|w| w == getpid_bytes),
            "should contain macOS SYS_getpid"
        );
        // Should contain macOS SYS_kill (0x2000025)
        let kill_bytes = (BSD_CLASS | MACOS_SYS_KILL).to_le_bytes();
        assert!(
            wrapper.code.windows(4).any(|w| w == kill_bytes),
            "should contain macOS SYS_kill"
        );
        // Should end with syscall (0F 05)
        assert!(
            wrapper.code.windows(2).any(|w| w == [0x0F, 0x05]),
            "should contain syscall instructions"
        );
    }

    #[test]
    fn test_persistent_wrapper_x86_64_with_forkserver() {
        let displaced = vec![0x90; 5];
        let wrapper = generate_macho_persistent_wrapper_x86_64(&PersistentWrapperParams {
            wrapper_va: 0x300000,
            persistent_data_va: 0x200000,
            data_va: 0x100000,
            persistent_addr: 0x100800,
            displaced_bytes: &displaced,
            displaced_len: 5,
            persistent_count: 1000,
            include_forkserver: true,
        })
        .expect("persistent wrapper x86_64 with forkserver should succeed");
        assert!(!wrapper.code.is_empty());
        // Forkserver adds significant code
        assert!(
            wrapper.code.len() > 200,
            "forkserver wrapper too small: {} bytes",
            wrapper.code.len()
        );
        // Should contain macOS SYS_fork (0x2000002)
        let fork_bytes = (BSD_CLASS | MACOS_SYS_FORK).to_le_bytes();
        assert!(
            wrapper.code.windows(4).any(|w| w == fork_bytes),
            "should contain macOS SYS_fork"
        );
        // Should contain macOS SYS_wait4 (0x2000007)
        let wait4_bytes = (BSD_CLASS | MACOS_SYS_WAIT4).to_le_bytes();
        assert!(
            wrapper.code.windows(4).any(|w| w == wait4_bytes),
            "should contain macOS SYS_wait4"
        );
    }

    #[test]
    fn test_persistent_wrapper_x86_64_sigstop_value() {
        // Verify SIGSTOP=17 (macOS) not 19 (Linux)
        let displaced = vec![0x90; 5];
        let wrapper = generate_macho_persistent_wrapper_x86_64(&PersistentWrapperParams {
            wrapper_va: 0x300000,
            persistent_data_va: 0x200000,
            data_va: 0x100000,
            persistent_addr: 0x100800,
            displaced_bytes: &displaced,
            displaced_len: 5,
            persistent_count: 1000,
            include_forkserver: false,
        })
        .expect("persistent wrapper x86_64 for sigstop test should succeed");
        // SIGSTOP=17 should appear as mov esi, 17 (BE 11 00 00 00)
        assert!(
            wrapper.code.windows(5).any(|w| w == [0xBE, 17, 0, 0, 0]),
            "should use macOS SIGSTOP=17, not Linux SIGSTOP=19"
        );
    }

    #[test]
    fn test_persistent_wrapper_aarch64_basic() {
        // ARM64 NOP (d503201f) as displaced instruction
        // VAs must be within ADR range (±1MB) — in practice they're in the same segment.
        let displaced = [0x1F, 0x20, 0x03, 0xD5];
        let wrapper = generate_macho_persistent_wrapper_aarch64(&PersistentWrapperParams {
            wrapper_va: 0x201000,
            persistent_data_va: 0x200000,
            data_va: 0x200000,
            persistent_addr: 0x100800,
            displaced_bytes: &displaced,
            displaced_len: 4,
            persistent_count: 1000,
            include_forkserver: false,
        })
        .expect("persistent wrapper aarch64 basic should succeed");
        assert!(!wrapper.code.is_empty());
        assert_eq!(
            wrapper.code.len() % 4,
            0,
            "ARM64 code must be 4-byte aligned"
        );
        // Should contain SVC #0x80 (01 10 00 D4)
        let svc = arm64::encode_svc_0x80();
        assert!(
            wrapper.code.windows(4).any(|w| w == svc),
            "should contain SVC #0x80"
        );
    }

    #[test]
    fn test_persistent_wrapper_aarch64_with_forkserver() {
        let displaced = [0x1F, 0x20, 0x03, 0xD5]; // NOP
        let wrapper = generate_macho_persistent_wrapper_aarch64(&PersistentWrapperParams {
            wrapper_va: 0x201000,
            persistent_data_va: 0x200000,
            data_va: 0x200000,
            persistent_addr: 0x100800,
            displaced_bytes: &displaced,
            displaced_len: 4,
            persistent_count: 1000,
            include_forkserver: true,
        })
        .expect("persistent wrapper aarch64 with forkserver should succeed");
        assert!(!wrapper.code.is_empty());
        assert_eq!(wrapper.code.len() % 4, 0);
        assert!(
            wrapper.code.len() > 200,
            "forkserver wrapper too small: {} bytes",
            wrapper.code.len()
        );
    }

    #[test]
    fn test_macos_kill_syscall_number() {
        assert_eq!(BSD_CLASS | MACOS_SYS_KILL, 0x2000025);
    }
}
