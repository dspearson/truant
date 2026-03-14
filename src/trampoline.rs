use std::collections::BTreeMap;

use anyhow::{Context, Result};
use iced_x86::{BlockEncoder, BlockEncoderOptions, Decoder, DecoderOptions, InstructionBlock};

use crate::disasm::BasicBlock;

/// Size of the base data area: 8 bytes for shm_ptr + 8 bytes for prev_loc (only 2 used but aligned).
pub const DATA_SIZE: u64 = 16;

/// Additional data area size for heap sanitiser (appended after coverage data).
/// Layout: [original_malloc_ptr: 8] [original_free_ptr: 8] [alloc_count: 8] [free_count: 8]
pub const HEAP_SAN_DATA_SIZE: u64 = 32;

/// Data area for persistent mode state.
///
/// Layout (160 bytes):
///   +0:   first_pass (u8, init=1)           [child-side]
///   +1:   child_stopped (u8, init=0)        [forkserver-side]
///   +2:   forkserver_started (u8, init=0)   [deferred forkserver flag]
///   +3:   padding (1 byte)
///   +4:   counter (u32)                     [child-side]
///   +8:   save_rax..save_r15 (128 bytes)    [child-side + deferred forkserver, 16 x 8]
///   +136: save_ret_addr (8 bytes)           [child-side]
///   +144: save_rsp (8 bytes)                [child-side]
///   +152: padding (8 bytes)
pub const PERSISTENT_DATA_SIZE: u64 = 160;

/// Offset within the data area for prev_loc (2-byte value, 8-byte aligned slot).
pub const PREV_LOC_OFFSET: u64 = 8;

/// Generated trampoline for a single basic block.
#[derive(Debug)]
pub struct Trampoline {
    /// Virtual address where this trampoline will be placed.
    pub va: u64,
    /// Machine code bytes.
    pub code: Vec<u8>,
}

/// The complete init code blob (SHM setup + optional forkserver).
#[derive(Debug)]
pub struct InitCode {
    /// Virtual address where the init code is placed.
    pub va: u64,
    /// Machine code bytes.
    pub code: Vec<u8>,
    /// VA where the forkserver loop begins (for entry point redirection).
    pub entry_va: u64,
}

/// Generate the coverage trampoline for a single basic block.
///
/// The trampoline:
/// 1. Saves flags and scratch registers
/// 2. Loads SHM pointer, does edge hash, increments map byte
/// 3. Updates prev_loc
/// 4. Restores registers
/// 5. Executes the displaced original instructions (relocated)
/// 6. Jumps back to the original block after the displaced bytes
///
/// `trampoline_va`: where this trampoline will live in the new segment.
/// `data_va`: where the 16-byte data area lives (shm_ptr at +0, prev_loc at +8).
/// `block`: the basic block being instrumented.
pub fn generate_trampoline(
    trampoline_va: u64,
    data_va: u64,
    block: &BasicBlock,
) -> Result<Trampoline> {
    let block_id = block.block_id as u32;
    let return_va = block.va + block.displaced_len as u64;

    // Build the coverage instrumentation stub as raw bytes.
    let mut stub = Vec::with_capacity(128);

    // Skip past the 128-byte red zone before pushing anything.
    // The x86-64 SysV ABI allows leaf functions to store data in [rsp-128..rsp]
    // without adjusting rsp. Our pushes would clobber this area.
    // lea rsp, [rsp - 128]  ; 8 bytes: 48 8D A4 24 80 FF FF FF
    stub.extend_from_slice(&[0x48, 0x8D, 0xA4, 0x24, 0x80, 0xFF, 0xFF, 0xFF]);

    // pushfq
    stub.push(0x9C);
    // push rax
    stub.push(0x50);
    // push rcx
    stub.push(0x51);

    // Compute RIP-relative offsets directly. RIP after each instruction points
    // to the NEXT instruction, so disp = data_va - (current_va + instr_len).

    // mov rax, qword [rip + disp_shm_ptr]    ; 7 bytes: REX.W 8B 05 <disp32>
    let mov_rax_shm_ip = trampoline_va + stub.len() as u64 + 7; // RIP after this instruction
    let shm_ptr_disp = (data_va as i64 - mov_rax_shm_ip as i64) as i32;
    stub.extend_from_slice(&[0x48, 0x8B, 0x05]);
    stub.extend_from_slice(&shm_ptr_disp.to_le_bytes());

    // test rax, rax    ; 3 bytes: 48 85 C0
    stub.extend_from_slice(&[0x48, 0x85, 0xC0]);

    // jz .skip         ; 2 bytes: 74 <rel8> — we'll fix the offset after
    let jz_offset_pos = stub.len() + 1;
    stub.extend_from_slice(&[0x74, 0x00]); // placeholder

    // movzx ecx, word [rip + disp_prev_loc]  ; 7 bytes: 0F B7 0D <disp32>
    let movzx_ip = trampoline_va + stub.len() as u64 + 7;
    let prev_loc_va = data_va + PREV_LOC_OFFSET;
    let prev_loc_disp = (prev_loc_va as i64 - movzx_ip as i64) as i32;
    stub.extend_from_slice(&[0x0F, 0xB7, 0x0D]);
    stub.extend_from_slice(&prev_loc_disp.to_le_bytes());

    // xor ecx, BLOCK_ID   ; 6 bytes: 81 F1 <imm32>
    stub.extend_from_slice(&[0x81, 0xF1]);
    stub.extend_from_slice(&block_id.to_le_bytes());

    // and ecx, 0xFFFF     ; 6 bytes: 81 E1 <imm32>
    stub.extend_from_slice(&[0x81, 0xE1]);
    stub.extend_from_slice(&0x0000FFFFu32.to_le_bytes());

    // inc byte [rax + rcx]  ; 3 bytes: FE 04 08
    stub.extend_from_slice(&[0xFE, 0x04, 0x08]);

    // mov word [rip + disp_prev_loc], BLOCK_ID >> 1
    // 66 C7 05 <disp32> <imm16> = 9 bytes
    let mov_prev_ip = trampoline_va + stub.len() as u64 + 9;
    let prev_loc_disp2 = (prev_loc_va as i64 - mov_prev_ip as i64) as i32;
    let half_id = (block_id >> 1) as u16;
    stub.extend_from_slice(&[0x66, 0xC7, 0x05]);
    stub.extend_from_slice(&prev_loc_disp2.to_le_bytes());
    stub.extend_from_slice(&half_id.to_le_bytes());

    // .skip: (fix up the jz offset)
    // jz_offset_pos points to the rel8 byte. jz is 2 bytes (74 rel8).
    // IP after jz = jz_offset_pos + 1. rel8 = skip_target - (jz_offset_pos + 1)
    let skip_target = stub.len();
    let jz_rel = (skip_target as isize - (jz_offset_pos as isize + 1)) as u8;
    stub[jz_offset_pos] = jz_rel;

    // pop rcx
    stub.push(0x59);
    // pop rax
    stub.push(0x58);
    // popfq
    stub.push(0x9D);

    // Restore rsp past the red zone skip.
    // lea rsp, [rsp + 128]  ; 8 bytes: 48 8D A4 24 80 00 00 00
    stub.extend_from_slice(&[0x48, 0x8D, 0xA4, 0x24, 0x80, 0x00, 0x00, 0x00]);

    // Now emit the relocated displaced instructions using iced-x86 BlockEncoder.
    let displaced_va = trampoline_va + stub.len() as u64;
    let relocated = relocate_instructions(&block.displaced_bytes, block.va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate instructions for block at 0x{:x}",
                block.va,
            )
        })?;

    stub.extend_from_slice(&relocated);

    // jmp return_va   ; 5 bytes: E9 <rel32>
    let jmp_ip = trampoline_va + stub.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    stub.push(0xE9);
    stub.extend_from_slice(&jmp_rel.to_le_bytes());

    Ok(Trampoline {
        va: trampoline_va,
        code: stub,
    })
}

/// Relocate displaced instructions from `orig_ip` to `new_ip` using iced-x86 BlockEncoder.
fn relocate_instructions(bytes: &[u8], orig_ip: u64, new_ip: u64) -> Result<Vec<u8>> {
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

/// Generate the init + forkserver code.
///
/// This code:
/// 1. Walks the envp array from the ELF entry stack to find __AFL_SHM_ID=<decimal>
/// 2. Calls shmat() via raw syscall to map the shared memory
/// 3. Stores the SHM pointer at `data_va`
/// 4. Runs the forkserver protocol (if enabled)
/// 5. Jumps to the original entry point
///
/// All done with raw syscalls — no PLT/GOT dependencies.
/// Uses the ELF entry stack layout (argc, argv, envp) instead of /proc/self/environ,
/// so there is no buffer size limitation regardless of environment size.
pub fn generate_init_code(
    init_va: u64,
    data_va: u64,
    original_entry: u64,
    enable_forkserver: bool,
    persistent_data_va: Option<u64>,
) -> Result<InitCode> {
    let mut code = Vec::with_capacity(512);
    let entry_va = init_va;

    // ============================================================
    // Part 1: Walk envp from the ELF entry stack, find __AFL_SHM_ID
    // ============================================================

    // Save registers. At the ELF entry point, rdx carries the rtld_fini pointer
    // (the only register with a defined value per the SysV ABI). We must preserve
    // it across the init code since syscalls clobber rdx. Save it first (LIFO).
    code.push(0x52); // push rdx
    code.push(0x55); // push rbp
    code.push(0x53); // push rbx
    code.extend_from_slice(&[0x41, 0x54]); // push r12
    code.extend_from_slice(&[0x41, 0x55]); // push r13
    code.extend_from_slice(&[0x41, 0x56]); // push r14

    // sub rsp, 16   ; scratch space for forkserver temporaries
    // 6 pushes = 48 bytes, + 16 = 64 total offset from original rsp
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x10]);

    // Original entry stack layout (before our prologue):
    //   [rsp]     = argc
    //   [rsp+8]   = argv[0] ... argv[argc-1]
    //   [rsp+8+8*argc] = NULL
    //   [rsp+16+8*argc] = envp[0] ... envp[N]
    //   [rsp+...] = NULL
    // After our prologue: original rsp = current rsp + 64
    const STACK_OFFSET: u8 = 64;

    // lea r12, [rip + delta_to_data_va]  ; 7 bytes: 4C 8D 25 <disp32>
    let lea_r12_pos = code.len();
    code.extend_from_slice(&[0x4C, 0x8D, 0x25, 0x00, 0x00, 0x00, 0x00]);
    let lea_r12_rip = init_va + code.len() as u64;
    let delta = (data_va as i64 - lea_r12_rip as i64) as i32;
    code[lea_r12_pos + 3..lea_r12_pos + 7].copy_from_slice(&delta.to_le_bytes());

    // --- Walk envp from the entry stack ---
    // mov rax, [rsp + STACK_OFFSET]        ; argc
    code.extend_from_slice(&[0x48, 0x8B, 0x44, 0x24, STACK_OFFSET]);
    // lea rbx, [rsp + STACK_OFFSET + 8]    ; &argv[0]
    code.extend_from_slice(&[0x48, 0x8D, 0x5C, 0x24, STACK_OFFSET + 8]);
    // lea rbx, [rbx + rax*8 + 8]           ; skip argv + NULL terminator → &envp[0]
    code.extend_from_slice(&[0x48, 0x8D, 0x5C, 0xC3, 0x08]);

    // Load needle "__AFL_SH" into rbp for fast comparison
    let needle_q = u64::from_le_bytes(*b"__AFL_SH");
    code.extend_from_slice(&[0x48, 0xBD]); // mov rbp, imm64
    code.extend_from_slice(&needle_q.to_le_bytes());

    // .envp_loop:
    let envp_loop = code.len();

    // mov rdi, [rbx]                       ; envp[i] pointer
    code.extend_from_slice(&[0x48, 0x8B, 0x3B]);
    // test rdi, rdi                        ; NULL = end of envp
    code.extend_from_slice(&[0x48, 0x85, 0xFF]);
    // jz .skip_shm
    let jz_skip_shm_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // cmp [rdi], rbp                       ; first 8 bytes == "__AFL_SH"?
    code.extend_from_slice(&[0x48, 0x39, 0x2F]);
    // jne .next_env
    let jne_next_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // cmp dword [rdi + 8], "M_ID"
    let needle2_dw = u32::from_le_bytes(*b"M_ID");
    code.extend_from_slice(&[0x81, 0x7F, 0x08]);
    code.extend_from_slice(&needle2_dw.to_le_bytes());
    // jne .next_env
    let jne2_next_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // cmp byte [rdi + 12], '='
    code.extend_from_slice(&[0x80, 0x7F, 0x0C, b'=']);
    // jne .next_env
    let jne3_next_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // Found! Parse decimal at rdi + 13.
    // lea rbx, [rdi + 13]
    code.extend_from_slice(&[0x48, 0x8D, 0x5F, 0x0D]);
    // xor r13, r13
    code.extend_from_slice(&[0x4D, 0x31, 0xED]);

    // .parse_loop:
    let parse_loop_start = code.len();
    // movzx eax, byte [rbx]
    code.extend_from_slice(&[0x0F, 0xB6, 0x03]);
    // sub al, '0'
    code.extend_from_slice(&[0x2C, 0x30]);
    // cmp al, 9
    code.extend_from_slice(&[0x3C, 0x09]);
    // ja .parse_done
    let ja_parse_done_pos = code.len();
    code.extend_from_slice(&[0x77, 0x00]); // ja rel8 placeholder
    // imul r13, r13, 10
    code.extend_from_slice(&[0x4D, 0x6B, 0xED, 0x0A]);
    // movzx eax, al
    code.extend_from_slice(&[0x0F, 0xB6, 0xC0]);
    // add r13, rax
    code.extend_from_slice(&[0x49, 0x01, 0xC5]);
    // inc rbx
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]);
    // jmp .parse_loop
    let jmp_parse_rel = (parse_loop_start as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_parse_rel as u8]);

    // .parse_done:
    let parse_done = code.len();
    code[ja_parse_done_pos + 1] = (parse_done - (ja_parse_done_pos + 2)) as u8;

    // --- SYS_shmat(r13, NULL, 0) ---
    // mov eax, 30 (SYS_shmat)
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&30u32.to_le_bytes());
    // mov rdi, r13 (shmid)
    code.extend_from_slice(&[0x4C, 0x89, 0xEF]);
    // xor esi, esi (shmaddr = NULL)
    code.extend_from_slice(&[0x31, 0xF6]);
    // xor edx, edx (shmflg = 0)
    code.extend_from_slice(&[0x31, 0xD2]);
    // syscall
    code.extend_from_slice(&[0x0F, 0x05]);

    // Check return: if rax >= -4096 (error), skip
    // cmp rax, -4096
    code.extend_from_slice(&[0x48, 0x3D]);
    code.extend_from_slice(&(-4096i32 as u32).to_le_bytes());
    // jae .skip_shm
    let jae_skip_shm_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x83, 0x00, 0x00, 0x00, 0x00]); // jae rel32 placeholder

    // Store SHM pointer: mov [r12], rax
    code.extend_from_slice(&[0x49, 0x89, 0x04, 0x24]);

    // Jump to forkserver/epilogue
    let jmp_to_forkserver_pos = code.len();
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32 placeholder

    // .next_env:
    let next_env = code.len();
    code[jne_next_pos + 1] = (next_env - (jne_next_pos + 2)) as u8;
    code[jne2_next_pos + 1] = (next_env - (jne2_next_pos + 2)) as u8;
    code[jne3_next_pos + 1] = (next_env - (jne3_next_pos + 2)) as u8;

    // add rbx, 8                           ; next envp pointer
    code.extend_from_slice(&[0x48, 0x83, 0xC3, 0x08]);
    // jmp .envp_loop
    let jmp_envp_rel = (envp_loop as i32) - (code.len() as i32 + 5);
    code.push(0xE9);
    code.extend_from_slice(&jmp_envp_rel.to_le_bytes());

    // .skip_shm: (SHM not available — continue to forkserver or epilogue)
    let skip_shm = code.len();
    // Fix up skip_shm jump placeholders
    let skip_shm_offset = |from: usize| -> i32 { (skip_shm as i32) - (from as i32 + 4) };
    let rel = skip_shm_offset(jz_skip_shm_pos + 2);
    code[jz_skip_shm_pos + 2..jz_skip_shm_pos + 6].copy_from_slice(&rel.to_le_bytes());
    let rel = skip_shm_offset(jae_skip_shm_pos + 2);
    code[jae_skip_shm_pos + 2..jae_skip_shm_pos + 6].copy_from_slice(&rel.to_le_bytes());

    if enable_forkserver {
        // .forkserver:
        let forkserver_start = code.len();
        // Fix jmp_to_forkserver
        let rel = (forkserver_start as i32) - (jmp_to_forkserver_pos as i32 + 5);
        code[jmp_to_forkserver_pos + 1..jmp_to_forkserver_pos + 5]
            .copy_from_slice(&rel.to_le_bytes());

        // Forkserver protocol:
        // 1. write(FORKSRV_FD+1, hello, 4) — announce to parent
        // 2. Loop: read(FORKSRV_FD, &cmd, 4) — wait for GO
        //    [persistent] check child_stopped → SIGCONT instead of fork
        //    fork() — child runs, parent waits
        //    write(FORKSRV_FD+1, child_pid, 4)
        //    waitpid(child_pid, &status, WUNTRACED)
        //    [persistent] if WIFSTOPPED → set child_stopped, loop (don't write status)
        //    write(FORKSRV_FD+1, &status, 4)
        //    goto 2
        //
        // FORKSRV_FD = 198

        // mov dword [rsp], 0 ; hello message (4 zero bytes)
        code.extend_from_slice(&[0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00]);

        // SYS_write(199, &hello, 4)
        code.extend_from_slice(&[0xB8]); // mov eax, 1 (SYS_write)
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&[0xBF]); // mov edi, 199 (FORKSRV_FD+1)
        code.extend_from_slice(&199u32.to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]); // lea rsi, [rsp]
        code.extend_from_slice(&[0xBA]); // mov edx, 4
        code.extend_from_slice(&4u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]); // syscall

        // test eax, eax / js .no_forkserver
        code.extend_from_slice(&[0x85, 0xC0]);
        let js_no_fork_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x88, 0x00, 0x00, 0x00, 0x00]); // js placeholder

        // .fork_loop:
        let fork_loop = code.len();

        // SYS_read(198, &cmd, 4) — wait for GO
        code.extend_from_slice(&[0xB8]); // mov eax, 0 (SYS_read)
        code.extend_from_slice(&0u32.to_le_bytes());
        code.extend_from_slice(&[0xBF]); // mov edi, 198 (FORKSRV_FD)
        code.extend_from_slice(&198u32.to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]); // lea rsi, [rsp]
        code.extend_from_slice(&[0xBA]); // mov edx, 4
        code.extend_from_slice(&4u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]); // syscall

        // cmp eax, 4 / jne .epilogue (parent died)
        code.extend_from_slice(&[0x83, 0xF8, 0x04]); // cmp eax, 4
        let jne_epilogue_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne placeholder

        // --- Persistent: check child_stopped before forking ---
        let jmp_past_sigcont_pos;
        if let Some(pd_va) = persistent_data_va {
            let child_stopped_va = pd_va + 1; // offset 1

            // cmp byte [rip+child_stopped], 0
            let cmp_ip = init_va + code.len() as u64 + 7;
            code.extend_from_slice(&[0x80, 0x3D]);
            code.extend_from_slice(
                &((child_stopped_va as i64 - cmp_ip as i64) as i32).to_le_bytes(),
            );
            code.push(0x00);

            // je .do_fork   ; child not stopped, fork normally
            let je_do_fork_pos = code.len();
            code.extend_from_slice(&[0x74, 0x00]); // je rel8 placeholder

            // Child is stopped — SIGCONT it instead of forking
            // SYS_kill(r13d, SIGCONT=18)  ; r13d still holds the child PID from last iteration
            code.extend_from_slice(&[0xB8]); // mov eax, 62 (SYS_kill)
            code.extend_from_slice(&62u32.to_le_bytes());
            code.extend_from_slice(&[0x44, 0x89, 0xEF]); // mov edi, r13d (child pid)
            code.extend_from_slice(&[0xBE]); // mov esi, 18 (SIGCONT)
            code.extend_from_slice(&18u32.to_le_bytes());
            code.extend_from_slice(&[0x0F, 0x05]); // syscall

            // Write child_pid to parent (reuse r13d in [rsp])
            // mov [rsp], r13d
            code.extend_from_slice(&[0x44, 0x89, 0x2C, 0x24]);
            // SYS_write(199, &child_pid, 4)
            code.extend_from_slice(&[0xB8]);
            code.extend_from_slice(&1u32.to_le_bytes());
            code.extend_from_slice(&[0xBF]);
            code.extend_from_slice(&199u32.to_le_bytes());
            code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]);
            code.extend_from_slice(&[0xBA]);
            code.extend_from_slice(&4u32.to_le_bytes());
            code.extend_from_slice(&[0x0F, 0x05]);

            // jmp .do_waitpid  ; skip the fork, go straight to waitpid
            jmp_past_sigcont_pos = Some(code.len());
            code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]); // jmp placeholder

            // .do_fork:
            let do_fork = code.len();
            code[je_do_fork_pos + 1] = (do_fork - (je_do_fork_pos + 2)) as u8;
        } else {
            jmp_past_sigcont_pos = None;
        }

        // SYS_fork()
        code.extend_from_slice(&[0xB8]); // mov eax, 57 (SYS_fork)
        code.extend_from_slice(&57u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // test eax, eax
        code.extend_from_slice(&[0x85, 0xC0]);
        // jz .child (child process jumps to original entry)
        let jz_child_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz placeholder

        // js .epilogue (fork failed)
        let js_epilogue_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x88, 0x00, 0x00, 0x00, 0x00]); // js placeholder

        // --- Parent: write child PID, wait, write status ---
        // mov [rsp], eax (child pid)
        code.extend_from_slice(&[0x89, 0x04, 0x24]);
        // mov r13d, eax (save pid)
        code.extend_from_slice(&[0x41, 0x89, 0xC5]);

        // SYS_write(199, &child_pid, 4)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&199u32.to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]);
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&4u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // .do_waitpid: (persistent SIGCONT path joins here)
        let do_waitpid = code.len();
        if let Some(pos) = jmp_past_sigcont_pos {
            let rel = (do_waitpid as i32) - (pos as i32 + 5);
            code[pos + 1..pos + 5].copy_from_slice(&rel.to_le_bytes());
        }

        // SYS_wait4(child_pid, &status, options, NULL)
        // options = WUNTRACED (2) when persistent, 0 otherwise
        let waitpid_options: u32 = if persistent_data_va.is_some() { 2 } else { 0 };
        code.extend_from_slice(&[0xB8]); // mov eax, 61 (SYS_wait4)
        code.extend_from_slice(&61u32.to_le_bytes());
        code.extend_from_slice(&[0x44, 0x89, 0xEF]); // mov edi, r13d
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]); // lea rsi, [rsp]
        code.extend_from_slice(&[0xBA]); // mov edx, options
        code.extend_from_slice(&waitpid_options.to_le_bytes());
        code.extend_from_slice(&[0x4D, 0x31, 0xD2]); // xor r10, r10 (rusage = NULL)
        code.extend_from_slice(&[0x0F, 0x05]);

        // --- Persistent: check WIFSTOPPED ---
        if let Some(pd_va) = persistent_data_va {
            let child_stopped_va = pd_va + 1;

            // WIFSTOPPED: (status & 0xFF) == 0x7F
            // mov eax, [rsp]   ; load status
            code.extend_from_slice(&[0x8B, 0x04, 0x24]);
            // and eax, 0xFF
            code.extend_from_slice(&[0x25]);
            code.extend_from_slice(&0xFFu32.to_le_bytes());
            // cmp eax, 0x7F
            code.extend_from_slice(&[0x3D]);
            code.extend_from_slice(&0x7Fu32.to_le_bytes());
            // jne .not_stopped
            let jne_not_stopped_pos = code.len();
            code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

            // WIFSTOPPED: set child_stopped = 1, write synthetic status=0
            // to parent, then loop back for next GO.
            // mov byte [rip+child_stopped], 1
            let mov_cs_ip = init_va + code.len() as u64 + 7;
            code.extend_from_slice(&[0xC6, 0x05]);
            code.extend_from_slice(
                &((child_stopped_va as i64 - mov_cs_ip as i64) as i32).to_le_bytes(),
            );
            code.push(0x01);

            // Write synthetic status=0 (normal exit) to parent so it
            // doesn't block waiting for the status read.
            // mov dword [rsp], 0
            code.extend_from_slice(&[0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00]);
            // SYS_write(199, &status, 4)
            code.extend_from_slice(&[0xB8]);
            code.extend_from_slice(&1u32.to_le_bytes());
            code.extend_from_slice(&[0xBF]);
            code.extend_from_slice(&199u32.to_le_bytes());
            code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]);
            code.extend_from_slice(&[0xBA]);
            code.extend_from_slice(&4u32.to_le_bytes());
            code.extend_from_slice(&[0x0F, 0x05]);

            // jmp .fork_loop
            let jmp_forkloop_rel = (fork_loop as i32) - (code.len() as i32 + 5);
            code.push(0xE9);
            code.extend_from_slice(&jmp_forkloop_rel.to_le_bytes());

            // .not_stopped: set child_stopped = 0
            let not_stopped = code.len();
            code[jne_not_stopped_pos + 1] = (not_stopped - (jne_not_stopped_pos + 2)) as u8;

            // mov byte [rip+child_stopped], 0
            let mov_cs2_ip = init_va + code.len() as u64 + 7;
            code.extend_from_slice(&[0xC6, 0x05]);
            code.extend_from_slice(
                &((child_stopped_va as i64 - mov_cs2_ip as i64) as i32).to_le_bytes(),
            );
            code.push(0x00);
        }

        // SYS_write(199, &status, 4)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&199u32.to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]);
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&4u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // jmp .fork_loop
        let jmp_fork_rel = (fork_loop as i32) - (code.len() as i32 + 5);
        code.push(0xE9);
        code.extend_from_slice(&jmp_fork_rel.to_le_bytes());

        // .child: (child process — close forkserver fds, jump to original entry)
        let child_label = code.len();
        let rel = (child_label as i32) - (jz_child_pos as i32 + 6);
        code[jz_child_pos + 2..jz_child_pos + 6].copy_from_slice(&rel.to_le_bytes());

        // SYS_close(198)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&3u32.to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&198u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // SYS_close(199)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&3u32.to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&199u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // Jump to epilogue (restore regs, jump to original entry)
        let jmp_epilogue_pos = code.len();
        code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]); // placeholder

        // .no_forkserver: (forkserver handshake failed — run without)
        let no_forkserver = code.len();
        let rel = (no_forkserver as i32) - (js_no_fork_pos as i32 + 6);
        code[js_no_fork_pos + 2..js_no_fork_pos + 6].copy_from_slice(&rel.to_le_bytes());

        // .epilogue:
        let epilogue = code.len();
        // Fix up jne_epilogue_pos, js_epilogue_pos, jmp_epilogue_pos
        let rel = (epilogue as i32) - (jne_epilogue_pos as i32 + 6);
        code[jne_epilogue_pos + 2..jne_epilogue_pos + 6].copy_from_slice(&rel.to_le_bytes());
        let rel = (epilogue as i32) - (js_epilogue_pos as i32 + 6);
        code[js_epilogue_pos + 2..js_epilogue_pos + 6].copy_from_slice(&rel.to_le_bytes());
        let rel = (epilogue as i32) - (jmp_epilogue_pos as i32 + 5);
        code[jmp_epilogue_pos + 1..jmp_epilogue_pos + 5].copy_from_slice(&rel.to_le_bytes());
    } else {
        // No forkserver: jmp_to_forkserver goes directly to epilogue
        let epilogue = code.len();
        let rel = (epilogue as i32) - (jmp_to_forkserver_pos as i32 + 5);
        code[jmp_to_forkserver_pos + 1..jmp_to_forkserver_pos + 5]
            .copy_from_slice(&rel.to_le_bytes());
    }

    // === Epilogue: restore stack + regs, jump to original entry ===
    // add rsp, 16
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x10]);

    // pop r14; pop r13; pop r12; pop rbx; pop rbp; pop rdx
    code.extend_from_slice(&[0x41, 0x5E]); // pop r14
    code.extend_from_slice(&[0x41, 0x5D]); // pop r13
    code.extend_from_slice(&[0x41, 0x5C]); // pop r12
    code.push(0x5B); // pop rbx
    code.push(0x5D); // pop rbp
    code.push(0x5A); // pop rdx (restore rtld_fini)

    // jmp original_entry — RIP-relative lea + indirect jump (PIE-safe)
    // lea rax, [rip + delta_to_original_entry]  ; 7 bytes: 48 8D 05 <disp32>
    let lea_rax_pos = code.len();
    code.extend_from_slice(&[0x48, 0x8D, 0x05, 0x00, 0x00, 0x00, 0x00]);
    let lea_rax_rip = init_va + code.len() as u64;
    let delta = (original_entry as i64 - lea_rax_rip as i64) as i32;
    code[lea_rax_pos + 3..lea_rax_pos + 7].copy_from_slice(&delta.to_le_bytes());
    // jmp rax
    code.extend_from_slice(&[0xFF, 0xE0]);

    Ok(InitCode {
        va: init_va,
        code,
        entry_va,
    })
}

/// Generate the SHM init code for shared objects (.so files).
///
/// Unlike `generate_init_code` which walks the ELF entry stack for envp,
/// this variant reads `/proc/self/environ` via raw syscalls. This works
/// correctly when the init code runs as a DT_INIT function called by the
/// dynamic linker (where the stack layout differs from ELF entry).
///
/// This code:
/// 1. Opens `/proc/self/environ` and scans for `__AFL_SHM_ID=<decimal>`
/// 2. Calls shmat() via raw syscall to map the shared memory
/// 3. Stores the SHM pointer at `data_va`
/// 4. Optionally chains to the original DT_INIT function
/// 5. Returns to the caller (dynamic linker)
///
/// No forkserver or persistent mode — those are handled by the harness binary.
pub fn generate_so_init_code(
    init_va: u64,
    data_va: u64,
    original_dt_init: Option<u64>,
) -> Result<InitCode> {
    let mut code = Vec::with_capacity(512);
    let entry_va = init_va;

    // ============================================================
    // Prologue: save callee-saved registers (SysV ABI)
    // DT_INIT is called as void (*)(void) by ld.so, so we must
    // preserve rbx, rbp, r12-r15 and return cleanly.
    // ============================================================
    code.push(0x55); // push rbp
    code.push(0x53); // push rbx
    code.extend_from_slice(&[0x41, 0x54]); // push r12
    code.extend_from_slice(&[0x41, 0x55]); // push r13
    code.extend_from_slice(&[0x41, 0x56]); // push r14
    code.extend_from_slice(&[0x41, 0x57]); // push r15

    // Allocate stack buffer for reading /proc/self/environ.
    // 8192 bytes buffer + 8 bytes padding for alignment = 8200, round to 8208 (16-aligned).
    const BUF_SIZE: u32 = 8192;
    const STACK_ALLOC: u32 = 8208;
    // sub rsp, STACK_ALLOC  ; 7 bytes: 48 81 EC <imm32>
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&STACK_ALLOC.to_le_bytes());

    // lea r12, [rip + delta_to_data_va]  ; r12 = &data_area
    let lea_r12_pos = code.len();
    code.extend_from_slice(&[0x4C, 0x8D, 0x25, 0x00, 0x00, 0x00, 0x00]);
    let lea_r12_rip = init_va + code.len() as u64;
    let delta = (data_va as i64 - lea_r12_rip as i64) as i32;
    code[lea_r12_pos + 3..lea_r12_pos + 7].copy_from_slice(&delta.to_le_bytes());

    // ============================================================
    // Open /proc/self/environ
    // ============================================================
    // lea rdi, [rip + path_string]  ; path (embedded at end of code)
    let lea_path_pos = code.len();
    code.extend_from_slice(&[0x48, 0x8D, 0x3D, 0x00, 0x00, 0x00, 0x00]); // placeholder
    // xor esi, esi  ; O_RDONLY = 0
    code.extend_from_slice(&[0x31, 0xF6]);
    // mov eax, 2    ; SYS_open
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&2u32.to_le_bytes());
    // syscall
    code.extend_from_slice(&[0x0F, 0x05]);
    // test eax, eax
    code.extend_from_slice(&[0x85, 0xC0]);
    // js .epilogue  ; open failed
    let js_epilogue_open_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x88, 0x00, 0x00, 0x00, 0x00]); // js rel32 placeholder

    // mov r13d, eax  ; save fd
    code.extend_from_slice(&[0x41, 0x89, 0xC5]);

    // Load needle "__AFL_SH" into rbp for comparison
    let needle_q = u64::from_le_bytes(*b"__AFL_SH");
    code.extend_from_slice(&[0x48, 0xBD]); // mov rbp, imm64
    code.extend_from_slice(&needle_q.to_le_bytes());

    // ============================================================
    // Read + scan loop
    // ============================================================
    // .read_loop:
    let read_loop = code.len();

    // SYS_read(r13d, rsp, BUF_SIZE)
    code.extend_from_slice(&[0xB8]); // mov eax, 0 (SYS_read)
    code.extend_from_slice(&0u32.to_le_bytes());
    code.extend_from_slice(&[0x44, 0x89, 0xEF]); // mov edi, r13d (fd)
    code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]); // lea rsi, [rsp] (buffer)
    code.extend_from_slice(&[0xBA]); // mov edx, BUF_SIZE
    code.extend_from_slice(&BUF_SIZE.to_le_bytes());
    code.extend_from_slice(&[0x0F, 0x05]); // syscall

    // test rax, rax
    code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    // jle .close_fd  ; EOF (0) or error (<0)
    let jle_close_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x8E, 0x00, 0x00, 0x00, 0x00]); // jle rel32 placeholder

    // Set up scan pointers: rbx = scan start, r14 = scan end
    // lea rbx, [rsp]
    code.extend_from_slice(&[0x48, 0x8D, 0x1C, 0x24]);
    // lea r14, [rsp + rax]  ; rsp + bytes_read
    code.extend_from_slice(&[0x4C, 0x8D, 0x34, 0x04]); // lea r14, [rsp + rax]

    // .scan_loop:
    let scan_loop = code.len();

    // lea rcx, [rbx + 13]  ; need 13 bytes for "__AFL_SHM_ID="
    code.extend_from_slice(&[0x48, 0x8D, 0x4B, 0x0D]);
    // cmp rcx, r14  ; 0x3B = CMP r64, r/m64 (rcx - r14)
    code.extend_from_slice(&[0x49, 0x3B, 0xCE]);
    // ja .read_more  ; not enough data left in this chunk
    let ja_read_more_pos = code.len();
    code.extend_from_slice(&[0x77, 0x00]); // ja rel8 placeholder

    // cmp qword [rbx], rbp  ; "__AFL_SH"?
    code.extend_from_slice(&[0x48, 0x39, 0x2B]);
    // jne .advance
    let jne_advance1_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // cmp dword [rbx + 8], "M_ID"
    let needle2 = u32::from_le_bytes(*b"M_ID");
    code.extend_from_slice(&[0x81, 0x7B, 0x08]);
    code.extend_from_slice(&needle2.to_le_bytes());
    // jne .advance
    let jne_advance2_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // cmp byte [rbx + 12], '='
    code.extend_from_slice(&[0x80, 0x7B, 0x0C, b'=']);
    // jne .advance
    let jne_advance3_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // ===== FOUND! Parse decimal SHM ID at rbx + 13 =====
    // lea rbx, [rbx + 13]
    code.extend_from_slice(&[0x48, 0x8D, 0x5B, 0x0D]);
    // xor r15, r15  ; shmid accumulator = 0
    code.extend_from_slice(&[0x4D, 0x31, 0xFF]);

    // .parse_loop:
    let parse_loop = code.len();
    // movzx eax, byte [rbx]
    code.extend_from_slice(&[0x0F, 0xB6, 0x03]);
    // sub al, '0'
    code.extend_from_slice(&[0x2C, 0x30]);
    // cmp al, 9
    code.extend_from_slice(&[0x3C, 0x09]);
    // ja .parse_done
    let ja_parse_done_pos = code.len();
    code.extend_from_slice(&[0x77, 0x00]); // ja rel8 placeholder
    // imul r15, r15, 10
    code.extend_from_slice(&[0x4D, 0x6B, 0xFF, 0x0A]);
    // movzx eax, al
    code.extend_from_slice(&[0x0F, 0xB6, 0xC0]);
    // add r15, rax
    code.extend_from_slice(&[0x49, 0x01, 0xC7]);
    // inc rbx
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]);
    // jmp .parse_loop
    let jmp_parse_rel = (parse_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_parse_rel as u8]);

    // .parse_done:
    let parse_done = code.len();
    code[ja_parse_done_pos + 1] = (parse_done - (ja_parse_done_pos + 2)) as u8;

    // Close fd before shmat
    // SYS_close(r13d)
    code.extend_from_slice(&[0xB8]); // mov eax, 3 (SYS_close)
    code.extend_from_slice(&3u32.to_le_bytes());
    code.extend_from_slice(&[0x44, 0x89, 0xEF]); // mov edi, r13d
    code.extend_from_slice(&[0x0F, 0x05]); // syscall

    // SYS_shmat(r15, NULL, 0)
    code.extend_from_slice(&[0xB8]); // mov eax, 30 (SYS_shmat)
    code.extend_from_slice(&30u32.to_le_bytes());
    code.extend_from_slice(&[0x4C, 0x89, 0xFF]); // mov rdi, r15 (shmid)
    code.extend_from_slice(&[0x31, 0xF6]); // xor esi, esi (shmaddr=NULL)
    code.extend_from_slice(&[0x31, 0xD2]); // xor edx, edx (shmflg=0)
    code.extend_from_slice(&[0x0F, 0x05]); // syscall

    // Check for error: cmp rax, -4096
    code.extend_from_slice(&[0x48, 0x3D]);
    code.extend_from_slice(&(-4096i32 as u32).to_le_bytes());
    // jae .epilogue
    let jae_epilogue_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x83, 0x00, 0x00, 0x00, 0x00]); // jae rel32 placeholder

    // Store SHM pointer: mov [r12], rax
    code.extend_from_slice(&[0x49, 0x89, 0x04, 0x24]);

    // jmp .epilogue
    let jmp_epilogue_pos = code.len();
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32 placeholder

    // .advance: (no match, try next byte)
    let advance = code.len();
    code[jne_advance1_pos + 1] = (advance - (jne_advance1_pos + 2)) as u8;
    code[jne_advance2_pos + 1] = (advance - (jne_advance2_pos + 2)) as u8;
    code[jne_advance3_pos + 1] = (advance - (jne_advance3_pos + 2)) as u8;
    // inc rbx
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]);
    // jmp .scan_loop
    let jmp_scan_rel = (scan_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_scan_rel as u8]);

    // .read_more: (not enough data in chunk, read another)
    let read_more = code.len();
    code[ja_read_more_pos + 1] = (read_more - (ja_read_more_pos + 2)) as u8;
    // jmp .read_loop
    let jmp_read_rel = (read_loop as i32) - (code.len() as i32 + 5);
    code.push(0xE9);
    code.extend_from_slice(&jmp_read_rel.to_le_bytes());

    // .close_fd: (EOF or error)
    let close_fd = code.len();
    let rel = (close_fd as i32) - (jle_close_pos as i32 + 6);
    code[jle_close_pos + 2..jle_close_pos + 6].copy_from_slice(&rel.to_le_bytes());
    // SYS_close(r13d)
    code.extend_from_slice(&[0xB8]); // mov eax, 3
    code.extend_from_slice(&3u32.to_le_bytes());
    code.extend_from_slice(&[0x44, 0x89, 0xEF]); // mov edi, r13d
    code.extend_from_slice(&[0x0F, 0x05]); // syscall

    // ============================================================
    // Epilogue: restore stack + regs, chain to original DT_INIT, return
    // ============================================================
    let epilogue = code.len();
    // Fix up all jumps to epilogue
    let fix_rel32 = |code: &mut Vec<u8>, pos: usize| {
        let rel = (epilogue as i32) - (pos as i32 + 6);
        code[pos + 2..pos + 6].copy_from_slice(&rel.to_le_bytes());
    };
    fix_rel32(&mut code, js_epilogue_open_pos);
    fix_rel32(&mut code, jae_epilogue_pos);
    let rel = (epilogue as i32) - (jmp_epilogue_pos as i32 + 5);
    code[jmp_epilogue_pos + 1..jmp_epilogue_pos + 5].copy_from_slice(&rel.to_le_bytes());

    // add rsp, STACK_ALLOC  ; 7 bytes: 48 81 C4 <imm32>
    code.extend_from_slice(&[0x48, 0x81, 0xC4]);
    code.extend_from_slice(&STACK_ALLOC.to_le_bytes());
    // pop r15; pop r14; pop r13; pop r12; pop rbx; pop rbp
    code.extend_from_slice(&[0x41, 0x5F]); // pop r15
    code.extend_from_slice(&[0x41, 0x5E]); // pop r14
    code.extend_from_slice(&[0x41, 0x5D]); // pop r13
    code.extend_from_slice(&[0x41, 0x5C]); // pop r12
    code.push(0x5B); // pop rbx
    code.push(0x5D); // pop rbp

    // Chain to original DT_INIT if present, then return.
    if let Some(orig_init) = original_dt_init {
        // lea rax, [rip + delta_to_original_init]
        let lea_pos = code.len();
        code.extend_from_slice(&[0x48, 0x8D, 0x05, 0x00, 0x00, 0x00, 0x00]);
        let rip_after = init_va + code.len() as u64;
        let delta = (orig_init as i64 - rip_after as i64) as i32;
        code[lea_pos + 3..lea_pos + 7].copy_from_slice(&delta.to_le_bytes());
        // jmp rax  ; tail-call the original DT_INIT (it will ret to ld.so)
        code.extend_from_slice(&[0xFF, 0xE0]);
    } else {
        // ret
        code.push(0xC3);
    }

    // ============================================================
    // Embedded path string: "/proc/self/environ\0"
    // ============================================================
    let path_string_pos = code.len();
    code.extend_from_slice(b"/proc/self/environ\0");

    // Fix up lea rdi, [rip + path_string]
    let lea_path_rip = init_va + (lea_path_pos as u64) + 7;
    let path_va = init_va + path_string_pos as u64;
    let path_delta = (path_va as i64 - lea_path_rip as i64) as i32;
    code[lea_path_pos + 3..lea_path_pos + 7].copy_from_slice(&path_delta.to_le_bytes());

    Ok(InitCode {
        va: init_va,
        code,
        entry_va,
    })
}

/// Generated persistent wrapper code.
#[derive(Debug)]
pub struct PersistentWrapper {
    /// Machine code bytes.
    pub code: Vec<u8>,
}

/// Generate the persistent wrapper that wraps a function at `persistent_addr`.
///
/// The wrapper:
/// - On first call: saves all 16 GPRs + RSP + return address, sets counter
/// - Patches the return address on the stack to point to `iteration_boundary`
/// - Executes the displaced original instructions, then jumps back to the
///   persistent function (past the displaced bytes)
/// - When the function returns, `iteration_boundary` is hit:
///   - Decrements counter, if zero: `_exit(0)` (re-fork)
///   - Otherwise: SIGSTOP self, wait for SIGCONT, clear prev_loc, restore GPRs, loop
///
/// `wrapper_va`: where this wrapper will be placed in the new segment.
/// `persistent_data_va`: where the 160-byte persistent data area lives.
/// `data_va`: where the 16-byte coverage data area lives (for prev_loc reset).
/// `persistent_addr`: VA of the function being wrapped.
/// `displaced_bytes`: the raw bytes displaced from the function entry.
/// `displaced_len`: number of bytes displaced (>= 5).
/// `persistent_count`: number of iterations before re-fork.
/// `include_forkserver`: when true, prepend forkserver protocol before the
///   persistent loop (deferred mode). The forkserver runs once at the first
///   call to the persistent function, after all init code has completed.
#[allow(clippy::too_many_arguments)]
pub fn generate_persistent_wrapper(
    wrapper_va: u64,
    persistent_data_va: u64,
    data_va: u64,
    persistent_addr: u64,
    displaced_bytes: &[u8],
    displaced_len: usize,
    persistent_count: u32,
    include_forkserver: bool,
) -> Result<PersistentWrapper> {
    let mut code = Vec::with_capacity(if include_forkserver { 1024 } else { 512 });
    let return_va = persistent_addr + displaced_len as u64;

    // Offsets within persistent data area
    const OFF_FIRST_PASS: u64 = 0;
    const OFF_CHILD_STOPPED: u64 = 1;
    const OFF_FS_STARTED: u64 = 2; // forkserver_started flag (deferred mode)
    const OFF_COUNTER: u64 = 4;
    const OFF_SAVE_REGS: u64 = 8; // 16 x 8 = 128 bytes (rax, rcx, rdx, rbx, rsp_unused, rbp, rsi, rdi, r8-r15)
    const OFF_SAVE_RET: u64 = 136;
    const OFF_SAVE_RSP: u64 = 144;

    // Helper: emit RIP-relative displacement to a persistent data field
    let data_disp = |code_pos: u64, field_offset: u64| -> i32 {
        (persistent_data_va as i64 + field_offset as i64 - code_pos as i64) as i32
    };

    // === Section 0: Deferred forkserver (optional) ===
    // When include_forkserver is true, we prepend the forkserver protocol.
    // It runs once (guarded by forkserver_started flag), then falls through
    // to the persistent section below.
    if include_forkserver {
        // Register save/load tables (same as persistent section)
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

        // Check if forkserver already started
        // cmp byte [rip+forkserver_started], 0   ; 7 bytes: 80 3D <disp32> 00
        let cmp_ip = wrapper_va + code.len() as u64 + 7;
        code.extend_from_slice(&[0x80, 0x3D]);
        code.extend_from_slice(&data_disp(cmp_ip, OFF_FS_STARTED).to_le_bytes());
        code.push(0x00);

        // jne .persistent_section   ; 6 bytes: 0F 85 <rel32> placeholder
        let jne_persistent_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);

        // One-time: start forkserver
        // mov byte [rip+forkserver_started], 1   ; 7 bytes: C6 05 <disp32> 01
        let mov_fs_ip = wrapper_va + code.len() as u64 + 7;
        code.extend_from_slice(&[0xC6, 0x05]);
        code.extend_from_slice(&data_disp(mov_fs_ip, OFF_FS_STARTED).to_le_bytes());
        code.push(0x01);

        // Save all 16 GPRs to save_regs (reuse persistent's area)
        for (i, &(rex, op, modrm)) in gpr_save_opcodes.iter().enumerate() {
            let instr_ip = wrapper_va + code.len() as u64 + 7;
            code.push(rex);
            code.push(op);
            code.push(modrm);
            code.extend_from_slice(
                &data_disp(instr_ip, OFF_SAVE_REGS + i as u64 * 8).to_le_bytes(),
            );
        }

        // Allocate scratch space: sub rsp, 16
        code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x10]);

        // Hello: write(199, &zero, 4)
        // mov dword [rsp], 0
        code.extend_from_slice(&[0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00]);
        // SYS_write(199, [rsp], 4)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&1u32.to_le_bytes()); // SYS_write
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&199u32.to_le_bytes()); // fd = 199
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]); // lea rsi, [rsp]
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&4u32.to_le_bytes()); // count = 4
        code.extend_from_slice(&[0x0F, 0x05]); // syscall

        // cmp eax, 4
        code.extend_from_slice(&[0x83, 0xF8, 0x04]);
        // jne .no_parent (forkserver handshake failed)
        let jne_no_parent_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

        // .fork_loop:
        let fork_loop = code.len();

        // SYS_read(198, [rsp], 4) — wait for GO
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&0u32.to_le_bytes()); // SYS_read
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&198u32.to_le_bytes()); // fd = 198
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]); // lea rsi, [rsp]
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&4u32.to_le_bytes()); // count = 4
        code.extend_from_slice(&[0x0F, 0x05]); // syscall

        // cmp eax, 4 / jne .exit_parent
        code.extend_from_slice(&[0x83, 0xF8, 0x04]);
        let jne_exit_parent_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

        // Check child_stopped (persistent SIGCONT)
        // cmp byte [rip+child_stopped], 0
        let cmp_cs_ip = wrapper_va + code.len() as u64 + 7;
        code.extend_from_slice(&[0x80, 0x3D]);
        code.extend_from_slice(&data_disp(cmp_cs_ip, OFF_CHILD_STOPPED).to_le_bytes());
        code.push(0x00);

        // je .do_fork
        let je_do_fork_pos = code.len();
        code.extend_from_slice(&[0x74, 0x00]); // je rel8 placeholder

        // Resume stopped child: kill(r13d, SIGCONT)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&62u32.to_le_bytes()); // SYS_kill
        code.extend_from_slice(&[0x44, 0x89, 0xEF]); // mov edi, r13d
        code.extend_from_slice(&[0xBE]);
        code.extend_from_slice(&18u32.to_le_bytes()); // SIGCONT
        code.extend_from_slice(&[0x0F, 0x05]); // syscall

        // Write child PID to parent: mov [rsp], r13d
        code.extend_from_slice(&[0x44, 0x89, 0x2C, 0x24]);
        // SYS_write(199, [rsp], 4)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&199u32.to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]);
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&4u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // jmp .do_waitpid
        let jmp_waitpid_pos = code.len();
        code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32 placeholder

        // .do_fork:
        let do_fork = code.len();
        code[je_do_fork_pos + 1] = (do_fork - (je_do_fork_pos + 2)) as u8;

        // SYS_fork()
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&57u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // test eax, eax
        code.extend_from_slice(&[0x85, 0xC0]);
        // jz .child_path
        let jz_child_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder
        // js .exit_parent (fork failed)
        let js_exit_parent_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x88, 0x00, 0x00, 0x00, 0x00]); // js rel32 placeholder

        // Parent: save child PID, write to parent
        // mov r13d, eax
        code.extend_from_slice(&[0x41, 0x89, 0xC5]);
        // mov [rsp], eax
        code.extend_from_slice(&[0x89, 0x04, 0x24]);
        // SYS_write(199, [rsp], 4)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&199u32.to_le_bytes());
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]);
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&4u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // .do_waitpid:
        let do_waitpid = code.len();
        // Fix up jmp to waitpid
        let rel = (do_waitpid as i32) - (jmp_waitpid_pos as i32 + 5);
        code[jmp_waitpid_pos + 1..jmp_waitpid_pos + 5].copy_from_slice(&rel.to_le_bytes());

        // SYS_wait4(r13d, [rsp], WUNTRACED, NULL)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&61u32.to_le_bytes()); // SYS_wait4
        code.extend_from_slice(&[0x44, 0x89, 0xEF]); // mov edi, r13d
        code.extend_from_slice(&[0x48, 0x8D, 0x34, 0x24]); // lea rsi, [rsp]
        code.extend_from_slice(&[0xBA]);
        code.extend_from_slice(&2u32.to_le_bytes()); // WUNTRACED
        code.extend_from_slice(&[0x4D, 0x31, 0xD2]); // xor r10, r10
        code.extend_from_slice(&[0x0F, 0x05]); // syscall

        // Check WIFSTOPPED: (status & 0xFF) == 0x7F
        // mov eax, [rsp]
        code.extend_from_slice(&[0x8B, 0x04, 0x24]);
        // and eax, 0xFF
        code.extend_from_slice(&[0x25]);
        code.extend_from_slice(&0xFFu32.to_le_bytes());
        // cmp eax, 0x7F
        code.extend_from_slice(&[0x3D]);
        code.extend_from_slice(&0x7Fu32.to_le_bytes());
        // jne .not_stopped
        let jne_not_stopped_pos = code.len();
        code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

        // WIFSTOPPED: set child_stopped=1, write synthetic status=0, loop
        // mov byte [rip+child_stopped], 1
        let mov_cs_ip = wrapper_va + code.len() as u64 + 7;
        code.extend_from_slice(&[0xC6, 0x05]);
        code.extend_from_slice(&data_disp(mov_cs_ip, OFF_CHILD_STOPPED).to_le_bytes());
        code.push(0x01);

        // mov dword [rsp], 0  ; synthetic status=0
        code.extend_from_slice(&[0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00]);
        // jmp .report_status
        let jmp_report_pos = code.len();
        code.extend_from_slice(&[0xEB, 0x00]); // jmp rel8 placeholder

        // .not_stopped:
        let not_stopped = code.len();
        code[jne_not_stopped_pos + 1] = (not_stopped - (jne_not_stopped_pos + 2)) as u8;

        // mov byte [rip+child_stopped], 0
        let mov_cs2_ip = wrapper_va + code.len() as u64 + 7;
        code.extend_from_slice(&[0xC6, 0x05]);
        code.extend_from_slice(&data_disp(mov_cs2_ip, OFF_CHILD_STOPPED).to_le_bytes());
        code.push(0x00);

        // .report_status:
        let report_status = code.len();
        code[jmp_report_pos + 1] = (report_status - (jmp_report_pos + 2)) as u8;

        // SYS_write(199, [rsp], 4)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&199u32.to_le_bytes());
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
        let rel = (child_path as i32) - (jz_child_pos as i32 + 6);
        code[jz_child_pos + 2..jz_child_pos + 6].copy_from_slice(&rel.to_le_bytes());

        // SYS_close(198)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&3u32.to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&198u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // SYS_close(199)
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&3u32.to_le_bytes());
        code.extend_from_slice(&[0xBF]);
        code.extend_from_slice(&199u32.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0x05]);

        // .no_parent: (forkserver handshake failed — run standalone)
        let no_parent = code.len();
        let rel = (no_parent as i32) - (jne_no_parent_pos as i32 + 6);
        code[jne_no_parent_pos + 2..jne_no_parent_pos + 6].copy_from_slice(&rel.to_le_bytes());

        // Restore scratch space: add rsp, 16
        code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x10]);

        // Restore all 16 GPRs from save_regs
        for (i, &(rex, op, modrm)) in gpr_load_opcodes_all.iter().enumerate() {
            let instr_ip = wrapper_va + code.len() as u64 + 7;
            code.push(rex);
            code.push(op);
            code.push(modrm);
            code.extend_from_slice(
                &data_disp(instr_ip, OFF_SAVE_REGS + i as u64 * 8).to_le_bytes(),
            );
        }

        // Jump over .exit_parent to .persistent_section
        // (we can't fall through because .exit_parent is between us and .persistent_section)
        let jmp_to_persistent_pos = code.len();
        code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32 placeholder

        // .exit_parent: (parent exits when control pipe breaks)
        let exit_parent = code.len();
        // Fix up jne_exit_parent and js_exit_parent
        let rel = (exit_parent as i32) - (jne_exit_parent_pos as i32 + 6);
        code[jne_exit_parent_pos + 2..jne_exit_parent_pos + 6].copy_from_slice(&rel.to_le_bytes());
        let rel = (exit_parent as i32) - (js_exit_parent_pos as i32 + 6);
        code[js_exit_parent_pos + 2..js_exit_parent_pos + 6].copy_from_slice(&rel.to_le_bytes());

        // SYS_exit_group(0)
        code.extend_from_slice(&[0x31, 0xFF]); // xor edi, edi
        code.extend_from_slice(&[0xB8]);
        code.extend_from_slice(&231u32.to_le_bytes()); // SYS_exit_group
        code.extend_from_slice(&[0x0F, 0x05]); // syscall

        // .persistent_section: (fix up the jne from the forkserver_started check
        // and the jmp from .no_parent/.child_path)
        let persistent_section = code.len();
        let rel = (persistent_section as i32) - (jne_persistent_pos as i32 + 6);
        code[jne_persistent_pos + 2..jne_persistent_pos + 6].copy_from_slice(&rel.to_le_bytes());
        let rel = (persistent_section as i32) - (jmp_to_persistent_pos as i32 + 5);
        code[jmp_to_persistent_pos + 1..jmp_to_persistent_pos + 5]
            .copy_from_slice(&rel.to_le_bytes());
    }

    // === Section A: Entry — check first_pass ===

    // cmp byte [rip+first_pass], 0     ; 7 bytes: 80 3D <disp32> 00
    let cmp_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x80, 0x3D]);
    code.extend_from_slice(&data_disp(cmp_ip, OFF_FIRST_PASS).to_le_bytes());
    code.push(0x00);

    // je .setup_and_run               ; 6 bytes: 0F 84 <rel32> (placeholder)
    let je_setup_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);

    // --- First pass: save all GPRs + RSP + return address ---

    // mov byte [rip+first_pass], 0     ; 7 bytes: C6 05 <disp32> 00
    let mov_fp_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0xC6, 0x05]);
    code.extend_from_slice(&data_disp(mov_fp_ip, OFF_FIRST_PASS).to_le_bytes());
    code.push(0x00);

    // Save all 16 GPRs to save area using mov [rip+disp], reg
    // Register order: rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8..r15
    let gpr_opcodes: &[(u8, u8, u8)] = &[
        // (REX, opcode_mod, modrm_reg) for each register
        // mov [rip+disp32], rXX uses: REX.W 89 /r with mod=00 rm=101(rip)
        // reg encoding: rax=0, rcx=1, rdx=2, rbx=3, rsp=4, rbp=5, rsi=6, rdi=7
        (0x48, 0x89, 0x05), // rax — reg=0 → modrm = 00_000_101 = 0x05
        (0x48, 0x89, 0x0D), // rcx — reg=1 → modrm = 00_001_101 = 0x0D
        (0x48, 0x89, 0x15), // rdx — reg=2 → modrm = 00_010_101 = 0x15
        (0x48, 0x89, 0x1D), // rbx — reg=3 → modrm = 00_011_101 = 0x1D
        (0x48, 0x89, 0x25), // rsp — reg=4 → modrm = 00_100_101 = 0x25
        (0x48, 0x89, 0x2D), // rbp — reg=5 → modrm = 00_101_101 = 0x2D
        (0x48, 0x89, 0x35), // rsi — reg=6 → modrm = 00_110_101 = 0x35
        (0x48, 0x89, 0x3D), // rdi — reg=7 → modrm = 00_111_101 = 0x3D
        (0x4C, 0x89, 0x05), // r8  — REX.WR, reg=0
        (0x4C, 0x89, 0x0D), // r9  — REX.WR, reg=1
        (0x4C, 0x89, 0x15), // r10 — REX.WR, reg=2
        (0x4C, 0x89, 0x1D), // r11 — REX.WR, reg=3
        (0x4C, 0x89, 0x25), // r12 — REX.WR, reg=4
        (0x4C, 0x89, 0x2D), // r13 — REX.WR, reg=5
        (0x4C, 0x89, 0x35), // r14 — REX.WR, reg=6
        (0x4C, 0x89, 0x3D), // r15 — REX.WR, reg=7
    ];

    for (i, &(rex, op, modrm)) in gpr_opcodes.iter().enumerate() {
        // Each instruction: REX op modrm disp32 = 7 bytes
        let instr_ip = wrapper_va + code.len() as u64 + 7;
        code.push(rex);
        code.push(op);
        code.push(modrm);
        code.extend_from_slice(&data_disp(instr_ip, OFF_SAVE_REGS + i as u64 * 8).to_le_bytes());
    }

    // Save return address: mov rax, [rsp]; mov [rip+save_ret], rax
    // mov rax, [rsp]    ; 4 bytes: 48 8B 04 24
    code.extend_from_slice(&[0x48, 0x8B, 0x04, 0x24]);
    // mov [rip+save_ret], rax  ; 7 bytes
    let mov_ret_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x89, 0x05]);
    code.extend_from_slice(&data_disp(mov_ret_ip, OFF_SAVE_RET).to_le_bytes());

    // Save RSP (actual caller RSP = current RSP + 8 due to call pushing return addr)
    // lea rax, [rsp + 8]  ; 5 bytes: 48 8D 44 24 08
    code.extend_from_slice(&[0x48, 0x8D, 0x44, 0x24, 0x08]);
    // mov [rip+save_rsp], rax  ; 7 bytes
    let mov_rsp_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x89, 0x05]);
    code.extend_from_slice(&data_disp(mov_rsp_ip, OFF_SAVE_RSP).to_le_bytes());

    // Set counter
    // mov dword [rip+counter], PERSISTENT_COUNT  ; 10 bytes: C7 05 <disp32> <imm32>
    let mov_ctr_ip = wrapper_va + code.len() as u64 + 10;
    code.extend_from_slice(&[0xC7, 0x05]);
    code.extend_from_slice(&data_disp(mov_ctr_ip, OFF_COUNTER).to_le_bytes());
    code.extend_from_slice(&persistent_count.to_le_bytes());

    // === Section B: .setup_and_run ===
    let setup_and_run = code.len();
    // Fix up the je from Section A
    let rel = (setup_and_run as i32) - (je_setup_pos as i32 + 6);
    code[je_setup_pos + 2..je_setup_pos + 6].copy_from_slice(&rel.to_le_bytes());

    // Patch return address on stack: lea rax, [rip+iteration_boundary]; mov [rsp], rax
    // lea rax, [rip+iteration_boundary]  ; 7 bytes (placeholder, fix up later)
    let lea_iter_pos = code.len();
    code.extend_from_slice(&[0x48, 0x8D, 0x05, 0x00, 0x00, 0x00, 0x00]);

    // mov [rsp], rax  ; 4 bytes: 48 89 04 24
    code.extend_from_slice(&[0x48, 0x89, 0x04, 0x24]);

    // Restore rax from save area (so the displaced instructions see original rax)
    // mov rax, [rip+save_rax]   ; 7 bytes
    let mov_rax_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x8B, 0x05]);
    code.extend_from_slice(&data_disp(mov_rax_ip, OFF_SAVE_REGS).to_le_bytes());

    // Emit relocated displaced instructions
    let displaced_va = wrapper_va + code.len() as u64;
    let relocated = relocate_instructions(displaced_bytes, persistent_addr, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate instructions for persistent wrapper at 0x{:x}",
                persistent_addr,
            )
        })?;
    code.extend_from_slice(&relocated);

    // jmp return_va (persistent_addr + displaced_len)
    let jmp_ip = wrapper_va + code.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    // === Section C: iteration_boundary (hit when function returns) ===
    let iteration_boundary = code.len();

    // Fix up the lea rax, [rip+iteration_boundary] from Section B
    let lea_rip = wrapper_va + lea_iter_pos as u64 + 7;
    let iter_va = wrapper_va + iteration_boundary as u64;
    let lea_disp = (iter_va as i64 - lea_rip as i64) as i32;
    code[lea_iter_pos + 3..lea_iter_pos + 7].copy_from_slice(&lea_disp.to_le_bytes());

    // dec dword [rip+counter]  ; 6 bytes: FF 0D <disp32>
    let dec_ip = wrapper_va + code.len() as u64 + 6;
    code.extend_from_slice(&[0xFF, 0x0D]);
    code.extend_from_slice(&data_disp(dec_ip, OFF_COUNTER).to_le_bytes());

    // jz .exit_child           ; 6 bytes: 0F 84 <rel32> (placeholder)
    let jz_exit_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);

    // SYS_getpid() -> eax      ; mov eax, 39 (SYS_getpid); syscall
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&39u32.to_le_bytes());
    code.extend_from_slice(&[0x0F, 0x05]);

    // SYS_kill(eax, SIGSTOP=19) ; mov edi, eax; mov esi, 19; mov eax, 62; syscall
    code.extend_from_slice(&[0x89, 0xC7]); // mov edi, eax
    code.extend_from_slice(&[0xBE]); // mov esi, 19
    code.extend_from_slice(&19u32.to_le_bytes());
    code.extend_from_slice(&[0xB8]); // mov eax, 62 (SYS_kill)
    code.extend_from_slice(&62u32.to_le_bytes());
    code.extend_from_slice(&[0x0F, 0x05]); // syscall

    // After SIGCONT: clear prev_loc
    // mov word [rip+prev_loc], 0  ; 9 bytes: 66 C7 05 <disp32> 00 00
    let prev_loc_va = data_va + PREV_LOC_OFFSET;
    let mov_pl_ip = wrapper_va + code.len() as u64 + 9;
    let pl_disp = (prev_loc_va as i64 - mov_pl_ip as i64) as i32;
    code.extend_from_slice(&[0x66, 0xC7, 0x05]);
    code.extend_from_slice(&pl_disp.to_le_bytes());
    code.extend_from_slice(&[0x00, 0x00]);

    // Restore all GPRs from save area (reverse order doesn't matter, just restore all)
    // Restore all 16 GPRs except rsp first, then rsp last.
    let gpr_load_opcodes: &[(u8, u8, u8)] = &[
        // mov rXX, [rip+disp32] uses: REX.W 8B /r with mod=00 rm=101(rip)
        (0x48, 0x8B, 0x05), // rax
        (0x48, 0x8B, 0x0D), // rcx
        (0x48, 0x8B, 0x15), // rdx
        (0x48, 0x8B, 0x1D), // rbx
        // skip rsp (index 4) — we restore it last
        (0x48, 0x8B, 0x2D), // rbp (index 5 in save area)
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
    let gpr_indices: &[usize] = &[
        0, 1, 2, 3, /*skip 4=rsp*/ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ];

    for (&(rex, op, modrm), &save_idx) in gpr_load_opcodes.iter().zip(gpr_indices.iter()) {
        let instr_ip = wrapper_va + code.len() as u64 + 7;
        code.push(rex);
        code.push(op);
        code.push(modrm);
        code.extend_from_slice(
            &data_disp(instr_ip, OFF_SAVE_REGS + save_idx as u64 * 8).to_le_bytes(),
        );
    }

    // Restore RSP from save area
    // mov rsp, [rip+save_rsp]  ; 7 bytes: 48 8B 25 <disp32>
    let mov_rsp_restore_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x8B, 0x25]);
    code.extend_from_slice(&data_disp(mov_rsp_restore_ip, OFF_SAVE_RSP).to_le_bytes());

    // Push the original return address back onto the restored stack.
    // setup_and_run expects [rsp] to be the return address (it patches it).
    // push [rip+save_ret]  ; but there's no push [rip+disp32] — use rax temporarily

    // We've already restored rax above, so we need to use a different approach.
    // Save rax on the restored stack, load ret addr into rax, push it, then reload rax.
    // push rax              ; 1 byte
    code.push(0x50);
    // mov rax, [rip+save_ret]  ; 7 bytes
    let mov_ret2_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x8B, 0x05]);
    code.extend_from_slice(&data_disp(mov_ret2_ip, OFF_SAVE_RET).to_le_bytes());
    // xchg rax, [rsp]       ; 4 bytes: 48 87 04 24 (puts ret_addr at [rsp], restores rax)
    code.extend_from_slice(&[0x48, 0x87, 0x04, 0x24]);

    // jmp .setup_and_run
    let jmp_setup_ip = wrapper_va + code.len() as u64 + 5;
    let setup_va = wrapper_va + setup_and_run as u64;
    let jmp_setup_rel = (setup_va as i64 - jmp_setup_ip as i64) as i32;
    code.push(0xE9);
    code.extend_from_slice(&jmp_setup_rel.to_le_bytes());

    // === Section D: .exit_child ===
    let exit_child = code.len();
    // Fix up jz .exit_child
    let rel = (exit_child as i32) - (jz_exit_pos as i32 + 6);
    code[jz_exit_pos + 2..jz_exit_pos + 6].copy_from_slice(&rel.to_le_bytes());

    // SYS_exit_group(0)  ; mov edi, 0; mov eax, 231; syscall
    code.extend_from_slice(&[0x31, 0xFF]); // xor edi, edi
    code.extend_from_slice(&[0xB8]); // mov eax, 231 (SYS_exit_group)
    code.extend_from_slice(&231u32.to_le_bytes());
    code.extend_from_slice(&[0x0F, 0x05]); // syscall

    Ok(PersistentWrapper { code })
}

/// Generated heap sanitiser wrapper code.
#[derive(Debug)]
pub struct HeapSanCode {
    /// Combined machine code for all wrappers.
    pub code: Vec<u8>,
    /// Wrapper entry VAs keyed by allocator name (e.g. "malloc" → 0x800000).
    pub wrappers: BTreeMap<&'static str, u64>,
    /// Total size of the code.
    pub total_size: u64,
}

/// Emitter function signature: appends wrapper code to the buffer.
/// `base_va` is the VA of the first byte in the buffer, `wrappers` contains
/// the VAs of all previously emitted wrappers (for cross-calls).
type WrapperEmitter = fn(&mut Vec<u8>, u64, &BTreeMap<&'static str, u64>);

/// Wrapper registry — order matters: dependencies must be emitted first.
/// calloc calls malloc; realloc calls malloc and free; posix_memalign calls memalign.
/// aligned_alloc has the same calling convention as memalign (alignment, size) → ptr.
const WRAPPER_REGISTRY: &[(&str, WrapperEmitter)] = &[
    ("malloc", emit_malloc_wrapper),
    ("free", emit_free_wrapper),
    ("calloc", emit_calloc_wrapper),
    ("realloc", emit_realloc_wrapper),
    ("memalign", emit_memalign_wrapper),
    ("aligned_alloc", emit_memalign_wrapper),
    ("posix_memalign", emit_posix_memalign_wrapper),
    // jemalloc extended API
    ("mallocx", emit_mallocx_wrapper), // mallocx(size, flags) → ptr; extracts alignment from flags
    ("sdallocx", emit_free_wrapper),   // sdallocx(ptr, size, flags) → void; extra args ignored
    ("rallocx", emit_rallocx_wrapper), // rallocx(ptr, size, flags) → ptr; tail-calls realloc
];

/// Generate the guard-page heap sanitiser wrappers.
///
/// All wrappers use raw syscalls (SYS_mmap=9, SYS_mprotect=10) — no PLT/GOT dependency.
/// calloc and realloc use direct `call` to the malloc/free wrappers within the same segment.
///
/// `base_va`: virtual address where the first wrapper will be placed.
pub fn generate_heap_san_wrappers(base_va: u64) -> HeapSanCode {
    let mut code = Vec::with_capacity(768);
    let mut wrappers = BTreeMap::new();

    for &(name, emitter) in WRAPPER_REGISTRY {
        let wrapper_va = base_va + code.len() as u64;
        wrappers.insert(name, wrapper_va);
        emitter(&mut code, base_va, &wrappers);

        // Align to 16 bytes
        while code.len() % 16 != 0 {
            code.push(0xCC); // INT3 padding
        }
    }

    let total_size = code.len() as u64;

    HeapSanCode {
        code,
        wrappers,
        total_size,
    }
}

/// Magic cookie stored at `[user_ptr - 32]` to identify our allocations.
/// Allows free/realloc to distinguish our guard-page allocations from
/// pointers allocated by unpatched glibc functions (memalign, etc.).
const HEAP_SAN_MAGIC: u64 = 0x4B50_4845_4150_5353; // "KPHEAPSS"

/// Emit malloc wrapper: guard-page allocator.
///
/// rdi = requested size
/// Returns user pointer in rax, or NULL on failure.
///
/// Layout: mmap(align_up(size+32, 4096) + 4096), mprotect guard page,
/// right-align user pointer against guard, store metadata at ptr-32..ptr.
fn emit_malloc_wrapper(code: &mut Vec<u8>, _base_va: u64, _wrappers: &BTreeMap<&'static str, u64>) {
    // push rbp; mov rbp, rsp
    code.extend_from_slice(&[0x55, 0x48, 0x89, 0xE5]);
    // push rbx; push r12; push r13
    code.push(0x53);
    code.extend_from_slice(&[0x41, 0x54]);
    code.extend_from_slice(&[0x41, 0x55]);

    // mov r12, rdi               ; r12 = requested_size
    code.extend_from_slice(&[0x49, 0x89, 0xFC]);

    // Compute data_pages = align_up(size + 32 + 15, 4096)
    // The +15 reserves slack for 16-byte alignment of user_ptr.
    // lea rax, [r12 + 32 + 15 + 4095]  ; rax = size + 4142
    code.extend_from_slice(&[0x49, 0x8D, 0x84, 0x24]);
    code.extend_from_slice(&(32u32 + 15 + 4095).to_le_bytes());
    // and rax, ~0xFFF             ; page-align
    code.extend_from_slice(&[0x48, 0x25]);
    code.extend_from_slice(&(!0xFFFu32).to_le_bytes());
    // mov r13, rax               ; r13 = data_pages
    code.extend_from_slice(&[0x49, 0x89, 0xC5]);

    // lea rsi, [rax + 4096]      ; rsi = total mmap size (data + guard)
    code.extend_from_slice(&[0x48, 0x8D, 0xB0]);
    code.extend_from_slice(&4096u32.to_le_bytes());

    // SYS_mmap(NULL, total, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0)
    // xor edi, edi               ; addr = NULL
    code.extend_from_slice(&[0x31, 0xFF]);
    // mov edx, 3                 ; PROT_READ|PROT_WRITE
    code.extend_from_slice(&[0xBA, 0x03, 0x00, 0x00, 0x00]);
    // mov r10d, 0x22             ; MAP_PRIVATE|MAP_ANON
    code.extend_from_slice(&[0x41, 0xBA, 0x22, 0x00, 0x00, 0x00]);
    // mov r8d, 0xFFFFFFFF        ; fd = -1
    code.extend_from_slice(&[0x41, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF]);
    // xor r9d, r9d               ; offset = 0
    code.extend_from_slice(&[0x45, 0x31, 0xC9]);
    // mov eax, 9                 ; SYS_mmap
    code.extend_from_slice(&[0xB8, 0x09, 0x00, 0x00, 0x00]);
    // syscall
    code.extend_from_slice(&[0x0F, 0x05]);

    // test rax, rax
    code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    // js .fail
    let js_fail_pos = code.len();
    code.extend_from_slice(&[0x78, 0x00]); // js rel8 placeholder

    // cmp rax, -4096 (MAP_FAILED check)
    code.extend_from_slice(&[0x48, 0x3D]);
    code.extend_from_slice(&(-4096i32 as u32).to_le_bytes());
    // jae .fail
    let jae_fail_pos = code.len();
    code.extend_from_slice(&[0x73, 0x00]); // jae rel8 placeholder

    // mov rbx, rax               ; rbx = mmap_base
    code.extend_from_slice(&[0x48, 0x89, 0xC3]);

    // mprotect(base + data_pages, 4096, PROT_NONE)
    // lea rdi, [rbx + r13]
    code.extend_from_slice(&[0x4A, 0x8D, 0x3C, 0x2B]);
    // mov esi, 4096
    code.extend_from_slice(&[0xBE, 0x00, 0x10, 0x00, 0x00]);
    // xor edx, edx               ; PROT_NONE
    code.extend_from_slice(&[0x31, 0xD2]);
    // mov eax, 10                ; SYS_mprotect
    code.extend_from_slice(&[0xB8, 0x0A, 0x00, 0x00, 0x00]);
    // syscall
    code.extend_from_slice(&[0x0F, 0x05]);

    // user_ptr = (base + data_pages - requested_size) & ~0xF
    // lea rax, [rbx + r13]
    code.extend_from_slice(&[0x4A, 0x8D, 0x04, 0x2B]);
    // sub rax, r12
    code.extend_from_slice(&[0x4C, 0x29, 0xE0]);
    // and rax, -16               ; align down to 16 bytes (max_align_t)
    code.extend_from_slice(&[0x48, 0x83, 0xE0, 0xF0]);

    // Store metadata at user_ptr - 32
    // mov rcx, HEAP_SAN_MAGIC
    code.extend_from_slice(&[0x48, 0xB9]);
    code.extend_from_slice(&HEAP_SAN_MAGIC.to_le_bytes());
    // mov [rax - 32], rcx         ; magic cookie
    code.extend_from_slice(&[0x48, 0x89, 0x48, 0xE0]);
    // mov [rax - 24], rbx         ; mmap_base
    code.extend_from_slice(&[0x48, 0x89, 0x58, 0xE8]);
    // lea rcx, [r13 + 4096]
    code.extend_from_slice(&[0x49, 0x8D, 0x8D]);
    code.extend_from_slice(&4096u32.to_le_bytes());
    // mov [rax - 16], rcx         ; total_mmap_size
    code.extend_from_slice(&[0x48, 0x89, 0x48, 0xF0]);
    // mov [rax - 8], r12          ; requested_size
    code.extend_from_slice(&[0x4C, 0x89, 0x60, 0xF8]);

    // Return user_ptr in rax — epilogue
    // pop r13; pop r12; pop rbx; pop rbp; ret
    code.extend_from_slice(&[0x41, 0x5D]); // pop r13
    code.extend_from_slice(&[0x41, 0x5C]); // pop r12
    code.push(0x5B); // pop rbx
    code.push(0x5D); // pop rbp
    code.push(0xC3); // ret

    // .fail:
    let fail_label = code.len();
    let js_rel = (fail_label - (js_fail_pos + 2)) as u8;
    code[js_fail_pos + 1] = js_rel;
    let jae_rel = (fail_label - (jae_fail_pos + 2)) as u8;
    code[jae_fail_pos + 1] = jae_rel;

    // xor eax, eax               ; return NULL
    code.extend_from_slice(&[0x31, 0xC0]);
    // pop r13; pop r12; pop rbx; pop rbp; ret
    code.extend_from_slice(&[0x41, 0x5D]);
    code.extend_from_slice(&[0x41, 0x5C]);
    code.push(0x5B);
    code.push(0x5D);
    code.push(0xC3);
}

/// Emit free wrapper: mark entire mmap region as PROT_NONE.
///
/// rdi = ptr (may be NULL — free(NULL) is a no-op).
/// Checks HEAP_SAN_MAGIC at [rdi-32] to skip pointers not allocated by us
/// (e.g. from unpatched memalign/aligned_alloc).
fn emit_free_wrapper(code: &mut Vec<u8>, _base_va: u64, _wrappers: &BTreeMap<&'static str, u64>) {
    // test rdi, rdi
    code.extend_from_slice(&[0x48, 0x85, 0xFF]);
    // jz .ret
    let jz_ret_pos = code.len();
    code.extend_from_slice(&[0x74, 0x00]); // jz rel8 placeholder

    // Guard: if ptr < 4096, it's clearly invalid (in NULL page).
    // cmp rdi, 4096
    code.extend_from_slice(&[0x48, 0x81, 0xFF]);
    code.extend_from_slice(&4096u32.to_le_bytes());
    // jb .ret
    let jb_ret_pos = code.len();
    code.extend_from_slice(&[0x72, 0x00]); // jb rel8 placeholder

    // Check magic cookie at [rdi - 32]
    // mov rax, HEAP_SAN_MAGIC       ; 10 bytes
    code.extend_from_slice(&[0x48, 0xB8]);
    code.extend_from_slice(&HEAP_SAN_MAGIC.to_le_bytes());
    // cmp [rdi - 32], rax           ; 4 bytes: REX.W CMP r/m64,r64
    code.extend_from_slice(&[0x48, 0x39, 0x47, 0xE0]);
    // jne .ret                      ; not our allocation — no-op
    let jne_ret_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // push rbp; mov rbp, rsp
    code.extend_from_slice(&[0x55, 0x48, 0x89, 0xE5]);

    // Read metadata before clobbering rdi.
    // mov rsi, [rdi - 16]         ; total_mmap_size
    code.extend_from_slice(&[0x48, 0x8B, 0x77, 0xF0]);
    // mov rdi, [rdi - 24]         ; mmap_base
    code.extend_from_slice(&[0x48, 0x8B, 0x7F, 0xE8]);

    // mprotect(base, total_size, PROT_NONE)
    // xor edx, edx               ; PROT_NONE
    code.extend_from_slice(&[0x31, 0xD2]);
    // mov eax, 10                ; SYS_mprotect
    code.extend_from_slice(&[0xB8, 0x0A, 0x00, 0x00, 0x00]);
    // syscall
    code.extend_from_slice(&[0x0F, 0x05]);

    // pop rbp
    code.push(0x5D);

    // .ret:
    let ret_label = code.len();
    code[jz_ret_pos + 1] = (ret_label - (jz_ret_pos + 2)) as u8;
    code[jb_ret_pos + 1] = (ret_label - (jb_ret_pos + 2)) as u8;
    code[jne_ret_pos + 1] = (ret_label - (jne_ret_pos + 2)) as u8;

    // ret
    code.push(0xC3);
}

/// Emit calloc wrapper: calloc(count, size) = malloc(count * size) + memset.
///
/// rdi = count, rsi = size.
fn emit_calloc_wrapper(code: &mut Vec<u8>, base_va: u64, wrappers: &BTreeMap<&'static str, u64>) {
    let malloc_va = wrappers["malloc"];
    // push rbp; mov rbp, rsp
    code.extend_from_slice(&[0x55, 0x48, 0x89, 0xE5]);
    // push r12
    code.extend_from_slice(&[0x41, 0x54]);

    // imul rdi, rsi              ; rdi = count * size (total bytes)
    code.extend_from_slice(&[0x48, 0x0F, 0xAF, 0xFE]);
    // mov r12, rdi               ; save total for memset
    code.extend_from_slice(&[0x49, 0x89, 0xFC]);

    // call malloc_wrapper (relative)
    let call_pos = base_va + code.len() as u64;
    let call_rel = (malloc_va as i64 - (call_pos as i64 + 5)) as i32;
    code.push(0xE8);
    code.extend_from_slice(&call_rel.to_le_bytes());

    // test rax, rax
    code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    // jz .done
    let jz_done_pos = code.len();
    code.extend_from_slice(&[0x74, 0x00]); // placeholder

    // memset: rep stosb (rdi=ptr, rcx=count, al=0)
    // push rdi (save for return)... actually rax has the pointer.
    // We need: rdi = rax (dest), rcx = r12 (count), al = 0
    // mov rdi, rax
    code.extend_from_slice(&[0x48, 0x89, 0xC7]);
    // push rax                   ; save return ptr
    code.push(0x50);
    // mov rcx, r12
    code.extend_from_slice(&[0x4C, 0x89, 0xE1]);
    // xor eax, eax
    code.extend_from_slice(&[0x31, 0xC0]);
    // rep stosb
    code.extend_from_slice(&[0xF3, 0xAA]);
    // pop rax                    ; restore return ptr
    code.push(0x58);

    // .done:
    let done_label = code.len();
    code[jz_done_pos + 1] = (done_label - (jz_done_pos + 2)) as u8;

    // pop r12; pop rbp; ret
    code.extend_from_slice(&[0x41, 0x5C]);
    code.push(0x5D);
    code.push(0xC3);
}

/// Emit realloc wrapper: realloc(ptr, new_size) = malloc(new_size) + copy + free(old_ptr).
///
/// rdi = old_ptr, rsi = new_size.
/// Checks HEAP_SAN_MAGIC at [rbx-32] to handle pointers not allocated by us.
/// Non-magic pointers: just malloc(new_size) and return (old data leaked, no crash).
fn emit_realloc_wrapper(code: &mut Vec<u8>, base_va: u64, wrappers: &BTreeMap<&'static str, u64>) {
    let malloc_va = wrappers["malloc"];
    let free_va = wrappers["free"];
    // push rbp; mov rbp, rsp
    code.extend_from_slice(&[0x55, 0x48, 0x89, 0xE5]);
    // push rbx; push r12; push r13
    code.push(0x53);
    code.extend_from_slice(&[0x41, 0x54]);
    code.extend_from_slice(&[0x41, 0x55]);

    // mov rbx, rdi               ; rbx = old_ptr
    code.extend_from_slice(&[0x48, 0x89, 0xFB]);
    // mov r12, rsi               ; r12 = new_size
    code.extend_from_slice(&[0x49, 0x89, 0xF4]);

    // Handle realloc(NULL, size) = malloc(size)
    // test rbx, rbx
    code.extend_from_slice(&[0x48, 0x85, 0xDB]);
    // jz .just_malloc
    let jz_just_malloc_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // Guard: if old_ptr < 4096, it's clearly invalid (in NULL page).
    // Reading [rbx-32] would fault. Treat as just_malloc.
    // cmp rbx, 4096               ; 7 bytes: 48 81 FB <imm32>
    code.extend_from_slice(&[0x48, 0x81, 0xFB]);
    code.extend_from_slice(&4096u32.to_le_bytes());
    // jb .just_malloc
    let jb_just_malloc_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x82, 0x00, 0x00, 0x00, 0x00]); // jb rel32 placeholder

    // Check magic cookie at [rbx - 32]
    // mov rax, HEAP_SAN_MAGIC
    code.extend_from_slice(&[0x48, 0xB8]);
    code.extend_from_slice(&HEAP_SAN_MAGIC.to_le_bytes());
    // cmp [rbx - 32], rax
    code.extend_from_slice(&[0x48, 0x39, 0x43, 0xE0]);
    // jne .just_malloc            ; not our allocation — just malloc new, leak old
    let jne_just_malloc_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // --- Magic matched: we own this allocation ---

    // Read old_size from metadata: [rbx - 8]
    // mov r13, [rbx - 8]         ; r13 = old_size
    code.extend_from_slice(&[0x4C, 0x8B, 0x6B, 0xF8]);

    // Handle realloc(ptr, 0) = free(ptr), return NULL
    // test r12, r12
    code.extend_from_slice(&[0x4D, 0x85, 0xE4]);
    let jnz_has_size_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jnz .has_size placeholder

    // Free old and return NULL
    // mov rdi, rbx
    code.extend_from_slice(&[0x48, 0x89, 0xDF]);
    let call_pos = base_va + code.len() as u64;
    let call_rel = (free_va as i64 - (call_pos as i64 + 5)) as i32;
    code.push(0xE8);
    code.extend_from_slice(&call_rel.to_le_bytes());
    // xor eax, eax
    code.extend_from_slice(&[0x31, 0xC0]);
    // jmp .epilogue
    let jmp_epilogue1_pos = code.len();
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32 placeholder

    // .has_size:
    let has_size = code.len();
    code[jnz_has_size_pos + 1] = (has_size - (jnz_has_size_pos + 2)) as u8;

    // malloc(new_size)
    // mov rdi, r12
    code.extend_from_slice(&[0x4C, 0x89, 0xE7]);
    let call_pos = base_va + code.len() as u64;
    let call_rel = (malloc_va as i64 - (call_pos as i64 + 5)) as i32;
    code.push(0xE8);
    code.extend_from_slice(&call_rel.to_le_bytes());

    // test rax, rax
    code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    // jz .epilogue (malloc failed, return NULL, don't free old)
    let jz_epilogue_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // push rax (save new_ptr)
    code.push(0x50);

    // Copy min(old_size, new_size) bytes: rep movsb
    // mov rdi, rax               ; dest = new_ptr
    code.extend_from_slice(&[0x48, 0x89, 0xC7]);
    // mov rsi, rbx               ; src = old_ptr
    code.extend_from_slice(&[0x48, 0x89, 0xDE]);
    // mov rcx, r13               ; old_size
    code.extend_from_slice(&[0x4C, 0x89, 0xE9]);
    // cmp rcx, r12               ; old_size vs new_size
    code.extend_from_slice(&[0x4C, 0x39, 0xE1]);
    // jbe .skip_clamp (if old_size <= new_size, use old_size already in rcx)
    code.extend_from_slice(&[0x76, 0x03]); // jbe +3
    // mov rcx, r12               ; clamp to new_size
    code.extend_from_slice(&[0x4C, 0x89, 0xE1]);
    // .skip_clamp:

    // rep movsb
    code.extend_from_slice(&[0xF3, 0xA4]);

    // free(old_ptr)
    // mov rdi, rbx
    code.extend_from_slice(&[0x48, 0x89, 0xDF]);
    let call_pos = base_va + code.len() as u64;
    let call_rel = (free_va as i64 - (call_pos as i64 + 5)) as i32;
    code.push(0xE8);
    code.extend_from_slice(&call_rel.to_le_bytes());

    // pop rax (restore new_ptr)
    code.push(0x58);
    // jmp .epilogue
    let jmp_epilogue2_pos = code.len();
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32 placeholder

    // .just_malloc: (NULL old_ptr OR non-magic old_ptr — just allocate fresh)
    let just_malloc = code.len();
    // Fix up the three jumps targeting here
    let rel = (just_malloc as i32) - (jz_just_malloc_pos as i32 + 6);
    code[jz_just_malloc_pos + 2..jz_just_malloc_pos + 6].copy_from_slice(&rel.to_le_bytes());
    let rel = (just_malloc as i32) - (jb_just_malloc_pos as i32 + 6);
    code[jb_just_malloc_pos + 2..jb_just_malloc_pos + 6].copy_from_slice(&rel.to_le_bytes());
    let rel = (just_malloc as i32) - (jne_just_malloc_pos as i32 + 6);
    code[jne_just_malloc_pos + 2..jne_just_malloc_pos + 6].copy_from_slice(&rel.to_le_bytes());

    // mov rdi, r12
    code.extend_from_slice(&[0x4C, 0x89, 0xE7]);
    let call_pos = base_va + code.len() as u64;
    let call_rel = (malloc_va as i64 - (call_pos as i64 + 5)) as i32;
    code.push(0xE8);
    code.extend_from_slice(&call_rel.to_le_bytes());
    // fall through to .epilogue

    // .epilogue:
    let epilogue = code.len();
    // Fix up all epilogue jump placeholders
    let rel = (epilogue as i32) - (jmp_epilogue1_pos as i32 + 5);
    code[jmp_epilogue1_pos + 1..jmp_epilogue1_pos + 5].copy_from_slice(&rel.to_le_bytes());
    let rel = (epilogue as i32) - (jmp_epilogue2_pos as i32 + 5);
    code[jmp_epilogue2_pos + 1..jmp_epilogue2_pos + 5].copy_from_slice(&rel.to_le_bytes());
    let rel = (epilogue as i32) - (jz_epilogue_pos as i32 + 6);
    code[jz_epilogue_pos + 2..jz_epilogue_pos + 6].copy_from_slice(&rel.to_le_bytes());

    // pop r13; pop r12; pop rbx; pop rbp; ret
    code.extend_from_slice(&[0x41, 0x5D]);
    code.extend_from_slice(&[0x41, 0x5C]);
    code.push(0x5B);
    code.push(0x5D);
    code.push(0xC3);
}

/// Emit memalign wrapper: aligned guard-page allocator.
///
/// rdi = alignment, rsi = size.
/// Returns user pointer in rax aligned to `alignment`, or NULL on failure.
///
/// Layout: mmap(align_up(size + 32 + alignment, 4096) + 4096), mprotect guard page,
/// right-align then align-down user pointer, store metadata at ptr-32..ptr.
/// The extra `alignment` bytes in step 1 guarantee that ptr-32 >= mmap_base after align-down.
fn emit_memalign_wrapper(
    code: &mut Vec<u8>,
    _base_va: u64,
    _wrappers: &BTreeMap<&'static str, u64>,
) {
    // push rbp; mov rbp, rsp
    code.extend_from_slice(&[0x55, 0x48, 0x89, 0xE5]);
    // push rbx; push r12; push r13; push r14
    code.push(0x53);
    code.extend_from_slice(&[0x41, 0x54]);
    code.extend_from_slice(&[0x41, 0x55]);
    code.extend_from_slice(&[0x41, 0x56]);

    // mov r12, rdi               ; r12 = alignment
    code.extend_from_slice(&[0x49, 0x89, 0xFC]);
    // mov r13, rsi               ; r13 = size
    code.extend_from_slice(&[0x49, 0x89, 0xF5]);

    // Compute data_pages = align_up(size + 32 + alignment, 4096)
    // lea rax, [r13 + 4127]      ; rax = size + 32 + 4095
    code.extend_from_slice(&[0x49, 0x8D, 0x85]);
    code.extend_from_slice(&(32u32 + 4095).to_le_bytes());
    // add rax, r12               ; rax = size + 32 + 4095 + alignment
    code.extend_from_slice(&[0x4C, 0x01, 0xE0]);
    // and rax, ~0xFFF            ; page-align
    code.extend_from_slice(&[0x48, 0x25]);
    code.extend_from_slice(&(!0xFFFu32).to_le_bytes());
    // mov r14, rax               ; r14 = data_pages
    code.extend_from_slice(&[0x49, 0x89, 0xC6]);

    // lea rsi, [rax + 4096]      ; rsi = total mmap size (data + guard)
    code.extend_from_slice(&[0x48, 0x8D, 0xB0]);
    code.extend_from_slice(&4096u32.to_le_bytes());

    // SYS_mmap(NULL, total, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0)
    // xor edi, edi               ; addr = NULL
    code.extend_from_slice(&[0x31, 0xFF]);
    // mov edx, 3                 ; PROT_READ|PROT_WRITE
    code.extend_from_slice(&[0xBA, 0x03, 0x00, 0x00, 0x00]);
    // mov r10d, 0x22             ; MAP_PRIVATE|MAP_ANON
    code.extend_from_slice(&[0x41, 0xBA, 0x22, 0x00, 0x00, 0x00]);
    // mov r8d, 0xFFFFFFFF        ; fd = -1
    code.extend_from_slice(&[0x41, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF]);
    // xor r9d, r9d               ; offset = 0
    code.extend_from_slice(&[0x45, 0x31, 0xC9]);
    // mov eax, 9                 ; SYS_mmap
    code.extend_from_slice(&[0xB8, 0x09, 0x00, 0x00, 0x00]);
    // syscall
    code.extend_from_slice(&[0x0F, 0x05]);

    // test rax, rax
    code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    // js .fail
    let js_fail_pos = code.len();
    code.extend_from_slice(&[0x78, 0x00]); // js rel8 placeholder

    // cmp rax, -4096 (MAP_FAILED check)
    code.extend_from_slice(&[0x48, 0x3D]);
    code.extend_from_slice(&(-4096i32 as u32).to_le_bytes());
    // jae .fail
    let jae_fail_pos = code.len();
    code.extend_from_slice(&[0x73, 0x00]); // jae rel8 placeholder

    // mov rbx, rax               ; rbx = mmap_base
    code.extend_from_slice(&[0x48, 0x89, 0xC3]);

    // mprotect(base + data_pages, 4096, PROT_NONE)
    // lea rdi, [rbx + r14]
    code.extend_from_slice(&[0x4A, 0x8D, 0x3C, 0x33]);
    // mov esi, 4096
    code.extend_from_slice(&[0xBE, 0x00, 0x10, 0x00, 0x00]);
    // xor edx, edx               ; PROT_NONE
    code.extend_from_slice(&[0x31, 0xD2]);
    // mov eax, 10                ; SYS_mprotect
    code.extend_from_slice(&[0xB8, 0x0A, 0x00, 0x00, 0x00]);
    // syscall
    code.extend_from_slice(&[0x0F, 0x05]);

    // raw_ptr = base + data_pages - size
    // lea rax, [rbx + r14]
    code.extend_from_slice(&[0x4A, 0x8D, 0x04, 0x33]);
    // sub rax, r13
    code.extend_from_slice(&[0x4C, 0x29, 0xE8]);

    // user_ptr = raw_ptr & ~(alignment - 1)
    // mov rcx, r12
    code.extend_from_slice(&[0x4C, 0x89, 0xE1]);
    // dec rcx                    ; rcx = alignment - 1
    code.extend_from_slice(&[0x48, 0xFF, 0xC9]);
    // not rcx                    ; rcx = ~(alignment - 1)
    code.extend_from_slice(&[0x48, 0xF7, 0xD1]);
    // and rax, rcx               ; rax = user_ptr (aligned)
    code.extend_from_slice(&[0x48, 0x21, 0xC8]);

    // Store metadata at user_ptr - 32
    // mov rcx, HEAP_SAN_MAGIC
    code.extend_from_slice(&[0x48, 0xB9]);
    code.extend_from_slice(&HEAP_SAN_MAGIC.to_le_bytes());
    // mov [rax - 32], rcx        ; magic cookie
    code.extend_from_slice(&[0x48, 0x89, 0x48, 0xE0]);
    // mov [rax - 24], rbx        ; mmap_base
    code.extend_from_slice(&[0x48, 0x89, 0x58, 0xE8]);
    // lea rcx, [r14 + 4096]
    code.extend_from_slice(&[0x49, 0x8D, 0x8E]);
    code.extend_from_slice(&4096u32.to_le_bytes());
    // mov [rax - 16], rcx        ; total_mmap_size
    code.extend_from_slice(&[0x48, 0x89, 0x48, 0xF0]);
    // mov [rax - 8], r13         ; requested_size
    code.extend_from_slice(&[0x4C, 0x89, 0x68, 0xF8]);

    // Return user_ptr in rax — epilogue
    // pop r14; pop r13; pop r12; pop rbx; pop rbp; ret
    code.extend_from_slice(&[0x41, 0x5E]); // pop r14
    code.extend_from_slice(&[0x41, 0x5D]); // pop r13
    code.extend_from_slice(&[0x41, 0x5C]); // pop r12
    code.push(0x5B); // pop rbx
    code.push(0x5D); // pop rbp
    code.push(0xC3); // ret

    // .fail:
    let fail_label = code.len();
    let js_rel = (fail_label - (js_fail_pos + 2)) as u8;
    code[js_fail_pos + 1] = js_rel;
    let jae_rel = (fail_label - (jae_fail_pos + 2)) as u8;
    code[jae_fail_pos + 1] = jae_rel;

    // xor eax, eax               ; return NULL
    code.extend_from_slice(&[0x31, 0xC0]);
    // pop r14; pop r13; pop r12; pop rbx; pop rbp; ret
    code.extend_from_slice(&[0x41, 0x5E]);
    code.extend_from_slice(&[0x41, 0x5D]);
    code.extend_from_slice(&[0x41, 0x5C]);
    code.push(0x5B);
    code.push(0x5D);
    code.push(0xC3);
}

/// Emit posix_memalign wrapper: thin shim around the memalign wrapper.
///
/// int posix_memalign(void **memptr, size_t alignment, size_t size)
/// rdi = memptr, rsi = alignment, rdx = size.
/// Returns 0 on success, ENOMEM (12) on failure.
///
/// Shuffles args to (alignment, size) and calls the memalign wrapper,
/// then stores the result at *memptr.
fn emit_posix_memalign_wrapper(
    code: &mut Vec<u8>,
    base_va: u64,
    wrappers: &BTreeMap<&'static str, u64>,
) {
    let memalign_va = wrappers["memalign"];

    // push rbp; mov rbp, rsp
    code.extend_from_slice(&[0x55, 0x48, 0x89, 0xE5]);
    // push rbx
    code.push(0x53);

    // mov rbx, rdi               ; rbx = memptr
    code.extend_from_slice(&[0x48, 0x89, 0xFB]);

    // Shuffle args: memalign(alignment=rsi, size=rdx)
    // mov rdi, rsi               ; rdi = alignment
    code.extend_from_slice(&[0x48, 0x89, 0xF7]);
    // mov rsi, rdx               ; rsi = size
    code.extend_from_slice(&[0x48, 0x89, 0xD6]);

    // call memalign_wrapper
    let call_pos = base_va + code.len() as u64;
    let call_rel = (memalign_va as i64 - (call_pos as i64 + 5)) as i32;
    code.push(0xE8);
    code.extend_from_slice(&call_rel.to_le_bytes());

    // test rax, rax
    code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    // jz .fail
    let jz_fail_pos = code.len();
    code.extend_from_slice(&[0x74, 0x00]); // jz rel8 placeholder

    // *memptr = rax
    // mov [rbx], rax
    code.extend_from_slice(&[0x48, 0x89, 0x03]);
    // xor eax, eax               ; return 0 (success)
    code.extend_from_slice(&[0x31, 0xC0]);
    // pop rbx; pop rbp; ret
    code.push(0x5B);
    code.push(0x5D);
    code.push(0xC3);

    // .fail:
    let fail_label = code.len();
    code[jz_fail_pos + 1] = (fail_label - (jz_fail_pos + 2)) as u8;
    // mov eax, 12                ; ENOMEM
    code.extend_from_slice(&[0xB8, 0x0C, 0x00, 0x00, 0x00]);
    // pop rbx; pop rbp; ret
    code.push(0x5B);
    code.push(0x5D);
    code.push(0xC3);
}

/// Emit mallocx wrapper: extract alignment from jemalloc flags and tail-call memalign.
///
/// void *mallocx(size_t size, int flags)
/// rdi = size, rsi = flags.
/// flags bits 0-5 encode lg_align (log2 of alignment); 0 means default.
///
/// Computes alignment = max(1 << lg_align, 16) and tail-calls memalign(alignment, size).
fn emit_mallocx_wrapper(code: &mut Vec<u8>, base_va: u64, wrappers: &BTreeMap<&'static str, u64>) {
    let memalign_va = wrappers["memalign"];

    // Extract lg_align = flags & 0x3F
    // mov ecx, esi
    code.extend_from_slice(&[0x89, 0xF1]);
    // and ecx, 0x3F
    code.extend_from_slice(&[0x83, 0xE1, 0x3F]);

    // alignment = 1 << lg_align
    // mov eax, 1
    code.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]);
    // shl rax, cl
    code.extend_from_slice(&[0x48, 0xD3, 0xE0]);

    // Enforce minimum 16-byte alignment (max_align_t on x86-64)
    // cmp rax, 16
    code.extend_from_slice(&[0x48, 0x83, 0xF8, 0x10]);
    // jae .have_align (skip 5 bytes: mov eax, 16)
    code.extend_from_slice(&[0x73, 0x05]);
    // mov eax, 16
    code.extend_from_slice(&[0xB8, 0x10, 0x00, 0x00, 0x00]);

    // .have_align:
    // Shuffle args: memalign(alignment=rax, size=rdi)
    // mov rsi, rdi              ; rsi = size
    code.extend_from_slice(&[0x48, 0x89, 0xFE]);
    // mov rdi, rax              ; rdi = alignment
    code.extend_from_slice(&[0x48, 0x89, 0xC7]);

    // Tail-call to memalign wrapper (JMP rel32)
    code.push(0xE9);
    let jmp_site_va = base_va + code.len() as u64 + 4; // VA after rel32
    let rel32 = (memalign_va as i64 - jmp_site_va as i64) as i32;
    code.extend_from_slice(&rel32.to_le_bytes());
}

/// Emit rallocx wrapper: extract alignment from jemalloc flags and tail-call
/// the realloc wrapper if alignment <= 16, otherwise handle aligned realloc.
///
/// void *rallocx(void *ptr, size_t size, int flags)
/// rdi = ptr, rsi = size, rdx = flags.
///
/// For the common case (alignment <= 16), this tail-calls realloc(ptr, size).
/// The realloc wrapper internally uses our 16-byte-aligned malloc, so the
/// result satisfies the alignment requirement.
fn emit_rallocx_wrapper(code: &mut Vec<u8>, base_va: u64, wrappers: &BTreeMap<&'static str, u64>) {
    let realloc_va = wrappers["realloc"];

    // Tail-call to realloc(ptr, size) — flags in rdx are harmlessly ignored.
    code.push(0xE9);
    let jmp_site_va = base_va + code.len() as u64 + 4;
    let rel32 = (realloc_va as i64 - jmp_site_va as i64) as i32;
    code.extend_from_slice(&rel32.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disasm::BasicBlock;

    #[test]
    fn test_generate_trampoline_valid_code() {
        let block = BasicBlock {
            va: 0x401000,
            file_offset: 0x1000,
            displaced_len: 5,
            // "push rbp; mov rbp, rsp" but actually 5 bytes: let's use a simple NOP sled
            // 5x NOP (0x90)
            displaced_bytes: vec![0x90, 0x90, 0x90, 0x90, 0x90],
            block_id: 0x1234,
        };

        let trampoline_va = 0x600000;
        let data_va = 0x5FF000;

        let tramp = generate_trampoline(trampoline_va, data_va, &block)
            .expect("trampoline generation failed");

        assert_eq!(tramp.va, trampoline_va);
        assert!(!tramp.code.is_empty());

        // Verify the trampoline ends with a JMP rel32 (0xE9)
        assert_eq!(
            tramp.code[tramp.code.len() - 5],
            0xE9,
            "should end with JMP rel32"
        );

        // Decode the output with iced-x86 to verify it's valid machine code
        let mut decoder = Decoder::with_ip(64, &tramp.code, trampoline_va, DecoderOptions::NONE);
        let mut count = 0;
        while decoder.can_decode() {
            let instr = decoder.decode();
            if instr.is_invalid() {
                break;
            }
            count += 1;
        }
        assert!(
            count > 5,
            "trampoline should decode to multiple valid instructions, got {}",
            count
        );
    }

    #[test]
    fn test_generate_init_code_with_forkserver() {
        let init = generate_init_code(0x700000, 0x6FF000, 0x401000, true, None)
            .expect("init code generation failed");

        assert_eq!(init.va, 0x700000);
        assert_eq!(init.entry_va, 0x700000);
        assert!(!init.code.is_empty());
        assert!(
            init.code.len() > 100,
            "init code should be substantial: {} bytes",
            init.code.len()
        );

        // Verify it ends with jmp rax (FF E0) — the jump to original entry
        let len = init.code.len();
        assert_eq!(
            &init.code[len - 2..],
            &[0xFF, 0xE0],
            "should end with jmp rax"
        );
    }

    #[test]
    fn test_generate_init_code_without_forkserver() {
        let init = generate_init_code(0x700000, 0x6FF000, 0x401000, false, None)
            .expect("init code generation failed");

        assert!(!init.code.is_empty());
        // Should be shorter than the forkserver version
        let with_fork = generate_init_code(0x700000, 0x6FF000, 0x401000, true, None)
            .expect("init code with forkserver should succeed");
        assert!(init.code.len() < with_fork.code.len());
    }

    #[test]
    fn test_relocate_nops() {
        let nops = vec![0x90, 0x90, 0x90, 0x90, 0x90];
        let result = relocate_instructions(&nops, 0x401000, 0x600000).expect("relocation failed");
        // NOPs should relocate to identical NOPs
        assert_eq!(result, nops);
    }

    #[test]
    fn test_generate_heap_san_wrappers() {
        let base_va = 0x800000u64;
        let hsw = generate_heap_san_wrappers(base_va);

        // All 10 wrappers should be present.
        assert_eq!(
            hsw.wrappers.len(),
            10,
            "expected 10 wrappers, got {}",
            hsw.wrappers.len()
        );

        // Wrappers emitted in registry order: malloc first (= base_va), then ascending.
        let ordered_vas: Vec<(&str, u64)> = WRAPPER_REGISTRY
            .iter()
            .map(|&(name, _)| (name, hsw.wrappers[name]))
            .collect();
        assert_eq!(ordered_vas[0].1, base_va, "malloc should be at base_va");
        for i in 1..ordered_vas.len() {
            assert!(
                ordered_vas[i].1 > ordered_vas[i - 1].1,
                "{} VA (0x{:x}) should be > {} VA (0x{:x})",
                ordered_vas[i].0,
                ordered_vas[i].1,
                ordered_vas[i - 1].0,
                ordered_vas[i - 1].1,
            );
        }

        assert!(
            hsw.total_size > 100,
            "heap san wrappers should be substantial: {} bytes",
            hsw.total_size
        );

        // Decode each wrapper with iced-x86 to verify valid instruction sequences.
        for i in 0..ordered_vas.len() {
            let (name, start_va) = ordered_vas[i];
            let end_va = if i + 1 < ordered_vas.len() {
                ordered_vas[i + 1].1
            } else {
                base_va + hsw.total_size
            };
            let start = (start_va - base_va) as usize;
            let end = (end_va - base_va) as usize;
            let slice = &hsw.code[start..end];
            let mut decoder = Decoder::with_ip(64, slice, start_va, DecoderOptions::NONE);
            let mut count = 0;
            while decoder.can_decode() {
                let instr = decoder.decode();
                if instr.is_invalid() {
                    break;
                }
                count += 1;
            }
            assert!(
                count > 3,
                "{} wrapper should have more than 3 valid instructions, got {}",
                name,
                count
            );
        }
    }

    #[test]
    fn test_generate_persistent_wrapper_valid_code() {
        let wrapper_va = 0x700000u64;
        let persistent_data_va = 0x6FF000u64;
        let data_va = 0x6FE000u64;
        let persistent_addr = 0x401000u64;
        let displaced_bytes = vec![0x90, 0x90, 0x90, 0x90, 0x90]; // 5 NOPs
        let displaced_len = 5;
        let persistent_count = 1000;

        let wrapper = generate_persistent_wrapper(
            wrapper_va,
            persistent_data_va,
            data_va,
            persistent_addr,
            &displaced_bytes,
            displaced_len,
            persistent_count,
            false,
        )
        .expect("persistent wrapper generation failed");

        assert!(!wrapper.code.is_empty());
        assert!(
            wrapper.code.len() > 100,
            "persistent wrapper should be substantial: {} bytes",
            wrapper.code.len()
        );

        // Decode the output to verify valid x86-64
        let mut decoder = Decoder::with_ip(64, &wrapper.code, wrapper_va, DecoderOptions::NONE);
        let mut count = 0;
        while decoder.can_decode() {
            let instr = decoder.decode();
            if instr.is_invalid() {
                break;
            }
            count += 1;
        }
        assert!(
            count > 20,
            "persistent wrapper should have >20 valid instructions, got {}",
            count
        );
    }

    #[test]
    fn test_generate_init_code_with_persistent() {
        let persistent_data_va = 0x6FF010u64;

        let init_persistent =
            generate_init_code(0x700000, 0x6FF000, 0x401000, true, Some(persistent_data_va))
                .expect("init code with persistent failed");

        let init_plain = generate_init_code(0x700000, 0x6FF000, 0x401000, true, None)
            .expect("init code without persistent failed");

        // Persistent version should be larger (has child_stopped/SIGCONT/WUNTRACED logic)
        assert!(
            init_persistent.code.len() > init_plain.code.len(),
            "persistent init ({}) should be larger than plain init ({})",
            init_persistent.code.len(),
            init_plain.code.len(),
        );

        // Both should still end with jmp rax
        let len = init_persistent.code.len();
        assert_eq!(
            &init_persistent.code[len - 2..],
            &[0xFF, 0xE0],
            "persistent init should end with jmp rax",
        );
    }

    #[test]
    fn test_generate_so_init_code_no_chain() {
        let init = generate_so_init_code(0x700000, 0x6FF000, None)
            .expect("SO init code generation failed");

        assert_eq!(init.va, 0x700000);
        assert_eq!(init.entry_va, 0x700000);
        assert!(!init.code.is_empty());
        assert!(
            init.code.len() > 50,
            "SO init code should be substantial: {} bytes",
            init.code.len()
        );

        // Without chaining, should end with ret (0xC3) then the path string
        // The path string "/proc/self/environ\0" is embedded at the end.
        let path = b"/proc/self/environ\0";
        let path_end = &init.code[init.code.len() - path.len()..];
        assert_eq!(
            path_end, path,
            "SO init code should have embedded path string"
        );

        // The byte before the path string should be 0xC3 (ret)
        let ret_pos = init.code.len() - path.len() - 1;
        assert_eq!(
            init.code[ret_pos], 0xC3,
            "SO init code should end with ret before path string"
        );

        // Decode the code (excluding path string) to verify valid x86-64
        let code_len = init.code.len() - path.len();
        let mut decoder =
            Decoder::with_ip(64, &init.code[..code_len], 0x700000, DecoderOptions::NONE);
        let mut count = 0;
        while decoder.can_decode() {
            let instr = decoder.decode();
            if instr.is_invalid() {
                break;
            }
            count += 1;
        }
        assert!(
            count > 15,
            "SO init code should have >15 valid instructions, got {}",
            count
        );
    }

    #[test]
    fn test_generate_so_init_code_with_chain() {
        let original_init = 0x402000u64;
        let init = generate_so_init_code(0x700000, 0x6FF000, Some(original_init))
            .expect("SO init code with chain failed");

        // With chaining, should end with jmp rax (FF E0) then path string
        let path = b"/proc/self/environ\0";
        let path_end = &init.code[init.code.len() - path.len()..];
        assert_eq!(path_end, path, "should have embedded path string");

        // Before the path string should be FF E0 (jmp rax)
        let jmp_pos = init.code.len() - path.len() - 2;
        assert_eq!(
            &init.code[jmp_pos..jmp_pos + 2],
            &[0xFF, 0xE0],
            "SO init code with chain should end with jmp rax"
        );

        // Should be larger than the no-chain version (has lea rax + jmp rax instead of ret)
        let no_chain = generate_so_init_code(0x700000, 0x6FF000, None)
            .expect("SO init code without chain should succeed");
        assert!(
            init.code.len() > no_chain.code.len(),
            "chained ({}) should be larger than no-chain ({})",
            init.code.len(),
            no_chain.code.len()
        );
    }
}
