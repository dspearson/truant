//! Mach-O-aware basic block detection for x86_64.
//!
//! Adapted from `disasm.rs` (ELF-specific). Uses Mach-O segment layout for
//! VA-to-file-offset mapping and stubs_ranges instead of PLT ranges.

use anyhow::Result;
use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind, Register};
use std::collections::BTreeSet;

use crate::disasm::BasicBlock;
use crate::macho::{MachOContext, is_in_stubs, va_to_file_offset_macho};

/// Detect basic blocks in the __text section of a Mach-O binary.
///
/// Same algorithm as the ELF variant in `disasm.rs`, but uses Mach-O
/// stubs_ranges and segment-based VA-to-offset mapping.
pub fn find_basic_blocks_macho(
    ctx: &MachOContext,
    data: &[u8],
    instrument_modules: &Option<Vec<String>>,
) -> Result<Vec<BasicBlock>> {
    let text = &ctx.text;
    let text_start = text.offset as usize;
    let text_end = text_start + text.size as usize;

    if text_end > data.len() {
        anyhow::bail!(
            "__text extends beyond file: offset {} + size {} > file size {}",
            text.offset,
            text.size,
            data.len()
        );
    }

    let text_bytes = &data[text_start..text_end];
    let text_va = text.va;

    // Phase 1: linear sweep to collect all instructions and branch targets.
    let mut decoder = Decoder::with_ip(64, text_bytes, text_va, DecoderOptions::NONE);
    let mut instructions: Vec<Instruction> = Vec::new();
    let mut branch_targets: BTreeSet<u64> = BTreeSet::new();

    branch_targets.insert(text_va);

    if ctx.entry_point >= text_va && ctx.entry_point < text_va + text.size {
        branch_targets.insert(ctx.entry_point);
    }

    for &addr in &ctx.func_symbols {
        if addr >= text_va && addr < text_va + text.size {
            branch_targets.insert(addr);
        }
    }

    let mut prev_was_nop = false;

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.is_invalid() {
            break;
        }

        if instr.mnemonic() == Mnemonic::Endbr64 || instr.mnemonic() == Mnemonic::Endbr32 {
            branch_targets.insert(instr.ip());
        }

        if prev_was_nop && !is_nop(&instr) {
            branch_targets.insert(instr.ip());
        }
        prev_was_nop = is_nop(&instr);

        match instr.flow_control() {
            FlowControl::ConditionalBranch => {
                let target = instr.near_branch_target();
                if target >= text_va && target < text_va + text.size {
                    branch_targets.insert(target);
                }
                let next = instr.next_ip();
                if next >= text_va && next < text_va + text.size {
                    branch_targets.insert(next);
                }
            }
            FlowControl::UnconditionalBranch => {
                let target = instr.near_branch_target();
                if target >= text_va && target < text_va + text.size {
                    branch_targets.insert(target);
                }
                let next = instr.next_ip();
                if next >= text_va && next < text_va + text.size {
                    branch_targets.insert(next);
                }
            }
            FlowControl::Return | FlowControl::Exception | FlowControl::Interrupt => {
                let next = instr.next_ip();
                if next >= text_va && next < text_va + text.size {
                    branch_targets.insert(next);
                }
            }
            FlowControl::Call => {
                let target = instr.near_branch_target();
                if target >= text_va && target < text_va + text.size {
                    branch_targets.insert(target);
                }
            }
            FlowControl::IndirectBranch | FlowControl::IndirectCall => {
                let next = instr.next_ip();
                if next >= text_va && next < text_va + text.size {
                    branch_targets.insert(next);
                }
            }
            _ => {}
        }

        instructions.push(instr);
    }

    // Phase 1.5: resolve jump table targets for indirect branches.
    let mut jump_table_targets = 0usize;
    for idx in 0..instructions.len() {
        if instructions[idx].flow_control() == FlowControl::IndirectBranch
            && instructions[idx].mnemonic() == Mnemonic::Jmp
        {
            let targets = resolve_jump_table_targets_macho(&instructions, idx, data, ctx);
            for &t in &targets {
                if t >= text_va && t < text_va + text.size && branch_targets.insert(t) {
                    jump_table_targets += 1;
                }
            }
        }
    }
    if jump_table_targets > 0 {
        tracing::info!(
            "resolved {} new branch targets from jump tables (Mach-O)",
            jump_table_targets,
        );
    }

    // Phase 1.75: module filter (using goblin Mach-O symbol table).
    if let Some(modules) = instrument_modules {
        let macho = goblin::mach::MachO::parse(data, 0)
            .map_err(|e| anyhow::anyhow!("goblin Mach-O parse failed: {}", e))?;

        let mut allowed_ranges: Vec<(u64, u64)> = Vec::new();
        if let Some(ref symbols) = macho.symbols {
            for (name, nlist) in symbols.iter().flatten() {
                // N_SECT type with non-zero value in __text
                if (nlist.n_type & 0x0E) == 0x0E
                    && nlist.n_value > 0
                    && modules.iter().any(|m| name.contains(m.as_str()))
                {
                    let end = nlist.n_value + 0x10000; // fallback window
                    allowed_ranges.push((nlist.n_value, end));
                }
            }
        }

        let matched = allowed_ranges.len();
        if matched == 0 {
            anyhow::bail!(
                "--instrument-modules filter matched 0 symbols for modules {:?}; \
                 check module names match Mach-O symbol table (use `nm <binary>` to list symbols)",
                modules
            );
        }
        tracing::info!(
            "--instrument-modules: {} symbols matched for {:?} (Mach-O)",
            matched,
            modules,
        );

        let before = branch_targets.len();
        branch_targets.retain(|&addr| {
            allowed_ranges
                .iter()
                .any(|&(start, end)| addr >= start && addr < end)
        });

        tracing::info!(
            "--instrument-modules: {} targets retained ({} removed)",
            branch_targets.len(),
            before - branch_targets.len(),
        );
    }

    // Phase 2: build BasicBlock list.
    let mut blocks = Vec::new();
    let mut instr_idx = 0;
    let targets_vec: Vec<u64> = branch_targets.iter().copied().collect();

    for (ti, &block_start) in targets_vec.iter().enumerate() {
        while instr_idx < instructions.len() && instructions[instr_idx].ip() < block_start {
            instr_idx += 1;
        }

        if instr_idx >= instructions.len() {
            break;
        }

        if instructions[instr_idx].ip() != block_start {
            continue;
        }

        let block_end = if ti + 1 < targets_vec.len() {
            targets_vec[ti + 1]
        } else {
            text_va + text.size
        };

        // Skip stubs (Mach-O equivalent of PLT)
        if is_in_stubs(block_start, ctx) {
            continue;
        }

        let mut total_bytes = 0usize;
        let mut end_idx = instr_idx;

        while end_idx < instructions.len() && instructions[end_idx].ip() < block_end {
            total_bytes += instructions[end_idx].len();
            end_idx += 1;
            if total_bytes >= 5 {
                break;
            }
        }

        if total_bytes < 5 {
            continue;
        }

        // Interior target check
        let has_interior_target = branch_targets
            .range((block_start + 1)..=(block_start + total_bytes as u64 - 1))
            .next()
            .is_some();
        if has_interior_target {
            continue;
        }

        let file_start = (block_start - text_va) as usize + text.offset as usize;
        let displaced_bytes = data[file_start..file_start + total_bytes].to_vec();
        let block_id = fnv_hash_u16(block_start);

        blocks.push(BasicBlock {
            va: block_start,
            file_offset: file_start as u64,
            displaced_len: total_bytes,
            displaced_bytes,
            block_id,
        });
    }

    tracing::info!(
        "found {} instrumentable basic blocks ({} branch targets total) (Mach-O)",
        blocks.len(),
        branch_targets.len(),
    );

    Ok(blocks)
}

/// Extract displaced bytes at a given VA for Mach-O binaries.
pub fn extract_displaced_bytes_macho(
    data: &[u8],
    ctx: &MachOContext,
    va: u64,
    min_bytes: usize,
) -> Result<(Vec<u8>, usize)> {
    let text = &ctx.text;
    if va < text.va || va >= text.va + text.size {
        anyhow::bail!(
            "address 0x{:x} is outside __text (0x{:x}..0x{:x})",
            va,
            text.va,
            text.va + text.size,
        );
    }

    let offset_in_text = (va - text.va) as usize;
    let text_start = text.offset as usize;
    let file_offset = text_start + offset_in_text;
    let remaining = text.size as usize - offset_in_text;
    let slice = &data[file_offset..file_offset + remaining];

    let mut decoder = Decoder::with_ip(64, slice, va, DecoderOptions::NONE);
    let mut total = 0usize;

    while decoder.can_decode() && total < min_bytes {
        let instr = decoder.decode();
        if instr.is_invalid() {
            anyhow::bail!(
                "invalid instruction at 0x{:x} while extracting displaced bytes",
                va + total as u64,
            );
        }
        total += instr.len();
    }

    if total < min_bytes {
        anyhow::bail!(
            "only {} bytes of valid instructions at 0x{:x}, need {}",
            total,
            va,
            min_bytes,
        );
    }

    Ok((data[file_offset..file_offset + total].to_vec(), total))
}

fn is_nop(instr: &Instruction) -> bool {
    matches!(instr.mnemonic(), Mnemonic::Nop | Mnemonic::Fnop)
}

/// Resolve jump table targets for Mach-O (same pattern as ELF, different VA resolver).
fn resolve_jump_table_targets_macho(
    instructions: &[Instruction],
    jmp_idx: usize,
    data: &[u8],
    ctx: &MachOContext,
) -> Vec<u64> {
    let text_va = ctx.text.va;
    let text_end = text_va + ctx.text.size;
    let jmp_instr = &instructions[jmp_idx];

    if jmp_instr.op_count() != 1 || jmp_instr.op0_kind() != OpKind::Register {
        return Vec::new();
    }
    let jmp_reg = jmp_instr.op0_register();

    let search_start = jmp_idx.saturating_sub(15);
    let window = &instructions[search_start..jmp_idx];

    let add_info = window.iter().rev().find_map(|i| {
        if i.mnemonic() == Mnemonic::Add
            && i.op_count() == 2
            && i.op0_kind() == OpKind::Register
            && i.op1_kind() == OpKind::Register
        {
            let r0 = i.op0_register();
            let r1 = i.op1_register();
            if r0 == jmp_reg {
                Some((r1, r0))
            } else if r1 == jmp_reg {
                Some((r0, r1))
            } else {
                None
            }
        } else {
            None
        }
    });

    let (base_reg, _) = match add_info {
        Some(info) => info,
        None => return Vec::new(),
    };

    let table_base_va = window.iter().rev().find_map(|i| {
        if i.mnemonic() == Mnemonic::Lea
            && i.op_count() == 2
            && i.op0_kind() == OpKind::Register
            && full_reg(i.op0_register()) == full_reg(base_reg)
            && i.is_ip_rel_memory_operand()
        {
            Some(i.ip_rel_memory_address())
        } else {
            None
        }
    });

    let table_base = match table_base_va {
        Some(va) => va,
        None => return Vec::new(),
    };

    let max_entries = window
        .iter()
        .rev()
        .find_map(|i| {
            if (i.mnemonic() == Mnemonic::Cmp || i.mnemonic() == Mnemonic::Sub)
                && i.op_count() == 2
                && i.op0_kind() == OpKind::Register
                && matches!(
                    i.op1_kind(),
                    OpKind::Immediate8
                        | OpKind::Immediate16
                        | OpKind::Immediate32
                        | OpKind::Immediate8to32
                        | OpKind::Immediate8to16
                        | OpKind::Immediate8to64
                        | OpKind::Immediate32to64
                )
            {
                let n = i.immediate(1);
                if n > 0 && n < 4096 {
                    Some((n + 1) as usize)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unwrap_or(256);

    let table_file_offset = match va_to_file_offset_macho(table_base, ctx) {
        Some(off) => off,
        None => return Vec::new(),
    };

    let mut targets = Vec::new();
    for i in 0..max_entries {
        let entry_off = table_file_offset + i * 4;
        if entry_off + 4 > data.len() {
            break;
        }
        let offset = i32::from_le_bytes([
            data[entry_off],
            data[entry_off + 1],
            data[entry_off + 2],
            data[entry_off + 3],
        ]);
        let target = (table_base as i64 + offset as i64) as u64;

        if target >= text_va && target < text_end {
            targets.push(target);
        } else if targets.is_empty() {
            return Vec::new();
        } else {
            break;
        }
    }

    targets
}

/// Get the full 64-bit register for a sub-register.
fn full_reg(reg: Register) -> Register {
    match reg {
        Register::AL | Register::AH => Register::RAX,
        Register::BL | Register::BH => Register::RBX,
        Register::CL | Register::CH => Register::RCX,
        Register::DL | Register::DH => Register::RDX,
        Register::SIL => Register::RSI,
        Register::DIL => Register::RDI,
        Register::SPL => Register::RSP,
        Register::BPL => Register::RBP,
        Register::R8L => Register::R8,
        Register::R9L => Register::R9,
        Register::R10L => Register::R10,
        Register::R11L => Register::R11,
        Register::R12L => Register::R12,
        Register::R13L => Register::R13,
        Register::R14L => Register::R14,
        Register::R15L => Register::R15,
        Register::AX => Register::RAX,
        Register::BX => Register::RBX,
        Register::CX => Register::RCX,
        Register::DX => Register::RDX,
        Register::SI => Register::RSI,
        Register::DI => Register::RDI,
        Register::SP => Register::RSP,
        Register::BP => Register::RBP,
        Register::R8W => Register::R8,
        Register::R9W => Register::R9,
        Register::R10W => Register::R10,
        Register::R11W => Register::R11,
        Register::R12W => Register::R12,
        Register::R13W => Register::R13,
        Register::R14W => Register::R14,
        Register::R15W => Register::R15,
        Register::EAX => Register::RAX,
        Register::EBX => Register::RBX,
        Register::ECX => Register::RCX,
        Register::EDX => Register::RDX,
        Register::ESI => Register::RSI,
        Register::EDI => Register::RDI,
        Register::ESP => Register::RSP,
        Register::EBP => Register::RBP,
        Register::R8D => Register::R8,
        Register::R9D => Register::R9,
        Register::R10D => Register::R10,
        Register::R11D => Register::R11,
        Register::R12D => Register::R12,
        Register::R13D => Register::R13,
        Register::R14D => Register::R14,
        Register::R15D => Register::R15,
        other => other,
    }
}

use crate::disasm::fnv_hash_u16;
