//! PE-aware basic block detection for x86 and x86_64.
//!
//! Adapted from `macho_disasm.rs`. Uses PE section layout for VA-to-file-offset
//! mapping. Same algorithm as ELF/Mach-O variants. Supports both PE32 (32-bit)
//! and PE32+ (64-bit) via `PeContext.is_64bit`.

use anyhow::Result;
use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, OpKind, Register};
use std::collections::BTreeSet;

use crate::disasm::BasicBlock;
use crate::pe::{PeContext, va_to_file_offset_pe};

/// Detect basic blocks in the .text section of a PE binary.
pub fn find_basic_blocks_pe(
    ctx: &PeContext,
    data: &[u8],
    instrument_modules: &Option<Vec<String>>,
) -> Result<Vec<BasicBlock>> {
    let text = &ctx.text;
    let text_start = text.offset as usize;
    let text_end = text_start + text.size as usize;

    if text_end > data.len() {
        anyhow::bail!(
            ".text extends beyond file: offset {} + size {} > file size {}",
            text.offset,
            text.size,
            data.len()
        );
    }

    let text_bytes = &data[text_start..text_end];
    let text_va = text.va;

    // Phase 1: linear sweep to collect all instructions and branch targets.
    let bitness = if ctx.is_64bit { 64 } else { 32 };
    let mut decoder = Decoder::with_ip(bitness, text_bytes, text_va, DecoderOptions::NONE);
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

    // Add function starts from .pdata (RUNTIME_FUNCTION table).
    // This prevents displacing code that spans function boundaries.
    for &addr in &ctx.pdata_functions {
        if addr >= text_va && addr < text_va + text.size {
            branch_targets.insert(addr);
        }
    }

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.is_invalid() {
            continue;
        }
        instructions.push(instr);

        match instr.flow_control() {
            FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch => {
                if instr.op_count() >= 1 && instr.op_kind(0) == OpKind::NearBranch64 {
                    let target = instr.near_branch_target();
                    if target >= text_va && target < text_va + text.size {
                        branch_targets.insert(target);
                    }
                }
                // Instruction after a branch is a block start.
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
                if instr.op_count() >= 1 && instr.op_kind(0) == OpKind::NearBranch64 {
                    let target = instr.near_branch_target();
                    if target >= text_va && target < text_va + text.size {
                        branch_targets.insert(target);
                    }
                }
            }
            FlowControl::IndirectBranch | FlowControl::IndirectCall => {
                // Can't resolve indirect targets statically.
            }
            _ => {}
        }
    }

    // Phase 2: build basic blocks from branch targets.
    let mut blocks = Vec::new();
    let targets: Vec<u64> = branch_targets.iter().copied().collect();

    for &target in &targets {
        // Filter by module if requested.
        if let Some(modules) = instrument_modules {
            // For PE, we use export symbol names for module filtering.
            // Skip if we have module filters but this isn't an export.
            let _ = modules; // PE module filtering is optional — instrument all by default.
        }

        let file_offset = match va_to_file_offset_pe(target, ctx) {
            Some(off) => off,
            None => continue,
        };

        if file_offset >= data.len() {
            continue;
        }

        // Decode instructions at block start to get displaced bytes (>= 5 for JMP rel32).
        let remaining = &data[file_offset..std::cmp::min(file_offset + 15, data.len())];
        let mut dec = Decoder::with_ip(bitness, remaining, target, DecoderOptions::NONE);
        let mut total_len = 0usize;
        let mut displaced = Vec::new();

        while total_len < 5 && dec.can_decode() {
            let instr = dec.decode();
            if instr.is_invalid() {
                break;
            }
            let instr_len = instr.len();

            // Skip RIP-relative instructions in displacement — they need relocation
            // which is complex. Also skip instructions that would be problematic.
            if uses_rip_relative(&instr) && total_len == 0 {
                // If the very first instruction is RIP-relative, we can't easily
                // displace it. Skip this block.
                break;
            }

            let start = file_offset + total_len;
            let end = start + instr_len;
            if end > data.len() {
                break;
            }
            displaced.extend_from_slice(&data[start..end]);
            total_len += instr_len;
        }

        if total_len < 5 {
            continue;
        }

        // Verify no branch target falls inside the displaced region (after
        // the block start). This prevents overwriting code that other branches
        // target — e.g., when two branch targets are within 5 bytes of each
        // other, the JMP rel32 patches would overlap and corrupt each other.
        let has_interior_target = branch_targets
            .range((target + 1)..=(target + total_len as u64 - 1))
            .next()
            .is_some();
        if has_interior_target {
            continue;
        }

        let block_id = fnv_hash_u16(target);

        blocks.push(BasicBlock {
            va: target,
            file_offset: file_offset as u64,
            displaced_len: total_len,
            displaced_bytes: displaced,
            block_id,
        });
    }

    Ok(blocks)
}

/// Extract displaced bytes at a given VA for hook patching.
pub fn extract_displaced_bytes_pe(
    data: &[u8],
    ctx: &PeContext,
    va: u64,
    min_bytes: usize,
) -> Result<(Vec<u8>, usize)> {
    let file_offset = va_to_file_offset_pe(va, ctx)
        .ok_or_else(|| anyhow::anyhow!("VA 0x{:x} has no file mapping in PE", va))?;

    if file_offset + min_bytes > data.len() {
        anyhow::bail!(
            "VA 0x{:x} too close to end of file for {} displaced bytes",
            va,
            min_bytes,
        );
    }

    let remaining = &data[file_offset..std::cmp::min(file_offset + 15, data.len())];
    let bitness = if ctx.is_64bit { 64 } else { 32 };
    let mut dec = Decoder::with_ip(bitness, remaining, va, DecoderOptions::NONE);
    let mut total_len = 0usize;
    let mut displaced = Vec::new();

    while total_len < min_bytes && dec.can_decode() {
        let instr = dec.decode();
        if instr.is_invalid() {
            anyhow::bail!("invalid instruction at VA 0x{:x} offset +{}", va, total_len);
        }
        let instr_len = instr.len();
        let start = file_offset + total_len;
        let end = start + instr_len;
        if end > data.len() {
            break;
        }
        displaced.extend_from_slice(&data[start..end]);
        total_len += instr_len;
    }

    if total_len < min_bytes {
        anyhow::bail!(
            "could not extract {} displaced bytes at VA 0x{:x} (got {})",
            min_bytes,
            va,
            total_len,
        );
    }

    Ok((displaced, total_len))
}

/// Check if an instruction uses RIP-relative addressing.
fn uses_rip_relative(instr: &Instruction) -> bool {
    for i in 0..instr.op_count() {
        if instr.op_kind(i) == OpKind::Memory && instr.memory_base() == Register::RIP {
            return true;
        }
    }
    false
}

/// FNV-1a hash of a VA, reduced to 16 bits.
fn fnv_hash_u16(va: u64) -> u16 {
    let bytes = va.to_le_bytes();
    let mut hash: u32 = 0x811c_9dc5;
    for &b in &bytes {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    (hash ^ (hash >> 16)) as u16
}
