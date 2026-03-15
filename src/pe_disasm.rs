//! PE-aware basic block detection for x86 and x86_64.
//!
//! Adapted from `macho_disasm.rs`. Uses PE section layout for VA-to-file-offset
//! mapping. Same algorithm as ELF/Mach-O variants. Supports both PE32 (32-bit)
//! and PE32+ (64-bit) via `PeContext.is_64bit`.

use anyhow::Result;
use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, OpKind, Register};
use std::collections::BTreeSet;

use crate::disasm::{BasicBlock, DisassemblyResult, SkipReason};
use crate::pe::{PeContext, va_to_file_offset_pe};

/// Check if an instruction is a NOP variant (alignment padding between functions).
fn is_nop(instr: &Instruction) -> bool {
    use iced_x86::Code;
    matches!(
        instr.code(),
        Code::Nopw   // 66 90
        | Code::Nopd  // 90
        | Code::Nop_rm16
        | Code::Nop_rm32
        | Code::Nop_rm64
    )
}

/// Check if an operand kind is a near branch (works for both 32-bit and 64-bit).
fn is_near_branch(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64
    )
}

/// Detect basic blocks in the .text section of a PE binary.
pub fn find_basic_blocks_pe(
    ctx: &PeContext,
    data: &[u8],
    instrument_modules: &Option<Vec<String>>,
) -> Result<DisassemblyResult> {
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

    // Scan ALL sections (including .text) for potential function pointers.
    // This catches indirect call targets (CRT init tables, vtables, function pointer
    // arrays, jump tables) that aren't visible via direct CALL/JMP analysis.
    // For .text, we only scan at aligned boundaries to reduce false positives.
    let ptr_size = if ctx.is_64bit { 8 } else { 4 };
    let text_end_va = text_va.saturating_add(text.size);
    for section in &ctx.sections {
        let sec_off = section.raw_offset as usize;
        let sec_end = sec_off + section.raw_size as usize;
        if sec_end > data.len() || sec_off >= data.len() {
            continue;
        }
        // Scan every byte offset — function pointer references can appear as
        // unaligned immediates in x86 instructions (e.g., mov reg, imm32).
        let mut i = sec_off;
        while i + ptr_size <= sec_end {
            let ptr_val = if ctx.is_64bit {
                u64::from_le_bytes(data[i..i + 8].try_into().unwrap_or([0; 8]))
            } else {
                u32::from_le_bytes(data[i..i + 4].try_into().unwrap_or([0; 4])) as u64
            };
            if ptr_val >= text_va && ptr_val < text_end_va {
                branch_targets.insert(ptr_val);
            }
            i += 1; // scan every byte offset for unaligned immediates
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
                if instr.op_count() >= 1 && is_near_branch(instr.op_kind(0)) {
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
                if instr.op_count() >= 1 && is_near_branch(instr.op_kind(0)) {
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
    let mut skipped: Vec<(u64, SkipReason)> = Vec::new();
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
            None => {
                skipped.push((target, SkipReason::NoFileMapping));
                continue;
            }
        };

        if file_offset >= data.len() {
            skipped.push((target, SkipReason::NoFileMapping));
            continue;
        }

        // Decode instructions at block start to get displaced bytes (>= 5 for JMP rel32).
        // Skip leading alignment NOPs — these are inter-function padding, and the real
        // function entry point after the NOPs may be an indirect call target. If the JMP
        // patch spans the NOP into the function entry, it corrupts the call target.
        let mut skip_offset = 0usize;
        let max_nop_skip = 4; // skip up to 4 bytes of alignment NOPs
        while skip_offset < max_nop_skip && file_offset + skip_offset < data.len() {
            let remaining = &data[file_offset + skip_offset
                ..std::cmp::min(file_offset + skip_offset + 15, data.len())];
            let mut peek = Decoder::with_ip(
                bitness,
                remaining,
                target + skip_offset as u64,
                DecoderOptions::NONE,
            );
            if !peek.can_decode() {
                break;
            }
            let instr = peek.decode();
            if instr.is_invalid() {
                break;
            }
            if is_nop(&instr) {
                skip_offset += instr.len();
            } else {
                break;
            }
        }
        if skip_offset > 0 && file_offset + skip_offset + 5 <= data.len() {
            // Ensure the post-NOP address is registered as a branch target so it
            // doesn't get overwritten by another block's patch.
            let adjusted_va = target + skip_offset as u64;
            if !branch_targets.contains(&adjusted_va) {
                skipped.push((target, SkipReason::AlignmentNop));
                continue;
            }
        }

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
                skipped.push((target, SkipReason::RipRelativeStart));
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
            skipped.push((target, SkipReason::TooShort));
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
            skipped.push((target, SkipReason::InteriorTarget));
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

    Ok(DisassemblyResult { blocks, skipped })
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
use crate::disasm::fnv_hash_u16;
