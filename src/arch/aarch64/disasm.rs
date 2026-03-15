use anyhow::{Context, Result};
use capstone::prelude::*;
use std::collections::{BTreeSet, HashSet};

use crate::disasm::{BasicBlock, DisassemblyResult};
use crate::traits::{BinaryContext, Disassembler};

/// AArch64 disassembler implementing the Disassembler trait.
///
/// Uses capstone-rs to perform a two-pass linear sweep over the .text section:
/// 1. Collect branch targets and detect ADRP+LDR+BR jump table patterns (forbidden regions).
/// 2. Build BasicBlock entries for all block-start addresses not in forbidden regions.
#[derive(Debug, Default)]
pub struct AArch64Disassembler;

impl AArch64Disassembler {
    pub fn new() -> Self {
        Self
    }
}

use crate::disasm::fnv_hash_u16;

impl Disassembler for AArch64Disassembler {
    fn find_basic_blocks(
        &self,
        binary_data: &[u8],
        ctx: &dyn BinaryContext,
        instrument_modules: &Option<Vec<String>>,
    ) -> Result<DisassemblyResult> {
        // Verify context is a supported type (ELF or Mach-O).
        let is_elf = ctx
            .as_any()
            .downcast_ref::<crate::elf_impl::context::ElfBinaryContext>()
            .is_some();
        let is_macho = ctx
            .as_any()
            .downcast_ref::<crate::macho_impl::MachOBinaryContext>()
            .is_some();
        if !is_elf && !is_macho {
            anyhow::bail!("AArch64Disassembler requires ElfBinaryContext or MachOBinaryContext");
        }

        let text_va = ctx.text_section_va();
        let text_offset = ctx.text_section_offset() as usize;
        let text_size = ctx.text_section_size() as usize;
        let entry_point = ctx.entry_point();
        let func_symbols = ctx.func_symbols();

        if text_offset + text_size > binary_data.len() {
            anyhow::bail!(
                ".text extends beyond file: offset {} + size {} > file size {}",
                text_offset,
                text_size,
                binary_data.len()
            );
        }

        let text_bytes = &binary_data[text_offset..text_offset + text_size];

        // Initialize capstone for AArch64.
        let cs = Capstone::new()
            .arm64()
            .mode(arch::arm64::ArchMode::Arm)
            .detail(true)
            .build()
            .context("failed to init capstone for AArch64")?;

        // Phase 1: Linear sweep to collect instructions and branch targets.
        let insns = cs
            .disasm_all(text_bytes, text_va)
            .context("capstone disassembly failed")?;

        let mut block_starts: BTreeSet<u64> = BTreeSet::new();
        let mut forbidden: HashSet<u64> = HashSet::new();

        // Seed block starts with text section start, entry point, and function symbols.
        block_starts.insert(text_va);
        if entry_point >= text_va && entry_point < text_va + text_size as u64 {
            block_starts.insert(entry_point);
        }
        for &sym_va in func_symbols {
            if sym_va >= text_va && sym_va < text_va + text_size as u64 {
                block_starts.insert(sym_va);
            }
        }

        // Collect instructions as a Vec for jump-table detection (need look-ahead).
        let insn_vec: Vec<_> = insns.iter().collect();

        for (idx, insn) in insn_vec.iter().enumerate() {
            let va = insn.address();
            let mnemonic = insn.mnemonic().unwrap_or("");

            // Classify instructions for block-start propagation.
            let is_unconditional_branch =
                matches!(mnemonic, "b" | "br" | "bl" | "blr" | "ret" | "eret");
            let is_conditional_branch =
                mnemonic.starts_with("b.") || matches!(mnemonic, "cbz" | "cbnz" | "tbz" | "tbnz");

            if is_unconditional_branch || is_conditional_branch {
                // The instruction after any branch starts a new block.
                let next_va = va + 4;
                if next_va >= text_va && next_va < text_va + text_size as u64 {
                    block_starts.insert(next_va);
                }

                // Extract branch target from operand detail.
                if let Ok(detail) = cs.insn_detail(insn) {
                    let arch_detail = detail.arch_detail();
                    let arm64_detail = arch_detail
                        .arm64()
                        .expect("instruction should have ARM64 detail when disassembled as ARM64");
                    for op in arm64_detail.operands() {
                        use capstone::arch::arm64::Arm64OperandType;
                        if let Arm64OperandType::Imm(target) = op.op_type {
                            let target_va = target as u64;
                            if target_va >= text_va && target_va < text_va + text_size as u64 {
                                block_starts.insert(target_va);
                            }
                        }
                    }
                }
            }

            // Jump table detection: ADRP+LDR+BR patterns use computed branch
            // targets that can't be statically resolved. Forbid all three.
            if mnemonic == "adrp" && idx + 2 < insn_vec.len() {
                let next_mnemonic = insn_vec[idx + 1].mnemonic().unwrap_or("");
                let next2_mnemonic = insn_vec[idx + 2].mnemonic().unwrap_or("");
                if next_mnemonic.starts_with("ldr") && next2_mnemonic == "br" {
                    forbidden.insert(va);
                    forbidden.insert(insn_vec[idx + 1].address());
                    forbidden.insert(insn_vec[idx + 2].address());
                    tracing::debug!(
                        "AArch64: jump table at 0x{:x} (adrp+ldr+br), marking forbidden",
                        va
                    );
                }
            }
        }

        // Phase 2: Build BasicBlock list.
        let mut blocks: Vec<BasicBlock> = Vec::new();

        for &block_va in &block_starts {
            if forbidden.contains(&block_va) {
                continue;
            }

            if block_va < text_va || block_va >= text_va + text_size as u64 {
                continue;
            }
            let relative_offset = (block_va - text_va) as usize;
            let file_offset = text_offset + relative_offset;

            // Need at least 4 bytes for one AArch64 instruction.
            if file_offset + 4 > binary_data.len() {
                continue;
            }

            // Check if block starts with ADRP followed by a dependent instruction
            // (ADD, LDR, STR). If so, displace both (8 bytes) to keep the pair
            // together — ADRP computes a page address that the next instruction
            // consumes, so they must be relocated as a unit.
            let raw = u32::from_le_bytes(
                binary_data[file_offset..file_offset + 4]
                    .try_into()
                    .unwrap_or([0; 4]),
            );
            let is_adrp = (raw & 0x9F00_0000) == 0x9000_0000;
            let displaced_len = if is_adrp
                && file_offset + 8 <= binary_data.len()
                && !block_starts.contains(&(block_va + 4))
            {
                // Verify the next instruction isn't a block start (which would
                // mean something branches to it independently).
                8
            } else {
                4
            };

            let displaced_bytes = binary_data[file_offset..file_offset + displaced_len].to_vec();

            blocks.push(BasicBlock {
                va: block_va,
                file_offset: file_offset as u64,
                displaced_len,
                displaced_bytes,
                block_id: fnv_hash_u16(block_va),
            });
        }

        // Apply instrument_modules filter if requested.
        if let Some(filter) = instrument_modules
            && !filter.is_empty()
        {
            let allowed_ranges: Vec<(u64, u64)> = if is_elf {
                let elf = goblin::elf::Elf::parse(binary_data)
                    .context("goblin ELF parse failed during instrument_modules filter")?;
                elf.syms
                    .iter()
                    .filter(|sym| sym.st_type() == goblin::elf::sym::STT_FUNC)
                    .filter_map(|sym| {
                        let name = elf.strtab.get_at(sym.st_name)?;
                        if filter.iter().any(|m| name.contains(m.as_str())) {
                            let end = if sym.st_size > 0 {
                                sym.st_value + sym.st_size
                            } else {
                                sym.st_value + 0x10000
                            };
                            Some((sym.st_value, end))
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                // Mach-O: use goblin Mach-O parser for symbol extraction
                let macho = goblin::mach::MachO::parse(binary_data, 0)
                    .context("goblin Mach-O parse failed during instrument_modules filter")?;
                let mut ranges = Vec::new();
                if let Some(ref symbols) = macho.symbols {
                    for (name, nlist) in symbols.iter().flatten() {
                        if nlist.is_undefined() || nlist.n_value == 0 {
                            continue;
                        }
                        if filter.iter().any(|m| name.contains(m.as_str())) {
                            let end = nlist.n_value + 0x10000;
                            ranges.push((nlist.n_value, end));
                        }
                    }
                }
                ranges
            };

            if allowed_ranges.is_empty() {
                anyhow::bail!(
                    "--instrument-modules filter matched 0 symbols for modules {:?}; \
                         check module names match symbol table (use `nm -D <binary>` to list symbols)",
                    filter
                );
            }

            let before = blocks.len();
            blocks.retain(|b| {
                allowed_ranges
                    .iter()
                    .any(|&(start, end)| b.va >= start && b.va < end)
            });

            tracing::info!(
                "--instrument-modules: {} blocks retained after filtering ({} removed)",
                blocks.len(),
                before - blocks.len(),
            );

            if blocks.is_empty() {
                anyhow::bail!(
                    "instrument_modules filter {:?} matched symbols but no blocks align",
                    filter
                );
            }
        }

        tracing::debug!(
            "AArch64Disassembler: found {} basic blocks in text section (va=0x{:x}, size={})",
            blocks.len(),
            text_va,
            text_size
        );

        Ok(DisassemblyResult {
            blocks,
            skipped: Vec::new(),
        })
    }
}
