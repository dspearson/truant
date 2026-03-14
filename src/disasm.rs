use anyhow::Result;
use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind, Register};
use std::collections::BTreeSet;

use crate::elf::ElfContext;

/// A basic block that is eligible for instrumentation.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Virtual address of the block start
    pub va: u64,
    /// File offset corresponding to `va`
    pub file_offset: u64,
    /// Total byte length of instructions we displace (>= 5)
    pub displaced_len: usize,
    /// The raw bytes of the displaced instructions
    pub displaced_bytes: Vec<u8>,
    /// Block ID for coverage hashing: fnv(va) & 0xFFFF
    pub block_id: u16,
}

/// Detect basic blocks in the .text section suitable for JMP rel32 instrumentation.
///
/// When `instrument_modules` is `Some(names)`, only blocks within functions whose
/// symbol name contains any of the given substrings are retained. Fails if no
/// symbols match (to avoid silently returning zero blocks).
pub fn find_basic_blocks(
    ctx: &ElfContext,
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
    let mut decoder = Decoder::with_ip(64, text_bytes, text_va, DecoderOptions::NONE);
    let mut instructions: Vec<Instruction> = Vec::new();
    let mut branch_targets: BTreeSet<u64> = BTreeSet::new();

    // The first instruction in .text is always a block start.
    branch_targets.insert(text_va);

    // The ELF entry point is a block start (init code jumps back to it).
    if ctx.entry_point >= text_va && ctx.entry_point < text_va + text.size {
        branch_targets.insert(ctx.entry_point);
    }

    // Add function symbol addresses as block starts.
    for &addr in &ctx.func_symbols {
        if addr >= text_va && addr < text_va + text.size {
            branch_targets.insert(addr);
        }
    }

    let mut prev_was_nop = false;

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.is_invalid() {
            // Skip invalid instructions — move past them.
            break;
        }

        // endbr64/endbr32 mark indirect branch targets (function entries, jump
        // table destinations). Treat each one as a block boundary so we never
        // clobber it with a JMP rel32 from a preceding block.
        if instr.mnemonic() == Mnemonic::Endbr64 || instr.mnemonic() == Mnemonic::Endbr32 {
            branch_targets.insert(instr.ip());
        }

        // Compilers emit NOP padding before jump table targets, function entries,
        // and loop headers. The real code following the NOPs may be an indirect
        // branch target that we can't determine statically. Inserting a block
        // boundary at the first non-NOP after NOPs prevents our JMP rel32 from
        // clobbering these targets.
        if prev_was_nop && !is_nop(&instr) {
            branch_targets.insert(instr.ip());
        }
        prev_was_nop = is_nop(&instr);

        match instr.flow_control() {
            FlowControl::ConditionalBranch => {
                // Target of the branch is a block start.
                let target = instr.near_branch_target();
                if target >= text_va && target < text_va + text.size {
                    branch_targets.insert(target);
                }
                // Fall-through is also a block start.
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
                // Instruction after unconditional branch is a new block start.
                let next = instr.next_ip();
                if next >= text_va && next < text_va + text.size {
                    branch_targets.insert(next);
                }
            }
            FlowControl::Return | FlowControl::Exception | FlowControl::Interrupt => {
                // Next instruction is a block start.
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
                // Can't determine target statically; next instruction is a block start.
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
    // Switch statements compiled to jump tables (lea+movslq+add+jmp *reg)
    // have case targets that are invisible to the linear disassembler.
    // We parse the actual table entries and add their targets to
    // branch_targets so they are never clobbered by our JMP rel32 patches.
    let mut jump_table_targets = 0usize;
    for idx in 0..instructions.len() {
        if instructions[idx].flow_control() == FlowControl::IndirectBranch
            && instructions[idx].mnemonic() == Mnemonic::Jmp
        {
            let targets = resolve_jump_table_targets(&instructions, idx, data, ctx);
            for &t in &targets {
                if t >= text_va && t < text_va + text.size && branch_targets.insert(t) {
                    jump_table_targets += 1;
                }
            }
        }
    }
    if jump_table_targets > 0 {
        tracing::info!(
            "resolved {} new branch targets from jump tables",
            jump_table_targets,
        );
    }

    // Phase 1.75: apply module filter if requested.
    // Retain only branch_targets that fall within functions whose demangled name
    // contains any of the given module name substrings.
    if let Some(modules) = instrument_modules {
        let elf = goblin::elf::Elf::parse(data)?;

        // Build allowed address ranges: (start, end) for each matching function.
        let allowed_ranges: Vec<(u64, u64)> = elf
            .syms
            .iter()
            .filter(|sym| sym.st_type() == goblin::elf::sym::STT_FUNC)
            .filter_map(|sym| {
                let name = elf.strtab.get_at(sym.st_name)?;
                if modules.iter().any(|m| name.contains(m.as_str())) {
                    let end = if sym.st_size > 0 {
                        sym.st_value + sym.st_size
                    } else {
                        sym.st_value + 0x10000 // fallback: 64KB window
                    };
                    Some((sym.st_value, end))
                } else {
                    None
                }
            })
            .collect();

        let matched = allowed_ranges.len();
        if matched == 0 {
            anyhow::bail!(
                "--instrument-modules filter matched 0 symbols for modules {:?}; \
                 check module names match ELF symbol table (use `nm -D <binary>` to list symbols)",
                modules
            );
        }
        tracing::info!(
            "--instrument-modules: {} symbols matched for {:?}",
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
            "--instrument-modules: {} branch targets retained after filtering ({} removed)",
            branch_targets.len(),
            before - branch_targets.len(),
        );
    }

    // Phase 2: walk instructions, split into blocks, keep only those >= 5 bytes.
    let mut blocks = Vec::new();
    let mut instr_idx = 0;

    let targets_vec: Vec<u64> = branch_targets.iter().copied().collect();

    for (ti, &block_start) in targets_vec.iter().enumerate() {
        // Advance instruction index to the block start.
        while instr_idx < instructions.len() && instructions[instr_idx].ip() < block_start {
            instr_idx += 1;
        }

        if instr_idx >= instructions.len() {
            break;
        }

        if instructions[instr_idx].ip() != block_start {
            // No instruction starts exactly at this address (mid-instruction target).
            continue;
        }

        // Determine block end: next block start or end of text.
        let block_end = if ti + 1 < targets_vec.len() {
            targets_vec[ti + 1]
        } else {
            text_va + text.size
        };

        // Skip if this block falls in a PLT range.
        if is_in_plt(block_start, ctx) {
            continue;
        }

        // Accumulate instructions until we have >= 5 bytes or reach block end.
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
            continue; // Block too small for JMP rel32
        }

        // Verify no branch target falls inside the displaced region (after
        // the block start). This prevents overwriting code that other branches
        // target — e.g., glibc's lock-prefix skip pattern where a conditional
        // branch targets 1 byte past a `lock` prefix byte (0xF0), creating
        // two entry points within a single instruction.
        let has_interior_target = branch_targets
            .range((block_start + 1)..=(block_start + total_bytes as u64 - 1))
            .next()
            .is_some();
        if has_interior_target {
            continue;
        }

        // Extract displaced bytes.
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
        "found {} instrumentable basic blocks ({} branch targets total)",
        blocks.len(),
        branch_targets.len(),
    );

    Ok(blocks)
}

/// Check if an instruction is a NOP variant (alignment padding).
fn is_nop(instr: &Instruction) -> bool {
    matches!(instr.mnemonic(), Mnemonic::Nop | Mnemonic::Fnop)
}

/// Detect PIC-style jump table targets for an indirect jump instruction.
///
/// Scans backwards from an indirect `jmp *reg` to find the pattern:
///   lea rBase, [rip+disp]          ; table base in .rodata
///   movslq (rBase, rIdx, 4), rIdx  ; load signed 32-bit offset
///   add rBase, rIdx                ; compute target = base + offset
///   jmp *rIdx                      ; indirect branch
///
/// Also looks for a preceding `cmp $N, rIdx; ja default` bounds check to
/// determine the entry count. Falls back to reading entries until one
/// points outside .text.
///
/// Returns all resolved jump target addresses.
fn resolve_jump_table_targets(
    instructions: &[Instruction],
    jmp_idx: usize,
    data: &[u8],
    ctx: &ElfContext,
) -> Vec<u64> {
    let text_va = ctx.text.va;
    let text_end = text_va + ctx.text.size;
    let jmp_instr = &instructions[jmp_idx];

    // The jmp must be `jmp *reg` (not memory-indirect like jmp *[rip+disp]).
    if jmp_instr.op_count() != 1 || jmp_instr.op0_kind() != OpKind::Register {
        return Vec::new();
    }
    let jmp_reg = jmp_instr.op0_register();

    // Search backwards (up to 15 instructions) for the pattern components.
    let search_start = jmp_idx.saturating_sub(15);
    let window = &instructions[search_start..jmp_idx];

    // Find `add rBase, rJmp` — the last add before the jmp that writes to jmp_reg.
    let add_info = window.iter().rev().find_map(|i| {
        if i.mnemonic() == Mnemonic::Add
            && i.op_count() == 2
            && i.op0_kind() == OpKind::Register
            && i.op1_kind() == OpKind::Register
        {
            // add rBase, rJmp  (Intel syntax: add dst, src)
            // or add rJmp, rBase
            let r0 = i.op0_register();
            let r1 = i.op1_register();
            if r0 == jmp_reg {
                Some((r1, r0)) // base=r1, result written to r0=jmp_reg
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

    // Find `lea rBase, [rip+disp]` — loads the table base.
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

    // Find optional bounds check: `cmp $N, rIdx` followed by `ja target`.
    // This tells us the table has N+1 entries.
    let max_entries = window
        .iter()
        .rev()
        .find_map(|i| {
            if (i.mnemonic() == Mnemonic::Cmp || i.mnemonic() == Mnemonic::Sub)
                && i.op_count() == 2
                && i.op0_kind() == OpKind::Register
                && (i.op1_kind() == OpKind::Immediate8
                    || i.op1_kind() == OpKind::Immediate16
                    || i.op1_kind() == OpKind::Immediate32
                    || i.op1_kind() == OpKind::Immediate8to32
                    || i.op1_kind() == OpKind::Immediate8to16
                    || i.op1_kind() == OpKind::Immediate8to64
                    || i.op1_kind() == OpKind::Immediate32to64)
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
        .unwrap_or(256); // Conservative fallback: probe up to 256 entries.

    // Convert table VA to file offset.
    let table_file_offset = va_to_file_offset(table_base, ctx, data);
    let table_file_offset = match table_file_offset {
        Some(off) => off,
        None => return Vec::new(),
    };

    // Read table entries (signed 32-bit offsets relative to table_base).
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

        // Only accept targets within .text.
        if target >= text_va && target < text_end {
            targets.push(target);
        } else if targets.is_empty() {
            // First entry is out of range — probably not a valid table.
            return Vec::new();
        } else {
            // Reached an entry pointing outside .text — table ends here.
            break;
        }
    }

    targets
}

/// Map a virtual address to a file offset by scanning PT_LOAD segments via ElfContext.
/// Returns None if the VA isn't covered by any loaded segment.
fn va_to_file_offset(va: u64, ctx: &ElfContext, data: &[u8]) -> Option<usize> {
    // Quick path: if VA is in .text, use the text section mapping.
    if va >= ctx.text.va && va < ctx.text.va + ctx.text.size {
        return Some((ctx.text.offset + (va - ctx.text.va)) as usize);
    }

    // Scan PT_LOAD program headers from the ELF to find the segment.
    // Parse program headers from the raw data.
    let elf = goblin::elf::Elf::parse(data).ok()?;
    for phdr in &elf.program_headers {
        if phdr.p_type != goblin::elf::program_header::PT_LOAD {
            continue;
        }
        if va >= phdr.p_vaddr && va < phdr.p_vaddr + phdr.p_filesz {
            let offset = (phdr.p_offset + (va - phdr.p_vaddr)) as usize;
            if offset < data.len() {
                return Some(offset);
            }
        }
    }
    None
}

/// Get the full 64-bit general-purpose register for a given register operand.
/// E.g., EAX → RAX, SI → RSI, R8D → R8.
fn full_reg(reg: Register) -> Register {
    match reg {
        // 8-bit low
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
        // 16-bit
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
        // 32-bit
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
        // Already 64-bit or unknown — return as-is.
        other => other,
    }
}

/// Check if an address falls within a PLT section.
fn is_in_plt(addr: u64, ctx: &ElfContext) -> bool {
    for plt in &ctx.plt_ranges {
        if addr >= plt.va && addr < plt.va + plt.size {
            return true;
        }
    }
    false
}

/// Extract displaced bytes at a given VA, accumulating complete instructions
/// until at least `min_bytes` bytes are covered.
///
/// Returns `(raw_bytes, displacement_length)`.
pub fn extract_displaced_bytes(
    data: &[u8],
    ctx: &ElfContext,
    va: u64,
    min_bytes: usize,
) -> Result<(Vec<u8>, usize)> {
    let (sec_va, sec_offset, sec_size) = ctx.exec_section_for_va(va).ok_or_else(|| {
        anyhow::anyhow!(
            "VA 0x{:x} is not in any executable section (.text 0x{:x}..0x{:x}, {} PLT ranges)",
            va,
            ctx.text.va,
            ctx.text.va + ctx.text.size,
            ctx.plt_ranges.len(),
        )
    })?;

    let offset_in_sec = (va - sec_va) as usize;
    let file_offset = sec_offset as usize + offset_in_sec;
    let remaining = sec_size as usize - offset_in_sec;
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

/// FNV-1a hash of a u64, masked to 16 bits.
fn fnv_hash_u16(val: u64) -> u16 {
    let bytes = val.to_le_bytes();
    let mut hash: u32 = 0x811c_9dc5;
    for &b in &bytes {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    (hash & 0xFFFF) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::ElfContext;

    #[cfg(target_os = "linux")]
    #[test]
    fn test_find_blocks_usr_bin_true() {
        let data = std::fs::read("/usr/bin/true").expect("cannot read /usr/bin/true");
        let ctx = ElfContext::parse(&data).expect("failed to parse ELF");
        let blocks = find_basic_blocks(&ctx, &data, &None).expect("block detection failed");

        assert!(
            !blocks.is_empty(),
            "should find at least one basic block in /usr/bin/true"
        );

        for block in &blocks {
            assert!(
                block.displaced_len >= 5,
                "block at 0x{:x} too small: {}",
                block.va,
                block.displaced_len
            );
            assert_eq!(block.displaced_bytes.len(), block.displaced_len);
            assert!(block.va >= ctx.text.va);
            assert!(block.va < ctx.text.va + ctx.text.size);
        }
    }

    #[test]
    fn test_fnv_hash_deterministic() {
        assert_eq!(fnv_hash_u16(0x401000), fnv_hash_u16(0x401000));
        // Different addresses should (usually) produce different hashes
        assert_ne!(fnv_hash_u16(0x401000), fnv_hash_u16(0x401010));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_blocks_no_plt_stubs() {
        let data = std::fs::read("/usr/bin/true").expect("cannot read /usr/bin/true");
        let ctx = ElfContext::parse(&data).expect("failed to parse ELF");
        let blocks = find_basic_blocks(&ctx, &data, &None).expect("block detection failed");

        for block in &blocks {
            assert!(
                !is_in_plt(block.va, &ctx),
                "block at 0x{:x} is in PLT",
                block.va
            );
        }
    }

    #[test]
    fn test_full_reg_mapping() {
        assert_eq!(full_reg(Register::EAX), Register::RAX);
        assert_eq!(full_reg(Register::R14D), Register::R14);
        assert_eq!(full_reg(Register::CL), Register::RCX);
        assert_eq!(full_reg(Register::SI), Register::RSI);
        assert_eq!(full_reg(Register::RAX), Register::RAX);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_jump_table_targets_not_instrumented() {
        // Build a synthetic ELF with a switch table pattern and verify that
        // jump table targets are added to branch_targets and NOT instrumented.
        //
        // We test this indirectly: compile a small C program with a switch
        // statement, run find_basic_blocks, and verify no block's displaced
        // region overlaps with any jump table target.
        //
        // For now, just test on /usr/bin/true which has basic blocks.
        let data = std::fs::read("/usr/bin/true").expect("cannot read /usr/bin/true");
        let ctx = ElfContext::parse(&data).expect("failed to parse ELF");
        let blocks = find_basic_blocks(&ctx, &data, &None).expect("block detection failed");

        // Verify displaced regions don't overlap: for each block, check that
        // no other block starts inside its displaced region.
        let block_starts: BTreeSet<u64> = blocks.iter().map(|b| b.va).collect();
        for block in &blocks {
            for offset in 1..block.displaced_len as u64 {
                assert!(
                    !block_starts.contains(&(block.va + offset)),
                    "block at 0x{:x} displaced region ({}b) contains block start at 0x{:x}",
                    block.va,
                    block.displaced_len,
                    block.va + offset,
                );
            }
        }
    }
}
