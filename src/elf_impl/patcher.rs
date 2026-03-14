use anyhow::{Context, Result, bail};

use crate::disasm::BasicBlock;
use crate::elf::{AllocEntryKind, ElfContext};
use crate::hook_trampoline::{self, TargetAbi};
use crate::hooks::{self, HookSource, ResolvedHook};
use crate::patcher::PatchResult;
use crate::traits::{BinaryContext, Patcher, TrampolineGenerator};
use crate::trampoline::{self, DATA_SIZE};

/// ELF-specific patcher implementing the Patcher trait.
///
/// Contains the full ELF patching implementation: segment layout, trampoline
/// generation, PT_NOTE conversion, entry point redirection, and hook application.
pub struct ElfPatcher {
    tramp_gen: Box<dyn TrampolineGenerator>,
}

impl std::fmt::Debug for ElfPatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElfPatcher")
            .field("tramp_gen", &self.tramp_gen)
            .finish()
    }
}

impl ElfPatcher {
    pub fn new(tramp_gen: Box<dyn TrampolineGenerator>) -> Self {
        Self { tramp_gen }
    }
}

impl Patcher for ElfPatcher {
    fn patch(
        &self,
        ctx: &dyn BinaryContext,
        blocks: &[BasicBlock],
        data: Vec<u8>,
        enable_forkserver: bool,
        enable_heap_san: bool,
        persistent_addr: Option<u64>,
        persistent_count: u32,
        defer: bool,
        hooks: &[ResolvedHook],
        no_coverage: bool,
    ) -> Result<PatchResult> {
        let elf_ctx = ctx
            .as_any()
            .downcast_ref::<crate::elf_impl::context::ElfBinaryContext>()
            .ok_or_else(|| anyhow::anyhow!("ElfPatcher requires ElfBinaryContext"))?;
        patch(
            elf_ctx.inner(),
            blocks,
            data,
            enable_forkserver,
            enable_heap_san,
            persistent_addr,
            persistent_count,
            defer,
            &*self.tramp_gen,
            hooks,
            no_coverage,
        )
    }
}

/// Align a value up to the given alignment.
fn align_up(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

/// Orchestrate the entire patching process:
/// 1. Compute new segment layout
/// 2. Generate init code + trampolines
/// 3. Patch basic blocks with JMP rel32
/// 4. Convert PT_NOTE → PT_LOAD
/// 5. Redirect entry point
#[allow(clippy::too_many_arguments)]
pub fn patch(
    ctx: &ElfContext,
    blocks: &[BasicBlock],
    mut data: Vec<u8>,
    enable_forkserver: bool,
    enable_heap_san: bool,
    persistent_addr: Option<u64>,
    persistent_count: u32,
    defer: bool,
    tramp_gen: &dyn TrampolineGenerator,
    resolved_hooks: &[ResolvedHook],
    no_coverage: bool,
) -> Result<PatchResult> {
    if blocks.is_empty() && !no_coverage {
        bail!("no basic blocks to instrument");
    }

    // Compute new segment VA, page-aligned and with a gap.
    let segment_va = align_up(ctx.highest_va_end, 0x1000) + 0x1000;

    // Layout within the new segment:
    // [data area: 16 bytes] [persistent_data: 160 (if enabled)] [init code]
    // [trampoline_0] ... [trampoline_N] [persistent_wrapper (if enabled)] [heap_san]
    let data_va = segment_va;

    // For shared objects, skip persistent/forkserver — those are handled by the harness.
    let persistent_data_va = if !ctx.is_shared_object && persistent_addr.is_some() {
        Some(segment_va + DATA_SIZE)
    } else {
        None
    };

    // Hook data area: N * 8 bytes for handler address slots (library symbol hooks).
    let num_hook_data_slots = hooks::count_library_hooks(resolved_hooks);
    let hook_data_size = (num_hook_data_slots as u64) * 8;

    let hook_data_va = if persistent_data_va.is_some() {
        segment_va + DATA_SIZE + trampoline::PERSISTENT_DATA_SIZE
    } else {
        segment_va + DATA_SIZE
    };

    // Toggle area: one byte per hook (aligned up to 8).
    let toggle_area_size = if resolved_hooks.is_empty() {
        0u64
    } else {
        ((resolved_hooks.len() as u64) + 7) & !7
    };
    let toggle_data_va = hook_data_va + hook_data_size;

    // Return-address slot area: 8 bytes per Return mode hook (for saving/restoring RA).
    let return_slot_count = hooks::count_return_hooks(resolved_hooks);
    let return_slot_area_size = (return_slot_count as u64) * 8;
    let return_slot_va = toggle_data_va + toggle_area_size;

    let init_va = return_slot_va + return_slot_area_size;

    // --- Init code + coverage trampolines (skip in no-coverage mode) ---
    let init_code;
    let mut trampolines: Vec<(&BasicBlock, trampoline::Trampoline)> = Vec::new();
    let mut blocks_skipped = 0;
    let mut current_va;

    if no_coverage {
        // No-coverage mode: no SHM init, no coverage trampolines.
        // Segment starts right after the hook data area.
        init_code = trampoline::InitCode {
            va: init_va,
            code: Vec::new(),
            entry_va: init_va,
        };
        current_va = init_va;
    } else {
        // Generate init code first to know its size.
        // Shared objects use a different init code that reads /proc/self/environ
        // instead of walking the ELF entry stack (which is not available in
        // DT_INIT context).
        init_code = if ctx.is_shared_object {
            tramp_gen
                .generate_so_init_code(init_va, data_va, ctx.dt_init)
                .context("failed to generate .so init code")?
        } else {
            tramp_gen
                .generate_init_code(
                    init_va,
                    data_va,
                    ctx.entry_point,
                    if defer { false } else { enable_forkserver },
                    persistent_data_va,
                )
                .context("failed to generate init code")?
        };

        let mut trampolines_start_va = init_va + init_code.code.len() as u64;
        // Align trampoline start to 16 bytes for cleanliness.
        trampolines_start_va = align_up(trampolines_start_va, 16);

        // Generate trampolines for each block.
        trampolines.reserve(blocks.len());
        current_va = trampolines_start_va;

        for block in blocks {
            match tramp_gen.generate_trampoline(current_va, data_va, block) {
                Ok(tramp) => {
                    current_va += tramp.code.len() as u64;
                    // Align next trampoline to 8 bytes.
                    current_va = align_up(current_va, 8);
                    trampolines.push((block, tramp));
                }
                Err(e) => {
                    tracing::warn!("skipping block at 0x{:x}: {}", block.va, e);
                    blocks_skipped += 1;
                }
            }
        }

        if trampolines.is_empty() {
            bail!("all basic blocks failed trampoline generation");
        }
    }

    // --- Persistent wrapper (optional, not for .so files) ---
    let is_aarch64 = tramp_gen.branch_instruction_size() == 4;
    let effective_persistent_addr = if ctx.is_shared_object {
        None
    } else {
        persistent_addr
    };
    let persistent_wrapper = if let Some(p_addr) = effective_persistent_addr {
        current_va = align_up(current_va, 16);
        let pd_va = persistent_data_va.unwrap();
        let min_displaced = if is_aarch64 { 4 } else { 5 };
        let (displaced_bytes, displaced_len) =
            crate::disasm::extract_displaced_bytes(&data, ctx, p_addr, min_displaced)
                .context("failed to extract displaced bytes at persistent_addr")?;

        let wrapper = if is_aarch64 {
            #[cfg(feature = "aarch64")]
            {
                crate::arch::aarch64::generate_persistent_wrapper_aarch64(
                    current_va,
                    pd_va,
                    data_va,
                    p_addr,
                    &displaced_bytes,
                    displaced_len,
                    persistent_count,
                    defer,
                )
                .context("failed to generate AArch64 persistent wrapper")?
            }
            #[cfg(not(feature = "aarch64"))]
            {
                bail!("AArch64 persistent mode requires --features aarch64")
            }
        } else {
            trampoline::generate_persistent_wrapper(
                current_va,
                pd_va,
                data_va,
                p_addr,
                &displaced_bytes,
                displaced_len,
                persistent_count,
                defer,
            )
            .context("failed to generate persistent wrapper")?
        };

        let wrapper_va = current_va;
        current_va += wrapper.code.len() as u64;

        tracing::info!(
            "persistent wrapper: {} bytes at 0x{:x}, target=0x{:x}, displaced={} bytes",
            wrapper.code.len(),
            wrapper_va,
            p_addr,
            displaced_len,
        );

        Some((wrapper, wrapper_va, p_addr, displaced_len))
    } else {
        None
    };

    // --- Heap sanitiser wrappers (optional) ---
    let mut heap_san_intercepted = 0;
    let heap_san_code = if enable_heap_san {
        let alloc_syms = crate::elf::find_allocator_symbols(ctx, &data);
        if alloc_syms.get("malloc").is_some() && alloc_syms.get("free").is_some() {
            // Align current_va to 16 for wrapper placement.
            current_va = align_up(current_va, 16);
            let hsw = trampoline::generate_heap_san_wrappers(current_va);
            current_va += hsw.total_size;
            tracing::info!(
                "heap san: {} allocator functions found, wrappers at 0x{:x}",
                alloc_syms.count(),
                *hsw.wrappers.values().min().unwrap(),
            );
            Some((alloc_syms, hsw))
        } else {
            tracing::warn!(
                "heap san: malloc/free not found (found {} allocator symbols), skipping",
                alloc_syms.count(),
            );
            None
        }
    } else {
        None
    };

    // --- Relocated .dynamic (when .so has no DT_INIT and insufficient DT_NULLs) ---
    // We copy .dynamic into our new segment with a DT_INIT entry injected, then
    // update PT_DYNAMIC to point here.
    if ctx.is_shared_object {
        tracing::debug!(
            "DT_INIT relocation check: dt_init_offset={:?}, dt_null_count={}, dynamic_file_offset={:?}, dynamic_size={:?}",
            ctx.dt_init_offset,
            ctx.dt_null_count,
            ctx.dynamic_file_offset,
            ctx.dynamic_size,
        );
    }
    let relocated_dynamic: Option<(u64, Vec<u8>)> = if let (
        true,
        None,
        true,
        Some(dyn_file_off),
        Some(dyn_size),
    ) = (
        ctx.is_shared_object,
        ctx.dt_init_offset,
        ctx.dt_null_count < 2,
        ctx.dynamic_file_offset,
        ctx.dynamic_size,
    ) {
        current_va = align_up(current_va, 8);
        let reloc_va = current_va;
        let dyn_off = dyn_file_off as usize;
        let dyn_sz = dyn_size as usize;
        let end = (dyn_off + dyn_sz).min(data.len());
        let mut relocated = data[dyn_off..end].to_vec();

        // Find the DT_NULL entry (last 16-byte record with tag==0) and replace
        // it with DT_INIT pointing to our init code VA.
        let mut found_null = false;
        let mut pos = 0;
        while pos + 16 <= relocated.len() {
            let tag = i64::from_le_bytes(relocated[pos..pos + 8].try_into().unwrap());
            if tag == goblin::elf::dynamic::DT_NULL as i64 {
                // Replace with DT_INIT
                let dt_init_tag = goblin::elf::dynamic::DT_INIT as i64;
                relocated[pos..pos + 8].copy_from_slice(&dt_init_tag.to_le_bytes());
                relocated[pos + 8..pos + 16].copy_from_slice(&init_code.entry_va.to_le_bytes());
                found_null = true;
                break;
            }
            pos += 16;
        }
        if found_null {
            // Append new DT_NULL terminator
            relocated.extend_from_slice(&[0u8; 16]);
            current_va += relocated.len() as u64;
            tracing::info!(
                "relocated .dynamic ({} bytes) to new segment at VA 0x{:x} with synthesised DT_INIT=0x{:x}",
                relocated.len(),
                reloc_va,
                init_code.entry_va,
            );
            Some((reloc_va, relocated))
        } else {
            tracing::warn!("could not find DT_NULL in .dynamic for relocation");
            None
        }
    } else {
        None
    };

    // --- Hook trampolines + shellcode blobs ---
    let mut hook_trampolines: Vec<(usize, trampoline::Trampoline)> = Vec::new(); // (hook_index, tramp)
    let mut shellcode_blobs: Vec<(u64, Vec<u8>)> = Vec::new(); // (va, bytes)
    let mut return_trampolines: Vec<trampoline::Trampoline> = Vec::new(); // return-hook-only
    let mut hooks_applied = 0usize;

    if !resolved_hooks.is_empty() {
        // First pass: embed shellcode blobs and record their VAs.
        // We need to know shellcode VAs before generating trampolines.
        current_va = align_up(current_va, 16);
        let mut shellcode_va_map: Vec<Option<u64>> = Vec::with_capacity(resolved_hooks.len());
        for hook in resolved_hooks {
            if let HookSource::Shellcode(ref sc) = hook.source {
                let sc_va = current_va;
                shellcode_blobs.push((sc_va, sc.clone()));
                current_va += sc.len() as u64;
                current_va = align_up(current_va, 8);
                shellcode_va_map.push(Some(sc_va));
            } else {
                shellcode_va_map.push(None);
            }
        }

        // Second pass: group hooks by target_va, generate trampolines.
        // Hooks sharing the same VA are chained into a single trampoline.
        current_va = align_up(current_va, 16);
        let is_aarch64 = tramp_gen.branch_instruction_size() == 4;
        let hook_abi = if is_aarch64 {
            TargetAbi::Aarch64
        } else {
            TargetAbi::SysV64
        };

        // Build VA -> [indices] map (preserving declaration order).
        let mut va_groups: std::collections::BTreeMap<u64, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, hook) in resolved_hooks.iter().enumerate() {
            va_groups.entry(hook.target_va).or_default().push(i);
        }

        for indices in va_groups.values() {
            if indices.len() == 1 {
                // Single hook — use standard generator.
                let i = indices[0];
                let hook = &resolved_hooks[i];
                let sc_va = shellcode_va_map[i];
                let toggle_va = Some(toggle_data_va + hook.toggle_index as u64);

                if hook.mode == hooks::HookMode::Return {
                    // Return mode: generate entry + return trampoline pair.
                    let entry_va = current_va;
                    // Conservative estimate for entry trampoline size (aligned to 8).
                    let estimated_entry_size = 256u64;
                    let ret_tramp_va = align_up(entry_va + estimated_entry_size, 8);
                    let slot_va = return_slot_va + hook.return_slot_index.unwrap() as u64 * 8;

                    match hook_trampoline::generate_return_hook_trampolines(
                        entry_va,
                        ret_tramp_va,
                        hook,
                        hook_data_va,
                        sc_va,
                        hook_abi,
                        toggle_va,
                        slot_va,
                    ) {
                        Ok((entry_tramp, ret_tramp)) => {
                            // Verify entry trampoline fits within estimate.
                            assert!(
                                (entry_tramp.code.len() as u64) <= estimated_entry_size,
                                "return hook entry trampoline ({} bytes) exceeded estimate ({})",
                                entry_tramp.code.len(),
                                estimated_entry_size,
                            );
                            // Entry trampoline used for site patching.
                            hook_trampolines.push((i, entry_tramp));
                            // Return trampoline only written to segment data (not site-patched).
                            current_va = ret_tramp.va + ret_tramp.code.len() as u64;
                            current_va = align_up(current_va, 8);
                            return_trampolines.push(ret_tramp);
                            hooks_applied += 1;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "return hook at 0x{:x}: trampoline generation failed: {}",
                                hook.target_va,
                                e,
                            );
                        }
                    }
                } else {
                    match hook_trampoline::generate_hook_trampoline(
                        current_va,
                        hook,
                        hook_data_va,
                        sc_va,
                        hook_abi,
                        toggle_va,
                    ) {
                        Ok(tramp) => {
                            current_va += tramp.code.len() as u64;
                            current_va = align_up(current_va, 8);
                            hook_trampolines.push((i, tramp));
                            hooks_applied += 1;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "hook at 0x{:x}: trampoline generation failed: {}",
                                hook.target_va,
                                e,
                            );
                        }
                    }
                }
            } else {
                // Chained hooks — generate a single trampoline calling all handlers.
                let chain_hooks: Vec<&crate::hooks::ResolvedHook> =
                    indices.iter().map(|&i| &resolved_hooks[i]).collect();
                let chain_sc_vas: Vec<Option<u64>> =
                    indices.iter().map(|&i| shellcode_va_map[i]).collect();
                let chain_toggle_vas: Vec<Option<u64>> = indices
                    .iter()
                    .map(|&i| Some(toggle_data_va + resolved_hooks[i].toggle_index as u64))
                    .collect();
                match hook_trampoline::generate_chained_hook_trampoline(
                    current_va,
                    &chain_hooks,
                    hook_data_va,
                    &chain_sc_vas,
                    hook_abi,
                    &chain_toggle_vas,
                ) {
                    Ok(tramp) => {
                        current_va += tramp.code.len() as u64;
                        current_va = align_up(current_va, 8);
                        // Use first index as representative for site patching.
                        hook_trampolines.push((indices[0], tramp));
                        hooks_applied += indices.len();
                    }
                    Err(e) => {
                        tracing::warn!(
                            "chained hook at 0x{:x}: trampoline generation failed: {}",
                            resolved_hooks[indices[0]].target_va,
                            e,
                        );
                    }
                }
            }
        }
    }

    let segment_end_va = current_va;
    let segment_size = segment_end_va - segment_va;

    // Build the new segment data.
    let mut segment_data = vec![0u8; segment_size as usize];

    // Data area: 16 bytes (shm_ptr=0, prev_loc=0) — already zeroed.

    // Toggle area: write initial enabled values (1 = enabled, 0 = disabled).
    for hook in resolved_hooks {
        let byte_offset = (toggle_data_va - segment_va) as usize + hook.toggle_index;
        if byte_offset < segment_data.len() {
            segment_data[byte_offset] = if hook.initial_enabled { 1 } else { 0 };
        }
    }

    // Persistent data area (if enabled): initialise first_pass=1
    if let Some(pd_va) = persistent_data_va {
        let pd_offset = (pd_va - segment_va) as usize;
        segment_data[pd_offset] = 1; // first_pass = 1
        // child_stopped (offset +1) = 0 — already zeroed
    }

    // Init code
    let init_offset = (init_va - segment_va) as usize;
    segment_data[init_offset..init_offset + init_code.code.len()].copy_from_slice(&init_code.code);

    // Trampolines
    for (_, tramp) in &trampolines {
        let offset = (tramp.va - segment_va) as usize;
        segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
    }

    // Persistent wrapper
    if let Some((ref wrapper, wrapper_va, _, _)) = persistent_wrapper {
        let offset = (wrapper_va - segment_va) as usize;
        segment_data[offset..offset + wrapper.code.len()].copy_from_slice(&wrapper.code);
    }

    // Relocated .dynamic
    if let Some((reloc_va, ref reloc_data)) = relocated_dynamic {
        let offset = (reloc_va - segment_va) as usize;
        segment_data[offset..offset + reloc_data.len()].copy_from_slice(reloc_data);
    }

    // Heap san wrappers
    if let Some((_, ref hsw)) = heap_san_code {
        let base = *hsw.wrappers.values().min().unwrap();
        let offset = (base - segment_va) as usize;
        segment_data[offset..offset + hsw.code.len()].copy_from_slice(&hsw.code);
    }

    // Shellcode blobs
    for (sc_va, sc_bytes) in &shellcode_blobs {
        let offset = (*sc_va - segment_va) as usize;
        segment_data[offset..offset + sc_bytes.len()].copy_from_slice(sc_bytes);
    }

    // Hook trampolines
    for (_, tramp) in &hook_trampolines {
        let offset = (tramp.va - segment_va) as usize;
        segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
    }

    // Return trampolines (return-hook-specific, not site-patched)
    for tramp in &return_trampolines {
        let offset = (tramp.va - segment_va) as usize;
        segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
    }

    // Patch basic blocks in the original binary with a branch to the trampoline.
    let branch_size = tramp_gen.branch_instruction_size();
    for (block, tramp) in &trampolines {
        let file_offset = block.file_offset as usize;

        match tramp_gen.encode_branch(block.va, tramp.va) {
            Ok(branch_bytes) => {
                data[file_offset..file_offset + branch_bytes.len()].copy_from_slice(&branch_bytes);

                // NOP-pad remaining displaced bytes (x86_64: 0x90 per byte).
                // On AArch64, displaced_len == branch_size == 4, so this loop is a no-op.
                for i in branch_size..block.displaced_len {
                    data[file_offset + i] = 0x90;
                }
            }
            Err(e) => {
                tracing::warn!(
                    "block at 0x{:x}: branch encoding failed: {}, skipping",
                    block.va,
                    e
                );
            }
        }
    }

    // Patch hook target sites with branch to hook trampolines.
    for (hook_idx, tramp) in &hook_trampolines {
        let hook = &resolved_hooks[*hook_idx];
        let file_offset = hook.file_offset as usize;

        match tramp_gen.encode_branch(hook.target_va, tramp.va) {
            Ok(branch_bytes) => {
                data[file_offset..file_offset + branch_bytes.len()].copy_from_slice(&branch_bytes);
                // NOP-pad remaining displaced bytes (x86_64: 0x90, AArch64: displaced_len == branch_size).
                for i in branch_size..hook.displaced_len {
                    data[file_offset + i] = 0x90;
                }
                tracing::info!(
                    "hook: patched 0x{:x} → branch 0x{:x} (mode={:?})",
                    hook.target_va,
                    tramp.va,
                    hook.mode,
                );
            }
            Err(e) => {
                tracing::warn!(
                    "hook at 0x{:x}: branch encoding failed: {}, skipping",
                    hook.target_va,
                    e,
                );
            }
        }
    }

    // Patch persistent_addr with branch to wrapper.
    if let Some((_, wrapper_va, p_addr, displaced_len)) = persistent_wrapper {
        let text = &ctx.text;
        let file_offset = (p_addr - text.va) as usize + text.offset as usize;

        if is_aarch64 {
            // ARM64: encode B imm26 (4 bytes)
            match tramp_gen.encode_branch(p_addr, wrapper_va) {
                Ok(branch_bytes) => {
                    data[file_offset..file_offset + branch_bytes.len()]
                        .copy_from_slice(&branch_bytes);
                    // NOP-fill remaining displaced bytes (ARM64 NOP = 0x1F2003D5)
                    for i in (4..displaced_len).step_by(4) {
                        data[file_offset + i..file_offset + i + 4]
                            .copy_from_slice(&[0x1F, 0x20, 0x03, 0xD5]);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "persistent entry at 0x{:x}: branch encoding failed: {}",
                        p_addr,
                        e,
                    );
                }
            }
        } else {
            // x86_64: JMP rel32 (5 bytes)
            let jmp_source = p_addr + 5;
            let rel32 = (wrapper_va as i64 - jmp_source as i64) as i32;
            data[file_offset] = 0xE9;
            data[file_offset + 1..file_offset + 5].copy_from_slice(&rel32.to_le_bytes());
            for i in 5..displaced_len {
                data[file_offset + i] = 0x90; // NOP
            }
        }

        tracing::info!(
            "patched persistent entry at 0x{:x} → branch to 0x{:x}",
            p_addr,
            wrapper_va,
        );
    }

    // Patch allocator entries for heap san.
    if let Some((ref alloc_syms, ref hsw)) = heap_san_code {
        for (name, alloc) in &alloc_syms.entries {
            let wrapper_va = match hsw.wrappers.get(name) {
                Some(&va) => va,
                None => continue,
            };
            let off = alloc.patch_offset as usize;
            match alloc.kind {
                AllocEntryKind::GotPlt => {
                    // Overwrite 8-byte GOT pointer with wrapper VA.
                    if off + 8 <= data.len() {
                        data[off..off + 8].copy_from_slice(&wrapper_va.to_le_bytes());
                        heap_san_intercepted += 1;
                        tracing::info!(
                            "heap san: patched {} GOT at 0x{:x} → 0x{:x}",
                            name,
                            alloc.patch_va,
                            wrapper_va
                        );
                    }
                }
                AllocEntryKind::FuncEntry => {
                    // Overwrite first 5 bytes with JMP rel32 to wrapper.
                    if off + 5 <= data.len() {
                        let rel32 = (wrapper_va as i64 - (alloc.patch_va as i64 + 5)) as i32;
                        data[off] = 0xE9;
                        data[off + 1..off + 5].copy_from_slice(&rel32.to_le_bytes());
                        heap_san_intercepted += 1;
                        tracing::info!(
                            "heap san: patched {} entry at 0x{:x} → 0x{:x}",
                            name,
                            alloc.patch_va,
                            wrapper_va
                        );
                    }
                }
            }
        }
    }

    // Pad file to page boundary before appending new segment.
    // The ELF spec requires p_offset % p_align == p_vaddr % p_align.
    // Since segment_va is page-aligned, p_offset must also be page-aligned.
    let page_aligned_end = align_up(data.len() as u64, 0x1000);
    let padding = (page_aligned_end - data.len() as u64) as usize;
    data.extend(std::iter::repeat_n(0u8, padding));
    let file_end = data.len() as u64;

    // Add a PT_LOAD program header for our new segment.
    // Prefer converting an existing PT_NOTE (non-destructive). If none exists,
    // append a new phdr entry at the end of the program header table.
    if let Some(ref note) = ctx.note_segment {
        patch_phdr_note_to_load(
            &mut data,
            note.phdr_offset,
            segment_va,
            file_end,
            segment_size,
        )?;
    } else {
        // Append a new program header entry after the existing table.
        let new_phdr_offset = ctx.e_phoff + (ctx.e_phnum as u64) * ctx.phdr_entry_size;
        let phdr_end = new_phdr_offset + ctx.phdr_entry_size;

        // Check there's space (must not overlap with section data).
        if phdr_end as usize > data.len() {
            bail!(
                "cannot append program header: offset 0x{:x} extends beyond file (0x{:x} bytes)",
                phdr_end,
                data.len(),
            );
        }

        // Write the new PT_LOAD entry.
        patch_phdr_note_to_load(
            &mut data,
            new_phdr_offset,
            segment_va,
            file_end,
            segment_size,
        )?;

        // Increment e_phnum in the ELF header (offset 56 in ELF64).
        let phnum_offset = 56usize;
        let new_phnum = ctx.e_phnum + 1;
        data[phnum_offset..phnum_offset + 2].copy_from_slice(&new_phnum.to_le_bytes());

        tracing::info!(
            "appended new PT_LOAD program header (e_phnum {} → {})",
            ctx.e_phnum,
            new_phnum,
        );
    }

    // Redirect entry point to our init code (skip in no-coverage mode).
    if !no_coverage {
        if ctx.is_shared_object {
            // For .so files, patch DT_INIT or synthesise one from a spare DT_NULL.
            if let Some(offset) = ctx.dt_init_offset {
                // Existing DT_INIT — overwrite its d_val with our init code VA.
                let offset = offset as usize;
                if offset + 8 <= data.len() {
                    data[offset..offset + 8].copy_from_slice(&init_code.entry_va.to_le_bytes());
                    tracing::info!("patched DT_INIT to 0x{:x}", init_code.entry_va);
                } else {
                    tracing::warn!("DT_INIT offset out of bounds, cannot patch");
                }
            } else if ctx.dt_null_count >= 2 {
                // No DT_INIT exists — synthesise one by converting a DT_NULL entry.
                // DT_NULL terminates the .dynamic array, so we need at least 2
                // (one to convert, one to remain as terminator).
                if let Some(null_offset) = ctx.first_dt_null_offset {
                    let off = null_offset as usize;
                    if off + 16 <= data.len() {
                        // Write DT_INIT tag (12) into d_tag field
                        let dt_init_tag = goblin::elf::dynamic::DT_INIT;
                        data[off..off + 8].copy_from_slice(&(dt_init_tag as i64).to_le_bytes());
                        // Write init code VA into d_val field
                        data[off + 8..off + 16].copy_from_slice(&init_code.entry_va.to_le_bytes());
                        tracing::info!(
                            "synthesised DT_INIT from DT_NULL at offset 0x{:x}, init VA=0x{:x}",
                            null_offset,
                            init_code.entry_va,
                        );
                    } else {
                        tracing::warn!("DT_NULL offset out of bounds, cannot synthesise DT_INIT");
                    }
                }
            } else if let Some((reloc_va, ref reloc_data)) = relocated_dynamic {
                // .dynamic was relocated to our new segment with DT_INIT injected.
                // Update PT_DYNAMIC to point to the relocated copy.
                if let Some(pt_dyn_off) = ctx.pt_dynamic_phdr_offset {
                    let off = pt_dyn_off as usize;
                    if off + 56 <= data.len() {
                        // ELF64 Phdr layout (56 bytes):
                        //   +0:  p_type   (4 bytes)  — keep as PT_DYNAMIC
                        //   +4:  p_flags  (4 bytes)  — keep
                        //   +8:  p_offset (8 bytes)  — update to new file offset
                        //   +16: p_vaddr  (8 bytes)  — update to reloc_va
                        //   +24: p_paddr  (8 bytes)  — update to reloc_va
                        //   +32: p_filesz (8 bytes)  — update to new size
                        //   +40: p_memsz  (8 bytes)  — update to new size
                        //   +48: p_align  (8 bytes)  — keep
                        let new_filesz = reloc_data.len() as u64;
                        // File offset: segment appended at file_end, reloc_va offset within segment
                        let new_file_offset = file_end + (reloc_va - segment_va);
                        data[off + 8..off + 16].copy_from_slice(&new_file_offset.to_le_bytes());
                        data[off + 16..off + 24].copy_from_slice(&reloc_va.to_le_bytes());
                        data[off + 24..off + 32].copy_from_slice(&reloc_va.to_le_bytes());
                        data[off + 32..off + 40].copy_from_slice(&new_filesz.to_le_bytes());
                        data[off + 40..off + 48].copy_from_slice(&new_filesz.to_le_bytes());
                        tracing::info!(
                            "updated PT_DYNAMIC to relocated .dynamic at VA=0x{:x}, offset=0x{:x}, size={}",
                            reloc_va,
                            new_file_offset,
                            new_filesz,
                        );
                    } else {
                        tracing::warn!("PT_DYNAMIC phdr offset out of bounds");
                    }
                } else {
                    tracing::warn!(
                        "no PT_DYNAMIC program header found — cannot update relocated .dynamic"
                    );
                }
            } else {
                tracing::warn!(
                    "no DT_INIT and insufficient DT_NULL slots ({}) — .so coverage init will not run",
                    ctx.dt_null_count,
                );
            }
        } else {
            // For executables, patch e_entry.
            let offset = ctx.e_entry_offset as usize;
            data[offset..offset + 8].copy_from_slice(&init_code.entry_va.to_le_bytes());
            tracing::info!("patched e_entry to 0x{:x}", init_code.entry_va);
        }
    }

    // Append the new segment data to the file.
    data.extend_from_slice(&segment_data);

    // Strip BTI from GNU_PROPERTY notes on AArch64.
    // Our instrumented trampolines lack BTI landing pads, so BTI enforcement
    // would cause SIGILL on indirect branches into the new segment.
    strip_bti_property(&mut data);

    let blocks_instrumented = trampolines.len();

    tracing::info!(
        "patched {} blocks, new segment at 0x{:x} ({} bytes){}",
        blocks_instrumented,
        segment_va,
        segment_size,
        if heap_san_intercepted > 0 {
            format!(", heap san: {} functions intercepted", heap_san_intercepted)
        } else {
            String::new()
        },
    );

    let hook_data_va = segment_va + DATA_SIZE;
    let num_hook_slots = hooks::count_library_hooks(resolved_hooks);
    let hook_data_size = (num_hook_slots * 8) as u64;
    let toggle_data_va = hook_data_va + hook_data_size;

    Ok(PatchResult {
        data,
        blocks_instrumented,
        blocks_skipped,
        segment_va,
        segment_size,
        heap_san_intercepted,
        hooks_applied,
        hook_data_va,
        toggle_data_va,
    })
}

/// Clear the BTI bit from any GNU_PROPERTY AArch64 feature note.
///
/// AArch64 binaries compiled with BTI have a GNU_PROPERTY_TYPE_0 note containing
/// GNU_PROPERTY_AARCH64_FEATURE_1_AND (type 0xc0000000) with bit 0 = BTI.
/// Our instrumentation trampolines lack BTI landing pads, so we clear this bit
/// to prevent SIGILL on indirect branches into the instrumented segment.
fn strip_bti_property(data: &mut [u8]) {
    const NT_GNU_PROPERTY_TYPE_0: u32 = 5;
    const AARCH64_FEATURE_1_AND: u32 = 0xc000_0000;
    const AARCH64_FEATURE_1_BTI: u32 = 1;
    let gnu_magic = b"GNU\0";

    // Scan for GNU note headers: [namesz=4][descsz][type=5]["GNU\0"][desc...]
    let mut pos = 0;
    while pos + 16 + 4 <= data.len() {
        let namesz = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        let descsz = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
        let ntype = u32::from_le_bytes(data[pos + 8..pos + 12].try_into().unwrap());

        if namesz == 4
            && ntype == NT_GNU_PROPERTY_TYPE_0
            && pos + 16 <= data.len()
            && data[pos + 12..pos + 16] == *gnu_magic
        {
            // Parse properties within the desc (starting at pos+16, length descsz).
            let desc_start = pos + 16;
            let desc_end = desc_start + descsz as usize;
            if desc_end > data.len() {
                break;
            }
            let mut pp = desc_start;
            while pp + 8 <= desc_end {
                let pr_type = u32::from_le_bytes(data[pp..pp + 4].try_into().unwrap());
                let pr_datasz = u32::from_le_bytes(data[pp + 4..pp + 8].try_into().unwrap());
                if pr_type == AARCH64_FEATURE_1_AND && pr_datasz >= 4 && pp + 8 + 4 <= desc_end {
                    let flags_off = pp + 8;
                    let flags =
                        u32::from_le_bytes(data[flags_off..flags_off + 4].try_into().unwrap());
                    if flags & AARCH64_FEATURE_1_BTI != 0 {
                        let new_flags = flags & !AARCH64_FEATURE_1_BTI;
                        data[flags_off..flags_off + 4].copy_from_slice(&new_flags.to_le_bytes());
                        tracing::info!(
                            "stripped BTI from GNU_PROPERTY at offset 0x{:x} (flags 0x{:x} → 0x{:x})",
                            flags_off,
                            flags,
                            new_flags
                        );
                        return;
                    }
                }
                // Advance: 8 (header) + datasz, aligned to 8 bytes.
                let prop_size = 8 + ((pr_datasz as usize + 7) & !7);
                pp += prop_size;
            }
        }
        pos += 4; // notes are 4-byte aligned
    }
}

/// Overwrite a PT_NOTE program header entry to become PT_LOAD (public for tests).
#[cfg(test)]
pub(crate) fn patch_phdr_note_to_load_test(
    data: &mut [u8],
    phdr_offset: u64,
    segment_va: u64,
    file_offset: u64,
    segment_size: u64,
) -> Result<()> {
    patch_phdr_note_to_load(data, phdr_offset, segment_va, file_offset, segment_size)
}

/// Overwrite a PT_NOTE program header entry to become PT_LOAD.
///
/// ELF64 Phdr layout (56 bytes):
///   p_type   (4 bytes, offset 0)
///   p_flags  (4 bytes, offset 4)
///   p_offset (8 bytes, offset 8)
///   p_vaddr  (8 bytes, offset 16)
///   p_paddr  (8 bytes, offset 24)
///   p_filesz (8 bytes, offset 32)
///   p_memsz  (8 bytes, offset 40)
///   p_align  (8 bytes, offset 48)
fn patch_phdr_note_to_load(
    data: &mut [u8],
    phdr_offset: u64,
    segment_va: u64,
    file_offset: u64,
    segment_size: u64,
) -> Result<()> {
    let off = phdr_offset as usize;

    if off + 56 > data.len() {
        bail!(
            "PT_NOTE phdr offset 0x{:x} extends beyond file",
            phdr_offset
        );
    }

    // p_type = PT_LOAD (1)
    data[off..off + 4].copy_from_slice(&1u32.to_le_bytes());
    // p_flags = PF_R | PF_W | PF_X (7)
    data[off + 4..off + 8].copy_from_slice(&7u32.to_le_bytes());
    // p_offset = file_offset (where we append)
    data[off + 8..off + 16].copy_from_slice(&file_offset.to_le_bytes());
    // p_vaddr = segment_va
    data[off + 16..off + 24].copy_from_slice(&segment_va.to_le_bytes());
    // p_paddr = segment_va (same)
    data[off + 24..off + 32].copy_from_slice(&segment_va.to_le_bytes());
    // p_filesz = segment_size
    data[off + 32..off + 40].copy_from_slice(&segment_size.to_le_bytes());
    // p_memsz = segment_size
    data[off + 40..off + 48].copy_from_slice(&segment_size.to_le_bytes());
    // p_align = 0x1000 (page alignment)
    data[off + 48..off + 56].copy_from_slice(&0x1000u64.to_le_bytes());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0x1000, 0x1000), 0x1000);
        assert_eq!(align_up(0x1001, 0x1000), 0x2000);
        assert_eq!(align_up(0, 0x1000), 0);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
    }

    #[test]
    fn test_patch_phdr_note_to_load() {
        let mut phdr = vec![0u8; 56];
        // Set p_type = PT_NOTE (4)
        phdr[0..4].copy_from_slice(&4u32.to_le_bytes());

        patch_phdr_note_to_load(&mut phdr, 0, 0x800000, 0x10000, 0x2000).unwrap();

        // Verify p_type = PT_LOAD (1)
        assert_eq!(u32::from_le_bytes(phdr[0..4].try_into().unwrap()), 1);
        // Verify p_flags = RWX (7)
        assert_eq!(u32::from_le_bytes(phdr[4..8].try_into().unwrap()), 7);
        // Verify p_vaddr
        assert_eq!(
            u64::from_le_bytes(phdr[16..24].try_into().unwrap()),
            0x800000
        );
        // Verify p_filesz
        assert_eq!(u64::from_le_bytes(phdr[32..40].try_into().unwrap()), 0x2000);
    }
}
