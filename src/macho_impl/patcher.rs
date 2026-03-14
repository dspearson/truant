use anyhow::{Context, Result, bail};

use crate::disasm::BasicBlock;
use crate::hook_trampoline::TargetAbi;
use crate::hooks::ResolvedHook;
use crate::macho::MachOContext;
use crate::macho_trampoline;
use crate::patcher::PatchResult;
use crate::traits::{BinaryContext, Patcher, TrampolineGenerator};
use crate::trampoline::{DATA_SIZE, PERSISTENT_DATA_SIZE};

/// Mach-O-specific patcher implementing the Patcher trait.
///
/// Handles LC_SEGMENT_64 insertion, entry point redirection (LC_MAIN or
/// LC_UNIXTHREAD), code signature removal, and segment data appending.
pub struct MachOPatcher {
    tramp_gen: Box<dyn TrampolineGenerator>,
}

impl std::fmt::Debug for MachOPatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MachOPatcher")
            .field("tramp_gen", &self.tramp_gen)
            .finish()
    }
}

impl MachOPatcher {
    pub fn new(tramp_gen: Box<dyn TrampolineGenerator>) -> Self {
        Self { tramp_gen }
    }
}

/// Mach-O LC_SEGMENT_64 command constants.
const LC_SEGMENT_64: u32 = 0x19;
const LC_SEGMENT_64_SIZE: u32 = 72; // sizeof(segment_command_64)

/// Align a value up to the given alignment.
fn align_up(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

/// Page size for the architecture (derived from Mach-O cputype).
fn page_size_for_arch(ctx: &MachOContext) -> u64 {
    // ARM64 macOS uses 16KB pages; x86_64 uses 4KB
    if ctx.cputype == 0x0100_000C {
        0x4000 // 16KB (ARM64 macOS)
    } else {
        0x1000 // 4KB (x86_64)
    }
}

impl Patcher for MachOPatcher {
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
        let macho_ctx = ctx
            .as_any()
            .downcast_ref::<crate::macho_impl::context::MachOBinaryContext>()
            .ok_or_else(|| anyhow::anyhow!("MachOPatcher requires MachOBinaryContext"))?;

        patch_macho(
            macho_ctx.inner(),
            blocks,
            data,
            enable_forkserver,
            enable_heap_san,
            &*self.tramp_gen,
            hooks,
            persistent_addr,
            persistent_count,
            defer,
            no_coverage,
        )
    }
}

/// Orchestrate Mach-O patching:
/// 1. Strip code signature (reclaim LC space)
/// 2. Compute new segment layout
/// 3. Generate init code + trampolines
/// 4. Insert LC_SEGMENT_64 load command
/// 5. Redirect entry point
/// 6. Patch basic blocks
/// 7. Append segment data
#[allow(clippy::too_many_arguments)]
fn patch_macho(
    ctx: &MachOContext,
    blocks: &[BasicBlock],
    mut data: Vec<u8>,
    enable_forkserver: bool,
    enable_heap_san: bool,
    tramp_gen: &dyn TrampolineGenerator,
    hooks: &[ResolvedHook],
    persistent_addr: Option<u64>,
    persistent_count: u32,
    defer: bool,
    no_coverage: bool,
) -> Result<PatchResult> {
    if blocks.is_empty() && !no_coverage {
        bail!("no basic blocks to instrument");
    }

    let page_sz = page_size_for_arch(ctx);

    // Step 1: Strip code signature if present (reclaims LC space).
    let (mut ncmds, mut sizeofcmds) = (ctx.ncmds, ctx.sizeofcmds);
    let mut available_lc_space = ctx.available_lc_space;

    if let Some(codesig_offset) = ctx.code_signature_lc_offset {
        let off = codesig_offset as usize;
        if off + ctx.code_signature_lc_size as usize <= data.len() {
            // Zero out the load command
            for b in &mut data[off..off + ctx.code_signature_lc_size as usize] {
                *b = 0;
            }
            ncmds -= 1;
            sizeofcmds -= ctx.code_signature_lc_size;
            available_lc_space += ctx.code_signature_lc_size as u64;

            // Update header
            data[ctx.ncmds_offset as usize..ctx.ncmds_offset as usize + 4]
                .copy_from_slice(&ncmds.to_le_bytes());
            data[ctx.sizeofcmds_offset as usize..ctx.sizeofcmds_offset as usize + 4]
                .copy_from_slice(&sizeofcmds.to_le_bytes());

            tracing::info!(
                "stripped LC_CODE_SIGNATURE ({} bytes) — re-sign with: codesign -f -s - <output>",
                ctx.code_signature_lc_size,
            );
        }
    }

    // Step 2: Verify we have room for a new LC_SEGMENT_64 (72 bytes).
    if available_lc_space < LC_SEGMENT_64_SIZE as u64 {
        bail!(
            "insufficient load command space for LC_SEGMENT_64: {} bytes available, {} needed.\n\
             Strip unused load commands or rebuild with -headerpad_max_install_names",
            available_lc_space,
            LC_SEGMENT_64_SIZE,
        );
    }

    // Step 2b: Parse chained fixups metadata for GOT-based shmat.
    // On ARM64, always use raw syscall: the __TR_COV segment is code-signed and
    // the kernel maps it read-execute.  Dyld cannot write the resolved shmat
    // address into a code-signed page → SIGBUS during fixup resolution.
    // For dylibs, always use raw syscall: dylib init functions don't go through
    // the same chained fixups mechanism as executables, and GOT slot resolution
    // in __TR_COV is unreliable in the dylib loading context.
    const CPU_TYPE_ARM64: u32 = 0x0100_000C;
    let use_got_shmat =
        ctx.chained_fixups_lc_offset.is_some() && ctx.cputype != CPU_TYPE_ARM64 && !ctx.is_dylib;
    let shmat_ordinal: u32 = if use_got_shmat {
        let fo = ctx.chained_fixups_dataoff as usize;
        u32::from_le_bytes(data[fo + 16..fo + 20].try_into().unwrap()) // imports_count
    } else {
        0
    };
    let data_area_size: u64 = if use_got_shmat { 24 } else { DATA_SIZE };

    // Detect ARM64 early — needed for segment layout decisions.
    let is_arm64 = ctx.cputype == CPU_TYPE_ARM64;
    let hook_abi = if is_arm64 {
        TargetAbi::Aarch64
    } else {
        TargetAbi::SysV64
    };

    let segment_va = if ctx.linkedit_vmaddr > 0 {
        ctx.linkedit_vmaddr // TR_COV takes LINKEDIT's spot; LINKEDIT shifts up
    } else {
        align_up(ctx.highest_va_end, page_sz) + page_sz // fallback if no LINKEDIT
    };

    // On ARM64 macOS, the kernel enforces W^X: a page mapped executable cannot be
    // written at the same time.  If __TR_COV has initprot=RWX, the kernel silently
    // strips W, so any STR to data_va (within the same page as the code) causes
    // SIGBUS.  Fix: give the 16-byte data area its own RW page (__TR_DAT) and keep
    // the init code + trampolines in a separate RX page (__TR_COV).
    // x86_64 macOS does not enforce W^X in userspace, so the original single-segment
    // RWX layout continues to work there.
    let has_library_hooks =
        crate::hooks::count_library_hooks(hooks) > 0 || crate::hooks::count_return_hooks(hooks) > 0;
    let arm64_split = is_arm64 && (!no_coverage || has_library_hooks);
    let data_va = segment_va;
    let init_va = if arm64_split {
        // Data lives in page 0 (RW); code starts at page 1 (RX).
        segment_va + page_sz
    } else {
        segment_va + data_area_size
    };

    // Step 4: Generate init code + trampolines (skip in no-coverage mode).

    let init_code;
    let mut trampolines: Vec<(&BasicBlock, crate::trampoline::Trampoline)> = Vec::new();
    let mut blocks_skipped = 0;
    let mut current_va;

    if no_coverage {
        init_code = crate::trampoline::InitCode {
            va: init_va,
            code: Vec::new(),
            entry_va: init_va,
        };
        current_va = init_va;
    } else {
        init_code = if ctx.is_dylib {
            let chain_to = ctx.mod_init_func_pointers.first().copied();
            if is_arm64 {
                macho_trampoline::generate_macho_dylib_init_aarch64(
                    init_va,
                    data_va,
                    chain_to,
                    use_got_shmat,
                )
                .context("failed to generate macOS ARM64 dylib init code")?
            } else {
                macho_trampoline::generate_macho_dylib_init_x86_64(
                    init_va,
                    data_va,
                    chain_to,
                    use_got_shmat,
                )
                .context("failed to generate macOS dylib init code")?
            }
        } else if ctx.lc_main_entryoff_offset.is_some() {
            if is_arm64 {
                macho_trampoline::generate_macho_exec_init_aarch64(
                    init_va,
                    data_va,
                    ctx.entry_point,
                    enable_forkserver,
                    use_got_shmat,
                )
                .context("failed to generate macOS ARM64 exec init code")?
            } else {
                macho_trampoline::generate_macho_exec_init_x86_64(
                    init_va,
                    data_va,
                    ctx.entry_point,
                    enable_forkserver,
                    use_got_shmat,
                )
                .context("failed to generate macOS exec init code")?
            }
        } else if ctx.lc_unixthread_pc_offset.is_some() {
            if is_arm64 {
                macho_trampoline::generate_macho_unixthread_init_aarch64(
                    init_va,
                    data_va,
                    ctx.entry_point,
                    enable_forkserver,
                    use_got_shmat,
                )
                .context("failed to generate macOS ARM64 LC_UNIXTHREAD init code")?
            } else {
                macho_trampoline::generate_macho_unixthread_init_x86_64(
                    init_va,
                    data_va,
                    ctx.entry_point,
                    enable_forkserver,
                    use_got_shmat,
                )
                .context("failed to generate macOS LC_UNIXTHREAD init code")?
            }
        } else {
            bail!("no LC_MAIN or LC_UNIXTHREAD found — cannot determine entry point");
        };

        // Step 5: Generate trampolines.
        let mut trampolines_start_va = init_va + init_code.code.len() as u64;
        trampolines_start_va = align_up(trampolines_start_va, 16);

        trampolines.reserve(blocks.len());
        current_va = trampolines_start_va;

        for block in blocks {
            match tramp_gen.generate_trampoline(current_va, data_va, block) {
                Ok(tramp) => {
                    current_va += tramp.code.len() as u64;
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

    // Hook data slots, toggle bytes, and return-address slots.
    //
    // On ARM64 macOS (arm64_split), these MUST live in the RW data page (__TR_DAT),
    // NOT in the RX code page (__TR_COV).  The preload library's constructor writes
    // function pointers into hook data slots at runtime — writing to an RX page
    // triggers SEGFAULT under W^X enforcement.
    let num_hook_slots = crate::hooks::count_library_hooks(hooks);
    let hook_data_size = (num_hook_slots * 8) as u64;
    let toggle_area_size = if hooks.is_empty() {
        0u64
    } else {
        ((hooks.len() as u64) + 7) & !7
    };
    let return_slot_count = crate::hooks::count_return_hooks(hooks);
    let return_slot_area_size = (return_slot_count as u64) * 8;

    let (hook_data_va, toggle_data_va, return_slot_va);
    if arm64_split {
        // Place writable hook metadata in the data page after the coverage data area.
        let mut data_cursor = data_va + data_area_size;
        data_cursor = align_up(data_cursor, 8);
        hook_data_va = data_cursor;
        data_cursor += hook_data_size;
        toggle_data_va = data_cursor;
        data_cursor += toggle_area_size;
        return_slot_va = data_cursor;
        data_cursor += return_slot_area_size;
        // Verify it fits in one data page.
        assert!(
            data_cursor <= data_va + page_sz,
            "hook metadata ({} bytes) exceeds data page size ({})",
            data_cursor - data_va,
            page_sz,
        );
    } else {
        // x86_64 / non-split: allocate in the combined segment as before.
        hook_data_va = current_va;
        current_va += hook_data_size;
        toggle_data_va = current_va;
        current_va += toggle_area_size;
        return_slot_va = current_va;
        current_va += return_slot_area_size;
    }
    current_va = align_up(current_va, 16);

    let mut hook_trampolines: Vec<(usize, crate::trampoline::Trampoline)> = Vec::new();
    let mut shellcode_blobs: Vec<(u64, Vec<u8>)> = Vec::new();

    // Place shellcode blobs first so we know their VAs for trampoline generation.
    let mut shellcode_va_map: Vec<Option<u64>> = Vec::with_capacity(hooks.len());
    for hook in hooks {
        if let crate::hooks::HookSource::Shellcode(ref sc) = hook.source {
            let sc_va = current_va;
            shellcode_blobs.push((sc_va, sc.clone()));
            current_va += sc.len() as u64;
            current_va = align_up(current_va, 16);
            shellcode_va_map.push(Some(sc_va));
        } else {
            shellcode_va_map.push(None);
        }
    }

    // Group hooks by target_va for chaining (same as ELF patcher).
    current_va = align_up(current_va, 16);
    let mut va_groups: std::collections::BTreeMap<u64, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, hook) in hooks.iter().enumerate() {
        va_groups.entry(hook.target_va).or_default().push(i);
    }

    // Separate vec for return trampolines (not site-patched).
    let mut return_trampolines: Vec<crate::trampoline::Trampoline> = Vec::new();

    for indices in va_groups.values() {
        if indices.len() == 1 {
            let i = indices[0];
            let hook = &hooks[i];
            let sc_va = shellcode_va_map[i];
            let toggle_va = Some(toggle_data_va + hook.toggle_index as u64);

            if hook.mode == crate::hooks::HookMode::Return {
                // Return mode: generate entry + return trampoline pair.
                let entry_va = current_va;
                let estimated_entry_size = 256u64;
                let ret_tramp_va = align_up(entry_va + estimated_entry_size, 8);
                let slot_va = return_slot_va + hook.return_slot_index.unwrap() as u64 * 8;

                match crate::hook_trampoline::generate_return_hook_trampolines(
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
                        assert!(
                            (entry_tramp.code.len() as u64) <= estimated_entry_size,
                            "return hook entry trampoline ({} bytes) exceeded estimate ({})",
                            entry_tramp.code.len(),
                            estimated_entry_size,
                        );
                        hook_trampolines.push((i, entry_tramp));
                        current_va = ret_tramp.va + ret_tramp.code.len() as u64;
                        current_va = align_up(current_va, 8);
                        return_trampolines.push(ret_tramp);
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
                match crate::hook_trampoline::generate_hook_trampoline(
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
            let chain_hooks: Vec<&ResolvedHook> = indices.iter().map(|&i| &hooks[i]).collect();
            let chain_sc_vas: Vec<Option<u64>> =
                indices.iter().map(|&i| shellcode_va_map[i]).collect();
            let chain_toggle_vas: Vec<Option<u64>> = indices
                .iter()
                .map(|&i| Some(toggle_data_va + hooks[i].toggle_index as u64))
                .collect();
            match crate::hook_trampoline::generate_chained_hook_trampoline(
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
                    hook_trampolines.push((indices[0], tramp));
                }
                Err(e) => {
                    tracing::warn!(
                        "chained hook at 0x{:x}: trampoline generation failed: {}",
                        hooks[indices[0]].target_va,
                        e,
                    );
                }
            }
        }
    }

    // Persistent mode: allocate data area and generate wrapper.
    let persistent_data_va;
    let persistent_wrapper = if let Some(p_addr) = persistent_addr {
        if ctx.is_dylib {
            bail!("persistent mode is not supported for dylibs");
        }
        current_va = align_up(current_va, 16);
        persistent_data_va = current_va;
        current_va += PERSISTENT_DATA_SIZE;
        current_va = align_up(current_va, 16);

        // Extract displaced bytes at the persistent function entry.
        let branch_sz = tramp_gen.branch_instruction_size();
        let (displaced_bytes, displaced_len) = if is_arm64 {
            // AArch64: fixed-width 4-byte instructions.
            let foff = crate::macho::va_to_file_offset_macho(p_addr, ctx).ok_or_else(|| {
                anyhow::anyhow!("persistent addr 0x{:x} has no file mapping", p_addr,)
            })?;
            let n = branch_sz.div_ceil(4) * 4; // round up to instruction boundary
            if foff + n > data.len() {
                bail!("persistent addr 0x{:x} too close to end of file", p_addr);
            }
            (data[foff..foff + n].to_vec(), n)
        } else {
            crate::macho_disasm::extract_displaced_bytes_macho(&data, ctx, p_addr, branch_sz)
                .with_context(|| {
                    format!(
                        "failed to extract displaced bytes at persistent addr 0x{:x}",
                        p_addr,
                    )
                })?
        };

        let include_forkserver = enable_forkserver && defer;
        let wrapper_va = current_va;
        let wrapper = if is_arm64 {
            macho_trampoline::generate_macho_persistent_wrapper_aarch64(
                wrapper_va,
                persistent_data_va,
                data_va,
                p_addr,
                &displaced_bytes,
                displaced_len,
                persistent_count,
                include_forkserver,
            )
            .context("failed to generate macOS ARM64 persistent wrapper")?
        } else {
            macho_trampoline::generate_macho_persistent_wrapper_x86_64(
                wrapper_va,
                persistent_data_va,
                data_va,
                p_addr,
                &displaced_bytes,
                displaced_len,
                persistent_count,
                include_forkserver,
            )
            .context("failed to generate macOS x86_64 persistent wrapper")?
        };

        current_va += wrapper.code.len() as u64;
        current_va = align_up(current_va, 16);

        tracing::info!(
            "persistent mode: wrapper at VA 0x{:x} ({} bytes), data at VA 0x{:x}, target 0x{:x}",
            wrapper_va,
            wrapper.code.len(),
            persistent_data_va,
            p_addr,
        );

        Some((wrapper_va, wrapper, displaced_len))
    } else {
        persistent_data_va = 0;
        None
    };

    // --- Heap sanitiser wrappers (optional) ---
    // The heap san wrappers use x86_64 raw syscalls (SYS_mmap, SYS_mprotect) —
    // these are Linux-only and will not work on macOS where syscall numbers differ.
    // For now we only support heap san on x86_64 Mach-O as a proof of concept
    // (the wrappers can be used in cross-compiled Linux test scenarios).
    // ARM64 macOS heap san requires macOS-specific syscall wrappers (deferred).
    let mut heap_san_intercepted = 0;
    let heap_san_code = if enable_heap_san && !is_arm64 {
        let alloc_syms = crate::macho::find_allocator_symbols_macho(ctx, &data);
        if alloc_syms.get("malloc").is_some() && alloc_syms.get("free").is_some() {
            current_va = align_up(current_va, 16);
            let hsw = crate::trampoline::generate_heap_san_wrappers(current_va);
            current_va += hsw.total_size;
            tracing::info!(
                "heap san (Mach-O): {} allocator functions found, wrappers at 0x{:x}",
                alloc_syms.count(),
                *hsw.wrappers.values().min().unwrap(),
            );
            Some((alloc_syms, hsw))
        } else {
            tracing::warn!(
                "heap san (Mach-O): malloc/free not found (found {} allocator symbols), skipping",
                alloc_syms.count(),
            );
            None
        }
    } else if enable_heap_san && is_arm64 {
        tracing::warn!(
            "heap san: ARM64 macOS not yet supported (Linux x86_64 syscalls only), skipping"
        );
        None
    } else {
        None
    };

    let segment_end_va = current_va;

    // For dylibs without an existing init section (__mod_init_func / __init_offsets),
    // we will emit LC_ROUTINES_64 (cmd=0x1A, 72 bytes) which instructs dyld to call
    // the specified VA before any section-based initialisers.  This is more robust than
    // synthesising a section in a custom segment, because dyld only processes
    // __mod_init_func / __init_offsets from __TEXT and __DATA; custom segment names
    // are ignored for initialisation purposes.
    //
    // LC_ROUTINES_64 does not require any fixup chain entries and works with both
    // legacy rebase/bind opcodes and LC_DYLD_CHAINED_FIXUPS.
    let needs_lc_routines_64 = ctx.is_dylib
        && ctx.mod_init_func_section.is_none()
        && ctx.init_offsets_section.is_none()
        && !no_coverage;
    // Segment size does not change — LC_ROUTINES_64 is a load command, not segment data.
    let final_end_va = segment_end_va;

    // For the ARM64 split layout the two segments span:
    //   __TR_DAT: [segment_va, segment_va + page_sz)  — 16-byte data area (RW)
    //   __TR_COV: [segment_va + page_sz, current_va)  — init code + trampolines (RX)
    // For x86_64 (or no-coverage) we keep a single combined segment as before.
    let (segment_size, segment_vmsize, code_segment_size, code_segment_vmsize);
    if arm64_split {
        // Data segment is exactly one page.
        segment_size = page_sz;
        segment_vmsize = page_sz;
        // Code segment spans from init_va to current_va.
        code_segment_size = final_end_va - init_va;
        code_segment_vmsize = align_up(code_segment_size, page_sz);
    } else {
        segment_size = final_end_va - segment_va;
        segment_vmsize = align_up(segment_size, page_sz);
        // Unused for x86_64 path.
        code_segment_size = 0;
        code_segment_vmsize = 0;
    }

    // Step 6: Build segment data.
    //
    // For ARM64 split: two separate byte buffers (data_page_data, code_page_data).
    // For x86_64: one combined buffer (segment_data).
    let mut segment_data: Vec<u8>;
    let mut code_segment_data: Vec<u8>;

    if arm64_split {
        // Data page: zeroed, 16-byte data area at offset 0.
        segment_data = vec![0u8; segment_size as usize];
        // Code page: zeroed, code starts at offset 0 relative to init_va.
        code_segment_data = vec![0u8; code_segment_size as usize];

        // GOT-based shmat is disabled for ARM64 (use_got_shmat is always false).
        // Init code (relative to code page start = init_va).
        code_segment_data[..init_code.code.len()].copy_from_slice(&init_code.code);

        // Trampolines (relative to code page).
        for (_, tramp) in &trampolines {
            let offset = (tramp.va - init_va) as usize;
            code_segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
        }

        // Toggle area (in data page, relative to data_va).
        for hook in hooks {
            let byte_offset = (toggle_data_va - data_va) as usize + hook.toggle_index;
            if byte_offset < segment_data.len() {
                segment_data[byte_offset] = if hook.initial_enabled { 1 } else { 0 };
            }
        }

        // Hook shellcode blobs (relative to code page).
        for (sc_va, sc_bytes) in &shellcode_blobs {
            let offset = (*sc_va - init_va) as usize;
            code_segment_data[offset..offset + sc_bytes.len()].copy_from_slice(sc_bytes);
        }

        // Hook trampolines (relative to code page).
        for (_, tramp) in &hook_trampolines {
            let offset = (tramp.va - init_va) as usize;
            code_segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
        }

        // Return trampolines (relative to code page).
        for tramp in &return_trampolines {
            let offset = (tramp.va - init_va) as usize;
            code_segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
        }

        // Persistent data + wrapper (relative to code page).
        if let Some((wrapper_va, ref wrapper, _)) = persistent_wrapper {
            let pd_offset = (persistent_data_va - init_va) as usize;
            if pd_offset < code_segment_data.len() {
                code_segment_data[pd_offset] = 1; // first_pass = 1
            }
            let w_offset = (wrapper_va - init_va) as usize;
            code_segment_data[w_offset..w_offset + wrapper.code.len()]
                .copy_from_slice(&wrapper.code);
        }

        // Heap san wrappers (relative to code page).
        if let Some((_, ref hsw)) = heap_san_code {
            let base = *hsw.wrappers.values().min().unwrap();
            let offset = (base - init_va) as usize;
            code_segment_data[offset..offset + hsw.code.len()].copy_from_slice(&hsw.code);
        }
    } else {
        code_segment_data = Vec::new(); // unused for x86_64

        segment_data = vec![0u8; segment_size as usize];

        // Write GOT slot bind entry for dyld to resolve _shmat at load time.
        // dyld_chained_ptr_64_bind: ordinal(24) | addend(8) | reserved(19) | next(12) | bind(1)
        if use_got_shmat {
            let bind_entry: u64 = (shmat_ordinal as u64 & 0xFFFFFF) | (1u64 << 63);
            segment_data[16..24].copy_from_slice(&bind_entry.to_le_bytes());
        }

        // Init code
        let init_offset = (init_va - segment_va) as usize;
        segment_data[init_offset..init_offset + init_code.code.len()]
            .copy_from_slice(&init_code.code);

        // Trampolines
        for (_, tramp) in &trampolines {
            let offset = (tramp.va - segment_va) as usize;
            segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
        }

        // Toggle area: write initial enabled values.
        for hook in hooks {
            let byte_offset = (toggle_data_va - segment_va) as usize + hook.toggle_index;
            if byte_offset < segment_data.len() {
                segment_data[byte_offset] = if hook.initial_enabled { 1 } else { 0 };
            }
        }

        // Hook shellcode blobs
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

        // Persistent data + wrapper
        if let Some((wrapper_va, ref wrapper, _)) = persistent_wrapper {
            // Write persistent data area (first_pass=1 at offset 0)
            let pd_offset = (persistent_data_va - segment_va) as usize;
            segment_data[pd_offset] = 1; // first_pass = 1

            // Write persistent wrapper code
            let w_offset = (wrapper_va - segment_va) as usize;
            segment_data[w_offset..w_offset + wrapper.code.len()].copy_from_slice(&wrapper.code);
        }

        // Heap san wrappers
        if let Some((_, ref hsw)) = heap_san_code {
            let base = *hsw.wrappers.values().min().unwrap();
            let offset = (base - segment_va) as usize;
            segment_data[offset..offset + hsw.code.len()].copy_from_slice(&hsw.code);
        }
    }

    // Note: when needs_lc_routines_64, the init function VA is registered via
    // LC_ROUTINES_64 (added to the load commands below) — no segment data needed.

    // Step 7: Patch basic blocks with branch to trampoline.
    let branch_size = tramp_gen.branch_instruction_size();
    for (block, tramp) in &trampolines {
        let file_offset = block.file_offset as usize;

        match tramp_gen.encode_branch(block.va, tramp.va) {
            Ok(branch_bytes) => {
                data[file_offset..file_offset + branch_bytes.len()].copy_from_slice(&branch_bytes);
                for i in branch_size..block.displaced_len {
                    data[file_offset + i] = 0x90;
                }
            }
            Err(e) => {
                tracing::warn!("block at 0x{:x}: branch encoding failed: {}", block.va, e);
            }
        }
    }

    // Step 7.5: Patch hook target sites with JMP rel32 to hook trampoline.
    let mut hooks_applied = 0;
    for (hook_idx, tramp) in &hook_trampolines {
        let hook = &hooks[*hook_idx];
        let file_offset = hook.file_offset as usize;
        match tramp_gen.encode_branch(hook.target_va, tramp.va) {
            Ok(branch_bytes) => {
                data[file_offset..file_offset + branch_bytes.len()].copy_from_slice(&branch_bytes);
                let branch_size = tramp_gen.branch_instruction_size();
                for i in branch_size..hook.displaced_len {
                    data[file_offset + i] = 0x90;
                }
                hooks_applied += 1;
            }
            Err(e) => {
                tracing::warn!(
                    "hook at 0x{:x}: branch encoding failed: {}",
                    hook.target_va,
                    e
                );
            }
        }
    }

    // Step 7.6: Patch persistent function entry with branch to wrapper.
    if let Some((wrapper_va, _, displaced_len)) = &persistent_wrapper {
        let p_addr = persistent_addr.unwrap();
        let file_offset = crate::macho::va_to_file_offset_macho(p_addr, ctx).ok_or_else(|| {
            anyhow::anyhow!("persistent addr 0x{:x}: VA to file offset failed", p_addr)
        })?;
        match tramp_gen.encode_branch(p_addr, *wrapper_va) {
            Ok(branch_bytes) => {
                data[file_offset..file_offset + branch_bytes.len()].copy_from_slice(&branch_bytes);
                let branch_size = tramp_gen.branch_instruction_size();
                let nop = if is_arm64 {
                    &[0x1F, 0x20, 0x03, 0xD5][..]
                } else {
                    &[0x90][..]
                };
                let mut i = branch_size;
                while i < *displaced_len {
                    let n = nop.len().min(*displaced_len - i);
                    data[file_offset + i..file_offset + i + n].copy_from_slice(&nop[..n]);
                    i += n;
                }
                tracing::info!(
                    "patched persistent function at 0x{:x} → wrapper at 0x{:x}",
                    p_addr,
                    wrapper_va
                );
            }
            Err(e) => {
                tracing::warn!(
                    "persistent addr 0x{:x}: branch encoding failed: {}",
                    p_addr,
                    e
                );
            }
        }
    }

    // Step 7.65: Patch allocator function entries for heap san.
    if let Some((ref alloc_syms, ref hsw)) = heap_san_code {
        for (name, alloc) in &alloc_syms.entries {
            let wrapper_va = match hsw.wrappers.get(name) {
                Some(&va) => va,
                None => continue,
            };
            let off = alloc.patch_offset as usize;
            // Mach-O allocator symbols are always FuncEntry (direct function patching).
            // Overwrite the first 5 bytes with JMP rel32 to wrapper.
            if off + 5 <= data.len() {
                let rel32 = (wrapper_va as i64 - (alloc.patch_va as i64 + 5)) as i32;
                data[off] = 0xE9;
                data[off + 1..off + 5].copy_from_slice(&rel32.to_le_bytes());
                heap_san_intercepted += 1;
                tracing::info!(
                    "heap san (Mach-O): patched {} entry at 0x{:x} → 0x{:x}",
                    name,
                    alloc.patch_va,
                    wrapper_va,
                );
            }
        }
    }

    // Step 7.7: Build chained fixups FIRST (needs original LINKEDIT data intact).
    // We MUST update seg_count whenever we add __TR_COV, because dyld validates
    // that the number of LC_SEGMENT_64 commands matches the chained fixups seg_count.
    // Failure to update causes dyld to reject the binary with "seg_count does not match".
    let new_fixups_blob: Option<Vec<u8>> = if use_got_shmat {
        let (blob, _ordinal) =
            build_chained_fixups_with_shmat(&data, ctx, segment_va, page_sz, ctx.libsystem_ordinal)
                .context("failed to rebuild chained fixups with _shmat")?;
        tracing::info!(
            "rebuilt chained fixups: {} → {} bytes (shmat ordinal={}, libsystem_ordinal={})",
            ctx.chained_fixups_datasize,
            blob.len(),
            shmat_ordinal,
            ctx.libsystem_ordinal,
        );
        Some(blob)
    } else if ctx.chained_fixups_lc_offset.is_some() && !no_coverage {
        // No GOT slot needed, but must still add empty seg entries.
        // ARM64 split adds __TR_DAT + __TR_COV (2 segments); x86_64 adds __TR_COV (1 segment).
        let extra_segs: u32 = if arm64_split { 2 } else { 1 };
        let blob = rebuild_chained_fixups_add_n_empty_segments(&data, ctx, extra_segs)
            .context("failed to rebuild chained fixups (empty segment entries)")?;
        tracing::info!(
            "rebuilt chained fixups ({} empty seg entries): {} → {} bytes",
            extra_segs,
            ctx.chained_fixups_datasize,
            blob.len(),
        );
        Some(blob)
    } else {
        None
    };

    // Step 8: Restructure file layout — place our new segment(s) BEFORE __LINKEDIT.
    // macOS requires __LINKEDIT to be the last segment (highest vmaddr, data to EOF).
    // ARM64 macOS enforces this at the kernel level (SIGKILL on launch).
    //
    // For ARM64 split: write __TR_DAT (data page) + __TR_COV (code pages) consecutively.
    // For x86_64: write a single __TR_COV blob (unchanged from before).
    let file_offset_of_segment; // fileoff of __TR_DAT (ARM64) or __TR_COV (x86_64)
    let file_offset_of_code_seg; // fileoff of __TR_COV (ARM64 only)
    let new_linkedit_fileoff: u64;
    let new_fixups_fileoff: u32;
    let new_fixups_size: u32;

    if ctx.linkedit_segment_offset.is_some() {
        // Save original LINKEDIT data (everything from linkedit_fileoff to EOF).
        let linkedit_data = data[ctx.linkedit_fileoff as usize..].to_vec();
        data.truncate(ctx.linkedit_fileoff as usize);

        // Pad to page boundary, append data segment.
        let pad1 = align_up(data.len() as u64, page_sz) - data.len() as u64;
        data.extend(std::iter::repeat_n(0u8, pad1 as usize));
        file_offset_of_segment = data.len() as u64;
        data.extend_from_slice(&segment_data);

        if arm64_split {
            // Pad to next page boundary, append code segment.
            let pad_code = align_up(data.len() as u64, page_sz) - data.len() as u64;
            data.extend(std::iter::repeat_n(0u8, pad_code as usize));
            file_offset_of_code_seg = data.len() as u64;
            data.extend_from_slice(&code_segment_data);
        } else {
            file_offset_of_code_seg = 0; // unused
        }

        // Pad to page boundary, re-append original LINKEDIT data.
        let pad2 = align_up(data.len() as u64, page_sz) - data.len() as u64;
        data.extend(std::iter::repeat_n(0u8, pad2 as usize));
        new_linkedit_fileoff = data.len() as u64;
        data.extend_from_slice(&linkedit_data);

        // Append new chained fixups blob at EOF (within LINKEDIT extent).
        if let Some(ref blob) = new_fixups_blob {
            new_fixups_fileoff = data.len() as u32;
            new_fixups_size = blob.len() as u32;
            data.extend_from_slice(blob);
        } else {
            new_fixups_fileoff = 0;
            new_fixups_size = 0;
        }
    } else {
        // No LINKEDIT — fall back to old layout (append at EOF).
        new_linkedit_fileoff = 0;
        if let Some(ref blob) = new_fixups_blob {
            new_fixups_fileoff = data.len() as u32;
            new_fixups_size = blob.len() as u32;
            data.extend_from_slice(blob);
        } else {
            new_fixups_fileoff = 0;
            new_fixups_size = 0;
        }
        let pad = align_up(data.len() as u64, page_sz) - data.len() as u64;
        data.extend(std::iter::repeat_n(0u8, pad as usize));
        file_offset_of_segment = data.len() as u64;
        data.extend_from_slice(&segment_data);
        if arm64_split {
            let pad_code = align_up(data.len() as u64, page_sz) - data.len() as u64;
            data.extend(std::iter::repeat_n(0u8, pad_code as usize));
            file_offset_of_code_seg = data.len() as u64;
            data.extend_from_slice(&code_segment_data);
        } else {
            file_offset_of_code_seg = 0;
        }
    }

    // Step 9: Insert LC_SEGMENT_64 load command(s) (and LC_ROUTINES_64 if needed).
    // Find insertion point: use the end of existing load commands.
    // We must write into the space after the last valid LC.
    //
    // ARM64 split: 2 × LC_SEGMENT_64 (__TR_DAT + __TR_COV) = 144 bytes.
    // x86_64:      1 × LC_SEGMENT_64 (__TR_COV) = 72 bytes.
    const LC_ROUTINES_64_SIZE: u32 = 72; // cmd(4) + cmdsize(4) + init_address(8) + init_module(8) + reserved[8*8]
    let lc_seg_count: u32 = if arm64_split { 2 } else { 1 };
    let lc_total_size = LC_SEGMENT_64_SIZE * lc_seg_count
        + if needs_lc_routines_64 {
            LC_ROUTINES_64_SIZE
        } else {
            0
        };
    let (lc_insert_offset, _lc_end_post_shift) =
        find_lc_insert_offset(&mut data, ctx, lc_total_size);

    if arm64_split {
        // __TR_DAT: RW page at segment_va — holds the 16-byte coverage data area.
        write_lc_segment_64_named(
            &mut data,
            lc_insert_offset,
            b"__TR_DAT\0\0\0\0\0\0\0\0",
            segment_va,
            segment_vmsize,
            file_offset_of_segment,
            segment_size,
            0x3, // initprot = VM_PROT_READ | VM_PROT_WRITE (no execute)
        );
        // __TR_COV: RWX page(s) at segment_va + page_sz — holds init code + trampolines.
        //
        // initprot=0x7 (RWX) instead of 0x5 (RX) is intentional on ARM64 macOS:
        // The code signing executable segment limit (CS_EXECSEG) only covers the
        // __TEXT segment.  A separate RX segment outside this range causes the kernel
        // to SIGKILL the process when it tries to execute code there.
        // RWX segments bypass CS_EXECSEG enforcement on ad-hoc signed binaries (no
        // hardened runtime).  W^X enforcement is only applied under CS_RUNTIME, which
        // is NOT set by ad-hoc codesign (`codesign -s -`).
        write_lc_segment_64_named(
            &mut data,
            lc_insert_offset + LC_SEGMENT_64_SIZE as usize,
            b"__TR_COV\0\0\0\0\0\0\0\0",
            segment_va + page_sz,
            code_segment_vmsize,
            file_offset_of_code_seg,
            code_segment_size,
            0x7, // initprot = VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE
        );
    } else {
        // Single combined RWX segment (x86_64 — W^X not enforced).
        write_lc_segment_64(
            &mut data,
            lc_insert_offset,
            segment_va,
            segment_vmsize,
            file_offset_of_segment,
            segment_size,
            0x7, // initprot = RWX
        );
    }

    // Insert LC_ROUTINES_64 immediately after the last LC_SEGMENT_64 for dylibs without
    // an existing init section.  dyld calls the `init_address` field as the dylib init
    // function before any section-based initialisers (__mod_init_func / __init_offsets).
    // LC_ROUTINES_64 (cmd = 0x1A) layout:
    //   [0]  u32 cmd          = 0x1A
    //   [4]  u32 cmdsize      = 72
    //   [8]  u64 init_address = init_code.entry_va
    //   [16] u64 init_module  = 0
    //   [24] u64[8] reserved  = 0
    if needs_lc_routines_64 {
        const LC_ROUTINES_64_CMD: u32 = 0x1A;
        let r = lc_insert_offset + (LC_SEGMENT_64_SIZE * lc_seg_count) as usize;
        // Zero the whole struct first
        data[r..r + LC_ROUTINES_64_SIZE as usize].fill(0);
        data[r..r + 4].copy_from_slice(&LC_ROUTINES_64_CMD.to_le_bytes());
        data[r + 4..r + 8].copy_from_slice(&LC_ROUTINES_64_SIZE.to_le_bytes());
        data[r + 8..r + 16].copy_from_slice(&init_code.entry_va.to_le_bytes());
        // init_module = 0, all reserved fields = 0 (already zeroed)
        tracing::info!(
            "inserted LC_ROUTINES_64 init_address=0x{:x} (dylib has no existing init section)",
            init_code.entry_va,
        );
    }

    // Update ncmds and sizeofcmds in the header.
    let added_ncmds: u32 = lc_seg_count + if needs_lc_routines_64 { 1 } else { 0 };
    ncmds += added_ncmds;
    sizeofcmds += lc_total_size;
    data[ctx.ncmds_offset as usize..ctx.ncmds_offset as usize + 4]
        .copy_from_slice(&ncmds.to_le_bytes());
    data[ctx.sizeofcmds_offset as usize..ctx.sizeofcmds_offset as usize + 4]
        .copy_from_slice(&sizeofcmds.to_le_bytes());

    // Step 9b: Update __LINKEDIT and all LCs that reference LINKEDIT file offsets.
    // Our new segment(s) are placed before LINKEDIT in the file, shifting all LINKEDIT
    // data to a higher file offset. Walk all load commands and adjust.
    if ctx.linkedit_segment_offset.is_some() && new_linkedit_fileoff > 0 {
        let linkedit_shift = new_linkedit_fileoff - ctx.linkedit_fileoff;
        // For ARM64 split: LINKEDIT comes after both __TR_DAT + __TR_COV.
        let total_tr_vmsize = segment_vmsize + if arm64_split { code_segment_vmsize } else { 0 };
        let new_linkedit_vmaddr = segment_va + total_tr_vmsize;
        let new_linkedit_filesize = data.len() as u64 - new_linkedit_fileoff;
        let new_linkedit_vmsize = align_up(new_linkedit_filesize, page_sz);

        update_linkedit_lc_offsets(
            &mut data,
            ctx,
            ncmds,
            linkedit_shift,
            new_linkedit_vmaddr,
            new_linkedit_vmsize,
            new_linkedit_fileoff,
            new_linkedit_filesize,
            new_fixups_fileoff,
            new_fixups_size,
        );

        tracing::info!(
            "relocated __LINKEDIT: VA 0x{:x} → 0x{:x}, fileoff 0x{:x} → 0x{:x} (shift +0x{:x})",
            ctx.linkedit_vmaddr,
            new_linkedit_vmaddr,
            ctx.linkedit_fileoff,
            new_linkedit_fileoff,
            linkedit_shift,
        );
    }

    // Step 10: Redirect entry point (skip in no-coverage mode).
    //
    // IMPORTANT: Step 9 inserted a 72-byte LC_SEGMENT_64 at `lc_insert_offset`,
    // shifting all subsequent load commands forward by 72 bytes. The offsets
    // stored in ctx (lc_main_entryoff_offset, lc_unixthread_pc_offset) are
    // pre-shift values. Adjust them if they fall at or after the insertion point.
    let lc_shift = lc_total_size as u64;
    if no_coverage {
        // No entry point redirect — hooks only.
    } else if ctx.is_dylib {
        redirect_dylib_entry(&mut data, ctx, init_code.entry_va)?;
    } else if let Some(lc_main_off) = ctx.lc_main_entryoff_offset {
        let adjusted_off = if lc_main_off >= lc_insert_offset as u64 {
            lc_main_off + lc_shift
        } else {
            lc_main_off
        };
        // LC_MAIN: entryoff = init_va - __TEXT vmaddr
        let new_entryoff = init_code.entry_va - ctx.text_segment_vmaddr;
        let off = adjusted_off as usize;
        if off + 8 <= data.len() {
            data[off..off + 8].copy_from_slice(&new_entryoff.to_le_bytes());
            tracing::info!(
                "patched LC_MAIN entryoff to 0x{:x} (init_va=0x{:x}, offset adjusted {}→{})",
                new_entryoff,
                init_code.entry_va,
                lc_main_off,
                adjusted_off,
            );
        }
    } else if let Some(pc_off) = ctx.lc_unixthread_pc_offset {
        let adjusted_off = if pc_off >= lc_insert_offset as u64 {
            pc_off + lc_shift
        } else {
            pc_off
        };
        // LC_UNIXTHREAD: patch PC register in thread state
        let off = adjusted_off as usize;
        if off + 8 <= data.len() {
            data[off..off + 8].copy_from_slice(&init_code.entry_va.to_le_bytes());
            tracing::info!(
                "patched LC_UNIXTHREAD PC to 0x{:x} (offset adjusted {}→{})",
                init_code.entry_va,
                pc_off,
                adjusted_off,
            );
        }
    }

    let blocks_instrumented = trampolines.len();

    tracing::info!(
        "Mach-O: patched {} blocks, new segment __TR_COV at VA 0x{:x} ({} bytes)",
        blocks_instrumented,
        segment_va,
        segment_size,
    );

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

/// Find the offset to insert a new load command.
///
/// Find the insertion offset for our LC_SEGMENT_64 and shift __LINKEDIT forward.
///
/// __LINKEDIT must be the last segment load command in the Mach-O header.
/// We insert __TR_COV immediately before __LINKEDIT by shifting __LINKEDIT
/// (and any LCs after it) forward by 72 bytes.
///
/// Returns (insert_at, lc_end_post_shift) where:
/// - insert_at: offset where the new LC should be written
/// - lc_end_post_shift: end of all valid LCs after the shift (suitable for
///   appending additional LCs)
fn find_lc_insert_offset(data: &mut [u8], ctx: &MachOContext, lc_size: u32) -> (usize, usize) {
    if let Some(linkedit_off) = ctx.linkedit_segment_offset {
        let insert_at = linkedit_off as usize;
        // Read current ncmds/sizeofcmds to find end of LC area
        let ncmds = u32::from_le_bytes(
            data[ctx.ncmds_offset as usize..ctx.ncmds_offset as usize + 4]
                .try_into()
                .unwrap(),
        );
        // Walk LCs to find the end of valid load commands
        let mut offset = ctx.header_size;
        for _ in 0..ncmds {
            if offset + 8 > data.len() {
                break;
            }
            let cmd = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            let cmdsize = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
            if cmd == 0 || cmdsize < 8 {
                break;
            }
            offset += cmdsize as usize;
        }
        let lc_end = offset;

        // Shift everything from linkedit_off..lc_end forward by 72 bytes
        let shift_len = lc_end - insert_at;
        if shift_len > 0 {
            data.copy_within(insert_at..lc_end, insert_at + lc_size as usize);
        }

        // After the shift, lc_end is lc_size higher
        (insert_at, lc_end + lc_size as usize)
    } else {
        // No __LINKEDIT found — append at end of existing LCs
        let mut offset = ctx.header_size;
        let ncmds = u32::from_le_bytes(
            data[ctx.ncmds_offset as usize..ctx.ncmds_offset as usize + 4]
                .try_into()
                .unwrap(),
        );
        for _ in 0..ncmds {
            if offset + 8 > data.len() {
                break;
            }
            let cmd = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            let cmdsize = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
            if cmd == 0 || cmdsize < 8 {
                break;
            }
            offset += cmdsize as usize;
        }
        (offset, offset)
    }
}

/// Write an LC_SEGMENT_64 load command at the given offset (segment named "__TR_COV").
///
/// `initprot` controls the initial page protection:
/// - 0x7 (RWX) for executables using GOT-based shmat (dyld resolves symbol into segment)
/// - 0x5 (RX)  for dylibs using raw syscall (no write needed; avoids W^X enforcement)
fn write_lc_segment_64(
    data: &mut [u8],
    offset: usize,
    vmaddr: u64,
    vmsize: u64,
    fileoff: u64,
    filesize: u64,
    initprot: u32,
) {
    write_lc_segment_64_named(
        data,
        offset,
        b"__TR_COV\0\0\0\0\0\0\0\0",
        vmaddr,
        vmsize,
        fileoff,
        filesize,
        initprot,
    );
}

/// Write an LC_SEGMENT_64 load command with an explicit segment name.
///
/// `segname` must be exactly 16 bytes (null-padded).
/// `maxprot` is always set to 0x7 (RWX) to give the kernel maximum flexibility;
/// `initprot` controls the actual initial mapping protection.
#[allow(clippy::too_many_arguments)]
fn write_lc_segment_64_named(
    data: &mut [u8],
    offset: usize,
    segname: &[u8; 16],
    vmaddr: u64,
    vmsize: u64,
    fileoff: u64,
    filesize: u64,
    initprot: u32,
) {
    // cmd (4 bytes)
    data[offset..offset + 4].copy_from_slice(&LC_SEGMENT_64.to_le_bytes());
    // cmdsize (4 bytes)
    data[offset + 4..offset + 8].copy_from_slice(&LC_SEGMENT_64_SIZE.to_le_bytes());
    // segname (16 bytes, null-padded)
    data[offset + 8..offset + 24].copy_from_slice(segname);
    // vmaddr (8 bytes)
    data[offset + 24..offset + 32].copy_from_slice(&vmaddr.to_le_bytes());
    // vmsize (8 bytes)
    data[offset + 32..offset + 40].copy_from_slice(&vmsize.to_le_bytes());
    // fileoff (8 bytes)
    data[offset + 40..offset + 48].copy_from_slice(&fileoff.to_le_bytes());
    // filesize (8 bytes)
    data[offset + 48..offset + 56].copy_from_slice(&filesize.to_le_bytes());
    // maxprot = VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE (0x7)
    data[offset + 56..offset + 60].copy_from_slice(&7u32.to_le_bytes());
    // initprot
    data[offset + 60..offset + 64].copy_from_slice(&initprot.to_le_bytes());
    // nsects = 0
    data[offset + 64..offset + 68].copy_from_slice(&0u32.to_le_bytes());
    // flags = 0
    data[offset + 68..offset + 72].copy_from_slice(&0u32.to_le_bytes());
}

/// LC_DYLD_CHAINED_FIXUPS cmd value.
const LC_DYLD_CHAINED_FIXUPS: u32 = 0x80000034;

/// Additional LC constants for LINKEDIT-referencing load commands.
const LC_SYMTAB: u32 = 0x02;
const LC_DYSYMTAB: u32 = 0x0B;
const LC_DYLD_INFO: u32 = 0x22;
const LC_DYLD_INFO_ONLY: u32 = 0x80000022;
const LC_SEGMENT_SPLIT_INFO: u32 = 0x1E;
const LC_FUNCTION_STARTS: u32 = 0x26;
const LC_DATA_IN_CODE: u32 = 0x29;
const LC_DYLD_EXPORTS_TRIE: u32 = 0x80000033;

/// Shift a u32 field at `offset` in `data` by `shift` bytes.
/// Skips fields that are zero (unused in LC_DYSYMTAB, etc.).
fn shift_lc_u32_field(data: &mut [u8], offset: usize, shift: u64) {
    if offset + 4 > data.len() {
        return;
    }
    let val = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    if val == 0 {
        return; // Skip unused fields
    }
    let new_val = val as u64 + shift;
    data[offset..offset + 4].copy_from_slice(&(new_val as u32).to_le_bytes());
}

/// Walk all load commands and shift file offset fields for LINKEDIT-referencing LCs.
///
/// When __TR_COV is inserted before __LINKEDIT in the file layout, all LINKEDIT
/// data moves to a higher file offset. This function adjusts every LC that
/// references data within LINKEDIT.
#[allow(clippy::too_many_arguments)]
fn update_linkedit_lc_offsets(
    data: &mut [u8],
    ctx: &MachOContext,
    ncmds: u32,
    linkedit_shift: u64,
    new_linkedit_vmaddr: u64,
    new_linkedit_vmsize: u64,
    new_linkedit_fileoff: u64,
    new_linkedit_filesize: u64,
    new_fixups_fileoff: u32,
    new_fixups_size: u32,
) {
    let mut lc_off = ctx.header_size;

    for _ in 0..ncmds {
        if lc_off + 8 > data.len() {
            break;
        }
        let cmd = u32::from_le_bytes(data[lc_off..lc_off + 4].try_into().unwrap());
        let cmdsize = u32::from_le_bytes(data[lc_off + 4..lc_off + 8].try_into().unwrap());
        if cmd == 0 || cmdsize < 8 {
            break;
        }

        match cmd {
            LC_DYLD_INFO | LC_DYLD_INFO_ONLY => {
                // dyld_info_command: rebase_off(+8), bind_off(+16), weak_bind_off(+24),
                // lazy_bind_off(+32), export_off(+40) — skip zeros (unused sections)
                for field_off in [8, 16, 24, 32, 40] {
                    shift_lc_u32_field(data, lc_off + field_off, linkedit_shift);
                }
            }
            LC_SYMTAB => {
                // symtab_command: symoff(+8), stroff(+16)
                shift_lc_u32_field(data, lc_off + 8, linkedit_shift);
                shift_lc_u32_field(data, lc_off + 16, linkedit_shift);
            }
            LC_DYSYMTAB => {
                // dysymtab_command: tocoff(+32), modtaboff(+40), extrefsymoff(+48),
                // indirectsymoff(+56), extreloff(+64), locreloff(+72) — skip zeros
                for field_off in [32, 40, 48, 56, 64, 72] {
                    shift_lc_u32_field(data, lc_off + field_off, linkedit_shift);
                }
            }
            LC_FUNCTION_STARTS | LC_DATA_IN_CODE | LC_DYLD_EXPORTS_TRIE | LC_SEGMENT_SPLIT_INFO => {
                // linkedit_data_command: dataoff(+8)
                shift_lc_u32_field(data, lc_off + 8, linkedit_shift);
            }
            LC_DYLD_CHAINED_FIXUPS => {
                if new_fixups_fileoff > 0 {
                    // Point to rebuilt blob at EOF (not just shifted).
                    data[lc_off + 8..lc_off + 12]
                        .copy_from_slice(&new_fixups_fileoff.to_le_bytes());
                    data[lc_off + 12..lc_off + 16].copy_from_slice(&new_fixups_size.to_le_bytes());
                } else {
                    // No rebuilt blob — just shift the existing offset.
                    shift_lc_u32_field(data, lc_off + 8, linkedit_shift);
                }
            }
            LC_SEGMENT_64 => {
                let segname = &data[lc_off + 8..lc_off + 24];
                if segname.starts_with(b"__LINKEDIT") {
                    // LC_SEGMENT_64: vmaddr(+24), vmsize(+32), fileoff(+40), filesize(+48)
                    data[lc_off + 24..lc_off + 32]
                        .copy_from_slice(&new_linkedit_vmaddr.to_le_bytes());
                    data[lc_off + 32..lc_off + 40]
                        .copy_from_slice(&new_linkedit_vmsize.to_le_bytes());
                    data[lc_off + 40..lc_off + 48]
                        .copy_from_slice(&new_linkedit_fileoff.to_le_bytes());
                    data[lc_off + 48..lc_off + 56]
                        .copy_from_slice(&new_linkedit_filesize.to_le_bytes());
                }
            }
            _ => {}
        }

        lc_off += cmdsize as usize;
    }
}

/// Rebuild the chained fixups blob adding `extra_count` empty entries for new segments.
///
/// dyld validates that seg_count in the chained fixups header matches the number of
/// LC_SEGMENT_64 commands.  When we add segments we must increment seg_count and
/// insert seg_info_offset = 0 (no fixups) entries at the correct position (before
/// __LINKEDIT).
///
/// `extra_count` is 1 for x86_64 (__TR_COV only) or 2 for ARM64 (__TR_DAT + __TR_COV).
fn rebuild_chained_fixups_add_n_empty_segments(
    data: &[u8],
    ctx: &MachOContext,
    extra_count: u32,
) -> Result<Vec<u8>> {
    let orig_off = ctx.chained_fixups_dataoff as usize;
    let orig_size = ctx.chained_fixups_datasize as usize;
    if orig_off + orig_size > data.len() {
        bail!("chained fixups data out of bounds");
    }
    let orig = &data[orig_off..orig_off + orig_size];

    // Parse header (28 bytes)
    let fixups_version = u32::from_le_bytes(orig[0..4].try_into().unwrap());
    let starts_offset = u32::from_le_bytes(orig[4..8].try_into().unwrap()) as usize;
    let imports_offset = u32::from_le_bytes(orig[8..12].try_into().unwrap()) as usize;
    let symbols_offset = u32::from_le_bytes(orig[12..16].try_into().unwrap()) as usize;
    let imports_count = u32::from_le_bytes(orig[16..20].try_into().unwrap());
    let imports_format = u32::from_le_bytes(orig[20..24].try_into().unwrap());
    let symbols_format = u32::from_le_bytes(orig[24..28].try_into().unwrap());

    // Parse starts_in_image
    let orig_seg_count =
        u32::from_le_bytes(orig[starts_offset..starts_offset + 4].try_into().unwrap());
    let new_seg_count = orig_seg_count + extra_count;
    let orig_starts_hdr_end = starts_offset + 4 + orig_seg_count as usize * 4;

    // Read original seg_info_offsets
    let mut orig_seg_offsets = Vec::with_capacity(orig_seg_count as usize);
    for i in 0..orig_seg_count as usize {
        let off = starts_offset + 4 + i * 4;
        orig_seg_offsets.push(u32::from_le_bytes(orig[off..off + 4].try_into().unwrap()));
    }

    // New segments are inserted before __LINKEDIT.
    let insert_idx = ctx
        .segments
        .iter()
        .position(|s| s.name == "__LINKEDIT")
        .unwrap_or(orig_seg_count as usize);

    // Copy existing starts_in_segment data
    let orig_sis_data = if orig_starts_hdr_end < imports_offset {
        &orig[orig_starts_hdr_end..imports_offset]
    } else {
        &orig[0..0]
    };

    // Build new seg_info_offsets: insert `extra_count` zeros at insert_idx,
    // shift all non-zero offsets by `extra_count * 4` (header grew by that many bytes).
    let shift = extra_count * 4;
    let mut new_seg_offsets = Vec::with_capacity(new_seg_count as usize);
    for (i, &off) in orig_seg_offsets.iter().enumerate() {
        if i == insert_idx {
            new_seg_offsets.extend(std::iter::repeat_n(0u32, extra_count as usize));
        }
        // Non-zero offsets shift because the header grew.
        new_seg_offsets.push(if off == 0 { 0 } else { off + shift });
    }
    if insert_idx >= orig_seg_count as usize {
        new_seg_offsets.extend(std::iter::repeat_n(0u32, extra_count as usize));
    }

    // Assemble new starts section (header + sis data unchanged except header shift)
    let mut starts_data = Vec::new();
    starts_data.extend_from_slice(&new_seg_count.to_le_bytes());
    for &off in &new_seg_offsets {
        starts_data.extend_from_slice(&off.to_le_bytes());
    }
    starts_data.extend_from_slice(orig_sis_data);

    // Recompute offsets (starts_offset unchanged; imports/symbols shift by extra_count*4)
    let new_starts_offset = starts_offset as u32;
    let padding_before_starts = starts_offset.saturating_sub(28);
    let new_imports_offset = new_starts_offset + starts_data.len() as u32;
    let new_symbols_offset = new_imports_offset + imports_count * 4;

    // Copy original imports and symbols verbatim
    let orig_imports_data = if imports_offset + imports_count as usize * 4 <= orig.len() {
        &orig[imports_offset..imports_offset + imports_count as usize * 4]
    } else {
        &orig[0..0]
    };
    let orig_symbols = if symbols_offset < orig.len() {
        &orig[symbols_offset..]
    } else {
        &orig[0..0]
    };

    // Assemble final blob
    let mut blob = Vec::with_capacity(orig_size + extra_count as usize * 4);
    // Header
    blob.extend_from_slice(&fixups_version.to_le_bytes());
    blob.extend_from_slice(&new_starts_offset.to_le_bytes());
    blob.extend_from_slice(&new_imports_offset.to_le_bytes());
    blob.extend_from_slice(&new_symbols_offset.to_le_bytes());
    blob.extend_from_slice(&imports_count.to_le_bytes());
    blob.extend_from_slice(&imports_format.to_le_bytes());
    blob.extend_from_slice(&symbols_format.to_le_bytes());
    // Padding
    blob.resize(blob.len() + padding_before_starts, 0);
    // Starts
    blob.extend_from_slice(&starts_data);
    // Imports + symbols
    blob.extend_from_slice(orig_imports_data);
    blob.extend_from_slice(orig_symbols);

    tracing::debug!(
        "chained fixups (empty segs): {} → {} bytes, seg_count {} → {}",
        orig_size,
        blob.len(),
        orig_seg_count,
        new_seg_count,
    );

    Ok(blob)
}

/// Build a new chained fixups blob that includes a `_shmat` import.
///
/// The new blob has:
/// - One extra segment entry in starts_in_image for __TR_COV
/// - A starts_in_segment for __TR_COV with one fixup at offset 16 (the GOT slot)
/// - A new DYLD_CHAINED_IMPORT entry for `_shmat` (lib_ordinal=1, libSystem)
/// - `_shmat\0` appended to the symbol strings
///
/// Returns `(new_blob, shmat_import_ordinal)`.
fn build_chained_fixups_with_shmat(
    data: &[u8],
    ctx: &MachOContext,
    segment_va: u64,
    page_sz: u64,
    libsystem_ordinal: u32,
) -> Result<(Vec<u8>, u32)> {
    let orig_off = ctx.chained_fixups_dataoff as usize;
    let orig_size = ctx.chained_fixups_datasize as usize;
    if orig_off + orig_size > data.len() {
        bail!("chained fixups data out of bounds");
    }
    let orig = &data[orig_off..orig_off + orig_size];

    // Parse header (28 bytes)
    let fixups_version = u32::from_le_bytes(orig[0..4].try_into().unwrap());
    let starts_offset = u32::from_le_bytes(orig[4..8].try_into().unwrap()) as usize;
    let imports_offset = u32::from_le_bytes(orig[8..12].try_into().unwrap()) as usize;
    let symbols_offset = u32::from_le_bytes(orig[12..16].try_into().unwrap()) as usize;
    let imports_count = u32::from_le_bytes(orig[16..20].try_into().unwrap());
    let imports_format = u32::from_le_bytes(orig[20..24].try_into().unwrap());
    let symbols_format = u32::from_le_bytes(orig[24..28].try_into().unwrap());

    if imports_format != 1 {
        bail!(
            "unsupported chained fixups imports_format: {} (only DYLD_CHAINED_IMPORT supported)",
            imports_format,
        );
    }

    // Parse starts_in_image
    let orig_seg_count =
        u32::from_le_bytes(orig[starts_offset..starts_offset + 4].try_into().unwrap());
    let new_seg_count = orig_seg_count + 1;
    let orig_starts_hdr_end = starts_offset + 4 + orig_seg_count as usize * 4;

    // Read original seg_info_offsets (relative to starts_in_image)
    let mut orig_seg_offsets = Vec::with_capacity(orig_seg_count as usize);
    for i in 0..orig_seg_count as usize {
        let off = starts_offset + 4 + i * 4;
        orig_seg_offsets.push(u32::from_le_bytes(orig[off..off + 4].try_into().unwrap()));
    }

    // Find pointer_format from first segment with fixups.
    // starts_in_segment layout: size(4) page_size(2) pointer_format(2) ...
    let mut pointer_format: u16 = 2; // DYLD_CHAINED_PTR_64
    for &off in &orig_seg_offsets {
        if off != 0 {
            let sis_base = starts_offset + off as usize;
            if sis_base + 8 <= orig.len() {
                pointer_format =
                    u16::from_le_bytes(orig[sis_base + 6..sis_base + 8].try_into().unwrap());
            }
            break;
        }
    }

    // Find __LINKEDIT segment index — __TR_COV inserts before it
    let tr_cov_idx = ctx
        .segments
        .iter()
        .position(|s| s.name == "__LINKEDIT")
        .unwrap_or(orig_seg_count as usize);

    // Copy existing starts_in_segment data (between offset array and imports)
    let orig_sis_data = if orig_starts_hdr_end < imports_offset {
        &orig[orig_starts_hdr_end..imports_offset]
    } else {
        &orig[0..0] // empty
    };

    // Build __TR_COV starts_in_segment (24 bytes: 22-byte header + 2-byte page_start)
    let tr_cov_segment_offset = segment_va - ctx.text_segment_vmaddr;
    let mut tr_cov_sis = Vec::with_capacity(24);
    tr_cov_sis.extend_from_slice(&24u32.to_le_bytes()); // size
    tr_cov_sis.extend_from_slice(&(page_sz as u16).to_le_bytes()); // page_size
    tr_cov_sis.extend_from_slice(&pointer_format.to_le_bytes()); // pointer_format
    tr_cov_sis.extend_from_slice(&tr_cov_segment_offset.to_le_bytes()); // segment_offset
    tr_cov_sis.extend_from_slice(&0u32.to_le_bytes()); // max_valid_pointer
    tr_cov_sis.extend_from_slice(&1u16.to_le_bytes()); // page_count = 1
    tr_cov_sis.extend_from_slice(&16u16.to_le_bytes()); // page_start[0] = 16 (GOT slot)

    // Build new seg_info_offsets
    // Adding one entry (+4 bytes) shifts all existing sis data by 4
    let new_starts_hdr_size = 4 + new_seg_count as usize * 4; // relative to starts_in_image
    let tr_cov_sis_offset = (new_starts_hdr_size + orig_sis_data.len()) as u32;

    let mut new_seg_offsets = Vec::with_capacity(new_seg_count as usize);
    for (i, &off) in orig_seg_offsets.iter().enumerate() {
        if i == tr_cov_idx {
            new_seg_offsets.push(tr_cov_sis_offset);
        }
        new_seg_offsets.push(if off == 0 { 0 } else { off + 4 });
    }
    if tr_cov_idx >= orig_seg_count as usize {
        new_seg_offsets.push(tr_cov_sis_offset);
    }

    // Assemble starts section
    let mut starts_data = Vec::new();
    starts_data.extend_from_slice(&new_seg_count.to_le_bytes());
    for &off in &new_seg_offsets {
        starts_data.extend_from_slice(&off.to_le_bytes());
    }
    starts_data.extend_from_slice(orig_sis_data);
    starts_data.extend_from_slice(&tr_cov_sis);

    // Copy original imports (4 bytes each for format 1)
    let orig_imports_data = &orig[imports_offset..imports_offset + imports_count as usize * 4];

    // Copy original symbols
    let orig_symbols = &orig[symbols_offset..];

    // Build new _shmat import: lib_ordinal(8) | weak_import(1) | name_offset(23)
    // lib_ordinal is 1-based index into LC_LOAD_DYLIB commands; use the libSystem ordinal.
    // shmat is in /usr/lib/libSystem.B.dylib (re-exported from libsystem_kernel.dylib).
    let shmat_name_offset = orig_symbols.len() as u32;
    let shmat_import: u32 = (libsystem_ordinal & 0xFF) /* lib_ordinal in bits 0-7, weak_import=0 in bit 8 */
        | (shmat_name_offset << 9);

    // Compute final offsets (all relative to blob start)
    let new_starts_offset = starts_offset as u32; // keep same padding as original
    let padding_before_starts = starts_offset.saturating_sub(28);
    let new_imports_offset = new_starts_offset + starts_data.len() as u32;
    let new_imports_count = imports_count + 1;
    let new_symbols_offset = new_imports_offset + new_imports_count * 4;

    // Assemble final blob
    let mut blob = Vec::with_capacity(orig_size + 64);

    // Header (28 bytes)
    blob.extend_from_slice(&fixups_version.to_le_bytes());
    blob.extend_from_slice(&new_starts_offset.to_le_bytes());
    blob.extend_from_slice(&new_imports_offset.to_le_bytes());
    blob.extend_from_slice(&new_symbols_offset.to_le_bytes());
    blob.extend_from_slice(&new_imports_count.to_le_bytes());
    blob.extend_from_slice(&imports_format.to_le_bytes());
    blob.extend_from_slice(&symbols_format.to_le_bytes());

    // Padding to starts_offset
    blob.resize(blob.len() + padding_before_starts, 0);

    // Starts data
    blob.extend_from_slice(&starts_data);

    // Imports (original + new)
    blob.extend_from_slice(orig_imports_data);
    blob.extend_from_slice(&shmat_import.to_le_bytes());

    // Symbols (original + "_shmat\0")
    blob.extend_from_slice(orig_symbols);
    blob.extend_from_slice(b"_shmat\0");

    tracing::debug!(
        "chained fixups blob: {} → {} bytes, seg_count {} → {}, imports {} → {}",
        orig_size,
        blob.len(),
        orig_seg_count,
        new_seg_count,
        imports_count,
        new_imports_count,
    );

    Ok((blob, imports_count)) // shmat ordinal = old imports_count (0-based)
}

/// Redirect dylib entry to our init code.
///
/// Strategy: If __mod_init_func exists and has pointers, overwrite the first
/// 8-byte pointer. If __init_offsets exists, overwrite the first 4-byte offset.
/// Our init chains to the original via JMP/BR.
/// If neither exists, do nothing here — the caller will add LC_ROUTINES_64.
fn redirect_dylib_entry(data: &mut [u8], ctx: &MachOContext, init_va: u64) -> Result<()> {
    // Prefer __mod_init_func (64-bit absolute pointers)
    if let Some(ref mod_init) = ctx.mod_init_func_section {
        if mod_init.section_size >= 8 {
            let off = mod_init.section_offset as usize;
            if off + 8 <= data.len() {
                data[off..off + 8].copy_from_slice(&init_va.to_le_bytes());
                tracing::info!(
                    "patched __mod_init_func[0] to 0x{:x} (chains to original)",
                    init_va,
                );
                return Ok(());
            }
        }
    }

    // Fall back to __init_offsets (32-bit offsets from __TEXT base)
    if let Some(ref init_off) = ctx.init_offsets_section {
        if init_off.section_size >= 4 {
            let off = init_off.section_offset as usize;
            if off + 4 <= data.len() {
                let offset32 = (init_va - init_off.text_segment_vmaddr) as u32;
                data[off..off + 4].copy_from_slice(&offset32.to_le_bytes());
                tracing::info!(
                    "patched __init_offsets[0] to offset 0x{:x} (VA 0x{:x}, chains to original)",
                    offset32,
                    init_va,
                );
                return Ok(());
            }
        }
    }

    // Neither exists — LC_ROUTINES_64 will be added by the caller.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disasm::BasicBlock;
    use crate::macho::MachOSegment;
    use crate::traits::TrampolineGenerator;
    use crate::trampoline::{InitCode, Trampoline};

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0x1000, 0x1000), 0x1000);
        assert_eq!(align_up(0x1001, 0x1000), 0x2000);
        assert_eq!(align_up(0, 0x4000), 0);
        assert_eq!(align_up(0x4001, 0x4000), 0x8000);
    }

    #[test]
    fn test_write_lc_segment_64() {
        let mut data = vec![0u8; 128];
        write_lc_segment_64(&mut data, 0, 0x200000, 0x4000, 0x8000, 0x1234, 0x7);

        // cmd = LC_SEGMENT_64
        assert_eq!(u32::from_le_bytes(data[0..4].try_into().unwrap()), 0x19);
        // cmdsize = 72
        assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()), 72);
        // segname starts with "__TR_COV"
        assert_eq!(&data[8..16], b"__TR_COV");
        // vmaddr
        assert_eq!(
            u64::from_le_bytes(data[24..32].try_into().unwrap()),
            0x200000
        );
        // vmsize
        assert_eq!(u64::from_le_bytes(data[32..40].try_into().unwrap()), 0x4000);
        // fileoff
        assert_eq!(u64::from_le_bytes(data[40..48].try_into().unwrap()), 0x8000);
        // filesize
        assert_eq!(u64::from_le_bytes(data[48..56].try_into().unwrap()), 0x1234);
        // maxprot = 7
        assert_eq!(u32::from_le_bytes(data[56..60].try_into().unwrap()), 7);
    }

    // --- shift_lc_u32_field tests ---

    #[test]
    fn test_shift_lc_u32_field_nonzero() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(&1000u32.to_le_bytes());
        shift_lc_u32_field(&mut data, 0, 0x2000);
        assert_eq!(
            u32::from_le_bytes(data[0..4].try_into().unwrap()),
            1000 + 0x2000,
        );
    }

    #[test]
    fn test_shift_lc_u32_field_zero_skipped() {
        let mut data = vec![0u8; 8];
        // Zero field should remain zero (unused LC_DYSYMTAB fields).
        shift_lc_u32_field(&mut data, 0, 0x5000);
        assert_eq!(u32::from_le_bytes(data[0..4].try_into().unwrap()), 0);
    }

    #[test]
    fn test_shift_lc_u32_field_out_of_bounds() {
        let mut data = vec![0u8; 2]; // too small for u32
        data[0] = 0xFF;
        shift_lc_u32_field(&mut data, 0, 0x1000);
        // Should not panic; data unchanged.
        assert_eq!(data[0], 0xFF);
    }

    #[test]
    fn test_shift_lc_u32_field_large_shift() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(&0x8000u32.to_le_bytes());
        // Shift by a large amount (simulating TR_COV + padding between LINKEDIT)
        shift_lc_u32_field(&mut data, 0, 0x10_0000);
        assert_eq!(
            u32::from_le_bytes(data[0..4].try_into().unwrap()),
            0x8000 + 0x10_0000,
        );
    }

    // --- update_linkedit_lc_offsets tests ---

    /// Helper: write a u32 at offset in data.
    fn put_u32(data: &mut [u8], off: usize, val: u32) {
        data[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }

    /// Helper: write a u64 at offset in data.
    fn put_u64(data: &mut [u8], off: usize, val: u64) {
        data[off..off + 8].copy_from_slice(&val.to_le_bytes());
    }

    /// Helper: read a u32 from offset.
    fn get_u32(data: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(data[off..off + 4].try_into().unwrap())
    }

    /// Helper: read a u64 from offset.
    fn get_u64(data: &[u8], off: usize) -> u64 {
        u64::from_le_bytes(data[off..off + 8].try_into().unwrap())
    }

    /// Build a synthetic Mach-O header with __TEXT, __LINKEDIT, LC_MAIN,
    /// LC_SYMTAB, and LC_FUNCTION_STARTS. Returns (data, ctx).
    ///
    /// Layout:
    ///   0x0000: mach_header_64 (32 bytes)
    ///   0x0020: LC_SEGMENT_64 __TEXT (72+80 = 152 bytes, with __text section)
    ///   0x00B8: LC_MAIN (24 bytes)
    ///   0x00D0: LC_SYMTAB (24 bytes)
    ///   0x00E8: LC_FUNCTION_STARTS (16 bytes)
    ///   0x00F8: LC_DYSYMTAB (80 bytes)
    ///   0x0148: LC_SEGMENT_64 __LINKEDIT (72 bytes)
    ///   ...padding to 0x1000...
    ///   0x1000: __text code (256 bytes: NOP sled + RET)
    ///   0x1100: padding to 0x2000
    ///   0x2000: LINKEDIT data (256 bytes of 0xAA pattern)
    fn build_macho_with_linkedit(cputype: u32) -> (Vec<u8>, MachOContext) {
        let page_sz: usize = if cputype == 0x0100_000C {
            0x4000
        } else {
            0x1000
        };
        let text_seg_vmaddr: u64 = 0x10000_0000;
        let text_seg_fileoff: u64 = 0;
        let text_seg_size: u64 = page_sz as u64;

        let text_section_va = text_seg_vmaddr + 0x800;
        let text_section_fileoff: u64 = 0x800;
        let text_section_size: u64 = 0x100;

        let linkedit_fileoff = page_sz as u64 * 2;
        let linkedit_data_len: u64 = 256;
        let linkedit_vmaddr = text_seg_vmaddr + linkedit_fileoff;
        let linkedit_vmsize: u64 = align_up(linkedit_data_len, page_sz as u64);

        let total_size = linkedit_fileoff as usize + linkedit_data_len as usize;
        let mut data = vec![0u8; total_size];

        let header_size = 32usize;
        let mut lc_off = header_size;

        // --- LC_SEGMENT_64 __TEXT (72 bytes + 80-byte section = 152) ---
        let text_lc_size = 152u32;
        put_u32(&mut data, lc_off, LC_SEGMENT_64);
        put_u32(&mut data, lc_off + 4, text_lc_size);
        data[lc_off + 8..lc_off + 14].copy_from_slice(b"__TEXT");
        put_u64(&mut data, lc_off + 24, text_seg_vmaddr);
        put_u64(&mut data, lc_off + 32, text_seg_size);
        put_u64(&mut data, lc_off + 40, text_seg_fileoff);
        put_u64(&mut data, lc_off + 48, text_seg_size);
        put_u32(&mut data, lc_off + 56, 7); // maxprot
        put_u32(&mut data, lc_off + 60, 5); // initprot
        put_u32(&mut data, lc_off + 64, 1); // nsects
        // Section __text
        let sect_off = lc_off + 72;
        data[sect_off..sect_off + 6].copy_from_slice(b"__text");
        data[sect_off + 16..sect_off + 22].copy_from_slice(b"__TEXT");
        put_u64(&mut data, sect_off + 32, text_section_va);
        put_u64(&mut data, sect_off + 40, text_section_size);
        put_u32(&mut data, sect_off + 48, text_section_fileoff as u32);
        lc_off += text_lc_size as usize;

        // --- LC_MAIN (24 bytes) ---
        let lc_main_off = lc_off;
        put_u32(&mut data, lc_off, 0x80000028); // LC_MAIN
        put_u32(&mut data, lc_off + 4, 24);
        put_u64(&mut data, lc_off + 8, 0x800); // entryoff
        lc_off += 24;

        // --- LC_SYMTAB (24 bytes) ---
        let symtab_symoff = linkedit_fileoff as u32 + 0; // symbols at start of LINKEDIT
        let symtab_stroff = linkedit_fileoff as u32 + 64; // strings 64 bytes in
        put_u32(&mut data, lc_off, LC_SYMTAB);
        put_u32(&mut data, lc_off + 4, 24);
        put_u32(&mut data, lc_off + 8, symtab_symoff); // symoff
        put_u32(&mut data, lc_off + 12, 2); // nsyms
        put_u32(&mut data, lc_off + 16, symtab_stroff); // stroff
        put_u32(&mut data, lc_off + 20, 32); // strsize
        lc_off += 24;

        // --- LC_FUNCTION_STARTS (16 bytes) ---
        let func_starts_off = linkedit_fileoff as u32 + 128;
        put_u32(&mut data, lc_off, LC_FUNCTION_STARTS);
        put_u32(&mut data, lc_off + 4, 16);
        put_u32(&mut data, lc_off + 8, func_starts_off); // dataoff
        put_u32(&mut data, lc_off + 12, 16); // datasize
        lc_off += 16;

        // --- LC_DYSYMTAB (80 bytes) ---
        let dysymtab_indirectsymoff = linkedit_fileoff as u32 + 192;
        put_u32(&mut data, lc_off, LC_DYSYMTAB);
        put_u32(&mut data, lc_off + 4, 80);
        // tocoff(+32)=0, modtaboff(+40)=0, extrefsymoff(+48)=0 (unused)
        put_u32(&mut data, lc_off + 56, dysymtab_indirectsymoff); // indirectsymoff
        put_u32(&mut data, lc_off + 60, 4); // nindirectsyms
        // extreloff(+64)=0, locreloff(+72)=0 (unused)
        lc_off += 80;

        // --- LC_SEGMENT_64 __LINKEDIT (72 bytes) ---
        let linkedit_seg_lc_off = lc_off;
        put_u32(&mut data, lc_off, LC_SEGMENT_64);
        put_u32(&mut data, lc_off + 4, 72);
        data[lc_off + 8..lc_off + 18].copy_from_slice(b"__LINKEDIT");
        put_u64(&mut data, lc_off + 24, linkedit_vmaddr);
        put_u64(&mut data, lc_off + 32, linkedit_vmsize);
        put_u64(&mut data, lc_off + 40, linkedit_fileoff);
        put_u64(&mut data, lc_off + 48, linkedit_data_len);
        put_u32(&mut data, lc_off + 56, 1); // maxprot = VM_PROT_READ
        put_u32(&mut data, lc_off + 60, 1); // initprot = VM_PROT_READ
        lc_off += 72;

        let ncmds = 6u32;
        let sizeofcmds = (lc_off - header_size) as u32;

        // --- Mach-O header ---
        put_u32(&mut data, 0, 0xFEEDFACF); // MH_MAGIC_64
        put_u32(&mut data, 4, cputype);
        put_u32(&mut data, 8, 3); // cpusubtype
        put_u32(&mut data, 12, 2); // MH_EXECUTE
        put_u32(&mut data, 16, ncmds);
        put_u32(&mut data, 20, sizeofcmds);

        // Fill __text with NOP+RET
        for i in 0..0xFF {
            data[text_section_fileoff as usize + i] = 0x90;
        }
        data[text_section_fileoff as usize + 0xFF] = 0xC3;

        // Fill LINKEDIT data with recognisable pattern
        for i in 0..linkedit_data_len as usize {
            data[linkedit_fileoff as usize + i] = 0xAA;
        }

        let ctx = MachOContext {
            text: crate::macho::TextSection {
                va: text_section_va,
                offset: text_section_fileoff,
                size: text_section_size,
            },
            entry_point: text_section_va,
            func_symbols: vec![text_section_va],
            highest_va_end: linkedit_vmaddr + linkedit_vmsize,
            is_dylib: false,
            is_64bit: true,
            segments: vec![
                MachOSegment {
                    name: "__TEXT".into(),
                    vmaddr: text_seg_vmaddr,
                    vmsize: text_seg_size,
                    fileoff: text_seg_fileoff,
                    filesize: text_seg_size,
                },
                MachOSegment {
                    name: "__LINKEDIT".into(),
                    vmaddr: linkedit_vmaddr,
                    vmsize: linkedit_vmsize,
                    fileoff: linkedit_fileoff,
                    filesize: linkedit_data_len,
                },
            ],
            stubs_ranges: vec![],
            header_size,
            lc_end_offset: lc_off as u64,
            first_section_offset: text_section_fileoff,
            available_lc_space: text_section_fileoff - lc_off as u64,
            lc_main_entryoff_offset: Some(lc_main_off as u64 + 8),
            lc_unixthread_pc_offset: None,
            text_segment_vmaddr: text_seg_vmaddr,
            mod_init_func_section: None,
            init_offsets_section: None,
            mod_init_func_pointers: vec![],
            code_signature_lc_offset: None,
            code_signature_lc_size: 0,
            linkedit_segment_offset: Some(linkedit_seg_lc_off as u64),
            linkedit_vmaddr,
            linkedit_fileoff,
            linkedit_filesize: linkedit_data_len,
            ncmds,
            sizeofcmds,
            ncmds_offset: 16,
            sizeofcmds_offset: 20,
            chained_fixups_lc_offset: None,
            chained_fixups_dataoff: 0,
            chained_fixups_datasize: 0,
            cputype,
            libsystem_ordinal: 1,
        };

        (data, ctx)
    }

    // --- update_linkedit_lc_offsets tests ---

    #[test]
    fn test_update_linkedit_lc_offsets_symtab() {
        let (mut data, ctx) = build_macho_with_linkedit(0x0100_0007); // x86_64
        let orig_symoff = get_u32(&data, ctx.header_size + 152 + 24 + 8);
        let orig_stroff = get_u32(&data, ctx.header_size + 152 + 24 + 16);
        let shift = 0x4000u64;

        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            ctx.linkedit_vmaddr + shift,
            align_up(ctx.linkedit_filesize + shift, 0x1000),
            ctx.linkedit_fileoff + shift,
            ctx.linkedit_filesize,
            0,
            0,
        );

        assert_eq!(
            get_u32(&data, ctx.header_size + 152 + 24 + 8),
            orig_symoff + shift as u32,
            "LC_SYMTAB symoff should be shifted"
        );
        assert_eq!(
            get_u32(&data, ctx.header_size + 152 + 24 + 16),
            orig_stroff + shift as u32,
            "LC_SYMTAB stroff should be shifted"
        );
    }

    #[test]
    fn test_update_linkedit_lc_offsets_function_starts() {
        let (mut data, ctx) = build_macho_with_linkedit(0x0100_0007);
        // LC_FUNCTION_STARTS is at offset: header + __TEXT(152) + LC_MAIN(24) + LC_SYMTAB(24)
        let func_starts_lc = ctx.header_size + 152 + 24 + 24;
        let orig_dataoff = get_u32(&data, func_starts_lc + 8);
        let shift = 0x8000u64;

        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            ctx.linkedit_vmaddr + shift,
            align_up(ctx.linkedit_filesize + shift, 0x1000),
            ctx.linkedit_fileoff + shift,
            ctx.linkedit_filesize,
            0,
            0,
        );

        assert_eq!(
            get_u32(&data, func_starts_lc + 8),
            orig_dataoff + shift as u32,
            "LC_FUNCTION_STARTS dataoff should be shifted"
        );
    }

    #[test]
    fn test_update_linkedit_lc_offsets_dysymtab_skip_zeros() {
        let (mut data, ctx) = build_macho_with_linkedit(0x0100_0007);
        // LC_DYSYMTAB is at: header + __TEXT(152) + LC_MAIN(24) + LC_SYMTAB(24) + LC_FUNC_STARTS(16)
        let dysymtab_lc = ctx.header_size + 152 + 24 + 24 + 16;
        let orig_indirectsymoff = get_u32(&data, dysymtab_lc + 56);
        let shift = 0x3000u64;

        // Verify unused fields are zero before shift
        assert_eq!(get_u32(&data, dysymtab_lc + 32), 0, "tocoff should be zero");
        assert_eq!(
            get_u32(&data, dysymtab_lc + 40),
            0,
            "modtaboff should be zero"
        );

        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            ctx.linkedit_vmaddr + shift,
            align_up(ctx.linkedit_filesize, 0x1000),
            ctx.linkedit_fileoff + shift,
            ctx.linkedit_filesize,
            0,
            0,
        );

        // Zero fields must remain zero (skip rule).
        assert_eq!(
            get_u32(&data, dysymtab_lc + 32),
            0,
            "tocoff should stay zero"
        );
        assert_eq!(
            get_u32(&data, dysymtab_lc + 40),
            0,
            "modtaboff should stay zero"
        );
        assert_eq!(
            get_u32(&data, dysymtab_lc + 48),
            0,
            "extrefsymoff should stay zero"
        );
        assert_eq!(
            get_u32(&data, dysymtab_lc + 64),
            0,
            "extreloff should stay zero"
        );
        assert_eq!(
            get_u32(&data, dysymtab_lc + 72),
            0,
            "locreloff should stay zero"
        );
        // Non-zero field should be shifted.
        assert_eq!(
            get_u32(&data, dysymtab_lc + 56),
            orig_indirectsymoff + shift as u32,
            "indirectsymoff should be shifted"
        );
    }

    #[test]
    fn test_update_linkedit_lc_offsets_linkedit_segment() {
        let (mut data, ctx) = build_macho_with_linkedit(0x0100_000C); // ARM64
        let linkedit_lc = ctx.linkedit_segment_offset.unwrap() as usize;
        let shift = 0x10000u64;
        let new_vmaddr = ctx.linkedit_vmaddr + shift;
        let new_fileoff = ctx.linkedit_fileoff + shift;
        let new_filesize = ctx.linkedit_filesize + 64;
        let new_vmsize = align_up(new_filesize, 0x4000);

        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            new_vmaddr,
            new_vmsize,
            new_fileoff,
            new_filesize,
            0,
            0,
        );

        assert_eq!(
            get_u64(&data, linkedit_lc + 24),
            new_vmaddr,
            "LINKEDIT vmaddr should be updated"
        );
        assert_eq!(
            get_u64(&data, linkedit_lc + 32),
            new_vmsize,
            "LINKEDIT vmsize should be updated"
        );
        assert_eq!(
            get_u64(&data, linkedit_lc + 40),
            new_fileoff,
            "LINKEDIT fileoff should be updated"
        );
        assert_eq!(
            get_u64(&data, linkedit_lc + 48),
            new_filesize,
            "LINKEDIT filesize should be updated"
        );
    }

    // --- Mock trampoline generator for patch_macho integration tests ---

    #[derive(Debug)]
    struct MockTrampolineGen;

    impl TrampolineGenerator for MockTrampolineGen {
        fn generate_trampoline(
            &self,
            trampoline_va: u64,
            _data_va: u64,
            block: &BasicBlock,
        ) -> anyhow::Result<Trampoline> {
            // Minimal trampoline: displaced bytes + JMP back
            let mut code = block.displaced_bytes.to_vec();
            code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]); // JMP rel32 placeholder
            Ok(Trampoline {
                va: trampoline_va,
                code,
            })
        }

        fn generate_init_code(
            &self,
            init_va: u64,
            _data_va: u64,
            _entry_point: u64,
            _forkserver: bool,
            _persistent_data_va: Option<u64>,
        ) -> anyhow::Result<InitCode> {
            Ok(InitCode {
                va: init_va,
                code: vec![0xC3],
                entry_va: init_va,
            })
        }

        fn generate_so_init_code(
            &self,
            init_va: u64,
            _data_va: u64,
            _dt_init: Option<u64>,
        ) -> anyhow::Result<InitCode> {
            Ok(InitCode {
                va: init_va,
                code: vec![0xC3],
                entry_va: init_va,
            })
        }

        fn encode_branch(&self, _src: u64, _tgt: u64) -> anyhow::Result<Vec<u8>> {
            Ok(vec![0xE9, 0x00, 0x00, 0x00, 0x00]) // JMP rel32
        }

        fn branch_instruction_size(&self) -> usize {
            5
        }
    }

    // --- Full patch_macho integration: segment ordering ---

    /// Walk load commands in a patched binary and extract segment info.
    fn read_segments(data: &[u8], header_size: usize) -> Vec<(String, u64, u64, u64, u64)> {
        let ncmds = get_u32(data, 16) as usize;
        let mut off = header_size;
        let mut segs = Vec::new();
        for _ in 0..ncmds {
            if off + 8 > data.len() {
                break;
            }
            let cmd = get_u32(data, off);
            let cmdsize = get_u32(data, off + 4);
            if cmd == 0 || cmdsize < 8 {
                break;
            }
            if cmd == LC_SEGMENT_64 && off + 72 <= data.len() {
                let name_bytes = &data[off + 8..off + 24];
                let name_end = name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
                let name = String::from_utf8_lossy(&name_bytes[..name_end]).to_string();
                let vmaddr = get_u64(data, off + 24);
                let vmsize = get_u64(data, off + 32);
                let fileoff = get_u64(data, off + 40);
                let filesize = get_u64(data, off + 48);
                segs.push((name, vmaddr, vmsize, fileoff, filesize));
            }
            off += cmdsize as usize;
        }
        segs
    }

    /// Read LC_SYMTAB offsets from patched binary.
    fn read_symtab(data: &[u8], header_size: usize) -> Option<(u32, u32)> {
        let ncmds = get_u32(data, 16) as usize;
        let mut off = header_size;
        for _ in 0..ncmds {
            if off + 8 > data.len() {
                break;
            }
            let cmd = get_u32(data, off);
            let cmdsize = get_u32(data, off + 4);
            if cmd == 0 || cmdsize < 8 {
                break;
            }
            if cmd == LC_SYMTAB {
                return Some((get_u32(data, off + 8), get_u32(data, off + 16)));
            }
            off += cmdsize as usize;
        }
        None
    }

    /// Read LC_FUNCTION_STARTS dataoff from patched binary.
    fn read_function_starts_offset(data: &[u8], header_size: usize) -> Option<u32> {
        let ncmds = get_u32(data, 16) as usize;
        let mut off = header_size;
        for _ in 0..ncmds {
            if off + 8 > data.len() {
                break;
            }
            let cmd = get_u32(data, off);
            let cmdsize = get_u32(data, off + 4);
            if cmd == 0 || cmdsize < 8 {
                break;
            }
            if cmd == LC_FUNCTION_STARTS {
                return Some(get_u32(data, off + 8));
            }
            off += cmdsize as usize;
        }
        None
    }

    /// Read LC_DYSYMTAB indirectsymoff from patched binary.
    fn read_dysymtab_indirectsymoff(data: &[u8], header_size: usize) -> Option<u32> {
        let ncmds = get_u32(data, 16) as usize;
        let mut off = header_size;
        for _ in 0..ncmds {
            if off + 8 > data.len() {
                break;
            }
            let cmd = get_u32(data, off);
            let cmdsize = get_u32(data, off + 4);
            if cmd == 0 || cmdsize < 8 {
                break;
            }
            if cmd == LC_DYSYMTAB {
                return Some(get_u32(data, off + 56));
            }
            off += cmdsize as usize;
        }
        None
    }

    #[test]
    fn test_patch_macho_x86_64_trcov_before_linkedit_vm() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_0007);
        let orig_linkedit_vmaddr = ctx.linkedit_vmaddr;

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs
            .iter()
            .find(|s| s.0 == "__TR_COV")
            .expect("__TR_COV missing");
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT missing");

        // TR_COV takes LINKEDIT's old vmaddr.
        assert_eq!(
            tr_cov.1, orig_linkedit_vmaddr,
            "TR_COV vmaddr should equal original LINKEDIT vmaddr"
        );
        // LINKEDIT must have higher vmaddr than TR_COV.
        assert!(
            linkedit.1 > tr_cov.1,
            "LINKEDIT vmaddr (0x{:x}) must be > TR_COV vmaddr (0x{:x})",
            linkedit.1,
            tr_cov.1
        );
        // LINKEDIT vmaddr = TR_COV vmaddr + TR_COV vmsize.
        assert_eq!(
            linkedit.1,
            tr_cov.1 + tr_cov.2,
            "LINKEDIT vmaddr should be TR_COV vmaddr + vmsize"
        );
    }

    #[test]
    fn test_patch_macho_x86_64_trcov_before_linkedit_file() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_0007);

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs.iter().find(|s| s.0 == "__TR_COV").unwrap();
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();

        // In file layout, TR_COV data must come before LINKEDIT data.
        assert!(
            tr_cov.3 < linkedit.3,
            "TR_COV fileoff (0x{:x}) must be < LINKEDIT fileoff (0x{:x})",
            tr_cov.3,
            linkedit.3
        );
        // TR_COV data + page-aligned padding must not overlap LINKEDIT.
        assert!(
            tr_cov.3 + align_up(tr_cov.4, 0x1000) <= linkedit.3,
            "TR_COV data region must not overlap LINKEDIT"
        );
    }

    #[test]
    fn test_patch_macho_x86_64_linkedit_extends_to_eof() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_0007);

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();

        // LINKEDIT fileoff + filesize must equal total file size (extends to EOF).
        assert_eq!(
            linkedit.3 + linkedit.4,
            result.data.len() as u64,
            "LINKEDIT must extend to EOF: fileoff(0x{:x}) + filesize(0x{:x}) = 0x{:x}, but file is 0x{:x}",
            linkedit.3,
            linkedit.4,
            linkedit.3 + linkedit.4,
            result.data.len()
        );
    }

    #[test]
    fn test_patch_macho_x86_64_linkedit_data_intact() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_0007);
        let orig_linkedit = data[ctx.linkedit_fileoff as usize..].to_vec();

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();
        let new_off = linkedit.3 as usize;

        // Original LINKEDIT data should appear at the new offset, byte-for-byte.
        let relocated = &result.data[new_off..new_off + orig_linkedit.len()];
        assert_eq!(
            relocated,
            orig_linkedit.as_slice(),
            "LINKEDIT data must be preserved after relocation"
        );
    }

    #[test]
    fn test_patch_macho_x86_64_lc_symtab_shifted() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_0007);
        let orig_symtab = read_symtab(&data, ctx.header_size).unwrap();

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();
        let shift = linkedit.3 - ctx.linkedit_fileoff;

        let new_symtab = read_symtab(&result.data, ctx.header_size).unwrap();
        assert_eq!(
            new_symtab.0,
            orig_symtab.0 + shift as u32,
            "symoff should be shifted by 0x{:x}",
            shift
        );
        assert_eq!(
            new_symtab.1,
            orig_symtab.1 + shift as u32,
            "stroff should be shifted by 0x{:x}",
            shift
        );
    }

    #[test]
    fn test_patch_macho_x86_64_lc_function_starts_shifted() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_0007);
        let orig_fs = read_function_starts_offset(&data, ctx.header_size).unwrap();

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();
        let shift = linkedit.3 - ctx.linkedit_fileoff;

        let new_fs = read_function_starts_offset(&result.data, ctx.header_size).unwrap();
        assert_eq!(
            new_fs,
            orig_fs + shift as u32,
            "LC_FUNCTION_STARTS dataoff should be shifted by 0x{:x}",
            shift
        );
    }

    #[test]
    fn test_patch_macho_x86_64_lc_dysymtab_shifted() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_0007);
        let orig_indirect = read_dysymtab_indirectsymoff(&data, ctx.header_size).unwrap();

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();
        let shift = linkedit.3 - ctx.linkedit_fileoff;

        let new_indirect = read_dysymtab_indirectsymoff(&result.data, ctx.header_size).unwrap();
        assert_eq!(
            new_indirect,
            orig_indirect + shift as u32,
            "indirectsymoff should be shifted by 0x{:x}",
            shift
        );
    }

    #[test]
    fn test_patch_macho_x86_64_lc_offsets_within_file() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_0007);

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let file_len = result.data.len() as u32;

        // Every shifted offset must be < file length.
        let symtab = read_symtab(&result.data, ctx.header_size).unwrap();
        assert!(symtab.0 < file_len, "symoff out of file bounds");
        assert!(symtab.1 < file_len, "stroff out of file bounds");

        let fs = read_function_starts_offset(&result.data, ctx.header_size).unwrap();
        assert!(fs < file_len, "function_starts dataoff out of file bounds");

        let indirect = read_dysymtab_indirectsymoff(&result.data, ctx.header_size).unwrap();
        assert!(indirect < file_len, "indirectsymoff out of file bounds");
    }

    // --- ARM64 architecture tests (16KB pages) ---

    #[test]
    fn test_patch_macho_arm64_trcov_before_linkedit_vm() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_000C);
        let orig_linkedit_vmaddr = ctx.linkedit_vmaddr;

        let result = patch_macho(
            &ctx,
            &[],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            true, // no_coverage — ARM64 MockTrampolineGen emits x86
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs
            .iter()
            .find(|s| s.0 == "__TR_COV")
            .expect("__TR_COV missing");
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT missing");

        assert_eq!(
            tr_cov.1, orig_linkedit_vmaddr,
            "ARM64: TR_COV vmaddr should equal original LINKEDIT vmaddr"
        );
        assert!(
            linkedit.1 > tr_cov.1,
            "ARM64: LINKEDIT vmaddr must be > TR_COV vmaddr"
        );
        assert_eq!(
            linkedit.1,
            tr_cov.1 + tr_cov.2,
            "ARM64: LINKEDIT vmaddr should be TR_COV vmaddr + vmsize"
        );
    }

    #[test]
    fn test_patch_macho_arm64_trcov_before_linkedit_file() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_000C);

        let result = patch_macho(
            &ctx,
            &[],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            true,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs.iter().find(|s| s.0 == "__TR_COV").unwrap();
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();

        assert!(
            tr_cov.3 < linkedit.3,
            "ARM64: TR_COV fileoff (0x{:x}) must be < LINKEDIT fileoff (0x{:x})",
            tr_cov.3,
            linkedit.3
        );
    }

    #[test]
    fn test_patch_macho_arm64_linkedit_extends_to_eof() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_000C);

        let result = patch_macho(
            &ctx,
            &[],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            true,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();

        assert_eq!(
            linkedit.3 + linkedit.4,
            result.data.len() as u64,
            "ARM64: LINKEDIT must extend to EOF"
        );
    }

    #[test]
    fn test_patch_macho_arm64_page_alignment() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_000C);

        let result = patch_macho(
            &ctx,
            &[],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            true,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs.iter().find(|s| s.0 == "__TR_COV").unwrap();
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();

        // ARM64 uses 16KB pages — all segment offsets must be 16KB-aligned.
        assert_eq!(tr_cov.1 % 0x4000, 0, "TR_COV vmaddr must be 16KB-aligned");
        assert_eq!(tr_cov.3 % 0x4000, 0, "TR_COV fileoff must be 16KB-aligned");
        assert_eq!(
            linkedit.1 % 0x4000,
            0,
            "LINKEDIT vmaddr must be 16KB-aligned"
        );
        assert_eq!(
            linkedit.3 % 0x4000,
            0,
            "LINKEDIT fileoff must be 16KB-aligned"
        );
        assert_eq!(tr_cov.2 % 0x4000, 0, "TR_COV vmsize must be 16KB-aligned");
        assert_eq!(
            linkedit.2 % 0x4000,
            0,
            "LINKEDIT vmsize must be 16KB-aligned"
        );
    }

    #[test]
    fn test_patch_macho_arm64_linkedit_data_intact() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_000C);
        let orig_linkedit = data[ctx.linkedit_fileoff as usize..].to_vec();

        let result = patch_macho(
            &ctx,
            &[],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            true,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();
        let new_off = linkedit.3 as usize;

        let relocated = &result.data[new_off..new_off + orig_linkedit.len()];
        assert_eq!(
            relocated,
            orig_linkedit.as_slice(),
            "ARM64: LINKEDIT data must be preserved after relocation"
        );
    }

    #[test]
    fn test_patch_macho_arm64_lc_offsets_shifted() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_000C);
        let orig_symtab = read_symtab(&data, ctx.header_size).unwrap();
        let orig_fs = read_function_starts_offset(&data, ctx.header_size).unwrap();
        let orig_indirect = read_dysymtab_indirectsymoff(&data, ctx.header_size).unwrap();

        let result = patch_macho(
            &ctx,
            &[],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            true,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();
        let shift = (linkedit.3 - ctx.linkedit_fileoff) as u32;

        let new_symtab = read_symtab(&result.data, ctx.header_size).unwrap();
        let new_fs = read_function_starts_offset(&result.data, ctx.header_size).unwrap();
        let new_indirect = read_dysymtab_indirectsymoff(&result.data, ctx.header_size).unwrap();

        assert_eq!(new_symtab.0, orig_symtab.0 + shift, "ARM64: symoff shift");
        assert_eq!(new_symtab.1, orig_symtab.1 + shift, "ARM64: stroff shift");
        assert_eq!(new_fs, orig_fs + shift, "ARM64: function_starts shift");
        assert_eq!(
            new_indirect,
            orig_indirect + shift,
            "ARM64: indirectsymoff shift"
        );
    }

    // --- No-coverage mode (hooks-only) ---

    #[test]
    fn test_patch_macho_no_coverage_linkedit_layout() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_0007);

        let result = patch_macho(
            &ctx,
            &[],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            true, // no_coverage
        )
        .expect("no-coverage patch should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs
            .iter()
            .find(|s| s.0 == "__TR_COV")
            .expect("__TR_COV missing");
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT missing");

        assert!(
            tr_cov.3 < linkedit.3,
            "no-coverage: TR_COV fileoff < LINKEDIT fileoff"
        );
        assert!(
            tr_cov.1 < linkedit.1,
            "no-coverage: TR_COV vmaddr < LINKEDIT vmaddr"
        );
        assert_eq!(
            linkedit.3 + linkedit.4,
            result.data.len() as u64,
            "no-coverage: LINKEDIT extends to EOF"
        );
    }

    // ====================================================================
    // Full LC coverage: DATA_IN_CODE, EXPORTS_TRIE, SEGMENT_SPLIT_INFO,
    // DYLD_CHAINED_FIXUPS, page alignment, codesign structural checks.
    // ====================================================================

    /// Generic LC reader: find a linkedit_data_command by cmd type, return (dataoff, datasize).
    fn read_linkedit_data_lc(
        data: &[u8],
        header_size: usize,
        target_cmd: u32,
    ) -> Option<(u32, u32)> {
        let ncmds = get_u32(data, 16) as usize;
        let mut off = header_size;
        for _ in 0..ncmds {
            if off + 8 > data.len() {
                break;
            }
            let cmd = get_u32(data, off);
            let cmdsize = get_u32(data, off + 4);
            if cmd == 0 || cmdsize < 8 {
                break;
            }
            if cmd == target_cmd && off + 16 <= data.len() {
                return Some((get_u32(data, off + 8), get_u32(data, off + 12)));
            }
            off += cmdsize as usize;
        }
        None
    }

    /// Build a synthetic Mach-O with ALL linkedit_data_command LC types
    /// plus an optional chained fixups blob. Returns (data, ctx).
    ///
    /// Includes: LC_SYMTAB, LC_DYSYMTAB, LC_FUNCTION_STARTS, LC_DATA_IN_CODE,
    /// LC_DYLD_EXPORTS_TRIE, LC_SEGMENT_SPLIT_INFO, and optionally
    /// LC_DYLD_CHAINED_FIXUPS with a minimal valid fixups blob.
    fn build_macho_all_lcs(cputype: u32, with_chained_fixups: bool) -> (Vec<u8>, MachOContext) {
        let page_sz: usize = if cputype == 0x0100_000C {
            0x4000
        } else {
            0x1000
        };
        let text_seg_vmaddr: u64 = 0x10000_0000;
        let text_seg_fileoff: u64 = 0;
        let text_seg_size: u64 = page_sz as u64;
        let text_section_va = text_seg_vmaddr + 0x800;
        let text_section_fileoff: u64 = 0x800;
        let text_section_size: u64 = 0x100;

        let linkedit_fileoff = page_sz as u64 * 2;
        let linkedit_data_len: u64 = 512; // larger to hold fixups blob
        let linkedit_vmaddr = text_seg_vmaddr + linkedit_fileoff;
        let linkedit_vmsize: u64 = align_up(linkedit_data_len, page_sz as u64);

        let total_size = linkedit_fileoff as usize + linkedit_data_len as usize;
        let mut data = vec![0u8; total_size];

        let header_size = 32usize;
        let mut lc_off = header_size;
        let mut ncmds = 0u32;

        // LC_SEGMENT_64 __TEXT (152 bytes)
        let text_lc_size = 152u32;
        put_u32(&mut data, lc_off, LC_SEGMENT_64);
        put_u32(&mut data, lc_off + 4, text_lc_size);
        data[lc_off + 8..lc_off + 14].copy_from_slice(b"__TEXT");
        put_u64(&mut data, lc_off + 24, text_seg_vmaddr);
        put_u64(&mut data, lc_off + 32, text_seg_size);
        put_u64(&mut data, lc_off + 40, text_seg_fileoff);
        put_u64(&mut data, lc_off + 48, text_seg_size);
        put_u32(&mut data, lc_off + 56, 7);
        put_u32(&mut data, lc_off + 60, 5);
        put_u32(&mut data, lc_off + 64, 1);
        let sect_off = lc_off + 72;
        data[sect_off..sect_off + 6].copy_from_slice(b"__text");
        data[sect_off + 16..sect_off + 22].copy_from_slice(b"__TEXT");
        put_u64(&mut data, sect_off + 32, text_section_va);
        put_u64(&mut data, sect_off + 40, text_section_size);
        put_u32(&mut data, sect_off + 48, text_section_fileoff as u32);
        lc_off += text_lc_size as usize;
        ncmds += 1;

        // LC_MAIN (24 bytes)
        let lc_main_off = lc_off;
        put_u32(&mut data, lc_off, 0x80000028);
        put_u32(&mut data, lc_off + 4, 24);
        put_u64(&mut data, lc_off + 8, 0x800);
        lc_off += 24;
        ncmds += 1;

        // LC_SYMTAB (24 bytes) — offsets into LINKEDIT
        let symoff = linkedit_fileoff as u32;
        let stroff = linkedit_fileoff as u32 + 64;
        put_u32(&mut data, lc_off, LC_SYMTAB);
        put_u32(&mut data, lc_off + 4, 24);
        put_u32(&mut data, lc_off + 8, symoff);
        put_u32(&mut data, lc_off + 12, 2);
        put_u32(&mut data, lc_off + 16, stroff);
        put_u32(&mut data, lc_off + 20, 32);
        lc_off += 24;
        ncmds += 1;

        // LC_DYSYMTAB (80 bytes)
        let indirectsymoff = linkedit_fileoff as u32 + 96;
        put_u32(&mut data, lc_off, LC_DYSYMTAB);
        put_u32(&mut data, lc_off + 4, 80);
        put_u32(&mut data, lc_off + 56, indirectsymoff);
        put_u32(&mut data, lc_off + 60, 4);
        lc_off += 80;
        ncmds += 1;

        // LC_FUNCTION_STARTS (16 bytes)
        let func_starts_dataoff = linkedit_fileoff as u32 + 128;
        put_u32(&mut data, lc_off, LC_FUNCTION_STARTS);
        put_u32(&mut data, lc_off + 4, 16);
        put_u32(&mut data, lc_off + 8, func_starts_dataoff);
        put_u32(&mut data, lc_off + 12, 8);
        lc_off += 16;
        ncmds += 1;

        // LC_DATA_IN_CODE (16 bytes)
        let data_in_code_dataoff = linkedit_fileoff as u32 + 160;
        put_u32(&mut data, lc_off, LC_DATA_IN_CODE);
        put_u32(&mut data, lc_off + 4, 16);
        put_u32(&mut data, lc_off + 8, data_in_code_dataoff);
        put_u32(&mut data, lc_off + 12, 8);
        lc_off += 16;
        ncmds += 1;

        // LC_DYLD_EXPORTS_TRIE (16 bytes)
        let exports_trie_dataoff = linkedit_fileoff as u32 + 192;
        put_u32(&mut data, lc_off, LC_DYLD_EXPORTS_TRIE);
        put_u32(&mut data, lc_off + 4, 16);
        put_u32(&mut data, lc_off + 8, exports_trie_dataoff);
        put_u32(&mut data, lc_off + 12, 16);
        lc_off += 16;
        ncmds += 1;

        // LC_SEGMENT_SPLIT_INFO (16 bytes)
        let split_info_dataoff = linkedit_fileoff as u32 + 224;
        put_u32(&mut data, lc_off, LC_SEGMENT_SPLIT_INFO);
        put_u32(&mut data, lc_off + 4, 16);
        put_u32(&mut data, lc_off + 8, split_info_dataoff);
        put_u32(&mut data, lc_off + 12, 8);
        lc_off += 16;
        ncmds += 1;

        // Optionally: LC_DYLD_CHAINED_FIXUPS (16 bytes) + fixups blob in LINKEDIT
        let chained_fixups_lc_offset;
        let chained_fixups_dataoff;
        let chained_fixups_datasize;
        if with_chained_fixups {
            chained_fixups_lc_offset = Some(lc_off as u64);
            chained_fixups_dataoff = linkedit_fileoff as u32 + 256;
            // Build minimal valid fixups blob at that offset.
            // Header (28) + padding (4) + starts_in_image (12) + import (4) + symbol (6) = 54
            let blob_off = chained_fixups_dataoff as usize;
            let starts_offset = 32u32;
            let imports_offset = 44u32;
            let symbols_offset = 48u32;
            let imports_count = 1u32;
            put_u32(&mut data, blob_off, 0); // fixups_version
            put_u32(&mut data, blob_off + 4, starts_offset);
            put_u32(&mut data, blob_off + 8, imports_offset);
            put_u32(&mut data, blob_off + 12, symbols_offset);
            put_u32(&mut data, blob_off + 16, imports_count);
            put_u32(&mut data, blob_off + 20, 1); // imports_format = DYLD_CHAINED_IMPORT
            put_u32(&mut data, blob_off + 24, 0); // symbols_format
            // starts_in_image at blob_off+32
            put_u32(&mut data, blob_off + 32, 2); // seg_count = 2
            put_u32(&mut data, blob_off + 36, 0); // seg_info_offset[0] = 0 (no fixups)
            put_u32(&mut data, blob_off + 40, 0); // seg_info_offset[1] = 0
            // import at blob_off+44
            put_u32(&mut data, blob_off + 44, 1); // lib_ordinal=1
            // symbol at blob_off+48
            data[blob_off + 48..blob_off + 53].copy_from_slice(b"test\0");
            chained_fixups_datasize = 54;

            put_u32(&mut data, lc_off, LC_DYLD_CHAINED_FIXUPS);
            put_u32(&mut data, lc_off + 4, 16);
            put_u32(&mut data, lc_off + 8, chained_fixups_dataoff);
            put_u32(&mut data, lc_off + 12, chained_fixups_datasize);
            lc_off += 16;
            ncmds += 1;
        } else {
            chained_fixups_lc_offset = None;
            chained_fixups_dataoff = 0;
            chained_fixups_datasize = 0;
        }

        // LC_SEGMENT_64 __LINKEDIT (72 bytes) — MUST be last
        let linkedit_seg_lc_off = lc_off;
        put_u32(&mut data, lc_off, LC_SEGMENT_64);
        put_u32(&mut data, lc_off + 4, 72);
        data[lc_off + 8..lc_off + 18].copy_from_slice(b"__LINKEDIT");
        put_u64(&mut data, lc_off + 24, linkedit_vmaddr);
        put_u64(&mut data, lc_off + 32, linkedit_vmsize);
        put_u64(&mut data, lc_off + 40, linkedit_fileoff);
        put_u64(&mut data, lc_off + 48, linkedit_data_len);
        put_u32(&mut data, lc_off + 56, 1);
        put_u32(&mut data, lc_off + 60, 1);
        lc_off += 72;
        ncmds += 1;

        let sizeofcmds = (lc_off - header_size) as u32;

        // Mach-O header
        put_u32(&mut data, 0, 0xFEEDFACF);
        put_u32(&mut data, 4, cputype);
        put_u32(&mut data, 8, 3);
        put_u32(&mut data, 12, 2);
        put_u32(&mut data, 16, ncmds);
        put_u32(&mut data, 20, sizeofcmds);

        // Fill __text
        for i in 0..0xFF {
            data[text_section_fileoff as usize + i] = 0x90;
        }
        data[text_section_fileoff as usize + 0xFF] = 0xC3;
        // Fill LINKEDIT with pattern, skipping the fixups blob region.
        let fixups_rel_start = if with_chained_fixups {
            256usize
        } else {
            usize::MAX
        };
        let fixups_rel_end = if with_chained_fixups { 256 + 54 } else { 0 };
        for i in 0..linkedit_data_len as usize {
            if i >= fixups_rel_start && i < fixups_rel_end {
                continue;
            }
            data[linkedit_fileoff as usize + i] = 0xBB;
        }

        let ctx = MachOContext {
            text: crate::macho::TextSection {
                va: text_section_va,
                offset: text_section_fileoff,
                size: text_section_size,
            },
            entry_point: text_section_va,
            func_symbols: vec![text_section_va],
            highest_va_end: linkedit_vmaddr + linkedit_vmsize,
            is_dylib: false,
            is_64bit: true,
            segments: vec![
                MachOSegment {
                    name: "__TEXT".into(),
                    vmaddr: text_seg_vmaddr,
                    vmsize: text_seg_size,
                    fileoff: text_seg_fileoff,
                    filesize: text_seg_size,
                },
                MachOSegment {
                    name: "__LINKEDIT".into(),
                    vmaddr: linkedit_vmaddr,
                    vmsize: linkedit_vmsize,
                    fileoff: linkedit_fileoff,
                    filesize: linkedit_data_len,
                },
            ],
            stubs_ranges: vec![],
            header_size,
            lc_end_offset: lc_off as u64,
            first_section_offset: text_section_fileoff,
            available_lc_space: text_section_fileoff - lc_off as u64,
            lc_main_entryoff_offset: Some(lc_main_off as u64 + 8),
            lc_unixthread_pc_offset: None,
            text_segment_vmaddr: text_seg_vmaddr,
            mod_init_func_section: None,
            init_offsets_section: None,
            mod_init_func_pointers: vec![],
            code_signature_lc_offset: None,
            code_signature_lc_size: 0,
            linkedit_segment_offset: Some(linkedit_seg_lc_off as u64),
            linkedit_vmaddr,
            linkedit_fileoff,
            linkedit_filesize: linkedit_data_len,
            ncmds,
            sizeofcmds,
            ncmds_offset: 16,
            sizeofcmds_offset: 20,
            chained_fixups_lc_offset,
            chained_fixups_dataoff,
            chained_fixups_datasize,
            cputype,
            libsystem_ordinal: 1,
        };

        (data, ctx)
    }

    // --- LC_DATA_IN_CODE ---

    #[test]
    fn test_update_linkedit_lc_offsets_data_in_code() {
        let (mut data, ctx) = build_macho_all_lcs(0x0100_0007, false);
        let orig = read_linkedit_data_lc(&data, ctx.header_size, LC_DATA_IN_CODE).unwrap();
        let shift = 0x6000u64;

        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            ctx.linkedit_vmaddr + shift,
            align_up(ctx.linkedit_filesize, 0x1000),
            ctx.linkedit_fileoff + shift,
            ctx.linkedit_filesize,
            0,
            0,
        );

        let new = read_linkedit_data_lc(&data, ctx.header_size, LC_DATA_IN_CODE).unwrap();
        assert_eq!(
            new.0,
            orig.0 + shift as u32,
            "LC_DATA_IN_CODE dataoff shifted"
        );
        assert_eq!(new.1, orig.1, "LC_DATA_IN_CODE datasize unchanged");
    }

    // --- LC_DYLD_EXPORTS_TRIE ---

    #[test]
    fn test_update_linkedit_lc_offsets_exports_trie() {
        let (mut data, ctx) = build_macho_all_lcs(0x0100_0007, false);
        let orig = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_EXPORTS_TRIE).unwrap();
        let shift = 0x5000u64;

        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            ctx.linkedit_vmaddr + shift,
            align_up(ctx.linkedit_filesize, 0x1000),
            ctx.linkedit_fileoff + shift,
            ctx.linkedit_filesize,
            0,
            0,
        );

        let new = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_EXPORTS_TRIE).unwrap();
        assert_eq!(
            new.0,
            orig.0 + shift as u32,
            "LC_DYLD_EXPORTS_TRIE dataoff shifted"
        );
        assert_eq!(new.1, orig.1, "LC_DYLD_EXPORTS_TRIE datasize unchanged");
    }

    // --- LC_SEGMENT_SPLIT_INFO ---

    #[test]
    fn test_update_linkedit_lc_offsets_segment_split_info() {
        let (mut data, ctx) = build_macho_all_lcs(0x0100_0007, false);
        let orig = read_linkedit_data_lc(&data, ctx.header_size, LC_SEGMENT_SPLIT_INFO).unwrap();
        let shift = 0x7000u64;

        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            ctx.linkedit_vmaddr + shift,
            align_up(ctx.linkedit_filesize, 0x1000),
            ctx.linkedit_fileoff + shift,
            ctx.linkedit_filesize,
            0,
            0,
        );

        let new = read_linkedit_data_lc(&data, ctx.header_size, LC_SEGMENT_SPLIT_INFO).unwrap();
        assert_eq!(
            new.0,
            orig.0 + shift as u32,
            "LC_SEGMENT_SPLIT_INFO dataoff shifted"
        );
        assert_eq!(new.1, orig.1, "LC_SEGMENT_SPLIT_INFO datasize unchanged");
    }

    // --- LC_DYLD_CHAINED_FIXUPS: new blob path ---

    #[test]
    fn test_update_linkedit_lc_offsets_chained_fixups_new_blob() {
        let (mut data, ctx) = build_macho_all_lcs(0x0100_0007, true);
        let shift = 0x4000u64;
        let new_blob_off = 0xDEADu32;
        let new_blob_size = 128u32;

        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            ctx.linkedit_vmaddr + shift,
            align_up(ctx.linkedit_filesize, 0x1000),
            ctx.linkedit_fileoff + shift,
            ctx.linkedit_filesize,
            new_blob_off,
            new_blob_size,
        );

        let fixups = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_CHAINED_FIXUPS).unwrap();
        assert_eq!(
            fixups.0, new_blob_off,
            "LC_DYLD_CHAINED_FIXUPS should point to new blob offset"
        );
        assert_eq!(
            fixups.1, new_blob_size,
            "LC_DYLD_CHAINED_FIXUPS should have new blob size"
        );
    }

    // --- LC_DYLD_CHAINED_FIXUPS: shift-only path (new_fixups_fileoff == 0) ---

    #[test]
    fn test_update_linkedit_lc_offsets_chained_fixups_shift_fallback() {
        let (mut data, ctx) = build_macho_all_lcs(0x0100_0007, true);
        let orig = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_CHAINED_FIXUPS).unwrap();
        let shift = 0x3000u64;

        // new_fixups_fileoff=0 triggers the shift-only fallback path.
        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            ctx.linkedit_vmaddr + shift,
            align_up(ctx.linkedit_filesize, 0x1000),
            ctx.linkedit_fileoff + shift,
            ctx.linkedit_filesize,
            0,
            0,
        );

        let fixups = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_CHAINED_FIXUPS).unwrap();
        assert_eq!(
            fixups.0,
            orig.0 + shift as u32,
            "chained fixups dataoff should be shifted when no new blob"
        );
        assert_eq!(
            fixups.1, orig.1,
            "chained fixups datasize unchanged in shift fallback"
        );
    }

    // --- Integration: all LC types through patch_macho ---

    #[test]
    fn test_patch_macho_all_lc_types_shifted_x86_64() {
        let (data, ctx) = build_macho_all_lcs(0x0100_0007, false);
        let orig_symtab = read_symtab(&data, ctx.header_size).unwrap();
        let orig_fs = read_linkedit_data_lc(&data, ctx.header_size, LC_FUNCTION_STARTS).unwrap();
        let orig_dic = read_linkedit_data_lc(&data, ctx.header_size, LC_DATA_IN_CODE).unwrap();
        let orig_et = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_EXPORTS_TRIE).unwrap();
        let orig_ssi =
            read_linkedit_data_lc(&data, ctx.header_size, LC_SEGMENT_SPLIT_INFO).unwrap();

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();
        let shift = (linkedit.3 - ctx.linkedit_fileoff) as u32;
        let file_len = result.data.len() as u32;

        // All offsets shifted by the same amount.
        let new_symtab = read_symtab(&result.data, ctx.header_size).unwrap();
        assert_eq!(new_symtab.0, orig_symtab.0 + shift, "symoff");
        assert_eq!(new_symtab.1, orig_symtab.1 + shift, "stroff");

        let new_fs =
            read_linkedit_data_lc(&result.data, ctx.header_size, LC_FUNCTION_STARTS).unwrap();
        assert_eq!(new_fs.0, orig_fs.0 + shift, "function_starts");

        let new_dic =
            read_linkedit_data_lc(&result.data, ctx.header_size, LC_DATA_IN_CODE).unwrap();
        assert_eq!(new_dic.0, orig_dic.0 + shift, "data_in_code");

        let new_et =
            read_linkedit_data_lc(&result.data, ctx.header_size, LC_DYLD_EXPORTS_TRIE).unwrap();
        assert_eq!(new_et.0, orig_et.0 + shift, "exports_trie");

        let new_ssi =
            read_linkedit_data_lc(&result.data, ctx.header_size, LC_SEGMENT_SPLIT_INFO).unwrap();
        assert_eq!(new_ssi.0, orig_ssi.0 + shift, "segment_split_info");

        // All offsets within bounds.
        for (name, off) in [
            ("symoff", new_symtab.0),
            ("stroff", new_symtab.1),
            ("func_starts", new_fs.0),
            ("data_in_code", new_dic.0),
            ("exports_trie", new_et.0),
            ("segment_split_info", new_ssi.0),
        ] {
            assert!(
                off < file_len,
                "{name} offset 0x{off:x} >= file len 0x{file_len:x}"
            );
        }
    }

    #[test]
    fn test_patch_macho_all_lc_types_shifted_arm64() {
        let (data, ctx) = build_macho_all_lcs(0x0100_000C, false);
        let orig_dic = read_linkedit_data_lc(&data, ctx.header_size, LC_DATA_IN_CODE).unwrap();
        let orig_et = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_EXPORTS_TRIE).unwrap();
        let orig_ssi =
            read_linkedit_data_lc(&data, ctx.header_size, LC_SEGMENT_SPLIT_INFO).unwrap();

        let result = patch_macho(
            &ctx,
            &[],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            true,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();
        let shift = (linkedit.3 - ctx.linkedit_fileoff) as u32;

        let new_dic =
            read_linkedit_data_lc(&result.data, ctx.header_size, LC_DATA_IN_CODE).unwrap();
        assert_eq!(new_dic.0, orig_dic.0 + shift, "ARM64: data_in_code");

        let new_et =
            read_linkedit_data_lc(&result.data, ctx.header_size, LC_DYLD_EXPORTS_TRIE).unwrap();
        assert_eq!(new_et.0, orig_et.0 + shift, "ARM64: exports_trie");

        let new_ssi =
            read_linkedit_data_lc(&result.data, ctx.header_size, LC_SEGMENT_SPLIT_INFO).unwrap();
        assert_eq!(new_ssi.0, orig_ssi.0 + shift, "ARM64: segment_split_info");
    }

    // --- Chained fixups integration through patch_macho ---

    #[test]
    fn test_patch_macho_chained_fixups_new_blob_at_eof() {
        let (data, ctx) = build_macho_all_lcs(0x0100_0007, true);

        let result = patch_macho(
            &ctx,
            &[],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            true, // no_coverage
        )
        .expect("patch_macho with chained fixups should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();
        let fixups =
            read_linkedit_data_lc(&result.data, ctx.header_size, LC_DYLD_CHAINED_FIXUPS).unwrap();

        // New fixups blob must be within LINKEDIT range.
        let linkedit_end = linkedit.3 + linkedit.4;
        assert!(
            fixups.0 as u64 >= linkedit.3,
            "fixups blob (0x{:x}) must start within LINKEDIT (0x{:x})",
            fixups.0,
            linkedit.3
        );
        assert!(
            (fixups.0 + fixups.1) as u64 <= linkedit_end,
            "fixups blob end (0x{:x}) must be within LINKEDIT end (0x{:x})",
            fixups.0 + fixups.1,
            linkedit_end
        );

        // Fixups blob must be different from original offset (it was rebuilt and appended).
        assert_ne!(
            fixups.0, ctx.chained_fixups_dataoff,
            "fixups dataoff should differ from original (data was relocated)"
        );
    }

    // --- x86_64 page alignment ---

    #[test]
    fn test_patch_macho_x86_64_page_alignment() {
        let (data, ctx) = build_macho_with_linkedit(0x0100_0007);

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs.iter().find(|s| s.0 == "__TR_COV").unwrap();
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();

        // x86_64 uses 4KB pages.
        assert_eq!(
            tr_cov.1 % 0x1000,
            0,
            "x86_64: TR_COV vmaddr must be 4KB-aligned"
        );
        assert_eq!(
            tr_cov.3 % 0x1000,
            0,
            "x86_64: TR_COV fileoff must be 4KB-aligned"
        );
        assert_eq!(
            linkedit.1 % 0x1000,
            0,
            "x86_64: LINKEDIT vmaddr must be 4KB-aligned"
        );
        assert_eq!(
            linkedit.3 % 0x1000,
            0,
            "x86_64: LINKEDIT fileoff must be 4KB-aligned"
        );
        assert_eq!(
            tr_cov.2 % 0x1000,
            0,
            "x86_64: TR_COV vmsize must be 4KB-aligned"
        );
        assert_eq!(
            linkedit.2 % 0x1000,
            0,
            "x86_64: LINKEDIT vmsize must be 4KB-aligned"
        );
    }

    // --- Structural codesign prerequisites ---

    #[test]
    fn test_patch_macho_linkedit_is_last_segment_x86_64() {
        let (data, ctx) = build_macho_all_lcs(0x0100_0007, false);

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();

        // LINKEDIT must have the highest vmaddr of all segments.
        for seg in &segs {
            if seg.0 != "__LINKEDIT" {
                assert!(
                    linkedit.1 >= seg.1 + seg.2,
                    "LINKEDIT vmaddr (0x{:x}) must be >= {} vmend (0x{:x})",
                    linkedit.1,
                    seg.0,
                    seg.1 + seg.2
                );
            }
        }

        // LINKEDIT must have the highest file offset.
        for seg in &segs {
            if seg.0 != "__LINKEDIT" {
                assert!(
                    linkedit.3 >= seg.3 + seg.4,
                    "LINKEDIT fileoff (0x{:x}) must be >= {} file end (0x{:x})",
                    linkedit.3,
                    seg.0,
                    seg.3 + seg.4
                );
            }
        }

        // LINKEDIT extends to EOF.
        assert_eq!(
            linkedit.3 + linkedit.4,
            result.data.len() as u64,
            "codesign: LINKEDIT must extend to EOF"
        );
    }

    #[test]
    fn test_patch_macho_linkedit_is_last_segment_arm64() {
        let (data, ctx) = build_macho_all_lcs(0x0100_000C, false);

        let result = patch_macho(
            &ctx,
            &[],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            true,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();

        for seg in &segs {
            if seg.0 != "__LINKEDIT" {
                assert!(
                    linkedit.1 >= seg.1 + seg.2,
                    "ARM64: LINKEDIT vmaddr (0x{:x}) must be >= {} vmend (0x{:x})",
                    linkedit.1,
                    seg.0,
                    seg.1 + seg.2
                );
                assert!(
                    linkedit.3 >= seg.3 + seg.4,
                    "ARM64: LINKEDIT fileoff (0x{:x}) must be >= {} file end (0x{:x})",
                    linkedit.3,
                    seg.0,
                    seg.3 + seg.4
                );
            }
        }

        assert_eq!(
            linkedit.3 + linkedit.4,
            result.data.len() as u64,
            "ARM64 codesign: LINKEDIT must extend to EOF"
        );
    }

    // --- Segment contiguity in VM space ---

    #[test]
    fn test_patch_macho_no_vm_gaps_x86_64() {
        let (data, ctx) = build_macho_all_lcs(0x0100_0007, false);

        let block = BasicBlock {
            va: ctx.text.va,
            file_offset: ctx.text.offset as u64,
            displaced_bytes: vec![0x90; 16],
            displaced_len: 5,
            block_id: 0x1234,
        };

        let result = patch_macho(
            &ctx,
            &[block],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            false,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs.iter().find(|s| s.0 == "__TR_COV").unwrap();
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();

        // TR_COV vmaddr + vmsize == LINKEDIT vmaddr (contiguous, no gap).
        assert_eq!(
            tr_cov.1 + tr_cov.2,
            linkedit.1,
            "TR_COV and LINKEDIT must be contiguous in VM space"
        );
    }

    #[test]
    fn test_patch_macho_no_vm_gaps_arm64() {
        let (data, ctx) = build_macho_all_lcs(0x0100_000C, false);

        let result = patch_macho(
            &ctx,
            &[],
            data,
            false,
            false,
            &MockTrampolineGen,
            &[],
            None,
            0,
            false,
            true,
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs.iter().find(|s| s.0 == "__TR_COV").unwrap();
        let linkedit = segs.iter().find(|s| s.0 == "__LINKEDIT").unwrap();

        assert_eq!(
            tr_cov.1 + tr_cov.2,
            linkedit.1,
            "ARM64: TR_COV and LINKEDIT must be contiguous in VM space"
        );
    }
}
