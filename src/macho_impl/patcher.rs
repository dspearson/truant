use anyhow::{Context, Result, bail};

use crate::disasm::BasicBlock;
use crate::hook_trampoline::TargetAbi;
use crate::hooks::ResolvedHook;
use crate::macho::MachOContext;
use crate::macho_trampoline;
use crate::patcher::{InstrumentationOptions, PatchResult};
use crate::traits::{BinaryContext, Patcher, TrampolineGenerator};
use crate::trampoline::{DATA_SIZE, PERSISTENT_DATA_SIZE};

/// Return type for `generate_init_and_trampolines`.
type InitAndTrampolines<'a> = (
    crate::trampoline::InitCode,
    Vec<(&'a BasicBlock, crate::trampoline::Trampoline)>,
    usize,
    u64,
);

/// Return type for `setup_hook_infrastructure`.
type HookInfra = (
    u64,
    u64,
    u64,
    u64,
    Vec<(usize, crate::trampoline::Trampoline)>,
    Vec<(u64, Vec<u8>)>,
    Vec<crate::trampoline::Trampoline>,
);

/// Return type for `generate_persistent_wrapper_helper`.
type PersistentResult = (
    u64,
    Option<(u64, crate::trampoline::PersistentWrapper, usize)>,
    u64,
);

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
        opts: &InstrumentationOptions,
        hooks: &[ResolvedHook],
    ) -> Result<PatchResult> {
        let macho_ctx = ctx
            .as_any()
            .downcast_ref::<crate::macho_impl::context::MachOBinaryContext>()
            .ok_or_else(|| anyhow::anyhow!("MachOPatcher requires MachOBinaryContext"))?;

        patch_macho(
            macho_ctx.inner(),
            blocks,
            data,
            opts,
            &*self.tramp_gen,
            hooks,
        )
    }
}

/// Layout parameters computed for the new Mach-O segments.
struct MachoLayout {
    is_arm64: bool,
    use_got_shmat: bool,
    arm64_split: bool,
    segment_va: u64,
    data_va: u64,
    init_va: u64,
    hook_abi: TargetAbi,
    shmat_ordinal: u32,
    page_sz: u64,
    data_area_size: u64,
}

/// Computed segment sizes for the data and code segments.
struct SegmentSizes {
    segment_size: u64,
    segment_vmsize: u64,
    code_segment_size: u64,
    code_segment_vmsize: u64,
    needs_lc_routines_64: bool,
}

/// File layout offsets after restructuring.
struct FileLayout {
    file_offset_of_segment: u64,
    file_offset_of_code_seg: u64,
    new_linkedit_fileoff: u64,
    new_fixups_fileoff: u32,
    new_fixups_size: u32,
    new_linkedit_vmaddr: u64,
    new_linkedit_vmsize: u64,
    new_linkedit_filesize: u64,
}

/// Non-essential load command types that can be stripped to reclaim header space.
/// These are informational or debugging LCs that dyld does not require for loading.
const LC_UUID: u32 = 0x1B;
const LC_BUILD_VERSION: u32 = 0x32;
const LC_SOURCE_VERSION: u32 = 0x2A;
const LC_LOAD_DYLINKER: u32 = 0x0E;
const LC_MAIN: u32 = 0x80000028;
const LC_UNIXTHREAD: u32 = 0x05;

/// LCs that are safe to strip — ordered from least to most useful so we strip
/// the most expendable ones first.
const STRIPPABLE_LCS: &[u32] = &[
    LC_UUID,            // 24 bytes — binary identity, not needed for execution
    LC_BUILD_VERSION,   // 32 bytes — build metadata
    LC_SOURCE_VERSION,  // 16 bytes — source version info
    LC_DATA_IN_CODE,    // 16 bytes — marks non-code regions, rarely populated
    LC_FUNCTION_STARTS, // 16 bytes — function boundary hints for stack traces
    LC_LOAD_DYLINKER,   // ~32 bytes — dyld path, kernel supplies it anyway
];

/// Reclaim load command space by stripping non-essential LCs.
///
/// Always strips LC_CODE_SIGNATURE (it will be re-signed after patching).
/// If that's not enough for `min_space`, progressively strips informational LCs
/// (UUID, build version, source version, data-in-code, function starts, dylinker)
/// and compacts the remaining LC area.
///
/// After compaction, re-scans the LC area to update cached offsets in `ctx`
/// (linkedit_segment_offset, lc_main_entryoff_offset, etc.).
///
/// Returns (ncmds, sizeofcmds, available_lc_space) after stripping.
/// Mutates `ctx` in place to update cached LC offsets after compaction.
fn reclaim_lc_space(data: &mut [u8], ctx: &mut MachOContext, min_space: u64) -> (u32, u32, u64) {
    let mut ncmds = ctx.ncmds;
    let mut sizeofcmds = ctx.sizeofcmds;
    let mut available_lc_space = ctx.available_lc_space;

    // Phase 1: Strip LC_CODE_SIGNATURE if present.
    if let Some(codesig_offset) = ctx.code_signature_lc_offset {
        let off = codesig_offset as usize;
        if off + ctx.code_signature_lc_size as usize <= data.len() {
            for b in &mut data[off..off + ctx.code_signature_lc_size as usize] {
                *b = 0;
            }
            ncmds -= 1;
            sizeofcmds -= ctx.code_signature_lc_size;
            available_lc_space += ctx.code_signature_lc_size as u64;

            tracing::info!(
                "stripped LC_CODE_SIGNATURE ({} bytes) — re-sign with: codesign -f -s - <output>",
                ctx.code_signature_lc_size,
            );
        }
    }

    // Phase 2: If still insufficient, strip non-essential LCs and compact.
    if available_lc_space < min_space {
        let lc_area_start = ctx.header_size;
        let lc_area_end = ctx.lc_end_offset as usize;

        // Collect (offset, size, cmd_type) for all LCs in the area.
        let mut lcs: Vec<(usize, usize, u32)> = Vec::new();
        let mut pos = lc_area_start;
        while pos + 8 <= lc_area_end {
            let cmd = u32::from_le_bytes(data[pos..pos + 4].try_into().expect("slice is 4 bytes"));
            let cmdsize =
                u32::from_le_bytes(data[pos + 4..pos + 8].try_into().expect("slice is 4 bytes"))
                    as usize;
            if cmdsize == 0 || pos + cmdsize > lc_area_end {
                break;
            }
            lcs.push((pos, cmdsize, cmd));
            pos += cmdsize;
        }

        // Mark non-essential LCs for removal until we have enough space.
        let mut to_strip: Vec<bool> = vec![false; lcs.len()];
        for &target_cmd in STRIPPABLE_LCS {
            if available_lc_space >= min_space {
                break;
            }
            for (i, &(_, size, cmd)) in lcs.iter().enumerate() {
                if cmd == target_cmd && !to_strip[i] {
                    to_strip[i] = true;
                    ncmds -= 1;
                    sizeofcmds -= size as u32;
                    available_lc_space += size as u64;
                    tracing::info!(
                        "stripped LC 0x{:x} ({} bytes) to reclaim header space",
                        target_cmd,
                        size,
                    );
                }
            }
        }

        // Compact: copy surviving LCs into a contiguous block, then re-scan
        // to find updated offsets for LINKEDIT, LC_MAIN, etc.
        if to_strip.iter().any(|&s| s) {
            let mut write_pos = lc_area_start;
            for (i, &(read_pos, size, _)) in lcs.iter().enumerate() {
                if to_strip[i] {
                    continue;
                }
                if write_pos != read_pos {
                    data.copy_within(read_pos..read_pos + size, write_pos);
                }
                write_pos += size;
            }
            // Zero out the freed tail.
            for b in &mut data[write_pos..lc_area_end] {
                *b = 0;
            }

            // Re-scan compacted LCs to update cached offsets in ctx.
            ctx.linkedit_segment_offset = None;
            ctx.lc_main_entryoff_offset = None;
            ctx.lc_unixthread_pc_offset = None;
            ctx.chained_fixups_lc_offset = None;

            let mut offset = lc_area_start;
            for _ in 0..ncmds {
                if offset + 8 > data.len() {
                    break;
                }
                let cmd = u32::from_le_bytes(
                    data[offset..offset + 4]
                        .try_into()
                        .expect("slice is 4 bytes"),
                );
                let cmdsize = u32::from_le_bytes(
                    data[offset + 4..offset + 8]
                        .try_into()
                        .expect("slice is 4 bytes"),
                ) as usize;
                if cmd == 0 || cmdsize < 8 {
                    break;
                }
                match cmd {
                    LC_SEGMENT_64 => {
                        if offset + 24 <= data.len()
                            && data[offset + 8..offset + 24].starts_with(b"__LINKEDIT\0")
                        {
                            ctx.linkedit_segment_offset = Some(offset as u64);
                        }
                    }
                    LC_MAIN => {
                        if cmdsize >= 24 {
                            ctx.lc_main_entryoff_offset = Some(offset as u64 + 8);
                        }
                    }
                    LC_UNIXTHREAD => {
                        if ctx.cputype == 0x0100_0007 && cmdsize >= 152 {
                            ctx.lc_unixthread_pc_offset = Some(offset as u64 + 144);
                        } else if ctx.cputype == 0x0100_000C && cmdsize >= 280 {
                            ctx.lc_unixthread_pc_offset = Some(offset as u64 + 272);
                        }
                    }
                    LC_DYLD_CHAINED_FIXUPS => {
                        if cmdsize >= 16 {
                            ctx.chained_fixups_lc_offset = Some(offset as u64);
                        }
                    }
                    _ => {}
                }
                offset += cmdsize;
            }
        }
    }

    // Update header fields.
    data[ctx.ncmds_offset as usize..ctx.ncmds_offset as usize + 4]
        .copy_from_slice(&ncmds.to_le_bytes());
    data[ctx.sizeofcmds_offset as usize..ctx.sizeofcmds_offset as usize + 4]
        .copy_from_slice(&sizeofcmds.to_le_bytes());

    (ncmds, sizeofcmds, available_lc_space)
}

/// Compute the layout for new segments: architecture, VAs, GOT strategy, page size.
fn compute_layout(
    ctx: &MachOContext,
    data: &[u8],
    hooks: &[ResolvedHook],
    no_coverage: bool,
    page_sz: u64,
) -> MachoLayout {
    const CPU_TYPE_ARM64: u32 = 0x0100_000C;
    let use_got_shmat =
        ctx.chained_fixups_lc_offset.is_some() && ctx.cputype != CPU_TYPE_ARM64 && !ctx.is_dylib;
    let shmat_ordinal: u32 = if use_got_shmat {
        let fo = ctx.chained_fixups_dataoff as usize;
        u32::from_le_bytes(data[fo + 16..fo + 20].try_into().expect("slice is 4 bytes"))
    } else {
        0
    };
    let data_area_size: u64 = if use_got_shmat { 24 } else { DATA_SIZE };

    let is_arm64 = ctx.cputype == CPU_TYPE_ARM64;
    let hook_abi = if is_arm64 {
        TargetAbi::Aarch64
    } else {
        TargetAbi::SysV64
    };

    let segment_va = if ctx.linkedit_vmaddr > 0 {
        ctx.linkedit_vmaddr
    } else {
        align_up(ctx.highest_va_end, page_sz) + page_sz
    };

    let has_library_hooks =
        crate::hooks::count_library_hooks(hooks) > 0 || crate::hooks::count_return_hooks(hooks) > 0;
    let arm64_split = is_arm64 && (!no_coverage || has_library_hooks);
    let data_va = segment_va;
    let init_va = if arm64_split {
        segment_va + page_sz
    } else {
        segment_va + data_area_size
    };

    MachoLayout {
        is_arm64,
        use_got_shmat,
        arm64_split,
        segment_va,
        data_va,
        init_va,
        hook_abi,
        shmat_ordinal,
        page_sz,
        data_area_size,
    }
}

/// Generate init code and coverage trampolines.
/// Returns (init_code, trampolines, blocks_skipped, current_va).
fn generate_init_and_trampolines<'a>(
    ctx: &MachOContext,
    blocks: &'a [BasicBlock],
    layout: &MachoLayout,
    enable_forkserver: bool,
    tramp_gen: &dyn TrampolineGenerator,
    no_coverage: bool,
) -> Result<InitAndTrampolines<'a>> {
    if no_coverage {
        let init_code = crate::trampoline::InitCode {
            va: layout.init_va,
            code: Vec::new(),
            entry_va: layout.init_va,
        };
        return Ok((init_code, Vec::new(), 0, layout.init_va));
    }

    let init_code = if ctx.is_dylib {
        let chain_to = ctx.mod_init_func_pointers.first().copied();
        if layout.is_arm64 {
            macho_trampoline::generate_macho_dylib_init_aarch64(
                layout.init_va,
                layout.data_va,
                chain_to,
                layout.use_got_shmat,
            )
            .context("failed to generate macOS ARM64 dylib init code")?
        } else {
            macho_trampoline::generate_macho_dylib_init_x86_64(
                layout.init_va,
                layout.data_va,
                chain_to,
                layout.use_got_shmat,
            )
            .context("failed to generate macOS dylib init code")?
        }
    } else if ctx.lc_main_entryoff_offset.is_some() {
        if layout.is_arm64 {
            macho_trampoline::generate_macho_exec_init_aarch64(
                layout.init_va,
                layout.data_va,
                ctx.entry_point,
                enable_forkserver,
                layout.use_got_shmat,
            )
            .context("failed to generate macOS ARM64 exec init code")?
        } else {
            macho_trampoline::generate_macho_exec_init_x86_64(
                layout.init_va,
                layout.data_va,
                ctx.entry_point,
                enable_forkserver,
                layout.use_got_shmat,
            )
            .context("failed to generate macOS exec init code")?
        }
    } else if ctx.lc_unixthread_pc_offset.is_some() {
        if layout.is_arm64 {
            macho_trampoline::generate_macho_unixthread_init_aarch64(
                layout.init_va,
                layout.data_va,
                ctx.entry_point,
                enable_forkserver,
                layout.use_got_shmat,
            )
            .context("failed to generate macOS ARM64 LC_UNIXTHREAD init code")?
        } else {
            macho_trampoline::generate_macho_unixthread_init_x86_64(
                layout.init_va,
                layout.data_va,
                ctx.entry_point,
                enable_forkserver,
                layout.use_got_shmat,
            )
            .context("failed to generate macOS LC_UNIXTHREAD init code")?
        }
    } else {
        bail!("no LC_MAIN or LC_UNIXTHREAD found — cannot determine entry point");
    };

    let mut trampolines_start_va = layout.init_va + init_code.code.len() as u64;
    trampolines_start_va = align_up(trampolines_start_va, 16);

    let mut trampolines = Vec::with_capacity(blocks.len());
    let mut current_va = trampolines_start_va;
    let mut blocks_skipped = 0;

    for block in blocks {
        match tramp_gen.generate_trampoline(current_va, layout.data_va, block) {
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

    if trampolines.is_empty() && !blocks.is_empty() {
        bail!("all basic blocks failed trampoline generation");
    }

    Ok((init_code, trampolines, blocks_skipped, current_va))
}

/// Set up hook infrastructure: hook data slots, toggle bytes, return slots,
/// shellcode blobs, and hook trampolines.
/// Returns (hook_data_va, toggle_data_va, current_va, hook_trampolines,
///          shellcode_blobs, return_trampolines).
fn setup_hook_infrastructure(
    hooks: &[ResolvedHook],
    layout: &MachoLayout,
    mut current_va: u64,
) -> HookInfra {
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
    if layout.arm64_split {
        let mut data_cursor = layout.data_va + layout.data_area_size;
        data_cursor = align_up(data_cursor, 8);
        hook_data_va = data_cursor;
        data_cursor += hook_data_size;
        toggle_data_va = data_cursor;
        data_cursor += toggle_area_size;
        return_slot_va = data_cursor;
        data_cursor += return_slot_area_size;
        assert!(
            data_cursor <= layout.data_va + layout.page_sz,
            "hook metadata ({} bytes) exceeds data page size ({})",
            data_cursor - layout.data_va,
            layout.page_sz,
        );
    } else {
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

    // Group hooks by target_va for chaining.
    current_va = align_up(current_va, 16);
    let mut va_groups: std::collections::BTreeMap<u64, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, hook) in hooks.iter().enumerate() {
        va_groups.entry(hook.target_va).or_default().push(i);
    }

    let mut return_trampolines: Vec<crate::trampoline::Trampoline> = Vec::new();

    for indices in va_groups.values() {
        if indices.len() == 1 {
            let i = indices[0];
            let hook = &hooks[i];
            let sc_va = shellcode_va_map[i];
            let toggle_va = Some(toggle_data_va + hook.toggle_index as u64);

            if hook.mode == crate::hooks::HookMode::Return {
                let entry_va = current_va;
                let estimated_entry_size = 256u64;
                let ret_tramp_va = align_up(entry_va + estimated_entry_size, 8);
                let slot_va = return_slot_va
                    + hook
                        .return_slot_index
                        .expect("return hook must have return_slot_index")
                        as u64
                        * 8;

                match crate::hook_trampoline::generate_return_hook_trampolines(
                    &crate::hook_trampoline::ReturnHookContext {
                        entry_va,
                        ret_tramp_va,
                        hook_data_va,
                        shellcode_va: sc_va,
                        toggle_va,
                        return_slot_va: slot_va,
                    },
                    hook,
                    layout.hook_abi,
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
                    layout.hook_abi,
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
                layout.hook_abi,
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

    (
        hook_data_va,
        toggle_data_va,
        current_va,
        return_slot_va,
        hook_trampolines,
        shellcode_blobs,
        return_trampolines,
    )
}

/// Generate persistent mode wrapper and data area.
/// Returns (persistent_data_va, persistent_wrapper, current_va).
fn generate_persistent_wrapper(
    ctx: &MachOContext,
    data: &[u8],
    layout: &MachoLayout,
    mut current_va: u64,
    opts: &InstrumentationOptions,
    tramp_gen: &dyn TrampolineGenerator,
) -> Result<PersistentResult> {
    let persistent_addr = opts.persistent_addr;
    let persistent_count = opts.persistent_count;
    let enable_forkserver = opts.enable_forkserver;
    let defer = opts.defer;
    let persistent_data_va;
    let persistent_wrapper = if let Some(p_addr) = persistent_addr {
        if ctx.is_dylib {
            bail!("persistent mode is not supported for dylibs");
        }
        current_va = align_up(current_va, 16);
        persistent_data_va = current_va;
        current_va += PERSISTENT_DATA_SIZE;
        current_va = align_up(current_va, 16);

        let branch_sz = tramp_gen.branch_instruction_size();
        let (displaced_bytes, displaced_len) = if layout.is_arm64 {
            let foff = crate::macho::va_to_file_offset_macho(p_addr, ctx).ok_or_else(|| {
                anyhow::anyhow!("persistent addr 0x{:x} has no file mapping", p_addr,)
            })?;
            let n = branch_sz.div_ceil(4) * 4;
            if foff + n > data.len() {
                bail!("persistent addr 0x{:x} too close to end of file", p_addr);
            }
            (data[foff..foff + n].to_vec(), n)
        } else {
            crate::macho_disasm::extract_displaced_bytes_macho(data, ctx, p_addr, branch_sz)
                .with_context(|| {
                    format!(
                        "failed to extract displaced bytes at persistent addr 0x{:x}",
                        p_addr,
                    )
                })?
        };

        let include_forkserver = enable_forkserver && defer;
        let wrapper_va = current_va;
        let pw_params = crate::trampoline::PersistentWrapperParams {
            wrapper_va,
            persistent_data_va,
            data_va: layout.data_va,
            persistent_addr: p_addr,
            displaced_bytes: &displaced_bytes,
            displaced_len,
            persistent_count,
            include_forkserver,
        };
        let wrapper = if layout.is_arm64 {
            macho_trampoline::generate_macho_persistent_wrapper_aarch64(&pw_params)
                .context("failed to generate macOS ARM64 persistent wrapper")?
        } else {
            macho_trampoline::generate_macho_persistent_wrapper_x86_64(&pw_params)
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

    Ok((persistent_data_va, persistent_wrapper, current_va))
}

/// Generate heap sanitiser wrappers (x86_64 only).
/// Returns (heap_san_code, current_va).
fn generate_heap_san(
    ctx: &MachOContext,
    data: &[u8],
    layout: &MachoLayout,
    mut current_va: u64,
    enable_heap_san: bool,
) -> (
    Option<(crate::elf::AllocatorSymbols, crate::trampoline::HeapSanCode)>,
    u64,
) {
    let heap_san_code = if enable_heap_san && !layout.is_arm64 {
        let alloc_syms = crate::macho::find_allocator_symbols_macho(ctx, data);
        if alloc_syms.get("malloc").is_some() && alloc_syms.get("free").is_some() {
            current_va = align_up(current_va, 16);
            let hsw = crate::trampoline::generate_heap_san_wrappers(current_va);
            current_va += hsw.total_size;
            tracing::info!(
                "heap san (Mach-O): {} allocator functions found, wrappers at 0x{:x}",
                alloc_syms.count(),
                *hsw.wrappers
                    .values()
                    .min()
                    .expect("heap san wrappers are non-empty"),
            );
            Some((alloc_syms, hsw))
        } else {
            tracing::warn!(
                "heap san (Mach-O): malloc/free not found (found {} allocator symbols), skipping",
                alloc_syms.count(),
            );
            None
        }
    } else if enable_heap_san && layout.is_arm64 {
        tracing::warn!(
            "heap san: ARM64 macOS not yet supported (Linux x86_64 syscalls only), skipping"
        );
        None
    } else {
        None
    };

    (heap_san_code, current_va)
}

/// Compute segment sizes and determine if LC_ROUTINES_64 is needed.
fn compute_segment_sizes(
    ctx: &MachOContext,
    layout: &MachoLayout,
    segment_end_va: u64,
    no_coverage: bool,
) -> SegmentSizes {
    let needs_lc_routines_64 = ctx.is_dylib
        && ctx.mod_init_func_section.is_none()
        && ctx.init_offsets_section.is_none()
        && !no_coverage;
    let final_end_va = segment_end_va;

    let (segment_size, segment_vmsize, code_segment_size, code_segment_vmsize);
    if layout.arm64_split {
        segment_size = layout.page_sz;
        segment_vmsize = layout.page_sz;
        code_segment_size = final_end_va - layout.init_va;
        code_segment_vmsize = align_up(code_segment_size, layout.page_sz);
    } else {
        segment_size = final_end_va - layout.segment_va;
        segment_vmsize = align_up(segment_size, layout.page_sz);
        code_segment_size = 0;
        code_segment_vmsize = 0;
    }

    SegmentSizes {
        segment_size,
        segment_vmsize,
        code_segment_size,
        code_segment_vmsize,
        needs_lc_routines_64,
    }
}

/// Assemble all binary artifacts into segment data buffers.
#[allow(clippy::too_many_arguments)]
fn assemble_segment_data(
    layout: &MachoLayout,
    sizes: &SegmentSizes,
    init_code: &crate::trampoline::InitCode,
    trampolines: &[(&BasicBlock, crate::trampoline::Trampoline)],
    hooks: &[ResolvedHook],
    toggle_data_va: u64,
    hook_trampolines: &[(usize, crate::trampoline::Trampoline)],
    shellcode_blobs: &[(u64, Vec<u8>)],
    return_trampolines: &[crate::trampoline::Trampoline],
    persistent_data_va: u64,
    persistent_wrapper: &Option<(u64, crate::trampoline::PersistentWrapper, usize)>,
    heap_san_code: &Option<(crate::elf::AllocatorSymbols, crate::trampoline::HeapSanCode)>,
) -> (Vec<u8>, Vec<u8>) {
    if layout.arm64_split {
        let mut segment_data = vec![0u8; sizes.segment_size as usize];
        let mut code_segment_data = vec![0u8; sizes.code_segment_size as usize];

        code_segment_data[..init_code.code.len()].copy_from_slice(&init_code.code);

        for (_, tramp) in trampolines {
            let offset = (tramp.va - layout.init_va) as usize;
            code_segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
        }

        for hook in hooks {
            let byte_offset = (toggle_data_va - layout.data_va) as usize + hook.toggle_index;
            if byte_offset < segment_data.len() {
                segment_data[byte_offset] = if hook.initial_enabled { 1 } else { 0 };
            }
        }

        for (sc_va, sc_bytes) in shellcode_blobs {
            let offset = (*sc_va - layout.init_va) as usize;
            code_segment_data[offset..offset + sc_bytes.len()].copy_from_slice(sc_bytes);
        }

        for (_, tramp) in hook_trampolines {
            let offset = (tramp.va - layout.init_va) as usize;
            code_segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
        }

        for tramp in return_trampolines {
            let offset = (tramp.va - layout.init_va) as usize;
            code_segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
        }

        if let Some((wrapper_va, ref wrapper, _)) = *persistent_wrapper {
            let pd_offset = (persistent_data_va - layout.init_va) as usize;
            if pd_offset < code_segment_data.len() {
                code_segment_data[pd_offset] = 1;
            }
            let w_offset = (wrapper_va - layout.init_va) as usize;
            code_segment_data[w_offset..w_offset + wrapper.code.len()]
                .copy_from_slice(&wrapper.code);
        }

        if let Some((_, ref hsw)) = *heap_san_code {
            let base = *hsw
                .wrappers
                .values()
                .min()
                .expect("heap san wrappers are non-empty");
            let offset = (base - layout.init_va) as usize;
            code_segment_data[offset..offset + hsw.code.len()].copy_from_slice(&hsw.code);
        }

        (segment_data, code_segment_data)
    } else {
        let mut segment_data = vec![0u8; sizes.segment_size as usize];

        if layout.use_got_shmat {
            let bind_entry: u64 = (layout.shmat_ordinal as u64 & 0xFFFFFF) | (1u64 << 63);
            segment_data[16..24].copy_from_slice(&bind_entry.to_le_bytes());
        }

        let init_offset = (layout.init_va - layout.segment_va) as usize;
        segment_data[init_offset..init_offset + init_code.code.len()]
            .copy_from_slice(&init_code.code);

        for (_, tramp) in trampolines {
            let offset = (tramp.va - layout.segment_va) as usize;
            segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
        }

        for hook in hooks {
            let byte_offset = (toggle_data_va - layout.segment_va) as usize + hook.toggle_index;
            if byte_offset < segment_data.len() {
                segment_data[byte_offset] = if hook.initial_enabled { 1 } else { 0 };
            }
        }

        for (sc_va, sc_bytes) in shellcode_blobs {
            let offset = (*sc_va - layout.segment_va) as usize;
            segment_data[offset..offset + sc_bytes.len()].copy_from_slice(sc_bytes);
        }

        for (_, tramp) in hook_trampolines {
            let offset = (tramp.va - layout.segment_va) as usize;
            segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
        }

        for tramp in return_trampolines {
            let offset = (tramp.va - layout.segment_va) as usize;
            segment_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
        }

        if let Some((wrapper_va, ref wrapper, _)) = *persistent_wrapper {
            let pd_offset = (persistent_data_va - layout.segment_va) as usize;
            segment_data[pd_offset] = 1;
            let w_offset = (wrapper_va - layout.segment_va) as usize;
            segment_data[w_offset..w_offset + wrapper.code.len()].copy_from_slice(&wrapper.code);
        }

        if let Some((_, ref hsw)) = *heap_san_code {
            let base = *hsw
                .wrappers
                .values()
                .min()
                .expect("heap san wrappers are non-empty");
            let offset = (base - layout.segment_va) as usize;
            segment_data[offset..offset + hsw.code.len()].copy_from_slice(&hsw.code);
        }

        (segment_data, Vec::new())
    }
}

/// Patch basic blocks, hooks, persistent entry, and heap san sites in the binary data.
#[allow(clippy::too_many_arguments)]
fn patch_code_sites(
    data: &mut [u8],
    ctx: &MachOContext,
    layout: &MachoLayout,
    trampolines: &[(&BasicBlock, crate::trampoline::Trampoline)],
    hook_trampolines: &[(usize, crate::trampoline::Trampoline)],
    hooks: &[ResolvedHook],
    persistent_wrapper: &Option<(u64, crate::trampoline::PersistentWrapper, usize)>,
    persistent_addr: Option<u64>,
    heap_san_code: &Option<(crate::elf::AllocatorSymbols, crate::trampoline::HeapSanCode)>,
    tramp_gen: &dyn TrampolineGenerator,
) -> Result<(usize, usize)> {
    let branch_size = tramp_gen.branch_instruction_size();

    // Patch basic blocks with branch to trampoline.
    for (block, tramp) in trampolines {
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

    // Patch hook target sites.
    let mut hooks_applied = 0;
    for (hook_idx, tramp) in hook_trampolines {
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

    // Patch persistent function entry with branch to wrapper.
    if let Some((wrapper_va, _, displaced_len)) = persistent_wrapper {
        let p_addr =
            persistent_addr.expect("persistent_addr is Some when persistent_wrapper is set");
        let file_offset = crate::macho::va_to_file_offset_macho(p_addr, ctx).ok_or_else(|| {
            anyhow::anyhow!("persistent addr 0x{:x}: VA to file offset failed", p_addr)
        })?;
        match tramp_gen.encode_branch(p_addr, *wrapper_va) {
            Ok(branch_bytes) => {
                data[file_offset..file_offset + branch_bytes.len()].copy_from_slice(&branch_bytes);
                let branch_size = tramp_gen.branch_instruction_size();
                let nop = if layout.is_arm64 {
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

    // Patch allocator function entries for heap san.
    let mut heap_san_intercepted = 0;
    if let Some((ref alloc_syms, ref hsw)) = *heap_san_code {
        for (name, alloc) in &alloc_syms.entries {
            let wrapper_va = match hsw.wrappers.get(name) {
                Some(&va) => va,
                None => continue,
            };
            let off = alloc.patch_offset as usize;
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

    Ok((hooks_applied, heap_san_intercepted))
}

/// Build chained fixups blob if needed (must happen before file layout restructuring).
fn build_fixups_blob(
    data: &[u8],
    ctx: &MachOContext,
    layout: &MachoLayout,
    no_coverage: bool,
) -> Result<Option<Vec<u8>>> {
    if layout.use_got_shmat {
        let (blob, _ordinal) = build_chained_fixups_with_shmat(
            data,
            ctx,
            layout.segment_va,
            layout.page_sz,
            ctx.libsystem_ordinal,
        )
        .context("failed to rebuild chained fixups with _shmat")?;
        tracing::info!(
            "rebuilt chained fixups: {} → {} bytes (shmat ordinal={}, libsystem_ordinal={})",
            ctx.chained_fixups_datasize,
            blob.len(),
            layout.shmat_ordinal,
            ctx.libsystem_ordinal,
        );
        Ok(Some(blob))
    } else if ctx.chained_fixups_lc_offset.is_some() && !no_coverage {
        let extra_segs: u32 = if layout.arm64_split { 2 } else { 1 };
        let blob = rebuild_chained_fixups_add_n_empty_segments(data, ctx, extra_segs)
            .context("failed to rebuild chained fixups (empty segment entries)")?;
        tracing::info!(
            "rebuilt chained fixups ({} empty seg entries): {} → {} bytes",
            extra_segs,
            ctx.chained_fixups_datasize,
            blob.len(),
        );
        Ok(Some(blob))
    } else {
        Ok(None)
    }
}

/// Restructure the file layout: insert segment data before __LINKEDIT.
fn restructure_file_layout(
    data: &mut Vec<u8>,
    ctx: &MachOContext,
    layout: &MachoLayout,
    segment_data: &[u8],
    code_segment_data: &[u8],
    new_fixups_blob: &Option<Vec<u8>>,
) -> FileLayout {
    let file_offset_of_segment;
    let file_offset_of_code_seg;
    let new_linkedit_fileoff: u64;
    let new_fixups_fileoff: u32;
    let new_fixups_size: u32;

    if ctx.linkedit_segment_offset.is_some() {
        let linkedit_data = data[ctx.linkedit_fileoff as usize..].to_vec();
        data.truncate(ctx.linkedit_fileoff as usize);

        let pad1 = align_up(data.len() as u64, layout.page_sz) - data.len() as u64;
        data.extend(std::iter::repeat_n(0u8, pad1 as usize));
        file_offset_of_segment = data.len() as u64;
        data.extend_from_slice(segment_data);

        if layout.arm64_split {
            let pad_code = align_up(data.len() as u64, layout.page_sz) - data.len() as u64;
            data.extend(std::iter::repeat_n(0u8, pad_code as usize));
            file_offset_of_code_seg = data.len() as u64;
            data.extend_from_slice(code_segment_data);
        } else {
            file_offset_of_code_seg = 0;
        }

        let pad2 = align_up(data.len() as u64, layout.page_sz) - data.len() as u64;
        data.extend(std::iter::repeat_n(0u8, pad2 as usize));
        new_linkedit_fileoff = data.len() as u64;
        data.extend_from_slice(&linkedit_data);

        if let Some(ref blob) = *new_fixups_blob {
            new_fixups_fileoff = data.len() as u32;
            new_fixups_size = blob.len() as u32;
            data.extend_from_slice(blob);
        } else {
            new_fixups_fileoff = 0;
            new_fixups_size = 0;
        }
    } else {
        new_linkedit_fileoff = 0;
        if let Some(ref blob) = *new_fixups_blob {
            new_fixups_fileoff = data.len() as u32;
            new_fixups_size = blob.len() as u32;
            data.extend_from_slice(blob);
        } else {
            new_fixups_fileoff = 0;
            new_fixups_size = 0;
        }
        let pad = align_up(data.len() as u64, layout.page_sz) - data.len() as u64;
        data.extend(std::iter::repeat_n(0u8, pad as usize));
        file_offset_of_segment = data.len() as u64;
        data.extend_from_slice(segment_data);
        if layout.arm64_split {
            let pad_code = align_up(data.len() as u64, layout.page_sz) - data.len() as u64;
            data.extend(std::iter::repeat_n(0u8, pad_code as usize));
            file_offset_of_code_seg = data.len() as u64;
            data.extend_from_slice(code_segment_data);
        } else {
            file_offset_of_code_seg = 0;
        }
    }

    FileLayout {
        file_offset_of_segment,
        file_offset_of_code_seg,
        new_linkedit_fileoff,
        new_fixups_fileoff,
        new_fixups_size,
        // These are set by the caller after computing segment sizes.
        new_linkedit_vmaddr: 0,
        new_linkedit_vmsize: 0,
        new_linkedit_filesize: 0,
    }
}

/// Insert LC_SEGMENT_64 load commands, LC_ROUTINES_64 if needed,
/// update headers, LINKEDIT offsets, and redirect the entry point.
#[allow(clippy::too_many_arguments)]
fn insert_load_commands_and_update_entry(
    data: &mut [u8],
    ctx: &MachOContext,
    layout: &MachoLayout,
    sizes: &SegmentSizes,
    file_layout: &FileLayout,
    init_code: &crate::trampoline::InitCode,
    mut ncmds: u32,
    mut sizeofcmds: u32,
    no_coverage: bool,
) -> Result<()> {
    const LC_ROUTINES_64_SIZE: u32 = 72;
    let lc_seg_count: u32 = if layout.arm64_split { 2 } else { 1 };
    let lc_total_size = LC_SEGMENT_64_SIZE * lc_seg_count
        + if sizes.needs_lc_routines_64 {
            LC_ROUTINES_64_SIZE
        } else {
            0
        };
    let (lc_insert_offset, _lc_end_post_shift) = find_lc_insert_offset(data, ctx, lc_total_size);

    if layout.arm64_split {
        write_lc_segment_64_named(
            data,
            lc_insert_offset,
            b"__TR_DAT\0\0\0\0\0\0\0\0",
            layout.segment_va,
            sizes.segment_vmsize,
            file_layout.file_offset_of_segment,
            sizes.segment_size,
            0x3,
        );
        write_lc_segment_64_named(
            data,
            lc_insert_offset + LC_SEGMENT_64_SIZE as usize,
            b"__TR_COV\0\0\0\0\0\0\0\0",
            layout.segment_va + layout.page_sz,
            sizes.code_segment_vmsize,
            file_layout.file_offset_of_code_seg,
            sizes.code_segment_vmsize,
            0x7,
        );
    } else {
        write_lc_segment_64(
            data,
            lc_insert_offset,
            layout.segment_va,
            sizes.segment_vmsize,
            file_layout.file_offset_of_segment,
            sizes.segment_vmsize,
            0x7,
        );
    }

    if sizes.needs_lc_routines_64 {
        const LC_ROUTINES_64_CMD: u32 = 0x1A;
        let r = lc_insert_offset + (LC_SEGMENT_64_SIZE * lc_seg_count) as usize;
        data[r..r + LC_ROUTINES_64_SIZE as usize].fill(0);
        data[r..r + 4].copy_from_slice(&LC_ROUTINES_64_CMD.to_le_bytes());
        data[r + 4..r + 8].copy_from_slice(&LC_ROUTINES_64_SIZE.to_le_bytes());
        data[r + 8..r + 16].copy_from_slice(&init_code.entry_va.to_le_bytes());
        tracing::info!(
            "inserted LC_ROUTINES_64 init_address=0x{:x} (dylib has no existing init section)",
            init_code.entry_va,
        );
    }

    // Update ncmds and sizeofcmds in the header.
    let added_ncmds: u32 = lc_seg_count + if sizes.needs_lc_routines_64 { 1 } else { 0 };
    ncmds += added_ncmds;
    sizeofcmds += lc_total_size;
    data[ctx.ncmds_offset as usize..ctx.ncmds_offset as usize + 4]
        .copy_from_slice(&ncmds.to_le_bytes());
    data[ctx.sizeofcmds_offset as usize..ctx.sizeofcmds_offset as usize + 4]
        .copy_from_slice(&sizeofcmds.to_le_bytes());

    // Update __LINKEDIT and all LCs that reference LINKEDIT file offsets.
    if ctx.linkedit_segment_offset.is_some() && file_layout.new_linkedit_fileoff > 0 {
        let linkedit_shift = file_layout.new_linkedit_fileoff - ctx.linkedit_fileoff;
        update_linkedit_lc_offsets(data, ctx, ncmds, linkedit_shift, file_layout);

        tracing::info!(
            "relocated __LINKEDIT: VA 0x{:x} → 0x{:x}, fileoff 0x{:x} → 0x{:x} (shift +0x{:x})",
            ctx.linkedit_vmaddr,
            file_layout.new_linkedit_vmaddr,
            ctx.linkedit_fileoff,
            file_layout.new_linkedit_fileoff,
            linkedit_shift,
        );
    }

    // Redirect entry point (skip in no-coverage mode).
    let lc_shift = lc_total_size as u64;
    if no_coverage {
        // No entry point redirect — hooks only.
    } else if ctx.is_dylib {
        redirect_dylib_entry(data, ctx, init_code.entry_va)?;
    } else if let Some(lc_main_off) = ctx.lc_main_entryoff_offset {
        let adjusted_off = if lc_main_off >= lc_insert_offset as u64 {
            lc_main_off + lc_shift
        } else {
            lc_main_off
        };
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

    Ok(())
}

/// Orchestrate Mach-O patching:
/// 1. Strip code signature (reclaim LC space)
/// 2. Compute new segment layout
/// 3. Generate init code + trampolines
/// 4. Set up hook infrastructure
/// 5. Generate persistent wrapper
/// 6. Generate heap san wrappers
/// 7. Assemble segment data
/// 8. Patch code sites
/// 9. Restructure file layout
/// 10. Insert load commands + update entry point
fn patch_macho(
    ctx: &MachOContext,
    blocks: &[BasicBlock],
    mut data: Vec<u8>,
    opts: &InstrumentationOptions,
    tramp_gen: &dyn TrampolineGenerator,
    hooks: &[ResolvedHook],
) -> Result<PatchResult> {
    let enable_forkserver = opts.enable_forkserver;
    let enable_heap_san = opts.enable_heap_san;
    let persistent_addr = opts.persistent_addr;
    let no_coverage = opts.no_coverage;
    if blocks.is_empty() && !no_coverage && hooks.is_empty() {
        bail!("no basic blocks to instrument");
    }

    // Clone ctx so we can update cached LC offsets after stripping/compaction.
    let mut ctx = ctx.clone();
    let ctx = &mut ctx;

    let page_sz = page_size_for_arch(ctx);

    // Step 1: Reclaim LC space by stripping non-essential load commands.
    // ARM64 split layout needs up to 3 * 72 = 216 bytes (2 segments + LC_ROUTINES_64).
    let min_lc_space = if ctx.cputype == 0x0100_000C {
        LC_SEGMENT_64_SIZE as u64 * 3
    } else {
        LC_SEGMENT_64_SIZE as u64
    };
    let (ncmds, sizeofcmds, available_lc_space) = reclaim_lc_space(&mut data, ctx, min_lc_space);

    // Step 2: Verify we have room for at least one LC_SEGMENT_64.
    if available_lc_space < LC_SEGMENT_64_SIZE as u64 {
        bail!(
            "insufficient load command space for LC_SEGMENT_64: {} bytes available, {} needed.\n\
             Strip unused load commands or rebuild with -headerpad_max_install_names",
            available_lc_space,
            LC_SEGMENT_64_SIZE,
        );
    }

    // Step 3: Compute layout parameters.
    let layout = compute_layout(ctx, &data, hooks, no_coverage, page_sz);

    // Step 4: Generate init code + coverage trampolines.
    let (init_code, trampolines, blocks_skipped, mut current_va) = generate_init_and_trampolines(
        ctx,
        blocks,
        &layout,
        enable_forkserver,
        tramp_gen,
        no_coverage,
    )?;

    // Step 5: Set up hook infrastructure.
    let (
        hook_data_va,
        toggle_data_va,
        new_current_va,
        _return_slot_va,
        hook_trampolines,
        shellcode_blobs,
        return_trampolines,
    ) = setup_hook_infrastructure(hooks, &layout, current_va);
    current_va = new_current_va;

    // Step 6: Generate persistent wrapper.
    let (persistent_data_va, persistent_wrapper, new_current_va) =
        generate_persistent_wrapper(ctx, &data, &layout, current_va, opts, tramp_gen)?;
    current_va = new_current_va;

    // Step 7: Generate heap san wrappers.
    let (heap_san_code, new_current_va) =
        generate_heap_san(ctx, &data, &layout, current_va, enable_heap_san);
    current_va = new_current_va;

    let segment_end_va = current_va;

    // Step 8: Compute segment sizes.
    let sizes = compute_segment_sizes(ctx, &layout, segment_end_va, no_coverage);

    // Step 9: Assemble segment data buffers.
    let (segment_data, code_segment_data) = assemble_segment_data(
        &layout,
        &sizes,
        &init_code,
        &trampolines,
        hooks,
        toggle_data_va,
        &hook_trampolines,
        &shellcode_blobs,
        &return_trampolines,
        persistent_data_va,
        &persistent_wrapper,
        &heap_san_code,
    );

    // Step 10: Patch code sites (basic blocks, hooks, persistent, heap san).
    let (hooks_applied, heap_san_intercepted) = patch_code_sites(
        &mut data,
        ctx,
        &layout,
        &trampolines,
        &hook_trampolines,
        hooks,
        &persistent_wrapper,
        persistent_addr,
        &heap_san_code,
        tramp_gen,
    )?;

    // Step 11: Build chained fixups blob (needs original LINKEDIT data intact).
    let new_fixups_blob = build_fixups_blob(&data, ctx, &layout, no_coverage)?;

    // Step 12: Restructure file layout — insert segments before __LINKEDIT.
    let mut file_layout = restructure_file_layout(
        &mut data,
        ctx,
        &layout,
        &segment_data,
        &code_segment_data,
        &new_fixups_blob,
    );

    // Compute LINKEDIT layout fields now that data is restructured.
    if ctx.linkedit_segment_offset.is_some() && file_layout.new_linkedit_fileoff > 0 {
        let total_tr_vmsize = sizes.segment_vmsize
            + if layout.arm64_split {
                sizes.code_segment_vmsize
            } else {
                0
            };
        file_layout.new_linkedit_vmaddr = layout.segment_va + total_tr_vmsize;
        file_layout.new_linkedit_filesize = data.len() as u64 - file_layout.new_linkedit_fileoff;
        file_layout.new_linkedit_vmsize =
            align_up(file_layout.new_linkedit_filesize, layout.page_sz);
    }

    // Step 13: Insert load commands, update LINKEDIT offsets, redirect entry point.
    insert_load_commands_and_update_entry(
        &mut data,
        ctx,
        &layout,
        &sizes,
        &file_layout,
        &init_code,
        ncmds,
        sizeofcmds,
        no_coverage,
    )?;

    let blocks_instrumented = trampolines.len();

    tracing::info!(
        "Mach-O: patched {} blocks, new segment __TR_COV at VA 0x{:x} ({} bytes)",
        blocks_instrumented,
        layout.segment_va,
        sizes.segment_size,
    );

    Ok(PatchResult {
        data,
        blocks_instrumented,
        blocks_skipped,
        segment_va: layout.segment_va,
        segment_size: sizes.segment_size,
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
                .expect("slice is 4 bytes"),
        );
        // Walk LCs to find the end of valid load commands
        let mut offset = ctx.header_size;
        for _ in 0..ncmds {
            if offset + 8 > data.len() {
                break;
            }
            let cmd = u32::from_le_bytes(
                data[offset..offset + 4]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            let cmdsize = u32::from_le_bytes(
                data[offset + 4..offset + 8]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
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
                .expect("slice is 4 bytes"),
        );
        for _ in 0..ncmds {
            if offset + 8 > data.len() {
                break;
            }
            let cmd = u32::from_le_bytes(
                data[offset..offset + 4]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            let cmdsize = u32::from_le_bytes(
                data[offset + 4..offset + 8]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
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
    use crate::binary_patch::PatchSet;
    let mut ps = PatchSet::new();
    ps.write_u32(offset, LC_SEGMENT_64, "LC_SEGMENT_64 cmd");
    ps.write_u32(offset + 4, LC_SEGMENT_64_SIZE, "LC_SEGMENT_64 cmdsize");
    ps.write(offset + 8, segname, "LC_SEGMENT_64 segname");
    ps.write_u64(offset + 24, vmaddr, "LC_SEGMENT_64 vmaddr");
    ps.write_u64(offset + 32, vmsize, "LC_SEGMENT_64 vmsize");
    ps.write_u64(offset + 40, fileoff, "LC_SEGMENT_64 fileoff");
    ps.write_u64(offset + 48, filesize, "LC_SEGMENT_64 filesize");
    ps.write_u32(offset + 56, 7, "LC_SEGMENT_64 maxprot");
    ps.write_u32(offset + 60, initprot, "LC_SEGMENT_64 initprot");
    ps.write_u32(offset + 64, 0, "LC_SEGMENT_64 nsects");
    ps.write_u32(offset + 68, 0, "LC_SEGMENT_64 flags");
    ps.apply(data)
        .expect("LC_SEGMENT_64 patches must not overlap");
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
    let val = u32::from_le_bytes(
        data[offset..offset + 4]
            .try_into()
            .expect("slice is 4 bytes"),
    );
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
fn update_linkedit_lc_offsets(
    data: &mut [u8],
    ctx: &MachOContext,
    ncmds: u32,
    linkedit_shift: u64,
    fl: &FileLayout,
) {
    let new_linkedit_vmaddr = fl.new_linkedit_vmaddr;
    let new_linkedit_vmsize = fl.new_linkedit_vmsize;
    let new_linkedit_fileoff = fl.new_linkedit_fileoff;
    let new_linkedit_filesize = fl.new_linkedit_filesize;
    let new_fixups_fileoff = fl.new_fixups_fileoff;
    let new_fixups_size = fl.new_fixups_size;
    let mut lc_off = ctx.header_size;

    for _ in 0..ncmds {
        if lc_off + 8 > data.len() {
            break;
        }
        let cmd = u32::from_le_bytes(
            data[lc_off..lc_off + 4]
                .try_into()
                .expect("slice is 4 bytes"),
        );
        let cmdsize = u32::from_le_bytes(
            data[lc_off + 4..lc_off + 8]
                .try_into()
                .expect("slice is 4 bytes"),
        );
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
    let fixups_version = u32::from_le_bytes(orig[0..4].try_into().expect("slice is 4 bytes"));
    let starts_offset =
        u32::from_le_bytes(orig[4..8].try_into().expect("slice is 4 bytes")) as usize;
    let imports_offset =
        u32::from_le_bytes(orig[8..12].try_into().expect("slice is 4 bytes")) as usize;
    let symbols_offset =
        u32::from_le_bytes(orig[12..16].try_into().expect("slice is 4 bytes")) as usize;
    let imports_count = u32::from_le_bytes(orig[16..20].try_into().expect("slice is 4 bytes"));
    let imports_format = u32::from_le_bytes(orig[20..24].try_into().expect("slice is 4 bytes"));
    let symbols_format = u32::from_le_bytes(orig[24..28].try_into().expect("slice is 4 bytes"));

    // Parse starts_in_image
    let orig_seg_count = u32::from_le_bytes(
        orig[starts_offset..starts_offset + 4]
            .try_into()
            .expect("slice is 4 bytes"),
    );
    let new_seg_count = orig_seg_count + extra_count;
    let orig_starts_hdr_end = starts_offset + 4 + orig_seg_count as usize * 4;

    // Read original seg_info_offsets
    let mut orig_seg_offsets = Vec::with_capacity(orig_seg_count as usize);
    for i in 0..orig_seg_count as usize {
        let off = starts_offset + 4 + i * 4;
        orig_seg_offsets.push(u32::from_le_bytes(
            orig[off..off + 4].try_into().expect("slice is 4 bytes"),
        ));
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
    let fixups_version = u32::from_le_bytes(orig[0..4].try_into().expect("slice is 4 bytes"));
    let starts_offset =
        u32::from_le_bytes(orig[4..8].try_into().expect("slice is 4 bytes")) as usize;
    let imports_offset =
        u32::from_le_bytes(orig[8..12].try_into().expect("slice is 4 bytes")) as usize;
    let symbols_offset =
        u32::from_le_bytes(orig[12..16].try_into().expect("slice is 4 bytes")) as usize;
    let imports_count = u32::from_le_bytes(orig[16..20].try_into().expect("slice is 4 bytes"));
    let imports_format = u32::from_le_bytes(orig[20..24].try_into().expect("slice is 4 bytes"));
    let symbols_format = u32::from_le_bytes(orig[24..28].try_into().expect("slice is 4 bytes"));

    if imports_format != 1 {
        bail!(
            "unsupported chained fixups imports_format: {} (only DYLD_CHAINED_IMPORT supported)",
            imports_format,
        );
    }

    // Parse starts_in_image
    let orig_seg_count = u32::from_le_bytes(
        orig[starts_offset..starts_offset + 4]
            .try_into()
            .expect("slice is 4 bytes"),
    );
    let new_seg_count = orig_seg_count + 1;
    let orig_starts_hdr_end = starts_offset + 4 + orig_seg_count as usize * 4;

    // Read original seg_info_offsets (relative to starts_in_image)
    let mut orig_seg_offsets = Vec::with_capacity(orig_seg_count as usize);
    for i in 0..orig_seg_count as usize {
        let off = starts_offset + 4 + i * 4;
        orig_seg_offsets.push(u32::from_le_bytes(
            orig[off..off + 4].try_into().expect("slice is 4 bytes"),
        ));
    }

    // Find pointer_format from first segment with fixups.
    // starts_in_segment layout: size(4) page_size(2) pointer_format(2) ...
    let mut pointer_format: u16 = 2; // DYLD_CHAINED_PTR_64
    for &off in &orig_seg_offsets {
        if off != 0 {
            let sis_base = starts_offset + off as usize;
            if sis_base + 8 <= orig.len() {
                pointer_format = u16::from_le_bytes(
                    orig[sis_base + 6..sis_base + 8]
                        .try_into()
                        .expect("slice is 2 bytes"),
                );
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
    if let Some(ref mod_init) = ctx.mod_init_func_section
        && mod_init.section_size >= 8
    {
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

    // Fall back to __init_offsets (32-bit offsets from __TEXT base)
    if let Some(ref init_off) = ctx.init_offsets_section
        && init_off.section_size >= 4
    {
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

    fn default_opts() -> InstrumentationOptions {
        InstrumentationOptions {
            enable_forkserver: false,
            enable_heap_san: false,
            persistent_addr: None,
            persistent_count: 0,
            defer: false,
            no_coverage: false,
        }
    }

    fn no_coverage_opts() -> InstrumentationOptions {
        InstrumentationOptions {
            no_coverage: true,
            ..default_opts()
        }
    }

    fn make_file_layout(
        new_linkedit_vmaddr: u64,
        new_linkedit_vmsize: u64,
        new_linkedit_fileoff: u64,
        new_linkedit_filesize: u64,
        new_fixups_fileoff: u32,
        new_fixups_size: u32,
    ) -> FileLayout {
        FileLayout {
            file_offset_of_segment: 0,
            file_offset_of_code_seg: 0,
            new_linkedit_fileoff,
            new_fixups_fileoff,
            new_fixups_size,
            new_linkedit_vmaddr,
            new_linkedit_vmsize,
            new_linkedit_filesize,
        }
    }

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
        assert_eq!(
            u32::from_le_bytes(data[0..4].try_into().expect("slice is 4 bytes")),
            0x19
        );
        // cmdsize = 72
        assert_eq!(
            u32::from_le_bytes(data[4..8].try_into().expect("slice is 4 bytes")),
            72
        );
        // segname starts with "__TR_COV"
        assert_eq!(&data[8..16], b"__TR_COV");
        // vmaddr
        assert_eq!(
            u64::from_le_bytes(data[24..32].try_into().expect("slice is 8 bytes")),
            0x200000
        );
        // vmsize
        assert_eq!(
            u64::from_le_bytes(data[32..40].try_into().expect("slice is 8 bytes")),
            0x4000
        );
        // fileoff
        assert_eq!(
            u64::from_le_bytes(data[40..48].try_into().expect("slice is 8 bytes")),
            0x8000
        );
        // filesize
        assert_eq!(
            u64::from_le_bytes(data[48..56].try_into().expect("slice is 8 bytes")),
            0x1234
        );
        // maxprot = 7
        assert_eq!(
            u32::from_le_bytes(data[56..60].try_into().expect("slice is 4 bytes")),
            7
        );
    }

    // --- shift_lc_u32_field tests ---

    #[test]
    fn test_shift_lc_u32_field_nonzero() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(&1000u32.to_le_bytes());
        shift_lc_u32_field(&mut data, 0, 0x2000);
        assert_eq!(
            u32::from_le_bytes(data[0..4].try_into().expect("slice is 4 bytes")),
            1000 + 0x2000,
        );
    }

    #[test]
    fn test_shift_lc_u32_field_zero_skipped() {
        let mut data = vec![0u8; 8];
        // Zero field should remain zero (unused LC_DYSYMTAB fields).
        shift_lc_u32_field(&mut data, 0, 0x5000);
        assert_eq!(
            u32::from_le_bytes(data[0..4].try_into().expect("slice is 4 bytes")),
            0
        );
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
            u32::from_le_bytes(data[0..4].try_into().expect("slice is 4 bytes")),
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
        u32::from_le_bytes(data[off..off + 4].try_into().expect("slice is 4 bytes"))
    }

    /// Helper: read a u64 from offset.
    fn get_u64(data: &[u8], off: usize) -> u64 {
        u64::from_le_bytes(data[off..off + 8].try_into().expect("slice is 8 bytes"))
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
            &make_file_layout(
                ctx.linkedit_vmaddr + shift,
                align_up(ctx.linkedit_filesize + shift, 0x1000),
                ctx.linkedit_fileoff + shift,
                ctx.linkedit_filesize,
                0,
                0,
            ),
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
            &make_file_layout(
                ctx.linkedit_vmaddr + shift,
                align_up(ctx.linkedit_filesize + shift, 0x1000),
                ctx.linkedit_fileoff + shift,
                ctx.linkedit_filesize,
                0,
                0,
            ),
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
            &make_file_layout(
                ctx.linkedit_vmaddr + shift,
                align_up(ctx.linkedit_filesize, 0x1000),
                ctx.linkedit_fileoff + shift,
                ctx.linkedit_filesize,
                0,
                0,
            ),
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
        let linkedit_lc = ctx
            .linkedit_segment_offset
            .expect("linkedit_segment_offset must be set") as usize;
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
            &make_file_layout(new_vmaddr, new_vmsize, new_fileoff, new_filesize, 0, 0),
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
            &default_opts(),
            &MockTrampolineGen,
            &[],
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
            &default_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs
            .iter()
            .find(|s| s.0 == "__TR_COV")
            .expect("__TR_COV segment must exist");
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");

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
            &default_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");

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
            &default_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");
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
        let orig_symtab = read_symtab(&data, ctx.header_size).expect("read_symtab should succeed");

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
            &default_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");
        let shift = linkedit.3 - ctx.linkedit_fileoff;

        let new_symtab =
            read_symtab(&result.data, ctx.header_size).expect("read_symtab should succeed");
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
        let orig_fs = read_function_starts_offset(&data, ctx.header_size)
            .expect("read_function_starts_offset should succeed");

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
            &default_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");
        let shift = linkedit.3 - ctx.linkedit_fileoff;

        let new_fs = read_function_starts_offset(&result.data, ctx.header_size)
            .expect("read_function_starts_offset should succeed");
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
        let orig_indirect = read_dysymtab_indirectsymoff(&data, ctx.header_size)
            .expect("read_dysymtab_indirectsymoff should succeed");

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
            &default_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");
        let shift = linkedit.3 - ctx.linkedit_fileoff;

        let new_indirect = read_dysymtab_indirectsymoff(&result.data, ctx.header_size)
            .expect("read_dysymtab_indirectsymoff should succeed");
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
            &default_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let file_len = result.data.len() as u32;

        // Every shifted offset must be < file length.
        let symtab =
            read_symtab(&result.data, ctx.header_size).expect("read_symtab should succeed");
        assert!(symtab.0 < file_len, "symoff out of file bounds");
        assert!(symtab.1 < file_len, "stroff out of file bounds");

        let fs = read_function_starts_offset(&result.data, ctx.header_size)
            .expect("read_function_starts_offset should succeed");
        assert!(fs < file_len, "function_starts dataoff out of file bounds");

        let indirect = read_dysymtab_indirectsymoff(&result.data, ctx.header_size)
            .expect("read_dysymtab_indirectsymoff should succeed");
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
            &no_coverage_opts(), // ARM64 MockTrampolineGen emits x86
            &MockTrampolineGen,
            &[],
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
            &no_coverage_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs
            .iter()
            .find(|s| s.0 == "__TR_COV")
            .expect("__TR_COV segment must exist");
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");

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
            &no_coverage_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");

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
            &no_coverage_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs
            .iter()
            .find(|s| s.0 == "__TR_COV")
            .expect("__TR_COV segment must exist");
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");

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
            &no_coverage_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");
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
        let orig_symtab = read_symtab(&data, ctx.header_size).expect("read_symtab should succeed");
        let orig_fs = read_function_starts_offset(&data, ctx.header_size)
            .expect("read_function_starts_offset should succeed");
        let orig_indirect = read_dysymtab_indirectsymoff(&data, ctx.header_size)
            .expect("read_dysymtab_indirectsymoff should succeed");

        let result = patch_macho(
            &ctx,
            &[],
            data,
            &no_coverage_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");
        let shift = (linkedit.3 - ctx.linkedit_fileoff) as u32;

        let new_symtab =
            read_symtab(&result.data, ctx.header_size).expect("read_symtab should succeed");
        let new_fs = read_function_starts_offset(&result.data, ctx.header_size)
            .expect("read_function_starts_offset should succeed");
        let new_indirect = read_dysymtab_indirectsymoff(&result.data, ctx.header_size)
            .expect("read_dysymtab_indirectsymoff should succeed");

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
            &no_coverage_opts(),
            &MockTrampolineGen,
            &[],
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
        let orig = read_linkedit_data_lc(&data, ctx.header_size, LC_DATA_IN_CODE)
            .expect("read_linkedit_data_lc should succeed");
        let shift = 0x6000u64;

        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            &make_file_layout(
                ctx.linkedit_vmaddr + shift,
                align_up(ctx.linkedit_filesize, 0x1000),
                ctx.linkedit_fileoff + shift,
                ctx.linkedit_filesize,
                0,
                0,
            ),
        );

        let new = read_linkedit_data_lc(&data, ctx.header_size, LC_DATA_IN_CODE)
            .expect("read_linkedit_data_lc should succeed");
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
        let orig = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_EXPORTS_TRIE)
            .expect("read_linkedit_data_lc should succeed");
        let shift = 0x5000u64;

        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            &make_file_layout(
                ctx.linkedit_vmaddr + shift,
                align_up(ctx.linkedit_filesize, 0x1000),
                ctx.linkedit_fileoff + shift,
                ctx.linkedit_filesize,
                0,
                0,
            ),
        );

        let new = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_EXPORTS_TRIE)
            .expect("read_linkedit_data_lc should succeed");
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
        let orig = read_linkedit_data_lc(&data, ctx.header_size, LC_SEGMENT_SPLIT_INFO)
            .expect("read_linkedit_data_lc should succeed");
        let shift = 0x7000u64;

        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            &make_file_layout(
                ctx.linkedit_vmaddr + shift,
                align_up(ctx.linkedit_filesize, 0x1000),
                ctx.linkedit_fileoff + shift,
                ctx.linkedit_filesize,
                0,
                0,
            ),
        );

        let new = read_linkedit_data_lc(&data, ctx.header_size, LC_SEGMENT_SPLIT_INFO)
            .expect("read_linkedit_data_lc should succeed");
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
            &make_file_layout(
                ctx.linkedit_vmaddr + shift,
                align_up(ctx.linkedit_filesize, 0x1000),
                ctx.linkedit_fileoff + shift,
                ctx.linkedit_filesize,
                new_blob_off,
                new_blob_size,
            ),
        );

        let fixups = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_CHAINED_FIXUPS)
            .expect("read_linkedit_data_lc should succeed");
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
        let orig = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_CHAINED_FIXUPS)
            .expect("read_linkedit_data_lc should succeed");
        let shift = 0x3000u64;

        // new_fixups_fileoff=0 triggers the shift-only fallback path.
        update_linkedit_lc_offsets(
            &mut data,
            &ctx,
            ctx.ncmds,
            shift,
            &make_file_layout(
                ctx.linkedit_vmaddr + shift,
                align_up(ctx.linkedit_filesize, 0x1000),
                ctx.linkedit_fileoff + shift,
                ctx.linkedit_filesize,
                0,
                0,
            ),
        );

        let fixups = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_CHAINED_FIXUPS)
            .expect("read_linkedit_data_lc should succeed");
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
        let orig_symtab = read_symtab(&data, ctx.header_size).expect("read_symtab should succeed");
        let orig_fs = read_linkedit_data_lc(&data, ctx.header_size, LC_FUNCTION_STARTS)
            .expect("read_linkedit_data_lc should succeed");
        let orig_dic = read_linkedit_data_lc(&data, ctx.header_size, LC_DATA_IN_CODE)
            .expect("read_linkedit_data_lc should succeed");
        let orig_et = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_EXPORTS_TRIE)
            .expect("read_linkedit_data_lc should succeed");
        let orig_ssi = read_linkedit_data_lc(&data, ctx.header_size, LC_SEGMENT_SPLIT_INFO)
            .expect("read_linkedit_data_lc should succeed");

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
            &default_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");
        let shift = (linkedit.3 - ctx.linkedit_fileoff) as u32;
        let file_len = result.data.len() as u32;

        // All offsets shifted by the same amount.
        let new_symtab =
            read_symtab(&result.data, ctx.header_size).expect("read_symtab should succeed");
        assert_eq!(new_symtab.0, orig_symtab.0 + shift, "symoff");
        assert_eq!(new_symtab.1, orig_symtab.1 + shift, "stroff");

        let new_fs = read_linkedit_data_lc(&result.data, ctx.header_size, LC_FUNCTION_STARTS)
            .expect("read_linkedit_data_lc should succeed");
        assert_eq!(new_fs.0, orig_fs.0 + shift, "function_starts");

        let new_dic = read_linkedit_data_lc(&result.data, ctx.header_size, LC_DATA_IN_CODE)
            .expect("read_linkedit_data_lc should succeed");
        assert_eq!(new_dic.0, orig_dic.0 + shift, "data_in_code");

        let new_et = read_linkedit_data_lc(&result.data, ctx.header_size, LC_DYLD_EXPORTS_TRIE)
            .expect("read_linkedit_data_lc should succeed");
        assert_eq!(new_et.0, orig_et.0 + shift, "exports_trie");

        let new_ssi = read_linkedit_data_lc(&result.data, ctx.header_size, LC_SEGMENT_SPLIT_INFO)
            .expect("read_linkedit_data_lc should succeed");
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
        let orig_dic = read_linkedit_data_lc(&data, ctx.header_size, LC_DATA_IN_CODE)
            .expect("read_linkedit_data_lc should succeed");
        let orig_et = read_linkedit_data_lc(&data, ctx.header_size, LC_DYLD_EXPORTS_TRIE)
            .expect("read_linkedit_data_lc should succeed");
        let orig_ssi = read_linkedit_data_lc(&data, ctx.header_size, LC_SEGMENT_SPLIT_INFO)
            .expect("read_linkedit_data_lc should succeed");

        let result = patch_macho(
            &ctx,
            &[],
            data,
            &no_coverage_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");
        let shift = (linkedit.3 - ctx.linkedit_fileoff) as u32;

        let new_dic = read_linkedit_data_lc(&result.data, ctx.header_size, LC_DATA_IN_CODE)
            .expect("read_linkedit_data_lc should succeed");
        assert_eq!(new_dic.0, orig_dic.0 + shift, "ARM64: data_in_code");

        let new_et = read_linkedit_data_lc(&result.data, ctx.header_size, LC_DYLD_EXPORTS_TRIE)
            .expect("read_linkedit_data_lc should succeed");
        assert_eq!(new_et.0, orig_et.0 + shift, "ARM64: exports_trie");

        let new_ssi = read_linkedit_data_lc(&result.data, ctx.header_size, LC_SEGMENT_SPLIT_INFO)
            .expect("read_linkedit_data_lc should succeed");
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
            &no_coverage_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho with chained fixups should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");
        let fixups = read_linkedit_data_lc(&result.data, ctx.header_size, LC_DYLD_CHAINED_FIXUPS)
            .expect("read_linkedit_data_lc should succeed");

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
            &default_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs
            .iter()
            .find(|s| s.0 == "__TR_COV")
            .expect("__TR_COV segment must exist");
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");

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
            &default_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");

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
            &no_coverage_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");

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
            &default_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs
            .iter()
            .find(|s| s.0 == "__TR_COV")
            .expect("__TR_COV segment must exist");
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");

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
            &no_coverage_opts(),
            &MockTrampolineGen,
            &[],
        )
        .expect("patch_macho should succeed");

        let segs = read_segments(&result.data, ctx.header_size);
        let tr_cov = segs
            .iter()
            .find(|s| s.0 == "__TR_COV")
            .expect("__TR_COV segment must exist");
        let linkedit = segs
            .iter()
            .find(|s| s.0 == "__LINKEDIT")
            .expect("__LINKEDIT segment must exist");

        assert_eq!(
            tr_cov.1 + tr_cov.2,
            linkedit.1,
            "ARM64: TR_COV and LINKEDIT must be contiguous in VM space"
        );
    }
}
