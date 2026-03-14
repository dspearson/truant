use anyhow::{Context, Result, bail};

use crate::disasm::BasicBlock;
use crate::hook_trampoline::TargetAbi;
use crate::hooks::{self, HookSource, ResolvedHook};
use crate::patcher::PatchResult;
use crate::pe::PeContext;
use crate::traits::{BinaryContext, Patcher, TrampolineGenerator};
use crate::trampoline::{DATA_SIZE, PREV_LOC_OFFSET};

/// Information about a hook library to resolve directly in PE init code.
///
/// Instead of generating a companion DLL (which cannot be compiled on Linux
/// for Windows), we embed LoadLibraryA + GetProcAddress calls in the init
/// code to resolve handler symbols at runtime.
pub struct PeHookLibraryInfo {
    /// Library path (embedded as null-terminated ASCII string in the section).
    pub library_path: String,
    /// Handler symbol names and their data slot VAs.
    pub handlers: Vec<(String, u64)>,
}

/// PE-specific patcher implementing the Patcher trait.
///
/// Appends a `.trcov` section to the PE file containing coverage trampolines,
/// init code, and hook data. Updates PE headers accordingly.
pub struct PePatcher {
    tramp_gen: Box<dyn TrampolineGenerator>,
    /// Optional library path for hook handler resolution (embedded in PE init code).
    hook_library_path: Option<String>,
}

impl std::fmt::Debug for PePatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PePatcher")
            .field("tramp_gen", &self.tramp_gen)
            .field("hook_library_path", &self.hook_library_path)
            .finish()
    }
}

impl PePatcher {
    pub fn new(tramp_gen: Box<dyn TrampolineGenerator>) -> Self {
        Self {
            tramp_gen,
            hook_library_path: None,
        }
    }

    /// Set the hook library path for PE init code handler resolution.
    pub fn set_hook_library_path(&mut self, path: String) {
        self.hook_library_path = Some(path);
    }
}

/// PE section header characteristics.
const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

/// Size of a PE section header entry (40 bytes).
const SECTION_HEADER_SIZE: usize = 40;

/// PE persistent mode data area (200 bytes).
///
/// Layout:
///   +0:   first_pass (u8, init=1)
///   +4:   counter (u32)
///   +8:   save_regs[16 x 8] (128 bytes: rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8-r15)
///   +136: save_ret_addr (u64)
///   +144: save_rsp (u64)
///   +152: ReadFile_ptr (u64, resolved by init code)
///   +160: WriteFile_ptr (u64, resolved by init code)
///   +168: ExitProcess_ptr (u64, resolved by init code)
///   +176: padding (24 bytes)
const PE_PERSISTENT_DATA_SIZE: u64 = 200;

/// PE32 persistent mode data area (64 bytes).
///
/// Layout:
///   +0:   first_pass (u8, init=1)
///   +4:   counter (u32)
///   +8:   save_regs[8 x 4] (32 bytes: eax, ecx, edx, ebx, esp, ebp, esi, edi)
///   +40:  save_ret_addr (u32)
///   +44:  save_esp (u32)
///   +48:  ReadFile_ptr (u32, resolved by init code)
///   +52:  WriteFile_ptr (u32, resolved by init code)
///   +56:  ExitProcess_ptr (u32, resolved by init code)
///   +60:  padding (4 bytes)
const PE32_PERSISTENT_DATA_SIZE: u64 = 64;

/// Align a value up to the given alignment.
fn align_up(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

impl Patcher for PePatcher {
    fn patch(
        &self,
        ctx: &dyn BinaryContext,
        blocks: &[BasicBlock],
        data: Vec<u8>,
        _enable_forkserver: bool,
        _enable_heap_san: bool,
        persistent_addr: Option<u64>,
        persistent_count: u32,
        _defer: bool,
        hooks: &[ResolvedHook],
        no_coverage: bool,
    ) -> Result<PatchResult> {
        let pe_ctx = ctx
            .as_any()
            .downcast_ref::<crate::pe_impl::context::PeBinaryContext>()
            .ok_or_else(|| anyhow::anyhow!("PePatcher requires PeBinaryContext"))?;

        patch_pe(
            pe_ctx.inner(),
            blocks,
            data,
            &*self.tramp_gen,
            hooks,
            no_coverage,
            self.hook_library_path.as_ref(),
            persistent_addr,
            persistent_count,
        )
    }
}

/// Filter and sort basic blocks, removing overlapping ones.
fn filter_and_sort_blocks(blocks: &[BasicBlock]) -> Vec<&BasicBlock> {
    let mut sorted: Vec<&BasicBlock> = blocks.iter().collect();
    sorted.sort_by_key(|b| b.va);
    let mut filtered: Vec<&BasicBlock> = Vec::with_capacity(sorted.len());
    for block in &sorted {
        if let Some(prev) = filtered.last()
            && block.va < prev.va + prev.displaced_len as u64
        {
            continue;
        }
        filtered.push(block);
    }
    if filtered.len() < sorted.len() {
        tracing::info!(
            "removed {} overlapping blocks ({} -> {})",
            sorted.len() - filtered.len(),
            sorted.len(),
            filtered.len(),
        );
    }
    filtered
}

/// Build the section data buffer containing init code, trampolines, persistent data,
/// hook data, shellcode blobs, and hook trampolines.
#[allow(clippy::too_many_arguments)]
fn build_section_data(
    section_raw_size: u64,
    new_section_va: u64,
    init_va: u64,
    init_code: &crate::trampoline::InitCode,
    trampolines: &[(&BasicBlock, crate::trampoline::Trampoline)],
    persistent_data_va: Option<u64>,
    persistent_wrapper: Option<crate::trampoline::PersistentWrapper>,
    persistent_wrapper_va: Option<u64>,
    toggle_data_va: u64,
    resolved_hooks: &[ResolvedHook],
    shellcode_blobs: &[(u64, Vec<u8>)],
    hook_trampolines: &[(usize, crate::trampoline::Trampoline)],
    return_trampolines: &[crate::trampoline::Trampoline],
) -> Vec<u8> {
    let mut section_data = vec![0u8; section_raw_size as usize];
    if !init_code.code.is_empty() {
        let init_offset = (init_va - new_section_va) as usize;
        section_data[init_offset..init_offset + init_code.code.len()]
            .copy_from_slice(&init_code.code);
    }
    for (_, tramp) in trampolines {
        let offset = (tramp.va - new_section_va) as usize;
        section_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
    }
    if let Some(p_data_va) = persistent_data_va {
        let p_offset = (p_data_va - new_section_va) as usize;
        section_data[p_offset] = 1;
    }
    if let Some(wrapper) = persistent_wrapper {
        let w_va = persistent_wrapper_va.expect("persistent_wrapper_va must be set");
        let w_offset = (w_va - new_section_va) as usize;
        section_data[w_offset..w_offset + wrapper.code.len()].copy_from_slice(&wrapper.code);
    }
    for hook in resolved_hooks {
        let offset = (toggle_data_va - new_section_va) as usize + hook.toggle_index;
        if offset < section_data.len() {
            section_data[offset] = if hook.initial_enabled { 1 } else { 0 };
        }
    }
    for (sc_va, sc_bytes) in shellcode_blobs {
        let offset = (*sc_va - new_section_va) as usize;
        section_data[offset..offset + sc_bytes.len()].copy_from_slice(sc_bytes);
    }
    for (_, tramp) in hook_trampolines {
        let offset = (tramp.va - new_section_va) as usize;
        section_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
    }
    for tramp in return_trampolines {
        let offset = (tramp.va - new_section_va) as usize;
        section_data[offset..offset + tramp.code.len()].copy_from_slice(&tramp.code);
    }
    section_data
}

/// Patch basic blocks with JMP rel32 to their trampolines.
fn patch_coverage_blocks(
    data: &mut [u8],
    trampolines: &[(&BasicBlock, crate::trampoline::Trampoline)],
    tramp_gen: &dyn TrampolineGenerator,
) -> Result<usize> {
    let mut blocks_instrumented = 0;
    for (block, tramp) in trampolines {
        let branch = tramp_gen
            .encode_branch(block.va, tramp.va)
            .with_context(|| format!("failed to encode branch at 0x{:x}", block.va))?;
        let fo = block.file_offset as usize;
        if fo + branch.len() > data.len() {
            tracing::warn!("block at 0x{:x}: file offset out of range", block.va);
            continue;
        }
        data[fo..fo + branch.len()].copy_from_slice(&branch);
        for i in branch.len()..block.displaced_len {
            data[fo + i] = 0x90;
        }
        blocks_instrumented += 1;
    }
    Ok(blocks_instrumented)
}

/// Patch hook targets with JMP rel32 to hook trampolines.
fn patch_hook_targets(
    data: &mut [u8],
    resolved_hooks: &[ResolvedHook],
    hook_trampolines: &[(usize, crate::trampoline::Trampoline)],
    tramp_gen: &dyn TrampolineGenerator,
) -> Result<usize> {
    let mut hooks_applied = 0;
    let mut hook_tramp_map: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for (i, tramp) in hook_trampolines {
        let hook = &resolved_hooks[*i];
        hook_tramp_map.insert(hook.target_va, tramp.va);
    }
    for hook in resolved_hooks {
        if let Some(&tramp_va) = hook_tramp_map.get(&hook.target_va) {
            let branch = tramp_gen
                .encode_branch(hook.target_va, tramp_va)
                .with_context(|| {
                    format!("failed to encode hook branch at 0x{:x}", hook.target_va)
                })?;
            let fo = hook.file_offset as usize;
            if fo + branch.len() > data.len() {
                tracing::warn!("hook at 0x{:x}: file offset out of range", hook.target_va);
                continue;
            }
            data[fo..fo + branch.len()].copy_from_slice(&branch);
            for i in branch.len()..hook.displaced_len {
                data[fo + i] = 0x90;
            }
            hooks_applied += 1;
            hook_tramp_map.remove(&hook.target_va);
        }
    }
    Ok(hooks_applied)
}

/// Patch the persistent address with JMP rel32 to persistent wrapper.
fn patch_persistent_addr(
    data: &mut [u8],
    ctx: &PeContext,
    persistent_addr: Option<u64>,
    persistent_wrapper_va: Option<u64>,
    persistent_displaced_len: usize,
    tramp_gen: &dyn TrampolineGenerator,
) -> Result<()> {
    if let (Some(p_addr), Some(wrapper_va)) = (persistent_addr, persistent_wrapper_va) {
        let branch = tramp_gen
            .encode_branch(p_addr, wrapper_va)
            .with_context(|| format!("failed to encode persistent branch at 0x{:x}", p_addr))?;
        let fo = crate::pe::va_to_file_offset_pe(p_addr, ctx)
            .ok_or_else(|| anyhow::anyhow!("persistent_addr 0x{:x} has no file mapping", p_addr))?;
        if fo + branch.len() > data.len() {
            bail!("persistent_addr 0x{:x}: file offset out of range", p_addr);
        }
        data[fo..fo + branch.len()].copy_from_slice(&branch);
        for i in branch.len()..persistent_displaced_len {
            data[fo + i] = 0x90;
        }
    }
    Ok(())
}

/// Update PE headers: NumberOfSections, SizeOfImage, CheckSum, and optionally EntryPoint.
fn update_pe_headers(
    data: &mut [u8],
    ctx: &PeContext,
    new_section_rva: u64,
    section_virtual_size: u64,
    section_alignment: u64,
    need_init_code: bool,
    init_code: &crate::trampoline::InitCode,
) {
    let new_num_sections = ctx.number_of_sections + 1;
    let nsec_off = ctx.number_of_sections_field_offset as usize;
    data[nsec_off..nsec_off + 2].copy_from_slice(&new_num_sections.to_le_bytes());

    let new_size_of_image =
        (new_section_rva + align_up(section_virtual_size, section_alignment)) as u32;
    let soi_off = ctx.size_of_image_field_offset as usize;
    data[soi_off..soi_off + 4].copy_from_slice(&new_size_of_image.to_le_bytes());

    let cksum_off = ctx.checksum_field_offset as usize;
    data[cksum_off..cksum_off + 4].copy_from_slice(&0u32.to_le_bytes());

    if need_init_code {
        let new_entry_rva = (init_code.entry_va - ctx.image_base) as u32;
        let ep_off = ctx.entry_point_field_offset as usize;
        data[ep_off..ep_off + 4].copy_from_slice(&new_entry_rva.to_le_bytes());
    }
}

/// Orchestrate PE patching:
/// 1. Compute new section layout
/// 2. Generate PE init code + trampolines
/// 3. Insert `.trcov` section header
/// 4. Redirect entry point
/// 5. Patch basic blocks with JMP rel32
/// 6. Append section data
/// 7. Update PE headers
#[allow(clippy::too_many_arguments)]
fn patch_pe(
    ctx: &PeContext,
    blocks: &[BasicBlock],
    mut data: Vec<u8>,
    tramp_gen: &dyn TrampolineGenerator,
    resolved_hooks: &[ResolvedHook],
    no_coverage: bool,
    hook_library_path: Option<&String>,
    persistent_addr: Option<u64>,
    persistent_count: u32,
) -> Result<PatchResult> {
    if blocks.is_empty() && !no_coverage {
        bail!("no basic blocks to instrument");
    }

    // Sort blocks by VA and remove overlapping ones.
    let blocks = filter_and_sort_blocks(blocks);

    let section_alignment = ctx.section_alignment as u64;
    let file_alignment = ctx.file_alignment as u64;

    // Compute new section VA: aligned after highest existing VA.
    // Use RVA (relative to image base) for PE internal calculations.
    let highest_rva_end = ctx.highest_va_end - ctx.image_base;
    let new_section_rva = align_up(highest_rva_end, section_alignment);
    let new_section_va = ctx.image_base + new_section_rva;

    // Layout within the new section:
    // [data area: 16 bytes] [persistent_data: 200 bytes if persistent]
    // [init code] [trampoline_0] ... [trampoline_N]
    // [persistent_wrapper] [hook data slots] [toggle bytes] [return slots] [hook trampolines]
    let data_va = new_section_va;
    let use_persistent = persistent_addr.is_some() && !ctx.is_dll;
    let persistent_data_va = if use_persistent {
        Some(data_va + DATA_SIZE)
    } else {
        None
    };
    let persistent_area_size = if use_persistent {
        if ctx.is_64bit {
            PE_PERSISTENT_DATA_SIZE
        } else {
            PE32_PERSISTENT_DATA_SIZE
        }
    } else {
        0
    };
    let init_va = data_va + DATA_SIZE + persistent_area_size;

    // Generate init code (PE-specific: GetEnvironmentVariableA + OpenFileMappingA + MapViewOfFile).
    let init_code;
    let mut trampolines: Vec<(&BasicBlock, crate::trampoline::Trampoline)> = Vec::new();
    let mut blocks_skipped = 0;
    let mut current_va;

    let num_hook_slots = hooks::count_library_hooks(resolved_hooks);

    let has_lib_hooks = num_hook_slots > 0;
    // Library hooks need init code to run even in no-coverage mode,
    // because LoadLibraryA + GetProcAddress must be called to resolve handlers.
    let need_init_code = !no_coverage || has_lib_hooks;

    if !need_init_code {
        init_code = crate::trampoline::InitCode {
            va: init_va,
            code: Vec::new(),
            entry_va: init_va,
        };
        current_va = init_va;
    } else {
        // Two-pass init code generation when library hooks are present:
        // 1. Generate without hooks to measure trampoline layout
        // 2. Re-generate with hooks using the now-known hook_data_va
        //
        // Collect library hook handler info for building PeHookLibraryInfo.
        let handler_protos: Vec<(String, usize)> = resolved_hooks
            .iter()
            .filter_map(|h| {
                if let HookSource::LibrarySymbol {
                    ref name,
                    data_slot_index,
                } = h.source
                {
                    Some((name.clone(), data_slot_index))
                } else {
                    None
                }
            })
            .collect();
        let has_lib_hooks = !handler_protos.is_empty();

        // Dispatch 32-bit vs 64-bit init code generation.
        let gen_init = |va, dv, ep, hl: Option<&PeHookLibraryInfo>, pdv| {
            if ctx.is_64bit {
                generate_pe_init_code(va, dv, ep, hl, pdv)
            } else {
                generate_pe32_init_code(va, dv, ep, hl, pdv)
            }
        };

        // Pass 1: generate init code without hook resolution to get base size.
        let init_pass1 = gen_init(init_va, data_va, ctx.entry_point, None, persistent_data_va)
            .context("failed to generate PE init code")?;

        // Compute trampoline layout using pass-1 init code size.
        let mut trampolines_start_va = init_va + init_pass1.code.len() as u64;
        trampolines_start_va = align_up(trampolines_start_va, 16);

        // Helper: generate a trampoline, dispatching 32-bit vs 64-bit.
        let gen_tramp = |va: u64,
                         dv: u64,
                         block: &crate::disasm::BasicBlock|
         -> Result<crate::trampoline::Trampoline> {
            if ctx.is_64bit {
                tramp_gen.generate_trampoline(va, dv, block)
            } else {
                generate_trampoline_pe32(va, dv, block)
            }
        };

        trampolines.reserve(blocks.len());
        current_va = trampolines_start_va;

        for &block in &blocks {
            match gen_tramp(current_va, data_va, block) {
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

        // Now we know where hook data starts. If there are library hooks,
        // re-generate init code with hook resolution embedded.
        if has_lib_hooks {
            // hook_data_va is right after the last trampoline.
            let hook_data_va_est = current_va;
            let hook_lib_info = PeHookLibraryInfo {
                library_path: hook_library_path.cloned().unwrap_or_default(),
                handlers: handler_protos
                    .iter()
                    .map(|(name, slot_idx)| {
                        (name.clone(), hook_data_va_est + (*slot_idx as u64) * 8)
                    })
                    .collect(),
            };

            let init_pass2 = gen_init(
                init_va,
                data_va,
                ctx.entry_point,
                Some(&hook_lib_info),
                persistent_data_va,
            )
            .context("failed to generate PE init code with hook resolution")?;

            // If pass-2 init code is a different size, we need to re-layout trampolines.
            if init_pass2.code.len() != init_pass1.code.len() {
                trampolines.clear();
                blocks_skipped = 0;
                let mut tramp_start = init_va + init_pass2.code.len() as u64;
                tramp_start = align_up(tramp_start, 16);
                current_va = tramp_start;

                for &block in &blocks {
                    match gen_tramp(current_va, data_va, block) {
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

                // hook_data_va may have shifted — do a third pass if needed.
                let hook_data_va_final = current_va;
                if hook_data_va_final != hook_data_va_est {
                    let hook_lib_final = PeHookLibraryInfo {
                        library_path: hook_lib_info.library_path.clone(),
                        handlers: handler_protos
                            .iter()
                            .map(|(name, slot_idx)| {
                                (name.clone(), hook_data_va_final + (*slot_idx as u64) * 8)
                            })
                            .collect(),
                    };
                    let init_pass3 = gen_init(
                        init_va,
                        data_va,
                        ctx.entry_point,
                        Some(&hook_lib_final),
                        persistent_data_va,
                    )
                    .context("failed to generate PE init code (pass 3)")?;

                    // Size must match pass2 since hook info hasn't changed (only VAs in
                    // embedded data, which are fixed-size). If not, bail.
                    assert_eq!(
                        init_pass3.code.len(),
                        init_pass2.code.len(),
                        "PE init code size changed between pass 2 and 3 — layout bug",
                    );
                    init_code = init_pass3;
                } else {
                    init_code = init_pass2;
                }
            } else {
                init_code = init_pass2;
            }
        } else {
            init_code = init_pass1;
        }
    }

    // Persistent wrapper (pipe-based, no fork).
    let mut persistent_wrapper: Option<crate::trampoline::PersistentWrapper> = None;
    let mut persistent_wrapper_va: Option<u64> = None;
    let mut persistent_displaced_len: usize = 0;
    if use_persistent {
        let p_addr = persistent_addr.expect("persistent_addr required when use_persistent is true");
        let p_data_va =
            persistent_data_va.expect("persistent_data_va required when use_persistent is true");
        current_va = align_up(current_va, 16);
        let wrapper_va = current_va;
        persistent_wrapper_va = Some(wrapper_va);

        let (displaced, displaced_len) = crate::pe_disasm::extract_displaced_bytes_pe(
            &data, ctx, p_addr, 5,
        )
        .with_context(|| {
            format!(
                "failed to extract displaced bytes at persistent_addr 0x{:x}",
                p_addr,
            )
        })?;
        persistent_displaced_len = displaced_len;

        let wrapper = if ctx.is_64bit {
            generate_pe_persistent_wrapper(
                wrapper_va,
                p_data_va,
                data_va,
                p_addr,
                &displaced,
                displaced_len,
                persistent_count,
            )
            .context("failed to generate PE persistent wrapper")?
        } else {
            generate_pe32_persistent_wrapper(
                wrapper_va,
                p_data_va,
                data_va,
                p_addr,
                &displaced,
                displaced_len,
                persistent_count,
            )
            .context("failed to generate PE32 persistent wrapper")?
        };

        current_va += wrapper.code.len() as u64;
        current_va = align_up(current_va, 16);
        persistent_wrapper = Some(wrapper);
    }

    // Hook data slots.
    let ptr_size: u64 = if ctx.is_64bit { 8 } else { 4 };
    let hook_abi = if ctx.is_64bit {
        TargetAbi::Win64
    } else {
        TargetAbi::Win32
    };
    let hook_data_size = (num_hook_slots as u64) * ptr_size;
    let toggle_area_size = if resolved_hooks.is_empty() {
        0u64
    } else {
        ((resolved_hooks.len() as u64) + 7) & !7
    };
    let return_slot_count = hooks::count_return_hooks(resolved_hooks);
    let return_slot_area_size = (return_slot_count as u64) * ptr_size;

    let hook_data_va = current_va;
    current_va += hook_data_size;
    let toggle_data_va = current_va;
    current_va += toggle_area_size;
    let return_slot_va = current_va;
    current_va += return_slot_area_size;
    current_va = align_up(current_va, 16);

    // Generate hook trampolines.
    let mut hook_trampolines: Vec<(usize, crate::trampoline::Trampoline)> = Vec::new();
    let mut shellcode_blobs: Vec<(u64, Vec<u8>)> = Vec::new();

    // Place shellcode blobs first.
    let mut shellcode_va_map: Vec<Option<u64>> = Vec::with_capacity(resolved_hooks.len());
    for hook in resolved_hooks {
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
    for (i, hook) in resolved_hooks.iter().enumerate() {
        va_groups.entry(hook.target_va).or_default().push(i);
    }

    let mut return_trampolines: Vec<crate::trampoline::Trampoline> = Vec::new();

    for indices in va_groups.values() {
        if indices.len() == 1 {
            let i = indices[0];
            let hook = &resolved_hooks[i];
            let sc_va = shellcode_va_map[i];
            let toggle_va = Some(toggle_data_va + hook.toggle_index as u64);

            if hook.mode == crate::hooks::HookMode::Return {
                let entry_va = current_va;
                let estimated_entry_size = 256u64;
                let ret_tramp_va = align_up(entry_va + estimated_entry_size, 8);
                let slot_va = return_slot_va
                    + hook
                        .return_slot_index
                        .expect("Return mode hook must have return_slot_index")
                        as u64
                        * ptr_size;

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
                            "return hook entry trampoline exceeded estimate",
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
            // Chained hooks on same VA.
            let chain_hooks: Vec<&ResolvedHook> =
                indices.iter().map(|&i| &resolved_hooks[i]).collect();
            let chain_sc_vas: Vec<Option<u64>> =
                indices.iter().map(|&i| shellcode_va_map[i]).collect();
            let chain_toggle_vas: Vec<Option<u64>> = indices
                .iter()
                .map(|&i| Some(toggle_data_va + resolved_hooks[i].toggle_index as u64))
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
                    // Use first index for the chained trampoline.
                    hook_trampolines.push((indices[0], tramp));
                }
                Err(e) => {
                    tracing::warn!(
                        "chained hooks at 0x{:x}: trampoline generation failed: {}",
                        chain_hooks[0].target_va,
                        e,
                    );
                }
            }
        }
    }

    // Compute total section size.
    let section_virtual_size = current_va - new_section_va;
    let section_raw_size = align_up(section_virtual_size, file_alignment);

    // Check we have room for a new section header.
    // The section header table follows the optional header. We need 40 bytes.
    let section_table_end =
        ctx.section_table_offset as usize + (ctx.number_of_sections as usize) * SECTION_HEADER_SIZE;
    let headers_end = ctx.size_of_headers as usize;

    if section_table_end + SECTION_HEADER_SIZE > headers_end {
        bail!(
            "no room for new section header: table ends at {} + 40 > headers end at {}.\n\
             The PE header area is full. Try stripping debug sections.",
            section_table_end,
            headers_end,
        );
    }

    // Build section data.
    let section_data = build_section_data(
        section_raw_size,
        new_section_va,
        init_va,
        &init_code,
        &trampolines,
        persistent_data_va,
        persistent_wrapper,
        persistent_wrapper_va,
        toggle_data_va,
        resolved_hooks,
        &shellcode_blobs,
        &hook_trampolines,
        &return_trampolines,
    );

    // Patch basic blocks with JMP rel32 to trampolines.
    let blocks_instrumented = patch_coverage_blocks(&mut data, &trampolines, tramp_gen)?;

    // Patch hook targets with JMP rel32 to hook trampolines.
    let hooks_applied =
        patch_hook_targets(&mut data, resolved_hooks, &hook_trampolines, tramp_gen)?;

    // Patch persistent_addr with JMP rel32 to persistent wrapper.
    patch_persistent_addr(
        &mut data,
        ctx,
        persistent_addr,
        persistent_wrapper_va,
        persistent_displaced_len,
        tramp_gen,
    )?;

    // Compute file offset for new section data.
    // Must be after all existing raw data, aligned to FileAlignment.
    let existing_file_end = data.len() as u64;
    let new_section_raw_offset = align_up(existing_file_end, file_alignment);

    // Pad data to new section raw offset.
    data.resize(new_section_raw_offset as usize, 0);
    // Append section data.
    data.extend_from_slice(&section_data);

    // Write new section header.
    let header_offset = section_table_end;
    let mut hdr = [0u8; SECTION_HEADER_SIZE];
    // Name: ".trcov\0\0" (8 bytes)
    hdr[0..6].copy_from_slice(b".trcov");
    // VirtualSize
    hdr[8..12].copy_from_slice(&(section_virtual_size as u32).to_le_bytes());
    // VirtualAddress (RVA)
    hdr[12..16].copy_from_slice(&(new_section_rva as u32).to_le_bytes());
    // SizeOfRawData
    hdr[16..20].copy_from_slice(&(section_raw_size as u32).to_le_bytes());
    // PointerToRawData
    hdr[20..24].copy_from_slice(&(new_section_raw_offset as u32).to_le_bytes());
    // Characteristics: CODE | INITIALIZED_DATA | EXECUTE | READ | WRITE
    let chars = IMAGE_SCN_CNT_CODE
        | IMAGE_SCN_CNT_INITIALIZED_DATA
        | IMAGE_SCN_MEM_EXECUTE
        | IMAGE_SCN_MEM_READ
        | IMAGE_SCN_MEM_WRITE;
    hdr[36..40].copy_from_slice(&chars.to_le_bytes());

    data[header_offset..header_offset + SECTION_HEADER_SIZE].copy_from_slice(&hdr);

    // Update PE headers.
    update_pe_headers(
        &mut data,
        ctx,
        new_section_rva,
        section_virtual_size,
        section_alignment,
        need_init_code,
        &init_code,
    );

    let segment_size = section_raw_size;

    Ok(PatchResult {
        data,
        blocks_instrumented,
        blocks_skipped,
        segment_va: new_section_va,
        segment_size,
        heap_san_intercepted: 0,
        hooks_applied,
        hook_data_va,
        toggle_data_va,
    })
}

/// Emit x64 code to parse the decimal SHM ID from the wide-string environment
/// variable value and construct the SHM name "afl_shm_<id>" on the stack.
fn emit_pe64_shm_name_build(code: &mut Vec<u8>) {
    // ===== FOUND! Parse decimal SHM ID from wide string at rsi + 26 =====
    // lea rsi, [rsi + 26]  ; skip past "__AFL_SHM_ID=" (13 wchars = 26 bytes)
    code.extend_from_slice(&[0x48, 0x8D, 0x76, 0x1A]);
    // xor r14, r14  ; shmid accumulator = 0
    code.extend_from_slice(&[0x4D, 0x31, 0xF6]);

    // .parse_loop:
    let parse_loop = code.len();
    // movzx eax, word [rsi]  ; load wide char
    code.extend_from_slice(&[0x0F, 0xB7, 0x06]);
    // sub ax, '0'  (0x30)
    code.extend_from_slice(&[0x66, 0x2D, 0x30, 0x00]);
    // cmp ax, 9
    code.extend_from_slice(&[0x66, 0x3D, 0x09, 0x00]);
    // ja .parse_done
    let ja_parse_done_pos = code.len();
    code.extend_from_slice(&[0x77, 0x00]); // ja rel8 placeholder
    // imul r14, r14, 10
    code.extend_from_slice(&[0x4D, 0x6B, 0xF6, 0x0A]);
    // movzx eax, ax
    code.extend_from_slice(&[0x0F, 0xB7, 0xC0]);
    // add r14, rax
    code.extend_from_slice(&[0x49, 0x01, 0xC6]);
    // add rsi, 2  ; next wchar
    code.extend_from_slice(&[0x48, 0x83, 0xC6, 0x02]);
    // jmp .parse_loop
    let jmp_parse_rel = (parse_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_parse_rel as u8]);

    // .parse_done:
    let parse_done = code.len();
    code[ja_parse_done_pos + 1] = (parse_done - (ja_parse_done_pos + 2)) as u8;

    // ============================================================
    // Step 3: Resolve kernel32.dll functions via PEB_LDR_DATA
    // Then call OpenFileMappingA and MapViewOfFile.
    //
    // Walk PEB.Ldr.InLoadOrderModuleList to find kernel32.dll.
    // Parse its export table to find OpenFileMappingA and MapViewOfFile.
    // ============================================================

    // First, build the SHM name "afl_shm_<id>" on the stack.
    // The ID is in r14. We need to convert it to a decimal ASCII string.
    // lea rdi, [rsp + 32]  ; buffer for name (after shadow space)
    code.extend_from_slice(&[0x48, 0x8D, 0x7C, 0x24, 0x20]);

    // Write "afl_shm_" prefix (8 bytes)
    // mov rax, "afl_shm_" as u64
    let prefix = u64::from_le_bytes(*b"afl_shm_");
    code.extend_from_slice(&[0x48, 0xB8]); // mov rax, imm64
    code.extend_from_slice(&prefix.to_le_bytes());
    // mov [rdi], rax
    code.extend_from_slice(&[0x48, 0x89, 0x07]);
    // lea rdi, [rdi + 8]
    code.extend_from_slice(&[0x48, 0x8D, 0x7F, 0x08]);

    // Convert r14 (SHM ID) to decimal ASCII at [rdi].
    // We'll use a simple div-10 loop, writing digits in reverse then reversing.
    // mov rax, r14
    code.extend_from_slice(&[0x4C, 0x89, 0xF0]);
    // mov rbx, rdi  ; save start of digits
    code.extend_from_slice(&[0x48, 0x89, 0xFB]);
    // test rax, rax
    code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    // jnz .conv_loop
    let jnz_conv_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jnz rel8 placeholder
    // Zero case: write '0'
    code.extend_from_slice(&[0xC6, 0x07, 0x30]); // mov byte [rdi], '0'
    code.extend_from_slice(&[0x48, 0xFF, 0xC7]); // inc rdi
    // jmp .conv_done
    let jmp_conv_done_pos = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp rel8 placeholder

    // .conv_loop:
    let conv_loop = code.len();
    code[jnz_conv_pos + 1] = (conv_loop - (jnz_conv_pos + 2)) as u8;
    // xor edx, edx
    code.extend_from_slice(&[0x31, 0xD2]);
    // mov rcx, 10
    code.extend_from_slice(&[0x48, 0xC7, 0xC1, 0x0A, 0x00, 0x00, 0x00]);
    // div rcx  ; rax = quotient, rdx = remainder
    code.extend_from_slice(&[0x48, 0xF7, 0xF1]);
    // add dl, '0'
    code.extend_from_slice(&[0x80, 0xC2, 0x30]);
    // mov [rdi], dl
    code.extend_from_slice(&[0x88, 0x17]);
    // inc rdi
    code.extend_from_slice(&[0x48, 0xFF, 0xC7]);
    // test rax, rax
    code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    // jnz .conv_loop
    let jnz_conv2_rel = (conv_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0x75, jnz_conv2_rel as u8]);

    // Now reverse the digits at [rbx..rdi).
    // mov byte [rdi], 0  ; null-terminate
    code.extend_from_slice(&[0xC6, 0x07, 0x00]);
    // dec rdi  ; rdi points to last digit
    code.extend_from_slice(&[0x48, 0xFF, 0xCF]);

    // .reverse_loop:
    let reverse_loop = code.len();
    // cmp rbx, rdi
    code.extend_from_slice(&[0x48, 0x39, 0xFB]);
    // jge .conv_done
    let jge_conv_done_pos = code.len();
    code.extend_from_slice(&[0x7D, 0x00]); // jge rel8 placeholder
    // mov al, [rbx]
    code.extend_from_slice(&[0x8A, 0x03]);
    // mov cl, [rdi]
    code.extend_from_slice(&[0x8A, 0x0F]);
    // mov [rbx], cl
    code.extend_from_slice(&[0x88, 0x0B]);
    // mov [rdi], al
    code.extend_from_slice(&[0x88, 0x07]);
    // inc rbx
    code.extend_from_slice(&[0x48, 0xFF, 0xC3]);
    // dec rdi
    code.extend_from_slice(&[0x48, 0xFF, 0xCF]);
    // jmp .reverse_loop
    let jmp_rev_rel = (reverse_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_rev_rel as u8]);

    // .conv_done:
    let conv_done = code.len();
    code[jmp_conv_done_pos + 1] = (conv_done - (jmp_conv_done_pos + 2)) as u8;
    code[jge_conv_done_pos + 1] = (conv_done - (jge_conv_done_pos + 2)) as u8;
}

/// Walk x64 PEB_LDR_DATA InLoadOrderModuleList to find kernel32.dll.
/// On success, r15 = kernel32 base address.
/// Returns the fixup position for the "kernel32 not found" jump to epilogue.
fn emit_pe64_kernel32_walk(code: &mut Vec<u8>) -> usize {
    // ============================================================
    // Step 4: Walk PEB_LDR_DATA to find kernel32.dll
    // PEB at gs:[0x60]
    // PEB.Ldr at PEB+0x18
    // PEB_LDR_DATA.InLoadOrderModuleList at Ldr+0x10
    // Each entry: LDR_DATA_TABLE_ENTRY
    //   +0x00: InLoadOrderLinks (LIST_ENTRY: Flink, Blink)
    //   +0x30: DllBase
    //   +0x58: BaseDllName (UNICODE_STRING: Length, MaxLength, Buffer)
    // ============================================================

    // mov rax, gs:[0x60]  ; PEB
    code.extend_from_slice(&[0x65, 0x48, 0x8B, 0x04, 0x25, 0x60, 0x00, 0x00, 0x00]);
    // mov rax, [rax + 0x18]  ; PEB.Ldr
    code.extend_from_slice(&[0x48, 0x8B, 0x40, 0x18]);
    // lea rbx, [rax + 0x10]  ; &InLoadOrderModuleList (list head)
    code.extend_from_slice(&[0x48, 0x8D, 0x58, 0x10]);
    // mov rdi, [rbx]  ; first entry (Flink)
    code.extend_from_slice(&[0x48, 0x8B, 0x3B]);

    // .ldr_loop:
    let ldr_loop = code.len();
    // cmp rdi, rbx  ; back to head?
    code.extend_from_slice(&[0x48, 0x39, 0xDF]);
    // je .epilogue  ; kernel32 not found
    let je_epilogue_k32_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // je rel32 placeholder

    // Check BaseDllName: LDR_DATA_TABLE_ENTRY.BaseDllName at offset 0x58.
    // UNICODE_STRING.Length at +0x00 (u16), Buffer at +0x08 (ptr).
    // movzx ecx, word [rdi + 0x58]  ; BaseDllName.Length (in bytes)
    code.extend_from_slice(&[0x0F, 0xB7, 0x4F, 0x58]);
    // cmp ecx, 24  ; "KERNEL32.DLL" = 12 wchars = 24 bytes (case-insensitive check below)
    // We also accept "kernel32.dll" (lowercase). Length must be 24.
    code.extend_from_slice(&[0x83, 0xF9, 0x18]);
    // jne .ldr_next
    let jne_ldr_next1_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // mov rsi, [rdi + 0x60]  ; BaseDllName.Buffer
    code.extend_from_slice(&[0x48, 0x8B, 0x77, 0x60]);

    // Case-insensitive compare of first 6 wchars ("kernel" or "KERNEL"):
    // We'll just check that the first wchar ORed with 0x20 == 'k' (0x6B)
    // and the 7th wchar is '3' (0x33).
    // movzx eax, word [rsi]  ; first wchar
    code.extend_from_slice(&[0x0F, 0xB7, 0x06]);
    // or al, 0x20  ; to lowercase
    code.extend_from_slice(&[0x0C, 0x20]);
    // cmp al, 'k'
    code.extend_from_slice(&[0x3C, 0x6B]);
    // jne .ldr_next
    let jne_ldr_next2_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // Check '3' at wchar index 6 (byte offset 12)
    // movzx eax, word [rsi + 12]
    code.extend_from_slice(&[0x0F, 0xB7, 0x46, 0x0C]);
    // cmp al, '3'
    code.extend_from_slice(&[0x3C, 0x33]);
    // jne .ldr_next
    let jne_ldr_next3_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // Check '2' at wchar index 7 (byte offset 14)
    // movzx eax, word [rsi + 14]
    code.extend_from_slice(&[0x0F, 0xB7, 0x46, 0x0E]);
    // cmp al, '2'
    code.extend_from_slice(&[0x3C, 0x32]);
    // jne .ldr_next
    let jne_ldr_next4_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // Found kernel32.dll! DllBase is at [rdi + 0x30].
    // mov r15, [rdi + 0x30]  ; r15 = kernel32 base
    code.extend_from_slice(&[0x4C, 0x8B, 0x7F, 0x30]);
    // jmp .resolve_exports
    let jmp_resolve_pos = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp rel8 placeholder

    // .ldr_next:
    let ldr_next = code.len();
    code[jne_ldr_next1_pos + 1] = (ldr_next - (jne_ldr_next1_pos + 2)) as u8;
    code[jne_ldr_next2_pos + 1] = (ldr_next - (jne_ldr_next2_pos + 2)) as u8;
    code[jne_ldr_next3_pos + 1] = (ldr_next - (jne_ldr_next3_pos + 2)) as u8;
    code[jne_ldr_next4_pos + 1] = (ldr_next - (jne_ldr_next4_pos + 2)) as u8;
    // mov rdi, [rdi]  ; Flink (next entry)
    code.extend_from_slice(&[0x48, 0x8B, 0x3F]);
    // jmp .ldr_loop
    let jmp_ldr_rel = (ldr_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_ldr_rel as u8]);

    // .resolve_exports:
    let resolve_exports = code.len();
    code[jmp_resolve_pos + 1] = (resolve_exports - (jmp_resolve_pos + 2)) as u8;

    je_epilogue_k32_pos
}

/// Resolve ReadFile, WriteFile, ExitProcess for PE x64 persistent mode.
/// Re-walks kernel32 exports using the base saved at [rbp - 0x68].
fn emit_pe64_persistent_resolve(code: &mut Vec<u8>, init_va: u64, p_data_va: u64) {
    let hash_read_file: u32 = djb2_hash(b"ReadFile");
    let hash_write_file: u32 = djb2_hash(b"WriteFile");
    let hash_exit_process: u32 = djb2_hash(b"ExitProcess");

    // Load kernel32 base from [rbp - 0x68].
    // mov r15, [rbp - 0x68]   ; 4C 8B 7D 98
    code.extend_from_slice(&[0x4C, 0x8B, 0x7D, 0x98]);

    // Parse export directory (same sequence as Step 5).
    // mov eax, [r15 + 0x3C]  ; e_lfanew
    code.extend_from_slice(&[0x41, 0x8B, 0x47, 0x3C]);
    // lea rbx, [r15 + rax]  ; PE header
    code.extend_from_slice(&[0x49, 0x8D, 0x1C, 0x07]);
    // mov eax, [rbx + 0x88]  ; Export directory RVA
    code.extend_from_slice(&[0x8B, 0x83, 0x88, 0x00, 0x00, 0x00]);
    // test eax, eax
    code.extend_from_slice(&[0x85, 0xC0]);
    // jz .skip_persistent_resolve
    let jz_skip_persist_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);

    // lea rsi, [r15 + rax]  ; export directory
    code.extend_from_slice(&[0x49, 0x8D, 0x34, 0x07]);

    // mov ecx, [rsi + 0x18]  ; NumberOfNames
    code.extend_from_slice(&[0x8B, 0x4E, 0x18]);
    // mov eax, [rsi + 0x20]  ; AddressOfNames RVA
    code.extend_from_slice(&[0x8B, 0x46, 0x20]);
    // lea r13, [r15 + rax]  ; AddressOfNames
    // REX.WRB (0x4D): W=64-bit, R=extend reg to r13, B=extend rm base to r15
    code.extend_from_slice(&[0x4D, 0x8D, 0x2C, 0x07]);
    // mov eax, [rsi + 0x24]  ; AddressOfNameOrdinals RVA
    code.extend_from_slice(&[0x8B, 0x46, 0x24]);
    // lea r14, [r15 + rax]  ; AddressOfNameOrdinals
    code.extend_from_slice(&[0x4D, 0x8D, 0x34, 0x07]);
    // mov eax, [rsi + 0x1C]  ; AddressOfFunctions RVA
    code.extend_from_slice(&[0x8B, 0x46, 0x1C]);
    // push rax  ; save AddressOfFunctions RVA
    code.push(0x50);
    // push r15  ; save kernel32 base
    code.extend_from_slice(&[0x41, 0x57]);

    // Use stack locals to track resolved pointers:
    // [rbp - 0x70] = ReadFile, [rbp - 0x78] = WriteFile, [rbp - 0x80] = ExitProcess
    // Init to 0.
    code.extend_from_slice(&[0x48, 0xC7, 0x45, 0x90, 0x00, 0x00, 0x00, 0x00]); // mov qword [rbp-0x70], 0
    code.extend_from_slice(&[0x48, 0xC7, 0x45, 0x88, 0x00, 0x00, 0x00, 0x00]); // mov qword [rbp-0x78], 0
    code.extend_from_slice(&[0x48, 0xC7, 0x45, 0x80, 0x00, 0x00, 0x00, 0x00]); // mov qword [rbp-0x80], 0

    // xor edx, edx  ; index = 0
    code.extend_from_slice(&[0x31, 0xD2]);

    // .persist_export_loop:
    let persist_export_loop = code.len();
    // cmp edx, ecx
    code.extend_from_slice(&[0x39, 0xCA]);
    // jge .persist_exports_done
    let jge_persist_done_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x8D, 0x00, 0x00, 0x00, 0x00]);

    // mov eax, [r13 + rdx*4]  ; name RVA
    code.extend_from_slice(&[0x41, 0x8B, 0x44, 0x95, 0x00]);
    // lea rdi, [r15 + rax]  ; name string
    code.extend_from_slice(&[0x49, 0x8D, 0x3C, 0x07]);

    // DJB2 hash of name.
    // push rdx
    code.push(0x52);
    // push rcx
    code.push(0x51);
    // mov eax, 5381
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&5381u32.to_le_bytes());

    // .persist_hash_loop:
    let persist_hash_loop = code.len();
    // movzx ebx, byte [rdi]
    code.extend_from_slice(&[0x0F, 0xB6, 0x1F]);
    // test bl, bl
    code.extend_from_slice(&[0x84, 0xDB]);
    // jz .persist_hash_done
    let jz_persist_hash_done = code.len();
    code.extend_from_slice(&[0x74, 0x00]);
    // imul eax, eax, 33
    code.extend_from_slice(&[0x6B, 0xC0, 0x21]);
    // add eax, ebx
    code.extend_from_slice(&[0x01, 0xD8]);
    // inc rdi
    code.extend_from_slice(&[0x48, 0xFF, 0xC7]);
    // jmp .persist_hash_loop
    let jmp_hash_rel = (persist_hash_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_hash_rel as u8]);

    // .persist_hash_done:
    let persist_hash_done = code.len();
    code[jz_persist_hash_done + 1] = (persist_hash_done - (jz_persist_hash_done + 2)) as u8;

    // pop rcx
    code.push(0x59);
    // pop rdx
    code.push(0x5A);

    // Compare hash with ReadFile.
    code.extend_from_slice(&[0x3D]); // cmp eax, hash_read_file
    code.extend_from_slice(&hash_read_file.to_le_bytes());
    let jne_check_wf_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne .check_write_file

    // Resolve ReadFile → [rbp-0x70]
    code.extend_from_slice(&[0x41, 0x0F, 0xB7, 0x04, 0x56]); // movzx eax, word [r14+rdx*2]
    code.extend_from_slice(&[0x8B, 0x5C, 0x24, 0x08]); // mov ebx, [rsp+8] ; AddrOfFunctions RVA
    code.extend_from_slice(&[0x49, 0x8D, 0x1C, 0x1F]); // lea rbx, [r15+rbx]
    code.extend_from_slice(&[0x8B, 0x04, 0x83]); // mov eax, [rbx+rax*4]
    code.extend_from_slice(&[0x49, 0x8D, 0x04, 0x07]); // lea rax, [r15+rax]
    code.extend_from_slice(&[0x48, 0x89, 0x45, 0x90]); // mov [rbp-0x70], rax
    let jmp_persist_next1 = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp .persist_export_next

    // .check_write_file:
    let check_wf = code.len();
    code[jne_check_wf_pos + 1] = (check_wf - (jne_check_wf_pos + 2)) as u8;

    code.extend_from_slice(&[0x3D]); // cmp eax, hash_write_file
    code.extend_from_slice(&hash_write_file.to_le_bytes());
    let jne_check_ep_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne .check_exit_process

    // Resolve WriteFile → [rbp-0x78]
    code.extend_from_slice(&[0x41, 0x0F, 0xB7, 0x04, 0x56]);
    code.extend_from_slice(&[0x8B, 0x5C, 0x24, 0x08]);
    code.extend_from_slice(&[0x49, 0x8D, 0x1C, 0x1F]);
    code.extend_from_slice(&[0x8B, 0x04, 0x83]);
    code.extend_from_slice(&[0x49, 0x8D, 0x04, 0x07]);
    code.extend_from_slice(&[0x48, 0x89, 0x45, 0x88]); // mov [rbp-0x78], rax
    let jmp_persist_next2 = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp .persist_export_next

    // .check_exit_process:
    let check_ep = code.len();
    code[jne_check_ep_pos + 1] = (check_ep - (jne_check_ep_pos + 2)) as u8;

    code.extend_from_slice(&[0x3D]); // cmp eax, hash_exit_process
    code.extend_from_slice(&hash_exit_process.to_le_bytes());
    let jne_persist_next_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne .persist_export_next

    // Resolve ExitProcess → [rbp-0x80]
    code.extend_from_slice(&[0x41, 0x0F, 0xB7, 0x04, 0x56]);
    code.extend_from_slice(&[0x8B, 0x5C, 0x24, 0x08]);
    code.extend_from_slice(&[0x49, 0x8D, 0x1C, 0x1F]);
    code.extend_from_slice(&[0x8B, 0x04, 0x83]);
    code.extend_from_slice(&[0x49, 0x8D, 0x04, 0x07]);
    code.extend_from_slice(&[0x48, 0x89, 0x45, 0x80]); // mov [rbp-0x80], rax

    // .persist_export_next:
    let persist_export_next = code.len();
    code[jmp_persist_next1 + 1] = (persist_export_next - (jmp_persist_next1 + 2)) as u8;
    code[jmp_persist_next2 + 1] = (persist_export_next - (jmp_persist_next2 + 2)) as u8;
    code[jne_persist_next_pos + 1] = (persist_export_next - (jne_persist_next_pos + 2)) as u8;

    // Early exit: check if all 3 found.
    // cmp qword [rbp-0x70], 0
    code.extend_from_slice(&[0x48, 0x83, 0x7D, 0x90, 0x00]);
    let jz_persist_cont1 = code.len();
    code.extend_from_slice(&[0x74, 0x00]); // jz .persist_continue
    // cmp qword [rbp-0x78], 0
    code.extend_from_slice(&[0x48, 0x83, 0x7D, 0x88, 0x00]);
    let jz_persist_cont2 = code.len();
    code.extend_from_slice(&[0x74, 0x00]); // jz .persist_continue
    // cmp qword [rbp-0x80], 0
    code.extend_from_slice(&[0x48, 0x83, 0x7D, 0x80, 0x00]);
    // jnz .persist_exports_done (all 3 found!)
    let jnz_persist_all_found = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);

    // .persist_continue:
    let persist_continue = code.len();
    code[jz_persist_cont1 + 1] = (persist_continue - (jz_persist_cont1 + 2)) as u8;
    code[jz_persist_cont2 + 1] = (persist_continue - (jz_persist_cont2 + 2)) as u8;

    // inc edx
    code.extend_from_slice(&[0xFF, 0xC2]);
    // jmp .persist_export_loop
    let jmp_loop_rel = (persist_export_loop as i32) - (code.len() as i32 + 5);
    code.push(0xE9);
    code.extend_from_slice(&jmp_loop_rel.to_le_bytes());

    // .persist_exports_done:
    let persist_exports_done = code.len();
    let rel1 = (persist_exports_done as i32) - (jge_persist_done_pos as i32 + 6);
    code[jge_persist_done_pos + 2..jge_persist_done_pos + 6].copy_from_slice(&rel1.to_le_bytes());
    let rel2 = (persist_exports_done as i32) - (jnz_persist_all_found as i32 + 6);
    code[jnz_persist_all_found + 2..jnz_persist_all_found + 6].copy_from_slice(&rel2.to_le_bytes());

    // Pop r15, rax (kernel32 base, AddressOfFunctions).
    code.extend_from_slice(&[0x41, 0x5F]); // pop r15
    code.push(0x58); // pop rax

    // Store resolved pointers to persistent data area using RIP-relative mov.
    // ReadFile → persistent_data_va + 152
    // mov rax, [rbp-0x70]
    code.extend_from_slice(&[0x48, 0x8B, 0x45, 0x90]);
    // mov [rip+disp32], rax   ; 7 bytes
    let ip_after = init_va + code.len() as u64 + 7;
    let disp = (p_data_va as i64 + 152 - ip_after as i64) as i32;
    code.extend_from_slice(&[0x48, 0x89, 0x05]);
    code.extend_from_slice(&disp.to_le_bytes());

    // WriteFile → persistent_data_va + 160
    code.extend_from_slice(&[0x48, 0x8B, 0x45, 0x88]); // mov rax, [rbp-0x78]
    let ip_after = init_va + code.len() as u64 + 7;
    let disp = (p_data_va as i64 + 160 - ip_after as i64) as i32;
    code.extend_from_slice(&[0x48, 0x89, 0x05]);
    code.extend_from_slice(&disp.to_le_bytes());

    // ExitProcess → persistent_data_va + 168
    code.extend_from_slice(&[0x48, 0x8B, 0x45, 0x80]); // mov rax, [rbp-0x80]
    let ip_after = init_va + code.len() as u64 + 7;
    let disp = (p_data_va as i64 + 168 - ip_after as i64) as i32;
    code.extend_from_slice(&[0x48, 0x89, 0x05]);
    code.extend_from_slice(&disp.to_le_bytes());

    // .skip_persistent_resolve:
    let skip_persist = code.len();
    let skip_rel = (skip_persist as i32) - (jz_skip_persist_pos as i32 + 6);
    code[jz_skip_persist_pos + 2..jz_skip_persist_pos + 6].copy_from_slice(&skip_rel.to_le_bytes());
}

/// Emit the x64 hook-only resolution block: fresh kernel32 walk for
/// LoadLibraryA + GetProcAddress when SHM setup was skipped.
/// Returns (hook_resolve_label, fixup_positions_to_real_epilogue).
fn emit_pe64_hook_only_resolve(
    code: &mut Vec<u8>,
    init_va: u64,
    hook_lib: &PeHookLibraryInfo,
) -> (usize, Vec<usize>) {
    let mut hr_fixup_to_real_epilogue: Vec<usize> = Vec::new();

    let label = code.len();

    // --- Walk PEB_LDR_DATA to find kernel32 (same algo as Step 4) ---
    code.extend_from_slice(&[0x65, 0x48, 0x8B, 0x04, 0x25, 0x60, 0x00, 0x00, 0x00]); // mov rax, gs:[0x60]
    code.extend_from_slice(&[0x48, 0x8B, 0x40, 0x18]); // mov rax, [rax+0x18]
    code.extend_from_slice(&[0x48, 0x8D, 0x58, 0x10]); // lea rbx, [rax+0x10]
    code.extend_from_slice(&[0x48, 0x8B, 0x3B]); // mov rdi, [rbx]

    let hr_ldr_loop = code.len();
    code.extend_from_slice(&[0x48, 0x39, 0xDF]); // cmp rdi, rbx
    let hr_je_no_k32 = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // je .real_epilogue
    hr_fixup_to_real_epilogue.push(hr_je_no_k32);

    code.extend_from_slice(&[0x0F, 0xB7, 0x4F, 0x58]); // movzx ecx, word [rdi+0x58]
    code.extend_from_slice(&[0x83, 0xF9, 0x18]); // cmp ecx, 24
    let hr_jne1 = code.len();
    code.extend_from_slice(&[0x75, 0x00]);
    code.extend_from_slice(&[0x48, 0x8B, 0x77, 0x60]); // mov rsi, [rdi+0x60]
    code.extend_from_slice(&[0x0F, 0xB7, 0x06]); // movzx eax, word [rsi]
    code.extend_from_slice(&[0x0C, 0x20]); // or al, 0x20
    code.extend_from_slice(&[0x3C, 0x6B]); // cmp al, 'k'
    let hr_jne2 = code.len();
    code.extend_from_slice(&[0x75, 0x00]);
    code.extend_from_slice(&[0x0F, 0xB7, 0x46, 0x0C]); // movzx eax, word [rsi+12]
    code.extend_from_slice(&[0x3C, 0x33]); // cmp al, '3'
    let hr_jne3 = code.len();
    code.extend_from_slice(&[0x75, 0x00]);
    code.extend_from_slice(&[0x0F, 0xB7, 0x46, 0x0E]); // movzx eax, word [rsi+14]
    code.extend_from_slice(&[0x3C, 0x32]); // cmp al, '2'
    let hr_jne4 = code.len();
    code.extend_from_slice(&[0x75, 0x00]);

    // Found! mov r15, [rdi+0x30]
    code.extend_from_slice(&[0x4C, 0x8B, 0x7F, 0x30]);
    let hr_jmp_found = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp .hr_found

    // .hr_ldr_next:
    let hr_ldr_next = code.len();
    code[hr_jne1 + 1] = (hr_ldr_next - (hr_jne1 + 2)) as u8;
    code[hr_jne2 + 1] = (hr_ldr_next - (hr_jne2 + 2)) as u8;
    code[hr_jne3 + 1] = (hr_ldr_next - (hr_jne3 + 2)) as u8;
    code[hr_jne4 + 1] = (hr_ldr_next - (hr_jne4 + 2)) as u8;
    code.extend_from_slice(&[0x48, 0x8B, 0x3F]); // mov rdi, [rdi]
    let rel = (hr_ldr_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, rel as u8]);

    // .hr_found: parse kernel32 exports for LoadLibraryA + GetProcAddress
    let hr_found = code.len();
    code[hr_jmp_found + 1] = (hr_found - (hr_jmp_found + 2)) as u8;

    code.extend_from_slice(&[0x41, 0x8B, 0x47, 0x3C]); // mov eax, [r15+0x3C]
    code.extend_from_slice(&[0x49, 0x8D, 0x1C, 0x07]); // lea rbx, [r15+rax]
    code.extend_from_slice(&[0x8B, 0x83, 0x88, 0x00, 0x00, 0x00]); // mov eax, [rbx+0x88]
    code.extend_from_slice(&[0x85, 0xC0]); // test eax, eax
    let hr_jz_noexp = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz .real_epilogue
    hr_fixup_to_real_epilogue.push(hr_jz_noexp);

    code.extend_from_slice(&[0x49, 0x8D, 0x34, 0x07]); // lea rsi, [r15+rax]
    code.extend_from_slice(&[0x8B, 0x4E, 0x18]); // mov ecx, [rsi+0x18] ; NumberOfNames
    code.extend_from_slice(&[0x8B, 0x46, 0x20]); // mov eax, [rsi+0x20] ; AddressOfNames RVA
    code.extend_from_slice(&[0x4D, 0x8D, 0x2C, 0x07]); // lea r13, [r15+rax]
    code.extend_from_slice(&[0x8B, 0x46, 0x24]); // mov eax, [rsi+0x24] ; Ordinals RVA
    code.extend_from_slice(&[0x4D, 0x8D, 0x34, 0x07]); // lea r14, [r15+rax]
    code.extend_from_slice(&[0x8B, 0x46, 0x1C]); // mov eax, [rsi+0x1C] ; AddressOfFunctions RVA
    // lea rsi, [r15+rax] — reuse rsi for AddressOfFunctions base
    code.extend_from_slice(&[0x49, 0x8D, 0x34, 0x07]);

    // Clear LoadLibraryA / GetProcAddress slots
    code.extend_from_slice(&[0x48, 0xC7, 0x45, 0xA8, 0x00, 0x00, 0x00, 0x00]); // [rbp-0x58]=0
    code.extend_from_slice(&[0x48, 0xC7, 0x45, 0xA0, 0x00, 0x00, 0x00, 0x00]); // [rbp-0x60]=0

    let hash_lla = djb2_hash(b"LoadLibraryA");
    let hash_gpa = djb2_hash(b"GetProcAddress");

    code.extend_from_slice(&[0x31, 0xD2]); // xor edx, edx

    let hr_exp_loop = code.len();
    code.extend_from_slice(&[0x39, 0xCA]); // cmp edx, ecx
    let hr_jge_exp_done = code.len();
    code.extend_from_slice(&[0x0F, 0x8D, 0x00, 0x00, 0x00, 0x00]); // jge .hr_exp_done

    code.extend_from_slice(&[0x41, 0x8B, 0x44, 0x95, 0x00]); // mov eax, [r13+rdx*4]
    code.extend_from_slice(&[0x49, 0x8D, 0x3C, 0x07]); // lea rdi, [r15+rax]

    // DJB2 hash
    code.push(0x52); // push rdx
    code.push(0x51); // push rcx
    code.push(0x56); // push rsi (save AddressOfFunctions base)
    code.extend_from_slice(&[0xB8, 0x05, 0x15, 0x00, 0x00]); // mov eax, 5381
    let hr_hl = code.len();
    code.extend_from_slice(&[0x0F, 0xB6, 0x1F]); // movzx ebx, byte [rdi]
    code.extend_from_slice(&[0x84, 0xDB]); // test bl, bl
    let hr_jz_hd = code.len();
    code.extend_from_slice(&[0x74, 0x00]);
    code.extend_from_slice(&[0x6B, 0xC0, 0x21]); // imul eax, eax, 33
    code.extend_from_slice(&[0x01, 0xD8]); // add eax, ebx
    code.extend_from_slice(&[0x48, 0xFF, 0xC7]); // inc rdi
    let r = (hr_hl as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, r as u8]);
    let hr_hd = code.len();
    code[hr_jz_hd + 1] = (hr_hd - (hr_jz_hd + 2)) as u8;
    code.push(0x5E); // pop rsi
    code.push(0x59); // pop rcx
    code.push(0x5A); // pop rdx

    // Check LoadLibraryA
    code.extend_from_slice(&[0x3D]);
    code.extend_from_slice(&hash_lla.to_le_bytes());
    let hr_jne_lla = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne .hr_check_gpa

    // Resolve: ordinal → function VA
    code.extend_from_slice(&[0x41, 0x0F, 0xB7, 0x04, 0x56]); // movzx eax, word [r14+rdx*2]
    code.extend_from_slice(&[0x8B, 0x04, 0x86]); // mov eax, [rsi+rax*4]
    // Replace the preceding mov with lea rdi, [r15+rax] to compute
    // function VA = kernel32 base + export RVA without clobbering r15.
    code.truncate(code.len() - 3);
    // lea rdi, [r15+rax]
    code.extend_from_slice(&[0x49, 0x8D, 0x3C, 0x07]);
    // mov [rbp-0x58], rdi
    code.extend_from_slice(&[0x48, 0x89, 0x7D, 0xA8]);
    let hr_jmp_next1 = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp .hr_exp_next

    // .hr_check_gpa:
    let hr_check_gpa = code.len();
    code[hr_jne_lla + 1] = (hr_check_gpa - (hr_jne_lla + 2)) as u8;

    code.extend_from_slice(&[0x3D]);
    code.extend_from_slice(&hash_gpa.to_le_bytes());
    let hr_jne_gpa = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne .hr_exp_next

    code.extend_from_slice(&[0x41, 0x0F, 0xB7, 0x04, 0x56]); // movzx eax, word [r14+rdx*2]
    code.extend_from_slice(&[0x8B, 0x04, 0x86]); // mov eax, [rsi+rax*4]
    code.extend_from_slice(&[0x49, 0x8D, 0x3C, 0x07]); // lea rdi, [r15+rax]
    code.extend_from_slice(&[0x48, 0x89, 0x7D, 0xA0]); // mov [rbp-0x60], rdi

    // .hr_exp_next:
    let hr_exp_next = code.len();
    code[hr_jmp_next1 + 1] = (hr_exp_next - (hr_jmp_next1 + 2)) as u8;
    code[hr_jne_gpa + 1] = (hr_exp_next - (hr_jne_gpa + 2)) as u8;

    // Check if both found
    code.extend_from_slice(&[0x48, 0x83, 0x7D, 0xA8, 0x00]); // cmp qword [rbp-0x58], 0
    let hr_jz_cont = code.len();
    code.extend_from_slice(&[0x74, 0x00]); // jz .hr_exp_inc
    code.extend_from_slice(&[0x48, 0x83, 0x7D, 0xA0, 0x00]); // cmp qword [rbp-0x60], 0
    let hr_jnz_both = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jnz .hr_exp_done (both found!)

    // .hr_exp_inc:
    let hr_exp_inc = code.len();
    code[hr_jz_cont + 1] = (hr_exp_inc - (hr_jz_cont + 2)) as u8;
    code.extend_from_slice(&[0xFF, 0xC2]); // inc edx
    let r2 = (hr_exp_loop as i32) - (code.len() as i32 + 5);
    code.push(0xE9);
    code.extend_from_slice(&r2.to_le_bytes());

    // .hr_exp_done:
    let hr_exp_done = code.len();
    let r3 = (hr_exp_done as i32) - (hr_jge_exp_done as i32 + 6);
    code[hr_jge_exp_done + 2..hr_jge_exp_done + 6].copy_from_slice(&r3.to_le_bytes());
    let r4 = (hr_exp_done as i32) - (hr_jnz_both as i32 + 6);
    code[hr_jnz_both + 2..hr_jnz_both + 6].copy_from_slice(&r4.to_le_bytes());

    // --- Now call LoadLibraryA + GetProcAddress for hook handlers ---
    // (Same logic as the main path at lines ~1831-1905)

    // Check both were found
    code.extend_from_slice(&[0x48, 0x83, 0x7D, 0xA8, 0x00]); // cmp [rbp-0x58], 0
    let hr_jz_nolib = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz .real_epilogue
    hr_fixup_to_real_epilogue.push(hr_jz_nolib);

    code.extend_from_slice(&[0x48, 0x83, 0x7D, 0xA0, 0x00]); // cmp [rbp-0x60], 0
    let hr_jz_nogpa = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    hr_fixup_to_real_epilogue.push(hr_jz_nogpa);

    // lea rcx, [rip + library_path] — need to embed the library path string
    // We'll embed it after this code and fix up the LEA.
    let hr_lea_lib = code.len();
    code.extend_from_slice(&[0x48, 0x8D, 0x0D, 0x00, 0x00, 0x00, 0x00]); // lea rcx, [rip+disp32]

    // sub rsp, 32
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x20]);
    // call [rbp-0x58] ; LoadLibraryA
    code.extend_from_slice(&[0xFF, 0x55, 0xA8]);
    // add rsp, 32
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20]);

    // test rax, rax
    code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    let hr_jz_loadfail = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    hr_fixup_to_real_epilogue.push(hr_jz_loadfail);

    // mov rbx, rax ; HMODULE
    code.extend_from_slice(&[0x48, 0x89, 0xC3]);

    // For each handler: GetProcAddress(rbx, symbol_name) → store at slot VA
    let mut hr_sym_leas: Vec<(usize, usize)> = Vec::new(); // (lea_pos, handler_index)
    for (i, (_name, slot_va)) in hook_lib.handlers.iter().enumerate() {
        let hr_lea_sym = code.len();
        code.extend_from_slice(&[0x48, 0x8D, 0x15, 0x00, 0x00, 0x00, 0x00]); // lea rdx, [rip+disp32]
        hr_sym_leas.push((hr_lea_sym, i));

        code.extend_from_slice(&[0x48, 0x89, 0xD9]); // mov rcx, rbx
        code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x20]); // sub rsp, 32
        code.extend_from_slice(&[0xFF, 0x55, 0xA0]); // call [rbp-0x60]
        code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20]); // add rsp, 32

        code.extend_from_slice(&[0x48, 0x85, 0xC0]); // test rax, rax
        let hr_jz_skip = code.len();
        code.extend_from_slice(&[0x74, 0x00]); // jz .hr_skip_store

        // lea r13, [rip + disp32] — data slot VA (RIP-relative for ASLR)
        let lea_slot_pos = code.len();
        code.extend_from_slice(&[0x4C, 0x8D, 0x2D, 0x00, 0x00, 0x00, 0x00]);
        let lea_slot_rip = init_va + code.len() as u64;
        let slot_disp = (*slot_va as i64 - lea_slot_rip as i64) as i32;
        code[lea_slot_pos + 3..lea_slot_pos + 7].copy_from_slice(&slot_disp.to_le_bytes());
        // mov [r13], rax
        code.extend_from_slice(&[0x49, 0x89, 0x45, 0x00]);

        // .hr_skip_store:
        let hr_skip_store = code.len();
        code[hr_jz_skip + 1] = (hr_skip_store - (hr_jz_skip + 2)) as u8;
    }

    // JMP over the embedded strings that follow.
    let hr_jmp_over_strings = code.len();
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]); // jmp rel32 placeholder

    // Embed strings: library path, then handler symbol names.
    // Fix up the LEA displacements.

    // Library path string
    let hr_lib_str = code.len();
    let lib_rip = init_va + (hr_lea_lib + 7) as u64;
    let lib_delta = (init_va + hr_lib_str as u64) as i64 - lib_rip as i64;
    code[hr_lea_lib + 3..hr_lea_lib + 7].copy_from_slice(&(lib_delta as i32).to_le_bytes());
    code.extend_from_slice(hook_lib.library_path.as_bytes());
    code.push(0x00); // null terminator

    // Handler symbol name strings
    for (lea_pos, handler_idx) in &hr_sym_leas {
        let str_offset = code.len();
        let sym_rip = init_va + (lea_pos + 7) as u64;
        let sym_delta = (init_va + str_offset as u64) as i64 - sym_rip as i64;
        code[lea_pos + 3..lea_pos + 7].copy_from_slice(&(sym_delta as i32).to_le_bytes());
        let (name, _) = &hook_lib.handlers[*handler_idx];
        code.extend_from_slice(name.as_bytes());
        code.push(0x00);
    }

    // Fix up JMP over strings
    let hr_after_strings = code.len();
    let jmp_rel = (hr_after_strings as i32) - (hr_jmp_over_strings as i32 + 5);
    code[hr_jmp_over_strings + 1..hr_jmp_over_strings + 5].copy_from_slice(&jmp_rel.to_le_bytes());

    // Fall through to real_epilogue

    (label, hr_fixup_to_real_epilogue)
}

/// Emit PE32 code to parse the decimal SHM ID and construct the SHM name.
fn emit_pe32_shm_name_build(code: &mut Vec<u8>) {
    const LOCAL_SIZE: u32 = 128;
    // ===== FOUND! Parse decimal SHM ID from wide string at edi + 26 =====
    // lea edi, [edi + 26]  ; skip past "__AFL_SHM_ID=" (13 wchars = 26 bytes)
    code.extend_from_slice(&[0x8D, 0x7F, 0x1A]);

    // Use ebx as shmid accumulator (we don't need needle pointer anymore).
    // xor ebx, ebx  ; shmid = 0
    code.extend_from_slice(&[0x31, 0xDB]);

    // .parse_loop:
    let parse_loop = code.len();
    // movzx eax, word [edi]  ; load wide char
    code.extend_from_slice(&[0x0F, 0xB7, 0x07]);
    // sub ax, '0'  (0x30)
    code.extend_from_slice(&[0x66, 0x2D, 0x30, 0x00]);
    // cmp ax, 9
    code.extend_from_slice(&[0x66, 0x3D, 0x09, 0x00]);
    // ja .parse_done
    let ja_parse_done_pos = code.len();
    code.extend_from_slice(&[0x77, 0x00]); // ja rel8 placeholder
    // imul ebx, ebx, 10
    code.extend_from_slice(&[0x6B, 0xDB, 0x0A]);
    // movzx eax, ax
    code.extend_from_slice(&[0x0F, 0xB7, 0xC0]);
    // add ebx, eax
    code.extend_from_slice(&[0x01, 0xC3]);
    // add edi, 2  ; next wchar
    code.extend_from_slice(&[0x83, 0xC7, 0x02]);
    // jmp .parse_loop
    let jmp_parse_rel = (parse_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_parse_rel as u8]);

    // .parse_done:
    let parse_done = code.len();
    code[ja_parse_done_pos + 1] = (parse_done - (ja_parse_done_pos + 2)) as u8;

    // ebx = SHM ID. Save it.
    // mov [ebp - 0x04], ebx  ; save SHM ID at local
    code.extend_from_slice(&[0x89, 0x5D, 0xFC]);

    // ============================================================
    // Step 3: Build SHM name "afl_shm_<id>" on stack
    // ============================================================

    // lea edi, [ebp - LOCAL_SIZE]  ; name buffer
    let name_ebp_offset: i32 = -(LOCAL_SIZE as i32);
    code.extend_from_slice(&[0x8D, 0xBD]);
    code.extend_from_slice(&name_ebp_offset.to_le_bytes());

    // Write "afl_shm_" prefix (8 bytes, written as two dwords)
    let prefix_bytes = b"afl_shm_";
    let prefix_lo = u32::from_le_bytes([
        prefix_bytes[0],
        prefix_bytes[1],
        prefix_bytes[2],
        prefix_bytes[3],
    ]);
    let prefix_hi = u32::from_le_bytes([
        prefix_bytes[4],
        prefix_bytes[5],
        prefix_bytes[6],
        prefix_bytes[7],
    ]);
    // mov dword [edi], prefix_lo
    code.extend_from_slice(&[0xC7, 0x07]);
    code.extend_from_slice(&prefix_lo.to_le_bytes());
    // mov dword [edi + 4], prefix_hi
    code.extend_from_slice(&[0xC7, 0x47, 0x04]);
    code.extend_from_slice(&prefix_hi.to_le_bytes());
    // add edi, 8
    code.extend_from_slice(&[0x83, 0xC7, 0x08]);

    // Convert ebx (SHM ID) to decimal ASCII at [edi].
    // mov eax, ebx
    code.extend_from_slice(&[0x89, 0xD8]);
    // mov ebx, edi  ; save start of digits
    code.extend_from_slice(&[0x89, 0xFB]);
    // test eax, eax
    code.extend_from_slice(&[0x85, 0xC0]);
    // jnz .conv_loop
    let jnz_conv_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jnz rel8 placeholder
    // Zero case: write '0'
    code.extend_from_slice(&[0xC6, 0x07, 0x30]); // mov byte [edi], '0'
    code.extend_from_slice(&[0x47]); // inc edi
    // jmp .conv_done
    let jmp_conv_done_pos = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp rel8 placeholder

    // .conv_loop:
    let conv_loop = code.len();
    code[jnz_conv_pos + 1] = (conv_loop - (jnz_conv_pos + 2)) as u8;
    // xor edx, edx
    code.extend_from_slice(&[0x31, 0xD2]);
    // mov ecx, 10
    code.extend_from_slice(&[0xB9, 0x0A, 0x00, 0x00, 0x00]);
    // div ecx  ; eax = quotient, edx = remainder
    code.extend_from_slice(&[0xF7, 0xF1]);
    // add dl, '0'
    code.extend_from_slice(&[0x80, 0xC2, 0x30]);
    // mov [edi], dl
    code.extend_from_slice(&[0x88, 0x17]);
    // inc edi
    code.push(0x47);
    // test eax, eax
    code.extend_from_slice(&[0x85, 0xC0]);
    // jnz .conv_loop
    let jnz_conv2_rel = (conv_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0x75, jnz_conv2_rel as u8]);

    // Null-terminate and reverse.
    // mov byte [edi], 0
    code.extend_from_slice(&[0xC6, 0x07, 0x00]);
    // dec edi  ; points to last digit
    code.push(0x4F); // dec edi

    // .reverse_loop:
    let reverse_loop = code.len();
    // cmp ebx, edi
    code.extend_from_slice(&[0x39, 0xFB]);
    // jge .conv_done
    let jge_conv_done_pos = code.len();
    code.extend_from_slice(&[0x7D, 0x00]); // jge rel8 placeholder
    // mov al, [ebx]
    code.extend_from_slice(&[0x8A, 0x03]);
    // mov cl, [edi]
    code.extend_from_slice(&[0x8A, 0x0F]);
    // mov [ebx], cl
    code.extend_from_slice(&[0x88, 0x0B]);
    // mov [edi], al
    code.extend_from_slice(&[0x88, 0x07]);
    // inc ebx
    code.push(0x43);
    // dec edi
    code.push(0x4F);
    // jmp .reverse_loop
    let jmp_rev_rel = (reverse_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_rev_rel as u8]);

    // .conv_done:
    let conv_done = code.len();
    code[jmp_conv_done_pos + 1] = (conv_done - (jmp_conv_done_pos + 2)) as u8;
    code[jge_conv_done_pos + 1] = (conv_done - (jge_conv_done_pos + 2)) as u8;
}

/// Walk PE32 PEB_LDR_DATA InLoadOrderModuleList to find kernel32.dll.
/// Returns the fixup position for the "kernel32 not found" jump to epilogue.
fn emit_pe32_kernel32_walk(code: &mut Vec<u8>) -> usize {
    // ============================================================
    // Step 4: Walk PEB_LDR_DATA to find kernel32.dll
    // PEB at fs:[0x30]
    // PEB.Ldr at PEB+0x0C
    // PEB_LDR_DATA.InLoadOrderModuleList at Ldr+0x0C
    // Each entry: LDR_DATA_TABLE_ENTRY (32-bit):
    //   +0x00: InLoadOrderLinks (LIST_ENTRY: Flink, Blink) (8 bytes)
    //   +0x18: DllBase (4 bytes)
    //   +0x2C: BaseDllName (UNICODE_STRING: Length u16, MaxLength u16, Buffer ptr)
    //          Buffer at +0x30 (offset +0x04 from UNICODE_STRING start at +0x2C)
    // ============================================================

    // mov eax, fs:[0x30]  ; PEB
    code.extend_from_slice(&[0x64, 0xA1, 0x30, 0x00, 0x00, 0x00]);
    // mov eax, [eax + 0x0C]  ; PEB.Ldr
    code.extend_from_slice(&[0x8B, 0x40, 0x0C]);
    // lea ebx, [eax + 0x0C]  ; &InLoadOrderModuleList (list head)
    code.extend_from_slice(&[0x8D, 0x58, 0x0C]);
    // mov edi, [ebx]  ; first entry (Flink)
    code.extend_from_slice(&[0x8B, 0x3B]);

    // .ldr_loop:
    let ldr_loop = code.len();
    // cmp edi, ebx  ; back to head?
    code.extend_from_slice(&[0x39, 0xDF]);
    // je .epilogue  ; kernel32 not found
    let je_epilogue_k32_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // je rel32 placeholder

    // Check BaseDllName: LDR_DATA_TABLE_ENTRY.BaseDllName at offset 0x2C (32-bit).
    // UNICODE_STRING.Length at +0x00 (u16), Buffer at +0x04 (ptr, 32-bit).
    // movzx ecx, word [edi + 0x2C]  ; BaseDllName.Length (in bytes)
    code.extend_from_slice(&[0x0F, 0xB7, 0x4F, 0x2C]);
    // cmp ecx, 24  ; "KERNEL32.DLL" = 12 wchars = 24 bytes
    code.extend_from_slice(&[0x83, 0xF9, 0x18]);
    // jne .ldr_next
    let jne_ldr_next1_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // mov ecx, [edi + 0x30]  ; BaseDllName.Buffer (at UNICODE_STRING + 0x04)
    code.extend_from_slice(&[0x8B, 0x4F, 0x30]);

    // Case-insensitive check: first wchar ORed with 0x20 == 'k' (0x6B)
    // movzx eax, word [ecx]
    code.extend_from_slice(&[0x0F, 0xB7, 0x01]);
    // or al, 0x20
    code.extend_from_slice(&[0x0C, 0x20]);
    // cmp al, 'k'
    code.extend_from_slice(&[0x3C, 0x6B]);
    // jne .ldr_next
    let jne_ldr_next2_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // Check '3' at wchar index 6 (byte offset 12)
    // movzx eax, word [ecx + 12]
    code.extend_from_slice(&[0x0F, 0xB7, 0x41, 0x0C]);
    // cmp al, '3'
    code.extend_from_slice(&[0x3C, 0x33]);
    // jne .ldr_next
    let jne_ldr_next3_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // Check '2' at wchar index 7 (byte offset 14)
    // movzx eax, word [ecx + 14]
    code.extend_from_slice(&[0x0F, 0xB7, 0x41, 0x0E]);
    // cmp al, '2'
    code.extend_from_slice(&[0x3C, 0x32]);
    // jne .ldr_next
    let jne_ldr_next4_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // Found kernel32.dll! DllBase is at [edi + 0x18] (32-bit).
    // mov esi, [edi + 0x18]  ; esi = kernel32 base (repurpose esi; data_va saved earlier)
    // Save esi (data_va) to stack before repurposing esi for kernel32 base.
    // data_va can also be reloaded from the immediate constant if needed.
    // push esi  ; save data_va
    code.push(0x56);
    // mov esi, [edi + 0x18]  ; esi = kernel32 base
    code.extend_from_slice(&[0x8B, 0x77, 0x18]);
    // jmp .resolve_exports
    let jmp_resolve_pos = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp rel8 placeholder

    // .ldr_next:
    let ldr_next = code.len();
    code[jne_ldr_next1_pos + 1] = (ldr_next - (jne_ldr_next1_pos + 2)) as u8;
    code[jne_ldr_next2_pos + 1] = (ldr_next - (jne_ldr_next2_pos + 2)) as u8;
    code[jne_ldr_next3_pos + 1] = (ldr_next - (jne_ldr_next3_pos + 2)) as u8;
    code[jne_ldr_next4_pos + 1] = (ldr_next - (jne_ldr_next4_pos + 2)) as u8;
    // mov edi, [edi]  ; Flink (next entry)
    code.extend_from_slice(&[0x8B, 0x3F]);
    // jmp .ldr_loop
    let jmp_ldr_rel = (ldr_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_ldr_rel as u8]);

    // .resolve_exports:
    let resolve_exports = code.len();
    code[jmp_resolve_pos + 1] = (resolve_exports - (jmp_resolve_pos + 2)) as u8;

    je_epilogue_k32_pos
}

/// Emit the PE32 hook-only resolution block.
/// Returns (hook_resolve_label, fixup_positions_to_real_epilogue).
fn emit_pe32_hook_only_resolve(
    code: &mut Vec<u8>,
    init_va: u64,
    hook_lib: &PeHookLibraryInfo,
) -> (usize, Vec<usize>) {
    let mut hr_fixup_to_real_epilogue: Vec<usize> = Vec::new();

    let label = code.len();

    // Walk PEB_LDR_DATA for kernel32 (same as Step 4, 32-bit)
    // mov eax, fs:[0x30]
    code.extend_from_slice(&[0x64, 0xA1, 0x30, 0x00, 0x00, 0x00]);
    // mov eax, [eax + 0x0C]  ; PEB.Ldr
    code.extend_from_slice(&[0x8B, 0x40, 0x0C]);
    // lea ebx, [eax + 0x0C]
    code.extend_from_slice(&[0x8D, 0x58, 0x0C]);
    // mov edi, [ebx]
    code.extend_from_slice(&[0x8B, 0x3B]);

    let hr_ldr_loop = code.len();
    // cmp edi, ebx
    code.extend_from_slice(&[0x39, 0xDF]);
    let hr_je_no_k32 = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // je .real_epilogue
    hr_fixup_to_real_epilogue.push(hr_je_no_k32);

    // movzx ecx, word [edi + 0x2C]
    code.extend_from_slice(&[0x0F, 0xB7, 0x4F, 0x2C]);
    // cmp ecx, 24
    code.extend_from_slice(&[0x83, 0xF9, 0x18]);
    let hr_jne1 = code.len();
    code.extend_from_slice(&[0x75, 0x00]);
    // mov ecx, [edi + 0x30]  ; BaseDllName.Buffer
    code.extend_from_slice(&[0x8B, 0x4F, 0x30]);
    // movzx eax, word [ecx]
    code.extend_from_slice(&[0x0F, 0xB7, 0x01]);
    // or al, 0x20
    code.extend_from_slice(&[0x0C, 0x20]);
    // cmp al, 'k'
    code.extend_from_slice(&[0x3C, 0x6B]);
    let hr_jne2 = code.len();
    code.extend_from_slice(&[0x75, 0x00]);
    // movzx eax, word [ecx + 12]
    code.extend_from_slice(&[0x0F, 0xB7, 0x41, 0x0C]);
    // cmp al, '3'
    code.extend_from_slice(&[0x3C, 0x33]);
    let hr_jne3 = code.len();
    code.extend_from_slice(&[0x75, 0x00]);
    // movzx eax, word [ecx + 14]
    code.extend_from_slice(&[0x0F, 0xB7, 0x41, 0x0E]);
    // cmp al, '2'
    code.extend_from_slice(&[0x3C, 0x32]);
    let hr_jne4 = code.len();
    code.extend_from_slice(&[0x75, 0x00]);

    // Found! mov esi, [edi + 0x18]  ; kernel32 base in esi
    code.extend_from_slice(&[0x8B, 0x77, 0x18]);
    let hr_jmp_found = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp .hr_found

    // .hr_ldr_next:
    let hr_ldr_next = code.len();
    code[hr_jne1 + 1] = (hr_ldr_next - (hr_jne1 + 2)) as u8;
    code[hr_jne2 + 1] = (hr_ldr_next - (hr_jne2 + 2)) as u8;
    code[hr_jne3 + 1] = (hr_ldr_next - (hr_jne3 + 2)) as u8;
    code[hr_jne4 + 1] = (hr_ldr_next - (hr_jne4 + 2)) as u8;
    // mov edi, [edi]
    code.extend_from_slice(&[0x8B, 0x3F]);
    let rel = (hr_ldr_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, rel as u8]);

    // .hr_found:
    let hr_found = code.len();
    code[hr_jmp_found + 1] = (hr_found - (hr_jmp_found + 2)) as u8;

    // Parse exports (PE32)
    // mov eax, [esi + 0x3C]
    code.extend_from_slice(&[0x8B, 0x46, 0x3C]);
    // lea ebx, [esi + eax]
    code.extend_from_slice(&[0x8D, 0x1C, 0x06]);
    // mov eax, [ebx + 0x78]  ; Export dir RVA
    code.extend_from_slice(&[0x8B, 0x43, 0x78]);
    // test eax, eax
    code.extend_from_slice(&[0x85, 0xC0]);
    let hr_jz_noexp = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    hr_fixup_to_real_epilogue.push(hr_jz_noexp);

    // lea ebx, [esi + eax]  ; export directory
    code.extend_from_slice(&[0x8D, 0x1C, 0x06]);
    // mov ecx, [ebx + 0x18]  ; NumberOfNames
    code.extend_from_slice(&[0x8B, 0x4B, 0x18]);
    // AddressOfNames
    code.extend_from_slice(&[0x8B, 0x43, 0x20]); // mov eax, [ebx + 0x20]
    code.extend_from_slice(&[0x8D, 0x04, 0x06]); // lea eax, [esi + eax]
    code.extend_from_slice(&[0x89, 0x45, 0xF8]); // mov [ebp - 0x08], eax
    // AddressOfNameOrdinals
    code.extend_from_slice(&[0x8B, 0x43, 0x24]); // mov eax, [ebx + 0x24]
    code.extend_from_slice(&[0x8D, 0x04, 0x06]); // lea eax, [esi + eax]
    code.extend_from_slice(&[0x89, 0x45, 0xF4]); // mov [ebp - 0x0C], eax
    // AddressOfFunctions
    code.extend_from_slice(&[0x8B, 0x43, 0x1C]); // mov eax, [ebx + 0x1C]
    code.extend_from_slice(&[0x8D, 0x04, 0x06]); // lea eax, [esi + eax]
    code.extend_from_slice(&[0x89, 0x45, 0xF0]); // mov [ebp - 0x10], eax

    // Clear slots
    code.extend_from_slice(&[0xC7, 0x45, 0xE4, 0x00, 0x00, 0x00, 0x00]); // [ebp-0x1C]=0
    code.extend_from_slice(&[0xC7, 0x45, 0xE0, 0x00, 0x00, 0x00, 0x00]); // [ebp-0x20]=0

    let hash_lla = djb2_hash(b"LoadLibraryA");
    let hash_gpa = djb2_hash(b"GetProcAddress");

    // xor edx, edx
    code.extend_from_slice(&[0x31, 0xD2]);

    let hr_exp_loop = code.len();
    // cmp edx, ecx
    code.extend_from_slice(&[0x39, 0xCA]);
    let hr_jge_exp_done = code.len();
    code.extend_from_slice(&[0x0F, 0x8D, 0x00, 0x00, 0x00, 0x00]);

    // mov eax, [ebp - 0x08]  ; AddressOfNames
    code.extend_from_slice(&[0x8B, 0x45, 0xF8]);
    // mov eax, [eax + edx*4]
    code.extend_from_slice(&[0x8B, 0x04, 0x90]);
    // lea edi, [esi + eax]
    code.extend_from_slice(&[0x8D, 0x3C, 0x06]);

    // DJB2 hash
    code.push(0x52); // push edx
    code.push(0x51); // push ecx
    code.push(0xB8); // mov eax, 5381
    code.extend_from_slice(&5381u32.to_le_bytes());
    let hr_hl = code.len();
    code.extend_from_slice(&[0x0F, 0xB6, 0x1F]); // movzx ebx, byte [edi]
    code.extend_from_slice(&[0x84, 0xDB]); // test bl, bl
    let hr_jz_hd = code.len();
    code.extend_from_slice(&[0x74, 0x00]);
    code.extend_from_slice(&[0x6B, 0xC0, 0x21]); // imul eax, eax, 33
    code.extend_from_slice(&[0x01, 0xD8]); // add eax, ebx
    code.push(0x47); // inc edi
    let r = (hr_hl as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, r as u8]);
    let hr_hd = code.len();
    code[hr_jz_hd + 1] = (hr_hd - (hr_jz_hd + 2)) as u8;
    code.push(0x59); // pop ecx
    code.push(0x5A); // pop edx

    // Check LoadLibraryA
    code.push(0x3D);
    code.extend_from_slice(&hash_lla.to_le_bytes());
    let hr_jne_lla = code.len();
    code.extend_from_slice(&[0x75, 0x00]);

    // Resolve
    code.extend_from_slice(&[0x8B, 0x5D, 0xF4]); // mov ebx, [ebp - 0x0C]
    code.extend_from_slice(&[0x0F, 0xB7, 0x04, 0x53]); // movzx eax, word [ebx + edx*2]
    code.extend_from_slice(&[0x8B, 0x5D, 0xF0]); // mov ebx, [ebp - 0x10]
    code.extend_from_slice(&[0x8B, 0x04, 0x83]); // mov eax, [ebx + eax*4]
    code.extend_from_slice(&[0x8D, 0x04, 0x06]); // lea eax, [esi + eax]
    code.extend_from_slice(&[0x89, 0x45, 0xE4]); // mov [ebp - 0x1C], eax
    let hr_jmp_next1 = code.len();
    code.extend_from_slice(&[0xEB, 0x00]);

    // Check GetProcAddress
    let hr_check_gpa = code.len();
    code[hr_jne_lla + 1] = (hr_check_gpa - (hr_jne_lla + 2)) as u8;

    code.push(0x3D);
    code.extend_from_slice(&hash_gpa.to_le_bytes());
    let hr_jne_gpa = code.len();
    code.extend_from_slice(&[0x75, 0x00]);

    code.extend_from_slice(&[0x8B, 0x5D, 0xF4]);
    code.extend_from_slice(&[0x0F, 0xB7, 0x04, 0x53]);
    code.extend_from_slice(&[0x8B, 0x5D, 0xF0]);
    code.extend_from_slice(&[0x8B, 0x04, 0x83]);
    code.extend_from_slice(&[0x8D, 0x04, 0x06]);
    code.extend_from_slice(&[0x89, 0x45, 0xE0]); // mov [ebp - 0x20], eax

    // .hr_exp_next:
    let hr_exp_next = code.len();
    code[hr_jmp_next1 + 1] = (hr_exp_next - (hr_jmp_next1 + 2)) as u8;
    code[hr_jne_gpa + 1] = (hr_exp_next - (hr_jne_gpa + 2)) as u8;

    // Check if both found
    code.extend_from_slice(&[0x83, 0x7D, 0xE4, 0x00]); // cmp dword [ebp-0x1C], 0
    let hr_jz_cont = code.len();
    code.extend_from_slice(&[0x74, 0x00]);
    code.extend_from_slice(&[0x83, 0x7D, 0xE0, 0x00]); // cmp dword [ebp-0x20], 0
    let hr_jnz_both = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);

    let hr_exp_inc = code.len();
    code[hr_jz_cont + 1] = (hr_exp_inc - (hr_jz_cont + 2)) as u8;
    // inc edx
    code.extend_from_slice(&[0xFF, 0xC2]);
    let r2 = (hr_exp_loop as i32) - (code.len() as i32 + 5);
    code.push(0xE9);
    code.extend_from_slice(&r2.to_le_bytes());

    // .hr_exp_done:
    let hr_exp_done = code.len();
    let r3 = (hr_exp_done as i32) - (hr_jge_exp_done as i32 + 6);
    code[hr_jge_exp_done + 2..hr_jge_exp_done + 6].copy_from_slice(&r3.to_le_bytes());
    let r4 = (hr_exp_done as i32) - (hr_jnz_both as i32 + 6);
    code[hr_jnz_both + 2..hr_jnz_both + 6].copy_from_slice(&r4.to_le_bytes());

    // Check both were found
    code.extend_from_slice(&[0x83, 0x7D, 0xE4, 0x00]); // cmp [ebp-0x1C], 0
    let hr_jz_nolib = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    hr_fixup_to_real_epilogue.push(hr_jz_nolib);

    code.extend_from_slice(&[0x83, 0x7D, 0xE0, 0x00]); // cmp [ebp-0x20], 0
    let hr_jz_nogpa = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    hr_fixup_to_real_epilogue.push(hr_jz_nogpa);

    // Call LoadLibraryA
    // mov eax, imm32  ; library path string VA (placeholder)
    let hr_mov_lib = code.len();
    code.push(0xB8);
    code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    // push eax
    code.push(0x50);
    // call [ebp - 0x1C]
    code.extend_from_slice(&[0xFF, 0x55, 0xE4]);
    // add esp, 4
    code.extend_from_slice(&[0x83, 0xC4, 0x04]);

    // test eax, eax
    code.extend_from_slice(&[0x85, 0xC0]);
    let hr_jz_loadfail = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    hr_fixup_to_real_epilogue.push(hr_jz_loadfail);

    // mov ebx, eax  ; HMODULE
    code.extend_from_slice(&[0x89, 0xC3]);

    // For each handler: GetProcAddress(hModule, lpProcName) -> store at slot VA
    let mut hr_sym_movs: Vec<(usize, usize)> = Vec::new(); // (mov_pos, handler_index)
    for (i, (_name, slot_va)) in hook_lib.handlers.iter().enumerate() {
        let hr_mov_sym = code.len();
        code.push(0xB8); // mov eax, imm32
        code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        hr_sym_movs.push((hr_mov_sym, i));

        // push eax  ; lpProcName
        code.push(0x50);
        // push ebx  ; hModule
        code.push(0x53);
        // call [ebp - 0x20]  ; GetProcAddress
        code.extend_from_slice(&[0xFF, 0x55, 0xE0]);
        // add esp, 8
        code.extend_from_slice(&[0x83, 0xC4, 0x08]);

        // test eax, eax
        code.extend_from_slice(&[0x85, 0xC0]);
        let hr_jz_skip = code.len();
        code.extend_from_slice(&[0x74, 0x00]);

        // mov [slot_va], eax  (absolute)
        code.push(0xA3);
        code.extend_from_slice(&(*slot_va as u32).to_le_bytes());

        let hr_skip_store = code.len();
        code[hr_jz_skip + 1] = (hr_skip_store - (hr_jz_skip + 2)) as u8;
    }

    // JMP over strings
    let hr_jmp_over_strings = code.len();
    code.extend_from_slice(&[0xE9, 0x00, 0x00, 0x00, 0x00]);

    // Embed library path string
    let hr_lib_str_offset = code.len();
    let hr_lib_str_va = init_va + hr_lib_str_offset as u64;
    code[hr_mov_lib + 1..hr_mov_lib + 5].copy_from_slice(&(hr_lib_str_va as u32).to_le_bytes());
    code.extend_from_slice(hook_lib.library_path.as_bytes());
    code.push(0x00);

    // Embed handler symbol name strings
    for (mov_pos, handler_idx) in &hr_sym_movs {
        let str_offset = code.len();
        let str_va = init_va + str_offset as u64;
        code[mov_pos + 1..mov_pos + 5].copy_from_slice(&(str_va as u32).to_le_bytes());
        let (name, _) = &hook_lib.handlers[*handler_idx];
        code.extend_from_slice(name.as_bytes());
        code.push(0x00);
    }

    // Fix up JMP over strings
    let hr_after_strings = code.len();
    let jmp_rel = (hr_after_strings as i32) - (hr_jmp_over_strings as i32 + 5);
    code[hr_jmp_over_strings + 1..hr_jmp_over_strings + 5].copy_from_slice(&jmp_rel.to_le_bytes());

    (label, hr_fixup_to_real_epilogue)
}

/// Generate PE-specific init code for SHM attachment.
///
/// On Windows, there is no shmat(). Instead we:
/// 1. Read `__AFL_SHM_ID` from the environment using GetEnvironmentVariableA
/// 2. Construct the SHM name "afl_shm_<id>"
/// 3. Open the file mapping with OpenFileMappingA
/// 4. Map it with MapViewOfFile
///
/// Since this binary will run on Windows but is cross-rewritten from Linux,
/// we can't link against kernel32.dll at rewrite time. Instead, we use a
/// self-contained approach: the init code walks the PEB to find kernel32.dll
/// base, then manually resolves the needed functions by hash.
///
/// For simplicity, we use the environment block from the PEB directly
/// (wide-string scanning), which avoids any DLL imports entirely.
fn generate_pe_init_code(
    init_va: u64,
    data_va: u64,
    original_entry: u64,
    hook_library: Option<&PeHookLibraryInfo>,
    persistent_data_va: Option<u64>,
) -> Result<crate::trampoline::InitCode> {
    let mut code = Vec::with_capacity(1024);
    let entry_va = init_va;

    // ============================================================
    // Windows x64 init code — PEB-based, import-free
    //
    // Strategy:
    // 1. Access PEB via gs:[0x60]
    // 2. Walk PEB_LDR_DATA to find kernel32.dll
    // 3. Parse its export directory to resolve:
    //    - GetEnvironmentVariableA (or W)
    //    - OpenFileMappingA
    //    - MapViewOfFile
    // 4. Call them to set up SHM
    //
    // For initial implementation, we use a simpler approach:
    // Walk the process environment block (PEB.ProcessParameters.Environment)
    // which is a block of wide-string "KEY=VALUE\0" entries terminated by \0\0.
    // Find __AFL_SHM_ID=<decimal>, parse the ID.
    // Then walk PEB_LDR_DATA InLoadOrderModuleList to find kernel32.dll,
    // resolve OpenFileMappingA and MapViewOfFile from its exports.
    // ============================================================

    // Save callee-saved registers (Microsoft x64 ABI: rbx, rsi, rdi, rbp, r12-r15).
    // Also allocate shadow space (32 bytes) for API calls.
    code.push(0x55); // push rbp
    code.extend_from_slice(&[0x48, 0x89, 0xE5]); // mov rbp, rsp
    code.push(0x53); // push rbx
    code.push(0x56); // push rsi
    code.push(0x57); // push rdi
    code.extend_from_slice(&[0x41, 0x54]); // push r12
    code.extend_from_slice(&[0x41, 0x55]); // push r13
    code.extend_from_slice(&[0x41, 0x56]); // push r14
    code.extend_from_slice(&[0x41, 0x57]); // push r15

    // Align stack to 16 bytes and allocate local space.
    // Entry RSP is 16n-8 (return addr pushed by PE loader's call).
    // 8 callee-saved pushes = 64 bytes → RSP = 16n - 8 - 64 = 16n - 72.
    // LOCAL_SIZE chosen so stack is 16-byte aligned after all pushes and sub.
    // Entry RSP = 16n-8 (return addr pushed by caller). After push rbp + 7 more
    // callee-saved pushes (64 bytes): RSP = 16n-8-64 = 16(n-4)-8.
    // sub rsp, 568: RSP = 16(n-4)-8-568 = 16(n-40). 16-aligned.
    const LOCAL_SIZE: u32 = 568;
    // sub rsp, LOCAL_SIZE
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&LOCAL_SIZE.to_le_bytes());

    // lea r12, [rip + delta_to_data_va]  ; r12 = &data_area
    let lea_r12_pos = code.len();
    code.extend_from_slice(&[0x4C, 0x8D, 0x25, 0x00, 0x00, 0x00, 0x00]);
    let lea_r12_rip = init_va + code.len() as u64;
    let delta = (data_va as i64 - lea_r12_rip as i64) as i32;
    code[lea_r12_pos + 3..lea_r12_pos + 7].copy_from_slice(&delta.to_le_bytes());

    // ============================================================
    // Step 1: Access PEB → ProcessParameters → Environment
    // PEB is at gs:[0x60] on x64 Windows
    // PEB.ProcessParameters at PEB+0x20
    // RTL_USER_PROCESS_PARAMETERS.Environment at +0x80
    // ============================================================

    // mov rax, gs:[0x60]  ; PEB
    code.extend_from_slice(&[0x65, 0x48, 0x8B, 0x04, 0x25, 0x60, 0x00, 0x00, 0x00]);
    // mov rax, [rax + 0x20]  ; PEB.ProcessParameters
    code.extend_from_slice(&[0x48, 0x8B, 0x40, 0x20]);
    // mov rsi, [rax + 0x80]  ; ProcessParameters.Environment (wide string block)
    code.extend_from_slice(&[0x48, 0x8B, 0xB0, 0x80, 0x00, 0x00, 0x00]);
    // test rsi, rsi
    code.extend_from_slice(&[0x48, 0x85, 0xF6]);
    // jz .epilogue  ; no environment
    let jz_epilogue_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // ============================================================
    // Step 2: Scan environment for L"__AFL_SHM_ID="
    // The environment block is a series of null-terminated wide strings,
    // terminated by an empty string (\0\0).
    // We're looking for the wide-char sequence: __AFL_SHM_ID=
    // ============================================================

    // Load first 8 wchars of needle: "__AFL_SH" as 16 bytes of wide chars
    // But this is complex. Simpler: scan byte-by-byte for the ASCII pattern
    // since Windows env vars use UTF-16LE, ASCII chars have the high byte = 0.
    // Search for: '_' 0x00 '_' 0x00 'A' 0x00 'F' 0x00 'L' 0x00 '_' 0x00
    //             'S' 0x00 'H' 0x00 'M' 0x00 '_' 0x00 'I' 0x00 'D' 0x00 '=' 0x00
    // That's 26 bytes (13 wide chars).

    // r13 = pointer to needle string (embedded at end of code)
    let lea_needle_pos = code.len();
    code.extend_from_slice(&[0x4C, 0x8D, 0x2D, 0x00, 0x00, 0x00, 0x00]); // lea r13, [rip + needle]

    // .env_scan_loop:
    let env_scan_loop = code.len();

    // Check for end of environment block: if [rsi] == 0x0000, done.
    // movzx eax, word [rsi]
    code.extend_from_slice(&[0x0F, 0xB7, 0x06]);
    // test ax, ax
    code.extend_from_slice(&[0x66, 0x85, 0xC0]);
    // jz .epilogue_env (empty string = end of block)
    let jz_epilogue_env_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // Compare 26 bytes at [rsi] with needle.
    // We'll do it in a small loop comparing 8 bytes at a time.
    // mov rdi, rsi  ; save scan position
    code.extend_from_slice(&[0x48, 0x89, 0xF7]);

    // Compare first 8 bytes: "__AF" as wide = 5F 00 5F 00 41 00 46 00
    // mov rax, [rsi]
    code.extend_from_slice(&[0x48, 0x8B, 0x06]);
    // mov rbx, [r13]
    code.extend_from_slice(&[0x49, 0x8B, 0x5D, 0x00]);
    // cmp rax, rbx
    code.extend_from_slice(&[0x48, 0x39, 0xD8]);
    // jne .next_env_var (use rel32 — next_env_var is far away)
    let jne_next1_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // Compare bytes 8..16: "L_SH" as wide = 4C 00 5F 00 53 00 48 00
    // mov rax, [rsi + 8]
    code.extend_from_slice(&[0x48, 0x8B, 0x46, 0x08]);
    // mov rbx, [r13 + 8]
    code.extend_from_slice(&[0x49, 0x8B, 0x5D, 0x08]);
    // cmp rax, rbx
    code.extend_from_slice(&[0x48, 0x39, 0xD8]);
    // jne .next_env_var
    let jne_next2_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // Compare bytes 16..24: "M_ID" as wide = 4D 00 5F 00 49 00 44 00
    // mov rax, [rsi + 16]
    code.extend_from_slice(&[0x48, 0x8B, 0x46, 0x10]);
    // mov rbx, [r13 + 16]
    code.extend_from_slice(&[0x49, 0x8B, 0x5D, 0x10]);
    // cmp rax, rbx
    code.extend_from_slice(&[0x48, 0x39, 0xD8]);
    // jne .next_env_var
    let jne_next3_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // Compare bytes 24..26: "=" as wide = 3D 00
    // movzx eax, word [rsi + 24]
    code.extend_from_slice(&[0x0F, 0xB7, 0x46, 0x18]);
    // cmp ax, 0x003D
    code.extend_from_slice(&[0x66, 0x3D, 0x3D, 0x00]);
    // jne .next_env_var
    let jne_next4_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // Parse decimal SHM ID and build SHM name on stack.
    emit_pe64_shm_name_build(&mut code);

    // Walk PEB_LDR_DATA to find kernel32.dll.
    let je_epilogue_k32_pos = emit_pe64_kernel32_walk(&mut code);

    // ============================================================
    // Step 5: Parse kernel32.dll export directory to find
    // OpenFileMappingA and MapViewOfFile.
    //
    // PE export directory:
    //   DOS header at base, e_lfanew at base+0x3C
    //   PE header at base+e_lfanew
    //   Optional header at PE+0x18
    //   Export directory RVA = DataDirectory[0].VirtualAddress (at OptHdr + 0x70 for PE32+)
    //   Export directory:
    //     +0x18: NumberOfNames
    //     +0x1C: AddressOfFunctions (RVA)
    //     +0x20: AddressOfNames (RVA)
    //     +0x24: AddressOfNameOrdinals (RVA)
    //
    // We hash each name and compare to known hashes of our target functions.
    // Using djb2 hash: hash = hash*33 + c
    // ============================================================

    // r15 = kernel32 base address
    // mov eax, [r15 + 0x3C]  ; e_lfanew
    code.extend_from_slice(&[0x41, 0x8B, 0x47, 0x3C]);
    // lea rbx, [r15 + rax]  ; PE header
    code.extend_from_slice(&[0x49, 0x8D, 0x1C, 0x07]);
    // mov eax, [rbx + 0x88]  ; Export directory RVA (PE32+: OptHdr starts at +0x18, DataDir at +0x70 from there = 0x88)
    code.extend_from_slice(&[0x8B, 0x83, 0x88, 0x00, 0x00, 0x00]);
    // test eax, eax
    code.extend_from_slice(&[0x85, 0xC0]);
    // jz .epilogue  ; no exports
    let jz_no_exports_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // lea rsi, [r15 + rax]  ; export directory
    code.extend_from_slice(&[0x49, 0x8D, 0x34, 0x07]);

    // mov ecx, [rsi + 0x18]  ; NumberOfNames
    code.extend_from_slice(&[0x8B, 0x4E, 0x18]);
    // mov eax, [rsi + 0x20]  ; AddressOfNames RVA
    code.extend_from_slice(&[0x8B, 0x46, 0x20]);
    // lea r13, [r15 + rax]  ; AddressOfNames
    // REX.WRB (0x4D): W=64-bit, R=extend reg to r13, B=extend rm base to r15
    code.extend_from_slice(&[0x4D, 0x8D, 0x2C, 0x07]);
    // mov eax, [rsi + 0x24]  ; AddressOfNameOrdinals RVA
    code.extend_from_slice(&[0x8B, 0x46, 0x24]);
    // r14 currently holds SHM ID -- save it before repurposing for AddressOfNameOrdinals.
    // push r14  ; save SHM ID
    code.extend_from_slice(&[0x41, 0x56]);
    code.extend_from_slice(&[0x4D, 0x8D, 0x34, 0x07]); // lea r14, [r15 + rax]

    // mov eax, [rsi + 0x1C]  ; AddressOfFunctions RVA
    code.extend_from_slice(&[0x8B, 0x46, 0x1C]);
    // Save AddressOfFunctions on stack
    // push rax
    code.push(0x50);
    // push r15  ; save kernel32 base
    code.extend_from_slice(&[0x41, 0x57]);

    // We'll find 2 (or 4 if hooks) functions. Store resolved addresses:
    // r8 = OpenFileMappingA, r9 = MapViewOfFile
    // [rbp - 0x58] = LoadLibraryA, [rbp - 0x60] = GetProcAddress (when hooks)
    //
    // [rsp + 0] = kernel32 base (pushed above)
    // [rsp + 8] = AddressOfFunctions RVA (pushed above)
    // [rsp + 16] = SHM ID (pushed above)
    // xor r8, r8  ; OpenFileMappingA resolved addr = 0 (not found yet)
    code.extend_from_slice(&[0x4D, 0x31, 0xC0]);
    // xor r9, r9  ; MapViewOfFile resolved addr = 0 (not found yet)
    code.extend_from_slice(&[0x4D, 0x31, 0xC9]);

    // Initialise LoadLibraryA / GetProcAddress slots to 0 (always, for simplicity).
    // mov qword [rbp - 0x58], 0
    code.extend_from_slice(&[0x48, 0xC7, 0x45, 0xA8, 0x00, 0x00, 0x00, 0x00]);
    // mov qword [rbp - 0x60], 0
    code.extend_from_slice(&[0x48, 0xC7, 0x45, 0xA0, 0x00, 0x00, 0x00, 0x00]);

    // DJB2 hashes (precomputed).
    let hash_open_file_mapping: u32 = djb2_hash(b"OpenFileMappingA");
    let hash_map_view_of_file: u32 = djb2_hash(b"MapViewOfFile");
    let hash_load_library_a: u32 = djb2_hash(b"LoadLibraryA");
    let hash_get_proc_address: u32 = djb2_hash(b"GetProcAddress");
    let need_hook_funcs = hook_library.is_some();

    // xor edx, edx  ; index = 0
    code.extend_from_slice(&[0x31, 0xD2]);

    // .export_loop:
    let export_loop = code.len();
    // cmp edx, ecx  ; index >= NumberOfNames?
    code.extend_from_slice(&[0x39, 0xCA]);
    // jge .exports_done
    let jge_exports_done_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x8D, 0x00, 0x00, 0x00, 0x00]); // jge rel32 placeholder

    // mov eax, [r13 + rdx*4]  ; name RVA
    code.extend_from_slice(&[0x41, 0x8B, 0x44, 0x95, 0x00]);
    // lea rdi, [r15 + rax]  ; name string
    code.extend_from_slice(&[0x49, 0x8D, 0x3C, 0x07]);

    // Compute DJB2 hash of the name string.
    // push rdx  ; save index
    code.push(0x52);
    // push rcx  ; save count
    code.push(0x51);
    // mov eax, 5381  ; hash = 5381
    code.extend_from_slice(&[0xB8]);
    code.extend_from_slice(&5381u32.to_le_bytes());

    // .hash_loop:
    let hash_loop = code.len();
    // movzx ebx, byte [rdi]
    code.extend_from_slice(&[0x0F, 0xB6, 0x1F]);
    // test bl, bl
    code.extend_from_slice(&[0x84, 0xDB]);
    // jz .hash_done
    let jz_hash_done_pos = code.len();
    code.extend_from_slice(&[0x74, 0x00]); // jz rel8 placeholder
    // imul eax, eax, 33
    code.extend_from_slice(&[0x6B, 0xC0, 0x21]);
    // add eax, ebx
    code.extend_from_slice(&[0x01, 0xD8]);
    // inc rdi
    code.extend_from_slice(&[0x48, 0xFF, 0xC7]);
    // jmp .hash_loop
    let jmp_hash_rel = (hash_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_hash_rel as u8]);

    // .hash_done:
    let hash_done = code.len();
    code[jz_hash_done_pos + 1] = (hash_done - (jz_hash_done_pos + 2)) as u8;

    // pop rcx  ; restore count
    code.push(0x59);
    // pop rdx  ; restore index
    code.push(0x5A);

    // Compare hash with known values.
    // Get function address: ordinal = [AddressOfNameOrdinals + index*2]
    //                       func RVA = [AddressOfFunctions + ordinal*4]
    //                       func addr = kernel32_base + func_rva

    // cmp eax, hash_open_file_mapping
    code.extend_from_slice(&[0x3D]);
    code.extend_from_slice(&hash_open_file_mapping.to_le_bytes());
    // jne .check_map_view
    let jne_check_map_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // Resolve OpenFileMappingA.
    // movzx eax, word [r14 + rdx*2]  ; ordinal
    code.extend_from_slice(&[0x41, 0x0F, 0xB7, 0x04, 0x56]);
    // mov ebx, [rsp + 8]  ; AddressOfFunctions RVA
    code.extend_from_slice(&[0x8B, 0x5C, 0x24, 0x08]);
    // lea rbx, [r15 + rbx]  ; AddressOfFunctions absolute
    code.extend_from_slice(&[0x49, 0x8D, 0x1C, 0x1F]);
    // mov eax, [rbx + rax*4]  ; function RVA
    code.extend_from_slice(&[0x8B, 0x04, 0x83]);
    // lea r8, [r15 + rax]  ; resolved address
    code.extend_from_slice(&[0x4D, 0x8D, 0x04, 0x07]);
    // jmp .export_next
    let jmp_export_next1_pos = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp rel8 placeholder

    // .check_map_view:
    let check_map_view = code.len();
    code[jne_check_map_pos + 1] = (check_map_view - (jne_check_map_pos + 2)) as u8;

    // cmp eax, hash_map_view_of_file
    code.extend_from_slice(&[0x3D]);
    code.extend_from_slice(&hash_map_view_of_file.to_le_bytes());
    // jne .check_load_library
    let jne_check_loadlib_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // Resolve MapViewOfFile.
    // movzx eax, word [r14 + rdx*2]  ; ordinal
    code.extend_from_slice(&[0x41, 0x0F, 0xB7, 0x04, 0x56]);
    // mov ebx, [rsp + 8]  ; AddressOfFunctions RVA
    code.extend_from_slice(&[0x8B, 0x5C, 0x24, 0x08]);
    // lea rbx, [r15 + rbx]
    code.extend_from_slice(&[0x49, 0x8D, 0x1C, 0x1F]);
    // mov eax, [rbx + rax*4]
    code.extend_from_slice(&[0x8B, 0x04, 0x83]);
    // lea r9, [r15 + rax]
    code.extend_from_slice(&[0x4D, 0x8D, 0x0C, 0x07]);
    // jmp .export_next
    let jmp_export_next2_pos = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp rel8 placeholder

    // .check_load_library:
    let check_load_library = code.len();
    code[jne_check_loadlib_pos + 1] = (check_load_library - (jne_check_loadlib_pos + 2)) as u8;

    if need_hook_funcs {
        // cmp eax, hash_load_library_a
        code.extend_from_slice(&[0x3D]);
        code.extend_from_slice(&hash_load_library_a.to_le_bytes());
        // jne .check_get_proc_address
        let jne_check_getproc_pos = code.len();
        code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

        // Resolve LoadLibraryA → store at [rbp - 0x58].
        // movzx eax, word [r14 + rdx*2]  ; ordinal
        code.extend_from_slice(&[0x41, 0x0F, 0xB7, 0x04, 0x56]);
        // mov ebx, [rsp + 8]  ; AddressOfFunctions RVA
        code.extend_from_slice(&[0x8B, 0x5C, 0x24, 0x08]);
        // lea rbx, [r15 + rbx]
        code.extend_from_slice(&[0x49, 0x8D, 0x1C, 0x1F]);
        // mov eax, [rbx + rax*4]
        code.extend_from_slice(&[0x8B, 0x04, 0x83]);
        // lea rax, [r15 + rax]  ; resolved address
        code.extend_from_slice(&[0x49, 0x8D, 0x04, 0x07]);
        // mov [rbp - 0x58], rax
        code.extend_from_slice(&[0x48, 0x89, 0x45, 0xA8]);
        // jmp .export_next
        let jmp_export_next3_pos = code.len();
        code.extend_from_slice(&[0xEB, 0x00]); // jmp rel8 placeholder

        // .check_get_proc_address:
        let check_get_proc_address = code.len();
        code[jne_check_getproc_pos + 1] =
            (check_get_proc_address - (jne_check_getproc_pos + 2)) as u8;

        // cmp eax, hash_get_proc_address
        code.extend_from_slice(&[0x3D]);
        code.extend_from_slice(&hash_get_proc_address.to_le_bytes());
        // jne .export_next
        let jne_export_next_pos = code.len();
        code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

        // Resolve GetProcAddress → store at [rbp - 0x60].
        // movzx eax, word [r14 + rdx*2]  ; ordinal
        code.extend_from_slice(&[0x41, 0x0F, 0xB7, 0x04, 0x56]);
        // mov ebx, [rsp + 8]  ; AddressOfFunctions RVA
        code.extend_from_slice(&[0x8B, 0x5C, 0x24, 0x08]);
        // lea rbx, [r15 + rbx]
        code.extend_from_slice(&[0x49, 0x8D, 0x1C, 0x1F]);
        // mov eax, [rbx + rax*4]
        code.extend_from_slice(&[0x8B, 0x04, 0x83]);
        // lea rax, [r15 + rax]
        code.extend_from_slice(&[0x49, 0x8D, 0x04, 0x07]);
        // mov [rbp - 0x60], rax
        code.extend_from_slice(&[0x48, 0x89, 0x45, 0xA0]);

        // .export_next:
        let export_next = code.len();
        code[jmp_export_next1_pos + 1] = (export_next - (jmp_export_next1_pos + 2)) as u8;
        code[jmp_export_next2_pos + 1] = (export_next - (jmp_export_next2_pos + 2)) as u8;
        code[jmp_export_next3_pos + 1] = (export_next - (jmp_export_next3_pos + 2)) as u8;
        code[jne_export_next_pos + 1] = (export_next - (jne_export_next_pos + 2)) as u8;

        // Early exit: check if all 4 functions found.
        // test r8, r8
        code.extend_from_slice(&[0x4D, 0x85, 0xC0]);
        // jz .export_continue
        let jz_export_continue_pos = code.len();
        code.extend_from_slice(&[0x74, 0x00]); // jz rel8 placeholder
        // test r9, r9
        code.extend_from_slice(&[0x4D, 0x85, 0xC9]);
        // jz .export_continue
        let jz_export_continue2_pos = code.len();
        code.extend_from_slice(&[0x74, 0x00]); // jz rel8 placeholder
        // cmp qword [rbp - 0x58], 0
        code.extend_from_slice(&[0x48, 0x83, 0x7D, 0xA8, 0x00]);
        // jz .export_continue
        let jz_export_continue3_pos = code.len();
        code.extend_from_slice(&[0x74, 0x00]); // jz rel8 placeholder
        // cmp qword [rbp - 0x60], 0
        code.extend_from_slice(&[0x48, 0x83, 0x7D, 0xA0, 0x00]);
        // jnz .exports_done  ; all 4 found!
        let jnz_both_found_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jnz rel32 placeholder

        // .export_continue:
        let export_continue = code.len();
        code[jz_export_continue_pos + 1] = (export_continue - (jz_export_continue_pos + 2)) as u8;
        code[jz_export_continue2_pos + 1] = (export_continue - (jz_export_continue2_pos + 2)) as u8;
        code[jz_export_continue3_pos + 1] = (export_continue - (jz_export_continue3_pos + 2)) as u8;

        // inc edx
        code.extend_from_slice(&[0xFF, 0xC2]);
        // jmp .export_loop
        let jmp_exp_loop_rel = (export_loop as i32) - (code.len() as i32 + 5);
        code.push(0xE9);
        code.extend_from_slice(&jmp_exp_loop_rel.to_le_bytes());

        // .exports_done:
        let exports_done = code.len();
        let rel1 = (exports_done as i32) - (jge_exports_done_pos as i32 + 6);
        code[jge_exports_done_pos + 2..jge_exports_done_pos + 6]
            .copy_from_slice(&rel1.to_le_bytes());
        let rel2 = (exports_done as i32) - (jnz_both_found_pos as i32 + 6);
        code[jnz_both_found_pos + 2..jnz_both_found_pos + 6].copy_from_slice(&rel2.to_le_bytes());
    } else {
        // No hooks: only need OpenFileMappingA and MapViewOfFile.
        // .export_next:
        let export_next = code.len();
        code[jmp_export_next1_pos + 1] = (export_next - (jmp_export_next1_pos + 2)) as u8;
        code[jmp_export_next2_pos + 1] = (export_next - (jmp_export_next2_pos + 2)) as u8;

        // Check if we found both.
        // test r8, r8
        code.extend_from_slice(&[0x4D, 0x85, 0xC0]);
        // jz .export_continue
        let jz_export_continue_pos = code.len();
        code.extend_from_slice(&[0x74, 0x00]); // jz rel8 placeholder
        // test r9, r9
        code.extend_from_slice(&[0x4D, 0x85, 0xC9]);
        // jnz .exports_done  ; both found!
        let jnz_both_found_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jnz rel32 placeholder

        // .export_continue:
        let export_continue = code.len();
        code[jz_export_continue_pos + 1] = (export_continue - (jz_export_continue_pos + 2)) as u8;

        // inc edx
        code.extend_from_slice(&[0xFF, 0xC2]);
        // jmp .export_loop
        let jmp_exp_loop_rel = (export_loop as i32) - (code.len() as i32 + 5);
        code.push(0xE9);
        code.extend_from_slice(&jmp_exp_loop_rel.to_le_bytes());

        // .exports_done:
        let exports_done = code.len();
        let rel1 = (exports_done as i32) - (jge_exports_done_pos as i32 + 6);
        code[jge_exports_done_pos + 2..jge_exports_done_pos + 6]
            .copy_from_slice(&rel1.to_le_bytes());
        let rel2 = (exports_done as i32) - (jnz_both_found_pos as i32 + 6);
        code[jnz_both_found_pos + 2..jnz_both_found_pos + 6].copy_from_slice(&rel2.to_le_bytes());
    } // end if/else need_hook_funcs

    // Save kernel32 base for persistent function resolution (before popping).
    if persistent_data_va.is_some() {
        // mov [rbp - 0x68], r15  ; save kernel32 base to stack local
        code.extend_from_slice(&[0x4C, 0x89, 0x7D, 0x98]);
    }

    // Pop saved values.
    // pop r15  ; kernel32 base (no longer needed)
    code.extend_from_slice(&[0x41, 0x5F]);
    // pop rax  ; AddressOfFunctions RVA (discard)
    code.push(0x58);
    // pop r14  ; SHM ID
    code.extend_from_slice(&[0x41, 0x5E]);

    // Check we resolved both functions.
    // test r8, r8
    code.extend_from_slice(&[0x4D, 0x85, 0xC0]);
    // jz .epilogue
    let jz_epilogue_nofunc_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder
    // test r9, r9
    code.extend_from_slice(&[0x4D, 0x85, 0xC9]);
    // jz .epilogue
    let jz_epilogue_nofunc2_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // ============================================================
    // Step 6: Call OpenFileMappingA(FILE_MAP_WRITE, FALSE, name)
    // Microsoft x64: rcx, rdx, r8, r9 + 32-byte shadow space
    // ============================================================

    // Save r8 (OpenFileMappingA) and r9 (MapViewOfFile) before calling.
    // push r8
    code.extend_from_slice(&[0x41, 0x50]);
    // push r9
    code.extend_from_slice(&[0x41, 0x51]);

    // sub rsp, 32  ; shadow space
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x20]);

    // FILE_MAP_WRITE = 0x0002
    // mov ecx, 2  ; dwDesiredAccess = FILE_MAP_WRITE
    code.extend_from_slice(&[0xB9, 0x02, 0x00, 0x00, 0x00]);
    // xor edx, edx  ; bInheritHandle = FALSE
    code.extend_from_slice(&[0x31, 0xD2]);
    // lea r8, [rbp - LOCAL_SIZE + 32]  ; name buffer ("afl_shm_<id>")
    // The name is at [rsp_after_prolog + 32] = [rbp - LOCAL_SIZE + 32]
    // But rbp was set before pushes and sub rsp, LOCAL_SIZE.
    // After function prolog: rbp = original rsp - 8 (push rbp)
    // pushes: rbx, rsi, rdi, r12, r13, r14, r15 = 7 more pushes = 56 bytes
    // sub rsp, 560 more bytes
    // Compute name buffer address from rbp: [rbp - (8 + 56 + LOCAL_SIZE) + 32].
    let name_rbp_offset: i32 = -(8 + 56 + LOCAL_SIZE as i32) + 32; // -592 (will be large neg)
    // lea r8, [rbp + name_rbp_offset]
    code.extend_from_slice(&[0x4C, 0x8D, 0x85]);
    code.extend_from_slice(&name_rbp_offset.to_le_bytes());

    // Pop saved function pointers back from stack into registers, then store
    // them in the local frame for easy access during the API calls below.
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20]); // add rsp, 32
    code.extend_from_slice(&[0x41, 0x59]); // pop r9
    code.extend_from_slice(&[0x41, 0x58]); // pop r8

    // Store function pointers in stack locals (rbp-relative).
    // mov [rbp - 0x48], r8  ; OpenFileMappingA
    code.extend_from_slice(&[0x4C, 0x89, 0x45, 0xB8]); // [rbp - 0x48]
    // mov [rbp - 0x50], r9  ; MapViewOfFile
    code.extend_from_slice(&[0x4C, 0x89, 0x4D, 0xB0]); // [rbp - 0x50]

    // Now call OpenFileMappingA(FILE_MAP_WRITE, FALSE, name_ptr)
    // sub rsp, 32  ; shadow space
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x20]);
    // mov ecx, 2  ; FILE_MAP_WRITE
    code.extend_from_slice(&[0xB9, 0x02, 0x00, 0x00, 0x00]);
    // xor edx, edx  ; FALSE
    code.extend_from_slice(&[0x31, 0xD2]);
    // lea r8, [rbp + name_rbp_offset]  ; name string
    code.extend_from_slice(&[0x4C, 0x8D, 0x85]);
    code.extend_from_slice(&name_rbp_offset.to_le_bytes());
    // call [rbp - 0x48]  ; OpenFileMappingA
    code.extend_from_slice(&[0xFF, 0x55, 0xB8]);
    // add rsp, 32
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20]);

    // Check return value (HANDLE in rax). NULL = failure.
    // test rax, rax
    code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    // jz .epilogue
    let jz_epilogue_open_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // ============================================================
    // Step 7: Call MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, 65536)
    // ============================================================

    // mov rbx, rax  ; save handle
    code.extend_from_slice(&[0x48, 0x89, 0xC3]);

    // sub rsp, 48  ; shadow space (32) + 5th arg on stack (8) + alignment (8)
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x30]);
    // mov rcx, rbx  ; hFileMappingObject
    code.extend_from_slice(&[0x48, 0x89, 0xD9]);
    // mov edx, 2  ; FILE_MAP_WRITE
    code.extend_from_slice(&[0xBA, 0x02, 0x00, 0x00, 0x00]);
    // xor r8d, r8d  ; dwFileOffsetHigh = 0
    code.extend_from_slice(&[0x45, 0x31, 0xC0]);
    // xor r9d, r9d  ; dwFileOffsetLow = 0
    code.extend_from_slice(&[0x45, 0x31, 0xC9]);
    // mov qword [rsp + 32], 65536  ; dwNumberOfBytesToMap
    code.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, 0x20, 0x00, 0x00, 0x01, 0x00]);
    // call [rbp - 0x50]  ; MapViewOfFile
    code.extend_from_slice(&[0xFF, 0x55, 0xB0]);
    // add rsp, 48
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x30]);

    // Check return value. NULL = failure.
    // test rax, rax
    code.extend_from_slice(&[0x48, 0x85, 0xC0]);
    // jz .epilogue
    let jz_epilogue_map_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // Store SHM pointer: mov [r12], rax
    code.extend_from_slice(&[0x49, 0x89, 0x04, 0x24]);

    // ============================================================
    // Step 7b (optional): Resolve ReadFile, WriteFile, ExitProcess
    // for PE persistent mode. Re-walks kernel32 exports using saved base.
    // ============================================================

    if let Some(p_data_va) = persistent_data_va {
        emit_pe64_persistent_resolve(&mut code, init_va, p_data_va);
    }

    // ============================================================
    // Step 8 (optional): Resolve hook library handlers via
    // LoadLibraryA + GetProcAddress (only when hook_library is set).
    // LoadLibraryA is at [rbp - 0x58], GetProcAddress at [rbp - 0x60].
    // ============================================================

    // LEA placeholders for embedded strings (library path, handler names).
    // We store them as RIP-relative LEAs that get fixed up at the end.
    let mut hook_string_fixups: Vec<(usize, usize)> = Vec::new(); // (lea_pos, string_index)

    let jz_epilogue_hook_positions: Vec<usize>;
    if let Some(hook_lib) = hook_library {
        let mut hook_epilogue_fixups: Vec<usize> = Vec::new();

        // Check LoadLibraryA was resolved.
        // cmp qword [rbp - 0x58], 0
        code.extend_from_slice(&[0x48, 0x83, 0x7D, 0xA8, 0x00]);
        // jz .epilogue  ; LoadLibraryA not found — skip hooks
        let jz_hook_skip1 = code.len();
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
        hook_epilogue_fixups.push(jz_hook_skip1);

        // Check GetProcAddress was resolved.
        // cmp qword [rbp - 0x60], 0
        code.extend_from_slice(&[0x48, 0x83, 0x7D, 0xA0, 0x00]);
        // jz .epilogue
        let jz_hook_skip2 = code.len();
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
        hook_epilogue_fixups.push(jz_hook_skip2);

        // Call LoadLibraryA(library_path_ptr).
        // lea rcx, [rip + library_path_string]  ; placeholder, fixed up later
        let lea_libpath_pos = code.len();
        code.extend_from_slice(&[0x48, 0x8D, 0x0D, 0x00, 0x00, 0x00, 0x00]);
        hook_string_fixups.push((lea_libpath_pos, 0)); // string index 0 = library path

        // sub rsp, 32  ; shadow space
        code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x20]);
        // call [rbp - 0x58]  ; LoadLibraryA
        code.extend_from_slice(&[0xFF, 0x55, 0xA8]);
        // add rsp, 32
        code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20]);

        // Check return (HMODULE in rax). NULL = failure.
        // test rax, rax
        code.extend_from_slice(&[0x48, 0x85, 0xC0]);
        // jz .epilogue
        let jz_hook_skip3 = code.len();
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
        hook_epilogue_fixups.push(jz_hook_skip3);

        // mov rbx, rax  ; rbx = HMODULE of hook library
        code.extend_from_slice(&[0x48, 0x89, 0xC3]);

        // For each handler: call GetProcAddress(rbx, symbol_name_ptr)
        // and store result at the handler's data slot VA.
        for (i, (_name, slot_va)) in hook_lib.handlers.iter().enumerate() {
            // lea rdx, [rip + handler_name_string]  ; placeholder
            let lea_sym_pos = code.len();
            code.extend_from_slice(&[0x48, 0x8D, 0x15, 0x00, 0x00, 0x00, 0x00]);
            hook_string_fixups.push((lea_sym_pos, i + 1)); // string index i+1

            // mov rcx, rbx  ; HMODULE
            code.extend_from_slice(&[0x48, 0x89, 0xD9]);
            // sub rsp, 32
            code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x20]);
            // call [rbp - 0x60]  ; GetProcAddress
            code.extend_from_slice(&[0xFF, 0x55, 0xA0]);
            // add rsp, 32
            code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20]);

            // test rax, rax  ; check if symbol was found
            code.extend_from_slice(&[0x48, 0x85, 0xC0]);
            // jz .skip_store  ; skip if NULL
            let jz_skip_store = code.len();
            code.extend_from_slice(&[0x74, 0x00]); // jz rel8 placeholder

            // Store resolved function pointer at data slot VA (RIP-relative for ASLR).
            // lea r13, [rip + disp32]
            let lea_slot_pos2 = code.len();
            code.extend_from_slice(&[0x4C, 0x8D, 0x2D, 0x00, 0x00, 0x00, 0x00]);
            let lea_slot_rip2 = init_va + code.len() as u64;
            let slot_disp2 = (*slot_va as i64 - lea_slot_rip2 as i64) as i32;
            code[lea_slot_pos2 + 3..lea_slot_pos2 + 7].copy_from_slice(&slot_disp2.to_le_bytes());
            // mov [r13], rax
            code.extend_from_slice(&[0x49, 0x89, 0x45, 0x00]);

            // .skip_store:
            let skip_store = code.len();
            code[jz_skip_store + 1] = (skip_store - (jz_skip_store + 2)) as u8;
        }

        jz_epilogue_hook_positions = hook_epilogue_fixups;
    } else {
        jz_epilogue_hook_positions = Vec::new();
    }

    // ============================================================
    // Epilogue: restore registers and jump to original entry point
    // ============================================================

    // .epilogue label: marks end of main code path.
    // The actual epilogue fixups happen after real_epilogue is defined below.
    let _epilogue = code.len();

    // Fix up .next_env_var jumps (jne rel32: opcode 0F 85 at pos, disp32 at pos+2..pos+6).
    // .next_env_var: advance rsi past current env var (scan for \0\0 then +2)
    let next_env_var = code.len();
    let fix_jne_rel32 = |code: &mut Vec<u8>, pos: usize| {
        let rel = (next_env_var as i32) - (pos as i32 + 6);
        code[pos + 2..pos + 6].copy_from_slice(&rel.to_le_bytes());
    };
    fix_jne_rel32(&mut code, jne_next1_pos);
    fix_jne_rel32(&mut code, jne_next2_pos);
    fix_jne_rel32(&mut code, jne_next3_pos);
    fix_jne_rel32(&mut code, jne_next4_pos);

    // Skip to next \0 (wide null) in the environment block.
    // .skip_wchar:
    let skip_wchar = code.len();
    // movzx eax, word [rsi]
    code.extend_from_slice(&[0x0F, 0xB7, 0x06]);
    // add rsi, 2
    code.extend_from_slice(&[0x48, 0x83, 0xC6, 0x02]);
    // test ax, ax
    code.extend_from_slice(&[0x66, 0x85, 0xC0]);
    // jnz .skip_wchar
    let jnz_skip_rel = (skip_wchar as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0x75, jnz_skip_rel as u8]);
    // jmp .env_scan_loop  ; try next var
    let jmp_env_scan_rel = (env_scan_loop as i32) - (code.len() as i32 + 5);
    code.push(0xE9);
    code.extend_from_slice(&jmp_env_scan_rel.to_le_bytes());

    // Now the actual epilogue code (restore + jump).
    // Note: epilogue label is set above, but the env_var skip code is between.
    // The jz/je jumps target the epilogue label set earlier. The actual
    // restore + jump code is emitted here as a separate section.

    // Hook-only resolve block.
    let hook_resolve_label: Option<usize>;
    let mut hr_fixup_to_real_epilogue: Vec<usize> = Vec::new();

    if let Some(hook_lib) = hook_library {
        let (label, hr_fixups) = emit_pe64_hook_only_resolve(&mut code, init_va, hook_lib);
        hook_resolve_label = Some(label);
        hr_fixup_to_real_epilogue = hr_fixups;
    } else {
        hook_resolve_label = None;
    }

    // .real_epilogue: restore registers and jump to original entry point
    let real_epilogue = code.len();

    // Restore stack.
    code.extend_from_slice(&[0x48, 0x81, 0xC4]); // add rsp, LOCAL_SIZE
    code.extend_from_slice(&LOCAL_SIZE.to_le_bytes());
    code.extend_from_slice(&[0x41, 0x5F]); // pop r15
    code.extend_from_slice(&[0x41, 0x5E]); // pop r14
    code.extend_from_slice(&[0x41, 0x5D]); // pop r13
    code.extend_from_slice(&[0x41, 0x5C]); // pop r12
    code.push(0x5F); // pop rdi
    code.push(0x5E); // pop rsi
    code.push(0x5B); // pop rbx
    code.push(0x5D); // pop rbp

    // JMP to original entry point.
    code.push(0xE9);
    let jmp_entry_rip = init_va + code.len() as u64 + 4;
    let jmp_entry_rel = (original_entry as i64 - jmp_entry_rip as i64) as i32;
    code.extend_from_slice(&jmp_entry_rel.to_le_bytes());

    // Fix up all early-exit epilogue jumps.
    // When hooks are present, early SHM failures go to hook_resolve_label
    // (which does its own kernel32 walk for LoadLibraryA/GetProcAddress).
    // When no hooks, they go straight to real_epilogue.
    let early_exit_target = hook_resolve_label.unwrap_or(real_epilogue);

    let fix_jmp_rel32 = |code: &mut Vec<u8>, pos: usize, target: usize| {
        let rel = (target as i32) - (pos as i32 + 6);
        code[pos + 2..pos + 6].copy_from_slice(&rel.to_le_bytes());
    };

    // SHM-path failures: go to hook resolve (or real_epilogue if no hooks)
    fix_jmp_rel32(&mut code, jz_epilogue_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_epilogue_env_pos, early_exit_target);
    fix_jmp_rel32(&mut code, je_epilogue_k32_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_no_exports_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_epilogue_nofunc_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_epilogue_nofunc2_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_epilogue_open_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_epilogue_map_pos, early_exit_target);

    // Hook resolution failures within the main SHM path: go to real_epilogue
    for &pos in &jz_epilogue_hook_positions {
        fix_jmp_rel32(&mut code, pos, real_epilogue);
    }

    // Hook-only resolution failures: go to real_epilogue
    for &pos in &hr_fixup_to_real_epilogue {
        fix_jmp_rel32(&mut code, pos, real_epilogue);
    }

    // Embed the wide-string needle "__AFL_SHM_ID=" after the code.
    let needle_offset = code.len();
    let needle_rip = init_va + (lea_needle_pos + 7) as u64;
    let needle_delta = (init_va + needle_offset as u64) as i64 - needle_rip as i64;
    code[lea_needle_pos + 3..lea_needle_pos + 7]
        .copy_from_slice(&(needle_delta as i32).to_le_bytes());

    // "__AFL_SHM_ID=" as UTF-16LE (13 wchars = 26 bytes)
    for ch in b"__AFL_SHM_ID=" {
        code.push(*ch);
        code.push(0x00);
    }

    // Embed hook library strings: [0] = library path, [1..] = handler symbol names.
    // Build a table of string offsets for fixup.
    let mut hook_string_offsets: Vec<usize> = Vec::new();
    if let Some(hook_lib) = hook_library {
        // Align before strings.
        while code.len() % 4 != 0 {
            code.push(0x00);
        }

        // String 0: library path (null-terminated ASCII).
        hook_string_offsets.push(code.len());
        code.extend_from_slice(hook_lib.library_path.as_bytes());
        code.push(0x00);

        // Strings 1..N: handler symbol names (null-terminated ASCII).
        for (name, _slot_va) in &hook_lib.handlers {
            hook_string_offsets.push(code.len());
            code.extend_from_slice(name.as_bytes());
            code.push(0x00);
        }

        // Fix up all RIP-relative LEA references to embedded strings.
        for &(lea_pos, string_idx) in &hook_string_fixups {
            let string_offset = hook_string_offsets[string_idx];
            let lea_rip = init_va + (lea_pos + 7) as u64; // RIP after the LEA instruction
            let string_va = init_va + string_offset as u64;
            let delta = (string_va as i64 - lea_rip as i64) as i32;
            code[lea_pos + 3..lea_pos + 7].copy_from_slice(&delta.to_le_bytes());
        }
    }

    // Pad to 8-byte alignment.
    while code.len() % 8 != 0 {
        code.push(0x00);
    }

    Ok(crate::trampoline::InitCode {
        va: init_va,
        code,
        entry_va,
    })
}

/// Generate PE32 (x86 32-bit) init code for SHM attachment.
///
/// Same strategy as `generate_pe_init_code` but for 32-bit x86:
/// - PEB via fs:[0x30] (not gs:[0x60])
/// - PEB.ProcessParameters at PEB+0x10 (not +0x20)
/// - RTL_USER_PROCESS_PARAMETERS.Environment at +0x48 (not +0x80)
/// - LDR at PEB+0x0C (not +0x18)
/// - InLoadOrderModuleList at Ldr+0x0C (not +0x10)
/// - LDR_DATA_TABLE_ENTRY: DllBase at +0x18 (not +0x30), BaseDllName at +0x2C (not +0x58)
/// - BaseDllName.Buffer at +0x04 from UNICODE_STRING (not +0x08)
/// - 32-bit registers, cdecl calling convention
/// - No RIP-relative addressing; use absolute immediates for data_va
/// - Export directory at OptHdr+0x60 (PE32, not PE32+ which uses +0x70)
#[allow(clippy::too_many_arguments)]
fn generate_pe32_init_code(
    init_va: u64,
    data_va: u64,
    original_entry: u64,
    hook_library: Option<&PeHookLibraryInfo>,
    _persistent_data_va: Option<u64>,
) -> Result<crate::trampoline::InitCode> {
    let mut code = Vec::with_capacity(1024);
    let entry_va = init_va;

    // ============================================================
    // Windows x86 (32-bit) init code -- PEB-based, import-free
    //
    // Strategy (same as x64):
    // 1. Access PEB via fs:[0x30]
    // 2. Scan environment for L"__AFL_SHM_ID="
    // 3. Walk PEB_LDR_DATA to find kernel32.dll
    // 4. Parse exports via DJB2 hash to resolve OpenFileMappingA, MapViewOfFile
    // 5. Call them to set up SHM
    // ============================================================

    // Save callee-saved registers (cdecl: ebx, esi, edi, ebp).
    code.push(0x55); // push ebp
    code.extend_from_slice(&[0x89, 0xE5]); // mov ebp, esp
    code.push(0x53); // push ebx
    code.push(0x56); // push esi
    code.push(0x57); // push edi

    // Allocate local space on stack.
    // Entry ESP is 4-byte aligned. After push ebp + mov ebp,esp + 3 pushes = 12 bytes.
    // We need space for:
    //   - SHM name buffer (32 bytes at [ebp - LOCAL_SIZE + 0])
    //   - resolved function pointers (16 bytes: OpenFileMappingA, MapViewOfFile, LoadLibraryA, GetProcAddress)
    //   - scratch space
    const LOCAL_SIZE: u32 = 128;
    // sub esp, LOCAL_SIZE
    code.extend_from_slice(&[0x81, 0xEC]);
    code.extend_from_slice(&LOCAL_SIZE.to_le_bytes());

    // Store data_va as an absolute immediate in esi.
    // mov esi, imm32  (data_va)
    // In PE32, addresses fit in 32 bits.
    code.push(0xBE); // mov esi, imm32
    code.extend_from_slice(&(data_va as u32).to_le_bytes());

    // ============================================================
    // Step 1: Access PEB -> ProcessParameters -> Environment
    // PEB at fs:[0x30] on x86 Windows
    // PEB.ProcessParameters at PEB+0x10
    // RTL_USER_PROCESS_PARAMETERS.Environment at +0x48
    // ============================================================

    // mov eax, fs:[0x30]  ; PEB
    code.extend_from_slice(&[0x64, 0xA1, 0x30, 0x00, 0x00, 0x00]);
    // mov eax, [eax + 0x10]  ; PEB.ProcessParameters
    code.extend_from_slice(&[0x8B, 0x40, 0x10]);
    // mov edi, [eax + 0x48]  ; ProcessParameters.Environment (wide string block)
    code.extend_from_slice(&[0x8B, 0x78, 0x48]);
    // test edi, edi
    code.extend_from_slice(&[0x85, 0xFF]);
    // jz .epilogue  ; no environment
    let jz_epilogue_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // ============================================================
    // Step 2: Scan environment for L"__AFL_SHM_ID="
    // ============================================================

    // Embed the needle at the end of the init code and reference it via
    // absolute address (no RIP-relative on x86). Placeholder patched later.
    let lea_needle_pos = code.len();
    code.push(0xBB); // mov ebx, imm32 (needle VA placeholder)
    code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    // edi = pointer into environment block (wide strings)
    // ebx = pointer to needle

    // .env_scan_loop:
    let env_scan_loop = code.len();

    // Check for end of environment block: if [edi] == 0x0000, done.
    // movzx eax, word [edi]
    code.extend_from_slice(&[0x0F, 0xB7, 0x07]);
    // test ax, ax
    code.extend_from_slice(&[0x66, 0x85, 0xC0]);
    // jz .epilogue_env (empty string = end of block)
    let jz_epilogue_env_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // Compare first 8 bytes: "__AF" as wide = 5F 00 5F 00 41 00 46 00
    // mov eax, [edi]
    code.extend_from_slice(&[0x8B, 0x07]);
    // cmp eax, [ebx]
    code.extend_from_slice(&[0x3B, 0x03]);
    // jne .next_env_var
    let jne_next1_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // Compare bytes 4..8
    // mov eax, [edi + 4]
    code.extend_from_slice(&[0x8B, 0x47, 0x04]);
    // cmp eax, [ebx + 4]
    code.extend_from_slice(&[0x3B, 0x43, 0x04]);
    // jne .next_env_var
    let jne_next2_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // Compare bytes 8..12
    // mov eax, [edi + 8]
    code.extend_from_slice(&[0x8B, 0x47, 0x08]);
    // cmp eax, [ebx + 8]
    code.extend_from_slice(&[0x3B, 0x43, 0x08]);
    // jne .next_env_var
    let jne_next3_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // Compare bytes 12..16
    // mov eax, [edi + 12]
    code.extend_from_slice(&[0x8B, 0x47, 0x0C]);
    // cmp eax, [ebx + 12]
    code.extend_from_slice(&[0x3B, 0x43, 0x0C]);
    // jne .next_env_var
    let jne_next4_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // Compare bytes 16..20
    // mov eax, [edi + 16]
    code.extend_from_slice(&[0x8B, 0x47, 0x10]);
    // cmp eax, [ebx + 16]
    code.extend_from_slice(&[0x3B, 0x43, 0x10]);
    // jne .next_env_var
    let jne_next5_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // Compare bytes 20..24
    // mov eax, [edi + 20]
    code.extend_from_slice(&[0x8B, 0x47, 0x14]);
    // cmp eax, [ebx + 20]
    code.extend_from_slice(&[0x3B, 0x43, 0x14]);
    // jne .next_env_var
    let jne_next6_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // Compare bytes 24..26: "=" as wide = 3D 00
    // movzx eax, word [edi + 24]
    code.extend_from_slice(&[0x0F, 0xB7, 0x47, 0x18]);
    // cmp ax, 0x003D
    code.extend_from_slice(&[0x66, 0x3D, 0x3D, 0x00]);
    // jne .next_env_var
    let jne_next7_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jne rel32 placeholder

    // Parse decimal SHM ID and build SHM name on stack.
    emit_pe32_shm_name_build(&mut code);

    // Walk PEB_LDR_DATA to find kernel32.dll.
    let je_epilogue_k32_pos = emit_pe32_kernel32_walk(&mut code);

    // ============================================================
    // Step 5: Parse kernel32.dll export directory
    // esi = kernel32 base address
    //
    // PE32 (not PE32+):
    //   e_lfanew at base+0x3C
    //   Optional header starts at PE+0x18
    //   Export directory RVA = DataDirectory[0].VA at OptHdr+0x60
    //     (OptHdr offset from PE signature = 0x18, DataDir[0] at +0x60 from OptHdr start,
    //      so from PE header: 0x18 + 0x60 = 0x78)
    //   Export directory:
    //     +0x18: NumberOfNames
    //     +0x1C: AddressOfFunctions (RVA)
    //     +0x20: AddressOfNames (RVA)
    //     +0x24: AddressOfNameOrdinals (RVA)
    // ============================================================

    // mov eax, [esi + 0x3C]  ; e_lfanew
    code.extend_from_slice(&[0x8B, 0x46, 0x3C]);
    // lea ebx, [esi + eax]  ; PE header
    code.extend_from_slice(&[0x8D, 0x1C, 0x06]);
    // mov eax, [ebx + 0x78]  ; Export directory RVA (PE32: OptHdr+0x60 = PE+0x18+0x60 = PE+0x78)
    code.extend_from_slice(&[0x8B, 0x43, 0x78]);
    // test eax, eax
    code.extend_from_slice(&[0x85, 0xC0]);
    // jz .epilogue  ; no exports
    let jz_no_exports_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // lea ebx, [esi + eax]  ; export directory (reuse ebx)
    code.extend_from_slice(&[0x8D, 0x1C, 0x06]);

    // mov ecx, [ebx + 0x18]  ; NumberOfNames
    code.extend_from_slice(&[0x8B, 0x4B, 0x18]);
    // mov eax, [ebx + 0x20]  ; AddressOfNames RVA
    code.extend_from_slice(&[0x8B, 0x43, 0x20]);
    // Save AddressOfNames absolute to stack: [ebp - 0x08]
    // lea eax, [esi + eax]
    code.extend_from_slice(&[0x8D, 0x04, 0x06]);
    // mov [ebp - 0x08], eax  ; AddressOfNames
    code.extend_from_slice(&[0x89, 0x45, 0xF8]);

    // mov eax, [ebx + 0x24]  ; AddressOfNameOrdinals RVA
    code.extend_from_slice(&[0x8B, 0x43, 0x24]);
    // lea eax, [esi + eax]
    code.extend_from_slice(&[0x8D, 0x04, 0x06]);
    // mov [ebp - 0x0C], eax  ; AddressOfNameOrdinals
    code.extend_from_slice(&[0x89, 0x45, 0xF4]);

    // mov eax, [ebx + 0x1C]  ; AddressOfFunctions RVA
    code.extend_from_slice(&[0x8B, 0x43, 0x1C]);
    // lea eax, [esi + eax]
    code.extend_from_slice(&[0x8D, 0x04, 0x06]);
    // mov [ebp - 0x10], eax  ; AddressOfFunctions
    code.extend_from_slice(&[0x89, 0x45, 0xF0]);

    // DJB2 hashes (same values regardless of bitness).
    let hash_open_file_mapping: u32 = djb2_hash(b"OpenFileMappingA");
    let hash_map_view_of_file: u32 = djb2_hash(b"MapViewOfFile");
    let hash_load_library_a: u32 = djb2_hash(b"LoadLibraryA");
    let hash_get_proc_address: u32 = djb2_hash(b"GetProcAddress");
    let need_hook_funcs = hook_library.is_some();

    // Resolved function pointers stored at:
    // [ebp - 0x14] = OpenFileMappingA
    // [ebp - 0x18] = MapViewOfFile
    // [ebp - 0x1C] = LoadLibraryA (if hooks)
    // [ebp - 0x20] = GetProcAddress (if hooks)
    // Init to 0.
    // mov dword [ebp - 0x14], 0
    code.extend_from_slice(&[0xC7, 0x45, 0xEC, 0x00, 0x00, 0x00, 0x00]);
    // mov dword [ebp - 0x18], 0
    code.extend_from_slice(&[0xC7, 0x45, 0xE8, 0x00, 0x00, 0x00, 0x00]);
    if need_hook_funcs {
        // mov dword [ebp - 0x1C], 0
        code.extend_from_slice(&[0xC7, 0x45, 0xE4, 0x00, 0x00, 0x00, 0x00]);
        // mov dword [ebp - 0x20], 0
        code.extend_from_slice(&[0xC7, 0x45, 0xE0, 0x00, 0x00, 0x00, 0x00]);
    }

    // xor edx, edx  ; index = 0
    code.extend_from_slice(&[0x31, 0xD2]);

    // .export_loop:
    let export_loop = code.len();
    // cmp edx, ecx  ; index >= NumberOfNames?
    code.extend_from_slice(&[0x39, 0xCA]);
    // jge .exports_done
    let jge_exports_done_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x8D, 0x00, 0x00, 0x00, 0x00]); // jge rel32 placeholder

    // mov eax, [ebp - 0x08]  ; AddressOfNames
    code.extend_from_slice(&[0x8B, 0x45, 0xF8]);
    // mov eax, [eax + edx*4]  ; name RVA
    code.extend_from_slice(&[0x8B, 0x04, 0x90]);
    // lea edi, [esi + eax]  ; name string
    code.extend_from_slice(&[0x8D, 0x3C, 0x06]);

    // Compute DJB2 hash of the name string.
    // push edx  ; save index
    code.push(0x52);
    // push ecx  ; save count
    code.push(0x51);
    // mov eax, 5381  ; hash = 5381
    code.push(0xB8);
    code.extend_from_slice(&5381u32.to_le_bytes());

    // .hash_loop:
    let hash_loop = code.len();
    // movzx ebx, byte [edi]
    code.extend_from_slice(&[0x0F, 0xB6, 0x1F]);
    // test bl, bl
    code.extend_from_slice(&[0x84, 0xDB]);
    // jz .hash_done
    let jz_hash_done_pos = code.len();
    code.extend_from_slice(&[0x74, 0x00]); // jz rel8 placeholder
    // imul eax, eax, 33
    code.extend_from_slice(&[0x6B, 0xC0, 0x21]);
    // add eax, ebx
    code.extend_from_slice(&[0x01, 0xD8]);
    // inc edi
    code.push(0x47);
    // jmp .hash_loop
    let jmp_hash_rel = (hash_loop as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0xEB, jmp_hash_rel as u8]);

    // .hash_done:
    let hash_done = code.len();
    code[jz_hash_done_pos + 1] = (hash_done - (jz_hash_done_pos + 2)) as u8;

    // pop ecx  ; restore count
    code.push(0x59);
    // pop edx  ; restore index
    code.push(0x5A);

    // Compare hash with known values.
    // Helper: resolve function address for current index.
    // ordinal = [AddressOfNameOrdinals + index*2]
    // func RVA = [AddressOfFunctions + ordinal*4]
    // func addr = kernel32_base + func_rva

    // cmp eax, hash_open_file_mapping
    code.push(0x3D);
    code.extend_from_slice(&hash_open_file_mapping.to_le_bytes());
    // jne .check_map_view
    let jne_check_map_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // Resolve OpenFileMappingA -> [ebp - 0x14]
    // mov ebx, [ebp - 0x0C]  ; AddressOfNameOrdinals
    code.extend_from_slice(&[0x8B, 0x5D, 0xF4]);
    // movzx eax, word [ebx + edx*2]  ; ordinal
    code.extend_from_slice(&[0x0F, 0xB7, 0x04, 0x53]);
    // mov ebx, [ebp - 0x10]  ; AddressOfFunctions
    code.extend_from_slice(&[0x8B, 0x5D, 0xF0]);
    // mov eax, [ebx + eax*4]  ; function RVA
    code.extend_from_slice(&[0x8B, 0x04, 0x83]);
    // lea eax, [esi + eax]  ; absolute address
    code.extend_from_slice(&[0x8D, 0x04, 0x06]);
    // mov [ebp - 0x14], eax
    code.extend_from_slice(&[0x89, 0x45, 0xEC]);
    // jmp .export_next
    let jmp_export_next1_pos = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp rel8 placeholder

    // .check_map_view:
    let check_map_view = code.len();
    code[jne_check_map_pos + 1] = (check_map_view - (jne_check_map_pos + 2)) as u8;

    // cmp eax, hash_map_view_of_file
    code.push(0x3D);
    code.extend_from_slice(&hash_map_view_of_file.to_le_bytes());
    // jne .check_load_library
    let jne_check_loadlib_pos = code.len();
    code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

    // Resolve MapViewOfFile -> [ebp - 0x18]
    code.extend_from_slice(&[0x8B, 0x5D, 0xF4]); // mov ebx, [ebp - 0x0C]
    code.extend_from_slice(&[0x0F, 0xB7, 0x04, 0x53]); // movzx eax, word [ebx + edx*2]
    code.extend_from_slice(&[0x8B, 0x5D, 0xF0]); // mov ebx, [ebp - 0x10]
    code.extend_from_slice(&[0x8B, 0x04, 0x83]); // mov eax, [ebx + eax*4]
    code.extend_from_slice(&[0x8D, 0x04, 0x06]); // lea eax, [esi + eax]
    code.extend_from_slice(&[0x89, 0x45, 0xE8]); // mov [ebp - 0x18], eax
    let jmp_export_next2_pos = code.len();
    code.extend_from_slice(&[0xEB, 0x00]); // jmp rel8 placeholder

    // .check_load_library:
    let check_load_library = code.len();
    code[jne_check_loadlib_pos + 1] = (check_load_library - (jne_check_loadlib_pos + 2)) as u8;

    if need_hook_funcs {
        // cmp eax, hash_load_library_a
        code.push(0x3D);
        code.extend_from_slice(&hash_load_library_a.to_le_bytes());
        let jne_check_getproc_pos = code.len();
        code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

        // Resolve LoadLibraryA -> [ebp - 0x1C]
        code.extend_from_slice(&[0x8B, 0x5D, 0xF4]);
        code.extend_from_slice(&[0x0F, 0xB7, 0x04, 0x53]);
        code.extend_from_slice(&[0x8B, 0x5D, 0xF0]);
        code.extend_from_slice(&[0x8B, 0x04, 0x83]);
        code.extend_from_slice(&[0x8D, 0x04, 0x06]);
        code.extend_from_slice(&[0x89, 0x45, 0xE4]); // mov [ebp - 0x1C], eax
        let jmp_export_next3_pos = code.len();
        code.extend_from_slice(&[0xEB, 0x00]); // jmp rel8 placeholder

        // .check_get_proc_address:
        let check_get_proc_address = code.len();
        code[jne_check_getproc_pos + 1] =
            (check_get_proc_address - (jne_check_getproc_pos + 2)) as u8;

        // cmp eax, hash_get_proc_address
        code.push(0x3D);
        code.extend_from_slice(&hash_get_proc_address.to_le_bytes());
        let jne_export_next_pos = code.len();
        code.extend_from_slice(&[0x75, 0x00]); // jne rel8 placeholder

        // Resolve GetProcAddress -> [ebp - 0x20]
        code.extend_from_slice(&[0x8B, 0x5D, 0xF4]);
        code.extend_from_slice(&[0x0F, 0xB7, 0x04, 0x53]);
        code.extend_from_slice(&[0x8B, 0x5D, 0xF0]);
        code.extend_from_slice(&[0x8B, 0x04, 0x83]);
        code.extend_from_slice(&[0x8D, 0x04, 0x06]);
        code.extend_from_slice(&[0x89, 0x45, 0xE0]); // mov [ebp - 0x20], eax

        // .export_next:
        let export_next = code.len();
        code[jmp_export_next1_pos + 1] = (export_next - (jmp_export_next1_pos + 2)) as u8;
        code[jmp_export_next2_pos + 1] = (export_next - (jmp_export_next2_pos + 2)) as u8;
        code[jmp_export_next3_pos + 1] = (export_next - (jmp_export_next3_pos + 2)) as u8;
        code[jne_export_next_pos + 1] = (export_next - (jne_export_next_pos + 2)) as u8;

        // Early exit: check if all 4 functions found.
        // cmp dword [ebp - 0x14], 0
        code.extend_from_slice(&[0x83, 0x7D, 0xEC, 0x00]);
        let jz_export_continue_pos = code.len();
        code.extend_from_slice(&[0x74, 0x00]);
        // cmp dword [ebp - 0x18], 0
        code.extend_from_slice(&[0x83, 0x7D, 0xE8, 0x00]);
        let jz_export_continue2_pos = code.len();
        code.extend_from_slice(&[0x74, 0x00]);
        // cmp dword [ebp - 0x1C], 0
        code.extend_from_slice(&[0x83, 0x7D, 0xE4, 0x00]);
        let jz_export_continue3_pos = code.len();
        code.extend_from_slice(&[0x74, 0x00]);
        // cmp dword [ebp - 0x20], 0
        code.extend_from_slice(&[0x83, 0x7D, 0xE0, 0x00]);
        // jnz .exports_done  ; all 4 found!
        let jnz_both_found_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);

        // .export_continue:
        let export_continue = code.len();
        code[jz_export_continue_pos + 1] = (export_continue - (jz_export_continue_pos + 2)) as u8;
        code[jz_export_continue2_pos + 1] = (export_continue - (jz_export_continue2_pos + 2)) as u8;
        code[jz_export_continue3_pos + 1] = (export_continue - (jz_export_continue3_pos + 2)) as u8;

        // inc edx
        code.extend_from_slice(&[0xFF, 0xC2]);
        // jmp .export_loop
        let jmp_exp_loop_rel = (export_loop as i32) - (code.len() as i32 + 5);
        code.push(0xE9);
        code.extend_from_slice(&jmp_exp_loop_rel.to_le_bytes());

        // .exports_done:
        let exports_done = code.len();
        let rel1 = (exports_done as i32) - (jge_exports_done_pos as i32 + 6);
        code[jge_exports_done_pos + 2..jge_exports_done_pos + 6]
            .copy_from_slice(&rel1.to_le_bytes());
        let rel2 = (exports_done as i32) - (jnz_both_found_pos as i32 + 6);
        code[jnz_both_found_pos + 2..jnz_both_found_pos + 6].copy_from_slice(&rel2.to_le_bytes());
    } else {
        // No hooks: only need OpenFileMappingA and MapViewOfFile.
        let export_next = code.len();
        code[jmp_export_next1_pos + 1] = (export_next - (jmp_export_next1_pos + 2)) as u8;
        code[jmp_export_next2_pos + 1] = (export_next - (jmp_export_next2_pos + 2)) as u8;

        // Check if we found both.
        code.extend_from_slice(&[0x83, 0x7D, 0xEC, 0x00]); // cmp dword [ebp - 0x14], 0
        let jz_export_continue_pos = code.len();
        code.extend_from_slice(&[0x74, 0x00]);
        code.extend_from_slice(&[0x83, 0x7D, 0xE8, 0x00]); // cmp dword [ebp - 0x18], 0
        let jnz_both_found_pos = code.len();
        code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]);

        let export_continue = code.len();
        code[jz_export_continue_pos + 1] = (export_continue - (jz_export_continue_pos + 2)) as u8;

        // inc edx
        code.extend_from_slice(&[0xFF, 0xC2]);
        // jmp .export_loop
        let jmp_exp_loop_rel = (export_loop as i32) - (code.len() as i32 + 5);
        code.push(0xE9);
        code.extend_from_slice(&jmp_exp_loop_rel.to_le_bytes());

        // .exports_done:
        let exports_done = code.len();
        let rel1 = (exports_done as i32) - (jge_exports_done_pos as i32 + 6);
        code[jge_exports_done_pos + 2..jge_exports_done_pos + 6]
            .copy_from_slice(&rel1.to_le_bytes());
        let rel2 = (exports_done as i32) - (jnz_both_found_pos as i32 + 6);
        code[jnz_both_found_pos + 2..jnz_both_found_pos + 6].copy_from_slice(&rel2.to_le_bytes());
    }

    // Pop saved data_va. esi was pushed unconditionally before the kernel32
    // export walk, so it must be popped unconditionally here.
    // pop esi  ; restore data_va
    code.push(0x5E);

    // Check we resolved both SHM functions.
    // cmp dword [ebp - 0x14], 0
    code.extend_from_slice(&[0x83, 0x7D, 0xEC, 0x00]);
    let jz_epilogue_nofunc_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    // cmp dword [ebp - 0x18], 0
    code.extend_from_slice(&[0x83, 0x7D, 0xE8, 0x00]);
    let jz_epilogue_nofunc2_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);

    // ============================================================
    // Step 6: Call OpenFileMappingA(FILE_MAP_WRITE, FALSE, name)
    // stdcall: push args right-to-left, callee cleans stack
    // ============================================================

    // lea eax, [ebp + name_ebp_offset]  ; name string buffer
    const LOCAL_SIZE_32: u32 = 128;
    let name_ebp_offset: i32 = -(LOCAL_SIZE_32 as i32);
    code.extend_from_slice(&[0x8D, 0x85]);
    code.extend_from_slice(&name_ebp_offset.to_le_bytes());

    // push eax  ; lpName
    code.push(0x50);
    // push 0  ; bInheritHandle = FALSE
    code.extend_from_slice(&[0x6A, 0x00]);
    // push 2  ; dwDesiredAccess = FILE_MAP_WRITE
    code.extend_from_slice(&[0x6A, 0x02]);
    // call [ebp - 0x14]  ; OpenFileMappingA
    code.extend_from_slice(&[0xFF, 0x55, 0xEC]);
    // add esp, 12  ; NOTE: stdcall callee already cleaned; this is redundant
    code.extend_from_slice(&[0x83, 0xC4, 0x0C]);

    // Check return value (HANDLE in eax). NULL = failure.
    // test eax, eax
    code.extend_from_slice(&[0x85, 0xC0]);
    let jz_epilogue_open_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);

    // ============================================================
    // Step 7: Call MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, 65536)
    // stdcall: 5 args pushed right-to-left, callee cleans stack
    // ============================================================

    // mov ebx, eax  ; save handle
    code.extend_from_slice(&[0x89, 0xC3]);

    // push 65536  ; dwNumberOfBytesToMap
    code.push(0x68);
    code.extend_from_slice(&65536u32.to_le_bytes());
    // push 0  ; dwFileOffsetLow
    code.extend_from_slice(&[0x6A, 0x00]);
    // push 0  ; dwFileOffsetHigh
    code.extend_from_slice(&[0x6A, 0x00]);
    // push 2  ; dwDesiredAccess = FILE_MAP_WRITE
    code.extend_from_slice(&[0x6A, 0x02]);
    // push ebx  ; hFileMappingObject
    code.push(0x53);
    // call [ebp - 0x18]  ; MapViewOfFile
    code.extend_from_slice(&[0xFF, 0x55, 0xE8]);
    // add esp, 20  ; NOTE: stdcall callee already cleaned; this is redundant
    code.extend_from_slice(&[0x83, 0xC4, 0x14]);

    // Check return value. NULL = failure.
    // test eax, eax
    code.extend_from_slice(&[0x85, 0xC0]);
    let jz_epilogue_map_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);

    // Store SHM pointer at data_va.
    // esi = data_va (restored earlier)
    // mov [esi], eax  ; store SHM base pointer (lower 4 bytes)
    code.extend_from_slice(&[0x89, 0x06]);
    // mov dword [esi + 4], 0  ; upper 4 bytes = 0 (32-bit pointer)
    code.extend_from_slice(&[0xC7, 0x46, 0x04, 0x00, 0x00, 0x00, 0x00]);

    // ============================================================
    // Step 8 (optional): Resolve hook library handlers via
    // LoadLibraryA + GetProcAddress (stdcall)
    // ============================================================

    let mut hook_string_fixups: Vec<(usize, usize)> = Vec::new(); // (mov_pos, string_index)
    let jz_epilogue_hook_positions: Vec<usize>;

    if let Some(hook_lib) = hook_library {
        let mut hook_epilogue_fixups: Vec<usize> = Vec::new();

        // Check LoadLibraryA was resolved.
        // cmp dword [ebp - 0x1C], 0
        code.extend_from_slice(&[0x83, 0x7D, 0xE4, 0x00]);
        let jz_hook_skip1 = code.len();
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
        hook_epilogue_fixups.push(jz_hook_skip1);

        // Check GetProcAddress was resolved.
        // cmp dword [ebp - 0x20], 0
        code.extend_from_slice(&[0x83, 0x7D, 0xE0, 0x00]);
        let jz_hook_skip2 = code.len();
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
        hook_epilogue_fixups.push(jz_hook_skip2);

        // Call LoadLibraryA(library_path_ptr).
        // mov eax, imm32  ; library path string VA (placeholder)
        let mov_libpath_pos = code.len();
        code.push(0xB8);
        code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // placeholder
        hook_string_fixups.push((mov_libpath_pos, 0)); // string index 0 = library path

        // push eax  ; lpLibFileName
        code.push(0x50);
        // call [ebp - 0x1C]  ; LoadLibraryA
        code.extend_from_slice(&[0xFF, 0x55, 0xE4]);
        // add esp, 4
        code.extend_from_slice(&[0x83, 0xC4, 0x04]);

        // Check return (HMODULE in eax).
        // test eax, eax
        code.extend_from_slice(&[0x85, 0xC0]);
        let jz_hook_skip3 = code.len();
        code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
        hook_epilogue_fixups.push(jz_hook_skip3);

        // mov ebx, eax  ; ebx = HMODULE of hook library
        code.extend_from_slice(&[0x89, 0xC3]);

        // For each handler: call GetProcAddress(hModule, lpProcName)
        // and store result at the handler's data slot VA.
        for (i, (_name, slot_va)) in hook_lib.handlers.iter().enumerate() {
            // mov eax, imm32  ; handler name string VA (placeholder)
            let mov_sym_pos = code.len();
            code.push(0xB8);
            code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
            hook_string_fixups.push((mov_sym_pos, i + 1));

            // push eax  ; lpProcName
            code.push(0x50);
            // push ebx  ; hModule
            code.push(0x53);
            // call [ebp - 0x20]  ; GetProcAddress
            code.extend_from_slice(&[0xFF, 0x55, 0xE0]);
            // add esp, 8
            code.extend_from_slice(&[0x83, 0xC4, 0x08]);

            // test eax, eax
            code.extend_from_slice(&[0x85, 0xC0]);
            let jz_skip_store = code.len();
            code.extend_from_slice(&[0x74, 0x00]); // jz rel8 placeholder

            // Store resolved function pointer at data slot VA.
            // mov dword [slot_va], eax  (absolute addressing, 32-bit)
            code.push(0xA3); // mov [imm32], eax
            code.extend_from_slice(&(*slot_va as u32).to_le_bytes());

            // .skip_store:
            let skip_store = code.len();
            code[jz_skip_store + 1] = (skip_store - (jz_skip_store + 2)) as u8;
        }

        jz_epilogue_hook_positions = hook_epilogue_fixups;
    } else {
        jz_epilogue_hook_positions = Vec::new();
    }

    // ============================================================
    // Epilogue: next_env_var skip + restore registers + jump to OEP
    // ============================================================

    let _epilogue = code.len();

    // .next_env_var: advance edi past current env var (scan for \0\0 then +2)
    let next_env_var = code.len();
    let fix_jne_rel32 = |code: &mut Vec<u8>, pos: usize| {
        let rel = (next_env_var as i32) - (pos as i32 + 6);
        code[pos + 2..pos + 6].copy_from_slice(&rel.to_le_bytes());
    };
    fix_jne_rel32(&mut code, jne_next1_pos);
    fix_jne_rel32(&mut code, jne_next2_pos);
    fix_jne_rel32(&mut code, jne_next3_pos);
    fix_jne_rel32(&mut code, jne_next4_pos);
    fix_jne_rel32(&mut code, jne_next5_pos);
    fix_jne_rel32(&mut code, jne_next6_pos);
    fix_jne_rel32(&mut code, jne_next7_pos);

    // Skip to next \0 (wide null) in the environment block.
    // .skip_wchar:
    let skip_wchar = code.len();
    // movzx eax, word [edi]
    code.extend_from_slice(&[0x0F, 0xB7, 0x07]);
    // add edi, 2
    code.extend_from_slice(&[0x83, 0xC7, 0x02]);
    // test ax, ax
    code.extend_from_slice(&[0x66, 0x85, 0xC0]);
    // jnz .skip_wchar
    let jnz_skip_rel = (skip_wchar as isize - (code.len() as isize + 2)) as i8;
    code.extend_from_slice(&[0x75, jnz_skip_rel as u8]);
    // jmp .env_scan_loop
    let jmp_env_scan_rel = (env_scan_loop as i32) - (code.len() as i32 + 5);
    code.push(0xE9);
    code.extend_from_slice(&jmp_env_scan_rel.to_le_bytes());

    // Hook-only resolve block.
    let hook_resolve_label: Option<usize>;
    let mut hr_fixup_to_real_epilogue: Vec<usize> = Vec::new();

    if let Some(hook_lib) = hook_library {
        let (label, hr_fixups) = emit_pe32_hook_only_resolve(&mut code, init_va, hook_lib);
        hook_resolve_label = Some(label);
        hr_fixup_to_real_epilogue = hr_fixups;
    } else {
        hook_resolve_label = None;
    }

    // .real_epilogue: restore registers and jump to original entry point
    let real_epilogue = code.len();

    // Restore stack and registers.
    // add esp, LOCAL_SIZE  (or use leave which does mov esp,ebp; pop ebp)
    // We used push ebx, push esi, push edi after push ebp / mov ebp,esp.
    // So: leave restores esp=ebp, pop ebp. But we also pushed ebx, esi, edi.
    // Those are between ebp and the local frame.
    // With ebp frame: esp points to locals. Pop locals, then pop edi, esi, ebx, then leave.

    // add esp, LOCAL_SIZE
    code.extend_from_slice(&[0x81, 0xC4]);
    code.extend_from_slice(&LOCAL_SIZE.to_le_bytes());
    code.push(0x5F); // pop edi
    code.push(0x5E); // pop esi
    code.push(0x5B); // pop ebx
    code.push(0x5D); // pop ebp

    // JMP to original entry point.
    code.push(0xE9);
    let jmp_entry_ip = init_va + code.len() as u64 + 4;
    let jmp_entry_rel = (original_entry as i64 - jmp_entry_ip as i64) as i32;
    code.extend_from_slice(&jmp_entry_rel.to_le_bytes());

    // Fix up all early-exit epilogue jumps.
    let early_exit_target = hook_resolve_label.unwrap_or(real_epilogue);

    let fix_jmp_rel32 = |code: &mut Vec<u8>, pos: usize, target: usize| {
        let rel = (target as i32) - (pos as i32 + 6);
        code[pos + 2..pos + 6].copy_from_slice(&rel.to_le_bytes());
    };

    fix_jmp_rel32(&mut code, jz_epilogue_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_epilogue_env_pos, early_exit_target);
    fix_jmp_rel32(&mut code, je_epilogue_k32_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_no_exports_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_epilogue_nofunc_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_epilogue_nofunc2_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_epilogue_open_pos, early_exit_target);
    fix_jmp_rel32(&mut code, jz_epilogue_map_pos, early_exit_target);

    // Hook resolution failures within the main SHM path: go to real_epilogue
    for &pos in &jz_epilogue_hook_positions {
        fix_jmp_rel32(&mut code, pos, real_epilogue);
    }

    // Hook-only resolution failures: go to real_epilogue
    for &pos in &hr_fixup_to_real_epilogue {
        fix_jmp_rel32(&mut code, pos, real_epilogue);
    }

    // Embed the wide-string needle "__AFL_SHM_ID=" after the code.
    let needle_offset = code.len();
    let needle_va = init_va + needle_offset as u64;
    // Fix up the mov ebx, imm32 at lea_needle_pos.
    code[lea_needle_pos + 1..lea_needle_pos + 5].copy_from_slice(&(needle_va as u32).to_le_bytes());

    // "__AFL_SHM_ID=" as UTF-16LE (13 wchars = 26 bytes)
    for ch in b"__AFL_SHM_ID=" {
        code.push(*ch);
        code.push(0x00);
    }

    // Embed hook library strings: [0] = library path, [1..] = handler symbol names.
    let mut hook_string_offsets: Vec<usize> = Vec::new();
    if let Some(hook_lib) = hook_library {
        while code.len() % 4 != 0 {
            code.push(0x00);
        }

        // String 0: library path
        hook_string_offsets.push(code.len());
        code.extend_from_slice(hook_lib.library_path.as_bytes());
        code.push(0x00);

        // Strings 1..N: handler symbol names
        for (name, _slot_va) in &hook_lib.handlers {
            hook_string_offsets.push(code.len());
            code.extend_from_slice(name.as_bytes());
            code.push(0x00);
        }

        // Fix up mov imm32 references to embedded strings.
        for &(mov_pos, string_idx) in &hook_string_fixups {
            let string_offset = hook_string_offsets[string_idx];
            let string_va = init_va + string_offset as u64;
            code[mov_pos + 1..mov_pos + 5].copy_from_slice(&(string_va as u32).to_le_bytes());
        }
    }

    // Pad to 8-byte alignment.
    while code.len() % 8 != 0 {
        code.push(0x00);
    }

    Ok(crate::trampoline::InitCode {
        va: init_va,
        code,
        entry_va,
    })
}

/// Relocate instructions from `orig_ip` to `new_ip` using iced-x86 BlockEncoder.
fn relocate_instructions_pe(bytes: &[u8], orig_ip: u64, new_ip: u64) -> Result<Vec<u8>> {
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

/// Relocate 32-bit instructions from `orig_ip` to `new_ip` using iced-x86 BlockEncoder.
fn relocate_instructions_pe32(bytes: &[u8], orig_ip: u64, new_ip: u64) -> Result<Vec<u8>> {
    use iced_x86::{BlockEncoder, BlockEncoderOptions, Decoder, DecoderOptions, InstructionBlock};
    let mut decoder = Decoder::with_ip(32, bytes, orig_ip, DecoderOptions::NONE);
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
    let result = BlockEncoder::encode(32, block, BlockEncoderOptions::NONE)
        .context("iced-x86 BlockEncoder failed (32-bit)")?;
    Ok(result.code_buffer)
}

/// Generate a 32-bit x86 coverage trampoline for PE32 binaries.
///
/// Same logic as the 64-bit `generate_trampoline` but uses:
/// - 32-bit registers (eax, ecx)
/// - Absolute addressing instead of RIP-relative
/// - No red zone skip (x86 doesn't have a red zone)
/// - No REX prefixes
fn generate_trampoline_pe32(
    trampoline_va: u64,
    data_va: u64,
    block: &crate::disasm::BasicBlock,
) -> Result<crate::trampoline::Trampoline> {
    let block_id = block.block_id as u32;
    let return_va = block.va + block.displaced_len as u64;

    let mut stub = Vec::with_capacity(128);

    // No red zone on x86-32. Just push flags and registers.
    // pushfd
    stub.push(0x9C);
    // push eax
    stub.push(0x50);
    // push ecx
    stub.push(0x51);

    // mov eax, dword [data_va]  ; load SHM pointer (absolute addressing)
    // A1 <imm32> = mov eax, [moffs32]
    stub.push(0xA1);
    stub.extend_from_slice(&(data_va as u32).to_le_bytes());

    // test eax, eax
    stub.extend_from_slice(&[0x85, 0xC0]);

    // jz .skip
    let jz_offset_pos = stub.len() + 1;
    stub.extend_from_slice(&[0x74, 0x00]); // placeholder

    // movzx ecx, word [prev_loc_va]  (absolute addressing)
    // 0F B7 0D <imm32>
    let prev_loc_va = data_va + PREV_LOC_OFFSET;
    stub.extend_from_slice(&[0x0F, 0xB7, 0x0D]);
    stub.extend_from_slice(&(prev_loc_va as u32).to_le_bytes());

    // xor ecx, BLOCK_ID
    stub.extend_from_slice(&[0x81, 0xF1]);
    stub.extend_from_slice(&block_id.to_le_bytes());

    // and ecx, 0xFFFF
    stub.extend_from_slice(&[0x81, 0xE1]);
    stub.extend_from_slice(&0x0000FFFFu32.to_le_bytes());

    // inc byte [eax + ecx]
    stub.extend_from_slice(&[0xFE, 0x04, 0x08]);

    // mov word [prev_loc_va], BLOCK_ID >> 1  (absolute addressing)
    // 66 C7 05 <imm32> <imm16>
    let half_id = (block_id >> 1) as u16;
    stub.extend_from_slice(&[0x66, 0xC7, 0x05]);
    stub.extend_from_slice(&(prev_loc_va as u32).to_le_bytes());
    stub.extend_from_slice(&half_id.to_le_bytes());

    // .skip: (fix up the jz offset)
    let skip_target = stub.len();
    let jz_rel = (skip_target as isize - (jz_offset_pos as isize + 1)) as u8;
    stub[jz_offset_pos] = jz_rel;

    // pop ecx
    stub.push(0x59);
    // pop eax
    stub.push(0x58);
    // popfd
    stub.push(0x9D);

    // Emit relocated displaced instructions.
    let displaced_va = trampoline_va + stub.len() as u64;
    let relocated = relocate_instructions_pe32(&block.displaced_bytes, block.va, displaced_va)
        .with_context(|| {
            format!(
                "failed to relocate 32-bit instructions for block at 0x{:x}",
                block.va,
            )
        })?;

    stub.extend_from_slice(&relocated);

    // jmp return_va
    let jmp_ip = trampoline_va + stub.len() as u64 + 5;
    let jmp_rel = (return_va as i64 - jmp_ip as i64) as i32;
    stub.push(0xE9);
    stub.extend_from_slice(&jmp_rel.to_le_bytes());

    Ok(crate::trampoline::Trampoline {
        va: trampoline_va,
        code: stub,
    })
}

/// Generate x86_64 machine code for a PE persistent mode wrapper.
///
/// This implements a pipe-based persistent loop (no fork — Windows has no fork).
/// The fuzzer communicates via pipe handles 198 (read) and 199 (write).
///
/// See `PE_PERSISTENT_DATA_SIZE` for the data layout.
#[allow(clippy::too_many_arguments)]
fn generate_pe_persistent_wrapper(
    wrapper_va: u64,
    persistent_data_va: u64,
    data_va: u64,
    persistent_addr: u64,
    displaced_bytes: &[u8],
    displaced_len: usize,
    persistent_count: u32,
) -> Result<crate::trampoline::PersistentWrapper> {
    let mut code = Vec::with_capacity(512);
    let return_va = persistent_addr + displaced_len as u64;

    // Persistent data field offsets.
    const OFF_FIRST_PASS: u64 = 0;
    const OFF_COUNTER: u64 = 4;
    const OFF_SAVE_REGS: u64 = 8;
    const OFF_SAVE_RET: u64 = 136;
    const OFF_SAVE_RSP: u64 = 144;
    const OFF_READ_FILE: u64 = 152;
    const OFF_WRITE_FILE: u64 = 160;
    const OFF_EXIT_PROCESS: u64 = 168;

    // Helper: RIP-relative displacement to a persistent data field.
    // code_pos is the RIP value AFTER the instruction (i.e., wrapper_va + current code.len() + instruction_size).
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
    let gpr_load_opcodes: &[(u8, u8, u8)] = &[
        (0x48, 0x8B, 0x05), // rax
        (0x48, 0x8B, 0x0D), // rcx
        (0x48, 0x8B, 0x15), // rdx
        (0x48, 0x8B, 0x1D), // rbx
        (0x48, 0x8B, 0x25), // rsp (not used for RSP restore)
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

    // ================================================================
    // Check first_pass flag
    // ================================================================

    // cmp byte [rip+first_pass], 0   ; 7 bytes
    let cmp_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x80, 0x3D]);
    code.extend_from_slice(&data_disp(cmp_ip, OFF_FIRST_PASS).to_le_bytes());
    code.push(0x00);

    // je .setup_and_run  (first_pass==0 → subsequent call)
    let je_setup_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // je rel32 placeholder

    // ================================================================
    // First call: save state, set counter, do pipe handshake
    // ================================================================

    // Clear first_pass: mov byte [rip+first_pass], 0  ; 7 bytes
    let mov_fp_ip = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0xC6, 0x05]);
    code.extend_from_slice(&data_disp(mov_fp_ip, OFF_FIRST_PASS).to_le_bytes());
    code.push(0x00);

    // Save all 16 GPRs to persistent data area.
    for (i, &(b0, b1, b2)) in gpr_save_opcodes.iter().enumerate() {
        let instr_len = 7u64; // 3-byte opcode + 4-byte disp32
        let ip_after = wrapper_va + code.len() as u64 + instr_len;
        code.extend_from_slice(&[b0, b1, b2]);
        code.extend_from_slice(&data_disp(ip_after, OFF_SAVE_REGS + (i as u64) * 8).to_le_bytes());
    }

    // Save return address: mov rax, [rsp] ; mov [rip+save_ret], rax
    // mov rax, [rsp]  ; 4 bytes
    code.extend_from_slice(&[0x48, 0x8B, 0x04, 0x24]);
    // mov [rip+save_ret], rax  ; 7 bytes
    let ip_after = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x89, 0x05]);
    code.extend_from_slice(&data_disp(ip_after, OFF_SAVE_RET).to_le_bytes());

    // Save RSP: mov [rip+save_rsp], rsp  ; 7 bytes
    let ip_after = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x89, 0x25]);
    code.extend_from_slice(&data_disp(ip_after, OFF_SAVE_RSP).to_le_bytes());

    // Set counter: mov dword [rip+counter], persistent_count  ; 10 bytes
    let ip_after = wrapper_va + code.len() as u64 + 10;
    code.extend_from_slice(&[0xC7, 0x05]);
    code.extend_from_slice(&data_disp(ip_after, OFF_COUNTER).to_le_bytes());
    code.extend_from_slice(&persistent_count.to_le_bytes());

    // ================================================================
    // Pipe handshake: write 4 zero bytes to fd 199, read 4 from fd 198
    // Uses WriteFile/ReadFile pointers resolved by init code.
    //
    // sub rsp, 48   ; shadow(32) + 5th param(8) + alignment(8)
    // ================================================================
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x30]); // sub rsp, 48

    // Write 4 zero bytes: mov dword [rsp+40], 0  (buffer at rsp+40)
    code.extend_from_slice(&[0xC7, 0x44, 0x24, 0x28, 0x00, 0x00, 0x00, 0x00]);

    // WriteFile(199, &buf, 4, &bytes_written, NULL)
    // mov rcx, 199  ; hFile
    code.extend_from_slice(&[0x48, 0xC7, 0xC1, 0xC7, 0x00, 0x00, 0x00]);
    // lea rdx, [rsp+40]  ; lpBuffer
    code.extend_from_slice(&[0x48, 0x8D, 0x54, 0x24, 0x28]);
    // mov r8d, 4  ; nNumberOfBytesToWrite
    code.extend_from_slice(&[0x41, 0xB8, 0x04, 0x00, 0x00, 0x00]);
    // lea r9, [rsp+44]  ; lpNumberOfBytesWritten
    code.extend_from_slice(&[0x4C, 0x8D, 0x4C, 0x24, 0x2C]);
    // mov qword [rsp+32], 0  ; lpOverlapped = NULL
    code.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, 0x20, 0x00, 0x00, 0x00, 0x00]);
    // call [rip+WriteFile_ptr]
    let call_ip = wrapper_va + code.len() as u64 + 6;
    code.extend_from_slice(&[0xFF, 0x15]);
    code.extend_from_slice(&data_disp(call_ip, OFF_WRITE_FILE).to_le_bytes());

    // Check WriteFile result. If failed (rax==0), skip to standalone mode.
    // test eax, eax
    code.extend_from_slice(&[0x85, 0xC0]);
    // jz .standalone  (no parent fuzzer)
    let jz_standalone_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // ReadFile(198, &buf, 4, &bytes_read, NULL) — wait for GO signal
    // mov rcx, 198  ; hFile
    code.extend_from_slice(&[0x48, 0xC7, 0xC1, 0xC6, 0x00, 0x00, 0x00]);
    // lea rdx, [rsp+40]  ; lpBuffer
    code.extend_from_slice(&[0x48, 0x8D, 0x54, 0x24, 0x28]);
    // mov r8d, 4  ; nNumberOfBytesToRead
    code.extend_from_slice(&[0x41, 0xB8, 0x04, 0x00, 0x00, 0x00]);
    // lea r9, [rsp+44]  ; lpNumberOfBytesRead
    code.extend_from_slice(&[0x4C, 0x8D, 0x4C, 0x24, 0x2C]);
    // mov qword [rsp+32], 0  ; lpOverlapped = NULL
    code.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, 0x20, 0x00, 0x00, 0x00, 0x00]);
    // call [rip+ReadFile_ptr]
    let call_ip = wrapper_va + code.len() as u64 + 6;
    code.extend_from_slice(&[0xFF, 0x15]);
    code.extend_from_slice(&data_disp(call_ip, OFF_READ_FILE).to_le_bytes());

    // .standalone:
    let standalone = code.len();
    let standalone_rel = (standalone as i32) - (jz_standalone_pos as i32 + 6);
    code[jz_standalone_pos + 2..jz_standalone_pos + 6]
        .copy_from_slice(&standalone_rel.to_le_bytes());

    // Deallocate scratch: add rsp, 48
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x30]);

    // ================================================================
    // .setup_and_run:
    // Patch return address on stack to point to iteration_boundary.
    // Restore rax, execute displaced instructions, JMP to function body.
    // ================================================================
    let setup_and_run = code.len();
    // Fix up the je from the first_pass check.
    let setup_rel = (setup_and_run as i32) - (je_setup_pos as i32 + 6);
    code[je_setup_pos + 2..je_setup_pos + 6].copy_from_slice(&setup_rel.to_le_bytes());

    // Load the iteration_boundary VA into rax and write it over the return address on stack.
    // We'll fix up this imm64 once we know iteration_boundary's position.
    // mov rax, imm64  ; 10 bytes
    let mov_iter_bound_pos = code.len();
    code.extend_from_slice(&[0x48, 0xB8]);
    code.extend_from_slice(&0u64.to_le_bytes()); // placeholder, fixed later

    // mov [rsp], rax  ; overwrite return address
    code.extend_from_slice(&[0x48, 0x89, 0x04, 0x24]);

    // Restore rax from save area so displaced instructions see original rax.
    // mov rax, [rip+save_regs+0]  ; 7 bytes
    let ip_after = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x8B, 0x05]);
    code.extend_from_slice(&data_disp(ip_after, OFF_SAVE_REGS).to_le_bytes());

    // Execute displaced instructions (relocated to current position).
    let displaced_new_ip = wrapper_va + code.len() as u64;
    let relocated = relocate_instructions_pe(displaced_bytes, persistent_addr, displaced_new_ip)
        .context("failed to relocate displaced instructions for PE persistent wrapper")?;
    code.extend_from_slice(&relocated);

    // JMP to persistent_addr + displaced_len (rest of original function).
    // jmp rel32
    code.push(0xE9);
    let jmp_rip = wrapper_va + code.len() as u64 + 4;
    let jmp_rel = (return_va as i64 - jmp_rip as i64) as i32;
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    // ================================================================
    // .iteration_boundary:
    // Hit when the persistent function returns.
    // ================================================================
    let iteration_boundary = code.len();
    let iteration_boundary_va = wrapper_va + iteration_boundary as u64;

    // Fix up the mov rax, iteration_boundary_va.
    code[mov_iter_bound_pos + 2..mov_iter_bound_pos + 10]
        .copy_from_slice(&iteration_boundary_va.to_le_bytes());

    // Decrement counter: sub dword [rip+disp32], 1  ; 7 bytes: 83 2D <disp32> 01
    let ip_after = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x83, 0x2D]);
    code.extend_from_slice(&data_disp(ip_after, OFF_COUNTER).to_le_bytes());
    code.push(0x01);

    // jnz .next_iteration  (counter > 0)
    let jnz_next_iter_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jnz rel32 placeholder

    // Counter == 0: call ExitProcess(0).
    // sub rsp, 40  ; shadow(32) + alignment(8)
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x28]);
    // xor ecx, ecx  ; uExitCode = 0
    code.extend_from_slice(&[0x31, 0xC9]);
    // call [rip+ExitProcess_ptr]
    let call_ip = wrapper_va + code.len() as u64 + 6;
    code.extend_from_slice(&[0xFF, 0x15]);
    code.extend_from_slice(&data_disp(call_ip, OFF_EXIT_PROCESS).to_le_bytes());
    // If ExitProcess returns (it shouldn't), just spin.
    // jmp $  ; EB FE
    code.extend_from_slice(&[0xEB, 0xFE]);

    // ================================================================
    // .next_iteration:
    // Write status to pipe, read GO, clear prev_loc, restore GPRs, loop.
    // ================================================================
    let next_iteration = code.len();
    let next_iter_rel = (next_iteration as i32) - (jnz_next_iter_pos as i32 + 6);
    code[jnz_next_iter_pos + 2..jnz_next_iter_pos + 6]
        .copy_from_slice(&next_iter_rel.to_le_bytes());

    // Allocate scratch: sub rsp, 48
    code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x30]);

    // Write 4-byte status (0) to fd 199.
    // mov dword [rsp+40], 0
    code.extend_from_slice(&[0xC7, 0x44, 0x24, 0x28, 0x00, 0x00, 0x00, 0x00]);
    // WriteFile(199, &buf, 4, &bytes_written, NULL)
    code.extend_from_slice(&[0x48, 0xC7, 0xC1, 0xC7, 0x00, 0x00, 0x00]); // mov rcx, 199
    code.extend_from_slice(&[0x48, 0x8D, 0x54, 0x24, 0x28]); // lea rdx, [rsp+40]
    code.extend_from_slice(&[0x41, 0xB8, 0x04, 0x00, 0x00, 0x00]); // mov r8d, 4
    code.extend_from_slice(&[0x4C, 0x8D, 0x4C, 0x24, 0x2C]); // lea r9, [rsp+44]
    code.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, 0x20, 0x00, 0x00, 0x00, 0x00]); // mov qword [rsp+32], 0
    let call_ip = wrapper_va + code.len() as u64 + 6;
    code.extend_from_slice(&[0xFF, 0x15]);
    code.extend_from_slice(&data_disp(call_ip, OFF_WRITE_FILE).to_le_bytes());

    // ReadFile(198, &buf, 4, &bytes_read, NULL) — wait for GO.
    code.extend_from_slice(&[0x48, 0xC7, 0xC1, 0xC6, 0x00, 0x00, 0x00]); // mov rcx, 198
    code.extend_from_slice(&[0x48, 0x8D, 0x54, 0x24, 0x28]); // lea rdx, [rsp+40]
    code.extend_from_slice(&[0x41, 0xB8, 0x04, 0x00, 0x00, 0x00]); // mov r8d, 4
    code.extend_from_slice(&[0x4C, 0x8D, 0x4C, 0x24, 0x2C]); // lea r9, [rsp+44]
    code.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, 0x20, 0x00, 0x00, 0x00, 0x00]); // mov qword [rsp+32], 0
    let call_ip = wrapper_va + code.len() as u64 + 6;
    code.extend_from_slice(&[0xFF, 0x15]);
    code.extend_from_slice(&data_disp(call_ip, OFF_READ_FILE).to_le_bytes());

    // Deallocate scratch: add rsp, 48
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x30]);

    // Clear prev_loc: mov qword [rip+prev_loc], 0  ; 11 bytes (48 C7 05 <disp32> 00000000)
    let ip_after = wrapper_va + code.len() as u64 + 11;
    let prev_loc_disp = (data_va as i64 + PREV_LOC_OFFSET as i64 - ip_after as i64) as i32;
    code.extend_from_slice(&[0x48, 0xC7, 0x05]);
    code.extend_from_slice(&prev_loc_disp.to_le_bytes());
    code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    // Restore all GPRs from save area (skip RSP — index 4).
    for (i, &(b0, b1, b2)) in gpr_load_opcodes.iter().enumerate() {
        if i == 4 {
            continue;
        } // skip RSP — restored separately
        let instr_len = 7u64;
        let ip_after = wrapper_va + code.len() as u64 + instr_len;
        code.extend_from_slice(&[b0, b1, b2]);
        code.extend_from_slice(&data_disp(ip_after, OFF_SAVE_REGS + (i as u64) * 8).to_le_bytes());
    }

    // Restore RSP from save area.
    // mov rsp, [rip+save_rsp]  ; 7 bytes
    let ip_after = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x8B, 0x25]);
    code.extend_from_slice(&data_disp(ip_after, OFF_SAVE_RSP).to_le_bytes());

    // Push saved return address onto restored stack.
    // First load it into rax (clobbers rax, but we'll re-load rax after the push).
    // mov rax, [rip+save_ret]  ; 7 bytes
    let ip_after = wrapper_va + code.len() as u64 + 7;
    code.extend_from_slice(&[0x48, 0x8B, 0x05]);
    code.extend_from_slice(&data_disp(ip_after, OFF_SAVE_RET).to_le_bytes());
    // push rax  ; puts saved return address on stack
    code.push(0x50);

    // JMP to setup_and_run.
    code.push(0xE9);
    let jmp_rip = wrapper_va + code.len() as u64 + 4;
    let jmp_rel = (wrapper_va as i64 + setup_and_run as i64 - jmp_rip as i64) as i32;
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    Ok(crate::trampoline::PersistentWrapper { code })
}

/// Generate 32-bit x86 machine code for a PE32 persistent mode wrapper.
///
/// Same logic as `generate_pe_persistent_wrapper` but uses 32-bit registers,
/// absolute addressing (no RIP-relative), cdecl calling convention, and
/// 4-byte pointers.
///
/// See `PE32_PERSISTENT_DATA_SIZE` for the data layout.
#[allow(clippy::too_many_arguments)]
fn generate_pe32_persistent_wrapper(
    wrapper_va: u64,
    persistent_data_va: u64,
    data_va: u64,
    persistent_addr: u64,
    displaced_bytes: &[u8],
    displaced_len: usize,
    persistent_count: u32,
) -> Result<crate::trampoline::PersistentWrapper> {
    let mut code = Vec::with_capacity(512);
    let return_va = persistent_addr + displaced_len as u64;

    // Persistent data field offsets (PE32 layout).
    const OFF_FIRST_PASS: u64 = 0;
    const OFF_COUNTER: u64 = 4;
    const OFF_SAVE_REGS: u64 = 8; // 8 x 4 = 32 bytes
    const OFF_SAVE_RET: u64 = 40;
    const OFF_SAVE_ESP: u64 = 44;
    const OFF_READ_FILE: u64 = 48;
    const OFF_WRITE_FILE: u64 = 52;
    const OFF_EXIT_PROCESS: u64 = 56;

    // Helper: absolute VA of a persistent data field.
    let data_addr = |field_offset: u64| -> u32 { (persistent_data_va + field_offset) as u32 };

    // GPR register indices: eax=0, ecx=1, edx=2, ebx=3, esp=4, ebp=5, esi=6, edi=7
    // mov [abs], reg  =>  89 <modrm> <disp32>  where modrm = 0x05 | (reg<<3), mod=00, r/m=101 (disp32)
    // mov reg, [abs]  =>  8B <modrm> <disp32>  where modrm = 0x05 | (reg<<3)
    let gpr_modrm: &[u8] = &[
        0x05, // eax (reg=0)
        0x0D, // ecx (reg=1)
        0x15, // edx (reg=2)
        0x1D, // ebx (reg=3)
        0x25, // esp (reg=4)
        0x2D, // ebp (reg=5)
        0x35, // esi (reg=6)
        0x3D, // edi (reg=7)
    ];

    // ================================================================
    // Check first_pass flag
    // ================================================================

    // cmp byte [first_pass], 0   ; 7 bytes: 80 3D <addr32> 00
    code.extend_from_slice(&[0x80, 0x3D]);
    code.extend_from_slice(&data_addr(OFF_FIRST_PASS).to_le_bytes());
    code.push(0x00);

    // je .setup_and_run  (first_pass==0 -> subsequent call)
    let je_setup_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // je rel32 placeholder

    // ================================================================
    // First call: save state, set counter, do pipe handshake
    // ================================================================

    // Clear first_pass: mov byte [first_pass], 0  ; 7 bytes: C6 05 <addr32> 00
    code.extend_from_slice(&[0xC6, 0x05]);
    code.extend_from_slice(&data_addr(OFF_FIRST_PASS).to_le_bytes());
    code.push(0x00);

    // Save all 8 GPRs to persistent data area.
    // mov [addr32], reg  =>  89 <modrm> <addr32>  ; 6 bytes each
    for (i, &modrm) in gpr_modrm.iter().enumerate() {
        code.extend_from_slice(&[0x89, modrm]);
        code.extend_from_slice(&data_addr(OFF_SAVE_REGS + (i as u64) * 4).to_le_bytes());
    }

    // Save return address: mov eax, [esp] ; mov [save_ret], eax
    // mov eax, [esp]  ; 3 bytes: 8B 04 24
    code.extend_from_slice(&[0x8B, 0x04, 0x24]);
    // mov [save_ret], eax  ; A3 <addr32>  ; 5 bytes
    code.push(0xA3);
    code.extend_from_slice(&data_addr(OFF_SAVE_RET).to_le_bytes());

    // Save ESP: mov [save_esp], esp  ; 89 25 <addr32>  ; 6 bytes
    code.extend_from_slice(&[0x89, 0x25]);
    code.extend_from_slice(&data_addr(OFF_SAVE_ESP).to_le_bytes());

    // Set counter: mov dword [counter], persistent_count  ; C7 05 <addr32> <imm32>  ; 10 bytes
    code.extend_from_slice(&[0xC7, 0x05]);
    code.extend_from_slice(&data_addr(OFF_COUNTER).to_le_bytes());
    code.extend_from_slice(&persistent_count.to_le_bytes());

    // ================================================================
    // Pipe handshake: write 4 zero bytes to fd 199, read 4 from fd 198
    // Uses WriteFile/ReadFile pointers resolved by init code.
    // stdcall: push args right-to-left; callee cleans stack.
    //
    // Set up a stack frame for scratch space.
    // sub esp, 16  ; 8 bytes scratch (buf at esp+0, written at esp+4, ...padding)
    // ================================================================
    code.extend_from_slice(&[0x83, 0xEC, 0x10]); // sub esp, 16

    // Zero the buffer: mov dword [esp], 0
    code.extend_from_slice(&[0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00]);

    // WriteFile(199, &buf, 4, &bytes_written, NULL)
    // stdcall: push right-to-left; callee cleans stack
    code.push(0x6A);
    code.push(0x00); // push 0  (lpOverlapped)
    // lea eax, [esp+8]  ; &bytes_written (adjusted for pushes: esp+4 original + 4 bytes pushed)
    code.extend_from_slice(&[0x8D, 0x44, 0x24, 0x08]);
    code.push(0x50); // push eax
    code.push(0x6A);
    code.push(0x04); // push 4  (nNumberOfBytesToWrite)
    // lea eax, [esp+12]  ; &buf (esp+0 original, but 3 pushes = 12 bytes on stack)
    code.extend_from_slice(&[0x8D, 0x44, 0x24, 0x0C]);
    code.push(0x50); // push eax
    code.push(0x68); // push 199
    code.extend_from_slice(&199u32.to_le_bytes());
    // call [WriteFile_ptr]  ; FF 15 <addr32>
    code.extend_from_slice(&[0xFF, 0x15]);
    code.extend_from_slice(&data_addr(OFF_WRITE_FILE).to_le_bytes());
    code.extend_from_slice(&[0x83, 0xC4, 0x14]); // add esp, 20 (5 args x 4)

    // Check WriteFile result. If failed (eax==0), skip to standalone mode.
    code.extend_from_slice(&[0x85, 0xC0]); // test eax, eax
    let jz_standalone_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]); // jz rel32 placeholder

    // ReadFile(198, &buf, 4, &bytes_read, NULL) -- wait for GO signal
    code.push(0x6A);
    code.push(0x00); // push 0  (lpOverlapped)
    // lea eax, [esp+8]  ; &bytes_read
    code.extend_from_slice(&[0x8D, 0x44, 0x24, 0x08]);
    code.push(0x50); // push eax
    code.push(0x6A);
    code.push(0x04); // push 4
    // lea eax, [esp+12]  ; &buf
    code.extend_from_slice(&[0x8D, 0x44, 0x24, 0x0C]);
    code.push(0x50); // push eax
    code.push(0x68); // push 198
    code.extend_from_slice(&198u32.to_le_bytes());
    // call [ReadFile_ptr]
    code.extend_from_slice(&[0xFF, 0x15]);
    code.extend_from_slice(&data_addr(OFF_READ_FILE).to_le_bytes());
    code.extend_from_slice(&[0x83, 0xC4, 0x14]); // add esp, 20

    // .standalone:
    let standalone = code.len();
    let standalone_rel = (standalone as i32) - (jz_standalone_pos as i32 + 6);
    code[jz_standalone_pos + 2..jz_standalone_pos + 6]
        .copy_from_slice(&standalone_rel.to_le_bytes());

    // Deallocate scratch: add esp, 16
    code.extend_from_slice(&[0x83, 0xC4, 0x10]);

    // ================================================================
    // .setup_and_run:
    // Patch return address on stack to point to iteration_boundary.
    // Restore eax, execute displaced instructions, JMP to function body.
    // ================================================================
    let setup_and_run = code.len();
    let setup_rel = (setup_and_run as i32) - (je_setup_pos as i32 + 6);
    code[je_setup_pos + 2..je_setup_pos + 6].copy_from_slice(&setup_rel.to_le_bytes());

    // Load the iteration_boundary VA into eax and write it over the return address on stack.
    // mov eax, imm32  ; 5 bytes: B8 <imm32>
    let mov_iter_bound_pos = code.len();
    code.push(0xB8);
    code.extend_from_slice(&0u32.to_le_bytes()); // placeholder, fixed later

    // mov [esp], eax  ; overwrite return address  ; 3 bytes: 89 04 24
    code.extend_from_slice(&[0x89, 0x04, 0x24]);

    // Restore eax from save area so displaced instructions see original eax.
    // mov eax, [save_regs+0]  ; A1 <addr32>  ; 5 bytes
    code.push(0xA1);
    code.extend_from_slice(&data_addr(OFF_SAVE_REGS).to_le_bytes());

    // Execute displaced instructions (relocated to current position).
    let displaced_new_ip = wrapper_va + code.len() as u64;
    let relocated = relocate_instructions_pe32(displaced_bytes, persistent_addr, displaced_new_ip)
        .context("failed to relocate displaced instructions for PE32 persistent wrapper")?;
    code.extend_from_slice(&relocated);

    // JMP to persistent_addr + displaced_len (rest of original function).
    code.push(0xE9);
    let jmp_rip = wrapper_va + code.len() as u64 + 4;
    let jmp_rel = (return_va as i64 - jmp_rip as i64) as i32;
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    // ================================================================
    // .iteration_boundary:
    // Hit when the persistent function returns.
    // ================================================================
    let iteration_boundary = code.len();
    let iteration_boundary_va = wrapper_va + iteration_boundary as u64;

    // Fix up the mov eax, iteration_boundary_va.
    code[mov_iter_bound_pos + 1..mov_iter_bound_pos + 5]
        .copy_from_slice(&(iteration_boundary_va as u32).to_le_bytes());

    // Decrement counter: sub dword [counter], 1  ; 83 2D <addr32> 01  ; 7 bytes
    code.extend_from_slice(&[0x83, 0x2D]);
    code.extend_from_slice(&data_addr(OFF_COUNTER).to_le_bytes());
    code.push(0x01);

    // jnz .next_iteration  (counter > 0)
    let jnz_next_iter_pos = code.len();
    code.extend_from_slice(&[0x0F, 0x85, 0x00, 0x00, 0x00, 0x00]); // jnz rel32 placeholder

    // Counter == 0: call ExitProcess(0).
    // stdcall: push 0, call [ExitProcess_ptr] (callee cleans stack; ExitProcess does not return)
    code.push(0x6A);
    code.push(0x00); // push 0  (uExitCode)
    code.extend_from_slice(&[0xFF, 0x15]); // call [ExitProcess_ptr]
    code.extend_from_slice(&data_addr(OFF_EXIT_PROCESS).to_le_bytes());
    // If ExitProcess returns (it shouldn't), just spin.
    code.extend_from_slice(&[0xEB, 0xFE]); // jmp $

    // ================================================================
    // .next_iteration:
    // Write status to pipe, read GO, clear prev_loc, restore GPRs, loop.
    // ================================================================
    let next_iteration = code.len();
    let next_iter_rel = (next_iteration as i32) - (jnz_next_iter_pos as i32 + 6);
    code[jnz_next_iter_pos + 2..jnz_next_iter_pos + 6]
        .copy_from_slice(&next_iter_rel.to_le_bytes());

    // Allocate scratch: sub esp, 16
    code.extend_from_slice(&[0x83, 0xEC, 0x10]);

    // Zero the buffer: mov dword [esp], 0
    code.extend_from_slice(&[0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00]);

    // WriteFile(199, &buf, 4, &bytes_written, NULL)
    code.push(0x6A);
    code.push(0x00); // push 0
    code.extend_from_slice(&[0x8D, 0x44, 0x24, 0x08]); // lea eax, [esp+8]
    code.push(0x50); // push eax
    code.push(0x6A);
    code.push(0x04); // push 4
    code.extend_from_slice(&[0x8D, 0x44, 0x24, 0x0C]); // lea eax, [esp+12]
    code.push(0x50); // push eax
    code.push(0x68); // push 199
    code.extend_from_slice(&199u32.to_le_bytes());
    code.extend_from_slice(&[0xFF, 0x15]); // call [WriteFile_ptr]
    code.extend_from_slice(&data_addr(OFF_WRITE_FILE).to_le_bytes());
    code.extend_from_slice(&[0x83, 0xC4, 0x14]); // add esp, 20

    // ReadFile(198, &buf, 4, &bytes_read, NULL)
    code.push(0x6A);
    code.push(0x00); // push 0
    code.extend_from_slice(&[0x8D, 0x44, 0x24, 0x08]); // lea eax, [esp+8]
    code.push(0x50); // push eax
    code.push(0x6A);
    code.push(0x04); // push 4
    code.extend_from_slice(&[0x8D, 0x44, 0x24, 0x0C]); // lea eax, [esp+12]
    code.push(0x50); // push eax
    code.push(0x68); // push 198
    code.extend_from_slice(&198u32.to_le_bytes());
    code.extend_from_slice(&[0xFF, 0x15]); // call [ReadFile_ptr]
    code.extend_from_slice(&data_addr(OFF_READ_FILE).to_le_bytes());
    code.extend_from_slice(&[0x83, 0xC4, 0x14]); // add esp, 20

    // Deallocate scratch: add esp, 16
    code.extend_from_slice(&[0x83, 0xC4, 0x10]);

    // Clear prev_loc: mov dword [prev_loc], 0  ; C7 05 <addr32> <imm32>  ; 10 bytes
    let prev_loc_va = data_va + PREV_LOC_OFFSET;
    code.extend_from_slice(&[0xC7, 0x05]);
    code.extend_from_slice(&(prev_loc_va as u32).to_le_bytes());
    code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    // Restore all GPRs from save area (skip ESP -- index 4).
    // mov reg, [addr32]  =>  8B <modrm> <addr32>  ; 6 bytes each
    for (i, &modrm) in gpr_modrm.iter().enumerate() {
        if i == 4 {
            continue;
        } // skip ESP -- restored separately
        code.extend_from_slice(&[0x8B, modrm]);
        code.extend_from_slice(&data_addr(OFF_SAVE_REGS + (i as u64) * 4).to_le_bytes());
    }

    // Restore ESP from save area.
    // mov esp, [save_esp]  ; 8B 25 <addr32>  ; 6 bytes
    code.extend_from_slice(&[0x8B, 0x25]);
    code.extend_from_slice(&data_addr(OFF_SAVE_ESP).to_le_bytes());

    // Push saved return address onto restored stack.
    // mov eax, [save_ret]  ; A1 <addr32>
    code.push(0xA1);
    code.extend_from_slice(&data_addr(OFF_SAVE_RET).to_le_bytes());
    // push eax  ; puts saved return address on stack
    code.push(0x50);

    // JMP to setup_and_run.
    code.push(0xE9);
    let jmp_rip = wrapper_va + code.len() as u64 + 4;
    let jmp_rel = (wrapper_va as i64 + setup_and_run as i64 - jmp_rip as i64) as i32;
    code.extend_from_slice(&jmp_rel.to_le_bytes());

    Ok(crate::trampoline::PersistentWrapper { code })
}

/// DJB2 hash function (for PE export name lookup).
fn djb2_hash(name: &[u8]) -> u32 {
    let mut hash: u32 = 5381;
    for &c in name {
        hash = hash.wrapping_mul(33).wrapping_add(c as u32);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_djb2_known_hashes() {
        // Verify our hash function produces correct values.
        let h1 = djb2_hash(b"OpenFileMappingA");
        let h2 = djb2_hash(b"MapViewOfFile");
        // These should be deterministic.
        assert_ne!(h1, 0);
        assert_ne!(h2, 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_pe_init_code_generation() {
        // Smoke test: generate init code and verify it's non-empty.
        let result = generate_pe_init_code(0x140010000, 0x140010000 - 16, 0x140001000, None, None);
        assert!(
            result.is_ok(),
            "PE init code generation failed: {:?}",
            result.err()
        );
        let init = result.expect("PE init code generation should succeed");
        assert!(!init.code.is_empty());
        assert!(
            init.code.len() > 100,
            "init code too short: {} bytes",
            init.code.len()
        );
    }

    #[test]
    fn test_pe_init_code_with_hooks() {
        let hook_lib = PeHookLibraryInfo {
            library_path: "my_hooks.dll".to_string(),
            handlers: vec![
                ("on_malloc".to_string(), 0x140020000),
                ("on_free".to_string(), 0x140020008),
            ],
        };
        let result = generate_pe_init_code(
            0x140010000,
            0x140010000 - 16,
            0x140001000,
            Some(&hook_lib),
            None,
        );
        assert!(
            result.is_ok(),
            "PE init code with hooks failed: {:?}",
            result.err()
        );
        let init = result.expect("PE init code with hooks should succeed");
        assert!(!init.code.is_empty());
        // With hooks, init code should be larger than without.
        let no_hooks =
            generate_pe_init_code(0x140010000, 0x140010000 - 16, 0x140001000, None, None)
                .expect("PE init code without hooks should succeed");
        assert!(
            init.code.len() > no_hooks.code.len(),
            "hook init code ({}) should be larger than no-hook ({})",
            init.code.len(),
            no_hooks.code.len(),
        );
        // Verify the library path is embedded in the code.
        let code = &init.code;
        let lib_bytes = b"my_hooks.dll\0";
        let found = code.windows(lib_bytes.len()).any(|w| w == lib_bytes);
        assert!(found, "library path not found in init code");
        // Verify handler names are embedded.
        assert!(code.windows(10).any(|w| w == b"on_malloc\0"));
        assert!(code.windows(8).any(|w| w == b"on_free\0"));
    }

    #[test]
    fn test_djb2_load_library_get_proc_address() {
        let h1 = djb2_hash(b"LoadLibraryA");
        let h2 = djb2_hash(b"GetProcAddress");
        assert_ne!(h1, 0);
        assert_ne!(h2, 0);
        assert_ne!(h1, h2);
        // Ensure no collision with the SHM functions.
        assert_ne!(h1, djb2_hash(b"OpenFileMappingA"));
        assert_ne!(h1, djb2_hash(b"MapViewOfFile"));
        assert_ne!(h2, djb2_hash(b"OpenFileMappingA"));
        assert_ne!(h2, djb2_hash(b"MapViewOfFile"));
    }

    #[test]
    fn test_djb2_persistent_no_collisions() {
        // Verify persistent function hashes don't collide with existing 4.
        let existing = [
            djb2_hash(b"OpenFileMappingA"),
            djb2_hash(b"MapViewOfFile"),
            djb2_hash(b"LoadLibraryA"),
            djb2_hash(b"GetProcAddress"),
        ];
        let persistent = [
            djb2_hash(b"ReadFile"),
            djb2_hash(b"WriteFile"),
            djb2_hash(b"ExitProcess"),
        ];
        // No collisions among persistent hashes.
        assert_ne!(persistent[0], persistent[1]);
        assert_ne!(persistent[0], persistent[2]);
        assert_ne!(persistent[1], persistent[2]);
        // No collisions with existing hashes.
        for &p in &persistent {
            for &e in &existing {
                assert_ne!(p, e, "persistent hash collision");
            }
        }
    }

    #[test]
    fn test_pe_init_code_with_persistent() {
        // Init code with persistent mode should be larger than without.
        let no_persistent =
            generate_pe_init_code(0x140010000, 0x140010000 - 16, 0x140001000, None, None)
                .expect("PE init code without persistent should succeed");
        let with_persistent = generate_pe_init_code(
            0x140010000,
            0x140010000 - 16,
            0x140001000,
            None,
            Some(0x14000F000),
        )
        .expect("PE init code with persistent should succeed");
        assert!(
            with_persistent.code.len() > no_persistent.code.len(),
            "persistent init ({}) should be larger than non-persistent ({})",
            with_persistent.code.len(),
            no_persistent.code.len(),
        );
    }

    #[test]
    fn test_pe_persistent_wrapper_generation() {
        // Generate a persistent wrapper and verify basic properties.
        let wrapper_va = 0x140020000u64;
        let persistent_data_va = 0x14001F000u64;
        let data_va = 0x14001E000u64;
        let persistent_addr = 0x140001000u64;
        // Simulated displaced bytes: 3x NOP + 2-byte NOP = 5 bytes, valid x86_64.
        let displaced = [0x90, 0x90, 0x90, 0x66, 0x90];
        let result = generate_pe_persistent_wrapper(
            wrapper_va,
            persistent_data_va,
            data_va,
            persistent_addr,
            &displaced,
            5,
            1000,
        );
        assert!(
            result.is_ok(),
            "PE persistent wrapper generation failed: {:?}",
            result.err()
        );
        let wrapper = result.expect("PE persistent wrapper generation should succeed");
        assert!(!wrapper.code.is_empty(), "persistent wrapper code is empty");
        assert!(
            wrapper.code.len() > 100,
            "persistent wrapper too short: {} bytes",
            wrapper.code.len(),
        );
        // Verify displaced bytes appear in the wrapper (after relocation).
        // The relocated NOP instructions should still be NOPs.
        let has_nop_sequence = wrapper.code.windows(3).any(|w| w == [0x90, 0x90, 0x90]);
        assert!(
            has_nop_sequence,
            "displaced NOPs not found in persistent wrapper"
        );
    }

    #[test]
    fn test_pe32_init_code_generation() {
        // Basic PE32 init code generation should succeed.
        let result = generate_pe32_init_code(0x00410000, 0x00410000 - 16, 0x00401000, None, None);
        assert!(
            result.is_ok(),
            "PE32 init code generation failed: {:?}",
            result.err()
        );
        let init = result.expect("PE32 init code generation should succeed");
        assert!(!init.code.is_empty(), "PE32 init code is empty");
        // Should contain the wide-string needle for __AFL_SHM_ID=.
        let needle = b"_\x00_\x00A\x00F\x00L\x00";
        let found = init.code.windows(needle.len()).any(|w| w == needle);
        assert!(found, "wide-string needle not found in PE32 init code");
        // Should contain fs:[0x30] PEB access bytes.
        let fs_peb = &[0x64, 0xA1, 0x30, 0x00, 0x00, 0x00];
        let found_fs = init.code.windows(fs_peb.len()).any(|w| w == fs_peb);
        assert!(found_fs, "fs:[0x30] PEB access not found in PE32 init code");
        // Should NOT contain gs: segment prefix (0x65 0x48) used by 64-bit.
        let gs_peb = &[0x65, 0x48, 0x8B];
        let found_gs = init.code.windows(gs_peb.len()).any(|w| w == gs_peb);
        assert!(
            !found_gs,
            "64-bit gs:[0x60] PEB access found in PE32 init code"
        );
        // Should NOT contain REX.W prefix (0x48).
        // (Some 0x48 bytes may appear as data, but the init code should not have REX prefixes.)
        // Just check it's smaller than the 64-bit version.
        let init64 = generate_pe_init_code(0x00410000, 0x00410000 - 16, 0x00401000, None, None)
            .expect("PE64 init code should succeed for comparison");
        assert!(
            init.code.len() < init64.code.len(),
            "PE32 init ({}) should be smaller than PE64 init ({})",
            init.code.len(),
            init64.code.len(),
        );
    }

    #[test]
    fn test_pe32_init_code_with_hooks() {
        let hook_lib = PeHookLibraryInfo {
            library_path: "my_hooks.dll".to_string(),
            handlers: vec![
                ("on_malloc".to_string(), 0x00420000),
                ("on_free".to_string(), 0x00420008),
            ],
        };
        let init = generate_pe32_init_code(
            0x00410000,
            0x00410000 - 16,
            0x00401000,
            Some(&hook_lib),
            None,
        )
        .expect("PE32 init code with hooks should succeed");
        let no_hooks = generate_pe32_init_code(0x00410000, 0x00410000 - 16, 0x00401000, None, None)
            .expect("PE32 init code without hooks should succeed");
        assert!(
            init.code.len() > no_hooks.code.len(),
            "PE32 hook init code ({}) should be larger than no-hook ({})",
            init.code.len(),
            no_hooks.code.len(),
        );
        // Verify the library path is embedded.
        let lib_bytes = b"my_hooks.dll\0";
        let found = init.code.windows(lib_bytes.len()).any(|w| w == lib_bytes);
        assert!(found, "library path not found in PE32 init code");
        // Verify handler names are embedded.
        assert!(init.code.windows(10).any(|w| w == b"on_malloc\0"));
        assert!(init.code.windows(8).any(|w| w == b"on_free\0"));
    }

    #[test]
    fn test_pe32_init_code_contains_afl_shm_needle() {
        let init = generate_pe32_init_code(0x00410000, 0x0040FFF0, 0x00401000, None, None)
            .expect("PE32 init code should succeed for SHM needle test");
        // Full wide-string needle: "__AFL_SHM_ID=" as UTF-16LE.
        let full_needle: Vec<u8> = b"__AFL_SHM_ID="
            .iter()
            .flat_map(|&c| vec![c, 0x00])
            .collect();
        let found = init
            .code
            .windows(full_needle.len())
            .any(|w| w == full_needle.as_slice());
        assert!(
            found,
            "full __AFL_SHM_ID= wide needle not found in PE32 init code"
        );
    }

    #[test]
    fn test_pe32_init_code_ends_with_jmp_to_oep() {
        let oep = 0x00401000u64;
        let init = generate_pe32_init_code(0x00410000, 0x0040FFF0, oep, None, None)
            .expect("PE32 init code should succeed for JMP OEP test");
        // The last 5 bytes should be a JMP rel32 to the original entry point.
        // Find the JMP (0xE9) near the end of executable code (before embedded data).
        // The code ends with JMP rel32 followed by embedded data (needle + padding).
        // Just verify there's at least one E9 with correct target.
        let code = &init.code;
        let mut found_jmp = false;
        for i in 0..code.len().saturating_sub(4) {
            if code[i] == 0xE9 {
                let rel = i32::from_le_bytes([code[i + 1], code[i + 2], code[i + 3], code[i + 4]]);
                let target = (init.va + i as u64 + 5).wrapping_add(rel as u64);
                if target == oep {
                    found_jmp = true;
                    break;
                }
            }
        }
        assert!(
            found_jmp,
            "JMP to original entry point not found in PE32 init code"
        );
    }

    #[test]
    fn test_pe32_persistent_wrapper_generation() {
        // Generate a PE32 persistent wrapper and verify basic properties.
        let wrapper_va = 0x00420000u64;
        let persistent_data_va = 0x0041F000u64;
        let data_va = 0x0041E000u64;
        let persistent_addr = 0x00401000u64;
        // Simulated displaced bytes: 3x NOP + 2-byte NOP = 5 bytes, valid x86-32.
        let displaced = [0x90, 0x90, 0x90, 0x66, 0x90];
        let result = generate_pe32_persistent_wrapper(
            wrapper_va,
            persistent_data_va,
            data_va,
            persistent_addr,
            &displaced,
            5,
            1000,
        );
        assert!(
            result.is_ok(),
            "PE32 persistent wrapper generation failed: {:?}",
            result.err()
        );
        let wrapper = result.expect("PE32 persistent wrapper generation should succeed");
        assert!(
            !wrapper.code.is_empty(),
            "PE32 persistent wrapper code is empty"
        );
        assert!(
            wrapper.code.len() > 100,
            "PE32 persistent wrapper too short: {} bytes",
            wrapper.code.len(),
        );
        // Verify displaced bytes appear in the wrapper (after relocation).
        let has_nop_sequence = wrapper.code.windows(3).any(|w| w == [0x90, 0x90, 0x90]);
        assert!(
            has_nop_sequence,
            "displaced NOPs not found in PE32 persistent wrapper"
        );
    }

    #[test]
    fn test_pe32_persistent_wrapper_contains_counter() {
        let wrapper_va = 0x00420000u64;
        let persistent_data_va = 0x0041F000u64;
        let data_va = 0x0041E000u64;
        let persistent_addr = 0x00401000u64;
        let displaced = [0x90, 0x90, 0x90, 0x66, 0x90];
        let persistent_count: u32 = 5000;
        let result = generate_pe32_persistent_wrapper(
            wrapper_va,
            persistent_data_va,
            data_va,
            persistent_addr,
            &displaced,
            5,
            persistent_count,
        );
        assert!(
            result.is_ok(),
            "PE32 persistent wrapper generation failed: {:?}",
            result.err()
        );
        let wrapper =
            result.expect("PE32 persistent wrapper generation should succeed for counter test");
        // The persistent_count value (5000 = 0x1388) should appear as a little-endian imm32.
        let count_bytes = persistent_count.to_le_bytes();
        let found = wrapper.code.windows(4).any(|w| w == count_bytes);
        assert!(
            found,
            "persistent_count {} not found in PE32 wrapper code",
            persistent_count
        );
    }
}
