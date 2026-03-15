use anyhow::{Context, Result, bail};

use crate::disasm::BasicBlock;
use crate::hook_trampoline::TargetAbi;
use crate::hooks::{self, HookSource, ResolvedHook};
use crate::patcher::{InstrumentationOptions, PatchResult};
use crate::pe::PeContext;
use crate::traits::{BinaryContext, Patcher, TrampolineGenerator};
use crate::trampoline::{DATA_SIZE, PREV_LOC_OFFSET};

use super::code_builder::{CodeBuilder, Label, PeArch};

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

// Auto-stripping of expendable section headers is provided by
// crate::pe::strip_expendable_section_headers.

impl Patcher for PePatcher {
    fn patch(
        &self,
        ctx: &dyn BinaryContext,
        blocks: &[BasicBlock],
        data: Vec<u8>,
        opts: &InstrumentationOptions,
        hooks: &[ResolvedHook],
    ) -> Result<PatchResult> {
        let pe_ctx = ctx
            .as_any()
            .downcast_ref::<crate::pe_impl::context::PeBinaryContext>()
            .ok_or_else(|| anyhow::anyhow!("PePatcher requires PeBinaryContext"))?;

        patch_pe(
            pe_ctx.inner(),
            blocks,
            data,
            opts,
            &*self.tramp_gen,
            hooks,
            self.hook_library_path.as_ref(),
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
/// Update PE headers after adding the .trcov section.
///
/// Uses `PatchSet` for overlap-checked, typed field writes — the same class
/// of bug that produced the PE32 SizeOfImage/CheckSum offset error is now
/// caught at patch-build time rather than producing a silently corrupt binary.
#[allow(clippy::too_many_arguments)]
fn update_pe_headers(
    data: &mut [u8],
    ctx: &PeContext,
    new_section_rva: u64,
    section_virtual_size: u64,
    section_alignment: u64,
    need_init_code: bool,
    init_code: &crate::trampoline::InitCode,
    current_section_count: usize,
) {
    use crate::binary_patch::PatchSet;

    let mut ps = PatchSet::new();

    let new_num_sections = (current_section_count + 1) as u16;
    ps.write_u16(
        ctx.number_of_sections_field_offset as usize,
        new_num_sections,
        "NumberOfSections",
    );

    let new_size_of_image =
        (new_section_rva + align_up(section_virtual_size, section_alignment)) as u32;
    ps.write_u32(
        ctx.size_of_image_field_offset as usize,
        new_size_of_image,
        "SizeOfImage",
    );

    ps.zero(ctx.checksum_field_offset as usize, 4, "CheckSum");

    if need_init_code {
        let new_entry_rva = (init_code.entry_va - ctx.image_base) as u32;
        ps.write_u32(
            ctx.entry_point_field_offset as usize,
            new_entry_rva,
            "AddressOfEntryPoint",
        );
    }

    // Disable ASLR: clear IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE.
    let dll_chars_off = if ctx.is_64bit {
        ctx.optional_header_offset as usize + 24 + 46
    } else {
        ctx.optional_header_offset as usize + 28 + 42
    };
    if dll_chars_off + 2 <= data.len() {
        let old = u16::from_le_bytes([data[dll_chars_off], data[dll_chars_off + 1]]);
        ps.write_u16(dll_chars_off, old & !0x0040, "DllCharacteristics");
    }

    // Zero base relocation data directory entry.
    let reloc_dd_off = if ctx.is_64bit {
        ctx.optional_header_offset as usize + 24 + 88 + 5 * 8
    } else {
        ctx.optional_header_offset as usize + 28 + 68 + 5 * 8
    };
    if reloc_dd_off + 8 <= data.len() {
        ps.zero(reloc_dd_off, 8, "BaseRelocationDirectory");
    }

    // Apply all patches with overlap detection.
    ps.apply(data).expect("PE header patches must not overlap");
}

/// Orchestrate PE patching:
/// 1. Compute new section layout
/// 2. Generate PE init code + trampolines
/// 3. Insert `.trcov` section header
/// 4. Redirect entry point
/// 5. Patch basic blocks with JMP rel32
/// 6. Append section data
/// 7. Update PE headers
fn patch_pe(
    ctx: &PeContext,
    blocks: &[BasicBlock],
    mut data: Vec<u8>,
    opts: &InstrumentationOptions,
    tramp_gen: &dyn TrampolineGenerator,
    resolved_hooks: &[ResolvedHook],
    hook_library_path: Option<&String>,
) -> Result<PatchResult> {
    let no_coverage = opts.no_coverage;
    let persistent_addr = opts.persistent_addr;
    let persistent_count = opts.persistent_count;
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
        let arch = if ctx.is_64bit {
            PeArch::X64
        } else {
            PeArch::X86
        };
        let gen_init = |va, dv, ep, hl: Option<&PeHookLibraryInfo>, pdv| {
            generate_pe_init_code_impl(va, dv, ep, hl, pdv, arch)
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

        let pw_params = crate::trampoline::PersistentWrapperParams {
            wrapper_va,
            persistent_data_va: p_data_va,
            data_va,
            persistent_addr: p_addr,
            displaced_bytes: &displaced,
            displaced_len,
            persistent_count,
            include_forkserver: false,
        };
        let wrapper = if ctx.is_64bit {
            generate_pe_persistent_wrapper(&pw_params)
                .context("failed to generate PE persistent wrapper")?
        } else {
            generate_pe32_persistent_wrapper(&pw_params)
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
                    &crate::hook_trampoline::ReturnHookContext {
                        entry_va,
                        ret_tramp_va,
                        hook_data_va,
                        shellcode_va: sc_va,
                        toggle_va,
                        return_slot_va: slot_va,
                    },
                    hook,
                    hook_abi,
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
    let mut current_section_count = ctx.number_of_sections as usize;
    let headers_end = ctx.size_of_headers as usize;

    let mut section_table_end =
        ctx.section_table_offset as usize + current_section_count * SECTION_HEADER_SIZE;

    if section_table_end + SECTION_HEADER_SIZE > headers_end {
        // Auto-strip expendable section headers to make room.
        let stripped = crate::pe::strip_expendable_section_headers(ctx, &mut data);
        if stripped > 0 {
            tracing::info!(
                "stripped {stripped} expendable section headers to make room for .trcov"
            );
            current_section_count -= stripped;
            section_table_end =
                ctx.section_table_offset as usize + current_section_count * SECTION_HEADER_SIZE;
            if section_table_end + SECTION_HEADER_SIZE > headers_end {
                bail!(
                    "no room for new section header even after stripping {stripped} sections: \
                     table ends at {section_table_end} + 40 > headers end at {headers_end}."
                );
            }
        } else {
            bail!(
                "no room for new section header: table ends at {} + 40 > headers end at {}.\n\
                 The PE header area is full and no expendable sections found to strip.",
                section_table_end,
                headers_end,
            );
        }
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
        current_section_count,
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

/// Unified PE init code generator for both x86 and x64 architectures.
///
/// On Windows, there is no shmat(). Instead we:
/// 1. Read `__AFL_SHM_ID` from the environment (PEB-based wide-string scan)
/// 2. Construct the SHM name "afl_shm_<id>"
/// 3. Walk PEB_LDR_DATA to find kernel32.dll
/// 4. Parse its export directory (DJB2 hash) to resolve OpenFileMappingA, MapViewOfFile
/// 5. Call them to set up SHM
/// 6. Optionally resolve persistent-mode functions (x64 only)
/// 7. Optionally resolve hook library handlers via LoadLibraryA + GetProcAddress
///
/// Uses `CodeBuilder` for all jump/branch operations — no manual fixups.
fn generate_pe_init_code_impl(
    init_va: u64,
    data_va: u64,
    original_entry: u64,
    hook_library: Option<&PeHookLibraryInfo>,
    persistent_data_va: Option<u64>,
    arch: PeArch,
) -> Result<crate::trampoline::InitCode> {
    let is64 = arch.is_64bit();
    let mut cb = CodeBuilder::new(init_va);
    let entry_va = init_va;

    // Pre-declare all labels used across sections.
    let epilogue_target = cb.label(); // early-exit target (hook_resolve or real_epilogue)
    let real_epilogue = cb.label();
    let next_env_var = cb.label();
    let env_scan_loop = cb.label();

    // ================================================================
    // Prologue: save callee-saved registers, allocate locals
    // ================================================================
    if is64 {
        cb.byte(0x55); // push rbp
        cb.raw(&[0x48, 0x89, 0xE5]); // mov rbp, rsp
        cb.byte(0x53); // push rbx
        cb.byte(0x56); // push rsi
        cb.byte(0x57); // push rdi
        cb.raw(&[0x41, 0x54]); // push r12
        cb.raw(&[0x41, 0x55]); // push r13
        cb.raw(&[0x41, 0x56]); // push r14
        cb.raw(&[0x41, 0x57]); // push r15
        // sub rsp, 568
        const LOCAL_SIZE_64: u32 = 568;
        cb.raw(&[0x48, 0x81, 0xEC]);
        cb.dword(LOCAL_SIZE_64);
    } else {
        cb.byte(0x55); // push ebp
        cb.raw(&[0x89, 0xE5]); // mov ebp, esp
        cb.byte(0x53); // push ebx
        cb.byte(0x56); // push esi
        cb.byte(0x57); // push edi
        // sub esp, 128
        const LOCAL_SIZE_32: u32 = 128;
        cb.raw(&[0x81, 0xEC]);
        cb.dword(LOCAL_SIZE_32);
    }

    // ================================================================
    // Load data area pointer
    // ================================================================
    let needle_fixup = if is64 {
        // lea r12, [rip + delta_to_data_va]
        let r12_fixup = cb.rip_rel_placeholder(&[0x4C, 0x8D, 0x25]);
        cb.patch_rip_rel(r12_fixup, data_va);

        // ================================================================
        // Step 1: PEB -> ProcessParameters -> Environment (x64)
        // ================================================================
        // mov rax, gs:[0x60]
        cb.raw(&[0x65, 0x48, 0x8B, 0x04, 0x25, 0x60, 0x00, 0x00, 0x00]);
        // mov rax, [rax + 0x20]
        cb.raw(&[0x48, 0x8B, 0x40, 0x20]);
        // mov rsi, [rax + 0x80]
        cb.raw(&[0x48, 0x8B, 0xB0, 0x80, 0x00, 0x00, 0x00]);
        // test rsi, rsi
        cb.raw(&[0x48, 0x85, 0xF6]);
        // jz epilogue_target
        cb.jcc_near(0x84, epilogue_target);

        // lea r13, [rip + needle] (placeholder)
        cb.rip_rel_placeholder(&[0x4C, 0x8D, 0x2D])
    } else {
        // mov esi, imm32 (data_va)
        cb.byte(0xBE);
        cb.dword(data_va as u32);

        // ================================================================
        // Step 1: PEB -> ProcessParameters -> Environment (x86)
        // ================================================================
        // mov eax, fs:[0x30]
        cb.raw(&[0x64, 0xA1, 0x30, 0x00, 0x00, 0x00]);
        // mov eax, [eax + 0x10]
        cb.raw(&[0x8B, 0x40, 0x10]);
        // mov edi, [eax + 0x48]
        cb.raw(&[0x8B, 0x78, 0x48]);
        // test edi, edi
        cb.raw(&[0x85, 0xFF]);
        // jz epilogue_target
        cb.jcc_near(0x84, epilogue_target);

        // mov ebx, imm32 (needle VA placeholder)
        cb.mov_imm32_placeholder(0xBB)
    };

    // ================================================================
    // Step 2: Scan environment for L"__AFL_SHM_ID="
    // ================================================================
    cb.bind(env_scan_loop);

    // Check end of environment block
    if is64 {
        // movzx eax, word [rsi]
        cb.raw(&[0x0F, 0xB7, 0x06]);
    } else {
        // movzx eax, word [edi]
        cb.raw(&[0x0F, 0xB7, 0x07]);
    }
    // test ax, ax
    cb.raw(&[0x66, 0x85, 0xC0]);
    // jz epilogue_target
    cb.jcc_near(0x84, epilogue_target);

    if is64 {
        // Needle compare: 3 qwords + 1 word
        // mov rdi, rsi
        cb.raw(&[0x48, 0x89, 0xF7]);

        // Compare first 8 bytes
        cb.raw(&[0x48, 0x8B, 0x06]); // mov rax, [rsi]
        cb.raw(&[0x49, 0x8B, 0x5D, 0x00]); // mov rbx, [r13]
        cb.raw(&[0x48, 0x39, 0xD8]); // cmp rax, rbx
        cb.jcc_near(0x85, next_env_var); // jne next_env_var

        // Compare bytes 8..16
        cb.raw(&[0x48, 0x8B, 0x46, 0x08]); // mov rax, [rsi + 8]
        cb.raw(&[0x49, 0x8B, 0x5D, 0x08]); // mov rbx, [r13 + 8]
        cb.raw(&[0x48, 0x39, 0xD8]); // cmp rax, rbx
        cb.jcc_near(0x85, next_env_var);

        // Compare bytes 16..24
        cb.raw(&[0x48, 0x8B, 0x46, 0x10]); // mov rax, [rsi + 16]
        cb.raw(&[0x49, 0x8B, 0x5D, 0x10]); // mov rbx, [r13 + 16]
        cb.raw(&[0x48, 0x39, 0xD8]); // cmp rax, rbx
        cb.jcc_near(0x85, next_env_var);

        // Compare bytes 24..26: "=" as wide
        cb.raw(&[0x0F, 0xB7, 0x46, 0x18]); // movzx eax, word [rsi + 24]
        cb.raw(&[0x66, 0x3D, 0x3D, 0x00]); // cmp ax, 0x003D
        cb.jcc_near(0x85, next_env_var);
    } else {
        // Needle compare: 6 dwords + 1 word
        // Compare first 4 bytes
        cb.raw(&[0x8B, 0x07]); // mov eax, [edi]
        cb.raw(&[0x3B, 0x03]); // cmp eax, [ebx]
        cb.jcc_near(0x85, next_env_var);

        // Compare bytes 4..8
        cb.raw(&[0x8B, 0x47, 0x04]); // mov eax, [edi + 4]
        cb.raw(&[0x3B, 0x43, 0x04]); // cmp eax, [ebx + 4]
        cb.jcc_near(0x85, next_env_var);

        // Compare bytes 8..12
        cb.raw(&[0x8B, 0x47, 0x08]); // mov eax, [edi + 8]
        cb.raw(&[0x3B, 0x43, 0x08]); // cmp eax, [ebx + 8]
        cb.jcc_near(0x85, next_env_var);

        // Compare bytes 12..16
        cb.raw(&[0x8B, 0x47, 0x0C]); // mov eax, [edi + 12]
        cb.raw(&[0x3B, 0x43, 0x0C]); // cmp eax, [ebx + 12]
        cb.jcc_near(0x85, next_env_var);

        // Compare bytes 16..20
        cb.raw(&[0x8B, 0x47, 0x10]); // mov eax, [edi + 16]
        cb.raw(&[0x3B, 0x43, 0x10]); // cmp eax, [ebx + 16]
        cb.jcc_near(0x85, next_env_var);

        // Compare bytes 20..24
        cb.raw(&[0x8B, 0x47, 0x14]); // mov eax, [edi + 20]
        cb.raw(&[0x3B, 0x43, 0x14]); // cmp eax, [ebx + 20]
        cb.jcc_near(0x85, next_env_var);

        // Compare bytes 24..26: "=" as wide
        cb.raw(&[0x0F, 0xB7, 0x47, 0x18]); // movzx eax, word [edi + 24]
        cb.raw(&[0x66, 0x3D, 0x3D, 0x00]); // cmp ax, 0x003D
        cb.jcc_near(0x85, next_env_var);
    }

    // ================================================================
    // SHM name build: parse decimal ID, construct "afl_shm_<id>"
    // ================================================================
    emit_shm_name_build(&mut cb, arch);

    // ================================================================
    // Step 4: Walk PEB_LDR_DATA to find kernel32.dll
    // ================================================================
    emit_kernel32_walk(&mut cb, arch, epilogue_target);

    // ================================================================
    // Step 5: Parse kernel32.dll export directory
    // ================================================================
    let need_hook_funcs = hook_library.is_some();
    emit_export_resolution(&mut cb, arch, need_hook_funcs, epilogue_target);

    // Save kernel32 base for persistent resolution (x64 only, before popping).
    if is64 && persistent_data_va.is_some() {
        // mov [rbp - 0x68], r15
        cb.raw(&[0x4C, 0x89, 0x7D, 0x98]);
    }

    // Pop saved values after export resolution.
    if is64 {
        cb.raw(&[0x41, 0x5F]); // pop r15
        cb.byte(0x58); // pop rax
        cb.raw(&[0x41, 0x5E]); // pop r14
    } else {
        cb.byte(0x5E); // pop esi (restore data_va)
    }

    // Check resolved SHM functions.
    if is64 {
        // test r8, r8; jz epilogue_target
        cb.raw(&[0x4D, 0x85, 0xC0]);
        cb.jcc_near(0x84, epilogue_target);
        // test r9, r9; jz epilogue_target
        cb.raw(&[0x4D, 0x85, 0xC9]);
        cb.jcc_near(0x84, epilogue_target);
    } else {
        // cmp dword [ebp - 0x14], 0; jz epilogue_target
        cb.raw(&[0x83, 0x7D, 0xEC, 0x00]);
        cb.jcc_near(0x84, epilogue_target);
        // cmp dword [ebp - 0x18], 0; jz epilogue_target
        cb.raw(&[0x83, 0x7D, 0xE8, 0x00]);
        cb.jcc_near(0x84, epilogue_target);
    }

    // ================================================================
    // Step 6: Call OpenFileMappingA(FILE_MAP_WRITE, FALSE, name)
    // ================================================================
    if is64 {
        const LOCAL_SIZE_64: u32 = 568;
        let name_rbp_offset: i32 = -(8 + 56 + LOCAL_SIZE_64 as i32) + 32;

        // Save r8, r9
        cb.raw(&[0x41, 0x50]); // push r8
        cb.raw(&[0x41, 0x51]); // push r9
        // sub rsp, 32
        cb.raw(&[0x48, 0x83, 0xEC, 0x20]);
        // mov ecx, 2
        cb.raw(&[0xB9, 0x02, 0x00, 0x00, 0x00]);
        // xor edx, edx
        cb.raw(&[0x31, 0xD2]);
        // lea r8, [rbp + name_rbp_offset]
        cb.raw(&[0x4C, 0x8D, 0x85]);
        cb.dword(name_rbp_offset as u32);
        // add rsp, 32 / pop r9 / pop r8
        cb.raw(&[0x48, 0x83, 0xC4, 0x20]);
        cb.raw(&[0x41, 0x59]); // pop r9
        cb.raw(&[0x41, 0x58]); // pop r8

        // Store function pointers in locals
        cb.raw(&[0x4C, 0x89, 0x45, 0xB8]); // mov [rbp - 0x48], r8
        cb.raw(&[0x4C, 0x89, 0x4D, 0xB0]); // mov [rbp - 0x50], r9

        // Actual call: sub rsp,32; setup args; call; add rsp,32
        cb.raw(&[0x48, 0x83, 0xEC, 0x20]);
        cb.raw(&[0xB9, 0x02, 0x00, 0x00, 0x00]); // mov ecx, 2
        cb.raw(&[0x31, 0xD2]); // xor edx, edx
        cb.raw(&[0x4C, 0x8D, 0x85]); // lea r8, [rbp + offset]
        cb.dword(name_rbp_offset as u32);
        cb.raw(&[0xFF, 0x55, 0xB8]); // call [rbp - 0x48]
        cb.raw(&[0x48, 0x83, 0xC4, 0x20]); // add rsp, 32

        // Check result
        cb.raw(&[0x48, 0x85, 0xC0]); // test rax, rax
        cb.jcc_near(0x84, epilogue_target);

        // ================================================================
        // Step 7: Call MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, 65536)
        // ================================================================
        cb.raw(&[0x48, 0x89, 0xC3]); // mov rbx, rax
        cb.raw(&[0x48, 0x83, 0xEC, 0x30]); // sub rsp, 48
        cb.raw(&[0x48, 0x89, 0xD9]); // mov rcx, rbx
        cb.raw(&[0xBA, 0x02, 0x00, 0x00, 0x00]); // mov edx, 2
        cb.raw(&[0x45, 0x31, 0xC0]); // xor r8d, r8d
        cb.raw(&[0x45, 0x31, 0xC9]); // xor r9d, r9d
        cb.raw(&[0x48, 0xC7, 0x44, 0x24, 0x20, 0x00, 0x00, 0x01, 0x00]); // mov qword [rsp+32], 65536
        cb.raw(&[0xFF, 0x55, 0xB0]); // call [rbp - 0x50]
        cb.raw(&[0x48, 0x83, 0xC4, 0x30]); // add rsp, 48

        cb.raw(&[0x48, 0x85, 0xC0]); // test rax, rax
        cb.jcc_near(0x84, epilogue_target);

        // Store SHM pointer: mov [r12], rax
        cb.raw(&[0x49, 0x89, 0x04, 0x24]);
    } else {
        // x86: stdcall OpenFileMappingA
        let name_ebp_offset: i32 = -128;
        // lea eax, [ebp + name_offset]
        cb.raw(&[0x8D, 0x85]);
        cb.dword(name_ebp_offset as u32);
        cb.byte(0x50); // push eax (lpName)
        cb.raw(&[0x6A, 0x00]); // push 0 (bInheritHandle)
        cb.raw(&[0x6A, 0x02]); // push 2 (FILE_MAP_WRITE)
        cb.raw(&[0xFF, 0x55, 0xEC]); // call [ebp - 0x14]
        cb.raw(&[0x83, 0xC4, 0x0C]); // add esp, 12

        cb.raw(&[0x85, 0xC0]); // test eax, eax
        cb.jcc_near(0x84, epilogue_target);

        // MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, 65536)
        cb.raw(&[0x89, 0xC3]); // mov ebx, eax
        cb.byte(0x68);
        cb.dword(65536); // push 65536
        cb.raw(&[0x6A, 0x00]); // push 0
        cb.raw(&[0x6A, 0x00]); // push 0
        cb.raw(&[0x6A, 0x02]); // push 2
        cb.byte(0x53); // push ebx
        cb.raw(&[0xFF, 0x55, 0xE8]); // call [ebp - 0x18]
        cb.raw(&[0x83, 0xC4, 0x14]); // add esp, 20

        cb.raw(&[0x85, 0xC0]); // test eax, eax
        cb.jcc_near(0x84, epilogue_target);

        // Store SHM pointer
        cb.raw(&[0x89, 0x06]); // mov [esi], eax
        cb.raw(&[0xC7, 0x46, 0x04, 0x00, 0x00, 0x00, 0x00]); // mov dword [esi+4], 0
    }

    // ================================================================
    // Step 7b (optional): Persistent resolve (x64 only)
    // ================================================================
    if let Some(p_data_va) = persistent_data_va.filter(|_| is64) {
        emit_persistent_resolve(&mut cb, p_data_va);
    }

    // ================================================================
    // Step 8 (optional): Hook library resolution via LoadLibraryA + GetProcAddress
    // ================================================================
    let mut hook_string_fixups: Vec<(usize, usize)> = Vec::new();
    if let Some(hook_lib) = hook_library {
        emit_hook_library_resolution(
            &mut cb,
            arch,
            hook_lib,
            real_epilogue,
            &mut hook_string_fixups,
        );
    }

    // ================================================================
    // next_env_var: skip current env var, loop back to env_scan_loop
    // ================================================================
    cb.bind(next_env_var);

    if is64 {
        let skip_wchar = cb.label();
        cb.bind(skip_wchar);
        cb.raw(&[0x0F, 0xB7, 0x06]); // movzx eax, word [rsi]
        cb.raw(&[0x48, 0x83, 0xC6, 0x02]); // add rsi, 2
        cb.raw(&[0x66, 0x85, 0xC0]); // test ax, ax
        cb.jcc_short(0x75, skip_wchar); // jnz skip_wchar
    } else {
        let skip_wchar = cb.label();
        cb.bind(skip_wchar);
        cb.raw(&[0x0F, 0xB7, 0x07]); // movzx eax, word [edi]
        cb.raw(&[0x83, 0xC7, 0x02]); // add edi, 2
        cb.raw(&[0x66, 0x85, 0xC0]); // test ax, ax
        cb.jcc_short(0x75, skip_wchar); // jnz skip_wchar
    }
    cb.jmp_near(env_scan_loop);

    // ================================================================
    // Hook-only resolve block (when SHM setup was skipped but hooks needed)
    // ================================================================
    if let Some(hook_lib) = hook_library {
        cb.bind(epilogue_target);
        emit_hook_only_resolve(&mut cb, arch, hook_lib, real_epilogue);
    } else {
        // No hooks: epilogue_target == real_epilogue
        cb.bind(epilogue_target);
    }

    // ================================================================
    // Real epilogue: restore registers, JMP to original entry point
    // ================================================================
    cb.bind(real_epilogue);

    if is64 {
        const LOCAL_SIZE_64: u32 = 568;
        cb.raw(&[0x48, 0x81, 0xC4]); // add rsp, LOCAL_SIZE
        cb.dword(LOCAL_SIZE_64);
        cb.raw(&[0x41, 0x5F]); // pop r15
        cb.raw(&[0x41, 0x5E]); // pop r14
        cb.raw(&[0x41, 0x5D]); // pop r13
        cb.raw(&[0x41, 0x5C]); // pop r12
        cb.byte(0x5F); // pop rdi
        cb.byte(0x5E); // pop rsi
        cb.byte(0x5B); // pop rbx
        cb.byte(0x5D); // pop rbp
    } else {
        const LOCAL_SIZE_32: u32 = 128;
        cb.raw(&[0x81, 0xC4]); // add esp, LOCAL_SIZE
        cb.dword(LOCAL_SIZE_32);
        cb.byte(0x5F); // pop edi
        cb.byte(0x5E); // pop esi
        cb.byte(0x5B); // pop ebx
        cb.byte(0x5D); // pop ebp
    }

    // JMP to original entry point (E9 rel32)
    cb.byte(0xE9);
    let jmp_rip = cb.va() + 4;
    let jmp_rel = (original_entry as i64 - jmp_rip as i64) as i32;
    cb.dword(jmp_rel as u32);

    // ================================================================
    // Embedded data: needle string, hook strings
    // ================================================================

    // Needle: "__AFL_SHM_ID=" as UTF-16LE (13 wchars = 26 bytes)
    let needle_va = cb.va();
    if is64 {
        cb.patch_rip_rel(needle_fixup, needle_va);
    } else {
        cb.patch_imm32(needle_fixup, needle_va as u32);
    }
    for ch in b"__AFL_SHM_ID=" {
        cb.byte(*ch);
        cb.byte(0x00);
    }

    // Hook library strings
    if let Some(hook_lib) = hook_library {
        cb.align_to(4);

        let mut hook_string_offsets: Vec<u64> = Vec::new();
        // String 0: library path
        hook_string_offsets.push(cb.va());
        cb.raw(hook_lib.library_path.as_bytes());
        cb.byte(0x00);

        // Strings 1..N: handler symbol names
        for (name, _) in &hook_lib.handlers {
            hook_string_offsets.push(cb.va());
            cb.raw(name.as_bytes());
            cb.byte(0x00);
        }

        // Fix up string references
        for &(fixup_pos, string_idx) in &hook_string_fixups {
            let string_va = hook_string_offsets[string_idx];
            if is64 {
                cb.patch_rip_rel(fixup_pos, string_va);
            } else {
                cb.patch_imm32(fixup_pos, string_va as u32);
            }
        }
    }

    cb.align_to(8);

    let code = cb.finish();
    Ok(crate::trampoline::InitCode {
        va: init_va,
        code,
        entry_va,
    })
}

/// Emit code to parse decimal SHM ID from wide-string env value and build "afl_shm_<id>" on stack.
fn emit_shm_name_build(cb: &mut CodeBuilder, arch: PeArch) {
    let is64 = arch.is_64bit();

    let parse_done = cb.label();
    let parse_loop = cb.label();
    let conv_loop = cb.label();
    let conv_done = cb.label();
    let reverse_loop = cb.label();

    if is64 {
        // lea rsi, [rsi + 26] ; skip "__AFL_SHM_ID=" (13 wchars)
        cb.raw(&[0x48, 0x8D, 0x76, 0x1A]);
        // xor r14, r14 ; shmid = 0
        cb.raw(&[0x4D, 0x31, 0xF6]);
    } else {
        // lea edi, [edi + 26]
        cb.raw(&[0x8D, 0x7F, 0x1A]);
        // xor ebx, ebx
        cb.raw(&[0x31, 0xDB]);
    }

    // .parse_loop:
    cb.bind(parse_loop);
    if is64 {
        cb.raw(&[0x0F, 0xB7, 0x06]); // movzx eax, word [rsi]
    } else {
        cb.raw(&[0x0F, 0xB7, 0x07]); // movzx eax, word [edi]
    }
    cb.raw(&[0x66, 0x2D, 0x30, 0x00]); // sub ax, '0'
    cb.raw(&[0x66, 0x3D, 0x09, 0x00]); // cmp ax, 9
    cb.jcc_short(0x77, parse_done); // ja parse_done

    if is64 {
        cb.raw(&[0x4D, 0x6B, 0xF6, 0x0A]); // imul r14, r14, 10
        cb.raw(&[0x0F, 0xB7, 0xC0]); // movzx eax, ax
        cb.raw(&[0x49, 0x01, 0xC6]); // add r14, rax
        cb.raw(&[0x48, 0x83, 0xC6, 0x02]); // add rsi, 2
    } else {
        cb.raw(&[0x6B, 0xDB, 0x0A]); // imul ebx, ebx, 10
        cb.raw(&[0x0F, 0xB7, 0xC0]); // movzx eax, ax
        cb.raw(&[0x01, 0xC3]); // add ebx, eax
        cb.raw(&[0x83, 0xC7, 0x02]); // add edi, 2
    }
    cb.jmp_short(parse_loop);

    // .parse_done:
    cb.bind(parse_done);

    if !is64 {
        // mov [ebp - 0x04], ebx ; save SHM ID
        cb.raw(&[0x89, 0x5D, 0xFC]);
    }

    // Build name buffer
    if is64 {
        // lea rdi, [rsp + 32]
        cb.raw(&[0x48, 0x8D, 0x7C, 0x24, 0x20]);
        // Write "afl_shm_" prefix as imm64
        let prefix = u64::from_le_bytes(*b"afl_shm_");
        cb.raw(&[0x48, 0xB8]); // mov rax, imm64
        cb.qword(prefix);
        cb.raw(&[0x48, 0x89, 0x07]); // mov [rdi], rax
        cb.raw(&[0x48, 0x8D, 0x7F, 0x08]); // lea rdi, [rdi + 8]
        // mov rax, r14
        cb.raw(&[0x4C, 0x89, 0xF0]);
        // mov rbx, rdi
        cb.raw(&[0x48, 0x89, 0xFB]);
        // test rax, rax
        cb.raw(&[0x48, 0x85, 0xC0]);
    } else {
        // lea edi, [ebp - 128]
        let name_ebp_offset: i32 = -128;
        cb.raw(&[0x8D, 0xBD]);
        cb.dword(name_ebp_offset as u32);
        // Write "afl_shm_" as two dwords
        let prefix_lo = u32::from_le_bytes(*b"afl_");
        let prefix_hi = u32::from_le_bytes(*b"shm_");
        cb.raw(&[0xC7, 0x07]); // mov dword [edi], prefix_lo
        cb.dword(prefix_lo);
        cb.raw(&[0xC7, 0x47, 0x04]); // mov dword [edi+4], prefix_hi
        cb.dword(prefix_hi);
        cb.raw(&[0x83, 0xC7, 0x08]); // add edi, 8
        // mov eax, ebx
        cb.raw(&[0x89, 0xD8]);
        // mov ebx, edi
        cb.raw(&[0x89, 0xFB]);
        // test eax, eax
        cb.raw(&[0x85, 0xC0]);
    }

    // jnz conv_loop
    cb.jcc_short(0x75, conv_loop);

    // Zero case: write '0'
    if is64 {
        cb.raw(&[0xC6, 0x07, 0x30]); // mov byte [rdi], '0'
        cb.raw(&[0x48, 0xFF, 0xC7]); // inc rdi
    } else {
        cb.raw(&[0xC6, 0x07, 0x30]); // mov byte [edi], '0'
        cb.byte(0x47); // inc edi
    }
    cb.jmp_short(conv_done);

    // .conv_loop: div-10 loop
    cb.bind(conv_loop);
    cb.raw(&[0x31, 0xD2]); // xor edx, edx
    if is64 {
        cb.raw(&[0x48, 0xC7, 0xC1, 0x0A, 0x00, 0x00, 0x00]); // mov rcx, 10
        cb.raw(&[0x48, 0xF7, 0xF1]); // div rcx
    } else {
        cb.raw(&[0xB9, 0x0A, 0x00, 0x00, 0x00]); // mov ecx, 10
        cb.raw(&[0xF7, 0xF1]); // div ecx
    }
    cb.raw(&[0x80, 0xC2, 0x30]); // add dl, '0'
    cb.raw(&[0x88, 0x17]); // mov [rdi/edi], dl
    if is64 {
        cb.raw(&[0x48, 0xFF, 0xC7]); // inc rdi
        cb.raw(&[0x48, 0x85, 0xC0]); // test rax, rax
    } else {
        cb.byte(0x47); // inc edi
        cb.raw(&[0x85, 0xC0]); // test eax, eax
    }
    cb.jcc_short(0x75, conv_loop); // jnz conv_loop

    // Null-terminate and reverse
    cb.raw(&[0xC6, 0x07, 0x00]); // mov byte [rdi/edi], 0
    if is64 {
        cb.raw(&[0x48, 0xFF, 0xCF]); // dec rdi
    } else {
        cb.byte(0x4F); // dec edi
    }

    // .reverse_loop:
    cb.bind(reverse_loop);
    if is64 {
        cb.raw(&[0x48, 0x39, 0xFB]); // cmp rbx, rdi
    } else {
        cb.raw(&[0x39, 0xFB]); // cmp ebx, edi
    }
    cb.jcc_short(0x7D, conv_done); // jge conv_done
    cb.raw(&[0x8A, 0x03]); // mov al, [rbx/ebx]
    cb.raw(&[0x8A, 0x0F]); // mov cl, [rdi/edi]
    cb.raw(&[0x88, 0x0B]); // mov [rbx/ebx], cl
    cb.raw(&[0x88, 0x07]); // mov [rdi/edi], al
    if is64 {
        cb.raw(&[0x48, 0xFF, 0xC3]); // inc rbx
        cb.raw(&[0x48, 0xFF, 0xCF]); // dec rdi
    } else {
        cb.byte(0x43); // inc ebx
        cb.byte(0x4F); // dec edi
    }
    cb.jmp_short(reverse_loop);

    // .conv_done:
    cb.bind(conv_done);
}

/// Walk PEB_LDR_DATA InLoadOrderModuleList to find kernel32.dll.
/// On success: x64 r15 = kernel32 base, x86 esi = kernel32 base (esi pushed first).
fn emit_kernel32_walk(cb: &mut CodeBuilder, arch: PeArch, epilogue: Label) {
    let is64 = arch.is_64bit();
    let ldr_loop = cb.label();
    let ldr_next = cb.label();
    let resolve_exports = cb.label();

    if is64 {
        // mov rax, gs:[0x60]
        cb.raw(&[0x65, 0x48, 0x8B, 0x04, 0x25, 0x60, 0x00, 0x00, 0x00]);
        // mov rax, [rax + 0x18]
        cb.raw(&[0x48, 0x8B, 0x40, 0x18]);
        // lea rbx, [rax + 0x10]
        cb.raw(&[0x48, 0x8D, 0x58, 0x10]);
        // mov rdi, [rbx]
        cb.raw(&[0x48, 0x8B, 0x3B]);
    } else {
        // mov eax, fs:[0x30]
        cb.raw(&[0x64, 0xA1, 0x30, 0x00, 0x00, 0x00]);
        // mov eax, [eax + 0x0C]
        cb.raw(&[0x8B, 0x40, 0x0C]);
        // lea ebx, [eax + 0x0C]
        cb.raw(&[0x8D, 0x58, 0x0C]);
        // mov edi, [ebx]
        cb.raw(&[0x8B, 0x3B]);
    }

    // .ldr_loop:
    cb.bind(ldr_loop);
    if is64 {
        cb.raw(&[0x48, 0x39, 0xDF]); // cmp rdi, rbx
    } else {
        cb.raw(&[0x39, 0xDF]); // cmp edi, ebx
    }
    cb.jcc_near(0x84, epilogue); // je epilogue (kernel32 not found)

    if is64 {
        // movzx ecx, word [rdi + 0x58]
        cb.raw(&[0x0F, 0xB7, 0x4F, 0x58]);
    } else {
        // movzx ecx, word [edi + 0x2C]
        cb.raw(&[0x0F, 0xB7, 0x4F, 0x2C]);
    }
    // cmp ecx, 24
    cb.raw(&[0x83, 0xF9, 0x18]);
    cb.jcc_short(0x75, ldr_next); // jne ldr_next

    if is64 {
        // mov rsi, [rdi + 0x60]
        cb.raw(&[0x48, 0x8B, 0x77, 0x60]);
        // movzx eax, word [rsi]
        cb.raw(&[0x0F, 0xB7, 0x06]);
    } else {
        // mov ecx, [edi + 0x30]
        cb.raw(&[0x8B, 0x4F, 0x30]);
        // movzx eax, word [ecx]
        cb.raw(&[0x0F, 0xB7, 0x01]);
    }
    cb.raw(&[0x0C, 0x20]); // or al, 0x20
    cb.raw(&[0x3C, 0x6B]); // cmp al, 'k'
    cb.jcc_short(0x75, ldr_next);

    if is64 {
        cb.raw(&[0x0F, 0xB7, 0x46, 0x0C]); // movzx eax, word [rsi + 12]
    } else {
        cb.raw(&[0x0F, 0xB7, 0x41, 0x0C]); // movzx eax, word [ecx + 12]
    }
    cb.raw(&[0x3C, 0x33]); // cmp al, '3'
    cb.jcc_short(0x75, ldr_next);

    if is64 {
        cb.raw(&[0x0F, 0xB7, 0x46, 0x0E]); // movzx eax, word [rsi + 14]
    } else {
        cb.raw(&[0x0F, 0xB7, 0x41, 0x0E]); // movzx eax, word [ecx + 14]
    }
    cb.raw(&[0x3C, 0x32]); // cmp al, '2'
    cb.jcc_short(0x75, ldr_next);

    // Found kernel32!
    if is64 {
        cb.raw(&[0x4C, 0x8B, 0x7F, 0x30]); // mov r15, [rdi + 0x30]
    } else {
        cb.byte(0x56); // push esi (save data_va)
        cb.raw(&[0x8B, 0x77, 0x18]); // mov esi, [edi + 0x18]
    }
    cb.jmp_short(resolve_exports);

    // .ldr_next:
    cb.bind(ldr_next);
    if is64 {
        cb.raw(&[0x48, 0x8B, 0x3F]); // mov rdi, [rdi]
    } else {
        cb.raw(&[0x8B, 0x3F]); // mov edi, [edi]
    }
    cb.jmp_short(ldr_loop);

    // .resolve_exports:
    cb.bind(resolve_exports);
}

/// Parse kernel32 export directory, resolve OpenFileMappingA, MapViewOfFile,
/// and optionally LoadLibraryA, GetProcAddress via DJB2 hash matching.
fn emit_export_resolution(
    cb: &mut CodeBuilder,
    arch: PeArch,
    need_hook_funcs: bool,
    epilogue: Label,
) {
    let is64 = arch.is_64bit();
    let exports_done = cb.label();
    let export_loop = cb.label();
    let export_next = cb.label();
    let export_continue = cb.label();

    // Parse PE export directory header.
    if is64 {
        cb.raw(&[0x41, 0x8B, 0x47, 0x3C]); // mov eax, [r15 + 0x3C]
        cb.raw(&[0x49, 0x8D, 0x1C, 0x07]); // lea rbx, [r15 + rax]
        cb.raw(&[0x8B, 0x83, 0x88, 0x00, 0x00, 0x00]); // mov eax, [rbx + 0x88]
    } else {
        cb.raw(&[0x8B, 0x46, 0x3C]); // mov eax, [esi + 0x3C]
        cb.raw(&[0x8D, 0x1C, 0x06]); // lea ebx, [esi + eax]
        cb.raw(&[0x8B, 0x43, 0x78]); // mov eax, [ebx + 0x78]
    }
    cb.raw(&[0x85, 0xC0]); // test eax, eax
    cb.jcc_near(0x84, epilogue); // jz epilogue (no exports)

    if is64 {
        // lea rsi, [r15 + rax]
        cb.raw(&[0x49, 0x8D, 0x34, 0x07]);
        // mov ecx, [rsi + 0x18]
        cb.raw(&[0x8B, 0x4E, 0x18]);
        // AddressOfNames
        cb.raw(&[0x8B, 0x46, 0x20]); // mov eax, [rsi + 0x20]
        // push r14 (save SHM ID)
        cb.raw(&[0x41, 0x56]);
        cb.raw(&[0x4D, 0x8D, 0x2C, 0x07]); // lea r13, [r15 + rax]
        // AddressOfNameOrdinals
        cb.raw(&[0x8B, 0x46, 0x24]); // mov eax, [rsi + 0x24]
        cb.raw(&[0x4D, 0x8D, 0x34, 0x07]); // lea r14, [r15 + rax]
        // AddressOfFunctions
        cb.raw(&[0x8B, 0x46, 0x1C]); // mov eax, [rsi + 0x1C]
        cb.byte(0x50); // push rax
        cb.raw(&[0x41, 0x57]); // push r15

        // Init OpenFileMappingA(r8), MapViewOfFile(r9)
        cb.raw(&[0x4D, 0x31, 0xC0]); // xor r8, r8
        cb.raw(&[0x4D, 0x31, 0xC9]); // xor r9, r9
        // Init LoadLibraryA/GetProcAddress slots
        cb.raw(&[0x48, 0xC7, 0x45, 0xA8, 0x00, 0x00, 0x00, 0x00]); // [rbp-0x58]=0
        cb.raw(&[0x48, 0xC7, 0x45, 0xA0, 0x00, 0x00, 0x00, 0x00]); // [rbp-0x60]=0
    } else {
        // lea ebx, [esi + eax]
        cb.raw(&[0x8D, 0x1C, 0x06]);
        cb.raw(&[0x8B, 0x4B, 0x18]); // mov ecx, [ebx + 0x18]
        // AddressOfNames
        cb.raw(&[0x8B, 0x43, 0x20]); // mov eax, [ebx + 0x20]
        cb.raw(&[0x8D, 0x04, 0x06]); // lea eax, [esi + eax]
        cb.raw(&[0x89, 0x45, 0xF8]); // mov [ebp - 0x08], eax
        // AddressOfNameOrdinals
        cb.raw(&[0x8B, 0x43, 0x24]); // mov eax, [ebx + 0x24]
        cb.raw(&[0x8D, 0x04, 0x06]); // lea eax, [esi + eax]
        cb.raw(&[0x89, 0x45, 0xF4]); // mov [ebp - 0x0C], eax
        // AddressOfFunctions
        cb.raw(&[0x8B, 0x43, 0x1C]); // mov eax, [ebx + 0x1C]
        cb.raw(&[0x8D, 0x04, 0x06]); // lea eax, [esi + eax]
        cb.raw(&[0x89, 0x45, 0xF0]); // mov [ebp - 0x10], eax

        // Init function slots
        cb.raw(&[0xC7, 0x45, 0xEC, 0x00, 0x00, 0x00, 0x00]); // [ebp-0x14]=0
        cb.raw(&[0xC7, 0x45, 0xE8, 0x00, 0x00, 0x00, 0x00]); // [ebp-0x18]=0
        if need_hook_funcs {
            cb.raw(&[0xC7, 0x45, 0xE4, 0x00, 0x00, 0x00, 0x00]); // [ebp-0x1C]=0
            cb.raw(&[0xC7, 0x45, 0xE0, 0x00, 0x00, 0x00, 0x00]); // [ebp-0x20]=0
        }
    }

    let hash_ofm = djb2_hash(b"OpenFileMappingA");
    let hash_mvf = djb2_hash(b"MapViewOfFile");
    let hash_lla = djb2_hash(b"LoadLibraryA");
    let hash_gpa = djb2_hash(b"GetProcAddress");

    cb.raw(&[0x31, 0xD2]); // xor edx, edx

    // .export_loop:
    cb.bind(export_loop);
    cb.raw(&[0x39, 0xCA]); // cmp edx, ecx
    cb.jcc_near(0x8D, exports_done); // jge exports_done

    // Load name pointer
    if is64 {
        cb.raw(&[0x41, 0x8B, 0x44, 0x95, 0x00]); // mov eax, [r13 + rdx*4]
        cb.raw(&[0x49, 0x8D, 0x3C, 0x07]); // lea rdi, [r15 + rax]
    } else {
        cb.raw(&[0x8B, 0x45, 0xF8]); // mov eax, [ebp - 0x08]
        cb.raw(&[0x8B, 0x04, 0x90]); // mov eax, [eax + edx*4]
        cb.raw(&[0x8D, 0x3C, 0x06]); // lea edi, [esi + eax]
    }

    // DJB2 hash of name
    emit_djb2_hash_loop(cb, arch);

    // Hash matching chain
    let check_mvf = cb.label();
    let check_ll = cb.label();

    // cmp eax, hash_open_file_mapping
    cb.byte(0x3D);
    cb.dword(hash_ofm);
    cb.jcc_short(0x75, check_mvf);

    // Resolve OpenFileMappingA
    emit_resolve_export(cb, arch, ExportDest::OpenFileMapping);
    cb.jmp_short(export_next);

    cb.bind(check_mvf);
    cb.byte(0x3D);
    cb.dword(hash_mvf);
    cb.jcc_short(0x75, check_ll);

    // Resolve MapViewOfFile
    emit_resolve_export(cb, arch, ExportDest::MapViewOfFile);
    cb.jmp_short(export_next);

    cb.bind(check_ll);

    if need_hook_funcs {
        let check_gpa = cb.label();
        // cmp eax, hash_load_library_a
        cb.byte(0x3D);
        cb.dword(hash_lla);
        cb.jcc_short(0x75, check_gpa);
        emit_resolve_export(cb, arch, ExportDest::LoadLibraryA);
        cb.jmp_short(export_next);

        cb.bind(check_gpa);
        cb.byte(0x3D);
        cb.dword(hash_gpa);
        cb.jcc_short(0x75, export_next);
        emit_resolve_export(cb, arch, ExportDest::GetProcAddress);
    }

    // .export_next: check if all found
    cb.bind(export_next);

    if need_hook_funcs {
        // Check all 4 functions found
        if is64 {
            cb.raw(&[0x4D, 0x85, 0xC0]); // test r8, r8
            cb.jcc_short(0x74, export_continue);
            cb.raw(&[0x4D, 0x85, 0xC9]); // test r9, r9
            cb.jcc_short(0x74, export_continue);
            cb.raw(&[0x48, 0x83, 0x7D, 0xA8, 0x00]); // cmp qword [rbp-0x58], 0
            cb.jcc_short(0x74, export_continue);
            cb.raw(&[0x48, 0x83, 0x7D, 0xA0, 0x00]); // cmp qword [rbp-0x60], 0
        } else {
            cb.raw(&[0x83, 0x7D, 0xEC, 0x00]); // cmp dword [ebp-0x14], 0
            cb.jcc_short(0x74, export_continue);
            cb.raw(&[0x83, 0x7D, 0xE8, 0x00]); // cmp dword [ebp-0x18], 0
            cb.jcc_short(0x74, export_continue);
            cb.raw(&[0x83, 0x7D, 0xE4, 0x00]); // cmp dword [ebp-0x1C], 0
            cb.jcc_short(0x74, export_continue);
            cb.raw(&[0x83, 0x7D, 0xE0, 0x00]); // cmp dword [ebp-0x20], 0
        }
        cb.jcc_near(0x85, exports_done); // jnz exports_done (all found)
    } else {
        // Check 2 functions found
        if is64 {
            cb.raw(&[0x4D, 0x85, 0xC0]); // test r8, r8
            cb.jcc_short(0x74, export_continue);
            cb.raw(&[0x4D, 0x85, 0xC9]); // test r9, r9
        } else {
            cb.raw(&[0x83, 0x7D, 0xEC, 0x00]); // cmp dword [ebp-0x14], 0
            cb.jcc_short(0x74, export_continue);
            cb.raw(&[0x83, 0x7D, 0xE8, 0x00]); // cmp dword [ebp-0x18], 0
        }
        cb.jcc_near(0x85, exports_done); // jnz exports_done
    }

    // .export_continue:
    cb.bind(export_continue);
    cb.raw(&[0xFF, 0xC2]); // inc edx
    cb.jmp_near(export_loop);

    // .exports_done:
    cb.bind(exports_done);
}

/// Which export function to resolve.
enum ExportDest {
    OpenFileMapping,
    MapViewOfFile,
    LoadLibraryA,
    GetProcAddress,
}

/// Emit the instruction sequence to resolve an export from the current index.
fn emit_resolve_export(cb: &mut CodeBuilder, arch: PeArch, dest: ExportDest) {
    if arch.is_64bit() {
        // movzx eax, word [r14 + rdx*2] ; ordinal
        cb.raw(&[0x41, 0x0F, 0xB7, 0x04, 0x56]);
        // mov ebx, [rsp + 8] ; AddressOfFunctions RVA
        cb.raw(&[0x8B, 0x5C, 0x24, 0x08]);
        // lea rbx, [r15 + rbx]
        cb.raw(&[0x49, 0x8D, 0x1C, 0x1F]);
        // mov eax, [rbx + rax*4]
        cb.raw(&[0x8B, 0x04, 0x83]);
        match dest {
            ExportDest::OpenFileMapping => {
                // lea r8, [r15 + rax]
                cb.raw(&[0x4D, 0x8D, 0x04, 0x07]);
            }
            ExportDest::MapViewOfFile => {
                // lea r9, [r15 + rax]
                cb.raw(&[0x4D, 0x8D, 0x0C, 0x07]);
            }
            ExportDest::LoadLibraryA => {
                // lea rax, [r15 + rax]; mov [rbp-0x58], rax
                cb.raw(&[0x49, 0x8D, 0x04, 0x07]);
                cb.raw(&[0x48, 0x89, 0x45, 0xA8]);
            }
            ExportDest::GetProcAddress => {
                // lea rax, [r15 + rax]; mov [rbp-0x60], rax
                cb.raw(&[0x49, 0x8D, 0x04, 0x07]);
                cb.raw(&[0x48, 0x89, 0x45, 0xA0]);
            }
        }
    } else {
        // mov ebx, [ebp - 0x0C] ; AddressOfNameOrdinals
        cb.raw(&[0x8B, 0x5D, 0xF4]);
        // movzx eax, word [ebx + edx*2]
        cb.raw(&[0x0F, 0xB7, 0x04, 0x53]);
        // mov ebx, [ebp - 0x10] ; AddressOfFunctions
        cb.raw(&[0x8B, 0x5D, 0xF0]);
        // mov eax, [ebx + eax*4]
        cb.raw(&[0x8B, 0x04, 0x83]);
        // lea eax, [esi + eax]
        cb.raw(&[0x8D, 0x04, 0x06]);
        match dest {
            ExportDest::OpenFileMapping => cb.raw(&[0x89, 0x45, 0xEC]), // mov [ebp-0x14], eax
            ExportDest::MapViewOfFile => cb.raw(&[0x89, 0x45, 0xE8]),   // mov [ebp-0x18], eax
            ExportDest::LoadLibraryA => cb.raw(&[0x89, 0x45, 0xE4]),    // mov [ebp-0x1C], eax
            ExportDest::GetProcAddress => cb.raw(&[0x89, 0x45, 0xE0]),  // mov [ebp-0x20], eax
        }
    }
}

/// Emit DJB2 hash computation loop over null-terminated string at rdi/edi.
/// Saves and restores rdx/rcx (edx/ecx). Result in eax.
fn emit_djb2_hash_loop(cb: &mut CodeBuilder, arch: PeArch) {
    let is64 = arch.is_64bit();
    let hash_loop = cb.label();
    let hash_done = cb.label();

    cb.byte(0x52); // push rdx/edx
    cb.byte(0x51); // push rcx/ecx
    cb.byte(0xB8); // mov eax, 5381
    cb.dword(5381);

    cb.bind(hash_loop);
    cb.raw(&[0x0F, 0xB6, 0x1F]); // movzx ebx, byte [rdi/edi]
    cb.raw(&[0x84, 0xDB]); // test bl, bl
    cb.jcc_short(0x74, hash_done); // jz hash_done
    cb.raw(&[0x6B, 0xC0, 0x21]); // imul eax, eax, 33
    cb.raw(&[0x01, 0xD8]); // add eax, ebx
    if is64 {
        cb.raw(&[0x48, 0xFF, 0xC7]); // inc rdi
    } else {
        cb.byte(0x47); // inc edi
    }
    cb.jmp_short(hash_loop);

    cb.bind(hash_done);
    cb.byte(0x59); // pop rcx/ecx
    cb.byte(0x5A); // pop rdx/edx
}

/// Emit x64-only persistent function resolution (ReadFile, WriteFile, ExitProcess).
fn emit_persistent_resolve(cb: &mut CodeBuilder, p_data_va: u64) {
    let hash_read_file = djb2_hash(b"ReadFile");
    let hash_write_file = djb2_hash(b"WriteFile");
    let hash_exit_process = djb2_hash(b"ExitProcess");

    let skip_persistent = cb.label();
    let persist_exports_done = cb.label();
    let persist_loop = cb.label();
    let persist_next = cb.label();
    let persist_continue = cb.label();

    // Load kernel32 base from [rbp - 0x68]
    cb.raw(&[0x4C, 0x8B, 0x7D, 0x98]); // mov r15, [rbp - 0x68]

    // Parse export directory
    cb.raw(&[0x41, 0x8B, 0x47, 0x3C]); // mov eax, [r15 + 0x3C]
    cb.raw(&[0x49, 0x8D, 0x1C, 0x07]); // lea rbx, [r15 + rax]
    cb.raw(&[0x8B, 0x83, 0x88, 0x00, 0x00, 0x00]); // mov eax, [rbx + 0x88]
    cb.raw(&[0x85, 0xC0]); // test eax, eax
    cb.jcc_near(0x84, skip_persistent);

    cb.raw(&[0x49, 0x8D, 0x34, 0x07]); // lea rsi, [r15 + rax]
    cb.raw(&[0x8B, 0x4E, 0x18]); // mov ecx, [rsi + 0x18]
    cb.raw(&[0x8B, 0x46, 0x20]); // mov eax, [rsi + 0x20]
    cb.raw(&[0x4D, 0x8D, 0x2C, 0x07]); // lea r13, [r15 + rax]
    cb.raw(&[0x8B, 0x46, 0x24]); // mov eax, [rsi + 0x24]
    cb.raw(&[0x4D, 0x8D, 0x34, 0x07]); // lea r14, [r15 + rax]
    cb.raw(&[0x8B, 0x46, 0x1C]); // mov eax, [rsi + 0x1C]
    cb.byte(0x50); // push rax
    cb.raw(&[0x41, 0x57]); // push r15

    // Init slots to 0
    cb.raw(&[0x48, 0xC7, 0x45, 0x90, 0x00, 0x00, 0x00, 0x00]); // [rbp-0x70]=0
    cb.raw(&[0x48, 0xC7, 0x45, 0x88, 0x00, 0x00, 0x00, 0x00]); // [rbp-0x78]=0
    cb.raw(&[0x48, 0xC7, 0x45, 0x80, 0x00, 0x00, 0x00, 0x00]); // [rbp-0x80]=0

    cb.raw(&[0x31, 0xD2]); // xor edx, edx

    cb.bind(persist_loop);
    cb.raw(&[0x39, 0xCA]); // cmp edx, ecx
    cb.jcc_near(0x8D, persist_exports_done); // jge done

    cb.raw(&[0x41, 0x8B, 0x44, 0x95, 0x00]); // mov eax, [r13+rdx*4]
    cb.raw(&[0x49, 0x8D, 0x3C, 0x07]); // lea rdi, [r15+rax]

    // DJB2 hash
    cb.byte(0x52); // push rdx
    cb.byte(0x51); // push rcx
    cb.byte(0xB8);
    cb.dword(5381); // mov eax, 5381

    let p_hash_loop = cb.label();
    let p_hash_done = cb.label();
    cb.bind(p_hash_loop);
    cb.raw(&[0x0F, 0xB6, 0x1F]); // movzx ebx, byte [rdi]
    cb.raw(&[0x84, 0xDB]); // test bl, bl
    cb.jcc_short(0x74, p_hash_done);
    cb.raw(&[0x6B, 0xC0, 0x21]); // imul eax, eax, 33
    cb.raw(&[0x01, 0xD8]); // add eax, ebx
    cb.raw(&[0x48, 0xFF, 0xC7]); // inc rdi
    cb.jmp_short(p_hash_loop);

    cb.bind(p_hash_done);
    cb.byte(0x59); // pop rcx
    cb.byte(0x5A); // pop rdx

    // Match ReadFile
    let check_wf = cb.label();
    cb.byte(0x3D);
    cb.dword(hash_read_file);
    cb.jcc_short(0x75, check_wf);
    // Resolve ReadFile -> [rbp-0x70]
    cb.raw(&[0x41, 0x0F, 0xB7, 0x04, 0x56]); // movzx eax, word [r14+rdx*2]
    cb.raw(&[0x8B, 0x5C, 0x24, 0x08]); // mov ebx, [rsp+8]
    cb.raw(&[0x49, 0x8D, 0x1C, 0x1F]); // lea rbx, [r15+rbx]
    cb.raw(&[0x8B, 0x04, 0x83]); // mov eax, [rbx+rax*4]
    cb.raw(&[0x49, 0x8D, 0x04, 0x07]); // lea rax, [r15+rax]
    cb.raw(&[0x48, 0x89, 0x45, 0x90]); // mov [rbp-0x70], rax
    cb.jmp_short(persist_next);

    // Match WriteFile
    cb.bind(check_wf);
    let check_ep = cb.label();
    cb.byte(0x3D);
    cb.dword(hash_write_file);
    cb.jcc_short(0x75, check_ep);
    cb.raw(&[0x41, 0x0F, 0xB7, 0x04, 0x56]);
    cb.raw(&[0x8B, 0x5C, 0x24, 0x08]);
    cb.raw(&[0x49, 0x8D, 0x1C, 0x1F]);
    cb.raw(&[0x8B, 0x04, 0x83]);
    cb.raw(&[0x49, 0x8D, 0x04, 0x07]);
    cb.raw(&[0x48, 0x89, 0x45, 0x88]); // mov [rbp-0x78], rax
    cb.jmp_short(persist_next);

    // Match ExitProcess
    cb.bind(check_ep);
    cb.byte(0x3D);
    cb.dword(hash_exit_process);
    cb.jcc_short(0x75, persist_next);
    cb.raw(&[0x41, 0x0F, 0xB7, 0x04, 0x56]);
    cb.raw(&[0x8B, 0x5C, 0x24, 0x08]);
    cb.raw(&[0x49, 0x8D, 0x1C, 0x1F]);
    cb.raw(&[0x8B, 0x04, 0x83]);
    cb.raw(&[0x49, 0x8D, 0x04, 0x07]);
    cb.raw(&[0x48, 0x89, 0x45, 0x80]); // mov [rbp-0x80], rax

    // .persist_next:
    cb.bind(persist_next);

    // Early exit: all 3 found?
    cb.raw(&[0x48, 0x83, 0x7D, 0x90, 0x00]); // cmp qword [rbp-0x70], 0
    cb.jcc_short(0x74, persist_continue);
    cb.raw(&[0x48, 0x83, 0x7D, 0x88, 0x00]); // cmp qword [rbp-0x78], 0
    cb.jcc_short(0x74, persist_continue);
    cb.raw(&[0x48, 0x83, 0x7D, 0x80, 0x00]); // cmp qword [rbp-0x80], 0
    cb.jcc_near(0x85, persist_exports_done); // jnz done

    cb.bind(persist_continue);
    cb.raw(&[0xFF, 0xC2]); // inc edx
    cb.jmp_near(persist_loop);

    // .persist_exports_done:
    cb.bind(persist_exports_done);

    cb.raw(&[0x41, 0x5F]); // pop r15
    cb.byte(0x58); // pop rax

    // Store resolved pointers via RIP-relative mov
    // ReadFile -> p_data_va + 152
    cb.raw(&[0x48, 0x8B, 0x45, 0x90]); // mov rax, [rbp-0x70]
    cb.rip_rel(&[0x48, 0x89, 0x05], p_data_va + 152);

    // WriteFile -> p_data_va + 160
    cb.raw(&[0x48, 0x8B, 0x45, 0x88]); // mov rax, [rbp-0x78]
    cb.rip_rel(&[0x48, 0x89, 0x05], p_data_va + 160);

    // ExitProcess -> p_data_va + 168
    cb.raw(&[0x48, 0x8B, 0x45, 0x80]); // mov rax, [rbp-0x80]
    cb.rip_rel(&[0x48, 0x89, 0x05], p_data_va + 168);

    cb.bind(skip_persistent);
}

/// Emit hook library resolution (LoadLibraryA + GetProcAddress) in the main SHM path.
fn emit_hook_library_resolution(
    cb: &mut CodeBuilder,
    arch: PeArch,
    hook_lib: &PeHookLibraryInfo,
    real_epilogue: Label,
    hook_string_fixups: &mut Vec<(usize, usize)>,
) {
    let is64 = arch.is_64bit();

    // Check LoadLibraryA and GetProcAddress were resolved.
    if is64 {
        cb.raw(&[0x48, 0x83, 0x7D, 0xA8, 0x00]); // cmp qword [rbp-0x58], 0
        cb.jcc_near(0x84, real_epilogue);
        cb.raw(&[0x48, 0x83, 0x7D, 0xA0, 0x00]); // cmp qword [rbp-0x60], 0
        cb.jcc_near(0x84, real_epilogue);
    } else {
        cb.raw(&[0x83, 0x7D, 0xE4, 0x00]); // cmp dword [ebp-0x1C], 0
        cb.jcc_near(0x84, real_epilogue);
        cb.raw(&[0x83, 0x7D, 0xE0, 0x00]); // cmp dword [ebp-0x20], 0
        cb.jcc_near(0x84, real_epilogue);
    }

    // LoadLibraryA(library_path)
    if is64 {
        let libpath_fixup = cb.rip_rel_placeholder(&[0x48, 0x8D, 0x0D]); // lea rcx, [rip+disp]
        hook_string_fixups.push((libpath_fixup, 0));
        cb.raw(&[0x48, 0x83, 0xEC, 0x20]); // sub rsp, 32
        cb.raw(&[0xFF, 0x55, 0xA8]); // call [rbp-0x58]
        cb.raw(&[0x48, 0x83, 0xC4, 0x20]); // add rsp, 32
        cb.raw(&[0x48, 0x85, 0xC0]); // test rax, rax
        cb.jcc_near(0x84, real_epilogue);
        cb.raw(&[0x48, 0x89, 0xC3]); // mov rbx, rax
    } else {
        let libpath_fixup = cb.mov_imm32_placeholder(0xB8); // mov eax, imm32
        hook_string_fixups.push((libpath_fixup, 0));
        cb.byte(0x50); // push eax
        cb.raw(&[0xFF, 0x55, 0xE4]); // call [ebp-0x1C]
        cb.raw(&[0x83, 0xC4, 0x04]); // add esp, 4
        cb.raw(&[0x85, 0xC0]); // test eax, eax
        cb.jcc_near(0x84, real_epilogue);
        cb.raw(&[0x89, 0xC3]); // mov ebx, eax
    }

    // For each handler: GetProcAddress(HMODULE, name) -> store at slot VA
    for (i, (_name, slot_va)) in hook_lib.handlers.iter().enumerate() {
        if is64 {
            let sym_fixup = cb.rip_rel_placeholder(&[0x48, 0x8D, 0x15]); // lea rdx, [rip+disp]
            hook_string_fixups.push((sym_fixup, i + 1));
            cb.raw(&[0x48, 0x89, 0xD9]); // mov rcx, rbx
            cb.raw(&[0x48, 0x83, 0xEC, 0x20]); // sub rsp, 32
            cb.raw(&[0xFF, 0x55, 0xA0]); // call [rbp-0x60]
            cb.raw(&[0x48, 0x83, 0xC4, 0x20]); // add rsp, 32
            cb.raw(&[0x48, 0x85, 0xC0]); // test rax, rax
            let skip_store = cb.label();
            cb.jcc_short(0x74, skip_store);
            // lea r13, [rip + slot_va]
            cb.rip_rel(&[0x4C, 0x8D, 0x2D], *slot_va);
            cb.raw(&[0x49, 0x89, 0x45, 0x00]); // mov [r13], rax
            cb.bind(skip_store);
        } else {
            let sym_fixup = cb.mov_imm32_placeholder(0xB8); // mov eax, imm32
            hook_string_fixups.push((sym_fixup, i + 1));
            cb.byte(0x50); // push eax
            cb.byte(0x53); // push ebx
            cb.raw(&[0xFF, 0x55, 0xE0]); // call [ebp-0x20]
            cb.raw(&[0x83, 0xC4, 0x08]); // add esp, 8
            cb.raw(&[0x85, 0xC0]); // test eax, eax
            let skip_store = cb.label();
            cb.jcc_short(0x74, skip_store);
            cb.byte(0xA3); // mov [imm32], eax
            cb.dword(*slot_va as u32);
            cb.bind(skip_store);
        }
    }
}

/// Emit hook-only resolve block: fresh kernel32 walk + LoadLibraryA/GetProcAddress resolution.
/// Used when SHM setup failed but hooks still need resolving.
fn emit_hook_only_resolve(
    cb: &mut CodeBuilder,
    arch: PeArch,
    hook_lib: &PeHookLibraryInfo,
    real_epilogue: Label,
) {
    let is64 = arch.is_64bit();

    // Reuse the kernel32 walk logic
    let hr_found = cb.label();
    let hr_ldr_loop = cb.label();
    let hr_ldr_next = cb.label();

    if is64 {
        cb.raw(&[0x65, 0x48, 0x8B, 0x04, 0x25, 0x60, 0x00, 0x00, 0x00]); // mov rax, gs:[0x60]
        cb.raw(&[0x48, 0x8B, 0x40, 0x18]); // mov rax, [rax+0x18]
        cb.raw(&[0x48, 0x8D, 0x58, 0x10]); // lea rbx, [rax+0x10]
        cb.raw(&[0x48, 0x8B, 0x3B]); // mov rdi, [rbx]
    } else {
        cb.raw(&[0x64, 0xA1, 0x30, 0x00, 0x00, 0x00]); // mov eax, fs:[0x30]
        cb.raw(&[0x8B, 0x40, 0x0C]); // mov eax, [eax+0x0C]
        cb.raw(&[0x8D, 0x58, 0x0C]); // lea ebx, [eax+0x0C]
        cb.raw(&[0x8B, 0x3B]); // mov edi, [ebx]
    }

    cb.bind(hr_ldr_loop);
    if is64 {
        cb.raw(&[0x48, 0x39, 0xDF]); // cmp rdi, rbx
    } else {
        cb.raw(&[0x39, 0xDF]); // cmp edi, ebx
    }
    cb.jcc_near(0x84, real_epilogue); // je real_epilogue (k32 not found)

    if is64 {
        cb.raw(&[0x0F, 0xB7, 0x4F, 0x58]); // movzx ecx, word [rdi+0x58]
    } else {
        cb.raw(&[0x0F, 0xB7, 0x4F, 0x2C]); // movzx ecx, word [edi+0x2C]
    }
    cb.raw(&[0x83, 0xF9, 0x18]); // cmp ecx, 24
    cb.jcc_short(0x75, hr_ldr_next);

    if is64 {
        cb.raw(&[0x48, 0x8B, 0x77, 0x60]); // mov rsi, [rdi+0x60]
        cb.raw(&[0x0F, 0xB7, 0x06]); // movzx eax, word [rsi]
    } else {
        cb.raw(&[0x8B, 0x4F, 0x30]); // mov ecx, [edi+0x30]
        cb.raw(&[0x0F, 0xB7, 0x01]); // movzx eax, word [ecx]
    }
    cb.raw(&[0x0C, 0x20]); // or al, 0x20
    cb.raw(&[0x3C, 0x6B]); // cmp al, 'k'
    cb.jcc_short(0x75, hr_ldr_next);
    if is64 {
        cb.raw(&[0x0F, 0xB7, 0x46, 0x0C]); // movzx eax, word [rsi+12]
    } else {
        cb.raw(&[0x0F, 0xB7, 0x41, 0x0C]); // movzx eax, word [ecx+12]
    }
    cb.raw(&[0x3C, 0x33]); // cmp al, '3'
    cb.jcc_short(0x75, hr_ldr_next);
    if is64 {
        cb.raw(&[0x0F, 0xB7, 0x46, 0x0E]); // movzx eax, word [rsi+14]
    } else {
        cb.raw(&[0x0F, 0xB7, 0x41, 0x0E]); // movzx eax, word [ecx+14]
    }
    cb.raw(&[0x3C, 0x32]); // cmp al, '2'
    cb.jcc_short(0x75, hr_ldr_next);

    // Found kernel32
    if is64 {
        cb.raw(&[0x4C, 0x8B, 0x7F, 0x30]); // mov r15, [rdi+0x30]
    } else {
        cb.raw(&[0x8B, 0x77, 0x18]); // mov esi, [edi+0x18]
    }
    cb.jmp_short(hr_found);

    cb.bind(hr_ldr_next);
    if is64 {
        cb.raw(&[0x48, 0x8B, 0x3F]); // mov rdi, [rdi]
    } else {
        cb.raw(&[0x8B, 0x3F]); // mov edi, [edi]
    }
    cb.jmp_short(hr_ldr_loop);

    cb.bind(hr_found);

    // Parse exports for LoadLibraryA + GetProcAddress
    if is64 {
        cb.raw(&[0x41, 0x8B, 0x47, 0x3C]); // mov eax, [r15+0x3C]
        cb.raw(&[0x49, 0x8D, 0x1C, 0x07]); // lea rbx, [r15+rax]
        cb.raw(&[0x8B, 0x83, 0x88, 0x00, 0x00, 0x00]); // mov eax, [rbx+0x88]
    } else {
        cb.raw(&[0x8B, 0x46, 0x3C]); // mov eax, [esi+0x3C]
        cb.raw(&[0x8D, 0x1C, 0x06]); // lea ebx, [esi+eax]
        cb.raw(&[0x8B, 0x43, 0x78]); // mov eax, [ebx+0x78]
    }
    cb.raw(&[0x85, 0xC0]); // test eax, eax
    cb.jcc_near(0x84, real_epilogue); // jz real_epilogue

    if is64 {
        cb.raw(&[0x49, 0x8D, 0x34, 0x07]); // lea rsi, [r15+rax]
        cb.raw(&[0x8B, 0x4E, 0x18]); // mov ecx, [rsi+0x18]
        cb.raw(&[0x8B, 0x46, 0x20]); // mov eax, [rsi+0x20]
        cb.raw(&[0x4D, 0x8D, 0x2C, 0x07]); // lea r13, [r15+rax]
        cb.raw(&[0x8B, 0x46, 0x24]); // mov eax, [rsi+0x24]
        cb.raw(&[0x4D, 0x8D, 0x34, 0x07]); // lea r14, [r15+rax]
        cb.raw(&[0x8B, 0x46, 0x1C]); // mov eax, [rsi+0x1C]
        // lea rsi, [r15+rax] (reuse rsi for AddressOfFunctions base)
        cb.raw(&[0x49, 0x8D, 0x34, 0x07]);
        // Clear slots
        cb.raw(&[0x48, 0xC7, 0x45, 0xA8, 0x00, 0x00, 0x00, 0x00]); // [rbp-0x58]=0
        cb.raw(&[0x48, 0xC7, 0x45, 0xA0, 0x00, 0x00, 0x00, 0x00]); // [rbp-0x60]=0
    } else {
        cb.raw(&[0x8D, 0x1C, 0x06]); // lea ebx, [esi+eax]
        cb.raw(&[0x8B, 0x4B, 0x18]); // mov ecx, [ebx+0x18]
        cb.raw(&[0x8B, 0x43, 0x20]); // mov eax, [ebx+0x20]
        cb.raw(&[0x8D, 0x04, 0x06]); // lea eax, [esi+eax]
        cb.raw(&[0x89, 0x45, 0xF8]); // [ebp-0x08]
        cb.raw(&[0x8B, 0x43, 0x24]); // mov eax, [ebx+0x24]
        cb.raw(&[0x8D, 0x04, 0x06]); // lea eax, [esi+eax]
        cb.raw(&[0x89, 0x45, 0xF4]); // [ebp-0x0C]
        cb.raw(&[0x8B, 0x43, 0x1C]); // mov eax, [ebx+0x1C]
        cb.raw(&[0x8D, 0x04, 0x06]); // lea eax, [esi+eax]
        cb.raw(&[0x89, 0x45, 0xF0]); // [ebp-0x10]
        // Clear slots
        cb.raw(&[0xC7, 0x45, 0xE4, 0x00, 0x00, 0x00, 0x00]); // [ebp-0x1C]=0
        cb.raw(&[0xC7, 0x45, 0xE0, 0x00, 0x00, 0x00, 0x00]); // [ebp-0x20]=0
    }

    let hash_lla = djb2_hash(b"LoadLibraryA");
    let hash_gpa = djb2_hash(b"GetProcAddress");

    cb.raw(&[0x31, 0xD2]); // xor edx, edx

    let hr_exp_loop = cb.label();
    let hr_exp_done = cb.label();
    let hr_exp_next = cb.label();
    let hr_exp_inc = cb.label();

    cb.bind(hr_exp_loop);
    cb.raw(&[0x39, 0xCA]); // cmp edx, ecx
    cb.jcc_near(0x8D, hr_exp_done);

    // Load name, hash it
    if is64 {
        cb.raw(&[0x41, 0x8B, 0x44, 0x95, 0x00]); // mov eax, [r13+rdx*4]
        cb.raw(&[0x49, 0x8D, 0x3C, 0x07]); // lea rdi, [r15+rax]
    } else {
        cb.raw(&[0x8B, 0x45, 0xF8]); // mov eax, [ebp-0x08]
        cb.raw(&[0x8B, 0x04, 0x90]); // mov eax, [eax+edx*4]
        cb.raw(&[0x8D, 0x3C, 0x06]); // lea edi, [esi+eax]
    }

    // DJB2 hash
    cb.byte(0x52);
    cb.byte(0x51); // push edx, ecx
    if is64 {
        cb.byte(0x56);
    } // push rsi (save AddressOfFunctions base for x64)
    cb.byte(0xB8);
    cb.dword(5381);
    let hr_hl = cb.label();
    let hr_hd = cb.label();
    cb.bind(hr_hl);
    cb.raw(&[0x0F, 0xB6, 0x1F]); // movzx ebx, byte [rdi]
    cb.raw(&[0x84, 0xDB]); // test bl, bl
    cb.jcc_short(0x74, hr_hd);
    cb.raw(&[0x6B, 0xC0, 0x21]); // imul eax, eax, 33
    cb.raw(&[0x01, 0xD8]); // add eax, ebx
    if is64 {
        cb.raw(&[0x48, 0xFF, 0xC7]); // inc rdi
    } else {
        cb.byte(0x47); // inc edi
    }
    cb.jmp_short(hr_hl);
    cb.bind(hr_hd);
    if is64 {
        cb.byte(0x5E);
    } // pop rsi
    cb.byte(0x59);
    cb.byte(0x5A); // pop ecx, edx

    // Check LoadLibraryA
    let hr_check_gpa = cb.label();
    cb.byte(0x3D);
    cb.dword(hash_lla);
    cb.jcc_short(0x75, hr_check_gpa);

    // Resolve LoadLibraryA
    if is64 {
        cb.raw(&[0x41, 0x0F, 0xB7, 0x04, 0x56]); // movzx eax, word [r14+rdx*2]
        cb.raw(&[0x8B, 0x04, 0x86]); // mov eax, [rsi+rax*4]
        cb.raw(&[0x49, 0x8D, 0x3C, 0x07]); // lea rdi, [r15+rax]
        cb.raw(&[0x48, 0x89, 0x7D, 0xA8]); // mov [rbp-0x58], rdi
    } else {
        cb.raw(&[0x8B, 0x5D, 0xF4]); // mov ebx, [ebp-0x0C]
        cb.raw(&[0x0F, 0xB7, 0x04, 0x53]); // movzx eax, word [ebx+edx*2]
        cb.raw(&[0x8B, 0x5D, 0xF0]); // mov ebx, [ebp-0x10]
        cb.raw(&[0x8B, 0x04, 0x83]); // mov eax, [ebx+eax*4]
        cb.raw(&[0x8D, 0x04, 0x06]); // lea eax, [esi+eax]
        cb.raw(&[0x89, 0x45, 0xE4]); // mov [ebp-0x1C], eax
    }
    cb.jmp_short(hr_exp_next);

    // Check GetProcAddress
    cb.bind(hr_check_gpa);
    cb.byte(0x3D);
    cb.dword(hash_gpa);
    cb.jcc_short(0x75, hr_exp_next);

    // Resolve GetProcAddress
    if is64 {
        cb.raw(&[0x41, 0x0F, 0xB7, 0x04, 0x56]);
        cb.raw(&[0x8B, 0x04, 0x86]);
        cb.raw(&[0x49, 0x8D, 0x3C, 0x07]);
        cb.raw(&[0x48, 0x89, 0x7D, 0xA0]); // mov [rbp-0x60], rdi
    } else {
        cb.raw(&[0x8B, 0x5D, 0xF4]);
        cb.raw(&[0x0F, 0xB7, 0x04, 0x53]);
        cb.raw(&[0x8B, 0x5D, 0xF0]);
        cb.raw(&[0x8B, 0x04, 0x83]);
        cb.raw(&[0x8D, 0x04, 0x06]);
        cb.raw(&[0x89, 0x45, 0xE0]); // mov [ebp-0x20], eax
    }

    // .hr_exp_next:
    cb.bind(hr_exp_next);

    // Check if both found
    if is64 {
        cb.raw(&[0x48, 0x83, 0x7D, 0xA8, 0x00]); // cmp [rbp-0x58], 0
        cb.jcc_short(0x74, hr_exp_inc);
        cb.raw(&[0x48, 0x83, 0x7D, 0xA0, 0x00]); // cmp [rbp-0x60], 0
    } else {
        cb.raw(&[0x83, 0x7D, 0xE4, 0x00]); // cmp [ebp-0x1C], 0
        cb.jcc_short(0x74, hr_exp_inc);
        cb.raw(&[0x83, 0x7D, 0xE0, 0x00]); // cmp [ebp-0x20], 0
    }
    cb.jcc_near(0x85, hr_exp_done); // jnz both found

    cb.bind(hr_exp_inc);
    cb.raw(&[0xFF, 0xC2]); // inc edx
    cb.jmp_near(hr_exp_loop);

    cb.bind(hr_exp_done);

    // Check both were found, call LoadLibraryA + GetProcAddress
    if is64 {
        cb.raw(&[0x48, 0x83, 0x7D, 0xA8, 0x00]);
        cb.jcc_near(0x84, real_epilogue);
        cb.raw(&[0x48, 0x83, 0x7D, 0xA0, 0x00]);
        cb.jcc_near(0x84, real_epilogue);
    } else {
        cb.raw(&[0x83, 0x7D, 0xE4, 0x00]);
        cb.jcc_near(0x84, real_epilogue);
        cb.raw(&[0x83, 0x7D, 0xE0, 0x00]);
        cb.jcc_near(0x84, real_epilogue);
    }

    // Call LoadLibraryA
    if is64 {
        let hr_lea_lib = cb.rip_rel_placeholder(&[0x48, 0x8D, 0x0D]); // lea rcx, [rip+disp]
        cb.raw(&[0x48, 0x83, 0xEC, 0x20]); // sub rsp, 32
        cb.raw(&[0xFF, 0x55, 0xA8]); // call [rbp-0x58]
        cb.raw(&[0x48, 0x83, 0xC4, 0x20]); // add rsp, 32
        cb.raw(&[0x48, 0x85, 0xC0]); // test rax, rax
        cb.jcc_near(0x84, real_epilogue);
        cb.raw(&[0x48, 0x89, 0xC3]); // mov rbx, rax

        // For each handler
        let mut hr_sym_fixups: Vec<(usize, usize)> = Vec::new();
        for (i, (_name, slot_va)) in hook_lib.handlers.iter().enumerate() {
            let sym_fixup = cb.rip_rel_placeholder(&[0x48, 0x8D, 0x15]); // lea rdx, [rip+disp]
            hr_sym_fixups.push((sym_fixup, i));
            cb.raw(&[0x48, 0x89, 0xD9]); // mov rcx, rbx
            cb.raw(&[0x48, 0x83, 0xEC, 0x20]); // sub rsp, 32
            cb.raw(&[0xFF, 0x55, 0xA0]); // call [rbp-0x60]
            cb.raw(&[0x48, 0x83, 0xC4, 0x20]); // add rsp, 32
            cb.raw(&[0x48, 0x85, 0xC0]); // test rax, rax
            let skip = cb.label();
            cb.jcc_short(0x74, skip);
            cb.rip_rel(&[0x4C, 0x8D, 0x2D], *slot_va); // lea r13, [rip+slot_va]
            cb.raw(&[0x49, 0x89, 0x45, 0x00]); // mov [r13], rax
            cb.bind(skip);
        }

        // JMP over embedded strings
        let hr_after_strings = cb.label();
        cb.jmp_near(hr_after_strings);

        // Embed strings
        let hr_lib_str_va = cb.va();
        cb.patch_rip_rel(hr_lea_lib, hr_lib_str_va);
        cb.raw(hook_lib.library_path.as_bytes());
        cb.byte(0x00);

        for (sym_fixup, handler_idx) in &hr_sym_fixups {
            let str_va = cb.va();
            cb.patch_rip_rel(*sym_fixup, str_va);
            let (name, _) = &hook_lib.handlers[*handler_idx];
            cb.raw(name.as_bytes());
            cb.byte(0x00);
        }

        cb.bind(hr_after_strings);
    } else {
        let hr_mov_lib = cb.mov_imm32_placeholder(0xB8); // mov eax, imm32
        cb.byte(0x50); // push eax
        cb.raw(&[0xFF, 0x55, 0xE4]); // call [ebp-0x1C]
        cb.raw(&[0x83, 0xC4, 0x04]); // add esp, 4
        cb.raw(&[0x85, 0xC0]); // test eax, eax
        cb.jcc_near(0x84, real_epilogue);
        cb.raw(&[0x89, 0xC3]); // mov ebx, eax

        let mut hr_sym_fixups: Vec<(usize, usize)> = Vec::new();
        for (i, (_name, slot_va)) in hook_lib.handlers.iter().enumerate() {
            let sym_fixup = cb.mov_imm32_placeholder(0xB8); // mov eax, imm32
            hr_sym_fixups.push((sym_fixup, i));
            cb.byte(0x50); // push eax
            cb.byte(0x53); // push ebx
            cb.raw(&[0xFF, 0x55, 0xE0]); // call [ebp-0x20]
            cb.raw(&[0x83, 0xC4, 0x08]); // add esp, 8
            cb.raw(&[0x85, 0xC0]); // test eax, eax
            let skip = cb.label();
            cb.jcc_short(0x74, skip);
            cb.byte(0xA3); // mov [imm32], eax
            cb.dword(*slot_va as u32);
            cb.bind(skip);
        }

        // JMP over strings
        let hr_after_strings = cb.label();
        cb.jmp_near(hr_after_strings);

        // Embed strings
        let hr_lib_str_va = cb.va();
        cb.patch_imm32(hr_mov_lib, hr_lib_str_va as u32);
        cb.raw(hook_lib.library_path.as_bytes());
        cb.byte(0x00);

        for (sym_fixup, handler_idx) in &hr_sym_fixups {
            let str_va = cb.va();
            cb.patch_imm32(*sym_fixup, str_va as u32);
            let (name, _) = &hook_lib.handlers[*handler_idx];
            cb.raw(name.as_bytes());
            cb.byte(0x00);
        }

        cb.bind(hr_after_strings);
    }

    // Fall through to real_epilogue
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
fn generate_pe_persistent_wrapper(
    params: &crate::trampoline::PersistentWrapperParams,
) -> Result<crate::trampoline::PersistentWrapper> {
    let wrapper_va = params.wrapper_va;
    let persistent_data_va = params.persistent_data_va;
    let data_va = params.data_va;
    let persistent_addr = params.persistent_addr;
    let displaced_bytes = params.displaced_bytes;
    let displaced_len = params.displaced_len;
    let persistent_count = params.persistent_count;
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
fn generate_pe32_persistent_wrapper(
    params: &crate::trampoline::PersistentWrapperParams,
) -> Result<crate::trampoline::PersistentWrapper> {
    let wrapper_va = params.wrapper_va;
    let persistent_data_va = params.persistent_data_va;
    let data_va = params.data_va;
    let persistent_addr = params.persistent_addr;
    let displaced_bytes = params.displaced_bytes;
    let displaced_len = params.displaced_len;
    let persistent_count = params.persistent_count;
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

/// Backward-compatible wrapper: generate PE64 init code.
#[cfg(test)]
fn generate_pe_init_code(
    init_va: u64,
    data_va: u64,
    original_entry: u64,
    hook_library: Option<&PeHookLibraryInfo>,
    persistent_data_va: Option<u64>,
) -> Result<crate::trampoline::InitCode> {
    generate_pe_init_code_impl(
        init_va,
        data_va,
        original_entry,
        hook_library,
        persistent_data_va,
        PeArch::X64,
    )
}

/// Backward-compatible wrapper: generate PE32 init code.
#[cfg(test)]
fn generate_pe32_init_code(
    init_va: u64,
    data_va: u64,
    original_entry: u64,
    hook_library: Option<&PeHookLibraryInfo>,
    _persistent_data_va: Option<u64>,
) -> Result<crate::trampoline::InitCode> {
    generate_pe_init_code_impl(
        init_va,
        data_va,
        original_entry,
        hook_library,
        _persistent_data_va,
        PeArch::X86,
    )
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
        let result = generate_pe_persistent_wrapper(&crate::trampoline::PersistentWrapperParams {
            wrapper_va,
            persistent_data_va,
            data_va,
            persistent_addr,
            displaced_bytes: &displaced,
            displaced_len: 5,
            persistent_count: 1000,
            include_forkserver: false,
        });
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
        let result =
            generate_pe32_persistent_wrapper(&crate::trampoline::PersistentWrapperParams {
                wrapper_va,
                persistent_data_va,
                data_va,
                persistent_addr,
                displaced_bytes: &displaced,
                displaced_len: 5,
                persistent_count: 1000,
                include_forkserver: false,
            });
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
        let result =
            generate_pe32_persistent_wrapper(&crate::trampoline::PersistentWrapperParams {
                wrapper_va,
                persistent_data_va,
                data_va,
                persistent_addr,
                displaced_bytes: &displaced,
                displaced_len: 5,
                persistent_count,
                include_forkserver: false,
            });
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

    // ================================================================
    // Additional coverage tests for the unified init code generator
    // ================================================================

    #[test]
    fn test_pe64_init_code_ends_with_jmp_to_oep() {
        let oep = 0x140001000u64;
        let init = generate_pe_init_code(0x140010000, 0x140010000 - 16, oep, None, None)
            .expect("PE64 init code should succeed for JMP OEP test");
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
            "JMP to original entry point not found in PE64 init code"
        );
    }

    #[test]
    fn test_pe64_init_code_contains_afl_shm_needle() {
        let init = generate_pe_init_code(0x140010000, 0x140010000 - 16, 0x140001000, None, None)
            .expect("PE64 init code should succeed for SHM needle test");
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
            "full __AFL_SHM_ID= wide needle not found in PE64 init code"
        );
    }

    #[test]
    fn test_pe64_init_code_contains_peb_access() {
        let init = generate_pe_init_code(0x140010000, 0x140010000 - 16, 0x140001000, None, None)
            .expect("PE64 init code should succeed for PEB access test");
        // gs:[0x60] access: 65 48 8B 04 25 60 00 00 00
        let gs_peb = [0x65u8, 0x48, 0x8B, 0x04, 0x25, 0x60, 0x00, 0x00, 0x00];
        let found = init.code.windows(gs_peb.len()).any(|w| w == gs_peb);
        assert!(found, "gs:[0x60] PEB access not found in PE64 init code");
    }

    #[test]
    fn test_pe32_init_code_contains_peb_access() {
        let init = generate_pe32_init_code(0x00410000, 0x0040FFF0, 0x00401000, None, None)
            .expect("PE32 init code should succeed for PEB access test");
        // fs:[0x30] access: 64 A1 30 00 00 00
        let fs_peb = [0x64u8, 0xA1, 0x30, 0x00, 0x00, 0x00];
        let found = init.code.windows(fs_peb.len()).any(|w| w == fs_peb);
        assert!(found, "fs:[0x30] PEB access not found in PE32 init code");
    }

    #[test]
    fn test_pe64_init_code_with_hooks_and_persistent() {
        let hook_lib = PeHookLibraryInfo {
            library_path: "hook.dll".to_string(),
            handlers: vec![
                ("on_entry".to_string(), 0x140020000),
                ("on_exit".to_string(), 0x140020008),
            ],
        };
        let init = generate_pe_init_code(
            0x140010000,
            0x140010000 - 16,
            0x140001000,
            Some(&hook_lib),
            Some(0x140030000),
        )
        .expect("PE64 init code with hooks + persistent should succeed");

        // Should be larger than hooks-only or persistent-only
        let hooks_only = generate_pe_init_code(
            0x140010000,
            0x140010000 - 16,
            0x140001000,
            Some(&hook_lib),
            None,
        )
        .unwrap();
        let persistent_only = generate_pe_init_code(
            0x140010000,
            0x140010000 - 16,
            0x140001000,
            None,
            Some(0x140030000),
        )
        .unwrap();

        assert!(
            init.code.len() > hooks_only.code.len(),
            "hooks+persistent ({}) should be larger than hooks-only ({})",
            init.code.len(),
            hooks_only.code.len()
        );
        assert!(
            init.code.len() > persistent_only.code.len(),
            "hooks+persistent ({}) should be larger than persistent-only ({})",
            init.code.len(),
            persistent_only.code.len()
        );

        // Verify hook strings are embedded
        assert!(init.code.windows(8).any(|w| w == b"hook.dll"));
        assert!(init.code.windows(8).any(|w| w == b"on_entry"));
        assert!(init.code.windows(7).any(|w| w == b"on_exit"));
    }

    #[test]
    fn test_pe64_hook_only_resolve_path_exists() {
        // When hooks are present, a hook-only resolve fallback block should be emitted.
        // This block re-walks kernel32 for LoadLibraryA + GetProcAddress even if SHM
        // setup fails. Verify its presence by checking the code is significantly larger
        // with hooks (the hook-only block adds ~300 bytes of kernel32 re-walk code).
        let hook_lib = PeHookLibraryInfo {
            library_path: "test.dll".to_string(),
            handlers: vec![("handler".to_string(), 0x140020000)],
        };
        let with_hooks = generate_pe_init_code(
            0x140010000,
            0x140010000 - 16,
            0x140001000,
            Some(&hook_lib),
            None,
        )
        .unwrap();
        let without_hooks =
            generate_pe_init_code(0x140010000, 0x140010000 - 16, 0x140001000, None, None).unwrap();

        // The hook-only resolve block adds kernel32 walk + export resolution + LoadLibraryA +
        // GetProcAddress calls + embedded strings. This should add at least 200 bytes.
        let delta = with_hooks.code.len() - without_hooks.code.len();
        assert!(
            delta >= 200,
            "hook-only resolve block too small: delta={delta} bytes (expected >= 200)"
        );

        // Verify the hook library string appears at least twice in the code:
        // once in the main path, once in the hook-only resolve block
        let dll_count = with_hooks
            .code
            .windows(8)
            .filter(|w| *w == b"test.dll")
            .count();
        assert!(
            dll_count >= 2,
            "expected hook library path to appear >= 2 times (main + hook-only), found {dll_count}"
        );
    }

    #[test]
    fn test_pe32_hook_only_resolve_path_exists() {
        let hook_lib = PeHookLibraryInfo {
            library_path: "test.dll".to_string(),
            handlers: vec![("handler".to_string(), 0x00420000)],
        };
        let with_hooks =
            generate_pe32_init_code(0x00410000, 0x0040FFF0, 0x00401000, Some(&hook_lib), None)
                .unwrap();
        let without_hooks =
            generate_pe32_init_code(0x00410000, 0x0040FFF0, 0x00401000, None, None).unwrap();

        let delta = with_hooks.code.len() - without_hooks.code.len();
        assert!(
            delta >= 200,
            "PE32 hook-only resolve block too small: delta={delta} bytes (expected >= 200)"
        );

        let dll_count = with_hooks
            .code
            .windows(8)
            .filter(|w| *w == b"test.dll")
            .count();
        assert!(
            dll_count >= 2,
            "PE32: expected hook library path >= 2 times, found {dll_count}"
        );
    }

    #[test]
    fn test_pe64_prologue_saves_callee_saved_regs() {
        let init =
            generate_pe_init_code(0x140010000, 0x140010000 - 16, 0x140001000, None, None).unwrap();
        let code = &init.code;

        // push rbp (0x55), push rbx (0x53), push rsi (0x56), push rdi (0x57)
        // push r12 (41 54), push r13 (41 55), push r14 (41 56), push r15 (41 57)
        assert_eq!(code[0], 0x55, "first byte should be push rbp");
        // mov rbp, rsp (48 89 E5)
        assert_eq!(&code[1..4], &[0x48, 0x89, 0xE5], "expected mov rbp, rsp");

        // Epilogue: verify pop sequence exists (in reverse order)
        // pop r15 (41 5F), pop r14 (41 5E), pop r13 (41 5D), pop r12 (41 5C),
        // pop rdi (5F), pop rsi (5E), pop rbx (5B), pop rbp (5D)
        let epilogue = [
            0x41u8, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5F, 0x5E, 0x5B, 0x5D,
        ];
        let found = code.windows(epilogue.len()).any(|w| w == epilogue);
        assert!(
            found,
            "callee-saved register restore sequence not found in PE64 code"
        );
    }

    #[test]
    fn test_pe32_prologue_saves_callee_saved_regs() {
        let init = generate_pe32_init_code(0x00410000, 0x0040FFF0, 0x00401000, None, None).unwrap();
        let code = &init.code;

        // push ebp (0x55), mov ebp,esp (89 E5), push ebx (53), push esi (56), push edi (57)
        assert_eq!(code[0], 0x55, "first byte should be push ebp");
        assert_eq!(&code[1..3], &[0x89, 0xE5], "expected mov ebp, esp");
        assert_eq!(code[3], 0x53, "expected push ebx");
        assert_eq!(code[4], 0x56, "expected push esi");
        assert_eq!(code[5], 0x57, "expected push edi");

        // Epilogue: pop edi (5F), pop esi (5E), pop ebx (5B), pop ebp (5D)
        let epilogue = [0x5Fu8, 0x5E, 0x5B, 0x5D];
        let found = code.windows(epilogue.len()).any(|w| w == epilogue);
        assert!(
            found,
            "callee-saved register restore sequence not found in PE32 code"
        );
    }

    #[test]
    fn test_pe_init_code_multiple_handlers() {
        // Test with many handlers to verify scaling
        let handlers: Vec<(String, u64)> = (0..8)
            .map(|i| (format!("handler_{i}"), 0x140020000 + i as u64 * 8))
            .collect();
        let hook_lib = PeHookLibraryInfo {
            library_path: "multi.dll".to_string(),
            handlers,
        };

        let init64 = generate_pe_init_code(
            0x140010000,
            0x140010000 - 16,
            0x140001000,
            Some(&hook_lib),
            None,
        )
        .expect("PE64 with 8 handlers should succeed");

        let init32 =
            generate_pe32_init_code(0x00410000, 0x0040FFF0, 0x00401000, Some(&hook_lib), None)
                .expect("PE32 with 8 handlers should succeed");

        // Verify all handler names are embedded
        for i in 0..8 {
            let name = format!("handler_{i}");
            assert!(
                init64
                    .code
                    .windows(name.len())
                    .any(|w| w == name.as_bytes()),
                "PE64: handler name '{name}' not found in code"
            );
            assert!(
                init32
                    .code
                    .windows(name.len())
                    .any(|w| w == name.as_bytes()),
                "PE32: handler name '{name}' not found in code"
            );
        }
    }

    #[test]
    fn test_djb2_hash_known_values() {
        // Verify DJB2 produces expected values for known inputs
        assert_eq!(djb2_hash(b""), 5381);
        assert_eq!(
            djb2_hash(b"a"),
            5381u32.wrapping_mul(33).wrapping_add(b'a' as u32)
        );
        // Verify determinism
        assert_eq!(
            djb2_hash(b"OpenFileMappingA"),
            djb2_hash(b"OpenFileMappingA")
        );
        // Verify the 6 critical function hashes are all distinct
        let hashes: Vec<u32> = [
            "OpenFileMappingA",
            "MapViewOfFile",
            "LoadLibraryA",
            "GetProcAddress",
            "ReadFile",
            "WriteFile",
        ]
        .iter()
        .map(|s| djb2_hash(s.as_bytes()))
        .collect();
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(
                    hashes[i], hashes[j],
                    "hash collision between functions {i} and {j}"
                );
            }
        }
    }
}
