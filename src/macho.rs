//! Mach-O binary parsing and metadata extraction for instrumentation.
//!
//! Extracts all structural information needed to instrument a Mach-O binary:
//! text section, entry point, segments, load command offsets, stub ranges,
//! code signature location, and dylib init function info.

use anyhow::{Context, Result, bail};

/// __TEXT,__text section location.
#[derive(Debug, Clone)]
pub struct TextSection {
    pub va: u64,
    pub offset: u64,
    pub size: u64,
}

/// A Mach-O LC_SEGMENT_64 segment.
#[derive(Debug, Clone)]
pub struct MachOSegment {
    pub name: String,
    pub vmaddr: u64,
    pub vmsize: u64,
    pub fileoff: u64,
    pub filesize: u64,
}

/// A stub range to exclude from instrumentation (__stubs, __stub_helper, __auth_stubs).
#[derive(Debug, Clone)]
pub struct StubRange {
    pub va: u64,
    pub size: u64,
}

/// Information about __DATA,__mod_init_func section.
#[derive(Debug, Clone)]
pub struct ModInitFuncInfo {
    pub section_offset: u64,
    pub section_size: u64,
    pub section_va: u64,
}

/// Information about __TEXT,__init_offsets section (S_INIT_FUNC_OFFSETS = 0x16).
/// Modern macOS compilers (Xcode 15+) emit 32-bit offsets from __TEXT base
/// instead of 64-bit absolute pointers in __mod_init_func.
#[derive(Debug, Clone)]
pub struct InitOffsetsInfo {
    pub section_offset: u64,
    pub section_size: u64,
    pub section_va: u64,
    /// __TEXT segment vmaddr, used to convert offsets to absolute VAs.
    pub text_segment_vmaddr: u64,
}

/// Parsed Mach-O binary context with all metadata needed for instrumentation.
#[derive(Debug)]
pub struct MachOContext {
    // Core
    pub text: TextSection,
    pub entry_point: u64,
    pub func_symbols: Vec<u64>,
    pub highest_va_end: u64,
    pub is_dylib: bool,
    pub is_64bit: bool,

    // Segment/section layout
    pub segments: Vec<MachOSegment>,
    pub stubs_ranges: Vec<StubRange>,

    // Load command patching offsets
    pub header_size: usize,
    pub lc_end_offset: u64,
    pub first_section_offset: u64,
    pub available_lc_space: u64,

    // Entry point patching
    pub lc_main_entryoff_offset: Option<u64>,
    pub lc_unixthread_pc_offset: Option<u64>,
    pub text_segment_vmaddr: u64,

    // Dylib init
    pub mod_init_func_section: Option<ModInitFuncInfo>,
    pub init_offsets_section: Option<InitOffsetsInfo>,
    pub mod_init_func_pointers: Vec<u64>,

    // Code signature
    pub code_signature_lc_offset: Option<u64>,
    pub code_signature_lc_size: u32,

    // __LINKEDIT
    pub linkedit_segment_offset: Option<u64>,
    pub linkedit_vmaddr: u64,
    pub linkedit_fileoff: u64,
    pub linkedit_filesize: u64,

    // Mach header patching
    pub ncmds: u32,
    pub sizeofcmds: u32,
    pub ncmds_offset: u64,
    pub sizeofcmds_offset: u64,

    // Chained fixups (LC_DYLD_CHAINED_FIXUPS) — for adding _shmat import
    pub chained_fixups_lc_offset: Option<u64>,
    pub chained_fixups_dataoff: u32,
    pub chained_fixups_datasize: u32,

    // CPU type from mach_header (for page size determination)
    pub cputype: u32,

    // Library ordinal of libSystem.B.dylib (1-based index into LC_LOAD_DYLIB commands).
    // Used when adding _shmat import via chained fixups so dyld resolves from the correct dylib.
    // Defaults to 1 if no LC_LOAD_DYLIB commands are present (e.g. static or minimal binaries).
    pub libsystem_ordinal: u32,
}

// Mach-O constants
const MH_MAGIC_64: u32 = 0xFEEDFACF;
const MH_CIGAM_64: u32 = 0xCFFAEDFE;
const MH_DYLIB: u32 = 6;
const MH_BUNDLE: u32 = 8;

const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x02;
const LC_MAIN: u32 = 0x80000028;
const LC_UNIXTHREAD: u32 = 0x05;
const LC_CODE_SIGNATURE: u32 = 0x1D;
const LC_DYLD_CHAINED_FIXUPS: u32 = 0x80000034;
const LC_LOAD_DYLIB: u32 = 0x0C;
const LC_LOAD_WEAK_DYLIB: u32 = 0x80000018;
const LC_REEXPORT_DYLIB: u32 = 0x8000001F;
const LC_LAZY_LOAD_DYLIB: u32 = 0x20;

const CPU_TYPE_X86_64: u32 = 0x0100_0007;
const CPU_TYPE_ARM64: u32 = 0x0100_000C;

impl MachOContext {
    /// Parse a Mach-O binary and extract all metadata needed for instrumentation.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 32 {
            bail!("file too short for Mach-O header ({} bytes)", data.len());
        }

        // Determine endianness and bitness from magic
        let magic = u32::from_le_bytes(data[0..4].try_into().expect("slice is 4 bytes"));
        let (is_64bit, is_big_endian) = match magic {
            MH_MAGIC_64 => (true, false),
            MH_CIGAM_64 => (true, true),
            0xFEEDFACE => (false, false),
            0xCEFAEDFE => (false, true),
            _ => bail!("not a Mach-O binary (magic: 0x{:08x})", magic),
        };

        if !is_64bit {
            bail!("32-bit Mach-O not supported (only 64-bit)");
        }
        if is_big_endian {
            bail!("big-endian Mach-O not supported");
        }

        // Parse mach_header_64 (32 bytes)
        let cputype = u32::from_le_bytes(data[4..8].try_into().expect("slice is 4 bytes"));
        let _cpusubtype = u32::from_le_bytes(data[8..12].try_into().expect("slice is 4 bytes"));
        let filetype = u32::from_le_bytes(data[12..16].try_into().expect("slice is 4 bytes"));
        let ncmds = u32::from_le_bytes(data[16..20].try_into().expect("slice is 4 bytes"));
        let sizeofcmds = u32::from_le_bytes(data[20..24].try_into().expect("slice is 4 bytes"));
        let _flags = u32::from_le_bytes(data[24..28].try_into().expect("slice is 4 bytes"));
        // reserved at 28..32

        let header_size: usize = 32; // sizeof(mach_header_64)
        let ncmds_offset = 16u64;
        let sizeofcmds_offset = 20u64;

        let is_dylib = filetype == MH_DYLIB || filetype == MH_BUNDLE;

        // Iterate load commands
        let mut offset = header_size;
        let _lc_area_end = header_size + sizeofcmds as usize;

        let mut text_section: Option<TextSection> = None;
        let mut text_segment_vmaddr: u64 = 0;
        let mut segments: Vec<MachOSegment> = Vec::new();
        let mut stubs_ranges: Vec<StubRange> = Vec::new();
        let mut highest_va_end: u64 = 0;
        let mut func_symbols: Vec<u64> = Vec::new();
        let mut entry_point: u64 = 0;
        let mut first_section_offset: u64 = u64::MAX;

        let mut lc_main_entryoff_offset: Option<u64> = None;
        let mut lc_unixthread_pc_offset: Option<u64> = None;
        let mut code_signature_lc_offset: Option<u64> = None;
        let mut code_signature_lc_size: u32 = 0;
        let mut linkedit_segment_offset: Option<u64> = None;
        let mut linkedit_vmaddr: u64 = 0;
        let mut linkedit_fileoff: u64 = 0;
        let mut linkedit_filesize: u64 = 0;
        let mut mod_init_func_section: Option<ModInitFuncInfo> = None;
        let mut init_offsets_section: Option<InitOffsetsInfo> = None;
        let mut mod_init_func_pointers: Vec<u64> = Vec::new();

        let mut symtab_offset: u32 = 0;
        let mut symtab_nsyms: u32 = 0;
        let mut strtab_offset: u32 = 0;
        let mut _strtab_size: u32 = 0;

        let mut chained_fixups_lc_offset: Option<u64> = None;
        let mut chained_fixups_dataoff: u32 = 0;
        let mut chained_fixups_datasize: u32 = 0;

        // Track dylib ordinals (1-based): ordinal 1 = first LC_LOAD_DYLIB
        let mut dylib_ordinal: u32 = 0;
        let mut libsystem_ordinal: u32 = 1; // default: assume libSystem is first

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

            if cmdsize < 8 || offset + cmdsize as usize > data.len() {
                break;
            }

            match cmd {
                LC_SEGMENT_64 => {
                    if cmdsize < 72 {
                        offset += cmdsize as usize;
                        continue;
                    }
                    // Parse LC_SEGMENT_64
                    let segname_bytes = &data[offset + 8..offset + 24];
                    let segname = std::str::from_utf8(
                        &segname_bytes[..segname_bytes.iter().position(|&b| b == 0).unwrap_or(16)],
                    )
                    .unwrap_or("")
                    .to_string();

                    let vmaddr = u64::from_le_bytes(
                        data[offset + 24..offset + 32]
                            .try_into()
                            .expect("slice is 8 bytes"),
                    );
                    let vmsize = u64::from_le_bytes(
                        data[offset + 32..offset + 40]
                            .try_into()
                            .expect("slice is 8 bytes"),
                    );
                    let fileoff = u64::from_le_bytes(
                        data[offset + 40..offset + 48]
                            .try_into()
                            .expect("slice is 8 bytes"),
                    );
                    let filesize = u64::from_le_bytes(
                        data[offset + 48..offset + 56]
                            .try_into()
                            .expect("slice is 8 bytes"),
                    );
                    let nsects = u32::from_le_bytes(
                        data[offset + 64..offset + 68]
                            .try_into()
                            .expect("slice is 4 bytes"),
                    );

                    let va_end = vmaddr + vmsize;
                    if va_end > highest_va_end {
                        highest_va_end = va_end;
                    }

                    if segname == "__TEXT" {
                        text_segment_vmaddr = vmaddr;
                    }

                    if segname == "__LINKEDIT" {
                        linkedit_segment_offset = Some(offset as u64);
                        linkedit_vmaddr = vmaddr;
                        linkedit_fileoff = fileoff;
                        linkedit_filesize = filesize;
                    }

                    segments.push(MachOSegment {
                        name: segname.clone(),
                        vmaddr,
                        vmsize,
                        fileoff,
                        filesize,
                    });

                    // Parse sections within this segment
                    let mut sect_off = offset + 72; // past LC_SEGMENT_64 header
                    for _ in 0..nsects {
                        if sect_off + 80 > data.len() {
                            break;
                        }

                        let sectname_bytes = &data[sect_off..sect_off + 16];
                        let sectname = std::str::from_utf8(
                            &sectname_bytes
                                [..sectname_bytes.iter().position(|&b| b == 0).unwrap_or(16)],
                        )
                        .unwrap_or("")
                        .to_string();

                        let seg_of_sect_bytes = &data[sect_off + 16..sect_off + 32];
                        let seg_of_sect = std::str::from_utf8(
                            &seg_of_sect_bytes
                                [..seg_of_sect_bytes.iter().position(|&b| b == 0).unwrap_or(16)],
                        )
                        .unwrap_or("")
                        .to_string();

                        let sect_addr = u64::from_le_bytes(
                            data[sect_off + 32..sect_off + 40]
                                .try_into()
                                .expect("slice is 8 bytes"),
                        );
                        let sect_size = u64::from_le_bytes(
                            data[sect_off + 40..sect_off + 48]
                                .try_into()
                                .expect("slice is 8 bytes"),
                        );
                        let sect_offset = u32::from_le_bytes(
                            data[sect_off + 48..sect_off + 52]
                                .try_into()
                                .expect("slice is 4 bytes"),
                        );

                        // Track first section file offset for LC space calculation
                        if sect_offset > 0 && (sect_offset as u64) < first_section_offset {
                            first_section_offset = sect_offset as u64;
                        }

                        // __TEXT,__text
                        if seg_of_sect == "__TEXT" && sectname == "__text" {
                            text_section = Some(TextSection {
                                va: sect_addr,
                                offset: sect_offset as u64,
                                size: sect_size,
                            });
                        }

                        // Stub sections to exclude from instrumentation
                        if sectname == "__stubs"
                            || sectname == "__stub_helper"
                            || sectname == "__auth_stubs"
                        {
                            stubs_ranges.push(StubRange {
                                va: sect_addr,
                                size: sect_size,
                            });
                        }

                        // __DATA,__mod_init_func (S_MOD_INIT_FUNC_POINTERS = 0x09)
                        if sectname == "__mod_init_func" {
                            mod_init_func_section = Some(ModInitFuncInfo {
                                section_offset: sect_offset as u64,
                                section_size: sect_size,
                                section_va: sect_addr,
                            });
                            // Read existing init function pointers (64-bit absolute)
                            let ptr_start = sect_offset as usize;
                            let ptr_end = ptr_start + sect_size as usize;
                            if ptr_end <= data.len() {
                                let mut p = ptr_start;
                                while p + 8 <= ptr_end {
                                    let ptr_val = u64::from_le_bytes(
                                        data[p..p + 8].try_into().expect("slice is 8 bytes"),
                                    );
                                    if ptr_val != 0 {
                                        mod_init_func_pointers.push(ptr_val);
                                    }
                                    p += 8;
                                }
                            }
                        }

                        // __TEXT,__init_offsets (S_INIT_FUNC_OFFSETS = 0x16)
                        // Contains 32-bit offsets from __TEXT segment base.
                        if sectname == "__init_offsets" {
                            init_offsets_section = Some(InitOffsetsInfo {
                                section_offset: sect_offset as u64,
                                section_size: sect_size,
                                section_va: sect_addr,
                                text_segment_vmaddr,
                            });
                            // Read 32-bit offsets, convert to absolute VA
                            let off_start = sect_offset as usize;
                            let off_end = off_start + sect_size as usize;
                            if off_end <= data.len() {
                                let mut p = off_start;
                                while p + 4 <= off_end {
                                    let offset32 = u32::from_le_bytes(
                                        data[p..p + 4].try_into().expect("slice is 4 bytes"),
                                    );
                                    if offset32 != 0 {
                                        let abs_va = text_segment_vmaddr + offset32 as u64;
                                        mod_init_func_pointers.push(abs_va);
                                    }
                                    p += 4;
                                }
                            }
                        }

                        sect_off += 80; // sizeof(section_64)
                    }
                }

                LC_SYMTAB => {
                    if cmdsize >= 24 {
                        symtab_offset = u32::from_le_bytes(
                            data[offset + 8..offset + 12]
                                .try_into()
                                .expect("slice is 4 bytes"),
                        );
                        symtab_nsyms = u32::from_le_bytes(
                            data[offset + 12..offset + 16]
                                .try_into()
                                .expect("slice is 4 bytes"),
                        );
                        strtab_offset = u32::from_le_bytes(
                            data[offset + 16..offset + 20]
                                .try_into()
                                .expect("slice is 4 bytes"),
                        );
                        _strtab_size = u32::from_le_bytes(
                            data[offset + 20..offset + 24]
                                .try_into()
                                .expect("slice is 4 bytes"),
                        );
                    }
                }

                LC_MAIN => {
                    if cmdsize >= 24 {
                        let entryoff = u64::from_le_bytes(
                            data[offset + 8..offset + 16]
                                .try_into()
                                .expect("slice is 8 bytes"),
                        );
                        // entryoff is relative to __TEXT segment vmaddr
                        entry_point = text_segment_vmaddr + entryoff;
                        lc_main_entryoff_offset = Some(offset as u64 + 8);
                    }
                }

                LC_UNIXTHREAD => {
                    // Thread state contains register values; PC is at a format-specific offset.
                    // x86_64: flavor=4, count=42, state starts at offset+16, RIP at state+128 (=offset+144)
                    // ARM64: flavor=6, count=68, state starts at offset+16, PC at state+256 (=offset+272)
                    if cmdsize >= 16 {
                        let flavor = u32::from_le_bytes(
                            data[offset + 8..offset + 12]
                                .try_into()
                                .expect("slice is 4 bytes"),
                        );
                        match cputype {
                            CPU_TYPE_X86_64 => {
                                // x86_THREAD_STATE64: flavor=4
                                if flavor == 4 && cmdsize >= 152 {
                                    // RIP is at offset 16 (state start) + 128 = 144 from LC start
                                    let rip = u64::from_le_bytes(
                                        data[offset + 144..offset + 152]
                                            .try_into()
                                            .expect("slice is 8 bytes"),
                                    );
                                    entry_point = rip;
                                    lc_unixthread_pc_offset = Some(offset as u64 + 144);
                                }
                            }
                            CPU_TYPE_ARM64 => {
                                // ARM_THREAD_STATE64: flavor=6
                                if flavor == 6 && cmdsize >= 280 {
                                    // PC is at offset 16 (state start) + 256 = 272 from LC start
                                    let pc = u64::from_le_bytes(
                                        data[offset + 272..offset + 280]
                                            .try_into()
                                            .expect("slice is 8 bytes"),
                                    );
                                    entry_point = pc;
                                    lc_unixthread_pc_offset = Some(offset as u64 + 272);
                                }
                            }
                            _ => {}
                        }
                    }
                }

                LC_CODE_SIGNATURE => {
                    code_signature_lc_offset = Some(offset as u64);
                    code_signature_lc_size = cmdsize;
                }

                LC_DYLD_CHAINED_FIXUPS => {
                    if cmdsize >= 16 {
                        chained_fixups_lc_offset = Some(offset as u64);
                        chained_fixups_dataoff = u32::from_le_bytes(
                            data[offset + 8..offset + 12]
                                .try_into()
                                .expect("slice is 4 bytes"),
                        );
                        chained_fixups_datasize = u32::from_le_bytes(
                            data[offset + 12..offset + 16]
                                .try_into()
                                .expect("slice is 4 bytes"),
                        );
                    }
                }

                // Count dylib load commands to determine each dylib's 1-based ordinal.
                // dylib_command layout: cmd(4) cmdsize(4) name_offset(4) ...
                // The dylib name is a C string at offset (name_offset) from the LC start.
                LC_LOAD_DYLIB | LC_LOAD_WEAK_DYLIB | LC_REEXPORT_DYLIB | LC_LAZY_LOAD_DYLIB => {
                    dylib_ordinal += 1;
                    // dylib_command: name is at offset 8+name_off from LC start (name_off at offset+8)
                    if cmdsize >= 16 {
                        let name_off = u32::from_le_bytes(
                            data[offset + 8..offset + 12]
                                .try_into()
                                .expect("slice is 4 bytes"),
                        ) as usize;
                        let name_start = offset + name_off;
                        if name_start < offset + cmdsize as usize && name_start < data.len() {
                            let name_end = data[name_start..offset + cmdsize as usize]
                                .iter()
                                .position(|&b| b == 0)
                                .map(|p| name_start + p)
                                .unwrap_or(offset + cmdsize as usize);
                            if let Ok(name) = std::str::from_utf8(&data[name_start..name_end]) {
                                // Match libSystem: /usr/lib/libSystem.B.dylib or /usr/lib/libSystem.dylib
                                if name.contains("libSystem") {
                                    libsystem_ordinal = dylib_ordinal;
                                }
                            }
                        }
                    }
                }

                _ => {}
            }

            offset += cmdsize as usize;
        }

        let lc_end_offset = offset as u64;

        // If LC_MAIN was encountered before __TEXT segment, re-calculate entry point
        // (LC_MAIN may appear before LC_SEGMENT_64 for __TEXT in some linker outputs)
        if let Some(entryoff_offset) = lc_main_entryoff_offset
            && text_segment_vmaddr > 0
        {
            let entryoff_file_pos = entryoff_offset as usize;
            if entryoff_file_pos + 8 <= data.len() {
                let entryoff = u64::from_le_bytes(
                    data[entryoff_file_pos..entryoff_file_pos + 8]
                        .try_into()
                        .expect("slice is 8 bytes"),
                );
                entry_point = text_segment_vmaddr + entryoff;
            }
        }

        let text = text_section.context("no __TEXT,__text section found")?;

        // Calculate available LC space
        if first_section_offset == u64::MAX {
            first_section_offset = lc_end_offset;
        }
        let available_lc_space = first_section_offset.saturating_sub(lc_end_offset);

        // Parse symbol table for function addresses
        if symtab_offset > 0 && symtab_nsyms > 0 {
            func_symbols =
                parse_nlist_func_symbols(data, symtab_offset, symtab_nsyms, strtab_offset, &text);
        }

        Ok(MachOContext {
            text,
            entry_point,
            func_symbols,
            highest_va_end,
            is_dylib,
            is_64bit,
            segments,
            stubs_ranges,
            header_size,
            lc_end_offset,
            first_section_offset,
            available_lc_space,
            lc_main_entryoff_offset,
            lc_unixthread_pc_offset,
            text_segment_vmaddr,
            mod_init_func_section,
            init_offsets_section,
            mod_init_func_pointers,
            code_signature_lc_offset,
            code_signature_lc_size,
            linkedit_segment_offset,
            linkedit_vmaddr,
            linkedit_fileoff,
            linkedit_filesize,
            ncmds,
            sizeofcmds,
            ncmds_offset,
            sizeofcmds_offset,
            chained_fixups_lc_offset,
            chained_fixups_dataoff,
            chained_fixups_datasize,
            cputype,
            libsystem_ordinal,
        })
    }
}

/// Parse nlist_64 entries from LC_SYMTAB to extract function symbol addresses.
fn parse_nlist_func_symbols(
    data: &[u8],
    symtab_offset: u32,
    nsyms: u32,
    _strtab_offset: u32,
    text: &TextSection,
) -> Vec<u64> {
    let mut result = Vec::new();
    let nlist_size = 16usize; // sizeof(nlist_64)
    let base = symtab_offset as usize;

    for i in 0..nsyms as usize {
        let off = base + i * nlist_size;
        if off + nlist_size > data.len() {
            break;
        }

        // nlist_64: n_strx(4) n_type(1) n_sect(1) n_desc(2) n_value(8)
        let n_type = data[off + 4];
        let n_value = u64::from_le_bytes(
            data[off + 8..off + 16]
                .try_into()
                .expect("slice is 8 bytes"),
        );

        // N_TYPE mask: 0x0E. N_SECT = 0x0E means symbol is defined in a section.
        // N_EXT (0x01) = external symbol.
        let n_type_masked = n_type & 0x0E;
        if n_type_masked == 0x0E && n_value > 0 {
            // Symbol defined in a section — check if it's in __text
            if n_value >= text.va && n_value < text.va + text.size {
                result.push(n_value);
            }
        }
    }

    result.sort_unstable();
    result.dedup();
    result
}

/// Map a virtual address to a file offset using Mach-O segment layout.
pub fn va_to_file_offset_macho(va: u64, ctx: &MachOContext) -> Option<usize> {
    // Quick path: VA in __text
    if va >= ctx.text.va && va < ctx.text.va + ctx.text.size {
        return Some((ctx.text.offset + (va - ctx.text.va)) as usize);
    }

    for seg in &ctx.segments {
        if seg.filesize > 0 && va >= seg.vmaddr && va < seg.vmaddr + seg.filesize {
            return Some((seg.fileoff + (va - seg.vmaddr)) as usize);
        }
    }
    None
}

/// All allocator function names we intercept (Mach-O variant).
/// Same targets as the ELF side, but Mach-O symbols have a leading underscore.
const MACHO_ALLOC_TARGETS: &[(&str, &str)] = &[
    ("_malloc", "malloc"),
    ("_free", "free"),
    ("_calloc", "calloc"),
    ("_realloc", "realloc"),
    ("_memalign", "memalign"),
    ("_aligned_alloc", "aligned_alloc"),
    ("_posix_memalign", "posix_memalign"),
    // jemalloc extended API
    ("_mallocx", "mallocx"),
    ("_sdallocx", "sdallocx"),
    ("_rallocx", "rallocx"),
    // jemalloc prefixed variants
    ("_je_malloc", "malloc"),
    ("_je_free", "free"),
    ("_je_calloc", "calloc"),
    ("_je_realloc", "realloc"),
    ("_je_memalign", "memalign"),
    ("_je_mallocx", "mallocx"),
    ("_je_sdallocx", "sdallocx"),
    ("_je_rallocx", "rallocx"),
];

/// Discover allocator functions in a Mach-O binary for heap sanitiser interception.
///
/// Scans the LC_SYMTAB nlist entries for function symbols matching allocator names.
/// Returns `AllocatorSymbols` with `FuncEntry` kind (direct function patching).
///
/// Mach-O symbols have a leading underscore (`_malloc`), which is stripped to produce
/// the canonical name used as key in `AllocatorSymbols::entries`.
pub fn find_allocator_symbols_macho(
    ctx: &MachOContext,
    data: &[u8],
) -> crate::elf::AllocatorSymbols {
    use crate::elf::{AllocEntry, AllocEntryKind, AllocatorSymbols};

    let mut syms = AllocatorSymbols::default();

    // We need to re-parse LC_SYMTAB to get string table access.
    // Walk load commands to find LC_SYMTAB.
    let header_size = 32usize; // sizeof(mach_header_64)
    let ncmds = u32::from_le_bytes(
        data.get(16..20)
            .and_then(|s| s.try_into().ok())
            .unwrap_or([0; 4]),
    );

    let mut symtab_off: u32 = 0;
    let mut symtab_nsyms: u32 = 0;
    let mut strtab_off: u32 = 0;
    let mut strtab_sz: u32 = 0;

    let mut offset = header_size;
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
        if cmdsize < 8 || offset + cmdsize as usize > data.len() {
            break;
        }

        if cmd == 0x02 /* LC_SYMTAB */ && cmdsize >= 24 {
            symtab_off = u32::from_le_bytes(
                data[offset + 8..offset + 12]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            symtab_nsyms = u32::from_le_bytes(
                data[offset + 12..offset + 16]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            strtab_off = u32::from_le_bytes(
                data[offset + 16..offset + 20]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            strtab_sz = u32::from_le_bytes(
                data[offset + 20..offset + 24]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            break;
        }
        offset += cmdsize as usize;
    }

    if symtab_off == 0 || symtab_nsyms == 0 || strtab_off == 0 {
        tracing::debug!("find_allocator_symbols_macho: no LC_SYMTAB found");
        return syms;
    }

    let strtab_end = strtab_off as usize + strtab_sz as usize;
    if strtab_end > data.len() {
        tracing::warn!("find_allocator_symbols_macho: strtab extends beyond file");
        return syms;
    }

    let nlist_size = 16usize; // sizeof(nlist_64)
    let base = symtab_off as usize;

    for i in 0..symtab_nsyms as usize {
        let off = base + i * nlist_size;
        if off + nlist_size > data.len() {
            break;
        }

        // nlist_64: n_strx(4) n_type(1) n_sect(1) n_desc(2) n_value(8)
        let n_strx = u32::from_le_bytes(data[off..off + 4].try_into().expect("slice is 4 bytes"));
        let n_type = data[off + 4];
        let n_value = u64::from_le_bytes(
            data[off + 8..off + 16]
                .try_into()
                .expect("slice is 8 bytes"),
        );

        // N_SECT (0x0E) means defined in a section; skip undefined/external-only symbols.
        let n_type_masked = n_type & 0x0E;
        if n_type_masked != 0x0E || n_value == 0 {
            continue;
        }

        // Resolve symbol name from string table.
        let str_off = strtab_off as usize + n_strx as usize;
        if str_off >= data.len() {
            continue;
        }
        let name_end = data[str_off..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| str_off + p)
            .unwrap_or(data.len().min(strtab_end));
        let name = match std::str::from_utf8(&data[str_off..name_end]) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Match against known allocator targets.
        let canonical = match MACHO_ALLOC_TARGETS
            .iter()
            .find(|&&(macho_name, _)| macho_name == name)
        {
            Some(&(_, canonical)) => canonical,
            None => continue,
        };

        // Convert VA to file offset.
        if let Some(file_off) = va_to_file_offset_macho(n_value, ctx) {
            syms.entries.entry(canonical).or_insert(AllocEntry {
                patch_offset: file_off as u64,
                patch_va: n_value,
                kind: AllocEntryKind::FuncEntry,
            });
        }
    }

    syms
}

/// Check if an address falls within a stub section.
pub fn is_in_stubs(addr: u64, ctx: &MachOContext) -> bool {
    for stub in &ctx.stubs_ranges {
        if addr >= stub.va && addr < stub.va + stub.size {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid Mach-O 64-bit x86_64 binary for testing.
    fn build_minimal_macho() -> Vec<u8> {
        let mut data = vec![0u8; 4096];

        // mach_header_64
        data[0..4].copy_from_slice(&MH_MAGIC_64.to_le_bytes()); // magic
        data[4..8].copy_from_slice(&CPU_TYPE_X86_64.to_le_bytes()); // cputype
        data[8..12].copy_from_slice(&0x03u32.to_le_bytes()); // cpusubtype (CPU_SUBTYPE_ALL)
        data[12..16].copy_from_slice(&0x02u32.to_le_bytes()); // filetype: MH_EXECUTE
        data[16..20].copy_from_slice(&2u32.to_le_bytes()); // ncmds = 2
        // sizeofcmds will be filled after building LCs
        // flags at 24..28 = 0
        // reserved at 28..32 = 0

        let header_size = 32usize;
        let mut lc_offset = header_size;

        // LC_SEGMENT_64 for __TEXT (contains __text section)
        let _text_seg_start = lc_offset;
        let text_lc_size = 72 + 80; // segment + 1 section
        data[lc_offset..lc_offset + 4].copy_from_slice(&LC_SEGMENT_64.to_le_bytes());
        data[lc_offset + 4..lc_offset + 8].copy_from_slice(&(text_lc_size as u32).to_le_bytes());
        // segname = "__TEXT"
        data[lc_offset + 8..lc_offset + 14].copy_from_slice(b"__TEXT");
        // vmaddr = 0x100000000
        data[lc_offset + 24..lc_offset + 32].copy_from_slice(&0x100000000u64.to_le_bytes());
        // vmsize = 0x1000
        data[lc_offset + 32..lc_offset + 40].copy_from_slice(&0x1000u64.to_le_bytes());
        // fileoff = 0
        data[lc_offset + 40..lc_offset + 48].copy_from_slice(&0u64.to_le_bytes());
        // filesize = 0x1000
        data[lc_offset + 48..lc_offset + 56].copy_from_slice(&0x1000u64.to_le_bytes());
        // maxprot = 7 (RWX)
        data[lc_offset + 56..lc_offset + 60].copy_from_slice(&7u32.to_le_bytes());
        // initprot = 5 (RX)
        data[lc_offset + 60..lc_offset + 64].copy_from_slice(&5u32.to_le_bytes());
        // nsects = 1
        data[lc_offset + 64..lc_offset + 68].copy_from_slice(&1u32.to_le_bytes());
        // flags = 0
        data[lc_offset + 68..lc_offset + 72].copy_from_slice(&0u32.to_le_bytes());

        // section_64: __text
        let sect_off = lc_offset + 72;
        data[sect_off..sect_off + 6].copy_from_slice(b"__text");
        data[sect_off + 16..sect_off + 22].copy_from_slice(b"__TEXT");
        // addr = 0x100000800
        data[sect_off + 32..sect_off + 40].copy_from_slice(&0x100000800u64.to_le_bytes());
        // size = 0x100
        data[sect_off + 40..sect_off + 48].copy_from_slice(&0x100u64.to_le_bytes());
        // offset = 0x800
        data[sect_off + 48..sect_off + 52].copy_from_slice(&0x800u32.to_le_bytes());

        lc_offset += text_lc_size;

        // LC_MAIN
        let main_lc_size = 24u32;
        data[lc_offset..lc_offset + 4].copy_from_slice(&LC_MAIN.to_le_bytes());
        data[lc_offset + 4..lc_offset + 8].copy_from_slice(&main_lc_size.to_le_bytes());
        // entryoff = 0x800 (relative to __TEXT vmaddr)
        data[lc_offset + 8..lc_offset + 16].copy_from_slice(&0x800u64.to_le_bytes());

        lc_offset += main_lc_size as usize;

        // Update sizeofcmds
        let sizeofcmds = (lc_offset - header_size) as u32;
        data[20..24].copy_from_slice(&sizeofcmds.to_le_bytes());

        // Put some x86_64 instructions at offset 0x800 (NOP sled + RET)
        let text_file_off = 0x800;
        for i in 0..0xFFusize {
            data[text_file_off + i] = 0x90; // NOP
        }
        data[text_file_off + 0xFF] = 0xC3; // RET

        data
    }

    #[test]
    fn test_parse_minimal_macho() {
        let data = build_minimal_macho();
        let ctx = MachOContext::parse(&data).expect("failed to parse minimal Mach-O");

        assert_eq!(ctx.text.va, 0x100000800);
        assert_eq!(ctx.text.offset, 0x800);
        assert_eq!(ctx.text.size, 0x100);
        assert_eq!(ctx.entry_point, 0x100000800);
        assert_eq!(ctx.text_segment_vmaddr, 0x100000000);
        assert!(!ctx.is_dylib);
        assert!(ctx.is_64bit);
        assert_eq!(ctx.ncmds, 2);
        assert!(ctx.lc_main_entryoff_offset.is_some());
        assert_eq!(ctx.highest_va_end, 0x100001000);
    }

    #[test]
    fn test_va_to_file_offset() {
        let data = build_minimal_macho();
        let ctx = MachOContext::parse(&data).expect("failed to parse minimal Mach-O");

        // VA in __text section
        let off = va_to_file_offset_macho(0x100000800, &ctx);
        assert_eq!(off, Some(0x800));

        let off = va_to_file_offset_macho(0x100000850, &ctx);
        assert_eq!(off, Some(0x850));

        // VA outside any segment
        let off = va_to_file_offset_macho(0xDEADBEEF, &ctx);
        assert_eq!(off, None);
    }

    #[test]
    fn test_stub_range_detection() {
        let ctx = MachOContext {
            text: TextSection {
                va: 0x1000,
                offset: 0x1000,
                size: 0x100,
            },
            entry_point: 0x1000,
            func_symbols: vec![],
            highest_va_end: 0x2000,
            is_dylib: false,
            is_64bit: true,
            segments: vec![],
            stubs_ranges: vec![
                StubRange {
                    va: 0x1500,
                    size: 0x20,
                },
                StubRange {
                    va: 0x1600,
                    size: 0x30,
                },
            ],
            header_size: 32,
            lc_end_offset: 200,
            first_section_offset: 4096,
            available_lc_space: 3896,
            lc_main_entryoff_offset: None,
            lc_unixthread_pc_offset: None,
            text_segment_vmaddr: 0x1000,
            mod_init_func_section: None,
            init_offsets_section: None,
            mod_init_func_pointers: vec![],
            code_signature_lc_offset: None,
            code_signature_lc_size: 0,
            linkedit_segment_offset: None,
            linkedit_vmaddr: 0,
            linkedit_fileoff: 0,
            linkedit_filesize: 0,
            chained_fixups_lc_offset: None,
            chained_fixups_dataoff: 0,
            chained_fixups_datasize: 0,
            ncmds: 1,
            sizeofcmds: 100,
            ncmds_offset: 16,
            sizeofcmds_offset: 20,
            cputype: 0x01000007, // CPU_TYPE_X86_64
            libsystem_ordinal: 1,
        };

        assert!(is_in_stubs(0x1500, &ctx));
        assert!(is_in_stubs(0x1510, &ctx));
        assert!(!is_in_stubs(0x1520, &ctx)); // past first range
        assert!(is_in_stubs(0x1600, &ctx));
        assert!(!is_in_stubs(0x1000, &ctx)); // not in stubs
    }

    #[test]
    fn test_available_lc_space() {
        let data = build_minimal_macho();
        let ctx = MachOContext::parse(&data).expect("failed to parse minimal Mach-O");

        // first_section_offset = 0x800 (__text), lc_end = header + sizeofcmds
        // available = 0x800 - lc_end
        assert!(ctx.available_lc_space > 0);
        assert!(
            ctx.available_lc_space >= 72,
            "should have room for at least one LC_SEGMENT_64"
        );
    }

    #[test]
    fn test_reject_32bit() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&0xFEEDFACEu32.to_le_bytes()); // 32-bit LE
        let result = MachOContext::parse(&data);
        assert!(result.is_err());
    }
}
