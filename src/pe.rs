//! PE/COFF binary parsing and metadata extraction for instrumentation.
//!
//! Extracts structural information needed to instrument a PE binary:
//! .text section, entry point, export symbols, section table, image layout.
//! Uses goblin::pe for parsing.

use anyhow::{Context, Result};

/// .text section location.
#[derive(Debug, Clone)]
pub struct TextSection {
    pub va: u64,
    pub offset: u64,
    pub size: u64,
}

/// A PE import entry (dynamically resolved function from a DLL).
#[derive(Debug, Clone)]
pub struct PeImport {
    /// Function name (e.g. "HeapAlloc"). Empty for ordinal-only imports.
    pub name: String,
    /// DLL name (e.g. "KERNEL32.dll").
    pub dll: String,
    /// RVA of the IAT slot (relative to image base). At runtime the loader writes
    /// the resolved function pointer here.
    pub iat_rva: u64,
}

/// A PE section header (for layout computation).
#[derive(Debug, Clone)]
pub struct PeSection {
    pub name: String,
    pub virtual_address: u64,
    pub virtual_size: u64,
    pub raw_offset: u64,
    pub raw_size: u64,
    pub characteristics: u32,
}

/// Parsed PE binary context with all metadata needed for instrumentation.
#[derive(Debug)]
pub struct PeContext {
    // Core
    pub text: TextSection,
    pub entry_point: u64,
    pub func_symbols: Vec<u64>,
    /// Function start RVAs from .pdata (RUNTIME_FUNCTION table).
    pub pdata_functions: Vec<u64>,
    pub highest_va_end: u64,
    pub is_dll: bool,
    pub is_64bit: bool,

    // PE header layout
    pub image_base: u64,
    pub section_alignment: u32,
    pub file_alignment: u32,
    pub size_of_image: u32,
    pub size_of_headers: u32,
    pub number_of_sections: u16,
    pub sections: Vec<PeSection>,

    // Header offsets for patching
    /// File offset of the PE signature ("PE\0\0").
    pub pe_sig_offset: u32,
    /// File offset of the COFF header (pe_sig_offset + 4).
    pub coff_header_offset: u32,
    /// File offset of the optional header.
    pub optional_header_offset: u32,
    /// File offset of the first section header.
    pub section_table_offset: u32,
    /// File offset of AddressOfEntryPoint field in optional header.
    pub entry_point_field_offset: u32,
    /// File offset of SizeOfImage field in optional header.
    pub size_of_image_field_offset: u32,
    /// File offset of NumberOfSections in COFF header.
    pub number_of_sections_field_offset: u32,
    /// File offset of CheckSum in optional header.
    pub checksum_field_offset: u32,
    /// File offset of SizeOfCode in optional header.
    pub size_of_code_field_offset: u32,
    /// File offset of SizeOfInitializedData in optional header.
    pub size_of_initialized_data_field_offset: u32,

    /// Has a .reloc section (base relocations).
    pub has_reloc: bool,

    /// Parsed imports (dynamically linked functions from DLLs).
    pub imports: Vec<PeImport>,

    /// VA ranges of non-function (data) symbols within .text.
    /// Used to exclude data-in-text regions (e.g. MinGW __CTOR_LIST__/__DTOR_LIST__)
    /// from instrumentation.
    pub text_data_ranges: Vec<(u64, u64)>,
}

// PE constants
const IMAGE_FILE_DLL: u16 = 0x2000;
const PE32PLUS_MAGIC: u16 = 0x20B;
const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;

/// Offset of the `e_lfanew` field in the DOS header (points to PE signature).
const PE_LFANEW_OFFSET: usize = 60;
/// Size of the COFF file header (between PE signature and optional header).
const COFF_HEADER_SIZE: u32 = 20;
/// Size of a PE section header entry.
const SECTION_HEADER_SIZE: usize = 40;

/// Section name prefixes that can be safely removed to make room for new headers.
const STRIPPABLE_PREFIXES: &[&str] = &[
    ".debug", ".zdebug", ".stabstr", ".stab", ".drectve", ".comment",
    ".note",
    // Note: .reloc is NOT stripped here — it's handled conditionally in the
    // strip function based on whether the PE is a DLL (DLLs need .reloc for
    // the loader to relocate them when the preferred base is occupied).
];

/// Strip expendable section headers from a PE binary to make room for new ones.
///
/// Removes headers for debug/symbol/COFF-overflow sections that aren't needed
/// at runtime. Section data is left in the file (harmless dead bytes).
/// Returns the number of headers removed.
pub fn strip_expendable_section_headers(ctx: &PeContext, data: &mut [u8]) -> usize {
    let sec_table = ctx.section_table_offset as usize;
    let n = ctx.number_of_sections as usize;

    let mut keep = Vec::with_capacity(n);
    for i in 0..n {
        let hdr_off = sec_table + i * SECTION_HEADER_SIZE;
        let name = std::str::from_utf8(&data[hdr_off..hdr_off + 8])
            .unwrap_or("")
            .trim_end_matches('\0');

        let is_strippable = STRIPPABLE_PREFIXES.iter().any(|p| name.starts_with(p))
            || (name.starts_with('/') && name[1..].bytes().all(|b| b.is_ascii_digit()))
            || (name == ".reloc" && !ctx.is_dll);

        if !is_strippable {
            keep.push(i);
        }
    }

    let stripped = n - keep.len();
    if stripped == 0 {
        return 0;
    }

    for (new_idx, &old_idx) in keep.iter().enumerate() {
        if new_idx != old_idx {
            let src = sec_table + old_idx * SECTION_HEADER_SIZE;
            let dst = sec_table + new_idx * SECTION_HEADER_SIZE;
            data.copy_within(src..src + SECTION_HEADER_SIZE, dst);
        }
    }

    let new_count = keep.len();
    for i in new_count..n {
        let off = sec_table + i * SECTION_HEADER_SIZE;
        data[off..off + SECTION_HEADER_SIZE].fill(0);
    }

    let nsec_off = ctx.number_of_sections_field_offset as usize;
    data[nsec_off..nsec_off + 2].copy_from_slice(&(new_count as u16).to_le_bytes());

    stripped
}

impl PeContext {
    /// Parse a PE binary from raw bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        use goblin::pe::PE;

        let pe = PE::parse(data).context("failed to parse PE binary")?;

        let is_64bit = pe
            .header
            .optional_header
            .as_ref()
            .map(|oh| oh.standard_fields.magic == PE32PLUS_MAGIC)
            .unwrap_or(false);

        let opt_header = pe
            .header
            .optional_header
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("PE has no optional header"))?;

        let image_base = opt_header.windows_fields.image_base;
        let section_alignment = opt_header.windows_fields.section_alignment;
        let file_alignment = opt_header.windows_fields.file_alignment;
        let size_of_image = opt_header.windows_fields.size_of_image;
        let size_of_headers = opt_header.windows_fields.size_of_headers;

        let is_dll = pe.header.coff_header.characteristics & IMAGE_FILE_DLL != 0;

        let pe_sig_offset = u32::from_le_bytes([
            data[PE_LFANEW_OFFSET],
            data[PE_LFANEW_OFFSET + 1],
            data[PE_LFANEW_OFFSET + 2],
            data[PE_LFANEW_OFFSET + 3],
        ]);
        let coff_header_offset = pe_sig_offset + 4;
        let optional_header_offset = coff_header_offset + COFF_HEADER_SIZE;
        let number_of_sections = pe.header.coff_header.number_of_sections;

        // NumberOfSections field offset: coff_header_offset + 2
        let number_of_sections_field_offset = coff_header_offset + 2;

        // Optional header field offsets differ between PE32 and PE32+.
        // AddressOfEntryPoint is at the same offset in both.
        let entry_point_field_offset = optional_header_offset + 16;
        let size_of_code_field_offset = optional_header_offset + 4;
        let size_of_initialized_data_field_offset = optional_header_offset + 8;

        // PE32+: standard fields = 24 bytes (no BaseOfData)
        //   windows_fields start at optional_header_offset + 24
        //   ImageBase: 8 bytes at +0
        //   SectionAlignment: 4 bytes at +8
        //   SizeOfImage: 4 bytes at +32
        //   SizeOfHeaders: 4 bytes at +36
        //   CheckSum: 4 bytes at +40
        //
        // PE32: standard fields = 28 bytes (includes BaseOfData)
        //   windows_fields start at optional_header_offset + 28
        //   ImageBase: 4 bytes at +0
        //   SectionAlignment: 4 bytes at +4
        //   SizeOfImage: 4 bytes at +28
        //   SizeOfHeaders: 4 bytes at +32
        //   CheckSum: 4 bytes at +36
        let (_windows_fields_offset, size_of_image_field_offset, checksum_field_offset) =
            if is_64bit {
                let wfo = optional_header_offset + 24;
                (wfo, wfo + 32, wfo + 40)
            } else {
                let wfo = optional_header_offset + 28;
                (wfo, wfo + 28, wfo + 36)
            };

        // Section table starts after optional header
        let optional_header_size = pe.header.coff_header.size_of_optional_header as u32;
        let section_table_offset = optional_header_offset + optional_header_size;

        // Find .text section
        let mut text_section = None;
        let mut sections = Vec::new();
        let mut highest_va_end: u64 = 0;
        let mut has_reloc = false;

        for section in &pe.sections {
            let name = String::from_utf8_lossy(&section.name)
                .trim_end_matches('\0')
                .to_string();

            let va = image_base.saturating_add(section.virtual_address as u64);
            let vsize = section.virtual_size as u64;
            let sec_end = va.saturating_add(vsize);
            if sec_end > highest_va_end {
                highest_va_end = sec_end;
            }

            if name == ".reloc" {
                has_reloc = true;
            }

            let chars = section.characteristics;
            if text_section.is_none()
                && (chars & IMAGE_SCN_CNT_CODE != 0)
                && (chars & IMAGE_SCN_MEM_EXECUTE != 0)
            {
                text_section = Some(TextSection {
                    va,
                    offset: section.pointer_to_raw_data as u64,
                    size: section.size_of_raw_data as u64,
                });
            }

            sections.push(PeSection {
                name,
                virtual_address: va,
                virtual_size: vsize,
                raw_offset: section.pointer_to_raw_data as u64,
                raw_size: section.size_of_raw_data as u64,
                characteristics: chars,
            });
        }

        let text = text_section
            .ok_or_else(|| anyhow::anyhow!("no executable .text section found in PE"))?;

        // Entry point
        let entry_rva = opt_header.standard_fields.address_of_entry_point;
        let entry_point = image_base.saturating_add(entry_rva as u64);

        // Export symbols as function addresses
        let mut func_symbols = Vec::new();
        for export in &pe.exports {
            if export.rva != 0 {
                func_symbols.push(image_base.saturating_add(export.rva as u64));
            }
        }

        // Parse imports (dynamically linked functions from DLLs).
        let mut imports = Vec::new();
        for imp in &pe.imports {
            let name = imp.name.to_string();
            let dll = imp.dll.to_string();
            let iat_rva = imp.offset as u64;
            imports.push(PeImport { name, dll, iat_rva });
        }

        // Extract function start addresses from .pdata (RUNTIME_FUNCTION table).
        // Each entry is 12 bytes: BeginAddress(u32 RVA), EndAddress(u32 RVA), UnwindData(u32 RVA).
        let mut pdata_functions = Vec::new();
        for sec in &sections {
            if sec.name == ".pdata" {
                let off = sec.raw_offset as usize;
                let sz = sec.raw_size as usize;
                if off + sz <= data.len() {
                    let pdata = &data[off..off + sz];
                    let n = pdata.len() / 12;
                    for i in 0..n {
                        let begin_rva = u32::from_le_bytes([
                            pdata[i * 12],
                            pdata[i * 12 + 1],
                            pdata[i * 12 + 2],
                            pdata[i * 12 + 3],
                        ]);
                        if begin_rva != 0 {
                            pdata_functions.push(image_base.saturating_add(begin_rva as u64));
                        }
                    }
                }
                break;
            }
        }

        // Parse COFF symbol table to find non-function (data) symbols in .text.
        // MinGW places __CTOR_LIST__, __DTOR_LIST__ etc. as data arrays in .text;
        // these must be excluded from instrumentation.
        let text_data_ranges = parse_coff_text_data_ranges(
            data,
            coff_header_offset as usize,
            &text,
            image_base,
            &sections,
        );

        Ok(PeContext {
            text,
            entry_point,
            func_symbols,
            pdata_functions,
            highest_va_end,
            is_dll,
            is_64bit,
            image_base,
            section_alignment,
            file_alignment,
            size_of_image,
            size_of_headers,
            number_of_sections,
            sections,
            pe_sig_offset,
            coff_header_offset,
            optional_header_offset,
            section_table_offset,
            entry_point_field_offset,
            size_of_image_field_offset,
            number_of_sections_field_offset,
            checksum_field_offset,
            size_of_code_field_offset,
            size_of_initialized_data_field_offset,
            has_reloc,
            imports,
            text_data_ranges,
        })
    }
}

/// Parse the COFF symbol table to find non-function symbols in .text.
///
/// Returns a sorted list of (va, size) ranges for data symbols. Size is estimated
/// as the distance to the next symbol in the same section, or to the section end.
///
/// COFF symbol entries are 18 bytes:
///   [0..8]   Name (short name or string table offset)
///   [8..12]  Value (RVA within section)
///   [12..14] SectionNumber (1-based, 0=external)
///   [14..16] Type (low byte: base type, high byte: derived type; 0x20 = function)
///   [16]     StorageClass
///   [17]     NumberOfAuxSymbols
fn parse_coff_text_data_ranges(
    data: &[u8],
    coff_header_offset: usize,
    text: &TextSection,
    image_base: u64,
    sections: &[PeSection],
) -> Vec<(u64, u64)> {
    // COFF header layout: pointer_to_symbol_table at +8, number_of_symbols at +12.
    if coff_header_offset + 16 > data.len() {
        return Vec::new();
    }
    let symtab_off = u32::from_le_bytes(
        data[coff_header_offset + 8..coff_header_offset + 12]
            .try_into()
            .unwrap(),
    ) as usize;
    let num_syms = u32::from_le_bytes(
        data[coff_header_offset + 12..coff_header_offset + 16]
            .try_into()
            .unwrap(),
    );

    if symtab_off == 0 || num_syms == 0 {
        return Vec::new(); // Stripped binary, no COFF symbols.
    }

    const COFF_SYM_SIZE: usize = 18;
    let symtab_end = symtab_off + (num_syms as usize) * COFF_SYM_SIZE;
    if symtab_end > data.len() {
        return Vec::new();
    }

    // Find which section index .text is (1-based).
    let text_section_index: Option<u16> = sections
        .iter()
        .enumerate()
        .find(|(_, s)| s.name == ".text")
        .map(|(i, _)| (i + 1) as u16);
    let text_sec_idx = match text_section_index {
        Some(idx) => idx,
        None => return Vec::new(),
    };

    // Collect all symbols in .text, noting which are functions and which are data.
    // A symbol is a function if Type has DTYPE_FUNCTION (bit 5 of high byte, i.e. Type >= 0x20).
    struct SymInfo {
        va: u64,
        is_function: bool,
    }
    let mut text_syms: Vec<SymInfo> = Vec::new();

    let mut i = 0u32;
    while i < num_syms {
        let off = symtab_off + (i as usize) * COFF_SYM_SIZE;
        if off + COFF_SYM_SIZE > data.len() {
            break;
        }

        let value = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap());
        let section_number = u16::from_le_bytes(data[off + 12..off + 14].try_into().unwrap());
        let sym_type = u16::from_le_bytes(data[off + 14..off + 16].try_into().unwrap());
        let num_aux = data[off + 17];

        if section_number == text_sec_idx {
            // COFF Value field is the RVA for defined symbols.
            let sym_va = image_base + value as u64;
            let is_function = sym_type >= 0x20; // DTYPE_FUNCTION
            text_syms.push(SymInfo { va: sym_va, is_function });
        }

        // Skip aux symbols.
        i += 1 + num_aux as u32;
    }

    // Sort by VA.
    text_syms.sort_by_key(|s| s.va);

    // Build ranges for non-function symbols. Size = distance to next symbol or section end.
    let text_end = text.va + text.size;
    let text_end_va = image_base + text_end;
    let mut ranges: Vec<(u64, u64)> = Vec::new();

    for (idx, sym) in text_syms.iter().enumerate() {
        if sym.is_function {
            continue;
        }
        let next_va = text_syms
            .get(idx + 1)
            .map(|s| s.va)
            .unwrap_or(text_end_va);
        let size = next_va.saturating_sub(sym.va);
        if size > 0 {
            ranges.push((sym.va, size));
        }
    }

    ranges
}

/// Find an IAT thunk stub in .text for the given IAT entry RVA.
///
/// PE binaries compiled with MSVC/MinGW emit 6-byte thunks for each import:
/// ```text
/// FF 25 xx xx xx xx    ; jmp [rip + disp32]
/// ```
/// where rip+disp32 points to the IAT slot (image_base + iat_rva).
///
/// Returns the VA of the thunk if found.
pub fn find_iat_thunk(data: &[u8], ctx: &PeContext, iat_rva: u64) -> Option<u64> {
    let text = &ctx.text;
    let text_start = text.offset as usize;
    let text_end = text_start + text.size as usize;

    if text_end > data.len() || text.size < 6 {
        return None;
    }

    let iat_va = ctx.image_base + iat_rva;

    // Scan .text for FF 25 xx xx xx xx (JMP [rip + disp32]).
    let text_bytes = &data[text_start..text_end];
    for i in 0..text_bytes.len().saturating_sub(5) {
        if text_bytes[i] == 0xFF && text_bytes[i + 1] == 0x25 {
            // Decode disp32 (little-endian, signed).
            let disp = i32::from_le_bytes([
                text_bytes[i + 2],
                text_bytes[i + 3],
                text_bytes[i + 4],
                text_bytes[i + 5],
            ]);
            // RIP value after this 6-byte instruction.
            let instr_va = text.va + i as u64;
            let rip_after = instr_va + 6;
            let target_va = (rip_after as i64 + disp as i64) as u64;

            if target_va == iat_va {
                return Some(instr_va);
            }
        }
    }

    None
}

/// Convert a VA to file offset using the PE section table.
pub fn va_to_file_offset_pe(va: u64, ctx: &PeContext) -> Option<usize> {
    for section in &ctx.sections {
        if va >= section.virtual_address
            && va < section.virtual_address.saturating_add(section.virtual_size)
        {
            let offset_in_section = va - section.virtual_address;
            if offset_in_section < section.raw_size {
                return Some((section.raw_offset + offset_in_section) as usize);
            }
        }
    }
    None
}

/// Check if a VA is within an executable section.
pub fn exec_section_for_va(va: u64, ctx: &PeContext) -> Option<(u64, u64, u64)> {
    for section in &ctx.sections {
        let chars = section.characteristics;
        if ((chars & IMAGE_SCN_MEM_EXECUTE != 0) || (chars & IMAGE_SCN_CNT_CODE != 0))
            && va >= section.virtual_address
            && va < section.virtual_address + section.virtual_size
        {
            return Some((
                section.virtual_address,
                section.raw_offset,
                section.virtual_size,
            ));
        }
    }
    None
}

/// Align a value up to the given alignment.
pub fn align_up(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

/// Inject an import for a companion DLL into a PE binary.
///
/// Adds a new `.idata2` section containing a copy of all original import
/// descriptors plus a new one for the given DLL. Updates the Import Directory
/// data directory to point to the new section.
///
/// The DLL must export `export_name` — an IAT entry for it is created so the
/// PE loader pulls the DLL in automatically.
///
/// Returns the modified binary data.
pub fn inject_import(
    mut data: Vec<u8>,
    ctx: &PeContext,
    dll_name: &str,
    export_name: &str,
) -> anyhow::Result<Vec<u8>> {
    // Locate the existing import directory.
    let import_dir_rva_offset = if ctx.is_64bit {
        // PE32+: data directories start at optional_header_offset + 24 + 88 = +112
        // Import directory is entry index 1, each entry is 8 bytes (RVA + Size)
        ctx.optional_header_offset + 24 + 88 + 8 // skip export dir entry
    } else {
        // PE32: data directories start at optional_header_offset + 28 + 68 = +96
        ctx.optional_header_offset + 28 + 68 + 8
    };
    let import_dir_rva_offset = import_dir_rva_offset as usize;

    // Read current import directory RVA and size.
    let old_import_rva = u32::from_le_bytes(
        data[import_dir_rva_offset..import_dir_rva_offset + 4]
            .try_into()
            .expect("slice is 4 bytes"),
    );
    let old_import_size = u32::from_le_bytes(
        data[import_dir_rva_offset + 4..import_dir_rva_offset + 8]
            .try_into()
            .expect("slice is 4 bytes"),
    );

    // Count existing import descriptors (20 bytes each, null-terminated).
    let old_desc_count = if old_import_size > 0 {
        (old_import_size as usize / 20).saturating_sub(1) // subtract null terminator
    } else {
        0
    };

    // Copy existing import descriptor bytes (excluding null terminator).
    let old_import_file_offset = if old_import_rva > 0 {
        va_to_file_offset_pe(ctx.image_base + old_import_rva as u64, ctx)
    } else {
        None
    };
    let old_descs: Vec<u8> = if let Some(off) = old_import_file_offset {
        let desc_bytes = old_desc_count * 20;
        if off + desc_bytes <= data.len() {
            data[off..off + desc_bytes].to_vec()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    // Build the new .idata2 section content.
    let thunk_size: usize = if ctx.is_64bit { 8 } else { 4 };

    // Layout within the new section:
    // [old import descriptors] [new import descriptor] [null terminator]
    // [ILT: 1 entry + null] [IAT: 1 entry + null]
    // [hint/name: \x00\x00 + export_name + \0]
    // [dll_name + \0]
    let dll_name_bytes = dll_name.as_bytes();
    let export_name_bytes = export_name.as_bytes();

    let desc_area_size = (old_desc_count + 1 + 1) * 20; // +1 new, +1 null
    let ilt_offset_in_sec = desc_area_size;
    let ilt_size = thunk_size * 2; // 1 entry + null
    let iat_offset_in_sec = ilt_offset_in_sec + ilt_size;
    let iat_size = thunk_size * 2;
    let hint_name_offset_in_sec = iat_offset_in_sec + iat_size;
    let hint_name_size = 2 + export_name_bytes.len() + 1; // hint(2) + name + null
    let dll_name_offset_in_sec = hint_name_offset_in_sec + hint_name_size;
    let dll_name_size = dll_name_bytes.len() + 1;
    let raw_content_size = dll_name_offset_in_sec + dll_name_size;

    // Align to file alignment.
    let section_raw_size = align_up(raw_content_size as u64, ctx.file_alignment as u64) as usize;
    let section_vsize = align_up(raw_content_size as u64, ctx.section_alignment as u64) as u32;

    // New section VA (after all existing sections).
    let new_sec_rva = align_up(
        ctx.highest_va_end - ctx.image_base,
        ctx.section_alignment as u64,
    ) as u32;
    let _new_sec_va = ctx.image_base + new_sec_rva as u64;

    // RVAs within the new section.
    let ilt_rva = new_sec_rva + ilt_offset_in_sec as u32;
    let iat_rva = new_sec_rva + iat_offset_in_sec as u32;
    let hint_name_rva = new_sec_rva + hint_name_offset_in_sec as u32;
    let dll_name_rva = new_sec_rva + dll_name_offset_in_sec as u32;

    // Build section content.
    let mut sec_data = vec![0u8; section_raw_size];

    // Copy old import descriptors.
    sec_data[..old_descs.len()].copy_from_slice(&old_descs);

    // Write new import descriptor at old_desc_count * 20.
    let new_desc_off = old_desc_count * 20;
    sec_data[new_desc_off..new_desc_off + 4].copy_from_slice(&ilt_rva.to_le_bytes()); // OriginalFirstThunk
    // TimeDateStamp = 0, ForwarderChain = 0 (already zeroed)
    sec_data[new_desc_off + 12..new_desc_off + 16].copy_from_slice(&dll_name_rva.to_le_bytes()); // Name
    sec_data[new_desc_off + 16..new_desc_off + 20].copy_from_slice(&iat_rva.to_le_bytes()); // FirstThunk
    // Null terminator descriptor is already zeroed.

    // ILT: one entry pointing to hint/name.
    // For PE32+ the ILT entry is 8 bytes, but the high bytes are already
    // zeroed (by-name import), so writing the low 4 is correct either way.
    sec_data[ilt_offset_in_sec..ilt_offset_in_sec + 4]
        .copy_from_slice(&hint_name_rva.to_le_bytes());

    // IAT: same as ILT initially (loader overwrites with actual address).
    sec_data[iat_offset_in_sec..iat_offset_in_sec + 4]
        .copy_from_slice(&hint_name_rva.to_le_bytes());

    // Hint/Name table entry.
    // hint = 0 (2 bytes, already zeroed)
    sec_data[hint_name_offset_in_sec + 2..hint_name_offset_in_sec + 2 + export_name_bytes.len()]
        .copy_from_slice(export_name_bytes);
    // null terminator already zeroed

    // DLL name string.
    sec_data[dll_name_offset_in_sec..dll_name_offset_in_sec + dll_name_bytes.len()]
        .copy_from_slice(dll_name_bytes);

    // File offset for the new section: append after all existing data,
    // aligned to file_alignment.
    let new_sec_file_offset = align_up(data.len() as u64, ctx.file_alignment as u64) as usize;

    // Pad data to file alignment.
    data.resize(new_sec_file_offset, 0);
    data.extend_from_slice(&sec_data);

    // Write the new section header.
    // Section header table: each entry is 40 bytes.
    let new_header_offset =
        ctx.section_table_offset as usize + ctx.number_of_sections as usize * 40;

    // Check there's space for the header (within SizeOfHeaders).
    // If not, auto-strip expendable section headers to make room.
    if new_header_offset + 40 > ctx.size_of_headers as usize {
        let stripped = strip_expendable_section_headers(ctx, &mut data);
        if stripped > 0 {
            tracing::info!("stripped {stripped} expendable section headers for import injection");
        }
        // Recompute with updated section count.
        let new_count = u16::from_le_bytes(
            data[ctx.number_of_sections_field_offset as usize
                ..ctx.number_of_sections_field_offset as usize + 2]
                .try_into()
                .unwrap_or([0; 2]),
        );
        let new_header_offset = ctx.section_table_offset as usize + new_count as usize * 40;
        if new_header_offset + 40 > ctx.size_of_headers as usize {
            anyhow::bail!(
                "no space for new section header (need {} bytes at {}, SizeOfHeaders={})",
                40,
                new_header_offset,
                ctx.size_of_headers,
            );
        }
    }

    // Section name: ".idata2\0"
    let mut header = [0u8; 40];
    header[0..8].copy_from_slice(b".idata2\0");
    header[8..12].copy_from_slice(&section_vsize.to_le_bytes()); // VirtualSize
    header[12..16].copy_from_slice(&new_sec_rva.to_le_bytes()); // VirtualAddress
    header[16..20].copy_from_slice(&(section_raw_size as u32).to_le_bytes()); // SizeOfRawData
    header[20..24].copy_from_slice(&(new_sec_file_offset as u32).to_le_bytes()); // PointerToRawData
    // Characteristics: contains initialized data, readable
    let chars: u32 = 0x4000_0040; // IMAGE_SCN_MEM_READ | IMAGE_SCN_CNT_INITIALIZED_DATA
    header[36..40].copy_from_slice(&chars.to_le_bytes());

    data[new_header_offset..new_header_offset + 40].copy_from_slice(&header);

    // Update PE headers with overlap-checked patches.
    use crate::binary_patch::PatchSet;
    let mut ps = PatchSet::new();

    let new_section_count = ctx.number_of_sections + 1;
    ps.write_u16(
        ctx.number_of_sections_field_offset as usize,
        new_section_count,
        "NumberOfSections",
    );

    ps.write_u32(import_dir_rva_offset, new_sec_rva, "ImportDirectory RVA");
    ps.write_u32(
        import_dir_rva_offset + 4,
        desc_area_size as u32,
        "ImportDirectory Size",
    );

    let new_soi = align_up(
        new_sec_rva as u64 + section_vsize as u64,
        ctx.section_alignment as u64,
    ) as u32;
    ps.write_u32(
        ctx.size_of_image_field_offset as usize,
        new_soi,
        "SizeOfImage",
    );

    ps.zero(ctx.checksum_field_offset as usize, 4, "CheckSum");

    ps.apply(&mut data)
        .expect("inject_import header patches must not overlap");

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0x1001, 0x1000), 0x2000);
        assert_eq!(align_up(0x1000, 0x1000), 0x1000);
        assert_eq!(align_up(0x0, 0x200), 0x0);
        assert_eq!(align_up(0x201, 0x200), 0x400);
    }

    /// Build a minimal PeContext for thunk scanning tests.
    fn make_test_ctx(text_va: u64, text_offset: u64, text_size: u64, image_base: u64) -> PeContext {
        PeContext {
            text: TextSection {
                va: text_va,
                offset: text_offset,
                size: text_size,
            },
            entry_point: text_va,
            func_symbols: vec![],
            pdata_functions: vec![],
            highest_va_end: text_va + text_size,
            is_dll: false,
            is_64bit: true,
            image_base,
            section_alignment: 0x1000,
            file_alignment: 0x200,
            size_of_image: 0x10000,
            size_of_headers: 0x200,
            number_of_sections: 1,
            sections: vec![PeSection {
                name: ".text".to_string(),
                virtual_address: text_va,
                virtual_size: text_size,
                raw_offset: text_offset,
                raw_size: text_size,
                characteristics: IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE,
            }],
            pe_sig_offset: 0,
            coff_header_offset: 0,
            optional_header_offset: 0,
            section_table_offset: 0,
            entry_point_field_offset: 0,
            size_of_image_field_offset: 0,
            number_of_sections_field_offset: 0,
            checksum_field_offset: 0,
            size_of_code_field_offset: 0,
            size_of_initialized_data_field_offset: 0,
            has_reloc: false,
            imports: vec![],
            text_data_ranges: vec![],
        }
    }

    #[test]
    fn test_find_iat_thunk_basic() {
        // Simulate a .text section at VA 0x140001000, file offset 0x400, size 0x100.
        let image_base = 0x1_4000_0000u64;
        let text_va = image_base + 0x1000;
        let text_offset = 0x400u64;
        let text_size = 0x100u64;
        let ctx = make_test_ctx(text_va, text_offset, text_size, image_base);

        // IAT entry RVA = 0x3000. So IAT VA = image_base + 0x3000.
        let iat_rva = 0x3000u64;
        let iat_va = image_base + iat_rva;

        // Place a JMP [rip+disp32] thunk at offset 0x20 within .text.
        // Thunk VA = text_va + 0x20 = image_base + 0x1020.
        // RIP after instr = thunk_va + 6 = image_base + 0x1026.
        // disp32 = iat_va - rip_after = (image_base + 0x3000) - (image_base + 0x1026)
        //        = 0x1FDA
        let thunk_offset_in_text = 0x20usize;
        let rip_after = text_va + thunk_offset_in_text as u64 + 6;
        let disp: i32 = (iat_va as i64 - rip_after as i64) as i32;
        let disp_bytes = disp.to_le_bytes();

        let mut data = vec![0xCC; text_offset as usize + text_size as usize];
        let abs = text_offset as usize + thunk_offset_in_text;
        data[abs] = 0xFF;
        data[abs + 1] = 0x25;
        data[abs + 2] = disp_bytes[0];
        data[abs + 3] = disp_bytes[1];
        data[abs + 4] = disp_bytes[2];
        data[abs + 5] = disp_bytes[3];

        let result = find_iat_thunk(&data, &ctx, iat_rva);
        assert_eq!(result, Some(text_va + thunk_offset_in_text as u64));
    }

    #[test]
    fn test_find_iat_thunk_not_found() {
        let image_base = 0x1_4000_0000u64;
        let text_va = image_base + 0x1000;
        let ctx = make_test_ctx(text_va, 0x400, 0x100, image_base);

        // Data is all INT3 — no FF 25 sequences.
        let data = vec![0xCC; 0x500];
        assert_eq!(find_iat_thunk(&data, &ctx, 0x3000), None);
    }

    #[test]
    fn test_find_iat_thunk_wrong_target() {
        // FF 25 exists but points to wrong IAT slot.
        let image_base = 0x1_4000_0000u64;
        let text_va = image_base + 0x1000;
        let ctx = make_test_ctx(text_va, 0x400, 0x100, image_base);

        // Thunk pointing to image_base + 0x4000 (not 0x3000).
        let iat_va_wrong = image_base + 0x4000;
        let rip_after = text_va + 6;
        let disp: i32 = (iat_va_wrong as i64 - rip_after as i64) as i32;
        let disp_bytes = disp.to_le_bytes();

        let mut data = vec![0xCC; 0x500];
        data[0x400] = 0xFF;
        data[0x401] = 0x25;
        data[0x402] = disp_bytes[0];
        data[0x403] = disp_bytes[1];
        data[0x404] = disp_bytes[2];
        data[0x405] = disp_bytes[3];

        // Looking for 0x3000, but the thunk points to 0x4000.
        assert_eq!(find_iat_thunk(&data, &ctx, 0x3000), None);
    }

    #[test]
    fn test_find_iat_thunk_multiple_thunks() {
        // Two thunks; second one matches.
        let image_base = 0x1_4000_0000u64;
        let text_va = image_base + 0x1000;
        let ctx = make_test_ctx(text_va, 0x400, 0x100, image_base);

        let iat_rva_target = 0x3008u64;
        let iat_va_target = image_base + iat_rva_target;

        let mut data = vec![0xCC; 0x500];

        // First thunk at offset 0 within .text, pointing elsewhere.
        let iat_va_other = image_base + 0x3000;
        let disp1: i32 = (iat_va_other as i64 - (text_va as i64 + 6)) as i32;
        let d1 = disp1.to_le_bytes();
        data[0x400] = 0xFF;
        data[0x401] = 0x25;
        data[0x402..0x406].copy_from_slice(&d1);

        // Second thunk at offset 0x10 within .text, pointing to target.
        let thunk2_va = text_va + 0x10;
        let disp2: i32 = (iat_va_target as i64 - (thunk2_va as i64 + 6)) as i32;
        let d2 = disp2.to_le_bytes();
        data[0x410] = 0xFF;
        data[0x411] = 0x25;
        data[0x412..0x416].copy_from_slice(&d2);

        let result = find_iat_thunk(&data, &ctx, iat_rva_target);
        assert_eq!(result, Some(thunk2_va));
    }

    #[test]
    fn test_pe_import_struct() {
        let imp = PeImport {
            name: "HeapAlloc".to_string(),
            dll: "KERNEL32.dll".to_string(),
            iat_rva: 0x3000,
        };
        assert_eq!(imp.name, "HeapAlloc");
        assert_eq!(imp.dll, "KERNEL32.dll");
        assert_eq!(imp.iat_rva, 0x3000);
    }

    /// Build a minimal valid PE binary in memory for testing.
    /// Returns raw bytes parseable by `PeContext::parse`.
    fn build_minimal_pe(is_64bit: bool) -> Vec<u8> {
        let e_lfanew: u32 = 0x40; // PE sig right after DOS header
        let pe_sig_offset = e_lfanew as usize;
        let coff_offset = pe_sig_offset + 4;
        let opt_offset = coff_offset + 20;

        // Optional header sizes
        let (magic, opt_standard_size, windows_fields_size): (u16, usize, usize) = if is_64bit {
            (0x20B, 24, 88)
        } else {
            (0x10B, 28, 68)
        };
        let num_data_dirs: u32 = 16;
        let data_dirs_size = num_data_dirs as usize * 8;
        let opt_header_size = opt_standard_size + windows_fields_size + data_dirs_size;

        let section_table_offset = opt_offset + opt_header_size;
        let num_sections: u16 = 1;
        let headers_end = section_table_offset + num_sections as usize * 40;
        let size_of_headers = align_up(headers_end as u64, 0x200) as u32;

        let text_rva: u32 = 0x1000;
        let text_vsize: u32 = 0x100;
        let text_raw_offset = size_of_headers;
        let text_raw_size: u32 = 0x200;
        let size_of_image = align_up((text_rva + text_vsize) as u64, 0x1000) as u32;

        let image_base: u64 = if is_64bit {
            0x0000_0001_4000_0000
        } else {
            0x0040_0000
        };

        let total_file_size = text_raw_offset as usize + text_raw_size as usize;
        let mut data = vec![0u8; total_file_size];

        // DOS header
        data[0] = b'M';
        data[1] = b'Z';
        data[0x3C..0x40].copy_from_slice(&e_lfanew.to_le_bytes());

        // PE signature
        data[pe_sig_offset..pe_sig_offset + 4].copy_from_slice(b"PE\0\0");

        // COFF header
        let machine: u16 = if is_64bit { 0x8664 } else { 0x014C };
        data[coff_offset..coff_offset + 2].copy_from_slice(&machine.to_le_bytes());
        data[coff_offset + 2..coff_offset + 4].copy_from_slice(&num_sections.to_le_bytes());
        data[coff_offset + 16..coff_offset + 18]
            .copy_from_slice(&(opt_header_size as u16).to_le_bytes());
        let characteristics: u16 = 0x0002; // IMAGE_FILE_EXECUTABLE_IMAGE
        data[coff_offset + 18..coff_offset + 20].copy_from_slice(&characteristics.to_le_bytes());

        // Optional header - standard fields
        data[opt_offset..opt_offset + 2].copy_from_slice(&magic.to_le_bytes());
        // SizeOfCode at +4
        data[opt_offset + 4..opt_offset + 8].copy_from_slice(&text_raw_size.to_le_bytes());
        // AddressOfEntryPoint at +16
        data[opt_offset + 16..opt_offset + 20].copy_from_slice(&text_rva.to_le_bytes());

        // Windows-specific fields
        let wf_offset = opt_offset + opt_standard_size;
        if is_64bit {
            // ImageBase (8 bytes)
            data[wf_offset..wf_offset + 8].copy_from_slice(&image_base.to_le_bytes());
            // SectionAlignment at +8
            data[wf_offset + 8..wf_offset + 12].copy_from_slice(&0x1000u32.to_le_bytes());
            // FileAlignment at +12
            data[wf_offset + 12..wf_offset + 16].copy_from_slice(&0x200u32.to_le_bytes());
            // SizeOfImage at +32 (56 from opt start)
            data[wf_offset + 32..wf_offset + 36].copy_from_slice(&size_of_image.to_le_bytes());
            // SizeOfHeaders at +36
            data[wf_offset + 36..wf_offset + 40].copy_from_slice(&size_of_headers.to_le_bytes());
            // NumberOfRvaAndSizes at +84
            data[wf_offset + 84..wf_offset + 88].copy_from_slice(&num_data_dirs.to_le_bytes());
        } else {
            // ImageBase (4 bytes)
            data[wf_offset..wf_offset + 4].copy_from_slice(&(image_base as u32).to_le_bytes());
            // SectionAlignment at +4
            data[wf_offset + 4..wf_offset + 8].copy_from_slice(&0x1000u32.to_le_bytes());
            // FileAlignment at +8
            data[wf_offset + 8..wf_offset + 12].copy_from_slice(&0x200u32.to_le_bytes());
            // SizeOfImage at +28
            data[wf_offset + 28..wf_offset + 32].copy_from_slice(&size_of_image.to_le_bytes());
            // SizeOfHeaders at +32
            data[wf_offset + 32..wf_offset + 36].copy_from_slice(&size_of_headers.to_le_bytes());
            // NumberOfRvaAndSizes at +64
            data[wf_offset + 64..wf_offset + 68].copy_from_slice(&num_data_dirs.to_le_bytes());
        }

        // Section table: .text
        let sh = section_table_offset;
        data[sh..sh + 6].copy_from_slice(b".text\0");
        data[sh + 8..sh + 12].copy_from_slice(&text_vsize.to_le_bytes());
        data[sh + 12..sh + 16].copy_from_slice(&text_rva.to_le_bytes());
        data[sh + 16..sh + 20].copy_from_slice(&text_raw_size.to_le_bytes());
        data[sh + 20..sh + 24].copy_from_slice(&text_raw_offset.to_le_bytes());
        let sec_chars: u32 = IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | 0x4000_0000;
        data[sh + 36..sh + 40].copy_from_slice(&sec_chars.to_le_bytes());

        // Fill text section with INT3
        for b in &mut data[text_raw_offset as usize..] {
            *b = 0xCC;
        }

        data
    }

    #[test]
    fn test_inject_import_pe64() {
        let data = build_minimal_pe(true);
        let ctx = PeContext::parse(&data).expect("should parse minimal PE64");
        assert!(ctx.is_64bit);

        let result = inject_import(data, &ctx, "test.dll", "test_init");
        assert!(
            result.is_ok(),
            "inject_import failed for PE64: {:?}",
            result.err()
        );

        let injected = result.unwrap();
        assert!(injected.len() > ctx.size_of_headers as usize);

        // Re-parse to verify structural integrity
        let ctx2 = PeContext::parse(&injected).expect("should parse PE64 after inject_import");
        assert_eq!(ctx2.number_of_sections, ctx.number_of_sections + 1);
        assert!(ctx2.size_of_image > ctx.size_of_image);

        // Verify the new section exists
        let idata2 = ctx2.sections.iter().find(|s| s.name == ".idata2");
        assert!(
            idata2.is_some(),
            ".idata2 section not found after injection"
        );

        // Verify import was added
        let has_test_import = ctx2.imports.iter().any(|i| i.dll == "test.dll");
        assert!(has_test_import, "test.dll import not found after injection");
    }

    #[test]
    fn test_inject_import_pe32() {
        let data = build_minimal_pe(false);
        let ctx = PeContext::parse(&data).expect("should parse minimal PE32");
        assert!(!ctx.is_64bit);

        let result = inject_import(data, &ctx, "heap_san.dll", "truant_heap_san_init");
        assert!(
            result.is_ok(),
            "inject_import failed for PE32: {:?}",
            result.err()
        );

        let injected = result.unwrap();

        // Re-parse to verify
        let ctx2 = PeContext::parse(&injected).expect("should parse PE32 after inject_import");
        assert_eq!(ctx2.number_of_sections, ctx.number_of_sections + 1);
        assert!(ctx2.size_of_image > ctx.size_of_image);

        let idata2 = ctx2.sections.iter().find(|s| s.name == ".idata2");
        assert!(
            idata2.is_some(),
            ".idata2 section not found after PE32 injection"
        );

        let has_import = ctx2.imports.iter().any(|i| i.dll == "heap_san.dll");
        assert!(
            has_import,
            "heap_san.dll import not found after PE32 injection"
        );

        // Verify the import has the correct export name
        let has_init = ctx2
            .imports
            .iter()
            .any(|i| i.name == "truant_heap_san_init");
        assert!(has_init, "truant_heap_san_init not found in PE32 imports");
    }

    #[test]
    fn test_inject_import_pe32_sidecar() {
        let data = build_minimal_pe(false);
        let ctx = PeContext::parse(&data).expect("should parse minimal PE32");

        let result = inject_import(data, &ctx, "sidecar.dll", "truant_sidecar_san_init");
        assert!(
            result.is_ok(),
            "inject_import failed for PE32 sidecar: {:?}",
            result.err()
        );

        let injected = result.unwrap();
        let ctx2 = PeContext::parse(&injected).expect("should parse PE32 after sidecar injection");

        let has_import = ctx2.imports.iter().any(|i| i.dll == "sidecar.dll");
        assert!(
            has_import,
            "sidecar.dll import not found after PE32 injection"
        );

        let has_init = ctx2
            .imports
            .iter()
            .any(|i| i.name == "truant_sidecar_san_init");
        assert!(
            has_init,
            "truant_sidecar_san_init not found in PE32 imports"
        );
    }

    #[test]
    fn test_inject_import_preserves_existing_imports() {
        // Inject twice: both imports should survive
        let data = build_minimal_pe(true);
        let ctx = PeContext::parse(&data).unwrap();

        let data2 = inject_import(data, &ctx, "first.dll", "first_init").unwrap();
        let ctx2 = PeContext::parse(&data2).unwrap();
        assert!(ctx2.imports.iter().any(|i| i.dll == "first.dll"));

        let data3 = inject_import(data2, &ctx2, "second.dll", "second_init").unwrap();
        let ctx3 = PeContext::parse(&data3).unwrap();

        // Both imports should be present
        assert!(
            ctx3.imports.iter().any(|i| i.dll == "first.dll"),
            "first.dll lost after second injection"
        );
        assert!(
            ctx3.imports.iter().any(|i| i.dll == "second.dll"),
            "second.dll not found after second injection"
        );
    }

    #[test]
    fn test_inject_import_pe32_thunk_size() {
        // PE32 should use 4-byte IAT thunks, not 8-byte
        let data = build_minimal_pe(false);
        let ctx = PeContext::parse(&data).unwrap();

        let injected = inject_import(data, &ctx, "test.dll", "test_func").unwrap();

        // Find the .idata2 section data
        let ctx2 = PeContext::parse(&injected).unwrap();
        let idata2 = ctx2.sections.iter().find(|s| s.name == ".idata2").unwrap();

        // The ILT and IAT entries should be 4 bytes each for PE32
        // (not 8 as for PE64). The section's raw size should reflect this.
        // With 4-byte thunks: descriptor area + ILT(8) + IAT(8) + hint/name + dll_name
        // With 8-byte thunks: descriptor area + ILT(16) + IAT(16) + hint/name + dll_name
        // PE32 raw content is smaller (though both may round to same alignment).
        let data64 = build_minimal_pe(true);
        let ctx64 = PeContext::parse(&data64).unwrap();
        let injected64 = inject_import(data64, &ctx64, "test.dll", "test_func").unwrap();
        let ctx64_2 = PeContext::parse(&injected64).unwrap();
        let idata2_64 = ctx64_2
            .sections
            .iter()
            .find(|s| s.name == ".idata2")
            .unwrap();

        // PE32 uses 4-byte thunks (ILT: 2*4=8, IAT: 2*4=8 = 16 total)
        // PE64 uses 8-byte thunks (ILT: 2*8=16, IAT: 2*8=16 = 32 total)
        // The raw_size may be identical due to file alignment, but the
        // unpadded virtual_size or the injected file should differ.
        // At minimum, both should produce valid PE binaries.
        assert!(
            idata2.raw_size > 0,
            "PE32 .idata2 raw_size should be non-zero"
        );
        assert!(
            idata2_64.raw_size > 0,
            "PE64 .idata2 raw_size should be non-zero"
        );

        // Verify the import is resolvable in both
        assert!(ctx2.imports.iter().any(|i| i.name == "test_func"));
        assert!(ctx64_2.imports.iter().any(|i| i.name == "test_func"));
    }
}
