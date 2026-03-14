use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use goblin::elf::reloc::reloc64;
use goblin::elf::{Elf, program_header, section_header};

/// Information about the .text section.
#[derive(Debug, Clone)]
pub struct TextSection {
    /// Virtual address of .text
    pub va: u64,
    /// File offset of .text
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
}

/// Information about a PT_NOTE segment that can be converted to PT_LOAD.
#[derive(Debug, Clone)]
pub struct NoteSegment {
    /// Index into the program header table
    pub phdr_index: usize,
    /// File offset of the program header entry
    pub phdr_offset: u64,
}

/// Range of addresses to exclude from instrumentation (PLT stubs).
#[derive(Debug, Clone)]
pub struct PltRange {
    pub va: u64,
    pub size: u64,
    pub offset: u64,
}

/// How an allocator function will be intercepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocEntryKind {
    /// Dynamic binary: overwrite .got.plt 8-byte pointer.
    GotPlt,
    /// Static binary: overwrite first 5 bytes of function with JMP rel32.
    FuncEntry,
}

/// A single allocator function discovered in the binary.
#[derive(Debug, Clone)]
pub struct AllocEntry {
    /// File offset to patch (GOT entry for GotPlt, function start for FuncEntry).
    pub patch_offset: u64,
    /// Virtual address of the patch site (for rel32 calculation with FuncEntry).
    pub patch_va: u64,
    /// How to intercept this function.
    pub kind: AllocEntryKind,
}

/// Discovered allocator symbols for heap sanitiser interception.
#[derive(Debug, Clone, Default)]
pub struct AllocatorSymbols {
    pub entries: BTreeMap<&'static str, AllocEntry>,
}

impl AllocatorSymbols {
    /// Number of allocator functions found.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Look up an allocator entry by name.
    pub fn get(&self, name: &str) -> Option<&AllocEntry> {
        self.entries.get(name)
    }
}

/// Extracted ELF context needed for rewriting.
#[derive(Debug)]
pub struct ElfContext {
    pub text: TextSection,
    pub entry_point: u64,
    pub note_segment: Option<NoteSegment>,
    /// Function symbol addresses (hints for block starts)
    pub func_symbols: Vec<u64>,
    /// Highest VA + memsz across all PT_LOAD segments
    pub highest_va_end: u64,
    /// Whether this is a shared object (.so / PIE)
    pub is_shared_object: bool,
    /// Whether the binary is dynamically linked (has PT_INTERP)
    pub is_dynamic: bool,
    /// DT_INIT value for .so files (virtual address of init function)
    pub dt_init: Option<u64>,
    /// File offset of the DT_INIT entry in .dynamic (for patching d_val)
    pub dt_init_offset: Option<u64>,
    /// File offset of the first DT_NULL entry in .dynamic (for synthesising DT_INIT)
    pub first_dt_null_offset: Option<u64>,
    /// Number of DT_NULL entries in .dynamic (need >= 2 to repurpose one)
    pub dt_null_count: usize,
    /// File offset and size of the .dynamic section (for relocation when DT_NULL count < 2)
    pub dynamic_file_offset: Option<u64>,
    pub dynamic_size: Option<u64>,
    /// File offset of the PT_DYNAMIC program header entry (for updating after relocation)
    pub pt_dynamic_phdr_offset: Option<u64>,
    /// PLT section ranges to skip during instrumentation
    pub plt_ranges: Vec<PltRange>,
    /// Total size of the program header table entry
    pub phdr_entry_size: u64,
    /// File offset of the ELF header's e_entry field
    pub e_entry_offset: u64,
    /// Whether ELF is 64-bit (we only support 64-bit)
    pub is_64bit: bool,
    /// ELF e_machine value (e.g. EM_X86_64=62, EM_AARCH64=183)
    pub e_machine: u16,
    /// File offset of the program header table (e_phoff)
    pub e_phoff: u64,
    /// Number of program header entries (e_phnum)
    pub e_phnum: u16,
}

impl ElfContext {
    /// Parse an ELF binary and extract all information needed for rewriting.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let elf = Elf::parse(data).context("failed to parse ELF")?;

        if !elf.is_64 {
            bail!("only 64-bit ELF binaries are supported");
        }

        // Find .text section
        let text = find_text_section(&elf).context("no .text section found")?;

        // Find PT_NOTE segment (candidate for conversion)
        let note_segment = find_note_segment(&elf);

        // Collect function symbols
        let func_symbols = collect_func_symbols(&elf);

        // Find highest VA end across all PT_LOAD segments
        let highest_va_end = elf
            .program_headers
            .iter()
            .filter(|ph| ph.p_type == program_header::PT_LOAD)
            .map(|ph| ph.p_vaddr + ph.p_memsz)
            .max()
            .unwrap_or(0);

        // PIE executables have e_type=ET_DYN but also have PT_INTERP.
        // True shared libraries (.so) have ET_DYN but no PT_INTERP.
        // Static-PIE executables also have ET_DYN and no PT_INTERP, but they
        // are executables (not libraries) — we detect them by checking for a
        // non-zero entry point within the text section.  For these binaries we
        // must patch e_entry (not DT_INIT) because the static CRT never
        // invokes DT_INIT.
        //
        // Android NDK .so files complicate this: they have ET_DYN, no PT_INTERP,
        // AND a non-zero e_entry pointing into .text. The distinguishing marker
        // is DT_SONAME — shared libraries have it, static-PIE executables don't.
        let has_interp = elf
            .program_headers
            .iter()
            .any(|ph| ph.p_type == program_header::PT_INTERP);
        let has_soname = elf
            .dynamic
            .as_ref()
            .map(|d| {
                d.dyns
                    .iter()
                    .any(|e| e.d_tag == goblin::elf::dynamic::DT_SONAME)
            })
            .unwrap_or(false);
        let is_et_dyn_no_interp = elf.header.e_type == goblin::elf::header::ET_DYN && !has_interp;
        let is_static_pie = is_et_dyn_no_interp
            && !has_soname
            && elf.header.e_entry != 0
            && elf.header.e_entry >= text.va
            && elf.header.e_entry < text.va + text.size;
        let is_shared_object = is_et_dyn_no_interp && !is_static_pie;

        // Extract DT_INIT, DT_NULL info for .so files
        let (dt_init, dt_init_offset, first_dt_null_offset, dt_null_count) = if is_shared_object {
            find_dt_init(&elf, data)
        } else {
            (None, None, None, 0usize)
        };

        // Extract .dynamic section location and PT_DYNAMIC phdr offset for .so relocation
        let (dynamic_file_offset, dynamic_size, pt_dynamic_phdr_offset) = if is_shared_object {
            let (doff, dsz) = find_dynamic_section_offset_and_size(&elf);
            let pt_dyn_off = elf
                .program_headers
                .iter()
                .enumerate()
                .find(|(_, ph)| ph.p_type == program_header::PT_DYNAMIC)
                .map(|(i, _)| elf.header.e_phoff + (i as u64) * (elf.header.e_phentsize as u64));
            (doff, dsz, pt_dyn_off)
        } else {
            (None, None, None)
        };

        // Find PLT ranges
        let plt_ranges = find_plt_ranges(&elf);

        // Program header entry size
        let phdr_entry_size = elf.header.e_phentsize as u64;

        // e_entry is at offset 24 in ELF64 header
        let e_entry_offset = 24;

        Ok(ElfContext {
            text,
            entry_point: elf.header.e_entry,
            note_segment,
            func_symbols,
            highest_va_end,
            is_shared_object,
            is_dynamic: has_interp,
            dt_init,
            dt_init_offset,
            first_dt_null_offset,
            dt_null_count,
            dynamic_file_offset,
            dynamic_size,
            pt_dynamic_phdr_offset,
            plt_ranges,
            phdr_entry_size,
            e_entry_offset,
            is_64bit: true,
            e_machine: elf.header.e_machine,
            e_phoff: elf.header.e_phoff,
            e_phnum: elf.header.e_phnum,
        })
    }

    /// Look up the executable section containing `va`.
    ///
    /// Returns `(section_va, section_file_offset, section_size)` if the VA falls
    /// within `.text` or any PLT range (`.plt`, `.plt.sec`, `.plt.got`).
    pub fn exec_section_for_va(&self, va: u64) -> Option<(u64, u64, u64)> {
        // Check .text first (most common case).
        let t = &self.text;
        if va >= t.va && va < t.va + t.size {
            return Some((t.va, t.offset, t.size));
        }
        // Check PLT ranges.
        for plt in &self.plt_ranges {
            if va >= plt.va && va < plt.va + plt.size {
                return Some((plt.va, plt.offset, plt.size));
            }
        }
        None
    }
}

fn find_text_section(elf: &Elf) -> Option<TextSection> {
    for sh in &elf.section_headers {
        let name = elf.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
        if name == ".text" && sh.sh_type == section_header::SHT_PROGBITS {
            return Some(TextSection {
                va: sh.sh_addr,
                offset: sh.sh_offset,
                size: sh.sh_size,
            });
        }
    }
    None
}

fn find_note_segment(elf: &Elf) -> Option<NoteSegment> {
    for (i, ph) in elf.program_headers.iter().enumerate() {
        if ph.p_type == program_header::PT_NOTE {
            // Calculate the file offset of this program header entry.
            let phdr_offset = elf.header.e_phoff + (i as u64) * (elf.header.e_phentsize as u64);
            return Some(NoteSegment {
                phdr_index: i,
                phdr_offset,
            });
        }
    }
    None
}

fn collect_func_symbols(elf: &Elf) -> Vec<u64> {
    let mut addrs = Vec::new();

    for sym in &elf.syms {
        if sym.st_type() == goblin::elf::sym::STT_FUNC && sym.st_value != 0 {
            addrs.push(sym.st_value);
        }
    }
    for sym in &elf.dynsyms {
        if sym.st_type() == goblin::elf::sym::STT_FUNC && sym.st_value != 0 {
            addrs.push(sym.st_value);
        }
    }

    addrs.sort_unstable();
    addrs.dedup();
    addrs
}

/// Scan .dynamic for DT_INIT and DT_NULL entries.
///
/// goblin's parser stops at the first DT_NULL, so it misses padding NULLs
/// left by strip_verneed or similar tools. We scan the raw bytes of the
/// .dynamic section to find ALL DT_NULL entries.
///
/// Returns (dt_init_va, dt_init_offset, first_dt_null_offset, dt_null_count).
fn find_dt_init(elf: &Elf, data: &[u8]) -> (Option<u64>, Option<u64>, Option<u64>, usize) {
    let mut dt_init: Option<u64> = None;
    let mut dt_init_offset: Option<u64> = None;
    let mut first_dt_null_offset: Option<u64> = None;
    let mut dt_null_count: usize = 0;

    // Use goblin for DT_INIT (it parses entries until DT_NULL).
    if let Some(ref dynamic) = elf.dynamic {
        let dynamic_offset = find_dynamic_section_offset(elf);
        for (i, dyn_entry) in dynamic.dyns.iter().enumerate() {
            if dyn_entry.d_tag == goblin::elf::dynamic::DT_INIT && dt_init.is_none() {
                dt_init = Some(dyn_entry.d_val);
                dt_init_offset = dynamic_offset.map(|off| off + (i as u64) * 16 + 8);
            }
        }
    }

    // Scan raw .dynamic bytes for ALL DT_NULL entries (including padding
    // after the first terminator that goblin doesn't report).
    let (dyn_offset, dyn_size) = find_dynamic_section_offset_and_size(elf);
    if let (Some(off), Some(sz)) = (dyn_offset, dyn_size) {
        let off = off as usize;
        let sz = sz as usize;
        let end = (off + sz).min(data.len());
        let mut pos = off;
        while pos + 16 <= end {
            let tag = i64::from_le_bytes(data[pos..pos + 8].try_into().expect("slice is 8 bytes"));
            if tag == goblin::elf::dynamic::DT_NULL as i64 {
                dt_null_count += 1;
                if first_dt_null_offset.is_none() {
                    first_dt_null_offset = Some(pos as u64);
                }
            }
            pos += 16;
        }
    }

    (dt_init, dt_init_offset, first_dt_null_offset, dt_null_count)
}

fn find_dynamic_section_offset(elf: &Elf) -> Option<u64> {
    for sh in &elf.section_headers {
        if sh.sh_type == section_header::SHT_DYNAMIC {
            return Some(sh.sh_offset);
        }
    }
    // Fallback: use PT_DYNAMIC program header
    for ph in &elf.program_headers {
        if ph.p_type == program_header::PT_DYNAMIC {
            return Some(ph.p_offset);
        }
    }
    None
}

/// Find the file offset AND size of the .dynamic section.
fn find_dynamic_section_offset_and_size(elf: &Elf) -> (Option<u64>, Option<u64>) {
    for sh in &elf.section_headers {
        if sh.sh_type == section_header::SHT_DYNAMIC {
            return (Some(sh.sh_offset), Some(sh.sh_size));
        }
    }
    for ph in &elf.program_headers {
        if ph.p_type == program_header::PT_DYNAMIC {
            return (Some(ph.p_offset), Some(ph.p_filesz));
        }
    }
    (None, None)
}

/// All allocator function names we intercept.
/// For the dynamic path these are the PLT symbol names.
/// For the static/stripped path these are matched against symtab/assertion strings.
const ALLOC_TARGETS: &[&str] = &[
    "malloc",
    "free",
    "calloc",
    "realloc",
    "memalign",
    "aligned_alloc",
    "posix_memalign",
    // jemalloc extended API — Rust's #[global_allocator] with tikv-jemallocator uses these
    // instead of standard free/realloc, so we must intercept them too.
    "mallocx",
    "sdallocx",
    "rallocx",
];

/// Check if a symbol name is one of our allocator targets, returning the canonical
/// target name (which is used as the key in `AllocatorSymbols::entries`).
fn is_alloc_target(name: &str) -> Option<&'static str> {
    ALLOC_TARGETS.iter().find(|&&t| t == name).copied()
}

/// Known jemalloc prefixes. When jemalloc is statically linked with a prefix
/// (e.g. `tikv-jemallocator` uses `_rjem_`), we strip the prefix and match
/// against `ALLOC_TARGETS`.
const JEMALLOC_PREFIXES: &[&str] = &["je_", "_rjem_"];

/// Try to match a symbol name as a prefixed jemalloc allocator function.
/// Returns the canonical target name (e.g. `"_rjem_malloc"` → `"malloc"`).
fn match_jemalloc_alias(name: &str) -> Option<&'static str> {
    for prefix in JEMALLOC_PREFIXES {
        if let Some(stripped) = name.strip_prefix(prefix) {
            return is_alloc_target(stripped);
        }
    }
    None
}

/// Discover allocator functions for heap sanitiser interception.
///
/// For dynamic binaries: scans `.rela.plt` for R_X86_64_JUMP_SLOT relocations pointing to
/// allocator symbols, recording the GOT entry VA and file offset.
///
/// For static binaries: scans `.symtab` for function symbols named malloc/free/calloc/realloc/memalign.
pub fn find_allocator_symbols(ctx: &ElfContext, data: &[u8]) -> AllocatorSymbols {
    let elf = match Elf::parse(data) {
        Ok(e) => e,
        Err(_) => return AllocatorSymbols::default(),
    };

    let mut syms = AllocatorSymbols::default();

    // Try dynamic path first (has .rela.plt + .dynsym).
    if !elf.pltrelocs.is_empty() {
        // We need VA → file offset mapping from PT_LOAD segments.
        let loads: Vec<_> = elf
            .program_headers
            .iter()
            .filter(|ph| ph.p_type == program_header::PT_LOAD)
            .collect();

        for reloc in elf.pltrelocs.iter() {
            if reloc.r_type != reloc64::R_X86_64_JUMP_SLOT {
                continue;
            }
            let sym = match elf.dynsyms.get(reloc.r_sym) {
                Some(s) => s,
                None => continue,
            };
            let name = match elf.dynstrtab.get_at(sym.st_name) {
                Some(n) => n,
                None => continue,
            };

            let target_name = match is_alloc_target(name).or_else(|| match_jemalloc_alias(name)) {
                Some(t) => t,
                None => continue,
            };

            // reloc.r_offset is the GOT entry VA. Convert to file offset.
            let got_va = reloc.r_offset;
            if let Some(file_off) = va_to_file_offset(&loads, got_va) {
                syms.entries.entry(target_name).or_insert(AllocEntry {
                    patch_offset: file_off,
                    patch_va: got_va,
                    kind: AllocEntryKind::GotPlt,
                });
            }
        }
    }

    // Fallback: static binary (no PLT relocs). Scan .symtab.
    if syms.count() == 0 {
        let loads: Vec<_> = elf
            .program_headers
            .iter()
            .filter(|ph| ph.p_type == program_header::PT_LOAD)
            .collect();

        for sym in &elf.syms {
            if sym.st_type() != goblin::elf::sym::STT_FUNC || sym.st_value == 0 {
                continue;
            }
            let name = match elf.strtab.get_at(sym.st_name) {
                Some(n) => n,
                None => continue,
            };

            // Match direct names, glibc internal aliases, and jemalloc prefixes.
            let target_name = match is_alloc_target(name) {
                Some(t) => t,
                None if name == "__libc_memalign" => "memalign",
                None => match match_jemalloc_alias(name) {
                    Some(t) => t,
                    None => continue,
                },
            };

            if let Some(file_off) = va_to_file_offset(&loads, sym.st_value) {
                syms.entries.entry(target_name).or_insert(AllocEntry {
                    patch_offset: file_off,
                    patch_va: sym.st_value,
                    kind: AllocEntryKind::FuncEntry,
                });
            }
        }
    }

    // Third fallback: string xref scan for stripped static binaries (glibc).
    // glibc embeds function-name strings ("__libc_malloc", etc.) for assertion
    // messages that survive stripping. Cross-reference LEA instructions in .text
    // against these strings to locate allocator entry points.
    if syms.count() == 0 {
        let loads: Vec<_> = elf
            .program_headers
            .iter()
            .filter(|ph| ph.p_type == program_header::PT_LOAD)
            .collect();
        syms = scan_for_allocators(data, &ctx.text, &loads);
    }

    // Fourth fallback: musl allocator detection for stripped static binaries.
    // musl's __libc_malloc_impl starts with `movabs rax, <near-PTRDIFF_MAX>` —
    // a pre-computed max allocation size constant that survives stripping. We use
    // this as an anchor, then locate allocator stubs via JMP-target analysis.
    if syms.count() == 0 {
        let loads: Vec<_> = elf
            .program_headers
            .iter()
            .filter(|ph| ph.p_type == program_header::PT_LOAD)
            .collect();
        syms = scan_for_musl_allocators(data, &ctx.text, &loads);
    }

    syms
}

/// Convert a virtual address to a file offset using PT_LOAD segment mappings.
fn va_to_file_offset(loads: &[&goblin::elf::ProgramHeader], va: u64) -> Option<u64> {
    for ph in loads {
        if va >= ph.p_vaddr && va < ph.p_vaddr + ph.p_filesz {
            return Some(ph.p_offset + (va - ph.p_vaddr));
        }
    }
    None
}

fn find_plt_ranges(elf: &Elf) -> Vec<PltRange> {
    let mut ranges = Vec::new();
    for sh in &elf.section_headers {
        let name = elf.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
        if name == ".plt" || name == ".plt.sec" || name == ".plt.got" {
            ranges.push(PltRange {
                va: sh.sh_addr,
                size: sh.sh_size,
                offset: sh.sh_offset,
            });
        }
    }
    ranges
}

/// Find all occurrences of `needle` in `haystack`, returning file offsets.
fn find_byte_pattern(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    let mut results = Vec::new();
    if needle.is_empty() || haystack.len() < needle.len() {
        return results;
    }
    let first = needle[0];
    let end = haystack.len() - needle.len();
    let mut i = 0;
    while i <= end {
        if haystack[i] == first && haystack[i..i + needle.len()] == *needle {
            results.push(i);
            i += needle.len();
        } else {
            i += 1;
        }
    }
    results
}

/// Convert a file offset to a virtual address using PT_LOAD segment mappings.
fn file_offset_to_va(loads: &[&goblin::elf::ProgramHeader], file_offset: u64) -> Option<u64> {
    for ph in loads {
        if file_offset >= ph.p_offset && file_offset < ph.p_offset + ph.p_filesz {
            return Some(ph.p_vaddr + (file_offset - ph.p_offset));
        }
    }
    None
}

/// Scan a stripped static binary for glibc allocator functions by cross-referencing
/// known assertion strings in .rodata against LEA instructions in .text.
///
/// glibc's malloc.c embeds function-name strings (e.g. `"__libc_malloc"`) for
/// assertion messages. These survive stripping. We find LEA instructions that load
/// these string addresses, then walk backward to locate function entry points.
fn scan_for_allocators(
    data: &[u8],
    text: &TextSection,
    loads: &[&goblin::elf::ProgramHeader],
) -> AllocatorSymbols {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic, OpKind, Register};

    // Primary anchors: glibc assertion-message function names.
    // _int_free is the inner free implementation with a DIFFERENT calling convention
    // from free(void*) — it takes (mstate, mchunkptr, int). We use it only to
    // locate __libc_free via caller analysis, never as a direct replacement.
    // _mid_memalign is the inner memalign implementation; __libc_memalign is a thin
    // wrapper around it that we locate via caller analysis.
    const ANCHORS: &[(&[u8], &str)] = &[
        (b"__libc_malloc\0", "malloc"),
        (b"__libc_free\0", "free"),
        (b"_int_free\0", "free_inner"),
        (b"__libc_calloc\0", "calloc"),
        (b"__libc_realloc\0", "realloc"),
        (b"_mid_memalign\0", "memalign_inner"),
    ];

    // Step 1: Find anchor string VAs.
    let mut string_vas: Vec<(u64, &str)> = Vec::new();
    for &(pattern, name) in ANCHORS {
        for file_offset in find_byte_pattern(data, pattern) {
            if let Some(va) = file_offset_to_va(loads, file_offset as u64) {
                string_vas.push((va, name));
            }
        }
    }

    if string_vas.is_empty() {
        tracing::debug!("string xref scan: no glibc assertion strings found");
        return AllocatorSymbols::default();
    }

    tracing::debug!(
        "string xref scan: found {} anchor strings",
        string_vas.len()
    );

    // Step 2: Decode .text, find LEA rip-relative instructions referencing string VAs.
    let text_start = text.offset as usize;
    let text_end = text_start + text.size as usize;
    if text_end > data.len() {
        return AllocatorSymbols::default();
    }
    let text_bytes = &data[text_start..text_end];

    let string_va_map: std::collections::HashMap<u64, &str> = string_vas.iter().copied().collect();

    let mut xrefs: Vec<(u64, &str)> = Vec::new();
    let mut decoder = Decoder::with_ip(64, text_bytes, text.va, DecoderOptions::NONE);

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.is_invalid() {
            continue;
        }

        if instr.mnemonic() != Mnemonic::Lea || instr.op_count() < 2 {
            continue;
        }

        if instr.op1_kind() == OpKind::Memory && instr.memory_base() == Register::RIP {
            // For RIP-relative, memory_displacement64() returns the resolved absolute
            // address when the decoder is initialised with an IP.
            let target_va = instr.memory_displacement64();
            if let Some(&name) = string_va_map.get(&target_va) {
                tracing::debug!(
                    "string xref scan: LEA at 0x{:x} references {} string at 0x{:x}",
                    instr.ip(),
                    name,
                    target_va
                );
                xrefs.push((instr.ip(), name));
            }
        }
    }

    if xrefs.is_empty() {
        tracing::debug!("string xref scan: no LEA xrefs found");
        return AllocatorSymbols::default();
    }

    // Step 3: Walk backward from each xref to find function prologue.
    let mut syms = AllocatorSymbols::default();
    let mut free_inner_va: Option<u64> = None;
    let mut memalign_inner_va: Option<u64> = None;

    for &(xref_va, name) in &xrefs {
        // Inner helper functions — record for caller analysis, don't use directly.
        if name == "free_inner" {
            if free_inner_va.is_none()
                && let Some(func_va) = find_function_start_backward(text_bytes, text.va, xref_va)
            {
                tracing::debug!("string xref scan: _int_free (inner) at VA 0x{:x}", func_va);
                free_inner_va = Some(func_va);
            }
            continue;
        }
        if name == "memalign_inner" {
            if memalign_inner_va.is_none()
                && let Some(func_va) = find_function_start_backward(text_bytes, text.va, xref_va)
            {
                tracing::debug!(
                    "string xref scan: _mid_memalign (inner) at VA 0x{:x}",
                    func_va
                );
                memalign_inner_va = Some(func_va);
            }
            continue;
        }

        let target_name = match is_alloc_target(name) {
            Some(t) => t,
            None => continue,
        };

        if syms.entries.contains_key(target_name) {
            continue; // Already found this allocator.
        }

        if let Some(func_va) = find_function_start_backward(text_bytes, text.va, xref_va)
            && let Some(file_off) = va_to_file_offset(loads, func_va)
        {
            tracing::debug!(
                "string xref scan: {} at VA 0x{:x} (file offset 0x{:x})",
                name,
                func_va,
                file_off
            );
            syms.entries.insert(
                target_name,
                AllocEntry {
                    patch_offset: file_off,
                    patch_va: func_va,
                    kind: AllocEntryKind::FuncEntry,
                },
            );
        }
    }

    // Step 4: Caller analysis for inner functions.
    //
    // If free wasn't found via __libc_free assertion string but we located _int_free,
    // find __libc_free by searching for callers that match the free(void*) pattern.
    if !syms.entries.contains_key("free")
        && let Some(inner_va) = free_inner_va
        && let Some(entry) = find_free_via_caller(data, text, loads, inner_va)
    {
        syms.entries.insert("free", entry);
    }

    // If memalign wasn't found via direct assertion but we located _mid_memalign,
    // find __libc_memalign by searching for callers of _mid_memalign.
    // Fallback: if caller analysis finds nothing (common when _mid_memalign is static
    // and inlined into __libc_memalign), use _mid_memalign itself — patching at its
    // entry catches all aligned allocation paths (memalign, aligned_alloc, valloc,
    // pvalloc, posix_memalign). The extra `address` arg in rdx is harmlessly ignored.
    if !syms.entries.contains_key("memalign")
        && let Some(inner_va) = memalign_inner_va
    {
        if let Some(entry) = find_memalign_via_caller(data, text, loads, inner_va) {
            syms.entries.insert("memalign", entry);
        } else if let Some(file_off) = va_to_file_offset(loads, inner_va) {
            tracing::debug!(
                "string xref scan: using _mid_memalign directly at VA 0x{:x} (inlined into __libc_memalign)",
                inner_va
            );
            syms.entries.insert(
                "memalign",
                AllocEntry {
                    patch_offset: file_off,
                    patch_va: inner_va,
                    kind: AllocEntryKind::FuncEntry,
                },
            );
        }
    }

    let found = syms.count();
    if found > 0 {
        tracing::info!(
            "string xref scan: discovered {}/{} allocator functions in stripped binary",
            found,
            ALLOC_TARGETS.len()
        );
    }

    syms
}

/// Find `__libc_free` by searching for callers of `_int_free` that start with
/// `test rdi, rdi` — the characteristic NULL-pointer check of `free(void *)`.
///
/// `_int_free` has the signature `_int_free(mstate, mchunkptr, int)`, so it cannot
/// be used as a direct replacement for `free`. But `__libc_free` is a thin wrapper
/// that calls `_int_free`, so we locate it via caller analysis.
fn find_free_via_caller(
    data: &[u8],
    text: &TextSection,
    loads: &[&goblin::elf::ProgramHeader],
    int_free_va: u64,
) -> Option<AllocEntry> {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic, Register};

    let text_start = text.offset as usize;
    let text_end = text_start + text.size as usize;
    if text_end > data.len() {
        return None;
    }
    let text_bytes = &data[text_start..text_end];

    // Scan .text for CALL/JMP rel32 instructions targeting _int_free.
    // __libc_free often tail-calls _int_free with JMP, so we must check both.
    let mut callers: Vec<u64> = Vec::new();
    let mut decoder = Decoder::with_ip(64, text_bytes, text.va, DecoderOptions::NONE);

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.is_invalid() {
            continue;
        }
        let is_call = instr.mnemonic() == Mnemonic::Call;
        let is_jmp = instr.mnemonic() == Mnemonic::Jmp;
        if (is_call || is_jmp) && instr.near_branch_target() == int_free_va {
            callers.push(instr.ip());
        }
    }

    if callers.is_empty() {
        tracing::debug!("find_free_via_caller: no CALL/JMP instructions targeting _int_free");
        return None;
    }

    tracing::debug!(
        "find_free_via_caller: found {} callers of _int_free",
        callers.len()
    );

    // For each caller, find the containing function and check if it matches
    // the __libc_free pattern. We look for two patterns (best match wins):
    //
    //  Pattern A (newer glibc): endbr64?; sub rsp, imm; mov rax, [rip+disp]
    //    (__free_hook check, then test rdi,rdi, then mov rax,[rdi-8])
    //
    //  Pattern B (older glibc): endbr64?; test rdi, rdi
    //    (NULL check first, no hook check)
    //
    // Both patterns distinguish __libc_free from internal helpers.
    // Pattern A is preferred because it's more specific — `mov [rdi-8]` is
    // the glibc chunk header read that only __libc_free does at function scope.

    let mut best_a: Option<(u64, u64)> = None; // (func_va, caller_va) for pattern A
    let mut best_b: Option<(u64, u64)> = None; // (func_va, caller_va) for pattern B

    for &caller_va in &callers {
        let func_va = match find_function_start_backward(text_bytes, text.va, caller_va) {
            Some(va) => va,
            None => continue,
        };

        // Skip if the function start IS _int_free itself (internal JMPs).
        if func_va == int_free_va {
            continue;
        }

        // Decode first ~10 instructions of the candidate function.
        let func_off = (func_va - text.va) as usize;
        let end = text_bytes.len().min(func_off + 64);
        let slice = &text_bytes[func_off..end];
        let mut dec = Decoder::with_ip(64, slice, func_va, DecoderOptions::NONE);

        let mut instrs = Vec::new();
        while dec.can_decode() && instrs.len() < 12 {
            let i = dec.decode();
            if i.is_invalid() {
                break;
            }
            instrs.push(i);
        }

        if instrs.is_empty() {
            continue;
        }

        // Skip optional endbr64 prefix.
        let start_idx = if instrs[0].mnemonic() == Mnemonic::Endbr64
            || instrs[0].mnemonic() == Mnemonic::Endbr32
        {
            1
        } else {
            0
        };

        // Pattern A: look for `mov rax, [rdi-8]` (chunk header read) in first ~10 instrs.
        // This is the definitive __libc_free signature — it reads the chunk size/flags
        // from the metadata word preceding the user pointer.
        let has_chunk_read = instrs[start_idx..].iter().any(|i| {
            i.mnemonic() == Mnemonic::Mov
                && i.op0_register() == Register::RAX
                && i.memory_base() == Register::RDI
                && i.memory_displacement64() == (-8i64 as u64)
        });

        if has_chunk_read && best_a.is_none() {
            best_a = Some((func_va, caller_va));
            continue;
        }

        // Pattern B: `test rdi, rdi` as (nearly) first instruction.
        if start_idx < instrs.len() {
            let check = &instrs[start_idx];
            if check.mnemonic() == Mnemonic::Test
                && check.op0_register() == Register::RDI
                && check.op1_register() == Register::RDI
                && best_b.is_none()
            {
                best_b = Some((func_va, caller_va));
            }
        }
    }

    // Prefer pattern A (chunk header read) over pattern B (bare NULL check).
    let (func_va, caller_va) = match best_a.or(best_b) {
        Some(pair) => pair,
        None => {
            tracing::debug!("find_free_via_caller: no caller matches free(void*) pattern");
            return None;
        }
    };

    if let Some(file_off) = va_to_file_offset(loads, func_va) {
        tracing::debug!(
            "find_free_via_caller: __libc_free at VA 0x{:x} (caller of _int_free at 0x{:x})",
            func_va,
            caller_va
        );
        return Some(AllocEntry {
            patch_offset: file_off,
            patch_va: func_va,
            kind: AllocEntryKind::FuncEntry,
        });
    }

    tracing::debug!("find_free_via_caller: no caller matches free(void*) pattern");
    None
}

/// Find `__libc_memalign` by searching for callers of `_mid_memalign`.
///
/// `_mid_memalign` is the inner implementation; `__libc_memalign` is the only
/// public-facing caller. We locate it by finding CALL/JMP instructions that
/// target `_mid_memalign` and returning the first caller whose function start
/// differs from `_mid_memalign` itself.
fn find_memalign_via_caller(
    data: &[u8],
    text: &TextSection,
    loads: &[&goblin::elf::ProgramHeader],
    mid_memalign_va: u64,
) -> Option<AllocEntry> {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};

    let text_start = text.offset as usize;
    let text_end = text_start + text.size as usize;
    if text_end > data.len() {
        return None;
    }
    let text_bytes = &data[text_start..text_end];

    // Scan .text for CALL/JMP rel32 instructions targeting _mid_memalign.
    let mut callers: Vec<u64> = Vec::new();
    let mut decoder = Decoder::with_ip(64, text_bytes, text.va, DecoderOptions::NONE);

    while decoder.can_decode() {
        let instr = decoder.decode();
        if instr.is_invalid() {
            continue;
        }
        let is_call = instr.mnemonic() == Mnemonic::Call;
        let is_jmp = instr.mnemonic() == Mnemonic::Jmp;
        if (is_call || is_jmp) && instr.near_branch_target() == mid_memalign_va {
            callers.push(instr.ip());
        }
    }

    if callers.is_empty() {
        tracing::debug!("find_memalign_via_caller: no CALL/JMP targeting _mid_memalign");
        return None;
    }

    tracing::debug!(
        "find_memalign_via_caller: found {} callers of _mid_memalign",
        callers.len()
    );

    // __libc_memalign is the only public-facing caller of _mid_memalign.
    // Find the first caller whose function start differs from _mid_memalign itself.
    for &caller_va in &callers {
        let func_va = match find_function_start_backward(text_bytes, text.va, caller_va) {
            Some(va) => va,
            None => continue,
        };

        // Skip if the function start IS _mid_memalign itself (internal calls).
        if func_va == mid_memalign_va {
            continue;
        }

        if let Some(file_off) = va_to_file_offset(loads, func_va) {
            tracing::debug!(
                "find_memalign_via_caller: __libc_memalign at VA 0x{:x} (caller at 0x{:x})",
                func_va,
                caller_va
            );
            return Some(AllocEntry {
                patch_offset: file_off,
                patch_va: func_va,
                kind: AllocEntryKind::FuncEntry,
            });
        }
    }

    tracing::debug!("find_memalign_via_caller: no suitable caller found");
    None
}

/// Scan a stripped static binary for musl allocator functions.
///
/// musl's `__libc_malloc_impl` starts with `movabs rax, <near-PTRDIFF_MAX>` — a
/// pre-computed max allocation size constant that survives stripping. We use this
/// as an anchor, then locate the public allocator stubs via JMP-target and
/// neighbour analysis.
///
/// Detection steps:
/// 1. Find `__libc_malloc_impl` via the `movabs rax, 0x7FFFFFFFFFFFxxxx` anchor
/// 2. Find `malloc` via JMP rel32 stubs targeting `__libc_malloc_impl`
/// 3. Find `free` as the JMP stub adjacent to `malloc` (target starts with `test rdi, rdi`)
/// 4. Find `aligned_alloc` adjacent to `malloc` (power-of-2 check pattern)
/// 5. Find `realloc` via the second `movabs` anchor + JMP stub
/// 6. Find `posix_memalign` via its EINVAL + min-alignment prologue
/// 7. Find `calloc` via `mul; jo` overflow check + call to `malloc`
fn scan_for_musl_allocators(
    data: &[u8],
    text: &TextSection,
    loads: &[&goblin::elf::ProgramHeader],
) -> AllocatorSymbols {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic, Register};

    let text_start = text.offset as usize;
    let text_end = text_start + text.size as usize;
    if text_end > data.len() {
        return AllocatorSymbols::default();
    }
    let text_bytes = &data[text_start..text_end];

    // ── Step 1: Find `movabs rax, 0x7FFFFFFFFFFFxxxx` anchors ──
    //
    // This instruction loads musl's pre-computed maximum allocation size. It appears
    // exactly twice: once in __libc_malloc_impl, once in __libc_realloc.
    // Pattern: 48 B8 xx xx xx xx FF FF FF 7F
    //   bytes [2..6] vary (the exact subtracted overhead), [6..10] are stable.
    let mut anchor_vas: Vec<u64> = Vec::new();

    for i in 0..text_bytes.len().saturating_sub(10) {
        if text_bytes[i] != 0x48 || text_bytes[i + 1] != 0xB8 {
            continue;
        }
        // Top 4 bytes of the immediate must be FF FF FF 7F (near PTRDIFF_MAX).
        if text_bytes[i + 6] != 0xFF
            || text_bytes[i + 7] != 0xFF
            || text_bytes[i + 8] != 0xFF
            || text_bytes[i + 9] != 0x7F
        {
            continue;
        }

        let va = text.va + i as u64;

        // Verify: the movabs is followed by pushes, then `cmp rax, r<arg>; jb`.
        let end = text_bytes.len().min(i + 64);
        let slice = &text_bytes[i..end];
        let mut dec = Decoder::with_ip(64, slice, va, DecoderOptions::NONE);

        let movabs = dec.decode();
        if movabs.is_invalid() || movabs.mnemonic() != Mnemonic::Mov {
            continue;
        }

        let mut push_count = 0;
        let mut found_cmp = false;
        for _ in 0..12 {
            if !dec.can_decode() {
                break;
            }
            let instr = dec.decode();
            if instr.is_invalid() {
                break;
            }
            match instr.mnemonic() {
                Mnemonic::Push => push_count += 1,
                Mnemonic::Sub | Mnemonic::Mov => {} // sub rsp / mov reg, reg
                Mnemonic::Cmp => {
                    // cmp rax, rdi/rsi (size in first or second arg)
                    let r0 = instr.op0_register();
                    let r1 = instr.op1_register();
                    if r0 == Register::RAX && (r1 == Register::RDI || r1 == Register::RSI) {
                        found_cmp = true;
                    }
                    break;
                }
                _ => break,
            }
        }

        if push_count >= 3 && found_cmp {
            tracing::debug!(
                "musl scan: movabs anchor at VA 0x{:x} ({} pushes, cmp verified)",
                va,
                push_count
            );
            anchor_vas.push(va);
        }
    }

    if anchor_vas.is_empty() {
        tracing::debug!("musl scan: no movabs rax anchor found");
        return AllocatorSymbols::default();
    }

    let malloc_impl_va = anchor_vas[0];
    let realloc_impl_movabs = anchor_vas.get(1).copied();

    tracing::debug!(
        "musl scan: __libc_malloc_impl at VA 0x{:x}{}",
        malloc_impl_va,
        realloc_impl_movabs
            .map(|va| format!(", __libc_realloc movabs at 0x{:x}", va))
            .unwrap_or_default()
    );

    let mut syms = AllocatorSymbols::default();

    // ── Step 2: Find `malloc` via JMP rel32 targeting __libc_malloc_impl ──
    //
    // musl's public `malloc` is a bare `JMP rel32` (5 bytes) that jumps directly
    // to __libc_malloc_impl. There may be two such stubs (__libc_malloc + malloc).
    let mut jmp_stubs: Vec<u64> = Vec::new();
    {
        let mut decoder = Decoder::with_ip(64, text_bytes, text.va, DecoderOptions::NONE);
        while decoder.can_decode() {
            let instr = decoder.decode();
            if instr.is_invalid() {
                continue;
            }
            if instr.mnemonic() == Mnemonic::Jmp
                && instr.len() == 5
                && instr.near_branch_target() == malloc_impl_va
            {
                jmp_stubs.push(instr.ip());
            }
        }
    }

    if jmp_stubs.is_empty() {
        tracing::debug!("musl scan: no JMP rel32 targeting __libc_malloc_impl");
        return AllocatorSymbols::default();
    }

    // Find the public `malloc` stub. In musl, `__libc_malloc` and `malloc` are
    // consecutive 5-byte JMP stubs to the same target. This back-to-back pair is
    // extremely distinctive. If no consecutive pair exists, fall back to a stub
    // preceded by a function boundary (ret/nop/int3).
    let malloc_va = {
        let mut found: Option<u64> = None;
        // Primary: consecutive JMP pair.
        for i in 0..jmp_stubs.len().saturating_sub(1) {
            if jmp_stubs[i + 1] == jmp_stubs[i] + 5 {
                found = Some(jmp_stubs[i + 1]);
                break;
            }
        }
        // Fallback: JMP preceded by function boundary.
        if found.is_none() {
            for &va in &jmp_stubs {
                let off = (va - text.va) as usize;
                if off == 0 {
                    continue;
                }
                let prev = text_bytes[off - 1];
                // ret (C3), NOP (90), INT3 (CC), or another JMP opcode's last byte
                if prev == 0xC3 || prev == 0x90 || prev == 0xCC {
                    found = Some(va);
                    break;
                }
            }
        }
        match found {
            Some(va) => va,
            None => {
                tracing::debug!(
                    "musl scan: no valid malloc stub among {} JMP candidates",
                    jmp_stubs.len()
                );
                return AllocatorSymbols::default();
            }
        }
    };
    tracing::debug!("musl scan: malloc at VA 0x{:x}", malloc_va);

    if let Some(file_off) = va_to_file_offset(loads, malloc_va) {
        syms.entries.insert(
            "malloc",
            AllocEntry {
                patch_offset: file_off,
                patch_va: malloc_va,
                kind: AllocEntryKind::FuncEntry,
            },
        );
    }

    // ── Step 3: Find `free` as a neighbouring JMP stub ──
    //
    // In musl, `free` is a bare JMP rel32 placed 5 or 10 bytes before `malloc`.
    // Its target (__libc_free) starts with `test rdi, rdi; je` (NULL-pointer check).
    for offset in [5u64, 10, 15] {
        let candidate_va = match malloc_va.checked_sub(offset) {
            Some(va) if va >= text.va => va,
            _ => continue,
        };
        let off = (candidate_va - text.va) as usize;
        if off + 5 > text_bytes.len() {
            continue;
        }

        // Must be a JMP rel32 opcode.
        if text_bytes[off] != 0xE9 {
            continue;
        }

        let rel32 = i32::from_le_bytes(
            text_bytes[off + 1..off + 5]
                .try_into()
                .expect("slice is 4 bytes"),
        );
        let target = (candidate_va as i64 + 5 + rel32 as i64) as u64;

        // Verify: target starts with `test rdi, rdi` (48 85 FF).
        if target < text.va {
            continue;
        }
        let target_off = (target - text.va) as usize;
        if target_off + 3 > text_bytes.len() {
            continue;
        }

        if text_bytes[target_off] == 0x48
            && text_bytes[target_off + 1] == 0x85
            && text_bytes[target_off + 2] == 0xFF
        {
            tracing::debug!(
                "musl scan: free at VA 0x{:x} → __libc_free at 0x{:x}",
                candidate_va,
                target
            );
            if let Some(file_off) = va_to_file_offset(loads, candidate_va) {
                syms.entries.insert(
                    "free",
                    AllocEntry {
                        patch_offset: file_off,
                        patch_va: candidate_va,
                        kind: AllocEntryKind::FuncEntry,
                    },
                );
            }
            break;
        }
    }

    // ── Step 4: Find `aligned_alloc` immediately after `malloc` ──
    //
    // musl's `aligned_alloc` starts with the power-of-2 validation:
    //   push rbx; mov rax, rdi; neg rax; and rax, rdi
    //   53 48 89 F8 48 F7 D8 48 21 F8
    let aligned_va = malloc_va + 5;
    let aligned_off = (aligned_va - text.va) as usize;
    if aligned_off + 10 <= text_bytes.len() {
        const POW2_CHECK: [u8; 10] = [0x53, 0x48, 0x89, 0xF8, 0x48, 0xF7, 0xD8, 0x48, 0x21, 0xF8];
        if text_bytes[aligned_off..aligned_off + 10] == POW2_CHECK {
            tracing::debug!(
                "musl scan: aligned_alloc at VA 0x{:x} (power-of-2 check)",
                aligned_va
            );
            if let Some(file_off) = va_to_file_offset(loads, aligned_va) {
                syms.entries.insert(
                    "aligned_alloc",
                    AllocEntry {
                        patch_offset: file_off,
                        patch_va: aligned_va,
                        kind: AllocEntryKind::FuncEntry,
                    },
                );
            }
        }
    }

    // ── Step 5: Find `realloc` via the second movabs anchor ──
    //
    // __libc_realloc contains the second movabs anchor, preceded by `test rdi, rdi; je`
    // (the NULL-pointer shortcut to malloc). The public `realloc` is a bare JMP to it.
    if let Some(movabs_va) = realloc_impl_movabs {
        // Find __libc_realloc's actual start (9 bytes before movabs if preceded by test+je).
        let realloc_impl_va = {
            let impl_off = (movabs_va - text.va) as usize;
            if impl_off >= 9 {
                let check = &text_bytes[impl_off - 9..impl_off];
                // test rdi, rdi (48 85 FF) + je rel32 (0F 84 xx xx xx xx)
                if check[0] == 0x48
                    && check[1] == 0x85
                    && check[2] == 0xFF
                    && check[3] == 0x0F
                    && check[4] == 0x84
                {
                    movabs_va - 9
                } else {
                    movabs_va
                }
            } else {
                movabs_va
            }
        };

        // Scan for JMP rel32 targeting __libc_realloc.
        let mut decoder = Decoder::with_ip(64, text_bytes, text.va, DecoderOptions::NONE);
        while decoder.can_decode() {
            let instr = decoder.decode();
            if instr.is_invalid() {
                continue;
            }
            if instr.mnemonic() == Mnemonic::Jmp
                && instr.len() == 5
                && instr.near_branch_target() == realloc_impl_va
            {
                let realloc_va = instr.ip();
                tracing::debug!(
                    "musl scan: realloc at VA 0x{:x} → __libc_realloc at 0x{:x}",
                    realloc_va,
                    realloc_impl_va
                );
                if let Some(file_off) = va_to_file_offset(loads, realloc_va) {
                    syms.entries.insert(
                        "realloc",
                        AllocEntry {
                            patch_offset: file_off,
                            patch_va: realloc_va,
                            kind: AllocEntryKind::FuncEntry,
                        },
                    );
                }
                break;
            }
        }
    }

    // ── Step 6: Find `posix_memalign` via its EINVAL + alignment check ──
    //
    // musl's posix_memalign starts with:
    //   mov ecx, 0x16; cmp rsi, 0x7; jbe .fail
    //   B9 16 00 00 00 48 83 FE 07
    {
        const POSIX_MEMALIGN_SIG: [u8; 9] = [0xB9, 0x16, 0x00, 0x00, 0x00, 0x48, 0x83, 0xFE, 0x07];
        for offset in find_byte_pattern(text_bytes, &POSIX_MEMALIGN_SIG) {
            let va = text.va + offset as u64;

            // Extra validation: after the jbe, there should be a push + call to aligned_alloc.
            // Decode a few instructions to verify structural consistency.
            let end = text_bytes.len().min(offset + 32);
            let slice = &text_bytes[offset..end];
            let mut dec = Decoder::with_ip(64, slice, va, DecoderOptions::NONE);

            let mut has_jbe = false;
            for _ in 0..5 {
                if !dec.can_decode() {
                    break;
                }
                let instr = dec.decode();
                if instr.is_invalid() {
                    break;
                }
                if instr.mnemonic() == Mnemonic::Jbe {
                    has_jbe = true;
                    break;
                }
            }

            if !has_jbe {
                continue;
            }

            tracing::debug!("musl scan: posix_memalign at VA 0x{:x}", va);
            if let Some(file_off) = va_to_file_offset(loads, va) {
                syms.entries.insert(
                    "posix_memalign",
                    AllocEntry {
                        patch_offset: file_off,
                        patch_va: va,
                        kind: AllocEntryKind::FuncEntry,
                    },
                );
            }
            break;
        }
    }

    // ── Step 7: Find `calloc` via overflow check + call to malloc ──
    //
    // musl's calloc uses `mul rdi; jo .overflow` to detect count*size overflow,
    // then calls the public `malloc`. Find CALL instructions targeting malloc_va,
    // then check their containing function for the `mul; jo` pattern.
    if syms.entries.contains_key("malloc") {
        let mut decoder = Decoder::with_ip(64, text_bytes, text.va, DecoderOptions::NONE);
        let mut call_sites: Vec<u64> = Vec::new();

        while decoder.can_decode() {
            let instr = decoder.decode();
            if instr.is_invalid() {
                continue;
            }
            if instr.mnemonic() == Mnemonic::Call && instr.near_branch_target() == malloc_va {
                call_sites.push(instr.ip());
            }
        }

        'calloc_search: for &call_va in &call_sites {
            let func_va = match find_function_start_backward(text_bytes, text.va, call_va) {
                Some(va) => va,
                None => continue,
            };

            // Skip if this is a function we've already identified.
            if syms.entries.values().any(|e| e.patch_va == func_va) {
                continue;
            }

            // Decode the first ~30 instructions looking for `mul` followed by `jo`.
            let func_off = (func_va - text.va) as usize;
            let end = text_bytes.len().min(func_off + 128);
            let slice = &text_bytes[func_off..end];
            let mut dec = Decoder::with_ip(64, slice, func_va, DecoderOptions::NONE);

            let mut saw_mul = false;
            for _ in 0..30 {
                if !dec.can_decode() {
                    break;
                }
                let instr = dec.decode();
                if instr.is_invalid() {
                    break;
                }
                match instr.mnemonic() {
                    Mnemonic::Mul | Mnemonic::Imul => saw_mul = true,
                    Mnemonic::Jo if saw_mul => {
                        tracing::debug!(
                            "musl scan: calloc at VA 0x{:x} (mul+jo + calls malloc)",
                            func_va
                        );
                        if let Some(file_off) = va_to_file_offset(loads, func_va) {
                            syms.entries.insert(
                                "calloc",
                                AllocEntry {
                                    patch_offset: file_off,
                                    patch_va: func_va,
                                    kind: AllocEntryKind::FuncEntry,
                                },
                            );
                        }
                        break 'calloc_search;
                    }
                    _ => {}
                }
            }
        }
    }

    let found = syms.count();
    if found > 0 {
        tracing::info!(
            "musl scan: discovered {}/{} allocator functions in stripped binary",
            found,
            ALLOC_TARGETS.len()
        );
    } else {
        tracing::debug!("musl scan: no allocator functions found");
    }

    syms
}

/// Scan backward from `xref_va` to find the entry point of the containing function.
///
/// Two-pass approach:
///   1. Scan backward for boundary markers (ret, int3, ud2) followed by alignment
///      padding. This is the primary strategy — it works for any prologue style.
///   2. Scan backward for distinctive function prologues: `endbr64` or two-or-more
///      consecutive `push` instructions. This catches functions preceded by
///      unconditional jumps or tail calls (no ret/int3/ud2 boundary).
///
/// Both passes validate by decoding forward to confirm instruction alignment reaches
/// the xref. Maximum backward scan: 4096 bytes.
fn find_function_start_backward(text_bytes: &[u8], text_va: u64, xref_va: u64) -> Option<u64> {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};

    let xref_off = (xref_va - text_va) as usize;
    if xref_off == 0 {
        return None;
    }
    let scan_start = xref_off.saturating_sub(4096);

    // Helper: decode forward from candidate offset, return true if we reach xref_va
    // instruction-aligned.
    let validate = |func_off: usize| -> bool {
        if func_off >= xref_off {
            return false;
        }
        let candidate_va = text_va + func_off as u64;
        let decode_end = text_bytes.len().min(xref_off + 15);
        let slice = &text_bytes[func_off..decode_end];
        let mut dec = Decoder::with_ip(64, slice, candidate_va, DecoderOptions::NONE);

        while dec.can_decode() {
            let instr = dec.decode();
            if instr.is_invalid() {
                return false;
            }
            if instr.ip() == xref_va {
                return true;
            }
            if instr.ip() > xref_va {
                return false;
            }
        }
        false
    };

    // Pass 1: Scan backward for distinctive prologues (endbr64, 2+ push)
    // that are preceded by NOP/int3 padding. Requiring padding prevents false
    // matches on push/endbr64 byte patterns inside other functions or instructions.
    //
    // Once a valid candidate is found, extend backward up to 64 bytes to capture
    // earlier pushes in the same prologue (compilers sometimes interleave pushes
    // with register moves like `push r13; mov %edx, %r13d; push r12`).
    let mut pos = xref_off;
    let mut best_prologue: Option<usize> = None;
    let mut extend_limit: Option<usize> = None;

    while pos > scan_start {
        pos -= 1;

        if let Some(limit) = extend_limit
            && pos < limit
        {
            break;
        }

        let is_candidate = is_endbr64(text_bytes, pos) || is_multi_push_prologue(text_bytes, pos);

        if is_candidate && is_preceded_by_nop(text_bytes, text_va, pos) && validate(pos) {
            best_prologue = Some(pos);
            if extend_limit.is_none() {
                extend_limit = Some(pos.saturating_sub(64));
            }
        }
    }

    if let Some(off) = best_prologue {
        return Some(text_va + off as u64);
    }

    // Pass 2: Scan backward for boundary markers (ret/int3/ud2) + padding.
    // For functions with non-standard prologues (e.g. TLS access, call).
    pos = xref_off;
    while pos > scan_start {
        pos -= 1;

        let is_boundary = match text_bytes[pos] {
            0xC3 => true,                                           // ret
            0xCC => true,                                           // int3
            0x0B if pos > 0 && text_bytes[pos - 1] == 0x0F => true, // ud2
            _ => false,
        };

        if !is_boundary {
            continue;
        }

        // Skip alignment padding after the boundary marker.
        let pad_start = pos + 1;
        let mut func_off = pad_start;
        while func_off < xref_off {
            match text_bytes[func_off] {
                0x90 | 0xCC => func_off += 1,
                0x0F | 0x66 => {
                    let end = text_bytes.len().min(func_off + 15);
                    let slice = &text_bytes[func_off..end];
                    let va = text_va + func_off as u64;
                    let mut dec = Decoder::with_ip(64, slice, va, DecoderOptions::NONE);
                    let instr = dec.decode();
                    if !instr.is_invalid() && instr.mnemonic() == Mnemonic::Nop {
                        func_off += instr.len();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        // Require padding between boundary and candidate — a bare boundary
        // byte without padding is likely a false positive (e.g. 0xCC ModR/M).
        if func_off <= pad_start {
            continue;
        }

        if validate(func_off) {
            return Some(text_va + func_off as u64);
        }
    }

    None
}

/// Check if the instruction immediately before `pos` is a NOP or int3.
/// This indicates alignment padding between functions — a strong signal that
/// `pos` is a function entry point.
fn is_preceded_by_nop(text_bytes: &[u8], text_va: u64, pos: usize) -> bool {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};

    // Single-byte NOP (0x90) or int3 (0xCC) padding.
    if pos > 0 && (text_bytes[pos - 1] == 0x90 || text_bytes[pos - 1] == 0xCC) {
        return true;
    }

    // Multi-byte NOPs (0F 1F XX, 66 0F 1F XX, etc.) — try decoding
    // 1-to-8 byte instructions ending right before pos.
    for try_len in 2..=8 {
        if pos < try_len {
            continue;
        }
        let start = pos - try_len;
        let slice = &text_bytes[start..pos];
        let va = text_va + start as u64;
        let mut dec = Decoder::with_ip(64, slice, va, DecoderOptions::NONE);
        let instr = dec.decode();
        if !instr.is_invalid() && instr.len() == try_len && instr.mnemonic() == Mnemonic::Nop {
            return true;
        }
    }

    false
}

/// Check for endbr64 (F3 0F 1E FA) at the given offset.
fn is_endbr64(text_bytes: &[u8], off: usize) -> bool {
    off + 4 <= text_bytes.len()
        && text_bytes[off] == 0xF3
        && text_bytes[off + 1] == 0x0F
        && text_bytes[off + 2] == 0x1E
        && text_bytes[off + 3] == 0xFA
}

/// Check if the bytes at `off` decode as 2+ consecutive push-register instructions.
/// This pattern is highly indicative of a function prologue and very unlikely to
/// appear at a misaligned offset within other instructions.
fn is_multi_push_prologue(text_bytes: &[u8], off: usize) -> bool {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};

    if off >= text_bytes.len() {
        return false;
    }
    let end = text_bytes.len().min(off + 30);
    let slice = &text_bytes[off..end];
    let mut dec = Decoder::with_ip(64, slice, 0, DecoderOptions::NONE);

    let mut push_count = 0;
    while dec.can_decode() {
        let instr = dec.decode();
        if instr.is_invalid() {
            break;
        }
        if instr.mnemonic() == Mnemonic::Push {
            push_count += 1;
        } else {
            break;
        }
    }
    push_count >= 2
}

#[cfg(test)]
mod tests {
    use super::{AllocEntryKind, ElfContext, find_allocator_symbols};

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_usr_bin_true() {
        let data = std::fs::read("/usr/bin/true").expect("cannot read /usr/bin/true");
        let ctx = ElfContext::parse(&data).expect("failed to parse ELF");

        assert!(ctx.text.size > 0, ".text section should have non-zero size");
        assert!(ctx.text.va > 0, ".text VA should be non-zero");
        assert!(ctx.entry_point > 0, "entry point should be non-zero");
        assert!(ctx.highest_va_end > 0, "highest VA end should be non-zero");
        assert!(ctx.is_64bit);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_func_symbols_non_empty() {
        let data = std::fs::read("/usr/bin/true").expect("cannot read /usr/bin/true");
        let ctx = ElfContext::parse(&data).expect("failed to parse ELF");
        let _ = ctx.func_symbols.len();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_plt_ranges_detected() {
        let data = std::fs::read("/usr/bin/true").expect("cannot read /usr/bin/true");
        let ctx = ElfContext::parse(&data).expect("failed to parse ELF");
        for plt in &ctx.plt_ranges {
            assert!(plt.size > 0);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_find_allocator_symbols_dynamic() {
        let data = std::fs::read("/usr/bin/true").expect("cannot read /usr/bin/true");
        let ctx = ElfContext::parse(&data).expect("failed to parse ELF");
        let syms = find_allocator_symbols(&ctx, &data);
        if let Some(entry) = syms.get("malloc") {
            assert_eq!(entry.kind, AllocEntryKind::GotPlt);
            assert!(entry.patch_offset > 0);
        }
        if let Some(entry) = syms.get("free") {
            assert_eq!(entry.kind, AllocEntryKind::GotPlt);
            assert!(entry.patch_offset > 0);
        }
    }
}
