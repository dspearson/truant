use crate::detect::CpuArchitecture;

/// Trait abstracting format-specific binary metadata needed for rewriting.
/// ELF implementation: ElfContext. Future Mach-O implementation: MachOContext.
pub trait BinaryContext: Send + Sync + std::fmt::Debug {
    /// CPU architecture of the binary.
    fn architecture(&self) -> CpuArchitecture;
    /// Virtual address of the binary's entry point (e_entry for ELF, __mh_execute_header start for Mach-O).
    fn entry_point(&self) -> u64;
    /// Virtual address of the executable code section (.text for ELF, __TEXT/__text for Mach-O).
    fn text_section_va(&self) -> u64;
    /// File offset of the executable code section.
    fn text_section_offset(&self) -> u64;
    /// Size of the executable code section.
    fn text_section_size(&self) -> u64;
    /// True if this is a shared object (.so / .dylib) rather than an executable.
    fn is_shared_object(&self) -> bool;
    /// True if dynamically linked (has interpreter/dynamic linker).
    fn is_dynamic(&self) -> bool;
    /// Highest virtual address end across all loadable segments.
    fn highest_va_end(&self) -> u64;
    /// Function symbol addresses (hints for basic block boundaries).
    fn func_symbols(&self) -> &[u64];
    /// DT_INIT virtual address (ELF .so only; None for executables and Mach-O).
    fn dt_init(&self) -> Option<u64>;
    /// True if 64-bit.
    fn is_64bit(&self) -> bool;
    /// Return a reference to self as `&dyn Any` for concrete-type downcasting.
    /// Standard Rust downcasting pattern — required for architecture-specific
    /// delegation (e.g. X86_64Disassembler downcast to ElfBinaryContext).
    fn as_any(&self) -> &dyn std::any::Any;
}
