use crate::detect::{CpuArchitecture, detect_architecture};
use crate::macho::MachOContext;
use crate::traits::BinaryContext;

/// Newtype wrapper around MachOContext that implements BinaryContext.
///
/// Mirrors ElfBinaryContext: enables `&dyn BinaryContext` usage while
/// allowing downcasting to access Mach-O-specific fields.
#[derive(Debug)]
pub struct MachOBinaryContext {
    ctx: MachOContext,
    arch: CpuArchitecture,
}

impl MachOBinaryContext {
    /// Parse a Mach-O binary from raw bytes.
    pub fn parse(data: &[u8]) -> anyhow::Result<Self> {
        let arch = detect_architecture(data)?;
        Ok(Self {
            ctx: MachOContext::parse(data)?,
            arch,
        })
    }

    /// Access the inner MachOContext for Mach-O-specific operations.
    pub fn inner(&self) -> &MachOContext {
        &self.ctx
    }
}

impl BinaryContext for MachOBinaryContext {
    fn architecture(&self) -> CpuArchitecture {
        self.arch
    }

    fn entry_point(&self) -> u64 {
        self.ctx.entry_point
    }

    fn text_section_va(&self) -> u64 {
        self.ctx.text.va
    }

    fn text_section_offset(&self) -> u64 {
        self.ctx.text.offset
    }

    fn text_section_size(&self) -> u64 {
        self.ctx.text.size
    }

    fn is_shared_object(&self) -> bool {
        self.ctx.is_dylib
    }

    fn is_dynamic(&self) -> bool {
        // Mach-O executables are always dynamically linked (via dyld)
        // unless they are a kernel extension or similar. For our purposes,
        // non-dylib Mach-O binaries are treated as dynamic executables.
        !self.ctx.is_dylib
    }

    fn highest_va_end(&self) -> u64 {
        self.ctx.highest_va_end
    }

    fn func_symbols(&self) -> &[u64] {
        &self.ctx.func_symbols
    }

    fn dt_init(&self) -> Option<u64> {
        // Not applicable to Mach-O (ELF concept).
        None
    }

    fn is_64bit(&self) -> bool {
        self.ctx.is_64bit
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
