use crate::detect::{CpuArchitecture, detect_architecture};
use crate::pe::PeContext;
use crate::traits::BinaryContext;

/// Newtype wrapper around PeContext that implements BinaryContext.
///
/// Mirrors ElfBinaryContext/MachOBinaryContext: enables `&dyn BinaryContext`
/// usage while allowing downcasting to access PE-specific fields.
#[derive(Debug)]
pub struct PeBinaryContext {
    ctx: PeContext,
    arch: CpuArchitecture,
}

impl PeBinaryContext {
    /// Parse a PE binary from raw bytes.
    pub fn parse(data: &[u8]) -> anyhow::Result<Self> {
        let arch = detect_architecture(data)?;
        Ok(Self {
            ctx: PeContext::parse(data)?,
            arch,
        })
    }

    /// Access the inner PeContext for PE-specific operations.
    pub fn inner(&self) -> &PeContext {
        &self.ctx
    }
}

impl BinaryContext for PeBinaryContext {
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
        self.ctx.is_dll
    }

    fn is_dynamic(&self) -> bool {
        // PE binaries are always dynamically linked (they use DLL imports).
        // Truly static PE binaries are rare.
        true
    }

    fn highest_va_end(&self) -> u64 {
        self.ctx.highest_va_end
    }

    fn func_symbols(&self) -> &[u64] {
        &self.ctx.func_symbols
    }

    fn dt_init(&self) -> Option<u64> {
        // Not applicable to PE (ELF concept).
        None
    }

    fn is_64bit(&self) -> bool {
        self.ctx.is_64bit
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
