use crate::detect::{CpuArchitecture, detect_architecture};
use crate::elf::ElfContext;
use crate::traits::BinaryContext;

/// Newtype wrapper around ElfContext that implements BinaryContext.
///
/// This allows ElfContext to be used as a `&dyn BinaryContext` in the trait-based
/// dispatch layer, while keeping the original ElfContext struct unchanged.
#[derive(Debug)]
pub struct ElfBinaryContext {
    ctx: ElfContext,
    arch: CpuArchitecture,
}

impl ElfBinaryContext {
    /// Parse an ELF binary from raw bytes, returning an ElfBinaryContext.
    pub fn parse(data: &[u8]) -> anyhow::Result<Self> {
        let arch = detect_architecture(data)?;
        Ok(Self {
            ctx: ElfContext::parse(data)?,
            arch,
        })
    }

    /// Access the inner ElfContext for ELF-specific operations.
    pub fn inner(&self) -> &ElfContext {
        &self.ctx
    }
}

impl BinaryContext for ElfBinaryContext {
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
        self.ctx.is_shared_object
    }

    fn is_dynamic(&self) -> bool {
        self.ctx.is_dynamic
    }

    fn highest_va_end(&self) -> u64 {
        self.ctx.highest_va_end
    }

    fn func_symbols(&self) -> &[u64] {
        &self.ctx.func_symbols
    }

    fn dt_init(&self) -> Option<u64> {
        self.ctx.dt_init
    }

    fn is_64bit(&self) -> bool {
        self.ctx.is_64bit
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
