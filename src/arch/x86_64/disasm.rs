use crate::disasm::BasicBlock;
use crate::traits::{BinaryContext, Disassembler};
use anyhow::Result;

/// x86_64 disassembler implementing the Disassembler trait.
///
/// Delegates to `crate::disasm::find_basic_blocks()`, extracting the concrete
/// `ElfBinaryContext` via `as_any()` downcasting. The existing disasm module
/// requires an `&ElfContext` directly; this wrapper bridges the trait boundary.
#[derive(Debug, Default)]
pub struct X86_64Disassembler;

impl X86_64Disassembler {
    pub fn new() -> Self {
        Self
    }
}

impl Disassembler for X86_64Disassembler {
    fn find_basic_blocks(
        &self,
        binary_data: &[u8],
        ctx: &dyn BinaryContext,
        instrument_modules: &Option<Vec<String>>,
    ) -> Result<Vec<BasicBlock>> {
        if let Some(elf_ctx) = ctx
            .as_any()
            .downcast_ref::<crate::elf_impl::context::ElfBinaryContext>()
        {
            crate::disasm::find_basic_blocks(elf_ctx.inner(), binary_data, instrument_modules)
        } else if let Some(macho_ctx) = ctx
            .as_any()
            .downcast_ref::<crate::macho_impl::MachOBinaryContext>()
        {
            crate::macho_disasm::find_basic_blocks_macho(
                macho_ctx.inner(),
                binary_data,
                instrument_modules,
            )
        } else if let Some(pe_ctx) = ctx
            .as_any()
            .downcast_ref::<crate::pe_impl::PeBinaryContext>()
        {
            crate::pe_disasm::find_basic_blocks_pe(pe_ctx.inner(), binary_data, instrument_modules)
        } else {
            anyhow::bail!("unsupported binary context type for x86_64 disassembly")
        }
    }
}
