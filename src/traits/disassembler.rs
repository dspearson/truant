use crate::disasm::DisassemblyResult;
use crate::traits::binary_context::BinaryContext;
use anyhow::Result;

/// Trait abstracting architecture-specific basic block detection.
/// x86_64 implementation: X86_64Disassembler (iced-x86).
/// AArch64 implementation: AArch64Disassembler (capstone-rs, Phase 40).
pub trait Disassembler: Send + Sync + std::fmt::Debug {
    /// Detect basic blocks in the binary's code section eligible for JMP instrumentation.
    /// `instrument_modules`: if Some, restrict to blocks within named functions.
    fn find_basic_blocks(
        &self,
        binary_data: &[u8],
        ctx: &dyn BinaryContext,
        instrument_modules: &Option<Vec<String>>,
    ) -> Result<DisassemblyResult>;
}
