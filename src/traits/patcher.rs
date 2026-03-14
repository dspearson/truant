use crate::disasm::BasicBlock;
use crate::hooks::ResolvedHook;
use crate::patcher::{InstrumentationOptions, PatchResult};
use crate::traits::binary_context::BinaryContext;
use anyhow::Result;

/// Trait abstracting format-specific binary patching.
/// ELF implementation: ElfPatcher (PT_NOTE→PT_LOAD conversion).
/// Mach-O implementation: MachOPatcher (Phase 41).
pub trait Patcher: Send + Sync + std::fmt::Debug {
    /// Patch the binary: convert segments, insert trampolines, redirect entry.
    fn patch(
        &self,
        ctx: &dyn BinaryContext,
        blocks: &[BasicBlock],
        data: Vec<u8>,
        opts: &InstrumentationOptions,
        hooks: &[ResolvedHook],
    ) -> Result<PatchResult>;
}
