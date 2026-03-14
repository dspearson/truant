use crate::disasm::BasicBlock;
use crate::hooks::ResolvedHook;
use crate::patcher::PatchResult;
use crate::traits::binary_context::BinaryContext;
use anyhow::Result;

/// Trait abstracting format-specific binary patching.
/// ELF implementation: ElfPatcher (PT_NOTE→PT_LOAD conversion).
/// Mach-O implementation: MachOPatcher (Phase 41).
pub trait Patcher: Send + Sync + std::fmt::Debug {
    /// Patch the binary: convert segments, insert trampolines, redirect entry.
    #[allow(clippy::too_many_arguments)]
    fn patch(
        &self,
        ctx: &dyn BinaryContext,
        blocks: &[BasicBlock],
        data: Vec<u8>,
        enable_forkserver: bool,
        enable_heap_san: bool,
        persistent_addr: Option<u64>,
        persistent_count: u32,
        defer: bool,
        hooks: &[ResolvedHook],
        no_coverage: bool,
    ) -> Result<PatchResult>;
}
