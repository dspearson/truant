use crate::disasm::BasicBlock;
use crate::trampoline::{InitCode, Trampoline};
use anyhow::Result;

/// Trait abstracting architecture-specific coverage trampoline code generation.
/// x86_64 implementation: X86_64TrampolineGenerator.
/// AArch64 implementation: AArch64TrampolineGenerator (Phase 40).
pub trait TrampolineGenerator: Send + Sync + std::fmt::Debug {
    /// Generate the coverage trampoline for a single basic block.
    fn generate_trampoline(
        &self,
        trampoline_va: u64,
        data_va: u64,
        block: &BasicBlock,
    ) -> Result<Trampoline>;

    /// Generate init code (SHM attachment + optional forkserver) for executables.
    fn generate_init_code(
        &self,
        init_va: u64,
        data_va: u64,
        entry_point: u64,
        enable_forkserver: bool,
        persistent_data_va: Option<u64>,
    ) -> Result<InitCode>;

    /// Generate init code for shared objects (reads __AFL_SHM_ID, attaches SHM, chains to original DT_INIT).
    fn generate_so_init_code(
        &self,
        init_va: u64,
        data_va: u64,
        dt_init: Option<u64>,
    ) -> Result<InitCode>;

    /// Encode a branch instruction from `source_va` to `target_va`.
    /// Returns the encoded bytes and the total patch size (including any NOP padding).
    /// x86_64: JMP rel32 (5 bytes), AArch64: B imm26 (4 bytes).
    fn encode_branch(&self, source_va: u64, target_va: u64) -> Result<Vec<u8>>;

    /// Size of the branch instruction used to patch block starts.
    /// x86_64: 5 (JMP rel32), AArch64: 4 (B imm26).
    fn branch_instruction_size(&self) -> usize;
}
