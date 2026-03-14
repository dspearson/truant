use crate::disasm::BasicBlock;
use crate::traits::TrampolineGenerator;
use crate::trampoline::{self, InitCode, Trampoline};
use anyhow::Result;

/// x86_64 trampoline generator implementing the TrampolineGenerator trait.
///
/// Delegates to the three top-level functions in `crate::trampoline`:
/// `generate_trampoline`, `generate_init_code`, and `generate_so_init_code`.
/// No downcasting needed — trampoline generation is architecture-specific but
/// not format-specific; all inputs arrive as plain values, not as BinaryContext.
#[derive(Debug)]
pub struct X86_64TrampolineGenerator;

impl Default for X86_64TrampolineGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl X86_64TrampolineGenerator {
    pub fn new() -> Self {
        Self
    }
}

impl TrampolineGenerator for X86_64TrampolineGenerator {
    fn generate_trampoline(
        &self,
        trampoline_va: u64,
        data_va: u64,
        block: &BasicBlock,
    ) -> Result<Trampoline> {
        trampoline::generate_trampoline(trampoline_va, data_va, block)
    }

    fn generate_init_code(
        &self,
        init_va: u64,
        data_va: u64,
        entry_point: u64,
        enable_forkserver: bool,
        persistent_data_va: Option<u64>,
    ) -> Result<InitCode> {
        trampoline::generate_init_code(
            init_va,
            data_va,
            entry_point,
            enable_forkserver,
            persistent_data_va,
        )
    }

    fn generate_so_init_code(
        &self,
        init_va: u64,
        data_va: u64,
        dt_init: Option<u64>,
    ) -> Result<InitCode> {
        trampoline::generate_so_init_code(init_va, data_va, dt_init)
    }

    fn encode_branch(&self, source_va: u64, target_va: u64) -> Result<Vec<u8>> {
        // JMP rel32: E9 <rel32>
        // rel32 = target_va - (source_va + 5)
        let jmp_source = source_va + 5;
        let delta = target_va as i64 - jmp_source as i64;
        if delta > i32::MAX as i64 || delta < i32::MIN as i64 {
            anyhow::bail!(
                "JMP rel32 out of range: source=0x{:x} target=0x{:x} delta={}",
                source_va,
                target_va,
                delta
            );
        }
        let rel32 = delta as i32;
        let mut buf = vec![0xE9u8];
        buf.extend_from_slice(&rel32.to_le_bytes());
        Ok(buf)
    }

    fn branch_instruction_size(&self) -> usize {
        5
    }
}
