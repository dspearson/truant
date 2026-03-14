pub mod x86_64;

pub use x86_64::{X86_64Disassembler, X86_64TrampolineGenerator};

#[cfg(feature = "aarch64")]
pub mod aarch64;

#[cfg(feature = "aarch64")]
pub use aarch64::{AArch64Disassembler, AArch64TrampolineGenerator};
