#[cfg(feature = "aarch64")]
pub mod relocation;

#[cfg(feature = "aarch64")]
pub mod disasm;

#[cfg(feature = "aarch64")]
pub mod trampoline_gen;

#[cfg(feature = "aarch64")]
pub mod hook_trampoline;

#[cfg(feature = "aarch64")]
pub use disasm::AArch64Disassembler;

#[cfg(feature = "aarch64")]
pub use trampoline_gen::AArch64TrampolineGenerator;

#[cfg(feature = "aarch64")]
pub use trampoline_gen::generate_persistent_wrapper_aarch64;
