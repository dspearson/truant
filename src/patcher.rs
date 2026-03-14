//! Shared types for the patching subsystem.

/// Result of the patching operation.
///
/// Shared across ELF, Mach-O, and PE patchers.
#[derive(Debug)]
pub struct PatchResult {
    /// The rewritten binary data.
    pub data: Vec<u8>,
    /// Number of basic blocks instrumented.
    pub blocks_instrumented: usize,
    /// Number of blocks skipped (relocation failure, etc.).
    pub blocks_skipped: usize,
    /// Virtual address of the new segment.
    pub segment_va: u64,
    /// Total size of the appended segment.
    pub segment_size: u64,
    /// Number of allocator functions intercepted by heap san (0-10).
    pub heap_san_intercepted: usize,
    /// Number of hooks applied.
    pub hooks_applied: usize,
    /// Virtual address where hook data slots start (for preload companion).
    pub hook_data_va: u64,
    /// Virtual address where toggle bytes start (for preload companion).
    pub toggle_data_va: u64,
}
