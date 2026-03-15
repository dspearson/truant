//! Typed binary patching with overlap detection.
//!
//! Replaces ad-hoc `data[offset..offset+N].copy_from_slice(...)` patterns
//! with a structured `PatchSet` that validates patches don't overlap before
//! applying them. This catches off-by-one offset bugs at patch-build time
//! rather than producing silently corrupt binaries.

/// A single patch: write `bytes` at `offset` in the binary data.
#[derive(Debug, Clone)]
pub struct BinaryPatch {
    pub offset: usize,
    pub bytes: Vec<u8>,
    pub description: &'static str,
}

/// A set of patches to apply atomically.
///
/// Validates that no two patches overlap before applying, preventing the
/// class of bugs where two header field writes clobber each other due to
/// incorrect offset calculations.
#[derive(Debug, Default)]
pub struct PatchSet {
    patches: Vec<BinaryPatch>,
}

impl PatchSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a patch: write `bytes` at `offset`.
    pub fn write(&mut self, offset: usize, bytes: &[u8], description: &'static str) {
        self.patches.push(BinaryPatch {
            offset,
            bytes: bytes.to_vec(),
            description,
        });
    }

    /// Queue a little-endian u16 write.
    pub fn write_u16(&mut self, offset: usize, value: u16, description: &'static str) {
        self.write(offset, &value.to_le_bytes(), description);
    }

    /// Queue a little-endian u32 write.
    pub fn write_u32(&mut self, offset: usize, value: u32, description: &'static str) {
        self.write(offset, &value.to_le_bytes(), description);
    }

    /// Queue a little-endian u64 write.
    pub fn write_u64(&mut self, offset: usize, value: u64, description: &'static str) {
        self.write(offset, &value.to_le_bytes(), description);
    }

    /// Queue zeroing N bytes at `offset`.
    pub fn zero(&mut self, offset: usize, len: usize, description: &'static str) {
        self.write(offset, &vec![0u8; len], description);
    }

    /// Check for overlapping patches. Returns the first overlap found, if any.
    fn find_overlap(&self) -> Option<(&BinaryPatch, &BinaryPatch)> {
        for i in 0..self.patches.len() {
            let a = &self.patches[i];
            let a_end = a.offset + a.bytes.len();
            for b in &self.patches[i + 1..] {
                let b_end = b.offset + b.bytes.len();
                if a.offset < b_end && b.offset < a_end {
                    return Some((a, b));
                }
            }
        }
        None
    }

    /// Apply all queued patches to `data`. Returns an error if any patches
    /// overlap or extend beyond the data buffer.
    pub fn apply(self, data: &mut [u8]) -> anyhow::Result<()> {
        if let Some((a, b)) = self.find_overlap() {
            anyhow::bail!(
                "overlapping patches: '{}' at 0x{:x}..0x{:x} overlaps '{}' at 0x{:x}..0x{:x}",
                a.description,
                a.offset,
                a.offset + a.bytes.len(),
                b.description,
                b.offset,
                b.offset + b.bytes.len(),
            );
        }

        for patch in &self.patches {
            let end = patch.offset + patch.bytes.len();
            if end > data.len() {
                anyhow::bail!(
                    "patch '{}' at 0x{:x}..0x{:x} extends beyond data (len=0x{:x})",
                    patch.description,
                    patch.offset,
                    end,
                    data.len(),
                );
            }
            data[patch.offset..end].copy_from_slice(&patch.bytes);
        }

        Ok(())
    }

    /// Number of queued patches.
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    /// Whether the patch set is empty.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_non_overlapping() {
        let mut data = vec![0u8; 16];
        let mut ps = PatchSet::new();
        ps.write_u32(0, 0xDEADBEEF, "field_a");
        ps.write_u32(8, 0xCAFEBABE, "field_b");
        ps.apply(&mut data).unwrap();
        assert_eq!(&data[0..4], &0xDEADBEEFu32.to_le_bytes());
        assert_eq!(&data[8..12], &0xCAFEBABEu32.to_le_bytes());
    }

    #[test]
    fn test_detect_overlap() {
        let mut data = vec![0u8; 16];
        let mut ps = PatchSet::new();
        ps.write_u32(0, 0x11111111, "field_a");
        ps.write_u32(2, 0x22222222, "field_b"); // overlaps at bytes 2-3
        let result = ps.apply(&mut data);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("overlapping"),
            "error should mention overlap"
        );
    }

    #[test]
    fn test_out_of_bounds() {
        let mut data = vec![0u8; 4];
        let mut ps = PatchSet::new();
        ps.write_u32(2, 0xFFFFFFFF, "field_a"); // extends to byte 6, but data is 4
        let result = ps.apply(&mut data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("beyond data"));
    }

    #[test]
    fn test_adjacent_non_overlapping() {
        let mut data = vec![0u8; 8];
        let mut ps = PatchSet::new();
        ps.write_u32(0, 0x11111111, "field_a");
        ps.write_u32(4, 0x22222222, "field_b"); // adjacent, not overlapping
        assert!(ps.apply(&mut data).is_ok());
    }

    #[test]
    fn test_zero() {
        let mut data = vec![0xFFu8; 8];
        let mut ps = PatchSet::new();
        ps.zero(2, 4, "clear_middle");
        ps.apply(&mut data).unwrap();
        assert_eq!(data, vec![0xFF, 0xFF, 0, 0, 0, 0, 0xFF, 0xFF]);
    }
}
