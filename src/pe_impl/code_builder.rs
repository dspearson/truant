use std::collections::HashMap;

/// Opaque label for forward/backward jump references.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Label(u32);

/// Kind of relocation fixup pending for a label.
enum Fixup {
    /// 1-byte signed relative displacement at `patch_offset`, relative to `patch_offset + 1`.
    Rel8 { patch_offset: usize },
    /// 4-byte signed relative displacement at `patch_offset`, relative to `patch_offset + 4`.
    Rel32 { patch_offset: usize },
}

/// Assembler-like code buffer with label/fixup support.
///
/// Eliminates manual jump fixup bookkeeping: declare labels with [`label()`],
/// emit jumps targeting them, and [`bind()`] resolves all pending fixups.
pub struct CodeBuilder {
    code: Vec<u8>,
    init_va: u64,
    next_label: u32,
    bound: HashMap<Label, usize>,
    pending: Vec<(Label, Fixup)>,
}

impl CodeBuilder {
    pub fn new(init_va: u64) -> Self {
        Self {
            code: Vec::with_capacity(2048),
            init_va,
            next_label: 0,
            bound: HashMap::new(),
            pending: Vec::new(),
        }
    }

    /// Create a new unresolved label.
    pub fn label(&mut self) -> Label {
        let l = Label(self.next_label);
        self.next_label += 1;
        l
    }

    /// Current byte offset into the code buffer.
    pub fn pos(&self) -> usize {
        self.code.len()
    }

    /// Current virtual address (`init_va + pos()`).
    pub fn va(&self) -> u64 {
        self.init_va + self.code.len() as u64
    }

    /// Bind a label to the current code position.
    /// Resolves all pending fixups targeting this label.
    pub fn bind(&mut self, label: Label) {
        let target = self.code.len();
        assert!(
            self.bound.insert(label, target).is_none(),
            "label {label:?} already bound"
        );
        self.pending.retain(|(l, fixup)| {
            if *l != label {
                return true;
            }
            match fixup {
                Fixup::Rel8 { patch_offset } => {
                    let rel = target as isize - (*patch_offset as isize + 1);
                    assert!(
                        (-128..=127).contains(&rel),
                        "rel8 overflow: {rel} at offset {patch_offset}"
                    );
                    self.code[*patch_offset] = rel as u8;
                }
                Fixup::Rel32 { patch_offset } => {
                    let rel = target as i32 - (*patch_offset as i32 + 4);
                    self.code[*patch_offset..*patch_offset + 4].copy_from_slice(&rel.to_le_bytes());
                }
            }
            false
        });
    }

    // ---- Jump emission ----

    /// Emit a conditional jump rel8 (e.g., `opcode=0x74` for jz, `0x75` for jnz).
    pub fn jcc_short(&mut self, opcode: u8, target: Label) {
        self.code.push(opcode);
        self.emit_rel8(target);
    }

    /// Emit a conditional jump rel32 (e.g., `opcode2=0x84` for jz, `0x85` for jnz).
    /// Encodes as `0F <opcode2> <rel32>`.
    pub fn jcc_near(&mut self, opcode2: u8, target: Label) {
        self.code.extend_from_slice(&[0x0F, opcode2]);
        self.emit_rel32(target);
    }

    /// Emit `jmp rel8`.
    pub fn jmp_short(&mut self, target: Label) {
        self.code.push(0xEB);
        self.emit_rel8(target);
    }

    /// Emit `jmp rel32`.
    pub fn jmp_near(&mut self, target: Label) {
        self.code.push(0xE9);
        self.emit_rel32(target);
    }

    // ---- Raw emission ----

    /// Emit raw bytes.
    pub fn raw(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    /// Emit a single byte.
    pub fn byte(&mut self, b: u8) {
        self.code.push(b);
    }

    /// Emit a little-endian u32.
    pub fn dword(&mut self, v: u32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    /// Emit a little-endian u64.
    pub fn qword(&mut self, v: u64) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    // ---- RIP-relative addressing (PE64) ----

    /// Emit a RIP-relative instruction targeting a known VA.
    /// `prefix` = opcode bytes before disp32 (e.g., `[0x4C, 0x8D, 0x25]` for `lea r12, [rip+disp32]`).
    pub fn rip_rel(&mut self, prefix: &[u8], target_va: u64) {
        self.code.extend_from_slice(prefix);
        let rip_after = self.init_va + self.code.len() as u64 + 4;
        let disp = (target_va as i64 - rip_after as i64) as i32;
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// Emit a RIP-relative instruction with a placeholder disp32.
    /// Returns the byte offset of the disp32 for later patching via [`patch_rip_rel`].
    pub fn rip_rel_placeholder(&mut self, prefix: &[u8]) -> usize {
        self.code.extend_from_slice(prefix);
        let offset = self.code.len();
        self.code.extend_from_slice(&[0x00; 4]);
        offset
    }

    /// Patch a RIP-relative placeholder (written by [`rip_rel_placeholder`]).
    pub fn patch_rip_rel(&mut self, disp_offset: usize, target_va: u64) {
        let rip_after = self.init_va + (disp_offset + 4) as u64;
        let disp = (target_va as i64 - rip_after as i64) as i32;
        self.code[disp_offset..disp_offset + 4].copy_from_slice(&disp.to_le_bytes());
    }

    // ---- Absolute addressing helpers (PE32) ----

    /// Emit `mov reg, imm32` with placeholder. Returns offset of the imm32.
    pub fn mov_imm32_placeholder(&mut self, opcode: u8) -> usize {
        self.code.push(opcode);
        let offset = self.code.len();
        self.code.extend_from_slice(&[0x00; 4]);
        offset
    }

    /// Patch an imm32 placeholder.
    pub fn patch_imm32(&mut self, offset: usize, value: u32) {
        self.code[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    // ---- Alignment ----

    /// Pad with zero bytes to the given alignment.
    pub fn align_to(&mut self, alignment: usize) {
        while !self.code.len().is_multiple_of(alignment) {
            self.code.push(0x00);
        }
    }

    // ---- Finalize ----

    /// Consume the builder and return the code buffer.
    /// Panics if any labels remain unresolved.
    pub fn finish(self) -> Vec<u8> {
        assert!(
            self.pending.is_empty(),
            "unresolved labels: {:?}",
            self.pending.iter().map(|(l, _)| l).collect::<Vec<_>>()
        );
        self.code
    }

    // ---- Internal ----

    fn emit_rel8(&mut self, target: Label) {
        let patch_offset = self.code.len();
        self.code.push(0x00);
        if let Some(&pos) = self.bound.get(&target) {
            let rel = pos as isize - (patch_offset as isize + 1);
            assert!((-128..=127).contains(&rel), "rel8 overflow: {rel}");
            self.code[patch_offset] = rel as u8;
        } else {
            self.pending.push((target, Fixup::Rel8 { patch_offset }));
        }
    }

    fn emit_rel32(&mut self, target: Label) {
        let patch_offset = self.code.len();
        self.code.extend_from_slice(&[0x00; 4]);
        if let Some(&pos) = self.bound.get(&target) {
            let rel = pos as i32 - (patch_offset as i32 + 4);
            self.code[patch_offset..patch_offset + 4].copy_from_slice(&rel.to_le_bytes());
        } else {
            self.pending.push((target, Fixup::Rel32 { patch_offset }));
        }
    }
}

/// PE architecture variant.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PeArch {
    X86,
    X64,
}

impl PeArch {
    pub fn is_64bit(self) -> bool {
        self == PeArch::X64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_jmp_short() {
        let mut cb = CodeBuilder::new(0x1000);
        let lbl = cb.label();
        cb.jmp_short(lbl);
        cb.raw(&[0x90]); // nop (1 byte between jump and target)
        cb.bind(lbl);
        cb.raw(&[0xCC]); // int3
        let code = cb.finish();
        assert_eq!(code, vec![0xEB, 0x01, 0x90, 0xCC]);
    }

    #[test]
    fn test_backward_jmp_short() {
        let mut cb = CodeBuilder::new(0x1000);
        let lbl = cb.label();
        cb.bind(lbl);
        cb.raw(&[0x90]); // nop
        cb.jmp_short(lbl);
        let code = cb.finish();
        // jmp_short: EB rel8, rel8 = 0 - (2+1) = -3 = 0xFD
        assert_eq!(code, vec![0x90, 0xEB, 0xFD]);
    }

    #[test]
    fn test_forward_jcc_near() {
        let mut cb = CodeBuilder::new(0x1000);
        let lbl = cb.label();
        cb.jcc_near(0x84, lbl); // jz rel32
        cb.bind(lbl);
        let code = cb.finish();
        // 0F 84 00 00 00 00 — disp32 = 0 (target is immediately after)
        assert_eq!(code, vec![0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_forward_jcc_short() {
        let mut cb = CodeBuilder::new(0x1000);
        let lbl = cb.label();
        cb.jcc_short(0x74, lbl); // jz rel8
        cb.raw(&[0x90, 0x90, 0x90]); // 3 nops
        cb.bind(lbl);
        let code = cb.finish();
        assert_eq!(code, vec![0x74, 0x03, 0x90, 0x90, 0x90]);
    }

    #[test]
    fn test_multiple_fixups_same_label() {
        let mut cb = CodeBuilder::new(0x1000);
        let done = cb.label();
        cb.jcc_short(0x74, done); // jz done
        cb.raw(&[0x90]);
        cb.jcc_short(0x75, done); // jnz done
        cb.bind(done);
        let code = cb.finish();
        assert_eq!(code, vec![0x74, 0x03, 0x90, 0x75, 0x00]);
    }

    #[test]
    fn test_rip_rel() {
        let mut cb = CodeBuilder::new(0x1000);
        // lea rax, [rip + disp32] targeting VA 0x2000
        cb.rip_rel(&[0x48, 0x8D, 0x05], 0x2000);
        let code = cb.finish();
        // After prefix (3 bytes) + disp32 (4 bytes) = 7 bytes total
        // RIP after instruction = 0x1000 + 7 = 0x1007
        // disp32 = 0x2000 - 0x1007 = 0x0FF9
        assert_eq!(code.len(), 7);
        let disp = i32::from_le_bytes(code[3..7].try_into().unwrap());
        assert_eq!(disp, 0x0FF9);
    }

    #[test]
    #[should_panic(expected = "unresolved labels")]
    fn test_unresolved_label_panics() {
        let mut cb = CodeBuilder::new(0x1000);
        let lbl = cb.label();
        cb.jmp_short(lbl);
        cb.finish(); // should panic
    }

    #[test]
    #[should_panic(expected = "already bound")]
    fn test_double_bind_panics() {
        let mut cb = CodeBuilder::new(0x1000);
        let lbl = cb.label();
        cb.bind(lbl);
        cb.bind(lbl); // should panic
    }

    #[test]
    fn test_jmp_near_forward() {
        let mut cb = CodeBuilder::new(0x1000);
        let lbl = cb.label();
        cb.jmp_near(lbl); // E9 rel32 (5 bytes)
        cb.raw(&[0x90, 0x90]); // 2 nops
        cb.bind(lbl);
        let code = cb.finish();
        // disp32 = target - (patch_offset + 4) = 7 - (1 + 4) = 2
        assert_eq!(code, vec![0xE9, 0x02, 0x00, 0x00, 0x00, 0x90, 0x90]);
    }

    #[test]
    fn test_backward_jcc_near() {
        let mut cb = CodeBuilder::new(0x1000);
        let lbl = cb.label();
        cb.bind(lbl);
        cb.raw(&[0x90]); // nop
        cb.jcc_near(0x85, lbl); // jnz rel32 backward
        let code = cb.finish();
        // 0F 85 rel32: patch at offset 3, target at 0
        // rel32 = 0 - (3 + 4) = -7 = 0xFFFFFFF9
        assert_eq!(code.len(), 7);
        assert_eq!(&code[0..3], &[0x90, 0x0F, 0x85]);
        let rel = i32::from_le_bytes(code[3..7].try_into().unwrap());
        assert_eq!(rel, -7);
    }

    #[test]
    fn test_mixed_forward_backward_refs() {
        // Forward ref to A, backward ref to B, then bind A
        let mut cb = CodeBuilder::new(0x1000);
        let a = cb.label();
        let b = cb.label();
        cb.jmp_short(a); // forward to A
        cb.bind(b); // B is here
        cb.raw(&[0x90, 0x90]); // 2 nops
        cb.bind(a); // A is here
        cb.jmp_short(b); // backward to B
        let code = cb.finish();
        // Byte 0: EB, byte 1: rel8 forward (A at offset 4, patch at 1, rel = 4 - 2 = 2)
        // Byte 2: B is here
        // Bytes 2-3: nops
        // Byte 4: A is here
        // Byte 4: EB, byte 5: rel8 backward (B at 2, patch at 5, rel = 2 - 6 = -4)
        assert_eq!(code, vec![0xEB, 0x02, 0x90, 0x90, 0xEB, 0xFC]);
    }

    #[test]
    fn test_many_fixups_to_one_label() {
        let mut cb = CodeBuilder::new(0x1000);
        let done = cb.label();
        for _ in 0..10 {
            cb.jcc_short(0x74, done); // 10 forward jz to same label
        }
        cb.bind(done);
        let code = cb.finish();
        assert_eq!(code.len(), 20); // 10 * 2 bytes each
        // Last jz: at offset 18, rel8 = 20 - 19 = 1... wait, 0
        // jz at offset 18: patch at 19, target at 20, rel = 20 - 20 = 0
        assert_eq!(code[19], 0x00);
        // First jz: at offset 0: patch at 1, target at 20, rel = 20 - 2 = 18
        assert_eq!(code[1], 18);
    }

    #[test]
    fn test_rip_rel_placeholder_and_patch() {
        let mut cb = CodeBuilder::new(0x1000);
        let fixup = cb.rip_rel_placeholder(&[0x48, 0x8D, 0x0D]); // lea rcx, [rip+disp32]
        cb.raw(&[0x90]); // nop after
        // Now patch: target VA = 0x2000
        // RIP after disp32 = 0x1000 + (fixup + 4) = 0x1000 + 7 = 0x1007
        cb.patch_rip_rel(fixup, 0x2000);
        let code = cb.finish();
        assert_eq!(code.len(), 8); // 3 prefix + 4 disp + 1 nop
        let disp = i32::from_le_bytes(code[3..7].try_into().unwrap());
        assert_eq!(disp, 0x2000i32 - 0x1007);
    }

    #[test]
    fn test_mov_imm32_placeholder_and_patch() {
        let mut cb = CodeBuilder::new(0x1000);
        let fixup = cb.mov_imm32_placeholder(0xB8); // mov eax, imm32
        cb.patch_imm32(fixup, 0xDEADBEEF);
        let code = cb.finish();
        assert_eq!(code, vec![0xB8, 0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn test_align_to() {
        let mut cb = CodeBuilder::new(0x1000);
        cb.raw(&[0x90, 0x90, 0x90]); // 3 bytes
        cb.align_to(8);
        let code = cb.finish();
        assert_eq!(code.len(), 8);
        assert_eq!(&code[3..], &[0x00, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_align_to_already_aligned() {
        let mut cb = CodeBuilder::new(0x1000);
        cb.raw(&[0x90; 8]); // already 8-aligned
        cb.align_to(8);
        let code = cb.finish();
        assert_eq!(code.len(), 8); // no padding added
    }
}
