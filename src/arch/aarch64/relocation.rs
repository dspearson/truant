//! Shared AArch64 PC-relative instruction relocation.
//!
//! Provides detection and relocation of ARM64 instructions that encode
//! PC-relative offsets: ADRP, ADR, LDR literal, B, BL, B.cond,
//! CBZ/CBNZ, TBZ/TBNZ.
//!
//! Used by both the hook trampoline generator and the coverage trampoline
//! generator to correctly relocate displaced instructions.

use anyhow::Result;

/// Sign-extend a value from `bits` width to i64.
pub(crate) fn sign_extend(val: u32, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((val as i64) << shift) >> shift
}

/// Return true if the instruction word is PC-relative.
pub(crate) fn is_pc_relative(raw: u32) -> bool {
    let is_adrp = (raw & 0x9F00_0000) == 0x9000_0000;
    let is_adr = (raw & 0x9F00_0000) == 0x1000_0000;
    let is_ldr_lit = (raw & 0x3B00_0000) == 0x1800_0000;
    let is_b = (raw & 0xFC00_0000) == 0x1400_0000;
    let is_bl = (raw & 0xFC00_0000) == 0x9400_0000;
    let is_bcond = (raw & 0xFF00_0010) == 0x5400_0000;
    let is_cbz_cbnz = (raw & 0x7E00_0000) == 0x3400_0000;
    let is_tbz_tbnz = (raw & 0x7E00_0000) == 0x3600_0000;
    is_adrp || is_adr || is_ldr_lit || is_b || is_bl || is_bcond || is_cbz_cbnz || is_tbz_tbnz
}

/// Relocate a PC-relative AArch64 instruction from `original_va` to `new_va`.
///
/// Decodes the instruction, computes its original target, then re-encodes with an
/// adjusted immediate so it reaches the same target from the new location.
/// Returns the relocated instruction word, or an error if the new offset is out of range.
pub(crate) fn relocate_pc_relative(raw: u32, original_va: u64, new_va: u64) -> Result<u32> {
    let is_adrp = (raw & 0x9F00_0000) == 0x9000_0000;
    let is_adr = (raw & 0x9F00_0000) == 0x1000_0000;
    let is_ldr_lit = (raw & 0x3B00_0000) == 0x1800_0000;
    let is_b = (raw & 0xFC00_0000) == 0x1400_0000;
    let is_bl = (raw & 0xFC00_0000) == 0x9400_0000;
    let is_bcond = (raw & 0xFF00_0010) == 0x5400_0000;
    let is_cbz_cbnz = (raw & 0x7E00_0000) == 0x3400_0000;
    let is_tbz_tbnz = (raw & 0x7E00_0000) == 0x3600_0000;

    if is_adrp {
        let rd = raw & 0x1F;
        let immlo = (raw >> 29) & 0x3;
        let immhi = (raw >> 5) & 0x7_FFFF;
        let imm21 = (immhi << 2) | immlo;
        let imm_signed = sign_extend(imm21, 21);
        let target_page = ((original_va & !0xFFF) as i64 + (imm_signed << 12)) as u64;

        let new_imm = ((target_page as i64) - (new_va as i64 & !0xFFF_i64)) >> 12;
        if !(-(1 << 20)..(1 << 20)).contains(&new_imm) {
            anyhow::bail!(
                "ADRP relocation out of range: target page 0x{:x}, new VA 0x{:x}, delta {} pages (max +/-1M pages)",
                target_page,
                new_va,
                new_imm
            );
        }
        let new_imm21 = (new_imm as u32) & 0x1F_FFFF;
        let new_immhi = (new_imm21 >> 2) & 0x7_FFFF;
        let new_immlo = new_imm21 & 0x3;
        Ok((1u32 << 31) | (new_immlo << 29) | (0b10000u32 << 24) | (new_immhi << 5) | rd)
    } else if is_adr {
        let rd = raw & 0x1F;
        let immlo = (raw >> 29) & 0x3;
        let immhi = (raw >> 5) & 0x7_FFFF;
        let imm21 = (immhi << 2) | immlo;
        let imm_signed = sign_extend(imm21, 21);
        let target = (original_va as i64 + imm_signed) as u64;

        let new_offset = target as i64 - new_va as i64;
        if !(-(1 << 20)..(1 << 20)).contains(&new_offset) {
            anyhow::bail!(
                "ADR relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {} (max +/-1M)",
                target,
                new_va,
                new_offset
            );
        }
        let new_imm21 = (new_offset as u32) & 0x1F_FFFF;
        let new_immhi = (new_imm21 >> 2) & 0x7_FFFF;
        let new_immlo = new_imm21 & 0x3;
        Ok((new_immlo << 29) | (0b10000u32 << 24) | (new_immhi << 5) | rd)
    } else if is_ldr_lit {
        let rt = raw & 0x1F;
        let opc_v = raw & 0xFF00_0000;
        let imm19 = (raw >> 5) & 0x7_FFFF;
        let offset_signed = sign_extend(imm19, 19) * 4;
        let target = (original_va as i64 + offset_signed) as u64;

        let new_offset = target as i64 - new_va as i64;
        if new_offset % 4 != 0 || new_offset / 4 < -(1 << 18) || new_offset / 4 >= (1 << 18) {
            anyhow::bail!(
                "LDR literal relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {}",
                target,
                new_va,
                new_offset
            );
        }
        let new_imm19 = ((new_offset / 4) as u32) & 0x7_FFFF;
        Ok(opc_v | (new_imm19 << 5) | rt)
    } else if is_b || is_bl {
        let opcode_bits = raw & 0xFC00_0000;
        let imm26 = raw & 0x03FF_FFFF;
        let offset_signed = sign_extend(imm26, 26) * 4;
        let target = (original_va as i64 + offset_signed) as u64;

        let new_offset = target as i64 - new_va as i64;
        if new_offset % 4 != 0 || new_offset / 4 < -(1 << 25) || new_offset / 4 >= (1 << 25) {
            anyhow::bail!(
                "B/BL relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {}",
                target,
                new_va,
                new_offset
            );
        }
        let new_imm26 = ((new_offset / 4) as u32) & 0x03FF_FFFF;
        Ok(opcode_bits | new_imm26)
    } else if is_bcond {
        let cond = raw & 0xF;
        let imm19 = (raw >> 5) & 0x7_FFFF;
        let offset_signed = sign_extend(imm19, 19) * 4;
        let target = (original_va as i64 + offset_signed) as u64;

        let new_offset = target as i64 - new_va as i64;
        if new_offset % 4 != 0 || new_offset / 4 < -(1 << 18) || new_offset / 4 >= (1 << 18) {
            anyhow::bail!(
                "B.cond relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {}",
                target,
                new_va,
                new_offset
            );
        }
        let new_imm19 = ((new_offset / 4) as u32) & 0x7_FFFF;
        Ok(0x5400_0000 | (new_imm19 << 5) | cond)
    } else if is_cbz_cbnz {
        let sf_op = raw & 0xFF00_0000;
        let rt = raw & 0x1F;
        let imm19 = (raw >> 5) & 0x7_FFFF;
        let offset_signed = sign_extend(imm19, 19) * 4;
        let target = (original_va as i64 + offset_signed) as u64;

        let new_offset = target as i64 - new_va as i64;
        if new_offset % 4 != 0 || new_offset / 4 < -(1 << 18) || new_offset / 4 >= (1 << 18) {
            anyhow::bail!(
                "CBZ/CBNZ relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {}",
                target,
                new_va,
                new_offset
            );
        }
        let new_imm19 = ((new_offset / 4) as u32) & 0x7_FFFF;
        Ok(sf_op | (new_imm19 << 5) | rt)
    } else if is_tbz_tbnz {
        let b5_op = raw & 0xFF00_0000;
        let b40 = (raw >> 19) & 0x1F;
        let rt = raw & 0x1F;
        let imm14 = (raw >> 5) & 0x3FFF;
        let offset_signed = sign_extend(imm14, 14) * 4;
        let target = (original_va as i64 + offset_signed) as u64;

        let new_offset = target as i64 - new_va as i64;
        if new_offset % 4 != 0 || new_offset / 4 < -(1 << 13) || new_offset / 4 >= (1 << 13) {
            anyhow::bail!(
                "TBZ/TBNZ relocation out of range: target 0x{:x}, new VA 0x{:x}, delta {}",
                target,
                new_va,
                new_offset
            );
        }
        let new_imm14 = ((new_offset / 4) as u32) & 0x3FFF;
        Ok(b5_op | (b40 << 19) | (new_imm14 << 5) | rt)
    } else {
        anyhow::bail!(
            "instruction 0x{:08x} is not a recognised PC-relative encoding",
            raw
        );
    }
}
