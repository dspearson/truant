use anyhow::{Result, bail};

/// Binary format of a target file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryFormat {
    Elf,
    MachO,
    Fat,
    Pe,
}

/// CPU architecture of a target binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArchitecture {
    X86,
    X86_64,
    AArch64,
}

/// Detect the binary format from the first 4 magic bytes.
///
/// Returns `Ok(BinaryFormat::Elf)` for ELF binaries (`\x7fELF`).
/// Returns `Ok(BinaryFormat::MachO)` for Mach-O binaries (all four magic variants).
/// Returns `Err` for unknown formats or files shorter than 4 bytes.
pub fn detect_format(data: &[u8]) -> Result<BinaryFormat> {
    if data.len() < 4 {
        bail!("file too short to detect format ({} bytes)", data.len());
    }
    let magic = &data[..4];
    // ELF magic: 0x7f 'E' 'L' 'F'
    if magic == b"\x7fELF" {
        return Ok(BinaryFormat::Elf);
    }
    // Mach-O big-endian 32-bit (MH_MAGIC): 0xFE 0xED 0xFA 0xCE
    if magic == b"\xfe\xed\xfa\xce" {
        return Ok(BinaryFormat::MachO);
    }
    // Mach-O big-endian 64-bit (MH_MAGIC_64): 0xFE 0xED 0xFA 0xCF
    if magic == b"\xfe\xed\xfa\xcf" {
        return Ok(BinaryFormat::MachO);
    }
    // Mach-O little-endian 32-bit (MH_CIGAM): 0xCE 0xFA 0xED 0xFE
    if magic == b"\xce\xfa\xed\xfe" {
        return Ok(BinaryFormat::MachO);
    }
    // Mach-O little-endian 64-bit (MH_CIGAM_64): 0xCF 0xFA 0xED 0xFE
    if magic == b"\xcf\xfa\xed\xfe" {
        return Ok(BinaryFormat::MachO);
    }
    // Fat (universal) binary: 0xCAFEBABE or 0xCAFEBABF (big-endian)
    if magic == b"\xca\xfe\xba\xbe" || magic == b"\xca\xfe\xba\xbf" {
        return Ok(BinaryFormat::Fat);
    }
    // PE/COFF: MZ magic (0x4D5A)
    if magic[0] == 0x4D && magic[1] == 0x5A {
        // Verify PE signature at e_lfanew offset
        if data.len() >= 64 {
            let e_lfanew = u32::from_le_bytes([data[60], data[61], data[62], data[63]]) as usize;
            if e_lfanew + 4 <= data.len() && &data[e_lfanew..e_lfanew + 4] == b"PE\0\0" {
                return Ok(BinaryFormat::Pe);
            }
        }
        // MZ header without valid PE signature — still report as PE
        return Ok(BinaryFormat::Pe);
    }
    bail!("unknown binary format (magic: {:02x?})", magic);
}

/// Detect the CPU architecture from a binary.
///
/// Parses the format-specific header after calling `detect_format`.
/// For ELF: inspects e_machine (0x3E → X86_64, 0xB7 → AArch64).
/// For Mach-O: inspects cputype field (parsed via goblin).
/// Returns `Err` for unsupported architectures or parse failures.
pub fn detect_architecture(data: &[u8]) -> Result<CpuArchitecture> {
    let format = detect_format(data)?;
    match format {
        BinaryFormat::Elf => {
            // ELF e_machine is at byte offset 18 (2 bytes, little-endian for x86_64)
            // Per the ELF spec: e_ident[16] + e_type[2] + e_machine[2]
            if data.len() < 20 {
                bail!("ELF too short to read e_machine");
            }
            let e_machine = u16::from_le_bytes([data[18], data[19]]);
            match e_machine {
                0x3E => Ok(CpuArchitecture::X86_64),
                0xB7 => Ok(CpuArchitecture::AArch64),
                other => bail!("unsupported ELF e_machine: 0x{:04x}", other),
            }
        }
        BinaryFormat::MachO => {
            // Use goblin for Mach-O parsing
            // cputype is at offset 0 in the Mach-O header (after magic)
            // CPU_TYPE_X86_64 = 0x01000007, CPU_TYPE_ARM64 = 0x0100000C
            if data.len() < 8 {
                bail!("Mach-O too short to read cputype");
            }
            // Determine endianness from magic
            // MH_MAGIC (0xFEEDFACE/CF) starts with 0xFE in file = big-endian
            // MH_CIGAM (0xCEFAEDFE/CFFAEDFE) starts with 0xCE/CF in file = little-endian
            let is_big_endian = data[0] == 0xFE;
            let cputype = if is_big_endian {
                u32::from_be_bytes([data[4], data[5], data[6], data[7]])
            } else {
                u32::from_le_bytes([data[4], data[5], data[6], data[7]])
            };
            // CPU_TYPE_X86_64 = 0x01000007
            // CPU_TYPE_ARM64  = 0x0100000C
            match cputype {
                0x01000007 => Ok(CpuArchitecture::X86_64),
                0x0100000C => Ok(CpuArchitecture::AArch64),
                other => bail!("unsupported Mach-O cputype: 0x{:08x}", other),
            }
        }
        BinaryFormat::Fat => {
            bail!("fat binary architecture detection requires slice extraction first")
        }
        BinaryFormat::Pe => {
            // PE COFF Machine field is at e_lfanew + 4 (PE signature) + 0 (COFF header Machine)
            if data.len() < 64 {
                bail!("PE too short to read e_lfanew");
            }
            let e_lfanew = u32::from_le_bytes([data[60], data[61], data[62], data[63]]) as usize;
            if e_lfanew + 6 > data.len() {
                bail!("PE too short to read COFF Machine field");
            }
            let machine = u16::from_le_bytes([data[e_lfanew + 4], data[e_lfanew + 5]]);
            match machine {
                0x014C => Ok(CpuArchitecture::X86),
                0x8664 => Ok(CpuArchitecture::X86_64),
                0xAA64 => Ok(CpuArchitecture::AArch64),
                other => bail!("unsupported PE Machine type: 0x{:04x}", other),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_elf() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(b"\x7fELF");
        let result = detect_format(&data);
        assert!(result.is_ok(), "expected Ok for ELF magic");
        assert_eq!(
            result.expect("ELF format detection should succeed"),
            BinaryFormat::Elf
        );
    }

    #[test]
    fn test_detect_format_macho_le32() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(b"\xfe\xed\xfa\xce");
        let result = detect_format(&data);
        assert!(result.is_ok(), "expected Ok for Mach-O LE32 magic");
        assert_eq!(
            result.expect("Mach-O LE32 format detection should succeed"),
            BinaryFormat::MachO
        );
    }

    #[test]
    fn test_detect_format_macho_le64() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(b"\xfe\xed\xfa\xcf");
        let result = detect_format(&data);
        assert!(result.is_ok(), "expected Ok for Mach-O LE64 magic");
        assert_eq!(
            result.expect("Mach-O LE64 format detection should succeed"),
            BinaryFormat::MachO
        );
    }

    #[test]
    fn test_detect_format_macho_be32() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(b"\xce\xfa\xed\xfe");
        let result = detect_format(&data);
        assert!(result.is_ok(), "expected Ok for Mach-O BE32 magic");
        assert_eq!(
            result.expect("Mach-O BE32 format detection should succeed"),
            BinaryFormat::MachO
        );
    }

    #[test]
    fn test_detect_format_unknown() {
        let data = vec![0u8; 8];
        let result = detect_format(&data);
        assert!(result.is_err(), "expected Err for unknown magic");
    }

    #[test]
    fn test_detect_format_too_short() {
        let data = b"\x7f";
        let result = detect_format(data);
        assert!(result.is_err(), "expected Err for file too short");
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn test_detect_arch_x86_64() {
        let data = std::fs::read("/usr/bin/true").expect("could not read /usr/bin/true");
        let result = detect_architecture(&data);
        assert!(
            result.is_ok(),
            "expected Ok for /usr/bin/true: {:?}",
            result.err()
        );
        assert_eq!(
            result.expect("x86_64 architecture detection should succeed"),
            CpuArchitecture::X86_64
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    #[test]
    fn test_detect_arch_aarch64_system() {
        let data = std::fs::read("/usr/bin/true").expect("could not read /usr/bin/true");
        let result = detect_architecture(&data);
        assert!(
            result.is_ok(),
            "expected Ok for /usr/bin/true: {:?}",
            result.err()
        );
        assert_eq!(
            result.expect("aarch64 architecture detection should succeed"),
            CpuArchitecture::AArch64
        );
    }

    #[test]
    fn test_detect_format_macho_cigam64() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(b"\xcf\xfa\xed\xfe");
        assert_eq!(detect_format(&data).unwrap(), BinaryFormat::MachO);
    }

    #[test]
    fn test_detect_format_fat() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(b"\xca\xfe\xba\xbe");
        assert_eq!(detect_format(&data).unwrap(), BinaryFormat::Fat);
    }

    #[test]
    fn test_detect_format_fat_64() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(b"\xca\xfe\xba\xbf");
        assert_eq!(detect_format(&data).unwrap(), BinaryFormat::Fat);
    }

    #[test]
    fn test_detect_format_pe() {
        // Minimal MZ + PE signature
        let mut data = vec![0u8; 128];
        data[0] = 0x4D; // 'M'
        data[1] = 0x5A; // 'Z'
        data[60..64].copy_from_slice(&64u32.to_le_bytes()); // e_lfanew = 64
        data[64..68].copy_from_slice(b"PE\0\0");
        assert_eq!(detect_format(&data).unwrap(), BinaryFormat::Pe);
    }

    #[test]
    fn test_detect_format_mz_without_pe_sig() {
        // MZ header but no valid PE signature — still detected as PE
        let mut data = vec![0u8; 8];
        data[0] = 0x4D;
        data[1] = 0x5A;
        assert_eq!(detect_format(&data).unwrap(), BinaryFormat::Pe);
    }

    #[test]
    fn test_detect_format_empty() {
        assert!(detect_format(&[]).is_err());
    }

    #[test]
    fn test_detect_format_three_bytes() {
        assert!(detect_format(&[0x7f, b'E', b'L']).is_err());
    }

    #[test]
    fn test_detect_arch_elf_x86_64() {
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[18..20].copy_from_slice(&0x3Eu16.to_le_bytes()); // e_machine = EM_X86_64
        assert_eq!(detect_architecture(&data).unwrap(), CpuArchitecture::X86_64);
    }

    #[test]
    fn test_detect_arch_elf_aarch64() {
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[18..20].copy_from_slice(&0xB7u16.to_le_bytes()); // e_machine = EM_AARCH64
        assert_eq!(
            detect_architecture(&data).unwrap(),
            CpuArchitecture::AArch64
        );
    }

    #[test]
    fn test_detect_arch_elf_unsupported() {
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[18..20].copy_from_slice(&0x28u16.to_le_bytes()); // e_machine = EM_ARM (32-bit)
        assert!(detect_architecture(&data).is_err());
    }

    #[test]
    fn test_detect_arch_macho_le_x86_64() {
        let mut data = vec![0u8; 12];
        data[0..4].copy_from_slice(b"\xcf\xfa\xed\xfe"); // MH_CIGAM_64 (little-endian)
        data[4..8].copy_from_slice(&0x0100_0007u32.to_le_bytes()); // CPU_TYPE_X86_64
        assert_eq!(detect_architecture(&data).unwrap(), CpuArchitecture::X86_64);
    }

    #[test]
    fn test_detect_arch_macho_le_arm64() {
        let mut data = vec![0u8; 12];
        data[0..4].copy_from_slice(b"\xcf\xfa\xed\xfe"); // MH_CIGAM_64 (little-endian)
        data[4..8].copy_from_slice(&0x0100_000Cu32.to_le_bytes()); // CPU_TYPE_ARM64
        assert_eq!(
            detect_architecture(&data).unwrap(),
            CpuArchitecture::AArch64
        );
    }

    #[test]
    fn test_detect_arch_macho_be_x86_64() {
        let mut data = vec![0u8; 12];
        data[0..4].copy_from_slice(b"\xfe\xed\xfa\xcf"); // MH_MAGIC_64 (big-endian)
        data[4..8].copy_from_slice(&0x0100_0007u32.to_be_bytes()); // CPU_TYPE_X86_64
        assert_eq!(detect_architecture(&data).unwrap(), CpuArchitecture::X86_64);
    }

    #[test]
    fn test_detect_arch_pe_x86_64() {
        let mut data = vec![0u8; 128];
        data[0] = 0x4D;
        data[1] = 0x5A;
        data[60..64].copy_from_slice(&64u32.to_le_bytes());
        data[64..68].copy_from_slice(b"PE\0\0");
        data[68..70].copy_from_slice(&0x8664u16.to_le_bytes()); // IMAGE_FILE_MACHINE_AMD64
        assert_eq!(detect_architecture(&data).unwrap(), CpuArchitecture::X86_64);
    }

    #[test]
    fn test_detect_arch_pe_x86() {
        let mut data = vec![0u8; 128];
        data[0] = 0x4D;
        data[1] = 0x5A;
        data[60..64].copy_from_slice(&64u32.to_le_bytes());
        data[64..68].copy_from_slice(b"PE\0\0");
        data[68..70].copy_from_slice(&0x014Cu16.to_le_bytes()); // IMAGE_FILE_MACHINE_I386
        assert_eq!(detect_architecture(&data).unwrap(), CpuArchitecture::X86);
    }

    #[test]
    fn test_detect_arch_pe_arm64() {
        let mut data = vec![0u8; 128];
        data[0] = 0x4D;
        data[1] = 0x5A;
        data[60..64].copy_from_slice(&64u32.to_le_bytes());
        data[64..68].copy_from_slice(b"PE\0\0");
        data[68..70].copy_from_slice(&0xAA64u16.to_le_bytes()); // IMAGE_FILE_MACHINE_ARM64
        assert_eq!(
            detect_architecture(&data).unwrap(),
            CpuArchitecture::AArch64
        );
    }

    #[test]
    fn test_detect_arch_elf_truncated() {
        let data = b"\x7fELF"; // 4 bytes — not enough for e_machine at offset 18
        assert!(detect_architecture(data).is_err());
    }

    #[test]
    fn test_detect_arch_fat_errors() {
        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(b"\xca\xfe\xba\xbe");
        assert!(detect_architecture(&data).is_err()); // Fat needs slice extraction
    }
}
