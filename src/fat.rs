//! Fat (universal) binary support: extract slices, instrument each, re-package.

use crate::RewriteConfig;
use anyhow::{Context, Result, bail};

/// A single slice in a fat binary.
#[derive(Debug)]
struct FatSlice {
    cpu_type: u32,
    cpu_subtype: u32,
    offset: usize,
    size: usize,
    align: u32,
}

/// Detect whether data starts with fat binary magic.
pub fn is_fat_binary(data: &[u8]) -> bool {
    if data.len() < 8 {
        return false;
    }
    let magic = u32::from_be_bytes(data[0..4].try_into().expect("slice is 4 bytes"));
    magic == 0xCAFEBABE || magic == 0xCAFEBABF
}

/// Parse fat header and extract slice descriptors.
fn parse_fat_header(data: &[u8]) -> Result<Vec<FatSlice>> {
    if data.len() < 8 {
        bail!("fat binary too short");
    }
    let magic = u32::from_be_bytes(data[0..4].try_into().expect("slice is 4 bytes"));
    let is_64 = magic == 0xCAFEBABF;
    let nfat_arch = u32::from_be_bytes(data[4..8].try_into().expect("slice is 4 bytes"));

    if nfat_arch == 0 || nfat_arch > 16 {
        bail!("invalid fat_header nfat_arch: {}", nfat_arch);
    }

    let mut slices = Vec::new();
    let mut offset = 8usize;

    for _ in 0..nfat_arch {
        if is_64 {
            // fat_arch_64: 32 bytes each
            if offset + 32 > data.len() {
                bail!("fat binary truncated in fat_arch_64 entries");
            }
            let cpu_type = u32::from_be_bytes(
                data[offset..offset + 4]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            let cpu_subtype = u32::from_be_bytes(
                data[offset + 4..offset + 8]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            let slice_offset = u64::from_be_bytes(
                data[offset + 8..offset + 16]
                    .try_into()
                    .expect("slice is 8 bytes"),
            ) as usize;
            let slice_size = u64::from_be_bytes(
                data[offset + 16..offset + 24]
                    .try_into()
                    .expect("slice is 8 bytes"),
            ) as usize;
            let align = u32::from_be_bytes(
                data[offset + 24..offset + 28]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            slices.push(FatSlice {
                cpu_type,
                cpu_subtype,
                offset: slice_offset,
                size: slice_size,
                align,
            });
            offset += 32;
        } else {
            // fat_arch: 20 bytes each
            if offset + 20 > data.len() {
                bail!("fat binary truncated in fat_arch entries");
            }
            let cpu_type = u32::from_be_bytes(
                data[offset..offset + 4]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            let cpu_subtype = u32::from_be_bytes(
                data[offset + 4..offset + 8]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            let slice_offset = u32::from_be_bytes(
                data[offset + 8..offset + 12]
                    .try_into()
                    .expect("slice is 4 bytes"),
            ) as usize;
            let slice_size = u32::from_be_bytes(
                data[offset + 12..offset + 16]
                    .try_into()
                    .expect("slice is 4 bytes"),
            ) as usize;
            let align = u32::from_be_bytes(
                data[offset + 16..offset + 20]
                    .try_into()
                    .expect("slice is 4 bytes"),
            );
            slices.push(FatSlice {
                cpu_type,
                cpu_subtype,
                offset: slice_offset,
                size: slice_size,
                align,
            });
            offset += 20;
        }
    }

    Ok(slices)
}

/// Rewrite a fat binary by instrumenting each Mach-O slice individually.
pub fn rewrite_fat(config: &RewriteConfig) -> Result<crate::RewriteResult> {
    let data = std::fs::read(&config.input)
        .with_context(|| format!("failed to read {}", config.input.display()))?;

    let slices = parse_fat_header(&data)?;
    tracing::info!("fat binary: {} slices", slices.len());

    let mut instrumented: Vec<(u32, u32, u32, Vec<u8>)> = Vec::new();
    let mut total_blocks = 0usize;
    let mut total_skipped = 0usize;
    let mut last_segment_va = 0u64;
    let mut last_segment_size = 0u64;

    for (i, slice) in slices.iter().enumerate() {
        let end = slice.offset + slice.size;
        if end > data.len() {
            bail!(
                "fat slice {} extends beyond file ({} + {} > {})",
                i,
                slice.offset,
                slice.size,
                data.len()
            );
        }
        let slice_data = &data[slice.offset..end];

        tracing::info!(
            "instrumenting fat slice {}: cputype=0x{:08x} offset=0x{:x} size={}",
            i,
            slice.cpu_type,
            slice.offset,
            slice.size,
        );

        // Write slice to temp file, instrument it, read back
        let tmp_in = std::path::PathBuf::from(format!("/tmp/truant_fat_slice_{}_in", i));
        let tmp_out = std::path::PathBuf::from(format!("/tmp/truant_fat_slice_{}_out", i));
        std::fs::write(&tmp_in, slice_data)?;

        let slice_config = RewriteConfig {
            input: tmp_in.clone(),
            output: tmp_out.clone(),
            forkserver: config.forkserver,
            dry_run: false,
            heap_san: false,
            sidecar_san: false,
            persistent_addr: None,
            persistent_count: config.persistent_count,
            defer: false,
            #[cfg(feature = "coverage")]
            #[cfg(feature = "coverage")]
            validate: false,
            instrument_modules: config.instrument_modules.clone(),
            hooks: config.hooks.clone(),
            no_coverage: config.no_coverage,
        };

        let result = crate::rewrite(&slice_config)
            .with_context(|| format!("failed to instrument fat slice {}", i))?;

        total_blocks += result.blocks_instrumented;
        total_skipped += result.blocks_skipped;
        last_segment_va = result.segment_va;
        last_segment_size = result.segment_size;

        let instrumented_data = std::fs::read(&tmp_out)?;
        instrumented.push((
            slice.cpu_type,
            slice.cpu_subtype,
            slice.align,
            instrumented_data,
        ));

        let _ = std::fs::remove_file(&tmp_in);
        let _ = std::fs::remove_file(&tmp_out);
    }

    // Re-package as fat binary
    let fat_data = build_fat_binary(&instrumented)?;
    std::fs::write(&config.output, &fat_data)
        .with_context(|| format!("failed to write {}", config.output.display()))?;

    // Set executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&config.output)?.permissions();
        perms.set_mode(perms.mode() | 0o111);
        std::fs::set_permissions(&config.output, perms)?;
    }

    // Ad-hoc codesign (fat binaries contain Mach-O slices)
    crate::codesign_if_available(&config.output);

    tracing::info!(
        "wrote fat binary {} ({} bytes, {} slices, {} total blocks)",
        config.output.display(),
        fat_data.len(),
        instrumented.len(),
        total_blocks,
    );

    Ok(crate::RewriteResult {
        blocks_instrumented: total_blocks,
        blocks_skipped: total_skipped,
        segment_va: last_segment_va,
        segment_size: last_segment_size,
        output_path: Some(config.output.clone()),
        heap_san_intercepted: 0,
        preload_lib_path: None,
        sidecar_san_lib_path: None,
        persistent_mode: config.persistent_addr.is_some(),
        hooks_applied: 0,
        hook_preload_lib_path: None,
    })
}

/// Build a fat binary from instrumented slices.
fn build_fat_binary(slices: &[(u32, u32, u32, Vec<u8>)]) -> Result<Vec<u8>> {
    let nfat = slices.len();
    // fat_header (8) + fat_arch entries (20 each)
    let header_size = 8 + nfat * 20;

    let mut result = Vec::new();

    // fat_header
    result.extend_from_slice(&0xCAFEBABEu32.to_be_bytes()); // magic
    result.extend_from_slice(&(nfat as u32).to_be_bytes()); // nfat_arch

    // Calculate slice offsets: each slice aligned to its alignment
    let mut current_offset = header_size;
    let mut offsets = Vec::new();

    for (_, _, align, data) in slices {
        let alignment = 1u64 << (*align).min(14); // cap at 16KB
        current_offset = ((current_offset as u64 + alignment - 1) & !(alignment - 1)) as usize;
        offsets.push(current_offset);
        current_offset += data.len();
    }

    // Write fat_arch entries
    for (i, (cpu_type, cpu_subtype, align, data)) in slices.iter().enumerate() {
        result.extend_from_slice(&cpu_type.to_be_bytes());
        result.extend_from_slice(&cpu_subtype.to_be_bytes());
        result.extend_from_slice(&(offsets[i] as u32).to_be_bytes());
        result.extend_from_slice(&(data.len() as u32).to_be_bytes());
        result.extend_from_slice(&align.to_be_bytes());
    }

    // Write slice data with padding
    for (i, (_, _, _, data)) in slices.iter().enumerate() {
        let target = offsets[i];
        while result.len() < target {
            result.push(0);
        }
        result.extend_from_slice(data);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_fat_binary() {
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(&0xCAFEBABEu32.to_be_bytes());
        assert!(is_fat_binary(&data));

        data[0..4].copy_from_slice(&0xCAFEBABFu32.to_be_bytes());
        assert!(is_fat_binary(&data));

        data[0..4].copy_from_slice(&0xFEEDFACFu32.to_le_bytes());
        assert!(!is_fat_binary(&data));
    }

    #[test]
    fn test_parse_fat_header() {
        let mut data = vec![0u8; 48];
        // magic = 0xCAFEBABE
        data[0..4].copy_from_slice(&0xCAFEBABEu32.to_be_bytes());
        // nfat_arch = 1
        data[4..8].copy_from_slice(&1u32.to_be_bytes());
        // fat_arch: cputype
        data[8..12].copy_from_slice(&0x01000007u32.to_be_bytes());
        // cpusubtype
        data[12..16].copy_from_slice(&3u32.to_be_bytes());
        // offset
        data[16..20].copy_from_slice(&28u32.to_be_bytes());
        // size
        data[20..24].copy_from_slice(&16u32.to_be_bytes());
        // align
        data[24..28].copy_from_slice(&14u32.to_be_bytes());

        let slices = parse_fat_header(&data).expect("fat header should parse successfully");
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].cpu_type, 0x01000007);
        assert_eq!(slices[0].offset, 28);
        assert_eq!(slices[0].size, 16);
    }
}
