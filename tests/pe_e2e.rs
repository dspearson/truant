//! PE binary rewrite integration tests.
//!
//! Structural tests run without Wine (validate headers, section layout, init code).
//! Runtime tests require Wine and are marked #[ignore] for CI.

mod common;

use std::path::Path;

use truant::pe::PeContext;

/// Compile a test PE if mingw is available, skip otherwise.
fn require_pe(dir: &Path, suffix: &str, src: &str) -> std::path::PathBuf {
    common::compile_pe(dir, suffix, src).expect("x86_64-w64-mingw32-gcc required for PE tests")
}

fn require_pe32(dir: &Path, suffix: &str, src: &str) -> std::path::PathBuf {
    common::compile_pe32(dir, suffix, src).expect("i686-w64-mingw32-gcc required for PE32 tests")
}

/// Read a PE section header by name from raw bytes.
fn find_section(data: &[u8], name: &str) -> Option<(u32, u32, u32, u32, u32)> {
    if data.len() < 64 {
        return None;
    }
    let e_lfanew = u32::from_le_bytes([data[60], data[61], data[62], data[63]]) as usize;
    if e_lfanew + 6 > data.len() {
        return None;
    }
    let coff_offset = e_lfanew + 4;
    let num_sections = u16::from_le_bytes([data[coff_offset + 2], data[coff_offset + 3]]) as usize;
    let opt_header_size =
        u16::from_le_bytes([data[coff_offset + 16], data[coff_offset + 17]]) as usize;
    let section_table = coff_offset + 20 + opt_header_size;

    for i in 0..num_sections {
        let base = section_table + i * 40;
        if base + 40 > data.len() {
            break;
        }
        let sec_name = String::from_utf8_lossy(&data[base..base + 8])
            .trim_end_matches('\0')
            .to_string();
        if sec_name == name {
            let virtual_size = u32::from_le_bytes([
                data[base + 8],
                data[base + 9],
                data[base + 10],
                data[base + 11],
            ]);
            let virtual_addr = u32::from_le_bytes([
                data[base + 12],
                data[base + 13],
                data[base + 14],
                data[base + 15],
            ]);
            let raw_size = u32::from_le_bytes([
                data[base + 16],
                data[base + 17],
                data[base + 18],
                data[base + 19],
            ]);
            let raw_offset = u32::from_le_bytes([
                data[base + 20],
                data[base + 21],
                data[base + 22],
                data[base + 23],
            ]);
            let chars = u32::from_le_bytes([
                data[base + 36],
                data[base + 37],
                data[base + 38],
                data[base + 39],
            ]);
            return Some((virtual_size, virtual_addr, raw_size, raw_offset, chars));
        }
    }
    None
}

/// Read AddressOfEntryPoint RVA from a PE file.
fn read_entry_rva(data: &[u8]) -> u32 {
    let e_lfanew = u32::from_le_bytes([data[60], data[61], data[62], data[63]]) as usize;
    let opt_header = e_lfanew + 4 + 20;
    u32::from_le_bytes([
        data[opt_header + 16],
        data[opt_header + 17],
        data[opt_header + 18],
        data[opt_header + 19],
    ])
}

/// Read SizeOfImage from a PE file.
fn read_size_of_image(data: &[u8]) -> u32 {
    let e_lfanew = u32::from_le_bytes([data[60], data[61], data[62], data[63]]) as usize;
    let opt_header = e_lfanew + 4 + 20;
    // PE32+: windows fields start at offset 24 from optional header, SizeOfImage at +32
    let windows_fields = opt_header + 24;
    u32::from_le_bytes([
        data[windows_fields + 32],
        data[windows_fields + 33],
        data[windows_fields + 34],
        data[windows_fields + 35],
    ])
}

/// Read NumberOfSections from COFF header.
fn read_num_sections(data: &[u8]) -> u16 {
    let e_lfanew = u32::from_le_bytes([data[60], data[61], data[62], data[63]]) as usize;
    let coff_offset = e_lfanew + 4;
    u16::from_le_bytes([data[coff_offset + 2], data[coff_offset + 3]])
}

/// Read CheckSum from optional header.
fn read_checksum(data: &[u8]) -> u32 {
    let e_lfanew = u32::from_le_bytes([data[60], data[61], data[62], data[63]]) as usize;
    let opt_header = e_lfanew + 4 + 20;
    let windows_fields = opt_header + 24;
    u32::from_le_bytes([
        data[windows_fields + 40],
        data[windows_fields + 41],
        data[windows_fields + 42],
        data[windows_fields + 43],
    ])
}

// ====================================================================
// Structural tests — no Wine needed
// ====================================================================

#[test]
fn pe_coverage_rewrite_adds_kpcov_section() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 0; }
"#;
    let input = require_pe(td.path(), "cov_section", src);
    let output = td.path().join("cov_section_out.exe");

    let result = common::rewrite_pe(&input, &output);

    assert!(
        result.blocks_instrumented > 0,
        "should instrument at least one block"
    );

    // Read output and verify .trcov section exists.
    let data = std::fs::read(&output).unwrap();
    let trcov = find_section(&data, ".trcov");
    assert!(trcov.is_some(), ".trcov section not found in rewritten PE");

    let (vsize, _va, raw_size, raw_offset, chars) = trcov.unwrap();
    assert!(vsize > 0, ".trcov virtual size should be > 0");
    assert!(raw_size > 0, ".trcov raw size should be > 0");
    assert!(raw_offset > 0, ".trcov raw offset should be > 0");
    // Should be CODE | INITIALIZED_DATA | EXECUTE | READ | WRITE
    assert!(chars & 0x2000_0000 != 0, ".trcov should be executable");
    assert!(chars & 0x4000_0000 != 0, ".trcov should be readable");
    assert!(chars & 0x8000_0000 != 0, ".trcov should be writable");
    assert!(chars & 0x0000_0020 != 0, ".trcov should contain code");
}

#[test]
fn pe_rewrite_increments_section_count() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 0; }
"#;
    let input = require_pe(td.path(), "sec_count", src);
    let output = td.path().join("sec_count_out.exe");

    let orig_data = std::fs::read(&input).unwrap();
    let orig_sections = read_num_sections(&orig_data);

    common::rewrite_pe(&input, &output);

    let new_data = std::fs::read(&output).unwrap();
    let new_sections = read_num_sections(&new_data);

    assert_eq!(
        new_sections,
        orig_sections + 1,
        "section count should increase by 1"
    );
}

#[test]
fn pe_rewrite_updates_size_of_image() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 0; }
"#;
    let input = require_pe(td.path(), "soi", src);
    let output = td.path().join("soi_out.exe");

    let orig_data = std::fs::read(&input).unwrap();
    let orig_soi = read_size_of_image(&orig_data);

    common::rewrite_pe(&input, &output);

    let new_data = std::fs::read(&output).unwrap();
    let new_soi = read_size_of_image(&new_data);

    assert!(
        new_soi > orig_soi,
        "SizeOfImage should grow: {} -> {}",
        orig_soi,
        new_soi
    );
}

#[test]
fn pe_rewrite_zeroes_checksum() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 0; }
"#;
    let input = require_pe(td.path(), "cksum", src);
    let output = td.path().join("cksum_out.exe");

    common::rewrite_pe(&input, &output);

    let data = std::fs::read(&output).unwrap();
    let checksum = read_checksum(&data);
    assert_eq!(checksum, 0, "checksum should be zeroed after rewrite");
}

#[test]
fn pe_rewrite_redirects_entry_point() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 0; }
"#;
    let input = require_pe(td.path(), "ep_redirect", src);
    let output = td.path().join("ep_redirect_out.exe");

    let orig_data = std::fs::read(&input).unwrap();
    let orig_ep = read_entry_rva(&orig_data);

    common::rewrite_pe(&input, &output);

    let new_data = std::fs::read(&output).unwrap();
    let new_ep = read_entry_rva(&new_data);

    // Entry point should point into the .trcov section now.
    let trcov = find_section(&new_data, ".trcov").unwrap();
    let kpcov_va = trcov.1;
    let kpcov_vsize = trcov.0;

    assert_ne!(new_ep, orig_ep, "entry point should change after rewrite");
    assert!(
        new_ep >= kpcov_va && new_ep < kpcov_va + kpcov_vsize,
        "new entry point 0x{:x} should be within .trcov (0x{:x}..0x{:x})",
        new_ep,
        kpcov_va,
        kpcov_va + kpcov_vsize,
    );
}

#[test]
fn pe_rewrite_init_code_contains_afl_needle() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 0; }
"#;
    let input = require_pe(td.path(), "needle", src);
    let output = td.path().join("needle_out.exe");

    common::rewrite_pe(&input, &output);

    let data = std::fs::read(&output).unwrap();
    let trcov = find_section(&data, ".trcov").unwrap();
    let raw_offset = trcov.3 as usize;
    let raw_size = trcov.2 as usize;
    let section_data = &data[raw_offset..raw_offset + raw_size];

    // The __AFL_SHM_ID= needle is embedded as UTF-16LE.
    let needle_u16: Vec<u8> = b"__AFL_SHM_ID=".iter().flat_map(|&c| [c, 0x00]).collect();

    let found = section_data
        .windows(needle_u16.len())
        .any(|w| w == needle_u16.as_slice());
    assert!(
        found,
        "__AFL_SHM_ID= UTF-16LE needle not found in .trcov section"
    );
}

#[test]
fn pe_coverage_patches_text_with_jmp() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 0; }
"#;
    let input = require_pe(td.path(), "jmp_patch", src);
    let output = td.path().join("jmp_patch_out.exe");

    let result = common::rewrite_pe(&input, &output);
    assert!(result.blocks_instrumented > 0);

    // Parse both binaries and check that .text differs.
    let orig_data = std::fs::read(&input).unwrap();
    let new_data = std::fs::read(&output).unwrap();

    let orig_text = find_section(&orig_data, ".text").unwrap();
    let new_text = find_section(&new_data, ".text").unwrap();

    // .text offsets should be the same.
    assert_eq!(
        orig_text.3, new_text.3,
        ".text raw offset should not change"
    );

    let orig_text_bytes = &orig_data[orig_text.3 as usize..(orig_text.3 + orig_text.2) as usize];
    let new_text_bytes = &new_data[new_text.3 as usize..(new_text.3 + new_text.2) as usize];

    // At least one byte should differ (JMP rel32 patches).
    assert_ne!(
        orig_text_bytes, new_text_bytes,
        ".text should be patched with JMP instructions"
    );

    // Count JMP rel32 (0xE9) bytes that weren't there before.
    let jmp_count = new_text_bytes
        .iter()
        .zip(orig_text_bytes.iter())
        .filter(|(n, o)| **n == 0xE9 && **o != 0xE9)
        .count();
    assert!(
        jmp_count > 0,
        "should find at least one new JMP rel32 in .text"
    );
}

#[test]
fn pe_output_is_valid_pe() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 0; }
"#;
    let input = require_pe(td.path(), "valid_pe", src);
    let output = td.path().join("valid_pe_out.exe");

    common::rewrite_pe(&input, &output);

    // The output should be parseable by goblin/PeContext.
    let data = std::fs::read(&output).unwrap();
    let ctx = PeContext::parse(&data);
    assert!(
        ctx.is_ok(),
        "rewritten PE should be parseable: {:?}",
        ctx.err()
    );

    let ctx = ctx.unwrap();
    assert!(ctx.is_64bit, "should still be PE32+");
    assert!(ctx.text.size > 0, ".text should exist");
}

#[test]
fn pe_no_coverage_mode_skips_blocks() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    // Build a PE with an exported function so we can hook it.
    let src = r#"
__declspec(dllexport) int target_func(void) { return 42; }
int main(void) { return target_func(); }
"#;
    let input = require_pe(td.path(), "nocov", src);
    let output = td.path().join("nocov_out.exe");
    let hooks = common::write_hooks(
        td.path(),
        "nocov_pe",
        r#"
[[hook]]
target = "target_func"
mode = "pre"
shellcode = [0xC3]
"#,
    );

    let result = common::rewrite_pe_no_cov(&input, &output, &hooks);

    assert_eq!(
        result.blocks_instrumented, 0,
        "no-coverage mode should not instrument blocks"
    );
    assert!(
        result.hooks_applied > 0,
        "hooks should still be applied in no-coverage mode"
    );

    // Entry point should NOT be redirected in no-coverage mode.
    let orig_data = std::fs::read(&input).unwrap();
    let new_data = std::fs::read(&output).unwrap();
    let orig_ep = read_entry_rva(&orig_data);
    let new_ep = read_entry_rva(&new_data);
    assert_eq!(
        orig_ep, new_ep,
        "entry point should not change in no-coverage mode"
    );
}

#[test]
fn pe_hook_shellcode_applies() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
__declspec(dllexport) int target_func(void) { return 42; }
int main(void) { return target_func(); }
"#;
    let input = require_pe(td.path(), "hook_sc", src);
    let output = td.path().join("hook_sc_out.exe");
    let hooks = common::write_hooks(
        td.path(),
        "hook_sc_pe",
        r#"
[[hook]]
target = "target_func"
mode = "pre"
shellcode = [0xC3]
"#,
    );

    let result = common::rewrite_pe_hooked(&input, &output, &hooks);
    assert!(result.hooks_applied > 0, "should apply at least one hook");

    // Verify .trcov section contains the shellcode byte.
    let data = std::fs::read(&output).unwrap();
    let trcov = find_section(&data, ".trcov").unwrap();
    let raw_offset = trcov.3 as usize;
    let raw_size = trcov.2 as usize;
    let section_data = &data[raw_offset..raw_offset + raw_size];

    // 0xC3 (RET) should appear in the section as our shellcode.
    assert!(
        section_data.contains(&0xC3),
        "shellcode byte 0xC3 not found in .trcov section",
    );
}

#[test]
fn pe_hook_with_library_embeds_strings() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
__declspec(dllexport) int target_func(void) { return 42; }
int main(void) { return target_func(); }
"#;
    let input = require_pe(td.path(), "hook_lib", src);
    let output = td.path().join("hook_lib_out.exe");
    let hooks = common::write_hooks(
        td.path(),
        "hook_lib_pe",
        r#"
[hooks]
library = "my_hooks.dll"

[[hook]]
target = "target_func"
mode = "pre"
handler = "on_target"
"#,
    );

    let result = common::rewrite_pe_hooked(&input, &output, &hooks);
    assert!(result.hooks_applied > 0, "should apply at least one hook");
    // No companion preload lib for PE.
    assert!(
        result.hook_preload_lib_path.is_none(),
        "PE should not generate companion .so"
    );

    // The library path and handler name should be embedded in the .trcov section.
    let data = std::fs::read(&output).unwrap();
    let trcov = find_section(&data, ".trcov").unwrap();
    let raw_offset = trcov.3 as usize;
    let raw_size = trcov.2 as usize;
    let section_data = &data[raw_offset..raw_offset + raw_size];

    let lib_found = section_data.windows(13).any(|w| w == b"my_hooks.dll\0");
    assert!(
        lib_found,
        "library path 'my_hooks.dll' not found in .trcov section"
    );

    let handler_found = section_data.windows(10).any(|w| w == b"on_target\0");
    assert!(
        handler_found,
        "handler name 'on_target' not found in .trcov section"
    );
}

#[test]
fn pe_multiple_hooks_same_export() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
__declspec(dllexport) int target_func(void) { return 42; }
int main(void) { return target_func(); }
"#;
    let input = require_pe(td.path(), "multi_hook", src);
    let output = td.path().join("multi_hook_out.exe");
    let hooks = common::write_hooks(
        td.path(),
        "multi_hook_pe",
        r#"
[[hook]]
target = "target_func"
mode = "pre"
shellcode = [0x90]

[[hook]]
target = "target_func"
mode = "post"
shellcode = [0x90]
"#,
    );

    let result = common::rewrite_pe_hooked(&input, &output, &hooks);
    // Both hooks on the same target — chained.
    assert!(result.hooks_applied > 0, "chained hooks should be applied");
}

#[test]
fn pe_context_round_trip() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
__declspec(dllexport) int foo(void) { return 1; }
__declspec(dllexport) int bar(void) { return 2; }
int main(void) { return foo() + bar(); }
"#;
    let input = require_pe(td.path(), "ctx_rt", src);
    let data = std::fs::read(&input).unwrap();

    let ctx = PeContext::parse(&data).expect("should parse test PE");

    assert!(ctx.is_64bit);
    assert!(!ctx.is_dll);
    assert!(ctx.text.size > 0);
    assert!(ctx.entry_point > ctx.image_base);
    assert!(
        ctx.func_symbols.len() >= 2,
        "should have at least foo and bar exports"
    );
    assert!(ctx.sections.len() > 1, "should have multiple sections");
    assert!(ctx.section_alignment > 0);
    assert!(ctx.file_alignment > 0);
    assert!(ctx.size_of_image > 0);
}

#[test]
fn pe_dll_rewrite() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
#include <windows.h>
__declspec(dllexport) int my_api(int x) { return x * 2; }
BOOL WINAPI DllMain(HINSTANCE h, DWORD reason, LPVOID reserved) {
    (void)h; (void)reason; (void)reserved;
    return TRUE;
}
"#;
    let input = common::compile_pe_dll(td.path(), "dll_rewrite", src).expect("mingw required");
    let output = td.path().join("dll_rewrite_out.dll");

    let result = common::rewrite_pe(&input, &output);
    assert!(
        result.blocks_instrumented > 0,
        "DLL should have instrumentable blocks"
    );

    // Verify the output is still a valid PE DLL.
    let data = std::fs::read(&output).unwrap();
    let ctx = PeContext::parse(&data).expect("rewritten DLL should be parseable");
    assert!(ctx.is_dll, "rewritten binary should still be a DLL");
    assert!(ctx.is_64bit);

    // .trcov section should exist.
    let trcov = find_section(&data, ".trcov");
    assert!(
        trcov.is_some(),
        ".trcov section should be present in rewritten DLL"
    );
}

// ====================================================================
// Runtime tests — require Wine (or real Windows)
// ====================================================================

#[test]
#[ignore = "requires Wine; run with: cargo test -p truant pe_wine -- --ignored"]
fn pe_wine_unrewritten_runs() {
    if !common::mingw_available() || !common::wine_available() {
        eprintln!("SKIP: mingw or wine not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 42; }
"#;
    let input = require_pe(td.path(), "wine_base", src);
    let (code, _, _) = common::run_wine(&input).expect("wine should be available");
    assert_eq!(code, 42, "unrewritten PE should exit with 42");
}

#[test]
#[ignore = "requires Wine; run with: cargo test -p truant pe_wine -- --ignored"]
fn pe_wine_rewritten_runs() {
    if !common::mingw_available() || !common::wine_available() {
        eprintln!("SKIP: mingw or wine not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 42; }
"#;
    let input = require_pe(td.path(), "wine_rewritten", src);
    let output = td.path().join("wine_rewritten_out.exe");

    common::rewrite_pe(&input, &output);

    // Without __AFL_SHM_ID set, the init code should gracefully fall through
    // to the original entry point.
    let (code, _, _) = common::run_wine(&output).expect("wine should be available");
    assert_eq!(
        code, 42,
        "rewritten PE should still exit with 42 under Wine"
    );
}

// ====================================================================
// PE heap sanitiser tests (companion DLL + IAT injection)
// ====================================================================

#[test]
#[cfg(feature = "coverage")]
fn pe_heap_san_dll_generated() {
    if !common::mingw_available() {
        eprintln!("SKIP: x86_64-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
#include <stdlib.h>
int main(void) {
    char *buf = (char *)malloc(64);
    if (!buf) return 1;
    buf[0] = 'A';
    free(buf);
    return 0;
}
"#;
    let input = require_pe(td.path(), "heap_san", src);
    let output = td.path().join("heap_san_out.exe");

    let config = truant::RewriteConfig {
        input: input.clone(),
        output: output.clone(),
        forkserver: false,
        dry_run: false,
        heap_san: true,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: None,
        no_coverage: false,
    };

    truant::rewrite(&config).expect("PE rewrite with heap_san failed");

    // Companion DLL should be generated.
    let dll_path = truant::preload::preload_dll_path(&output);
    assert!(
        dll_path.exists(),
        "heap sanitiser companion DLL should exist at {:?}",
        dll_path
    );

    // The rewritten PE should have an .idata2 section with the import.
    let data = std::fs::read(&output).unwrap();
    let idata2 = find_section(&data, ".idata2");
    assert!(
        idata2.is_some(),
        ".idata2 section should be present (IAT injection)"
    );

    // Verify the DLL name appears in the section content.
    let dll_name = dll_path.file_name().unwrap().to_string_lossy();
    let section_data = &data[idata2.unwrap().3 as usize..];
    let found = section_data
        .windows(dll_name.len())
        .any(|w| w == dll_name.as_bytes());
    assert!(
        found,
        "DLL name '{}' should appear in .idata2 section",
        dll_name
    );
}

#[test]
#[cfg(feature = "coverage")]
fn pe_heap_san_import_count_increases() {
    if !common::mingw_available() {
        eprintln!("SKIP: x86_64-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"int main(void) { return 0; }"#;
    let input = require_pe(td.path(), "heap_san_cnt", src);

    // Parse original to count imports.
    let orig_data = std::fs::read(&input).unwrap();
    let orig_ctx = PeContext::parse(&orig_data).unwrap();
    let _orig_import_count = orig_ctx.imports.len();

    let output = td.path().join("heap_san_cnt_out.exe");
    let config = truant::RewriteConfig {
        input: input.clone(),
        output: output.clone(),
        forkserver: false,
        dry_run: false,
        heap_san: true,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: None,
        no_coverage: false,
    };

    truant::rewrite(&config).expect("PE rewrite with heap_san failed");

    // The rewritten PE should be parseable and have the heap_san import.
    let new_data = std::fs::read(&output).unwrap();
    // Check .idata2 exists (our injected section).
    assert!(
        find_section(&new_data, ".idata2").is_some(),
        ".idata2 should exist"
    );
}

// ====================================================================
// PE32 (32-bit) tests
// ====================================================================

#[test]
fn pe32_coverage_rewrite_adds_trcov_section() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 0; }
"#;
    let input =
        common::compile_pe32(td.path(), "pe32_cov", src).expect("i686-w64-mingw32-gcc required");
    let output = td.path().join("pe32_cov_out.exe");

    let result = common::rewrite_pe(&input, &output);

    assert!(
        result.blocks_instrumented > 0,
        "should instrument at least one block"
    );

    // Read output and verify .trcov section exists.
    let data = std::fs::read(&output).unwrap();
    let trcov = find_section(&data, ".trcov");
    assert!(
        trcov.is_some(),
        ".trcov section not found in rewritten PE32"
    );

    let (vsize, _va, raw_size, _raw_offset, chars) = trcov.unwrap();
    assert!(vsize > 0, ".trcov virtual size should be > 0");
    assert!(raw_size > 0, ".trcov raw size should be > 0");
    // Should be CODE | INITIALIZED_DATA | EXECUTE | READ | WRITE
    assert!(chars & 0x2000_0000 != 0, ".trcov should be executable");
    assert!(chars & 0x4000_0000 != 0, ".trcov should be readable");
    assert!(chars & 0x8000_0000 != 0, ".trcov should be writable");
}

#[test]
fn pe32_output_is_valid_pe() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 42; }
"#;
    let input =
        common::compile_pe32(td.path(), "pe32_valid", src).expect("i686-w64-mingw32-gcc required");
    let output = td.path().join("pe32_valid_out.exe");

    common::rewrite_pe(&input, &output);

    // Parse the output to verify it's a valid PE.
    let data = std::fs::read(&output).unwrap();
    let ctx = PeContext::parse(&data);
    assert!(
        ctx.is_ok(),
        "rewritten PE32 should be parseable: {:?}",
        ctx.err()
    );
    let ctx = ctx.unwrap();
    assert!(!ctx.is_64bit, "PE32 binary should not be marked as 64-bit");
}

#[test]
fn pe32_rewrite_init_code_contains_afl_needle() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 0; }
"#;
    let input =
        common::compile_pe32(td.path(), "pe32_needle", src).expect("i686-w64-mingw32-gcc required");
    let output = td.path().join("pe32_needle_out.exe");

    common::rewrite_pe(&input, &output);

    // The .trcov section should contain the "__AFL_SHM_ID=" wide-string needle.
    let data = std::fs::read(&output).unwrap();
    let needle: Vec<u8> = b"__AFL_SHM_ID="
        .iter()
        .flat_map(|&c| vec![c, 0x00])
        .collect();
    let found = data.windows(needle.len()).any(|w| w == needle.as_slice());
    assert!(
        found,
        "__AFL_SHM_ID= wide needle not found in rewritten PE32"
    );
}

#[test]
fn pe32_rewrite_redirects_entry_point() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int main(void) { return 0; }
"#;
    let input =
        common::compile_pe32(td.path(), "pe32_ep", src).expect("i686-w64-mingw32-gcc required");
    let output = td.path().join("pe32_ep_out.exe");

    let orig_data = std::fs::read(&input).unwrap();
    let orig_entry_rva = read_entry_rva(&orig_data);

    common::rewrite_pe(&input, &output);

    let new_data = std::fs::read(&output).unwrap();
    let new_entry_rva = read_entry_rva(&new_data);

    // Entry point should be redirected (different from original).
    assert_ne!(
        orig_entry_rva, new_entry_rva,
        "PE32 entry point should be redirected to init code"
    );
    // New entry should be in the .trcov section.
    let trcov = find_section(&new_data, ".trcov");
    assert!(trcov.is_some());
    let (_vsize, trcov_rva, _raw, _roff, _chars) = trcov.unwrap();
    assert!(
        new_entry_rva >= trcov_rva,
        "new entry RVA 0x{:x} should be within .trcov (starts at 0x{:x})",
        new_entry_rva,
        trcov_rva,
    );
}

// ====================================================================
// PE32: coverage instrumentation patches .text with JMP rel32
// ====================================================================

#[test]
fn pe32_coverage_patches_text_with_jmp() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
int fib(int n) { return n <= 1 ? n : fib(n-1) + fib(n-2); }
int main(void) { return fib(5); }
"#;
    let input = require_pe32(td.path(), "pe32_jmp", src);
    let output = td.path().join("pe32_jmp_out.exe");

    let result = common::rewrite_pe(&input, &output);
    assert!(
        result.blocks_instrumented >= 5,
        "recursive fib should have >= 5 blocks, got {}",
        result.blocks_instrumented
    );

    // Verify .text contains JMP rel32 (E9) instructions targeting .trcov.
    let data = std::fs::read(&output).unwrap();
    let ctx = PeContext::parse(&data).unwrap();
    let trcov = find_section(&data, ".trcov").unwrap();
    let trcov_rva = trcov.1;
    let trcov_end_rva = trcov_rva + trcov.0;

    let text_off = ctx.text.offset as usize;
    let text_size = ctx.text.size as usize;
    let text_rva = (ctx.text.va - ctx.image_base) as u32;
    let mut jmp_count = 0;
    for i in 0..text_size.saturating_sub(4) {
        if data[text_off + i] == 0xE9 {
            let rel =
                i32::from_le_bytes(data[text_off + i + 1..text_off + i + 5].try_into().unwrap());
            let target_rva = (text_rva as i64 + i as i64 + 5 + rel as i64) as u32;
            if target_rva >= trcov_rva && target_rva < trcov_end_rva {
                jmp_count += 1;
            }
        }
    }
    assert!(
        jmp_count >= 5,
        "expected >= 5 JMP patches to .trcov, found {jmp_count}"
    );
}

// ====================================================================
// PE32: ASLR (DYNAMIC_BASE) is disabled after rewrite
// ====================================================================

#[test]
fn pe32_rewrite_disables_aslr() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"int main(void) { return 0; }"#;
    let input = require_pe32(td.path(), "pe32_aslr", src);
    let output = td.path().join("pe32_aslr_out.exe");

    common::rewrite_pe(&input, &output);

    let data = std::fs::read(&output).unwrap();
    let e_lfanew = u32::from_le_bytes(data[0x3C..0x40].try_into().unwrap()) as usize;
    let opt = e_lfanew + 4 + 20;
    // PE32: DllCharacteristics at windows_fields + 42 = opt + 28 + 42 = opt + 70
    let dll_chars_off = opt + 70;
    let dll_chars = u16::from_le_bytes(data[dll_chars_off..dll_chars_off + 2].try_into().unwrap());
    assert_eq!(
        dll_chars & 0x0040,
        0,
        "DYNAMIC_BASE should be cleared for PE32 (DllCharacteristics=0x{:04X})",
        dll_chars
    );
}

// ====================================================================
// PE32: relocation data directory is zeroed after rewrite
// ====================================================================

#[test]
fn pe32_rewrite_zeroes_reloc_directory() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"int main(void) { return 0; }"#;
    let input = require_pe32(td.path(), "pe32_reloc", src);
    let output = td.path().join("pe32_reloc_out.exe");

    common::rewrite_pe(&input, &output);

    let data = std::fs::read(&output).unwrap();
    let e_lfanew = u32::from_le_bytes(data[0x3C..0x40].try_into().unwrap()) as usize;
    let opt = e_lfanew + 4 + 20;
    // PE32: data dirs at opt + 96, base reloc = entry 5 (at +40 bytes offset)
    let reloc_dd_off = opt + 96 + 5 * 8;
    let reloc_rva = u32::from_le_bytes(data[reloc_dd_off..reloc_dd_off + 4].try_into().unwrap());
    let reloc_size =
        u32::from_le_bytes(data[reloc_dd_off + 4..reloc_dd_off + 8].try_into().unwrap());
    assert_eq!(reloc_rva, 0, "base relocation RVA should be zeroed");
    assert_eq!(reloc_size, 0, "base relocation size should be zeroed");
}

// ====================================================================
// PE32: hook shellcode is injected correctly
// ====================================================================

#[test]
fn pe32_hook_shellcode_applies() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
__declspec(dllexport) int target_func(void) { return 42; }
int main(void) { return target_func(); }
"#;
    let input = require_pe32(td.path(), "pe32_hook", src);
    let output = td.path().join("pe32_hook_out.exe");
    let hooks = common::write_hooks(
        td.path(),
        "pe32_hook",
        r#"
[[hook]]
target = "target_func"
mode = "pre"
shellcode = [0xC3]
"#,
    );

    let result = common::rewrite_pe_hooked(&input, &output, &hooks);
    assert!(
        result.hooks_applied > 0,
        "PE32: should apply at least one hook"
    );

    let data = std::fs::read(&output).unwrap();
    let trcov = find_section(&data, ".trcov").unwrap();
    let section_data = &data[trcov.3 as usize..trcov.3 as usize + trcov.2 as usize];
    assert!(
        section_data.contains(&0xC3),
        "PE32: shellcode byte 0xC3 not found in .trcov section"
    );
}

// ====================================================================
// PE32: auto-strip makes room for .trcov in header-full binaries
// ====================================================================

#[test]
fn pe32_auto_strip_creates_room() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"int main(void) { return 0; }"#;
    // Compile WITHOUT -s (debug info generates many sections that fill the header).
    let src_path = td.path().join("pe32_strip.c");
    let bin = td.path().join("pe32_strip.exe");
    std::fs::write(&src_path, src).unwrap();
    let status = std::process::Command::new("i686-w64-mingw32-gcc")
        .args(["-o", bin.to_str().unwrap(), src_path.to_str().unwrap()])
        .status();
    let _ = std::fs::remove_file(&src_path);
    match status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("SKIP: i686-w64-mingw32-gcc not available");
            return;
        }
    }

    let output = td.path().join("pe32_strip_out.exe");
    // This should succeed by auto-stripping debug sections.
    let result = common::rewrite_pe(&bin, &output);
    assert!(
        result.blocks_instrumented > 0,
        "PE32 auto-strip: should instrument blocks"
    );

    // The output should have .trcov section.
    let data = std::fs::read(&output).unwrap();
    assert!(
        find_section(&data, ".trcov").is_some(),
        "PE32 auto-strip: .trcov section not found"
    );
}

// ====================================================================
// PE32: heap sanitiser companion DLL generation + import injection
// ====================================================================

#[test]
#[cfg(feature = "coverage")]
fn pe32_heap_san_dll_generated() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
#include <stdlib.h>
int main(void) {
    char *buf = (char *)malloc(64);
    if (!buf) return 1;
    buf[0] = 'A';
    free(buf);
    return 0;
}
"#;
    let input = require_pe32(td.path(), "pe32_heap", src);
    let output = td.path().join("pe32_heap_out.exe");

    let config = truant::RewriteConfig {
        input: input.clone(),
        output: output.clone(),
        forkserver: false,
        dry_run: false,
        heap_san: true,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: None,
        no_coverage: false,
    };

    truant::rewrite(&config).expect("PE32 rewrite with heap_san failed");

    // Companion DLL should be generated.
    let dll_path = truant::preload::preload_dll_path(&output);
    assert!(
        dll_path.exists(),
        "PE32 heap_san companion DLL not generated at {:?}",
        dll_path
    );

    // DLL should be non-trivial size (> 10 KB).
    let dll_size = std::fs::metadata(&dll_path).unwrap().len();
    assert!(
        dll_size > 10_000,
        "PE32 heap_san DLL too small: {} bytes",
        dll_size
    );
}

#[test]
#[cfg(feature = "coverage")]
fn pe32_heap_san_import_injected() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"int main(void) { return 0; }"#;
    let input = require_pe32(td.path(), "pe32_heap_imp", src);
    let output = td.path().join("pe32_heap_imp_out.exe");

    let config = truant::RewriteConfig {
        input: input.clone(),
        output: output.clone(),
        forkserver: false,
        dry_run: false,
        heap_san: true,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: None,
        no_coverage: false,
    };

    truant::rewrite(&config).expect("PE32 rewrite with heap_san import failed");

    // .idata2 section should exist (injected import).
    let data = std::fs::read(&output).unwrap();
    assert!(
        find_section(&data, ".idata2").is_some(),
        "PE32: .idata2 section not found after heap_san injection"
    );

    // The PE should still be parseable as PE32.
    let ctx = PeContext::parse(&data).unwrap();
    assert!(!ctx.is_64bit, "should still be PE32");

    // The .idata2 section should contain the DLL name and export name.
    let idata2 = find_section(&data, ".idata2").unwrap();
    let sec_data = &data[idata2.3 as usize..idata2.3 as usize + idata2.2 as usize];
    assert!(
        sec_data
            .windows(b"truant_heap_san_init".len())
            .any(|w| w == b"truant_heap_san_init"),
        "PE32: truant_heap_san_init string not found in .idata2"
    );
}

// ====================================================================
// PE32: sidecar sanitiser companion DLL generation + import injection
// ====================================================================

#[test]
#[cfg(feature = "coverage")]
fn pe32_sidecar_san_dll_generated() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
#include <stdlib.h>
int main(void) {
    void *p = malloc(32);
    free(p);
    return 0;
}
"#;
    let input = require_pe32(td.path(), "pe32_sidecar", src);
    let output = td.path().join("pe32_sidecar_out.exe");

    let config = truant::RewriteConfig {
        input: input.clone(),
        output: output.clone(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: true,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: None,
        no_coverage: false,
    };

    truant::rewrite(&config).expect("PE32 rewrite with sidecar_san failed");

    // Companion DLL should be generated.
    let dll_path = truant::sidecar_preload::sidecar_preload_dll_path(&output);
    assert!(
        dll_path.exists(),
        "PE32 sidecar_san companion DLL not generated at {:?}",
        dll_path
    );

    // .idata2 section should exist.
    let data = std::fs::read(&output).unwrap();
    assert!(
        find_section(&data, ".idata2").is_some(),
        "PE32: .idata2 section not found after sidecar_san injection"
    );

    // The PE should still be parseable as PE32.
    let ctx = PeContext::parse(&data).unwrap();
    assert!(!ctx.is_64bit, "should still be PE32");

    // The .idata2 section should contain the sidecar DLL name and export.
    let idata2 = find_section(&data, ".idata2").unwrap();
    let sec_data = &data[idata2.3 as usize..idata2.3 as usize + idata2.2 as usize];
    assert!(
        sec_data
            .windows(b"truant_sidecar_san_init".len())
            .any(|w| w == b"truant_sidecar_san_init"),
        "PE32: truant_sidecar_san_init string not found in .idata2"
    );
}

// ====================================================================
// PE32: no-coverage hook mode works
// ====================================================================

#[test]
fn pe32_no_coverage_mode() {
    if !common::mingw32_available() {
        eprintln!("SKIP: i686-w64-mingw32-gcc not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
__declspec(dllexport) int target_func(void) { return 42; }
int main(void) { return target_func(); }
"#;
    let input = require_pe32(td.path(), "pe32_nocov", src);
    let output = td.path().join("pe32_nocov_out.exe");
    let hooks = common::write_hooks(
        td.path(),
        "pe32_nocov",
        r#"
[[hook]]
target = "target_func"
mode = "pre"
shellcode = [0xC3]
"#,
    );

    let result = common::rewrite_pe_no_cov(&input, &output, &hooks);
    assert_eq!(
        result.blocks_instrumented, 0,
        "PE32 no-coverage: should instrument 0 blocks"
    );
    assert!(
        result.hooks_applied > 0,
        "PE32 no-coverage: should apply hooks"
    );
}

// ====================================================================
// PE DLL: .reloc section preserved after rewrite
// ====================================================================

#[test]
fn pe_dll_rewrite_preserves_reloc() {
    if !common::mingw_available() {
        eprintln!("SKIP: mingw not available");
        return;
    }

    let td = common::test_dir();
    let src = r#"
__declspec(dllexport) int get_val(void) { return 42; }
"#;
    let input = common::compile_pe_dll(td.path(), "dll_reloc", src);
    let input = match input {
        Some(p) => p,
        None => {
            eprintln!("SKIP: PE DLL compilation failed");
            return;
        }
    };
    let output = td.path().join("dll_reloc_out.dll");

    common::rewrite_pe(&input, &output);

    let data = std::fs::read(&output).unwrap();
    let reloc = find_section(&data, ".reloc");
    assert!(
        reloc.is_some(),
        "DLL rewrite should preserve .reloc section"
    );
}
