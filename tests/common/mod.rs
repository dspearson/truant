//! Shared test helpers for all truant integration tests.
//!
//! This module is the single source of truth for compile/rewrite/run helpers
//! used by hook_e2e.rs and hook_unit.rs.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use truant::{RewriteConfig, rewrite};

/// Create a temporary directory for a test. The directory is automatically
/// removed when the returned `TempDir` is dropped. Callers must keep the
/// `TempDir` alive for as long as files inside it are needed.
pub fn test_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("failed to create temp dir for test")
}

/// Platform-appropriate C compiler and flags for test binaries.
/// On Linux: `gcc -no-pie` for non-PIE ELF.
/// On macOS: `cc -Wl,-headerpad_max_install_names` (non-PIE not applicable on ARM64).
/// On Windows: `gcc` (MinGW produces PE).
fn cc_command() -> (String, Vec<String>) {
    if cfg!(target_os = "macos") {
        (
            "cc".to_string(),
            vec!["-Wl,-headerpad_max_install_names".to_string()],
        )
    } else if cfg!(target_os = "windows") {
        (
            "gcc".to_string(),
            vec!["-Wl,--export-all-symbols".to_string()],
        )
    } else {
        ("gcc".to_string(), vec!["-no-pie".to_string()])
    }
}

/// Platform-appropriate binary filename (appends .exe on Windows).
pub fn bin_path(dir: &Path, suffix: &str) -> PathBuf {
    if cfg!(target_os = "windows") {
        dir.join(format!("{}.exe", suffix))
    } else {
        dir.join(suffix)
    }
}

/// Compile a non-PIE test binary from custom C source into `dir`.
pub fn compile_bin(dir: &Path, suffix: &str, src: &str) -> PathBuf {
    let src_path = dir.join(format!("{}.c", suffix));
    let bin = bin_path(dir, suffix);
    std::fs::write(&src_path, src).unwrap();
    let (cc, base_flags) = cc_command();
    let status = Command::new(&cc)
        .arg("-o")
        .arg(bin.to_str().unwrap())
        .args(&base_flags)
        .arg(src_path.to_str().unwrap())
        .status()
        .unwrap_or_else(|_| panic!("{} not available", cc));
    assert!(status.success(), "failed to compile test binary {}", suffix);
    let _ = std::fs::remove_file(&src_path);
    bin
}

/// Compile a non-PIE test binary with extra flags into `dir`.
pub fn compile_bin_flags(dir: &Path, suffix: &str, src: &str, flags: &[&str]) -> PathBuf {
    let src_path = dir.join(format!("{}.c", suffix));
    let bin = bin_path(dir, suffix);
    std::fs::write(&src_path, src).unwrap();
    let (cc, base_flags) = cc_command();
    let mut cmd = Command::new(&cc);
    cmd.arg("-o").arg(bin.to_str().unwrap());
    cmd.args(&base_flags);
    cmd.args(flags);
    cmd.arg(src_path.to_str().unwrap());
    let status = cmd
        .status()
        .unwrap_or_else(|_| panic!("{} not available", cc));
    assert!(status.success(), "failed to compile test binary {}", suffix);
    let _ = std::fs::remove_file(&src_path);
    bin
}

/// Compile a shared library (.so / .dylib / .dll) from C source into `dir`.
pub fn compile_so(dir: &Path, suffix: &str, src: &str) -> PathBuf {
    let src_path = dir.join(format!("{}.c", suffix));
    let ext = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };
    let lib = dir.join(format!("{}.{}", suffix, ext));
    std::fs::write(&src_path, src).unwrap();
    let (cc, _) = cc_command();
    let status = Command::new(&cc)
        .args([
            "-shared",
            "-fPIC",
            "-o",
            lib.to_str().unwrap(),
            src_path.to_str().unwrap(),
        ])
        .status()
        .unwrap_or_else(|_| panic!("{} not available", cc));
    assert!(
        status.success(),
        "failed to compile shared library {}",
        suffix
    );
    let _ = std::fs::remove_file(&src_path);
    lib
}

/// Write a hook config TOML into `dir`.
pub fn write_hooks(dir: &Path, suffix: &str, content: &str) -> PathBuf {
    let path = dir.join(format!("{}.toml", suffix));
    std::fs::write(&path, content).unwrap();
    path
}

/// Find a symbol's VA in a non-stripped binary via `nm`.
/// On macOS, also tries with a `_` prefix (Mach-O name mangling).
pub fn find_symbol_va(binary: &Path, name: &str) -> Option<u64> {
    let output = Command::new("nm").arg(binary).output().expect("nm failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let prefixed = format!("_{}", name);
    stdout
        .lines()
        .find(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            parts.len() >= 3
                && (parts[1] == "T" || parts[1] == "t")
                && (parts[2] == name || parts[2] == prefixed)
        })
        .and_then(|line| {
            let addr_str = line.split_whitespace().next()?;
            u64::from_str_radix(addr_str, 16).ok()
        })
}

/// Find a PLT stub VA for a given symbol name via `objdump`.
///
/// Checks `.plt`, `.plt.sec` (CET-enabled toolchains), and `.plt.got`
/// sections since different GCC/binutils versions place stubs differently.
pub fn find_plt_va(binary: &Path, name: &str) -> Option<u64> {
    let needle = format!("<{}@plt>:", name);
    for section in &[".plt", ".plt.sec", ".plt.got"] {
        let output = Command::new("objdump")
            .args(["-d", "-j", section, binary.to_str().unwrap()])
            .output()
            .expect("objdump failed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Some(va) = stdout
            .lines()
            .find(|line| line.contains(&needle))
            .and_then(|line| {
                let addr_str = line.trim().split_whitespace().next()?;
                u64::from_str_radix(addr_str, 16).ok()
            })
        {
            return Some(va);
        }
    }
    None
}

/// Rewrite a binary with hooks (coverage enabled).
pub fn rewrite_hooked(input: &Path, output: &Path, hooks: &Path) -> truant::RewriteResult {
    let config = RewriteConfig {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: Some(hooks.to_path_buf()),
        no_coverage: false,
    };
    rewrite(&config).expect("rewrite with hooks failed")
}

/// Rewrite a binary with hooks in no-coverage mode.
pub fn rewrite_no_cov(input: &Path, output: &Path, hooks: &Path) -> truant::RewriteResult {
    let config = RewriteConfig {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: Some(hooks.to_path_buf()),
        no_coverage: true,
    };
    rewrite(&config).expect("rewrite with hooks (no-coverage) failed")
}

/// Run binary and return exit code.
pub fn run(binary: &Path) -> i32 {
    Command::new(binary)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {}: {}", binary.display(), e))
        .code()
        .expect("process terminated by signal")
}

/// Run binary and capture stdout.
pub fn run_stdout(binary: &Path) -> String {
    let out = Command::new(binary)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {}", binary.display(), e));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Run binary with LD_PRELOAD and return exit code.
pub fn run_with_preload(binary: &Path, preload: &Path) -> i32 {
    Command::new(binary)
        .env("LD_PRELOAD", preload)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {}: {}", binary.display(), e))
        .code()
        .expect("process terminated by signal")
}

/// Compile a PE binary from C source using x86_64-w64-mingw32-gcc into `dir`.
/// Returns None if the cross-compiler is not available.
pub fn compile_pe(dir: &Path, suffix: &str, src: &str) -> Option<PathBuf> {
    let src_path = dir.join(format!("{}.c", suffix));
    let bin = dir.join(format!("{}.exe", suffix));
    std::fs::write(&src_path, src).unwrap();
    let result = Command::new("x86_64-w64-mingw32-gcc")
        .args(["-o", bin.to_str().unwrap(), src_path.to_str().unwrap()])
        .status();
    let _ = std::fs::remove_file(&src_path);
    match result {
        Ok(status) if status.success() => Some(bin),
        _ => None,
    }
}

/// Compile a PE binary with extra flags into `dir`.
pub fn compile_pe_flags(dir: &Path, suffix: &str, src: &str, flags: &[&str]) -> Option<PathBuf> {
    let src_path = dir.join(format!("{}.c", suffix));
    let bin = dir.join(format!("{}.exe", suffix));
    std::fs::write(&src_path, src).unwrap();
    let mut cmd = Command::new("x86_64-w64-mingw32-gcc");
    cmd.args(["-o", bin.to_str().unwrap()]);
    cmd.args(flags);
    cmd.arg(src_path.to_str().unwrap());
    let result = cmd.status();
    let _ = std::fs::remove_file(&src_path);
    match result {
        Ok(status) if status.success() => Some(bin),
        _ => None,
    }
}

/// Compile a Windows DLL from C source using x86_64-w64-mingw32-gcc into `dir`.
pub fn compile_pe_dll(dir: &Path, suffix: &str, src: &str) -> Option<PathBuf> {
    let src_path = dir.join(format!("{}.c", suffix));
    let lib = dir.join(format!("{}.dll", suffix));
    std::fs::write(&src_path, src).unwrap();
    let result = Command::new("x86_64-w64-mingw32-gcc")
        .args([
            "-shared",
            "-o",
            lib.to_str().unwrap(),
            src_path.to_str().unwrap(),
        ])
        .status();
    let _ = std::fs::remove_file(&src_path);
    match result {
        Ok(status) if status.success() => Some(lib),
        _ => None,
    }
}

/// Rewrite a PE binary with coverage instrumentation.
pub fn rewrite_pe(input: &Path, output: &Path) -> truant::RewriteResult {
    let config = RewriteConfig {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
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
    rewrite(&config).expect("PE rewrite failed")
}

/// Rewrite a PE binary with hooks.
pub fn rewrite_pe_hooked(input: &Path, output: &Path, hooks: &Path) -> truant::RewriteResult {
    let config = RewriteConfig {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: Some(hooks.to_path_buf()),
        no_coverage: false,
    };
    rewrite(&config).expect("PE rewrite with hooks failed")
}

/// Rewrite a PE binary in no-coverage mode with hooks.
pub fn rewrite_pe_no_cov(input: &Path, output: &Path, hooks: &Path) -> truant::RewriteResult {
    let config = RewriteConfig {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: None,
        hooks: Some(hooks.to_path_buf()),
        no_coverage: true,
    };
    rewrite(&config).expect("PE rewrite no-cov with hooks failed")
}

/// Run a PE binary under Wine and return (exit_code, stdout, stderr).
/// Returns None if Wine is not available or fails to start.
pub fn run_wine(binary: &Path) -> Option<(i32, String, String)> {
    let result = Command::new("wine64")
        .arg(binary)
        .env("WINEDEBUG", "-all")
        .output();
    match result {
        Ok(output) => {
            let code = output.status.code().unwrap_or(-1);
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            Some((code, stdout, stderr))
        }
        Err(_) => None,
    }
}

/// Check if Wine is available and functional (can run a trivial PE).
pub fn wine_available() -> bool {
    Command::new("wine64")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if x86_64-w64-mingw32-gcc is available.
pub fn mingw_available() -> bool {
    Command::new("x86_64-w64-mingw32-gcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if i686-w64-mingw32-gcc is available (32-bit PE cross-compiler).
pub fn mingw32_available() -> bool {
    Command::new("i686-w64-mingw32-gcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Compile a 32-bit PE binary from C source using i686-w64-mingw32-gcc into `dir`.
/// Returns None if the cross-compiler is not available.
///
/// Uses `-s` to strip debug info and `-Wl,--file-alignment,512` to keep header
/// room for an additional section (mingw PE32 defaults fill the header space).
pub fn compile_pe32(dir: &Path, suffix: &str, src: &str) -> Option<PathBuf> {
    let src_path = dir.join(format!("{}.c", suffix));
    let bin = dir.join(format!("{}.exe", suffix));
    std::fs::write(&src_path, src).unwrap();
    let result = Command::new("i686-w64-mingw32-gcc")
        .args([
            "-s",
            "-o",
            bin.to_str().unwrap(),
            src_path.to_str().unwrap(),
        ])
        .status();
    let _ = std::fs::remove_file(&src_path);
    match result {
        Ok(status) if status.success() => Some(bin),
        _ => None,
    }
}
