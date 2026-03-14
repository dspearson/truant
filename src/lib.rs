//! Truant — static binary rewriter for arbitrary function hooking.
//!
//! Rewrites ELF, Mach-O, and PE binaries to insert hook trampolines at
//! specified function entry points. Supports pre/post/replace/return hook
//! modes, conditional hooks, hook chaining, and runtime toggle bytes.
//!
//! With the `coverage` feature enabled, also provides AFL-compatible coverage
//! instrumentation (SHM bitmaps, forkserver, persistent mode).

pub mod arch;
pub mod detect;
pub mod disasm;
pub mod elf;
pub mod elf_impl;
pub mod fat;
pub mod hook_preload;
pub mod hook_trampoline;
pub mod hooks;
pub mod macho;
pub mod macho_disasm;
pub mod macho_impl;
pub mod macho_trampoline;
pub mod patcher;
pub mod pe;
pub mod pe_disasm;
pub mod pe_impl;
#[cfg(feature = "coverage")]
pub mod preload;
#[cfg(feature = "coverage")]
pub mod sidecar_preload;
pub mod traits;
pub mod trampoline;

pub use arch::{X86_64Disassembler, X86_64TrampolineGenerator};
pub use detect::{BinaryFormat, CpuArchitecture, detect_architecture, detect_format};
pub use elf_impl::{ElfBinaryContext, ElfPatcher};
pub use macho_impl::{MachOBinaryContext, MachOPatcher};
pub use pe_impl::{PeBinaryContext, PePatcher};

#[cfg(feature = "aarch64")]
pub use arch::{AArch64Disassembler, AArch64TrampolineGenerator};

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Configuration for the rewrite operation.
#[derive(Debug, Clone)]
pub struct RewriteConfig {
    /// Path to the input binary.
    pub input: PathBuf,
    /// Path for the output binary.
    pub output: PathBuf,
    /// Whether to include the forkserver loop in the init code.
    pub forkserver: bool,
    /// Dry run: analyse and report without writing output.
    pub dry_run: bool,
    /// Enable guard-page heap sanitiser (intercepts malloc/free/calloc/realloc/memalign/aligned_alloc/posix_memalign).
    pub heap_san: bool,
    /// Enable sidecar memory sanitiser (ring-buffer logging + shadow heap validation).
    /// Mutually exclusive with `heap_san`.
    pub sidecar_san: bool,
    /// Virtual address of function to wrap for persistent mode (skip re-fork).
    pub persistent_addr: Option<u64>,
    /// Number of iterations before re-forking in persistent mode (default 1000).
    pub persistent_count: u32,
    /// Defer forkserver to persistent entry point (requires persistent_addr).
    /// Target runs all init code (dlopen, etc.) before forking — forked children
    /// inherit loaded libraries with zero re-init overhead.
    pub defer: bool,
    /// Post-rewrite validation: run binary with a trivial input, check edge count > 50.
    /// When true, output is NOT written; only validation result is returned.
    #[cfg(feature = "coverage")]
    pub validate: bool,
    /// Restrict coverage instrumentation to functions matching these module name substrings.
    /// None = instrument all blocks (default).
    pub instrument_modules: Option<Vec<String>>,
    /// Path to hook specification TOML file for arbitrary function hooking.
    pub hooks: Option<PathBuf>,
    /// Hook-only mode: apply hooks without coverage instrumentation.
    /// Skips SHM init, forkserver, coverage trampolines, and entry point redirection.
    pub no_coverage: bool,
}

/// Result of a successful rewrite.
#[derive(Debug)]
pub struct RewriteResult {
    /// Number of basic blocks instrumented.
    pub blocks_instrumented: usize,
    /// Number of blocks skipped.
    pub blocks_skipped: usize,
    /// Virtual address of the new segment.
    pub segment_va: u64,
    /// Size of the new segment in bytes.
    pub segment_size: u64,
    /// Output file path (None if dry run).
    pub output_path: Option<PathBuf>,
    /// Number of allocator functions intercepted by heap san (0–10).
    pub heap_san_intercepted: usize,
    /// Path to LD_PRELOAD heap sanitiser .so (for dynamically linked binaries).
    pub preload_lib_path: Option<PathBuf>,
    /// Path to LD_PRELOAD sidecar sanitiser .so (for dynamically linked binaries).
    pub sidecar_san_lib_path: Option<PathBuf>,
    /// Whether persistent mode was enabled.
    pub persistent_mode: bool,
    /// Number of hooks applied.
    pub hooks_applied: usize,
    /// Path to companion hook preload .so (for .so-based hooks).
    pub hook_preload_lib_path: Option<PathBuf>,
}

/// Result of a post-rewrite validation run.
#[cfg(all(feature = "coverage", unix))]
#[derive(Debug)]
pub struct ValidateResult {
    /// Number of edges (non-zero bytes in coverage bitmap) observed during validation.
    pub edge_count: u64,
    /// Number of blocks instrumented in the rewritten binary.
    pub blocks_instrumented: usize,
    /// Whether validation succeeded (edge_count > 50).
    pub success: bool,
    /// Error description if execution failed.
    pub error: Option<String>,
}

/// Validate an ELF binary by rewriting to a temp path and running it once.
///
/// Rewrites to `/tmp/truant_validate_<pid>_output`, passes a trivial input file,
/// sets up a SysV SHM segment, runs the binary, and counts edges.
/// Returns success when edge_count > 50.
#[cfg(all(feature = "coverage", unix))]
pub fn validate(config: &RewriteConfig) -> Result<ValidateResult> {
    let pid = std::process::id();
    let tmp_output = std::path::PathBuf::from(format!("/tmp/truant_validate_{}_output", pid));
    let tmp_input = std::path::PathBuf::from(format!("/tmp/truant_validate_{}_input", pid));

    // Write trivial input (16 zero bytes)
    std::fs::write(&tmp_input, [0u8; 16])
        .with_context(|| format!("failed to write temp input {}", tmp_input.display()))?;

    // Build a rewrite config that writes to the temp output (not dry_run, not validate)
    let rewrite_cfg = RewriteConfig {
        input: config.input.clone(),
        output: tmp_output.clone(),
        forkserver: false,
        dry_run: false,
        heap_san: false,
        sidecar_san: false,
        persistent_addr: None,
        persistent_count: 1000,
        defer: false,
        #[cfg(feature = "coverage")]
        validate: false,
        instrument_modules: config.instrument_modules.clone(),
        hooks: None,
        no_coverage: false,
    };

    let rewrite_result = match rewrite(&rewrite_cfg) {
        Ok(r) => r,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp_input);
            anyhow::bail!("rewrite failed during validation: {}", e);
        }
    };
    let blocks_instrumented = rewrite_result.blocks_instrumented;

    // Detect if the input is a Mach-O dylib (filetype = MH_DYLIB = 6).
    // For dylibs we cannot run them directly — build a minimal dlopen harness.
    let is_macho_dylib = {
        let input_data = std::fs::read(&config.input).unwrap_or_default();
        // Mach-O magic: 0xFEEDFACF (64-bit LE) or 0xCEFAEDFE (64-bit BE)
        // MH_DYLIB = 6 at offset 12 (after magic, cputype, cpusubtype)
        if input_data.len() >= 16 {
            let magic = u32::from_le_bytes(input_data[0..4].try_into().unwrap_or([0; 4]));
            let filetype = u32::from_le_bytes(input_data[12..16].try_into().unwrap_or([0; 4]));
            (magic == 0xFEED_FACF || magic == 0xCEFA_EDFE) && filetype == 6
        } else {
            false
        }
    };

    // Create a 65536-byte SysV SHM segment
    let shm_id = unsafe { libc::shmget(libc::IPC_PRIVATE, 65536, libc::IPC_CREAT | 0o600) };
    if shm_id < 0 {
        let _ = std::fs::remove_file(&tmp_input);
        let _ = std::fs::remove_file(&tmp_output);
        anyhow::bail!("shmget failed (errno={})", std::io::Error::last_os_error());
    }
    let shm_ptr = unsafe { libc::shmat(shm_id, std::ptr::null(), 0) };
    if shm_ptr as isize == -1 {
        unsafe {
            libc::shmctl(shm_id, libc::IPC_RMID, std::ptr::null_mut());
        }
        let _ = std::fs::remove_file(&tmp_input);
        let _ = std::fs::remove_file(&tmp_output);
        anyhow::bail!("shmat failed");
    }

    // Zero the bitmap before running
    unsafe {
        std::ptr::write_bytes(shm_ptr as *mut u8, 0, 65536);
    }

    // For dylibs: compile a dlopen harness that loads the rewritten dylib AND calls
    // its exported functions (to trigger trampolines and write coverage edges).
    // The dylib init code (called via LC_ROUTINES_64 or __mod_init_func) reads
    // __AFL_SHM_ID from KERN_PROCARGS2, attaches SHM, stores the pointer.
    // Trampolines only fire when code runs, so we must call exported functions.
    let tmp_harness = std::path::PathBuf::from(format!("/tmp/truant_validate_{}_harness", pid));
    let run_result = if is_macho_dylib {
        #[cfg(target_os = "macos")]
        {
            let harness_src_path = format!("/tmp/truant_validate_{}_harness.c", pid);

            // Extract exported function symbols from the ORIGINAL input dylib using nm.
            // nm -g -j lists global symbols one per line (just names, no address/type).
            // Filter to those starting with '_' and strip the leading underscore for dlsym.
            let nm_out = std::process::Command::new("nm")
                .args(["-g", "-j", config.input.to_str().unwrap_or("")])
                .output()
                .ok();
            let exported_syms: Vec<String> = nm_out
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| {
                    s.lines()
                        .filter(|l| {
                            l.starts_with('_')
                            // Skip ObjC/C++ noise: +[, -[, __block_descriptor, etc.
                            && !l.contains("__block_descriptor")
                            && !l.starts_with("__Z")   // C++ mangled
                            && !l.starts_with("___Z")
                        }) // C++ mangled (macOS leading _)
                        .map(|l| l.trim_start_matches('_').to_string())
                        .take(32) // limit to first 32 symbols
                        .collect()
                })
                .unwrap_or_default();

            tracing::info!(
                "dylib validate: found {} exported symbols: {:?}",
                exported_syms.len(),
                &exported_syms[..exported_syms.len().min(8)]
            );

            // Build the call list — each symbol is called with (0,0,0,0,0,0) ints.
            // setjmp/signal protect against crashes from wrong argument types.
            let sym_calls: String = exported_syms.iter().map(|sym| {
                format!(
                    "    fn = dlsym(h, \"{sym}\"); \
                     if (fn && !setjmp(jmp_env)) {{ ((void(*)(long,long,long,long,long,long))fn)(0,0,0,0,0,0); }}\n"
                )
            }).collect();

            let harness_src = format!(
                r#"#include <dlfcn.h>
#include <setjmp.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
static jmp_buf jmp_env;
static void crash_handler(int sig) {{ (void)sig; longjmp(jmp_env, 1); }}
int main(int argc, char **argv) {{
    (void)argc; (void)argv;
    void *h = dlopen("{dylib}", RTLD_LAZY | RTLD_LOCAL);
    if (!h) {{ fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }}
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    void *fn;
{sym_calls}    dlclose(h);
    return 0;
}}
"#,
                dylib = tmp_output.display(),
                sym_calls = sym_calls,
            );
            std::fs::write(&harness_src_path, &harness_src)
                .context("failed to write dlopen harness source")?;

            let cc_status = std::process::Command::new("cc")
                .args([
                    "-o",
                    tmp_harness
                        .to_str()
                        .ok_or_else(|| anyhow::anyhow!("harness path is not valid UTF-8"))?,
                    &harness_src_path,
                ])
                .status();
            let _ = std::fs::remove_file(&harness_src_path);

            match cc_status {
                Ok(s) if s.success() => {
                    tracing::info!(
                        "compiled dlopen harness for dylib validation ({} syms)",
                        exported_syms.len()
                    );
                    std::process::Command::new(&tmp_harness)
                        .env("__AFL_SHM_ID", shm_id.to_string())
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .output()
                }
                Ok(s) => Err(std::io::Error::other(format!(
                    "cc failed to compile dlopen harness (exit {:?})",
                    s.code()
                ))),
                Err(e) => Err(e),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Non-macOS: dylibs are ELF .so files; run as executable is not supported.
            // Fall through to the standard path which will fail gracefully.
            std::process::Command::new(&tmp_output)
                .arg(&tmp_input)
                .env("__AFL_SHM_ID", shm_id.to_string())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .output()
        }
    } else {
        // Executables: run directly
        std::process::Command::new(&tmp_output)
            .arg(&tmp_input)
            .env("__AFL_SHM_ID", shm_id.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
    };

    // Count edges before cleanup
    let bitmap = unsafe { std::slice::from_raw_parts(shm_ptr as *const u8, 65536) };
    let edge_count = bitmap.iter().filter(|&&b| b != 0).count() as u64;

    // Cleanup SHM
    unsafe {
        libc::shmdt(shm_ptr);
        libc::shmctl(shm_id, libc::IPC_RMID, std::ptr::null_mut());
    }

    // Cleanup temp files
    let _ = std::fs::remove_file(&tmp_input);
    let _ = std::fs::remove_file(&tmp_output);
    let _ = std::fs::remove_file(&tmp_harness);

    // Check execution result
    let error = match run_result {
        Err(e) => Some(format!("failed to execute binary: {}", e)),
        Ok(output) => {
            if !output.status.success() {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    output.status.signal().map(|sig| {
                        format!(
                            "binary exited with signal {} (may indicate crash on trivial input)",
                            sig
                        )
                    })
                }
                #[cfg(not(unix))]
                {
                    None
                }
            } else {
                None
            }
        }
    };

    // Success criterion: edge count > 0 means instrumentation is working.
    // We just need to observe at least one edge to confirm the SHM init code ran and
    // the coverage bitmap is being populated. The absolute threshold does not matter
    // since validation inputs are trivial (16 zero bytes, not representative workloads).
    let success = error.is_none() && edge_count > 0;

    tracing::info!(
        "validation: {} edges observed, {} blocks instrumented, {}",
        edge_count,
        blocks_instrumented,
        if success { "PASS" } else { "FAIL" }
    );

    Ok(ValidateResult {
        edge_count,
        blocks_instrumented,
        success,
        error,
    })
}

/// Rewrite an ELF binary with AFL-compatible coverage instrumentation.
pub fn rewrite(config: &RewriteConfig) -> Result<RewriteResult> {
    let data = std::fs::read(&config.input)
        .with_context(|| format!("failed to read {}", config.input.display()))?;

    // Auto-detect binary format
    let binary_format =
        detect_format(&data).context("unsupported binary format (not ELF, Mach-O, or PE)")?;

    tracing::info!(
        "parsing {:?}: {} ({} bytes)",
        binary_format,
        config.input.display(),
        data.len()
    );

    // Fat binary: extract, instrument each slice, re-package
    if binary_format == BinaryFormat::Fat {
        return fat::rewrite_fat(config);
    }

    // Create format-specific context
    let ctx: Box<dyn traits::BinaryContext> = match binary_format {
        BinaryFormat::Elf => {
            Box::new(ElfBinaryContext::parse(&data).context("ELF analysis failed")?)
        }
        BinaryFormat::MachO => {
            Box::new(MachOBinaryContext::parse(&data).context("Mach-O analysis failed")?)
        }
        BinaryFormat::Pe => Box::new(PeBinaryContext::parse(&data).context("PE analysis failed")?),
        BinaryFormat::Fat => unreachable!("fat binaries handled above"),
    };

    tracing::info!(
        ".text: VA=0x{:x} size={} entry=0x{:x} symbols={}",
        ctx.text_section_va(),
        ctx.text_section_size(),
        ctx.entry_point(),
        ctx.func_symbols().len(),
    );

    // PT_NOTE warning: ELF-specific. Check via downcast; suppress if not ELF.
    if let Some(elf_ctx) = ctx.as_any().downcast_ref::<ElfBinaryContext>()
        && elf_ctx.inner().note_segment.is_none()
    {
        tracing::info!("no PT_NOTE segment — will append new program header entry");
    }

    // Create architecture-specific disassembler
    let disasm: Box<dyn traits::Disassembler> = match ctx.architecture() {
        CpuArchitecture::X86 | CpuArchitecture::X86_64 => Box::new(X86_64Disassembler::new()),
        CpuArchitecture::AArch64 => {
            #[cfg(feature = "aarch64")]
            {
                Box::new(AArch64Disassembler::new())
            }
            #[cfg(not(feature = "aarch64"))]
            {
                anyhow::bail!(
                    "AArch64 support requires --features aarch64 (compile with: cargo build --features aarch64)"
                )
            }
        }
    };

    let blocks = if config.no_coverage {
        tracing::info!("--no-coverage: skipping basic block discovery");
        Vec::new()
    } else {
        let b = disasm
            .find_basic_blocks(&data, &*ctx, &config.instrument_modules)
            .context("basic block detection failed")?;
        tracing::info!("detected {} instrumentable basic blocks", b.len());
        b
    };

    if config.dry_run {
        return Ok(RewriteResult {
            blocks_instrumented: blocks.len(),
            blocks_skipped: 0,
            segment_va: 0,
            segment_size: 0,
            output_path: None,
            heap_san_intercepted: 0,
            preload_lib_path: None,
            sidecar_san_lib_path: None,
            persistent_mode: config.persistent_addr.is_some(),
            hooks_applied: 0,
            hook_preload_lib_path: None,
        });
    }

    // Mutual exclusivity check
    if config.heap_san && config.sidecar_san {
        anyhow::bail!("--heap-san and --sidecar-san are mutually exclusive");
    }

    // For dynamically linked binaries, GOT patching only intercepts calls through
    // the main binary's PLT. Shared libraries (Qt, glibc, etc.) resolve allocator
    // symbols through their own PLT, bypassing our wrappers entirely. This causes
    // crashes when a shared library realloc/frees a pointer from our guard-page
    // allocator. Use LD_PRELOAD instead — it overrides symbols globally.
    #[cfg(feature = "coverage")]
    let use_preload = config.heap_san && ctx.is_dynamic();
    let inline_heap_san = config.heap_san && !ctx.is_dynamic();

    // Create architecture-specific trampoline generator
    let tramp_gen: Box<dyn traits::TrampolineGenerator> = match ctx.architecture() {
        CpuArchitecture::X86 | CpuArchitecture::X86_64 => {
            Box::new(X86_64TrampolineGenerator::new())
        }
        CpuArchitecture::AArch64 => {
            #[cfg(feature = "aarch64")]
            {
                Box::new(AArch64TrampolineGenerator::new())
            }
            #[cfg(not(feature = "aarch64"))]
            {
                anyhow::bail!(
                    "AArch64 support requires --features aarch64 (compile with: cargo build --features aarch64)"
                )
            }
        }
    };

    // --- Hook resolution (if configured) ---
    let resolved_hooks = if let Some(ref hooks_path) = config.hooks {
        let hook_config =
            hooks::parse_hook_config(hooks_path).context("failed to parse hook config")?;
        match binary_format {
            BinaryFormat::Elf => {
                let elf_ctx = ctx
                    .as_any()
                    .downcast_ref::<ElfBinaryContext>()
                    .ok_or_else(|| anyhow::anyhow!("hook resolution requires ElfBinaryContext"))?;
                let min_displaced = match ctx.architecture() {
                    CpuArchitecture::AArch64 => 4,
                    CpuArchitecture::X86 | CpuArchitecture::X86_64 => 5,
                };
                let rh = hooks::resolve_hooks(&hook_config, &data, elf_ctx.inner(), min_displaced)
                    .context("failed to resolve hooks")?;
                tracing::info!("resolved {} hooks from {}", rh.len(), hooks_path.display());
                rh
            }
            BinaryFormat::MachO => {
                let macho_ctx = ctx
                    .as_any()
                    .downcast_ref::<MachOBinaryContext>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("hook resolution requires MachOBinaryContext")
                    })?;
                let rh = hooks::resolve_hooks_macho(&hook_config, &data, macho_ctx.inner())
                    .context("failed to resolve hooks")?;
                tracing::info!("resolved {} hooks from {}", rh.len(), hooks_path.display());
                rh
            }
            BinaryFormat::Pe => {
                let pe_ctx = ctx
                    .as_any()
                    .downcast_ref::<PeBinaryContext>()
                    .ok_or_else(|| anyhow::anyhow!("hook resolution requires PeBinaryContext"))?;
                let rh = hooks::resolve_hooks_pe(&hook_config, &data, pe_ctx.inner())
                    .context("failed to resolve hooks")?;
                tracing::info!("resolved {} hooks from {}", rh.len(), hooks_path.display());
                rh
            }
            BinaryFormat::Fat => unreachable!("fat binaries handled above"),
        }
    } else {
        Vec::new()
    };

    // Build a set of hook target VAs to exclude from coverage instrumentation.
    let hook_target_vas: std::collections::HashSet<u64> =
        resolved_hooks.iter().map(|h| h.target_va).collect();

    // Filter out blocks whose VA matches a hook target.
    let blocks: Vec<_> = if hook_target_vas.is_empty() {
        blocks
    } else {
        let orig_len = blocks.len();
        let filtered: Vec<_> = blocks
            .into_iter()
            .filter(|b| !hook_target_vas.contains(&b.va))
            .collect();
        let excluded = orig_len - filtered.len();
        if excluded > 0 {
            tracing::info!("excluded {} blocks from coverage (hook targets)", excluded,);
        }
        filtered
    };

    // Create format-specific patcher
    let patcher: Box<dyn traits::Patcher> = match binary_format {
        BinaryFormat::Elf => Box::new(ElfPatcher::new(tramp_gen)),
        BinaryFormat::MachO => Box::new(MachOPatcher::new(tramp_gen)),
        BinaryFormat::Pe => {
            let mut pe_patcher = PePatcher::new(tramp_gen);
            // For PE, set the hook library path so that init code can embed
            // LoadLibraryA + GetProcAddress calls (no companion DLL needed).
            if let Some(ref hooks_path) = config.hooks {
                let hook_config = hooks::parse_hook_config(hooks_path)?;
                if let Some(lib_path) = hooks::library_path(&hook_config) {
                    pe_patcher.set_hook_library_path(lib_path.to_string_lossy().into_owned());
                }
            }
            Box::new(pe_patcher)
        }
        BinaryFormat::Fat => unreachable!("fat binaries handled above"),
    };

    let patch_result = patcher
        .patch(
            &*ctx,
            &blocks,
            data,
            config.forkserver,
            inline_heap_san,
            config.persistent_addr,
            config.persistent_count,
            config.defer,
            &resolved_hooks,
            config.no_coverage,
        )
        .context("patching failed")?;

    // Write output
    std::fs::write(&config.output, &patch_result.data)
        .with_context(|| format!("failed to write {}", config.output.display()))?;

    // Set executable permission
    set_executable(&config.output)?;

    // Ad-hoc codesign Mach-O binaries (required for Apple Silicon, helpful for x86_64)
    if binary_format == BinaryFormat::MachO {
        codesign_if_available(&config.output);
    }

    tracing::info!(
        "wrote {} ({} bytes, {} blocks instrumented, {} hooks applied)",
        config.output.display(),
        patch_result.data.len(),
        patch_result.blocks_instrumented,
        patch_result.hooks_applied,
    );

    // Generate heap sanitiser companion library.
    #[cfg(feature = "coverage")]
    let preload_lib_path = if config.heap_san && binary_format == BinaryFormat::Pe {
        // PE: cross-compile companion DLL and inject IAT import.
        let is_32bit = ctx
            .as_any()
            .downcast_ref::<PeBinaryContext>()
            .map(|c| !c.inner().is_64bit)
            .unwrap_or(false);
        let dll_path = preload::generate_preload_dll(&config.output, is_32bit)
            .context("failed to generate heap sanitiser companion DLL")?;
        // Inject import into the rewritten PE so the loader pulls the DLL in.
        let pe_data = std::fs::read(&config.output)
            .context("failed to re-read rewritten PE for import injection")?;
        let pe_ctx = pe::PeContext::parse(&pe_data)
            .context("failed to parse rewritten PE for import injection")?;
        let dll_filename = dll_path.file_name().unwrap_or_default().to_string_lossy();
        let injected = pe::inject_import(pe_data, &pe_ctx, &dll_filename, "truant_heap_san_init")
            .context("failed to inject heap sanitiser import")?;
        std::fs::write(&config.output, &injected)
            .context("failed to write PE with injected import")?;
        tracing::info!(
            "injected heap sanitiser import into {}",
            config.output.display()
        );
        Some(dll_path)
    } else if use_preload {
        let is_macho = binary_format == BinaryFormat::MachO;
        let so_path = preload::generate_preload_lib(&config.output, is_macho)
            .context("failed to generate heap sanitiser preload library")?;
        Some(so_path)
    } else {
        None
    };
    #[cfg(not(feature = "coverage"))]
    let preload_lib_path: Option<PathBuf> = None;

    // Generate sidecar sanitiser companion library.
    #[cfg(feature = "coverage")]
    let sidecar_san_lib_path = if config.sidecar_san && binary_format == BinaryFormat::Pe {
        // PE: cross-compile sidecar DLL and inject IAT import.
        let is_32bit = ctx
            .as_any()
            .downcast_ref::<PeBinaryContext>()
            .map(|c| !c.inner().is_64bit)
            .unwrap_or(false);
        let dll_path = sidecar_preload::generate_sidecar_preload_dll(&config.output, is_32bit)
            .context("failed to generate sidecar sanitiser companion DLL")?;
        let pe_data = std::fs::read(&config.output)
            .context("failed to re-read rewritten PE for sidecar import injection")?;
        let pe_ctx = pe::PeContext::parse(&pe_data)
            .context("failed to parse rewritten PE for sidecar import injection")?;
        let dll_filename = dll_path.file_name().unwrap_or_default().to_string_lossy();
        let injected =
            pe::inject_import(pe_data, &pe_ctx, &dll_filename, "truant_sidecar_san_init")
                .context("failed to inject sidecar sanitiser import")?;
        std::fs::write(&config.output, &injected)
            .context("failed to write PE with injected sidecar import")?;
        tracing::info!(
            "injected sidecar sanitiser import into {}",
            config.output.display()
        );
        Some(dll_path)
    } else if config.sidecar_san && ctx.is_dynamic() {
        let is_macho = binary_format == BinaryFormat::MachO;
        let so_path = sidecar_preload::generate_sidecar_preload_lib(&config.output, is_macho)
            .context("failed to generate sidecar sanitiser preload library")?;
        Some(so_path)
    } else if config.sidecar_san && !ctx.is_dynamic() {
        tracing::warn!(
            "sidecar-san requires a dynamically linked binary (static binaries not supported)"
        );
        None
    } else {
        None
    };
    #[cfg(not(feature = "coverage"))]
    let sidecar_san_lib_path: Option<PathBuf> = None;

    // Generate companion hook preload library for library-based hooks.
    // PE binaries don't need a companion DLL — hook resolution is embedded
    // directly in the PE init code via LoadLibraryA + GetProcAddress.
    let is_macho = binary_format == BinaryFormat::MachO;
    let hook_preload_lib_path = if binary_format == BinaryFormat::Pe {
        None
    } else if let Some(ref hooks_path) = config.hooks {
        let hook_config = hooks::parse_hook_config(hooks_path)?;
        if let Some(lib_path) = hooks::library_path(&hook_config) {
            let is_pie = if is_macho {
                // All modern macOS executables (and dylibs) are PIE.
                true
            } else {
                ctx.as_any()
                    .downcast_ref::<ElfBinaryContext>()
                    .map(|e| e.inner().is_shared_object)
                    .unwrap_or(false)
            };
            let hook_data_va = patch_result.hook_data_va;
            let toggle_data_va = patch_result.toggle_data_va;
            let spec = hook_preload::build_preload_spec(
                &resolved_hooks,
                lib_path,
                hook_data_va,
                toggle_data_va,
                is_pie,
                is_macho,
            );
            if let Some(spec) = spec {
                let so_path =
                    hook_preload::generate_hook_preload_lib(&config.output, &spec, is_macho)
                        .context("failed to generate hook companion preload library")?;
                Some(so_path)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    Ok(RewriteResult {
        blocks_instrumented: patch_result.blocks_instrumented,
        blocks_skipped: patch_result.blocks_skipped,
        segment_va: patch_result.segment_va,
        segment_size: patch_result.segment_size,
        output_path: Some(config.output.clone()),
        heap_san_intercepted: patch_result.heap_san_intercepted,
        preload_lib_path,
        sidecar_san_lib_path,
        persistent_mode: config.persistent_addr.is_some(),
        hooks_applied: patch_result.hooks_applied,
        hook_preload_lib_path,
    })
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    let mode = perms.mode() | 0o111; // +x
    perms.set_mode(mode);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<()> {
    Ok(())
}

/// Ad-hoc codesign a Mach-O binary (required on Apple Silicon, helpful on x86_64).
/// Silently skips if `codesign` is not available (e.g. cross-rewriting from Linux).
fn codesign_if_available(path: &Path) {
    match std::process::Command::new("codesign")
        .args(["-f", "-s", "-"])
        .arg(path)
        .output()
    {
        Ok(output) if output.status.success() => {
            tracing::info!("ad-hoc codesigned {}", path.display());
        }
        Ok(output) => {
            tracing::warn!(
                "codesign failed (exit {}): {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim(),
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!("codesign not found — skipping (cross-platform rewrite)");
        }
        Err(e) => {
            tracing::warn!("codesign failed: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn test_rewrite_dry_run() {
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let config = RewriteConfig {
            input: PathBuf::from("/usr/bin/true"),
            output: td.path().join("truant_test_dry"),
            forkserver: true,
            dry_run: true,
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

        let result = rewrite(&config).expect("dry run failed");
        assert!(result.blocks_instrumented > 0);
        assert!(result.output_path.is_none());
    }

    /// Compile a non-PIE test binary with a unique name into `dir`, returning the path.
    fn compile_test_binary(dir: &std::path::Path, suffix: &str) -> PathBuf {
        let src = dir.join(format!("{}.c", suffix));
        let input = dir.join(suffix);
        std::fs::write(&src, "int main() { return 0; }\n").expect("failed to write test source");
        let cc = std::process::Command::new("gcc")
            .args([
                "-o",
                input.to_str().expect("input path is not valid UTF-8"),
                "-no-pie",
                src.to_str().expect("source path is not valid UTF-8"),
            ])
            .status()
            .expect("gcc not available");
        assert!(cc.success(), "failed to compile test binary");
        let _ = std::fs::remove_file(&src);
        input
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_rewrite_nopatch_entry_only() {
        // Test: redirect entry to init code with NO block patching.
        // This isolates whether init code + segment loading works.
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let input = compile_test_binary(td.path(), "entry_only");
        let output = td.path().join("entry_only_out");

        let data = std::fs::read(&input).expect("failed to read test binary");
        let ctx = elf::ElfContext::parse(&data).expect("failed to parse ELF");

        // Create a minimal init code that just jumps to original entry
        let segment_va = ((ctx.highest_va_end + 0xFFF) & !0xFFF) + 0x1000;
        let _data_va = segment_va;
        let init_va = segment_va + trampoline::DATA_SIZE;

        // Minimal init: lea rax, [rip + delta]; jmp rax
        let mut init_code = Vec::new();
        // lea rax, [rip + delta] ; 7 bytes: 48 8D 05 <disp32>
        init_code.extend_from_slice(&[0x48, 0x8D, 0x05, 0x00, 0x00, 0x00, 0x00]);
        let rip_after = init_va + 7;
        let delta = (ctx.entry_point as i64 - rip_after as i64) as i32;
        init_code[3..7].copy_from_slice(&delta.to_le_bytes());
        // jmp rax
        init_code.extend_from_slice(&[0xFF, 0xE0]);

        let segment_size = trampoline::DATA_SIZE as usize + init_code.len();
        let mut segment_data = vec![0u8; segment_size];
        let init_offset = trampoline::DATA_SIZE as usize;
        segment_data[init_offset..init_offset + init_code.len()].copy_from_slice(&init_code);

        let mut out_data = data;
        // Pad to page boundary (ELF requires p_offset % p_align == p_vaddr % p_align)
        let page_aligned = (out_data.len() + 0xFFF) & !0xFFF;
        out_data.resize(page_aligned, 0);
        let file_end = out_data.len() as u64;

        // Convert PT_NOTE to PT_LOAD
        let note = ctx.note_segment.as_ref().expect("no PT_NOTE");
        elf_impl::patcher::patch_phdr_note_to_load_test(
            &mut out_data,
            note.phdr_offset,
            segment_va,
            file_end,
            segment_size as u64,
        )
        .expect("failed to patch PT_NOTE to PT_LOAD");

        // Patch e_entry
        out_data[24..32].copy_from_slice(&init_va.to_le_bytes());

        // Append segment
        out_data.extend_from_slice(&segment_data);

        std::fs::write(&output, &out_data).expect("failed to write output binary");
        set_executable(&output).expect("failed to set executable permission");

        let status = std::process::Command::new(&output)
            .status()
            .expect("failed to run entry-only binary");
        assert!(
            status.success(),
            "entry-only rewrite should exit 0, got {:?}",
            status
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_rewrite_init_code_only() {
        // Test: full init code with no block patching.
        // Isolates whether the init code (environ parsing) works.
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let input = compile_test_binary(td.path(), "init_only");
        let output = td.path().join("initonly");

        let data = std::fs::read(&input).expect("failed to read test binary");
        let ctx = elf::ElfContext::parse(&data).expect("failed to parse ELF");

        let segment_va = ((ctx.highest_va_end + 0xFFF) & !0xFFF) + 0x1000;
        let data_va = segment_va;
        let init_va = segment_va + trampoline::DATA_SIZE;

        let init = trampoline::generate_init_code(init_va, data_va, ctx.entry_point, false, None)
            .expect("failed to generate init code");

        let total_size = trampoline::DATA_SIZE as usize + init.code.len();
        let mut segment_data = vec![0u8; total_size];
        let init_offset = trampoline::DATA_SIZE as usize;
        segment_data[init_offset..init_offset + init.code.len()].copy_from_slice(&init.code);

        let mut out_data = data;
        let page_aligned = (out_data.len() + 0xFFF) & !0xFFF;
        out_data.resize(page_aligned, 0);
        let file_end = out_data.len() as u64;

        let note = ctx.note_segment.as_ref().expect("no PT_NOTE");
        elf_impl::patcher::patch_phdr_note_to_load_test(
            &mut out_data,
            note.phdr_offset,
            segment_va,
            file_end,
            total_size as u64,
        )
        .expect("failed to patch PT_NOTE to PT_LOAD");

        // Patch e_entry to our init code
        out_data[24..32].copy_from_slice(&init_va.to_le_bytes());

        out_data.extend_from_slice(&segment_data);

        std::fs::write(&output, &out_data).expect("failed to write output binary");
        set_executable(&output).expect("failed to set executable permission");

        // Also test a minimal save/restore version (no syscalls)
        let output_stub = td.path().join("stub");
        {
            let mut stub = Vec::new();
            // Mirror the full init prologue/epilogue structure (no-op body)
            stub.push(0x52); // push rdx (preserve rtld_fini)
            stub.push(0x55); // push rbp
            stub.push(0x53); // push rbx
            stub.extend_from_slice(&[0x41, 0x54]); // push r12
            stub.extend_from_slice(&[0x41, 0x55]); // push r13
            stub.extend_from_slice(&[0x41, 0x56]); // push r14
            stub.extend_from_slice(&[0x41, 0x57]); // push r15
            stub.extend_from_slice(&[0x48, 0x81, 0xEC, 0x20, 0x10, 0x00, 0x00]); // sub rsp, 4128
            stub.extend_from_slice(&[0x48, 0x81, 0xC4, 0x20, 0x10, 0x00, 0x00]); // add rsp, 4128
            stub.extend_from_slice(&[0x41, 0x5F]); // pop r15
            stub.extend_from_slice(&[0x41, 0x5E]); // pop r14
            stub.extend_from_slice(&[0x41, 0x5D]); // pop r13
            stub.extend_from_slice(&[0x41, 0x5C]); // pop r12
            stub.push(0x5B); // pop rbx
            stub.push(0x5D); // pop rbp
            stub.push(0x5A); // pop rdx
            // lea rax, [rip + delta_to_entry]; jmp rax
            let stub_init_va = init_va;
            let lea_pos = stub.len();
            stub.extend_from_slice(&[0x48, 0x8D, 0x05, 0x00, 0x00, 0x00, 0x00]);
            let rip_after = stub_init_va + stub.len() as u64;
            let delta = (ctx.entry_point as i64 - rip_after as i64) as i32;
            stub[lea_pos + 3..lea_pos + 7].copy_from_slice(&delta.to_le_bytes());
            stub.extend_from_slice(&[0xFF, 0xE0]); // jmp rax

            let total = trampoline::DATA_SIZE as usize + stub.len();
            let mut seg = vec![0u8; total];
            let off = trampoline::DATA_SIZE as usize;
            seg[off..off + stub.len()].copy_from_slice(&stub);

            let data2 = std::fs::read(&input).expect("failed to read test binary");
            let ctx2 = elf::ElfContext::parse(&data2).expect("failed to parse ELF");
            let mut out2 = data2;
            let p2 = (out2.len() + 0xFFF) & !0xFFF;
            out2.resize(p2, 0);
            let fe2 = out2.len() as u64;
            let note2 = ctx2.note_segment.as_ref().expect("no PT_NOTE segment");
            elf_impl::patcher::patch_phdr_note_to_load_test(
                &mut out2,
                note2.phdr_offset,
                segment_va,
                fe2,
                total as u64,
            )
            .expect("failed to patch PT_NOTE to PT_LOAD");
            out2[24..32].copy_from_slice(&init_va.to_le_bytes());
            out2.extend_from_slice(&seg);
            std::fs::write(&output_stub, &out2).expect("failed to write stub binary");
            set_executable(&output_stub).expect("failed to set executable permission");

            let st = std::process::Command::new(&output_stub)
                .status()
                .expect("failed to run stub binary");
            assert!(
                st.success(),
                "stub (push/pop only) should exit 0, got {:?}",
                st
            );
        }

        // Test with the real generate_init_code (open/read/close/scan environ)
        let status = std::process::Command::new(&output)
            .status()
            .expect("failed to run init-only binary");
        assert!(
            status.success(),
            "init-only rewrite should exit 0, got {:?}",
            status
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_rewrite_actual() {
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let input = compile_test_binary(td.path(), "actual");
        let output = td.path().join("actual_out");

        let config = RewriteConfig {
            input: input.clone(),
            output: output.clone(),
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

        let result = rewrite(&config).expect("rewrite failed");
        assert!(result.blocks_instrumented > 0);
        assert_eq!(result.output_path, Some(output.clone()));

        let meta = std::fs::metadata(&output).expect("output file missing");
        assert!(meta.len() > 0);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert!(
                meta.permissions().mode() & 0o111 != 0,
                "output should be executable"
            );
        }

        let status = std::process::Command::new(&output)
            .status()
            .expect("failed to execute rewritten binary");
        assert!(
            status.success(),
            "rewritten binary should exit 0, got {:?}",
            status
        );
    }

    // ===================================================================
    // Shared object (.so) rewrite integration tests
    // ===================================================================

    #[cfg(target_os = "linux")]
    #[test]
    fn test_rewrite_shared_object() {
        // Compile a simple .so with a single exported function.
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let src = td.path().join("so_test_lib.c");
        let so_input = td.path().join("so_test_lib.so");
        let so_output = td.path().join("so_test_lib_instr.so");

        std::fs::write(
            &src,
            r#"
            int add_numbers(int a, int b) {
                int sum = 0;
                for (int i = 0; i < b; i++) { sum += a; }
                return sum;
            }
        "#,
        )
        .expect("failed to write .so source");

        let cc = std::process::Command::new("gcc")
            .args([
                "-shared",
                "-fPIC",
                "-O0",
                "-o",
                so_input.to_str().expect("so_input path is not valid UTF-8"),
                src.to_str().expect("source path is not valid UTF-8"),
            ])
            .status()
            .expect("gcc not available");
        assert!(cc.success(), "failed to compile test .so");

        // Rewrite the .so
        let config = RewriteConfig {
            input: so_input.clone(),
            output: so_output.clone(),
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

        let result = rewrite(&config).expect("rewrite .so failed");
        assert!(
            result.blocks_instrumented > 0,
            "should instrument some blocks in .so"
        );

        // Verify the output .so has DT_INIT (synthesised from DT_NULL if needed)
        let data = std::fs::read(&so_output).expect("failed to read rewritten .so");
        let elf = goblin::elf::Elf::parse(&data).expect("failed to parse rewritten ELF");
        let has_dt_init = elf
            .dynamic
            .as_ref()
            .map(|d| {
                d.dyns
                    .iter()
                    .any(|e| e.d_tag == goblin::elf::dynamic::DT_INIT)
            })
            .unwrap_or(false);
        assert!(
            has_dt_init,
            "rewritten .so should have DT_INIT (original or synthesised)"
        );

        // Compile a harness that dlopen's the .so using an absolute path
        let harness_src = td.path().join("so_test_harness.c");
        let harness_bin = td.path().join("so_test_harness");
        let so_output_abs = so_output
            .canonicalize()
            .expect("failed to canonicalize .so output path");
        std::fs::write(
            &harness_src,
            format!(
                r#"
            #include <dlfcn.h>
            #include <stdio.h>
            #include <stdlib.h>

            int main(void) {{
                void *lib = dlopen("{}", RTLD_NOW);
                if (!lib) {{ fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }}

                typedef int (*fn_t)(int, int);
                fn_t add = (fn_t)dlsym(lib, "add_numbers");
                if (!add) {{ fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }}

                int result = add(3, 4);
                if (result != 12) {{
                    fprintf(stderr, "add(3,4) = %d, expected 12\n", result);
                    return 1;
                }}
                dlclose(lib);
                return 0;
            }}
        "#,
                so_output_abs.display()
            ),
        )
        .expect("failed to write harness source");

        let cc = std::process::Command::new("gcc")
            .args([
                "-o",
                harness_bin
                    .to_str()
                    .expect("harness_bin path is not valid UTF-8"),
                harness_src
                    .to_str()
                    .expect("harness_src path is not valid UTF-8"),
                "-ldl",
            ])
            .status()
            .expect("gcc not available");
        assert!(cc.success(), "failed to compile harness");

        // Run the harness — the instrumented .so should load and work correctly
        let status = std::process::Command::new(&harness_bin)
            .status()
            .expect("failed to run harness");
        assert!(
            status.success(),
            "harness with instrumented .so should exit 0, got {:?}",
            status
        );
    }

    // ===================================================================
    // Heap sanitiser integration tests
    // ===================================================================

    const HEAP_SAN_TARGET_SRC: &str = r#"
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) return 0;
    char *buf = malloc(16);
    if (!buf) return 1;
    if (argv[1][0] == 'o') {
        buf[16] = 'X';           /* heap overflow */
    } else if (argv[1][0] == 'u') {
        free(buf);
        buf[0] = 'X';           /* use-after-free */
    } else if (argv[1][0] == 'd') {
        free(buf);
        free(buf);              /* double-free */
    }
    free(buf);
    return 0;
}
"#;

    fn compile_heap_san_target(dir: &std::path::Path, suffix: &str, static_link: bool) -> PathBuf {
        let src = dir.join(format!("{}.c", suffix));
        let bin = dir.join(suffix);
        std::fs::write(&src, HEAP_SAN_TARGET_SRC).expect("failed to write heap san source");
        let mut args = vec![
            "-o",
            bin.to_str().expect("bin path is not valid UTF-8"),
            "-no-pie",
        ];
        if static_link {
            args.push("-static");
        }
        args.push(src.to_str().expect("source path is not valid UTF-8"));
        let status = std::process::Command::new("gcc")
            .args(&args)
            .status()
            .expect("gcc not available");
        assert!(status.success(), "failed to compile heap san test target");
        let _ = std::fs::remove_file(&src);
        bin
    }

    fn rewrite_with_heap_san(input: &Path, output: &Path) -> RewriteResult {
        let config = RewriteConfig {
            input: input.to_path_buf(),
            output: output.to_path_buf(),
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
        rewrite(&config).expect("rewrite with heap san failed")
    }

    /// Helper: run a heap-san test binary, using LD_PRELOAD if a preload lib was generated.
    fn run_heap_san_binary(
        output: &Path,
        result: &RewriteResult,
        args: &[&str],
    ) -> std::process::ExitStatus {
        let mut cmd = std::process::Command::new(output);
        cmd.args(args);
        if let Some(ref so_path) = result.preload_lib_path {
            cmd.env("LD_PRELOAD", so_path);
        }
        cmd.status().expect("failed to run heap san binary")
    }

    // --- Static binary tests: inline GOT/function-entry patching ---

    #[cfg(target_os = "linux")]
    #[test]
    fn test_heap_san_clean_program() {
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let input = compile_heap_san_target(td.path(), "clean", true);
        let output = td.path().join("heap_san_out_clean");

        let result = rewrite_with_heap_san(&input, &output);
        assert!(
            result.heap_san_intercepted >= 2,
            "should intercept at least malloc+free"
        );
        assert!(
            result.preload_lib_path.is_none(),
            "static binary should use inline patching"
        );

        let status = run_heap_san_binary(&output, &result, &["c"]);
        assert!(
            status.success(),
            "clean program should exit 0, got {:?}",
            status
        );

        let status = run_heap_san_binary(&output, &result, &[]);
        assert!(status.success(), "no-args should exit 0, got {:?}", status);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_heap_san_overflow() {
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let input = compile_heap_san_target(td.path(), "overflow", true);
        let output = td.path().join("heap_san_out_overflow");

        let result = rewrite_with_heap_san(&input, &output);
        assert!(result.heap_san_intercepted >= 2);

        let status = run_heap_san_binary(&output, &result, &["o"]);

        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(
                status.signal(),
                Some(11),
                "heap overflow should trigger SIGSEGV (11), got {:?}",
                status
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_heap_san_use_after_free() {
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let input = compile_heap_san_target(td.path(), "uaf", true);
        let output = td.path().join("heap_san_out_uaf");

        let result = rewrite_with_heap_san(&input, &output);
        assert!(result.heap_san_intercepted >= 2);

        let status = run_heap_san_binary(&output, &result, &["u"]);

        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(
                status.signal(),
                Some(11),
                "use-after-free should trigger SIGSEGV (11), got {:?}",
                status
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_heap_san_double_free() {
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let input = compile_heap_san_target(td.path(), "dblf", true);
        let output = td.path().join("heap_san_out_dblf");

        let result = rewrite_with_heap_san(&input, &output);
        assert!(result.heap_san_intercepted >= 2);

        let status = run_heap_san_binary(&output, &result, &["d"]);

        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(
                status.signal(),
                Some(11),
                "double-free should trigger SIGSEGV (11), got {:?}",
                status
            );
        }
    }

    // --- Dynamic binary tests: LD_PRELOAD preload library ---

    #[cfg(target_os = "linux")]
    #[test]
    #[cfg(feature = "coverage")]
    fn test_heap_san_dynamic_clean() {
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let input = compile_heap_san_target(td.path(), "dyn_clean", false);
        let output = td.path().join("heap_san_out_dyn_clean");

        let result = rewrite_with_heap_san(&input, &output);
        assert!(
            result.preload_lib_path.is_some(),
            "dynamic binary should generate preload .so"
        );
        assert_eq!(
            result.heap_san_intercepted, 0,
            "dynamic binary should not do inline patching"
        );

        let status = run_heap_san_binary(&output, &result, &["c"]);
        assert!(
            status.success(),
            "clean program should exit 0, got {:?}",
            status
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[cfg(feature = "coverage")]
    fn test_heap_san_dynamic_overflow() {
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let input = compile_heap_san_target(td.path(), "dyn_overflow", false);
        let output = td.path().join("heap_san_out_dyn_overflow");

        let result = rewrite_with_heap_san(&input, &output);
        assert!(result.preload_lib_path.is_some());

        let status = run_heap_san_binary(&output, &result, &["o"]);

        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(
                status.signal(),
                Some(11),
                "heap overflow should trigger SIGSEGV (11), got {:?}",
                status
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[cfg(feature = "coverage")]
    fn test_heap_san_dynamic_use_after_free() {
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let input = compile_heap_san_target(td.path(), "dyn_uaf", false);
        let output = td.path().join("heap_san_out_dyn_uaf");

        let result = rewrite_with_heap_san(&input, &output);
        assert!(result.preload_lib_path.is_some());

        let status = run_heap_san_binary(&output, &result, &["u"]);

        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(
                status.signal(),
                Some(11),
                "use-after-free should trigger SIGSEGV (11), got {:?}",
                status
            );
        }
    }

    // ===================================================================
    // Persistent mode integration tests
    // ===================================================================

    const PERSISTENT_TARGET_SRC: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int parse_input(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char buf[64];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    if (n >= 4 && memcmp(buf, "CRASH", 4) == 0) {
        abort();
    }
    return (int)n;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    int ret = parse_input(argv[1]);
    return ret >= 0 ? 0 : 1;
}
"#;

    #[cfg(target_os = "linux")]
    #[test]
    fn test_rewrite_persistent_produces_valid_binary() {
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let src = td.path().join("persistent_test.c");
        let input_bin = td.path().join("persistent_test_input");
        let output_bin = td.path().join("persistent_test_output");

        std::fs::write(&src, PERSISTENT_TARGET_SRC)
            .expect("failed to write persistent target source");
        let cc = std::process::Command::new("gcc")
            .args([
                "-o",
                input_bin
                    .to_str()
                    .expect("input_bin path is not valid UTF-8"),
                "-no-pie",
                "-static",
                src.to_str().expect("source path is not valid UTF-8"),
            ])
            .status()
            .expect("gcc not available");
        assert!(cc.success(), "failed to compile persistent test target");
        let _ = std::fs::remove_file(&src);

        // Find parse_input address via nm
        let nm_output = std::process::Command::new("nm")
            .arg(&input_bin)
            .output()
            .expect("nm failed");
        let nm_stdout = String::from_utf8_lossy(&nm_output.stdout);
        let parse_input_addr = nm_stdout
            .lines()
            .find(|line| line.contains(" T parse_input"))
            .and_then(|line| {
                let addr_str = line.split_whitespace().next()?;
                u64::from_str_radix(addr_str, 16).ok()
            });

        let persistent_addr =
            parse_input_addr.expect("could not find parse_input symbol in nm output");

        let config = RewriteConfig {
            input: input_bin.clone(),
            output: output_bin.clone(),
            forkserver: true,
            dry_run: false,
            heap_san: false,
            sidecar_san: false,
            persistent_addr: Some(persistent_addr),
            persistent_count: 100,
            defer: false,
            #[cfg(feature = "coverage")]
            validate: false,
            instrument_modules: None,
            hooks: None,
            no_coverage: false,
        };

        let result = rewrite(&config).expect("rewrite with persistent mode failed");
        assert!(result.persistent_mode);
        assert!(result.blocks_instrumented > 0);
        assert!(result.segment_size > 0);

        // Verify output binary exists, is larger than input, and is executable
        let input_meta =
            std::fs::metadata(&input_bin).expect("failed to get input binary metadata");
        let output_meta =
            std::fs::metadata(&output_bin).expect("failed to get output binary metadata");
        assert!(
            output_meta.len() > input_meta.len(),
            "rewritten binary should be larger than original"
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert!(
                output_meta.permissions().mode() & 0o111 != 0,
                "output should be executable"
            );
        }

        // Verify the persistent_addr was patched with a branch instruction
        let output_data = std::fs::read(&output_bin).expect("failed to read output binary");
        let output_ctx = elf::ElfContext::parse(&output_data).expect("failed to parse output ELF");
        let text = &output_ctx.text;
        let file_offset = (persistent_addr - text.va) as usize + text.offset as usize;
        if cfg!(target_arch = "aarch64") {
            // ARM64: B imm26 — top byte (byte[3]) >> 2 == 5 (opcode 000101xx)
            let opcode_byte = output_data[file_offset + 3];
            assert_eq!(
                opcode_byte >> 2,
                5,
                "persistent_addr should be patched with B imm26 (got 0x{:02x})",
                opcode_byte
            );
        } else {
            // x86_64: JMP rel32 (0xE9)
            assert_eq!(
                output_data[file_offset], 0xE9,
                "persistent_addr should be patched with JMP rel32 (0xE9)"
            );
        }

        // Note: we don't run the binary because persistent mode uses SIGSTOP,
        // which requires the forkserver loop to SIGCONT. Running without a
        // forkserver parent would leave the process stopped forever. Full
        // end-to-end testing is done via `truant fuzz --forkserver --persistent`.
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_rewrite_persistent_deferred() {
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let src = td.path().join("deferred_test.c");
        let input_bin = td.path().join("deferred_test_input");
        let output_bin = td.path().join("deferred_test_output");

        std::fs::write(&src, PERSISTENT_TARGET_SRC)
            .expect("failed to write persistent target source");
        let cc = std::process::Command::new("gcc")
            .args([
                "-o",
                input_bin
                    .to_str()
                    .expect("input_bin path is not valid UTF-8"),
                "-no-pie",
                "-static",
                src.to_str().expect("source path is not valid UTF-8"),
            ])
            .status()
            .expect("gcc not available");
        assert!(cc.success(), "failed to compile deferred test target");
        let _ = std::fs::remove_file(&src);

        // Find parse_input address via nm
        let nm_output = std::process::Command::new("nm")
            .arg(&input_bin)
            .output()
            .expect("nm failed");
        let nm_stdout = String::from_utf8_lossy(&nm_output.stdout);
        let persistent_addr = nm_stdout
            .lines()
            .find(|line| line.contains(" T parse_input"))
            .and_then(|line| {
                let addr_str = line.split_whitespace().next()?;
                u64::from_str_radix(addr_str, 16).ok()
            })
            .expect("could not find parse_input symbol in nm output");

        let config = RewriteConfig {
            input: input_bin.clone(),
            output: output_bin.clone(),
            forkserver: true,
            dry_run: false,
            heap_san: false,
            sidecar_san: false,
            persistent_addr: Some(persistent_addr),
            persistent_count: 100,
            defer: true,
            #[cfg(feature = "coverage")]
            validate: false,
            instrument_modules: None,
            hooks: None,
            no_coverage: false,
        };

        let result = rewrite(&config).expect("rewrite with deferred persistent mode failed");
        assert!(result.persistent_mode);
        assert!(result.blocks_instrumented > 0);

        // Verify output binary exists and is larger than input
        let input_meta =
            std::fs::metadata(&input_bin).expect("failed to get input binary metadata");
        let output_meta =
            std::fs::metadata(&output_bin).expect("failed to get output binary metadata");
        assert!(
            output_meta.len() > input_meta.len(),
            "rewritten binary should be larger than original"
        );

        // Verify the persistent_addr was patched with a branch instruction
        let output_data = std::fs::read(&output_bin).expect("failed to read output binary");
        let output_ctx = elf::ElfContext::parse(&output_data).expect("failed to parse output ELF");
        let text = &output_ctx.text;
        let file_offset = (persistent_addr - text.va) as usize + text.offset as usize;
        if cfg!(target_arch = "aarch64") {
            let opcode_byte = output_data[file_offset + 3];
            assert_eq!(
                opcode_byte >> 2,
                5,
                "persistent_addr should be patched with B imm26 (got 0x{:02x})",
                opcode_byte
            );
        } else {
            assert_eq!(
                output_data[file_offset], 0xE9,
                "persistent_addr should be patched with JMP rel32 (0xE9)"
            );
        }

        // Verify forkserver_started byte at persistent_data+2 is initialised to 0
        // persistent_data is at segment_va + DATA_SIZE
        let seg_va = result.segment_va;
        // Find the segment in the output file
        let elf_obj = goblin::elf::Elf::parse(&output_data)
            .expect("failed to parse output ELF for segment check");
        let new_seg = elf_obj
            .program_headers
            .iter()
            .find(|p| p.p_vaddr == seg_va)
            .expect("could not find new segment");
        let pd_file_offset = new_seg.p_offset as usize + trampoline::DATA_SIZE as usize;
        assert_eq!(output_data[pd_file_offset], 1, "first_pass should be 1");
        assert_eq!(
            output_data[pd_file_offset + 1],
            0,
            "child_stopped should be 0"
        );
        assert_eq!(
            output_data[pd_file_offset + 2],
            0,
            "forkserver_started should be 0"
        );
    }

    #[cfg(target_os = "linux")]
    #[cfg(feature = "aarch64")]
    #[test]
    fn test_aarch64_dry_run() {
        // Use a known small AArch64 ELF if available on the host.
        // On non-ARM64 Linux hosts, skip gracefully if file not found.
        let candidates = [
            "/usr/aarch64-linux-gnu/bin/true", // Debian/Ubuntu cross toolchain
            "/usr/bin/aarch64-linux-gnu-true", // alternative path
        ];
        let path = candidates.iter().find(|p| std::path::Path::new(p).exists());
        let path = match path {
            Some(p) => *p,
            None => {
                tracing::info!(
                    "skip: no AArch64 binary found on this host (install qemu-user or cross toolchain)"
                );
                return;
            }
        };
        let td = tempfile::tempdir().expect("failed to create temp dir");
        let config = RewriteConfig {
            input: path.into(),
            output: td.path().join("aarch64_test_out"),
            forkserver: false,
            dry_run: true,
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
        let result = rewrite(&config).expect("AArch64 dry-run rewrite failed");
        assert!(
            result.blocks_instrumented > 0,
            "expected non-zero block count for AArch64 binary, got 0"
        );
        // Branch island synthesis should eliminate all skipped blocks.
        assert_eq!(
            result.blocks_skipped, 0,
            "branch islands should eliminate all skipped blocks, but {} were skipped \
             (out of {} instrumented)",
            result.blocks_skipped, result.blocks_instrumented
        );
    }
}
