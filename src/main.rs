//! Truant — static binary rewriter for arbitrary function hooking.
//!
//! Usage:
//!   truant <input> -o <output> [--hooks <spec.toml>] [options]
//!   truant <input> --info
//!   truant <input> --dry-run

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Arg, ArgAction, Command};

fn main() -> Result<()> {
    let cli = Command::new("truant")
        .about("Truant — static binary rewriter for arbitrary function hooking")
        .version(env!("CARGO_PKG_VERSION"))
        .arg(
            Arg::new("input")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf))
                .help("Input binary (ELF, Mach-O, or PE)"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .action(ArgAction::SetTrue)
                .global(true)
                .help("Enable verbose logging"),
        )
        .arg(
            Arg::new("info")
                .long("info")
                .action(ArgAction::SetTrue)
                .help("Display binary format and architecture info, then exit"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_parser(clap::value_parser!(PathBuf))
                .help("Output rewritten binary"),
        )
        .arg(
            Arg::new("hooks")
                .long("hooks")
                .value_parser(clap::value_parser!(PathBuf))
                .help("TOML hook specification file"),
        )
        .arg(
            Arg::new("no_coverage")
                .long("no-coverage")
                .action(ArgAction::SetTrue)
                .help("Apply hooks only — skip coverage instrumentation"),
        )
        .arg(
            Arg::new("no_forkserver")
                .long("no-forkserver")
                .action(ArgAction::SetTrue)
                .help("Omit forkserver from init code"),
        )
        .arg(
            Arg::new("dry_run")
                .long("dry-run")
                .action(ArgAction::SetTrue)
                .help("Analyse and report without writing output"),
        )
        .arg(
            Arg::new("validate")
                .long("validate")
                .action(ArgAction::SetTrue)
                .help("Rewrite to temp, run, and check edge count"),
        )
        .arg(
            Arg::new("heap_san")
                .long("heap-san")
                .action(ArgAction::SetTrue)
                .help("Enable guard-page heap sanitiser (GOT/PLT patching)"),
        )
        .arg(
            Arg::new("sidecar_san")
                .long("sidecar-san")
                .action(ArgAction::SetTrue)
                .help("Generate sidecar LD_PRELOAD sanitiser library"),
        )
        .arg(
            Arg::new("persistent_addr")
                .long("persistent-addr")
                .value_parser(parse_hex_u64)
                .help("Hex address for persistent mode loop (e.g. 0x401000)"),
        )
        .arg(
            Arg::new("persistent_count")
                .long("persistent-count")
                .value_parser(clap::value_parser!(u64))
                .default_value("1000")
                .help("Iterations per forkserver child in persistent mode"),
        )
        .arg(
            Arg::new("no_defer")
                .long("no-defer")
                .action(ArgAction::SetTrue)
                .help("Place forkserver at e_entry instead of persistent-addr"),
        )
        .arg(
            Arg::new("instrument_modules")
                .long("instrument-modules")
                .num_args(1..)
                .help("Only instrument named shared libraries (multi-module)"),
        );

    let matches = cli.get_matches();

    // Initialise tracing.
    let filter = if matches.get_flag("verbose") {
        "debug"
    } else {
        "info"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_writer(std::io::stderr)
        .init();

    let input: &PathBuf = matches
        .get_one("input")
        .context("input binary is required")?;

    // --info mode: display format/arch and exit.
    if matches.get_flag("info") {
        return cmd_info(input);
    }

    // Default: rewrite.
    cmd_rewrite(input, &matches)
}

fn parse_hex_u64(s: &str) -> Result<u64, String> {
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    u64::from_str_radix(s, 16).map_err(|e| format!("invalid hex address: {}", e))
}

// ---------------------------------------------------------------------------
// --info
// ---------------------------------------------------------------------------

fn cmd_info(input: &PathBuf) -> Result<()> {
    let data =
        std::fs::read(input).with_context(|| format!("failed to read {}", input.display()))?;

    let format = truant::detect_format(&data);
    let arch = truant::detect_architecture(&data);

    println!("File:         {}", input.display());
    println!("Format:       {:?}", format);
    println!("Architecture: {:?}", arch);
    println!("Size:         {} bytes", data.len());

    Ok(())
}

// ---------------------------------------------------------------------------
// rewrite (default operation)
// ---------------------------------------------------------------------------

fn cmd_rewrite(input: &Path, matches: &clap::ArgMatches) -> Result<()> {
    let validate = matches.get_flag("validate");
    let dry_run = matches.get_flag("dry_run");

    let output: PathBuf = if validate || dry_run {
        matches
            .get_one::<PathBuf>("output")
            .cloned()
            .unwrap_or_else(|| {
                if cfg!(windows) {
                    PathBuf::from("NUL")
                } else {
                    PathBuf::from("/dev/null")
                }
            })
    } else {
        matches
            .get_one::<PathBuf>("output")
            .cloned()
            .context("--output is required (or use --validate / --dry-run / --info)")?
    };

    let no_forkserver = matches.get_flag("no_forkserver");
    let heap_san = matches.get_flag("heap_san");
    let sidecar_san = matches.get_flag("sidecar_san");
    let persistent_count = matches
        .get_one::<u64>("persistent_count")
        .copied()
        .unwrap_or(1000) as u32;

    let persistent_addr: Option<u64> = matches.get_one::<u64>("persistent_addr").copied();

    let no_defer = matches.get_flag("no_defer");
    let defer = persistent_addr.is_some() && !no_defer;

    let instrument_modules: Option<Vec<String>> = matches
        .get_many::<String>("instrument_modules")
        .map(|vals| vals.cloned().collect());

    let hooks: Option<PathBuf> = matches.get_one::<PathBuf>("hooks").cloned();

    let no_coverage = matches.get_flag("no_coverage");

    let config = truant::RewriteConfig {
        input: input.to_path_buf(),
        output: output.clone(),
        forkserver: !no_forkserver,
        dry_run,
        heap_san,
        sidecar_san,
        persistent_addr,
        persistent_count,
        defer,
        #[cfg(feature = "coverage")]
        validate,
        instrument_modules,
        hooks,
        no_coverage,
    };

    if validate && no_coverage {
        anyhow::bail!(
            "--validate and --no-coverage are incompatible: validation requires coverage edges"
        );
    }

    #[cfg(all(feature = "coverage", unix))]
    if validate {
        let validated = truant::validate(&config)?;
        println!(
            "validation: {} edges, {} blocks instrumented",
            validated.edge_count, validated.blocks_instrumented,
        );
        if validated.success {
            println!("validation: PASS (edge count > 0)");
        } else {
            if let Some(ref err) = validated.error {
                eprintln!("validation: FAIL -- {}", err);
            } else {
                eprintln!(
                    "validation: FAIL -- edge count {} == 0 (no coverage observed)",
                    validated.edge_count
                );
            }
            std::process::exit(1);
        }
        return Ok(());
    }

    #[cfg(not(all(feature = "coverage", unix)))]
    if validate {
        anyhow::bail!(
            "--validate requires the 'coverage' feature on Unix (compile with: cargo build --features coverage)"
        );
    }

    let result = truant::rewrite(&config)?;

    if dry_run {
        println!(
            "dry run: {} basic blocks detected",
            result.blocks_instrumented
        );
    } else {
        println!(
            "rewritten: {} ({} blocks instrumented, {} skipped)",
            output.display(),
            result.blocks_instrumented,
            result.blocks_skipped,
        );
        if result.blocks_skipped > 0 {
            tracing::warn!(
                "{} blocks skipped due to relocation issues",
                result.blocks_skipped
            );
        }
        println!(
            "  new segment: VA=0x{:x} size={} bytes",
            result.segment_va, result.segment_size,
        );
        if result.persistent_mode {
            println!(
                "  persistent mode: addr=0x{:x} count={}{}",
                persistent_addr.expect("persistent_addr must be set when persistent_mode is true"),
                persistent_count,
                if defer { " (deferred forkserver)" } else { "" },
            );
        }
        if result.heap_san_intercepted > 0 {
            println!(
                "  heap sanitiser: {} allocator functions intercepted",
                result.heap_san_intercepted,
            );
        }
        if let Some(ref so_path) = result.preload_lib_path {
            println!("  heap sanitiser: LD_PRELOAD library generated");
            println!();
            println!("  Run with:");
            println!("    LD_PRELOAD={} {}", so_path.display(), output.display());
        }
        if let Some(ref so_path) = result.sidecar_san_lib_path {
            println!("  sidecar sanitiser: LD_PRELOAD library generated");
            println!();
            println!("  Auto-detected when present next to target binary.");
            println!("  Manual run:");
            println!("    LD_PRELOAD={} {}", so_path.display(), output.display());
        }
        if result.hooks_applied > 0 {
            println!("  hooks: {} applied", result.hooks_applied);
        }
        if let Some(ref so_path) = result.hook_preload_lib_path {
            println!("  hook companion: LD_PRELOAD library generated");
            println!();
            println!("  Run with:");
            println!("    LD_PRELOAD={} {}", so_path.display(), output.display());
        }
    }

    Ok(())
}
