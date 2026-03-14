//! Criterion benchmarks for truant.
//!
//! Covers trampoline generation (x86_64 + AArch64), init code generation,
//! hook trampoline generation, full ELF rewrite pipeline, and block discovery.

use criterion::{Criterion, criterion_group, criterion_main};
use std::path::PathBuf;

use truant::disasm::{BasicBlock, find_basic_blocks};
use truant::elf::ElfContext;
use truant::hook_trampoline::{TargetAbi, generate_hook_trampoline};
use truant::hooks::{HookMode, HookSource, ResolvedHook};
use truant::trampoline::{generate_init_code, generate_trampoline};
use truant::{RewriteConfig, rewrite};

// ---------------------------------------------------------------------------
// Group 1: Trampoline generation
// ---------------------------------------------------------------------------

fn bench_trampoline_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("trampoline_generation");

    // x86_64 coverage trampoline with a 5-byte NOP displaced instruction.
    let block_x86 = BasicBlock {
        va: 0x401000,
        file_offset: 0x1000,
        displaced_len: 5,
        displaced_bytes: vec![0x0F, 0x1F, 0x44, 0x00, 0x00],
        block_id: 0x1234,
    };

    group.bench_function("x86_64", |b| {
        b.iter(|| {
            generate_trampoline(
                0x600000, // trampoline_va
                0x500000, // data_va
                &block_x86,
            )
            .unwrap()
        })
    });

    // AArch64 coverage trampoline with a 4-byte NOP displaced instruction.
    #[cfg(feature = "aarch64")]
    {
        use truant::AArch64TrampolineGenerator;
        use truant::traits::TrampolineGenerator;

        let block_aarch64 = BasicBlock {
            va: 0x401000,
            file_offset: 0x1000,
            displaced_len: 4,
            displaced_bytes: vec![0x1F, 0x20, 0x03, 0xD5],
            block_id: 0x1234,
        };
        let tramp_gen = AArch64TrampolineGenerator::new();

        group.bench_function("aarch64", |b| {
            b.iter(|| {
                tramp_gen
                    .generate_trampoline(
                        0x600000, // trampoline_va
                        0x500000, // data_va
                        &block_aarch64,
                    )
                    .unwrap()
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: Init code generation
// ---------------------------------------------------------------------------

fn bench_init_code_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("init_code_generation");

    // x86_64 init code with forkserver enabled, 65536-byte map.
    group.bench_function("x86_64_forkserver", |b| {
        b.iter(|| {
            generate_init_code(
                0x600000, // init_va
                0x500000, // data_va
                0x401000, // original_entry
                true,     // enable_forkserver
                None,     // persistent_data_va
            )
            .unwrap()
        })
    });

    // AArch64 init code with forkserver.
    #[cfg(feature = "aarch64")]
    {
        use truant::AArch64TrampolineGenerator;
        use truant::traits::TrampolineGenerator;

        let tramp_gen = AArch64TrampolineGenerator::new();
        group.bench_function("aarch64_forkserver", |b| {
            b.iter(|| {
                tramp_gen
                    .generate_init_code(
                        0x600000, // init_va
                        0x500000, // data_va
                        0x401000, // original_entry
                        true,     // enable_forkserver
                        None,     // persistent_data_va
                    )
                    .unwrap()
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: Hook trampoline generation
// ---------------------------------------------------------------------------

fn bench_hook_trampoline_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hook_trampoline_generation");

    // x86_64 pre-hook with shellcode [0xC3] and 5-byte NOP displaced.
    let hook_x86 = ResolvedHook {
        target_va: 0x401000,
        file_offset: 0x1000,
        displaced_bytes: vec![0x0F, 0x1F, 0x44, 0x00, 0x00],
        displaced_len: 5,
        mode: HookMode::Pre,
        source: HookSource::Shellcode(vec![0xC3]),
        condition: None,
        toggle_index: 0,
        initial_enabled: true,
        return_slot_index: None,
    };

    group.bench_function("x86_64_pre_hook", |b| {
        b.iter(|| {
            generate_hook_trampoline(
                0x600000, // trampoline_va
                &hook_x86,
                0x500000,       // hook_data_va
                Some(0x700000), // shellcode_va
                TargetAbi::SysV64,
                Some(0x500100), // toggle_va
            )
            .unwrap()
        })
    });

    // AArch64 pre-hook with shellcode [0xC3] and 4-byte NOP displaced.
    #[cfg(feature = "aarch64")]
    {
        let hook_aarch64 = ResolvedHook {
            target_va: 0x401000,
            file_offset: 0x1000,
            displaced_bytes: vec![0x1F, 0x20, 0x03, 0xD5],
            displaced_len: 4,
            mode: HookMode::Pre,
            source: HookSource::Shellcode(vec![0xC3]),
            condition: None,
            toggle_index: 0,
            initial_enabled: true,
            return_slot_index: None,
        };

        group.bench_function("aarch64_pre_hook", |b| {
            b.iter(|| {
                generate_hook_trampoline(
                    0x600000, // trampoline_va
                    &hook_aarch64,
                    0x500000,       // hook_data_va
                    Some(0x700000), // shellcode_va
                    TargetAbi::Aarch64,
                    Some(0x500100), // toggle_va
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: Full pipeline (ELF rewrite + block discovery)
// ---------------------------------------------------------------------------

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    // These benchmarks involve file I/O, so use longer times.
    group.warm_up_time(std::time::Duration::from_secs(3));
    group.measurement_time(std::time::Duration::from_secs(10));

    let input_path = PathBuf::from("/usr/bin/true");
    if !input_path.exists() {
        eprintln!("Skipping full_pipeline benchmarks: /usr/bin/true not found");
        group.finish();
        return;
    }

    // Benchmark: block discovery on /usr/bin/true.
    let data = std::fs::read(&input_path).expect("failed to read /usr/bin/true");
    let ctx = ElfContext::parse(&data).expect("failed to parse ELF context for /usr/bin/true");

    group.bench_function("block_discovery", |b| {
        b.iter(|| find_basic_blocks(&ctx, &data, &None).unwrap())
    });

    // Benchmark: full rewrite pipeline on /usr/bin/true.
    group.bench_function("full_elf_rewrite", |b| {
        let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let output_path = tmp_dir.path().join("true_rewritten");

        b.iter(|| {
            let config = RewriteConfig {
                input: input_path.clone(),
                output: output_path.clone(),
                forkserver: true,
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
            rewrite(&config).unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_trampoline_generation,
    bench_init_code_generation,
    bench_hook_trampoline_generation,
    bench_full_pipeline,
);
criterion_main!(benches);
