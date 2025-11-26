//! Parallel single-input encoding benchmarks.
//!
//! ## Benchmark Organization
//!
//! - **Core**: Serial vs parallel vs streaming comparison at key sizes
//! - **Optimization**: Tests for specific optimizations (safe boundaries, LazyEncoding, block sizes)
//! - **Edge Cases**: Worst-case and pathological inputs
//!
//! ## Running Benchmarks
//!
//! ```bash
//! # Run all benchmarks (fast, ~5 minutes)
//! cargo bench --bench parallel_single_benchmark
//!
//! # Run specific group
//! cargo bench --bench parallel_single_benchmark -- core_comparison
//! cargo bench --bench parallel_single_benchmark -- optimization
//! cargo bench --bench parallel_single_benchmark -- edge_cases
//! ```

#[macro_use]
extern crate criterion;

mod common;

use criterion::{BenchmarkId, Criterion, Throughput};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::tokenizer::parallel_encode::{ParallelConfig, ParallelMode};
use tokenizers::{decoders, Model, Tokenizer, TokenizerImpl};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Standard sizes for most benchmarks (covers key thresholds)
const SIZES_STANDARD: &[usize] = &[
    10_000,   // 10 KB - parallel threshold
    100_000,  // 100 KB - recursive sweet spot
    500_000,  // 500 KB - crossover point
    1_000_000, // 1 MB - streaming threshold
    4_000_000, // 4 MB - large input
];

/// Sizes for quick sanity checks
const SIZES_QUICK: &[usize] = &[100_000, 1_000_000];

/// Block sizes to test for streaming optimization (including very small)
const BLOCK_SIZES_KB: &[usize] = &[4, 8, 16, 32, 64];

// =============================================================================
// LAZY TOKENIZERS (created once, reused)
// =============================================================================

type BertTokenizer = TokenizerImpl<
    WordPiece,
    BertNormalizer,
    BertPreTokenizer,
    BertProcessing,
    decoders::wordpiece::WordPiece,
>;

static BERT_TOKENIZER: OnceLock<BertTokenizer> = OnceLock::new();
static BPE_TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

fn get_bert_tokenizer() -> &'static BertTokenizer {
    BERT_TOKENIZER.get_or_init(|| {
        let wp = WordPiece::from_file("data/bert-base-uncased-vocab.txt")
            .build()
            .expect("Vocab file not found, run `make test` first");

        let sep_id = *wp.get_vocab().get("[SEP]").unwrap();
        let cls_id = *wp.get_vocab().get("[CLS]").unwrap();

        let mut tokenizer = TokenizerImpl::new(wp);
        tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
        tokenizer.with_normalizer(Some(BertNormalizer::default()));
        tokenizer.with_decoder(Some(decoders::wordpiece::WordPiece::default()));
        tokenizer.with_post_processor(Some(BertProcessing::new(
            ("[SEP]".to_string(), sep_id),
            ("[CLS]".to_string(), cls_id),
        )));
        tokenizer
    })
}

fn get_bpe_tokenizer() -> &'static Tokenizer {
    BPE_TOKENIZER.get_or_init(|| {
        let bpe = BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
            .build()
            .expect("BPE files not found, run `make test` first");

        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        tokenizer.with_post_processor(Some(ByteLevel::default()));
        tokenizer
    })
}

// =============================================================================
// TEXT GENERATORS
// =============================================================================

/// Generate prose text (normal word boundaries)
fn generate_prose(size_bytes: usize) -> String {
    let base = "The quick brown fox jumps over the lazy dog. \
                This is a sample sentence for benchmarking. ";
    let mut text = base.repeat((size_bytes / base.len()) + 1);
    text.truncate(size_bytes);
    text
}

/// Generate text WITH clear paragraph/sentence boundaries (triggers safe boundary detection)
fn generate_with_safe_boundaries(size_bytes: usize) -> String {
    let base = "This is the first sentence. Here is another one. And a third sentence too.\n\n\
                New paragraph begins here. It has multiple sentences. Each one is clear.\n\n\
                Yet another paragraph. The boundaries are obvious. Split points are safe.\n\n";
    let mut text = base.repeat((size_bytes / base.len()) + 1);
    text.truncate(size_bytes);
    text
}

/// Generate text WITHOUT clear boundaries (continuous prose, no paragraph breaks)
fn generate_without_safe_boundaries(size_bytes: usize) -> String {
    let base = "word after word flows continuously without any paragraph breaks or \
                sentence endings that would trigger safe boundary detection in the \
                parallel encoding algorithm which means larger overlaps are needed ";
    let mut text = base.repeat((size_bytes / base.len()) + 1);
    text.truncate(size_bytes);
    text
}

/// Generate pathological input (repeated character)
fn generate_repeated_char(size_bytes: usize) -> String {
    "a".repeat(size_bytes)
}

/// Format size for display
fn format_size(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{}MB", bytes / 1_000_000)
    } else {
        format!("{}KB", bytes / 1_000)
    }
}

// =============================================================================
// TIMING HELPERS
// =============================================================================

/// Measure average time over multiple runs
fn measure_time<F>(iterations: u32, mut f: F) -> Duration
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed() / iterations
}

/// Measure and print speedup comparison
fn print_speedup_comparison(
    label: &str,
    serial_time: Duration,
    parallel_time: Duration,
    streaming_time: Option<Duration>,
) {
    let parallel_speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();

    if let Some(streaming) = streaming_time {
        let streaming_speedup = serial_time.as_secs_f64() / streaming.as_secs_f64();
        let best = if streaming_speedup > parallel_speedup {
            "Streaming"
        } else {
            "Recursive"
        };
        println!(
            "{:>6}: Serial {:>8.2?} | Recursive {:>8.2?} ({:.2}x) | Streaming {:>8.2?} ({:.2}x) | Best: {}",
            label, serial_time, parallel_time, parallel_speedup, streaming, streaming_speedup, best
        );
    } else {
        println!(
            "{:>6}: Serial {:>8.2?} | Parallel {:>8.2?} ({:.2}x)",
            label, serial_time, parallel_time, parallel_speedup
        );
    }
}

// =============================================================================
// BENCHMARK MACROS
// =============================================================================

/// Benchmark all three modes (serial, parallel, streaming) for a given text
macro_rules! bench_all_modes {
    ($group:expr, $tokenizer:expr, $text:expr, $label:expr) => {{
        $group.bench_with_input(BenchmarkId::new("serial", $label), $text, |b, text| {
            b.iter(|| std::hint::black_box($tokenizer.encode(text.as_str(), false)))
        });
        $group.bench_with_input(BenchmarkId::new("recursive", $label), $text, |b, text| {
            b.iter(|| std::hint::black_box($tokenizer.encode_parallel_single(text.as_str(), false)))
        });
        $group.bench_with_input(BenchmarkId::new("streaming", $label), $text, |b, text| {
            b.iter(|| std::hint::black_box($tokenizer.encode_streaming(text.as_str(), false)))
        });
    }};
}

/// Benchmark serial vs parallel only
macro_rules! bench_serial_parallel {
    ($group:expr, $tokenizer:expr, $text:expr, $label:expr) => {{
        $group.bench_with_input(BenchmarkId::new("serial", $label), $text, |b, text| {
            b.iter(|| std::hint::black_box($tokenizer.encode(text.as_str(), false)))
        });
        $group.bench_with_input(BenchmarkId::new("parallel", $label), $text, |b, text| {
            b.iter(|| std::hint::black_box($tokenizer.encode_parallel_single(text.as_str(), false)))
        });
    }};
}

// =============================================================================
// CORE BENCHMARKS: Serial vs Parallel vs Streaming
// =============================================================================

/// Primary benchmark: Compare all modes at standard sizes
///
/// This is the main benchmark that shows speedup across input sizes.
fn bench_core_comparison(c: &mut Criterion) {
    let tokenizer = get_bert_tokenizer();
    let mut group = c.benchmark_group("core_comparison");

    println!("\n=== Core Comparison (Serial vs Recursive vs Streaming) ===");

    for &size in SIZES_STANDARD {
        let text = generate_prose(size);
        let label = format_size(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Quick timing for printout
        let serial = measure_time(3, || {
            std::hint::black_box(tokenizer.encode(text.as_str(), false).unwrap());
        });
        let recursive = measure_time(3, || {
            std::hint::black_box(tokenizer.encode_parallel_single(text.as_str(), false).unwrap());
        });
        let streaming = measure_time(3, || {
            std::hint::black_box(tokenizer.encode_streaming(text.as_str(), false).unwrap());
        });
        print_speedup_comparison(&label, serial, recursive, Some(streaming));

        // Criterion benchmarks
        bench_all_modes!(group, tokenizer, &text, &label);
    }

    group.finish();
}

/// Benchmark byte-level BPE tokenizer (different token characteristics)
fn bench_core_bpe(c: &mut Criterion) {
    let tokenizer = get_bpe_tokenizer();
    let mut group = c.benchmark_group("core_bpe");

    for &size in SIZES_QUICK {
        let text = generate_prose(size);
        let label = format_size(size);
        group.throughput(Throughput::Bytes(size as u64));
        bench_serial_parallel!(group, tokenizer, &text, &label);
    }

    group.finish();
}

// =============================================================================
// OPTIMIZATION BENCHMARKS: Test specific optimizations
// =============================================================================

/// Benchmark safe boundary detection benefit
///
/// Text with clear paragraph/sentence boundaries should be FASTER
/// because the algorithm uses smaller overlaps (100-500 bytes vs 500-5000 bytes).
fn bench_optimization_safe_boundaries(c: &mut Criterion) {
    let tokenizer = get_bert_tokenizer();
    let mut group = c.benchmark_group("optimization_safe_boundaries");

    let size = 500_000; // 500KB - good size for recursive mode

    let with_boundaries = generate_with_safe_boundaries(size);
    let without_boundaries = generate_without_safe_boundaries(size);

    group.throughput(Throughput::Bytes(size as u64));

    println!("\n=== Safe Boundary Detection Benefit (500KB) ===");

    // Measure speedup difference
    let with_serial = measure_time(3, || {
        std::hint::black_box(tokenizer.encode(with_boundaries.as_str(), false).unwrap());
    });
    let with_parallel = measure_time(3, || {
        std::hint::black_box(tokenizer.encode_parallel_single(with_boundaries.as_str(), false).unwrap());
    });
    let without_serial = measure_time(3, || {
        std::hint::black_box(tokenizer.encode(without_boundaries.as_str(), false).unwrap());
    });
    let without_parallel = measure_time(3, || {
        std::hint::black_box(tokenizer.encode_parallel_single(without_boundaries.as_str(), false).unwrap());
    });

    let with_speedup = with_serial.as_secs_f64() / with_parallel.as_secs_f64();
    let without_speedup = without_serial.as_secs_f64() / without_parallel.as_secs_f64();

    println!("WITH safe boundaries:    {:.2}x speedup", with_speedup);
    println!("WITHOUT safe boundaries: {:.2}x speedup", without_speedup);
    println!(
        "Safe boundary benefit:   {:.1}% faster parallel",
        (with_speedup / without_speedup - 1.0) * 100.0
    );

    // Criterion benchmarks
    group.bench_function("with_boundaries/serial", |b| {
        b.iter(|| tokenizer.encode(with_boundaries.as_str(), false))
    });
    group.bench_function("with_boundaries/parallel", |b| {
        b.iter(|| tokenizer.encode_parallel_single(with_boundaries.as_str(), false))
    });
    group.bench_function("without_boundaries/serial", |b| {
        b.iter(|| tokenizer.encode(without_boundaries.as_str(), false))
    });
    group.bench_function("without_boundaries/parallel", |b| {
        b.iter(|| tokenizer.encode_parallel_single(without_boundaries.as_str(), false))
    });

    group.finish();
}

/// Benchmark LazyEncoding benefit at different recursion depths
///
/// LazyEncoding turns O(n × depth) offset operations into O(n).
/// Deeper recursion should show more benefit.
fn bench_optimization_lazy_encoding(c: &mut Criterion) {
    let tokenizer = get_bert_tokenizer();
    let mut group = c.benchmark_group("optimization_lazy_encoding");

    let size = 500_000; // 500KB
    let text = generate_prose(size);
    group.throughput(Throughput::Bytes(size as u64));

    println!("\n=== LazyEncoding Benefit by Recursion Depth (500KB) ===");

    // Test different forced depths
    for depth in [1, 2, 3, 4] {
        let config = ParallelConfig {
            max_depth: depth,
            mode: ParallelMode::Recursive,
            min_tokens_per_chunk: 5_000, // Allow more splits
            ..ParallelConfig::default()
        };

        let time = measure_time(3, || {
            std::hint::black_box(
                tokenizer
                    .encode_parallel_with_config(text.as_str(), false, config)
                    .unwrap(),
            );
        });

        println!("Depth {}: {:?}", depth, time);

        group.bench_function(BenchmarkId::new("depth", depth), |b| {
            b.iter(|| tokenizer.encode_parallel_with_config(text.as_str(), false, config))
        });
    }

    group.finish();
}

/// Benchmark streaming block size tuning
///
/// Different block sizes affect cache efficiency. 128KB (default) should fit L2.
fn bench_optimization_block_sizes(c: &mut Criterion) {
    let tokenizer = get_bert_tokenizer();
    let mut group = c.benchmark_group("optimization_block_sizes");

    let size = 4_000_000; // 4MB - good for streaming
    let text = generate_prose(size);
    group.throughput(Throughput::Bytes(size as u64));

    println!("\n=== Streaming Block Size Tuning (4MB) ===");

    for &block_kb in BLOCK_SIZES_KB {
        let config = ParallelConfig::streaming().with_block_size(block_kb * 1024);

        let time = measure_time(3, || {
            std::hint::black_box(
                tokenizer
                    .encode_parallel_with_config(text.as_str(), false, config)
                    .unwrap(),
            );
        });

        println!("Block {}KB: {:?}", block_kb, time);

        group.bench_function(BenchmarkId::new("block_kb", block_kb), |b| {
            b.iter(|| tokenizer.encode_parallel_with_config(text.as_str(), false, config))
        });
    }

    group.finish();
}

// =============================================================================
// EDGE CASE BENCHMARKS
// =============================================================================

/// Benchmark overhead for small inputs (should be minimal)
fn bench_edge_small_overhead(c: &mut Criterion) {
    let tokenizer = get_bert_tokenizer();
    let mut group = c.benchmark_group("edge_small_overhead");

    // Test sizes below and at threshold
    let sizes = [1_000, 5_000, 10_000];

    for &size in &sizes {
        let text = generate_prose(size);
        let label = format_size(size);

        group.bench_with_input(BenchmarkId::new("serial", &label), &text, |b, text| {
            b.iter(|| tokenizer.encode(text.as_str(), false))
        });
        group.bench_with_input(BenchmarkId::new("parallel", &label), &text, |b, text| {
            b.iter(|| tokenizer.encode_parallel_single(text.as_str(), false))
        });
    }

    group.finish();
}

/// Benchmark worst-case inputs (pathological patterns)
fn bench_edge_worst_case(c: &mut Criterion) {
    let tokenizer = get_bert_tokenizer();
    let mut group = c.benchmark_group("edge_worst_case");

    let size = 50_000;

    // Worst case: repeated character (creates very long tokens)
    let repeated = generate_repeated_char(size);
    group.bench_function("repeated_char/serial", |b| {
        b.iter(|| tokenizer.encode(repeated.as_str(), false))
    });
    group.bench_function("repeated_char/parallel", |b| {
        b.iter(|| tokenizer.encode_parallel_single(repeated.as_str(), false))
    });

    // Worst case: no whitespace
    let no_ws = "abcdefghij".repeat(size / 10);
    group.bench_function("no_whitespace/serial", |b| {
        b.iter(|| tokenizer.encode(no_ws.as_str(), false))
    });
    group.bench_function("no_whitespace/parallel", |b| {
        b.iter(|| tokenizer.encode_parallel_single(no_ws.as_str(), false))
    });

    group.finish();
}

/// Benchmark with special tokens (post-processing overhead)
fn bench_edge_special_tokens(c: &mut Criterion) {
    let tokenizer = get_bert_tokenizer();
    let mut group = c.benchmark_group("edge_special_tokens");

    let size = 100_000;
    let text = generate_prose(size);
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("without_special/parallel", |b| {
        b.iter(|| tokenizer.encode_parallel_single(text.as_str(), false))
    });
    group.bench_function("with_special/parallel", |b| {
        b.iter(|| tokenizer.encode_parallel_single(text.as_str(), true))
    });

    group.finish();
}

// =============================================================================
// SCALABILITY BENCHMARK (for documentation/reporting)
// =============================================================================

/// Comprehensive scalability report (prints speedup table)
fn bench_scalability_report(c: &mut Criterion) {
    let tokenizer = get_bert_tokenizer();
    let mut group = c.benchmark_group("scalability_report");

    // Reduced sizes for faster benchmarking (skip 8MB which takes too long)
    let sizes = [100_000, 500_000, 1_000_000, 4_000_000];

    println!("\n=== Scalability Report ===");
    println!("{:<8} {:>10} {:>12} {:>12} {:>12} {:>8}",
             "Size", "Serial", "Recursive", "Streaming", "Best", "Speedup");
    println!("{}", "-".repeat(72));

    for &size in &sizes {
        let text = generate_prose(size);
        let label = format_size(size);
        group.throughput(Throughput::Bytes(size as u64));

        let serial = measure_time(3, || {
            std::hint::black_box(tokenizer.encode(text.as_str(), false).unwrap());
        });
        let recursive = measure_time(3, || {
            std::hint::black_box(tokenizer.encode_parallel_single(text.as_str(), false).unwrap());
        });
        let streaming = measure_time(3, || {
            std::hint::black_box(tokenizer.encode_streaming(text.as_str(), false).unwrap());
        });

        let recursive_speedup = serial.as_secs_f64() / recursive.as_secs_f64();
        let streaming_speedup = serial.as_secs_f64() / streaming.as_secs_f64();
        let (best, best_speedup) = if streaming_speedup > recursive_speedup {
            ("Streaming", streaming_speedup)
        } else {
            ("Recursive", recursive_speedup)
        };

        println!(
            "{:<8} {:>10.2?} {:>10.2?} ({:.2}x) {:>10.2?} ({:.2}x) {:>8} {:.2}x",
            label, serial, recursive, recursive_speedup, streaming, streaming_speedup, best, best_speedup
        );

        // Just benchmark the best mode for this size
        group.bench_function(&label, |b| {
            b.iter(|| tokenizer.encode_parallel_single(text.as_str(), false))
        });
    }

    group.finish();
}

// =============================================================================
// CRITERION GROUPS
// =============================================================================

criterion_group!(
    name = core;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(3));
    targets = bench_core_comparison, bench_core_bpe
);

criterion_group!(
    name = optimization;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(3));
    targets = bench_optimization_safe_boundaries,
              bench_optimization_lazy_encoding,
              bench_optimization_block_sizes
);

criterion_group!(
    name = edge_cases;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(2));
    targets = bench_edge_small_overhead,
              bench_edge_worst_case,
              bench_edge_special_tokens
);

criterion_group!(
    name = scalability;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(2));
    targets = bench_scalability_report
);

criterion_main!(core, optimization, edge_cases, scalability);
