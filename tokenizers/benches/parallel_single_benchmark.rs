#[macro_use]
extern crate criterion;

mod common;

use criterion::{BenchmarkId, Criterion, Throughput};
use std::time::{Duration, Instant};
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{decoders, Model, Tokenizer, TokenizerImpl};

type BertTokenizer = TokenizerImpl<
    WordPiece,
    BertNormalizer,
    BertPreTokenizer,
    BertProcessing,
    decoders::wordpiece::WordPiece,
>;

/// Create a BERT tokenizer for benchmarking
fn create_bert_tokenizer() -> BertTokenizer {
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
}

/// Create a byte-level BPE tokenizer for benchmarking
fn create_byte_level_tokenizer() -> Tokenizer {
    let bpe = BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .build()
        .expect("BPE files not found, run `make test` first");

    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer.with_post_processor(Some(ByteLevel::default()));
    tokenizer
}

/// Generate test text of specified size with realistic content
fn generate_text(size_bytes: usize) -> String {
    let base_text = "The quick brown fox jumps over the lazy dog. \
                     This is a sample sentence for benchmarking tokenization performance. \
                     It contains various words, punctuation, and structures to simulate real text. ";

    let unit_size = base_text.len();
    let repeats = (size_bytes + unit_size - 1) / unit_size;
    let mut text = base_text.repeat(repeats);
    text.truncate(size_bytes);
    text
}

/// Benchmark serial encoding for different input sizes
fn bench_serial_encoding_sizes(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("parallel_vs_serial/bert/serial");

    // Test various input sizes from small to very large
    let sizes = vec![
        1_000,      // 1 KB - below threshold
        5_000,      // 5 KB - below threshold
        10_000,     // 10 KB - at threshold
        20_000,     // 20 KB - 2x threshold
        50_000,     // 50 KB - medium
        100_000,    // 100 KB - large
        250_000,    // 250 KB - very large
        500_000,    // 500 KB - stress test
    ];

    for size in sizes.iter() {
        let text = generate_text(*size);
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}KB", size / 1000)),
            &text,
            |b, text| {
                b.iter_custom(|iters| {
                    let mut duration = Duration::new(0, 0);
                    for _ in 0..iters {
                        let start = Instant::now();
                        let _ = std::hint::black_box(tokenizer.encode(text.as_str(), false));
                        duration += start.elapsed();
                    }
                    duration
                })
            },
        );
    }

    group.finish();
}

/// Benchmark parallel encoding for different input sizes
fn bench_parallel_encoding_sizes(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("parallel_vs_serial/bert/parallel");

    let sizes = vec![
        1_000,      // 1 KB - should fallback to serial
        5_000,      // 5 KB - should fallback to serial
        10_000,     // 10 KB - parallel threshold
        20_000,     // 20 KB - 2x threshold
        50_000,     // 50 KB - medium
        100_000,    // 100 KB - large
        250_000,    // 250 KB - very large
        500_000,    // 500 KB - stress test
    ];

    for size in sizes.iter() {
        let text = generate_text(*size);
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}KB", size / 1000)),
            &text,
            |b, text| {
                b.iter_custom(|iters| {
                    let mut duration = Duration::new(0, 0);
                    for _ in 0..iters {
                        let start = Instant::now();
                        let _ = std::hint::black_box(tokenizer.encode_parallel_single(text.as_str(), false));
                        duration += start.elapsed();
                    }
                    duration
                })
            },
        );
    }

    group.finish();
}

/// Direct comparison: serial vs parallel at various sizes
fn bench_serial_vs_parallel_comparison(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("parallel_comparison/bert");

    // Focus on sizes where parallel should help
    let sizes = vec![
        10_000,     // Threshold
        25_000,     // 2.5x threshold
        50_000,     // 5x threshold
        100_000,    // 10x threshold
        200_000,    // 20x threshold
    ];

    for size in sizes.iter() {
        let text = generate_text(*size);
        group.throughput(Throughput::Bytes(*size as u64));

        // Serial encoding
        group.bench_with_input(
            BenchmarkId::new("serial", format!("{}KB", size / 1000)),
            &text,
            |b, text| {
                b.iter(|| {
                    std::hint::black_box(tokenizer.encode(text.as_str(), false))
                })
            },
        );

        // Parallel encoding
        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}KB", size / 1000)),
            &text,
            |b, text| {
                b.iter(|| {
                    std::hint::black_box(tokenizer.encode_parallel_single(text.as_str(), false))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark byte-level BPE (has longer tokens, different characteristics)
fn bench_byte_level_bpe_parallel(c: &mut Criterion) {
    let tokenizer = create_byte_level_tokenizer();
    let mut group = c.benchmark_group("parallel_comparison/bpe");

    let sizes = vec![
        10_000,
        50_000,
        100_000,
        250_000,
    ];

    for size in sizes.iter() {
        let text = generate_text(*size);
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("serial", format!("{}KB", size / 1000)),
            &text,
            |b, text| {
                b.iter(|| {
                    std::hint::black_box(tokenizer.encode(text.as_str(), false))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}KB", size / 1000)),
            &text,
            |b, text| {
                b.iter(|| {
                    std::hint::black_box(tokenizer.encode_parallel_single(text.as_str(), false))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark overhead for small inputs (should be minimal)
fn bench_overhead_small_inputs(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("parallel_overhead");

    // Test sizes below threshold where parallel should fallback quickly
    let sizes = vec![100, 500, 1000, 2000, 5000, 9000, 10000];

    for size in sizes.iter() {
        let text = generate_text(*size);

        group.bench_with_input(
            BenchmarkId::new("serial", *size),
            &text,
            |b, text| b.iter(|| tokenizer.encode(text.as_str(), false)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", *size),
            &text,
            |b, text| b.iter(|| tokenizer.encode_parallel_single(text.as_str(), false)),
        );
    }

    group.finish();
}

/// Benchmark with realistic document content
fn bench_realistic_documents(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("parallel_realistic");

    // Try to load real data if available, otherwise generate
    let small_doc = std::fs::read_to_string("data/small.txt")
        .unwrap_or_else(|_| generate_text(5000));
    let big_doc = std::fs::read_to_string("data/big.txt")
        .unwrap_or_else(|_| generate_text(100000));

    group.throughput(Throughput::Bytes(small_doc.len() as u64));
    group.bench_function("small_document_serial", |b| {
        b.iter(|| tokenizer.encode(small_doc.as_str(), false))
    });

    group.bench_function("small_document_parallel", |b| {
        b.iter(|| tokenizer.encode_parallel_single(small_doc.as_str(), false))
    });

    group.throughput(Throughput::Bytes(big_doc.len() as u64));
    group.bench_function("big_document_serial", |b| {
        b.iter(|| tokenizer.encode(big_doc.as_str(), false))
    });

    group.bench_function("big_document_parallel", |b| {
        b.iter(|| tokenizer.encode_parallel_single(big_doc.as_str(), false))
    });

    group.finish();
}

/// Benchmark the crossover point (where parallel becomes beneficial)
fn bench_crossover_analysis(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("parallel_crossover");
    group.sample_size(50); // Reduce sample size for faster benchmarking

    // Fine-grained sizes around the threshold
    let sizes: Vec<usize> = (8..=12).map(|kb| kb * 1000).collect();

    for size in sizes.iter() {
        let text = generate_text(*size);

        group.bench_with_input(
            BenchmarkId::new("serial", *size),
            &text,
            |b, text| b.iter(|| tokenizer.encode(text.as_str(), false)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", *size),
            &text,
            |b, text| b.iter(|| tokenizer.encode_parallel_single(text.as_str(), false)),
        );
    }

    group.finish();
}

/// Benchmark scalability: how does speedup change with input size?
fn bench_scalability(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("parallel_scalability");
    group.sample_size(20); // Fewer samples for large inputs

    // Exponentially increasing sizes to show scalability
    let sizes = vec![
        10_000,    // 10 KB
        50_000,    // 50 KB
        100_000,   // 100 KB
        500_000,   // 500 KB
        1_000_000, // 1 MB
    ];

    for size in sizes.iter() {
        let text = generate_text(*size);
        group.throughput(Throughput::Bytes(*size as u64));

        // Measure serial
        let serial_time = {
            let start = Instant::now();
            for _ in 0..5 {
                std::hint::black_box(tokenizer.encode(text.as_str(), false).unwrap());
            }
            start.elapsed() / 5
        };

        // Measure parallel
        let parallel_time = {
            let start = Instant::now();
            for _ in 0..5 {
                std::hint::black_box(tokenizer.encode_parallel_single(text.as_str(), false).unwrap());
            }
            start.elapsed() / 5
        };

        let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();

        group.bench_function(
            BenchmarkId::new("speedup", format!("{}KB", size / 1000)),
            |b| {
                b.iter_custom(|iters| {
                    let mut duration = Duration::new(0, 0);
                    for _ in 0..iters {
                        let start = Instant::now();
                        std::hint::black_box(tokenizer.encode_parallel_single(text.as_str(), false).unwrap());
                        duration += start.elapsed();
                    }
                    duration
                })
            },
        );

        println!("Size: {}KB, Serial: {:?}, Parallel: {:?}, Speedup: {:.2}x",
                 size / 1000, serial_time, parallel_time, speedup);
    }

    group.finish();
}

/// Benchmark different content types
fn bench_content_types(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("parallel_content_types");
    let size = 50_000;

    // Different content patterns
    let content_types = vec![
        ("normal_prose", "The quick brown fox jumps over the lazy dog. "),
        ("technical", "async fn compute_hash(data: &[u8]) -> Result<Hash, Error> { "),
        ("numbers", "1234567890 0987654321 1111111111 2222222222 "),
        ("mixed_unicode", "Hello 世界! Testing émojis 🎉 and Unicode 你好. "),
        ("whitespace_heavy", "word   word    word     word      word       "),
    ];

    for (name, pattern) in content_types.iter() {
        let text = pattern.repeat(size / pattern.len());
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("serial", name),
            &text,
            |b, text| b.iter(|| tokenizer.encode(text.as_str(), false)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", name),
            &text,
            |b, text| b.iter(|| tokenizer.encode_parallel_single(text.as_str(), false)),
        );
    }

    group.finish();
}

/// Benchmark overhead breakdown (filtering, merging, word ID assignment)
fn bench_overhead_breakdown(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("parallel_overhead_breakdown");

    let size = 100_000;
    let text = generate_text(size);
    group.throughput(Throughput::Bytes(size as u64));

    // Benchmark full serial encoding
    group.bench_function("full_serial_encode", |b| {
        b.iter(|| tokenizer.encode(text.as_str(), false))
    });

    // Benchmark full parallel encoding
    group.bench_function("full_parallel_encode", |b| {
        b.iter(|| tokenizer.encode_parallel_single(text.as_str(), false))
    });

    group.finish();
}

/// Benchmark with special tokens (post-processing overhead)
fn bench_with_special_tokens(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("parallel_with_special_tokens");

    let sizes = vec![10_000, 50_000, 100_000];

    for size in sizes.iter() {
        let text = generate_text(*size);
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("serial", format!("{}KB", size / 1000)),
            &text,
            |b, text| b.iter(|| tokenizer.encode(text.as_str(), true)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}KB", size / 1000)),
            &text,
            |b, text| b.iter(|| tokenizer.encode_parallel_single(text.as_str(), true)),
        );
    }

    group.finish();
}

/// Benchmark worst-case scenarios
fn bench_worst_case_inputs(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("parallel_worst_case");

    let size = 50_000;

    // Worst case 1: Repeated character (creates very long tokens)
    let repeated_char = "a".repeat(size);
    group.bench_function("repeated_char_serial", |b| {
        b.iter(|| tokenizer.encode(repeated_char.as_str(), false))
    });
    group.bench_function("repeated_char_parallel", |b| {
        b.iter(|| tokenizer.encode_parallel_single(repeated_char.as_str(), false))
    });

    // Worst case 2: No whitespace (long continuous text)
    let no_whitespace = "abcdefghij".repeat(size / 10);
    group.bench_function("no_whitespace_serial", |b| {
        b.iter(|| tokenizer.encode(no_whitespace.as_str(), false))
    });
    group.bench_function("no_whitespace_parallel", |b| {
        b.iter(|| tokenizer.encode_parallel_single(no_whitespace.as_str(), false))
    });

    // Best case: Well-structured text with clear word boundaries
    let structured = "word ".repeat(size / 5);
    group.bench_function("structured_serial", |b| {
        b.iter(|| tokenizer.encode(structured.as_str(), false))
    });
    group.bench_function("structured_parallel", |b| {
        b.iter(|| tokenizer.encode_parallel_single(structured.as_str(), false))
    });

    group.finish();
}

/// Benchmark streaming mode for very large inputs
fn bench_streaming_large_inputs(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("streaming_vs_recursive");
    group.sample_size(10); // Fewer samples for very large inputs

    // Sizes where streaming should help (4MB+)
    let sizes = vec![
        1_000_000,   // 1 MB - baseline
        2_000_000,   // 2 MB
        4_000_000,   // 4 MB - streaming threshold
        8_000_000,   // 8 MB
    ];

    for size in sizes.iter() {
        let text = generate_text(*size);
        group.throughput(Throughput::Bytes(*size as u64));

        // Serial encoding (baseline)
        group.bench_with_input(
            BenchmarkId::new("serial", format!("{}MB", size / 1_000_000)),
            &text,
            |b, text| {
                b.iter(|| {
                    std::hint::black_box(tokenizer.encode(text.as_str(), false))
                })
            },
        );

        // Recursive parallel encoding
        group.bench_with_input(
            BenchmarkId::new("recursive", format!("{}MB", size / 1_000_000)),
            &text,
            |b, text| {
                b.iter(|| {
                    std::hint::black_box(tokenizer.encode_parallel_single(text.as_str(), false))
                })
            },
        );

        // Streaming encoding
        group.bench_with_input(
            BenchmarkId::new("streaming", format!("{}MB", size / 1_000_000)),
            &text,
            |b, text| {
                b.iter(|| {
                    std::hint::black_box(tokenizer.encode_streaming(text.as_str(), false))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark streaming scalability with very large inputs
fn bench_streaming_scalability(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("streaming_scalability");
    group.sample_size(10);

    // Test streaming with different block sizes
    let size = 8_000_000; // 8 MB
    let text = generate_text(size);
    group.throughput(Throughput::Bytes(size as u64));

    // Measure different modes
    let serial_time = {
        let start = Instant::now();
        for _ in 0..3 {
            std::hint::black_box(tokenizer.encode(text.as_str(), false).unwrap());
        }
        start.elapsed() / 3
    };

    let recursive_time = {
        let start = Instant::now();
        for _ in 0..3 {
            std::hint::black_box(tokenizer.encode_parallel_single(text.as_str(), false).unwrap());
        }
        start.elapsed() / 3
    };

    let streaming_time = {
        let start = Instant::now();
        for _ in 0..3 {
            std::hint::black_box(tokenizer.encode_streaming(text.as_str(), false).unwrap());
        }
        start.elapsed() / 3
    };

    let recursive_speedup = serial_time.as_secs_f64() / recursive_time.as_secs_f64();
    let streaming_speedup = serial_time.as_secs_f64() / streaming_time.as_secs_f64();

    println!("\n=== Streaming Scalability (8MB input) ===");
    println!("Serial:     {:?}", serial_time);
    println!("Recursive:  {:?} ({:.2}x speedup)", recursive_time, recursive_speedup);
    println!("Streaming:  {:?} ({:.2}x speedup)", streaming_time, streaming_speedup);
    println!("Streaming vs Recursive: {:.1}% faster",
             (recursive_time.as_secs_f64() / streaming_time.as_secs_f64() - 1.0) * 100.0);

    group.bench_function("serial_8MB", |b| {
        b.iter(|| tokenizer.encode(text.as_str(), false))
    });

    group.bench_function("recursive_8MB", |b| {
        b.iter(|| tokenizer.encode_parallel_single(text.as_str(), false))
    });

    group.bench_function("streaming_8MB", |b| {
        b.iter(|| tokenizer.encode_streaming(text.as_str(), false))
    });

    group.finish();
}

/// Benchmark cache efficiency with streaming
fn bench_cache_efficiency(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("cache_efficiency");
    group.sample_size(10);

    // Test the cache crossover point
    let sizes = vec![
        2_000_000,   // 2 MB - below streaming threshold
        3_000_000,   // 3 MB - near threshold
        4_000_000,   // 4 MB - at threshold
        5_000_000,   // 5 MB - above threshold
        6_000_000,   // 6 MB
    ];

    for size in sizes.iter() {
        let text = generate_text(*size);
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("recursive", format!("{}MB", size / 1_000_000)),
            &text,
            |b, text| b.iter(|| tokenizer.encode_parallel_single(text.as_str(), false)),
        );

        group.bench_with_input(
            BenchmarkId::new("streaming", format!("{}MB", size / 1_000_000)),
            &text,
            |b, text| b.iter(|| tokenizer.encode_streaming(text.as_str(), false)),
        );
    }

    group.finish();
}

/// Benchmark streaming mode across ALL input sizes (not just 1MB+)
/// This helps determine if streaming could be beneficial at smaller sizes too
fn bench_streaming_all_sizes(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let mut group = c.benchmark_group("streaming_all_sizes");
    group.sample_size(20);

    // Test streaming vs recursive vs serial at all sizes
    let sizes = vec![
        50_000,      // 50 KB
        100_000,     // 100 KB
        250_000,     // 250 KB
        500_000,     // 500 KB
        1_000_000,   // 1 MB (current streaming threshold)
        2_000_000,   // 2 MB
        4_000_000,   // 4 MB
    ];

    println!("\n=== Streaming vs Recursive vs Serial (All Sizes) ===");

    for size in sizes.iter() {
        let text = generate_text(*size);
        group.throughput(Throughput::Bytes(*size as u64));

        let size_label = if *size >= 1_000_000 {
            format!("{}MB", size / 1_000_000)
        } else {
            format!("{}KB", size / 1_000)
        };

        // Measure all three modes for comparison
        let serial_time = {
            let start = Instant::now();
            for _ in 0..3 {
                std::hint::black_box(tokenizer.encode(text.as_str(), false).unwrap());
            }
            start.elapsed() / 3
        };

        let recursive_time = {
            let start = Instant::now();
            for _ in 0..3 {
                std::hint::black_box(tokenizer.encode_parallel_single(text.as_str(), false).unwrap());
            }
            start.elapsed() / 3
        };

        let streaming_time = {
            let start = Instant::now();
            for _ in 0..3 {
                std::hint::black_box(tokenizer.encode_streaming(text.as_str(), false).unwrap());
            }
            start.elapsed() / 3
        };

        let recursive_speedup = serial_time.as_secs_f64() / recursive_time.as_secs_f64();
        let streaming_speedup = serial_time.as_secs_f64() / streaming_time.as_secs_f64();

        println!(
            "{:>6}: Serial {:>8.2?} | Recursive {:>8.2?} ({:.2}x) | Streaming {:>8.2?} ({:.2}x) | Best: {}",
            size_label,
            serial_time,
            recursive_time,
            recursive_speedup,
            streaming_time,
            streaming_speedup,
            if streaming_speedup > recursive_speedup { "Streaming" } else { "Recursive" }
        );

        // Benchmark serial
        group.bench_with_input(
            BenchmarkId::new("serial", &size_label),
            &text,
            |b, text| b.iter(|| tokenizer.encode(text.as_str(), false)),
        );

        // Benchmark recursive (encode_parallel_single with auto mode, but at small sizes uses recursive)
        group.bench_with_input(
            BenchmarkId::new("recursive", &size_label),
            &text,
            |b, text| b.iter(|| tokenizer.encode_parallel_single(text.as_str(), false)),
        );

        // Benchmark streaming (forced)
        group.bench_with_input(
            BenchmarkId::new("streaming", &size_label),
            &text,
            |b, text| b.iter(|| tokenizer.encode_streaming(text.as_str(), false)),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_serial_encoding_sizes,
    bench_parallel_encoding_sizes,
    bench_serial_vs_parallel_comparison,
    bench_byte_level_bpe_parallel,
    bench_overhead_small_inputs,
    bench_realistic_documents,
    bench_crossover_analysis,
    bench_scalability,
    bench_content_types,
    bench_overhead_breakdown,
    bench_with_special_tokens,
    bench_worst_case_inputs,
    bench_streaming_large_inputs,
    bench_streaming_scalability,
    bench_cache_efficiency,
    bench_streaming_all_sizes,
);

criterion_main!(benches);
