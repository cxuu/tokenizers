//! Serial `encode` versus `encode_parallel_single` on one long input.
//!
//! cargo bench --bench parallel_single_benchmark
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::time::Duration;
use tokenizers::Tokenizer;

const TOKENIZERS: &[(&str, &str)] = &[
    ("bert", "data/bert-wiki.json"),
    ("roberta", "data/roberta.json"),
    ("llama3", "data/llama-3-tokenizer.json"),
    ("albert", "data/albert-base-v1-tokenizer.json"),
];
const SIZES: &[usize] = &[100_000, 500_000, 1_000_000, 4_000_000];

fn prefix(text: &str, len: usize) -> &str {
    let mut end = len.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn bench_parallel_single(c: &mut Criterion) {
    let text = std::fs::read_to_string("data/big.txt").expect("run `make test` to download data");
    for (name, path) in TOKENIZERS {
        let tokenizer = Tokenizer::from_file(path).unwrap();
        let mut group = c.benchmark_group(format!("parallel-single-{name}"));
        group
            .sample_size(10)
            .measurement_time(Duration::from_secs(5));
        for &size in SIZES {
            let input = prefix(&text, size);
            group.throughput(Throughput::Bytes(input.len() as u64));
            group.bench_with_input(BenchmarkId::new("serial", size), input, |b, input| {
                b.iter(|| tokenizer.encode(black_box(input), false).unwrap())
            });
            group.bench_with_input(BenchmarkId::new("parallel", size), input, |b, input| {
                b.iter(|| {
                    tokenizer
                        .encode_parallel_single(black_box(input), false)
                        .unwrap()
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_parallel_single);
criterion_main!(benches);
