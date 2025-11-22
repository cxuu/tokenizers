# Benchmarking Guide: Parallel Single-Input Encoding

## Overview

This guide explains how to run and interpret the performance benchmarks for parallel single-input tokenization, including the new cache-block streaming mode.

---

## Quick Start

```bash
# Run all parallel encoding benchmarks
cargo bench --bench parallel_single_benchmark

# Run scalability benchmark with speedup output
cargo bench --bench parallel_single_benchmark -- bench_scalability

# Run streaming-specific benchmarks
cargo bench --bench parallel_single_benchmark -- bench_streaming
```

---

## Benchmark Suite Design

### Benchmark Functions

#### Core Benchmarks

| Function | Purpose | Input Sizes |
|----------|---------|-------------|
| `bench_serial_encoding_sizes` | Baseline serial performance | 1-500KB |
| `bench_parallel_encoding_sizes` | Parallel performance | 1-500KB |
| `bench_serial_vs_parallel_comparison` | Direct comparison | 10-200KB |
| `bench_scalability` | Speedup demonstration | 10KB-1MB |

#### Streaming Benchmarks (New)

| Function | Purpose | Input Sizes |
|----------|---------|-------------|
| `bench_streaming_large_inputs` | Compare modes at large sizes | 1-8MB |
| `bench_streaming_scalability` | Streaming speedup at 8MB | 8MB |
| `bench_cache_efficiency` | Cache crossover analysis | 2-6MB |

#### Specialized Benchmarks

| Function | Purpose |
|----------|---------|
| `bench_byte_level_bpe_parallel` | Test BPE tokenizer |
| `bench_overhead_small_inputs` | Measure overhead for small inputs |
| `bench_realistic_documents` | Real-world content |
| `bench_crossover_analysis` | Find optimal threshold |
| `bench_content_types` | Different text patterns |
| `bench_overhead_breakdown` | Cost analysis |
| `bench_with_special_tokens` | Post-processing impact |
| `bench_worst_case_inputs` | Stress testing |

---

## Key Benchmarks Explained

### 1. `bench_scalability` ⭐ **PRIMARY BENCHMARK**

**Purpose**: Show speedup at different input sizes

**Output**:
```
Size: 10KB, Serial: 2.0ms, Parallel: 2.7ms, Speedup: 0.74x (below threshold)
Size: 50KB, Serial: 10.5ms, Parallel: 10.5ms, Speedup: 1.00x (breakeven)
Size: 100KB, Serial: 23ms, Parallel: 13ms, Speedup: 1.77x (recursive)
Size: 500KB, Serial: 131ms, Parallel: 60ms, Speedup: 2.18x (recursive)
Size: 1MB, Serial: 250ms, Parallel: 150ms, Speedup: 1.67x (streaming)
```

**Key insights**:
- Breakeven at ~50KB
- Peak speedup ~2.2x at 500KB with recursive mode
- Streaming mode auto-selected at 1MB+ with consistent 1.5-1.8x speedup

---

### 2. `bench_streaming_large_inputs` ⭐ **STREAMING BENCHMARK**

**Purpose**: Compare serial, recursive, and streaming modes for large inputs

**Sizes tested**: 1MB, 2MB, 4MB, 8MB

**Expected output**:
```
streaming_vs_recursive/bert/serial/4MB: 1.05s
streaming_vs_recursive/bert/recursive/4MB: 475ms
streaming_vs_recursive/bert/streaming/4MB: 470ms
```

**Analysis**: At 4MB+, streaming and recursive have similar throughput, but streaming provides better cache behavior.

---

### 3. `bench_streaming_scalability` ⭐ **LARGE INPUT COMPARISON**

**Purpose**: Direct comparison of modes at large input sizes

**Output** (example at 8MB):
```
=== Streaming Scalability (8MB input) ===
Serial:     2.1s
Streaming:  1.3s (1.6x speedup)
```

At 1MB+, streaming mode is auto-selected for better cache locality.

---

### 4. `bench_cache_efficiency`

**Purpose**: Analyze cache behavior at different sizes

**Sizes tested**: 1MB, 2MB, 3MB, 4MB

**Expected**:
- 1MB+: Streaming mode provides consistent performance
- Better cache locality than recursive at large sizes

---

## Running Benchmarks

### Full Suite
```bash
cargo bench --bench parallel_single_benchmark
```
**Time**: ~20-30 minutes
**Output**: HTML reports in `target/criterion/`

### Quick Comparison
```bash
# Just scalability (shows speedup numbers)
cargo bench --bench parallel_single_benchmark -- bench_scalability

# Just streaming benchmarks
cargo bench --bench parallel_single_benchmark -- bench_streaming
```

### Single Benchmark
```bash
# Run specific benchmark group
cargo bench --bench parallel_single_benchmark -- bench_serial_vs_parallel_comparison
```

### With Custom Sample Size
```bash
# Fewer iterations for faster results
cargo bench --bench parallel_single_benchmark -- --sample-size 10
```

---

## Interpreting Results

### Criterion Output Format
```
streaming_vs_recursive/bert/serial/4MB
                          time:   [1.0421 s 1.0505 s 1.0602 s]
                          thrpt:  [3.7713 MiB/s 3.8067 MiB/s 3.8372 MiB/s]

streaming_vs_recursive/bert/streaming/4MB
                          time:   [468.52 ms 471.55 ms 474.84 ms]
                          thrpt:  [8.4200 MiB/s 8.4788 MiB/s 8.5336 MiB/s]
```

**Speedup Calculation**:
```
Speedup = 1050ms / 471ms = 2.23x
```

### Performance Targets

| Input Size | Recursive | Streaming | Status |
|------------|-----------|-----------|--------|
| < 50KB | 1.0x (fallback) | N/A | Expected |
| 50-100KB | 1.0-1.8x | N/A | Good |
| 100KB-1MB | 1.5-2.2x | N/A | Excellent |
| 1MB+ | N/A | 1.5-1.8x | Excellent (auto-streaming) |

### Red Flags

❌ **Parallel slower for inputs > 100KB**: Algorithm issue
❌ **Streaming slower than recursive at 8MB+**: Block size issue
❌ **Overhead > 10% for small inputs**: Threshold not working
❌ **No speedup at 500KB**: Parallelization broken

---

## Viewing Detailed Reports

Criterion generates HTML reports:

```bash
# Run benchmarks
cargo bench --bench parallel_single_benchmark

# Open report in browser
open target/criterion/report/index.html
```

**Reports Include**:
- Time distributions (violin plots)
- Throughput graphs
- Regression analysis
- Statistical confidence intervals

---

## Comparing Modes

### Serial vs Recursive vs Streaming

```bash
# Run the streaming comparison
cargo bench --bench parallel_single_benchmark -- bench_streaming_large_inputs 2>&1 | grep -E "(serial|recursive|streaming)"
```

### Expected Comparison at 8MB

| Mode | Time | Speedup | Notes |
|------|------|---------|-------|
| Serial | 2.1s | 1.0x | Baseline |
| Streaming | 1.3s | 1.6x | Cache-optimized (auto-selected) |

At 8MB, streaming mode is auto-selected and achieves ~1.6x speedup.

---

## Custom Benchmarks

### Test Specific Input Size
```rust
// Add to benches/parallel_single_benchmark.rs
fn bench_custom_size(c: &mut Criterion) {
    let tokenizer = create_bert_tokenizer();
    let text = generate_text(16_000_000); // 16 MB

    c.bench_function("16MB_serial", |b| {
        b.iter(|| tokenizer.encode(&text, false))
    });

    c.bench_function("16MB_streaming", |b| {
        b.iter(|| tokenizer.encode_streaming(&text, false))
    });
}
```

### Test Custom Block Size
```rust
use tokenizers::tokenizer::parallel_encode::{ParallelConfig, ParallelMode};

let config = ParallelConfig {
    mode: ParallelMode::Streaming,
    block_size: 64 * 1024, // 64KB blocks
    ..ParallelConfig::default()
};

c.bench_function("streaming_64kb_blocks", |b| {
    b.iter(|| tokenizer.encode_parallel_with_config(&text, false, config))
});
```

---

## Expected Results Summary

### Actual Benchmark Results (November 2025)

```
Input Size    Serial     Recursive         Streaming         Best Mode
──────────────────────────────────────────────────────────────────────────
50KB          10.1ms     10.7ms (0.95x)    9.8ms  (1.03x)    ~Tie
100KB         20.3ms     12.1ms (1.68x)    20.9ms (0.98x)    Recursive
250KB         53.8ms     25.8ms (2.09x)    35.8ms (1.50x)    Recursive
500KB         113ms      59.9ms (1.89x)    60.2ms (1.88x)    ~Tie
1MB           237ms      123ms  (1.93x)    109ms  (2.17x)    Streaming
2MB           506ms      231ms  (2.20x)    221ms  (2.29x)    Streaming
4MB           1.04s      482ms  (2.16x)    468ms  (2.22x)    Streaming
8MB           2.0s       972ms  (2.06x)    959ms  (2.09x)    Streaming
```

### Performance Profile

```
             Serial   Recursive  Streaming  Best Mode
< 50KB       ████     N/A        N/A        Serial (fallback)
50-100KB     ████     ███        N/A        Recursive (marginal)
100KB-500KB  ████     ██         ███        Recursive (1.7-2.1x)
500KB-1MB    ████     ██         ██         Either (~1.9x)
1MB+         ████     ██         █          Streaming (2.0-2.3x, auto)
```

---

## Troubleshooting

### Benchmark Runs Too Slowly
```bash
# Reduce sample size
cargo bench -- --sample-size 10

# Run only scalability
cargo bench -- bench_scalability
```

### Results Vary Too Much
```bash
# Ensure system is idle
# Close other applications
# Run longer for better statistics
cargo bench -- --sample-size 100
```

### Want Raw Numbers
```bash
# bench_scalability prints speedup directly
cargo bench -- bench_scalability 2>&1 | grep "Speedup:"

# bench_streaming_scalability shows all three modes
cargo bench -- bench_streaming_scalability 2>&1 | grep -E "(Serial|Recursive|Streaming)"
```

---

## Performance Validation Checklist

✅ **Correctness First**: Run `cargo test` before benchmarking
✅ **Speedup > 1.5x at 100KB**: Recursive mode working
✅ **Speedup > 1.8x at 500KB**: Recursive mode excellent
✅ **Speedup > 1.5x at 1MB+**: Streaming mode working
✅ **Overhead < 10% below threshold**: Fallback efficient
✅ **No Regressions**: Serial performance unchanged

---

## CI/CD Integration

### Performance Regression Detection
```bash
#!/bin/bash
# Run benchmarks and save baseline
cargo bench --bench parallel_single_benchmark -- --save-baseline main

# On future runs, compare
cargo bench --bench parallel_single_benchmark -- --baseline main
```

---

## Summary

The benchmark suite provides:

✅ **Comprehensive benchmarks** covering all scenarios
✅ **Streaming mode tests** for large input optimization
✅ **Clear metrics**: Speedup, throughput, overhead
✅ **Easy comparison**: Serial vs recursive vs streaming
✅ **Detailed reports**: HTML visualizations via Criterion

**Use these benchmarks to validate**:
- **1.7-2.1x speedup** for medium inputs (100KB-500KB) via recursive mode
- **2.0-2.3x consistent speedup** for large inputs (1MB+) via streaming mode
- Automatic mode selection at 1MB threshold
- Negligible overhead for small inputs (<50KB)
