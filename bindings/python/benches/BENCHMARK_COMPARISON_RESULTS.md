# Tokenizer Benchmark Results: 4MB Input

## Overview

This benchmark compares the tokenization speed of different tokenizer libraries
on a single 4MB input (simulating a large document or batch).

**Note**: tiktoken and HuggingFace use the **same LLaMA 3 tokenizer** for
apples-to-apples comparison. SentencePiece uses T5 (different tokenizer family).

### Test Configuration

- **Input Size**: 4.00 MB (4,194,320 bytes)
- **Iterations**: 10
- **Warmup**: 3 iterations

## Results

| Tokenizer | Time (ms) | Throughput (MB/s) | Tokens | Notes |
|-----------|-----------|-------------------|--------|-------|
| tiktoken/llama3 | 334.7 ± 2.2 | **12.53** | 672,020 | Rust impl, highly optimized |
| huggingface/streaming | 549.8 ± 23.7 | **7.63** | 672,020 | Rust impl, cache-optimized parallel |
| huggingface/parallel | 575.0 ± 27.1 | **7.29** | 672,020 | Rust impl, recursive parallel |
| huggingface/serial | 1468.9 ± 31.1 | **2.86** | 672,020 | Rust impl, serial |

## Analysis

**Fastest**: tiktoken/llama3 at 12.53 MB/s

### HuggingFace Parallel Encoding Speedup

| Mode | Time | Speedup vs Serial |
|------|------|-------------------|
| Serial | 1468.9 ms | 1.00x (baseline) |
| Parallel | 575.0 ms | **2.55x** |
| Streaming | 549.8 ms | **2.67x** |

### tiktoken vs HuggingFace Streaming

tiktoken is **1.6x faster** than HuggingFace streaming mode.

This gap is primarily due to:
1. tiktoken's highly optimized BPE implementation
2. Different regex/pre-tokenization strategies
3. Python binding overhead differences

## How to Reproduce

```bash
# Install dependencies
pip install tiktoken sentencepiece tokenizers huggingface_hub transformers

# Build local tokenizers with parallel encoding support
cd bindings/python
pip install maturin
maturin develop --release

# Run benchmark with LLaMA 3
cd benches
python bench_4mb_comparison.py --use-llama3 --iterations 10

# Or with specific thread count
RAYON_NUM_THREADS=8 python bench_4mb_comparison.py --use-llama3
```
