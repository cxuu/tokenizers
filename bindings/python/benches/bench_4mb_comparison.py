#!/usr/bin/env python3
"""
Benchmark comparing tokenizers on 4MB input.

Compares (using the same LLaMA 3 tokenizer where possible):
- tiktoken (using LLaMA 3 BPE model)
- HuggingFace tokenizers (serial, parallel, streaming)

Also includes:
- SentencePiece (different tokenizer family, for reference)

Usage:
    pip install tiktoken sentencepiece tokenizers huggingface_hub
    python bench_4mb_comparison.py

For reproducible results:
    RAYON_NUM_THREADS=8 python bench_4mb_comparison.py
"""

import os
import sys
import time
import json
import statistics
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict

# Target input size
TARGET_SIZE_MB = 4
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024

# Benchmark configuration
WARMUP_ITERATIONS = 3
BENCHMARK_ITERATIONS = 10

# LLaMA 3 model ID
LLAMA3_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    tokenizer_name: str
    model_name: str
    input_size_bytes: int
    input_size_mb: float
    token_count: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_mb_s: float
    throughput_tokens_s: float
    iterations: int


def generate_test_input(size_bytes: int) -> str:
    """
    Generate test input of approximately the given size.
    Uses a mix of English prose and common patterns.
    """
    # Base text with varied content
    base_texts = [
        "The quick brown fox jumps over the lazy dog. ",
        "In a world of artificial intelligence, tokenization remains fundamental. ",
        "Machine learning models process text through sophisticated pipelines. ",
        "Natural language processing has revolutionized how we interact with computers. ",
        "Deep learning architectures transform raw text into meaningful representations. ",
        "Transformers have become the backbone of modern NLP systems. ",
        "Attention mechanisms allow models to focus on relevant parts of input. ",
        "The tokenization step is crucial for efficient model inference. ",
    ]

    # Build text by cycling through base texts
    result = []
    current_size = 0
    idx = 0

    while current_size < size_bytes:
        text = base_texts[idx % len(base_texts)]
        result.append(text)
        current_size += len(text.encode('utf-8'))
        idx += 1

    return ''.join(result)


def benchmark_function(
    name: str,
    encode_fn: Callable[[str], List[int]],
    text: str,
    warmup: int = WARMUP_ITERATIONS,
    iterations: int = BENCHMARK_ITERATIONS
) -> Dict[str, Any]:
    """
    Benchmark an encoding function.

    Returns timing statistics and token count.
    """
    # Warmup runs
    for _ in range(warmup):
        encode_fn(text)

    # Timed runs
    times = []
    token_count = 0

    for _ in range(iterations):
        start = time.perf_counter()
        tokens = encode_fn(text)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        token_count = len(tokens)

    input_size_bytes = len(text.encode('utf-8'))
    mean_time = statistics.mean(times)

    return {
        'tokenizer_name': name,
        'times_ms': [t * 1000 for t in times],
        'mean_time_ms': mean_time * 1000,
        'std_time_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
        'token_count': token_count,
        'input_size_bytes': input_size_bytes,
        'throughput_mb_s': input_size_bytes / mean_time / 1e6,
        'throughput_tokens_s': token_count / mean_time,
    }


def load_tiktoken_llama3():
    """Load tiktoken with LLaMA 3 BPE model."""
    try:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        from huggingface_hub import hf_hub_download

        # Download LLaMA 3 tokenizer model
        filename = hf_hub_download(LLAMA3_MODEL_ID, "original/tokenizer.model")
        mergeable_ranks = load_tiktoken_bpe(filename)

        # LLaMA 3 regex pattern
        pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

        # Special tokens
        num_reserved_special_tokens = 256
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, num_reserved_special_tokens - 5)
        ]
        num_base_tokens = len(mergeable_ranks)
        special_tokens_dict = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        enc = tiktoken.Encoding(
            name="llama3",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens_dict,
        )

        return enc, lambda text: enc.encode(text)
    except ImportError as e:
        print(f"Warning: tiktoken not installed. Install with: pip install tiktoken")
        return None, None
    except Exception as e:
        print(f"Warning: Failed to load tiktoken with LLaMA 3: {e}")
        print("  This may require HF_TOKEN for gated model access")
        return None, None


def load_tiktoken_default(model: str = "cl100k_base"):
    """Load tiktoken with default encoding (fallback)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(model)
        return enc, lambda text: enc.encode(text)
    except ImportError:
        print("Warning: tiktoken not installed. Install with: pip install tiktoken")
        return None, None
    except Exception as e:
        print(f"Warning: Failed to load tiktoken ({model}): {e}")
        return None, None


def load_sentencepiece(model_path: Optional[str] = None):
    """Load SentencePiece model."""
    try:
        import sentencepiece as spm

        if model_path is None:
            # Try to find or download a model
            try:
                from transformers import T5Tokenizer
                t5 = T5Tokenizer.from_pretrained('google-t5/t5-base')
                # Get the model path from the tokenizer
                model_path = t5.vocab_file
            except Exception as e:
                print(f"Warning: Could not load SentencePiece model: {e}")
                return None, None

        sp = spm.SentencePieceProcessor(model_file=model_path)
        return sp, lambda text: sp.encode(text)
    except ImportError:
        print("Warning: sentencepiece not installed. Install with: pip install sentencepiece")
        return None, None
    except Exception as e:
        print(f"Warning: Failed to load SentencePiece: {e}")
        return None, None


def load_huggingface(model: str = LLAMA3_MODEL_ID):
    """Load HuggingFace tokenizer."""
    try:
        from tokenizers import Tokenizer

        # Try loading from pretrained
        try:
            tokenizer = Tokenizer.from_pretrained(model)
        except Exception:
            # Try local path
            tokenizer = Tokenizer.from_file(model)

        return tokenizer
    except ImportError:
        print("Warning: tokenizers not installed. Install with: pip install tokenizers")
        return None
    except Exception as e:
        print(f"Warning: Failed to load HuggingFace tokenizer ({model}): {e}")
        return None


def format_result(result: Dict[str, Any]) -> str:
    """Format a benchmark result for display."""
    return (
        f"  {result['tokenizer_name']:<40} "
        f"{result['mean_time_ms']:>8.1f}ms ± {result['std_time_ms']:>5.1f}ms  "
        f"{result['throughput_mb_s']:>6.2f} MB/s  "
        f"{result['token_count']:>10,} tokens"
    )


def run_benchmarks(
    input_size_mb: float = TARGET_SIZE_MB,
    use_llama3: bool = True,
    tiktoken_model: str = "cl100k_base",
    hf_model: str = LLAMA3_MODEL_ID,
    sentencepiece_model: Optional[str] = None,
    iterations: int = BENCHMARK_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
    output_json: Optional[str] = None,
) -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""

    input_size_bytes = int(input_size_mb * 1024 * 1024)

    print(f"\n{'='*80}")
    print(f"Tokenizer Benchmark: {input_size_mb}MB Input")
    print(f"{'='*80}")
    print(f"Warmup iterations: {warmup}")
    print(f"Benchmark iterations: {iterations}")
    print(f"RAYON_NUM_THREADS: {os.environ.get('RAYON_NUM_THREADS', 'not set')}")
    print(f"Using LLaMA 3: {use_llama3}")
    print()

    # Generate test input
    print(f"Generating {input_size_mb}MB test input...")
    test_text = generate_test_input(input_size_bytes)
    actual_size = len(test_text.encode('utf-8'))
    print(f"  Actual size: {actual_size / 1024 / 1024:.2f} MB ({actual_size:,} bytes)")
    print()

    results = []

    # Benchmark tiktoken
    print("Loading tiktoken...")
    if use_llama3:
        enc, encode_fn = load_tiktoken_llama3()
        tiktoken_name = "tiktoken/llama3"
    else:
        enc, encode_fn = load_tiktoken_default(tiktoken_model)
        tiktoken_name = f"tiktoken/{tiktoken_model}"

    if encode_fn:
        print(f"  Running {tiktoken_name}...")
        result = benchmark_function(
            tiktoken_name,
            encode_fn,
            test_text,
            warmup=warmup,
            iterations=iterations
        )
        result['model_name'] = "llama3" if use_llama3 else tiktoken_model
        results.append(result)
        print(format_result(result))
    print()

    # Benchmark HuggingFace tokenizers
    print(f"Loading HuggingFace tokenizers ({hf_model})...")
    tokenizer = load_huggingface(hf_model)

    if tokenizer:
        # Serial encoding
        print(f"  Running HuggingFace (serial)...")
        result = benchmark_function(
            "huggingface/serial",
            lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
            test_text,
            warmup=warmup,
            iterations=iterations
        )
        result['model_name'] = hf_model
        results.append(result)
        print(format_result(result))

        # Check if parallel encoding methods exist
        if hasattr(tokenizer, 'encode_parallel'):
            # Parallel encoding
            print(f"  Running HuggingFace (parallel)...")
            result = benchmark_function(
                "huggingface/parallel",
                lambda text: tokenizer.encode_parallel(text, add_special_tokens=False).ids,
                test_text,
                warmup=warmup,
                iterations=iterations
            )
            result['model_name'] = hf_model
            results.append(result)
            print(format_result(result))

        if hasattr(tokenizer, 'encode_streaming'):
            # Streaming encoding
            print(f"  Running HuggingFace (streaming)...")
            result = benchmark_function(
                "huggingface/streaming",
                lambda text: tokenizer.encode_streaming(text, add_special_tokens=False).ids,
                test_text,
                warmup=warmup,
                iterations=iterations
            )
            result['model_name'] = hf_model
            results.append(result)
            print(format_result(result))

        if hasattr(tokenizer, 'encode_parallel_zero_overlap'):
            # Zero-overlap pre-token parallel encoding
            print(f"  Running HuggingFace (zero-overlap)...")
            result = benchmark_function(
                "huggingface/zero-overlap",
                lambda text: tokenizer.encode_parallel_zero_overlap(text, add_special_tokens=False).ids,
                test_text,
                warmup=warmup,
                iterations=iterations
            )
            result['model_name'] = hf_model
            results.append(result)
            print(format_result(result))
    print()

    # Benchmark SentencePiece (different tokenizer family)
    if sentencepiece_model or not use_llama3:
        print("Loading SentencePiece (different tokenizer family)...")
        sp, encode_fn = load_sentencepiece(sentencepiece_model)
        if encode_fn:
            print(f"  Running SentencePiece...")
            result = benchmark_function(
                "sentencepiece/t5",
                encode_fn,
                test_text,
                warmup=warmup,
                iterations=iterations
            )
            result['model_name'] = "t5"
            results.append(result)
            print(format_result(result))
        print()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Tokenizer':<40} {'Time (ms)':<18} {'Throughput':<12} {'Tokens':<12}")
    print("-" * 80)

    # Sort by throughput
    sorted_results = sorted(results, key=lambda x: x['throughput_mb_s'], reverse=True)

    for r in sorted_results:
        print(
            f"{r['tokenizer_name']:<40} "
            f"{r['mean_time_ms']:>7.1f} ± {r['std_time_ms']:>5.1f}   "
            f"{r['throughput_mb_s']:>6.2f} MB/s  "
            f"{r['token_count']:>10,}"
        )

    if len(sorted_results) >= 2:
        fastest = sorted_results[0]
        print(f"\nFastest: {fastest['tokenizer_name']} at {fastest['throughput_mb_s']:.2f} MB/s")

        # Show speedup of parallel vs serial for HuggingFace
        hf_serial = next((r for r in results if r['tokenizer_name'] == 'huggingface/serial'), None)
        hf_parallel = next((r for r in results if r['tokenizer_name'] == 'huggingface/parallel'), None)
        hf_streaming = next((r for r in results if r['tokenizer_name'] == 'huggingface/streaming'), None)

        if hf_serial and hf_parallel:
            speedup = hf_serial['mean_time_ms'] / hf_parallel['mean_time_ms']
            print(f"HuggingFace parallel speedup: {speedup:.2f}x over serial")

        if hf_serial and hf_streaming:
            speedup = hf_serial['mean_time_ms'] / hf_streaming['mean_time_ms']
            print(f"HuggingFace streaming speedup: {speedup:.2f}x over serial")

    # Convert to BenchmarkResult objects
    benchmark_results = []
    for r in results:
        benchmark_results.append(BenchmarkResult(
            tokenizer_name=r['tokenizer_name'],
            model_name=r.get('model_name', ''),
            input_size_bytes=r['input_size_bytes'],
            input_size_mb=r['input_size_bytes'] / 1024 / 1024,
            token_count=r['token_count'],
            mean_time_ms=r['mean_time_ms'],
            std_time_ms=r['std_time_ms'],
            min_time_ms=r['min_time_ms'],
            max_time_ms=r['max_time_ms'],
            throughput_mb_s=r['throughput_mb_s'],
            throughput_tokens_s=r['throughput_tokens_s'],
            iterations=iterations,
        ))

    # Save JSON if requested
    if output_json:
        output_data = {
            'config': {
                'input_size_mb': input_size_mb,
                'warmup_iterations': warmup,
                'benchmark_iterations': iterations,
                'rayon_threads': os.environ.get('RAYON_NUM_THREADS'),
                'use_llama3': use_llama3,
            },
            'results': [asdict(r) for r in benchmark_results],
        }
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    return benchmark_results


def generate_markdown_report(results: List[BenchmarkResult], output_path: str, use_llama3: bool = True):
    """Generate a markdown report from benchmark results."""

    # Sort by throughput
    sorted_results = sorted(results, key=lambda x: x.throughput_mb_s, reverse=True)

    md = []
    md.append("# Tokenizer Benchmark Results: 4MB Input")
    md.append("")
    md.append("## Overview")
    md.append("")
    md.append("This benchmark compares the tokenization speed of different tokenizer libraries")
    md.append("on a single 4MB input (simulating a large document or batch).")
    md.append("")

    if use_llama3:
        md.append("**Note**: tiktoken and HuggingFace use the **same LLaMA 3 tokenizer** for")
        md.append("apples-to-apples comparison. SentencePiece uses T5 (different tokenizer family).")
        md.append("")

    md.append("### Test Configuration")
    md.append("")
    md.append(f"- **Input Size**: {results[0].input_size_mb:.2f} MB ({results[0].input_size_bytes:,} bytes)")
    md.append(f"- **Iterations**: {results[0].iterations}")
    md.append(f"- **Warmup**: 3 iterations")
    md.append("")
    md.append("## Results")
    md.append("")
    md.append("| Tokenizer | Time (ms) | Throughput (MB/s) | Tokens | Notes |")
    md.append("|-----------|-----------|-------------------|--------|-------|")

    for r in sorted_results:
        notes = ""
        if "tiktoken" in r.tokenizer_name:
            notes = "Rust impl, highly optimized"
        elif "streaming" in r.tokenizer_name:
            notes = "Rust impl, cache-optimized parallel"
        elif "parallel" in r.tokenizer_name:
            notes = "Rust impl, recursive parallel"
        elif "serial" in r.tokenizer_name:
            notes = "Rust impl, serial"
        elif "sentencepiece" in r.tokenizer_name:
            notes = "C++ impl, different model"

        md.append(
            f"| {r.tokenizer_name} | "
            f"{r.mean_time_ms:.1f} ± {r.std_time_ms:.1f} | "
            f"**{r.throughput_mb_s:.2f}** | "
            f"{r.token_count:,} | "
            f"{notes} |"
        )

    md.append("")
    md.append("## Analysis")
    md.append("")

    if sorted_results:
        fastest = sorted_results[0]
        md.append(f"**Fastest**: {fastest.tokenizer_name} at {fastest.throughput_mb_s:.2f} MB/s")
        md.append("")

        # Compare HuggingFace modes
        hf_serial = next((r for r in results if 'serial' in r.tokenizer_name), None)
        hf_parallel = next((r for r in results if 'parallel' in r.tokenizer_name and 'streaming' not in r.tokenizer_name), None)
        hf_streaming = next((r for r in results if 'streaming' in r.tokenizer_name), None)

        if hf_serial and hf_streaming:
            md.append("### HuggingFace Parallel Encoding Speedup")
            md.append("")
            md.append("| Mode | Time | Speedup vs Serial |")
            md.append("|------|------|-------------------|")
            md.append(f"| Serial | {hf_serial.mean_time_ms:.1f} ms | 1.00x (baseline) |")
            if hf_parallel:
                speedup = hf_serial.mean_time_ms / hf_parallel.mean_time_ms
                md.append(f"| Parallel | {hf_parallel.mean_time_ms:.1f} ms | **{speedup:.2f}x** |")
            if hf_streaming:
                speedup = hf_serial.mean_time_ms / hf_streaming.mean_time_ms
                md.append(f"| Streaming | {hf_streaming.mean_time_ms:.1f} ms | **{speedup:.2f}x** |")
            md.append("")

        # Compare to tiktoken
        tiktoken_result = next((r for r in results if 'tiktoken' in r.tokenizer_name), None)
        if tiktoken_result and hf_streaming:
            ratio = tiktoken_result.throughput_mb_s / hf_streaming.throughput_mb_s
            md.append(f"### tiktoken vs HuggingFace Streaming")
            md.append("")
            md.append(f"tiktoken is **{ratio:.1f}x faster** than HuggingFace streaming mode.")
            md.append("")
            md.append("This gap is primarily due to:")
            md.append("1. tiktoken's highly optimized BPE implementation")
            md.append("2. Different regex/pre-tokenization strategies")
            md.append("3. Python binding overhead differences")
            md.append("")

    md.append("## How to Reproduce")
    md.append("")
    md.append("```bash")
    md.append("# Install dependencies")
    md.append("pip install tiktoken sentencepiece tokenizers huggingface_hub transformers")
    md.append("")
    md.append("# Build local tokenizers with parallel encoding support")
    md.append("cd bindings/python")
    md.append("pip install maturin")
    md.append("maturin develop --release")
    md.append("")
    md.append("# Run benchmark with LLaMA 3")
    md.append("cd benches")
    md.append("python bench_4mb_comparison.py --use-llama3 --iterations 10")
    md.append("")
    md.append("# Or with specific thread count")
    md.append("RAYON_NUM_THREADS=8 python bench_4mb_comparison.py --use-llama3")
    md.append("```")
    md.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(md))

    print(f"Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tokenizers on 4MB input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with LLaMA 3 (apples-to-apples comparison)
  python bench_4mb_comparison.py --use-llama3

  # Run with default tiktoken (cl100k_base)
  python bench_4mb_comparison.py --no-llama3

  # Custom iterations and output
  python bench_4mb_comparison.py --iterations 20 --output-json results.json

  # With specific thread count
  RAYON_NUM_THREADS=8 python bench_4mb_comparison.py
        """
    )

    parser.add_argument(
        "--input-size-mb",
        type=float,
        default=TARGET_SIZE_MB,
        help=f"Input size in MB (default: {TARGET_SIZE_MB})"
    )
    parser.add_argument(
        "--use-llama3",
        action="store_true",
        default=True,
        help="Use LLaMA 3 tokenizer for tiktoken and HuggingFace (default: True)"
    )
    parser.add_argument(
        "--no-llama3",
        action="store_true",
        help="Use default tokenizers instead of LLaMA 3"
    )
    parser.add_argument(
        "--tiktoken-model",
        default="cl100k_base",
        help="tiktoken encoding to use when not using LLaMA 3 (default: cl100k_base)"
    )
    parser.add_argument(
        "--hf-model",
        default=LLAMA3_MODEL_ID,
        help=f"HuggingFace model to use (default: {LLAMA3_MODEL_ID})"
    )
    parser.add_argument(
        "--sentencepiece-model",
        help="Path to SentencePiece model file"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=BENCHMARK_ITERATIONS,
        help=f"Number of benchmark iterations (default: {BENCHMARK_ITERATIONS})"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_ITERATIONS,
        help=f"Number of warmup iterations (default: {WARMUP_ITERATIONS})"
    )
    parser.add_argument(
        "--output-json",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--output-markdown",
        default="BENCHMARK_COMPARISON_RESULTS.md",
        help="Save markdown report (default: BENCHMARK_COMPARISON_RESULTS.md)"
    )

    args = parser.parse_args()

    use_llama3 = args.use_llama3 and not args.no_llama3

    results = run_benchmarks(
        input_size_mb=args.input_size_mb,
        use_llama3=use_llama3,
        tiktoken_model=args.tiktoken_model,
        hf_model=args.hf_model,
        sentencepiece_model=args.sentencepiece_model,
        iterations=args.iterations,
        warmup=args.warmup,
        output_json=args.output_json,
    )

    if results and args.output_markdown:
        generate_markdown_report(results, args.output_markdown, use_llama3)


if __name__ == "__main__":
    main()
