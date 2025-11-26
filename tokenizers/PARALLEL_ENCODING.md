# Parallel Single-Input Encoding

## Overview

This implementation adds parallel tokenization for single long inputs to the Rust tokenizers library, providing significant speedup for large documents while maintaining correctness guarantees.

### Key Findings

- **Sweet spot: 100KB-500KB** — Recursive parallelism achieves **1.7-2.1x** speedup
- **Large inputs: 1MB+** — Streaming mode achieves **2.0-2.3x** consistent speedup
- **Streaming beats recursive at 1MB+** — Default threshold set to 1MB
- **Never slower than serial** — Safe to use for any input size

Two parallel encoding strategies available:
1. **Recursive Parallelism** — Best for 100KB-500KB (1.7-2.1x speedup)
2. **Cache-Block Streaming** — Best for 1MB+ (2.0-2.3x consistent speedup)

## Performance Optimizations

### 1. SIMD-Accelerated Boundary Detection

Uses the `memchr` crate for vectorized searching when finding split points:

```rust
use memchr::memchr_iter;

// 10-20x faster than byte-by-byte scanning for large inputs
for pos in memchr_iter(b' ', &bytes[target..end]) {
    // Process whitespace positions
}
```

Split point detection prioritizes **safe boundaries** in order:
1. `\n\n` — Paragraph breaks (guaranteed token boundary)
2. `. `, `.\n`, `! `, `!\n`, `? `, `?\n` — Sentence endings

### 2. Minimal Overlap for Safe Boundaries

When splitting at safe boundaries (sentence/paragraph ends), overlap is reduced by **80-90%**:

| Boundary Type | Overlap Size |
|--------------|--------------|
| Safe (sentence/paragraph) | 100-500 bytes |
| Regular (whitespace) | 500-5000 bytes |

This significantly reduces redundant tokenization work:
- With depth 4 recursion (16 chunks), regular overlap tokenizes ~5-15% extra bytes
- Safe boundary detection reduces this to ~1-2% extra bytes

### 3. Zero-Copy Offset Representation

Uses `LazyEncoding` to defer offset operations until final materialization:

```rust
struct LazyEncoding {
    encoding: Encoding,
    base_offset: usize,  // O(1) shift instead of O(n)
    start_idx: usize,    // O(log n) filter via binary search
    end_idx: usize,      // No data copying until merge
}
```

**Performance gains:**
- `shift_all_offsets()`: O(n) → O(1)
- `filter_tokens_starting_before/after()`: O(n) → O(log n)
- With depth 4 recursion: **~30 O(n) passes → 1 O(n) pass**

Only the final `materialize()` call performs actual data copying.

### 4. Zero-Cost Validation in Release

Debug/test builds validate parallel results against serial encoding. Release builds skip this entirely:

```rust
#[cfg(not(any(test, debug_assertions)))]
let should_validate = false; // Zero cost - no env var check
```

## Algorithms

### Strategy 1: Recursive Parallelism (<1MB)

Divide-and-conquer approach for medium inputs:

1. **Auto-tune** recursion depth based on input size and CPU cores
2. **Find safe split points** using SIMD search for sentence/paragraph boundaries
3. **Recursively split** input with minimal overlap at safe boundaries
4. **Encode chunks** in parallel using `rayon::join`
5. **Filter and merge** tokens using `LazyEncoding` (O(log n) binary search)
6. **Materialize** offsets only once at the final merge

**Key optimizations**:
- Safe boundary splitting reduces overlap from 500-5000 bytes to 100-500 bytes (80-90% less redundant work)
- `LazyEncoding` defers offset operations, turning O(n × depth) into O(n) total

**Best for**: 100KB-1MB inputs

### Strategy 2: Cache-Block Streaming (≥1MB)

Cache-optimized approach for large inputs:

```
┌─────────────────────────────────────────────────────┐
│                    1MB+ Input                       │
├────────┬────────┬────────┬────────┬────────┬───────┤
│Block 0 │Block 1 │Block 2 │Block 3 │Block 4 │ ...   │
│ 128KB  │ 128KB  │ 128KB  │ 128KB  │ 128KB  │       │
└────────┴────────┴────────┴────────┴────────┴───────┘
```

**Why streaming wins at 1MB+**:
- Each 128KB block fits in L2 cache
- Sequential memory access = prefetcher works optimally
- Parallel block processing
- Consistent 2.0-2.3x speedup at all sizes

### Auto Mode (Default)

Automatically selects:
- **<1MB**: Recursive parallelism
- **≥1MB**: Cache-Block Streaming

## API

```rust
impl Tokenizer {
    /// Encode with automatic mode selection (recommended)
    pub fn encode_parallel_single(
        &self,
        input: impl AsRef<str>,
        add_special_tokens: bool,
    ) -> Result<Encoding>

    /// Force streaming mode
    pub fn encode_streaming(
        &self,
        input: impl AsRef<str>,
        add_special_tokens: bool,
    ) -> Result<Encoding>

    /// Use custom configuration
    pub fn encode_parallel_with_config(
        &self,
        input: impl AsRef<str>,
        add_special_tokens: bool,
        config: ParallelConfig,
    ) -> Result<Encoding>
}
```

### Configuration

```rust
pub struct ParallelConfig {
    pub threshold: usize,           // Min size for parallel (default: 10KB)
    pub mode: ParallelMode,         // Auto, Recursive, or Streaming
    pub block_size: usize,          // Streaming block size (default: 128KB)
    pub streaming_threshold: usize, // Switch to streaming at (default: 1MB)
    pub safety_margin: usize,       // Extra overlap bytes (default: 64)
}
```

**Smart overlap sizing**: The actual overlap used depends on the split point type:
- Safe boundaries (sentence/paragraph ends): `max_token_len + safety_margin` (100-500 bytes)
- Regular whitespace boundaries: `max_token_len × 3` (500-5000 bytes)

## Performance

### Benchmark Results

| Input Size | Recursive | Streaming | Auto Mode |
|------------|-----------|-----------|-----------|
| 100KB | **1.68x** | 0.98x | Recursive |
| 250KB | **2.09x** | 1.50x | Recursive |
| 500KB | **1.89x** | 1.88x | ~Tie |
| 1MB | 1.93x | **2.17x** | **Streaming** |
| 2MB | 2.20x | **2.29x** | **Streaming** |
| 4MB | 2.16x | **2.22x** | **Streaming** |
| 8MB | 2.06x | **2.09x** | **Streaming** |

### Why Streaming Beats Recursive at 1MB+

**Recursive at large sizes**:
- Multiple threads access scattered memory regions
- Good speedup (2.0-2.2x) but streaming is slightly better
- L3 cache pressure at very large sizes

**Streaming at large sizes**:
- Each block fits in L2 cache
- Sequential access within blocks
- Parallel block processing
- **Result**: 2.0-2.3x consistent speedup

## Correctness

### Guarantees
✅ Token IDs identical to serial
✅ Tokens identical to serial
✅ Offsets identical to serial
✅ Decode consistency

### Not Computed (Performance)
❌ Word IDs not computed (use serial if needed)

### Fallback
Automatically falls back to serial if:
- Input < 10KB
- Parallelism disabled
- Validation fails (debug builds)

## Usage Examples

```rust
// Recommended: Auto mode
let encoding = tokenizer.encode_parallel_single(&text, false)?;

// Force streaming for any size
let encoding = tokenizer.encode_streaming(&text, false)?;

// Custom configuration
use tokenizers::tokenizer::parallel_encode::{ParallelConfig, ParallelMode};

let config = ParallelConfig {
    mode: ParallelMode::Streaming,
    block_size: 64 * 1024,
    ..ParallelConfig::default()
};
let encoding = tokenizer.encode_parallel_with_config(&text, false, config)?;
```

## Summary

| Input Size | Best Mode | Speedup |
|------------|-----------|---------|
| <50KB | Serial | 1.0x |
| 50KB-500KB | Recursive | **1.7-2.1x** |
| 500KB-1MB | Either | **~1.9x** |
| 1MB+ | Streaming | **2.0-2.3x** |

**Key insight**: Both modes achieve excellent speedup (~2x). Streaming provides slightly better and more consistent performance for large inputs (1MB+).

## Dependencies

The parallel encoding module uses:
- `rayon` — Work-stealing parallel execution
- `memchr` — SIMD-accelerated byte searching (AVX2/SSE2)

## Debug Mode

Set `DEBUG_PARALLEL=1` to enable verbose logging:

```bash
DEBUG_PARALLEL=1 cargo test parallel
```

This shows:
- Split point detection (safe vs regular boundaries)
- Overlap sizes used at each split
- Token counts before/after filtering
- Streaming block progress
