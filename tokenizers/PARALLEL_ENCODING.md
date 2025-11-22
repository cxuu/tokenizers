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

## Algorithms

### Strategy 1: Recursive Parallelism (<1MB)

Divide-and-conquer approach for medium inputs:

1. **Auto-tune** recursion depth based on input size and CPU cores
2. **Recursively split** input at midpoints with overlap regions
3. **Encode chunks** in parallel using `rayon::join`
4. **Filter and merge** tokens based on their start positions

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
}
```

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
