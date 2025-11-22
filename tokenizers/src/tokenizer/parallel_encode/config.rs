//! Configuration types and constants for parallel encoding.

// =============================================================================
// CONSTANTS - Thresholds
// =============================================================================

/// Default minimum input length (in bytes) before parallel encoding is used.
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 10_000;

/// Threshold for cache-block streaming (1MB+ inputs benefit from streaming).
/// Benchmarks show streaming outperforms recursive at 1MB+ due to better cache locality.
pub const DEFAULT_STREAMING_THRESHOLD: usize = 1024 * 1024;

// =============================================================================
// CONSTANTS - Chunk/Block Sizes
// =============================================================================

/// Default block size for cache-block streaming (8KB optimal for L1 cache).
/// Benchmarks show 8KB blocks achieve best throughput (~10 MiB/s at 4MB input).
pub const DEFAULT_BLOCK_SIZE: usize = 8 * 1024;

/// Default overlap size for cache-block streaming (1KB handles most token boundaries).
pub const DEFAULT_STREAMING_OVERLAP: usize = 1024;

/// Minimum tokens per chunk for recursive splitting.
/// Below this, parallel overhead dominates encoding time.
pub const DEFAULT_MIN_TOKENS_PER_CHUNK: usize = 10_000;

// =============================================================================
// CONSTANTS - Algorithm Parameters
// =============================================================================

/// Safety margin beyond the maximum token length for overlap.
pub const SAFETY_MARGIN: usize = 64;

/// Estimated bytes per token (conservative for English + BPE/WordPiece).
/// Used to estimate token count from input size.
pub const DEFAULT_BYTES_PER_TOKEN: f64 = 4.5;

/// Maximum recursion depth (safety cap).
/// Depth 4 = 16 chunks maximum.
pub const MAX_RECURSION_DEPTH: usize = 4;

// =============================================================================
// CONFIGURATION TYPES
// =============================================================================

/// Parallel encoding mode selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelMode {
    /// Recursive splitting (default) - best for 50KB-4MB inputs
    Recursive,
    /// Cache-block streaming - optimized for 4MB+ inputs with better cache locality
    Streaming,
    /// Auto-select based on input size (Streaming for 4MB+, Recursive otherwise)
    Auto,
}

impl Default for ParallelMode {
    fn default() -> Self {
        Self::Auto
    }
}

/// Configuration for parallel single-input encoding.
#[derive(Debug, Clone, Copy)]
pub struct ParallelConfig {
    /// Minimum input length in bytes to trigger parallel encoding
    pub threshold: usize,
    /// Additional safety margin in bytes beyond max token length
    pub safety_margin: usize,
    /// Minimum tokens per chunk for recursive splitting (default: 10,000)
    pub min_tokens_per_chunk: usize,
    /// Estimated bytes per token for depth calculation (default: 4.5)
    pub bytes_per_token_estimate: f64,
    /// Maximum recursion depth (0 = auto-detect based on cores, or explicit limit)
    pub max_depth: usize,
    /// Parallel encoding mode (Auto, Recursive, or Streaming)
    pub mode: ParallelMode,
    /// Block size for streaming mode (default: 128KB for L2 cache)
    pub block_size: usize,
    /// Overlap size for streaming mode (default: 1KB)
    pub streaming_overlap: usize,
    /// Threshold for auto-switching to streaming mode (default: 4MB)
    pub streaming_threshold: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_PARALLEL_THRESHOLD,
            safety_margin: SAFETY_MARGIN,
            min_tokens_per_chunk: DEFAULT_MIN_TOKENS_PER_CHUNK,
            bytes_per_token_estimate: DEFAULT_BYTES_PER_TOKEN,
            max_depth: 0, // Auto-detect
            mode: ParallelMode::Auto,
            block_size: DEFAULT_BLOCK_SIZE,
            streaming_overlap: DEFAULT_STREAMING_OVERLAP,
            streaming_threshold: DEFAULT_STREAMING_THRESHOLD,
        }
    }
}

impl ParallelConfig {
    /// Create a config optimized for streaming mode with cache locality.
    pub fn streaming() -> Self {
        Self {
            mode: ParallelMode::Streaming,
            ..Default::default()
        }
    }

    /// Create a config that forces recursive mode.
    pub fn recursive() -> Self {
        Self {
            mode: ParallelMode::Recursive,
            ..Default::default()
        }
    }

    /// Set the block size for streaming mode.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Set the overlap size for streaming mode.
    pub fn with_streaming_overlap(mut self, overlap: usize) -> Self {
        self.streaming_overlap = overlap;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert_eq!(config.threshold, DEFAULT_PARALLEL_THRESHOLD);
        assert_eq!(config.safety_margin, SAFETY_MARGIN);
        assert_eq!(config.min_tokens_per_chunk, DEFAULT_MIN_TOKENS_PER_CHUNK);
        assert!((config.bytes_per_token_estimate - DEFAULT_BYTES_PER_TOKEN).abs() < 0.001);
        assert_eq!(config.max_depth, 0);
        assert_eq!(config.mode, ParallelMode::Auto);
        assert_eq!(config.block_size, DEFAULT_BLOCK_SIZE);
        assert_eq!(config.streaming_overlap, DEFAULT_STREAMING_OVERLAP);
        assert_eq!(config.streaming_threshold, DEFAULT_STREAMING_THRESHOLD);
    }

    #[test]
    fn test_parallel_config_streaming() {
        let config = ParallelConfig::streaming();
        assert_eq!(config.mode, ParallelMode::Streaming);
        assert_eq!(config.block_size, 8 * 1024); // 8KB optimal block size
    }

    #[test]
    fn test_parallel_config_recursive() {
        let config = ParallelConfig::recursive();
        assert_eq!(config.mode, ParallelMode::Recursive);
    }

    #[test]
    fn test_parallel_config_with_block_size() {
        let config = ParallelConfig::streaming().with_block_size(64 * 1024);
        assert_eq!(config.block_size, 64 * 1024);
    }

    #[test]
    fn test_parallel_config_with_streaming_overlap() {
        let config = ParallelConfig::streaming().with_streaming_overlap(2048);
        assert_eq!(config.streaming_overlap, 2048);
    }
}
