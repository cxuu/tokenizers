//! Parallel encoding for single long inputs.
//!
//! This module implements multiple strategies to parallelize tokenization of a single long input:
//!
//! ## Strategy 1: Recursive Splitting (Default)
//! 1. Recursively splitting the input based on available cores and input size
//! 2. Encoding chunks in parallel with overlapping regions
//! 3. Merging results by using left encoding up to midpoint and right encoding after
//! 4. Filtering and merging the results deterministically
//!
//! ## Strategy 2: Cache-Block Streaming (New)
//! Optimized for cache locality on very large inputs (4MB+):
//! 1. Process input in L2-cache-sized blocks (~128KB) sequentially
//! 2. Use small overlap windows (~1KB) to handle token boundaries
//! 3. Stream results directly to output, avoiding cache thrashing
//! 4. Each block fits entirely in L2 cache with vocabulary lookups
//!
//! This ensures the output is identical to serial encoding while providing speedup
//! for long inputs on multi-core systems.

use crate::tokenizer::{Encoding, Result};
use crate::utils::parallelism::get_parallelism;
use memchr::memchr_iter;
use rayon;
use rayon::prelude::*;

/// Default minimum input length (in bytes) before parallel encoding is used.
const DEFAULT_PARALLEL_THRESHOLD: usize = 10_000;

/// Safety margin beyond the maximum token length for overlap.
const SAFETY_MARGIN: usize = 64;

/// Minimum tokens per chunk for recursive splitting.
/// Below this, parallel overhead dominates encoding time.
const DEFAULT_MIN_TOKENS_PER_CHUNK: usize = 10_000;

/// Estimated bytes per token (conservative for English + BPE/WordPiece).
/// Used to estimate token count from input size.
const DEFAULT_BYTES_PER_TOKEN: f64 = 4.5;

/// Maximum recursion depth (safety cap).
/// Depth 4 = 16 chunks maximum.
const MAX_RECURSION_DEPTH: usize = 4;

/// Default block size for cache-block streaming (128KB fits in L2 cache).
const DEFAULT_BLOCK_SIZE: usize = 128 * 1024;

/// Default overlap size for cache-block streaming (1KB handles most token boundaries).
const DEFAULT_STREAMING_OVERLAP: usize = 1024;

/// Threshold for cache-block streaming (1MB+ inputs benefit from streaming).
/// Benchmarks show streaming outperforms recursive at 1MB+ due to better cache locality.
const DEFAULT_STREAMING_THRESHOLD: usize = 1024 * 1024;

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

/// Compute the optimal recursion depth based on input size and available cores.
///
/// Uses token-based calculation: each chunk should have at least `min_tokens_per_chunk`
/// tokens to justify the parallel overhead.
fn compute_optimal_depth(
    input_bytes: usize,
    num_cores: usize,
    min_tokens_per_chunk: usize,
    bytes_per_token: f64,
) -> usize {
    if num_cores <= 1 {
        return 0; // No parallelism benefit with single core
    }

    // Estimate total tokens from input size
    let estimated_tokens = (input_bytes as f64 / bytes_per_token) as usize;

    // How many chunks can we make while keeping each >= min_tokens_per_chunk?
    let max_chunks_by_tokens = estimated_tokens / min_tokens_per_chunk;

    // depth = floor(log2(max_chunks))
    let depth_by_tokens = if max_chunks_by_tokens >= 2 {
        (max_chunks_by_tokens as f64).log2().floor() as usize
    } else {
        0 // Input too small for splitting
    };

    // Also limit by available cores (no benefit beyond core count)
    let depth_by_cores = (num_cores as f64).log2().ceil() as usize;

    depth_by_tokens.min(depth_by_cores).min(MAX_RECURSION_DEPTH)
}

/// Find a UTF-8 character boundary at or after the given byte position.
fn find_char_boundary_forward(s: &str, pos: usize) -> usize {
    let mut pos = pos.min(s.len());
    while pos < s.len() && !s.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

/// Find a UTF-8 character boundary at or before the given byte position.
#[allow(dead_code)]
fn find_char_boundary_backward(s: &str, pos: usize) -> usize {
    let mut pos = pos.min(s.len());
    while pos > 0 && !s.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

// NOTE: Word ID computation has been completely removed from parallel encoding.
// Computing word IDs (either via pre-computation or heuristics) adds overhead
// that reduces the performance benefits of parallelization.
//
// Word IDs are rarely needed for most use cases (LLM inference, embeddings, etc.).
// If word IDs are required, use serial encoding instead: tokenizer.encode()

/// Result of finding a split point, including whether it's a "safe" boundary.
#[derive(Debug, Clone, Copy)]
struct SplitPoint {
    position: usize,
    /// If true, this is a safe boundary (sentence/paragraph end) that needs minimal overlap
    is_safe_boundary: bool,
}

/// Find a good split point near the target position using SIMD-accelerated search.
/// Prefers safe boundaries (sentence/paragraph ends) which need minimal overlap,
/// falls back to whitespace boundaries.
fn find_split_point_safe(input: &str, target: usize, search_window: usize) -> SplitPoint {
    let bytes = input.as_bytes();
    let target = target.min(input.len());
    let search_start = target.saturating_sub(search_window);
    let search_end = (target + search_window).min(input.len());

    // Phase 1: Look for safe boundaries (paragraph breaks, sentence ends)
    // These are guaranteed token boundaries where minimal overlap is needed

    // First priority: double newline (paragraph break) - safest boundary
    if let Some(pos) = find_pattern_near(bytes, target, search_start, search_end, b"\n\n") {
        return SplitPoint {
            position: pos + 2, // Split after the paragraph break
            is_safe_boundary: true,
        };
    }

    // Second priority: sentence-ending punctuation followed by whitespace
    // Use SIMD to find periods, then check context
    for &(punct, follow) in &[(b'.', b' '), (b'.', b'\n'), (b'!', b' '), (b'!', b'\n'), (b'?', b' '), (b'?', b'\n')] {
        if let Some(pos) = find_punct_boundary_near(bytes, target, search_start, search_end, punct, follow) {
            return SplitPoint {
                position: pos + 2, // Split after "X " or "X\n"
                is_safe_boundary: true,
            };
        }
    }

    // Phase 2: Fall back to any whitespace using SIMD
    // Search forward from target first (prefer not to backtrack)
    let forward_slice = &bytes[target..search_end];
    for &ws in &[b' ', b'\n', b'\t'] {
        for pos in memchr_iter(ws, forward_slice) {
            let global_pos = target + pos;
            // Verify it's a char boundary
            if input.is_char_boundary(global_pos) {
                return SplitPoint {
                    position: global_pos,
                    is_safe_boundary: false,
                };
            }
        }
    }

    // Search backward if no forward whitespace found
    let backward_slice = &bytes[search_start..target];
    for &ws in &[b' ', b'\n', b'\t'] {
        // Find last occurrence by iterating
        let mut last_pos = None;
        for pos in memchr_iter(ws, backward_slice) {
            last_pos = Some(search_start + pos);
        }
        if let Some(pos) = last_pos {
            if input.is_char_boundary(pos + 1) {
                return SplitPoint {
                    position: pos + 1, // Position after the whitespace
                    is_safe_boundary: false,
                };
            }
        }
    }

    // Final fallback: UTF-8 character boundary
    SplitPoint {
        position: find_char_boundary_forward(input, target),
        is_safe_boundary: false,
    }
}

/// SIMD-accelerated search for a 2-byte pattern near target position.
#[inline]
fn find_pattern_near(bytes: &[u8], target: usize, start: usize, end: usize, pattern: &[u8]) -> Option<usize> {
    if pattern.len() < 1 || end <= start {
        return None;
    }

    let search_slice = &bytes[start..end];
    let first_byte = pattern[0];

    // Use SIMD to find first byte, then verify rest of pattern
    let mut best: Option<usize> = None;
    let mut best_dist = usize::MAX;

    for pos in memchr_iter(first_byte, search_slice) {
        let global_pos = start + pos;
        // Check if full pattern matches
        if global_pos + pattern.len() <= bytes.len()
            && &bytes[global_pos..global_pos + pattern.len()] == pattern
        {
            let dist = if global_pos >= target {
                global_pos - target
            } else {
                target - global_pos
            };
            if dist < best_dist {
                best_dist = dist;
                best = Some(global_pos);
            }
        }
    }

    best
}

/// SIMD-accelerated search for punctuation followed by specific character.
#[inline]
fn find_punct_boundary_near(bytes: &[u8], target: usize, start: usize, end: usize, punct: u8, follow: u8) -> Option<usize> {
    let search_slice = &bytes[start..end.saturating_sub(1)]; // -1 to have room for follow char

    let mut best: Option<usize> = None;
    let mut best_dist = usize::MAX;

    for pos in memchr_iter(punct, search_slice) {
        let global_pos = start + pos;
        // Check if followed by expected character
        if global_pos + 1 < bytes.len() && bytes[global_pos + 1] == follow {
            let dist = if global_pos >= target {
                global_pos - target
            } else {
                target - global_pos
            };
            if dist < best_dist {
                best_dist = dist;
                best = Some(global_pos);
            }
        }
    }

    best
}

/// Legacy wrapper for compatibility - returns just the position.
#[allow(dead_code)]
fn find_split_point(input: &str, target: usize, search_window: usize) -> usize {
    find_split_point_safe(input, target, search_window).position
}

/// Calculate the overlap size based on max token length and whether we're at a safe boundary.
///
/// For safe boundaries (sentence/paragraph ends), we use minimal overlap since tokenization
/// is guaranteed to be consistent. For other boundaries, we use larger overlap.
fn calculate_overlap_size(max_token_len: usize, safety_margin: usize, is_safe_boundary: bool) -> usize {
    if is_safe_boundary {
        // Safe boundaries (sentence/paragraph ends) need minimal overlap
        // Just enough to handle the longest possible token
        (max_token_len + safety_margin).max(100).min(500)
    } else {
        // Regular boundaries need larger overlap for safety
        // Use 3x multiplier with reasonable bounds
        ((max_token_len.max(safety_margin)) * 3)
            .max(500) // Minimum overlap
            .min(5000) // Maximum overlap to prevent wasted work
    }
}

/// Internal recursive encoding function.
///
/// Recursively splits the input and encodes chunks in parallel until reaching
/// the base case (max depth or chunk too small).
///
/// The algorithm:
/// 1. Split input at midpoint with overlap on both sides
/// 2. Left chunk: [0, mid + overlap], Right chunk: [mid - overlap, end]
/// 3. Both chunks encode the overlap region [mid - overlap, mid + overlap]
/// 4. After encoding, use LEFT's tokens for positions < mid (split point)
/// 5. Use RIGHT's tokens for positions >= mid
/// 6. This ensures correct tokenization because both sides have enough context
fn encode_recursive<F>(
    encode_fn: &F,
    max_token_len: usize,
    input: &str,
    input_global_offset: usize, // Offset of this chunk in the original input
    current_depth: usize,
    max_depth: usize,
    config: &ParallelConfig,
) -> Result<Encoding>
where
    F: Fn(&str, bool) -> Result<Encoding> + Send + Sync,
{
    let debug = std::env::var("DEBUG_PARALLEL").is_ok();

    // Base case: reached max depth or input too small
    let estimated_tokens = (input.len() as f64 / config.bytes_per_token_estimate) as usize;
    let min_bytes_for_split =
        (config.min_tokens_per_chunk as f64 * config.bytes_per_token_estimate * 2.0) as usize;

    if current_depth >= max_depth
        || input.len() < min_bytes_for_split
        || estimated_tokens < config.min_tokens_per_chunk * 2
    {
        if debug {
            println!(
                "[RECURSIVE depth={}] Base case: encoding {} bytes (est. {} tokens) at global offset {}",
                current_depth,
                input.len(),
                estimated_tokens,
                input_global_offset
            );
        }

        let mut encoding = encode_fn(input, false)?;

        // Shift offsets to global coordinates if not at root
        if input_global_offset > 0 {
            encoding.shift_all_offsets(input_global_offset as isize);
        }

        return Ok(encoding);
    }

    if debug {
        println!(
            "[RECURSIVE depth={}] Splitting {} bytes (est. {} tokens) at global offset {}",
            current_depth,
            input.len(),
            estimated_tokens,
            input_global_offset
        );
    }

    // Find split point near midpoint using SIMD-accelerated search
    // Prefer safe boundaries (sentence/paragraph ends) which need minimal overlap
    let mid = input.len() / 2;
    let split_result = find_split_point_safe(input, mid, 500); // Larger window to find safe boundaries
    let split_point = split_result.position;

    // Calculate overlap size - much smaller for safe boundaries
    let overlap_size = calculate_overlap_size(max_token_len, config.safety_margin, split_result.is_safe_boundary);

    if debug && split_result.is_safe_boundary {
        println!(
            "[RECURSIVE depth={}] Found safe boundary at {}, using minimal overlap ({})",
            current_depth, split_point, overlap_size
        );
    }

    // Create overlapping chunks:
    // Left chunk: [0, split_point + overlap_size]
    // Right chunk: [split_point - overlap_size, end]
    // Both chunks encode the region around split_point with enough overlap
    let left_end = find_char_boundary_forward(input, (split_point + overlap_size).min(input.len()));
    let right_start = find_char_boundary_forward(input, split_point.saturating_sub(overlap_size));

    let left_chunk = &input[..left_end];
    let right_chunk = &input[right_start..];

    // The merge point is where we switch from left to right encoding
    // Use the split_point (in global coordinates) as the boundary
    let global_split_point = input_global_offset + split_point;

    if debug {
        println!(
            "[RECURSIVE depth={}] Split at {} (global {}), left=0..{}, right={}..{}, overlap=[{}, {}]",
            current_depth,
            split_point,
            global_split_point,
            left_end,
            right_start,
            input.len(),
            right_start,
            left_end
        );
    }

    // Parallel recursive calls for left and right
    let (left_result, right_result) = rayon::join(
        || {
            encode_recursive(
                encode_fn,
                max_token_len,
                left_chunk,
                input_global_offset, // Left chunk starts at same global offset
                current_depth + 1,
                max_depth,
                config,
            )
        },
        || {
            encode_recursive(
                encode_fn,
                max_token_len,
                right_chunk,
                input_global_offset + right_start, // Right chunk offset
                current_depth + 1,
                max_depth,
                config,
            )
        },
    );

    let mut left_enc = left_result?;
    let mut right_enc = right_result?;

    if debug {
        println!(
            "[RECURSIVE depth={}] Before filtering - Left: {} tokens, Right: {} tokens",
            current_depth,
            left_enc.len(),
            right_enc.len()
        );
    }

    // Count tokens that should be in each half based on their start position
    let left_tokens_expected = left_enc
        .get_offsets()
        .iter()
        .filter(|(start, _)| *start < global_split_point)
        .count();
    let right_tokens_expected = right_enc
        .get_offsets()
        .iter()
        .filter(|(start, _)| *start >= global_split_point)
        .count();

    // Filter tokens based on where they START:
    // - Left: keep tokens that START before the split point
    // - Right: keep tokens that START at or after the split point
    // This ensures each token is assigned to exactly one side with no duplicates or gaps
    left_enc.filter_tokens_starting_before(global_split_point);
    right_enc.filter_tokens_starting_at_or_after(global_split_point);

    if debug {
        println!(
            "[RECURSIVE depth={}] After filtering - Left: {} tokens, Right: {} tokens",
            current_depth,
            left_enc.len(),
            right_enc.len()
        );
    }

    // Safety check: if expected tokens are missing after filtering, something went wrong
    // This can happen when tokens span way beyond the overlap region
    // Fall back to serial encoding for this chunk
    if (left_tokens_expected > 0 && left_enc.is_empty())
        || (right_tokens_expected > 0 && right_enc.is_empty())
    {
        if debug {
            println!(
                "[RECURSIVE depth={}] WARNING: Tokens lost during filtering (left: {} expected, {} got; right: {} expected, {} got), falling back to serial",
                current_depth,
                left_tokens_expected,
                left_enc.len(),
                right_tokens_expected,
                right_enc.len()
            );
        }
        let mut encoding = encode_fn(input, false)?;
        if input_global_offset > 0 {
            encoding.shift_all_offsets(input_global_offset as isize);
        }
        return Ok(encoding);
    }

    // Additional safety check: ensure left and right properly cover the split boundary
    // Left's last token should end near the split point, and right's first token should start near it
    let left_max_end = left_enc.get_offsets().last().map(|(_, end)| *end).unwrap_or(0);
    let right_min_start = right_enc.get_offsets().first().map(|(start, _)| *start).unwrap_or(input_global_offset + input.len());

    // If there's a significant gap between left's end and right's start, fall back
    // Allow some gap for whitespace-only regions, but not more than the overlap size
    let gap = right_min_start.saturating_sub(left_max_end);
    if gap > overlap_size {
        if debug {
            println!(
                "[RECURSIVE depth={}] WARNING: Gap between left (end={}) and right (start={}) is {} bytes, exceeds overlap {}, falling back to serial",
                current_depth,
                left_max_end,
                right_min_start,
                gap,
                overlap_size
            );
        }
        let mut encoding = encode_fn(input, false)?;
        if input_global_offset > 0 {
            encoding.shift_all_offsets(input_global_offset as isize);
        }
        return Ok(encoding);
    }

    // Merge: left + right
    let mut result = left_enc;
    result.merge_with(right_enc, false);

    if debug {
        println!(
            "[RECURSIVE depth={}] After merge: {} total tokens",
            current_depth,
            result.len()
        );
    }

    // Validate for overlapping offsets
    let has_overlapping_offsets = result.get_offsets().windows(2).any(|w| {
        let (_start1, end1) = w[0];
        let (start2, _end2) = w[1];
        start2 < end1
    });

    if has_overlapping_offsets {
        if debug {
            println!(
                "[RECURSIVE depth={}] WARNING: Detected overlapping offsets, falling back to serial",
                current_depth
            );
        }
        let mut encoding = encode_fn(input, false)?;
        if input_global_offset > 0 {
            encoding.shift_all_offsets(input_global_offset as isize);
        }
        return Ok(encoding);
    }

    Ok(result)
}

/// Cache-block streaming encode for very large inputs (4MB+).
///
/// This function processes the input in L2-cache-sized blocks to optimize for
/// cache locality. Each block is processed sequentially with a small overlap
/// window to ensure correct handling of token boundaries.
///
/// # Arguments
///
/// * `encode_fn` - Function that encodes a string slice
/// * `max_token_len` - Maximum token length in bytes from the vocabulary
/// * `input` - The input string to tokenize
/// * `config` - Configuration with block_size and streaming_overlap
///
/// # Returns
///
/// An `Encoding` with results streamed from cache-resident blocks.
#[allow(dead_code)]
fn encode_streaming<F>(
    encode_fn: &F,
    max_token_len: usize,
    input: &str,
    config: &ParallelConfig,
) -> Result<Encoding>
where
    F: Fn(&str, bool) -> Result<Encoding> + Send + Sync,
{
    let debug = std::env::var("DEBUG_PARALLEL").is_ok();
    let block_size = config.block_size;
    let overlap = config.streaming_overlap.max(max_token_len * 3);

    if debug {
        println!(
            "[STREAMING] Input: {} bytes, Block size: {} bytes, Overlap: {} bytes",
            input.len(),
            block_size,
            overlap
        );
    }

    // Estimate total tokens for pre-allocation
    let estimated_tokens = (input.len() as f64 / config.bytes_per_token_estimate) as usize;
    let mut result = Encoding::with_capacity(estimated_tokens);

    let mut offset = 0;
    let mut block_count = 0;

    while offset < input.len() {
        block_count += 1;

        // Calculate block boundaries
        let block_end = (offset + block_size).min(input.len());
        let process_end = if block_end < input.len() {
            // Include overlap for next block
            find_char_boundary_forward(input, (block_end + overlap).min(input.len()))
        } else {
            input.len()
        };

        // Extract block with potential overlap
        let block = &input[offset..process_end];

        if debug {
            println!(
                "[STREAMING] Block {}: offset={}, block_end={}, process_end={}, block_len={}",
                block_count,
                offset,
                block_end,
                process_end,
                block.len()
            );
        }

        // Encode this block
        let mut block_encoding = encode_fn(block, false)?;

        // Shift offsets to global coordinates
        if offset > 0 {
            block_encoding.shift_all_offsets(offset as isize);
        }

        // Filter tokens: keep only tokens that END before or at the block boundary
        // (tokens in the overlap region will be processed again with the next block)
        // Exception: if this is the last block, keep all tokens
        if block_end < input.len() {
            // Find the effective boundary: last token that starts before block_end
            // We use block_end (not process_end) as the boundary for filtering
            let global_block_end = offset + block_size;

            // Keep tokens that START before the block boundary
            // This ensures we don't lose any tokens and don't duplicate them
            let tokens_before = block_encoding
                .get_offsets()
                .iter()
                .filter(|(start, _)| *start < global_block_end)
                .count();

            block_encoding.filter_tokens_starting_before(global_block_end);

            if debug {
                println!(
                    "[STREAMING] Block {}: Kept {} tokens (of {} before filter)",
                    block_count,
                    block_encoding.len(),
                    tokens_before
                );
            }
        }

        // Stream tokens to result
        if result.is_empty() {
            result = block_encoding;
        } else {
            result.merge_with(block_encoding, false);
        }

        if debug {
            println!(
                "[STREAMING] Block {}: Result now has {} tokens",
                block_count,
                result.len()
            );
        }

        // Move to next block
        offset = block_end;
    }

    if debug {
        println!(
            "[STREAMING] Complete: {} blocks, {} total tokens",
            block_count,
            result.len()
        );
    }

    Ok(result)
}

/// Cache-block streaming encode with parallel block processing.
///
/// This function combines the cache locality benefits of streaming with
/// parallel processing of multiple blocks at once using rayon.
///
/// # Strategy
/// 1. Divide input into cache-sized blocks with overlaps
/// 2. Process blocks in parallel using rayon
/// 3. Merge results sequentially with proper boundary handling
///
/// # Arguments
///
/// * `encode_fn` - Function that encodes a string slice
/// * `max_token_len` - Maximum token length in bytes from the vocabulary
/// * `input` - The input string to tokenize
/// * `config` - Configuration with block_size and streaming_overlap
fn encode_streaming_parallel<F>(
    encode_fn: &F,
    max_token_len: usize,
    input: &str,
    config: &ParallelConfig,
) -> Result<Encoding>
where
    F: Fn(&str, bool) -> Result<Encoding> + Send + Sync,
{
    let debug = std::env::var("DEBUG_PARALLEL").is_ok();
    let block_size = config.block_size;
    let overlap = config.streaming_overlap.max(max_token_len * 3);

    // Calculate block boundaries
    // Each block has: (start_offset, logical_start, logical_end, end_offset)
    // - start_offset: where to start encoding (includes left overlap)
    // - logical_start: where this block's tokens should start (filter boundary)
    // - logical_end: where this block's tokens should end (filter boundary)
    // - end_offset: where to end encoding (includes right overlap)
    let mut blocks: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut logical_pos = 0;

    while logical_pos < input.len() {
        let logical_end = (logical_pos + block_size).min(input.len());

        // Include overlap on the left (except for first block)
        let start_offset = if logical_pos > 0 {
            find_char_boundary_forward(input, logical_pos.saturating_sub(overlap))
        } else {
            0
        };

        // Include overlap on the right (except for last block)
        let end_offset = if logical_end < input.len() {
            find_char_boundary_forward(input, (logical_end + overlap).min(input.len()))
        } else {
            input.len()
        };

        blocks.push((start_offset, logical_pos, logical_end, end_offset));
        logical_pos = logical_end;
    }

    if debug {
        println!(
            "[STREAMING_PARALLEL] Input: {} bytes, {} blocks",
            input.len(),
            blocks.len()
        );
    }

    // Encode all blocks in parallel
    // Each block encodes with overlap but only keeps tokens in its logical range
    let block_results: Vec<Result<(usize, usize, Encoding)>> = blocks
        .into_par_iter()
        .map(|(start_offset, logical_start, logical_end, end_offset)| {
            let block = &input[start_offset..end_offset];
            let mut block_encoding = encode_fn(block, false)?;

            // Shift offsets to global coordinates
            if start_offset > 0 {
                block_encoding.shift_all_offsets(start_offset as isize);
            }

            Ok((logical_start, logical_end, block_encoding))
        })
        .collect();

    // Merge results sequentially
    let estimated_tokens = (input.len() as f64 / config.bytes_per_token_estimate) as usize;
    let mut result = Encoding::with_capacity(estimated_tokens);

    for (idx, block_result) in block_results.into_iter().enumerate() {
        let (logical_start, logical_end, mut block_encoding) = block_result?;
        let is_first_block = logical_start == 0;
        let is_last_block = logical_end >= input.len();

        // Filter tokens to only keep those in the logical range [logical_start, logical_end)
        // This removes tokens from the overlap regions
        if !is_first_block {
            block_encoding.filter_tokens_starting_at_or_after(logical_start);
        }
        if !is_last_block {
            block_encoding.filter_tokens_starting_before(logical_end);
        }

        if debug {
            println!(
                "[STREAMING_PARALLEL] Block {}: logical=[{}, {}), kept {} tokens",
                idx,
                logical_start,
                logical_end,
                block_encoding.len()
            );
        }

        // Merge into result
        if result.is_empty() {
            result = block_encoding;
        } else {
            result.merge_with(block_encoding, false);
        }
    }

    if debug {
        println!(
            "[STREAMING_PARALLEL] Complete: {} total tokens",
            result.len()
        );
    }

    Ok(result)
}

/// Encode a single long input in parallel using recursive overlapping chunks.
///
/// # Arguments
///
/// * `encode_fn` - Function that encodes a string slice (without special tokens)
/// * `post_process_fn` - Function that applies post-processing with special tokens
/// * `max_token_len` - Maximum token length in bytes from the model vocabulary
/// * `input` - The input string to tokenize
/// * `add_special_tokens` - Whether to add special tokens via post-processing
/// * `config` - Configuration for parallel encoding
///
/// # Returns
///
/// An `Encoding` with token IDs, tokens, and offsets identical to serial encoding.
/// Word IDs are NOT computed (will be None) for maximum performance.
pub fn encode_parallel_single<F, P>(
    encode_fn: F,
    post_process_fn: P,
    max_token_len: usize,
    input: &str,
    add_special_tokens: bool,
    config: ParallelConfig,
) -> Result<Encoding>
where
    F: Fn(&str, bool) -> Result<Encoding> + Send + Sync,
    P: Fn(Encoding, Option<Encoding>, bool) -> Result<Encoding>,
{
    let debug = std::env::var("DEBUG_PARALLEL").is_ok();

    // Early exit for short inputs or if parallelism is disabled
    if input.len() < config.threshold || !get_parallelism() {
        return encode_fn(input, add_special_tokens);
    }

    // Determine which mode to use
    let use_streaming = match config.mode {
        ParallelMode::Streaming => true,
        ParallelMode::Recursive => false,
        ParallelMode::Auto => input.len() >= config.streaming_threshold,
    };

    if debug {
        let estimated_tokens = (input.len() as f64 / config.bytes_per_token_estimate) as usize;
        println!(
            "[PARALLEL_ENCODE] Input: {} bytes (~{} tokens), mode: {:?}, streaming: {}",
            input.len(),
            estimated_tokens,
            config.mode,
            use_streaming
        );
    }

    let result = if use_streaming {
        // Use cache-block streaming for very large inputs
        if debug {
            println!(
                "[PARALLEL_ENCODE] Using streaming mode (block_size: {}KB, overlap: {})",
                config.block_size / 1024,
                config.streaming_overlap
            );
        }
        encode_streaming_parallel(&encode_fn, max_token_len, input, &config)?
    } else {
        // Use recursive parallel encoding
        let num_cores = rayon::current_num_threads();
        let auto_depth = compute_optimal_depth(
            input.len(),
            num_cores,
            config.min_tokens_per_chunk,
            config.bytes_per_token_estimate,
        );

        // Use configured max_depth if set, otherwise use auto-detected depth
        let max_depth = if config.max_depth > 0 {
            config.max_depth.min(MAX_RECURSION_DEPTH)
        } else {
            auto_depth
        };

        if debug {
            println!(
                "[PARALLEL_ENCODE] Using recursive mode (cores: {}, depth: {})",
                num_cores, max_depth
            );
        }

        // If depth is 0, just use serial encoding
        if max_depth == 0 {
            return encode_fn(input, add_special_tokens);
        }

        // Perform recursive parallel encoding
        encode_recursive(
            &encode_fn,
            max_token_len,
            input,
            0, // Start at global offset 0
            0, // Start at depth 0
            max_depth,
            &config,
        )?
    };

    // Validation: compare with serial encoding in test/debug builds only
    // In release builds, skip validation entirely for zero overhead
    #[cfg(any(test, debug_assertions))]
    let should_validate = true;
    #[cfg(not(any(test, debug_assertions)))]
    let should_validate = false; // Zero cost in release - no env var check

    if should_validate {
        let reference = encode_fn(input, false)?;
        if result.get_ids() != reference.get_ids() {
            if debug {
                // Find first difference
                for (i, (r_id, p_id)) in reference
                    .get_ids()
                    .iter()
                    .zip(result.get_ids().iter())
                    .enumerate()
                {
                    if r_id != p_id {
                        println!(
                            "[PARALLEL_ENCODE] Mismatch at position {}: ref={} ('{}'), parallel={} ('{}')",
                            i,
                            r_id,
                            reference.get_tokens()[i],
                            p_id,
                            result.get_tokens()[i]
                        );
                        break;
                    }
                }
                println!(
                    "[PARALLEL_ENCODE] WARNING: Validation failed (mode={:?}), falling back to serial",
                    if use_streaming { "streaming" } else { "recursive" }
                );
            }
            // Use the reference (serial) result
            return if add_special_tokens {
                post_process_fn(reference, None, true)
            } else {
                Ok(reference)
            };
        }
    }

    // Apply post-processing with special tokens if needed
    if add_special_tokens {
        post_process_fn(result, None, true)
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_char_boundary_forward() {
        let s = "Hello 世界";
        assert_eq!(find_char_boundary_forward(s, 0), 0);
        assert_eq!(find_char_boundary_forward(s, 6), 6);
        // Middle of multi-byte char '世' (starts at 6, is 3 bytes)
        assert_eq!(find_char_boundary_forward(s, 7), 9);
        assert_eq!(find_char_boundary_forward(s, 100), s.len());
    }

    #[test]
    fn test_find_char_boundary_backward() {
        let s = "Hello 世界";
        assert_eq!(find_char_boundary_backward(s, 0), 0);
        assert_eq!(find_char_boundary_backward(s, 6), 6);
        // Middle of multi-byte char '世' (starts at 6, is 3 bytes)
        assert_eq!(find_char_boundary_backward(s, 7), 6);
        assert_eq!(find_char_boundary_backward(s, 100), s.len());
    }

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
        assert_eq!(config.block_size, DEFAULT_BLOCK_SIZE);
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

    #[test]
    fn test_compute_optimal_depth() {
        // Single core: always 0
        assert_eq!(compute_optimal_depth(1_000_000, 1, 10_000, 4.5), 0);

        // Small input (45KB = ~10k tokens): depth 0
        assert_eq!(compute_optimal_depth(45_000, 8, 10_000, 4.5), 0);

        // 90KB (~20k tokens) with 8 cores: depth 1 (2 chunks of 10k each)
        assert_eq!(compute_optimal_depth(90_000, 8, 10_000, 4.5), 1);

        // 180KB (~40k tokens) with 8 cores: depth 2 (4 chunks of 10k each)
        assert_eq!(compute_optimal_depth(180_000, 8, 10_000, 4.5), 2);

        // 360KB (~80k tokens) with 8 cores: depth 3 (8 chunks)
        assert_eq!(compute_optimal_depth(360_000, 8, 10_000, 4.5), 3);

        // 720KB (~160k tokens) with 8 cores: depth 3 (core-limited)
        assert_eq!(compute_optimal_depth(720_000, 8, 10_000, 4.5), 3);

        // 1MB with 16 cores: depth 4 (16 chunks)
        assert_eq!(compute_optimal_depth(1_000_000, 16, 10_000, 4.5), 4);

        // Very large input: capped at MAX_RECURSION_DEPTH (4)
        assert_eq!(compute_optimal_depth(10_000_000, 32, 10_000, 4.5), 4);

        // 2 cores: max depth 1
        assert_eq!(compute_optimal_depth(1_000_000, 2, 10_000, 4.5), 1);

        // 4 cores: max depth 2
        assert_eq!(compute_optimal_depth(1_000_000, 4, 10_000, 4.5), 2);
    }

    #[test]
    fn test_calculate_overlap_size() {
        // Regular boundary (not safe): uses 3x multiplier with bounds
        assert_eq!(calculate_overlap_size(100, 64, false), 500); // Hits minimum
        assert_eq!(calculate_overlap_size(1000, 64, false), 3000); // 1000 * 3
        assert_eq!(calculate_overlap_size(2000, 64, false), 5000); // Capped at 5000

        // Safe boundary (sentence/paragraph end): uses minimal overlap
        assert_eq!(calculate_overlap_size(100, 64, true), 164); // 100 + 64
        assert_eq!(calculate_overlap_size(50, 30, true), 100); // Hits minimum of 100
        assert_eq!(calculate_overlap_size(500, 64, true), 500); // Capped at 500
    }

    #[test]
    fn test_find_split_point_safe() {
        // Test paragraph break detection
        // "Hello world.\n\n" - paragraph break is at positions 12-13, so after is 14
        let input = "Hello world.\n\nThis is a new paragraph.";
        let result = find_split_point_safe(input, 15, 20);
        assert!(result.is_safe_boundary);
        assert_eq!(result.position, 14); // After "\n\n"

        // Test sentence boundary detection
        // "First sentence. " - ". " is at positions 14-15, so after is 16
        let input2 = "First sentence. Second sentence here.";
        let result2 = find_split_point_safe(input2, 16, 10);
        assert!(result2.is_safe_boundary);
        assert_eq!(result2.position, 16); // After ". "

        // Test fallback to whitespace
        let input3 = "word1 word2 word3";
        let result3 = find_split_point_safe(input3, 8, 5);
        assert!(!result3.is_safe_boundary);
    }
}
