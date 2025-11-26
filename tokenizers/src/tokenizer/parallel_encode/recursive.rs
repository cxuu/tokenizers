//! Recursive parallel encoding strategy.
//!
//! This module implements the recursive splitting algorithm:
//! 1. Recursively splitting the input based on available cores and input size
//! 2. Encoding chunks in parallel with overlapping regions
//! 3. Merging results by using left encoding up to midpoint and right encoding after
//! 4. Filtering and merging the results deterministically

use crate::tokenizer::{Encoding, Result};

use super::config::{ParallelConfig, MAX_RECURSION_DEPTH};
use super::debug_log;
use super::lazy_encoding::LazyEncoding;
use super::split_point::{
    calculate_overlap_size, find_char_boundary_forward, find_split_point_safe,
};

// =============================================================================
// DEPTH CALCULATION
// =============================================================================

/// Compute the optimal recursion depth based on input size and available cores.
///
/// Uses token-based calculation: each chunk should have at least `min_tokens_per_chunk`
/// tokens to justify the parallel overhead.
///
/// # Arguments
///
/// * `input_bytes` - Size of input in bytes
/// * `num_cores` - Number of available CPU cores
/// * `min_tokens_per_chunk` - Minimum tokens per chunk to justify parallelism
/// * `bytes_per_token` - Estimated bytes per token
///
/// # Returns
///
/// The optimal recursion depth (0 = no parallelism).
pub fn compute_optimal_depth(
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

// =============================================================================
// SERIAL FALLBACK HELPER
// =============================================================================

/// Encode serially and wrap in LazyEncoding with offset.
///
/// This is extracted as a helper to avoid repetition in fallback cases.
#[inline]
pub fn encode_serial_lazy<F>(
    encode_fn: &F,
    input: &str,
    global_offset: usize,
) -> Result<LazyEncoding>
where
    F: Fn(&str, bool) -> Result<Encoding> + Send + Sync,
{
    let encoding = encode_fn(input, false)?;
    let mut lazy = LazyEncoding::new(encoding);
    if global_offset > 0 {
        lazy.shift_offset(global_offset);
    }
    Ok(lazy)
}

// =============================================================================
// RECURSIVE ENCODING
// =============================================================================

/// Internal recursive encoding function using LazyEncoding for O(1) offset shifts.
///
/// Returns a LazyEncoding that defers offset materialization until the final merge.
/// This turns O(n × depth) offset operations into O(n) total.
///
/// # Algorithm
///
/// 1. Split input at midpoint with overlap on both sides
/// 2. Left chunk: [0, mid + overlap], Right chunk: [mid - overlap, end]
/// 3. Both chunks encode the overlap region [mid - overlap, mid + overlap]
/// 4. After encoding, use LEFT's tokens for positions < mid (split point)
/// 5. Use RIGHT's tokens for positions >= mid
/// 6. This ensures correct tokenization because both sides have enough context
pub fn encode_recursive_lazy<F>(
    encode_fn: &F,
    max_token_len: usize,
    input: &str,
    input_global_offset: usize,
    current_depth: usize,
    max_depth: usize,
    config: &ParallelConfig,
) -> Result<LazyEncoding>
where
    F: Fn(&str, bool) -> Result<Encoding> + Send + Sync,
{
    // Base case: reached max depth or input too small
    let estimated_tokens = (input.len() as f64 / config.bytes_per_token_estimate) as usize;
    let min_bytes_for_split =
        (config.min_tokens_per_chunk as f64 * config.bytes_per_token_estimate * 2.0) as usize;

    if current_depth >= max_depth
        || input.len() < min_bytes_for_split
        || estimated_tokens < config.min_tokens_per_chunk * 2
    {
        debug_log!(
            "[RECURSIVE depth={}] Base case: encoding {} bytes (est. {} tokens) at global offset {}",
            current_depth,
            input.len(),
            estimated_tokens,
            input_global_offset
        );

        return encode_serial_lazy(encode_fn, input, input_global_offset);
    }

    debug_log!(
        "[RECURSIVE depth={}] Splitting {} bytes (est. {} tokens) at global offset {}",
        current_depth,
        input.len(),
        estimated_tokens,
        input_global_offset
    );

    // Find split point near midpoint using SIMD-accelerated search
    let mid = input.len() / 2;
    let split_result = find_split_point_safe(input, mid, 500);
    let split_point = split_result.position;

    // Calculate overlap size - much smaller for safe boundaries
    let overlap_size =
        calculate_overlap_size(max_token_len, config.safety_margin, split_result.is_safe_boundary);

    if split_result.is_safe_boundary {
        debug_log!(
            "[RECURSIVE depth={}] Found safe boundary at {}, using minimal overlap ({})",
            current_depth,
            split_point,
            overlap_size
        );
    }

    // Create overlapping chunks
    let left_end = find_char_boundary_forward(input, (split_point + overlap_size).min(input.len()));
    let right_start = find_char_boundary_forward(input, split_point.saturating_sub(overlap_size));

    let left_chunk = &input[..left_end];
    let right_chunk = &input[right_start..];

    let global_split_point = input_global_offset + split_point;

    debug_log!(
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

    // Parallel recursive calls
    let (left_result, right_result) = rayon::join(
        || {
            encode_recursive_lazy(
                encode_fn,
                max_token_len,
                left_chunk,
                input_global_offset,
                current_depth + 1,
                max_depth,
                config,
            )
        },
        || {
            encode_recursive_lazy(
                encode_fn,
                max_token_len,
                right_chunk,
                input_global_offset + right_start,
                current_depth + 1,
                max_depth,
                config,
            )
        },
    );

    let mut left_lazy = left_result?;
    let mut right_lazy = right_result?;

    debug_log!(
        "[RECURSIVE depth={}] Before filtering - Left: {} tokens, Right: {} tokens",
        current_depth,
        left_lazy.len(),
        right_lazy.len()
    );

    // O(log n) count using binary search
    let left_tokens_expected = left_lazy.count_starting_before(global_split_point);
    let right_tokens_expected = right_lazy.count_starting_at_or_after(global_split_point);

    // O(log n) filter using binary search - just updates indices, no data copying
    left_lazy.filter_starting_before(global_split_point);
    right_lazy.filter_starting_at_or_after(global_split_point);

    debug_log!(
        "[RECURSIVE depth={}] After filtering - Left: {} tokens, Right: {} tokens",
        current_depth,
        left_lazy.len(),
        right_lazy.len()
    );

    // Safety check: tokens lost during filtering
    if (left_tokens_expected > 0 && left_lazy.is_empty())
        || (right_tokens_expected > 0 && right_lazy.is_empty())
    {
        debug_log!(
            "[RECURSIVE depth={}] WARNING: Tokens lost, falling back to serial",
            current_depth
        );
        return encode_serial_lazy(encode_fn, input, input_global_offset);
    }

    // Safety check: gap between left and right
    let left_max_end = left_lazy.last_offset().map(|(_, end)| end).unwrap_or(0);
    let right_min_start = right_lazy
        .first_offset()
        .map(|(start, _)| start)
        .unwrap_or(input_global_offset + input.len());

    let gap = right_min_start.saturating_sub(left_max_end);
    if gap > overlap_size {
        debug_log!(
            "[RECURSIVE depth={}] WARNING: Gap {} exceeds overlap {}, falling back to serial",
            current_depth,
            gap,
            overlap_size
        );
        return encode_serial_lazy(encode_fn, input, input_global_offset);
    }

    // Merge: materialize both and combine
    let result = left_lazy.merge_materialized(right_lazy);

    debug_log!(
        "[RECURSIVE depth={}] After merge: {} total tokens",
        current_depth,
        result.len()
    );

    // Validate for overlapping offsets
    let has_overlapping_offsets = result.get_offsets().windows(2).any(|w| {
        let (_start1, end1) = w[0];
        let (start2, _end2) = w[1];
        start2 < end1
    });

    if has_overlapping_offsets {
        debug_log!(
            "[RECURSIVE depth={}] WARNING: Overlapping offsets, falling back to serial",
            current_depth
        );
        return encode_serial_lazy(encode_fn, input, input_global_offset);
    }

    // Wrap the merged result in a new LazyEncoding (no offset shift needed)
    Ok(LazyEncoding::new(result))
}

/// Public wrapper that materializes the lazy encoding.
///
/// This is the entry point for recursive parallel encoding.
pub fn encode_recursive<F>(
    encode_fn: &F,
    max_token_len: usize,
    input: &str,
    input_global_offset: usize,
    current_depth: usize,
    max_depth: usize,
    config: &ParallelConfig,
) -> Result<Encoding>
where
    F: Fn(&str, bool) -> Result<Encoding> + Send + Sync,
{
    // Use lazy encoding internally, materialize at the end
    let lazy = encode_recursive_lazy(
        encode_fn,
        max_token_len,
        input,
        input_global_offset,
        current_depth,
        max_depth,
        config,
    )?;
    Ok(lazy.materialize())
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
}
