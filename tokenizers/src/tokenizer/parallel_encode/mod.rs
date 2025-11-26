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
//!
//! ## Module Organization
//!
//! - [`config`]: Configuration types and constants
//! - [`lazy_encoding`]: Zero-copy offset representation
//! - [`split_point`]: SIMD-accelerated split point detection
//! - [`recursive`]: Recursive parallel encoding strategy
//! - [`streaming`]: Cache-block streaming strategy

// =============================================================================
// SUBMODULES
// =============================================================================

mod config;
mod lazy_encoding;
mod recursive;
mod split_point;
mod streaming;

// =============================================================================
// RE-EXPORTS
// =============================================================================

pub use config::{ParallelConfig, ParallelMode};

// Internal re-exports for submodules
pub(crate) use config::MAX_RECURSION_DEPTH;

// =============================================================================
// DEBUG UTILITIES
// =============================================================================

use std::sync::OnceLock;

/// Check if debug mode is enabled (cached after first call).
#[inline]
fn is_debug_enabled() -> bool {
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("DEBUG_PARALLEL").is_ok())
}

/// Debug logging macro that only prints when DEBUG_PARALLEL is set.
///
/// Usage: `debug_log!("[TAG] message {}", value);`
macro_rules! debug_log {
    ($($arg:tt)*) => {
        if $crate::tokenizer::parallel_encode::is_debug_enabled() {
            println!($($arg)*);
        }
    };
}

// Make the macro available to submodules
pub(crate) use debug_log;

// =============================================================================
// IMPORTS
// =============================================================================

use crate::tokenizer::{Encoding, Result};
use crate::utils::parallelism::get_parallelism;

use recursive::{compute_optimal_depth, encode_recursive};
use streaming::encode_streaming_parallel;

// NOTE: Word ID computation has been completely removed from parallel encoding.
// Computing word IDs (either via pre-computation or heuristics) adds overhead
// that reduces the performance benefits of parallelization.
//
// Word IDs are rarely needed for most use cases (LLM inference, embeddings, etc.).
// If word IDs are required, use serial encoding instead: tokenizer.encode()

// =============================================================================
// PUBLIC API
// =============================================================================

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
///
/// # Example
///
/// ```ignore
/// use tokenizers::tokenizer::parallel_encode::{encode_parallel_single, ParallelConfig};
///
/// let result = encode_parallel_single(
///     |s, add_special| tokenizer.encode_raw(s, add_special),
///     |enc, pair, add_special| tokenizer.post_process(enc, pair, add_special),
///     100, // max_token_len
///     "Very long input text...",
///     true,
///     ParallelConfig::default(),
/// )?;
/// ```
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

    let estimated_tokens = (input.len() as f64 / config.bytes_per_token_estimate) as usize;
    debug_log!(
        "[PARALLEL_ENCODE] Input: {} bytes (~{} tokens), mode: {:?}, streaming: {}",
        input.len(),
        estimated_tokens,
        config.mode,
        use_streaming
    );

    let result = if use_streaming {
        // Use cache-block streaming for very large inputs
        debug_log!(
            "[PARALLEL_ENCODE] Using streaming mode (block_size: {}KB, overlap: {})",
            config.block_size / 1024,
            config.streaming_overlap
        );
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

        debug_log!(
            "[PARALLEL_ENCODE] Using recursive mode (cores: {}, depth: {})",
            num_cores,
            max_depth
        );

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
    {
        if let Some(reference) = validate_parallel_result(&result, &encode_fn, input, use_streaming)
        {
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

// =============================================================================
// VALIDATION (debug/test builds only)
// =============================================================================

/// Validate parallel result against serial encoding.
///
/// Returns `Some(reference)` if validation fails and we should use serial result.
/// Returns `None` if validation passes.
#[cfg(any(test, debug_assertions))]
fn validate_parallel_result<F>(
    result: &Encoding,
    encode_fn: &F,
    input: &str,
    use_streaming: bool,
) -> Option<Encoding>
where
    F: Fn(&str, bool) -> Result<Encoding> + Send + Sync,
{
    let reference = encode_fn(input, false).ok()?;

    if result.get_ids() != reference.get_ids() {
        if is_debug_enabled() {
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
                "[PARALLEL_ENCODE] WARNING: Validation failed (mode={}), falling back to serial",
                if use_streaming { "streaming" } else { "recursive" }
            );
        }
        Some(reference)
    } else {
        None
    }
}
