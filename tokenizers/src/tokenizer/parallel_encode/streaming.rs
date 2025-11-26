//! Cache-block streaming encoding strategy.
//!
//! Optimized for cache locality on very large inputs (4MB+):
//! 1. Process input in L2-cache-sized blocks (~128KB) sequentially
//! 2. Use small overlap windows (~1KB) to handle token boundaries
//! 3. Stream results directly to output, avoiding cache thrashing
//! 4. Each block fits entirely in L2 cache with vocabulary lookups

use crate::tokenizer::{Encoding, Result};
use rayon::prelude::*;

use super::config::ParallelConfig;
use super::debug_log;
use super::split_point::find_char_boundary_forward;

// =============================================================================
// BLOCK BOUNDARY CALCULATION
// =============================================================================

/// Describes the boundaries of a single block for processing.
///
/// Each block has:
/// - `start_offset`: where to start encoding (includes left overlap)
/// - `logical_start`: where this block's tokens should start (filter boundary)
/// - `logical_end`: where this block's tokens should end (filter boundary)
/// - `end_offset`: where to end encoding (includes right overlap)
#[derive(Debug, Clone, Copy)]
pub struct BlockBoundaries {
    /// Where to start encoding (includes left overlap for context)
    pub start_offset: usize,
    /// Where this block's tokens should start (filter boundary)
    pub logical_start: usize,
    /// Where this block's tokens should end (filter boundary)
    pub logical_end: usize,
    /// Where to end encoding (includes right overlap for context)
    pub end_offset: usize,
}

/// Calculate block boundaries for streaming encoding.
///
/// This is a shared utility used by both sequential and parallel streaming modes.
///
/// # Arguments
///
/// * `input` - The input string to process
/// * `block_size` - Size of each logical block in bytes
/// * `overlap` - Overlap size in bytes for handling token boundaries
///
/// # Returns
///
/// A vector of `BlockBoundaries` describing each block to process.
pub fn calculate_block_boundaries(
    input: &str,
    block_size: usize,
    overlap: usize,
) -> Vec<BlockBoundaries> {
    let mut blocks = Vec::new();
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

        blocks.push(BlockBoundaries {
            start_offset,
            logical_start: logical_pos,
            logical_end,
            end_offset,
        });

        logical_pos = logical_end;
    }

    blocks
}

// =============================================================================
// SEQUENTIAL STREAMING
// =============================================================================

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
pub fn encode_streaming<F>(
    encode_fn: &F,
    max_token_len: usize,
    input: &str,
    config: &ParallelConfig,
) -> Result<Encoding>
where
    F: Fn(&str, bool) -> Result<Encoding> + Send + Sync,
{
    let block_size = config.block_size;
    let overlap = config.streaming_overlap.max(max_token_len * 3);

    debug_log!(
        "[STREAMING] Input: {} bytes, Block size: {} bytes, Overlap: {} bytes",
        input.len(),
        block_size,
        overlap
    );

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

        debug_log!(
            "[STREAMING] Block {}: offset={}, block_end={}, process_end={}, block_len={}",
            block_count,
            offset,
            block_end,
            process_end,
            block.len()
        );

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

            debug_log!(
                "[STREAMING] Block {}: Kept {} tokens (of {} before filter)",
                block_count,
                block_encoding.len(),
                tokens_before
            );
        }

        // Stream tokens to result
        if result.is_empty() {
            result = block_encoding;
        } else {
            result.merge_with(block_encoding, false);
        }

        debug_log!(
            "[STREAMING] Block {}: Result now has {} tokens",
            block_count,
            result.len()
        );

        // Move to next block
        offset = block_end;
    }

    debug_log!(
        "[STREAMING] Complete: {} blocks, {} total tokens",
        block_count,
        result.len()
    );

    Ok(result)
}

// =============================================================================
// PARALLEL STREAMING
// =============================================================================

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
pub fn encode_streaming_parallel<F>(
    encode_fn: &F,
    max_token_len: usize,
    input: &str,
    config: &ParallelConfig,
) -> Result<Encoding>
where
    F: Fn(&str, bool) -> Result<Encoding> + Send + Sync,
{
    let block_size = config.block_size;
    let overlap = config.streaming_overlap.max(max_token_len * 3);

    // Calculate all block boundaries using the shared utility
    let blocks = calculate_block_boundaries(input, block_size, overlap);

    debug_log!(
        "[STREAMING_PARALLEL] Input: {} bytes, {} blocks",
        input.len(),
        blocks.len()
    );

    // Encode all blocks in parallel
    // Each block encodes with overlap but only keeps tokens in its logical range
    let block_results: Vec<Result<(usize, usize, Encoding)>> = blocks
        .into_par_iter()
        .map(|boundaries| {
            let block = &input[boundaries.start_offset..boundaries.end_offset];
            let mut block_encoding = encode_fn(block, false)?;

            // Shift offsets to global coordinates
            if boundaries.start_offset > 0 {
                block_encoding.shift_all_offsets(boundaries.start_offset as isize);
            }

            Ok((
                boundaries.logical_start,
                boundaries.logical_end,
                block_encoding,
            ))
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

        debug_log!(
            "[STREAMING_PARALLEL] Block {}: logical=[{}, {}), kept {} tokens",
            idx,
            logical_start,
            logical_end,
            block_encoding.len()
        );

        // Merge into result
        if result.is_empty() {
            result = block_encoding;
        } else {
            result.merge_with(block_encoding, false);
        }
    }

    debug_log!(
        "[STREAMING_PARALLEL] Complete: {} total tokens",
        result.len()
    );

    Ok(result)
}
