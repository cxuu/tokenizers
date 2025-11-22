//! Split point detection with SIMD-accelerated search.
//!
//! This module provides utilities for finding optimal split points in text
//! for parallel tokenization, including UTF-8 boundary handling.

use memchr::memchr_iter;

// =============================================================================
// UTF-8 BOUNDARY UTILITIES
// =============================================================================

/// Find a UTF-8 character boundary at or after the given byte position.
pub fn find_char_boundary_forward(s: &str, pos: usize) -> usize {
    let mut pos = pos.min(s.len());
    while pos < s.len() && !s.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

/// Find a UTF-8 character boundary at or before the given byte position.
#[allow(dead_code)]
pub fn find_char_boundary_backward(s: &str, pos: usize) -> usize {
    let mut pos = pos.min(s.len());
    while pos > 0 && !s.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

// =============================================================================
// SPLIT POINT DETECTION
// =============================================================================

/// Result of finding a split point, including whether it's a "safe" boundary.
#[derive(Debug, Clone, Copy)]
pub struct SplitPoint {
    /// The byte position of the split point
    pub position: usize,
    /// If true, this is a safe boundary (sentence/paragraph end) that needs minimal overlap
    pub is_safe_boundary: bool,
}

/// Find a good split point near the target position using SIMD-accelerated search.
///
/// Prefers safe boundaries (sentence/paragraph ends) which need minimal overlap,
/// falls back to whitespace boundaries.
///
/// # Arguments
///
/// * `input` - The input string to search
/// * `target` - The target byte position for the split
/// * `search_window` - How many bytes to search in each direction
///
/// # Returns
///
/// A `SplitPoint` with the position and whether it's a safe boundary.
pub fn find_split_point_safe(input: &str, target: usize, search_window: usize) -> SplitPoint {
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
    for &(punct, follow) in &[
        (b'.', b' '),
        (b'.', b'\n'),
        (b'!', b' '),
        (b'!', b'\n'),
        (b'?', b' '),
        (b'?', b'\n'),
    ] {
        if let Some(pos) =
            find_punct_boundary_near(bytes, target, search_start, search_end, punct, follow)
        {
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

/// Legacy wrapper for compatibility - returns just the position.
#[allow(dead_code)]
pub fn find_split_point(input: &str, target: usize, search_window: usize) -> usize {
    find_split_point_safe(input, target, search_window).position
}

// =============================================================================
// SIMD SEARCH HELPERS
// =============================================================================

/// SIMD-accelerated search for a 2-byte pattern near target position.
#[inline]
fn find_pattern_near(
    bytes: &[u8],
    target: usize,
    start: usize,
    end: usize,
    pattern: &[u8],
) -> Option<usize> {
    if pattern.is_empty() || end <= start {
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
fn find_punct_boundary_near(
    bytes: &[u8],
    target: usize,
    start: usize,
    end: usize,
    punct: u8,
    follow: u8,
) -> Option<usize> {
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

// =============================================================================
// OVERLAP CALCULATION
// =============================================================================

/// Calculate the overlap size based on max token length and whether we're at a safe boundary.
///
/// For safe boundaries (sentence/paragraph ends), we use minimal overlap since tokenization
/// is guaranteed to be consistent. For other boundaries, we use larger overlap.
///
/// # Arguments
///
/// * `max_token_len` - Maximum token length in bytes from the vocabulary
/// * `safety_margin` - Additional safety margin in bytes
/// * `is_safe_boundary` - Whether the split is at a safe boundary (sentence/paragraph end)
///
/// # Returns
///
/// The overlap size in bytes.
pub fn calculate_overlap_size(
    max_token_len: usize,
    safety_margin: usize,
    is_safe_boundary: bool,
) -> usize {
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

// =============================================================================
// TESTS
// =============================================================================

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
