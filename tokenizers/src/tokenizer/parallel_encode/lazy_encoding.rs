//! Zero-copy offset representation for parallel encoding.
//!
//! LazyEncoding wraps an Encoding with deferred offset operations:
//! - `base_offset`: Added to all offsets on materialization (shift is O(1))
//! - `start_idx` / `end_idx`: Virtual range for filtering (no copies until merge)
//!
//! This turns O(n × depth) offset operations into O(n) at final materialization.

use crate::tokenizer::Encoding;

/// A lazy wrapper around Encoding that defers offset shifting and filtering.
///
/// Instead of immediately modifying the encoding's offsets (O(n) per operation),
/// this stores metadata that's applied only during final materialization.
#[derive(Debug)]
pub struct LazyEncoding {
    encoding: Encoding,
    /// Offset to add to all positions (applied on materialization)
    base_offset: usize,
    /// Start index of the valid token range (inclusive)
    start_idx: usize,
    /// End index of the valid token range (exclusive)
    end_idx: usize,
}

impl LazyEncoding {
    /// Wrap an encoding in a lazy container.
    pub fn new(encoding: Encoding) -> Self {
        let len = encoding.len();
        Self {
            encoding,
            base_offset: 0,
            start_idx: 0,
            end_idx: len,
        }
    }

    /// O(1) offset shift - just update the base offset.
    #[inline]
    pub fn shift_offset(&mut self, delta: usize) {
        self.base_offset += delta;
    }

    /// Number of tokens in the current virtual range.
    #[inline]
    pub fn len(&self) -> usize {
        self.end_idx.saturating_sub(self.start_idx)
    }

    /// Check if the virtual range is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start_idx >= self.end_idx
    }

    /// Get offset at index (applying base_offset).
    #[inline]
    pub fn get_offset(&self, idx: usize) -> (usize, usize) {
        let actual_idx = self.start_idx + idx;
        let (start, end) = self.encoding.get_offsets()[actual_idx];
        (start + self.base_offset, end + self.base_offset)
    }

    /// Get the first offset in the virtual range (with base_offset applied).
    pub fn first_offset(&self) -> Option<(usize, usize)> {
        if self.is_empty() {
            None
        } else {
            Some(self.get_offset(0))
        }
    }

    /// Get the last offset in the virtual range (with base_offset applied).
    pub fn last_offset(&self) -> Option<(usize, usize)> {
        if self.is_empty() {
            None
        } else {
            Some(self.get_offset(self.len() - 1))
        }
    }

    /// Count tokens starting before position (O(log n) binary search).
    pub fn count_starting_before(&self, position: usize) -> usize {
        let offsets = self.encoding.get_offsets();
        let range = &offsets[self.start_idx..self.end_idx];

        // Binary search for first token starting at or after position
        let adjusted_pos = position.saturating_sub(self.base_offset);
        range.partition_point(|(start, _)| *start < adjusted_pos)
    }

    /// Count tokens starting at or after position (O(log n) binary search).
    #[allow(dead_code)]
    pub fn count_starting_at_or_after(&self, position: usize) -> usize {
        self.len() - self.count_starting_before(position)
    }

    /// Filter to keep only tokens starting before position (O(log n)).
    /// Uses binary search to find the cut point, no data copying.
    pub fn filter_starting_before(&mut self, position: usize) {
        let offsets = self.encoding.get_offsets();
        let range = &offsets[self.start_idx..self.end_idx];

        // Binary search: find first token with start >= position (adjusted for base)
        let adjusted_pos = position.saturating_sub(self.base_offset);
        let cut_idx = range.partition_point(|(start, _)| *start < adjusted_pos);

        self.end_idx = self.start_idx + cut_idx;
    }

    /// Filter to keep only tokens starting at or after position (O(log n)).
    /// Uses binary search to find the cut point, no data copying.
    pub fn filter_starting_at_or_after(&mut self, position: usize) {
        let offsets = self.encoding.get_offsets();
        let range = &offsets[self.start_idx..self.end_idx];

        // Binary search: find first token with start >= position (adjusted for base)
        let adjusted_pos = position.saturating_sub(self.base_offset);
        let cut_idx = range.partition_point(|(start, _)| *start < adjusted_pos);

        self.start_idx += cut_idx;
    }

    /// Materialize this lazy encoding into a concrete Encoding.
    /// This applies the base offset and extracts the valid range.
    /// Only called once at the end of parallel encoding.
    pub fn materialize(self) -> Encoding {
        // Fast path: no modifications needed
        if self.base_offset == 0 && self.start_idx == 0 && self.end_idx == self.encoding.len() {
            return self.encoding;
        }

        // Extract the range and apply offset in one pass
        let ids: Vec<u32> = self.encoding.get_ids()[self.start_idx..self.end_idx].to_vec();
        let tokens: Vec<String> = self.encoding.get_tokens()[self.start_idx..self.end_idx].to_vec();
        let type_ids: Vec<u32> =
            self.encoding.get_type_ids()[self.start_idx..self.end_idx].to_vec();
        let words: Vec<Option<u32>> =
            self.encoding.get_word_ids()[self.start_idx..self.end_idx].to_vec();
        let special_mask: Vec<u32> = self.encoding.get_special_tokens_mask()[self.start_idx..self.end_idx].to_vec();
        let attention_mask: Vec<u32> =
            self.encoding.get_attention_mask()[self.start_idx..self.end_idx].to_vec();

        let offsets: Vec<(usize, usize)> = self.encoding.get_offsets()[self.start_idx..self.end_idx]
            .iter()
            .map(|(s, e)| (s + self.base_offset, e + self.base_offset))
            .collect();

        Encoding::new(
            ids,
            type_ids,
            tokens,
            words,
            offsets,
            special_mask,
            attention_mask,
            vec![],             // Overflowing not used in parallel encoding
            Default::default(), // Sequence ranges rebuilt if needed
        )
    }

    /// Merge two lazy encodings by materializing and combining.
    pub fn merge_materialized(self, other: LazyEncoding) -> Encoding {
        let mut left = self.materialize();
        let right = other.materialize();
        left.merge_with(right, false);
        left
    }
}
