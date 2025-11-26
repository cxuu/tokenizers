//! Parallel encoding of a single long input, with output identical to serial encoding.
//!
//! The input is cut into segments that are encoded independently and concatenated. A cut is
//! only made at a byte `c` where `input[c]` is a space and `input[c - 1]` and `input[c + 1]`
//! are ASCII letters, and only when every stage of the pipeline can show that encoding
//! `input[..c]` and `input[c..]` separately gives the same splits and tokens as encoding
//! `input`:
//!
//! - [`Normalizer::map_cut_separator`]: normalizing `a + b` gives `normalize(a)` followed by
//!   [`Normalizer::normalize_continuation`]`(b)`.
//! - [`AddedVocabulary`]: no added token can match across the cut or change its extent
//!   because of what is on the other side.
//! - [`PreTokenizer::map_cut`]: splitting `a + b` gives the splits of `a` followed by the
//!   splits of `b`, where the last split of `a` and the first split of `b` may form a single
//!   split ([`Cut::Inside`]).
//! - [`Model::supports_cut`]: tokenizing is deterministic, and for [`Cut::Inside`] tokenizing
//!   `a + b` gives `tokenize(a) + tokenize(b)`.
//!
//! Segments are encoded as slices of the input, so their offsets are already global and any
//! stage that looks at offsets sees the same values as in serial encoding.
//!
//! Two things are checked at every cut, falling back to serial encoding when they fail:
//!
//! - The rules above assume that after normalization the cut is still between an ASCII letter
//!   and the separator, followed by an ASCII letter. Normalizers keep each of these chars as
//!   is, but can combine the letter after the separator with what follows it (NFC turns `e`
//!   and U+0301 into `é`).
//! - Alignments are the one state that transforming a string carries from char to char: some
//!   normalizers (e.g. `Precompiled` removing the first char of a string) leave every following
//!   alignment lagging. Alignments only lag, so if the letter before a cut is aligned with its
//!   original byte after pre-tokenization, nothing lags across the cut.
//!
//! When a stage can't show this, the input is pre-tokenized serially and only the model runs
//! in parallel over the splits, which is exact for any deterministic model.

use super::{
    AddedVocabulary, Decoder, Encoding, Model, NormalizedString, Normalizer, OffsetType,
    PostProcessor, PreTokenizedString, PreTokenizer, Result, TokenizerImpl,
};
use crate::utils::parallelism::*;

/// How a candidate cut looks to a pipeline stage. See the [module documentation](self).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cut {
    /// The cut is inside a split, right before `sep`. The split has an ASCII letter right
    /// before the cut and another one right after `sep`.
    Inside { sep: char },
    /// The cut is between two splits.
    Boundary,
}

/// Strategy used by [`TokenizerImpl::encode_parallel_with_config`] for a tokenizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelPlan {
    /// Cut the input into segments encoded in parallel. `inside` is true when a cut falls
    /// inside a split, which the model then has to support.
    Segments { inside: bool },
    /// Pre-tokenize serially, then run the model on the splits in parallel.
    Model,
    /// Encode serially.
    Serial,
}

#[derive(Clone, Debug)]
pub struct ParallelConfig {
    /// Inputs shorter than this many bytes are encoded serially.
    pub min_input_bytes: usize,
    /// Segments are at least this many bytes long.
    pub min_segment_bytes: usize,
    /// Number of segments to create per rayon thread.
    pub segments_per_thread: usize,
    /// Offsets to compute, as with `encode` (`Byte`), `encode_char_offsets` (`Char`) or
    /// `encode_fast` (`None`).
    pub offsets: OffsetType,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_input_bytes: 32 * 1024,
            min_segment_bytes: 8 * 1024,
            segments_per_thread: 4,
            offsets: OffsetType::Byte,
        }
    }
}

impl ParallelConfig {
    #[must_use]
    pub fn with_min_input_bytes(mut self, bytes: usize) -> Self {
        self.min_input_bytes = bytes;
        self
    }

    #[must_use]
    pub fn with_min_segment_bytes(mut self, bytes: usize) -> Self {
        self.min_segment_bytes = bytes;
        self
    }

    #[must_use]
    pub fn with_segments_per_thread(mut self, segments: usize) -> Self {
        self.segments_per_thread = segments;
        self
    }

    #[must_use]
    pub fn with_offsets(mut self, offsets: OffsetType) -> Self {
        self.offsets = offsets;
        self
    }
}

/// The plan, and what the space at a cut becomes after normalization.
fn plan<M: Model, N: Normalizer, PT: PreTokenizer>(
    model: &M,
    normalizer: Option<&N>,
    pre_tokenizer: Option<&PT>,
    added_vocabulary: &AddedVocabulary,
) -> (ParallelPlan, char) {
    if !model.supports_cut(Cut::Boundary) {
        return (ParallelPlan::Serial, ' ');
    }
    let segments = || -> Option<(bool, char)> {
        let sep = match normalizer {
            Some(normalizer) => normalizer.map_cut_separator(' ')?,
            None => ' ',
        };
        if sep.is_ascii_alphanumeric() || !added_vocabulary.supports_cut(sep) {
            return None;
        }
        let pre_tokenize = |cut| match pre_tokenizer {
            Some(pre_tokenizer) => pre_tokenizer.map_cut(cut),
            None => Some(cut),
        };
        // An added token absorbing the separator with `lstrip` makes the cut a boundary.
        if pre_tokenize(Cut::Boundary)? != Cut::Boundary {
            return None;
        }
        let inside = match pre_tokenize(Cut::Inside { sep })? {
            Cut::Boundary => false,
            cut => model.supports_cut(cut).then_some(true)?,
        };
        Some((inside, sep))
    };
    match segments() {
        Some((inside, sep)) => (ParallelPlan::Segments { inside }, sep),
        None => (ParallelPlan::Model, ' '),
    }
}

fn is_cut(bytes: &[u8], c: usize) -> bool {
    c > 0
        && c + 1 < bytes.len()
        && bytes[c] == b' '
        && bytes[c - 1].is_ascii_alphabetic()
        && bytes[c + 1].is_ascii_alphabetic()
}

/// Cut positions splitting `input` into roughly `segments` pieces of at least `min_segment` bytes.
fn find_cuts(input: &str, segments: usize, min_segment: usize) -> Vec<usize> {
    let bytes = input.as_bytes();
    let size = (bytes.len() / segments.max(1)).max(min_segment).max(1);
    let mut cuts = vec![];
    let mut target = size;
    while target + min_segment <= bytes.len() {
        let Some(cut) = memchr::memchr_iter(b' ', &bytes[target..])
            .map(|i| target + i)
            .find(|&c| is_cut(bytes, c))
        else {
            break;
        };
        if cut + min_segment > bytes.len() {
            break;
        }
        cuts.push(cut);
        target = cut + size;
    }
    cuts
}

struct Segment {
    encoding: Encoding,
    /// Whether, after normalization, the text right after the start of the segment is the
    /// separator then an ASCII letter, and the text right before its end an ASCII letter, all
    /// aligned with the original text. The rules in the module documentation assume it.
    starts_with_cut: bool,
    ends_with_cut: bool,
    splits: usize,
    /// Whether the first split is text starting at the segment start.
    text_at_start: bool,
    /// Whether the last split is text ending at the segment end.
    text_at_end: bool,
}

/// Byte to char offset conversion with the same results as `BytesToCharOffsetConverter`.
struct CharOffsets(Vec<usize>);

impl CharOffsets {
    fn new(sequence: &str) -> Self {
        let mut map = Vec::with_capacity(sequence.len());
        for (i, c) in sequence.chars().enumerate() {
            map.extend(std::iter::repeat_n(i, c.len_utf8()));
        }
        Self(map)
    }

    fn convert(&self, offsets: (usize, usize)) -> Option<(usize, usize)> {
        match (self.0.get(offsets.0), self.0.get(offsets.1)) {
            (Some(start), Some(end)) => Some((*start, *end)),
            (Some(start), None) => {
                let last = self.0.get(offsets.1 - 1).copied().unwrap_or(start + 1);
                Some((*start, last + 1))
            }
            _ => None,
        }
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: Model + Send + Sync,
    N: Normalizer + Send + Sync,
    PT: PreTokenizer + Send + Sync,
    PP: PostProcessor + Send + Sync,
    D: Decoder + Send + Sync,
{
    /// The strategy `encode_parallel_single` uses for long inputs with this tokenizer.
    pub fn parallel_plan(&self) -> ParallelPlan {
        self.parallel_plan_and_separator().0
    }

    fn parallel_plan_and_separator(&self) -> (ParallelPlan, char) {
        plan(
            &self.model,
            self.normalizer.as_ref(),
            self.pre_tokenizer.as_ref(),
            &self.added_vocabulary,
        )
    }

    /// Encode a single sequence using multiple threads. The result is identical to `encode`.
    pub fn encode_parallel_single(
        &self,
        input: impl AsRef<str>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        self.encode_parallel_with_config(input, add_special_tokens, &ParallelConfig::default())
    }

    /// Encode a single sequence using multiple threads, with offsets relative to chars. The
    /// result is identical to `encode_char_offsets`.
    pub fn encode_parallel_single_char_offsets(
        &self,
        input: impl AsRef<str>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let config = ParallelConfig::default().with_offsets(OffsetType::Char);
        self.encode_parallel_with_config(input, add_special_tokens, &config)
    }

    /// Encode a single sequence using multiple threads. The result is identical to `encode`,
    /// `encode_char_offsets` or `encode_fast`, depending on `config.offsets`.
    pub fn encode_parallel_with_config(
        &self,
        input: impl AsRef<str>,
        add_special_tokens: bool,
        config: &ParallelConfig,
    ) -> Result<Encoding> {
        let input = input.as_ref();
        let serial = || {
            let encoding = self.encode_single_sequence(input.into(), 0, config.offsets)?;
            self.post_process(encoding, None, add_special_tokens)
        };
        if input.len() < config.min_input_bytes || !get_parallelism() {
            return serial();
        }
        let (plan, sep) = self.parallel_plan_and_separator();
        let encoding = match plan {
            ParallelPlan::Serial => return serial(),
            ParallelPlan::Model => self.encode_splits_parallel(input, config.offsets)?,
            ParallelPlan::Segments { inside } => {
                let segments = rayon::current_num_threads() * config.segments_per_thread;
                let cuts = find_cuts(input, segments, config.min_segment_bytes);
                if cuts.is_empty() {
                    self.encode_splits_parallel(input, config.offsets)?
                } else {
                    match self.encode_segments(input, &cuts, inside, sep, config.offsets) {
                        Ok(encoding) => encoding,
                        // Let serial encoding report the same error it always would.
                        Err(_) => return serial(),
                    }
                }
            }
        };
        self.post_process(encoding, None, add_special_tokens)
    }

    fn encode_splits_parallel(&self, input: &str, offsets: OffsetType) -> Result<Encoding> {
        let normalized = self
            .added_vocabulary
            .extract_and_normalize(self.normalizer.as_ref(), input);
        let mut pre_tokenized = self.do_pre_tokenize(normalized)?;
        pre_tokenized.tokenize_parallel(|normalized| self.model.tokenize(normalized.get()))?;
        pre_tokenized.into_encoding(None, 0, offsets)
    }

    fn encode_segment(
        &self,
        input: &str,
        start: usize,
        end: usize,
        sep: char,
        offsets: OffsetType,
    ) -> Result<Segment> {
        let sequence = NormalizedString::from(&input[start..end]).with_original_shift(start);
        let (normalized, starts_with_cut) = self.added_vocabulary.extract_and_normalize_string(
            self.normalizer.as_ref(),
            sequence,
            (start > 0).then_some(sep),
        );
        let mut pre_tokenized: PreTokenizedString = self.do_pre_tokenize(normalized)?;
        let ends_with_cut = pre_tokenized.ends_with_cut(end);
        let (splits, text_at_start, text_at_end) = pre_tokenized.edges(start, end);
        pre_tokenized.tokenize(|normalized| self.model.tokenize(normalized.get()))?;
        Ok(Segment {
            encoding: pre_tokenized.into_encoding(None, 0, offsets)?,
            starts_with_cut,
            ends_with_cut,
            splits,
            text_at_start,
            text_at_end,
        })
    }

    fn encode_segments(
        &self,
        input: &str,
        cuts: &[usize],
        inside: bool,
        sep: char,
        offsets: OffsetType,
    ) -> Result<Encoding> {
        let segment_offsets = match offsets {
            OffsetType::None => OffsetType::None,
            OffsetType::Byte | OffsetType::Char => OffsetType::Byte,
        };
        let bounds: Vec<(usize, usize)> = std::iter::once(0)
            .chain(cuts.iter().copied())
            .zip(cuts.iter().copied().chain(std::iter::once(input.len())))
            .collect();
        let segments = bounds
            .into_maybe_par_iter()
            .map(|(start, end)| self.encode_segment(input, start, end, sep, segment_offsets))
            .collect::<Result<Vec<_>>>()?;
        let last = segments.len() - 1;
        if segments[..last].iter().any(|s| !s.ends_with_cut)
            || segments[1..].iter().any(|s| !s.starts_with_cut)
        {
            return Err("normalized text around a cut isn't as expected".into());
        }

        let len = segments.iter().map(|s| s.encoding.len()).sum();
        let mut ids = Vec::with_capacity(len);
        let mut tokens = Vec::with_capacity(len);
        let mut words = Vec::with_capacity(len);
        let mut offsets_out = Vec::with_capacity(len);
        let mut word_base = 0u32;
        let mut previous: Option<(usize, bool)> = None;
        for segment in segments {
            if segment.splits == 0 {
                return Err("parallel encoding produced an empty segment".into());
            }
            if let Some((splits, text_at_end)) = previous {
                // With `inside`, a split crossing the cut is one word in serial encoding.
                let joined = inside && text_at_end && segment.text_at_start;
                word_base += splits as u32 - u32::from(joined);
            }
            previous = Some((segment.splits, segment.text_at_end));
            let (segment_ids, segment_tokens, segment_words, segment_offsets) =
                segment.encoding.into_token_parts();
            ids.extend(segment_ids);
            tokens.extend(segment_tokens);
            words.extend(segment_words.into_iter().map(|w| w.map(|w| w + word_base)));
            offsets_out.extend(segment_offsets);
        }
        if offsets == OffsetType::Char {
            let chars = CharOffsets::new(input);
            offsets_out
                .maybe_par_iter_mut()
                .for_each(|o| *o = chars.convert(*o).unwrap_or(*o));
        }
        Ok(Encoding::new(
            ids,
            vec![0; len],
            tokens,
            words,
            offsets_out,
            vec![0; len],
            vec![1; len],
            vec![],
            Default::default(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::pre_tokenizer::BytesToCharOffsetConverter;

    #[test]
    fn cuts_are_spaces_between_ascii_letters() {
        let input = "ab cd 1 ef  gh é ij\nkl mn";
        let cuts = find_cuts(input, 100, 1);
        assert!(!cuts.is_empty());
        for c in cuts {
            assert!(is_cut(input.as_bytes(), c), "{}", c);
        }
        assert_eq!(find_cuts("abc def", 1, 4), Vec::<usize>::new());
        assert_eq!(find_cuts("", 4, 1), Vec::<usize>::new());
    }

    #[test]
    fn char_offsets_match_serial_converter() {
        let inputs = [
            "",
            "a",
            "héllo wörld",
            "日本語 text 🎉👍🏽",
            "\u{0301}e\u{0301}",
        ];
        for input in inputs {
            let reference = BytesToCharOffsetConverter::new(input);
            let ours = CharOffsets::new(input);
            for start in 0..=input.len() + 2 {
                for end in start..=input.len() + 2 {
                    if end == 0 {
                        continue;
                    }
                    assert_eq!(ours.convert((start, end)), reference.convert((start, end)));
                }
            }
        }
    }
}
