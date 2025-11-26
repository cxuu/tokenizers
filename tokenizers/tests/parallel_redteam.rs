//! Red-team differential tests for `encode_parallel_with_config`.
//!
//! Every tokenizer here is built programmatically from small hand-made vocabularies, covering
//! every normalizer, pre-tokenizer, model and added-token flag, alone and combined. Each check
//! cuts the input at every eligible space and compares against the matching serial method for
//! every offset type, with and without special tokens. Mismatching inputs are shrunk before
//! being reported.
//!
//! Environment: `REDTEAM_CASES` (random inputs per `Segments` pipeline, default 25),
//! `REDTEAM_PIPELINES` (random pipelines, default 400), `REDTEAM_REPORT_PANICS` (print inputs
//! that make serial encoding panic; a panic on both sides with the same message counts as equal).
//!
//! Tests named `bug_*` reproduce mismatches found by this red team: they fail until the
//! corresponding rule is fixed.

use ahash::AHashMap;
use std::collections::BTreeSet;
use tokenizers::models::bpe::BPE;
use tokenizers::models::unigram::Unigram;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::replace::ReplacePattern;
use tokenizers::normalizers::{
    BertNormalizer, ByteLevel as ByteLevelNormalizer, Lowercase, Nmt, NormalizerWrapper, Prepend,
    Replace, Sequence as NormalizerSequence, Strip, StripAccents, NFC, NFD, NFKC, NFKD,
};
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::delimiter::CharDelimiterSplit;
use tokenizers::pre_tokenizers::digits::Digits;
use tokenizers::pre_tokenizers::fixed_length::FixedLength;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::pre_tokenizers::punctuation::Punctuation;
use tokenizers::pre_tokenizers::sequence::Sequence as PreTokenizerSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::unicode_scripts::UnicodeScripts;
use tokenizers::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::processors::roberta::RobertaProcessing;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::tokenizer::{
    AddedToken, OffsetType, ParallelConfig, ParallelPlan, SplitDelimiterBehavior, Tokenizer,
};
use tokenizers::utils::padding::{PaddingParams, PaddingStrategy};
use tokenizers::utils::truncation::{TruncationParams, TruncationStrategy};
use tokenizers::{Encoding, Result};

use SplitDelimiterBehavior::{Contiguous, Isolated, MergedWithNext, MergedWithPrevious, Removed};

const SEGMENTS_FALSE: ParallelPlan = ParallelPlan::Segments { inside: false };
const SEGMENTS_TRUE: ParallelPlan = ParallelPlan::Segments { inside: true };
const BEHAVIORS: [SplitDelimiterBehavior; 5] = [
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
];

// ---------------------------------------------------------------------------------------------
// Comparison

fn config(offsets: OffsetType) -> ParallelConfig {
    ParallelConfig::default()
        .with_min_input_bytes(0)
        .with_min_segment_bytes(1)
        .with_segments_per_thread(1 << 20)
        .with_offsets(offsets)
}

fn serial(
    tokenizer: &Tokenizer,
    text: &str,
    add_special: bool,
    offsets: OffsetType,
) -> Result<Encoding> {
    match offsets {
        OffsetType::Byte => tokenizer.encode(text, add_special),
        OffsetType::Char => tokenizer.encode_char_offsets(text, add_special),
        OffsetType::None => tokenizer.encode_fast(text, add_special),
    }
}

fn show(encoding: &Encoding) -> String {
    encoding
        .get_tokens()
        .iter()
        .zip(encoding.get_offsets())
        .zip(encoding.get_word_ids())
        .zip(encoding.get_ids())
        .map(|(((token, offsets), word), id)| match word {
            Some(word) => format!("{token:?}#{id}@{}..{}w{word}", offsets.0, offsets.1),
            None => format!("{token:?}#{id}@{}..{}w-", offsets.0, offsets.1),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn show_result(result: &Result<Encoding>) -> String {
    match result {
        Ok(encoding) => {
            let mut s = show(encoding);
            if !encoding.get_overflowing().is_empty() {
                s += &format!(" (+{} overflowing)", encoding.get_overflowing().len());
            }
            s
        }
        Err(e) => format!("Err({e})"),
    }
}

/// Runs `f`, turning a panic into an error carrying its message.
fn catch(f: impl FnOnce() -> Result<Encoding>) -> Result<Encoding> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|panic| {
        let message = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        Err(format!("panic: {message}").into())
    })
}

/// The first configuration for which parallel and serial encoding disagree, described.
fn mismatch(tokenizer: &Tokenizer, text: &str) -> Option<String> {
    for offsets in [OffsetType::Byte, OffsetType::Char, OffsetType::None] {
        for add_special in [false, true] {
            let expected = catch(|| serial(tokenizer, text, add_special, offsets));
            let actual = catch(|| {
                tokenizer.encode_parallel_with_config(text, add_special, &config(offsets))
            });
            let same = match (&expected, &actual) {
                (Ok(e), Ok(a)) => e == a,
                (Err(e), Err(a)) => e.to_string() == a.to_string(),
                _ => false,
            };
            if !same {
                return Some(format!(
                    "offsets={offsets:?} add_special={add_special} input={text:?}\n    serial:   {}\n    parallel: {}",
                    show_result(&expected),
                    show_result(&actual)
                ));
            }
        }
    }
    None
}

/// Shrinks `text` while it still triggers a mismatch.
fn minimize(tokenizer: &Tokenizer, text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    let mut size = chars.len().div_ceil(2).max(1);
    loop {
        let mut changed = false;
        let mut i = 0;
        while i + size <= chars.len() {
            let candidate: String = chars[..i].iter().chain(&chars[i + size..]).collect();
            if !candidate.is_empty() && mismatch(tokenizer, &candidate).is_some() {
                chars = candidate.chars().collect();
                changed = true;
            } else {
                i += size;
            }
        }
        if size == 1 && !changed {
            break;
        }
        if !changed {
            size = (size / 2).max(1);
        }
    }
    chars.into_iter().collect()
}

#[track_caller]
fn assert_identical(tokenizer: &Tokenizer, text: &str) {
    if let Some(m) = mismatch(tokenizer, text) {
        panic!("parallel and serial encodings differ: {}", m);
    }
}

// ---------------------------------------------------------------------------------------------
// Inputs

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

/// Words that start and end with an ASCII letter, so that single spaces between them are cuts.
const WORDS: &[&str] = &[
    "a",
    "b",
    "x",
    "y",
    "z",
    "ab",
    "cd",
    "xy",
    "the",
    "The",
    "THE",
    "hello",
    "World",
    "abc",
    "and",
    "ing",
    "e\u{301}",
    "e\u{301}t\u{301}e",
    "a\u{308}b",
    "y\u{301}z",
    "w\u{301}",
    "ca\u{301}fe",
    "nai\u{308}ve",
    "don't",
    "it's",
    "ab1c",
    "aİb",
    "aßb",
    "aﬁb",
    "aＡb",
    "a日b",
    "a語b",
    "a👍b",
    "a\u{200d}b",
    "a\u{600}b",
    "a\u{0}b",
    "a_b",
    "a▁b",
    "aĠb",
    "x\u{301}",
    "e\u{323}",
    "aé!b",
    "x-y",
    "xx-y",
    "a.b",
    "a\u{3000}b",
    "a\u{85}b",
    "aΣb",
    "aσb",
];

const OTHERS: &[&str] = &[
    "日本",
    "語",
    "👍🏽",
    "👨‍👩‍👧",
    "🇫🇷",
    "1234",
    "42",
    "7",
    ".",
    ",",
    "!",
    "?",
    "...",
    "``",
    "''",
    "\"",
    "(",
    ")",
    "-",
    "_",
    "▁",
    "Ġ",
    "\u{a0}",
    "\u{3000}",
    "\u{200b}",
    "\u{feff}",
    "\u{fffd}",
    "\u{85}",
    "\u{2028}",
    "\u{301}",
    "é",
    "İ",
    "ß",
    "Σ",
    "[CLS]",
    "<mask>",
    "<|endoftext|>",
    "</s>",
    "<pad>",
    "-x-",
    "x-",
    "..",
    "--",
    "==",
    "@@",
    "y é",
    " é",
    "\u{0}",
    "\u{7f}",
    "\r\n",
    "\n\n",
    "    ",
    "x- ",
    " x-",
    "'s",
];

const SEPARATORS: &[&str] = &[
    "  ", "\t", "\n", "\r\n", " \u{301}", "\u{a0}", "\u{3000}", "", "-", ".", ", ", " ▁", "▁",
    " - ", " . ", "   ",
];

fn random_text(rng: &mut Rng) -> String {
    let n = 1 + rng.below(40);
    let mut text = String::new();
    for i in 0..n {
        if i > 0 {
            text.push_str(if rng.chance(75) {
                " "
            } else {
                rng.pick(SEPARATORS)
            });
        }
        text.push_str(if rng.chance(80) {
            rng.pick(WORDS)
        } else {
            rng.pick(OTHERS)
        });
    }
    if rng.chance(10) {
        text.insert(0, ' ');
    }
    if rng.chance(10) {
        text.push(' ');
    }
    text
}

fn random_texts(cases: usize, seed: u64) -> Vec<String> {
    let mut rng = Rng(seed | 1);
    (0..cases).map(|_| random_text(&mut rng)).collect()
}

fn adversarial_texts() -> Vec<String> {
    let texts = [
        "a b",
        "ab cd ef gh",
        "x y z w v u t",
        " x y ",
        "the the the the",
        "hello World THE end",
        "xy e\u{301}z",
        "xy e\u{301} z",
        "xy e\u{301}",
        "a y\u{301} z",
        "x w\u{301} y",
        "x e\u{323} y",
        "ab  cd\tef\ngh\r\nij",
        "ab ▁cd▁ ef ▁ gh",
        "ab Ġcd Ġ ef",
        "a <mask> b <mask>c d<mask> e",
        "x [CLS] y [CLS]y x[CLS] y",
        "x -x- y x-x-y -x- z",
        "ab x- cd xx- ef x-y",
        "ab x-cd",
        "a  x- b",
        "a 1 b 12 c 345 d 6789 e",
        "a , b . c ! d ? e",
        "a. b .c a.b",
        "ab ''cd'' ef ``gh`` ij",
        "don't stop it's x",
        "a .. b -- c == d @@ e",
        "a ..b c-- d ==e f@@ g",
        "a 日本 b 語 c 한 d",
        "a 👍🏽 b 👨‍👩‍👧 c 🇫🇷 d",
        "a\u{a0}b c\u{3000}d e\u{200b}f g",
        "a\u{0}b c\u{7f}d e\u{85}f g\u{fffd}h i\u{feff}j",
        "x \u{600}y z",
        "aİb cßd eﬁf gＡh",
        "aΣ bΣ cσ",
        "<|endoftext|> a b <|endoftext|>c d",
        "a </s> b </s>",
        "x <pad> y <pad>z",
        "a y é b",
        "ab y é cd",
        "abc é d",
        "x  y   z    w",
        "x\ty\nz",
        "end with space ",
        " start with space",
        "   ",
        "a",
        "ab",
    ];
    let mut texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
    texts.push(format!("start {} end", "ab cd ".repeat(40)));
    texts.push(format!("w {} w", "a".repeat(40)));
    texts
}

// ---------------------------------------------------------------------------------------------
// Vocabularies and models

const EXTRA_CHARS: &str = "▁Ġ_éèêëýÿáíóúñçàâäöüßẃẹ\u{301}\u{308}\u{307}\u{323}ıİσςΣﬁＡａ日本한\u{a0}\u{3000}\u{200b}\u{200d}\u{feff}\u{fffd}\u{600}\u{85}\u{2028}\t\n\r";

const MERGE_TARGETS: &[&str] = &[
    "th",
    "the",
    "he",
    "in",
    "ing",
    "er",
    "an",
    "and",
    "ab",
    "abc",
    "xy",
    "xyz",
    "cd",
    "ef",
    "on",
    "re",
    "es",
    "ll",
    "hello",
    "wor",
    "world",
    "ca",
    "caf",
    "café",
    "e\u{301}",
    "12",
    "123",
    "..",
    "'s",
    "▁t",
    "▁th",
    "▁the",
    "▁a",
    "▁ab",
    "▁x",
    "▁xy",
    "▁c",
    "▁cd",
    "▁▁",
    "Ġt",
    "Ġth",
    "Ġthe",
    "Ġa",
    "Ġab",
    "Ġx",
    "Ġxy",
    "Ġc",
    "Ġcd",
    "ĠĠ",
    " t",
    " th",
    " the",
    " a",
    " ab",
    " x",
    " xy",
    " c",
    " cd",
    "  ",
    "_t",
    "_x",
    "-x",
    "x-",
    "!!",
    "\u{301}\u{301}",
];

fn alphabet() -> BTreeSet<char> {
    let mut chars: BTreeSet<char> = (' '..='~').collect();
    chars.extend(EXTRA_CHARS.chars());
    chars.extend(ByteLevel::alphabet());
    chars
}

fn add(vocab: &mut AHashMap<String, u32>, token: String) {
    let id = vocab.len() as u32;
    vocab.entry(token).or_insert(id);
}

fn bpe_vocab(extra_targets: &[&str]) -> (AHashMap<String, u32>, Vec<(String, String)>) {
    let mut vocab = AHashMap::new();
    add(&mut vocab, "<unk>".into());
    for b in 0..=255u8 {
        add(&mut vocab, format!("<0x{b:02X}>"));
    }
    for c in alphabet() {
        add(&mut vocab, c.to_string());
    }
    let mut merges = vec![];
    for target in MERGE_TARGETS.iter().chain(extra_targets) {
        let chars: Vec<char> = target.chars().collect();
        for i in 1..chars.len() {
            let merged: String = chars[..=i].iter().collect();
            if !vocab.contains_key(&merged) {
                merges.push((chars[..i].iter().collect(), chars[i].to_string()));
                add(&mut vocab, merged);
            }
        }
    }
    (vocab, merges)
}

type BpeBuilder = tokenizers::models::bpe::BpeBuilder;

/// A BPE without unknown token drops unknown chars, see `bug_bpe_dropped_unknown_char_*`. The
/// other BPE models have one so that this bug doesn't hide others.
fn bpe_no_unk(
    extra_targets: &[&str],
    configure: impl FnOnce(BpeBuilder) -> BpeBuilder,
) -> ModelWrapper {
    let (vocab, merges) = bpe_vocab(extra_targets);
    configure(BPE::builder().vocab_and_merges(vocab, merges))
        .build()
        .unwrap()
        .into()
}

fn bpe_with(
    extra_targets: &[&str],
    configure: impl FnOnce(BpeBuilder) -> BpeBuilder,
) -> ModelWrapper {
    bpe_no_unk(extra_targets, |b| configure(b.unk_token("<unk>".into())))
}

fn bpe() -> ModelWrapper {
    bpe_with(&[], |b| b)
}

/// BPE with `continuing_subword_prefix` (`##`) or `end_of_word_suffix` (`</w>`).
fn bpe_affix(prefix: bool) -> ModelWrapper {
    let mut vocab = AHashMap::new();
    add(&mut vocab, "<unk>".into());
    for c in alphabet() {
        add(&mut vocab, c.to_string());
        add(
            &mut vocab,
            if prefix {
                format!("##{c}")
            } else {
                format!("{c}</w>")
            },
        );
    }
    let pairs: &[(&str, &str)] = if prefix {
        &[
            ("t", "##h"),
            ("th", "##e"),
            ("▁", "##t"),
            ("▁t", "##h"),
            ("x", "##y"),
            (" ", "##a"),
        ]
    } else {
        &[
            ("t", "h"),
            ("th", "e</w>"),
            ("▁", "t"),
            ("▁t", "h"),
            ("x", "y</w>"),
            ("y", " </w>"),
        ]
    };
    let mut merges = vec![];
    for (a, b) in pairs {
        let merged = format!("{a}{}", if prefix { &b[2..] } else { b });
        add(&mut vocab, merged);
        merges.push((a.to_string(), b.to_string()));
    }
    let builder = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token("<unk>".into());
    let builder = if prefix {
        builder.continuing_subword_prefix("##".into())
    } else {
        builder.end_of_word_suffix("</w>".into())
    };
    builder.build().unwrap().into()
}

fn word_vocab(unk: &str) -> AHashMap<String, u32> {
    let mut vocab = AHashMap::new();
    add(&mut vocab, unk.into());
    for c in alphabet() {
        add(&mut vocab, c.to_string());
    }
    for word in MERGE_TARGETS.iter().chain(WORDS) {
        add(&mut vocab, word.to_string());
    }
    vocab
}

fn wordlevel() -> ModelWrapper {
    WordLevel::builder()
        .vocab(word_vocab("<unk>"))
        .unk_token("<unk>".into())
        .build()
        .unwrap()
        .into()
}

fn wordpiece() -> ModelWrapper {
    let mut vocab = word_vocab("[UNK]");
    for c in alphabet() {
        add(&mut vocab, format!("##{c}"));
    }
    for target in ["he", "ing", "llo", "ld"] {
        add(&mut vocab, format!("##{target}"));
    }
    WordPiece::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .max_input_chars_per_word(6)
        .build()
        .unwrap()
        .into()
}

fn unigram() -> ModelWrapper {
    let mut vocab = vec![("<unk>".to_string(), 0.0)];
    for (i, c) in alphabet().into_iter().enumerate() {
        vocab.push((c.to_string(), -5.0 - (i % 7) as f64 * 0.1));
    }
    for (i, word) in MERGE_TARGETS.iter().enumerate() {
        if word.chars().count() > 1 {
            vocab.push((word.to_string(), -3.0 - (i % 5) as f64 * 0.3));
        }
    }
    Unigram::from(vocab, Some(0), false).unwrap().into()
}

fn models() -> Vec<(&'static str, ModelWrapper)> {
    vec![
        ("bpe", bpe()),
        (
            "bpe(byte_fallback,no unk)",
            bpe_no_unk(&[], |b| b.byte_fallback(true)),
        ),
        ("bpe(fuse)", bpe_with(&[], |b| b.fuse_unk(true))),
        (
            "bpe(byte_fallback,fuse)",
            bpe_with(&[], |b| b.fuse_unk(true).byte_fallback(true)),
        ),
        (
            "bpe(ignore_merges)",
            bpe_with(&[], |b| b.ignore_merges(true)),
        ),
        ("bpe(dropout=0)", bpe_with(&[], |b| b.dropout(0.0))),
        ("bpe(no cache)", bpe_with(&[], |b| b.cache_capacity(0))),
        ("bpe(cross)", bpe_with(&["y ", "y▁", "yĠ", "y_"], |b| b)),
        ("bpe(prefix ##)", bpe_affix(true)),
        ("bpe(suffix </w>)", bpe_affix(false)),
        ("wordlevel", wordlevel()),
        ("wordpiece", wordpiece()),
        ("unigram", unigram()),
    ]
}

// ---------------------------------------------------------------------------------------------
// Pipeline components

fn replace(pattern: &str, content: &str) -> NormalizerWrapper {
    Replace::new(pattern, content).unwrap().into()
}

fn replace_regex(pattern: &str, content: &str) -> NormalizerWrapper {
    Replace::new(ReplacePattern::Regex(pattern.into()), content)
        .unwrap()
        .into()
}

fn prepend(s: &str) -> NormalizerWrapper {
    Prepend::new(s.into()).into()
}

fn nseq(normalizers: Vec<NormalizerWrapper>) -> NormalizerWrapper {
    NormalizerSequence::new(normalizers).into()
}

fn split(pattern: &str, behavior: SplitDelimiterBehavior, invert: bool) -> PreTokenizerWrapper {
    Split::new(pattern, behavior, invert).unwrap().into()
}

fn split_regex(
    pattern: &str,
    behavior: SplitDelimiterBehavior,
    invert: bool,
) -> PreTokenizerWrapper {
    Split::new(SplitPattern::Regex(pattern.into()), behavior, invert)
        .unwrap()
        .into()
}

fn pseq(pre_tokenizers: Vec<PreTokenizerWrapper>) -> PreTokenizerWrapper {
    PreTokenizerSequence::new(pre_tokenizers).into()
}

fn metaspace(scheme: PrependScheme, split: bool) -> PreTokenizerWrapper {
    Metaspace::new('▁', scheme, split).into()
}

fn precompiled() -> Option<NormalizerWrapper> {
    let json = std::fs::read_to_string("data/albert-base-v1-tokenizer.json").ok()?;
    let json: serde_json::Value = serde_json::from_str(&json).ok()?;
    let normalizer = json["normalizer"]["normalizers"]
        .as_array()?
        .iter()
        .find(|n| n["type"] == "Precompiled")?
        .clone();
    serde_json::from_value(normalizer).ok()
}

const GPT2: &str =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
const CL100K: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const QWEN2: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const O200K: &str = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const NEMO: &str = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const DEEPSEEK_V3: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
const DEEPSEEK_CODER: &str = "\\s?\\p{L}+";
const BLOOM: &str = " ?[^(\\s|[.,!?\u{2026}\u{3002}\u{ff0c}\u{3001}\u{964}\u{6d4}\u{60c}])]+";
const CLIP: &str = "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+";
const SPACE_LED: [&str; 9] = [
    GPT2,
    CL100K,
    QWEN2,
    O200K,
    NEMO,
    DEEPSEEK_V3,
    DEEPSEEK_CODER,
    BLOOM,
    CLIP,
];
const SPACE_FREE: [&str; 6] = [
    "\\p{N}{1,3}",
    "[0-9][0-9][0-9]",
    "[\r\n]",
    "\\s?\\p{P}+",
    "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
    "[\u{4e00}-\u{9fa5}\u{800}-\u{4e00}\u{ac00}-\u{d7ff}]+",
];

type Named<T> = (String, Option<T>);

fn named<T>(name: impl Into<String>, value: impl Into<T>) -> Named<T> {
    (name.into(), Some(value.into()))
}

fn normalizers() -> Vec<Named<NormalizerWrapper>> {
    let mut list: Vec<Named<NormalizerWrapper>> = vec![
        ("none".into(), None),
        named("bert", BertNormalizer::default()),
        named(
            "bert(keep accents,case)",
            BertNormalizer::new(true, true, Some(false), false),
        ),
        named(
            "bert(no clean,no cjk,strip,lower)",
            BertNormalizer::new(false, false, Some(true), true),
        ),
        named("strip(both)", Strip::new(true, true)),
        named("strip(left)", Strip::new(true, false)),
        named("strip(right)", Strip::new(false, true)),
        named("strip_accents", StripAccents),
        named("nfc", NFC),
        named("nfd", NFD),
        named("nfkc", NFKC),
        named("nfkd", NFKD),
        named("nmt", Nmt),
        named("lowercase", Lowercase),
        named("bytelevel", ByteLevelNormalizer::new()),
        named("replace(' '→'▁')", replace(" ", "▁")),
        named("replace(' '→'_')", replace(" ", "_")),
        named("replace(' '→'\\t')", replace(" ", "\t")),
        named("replace('é'→'e')", replace("é", "e")),
        named("replace('``'→'\"')", replace("``", "\"")),
        named("replace('\\t'→' ')", replace("\t", " ")),
        named("replace('-'→' ')", replace("-", " ")),
        named("replace('.'→'')", replace(".", "")),
        named("replace(/ {2,}/→' ')", replace_regex(" {2,}", " ")),
        named("replace(/\\s+/→' ')", replace_regex("\\s+", " ")),
        named("replace(/\\s+/→'▁')", replace_regex("\\s+", "▁")),
        named("prepend('▁')", prepend("▁")),
        named("prepend(' ')", prepend(" ")),
        named(
            "seq[prepend('▁'),replace(' '→'▁')]",
            nseq(vec![prepend("▁"), replace(" ", "▁")]),
        ),
        named(
            "seq[replace(' '→'▁'),prepend('▁')]",
            nseq(vec![replace(" ", "▁"), prepend("▁")]),
        ),
        named(
            "seq[nfkd,strip_accents,lowercase]",
            nseq(vec![NFKD.into(), StripAccents.into(), Lowercase.into()]),
        ),
        named(
            "seq[strip(both),nfc]",
            nseq(vec![Strip::new(true, true).into(), NFC.into()]),
        ),
        named(
            "seq[nmt,nfkc,replace(/ {2,}/)]",
            nseq(vec![Nmt.into(), NFKC.into(), replace_regex(" {2,}", " ")]),
        ),
        named(
            "seq[lowercase,prepend('▁'),replace(' '→'▁')]",
            nseq(vec![Lowercase.into(), prepend("▁"), replace(" ", "▁")]),
        ),
        named(
            "seq[replace(' '→'▁'),strip(both)]",
            nseq(vec![replace(" ", "▁"), Strip::new(true, true).into()]),
        ),
        named(
            "seq[prepend(' '),strip(left)]",
            nseq(vec![prepend(" "), Strip::new(true, false).into()]),
        ),
        named(
            "seq[strip(both),prepend('▁'),replace(' '→'▁')]",
            nseq(vec![
                Strip::new(true, true).into(),
                prepend("▁"),
                replace(" ", "▁"),
            ]),
        ),
        named(
            "seq[replace('\\t'→' '),replace(/ {2,}/)]",
            nseq(vec![replace("\t", " "), replace_regex(" {2,}", " ")]),
        ),
        named(
            "seq[replace('-'→' '),strip(right)]",
            nseq(vec![replace("-", " "), Strip::new(false, true).into()]),
        ),
        named(
            "seq[bert,replace(/\\s+/→'▁')]",
            nseq(vec![
                BertNormalizer::default().into(),
                replace_regex("\\s+", "▁"),
            ]),
        ),
        named(
            "seq[replace(' '→'▁'),nmt]",
            nseq(vec![replace(" ", "▁"), Nmt.into()]),
        ),
        named(
            "seq[prepend('▁'),replace(' '→'▁'),replace('▁▁'→'▁')]",
            nseq(vec![prepend("▁"), replace(" ", "▁"), replace("▁▁", "▁")]),
        ),
        named("seq[prepend('x')]", nseq(vec![prepend("x")])),
    ];
    if let Some(p) = precompiled() {
        list.push(("precompiled".into(), Some(p.clone())));
        list.push((
            "seq[precompiled,replace(/ {2,}/),prepend('▁'),replace(' '→'▁')]".into(),
            Some(nseq(vec![
                p,
                replace_regex(" {2,}", " "),
                prepend("▁"),
                replace(" ", "▁"),
            ])),
        ));
    }
    list
}

fn pre_tokenizers() -> Vec<Named<PreTokenizerWrapper>> {
    let mut list: Vec<Named<PreTokenizerWrapper>> = vec![
        ("none".into(), None),
        named("bert", BertPreTokenizer),
        named("whitespace", Whitespace),
        named("whitespace_split", WhitespaceSplit),
        named("delimiter(' ')", CharDelimiterSplit::new(' ')),
        named("delimiter('-')", CharDelimiterSplit::new('-')),
        named("delimiter('▁')", CharDelimiterSplit::new('▁')),
        named("delimiter('x')", CharDelimiterSplit::new('x')),
        named("digits(individual)", Digits::new(true)),
        named("digits(contiguous)", Digits::new(false)),
        named("unicode_scripts", UnicodeScripts::new()),
        named("fixed_length(3)", FixedLength::new(3)),
        named(
            "metaspace('_',first,split)",
            Metaspace::new('_', PrependScheme::First, true),
        ),
        named(
            "metaspace('x',never,split)",
            Metaspace::new('x', PrependScheme::Never, true),
        ),
        named(
            "metaspace(' ',always,split)",
            Metaspace::new(' ', PrependScheme::Always, true),
        ),
        named(
            "seq[whitespace_split,metaspace]",
            pseq(vec![
                WhitespaceSplit.into(),
                metaspace(PrependScheme::Always, true),
            ]),
        ),
        named(
            "seq[split(gpt2),bytelevel(no prefix,no regex)]",
            pseq(vec![
                split_regex(GPT2, Isolated, false),
                ByteLevel::new(false, false, false).into(),
            ]),
        ),
        named(
            "seq[digits,punctuation,bytelevel(no prefix,no regex)]",
            pseq(vec![
                Digits::new(true).into(),
                Punctuation::new(Isolated).into(),
                ByteLevel::new(false, false, false).into(),
            ]),
        ),
        named(
            "seq[punctuation(next),digits]",
            pseq(vec![
                Punctuation::new(MergedWithNext).into(),
                Digits::new(false).into(),
            ]),
        ),
        named(
            "seq[metaspace(never,nosplit),punctuation]",
            pseq(vec![
                metaspace(PrependScheme::Never, false),
                Punctuation::new(Isolated).into(),
            ]),
        ),
        named(
            "seq[split('-'),metaspace(first,split)]",
            pseq(vec![
                split("-", Isolated, false),
                metaspace(PrependScheme::First, true),
            ]),
        ),
        named(
            "seq[whitespace_split,unicode_scripts]",
            pseq(vec![WhitespaceSplit.into(), UnicodeScripts::new().into()]),
        ),
        named(
            "seq[bert,digits]",
            pseq(vec![BertPreTokenizer.into(), Digits::new(true).into()]),
        ),
        named(
            "seq[split(' ',next),bytelevel(prefix,no regex)]",
            pseq(vec![
                split(" ", MergedWithNext, false),
                ByteLevel::new(true, false, false).into(),
            ]),
        ),
        named(
            "seq[bytelevel(no prefix,no regex),split('Ġ',isolated)]",
            pseq(vec![
                ByteLevel::new(false, false, false).into(),
                split("Ġ", Isolated, false),
            ]),
        ),
        named(
            "seq[metaspace(never,nosplit),bytelevel(prefix,no regex)]",
            pseq(vec![
                metaspace(PrependScheme::Never, false),
                ByteLevel::new(true, false, false).into(),
            ]),
        ),
        named(
            "seq[bytelevel(no prefix,no regex),metaspace(always,split)]",
            pseq(vec![
                ByteLevel::new(false, false, false).into(),
                metaspace(PrependScheme::Always, true),
            ]),
        ),
    ];
    for scheme in [
        PrependScheme::Always,
        PrependScheme::First,
        PrependScheme::Never,
    ] {
        for s in [true, false] {
            list.push(named(
                format!("metaspace({scheme:?},split={s})"),
                metaspace(scheme, s),
            ));
        }
    }
    for aps in [false, true] {
        for regex in [false, true] {
            list.push(named(
                format!("bytelevel(prefix={aps},regex={regex})"),
                ByteLevel::new(aps, true, regex),
            ));
        }
    }
    for behavior in BEHAVIORS {
        list.push(named(
            format!("punctuation({behavior:?})"),
            Punctuation::new(behavior),
        ));
        for invert in [false, true] {
            for pattern in [" ", "-", "▁", "é", "x"] {
                list.push(named(
                    format!("split({pattern:?},{behavior:?},invert={invert})"),
                    split(pattern, behavior, invert),
                ));
            }
        }
        for regex in SPACE_FREE {
            list.push(named(
                format!("split(/{}/,{behavior:?})", regex.escape_debug()),
                split_regex(regex, behavior, false),
            ));
        }
    }
    for (i, regex) in SPACE_LED.iter().enumerate() {
        for (behavior, invert) in [
            (Isolated, false),
            (Removed, true),
            (MergedWithNext, false),
            (Removed, false),
        ] {
            list.push(named(
                format!("split(space_led#{i},{behavior:?},invert={invert})"),
                split_regex(regex, behavior, invert),
            ));
        }
    }
    list
}

#[derive(Clone)]
struct Added {
    name: &'static str,
    special: Vec<AddedToken>,
    normal: Vec<AddedToken>,
    encode_special: bool,
}

fn added_configs() -> Vec<Added> {
    let special = vec![
        AddedToken::from("[CLS]", true),
        AddedToken::from("[SEP]", true),
        AddedToken::from("<|endoftext|>", true),
        AddedToken::from("</s>", true),
        AddedToken::from("<mask>", true).lstrip(true).rstrip(true),
    ];
    let normalized = vec![
        AddedToken::from("<pad>", false),
        AddedToken::from("-x-", false),
        AddedToken::from("x-", false),
        AddedToken::from("..", false).single_word(true),
        AddedToken::from("--", false).lstrip(true),
        AddedToken::from("==", false).rstrip(true),
        AddedToken::from("@@", false)
            .lstrip(true)
            .rstrip(true)
            .single_word(true),
        AddedToken::from("  ", false),
    ];
    let raw = vec![
        AddedToken::from("x-", false).normalized(false),
        AddedToken::from("-x-", false)
            .normalized(false)
            .single_word(true),
        AddedToken::from("..", false).normalized(false).lstrip(true),
        AddedToken::from("==", false)
            .normalized(false)
            .rstrip(true)
            .single_word(true),
        AddedToken::from("é-", false).normalized(false),
    ];
    let lstrip_letter = vec![
        AddedToken::from("x-", false).lstrip(true),
        AddedToken::from("x-", true).lstrip(true).rstrip(true),
    ];
    vec![
        Added {
            name: "none",
            special: vec![],
            normal: vec![],
            encode_special: false,
        },
        Added {
            name: "special",
            special: special.clone(),
            normal: vec![],
            encode_special: false,
        },
        Added {
            name: "special+encode_special",
            special: special.clone(),
            normal: vec![],
            encode_special: true,
        },
        Added {
            name: "normalized",
            special: vec![],
            normal: normalized.clone(),
            encode_special: false,
        },
        Added {
            name: "raw",
            special: vec![],
            normal: raw.clone(),
            encode_special: false,
        },
        Added {
            name: "lstrip_letter",
            special: vec![],
            normal: lstrip_letter,
            encode_special: false,
        },
        Added {
            name: "all",
            special,
            normal: [normalized, raw].concat(),
            encode_special: false,
        },
    ]
}

fn post_processors() -> Vec<Named<PostProcessorWrapper>> {
    vec![
        ("none".into(), None),
        named(
            "bert",
            BertProcessing::new(("[SEP]".into(), 1), ("[CLS]".into(), 2)),
        ),
        named(
            "roberta(trim)",
            RobertaProcessing::new(("</s>".into(), 1), ("<s>".into(), 2)).trim_offsets(true),
        ),
        named("bytelevel(trim)", ByteLevel::new(true, true, true)),
        named(
            "template",
            TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]")
                .unwrap()
                .special_tokens(vec![("[CLS]", 2), ("[SEP]", 1)])
                .build()
                .unwrap(),
        ),
    ]
}

struct Pipeline {
    name: String,
    tokenizer: Tokenizer,
}

fn build(
    normalizer: &Named<NormalizerWrapper>,
    pre_tokenizer: &Named<PreTokenizerWrapper>,
    model: &(&'static str, ModelWrapper),
    added: &Added,
    post: &Named<PostProcessorWrapper>,
) -> Pipeline {
    let mut tokenizer = Tokenizer::new(model.1.clone());
    tokenizer.with_normalizer(normalizer.1.clone());
    tokenizer.with_pre_tokenizer(pre_tokenizer.1.clone());
    tokenizer.with_post_processor(post.1.clone());
    tokenizer.add_special_tokens(&added.special);
    tokenizer.add_tokens(&added.normal);
    tokenizer.set_encode_special_tokens(added.encode_special);
    Pipeline {
        name: format!(
            "normalizer={} pre_tokenizer={} model={} added={} post={}",
            normalizer.0, pre_tokenizer.0, model.0, added.name, post.0
        ),
        tokenizer,
    }
}

#[derive(Default)]
struct Report {
    plans: AHashMap<String, usize>,
    failures: Vec<String>,
}

impl Report {
    fn segments(&self) -> usize {
        self.plans
            .iter()
            .filter(|(plan, _)| plan.starts_with("Segments"))
            .map(|(_, n)| n)
            .sum()
    }

    fn finish(self, test: &str) {
        let mut plans: Vec<_> = self.plans.iter().collect();
        plans.sort();
        eprintln!("{test}: plans {plans:?}");
        assert!(
            self.segments() > 0,
            "{}: no pipeline exercises segments",
            test
        );
        if !self.failures.is_empty() {
            panic!(
                "{test}: {} pipelines differ from serial encoding:\n{}",
                self.failures.len(),
                self.failures.join("\n")
            );
        }
    }
}

fn cases() -> usize {
    std::env::var("REDTEAM_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25)
}

fn run(report: &mut Report, pipeline: Pipeline, seed: u64) {
    let plan = pipeline.tokenizer.parallel_plan();
    *report.plans.entry(format!("{plan:?}")).or_default() += 1;
    let texts: Vec<String> = match plan {
        ParallelPlan::Segments { .. } => adversarial_texts()
            .into_iter()
            .chain(random_texts(cases(), seed))
            .collect(),
        ParallelPlan::Model => adversarial_texts().into_iter().step_by(5).collect(),
        ParallelPlan::Serial => vec!["a b c".into()],
    };
    for text in texts {
        if std::env::var("REDTEAM_REPORT_PANICS").is_ok() {
            if let Err(e) = catch(|| pipeline.tokenizer.encode(text.as_str(), false)) {
                if e.to_string().starts_with("panic") {
                    eprintln!("serial {e} for {} on {text:?}", pipeline.name);
                }
            }
        }
        if mismatch(&pipeline.tokenizer, &text).is_some() {
            let minimal = minimize(&pipeline.tokenizer, &text);
            let detail = mismatch(&pipeline.tokenizer, &minimal).unwrap();
            report
                .failures
                .push(format!("- {} plan={plan:?}\n  {detail}", pipeline.name));
            return;
        }
    }
}

fn none_post() -> Named<PostProcessorWrapper> {
    ("none".into(), None)
}

fn model_named<'a>(
    models: &'a [(&'static str, ModelWrapper)],
    name: &str,
) -> &'a (&'static str, ModelWrapper) {
    models.iter().find(|(n, _)| *n == name).unwrap()
}

fn find<T: Clone>(list: &[Named<T>], name: &str) -> Named<T> {
    list.iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("no component named {}", name))
        .clone()
}

fn hash(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

// ---------------------------------------------------------------------------------------------
// Broad differential tests

#[test]
fn every_normalizer() {
    let pre = pre_tokenizers();
    let pres = [
        ("none".to_string(), None),
        find(&pre, "metaspace(Always,split=true)"),
        find(&pre, "metaspace(Never,split=false)"),
        find(&pre, "bytelevel(prefix=false,regex=false)"),
        find(&pre, "bytelevel(prefix=true,regex=true)"),
        find(&pre, "whitespace_split"),
        find(&pre, "punctuation(Isolated)"),
        find(&pre, "split(space_led#1,Isolated,invert=false)"),
    ];
    let models = models();
    let added = added_configs();
    let mut report = Report::default();
    for normalizer in normalizers() {
        for pre_tokenizer in &pres {
            for model in [&models[0], &models[3]] {
                for added in [&added[0], &added[6]] {
                    let pipeline = build(&normalizer, pre_tokenizer, model, added, &none_post());
                    let seed = hash(&pipeline.name);
                    run(&mut report, pipeline, seed);
                }
            }
        }
    }
    report.finish("every_normalizer");
}

#[test]
fn every_pre_tokenizer() {
    let norm = normalizers();
    let norms = [
        ("none".to_string(), None),
        find(&norm, "lowercase"),
        find(&norm, "replace(' '→'▁')"),
        find(&norm, "seq[prepend('▁'),replace(' '→'▁')]"),
        find(&norm, "nfkc"),
        find(&norm, "strip(both)"),
    ];
    let models = models();
    let added = added_configs();
    let mut report = Report::default();
    for pre_tokenizer in pre_tokenizers() {
        for normalizer in &norms {
            for model in [&models[0], model_named(&models, "wordlevel")] {
                for added in [&added[0], &added[6]] {
                    let pipeline = build(normalizer, &pre_tokenizer, model, added, &none_post());
                    let seed = hash(&pipeline.name);
                    run(&mut report, pipeline, seed);
                }
            }
        }
    }
    report.finish("every_pre_tokenizer");
}

#[test]
fn every_model() {
    let norm = normalizers();
    let pre = pre_tokenizers();
    let pipelines = [
        (("none".to_string(), None), ("none".to_string(), None)),
        (("none".to_string(), None), find(&pre, "whitespace_split")),
        (
            find(&norm, "seq[prepend('▁'),replace(' '→'▁')]"),
            ("none".to_string(), None),
        ),
        (
            ("none".to_string(), None),
            find(&pre, "metaspace(First,split=false)"),
        ),
        (
            ("none".to_string(), None),
            find(&pre, "bytelevel(prefix=false,regex=false)"),
        ),
        (find(&norm, "bert"), find(&pre, "bert")),
        (
            ("none".to_string(), None),
            find(&pre, "seq[split(gpt2),bytelevel(no prefix,no regex)]"),
        ),
        (find(&norm, "nfkc"), find(&pre, "digits(individual)")),
    ];
    let added = added_configs();
    let mut report = Report::default();
    for model in models() {
        for (normalizer, pre_tokenizer) in &pipelines {
            for added in &added {
                let pipeline = build(normalizer, pre_tokenizer, &model, added, &none_post());
                let seed = hash(&pipeline.name);
                run(&mut report, pipeline, seed);
            }
        }
    }
    report.finish("every_model");
}

#[test]
fn every_added_token_config_and_post_processor() {
    let norm = normalizers();
    let pre = pre_tokenizers();
    let pipelines = [
        (("none".to_string(), None), ("none".to_string(), None)),
        (find(&norm, "lowercase"), find(&pre, "whitespace_split")),
        (
            find(&norm, "seq[prepend('▁'),replace(' '→'▁')]"),
            ("none".to_string(), None),
        ),
        (
            find(&norm, "strip(both)"),
            find(&pre, "metaspace(Always,split=false)"),
        ),
        (
            find(&norm, "prepend('▁')"),
            find(&pre, "metaspace(Never,split=true)"),
        ),
        (
            find(&norm, "nfkd"),
            find(&pre, "bytelevel(prefix=true,regex=true)"),
        ),
        (
            find(&norm, "replace(/\\s+/→' ')"),
            find(&pre, "split(space_led#0,Isolated,invert=false)"),
        ),
        (
            find(&norm, "strip(right)"),
            find(&pre, "punctuation(MergedWithNext)"),
        ),
    ];
    let models = models();
    let mut report = Report::default();
    for added in added_configs() {
        for post in post_processors() {
            for (normalizer, pre_tokenizer) in &pipelines {
                for model in [&models[0], &models[3], model_named(&models, "wordpiece")] {
                    let pipeline = build(normalizer, pre_tokenizer, model, &added, &post);
                    let seed = hash(&pipeline.name);
                    run(&mut report, pipeline, seed);
                }
            }
        }
    }
    report.finish("every_added_token_config_and_post_processor");
}

#[test]
fn random_pipelines() {
    let norms = normalizers();
    let pres = pre_tokenizers();
    let models = models();
    let added = added_configs();
    let posts = post_processors();
    let mut rng = Rng(0x2545f4914f6cdd1d);
    let mut report = Report::default();
    let count = std::env::var("REDTEAM_PIPELINES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    for _ in 0..count {
        // Two-stage sequences of random components, so combinations are exercised too.
        let normalizer = if rng.chance(30) {
            let (a, b) = (
                &norms[rng.below(norms.len())],
                &norms[rng.below(norms.len())],
            );
            match (&a.1, &b.1) {
                (Some(x), Some(y)) => (
                    format!("seq[{},{}]", a.0, b.0),
                    Some(nseq(vec![x.clone(), y.clone()])),
                ),
                _ => a.clone(),
            }
        } else {
            norms[rng.below(norms.len())].clone()
        };
        let pre_tokenizer = if rng.chance(30) {
            let (a, b) = (&pres[rng.below(pres.len())], &pres[rng.below(pres.len())]);
            match (&a.1, &b.1) {
                (Some(x), Some(y)) => (
                    format!("seq[{},{}]", a.0, b.0),
                    Some(pseq(vec![x.clone(), y.clone()])),
                ),
                _ => a.clone(),
            }
        } else {
            pres[rng.below(pres.len())].clone()
        };
        let model = &models[rng.below(models.len())];
        let added = &added[rng.below(added.len())];
        let post = &posts[rng.below(posts.len())];
        let pipeline = build(&normalizer, &pre_tokenizer, model, added, post);
        let seed = rng.next();
        run(&mut report, pipeline, seed);
    }
    report.finish("random_pipelines");
}

#[test]
fn truncation_and_padding() {
    let norm = normalizers();
    let pre = pre_tokenizers();
    let models = models();
    let added = added_configs();
    let posts = post_processors();
    let text = adversarial_texts().join(" ");
    for post in &posts {
        for (normalizer, pre_tokenizer) in [
            (("none".to_string(), None), ("none".to_string(), None)),
            (
                find(&norm, "seq[prepend('▁'),replace(' '→'▁')]"),
                ("none".to_string(), None),
            ),
            (
                find(&norm, "lowercase"),
                find(&pre, "bytelevel(prefix=true,regex=true)"),
            ),
        ] {
            let mut pipeline = build(&normalizer, &pre_tokenizer, &models[0], &added[1], post);
            assert!(matches!(
                pipeline.tokenizer.parallel_plan(),
                ParallelPlan::Segments { .. }
            ));
            for (max_length, stride, strategy) in [
                (17, 3, TruncationStrategy::LongestFirst),
                (64, 0, TruncationStrategy::OnlyFirst),
            ] {
                pipeline
                    .tokenizer
                    .with_truncation(Some(TruncationParams {
                        max_length,
                        stride,
                        strategy,
                        ..Default::default()
                    }))
                    .unwrap()
                    .with_padding(Some(PaddingParams {
                        strategy: PaddingStrategy::Fixed(80),
                        pad_to_multiple_of: Some(8),
                        ..Default::default()
                    }));
                if let Some(m) = mismatch(&pipeline.tokenizer, &text) {
                    panic!("{}: {m}", pipeline.name);
                }
            }
        }
    }
}

#[test]
fn default_config_on_long_inputs() {
    let norm = normalizers();
    let pre = pre_tokenizers();
    let models = models();
    let added = added_configs();
    let text: String = random_texts(4000, 0x51ab).join(" ");
    assert!(text.len() > 100_000);
    for (normalizer, pre_tokenizer, model) in [
        (
            ("none".to_string(), None),
            ("none".to_string(), None),
            &models[0],
        ),
        (
            find(&norm, "seq[prepend('▁'),replace(' '→'▁')]"),
            ("none".to_string(), None),
            &models[3],
        ),
        (
            find(&norm, "nfkc"),
            find(&pre, "bytelevel(prefix=true,regex=true)"),
            &models[0],
        ),
        (
            find(&norm, "bert"),
            find(&pre, "bert"),
            model_named(&models, "wordpiece"),
        ),
        (
            find(&norm, "strip(both)"),
            find(&pre, "metaspace(Always,split=false)"),
            &models[0],
        ),
        (
            ("none".to_string(), None),
            find(&pre, "split(space_led#3,Isolated,invert=false)"),
            model_named(&models, "unigram"),
        ),
    ] {
        let pipeline = build(
            &normalizer,
            &pre_tokenizer,
            model,
            &added[1],
            &post_processors()[4],
        );
        assert!(
            matches!(
                pipeline.tokenizer.parallel_plan(),
                ParallelPlan::Segments { .. }
            ),
            "{}",
            pipeline.name
        );
        for add_special in [false, true] {
            let expected = pipeline
                .tokenizer
                .encode(text.as_str(), add_special)
                .unwrap();
            let actual = pipeline
                .tokenizer
                .encode_parallel_single(text.as_str(), add_special)
                .unwrap();
            assert!(
                expected == actual,
                "{}: mismatch with the default config",
                pipeline.name
            );
            let expected = pipeline
                .tokenizer
                .encode_char_offsets(text.as_str(), add_special)
                .unwrap();
            let actual = pipeline
                .tokenizer
                .encode_parallel_single_char_offsets(text.as_str(), add_special)
                .unwrap();
            assert!(
                expected == actual,
                "{}: mismatch with the default config (char offsets)",
                pipeline.name
            );
        }
    }
}

#[test]
fn errors_match_serial() {
    // Unknown words with an unknown token missing from the vocabulary make WordLevel fail.
    let model: ModelWrapper = WordLevel::builder()
        .vocab(word_vocab("<unk>"))
        .unk_token("<missing>".into())
        .build()
        .unwrap()
        .into();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    assert_eq!(tokenizer.parallel_plan(), SEGMENTS_FALSE);
    for text in [
        "ab cd ef",
        "ab cd qqq ef",
        "qqq ab",
        "ab qqq",
        "a b c d e f g h i j qqq",
    ] {
        assert_identical(&tokenizer, text);
    }
    // Same with BPE: an unknown char with an unknown token missing from the vocabulary.
    let model = bpe_with(&[], |b| b.unk_token("<missing>".into()));
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_normalizer(Some(nseq(vec![prepend("▁"), replace(" ", "▁")])));
    assert_eq!(tokenizer.parallel_plan(), SEGMENTS_TRUE);
    for text in [
        "ab cd ef",
        "ab cd 語 ef",
        "語 ab",
        "ab 語",
        "a b c d e f g h i j 語",
    ] {
        assert_identical(&tokenizer, text);
    }
}

#[test]
fn most_components_allow_segments() {
    let models = models();
    let added = added_configs();
    let mut segments = 0;
    for normalizer in normalizers() {
        for pre_tokenizer in pre_tokenizers() {
            let pipeline = build(
                &normalizer,
                &pre_tokenizer,
                &models[0],
                &added[0],
                &none_post(),
            );
            if matches!(
                pipeline.tokenizer.parallel_plan(),
                ParallelPlan::Segments { .. }
            ) {
                segments += 1;
            }
        }
    }
    eprintln!("normalizer x pre-tokenizer grid with BPE: {segments} Segments plans");
    assert!(segments > 2000, "{}", segments);
}

// ---------------------------------------------------------------------------------------------
// Plans

fn tokenizer(
    model: ModelWrapper,
    normalizer: Option<NormalizerWrapper>,
    pre_tokenizer: Option<PreTokenizerWrapper>,
) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_normalizer(normalizer);
    tokenizer.with_pre_tokenizer(pre_tokenizer);
    tokenizer
}

#[test]
fn plans() {
    use ParallelPlan::{Model, Serial};
    let plan =
        |model: ModelWrapper, n: Option<NormalizerWrapper>, p: Option<PreTokenizerWrapper>| {
            tokenizer(model, n, p).parallel_plan()
        };
    // Inside cuts need a decomposable BPE.
    assert_eq!(plan(bpe(), None, None), SEGMENTS_TRUE);
    assert_eq!(plan(wordlevel(), None, None), Model);
    assert_eq!(plan(wordpiece(), None, None), Model);
    assert_eq!(plan(unigram(), None, None), Model);
    assert_eq!(
        plan(wordlevel(), None, Some(WhitespaceSplit.into())),
        SEGMENTS_FALSE
    );
    assert_eq!(
        plan(
            unigram(),
            None,
            Some(metaspace(PrependScheme::Always, true))
        ),
        SEGMENTS_FALSE
    );
    // A merge joining a letter to the separator, directly or after other merges, is detected.
    assert_eq!(plan(bpe_with(&["y "], |b| b), None, None), Model);
    assert_eq!(
        plan(bpe_with(&["y▁"], |b| b), Some(replace(" ", "▁")), None),
        Model
    );
    assert_eq!(plan(bpe_with(&["y▁"], |b| b), None, None), SEGMENTS_TRUE);
    assert_eq!(
        plan(bpe_with(&["xy▁z"], |b| b), Some(replace(" ", "▁")), None),
        Model
    );
    assert_eq!(
        plan(bpe_with(&["y "], |b| b), None, Some(WhitespaceSplit.into())),
        SEGMENTS_FALSE
    );
    assert_eq!(
        plan(
            bpe_with(&["yĠ"], |b| b),
            None,
            Some(ByteLevel::new(false, false, false).into())
        ),
        Model
    );
    // BPE options that make a word not decomposable, or non-deterministic.
    assert_eq!(
        plan(bpe_with(&[], |b| b.ignore_merges(true)), None, None),
        Model
    );
    assert_eq!(
        plan(bpe_with(&[], |b| b.dropout(0.0)), None, None),
        SEGMENTS_TRUE
    );
    assert_eq!(plan(bpe_with(&[], |b| b.dropout(0.5)), None, None), Serial);
    assert_eq!(
        plan(
            bpe_with(&[], |b| b.dropout(0.5)),
            None,
            Some(WhitespaceSplit.into())
        ),
        Serial
    );
    assert_eq!(plan(bpe_affix(false), None, None), Model);
    assert_eq!(plan(bpe_affix(true), None, None), Model);
    assert_eq!(
        plan(bpe_affix(true), None, Some(WhitespaceSplit.into())),
        SEGMENTS_FALSE
    );
    // Separator not in the vocabulary: no inside cut.
    let mut vocab: AHashMap<String, u32> = AHashMap::new();
    for c in ('a'..='z').chain('A'..='Z') {
        add(&mut vocab, c.to_string());
    }
    let no_space: ModelWrapper = BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .build()
        .unwrap()
        .into();
    assert_eq!(plan(no_space, None, None), Model);
    // Pre-tokenizers.
    assert_eq!(
        plan(bpe(), None, Some(metaspace(PrependScheme::Always, true))),
        SEGMENTS_FALSE
    );
    assert_eq!(
        plan(bpe(), None, Some(metaspace(PrependScheme::Always, false))),
        SEGMENTS_TRUE
    );
    assert_eq!(
        plan(bpe(), None, Some(ByteLevel::new(true, true, true).into())),
        SEGMENTS_FALSE
    );
    assert_eq!(
        plan(
            bpe(),
            None,
            Some(ByteLevel::new(false, false, false).into())
        ),
        SEGMENTS_TRUE
    );
    assert_eq!(
        plan(bpe(), None, Some(split_regex(GPT2, Isolated, false))),
        SEGMENTS_FALSE
    );
    assert_eq!(
        plan(bpe(), None, Some(split_regex(GPT2, MergedWithNext, false))),
        Model
    );
    assert_eq!(
        plan(bpe(), None, Some(split(" ", MergedWithPrevious, false))),
        Model
    );
    assert_eq!(
        plan(bpe(), None, Some(split(" ", MergedWithNext, false))),
        SEGMENTS_FALSE
    );
    assert_eq!(plan(bpe(), None, Some(UnicodeScripts::new().into())), Model);
    assert_eq!(plan(bpe(), None, Some(FixedLength::new(3).into())), Model);
    assert_eq!(
        plan(
            bpe(),
            None,
            Some(Punctuation::new(MergedWithPrevious).into())
        ),
        SEGMENTS_TRUE
    );
    assert_eq!(
        plan(
            bpe(),
            Some(replace(" ", "_")),
            Some(Punctuation::new(MergedWithPrevious).into())
        ),
        Model
    );
    // Normalizers.
    assert_eq!(
        plan(
            bpe(),
            Some(nseq(vec![prepend("▁"), replace(" ", "▁")])),
            None
        ),
        SEGMENTS_TRUE
    );
    assert_eq!(plan(bpe(), Some(replace(" ", "x")), None), Model);
    assert_eq!(plan(bpe(), Some(replace(" ", "1")), None), Model);
    assert_eq!(plan(bpe(), Some(replace(" ", "▁▁")), None), Model);
    assert_eq!(plan(bpe(), Some(replace("a", "b")), None), Model);
    assert_eq!(plan(bpe(), Some(replace_regex("[ ]", "▁")), None), Model);
    // The vocabulary merges `e` + U+0301, so U+0301 can't be an inside separator.
    assert_eq!(
        plan(
            bpe(),
            Some(replace(" ", "\u{301}")),
            Some(WhitespaceSplit.into())
        ),
        Model
    );
    assert_eq!(
        plan(
            bpe(),
            Some(replace(" ", "\u{308}")),
            Some(WhitespaceSplit.into())
        ),
        SEGMENTS_TRUE
    );
    assert_eq!(
        plan(
            bpe(),
            Some(nseq(vec![replace(" ", "\u{301}"), NFC.into()])),
            None
        ),
        Model
    );
    assert_eq!(
        plan(bpe(), Some(ByteLevelNormalizer::new().into()), None),
        SEGMENTS_TRUE
    );
    // Added tokens.
    let with_tokens = |tokens: &[AddedToken]| {
        let mut t = tokenizer(bpe(), None, Some(WhitespaceSplit.into()));
        t.add_tokens(tokens);
        t.parallel_plan()
    };
    assert_eq!(with_tokens(&[AddedToken::from("ab", false)]), Model);
    assert_eq!(
        with_tokens(&[AddedToken::from(" -", false)]),
        SEGMENTS_FALSE
    );
    assert_eq!(with_tokens(&[AddedToken::from("x y-", false)]), Model);
    assert_eq!(with_tokens(&[AddedToken::from(" x-", false)]), Model);
    assert_eq!(with_tokens(&[AddedToken::from("x ", false)]), Model);
    assert_eq!(
        with_tokens(&[AddedToken::from("  ", false)]),
        SEGMENTS_FALSE
    );
    assert_eq!(with_tokens(&[AddedToken::from(" ", false)]), Model);
    // The rule that the next tests show unsound: a letter, the separator and a non-ASCII letter.
    assert_eq!(
        with_tokens(&[AddedToken::from("y é", false)]),
        SEGMENTS_FALSE
    );
    assert_eq!(
        with_tokens(&[AddedToken::from(" é", false)]),
        SEGMENTS_FALSE
    );
}

#[test]
fn separator_alone_before_added_token() {
    // A raw added token right after the separator leaves the separator alone as the first piece
    // of the right segment, outside the `sep + letter` shape the contracts are written for.
    let norm = normalizers();
    let models = models();
    let mut report = Report::default();
    let pres = pre_tokenizers();
    let selected = [
        "none",
        "strip(both)",
        "strip(right)",
        "prepend('▁')",
        "seq[prepend('▁'),replace(' '→'▁')]",
        "seq[strip(both),prepend('▁'),replace(' '→'▁')]",
        "replace(/\\s+/→'▁')",
        "nfkc",
        "bert",
        "precompiled",
    ];
    for normalizer in norm.iter().filter(|(n, _)| selected.contains(&n.as_str())) {
        for (pre_name, pre_tokenizer) in &pres {
            let pre_tokenizer = (pre_name.clone(), pre_tokenizer.clone());
            for model in [&models[0], &models[3]] {
                let mut tokenizer = Tokenizer::new(model.1.clone());
                tokenizer.with_normalizer(normalizer.1.clone());
                tokenizer.with_pre_tokenizer(pre_tokenizer.1.clone());
                tokenizer.add_tokens(&[
                    AddedToken::from("x-", false).normalized(false),
                    AddedToken::from("é-", false).normalized(false),
                ]);
                let name = format!(
                    "normalizer={} pre_tokenizer={} model={}",
                    normalizer.0, pre_name, model.0
                );
                let plan = tokenizer.parallel_plan();
                if !matches!(plan, ParallelPlan::Segments { .. }) {
                    continue;
                }
                *report.plans.entry(format!("{plan:?}")).or_default() += 1;
                for text in [
                    "ab x- cd",
                    "ab x-cd ef",
                    "ab x- x- cd",
                    "a b x-",
                    "ab x-",
                    "ab  x- cd",
                    "ab x- é- cd",
                    "the x-y the",
                ] {
                    if let Some(m) = mismatch(&tokenizer, text) {
                        report
                            .failures
                            .push(format!("- {name} plan={plan:?}\n  {m}"));
                        break;
                    }
                }
            }
        }
    }
    report.finish("separator_alone_before_added_token");
}

// ---------------------------------------------------------------------------------------------
// Mismatches found by the red team

/// A BPE without unknown token drops unknown chars without counting their bytes, so the
/// offsets of every later token of the word are shifted left. `never_merges_across` only checks
/// the chars around the cut: with an inside cut the shift stops at the cut, and the tokens
/// right of it get their true offsets instead of the shifted serial ones.
#[test]
fn bug_bpe_dropped_unknown_char_shifts_offsets_across_cut() {
    let tokenizer = tokenizer(bpe_no_unk(&[], |b| b), None, None);
    // Fixed: BPE needs an unknown token to support a cut inside a word.
    assert_eq!(tokenizer.parallel_plan(), ParallelPlan::Model);
    assert_identical(&tokenizer, "\u{7f}d j");
}

/// `with_normalizer` (and the Python `normalizer` setter) doesn't rebuild the trie of
/// normalized added tokens, which keeps patterns normalized by the previous normalizer.
/// `AddedVocabulary::supports_cut` re-normalizes the token contents with the current normalizer
/// instead of checking the patterns actually matched: here it checks `y-z-`, but the trie
/// holds `y z `, which matches across both cuts.
#[test]
fn bug_stale_added_token_trie_after_normalizer_change() {
    let mut tokenizer = tokenizer(
        wordlevel(),
        Some(replace("-", " ")),
        Some(WhitespaceSplit.into()),
    );
    tokenizer.add_tokens(&[AddedToken::from("y-z-", false)]);
    tokenizer.with_normalizer(None::<NormalizerWrapper>);
    // Fixed: `supports_cut` checks the patterns in the tries.
    assert_eq!(tokenizer.parallel_plan(), ParallelPlan::Model);
    assert_identical(&tokenizer, "xy z w");
}

/// The same through a typical SentencePiece-style pipeline (`Prepend` + `Replace`).
#[test]
fn bug_bpe_dropped_unknown_char_shifts_offsets_across_cut_llama_style() {
    let tokenizer = tokenizer(
        bpe_no_unk(&[], |b| b),
        Some(nseq(vec![prepend("▁"), replace(" ", "▁")])),
        None,
    );
    assert_eq!(tokenizer.parallel_plan(), ParallelPlan::Model);
    assert_identical(&tokenizer, "語 the end");
}

/// NFC composes the ASCII letter after the separator with a following combining mark, so the
/// normalized text around the cut is `y`, ` `, `é`: not "an ASCII letter, the separator, an
/// ASCII letter" as `pattern_supports_cut` assumes. A normalized added token `y é` then matches
/// across the cut in serial encoding but in neither segment.
#[test]
fn bug_nfc_added_token_across_cut() {
    let mut tokenizer = tokenizer(wordlevel(), Some(NFC.into()), Some(WhitespaceSplit.into()));
    tokenizer.add_tokens(&[AddedToken::from("y é", false)]);
    assert_eq!(tokenizer.parallel_plan(), SEGMENTS_FALSE);
    assert_identical(&tokenizer, "xy e\u{301}z");
}

/// Same with NFKC, and with an inside cut (no pre-tokenizer, BPE).
#[test]
fn bug_nfkc_added_token_across_cut_inside() {
    let mut tokenizer = tokenizer(bpe(), Some(NFKC.into()), None);
    tokenizer.add_tokens(&[AddedToken::from("y é", false)]);
    assert_eq!(tokenizer.parallel_plan(), SEGMENTS_TRUE);
    assert_identical(&tokenizer, "xy e\u{301}z");
}

/// A normalized added token starting with the separator: in the right segment the match
/// starts at offset 0, so `single_word` sees the start of the string instead of the letter
/// before the cut.
#[test]
fn bug_nfc_single_word_added_token_at_cut() {
    let mut tokenizer = tokenizer(wordlevel(), Some(NFC.into()), Some(WhitespaceSplit.into()));
    tokenizer.add_tokens(&[AddedToken::from(" é", false).single_word(true)]);
    assert_eq!(tokenizer.parallel_plan(), SEGMENTS_FALSE);
    assert_identical(&tokenizer, "xy e\u{301} z");
}

/// The Precompiled normalizer transforms whole graphemes, so it composes like NFC.
#[test]
fn bug_precompiled_added_token_across_cut() {
    let Some(precompiled) = precompiled() else {
        eprintln!("skipping: data/albert-base-v1-tokenizer.json is missing");
        return;
    };
    let mut tokenizer = tokenizer(wordlevel(), Some(precompiled), Some(WhitespaceSplit.into()));
    tokenizer.add_tokens(&[AddedToken::from("y é", false)]);
    assert_eq!(tokenizer.parallel_plan(), SEGMENTS_FALSE);
    assert_identical(&tokenizer, "xy e\u{301}z");
}

/// Once NFC has replaced the letter after the separator, a later `Replace` can turn it into
/// punctuation, and `\s?\p{P}+` (allowed by `SPACE_FREE_REGEXES` as never touching the cut)
/// then matches the separator with it: serial splits at the cut while the plan says the cut is
/// inside a split, so word ids are off by one.
#[test]
fn bug_nfc_then_replace_breaks_space_free_regex() {
    let tokenizer = tokenizer(
        bpe(),
        Some(nseq(vec![NFC.into(), replace("é", "!")])),
        Some(split_regex("\\s?\\p{P}+", Isolated, false)),
    );
    assert_eq!(tokenizer.parallel_plan(), SEGMENTS_TRUE);
    assert_identical(&tokenizer, "xy e\u{301}z");
}
