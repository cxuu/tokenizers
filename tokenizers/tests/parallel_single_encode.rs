//! `encode_parallel_with_config` must return exactly what serial encoding returns.
//!
//! Tokenizers come from `data/` and, if present, `data/parallel_corpus/` (any
//! `tokenizer.json` files). Segments are kept tiny so that inputs get cut at almost every
//! eligible space. `PARALLEL_FUZZ_CASES` sets the number of random inputs per tokenizer.

use std::path::{Path, PathBuf};
use tokenizers::tokenizer::{OffsetType, ParallelConfig, ParallelPlan, Tokenizer};
use tokenizers::utils::padding::{PaddingParams, PaddingStrategy};
use tokenizers::utils::truncation::{TruncationParams, TruncationStrategy};
use tokenizers::{Encoding, Result};

fn tokenizer_files() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = [
        "data/bert-wiki.json",
        "data/roberta.json",
        "data/llama-3-tokenizer.json",
        "data/albert-base-v1-tokenizer.json",
        "data/tokenizer-wiki.json",
        "data/unigram.json",
    ]
    .iter()
    .map(PathBuf::from)
    .filter(|p| p.exists())
    .collect();
    if let Ok(dir) = std::fs::read_dir("data/parallel_corpus") {
        let mut corpus: Vec<PathBuf> = dir
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        corpus.sort();
        files.extend(corpus);
    }
    assert!(!files.is_empty(), "no tokenizer files, run `make test`");
    files
}

fn load(path: &Path) -> Option<Tokenizer> {
    match Tokenizer::from_file(path) {
        Ok(tokenizer) => Some(tokenizer),
        Err(e) => {
            eprintln!("skipping {}: {e}", path.display());
            None
        }
    }
}

fn big_text(len: usize) -> String {
    let text = std::fs::read_to_string("data/big.txt").unwrap_or_default();
    let mut end = len.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

/// Inputs that stress what a cut could break.
fn adversarial_texts() -> Vec<String> {
    let mut texts: Vec<String> = [
        "Hello world",
        "a b",
        "The quick brown fox jumps over the lazy dog. The dog was very lazy indeed.",
        "word  two   spaces    four\tTab\nnewline\r\ncrlf \n\n paragraph",
        "x y z ".repeat(50).as_str(),
        "don't won't it's they're I'm we'll he'd THEY'RE",
        "naïve café résumé coöperate Ångström e\u{301}t\u{301}e a\u{301} b\u{301}",
        "İstanbul ǅemal ΟΔΟΣ ΟΔΟΣ οδος straße ﬁne ＡＢＣ ｄｅｆ",
        "日本語の文章 中文 한국어 text mixed 日本 語 abc 漢字",
        "emoji 👍🏽 family 👨‍👩‍👧 flags 🇫🇷🇩🇪 keycap 1️⃣ end",
        "zero\u{200b}width joiner\u{200d}test bom\u{feff}x nbsp\u{a0}y ideo\u{3000}z",
        "ctrl\u{0}nul\u{1}soh\u{7f}del\u{fffd}repl a\u{85}b c\u{2028}d",
        "rtl עברית text العربية mixed",
        "spm ▁marker a▁b ▁ c ▁▁ d Ġbyte aĠb",
        "<|endoftext|> hello <|endoftext|>world a<|endoftext|>b [CLS] a [SEP] b [MASK] c",
        "<s> a </s> b <unk> c <pad> d <mask> e en_XX f <s>NOTUSED g <|im_start|>user a b<|im_end|>",
        "fn main() { let x = vec![1, 2, 3]; println!(\"{x:?}\"); } // comment here",
        "https://example.com/path?query=a b&c=d mail me at a.b@c.de now",
        "1234567890 3.14159 1,000,000 2024-01-01 a1 b2 c3 x 12 y 345 z 6789",
        "``quoted'' text ``more'' a ''b'' c",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // Long runs: an overlap-based merge gets these wrong.
    texts.push(format!("x{}y z w", " ".repeat(4001)));
    texts.push(format!("a {} b c", "=".repeat(3001)));
    texts.push(format!(
        "pi is {} ok then",
        "31415926535897932384626433832795".repeat(40)
    ));
    texts.push(format!("seq is {} end of it", "acgt".repeat(40)));
    texts.push(format!("word {} word", "a".repeat(300)));
    texts.push(format!("start {} end", "ab ".repeat(500)));
    texts.push(big_text(30_000));
    texts
}

/// Deterministic random inputs built from fragments that matter for tokenization.
fn random_texts(cases: usize, seed: u64) -> Vec<String> {
    const FRAGMENTS: &[&str] = &[
        "a",
        "b",
        "Z",
        "the",
        "Hello",
        "WORLD",
        "x",
        " ",
        " ",
        " ",
        "  ",
        "\t",
        "\n",
        "\r\n",
        "\n\n",
        ".",
        ",",
        "!",
        "?",
        "'s",
        "'",
        "\"",
        "-",
        "_",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "0",
        "7",
        "42",
        "123456",
        "é",
        "e\u{301}",
        "ß",
        "İ",
        "Σ",
        "ﬁ",
        "Ａ",
        "日本",
        "語",
        "한",
        "👍",
        "🏽",
        "‍",
        "🇫",
        "\u{200b}",
        "\u{a0}",
        "\u{3000}",
        "\u{0}",
        "\u{7f}",
        "▁",
        "Ġ",
        "<|endoftext|>",
        "[CLS]",
        "[MASK]",
        "<s>",
        "</s>",
        "<mask>",
        "en_XX",
        "<unk>",
        "``",
        "''",
        "===",
        "    ",
        "http://a.b/c",
        // Spaces between letters, where inputs get cut.
        "ab cd",
        "x y",
        "the end",
        "I am",
    ];
    let mut state = seed;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..cases)
        .map(|_| {
            let len = 1 + (next() % 300) as usize;
            (0..len)
                .map(|_| FRAGMENTS[(next() % FRAGMENTS.len() as u64) as usize])
                .collect()
        })
        .collect()
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

fn check(name: &str, tokenizer: &Tokenizer, text: &str, segment: usize) {
    for offsets in [OffsetType::Byte, OffsetType::Char, OffsetType::None] {
        for add_special in [false, true] {
            let config = ParallelConfig::default()
                .with_min_input_bytes(0)
                .with_min_segment_bytes(segment)
                .with_segments_per_thread(1 << 20)
                .with_offsets(offsets);
            let expected = serial(tokenizer, text, add_special, offsets);
            let actual = tokenizer.encode_parallel_with_config(text, add_special, &config);
            match (expected, actual) {
                (Ok(expected), Ok(actual)) => assert!(
                    expected == actual,
                    "{name}: mismatch for {offsets:?} add_special={add_special} segment={segment} \
                     on {:?}\nserial:   {:?}\nparallel: {:?}",
                    text.chars().take(200).collect::<String>(),
                    expected.get_tokens().iter().take(40).collect::<Vec<_>>(),
                    actual.get_tokens().iter().take(40).collect::<Vec<_>>(),
                ),
                (Err(expected), Err(actual)) => {
                    assert_eq!(
                        expected.to_string(),
                        actual.to_string(),
                        "{name}: different errors"
                    )
                }
                (expected, actual) => panic!(
                    "{name}: serial {:?} but parallel {:?} on {text:?}",
                    expected.map(|e| e.len()),
                    actual.map(|e| e.len())
                ),
            }
        }
    }
}

#[test]
fn identical_to_serial_on_adversarial_inputs() {
    let texts = adversarial_texts();
    for path in tokenizer_files() {
        let Some(tokenizer) = load(&path) else {
            continue;
        };
        let name = path.display().to_string();
        for text in &texts {
            for segment in [1, 5, 64] {
                check(&name, &tokenizer, text, segment);
            }
        }
    }
}

#[test]
fn identical_to_serial_on_random_inputs() {
    let cases = std::env::var("PARALLEL_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    for (i, path) in tokenizer_files().into_iter().enumerate() {
        let Some(tokenizer) = load(&path) else {
            continue;
        };
        let name = path.display().to_string();
        for text in random_texts(cases, 0x9e3779b97f4a7c15 ^ i as u64) {
            check(&name, &tokenizer, &text, 1);
        }
    }
}

#[test]
fn identical_to_serial_with_truncation_and_padding() {
    let text = big_text(20_000);
    for path in tokenizer_files() {
        let Some(mut tokenizer) = load(&path) else {
            continue;
        };
        let name = path.display().to_string();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 1000,
                stride: 17,
                strategy: TruncationStrategy::LongestFirst,
                ..Default::default()
            }))
            .unwrap()
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(1100),
                ..Default::default()
            }));
        check(&name, &tokenizer, &text, 16);
    }
}

#[test]
fn default_config_identical_on_long_input() {
    let text = big_text(2_000_000);
    for path in tokenizer_files() {
        let Some(tokenizer) = load(&path) else {
            continue;
        };
        let name = path.display().to_string();
        let expected = tokenizer.encode(text.as_str(), true).unwrap();
        let actual = tokenizer
            .encode_parallel_single(text.as_str(), true)
            .unwrap();
        assert!(expected == actual, "{}: mismatch on long input", name);
        let expected = tokenizer.encode_char_offsets(text.as_str(), false).unwrap();
        let actual = tokenizer
            .encode_parallel_single_char_offsets(text.as_str(), false)
            .unwrap();
        assert!(
            expected == actual,
            "{}: mismatch on long input with char offsets",
            name
        );
    }
}

#[test]
fn plans_of_bundled_tokenizers() {
    let expected = [
        (
            "data/bert-wiki.json",
            ParallelPlan::Segments { inside: false },
        ),
        (
            "data/roberta.json",
            ParallelPlan::Segments { inside: false },
        ),
        (
            "data/llama-3-tokenizer.json",
            ParallelPlan::Segments { inside: false },
        ),
        (
            "data/albert-base-v1-tokenizer.json",
            ParallelPlan::Segments { inside: false },
        ),
    ];
    for (path, plan) in expected {
        let Some(tokenizer) = load(Path::new(path)) else {
            continue;
        };
        assert_eq!(tokenizer.parallel_plan(), plan, "{}", path);
    }
}
