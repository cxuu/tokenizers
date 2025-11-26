//! Time each phase of single-sequence encoding to see what can be parallelized.
//!
//! cargo run --release --example phase_breakdown -- data/bert-wiki.json data/big.txt 4000000
use std::time::Instant;
use tokenizers::{Model, OffsetReferential, OffsetType, PreTokenizer, Tokenizer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tok = Tokenizer::from_file(&args[1]).unwrap();
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let n: usize = args
        .get(3)
        .map(|s| s.parse().unwrap())
        .unwrap_or(text.len());
    let mut end = n.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let text = &text[..end];

    for _ in 0..2 {
        let t0 = Instant::now();
        let mut pre = tok
            .get_added_vocabulary()
            .extract_and_normalize(tok.get_normalizer(), text);
        let t1 = Instant::now();
        if let Some(p) = tok.get_pre_tokenizer() {
            p.pre_tokenize(&mut pre).unwrap();
        }
        let t2 = Instant::now();
        let splits = pre
            .get_splits(OffsetReferential::Original, OffsetType::None)
            .len();
        pre.tokenize(|n| tok.get_model().tokenize(n.get())).unwrap();
        let t3 = Instant::now();
        let enc = pre.into_encoding(None, 0, OffsetType::Byte).unwrap();
        let t4 = Instant::now();
        let enc = tok.post_process(enc, None, false).unwrap();
        let t5 = Instant::now();
        let full = Instant::now();
        let serial = tok.encode(text, false).unwrap();
        let full = full.elapsed();
        assert_eq!(serial.get_ids(), enc.get_ids());
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        println!(
            "{} bytes={} splits={} tokens={} | added+norm {:.1}ms  pretok {:.1}ms  model {:.1}ms  into_enc {:.1}ms  post {:.1}ms | encode() {:.1}ms",
            args[1],
            text.len(),
            splits,
            enc.len(),
            ms(t0, t1),
            ms(t1, t2),
            ms(t2, t3),
            ms(t3, t4),
            ms(t4, t5),
            full.as_secs_f64() * 1e3
        );
    }
}
