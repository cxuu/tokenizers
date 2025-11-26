# Parallel encoding of a single input

`encode_parallel_single` encodes one long input on multiple threads and returns exactly what
`encode` returns: the same ids, tokens, offsets, word ids, masks, truncation and padding. It
is not an approximation: it either proves a parallel strategy exact for the tokenizer, or
uses a slower exact one.

```rust
let encoding = tokenizer.encode_parallel_single(&text, false)?; // == tokenizer.encode(&text, false)?
let encoding = tokenizer.encode_parallel_single_char_offsets(&text, false)?; // == encode_char_offsets
let config = ParallelConfig::default().with_min_input_bytes(64 * 1024);
let encoding = tokenizer.encode_parallel_with_config(&text, false, &config)?;
let plan = tokenizer.parallel_plan(); // Segments { inside }, Model or Serial
```

```python
encoding = tokenizer.encode_parallel(text)  # == tokenizer.encode(text)
```

## How it works

The input is cut into segments that are encoded independently, then concatenated. A cut is
only made at a space with an ASCII letter on each side (`a|␣b`), and only if every stage of
the pipeline proves that encoding `input[..c]` and `input[c..]` separately gives the same
result as encoding `input`. Each stage answers through a trait method whose default is "not
provable", so custom components never get cut:

| Stage | Method | Proven safe |
|---|---|---|
| Normalizer | `map_cut_separator`, `normalize_continuation` | Per-char normalizers (BERT, NFC/NFD/NFKC/NFKD, Lowercase, StripAccents, Nmt, ByteLevel), `Precompiled` when it keeps ASCII letters and the space as single chars, `Replace` with a literal that can't touch the cut or replaces the space by one char, or with `" {2,}"` / `\s+`, `Strip`, `Prepend` (skipped at the start of a segment) |
| Added tokens | `AddedVocabulary::supports_cut` | No pattern in the matchers ends with an ASCII letter or can match across the separator |
| Pre-tokenizer | `map_cut` | BERT, Whitespace(Split), ByteLevel, Metaspace, Punctuation, Digits, CharDelimiterSplit, Sequence, `Split` with a literal or with an allowlisted regex (GPT-2, cl100k, o200k, Qwen 2, DeepSeek, BLOOM, CLIP, ...) |
| Model | `supports_cut` | Any deterministic model when the cut is a split boundary; BPE with an unknown token inside a split when no merge can join a symbol ending with a letter to one starting with the separator |

When the pre-tokenizer doesn't split at the cut (Llama 2, Mistral, Gemma: no pre-tokenizer
or Metaspace without split), the cut falls inside a split, and only BPE qualifies: its merge
table is checked once (and cached) for a merge that could cross the cut. It needs an unknown
token, since without one unknown chars are dropped and the offsets after them lag.

Segments are encoded as slices of the input that keep global offsets, so offsets come out
right without shifting and offset-dependent stages (Metaspace `First`) behave as in serial.
Word ids are rebased per segment, and post-processing (special tokens, truncation, padding)
runs once on the concatenation.

Two things are checked at every cut, falling back to serial encoding when they fail:

- The rules assume the normalized text around the cut is still an ASCII letter, the
  separator, an ASCII letter. Normalizers can combine the letter after the separator with
  what follows (NFC turns `e` and U+0301 into `é`), so both segments check this shape after
  normalization, each char aligned with its original byte.
- Some normalizers carry alignment errors along a string: `Precompiled` drops the removal of
  a first char, so every following offset lags. Alignments only lag, so the letter before
  the cut must still be aligned with its original byte after pre-tokenization.

If anything errors, encoding falls back to serial to return the same error.

When no cut is provable (or the input has no `a␣b`, like Chinese text), the input is
pre-tokenized serially and only the model runs in parallel over the splits (`Model`). BPE
with dropout encodes serially.

## Coverage

Of 56 `tokenizer.json` files from the Hub, 54 get segments, 7 of them with cuts inside splits
(Llama 2, CodeLlama, TinyLlama, Phi-3, Mistral 7B, Gemma 2 and 3). CamemBERT and mBART-50 use
the `Model` strategy because they have added tokens ending with a letter (`<s>NOTUSED`,
`en_XX`). Run `cargo run --release --example parallel_plans -- <files>` to see the strategy
of any tokenizer.

## Testing

- `tests/parallel_single_encode.rs` compares every field of the encoding with `encode`,
  `encode_char_offsets` and `encode_fast`, with and without special tokens, truncation and
  padding, on adversarial and seeded random inputs, cutting at every possible space. It uses
  `data/*.json` and any `tokenizer.json` in `data/parallel_corpus/`. Set
  `PARALLEL_FUZZ_CASES` for more random inputs.
- `tests/parallel_redteam.rs` builds tokenizers from every normalizer, pre-tokenizer, model
  and added-token flag, alone and combined, plus regression tests for the mismatches it found
  (combining letters after the separator, BPE dropping unknown chars, a stale added-token
  matcher after changing the normalizer).
- `bindings/python/tests/bindings/test_parallel_encoding.py` does the same from Python.

## Performance

Apple M-series, 12 cores, release build, `data/big.txt`, byte offsets (`cargo bench --bench
parallel_single_benchmark`):

| Tokenizer | 100 KB | 500 KB | 1 MB | 4 MB |
|---|---|---|---|---|
| BERT | 2.5x | 4.0x | 4.3x | 4.6x |
| Llama 3 | 3.8x | 4.6x | 4.8x | 4.8x |
| ALBERT (Unigram) | 2.6x | 3.2x | 4.0x | 3.5x |
| RoBERTa | 2.3x | 2.5x | 2.6x | 2.6x |

From Python (`benches/bench_parallel_single.py`, char offsets), 4 MB: GPT-2 3.3x, Qwen 2.5
3.2x, gpt-4o 3.1x, T5 3.7x, Gemma 2 3.6x, Llama 2 6.3x.

Encoding one segment per thread isn't enough to use all cores well, since segments differ in
cost; `segments_per_thread` (default 4) balances the load. Inputs under `min_input_bytes`
(default 32 KB) are encoded serially.

The model is only 10-30% of serial encoding time, which is why cutting the raw text matters:
`cargo run --release --example phase_breakdown -- data/llama-3-tokenizer.json data/big.txt
4000000` times each step. From Python, each model call takes a shared lock in the bindings,
which limits scaling compared to Rust.
