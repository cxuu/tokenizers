import glob
import os

import pytest

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

# Tokenizer files from the Rust crate's test data, when present.
RUST_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tokenizers", "data")
FILES = sorted(
    [
        os.path.join(RUST_DATA, name)
        for name in ["bert-wiki.json", "roberta.json", "llama-3-tokenizer.json", "albert-base-v1-tokenizer.json"]
    ]
    + glob.glob(os.path.join(RUST_DATA, "parallel_corpus", "*.json"))
)
FILES = [f for f in FILES if os.path.exists(f)]


def text(n):
    path = os.path.join(RUST_DATA, "big.txt")
    base = open(path, encoding="utf-8").read() if os.path.exists(path) else "Hello world, héllo 世界! " * 1000
    while len(base) < n:
        base += base
    extra = " naïve 日本語 👍🏽 <|endoftext|> [MASK] <s> x" + " " * 300 + "y " + "=" * 300 + " z " + "1" * 400
    return base[:n] + extra + base[: n // 10]


def assert_same(expected, actual):
    assert actual.ids == expected.ids
    assert actual.tokens == expected.tokens
    assert actual.offsets == expected.offsets
    assert actual.word_ids == expected.word_ids
    assert actual.type_ids == expected.type_ids
    assert actual.attention_mask == expected.attention_mask
    assert actual.special_tokens_mask == expected.special_tokens_mask
    assert actual.sequence_ids == expected.sequence_ids
    assert len(actual.overflowing) == len(expected.overflowing)


@pytest.mark.parametrize("path", FILES, ids=os.path.basename)
@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_identical_to_encode(path, add_special_tokens):
    tokenizer = Tokenizer.from_file(path)
    sequence = text(300_000)
    assert_same(
        tokenizer.encode(sequence, add_special_tokens=add_special_tokens),
        tokenizer.encode_parallel(sequence, add_special_tokens=add_special_tokens),
    )


def test_short_input_and_custom_components():
    class Upper:
        def pre_tokenize(self, pretok):
            pretok.split(lambda i, s: s.split(" ", "removed"))

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(Upper())
    for sequence in ["", "A a", "a " * 50_000]:
        assert_same(tokenizer.encode(sequence), tokenizer.encode_parallel(sequence))
