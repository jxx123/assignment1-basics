from collections.abc import Iterable, Iterator
import pickle
import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def make_pattern(special_tokens=None):
    base_pat = PAT
    if not special_tokens:
        return base_pat
    # Escape special characters in tokens and sort by length (longest first)
    escaped_tokens = sorted(
        [re.escape(token) for token in special_tokens], key=len, reverse=True
    )
    # Combine special tokens with base pattern
    return "|".join(escaped_tokens + [base_pat])


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str):
        pat = make_pattern(special_tokens=self._special_tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, b):
        pass
