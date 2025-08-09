from collections.abc import Iterable, Iterator
import pickle
import regex as re
import os


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self._vocab = vocab
        # self._merges = {(vocab[first], vocab[second]) for first, second in merges}
        self._ranks = {pair: i for i, pair in enumerate(merges)}
        if special_tokens is None:
            self._special_tokens = []
        else:
            self._special_tokens = special_tokens
        self._byte_to_int = {v: k for k, v in self._vocab.items()}
        # self._merge_map = {first: second for first, second in merges}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _merge(self, pretoken: bytes) -> list[int]:
        if len(pretoken) == 1:
            return [self._byte_to_int[pretoken]]
        tokens = [bytes([tok]) for tok in pretoken]
        while True:
            pairs = {(tokens[i], tokens[i + 1]): i for i in range(len(tokens) - 1)}
            if not pairs:
                break

            # candidate = None
            # curr_rank = float("inf")
            # for pair in pairs:
            #     if pair in self._ranks and self._ranks[pair] < curr_rank:
            #         curr_rank = self._ranks[pair]
            #         candidate = pair

            # if candidate is None:
            #     break

            candidate = min(pairs, key=lambda p: self._ranks.get(p, float("inf")))
            if candidate not in self._ranks:
                break

            i = pairs[candidate]
            tokens = tokens[:i] + [b"".join(candidate)] + tokens[i + 2 :]
        return [self._byte_to_int[tok] for tok in tokens]

    def encode(self, text: str):
        if self._special_tokens:
            # Sort special tokens by length (descending) to handle overlapping tokens
            sorted_special_tokens = sorted(self._special_tokens, key=len, reverse=True)
            special_tok_pat = "|".join(
                [re.escape(special_token) for special_token in sorted_special_tokens]
            )
            chunks = re.split(f"({special_tok_pat})", text)
        else:
            chunks = [text]
        tokens = []
        for chunk in chunks:
            # print(f"chunk: {chunk}")
            if chunk in self._special_tokens:
                tokens.append(self._byte_to_int[chunk.encode("utf-8")])
            else:
                for match in re.finditer(PAT, chunk):
                    pretoken = match.group()
                    pretoken = pretoken.encode("utf-8")
                    # print(f"pretoken: {pretoken}")
                    curr_tokens = self._merge(pretoken)
                    tokens.extend(curr_tokens)
        # print(f"{[self._vocab[tok] for tok in tokens]}")
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            tokens = self.encode(text)
            for tok in tokens:
                yield tok

    def decode(self, ids: list[int]):
        # return "".join([self._vocab[idx].decode("utf-8") for idx in ids])
        bytes_sequence = b"".join([self._vocab[idx] for idx in ids])
        return bytes_sequence.decode("utf-8", errors="replace")


if __name__ == "__main__":
    tokenizer_path = "/home/jinyu/stanford_cs336/assignment1-basics/data/tokenizer/TinyStoriesV2-GPT4-train"
    tokenizer = BPETokenizer.from_files(
        os.path.join(tokenizer_path, "vocab.pkl"),
        os.path.join(tokenizer_path, "merges.pkl"),
        special_tokens=["<|endoftext|>"],
    )
    max_len = -1
    longest_tok = None
    for tok in tokenizer._vocab.values():
        if len(tok) > max_len:
            max_len = len(tok)
            longest_tok = tok
    print(f"Longest token: {longest_tok}")

    tokens = tokenizer.encode("hello world! <|endoftext|> who are you?")
    print(tokens)
    print(tokenizer.decode(tokens))
    # print(tokens)

    # with open("data/owt_valid.txt", "r") as f:
    #     for _ in range(5):
    #         text = next(f)
    #         tokens = tokenizer.encode(text)
    #         print(tokens)
