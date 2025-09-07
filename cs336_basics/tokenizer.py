from collections.abc import Iterable, Iterator
import pickle
import regex as re
import os
import argparse
import numpy as np

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self._vocab = vocab
        self._ranks = {pair: i for i, pair in enumerate(merges)}
        if special_tokens is None:
            self._special_tokens = []
        else:
            self._special_tokens = special_tokens
        self._byte_to_int = {v: k for k, v in self._vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    @property
    def eot_token(self):
        return self._byte_to_int[b"<|endoftext|>"]

    def _merge(self, pretoken: bytes) -> list[int]:
        if len(pretoken) == 1:
            return [self._byte_to_int[pretoken]]
        tokens = [bytes([tok]) for tok in pretoken]
        while True:
            pairs = {(tokens[i], tokens[i + 1]): i for i in range(len(tokens) - 1)}
            if not pairs:
                break

            candidate = min(
                pairs, key=lambda p: self._ranks.get(p, float("inf")))
            if candidate not in self._ranks:
                break

            i = pairs[candidate]
            tokens = tokens[:i] + [b"".join(candidate)] + tokens[i + 2:]
        return [self._byte_to_int[tok] for tok in tokens]

    def encode(self, text: str):
        if self._special_tokens:
            # Sort special tokens by length (descending) to handle overlapping tokens
            sorted_special_tokens = sorted(
                self._special_tokens, key=len, reverse=True)
            special_tok_pat = "|".join(
                [re.escape(special_token)
                 for special_token in sorted_special_tokens]
            )
            chunks = re.split(f"({special_tok_pat})", text)
        else:
            chunks = [text]
        tokens = []
        for chunk in chunks:
            if chunk in self._special_tokens:
                tokens.append(self._byte_to_int[chunk.encode("utf-8")])
            else:
                for match in re.finditer(PAT, chunk):
                    pretoken = match.group()
                    pretoken = pretoken.encode("utf-8")
                    curr_tokens = self._merge(pretoken)
                    tokens.extend(curr_tokens)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            tokens = self.encode(text)
            for tok in tokens:
                yield tok

    def decode(self, ids: list[int]):
        bytes_sequence = b"".join([self._vocab[idx] for idx in ids])
        return bytes_sequence.decode("utf-8", errors="replace")

    def encode_to_file(self, input_file_path: str, output_file_path: str, batch_size: int = 50000):
        """
        Optimized encoding that writes tokens directly to file using batched writes.
        Saves tokens in numpy uint16 format for compatibility with data loader.

        Args:
            input_file_path: Path to input text file
            output_file_path: Path to output binary file
            batch_size: Number of tokens to batch before writing (default: 50000)
        """
        with open(input_file_path, "r") as fin:
            with open(output_file_path, "wb") as fout:
                batch = []

                for tok in self.encode_iterable(fin):
                    batch.append(tok)

                    if len(batch) >= batch_size:
                        # Convert to numpy array and write as binary
                        tokens_array = np.array(batch, dtype=np.uint16)
                        tokens_array.tofile(fout)
                        batch = []

                # Write remaining tokens
                if batch:
                    tokens_array = np.array(batch, dtype=np.uint16)
                    tokens_array.tofile(fout)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--tokenizer_path", type=str,
                      default="./data/tokenizer/TinyStoriesV2-GPT4-train")
    args.add_argument("--input_file_path", type=str,
                      default="./data/TinyStoriesV2-GPT4-train.txt")
    args.add_argument("--output_file_path", type=str,
                      default="./data/TinyStoriesV2-GPT4-train.bin")
    args.add_argument("--batch_size", type=int, default=50000,
                      help="Number of tokens to batch before writing (default: 50000)")
    args = args.parse_args()

    tokenizer_path = args.tokenizer_path
    tokenizer = BPETokenizer.from_files(
        os.path.join(tokenizer_path, "vocab.pkl"),
        os.path.join(tokenizer_path, "merges.pkl"),
        special_tokens=["<|endoftext|>"],
    )

    # Use optimized batch method
    # tokenizer.encode_to_file(
    #     args.input_file_path,
    #     args.output_file_path,
    #     batch_size=args.batch_size
    # )

    text = """
    Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars. Tom was very proud of his tall tower. He wanted to make it even taller, so he reached for more blocks.
"Tom, can I have some blocks too?" Lily asked. She wanted to make a bridge for her cars.
"No, these are mine. Go find your own," Tom said. He did not want to share with his sister. He pulled the blocks closer to him.
Lily felt sad and angry. She did not think Tom was being nice. She looked at his tower and had an idea. She decided to pull one of the blocks at the bottom of the tower.
Suddenly, the tower fell down with a loud crash. All the blocks and cars scattered on the floor. Tom and Lily were shocked. They felt the floor shake and heard a rumble. It was an earthquake!
"Mommy! Daddy!" they cried. They were scared and ran to their parents, who were in the kitchen.
"Are you okay, kids?" Mommy asked. She hugged them and checked if they were hurt.
"We're okay, Mommy. But our toys are broken," Lily said.
"I'm sorry, Lily. But toys are not important. You are important. We are safe and together. That's what matters," Mommy said.
Tom felt sorry for what he did. He realized he was selfish and mean to his sister. He saw how scared she was during the earthquake. He wanted to make her happy.
"Lily, I'm sorry I did not share with you. You can have all the blocks you want. I love you, sister," Tom said.
Lily smiled and hugged him. She forgave him and thanked him. She loved him too.
They went back to the living room and cleaned up their toys. They decided to build something together. They made a big house with a garden and a fence. They put their cars and dolls inside. They were happy and proud of their work.
Mommy and Daddy came to see their house. They praised them and gave them a treat. It was a lemon cake. It was sour, but they liked it. They learned that sharing is caring, and that family is sweet.
<|endoftext|>"""

    tokens = tokenizer.encode(text)

    # print(f"Longest token: {max(tokenizer._vocab.values(), key=len)}")

    # tokenizer.encode_iterable()
    print(tokens)
    print(tokenizer.decode(tokens))

    # with open("data/owt_valid.txt", "r") as f:
    #     for _ in range(5):
    #         text = next(f)
    #         tokens = tokenizer.encode(text)
    #         print(tokens)
