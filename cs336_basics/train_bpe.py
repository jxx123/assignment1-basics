import os
from typing import BinaryIO
import argparse
import regex as re
from collections import defaultdict
import sys
import multiprocessing


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = ["<|endoftext|>"]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(corpus: str, special_tokens: list[str]) -> dict[tuple, int]:
    """
    Pre-tokenize the corpus by counting occurrences of each pre-token.
    """
    docs = re.split(
        "|".join([re.escape(special_token) for special_token in special_tokens]), corpus
    )
    token_stats = defaultdict(int)
    for doc in docs:
        for match in re.finditer(PAT, doc):
            tok = match.group()
            tok_ids = tok.encode("utf-8")
            token_stats[tuple(tok_ids)] += 1
    return token_stats


def compute_pair_freq(token_stats):
    pair_freq = defaultdict(int)
    for tokens, count in token_stats.items():
        for first, second in zip(tokens, tokens[1:]):
            pair_freq[(first, second)] += count
    return pair_freq


def merge_tokens(
    tokens: tuple, pair_to_merge: tuple[int, int], new_token_id: int
) -> tuple:
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair_to_merge:
            new_tokens.append(new_token_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return tuple(new_tokens)


def train_bpe(
    token_stats: dict[tuple, int],
    vocab_size: int,
    special_tokens: list[str],
    num_iters: int = sys.maxsize,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    base_vocab_size = 256
    vocab = {i: bytes([i]) for i in range(base_vocab_size)}
    for i, special_token in enumerate(special_tokens):
        vocab[i + base_vocab_size] = special_token.encode("utf-8")

    new_token_id = base_vocab_size + len(special_tokens)

    merges = []
    it = 0

    # Initialize pair frequencies once
    pair_freq: dict[tuple[int, int], int] = compute_pair_freq(token_stats)

    while new_token_id < vocab_size and it < num_iters:
        # print(f"Iter {it}: ")
        # print(f"pair_freq: {pair_freq}")
        if pair_freq:
            # Tie-breaking using token IDs directly in ascending order
            pair_to_merge = max(
                pair_freq, key=lambda x: (pair_freq[x], (vocab[x[0]], vocab[x[1]]))
            )

            vocab[new_token_id] = vocab[pair_to_merge[0]] + vocab[pair_to_merge[1]]
            merges.append((vocab[pair_to_merge[0]], vocab[pair_to_merge[1]]))
            # print(
            #     f"pair to merge: {(vocab[pair_to_merge[0]], vocab[pair_to_merge[1]])}"
            # )
        else:
            # no pairs to merge
            print("No more pairs to merge.")
            break

        # Perform merge and incrementally update pair frequencies
        # print("Merging tokens ...")
        new_token_stats = {}

        # Only update frequencies for token sequences that contain the pair to merge
        for tokens, count in token_stats.items():
            new_tokens = merge_tokens(tokens, pair_to_merge, new_token_id)
            new_token_stats[new_tokens] = count

            # If the token sequence changed, update pair frequencies
            if new_tokens != tokens:
                # Remove old pair frequencies for this token sequence
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_freq[pair] -= count
                    if pair_freq[pair] == 0:
                        del pair_freq[pair]

                # Add new pair frequencies for the merged token sequence
                for i in range(len(new_tokens) - 1):
                    pair = (new_tokens[i], new_tokens[i + 1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + count

        new_token_id += 1
        it += 1
        token_stats = new_token_stats
        # print("Merging is done.")
    return vocab, merges


def train_pipeline(
    input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4
):
    split_special_token_bytes = special_tokens[0].encode("utf-8")
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token_bytes)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    with multiprocessing.Pool() as pool:
        token_stats_list = pool.starmap(
            pretokenize, [(chunk, special_tokens) for chunk in chunks]
        )
    # print("Multiprocess is done!")

    token_stats = defaultdict(int)
    for stats in token_stats_list:
        for k, v in stats.items():
            token_stats[k] += v

    vocab, merges = train_bpe(token_stats, vocab_size, special_tokens)
    return vocab, merges


## Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-tokenization example for chunking a file."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/TinyStoriesV2-GPT4-valid.txt",
        help="Path to the file to be chunked.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1000,
        help="Size of the vocabulary",
    )
    parser.add_argument(
        "--special_tokens",
        type=list,
        default=SPECIAL_TOKENS,
        help="Special Tokens",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of processes to use for pre-tokenization.",
    )

    args = parser.parse_args()
    vocab, merges = train_pipeline(
        args.file_path,
        args.vocab_size,
        args.special_tokens,
        num_processes=args.num_processes,
    )
    print(vocab)
    print(merges)
