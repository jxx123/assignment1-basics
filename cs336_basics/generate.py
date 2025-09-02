import torch
import os
from cs336_basics import transformer, train, tokenizer
import argparse


def generate(model, tokenizer, prompt: str | list[str], max_tokens: int = 1000, temperature=1.0, top_p: float | None = None, device: str = 'cpu'):
    model.eval()
    if isinstance(prompt, str):
        prompt = [prompt]

    ids = [tokenizer.encode(p) for p in prompt]
    max_len = max([len(seq) for seq in ids])
    # TODO: not sure if paddding zeros in the front is the right thing to do.
    ids = [[0] * (max_len - len(seq)) + seq for seq in ids]
    ids = torch.tensor(ids, dtype=torch.int, device=device)

    for _ in range(max_tokens):
        logits = model(ids)  # B, T, V
        logits = logits / temperature
        logits = logits[:, -1, :]  # Get the latest token, B, V
        probs = transformer.softmax(logits, dim=-1)  # B, V
        # print('probs.shape', probs.shape)

        if top_p:
            sorted_probs, sorted_idx = torch.sort(
                probs, dim=-1, descending=True)  # B, V

            cum_probs = torch.cumsum(sorted_probs, dim=-1)  # B, V
            # print('cum_probs', cum_probs)
            mask = cum_probs > top_p
            # shift mask to the right to guarantee at least one token selected
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            masked_sorted_idx = sorted_idx[mask]
            # print('masked_sorted_idx', masked_sorted_idx.shape)
            probs[:, masked_sorted_idx] = 0.0
            # print('probs.sum after mask', probs.sum(dim=-1))
            probs = probs / probs.sum(dim=-1, keepdim=True)  # B, V

        next_ids = torch.multinomial(probs, num_samples=1)  # B, 1
        next_tokens = next_ids.squeeze(-1).numpy().tolist()
        yield tokenizer.decode(next_tokens)
        ids = torch.concat((ids, next_ids), dim=-1)  # B, T + 1
        # print('ids after stack', ids.shape)
        ids = ids[:, - model.context_length:]


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--tokenizer_path", type=str,
                      default="./data/tokenizer/TinyStoriesV2-GPT4-train")
    args.add_argument("--checkpoint_path", type=str,
                      default="./checkpoints/checkpoint_2000.pt")
    args.add_argument("--prompt", type=str,
                      default="hello")
    args.add_argument("--max_tokens", type=int,
                      default=100)
    args.add_argument("--temperature", type=float,
                      default=1.0)
    args.add_argument("--top_p", type=float,
                      default=None)

    args = args.parse_args()

    model = train.load_model_from_checkpoint(args.checkpoint_path)
    tokenizer_path = args.tokenizer_path
    tok = tokenizer.BPETokenizer.from_files(
        os.path.join(tokenizer_path, "vocab.pkl"),
        os.path.join(tokenizer_path, "merges.pkl"),
        special_tokens=["<|endoftext|>"],
    )
    for text in generate(model, tok, args.prompt, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p):
        print(text[0], end="")
