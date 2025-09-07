import torch
import os
from cs336_basics import transformer, train, tokenizer
import argparse


def generate(model, tokenizer, prompt: str, max_tokens: int = 1000, temperature=1.0, top_p: float | None = None, device: str = 'cpu'):
    # model.to(device)
    model.eval()

    # Encode the prompt to token ids
    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.int, device=device).unsqueeze(
        0)  # Add batch dimension: 1, T

    for _ in range(max_tokens):
        logits = model(ids)  # 1, T, V
        logits = logits / temperature
        logits = logits[0, -1, :]  # Get the latest token, remove batch dim: V
        probs = transformer.softmax(logits, dim=-1)  # V

        if top_p:
            sorted_probs, sorted_idx = torch.sort(
                probs, dim=-1, descending=True)  # V

            cum_probs = torch.cumsum(sorted_probs, dim=-1)  # V
            mask = cum_probs > top_p
            # shift mask to the right to guarantee at least one token selected
            mask[1:] = mask[:-1].clone()
            mask[0] = False
            masked_sorted_idx = sorted_idx[mask]
            probs[masked_sorted_idx] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)  # V

        next_id = torch.multinomial(probs, num_samples=1)  # 1
        next_token = next_id.item()
        if next_token == tokenizer.eot_token:
            return
        yield tokenizer.decode([next_token])
        ids = torch.concat((ids, next_id.unsqueeze(
            0).to(device)), dim=-1)  # 1, T + 1
        ids = ids if ids.shape[1] < model.context_length else ids[:, -
                                                                  model.context_length:]


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
    args.add_argument("--device", type=str,
                      default="cpu")
    args = args.parse_args()

    model = train.load_model_from_checkpoint(
        args.checkpoint_path, compile=True, device=args.device)

    tokenizer_path = args.tokenizer_path
    tok = tokenizer.BPETokenizer.from_files(
        os.path.join(tokenizer_path, "vocab.pkl"),
        os.path.join(tokenizer_path, "merges.pkl"),
        special_tokens=["<|endoftext|>"],
    )
    for text in generate(model, tok, args.prompt, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, device=args.device):
        print(text, end="")
