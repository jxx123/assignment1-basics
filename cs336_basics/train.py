import torch
from einops import einsum, rearrange


def cross_entropy(logits, targets):
    # logits: b, t, v
    # targets: b, t
    logits = logits - logits.max(dim=-1, keepdim=True)  # b, t, v
    one_hot = (targets[..., None] == torch.arange(logits.shape[-1])).int()
