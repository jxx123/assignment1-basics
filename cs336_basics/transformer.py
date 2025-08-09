import torch
import torch.nn as nn
import numpy as np
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        w = torch.empty(out_features, in_features, dtype=dtype, device=device)
        std = np.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(w, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W = nn.Parameter(w)  # out, in

    def forward(self, x):
        return einsum(x, self.W, "... in, out in -> ... out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(w, mean=0, std=1, a=-3, b=3)
        self.W = nn.Parameter(w)

    def forward(self, x):
        one_hot = rearrange(x, "... -> ... 1") == rearrange(
            torch.arange(self.num_embeddings), "num_embed -> 1 num_embed"
        )
        return einsum(
            one_hot.float(), self.W, "... num_embed, num_embed d_model -> ... d_model"
        )


if __name__ == "__main__":
    # linear = Linear(3, 5, dtype=torch.bfloat16, device=torch.device("cuda"))
    # print(linear.state_dict())
    # print(linear.state_dict()["W"].device)
    # print(linear.W.dtype)

    embed = Embedding(3, 5)
    y = embed(torch.randint(0, 2, (2, 6)))
