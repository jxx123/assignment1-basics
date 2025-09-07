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
        self.weight = nn.Parameter(w)  # out, in

    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


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
        w = torch.empty(num_embeddings, embedding_dim,
                        dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(w, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(w)

    def forward(self, x):
        return self.weight[x]
        # one_hot = rearrange(x, "... -> ... 1") == rearrange(
        #     torch.arange(self.num_embeddings), "num_embed -> 1 num_embed"
        # )
        # return einsum(
        #     one_hot.float(), self.weight, "... num_embed, num_embed d_model -> ... d_model"
        # )


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(
            d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # sum over the last dimension (d_model)
        rms = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True) /
                         self.d_model + self.eps)  # batch, seq, 1
        out = x / rms * self.weight
        return out.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = round(int(8 / 3 * d_model) / 64) * 64
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        def swish(x): return x * torch.sigmoid(x)
        x1 = swish(self.w1(x)) * self.w3(x)  # ..., d_ff
        x2 = self.w2(x1)
        return x2


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        assert d_k % 2 == 0, "d_k must be an even number."
        self.max_seq_len = max_seq_len
        self.device = device
        # Precompute the cos and sin matrix
        num = torch.arange(max_seq_len, device=device, dtype=dtype)
        den = theta ** (- 2 * (torch.arange(1, d_k // 2 +
                        1, device=device, dtype=dtype) - 1) / d_k)
        theta_i_k = einsum(num, den, "max_seq, half_d -> max_seq half_d")
        cos = torch.cos(theta_i_k)
        sin = torch.sin(theta_i_k)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x, token_positions=None):
        T, d = x.shape[-2], x.shape[-1]
        assert x.shape[-1] == self.d_k, "embedding dimension mismatch"
        assert x.shape[-2] <= self.max_seq_len, "sequence length is too large"
        # cos0, cos0, cos1, cos1 ...
        cos = rearrange(torch.stack(
            [self.cos, self.cos], dim=-1), "max_seq d n -> max_seq (d n)")
        sin = rearrange(torch.stack(
            [self.sin, self.sin], dim=-1), "max_seq d n -> max_seq (d n)")
        x_sin = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1)
        x_sin = rearrange(x_sin, "... d n -> ... (d n)")
        if token_positions is not None:
            return x * cos[token_positions, :] + x_sin * sin[token_positions, :]

        return x * cos[:T, :] + x_sin * sin[:T, :]


def softmax(x, dim: int):
    max_x = x.max(dim=dim, keepdim=True)
    x = x - max_x.values
    return torch.exp(x) / torch.exp(x).sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q, k, v, mask):
    d_k = q.shape[-1]
    attn = einsum(q, k, "... m d_k, ... n d_k -> ... m n") / np.sqrt(d_k)
    m = torch.where(mask, 0, -float('inf'))
    attn = attn + m
    attn = softmax(attn, -1) @ v
    return attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        if theta:
            self.rope = RoPE(theta, d_model // num_heads, max_seq_len)
        else:
            self.rope = None
        if max_seq_len is None:
            max_seq_len = 1000
        mask = torch.arange(max_seq_len)[
            :, None] >= torch.arange(max_seq_len)[None, :]
        # print('mask', mask)
        self.register_buffer('mask', mask, persistent=False)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        T = x.shape[-2]
        Q = rearrange(self.q_proj(
            x), "... T (h d) -> ... h T d", h=self.num_heads)
        K = rearrange(self.k_proj(
            x), "... T (h d) -> ... h T d", h=self.num_heads)
        V = rearrange(self.v_proj(
            x), "... T (h d) -> ... h T d", h=self.num_heads)
        if self.rope:
            Q = self.rope(Q, token_positions=token_positions)
            K = self.rope(K, token_positions=token_positions)
        mask = self.mask[:T, :T]
        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(attn, "... h T d -> ... T (h d)")
        return self.output_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, max_seq_len, theta: float = 10000.0, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(
            d_model, d_ff, num_heads, context_length, theta=rope_theta) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, ids):
        x = self.token_embeddings(ids)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


if __name__ == "__main__":
    # linear = Linear(3, 5, dtype=torch.bfloat16, device=torch.device("cuda"))
    # print(linear.state_dict())
    # print(linear.state_dict()["W"].device)
    # print(linear.W.dtype)

    # embed = Embedding(3, 5)
    # y = embed(torch.randint(0, 2, (2, 6)))

    # rmsnorm = RMSNorm(5)
    # x = torch.randn(10, 3, 5)
    # y = rmsnorm(x)

    # swiglu = SwiGLU(3, d_ff=8)
    # x = torch.randn((10, 3))
    # y = swiglu(x)
    # print(swiglu.state_dict().keys())
    # print(y.shape)

    # rope = RoPE(10000, 6, 10)
    # x = torch.randn((10, 4, 6))
    # token_positions = (torch.ones(10)[:, None]
    #                    * torch.arange(4)[None, :]).int()
    # print(token_positions)

    # y = rope(x, token_positions)
    # print(rope.cos.shape)
    # print(rope.sin.shape)
    # print(y.shape)

    # x = torch.randn(10, 3)
    # y = softmax(x, 1)
    # print(y)

    # q = torch.randn(10, 3)
    # k = torch.randn(5, 3)
    # v = torch.randn(5, 5)
    # mask = torch.randint(0, 2, size=(10, 5)) == 1
    # print(mask)
    # print(mask.shape)
    # attn = scaled_dot_product_attention(q, k, v, mask)
    # print(attn)

    # x = torch.randn(10, 5, 6)
    # mhsa = MultiHeadSelfAttention(
    #     d_model=6, num_heads=3, theta=1000, max_seq_len=100)
    # y = mhsa(x)
    # print(y)

    # block = TransformerBlock(8, 12, 2, 10, theta=10000)
    # print(block.state_dict())

    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 6400
    rope_theta = 10000
    lm = TransformerLM(vocab_size, context_length, d_model,
                       num_layers, num_heads, d_ff, rope_theta)
    parameter_count = 0

    def count_parameters(model):
        """Count total number of parameters in the model"""
        return sum(p.numel() for p in model.parameters())

    total_params = count_parameters(lm)
    print(f"Total parameters: {total_params:,}")
