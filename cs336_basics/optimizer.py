from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import matplotlib.pyplot as plt


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr, 'foo': 1}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # print(group['params'])
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr * grad / math.sqrt(t + 1)
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], weight_decay: float, eps: float = 1e-8):
        defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'eps': eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            params = group['params']
            for p in params:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros(
                    p.shape, dtype=p.dtype, device=p.device))
                v = state.get("v", torch.zeros(
                    p.shape, dtype=p.dtype, device=p.device))
                grad = p.grad.data
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2
                alpha_t = lr * \
                    math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data -= alpha_t * m / (v.sqrt() + eps)
                p.data -= lr * weight_decay * p.data
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1


def get_lr_cosine_schedule(t, lr_max, lr_min, warmup_steps, annealing_steps):
    if t < warmup_steps:
        return t / warmup_steps * lr_max

    if t < annealing_steps:
        lr = lr_min + 0.5 * (1 + math.cos((t - warmup_steps) /
                                          (annealing_steps - warmup_steps) * math.pi)) * (lr_max - lr_min)
        return lr

    return lr_min


def clip_gradient(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return None

    total_norm = torch.sqrt(sum([grad.square().sum() for grad in grads]))

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for grad in grads:
            grad.data = grad.data * clip_coef


if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    print('before', weights.square().sum().sqrt())
    clip_gradient([weights], 1)
    print('after', weights.square().sum().sqrt())
