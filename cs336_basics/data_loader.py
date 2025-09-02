import numpy.typing as npt
import torch
import numpy as np


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    size = dataset.size
    start_idx = torch.randint(0, size - context_length, size=(batch_size,))
    batch_x = torch.stack([torch.tensor(dataset[i:i + context_length], dtype=torch.int64)
                          for i in start_idx]).to(device)
    batch_y = torch.stack([torch.tensor(dataset[i + 1: i + context_length + 1], dtype=torch.int64)
                          for i in start_idx]).to(device)
    return batch_x, batch_y
