import torch
from torch_directml import device

DEVICE = device()

mask = (torch.triu(torch.ones((3, 3), device=DEVICE))).transpose(0, 1) # == 1).transpose(0, 1)
print(mask)
mask = mask.float()
print(mask)
mask = mask.masked_fill(mask == 0, float("-inf"))
print(mask)
mask = mask.masked_fill(mask == 1, float(0.0))
print(mask)
