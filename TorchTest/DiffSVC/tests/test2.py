import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(x.repeat(2, 3))
print(x.repeat(3, 1, 1))

print(x.shape[-1])
print(x.repeat(3, 2).shape)
print(x.repeat(3, 2).shape[-1])
