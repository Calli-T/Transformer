import torch
import torch.nn.functional as F

# 3차원 텐서 생성
x = torch.randn(2, 3, 4)

# 마지막 차원에 양쪽으로 1씩 패딩
pad = (1, 1)
y = F.pad(x, pad)
print(y.shape)

# 마지막 두 차원에 대해 각각 (1, 1), (2, 2)만큼 패딩
pad = (1, 1, 2, 2)
y = F.pad(x, pad)
print(y.shape)

# 마지막 세 차원에 대해 각각 (0, 1), (2, 1), (3, 3)만큼 패딩
pad = (0, 1, 2, 1, 3, 3)
y = F.pad(x, pad)
print(y.shape)