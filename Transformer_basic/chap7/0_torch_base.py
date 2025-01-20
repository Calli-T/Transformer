import torch
import math
#print(torch.arange(50).unsqueeze(1))
print(torch.exp(torch.tensor([0, 1, 2]))) # 자연로그의 밑을 n배
print(torch.arange(0, 128, 2))
print(math.log(10000))
print(-math.log(10000.0) / 128) # -ln(10000)/128
print(math.log(2)) # ln 2