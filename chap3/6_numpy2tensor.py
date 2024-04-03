import torch
import numpy as np

ndarray = np.array([1, 2, 3], dtype=np.uint8)
print(torch.tensor(ndarray))
print(torch.Tensor(ndarray))
print(torch.from_numpy(ndarray))
# 방식이 3개

print()
print(torch.IntTensor(ndarray)) # 얘는 int32로 해준다

# 53p