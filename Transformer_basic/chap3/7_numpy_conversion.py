import torch
import torch_directml

dml = torch_directml.device()
tensor = torch.FloatTensor([1, 2, 3]).to(dml)
ndarray = tensor.detach().cpu().numpy()
# cpu 장치만 가능하다
# detach은 현재 연산 그래프에서 분리된 새로운 텐서 반환
# 새로운 텐서 생성후 넘파이로 변환해야한다.
# 54p

print(ndarray)
print(type(ndarray))