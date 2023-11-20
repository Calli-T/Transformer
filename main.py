import torch
import torch_directml

dml = torch_directml.device()
# print(dml)

tensor1 = torch.tensor([1]).to(dml)  # Note that dml is a variable, not a string!
tensor2 = torch.tensor([2]).to(dml)

dml_algebra = tensor1 + tensor2
print(dml_algebra.item())

print(torch_directml.is_available())
print(torch_directml.device_name(0))
print(torch_directml.device_count())

print(tensor2.to(dml).device)

# 위는 그래픽 카드 인식 확인, 아래는 추가 패키지 필요할 때 설치하라고 적어둔 내용들

# install numpy pandas tensorboard matplotlib tqdm pyyaml