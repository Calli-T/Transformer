import torch
import torch_directml

# 파이토치 텐서 장치 변환법 51p
dml = torch_directml.device()
cpu = torch.FloatTensor([1, 2, 3])
# gpu = cpu.cuda()
gpu = cpu.to(dml)
gpu2cpu = gpu.cpu()
# cpu2cpu = cpu.to("cuda")

print(cpu.device)
print(gpu.device)
print(gpu2cpu.device)
