import torch
import torch_directml

# 텐서 타입지정
tensor = torch.rand((3, 3), dtype=torch.float)
print(tensor)

tensor = torch.rand((5, 5), dtype=float)  # 깡 float으로 하면 float 64
print(tensor)

# 텐서 장치 설정
device = "directml" if torch_directml.is_available() else "cpu" # 가능한가
cpu = torch.FloatTensor([1, 2, 3]) # cpu용 텐서생산
gpu = torch.rand((2, 3), dtype=float).to(torch_directml.device()) # gpu용 텐서생산

print(device)
print(cpu)
print(cpu.device)
print(gpu)
print(gpu.device)

# 차원이동
tensor = gpu.reshape(3, 2)
print(tensor)
print(tensor.shape)

# directml 버전은 이런식으로 쓰면된다
#dml = torch_directml.device()
#gpu = gpu.to(dml)