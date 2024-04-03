import torch
from torch import nn
from torch_directml import device

dml = device()


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1, device=dml)

    def forward(self, x):
        x = self.layer(x)
        return x


model = CustomModel().to(dml)

# 모델 params 가져오는건 2단계를 거쳐서
model_state_dict = torch.load('./models/model-15_state_dict.pt', map_location=dml)
model.load_state_dict(model_state_dict)

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor([[1 ** 2, 1], [5 ** 2, 5], [11 ** 2, 11]]).to(dml)
    outputs = model(inputs)
    print(outputs)

# 요약 torch.save(모델명 or param, 경로)/torch.load(모델명 or param의 경로, map_loation=[device_name])
# params를 불러온 경우 model.load_state_dict(params)