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


# 모델 구조 불러오기 & 모델 가중치 불러오기
model = torch.load('./models/model_15.pt', map_location=dml)
print(model)

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor([[1 ** 2, 1], [5 ** 2, 5], [11 ** 2, 11]]).to(dml)
    outputs = model(inputs)
    print(outputs)

#  모델 다시 저장
# model 이나 model.state_dict(), 경로 2개가 매개변수
torch.save(model, './models/model_16.pt')
torch.save(model.state_dict(), './models/model-16_state_dict.pt')

'''
모델 전체 파일 있으나 구조를 못볼 경우
load만 하고 에러메시지를 본다음 모델명에 맞춰 클래스를 선언하고
pass로 내용을 채운뒤
print(model)을 하면 모양을 알 수 있다
'''

# ~116p