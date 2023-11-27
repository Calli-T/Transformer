import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import torch_directml

device = torch_directml.device() if torch_directml.is_available() else "cpu"
print(device)


# torch dataset을 상속한 사용자 정의 데이터세트
class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:, 0].values
        self.y = df.iloc[:, 1].values
        self.length = len(df)

    # 이차방정식이므로....
    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index] ** 2, self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y

    def __len__(self):
        return self.length


# 모델 초기화와 스탭당 행동정의
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1, device=device)

    def forward(self, x):
        x = self.layer(x)
        return x


# 데이터 셋과 데이터 로더
train_dataset = CustomDataset("nonlinear.csv")
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

# 모델, 손실함수, 최적화
model = CustomModel()
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# 실행
for epoch in range(10000):
    cost = 0.0

    for x, y in train_dataloader:
        # device에 맞는 벡터로
        x = x.to(device)
        y = y.to(device)

        # 값에 넣고
        output = model(x)
        # 손실함수 계산
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost / len(train_dataloader)

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch: {epoch + 1:4d}, Model: {list(model.parameters())}, Cost: {cost:3f}')

# no_grad는 자동 미분 설정 x로 추론에 적합한 상태로 변경
# .eval()로 모델을 평가모드로 변경
# .predict method에 가까운듯?
with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor([[1 ** 2, 5], [5 ** 2, 5], [11 * 2, 11]]).to(device)
    outputs = model(inputs)
    print(outputs)

# 모델 저장과 특정 시점의 모델 저장
torch.save(model, './models/model_14.pt')
torch.save(model.state_dict(), './models/modeL-14_state_dict.pt')
