# 미완성

import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import random_split

import torch_directml

device = torch_directml.device() if torch_directml.is_available() else "cpu"
print(device)


class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:, 0].values
        self.y = df.iloc[:, 1].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index] ** 2, self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y

    def __len__(self):
        return self.length


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1, device=device)

    def forward(self, x):
        x = self.layer(x)
        return x


'''
train_dataset = CustomDataset("nonlinear.csv")
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
'''
dataset = CustomDataset('nonlinear.csv')
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - validation_size

# ---------------------------------------------------------
train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
# random_split(데이터셋, [길이1, 길이2...])
# 길이 배열 데이터 총합과 같아야함
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, drop_last=True)
# ---------------------------------------------------------

model = CustomModel()
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# 실행
for epoch in range(10000):
    cost = 0.0

    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost / len(train_dataloader)

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch: {epoch + 1:4d}, Model: {list(model.parameters())}, Cost: {cost:3f}')

# with torch.no_grad()와 함께라면, 자동미분 없는 검증 가능
with torch.no_grad():
    model.eval()
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)


        outputs = model(x)
        print(f'x: {x}')
        print(f'y: {y}')
        print(f'outputs: {outputs}')
        print('--------------------------------')

# 모델 저장과 특정 시점의 모델 저장
'''
torch.save(model, './models/model_14.pt')
torch.save(model.state_dict(), './models/modeL-14_state_dict.pt')
'''

# 108p
