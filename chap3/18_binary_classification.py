import torch
from torch import optim
from torch import nn
import pandas as pd
import torch_directml
from torch.utils.data import Dataset, DataLoader, random_split

device = torch_directml.device()


class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        # 데이터 프레임에서 한 열로 자르면 리스트가 나오나?
        self.x1 = df.iloc[:, 0].values
        self.x2 = df.iloc[:, 1].values
        self.x3 = df.iloc[:, 2].values
        self.y = df.iloc[:, 3].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x1[index], self.x2[index], self.x3[index]])
        y = torch.FloatTensor([int(self.y[index])])
        return x, y

    def __len__(self):
        return self.length


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.layer1_linear = nn.Linear(3, 1)
        #self.layer1_activation = nn.Sigmoid()
        self.layer = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.layer(x)
        #x = self.layer1_activation(self.layer1_linear(x))

        return x


dataset = CustomDataset('binary_cross_normalized.csv')
datasize_size = len(dataset)
train_size = int(datasize_size * 0.8)
validation_size = int(datasize_size * 0.1)
test_size = datasize_size - train_size - validation_size

train_datatset, validation_datatset, test_datatset = random_split(dataset,
                                                                  [train_size, validation_size, test_size],
                                                                  torch.manual_seed(42))

train_dataloader = DataLoader(train_datatset, batch_size=32, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(validation_datatset, batch_size=4, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_datatset, batch_size=4, shuffle=True, drop_last=True)

model = CustomModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

print(train_datatset)

for epoch in range(50000):
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

    cost = cost / len(train_dataloader)

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch: {epoch + 1:4d}, Model:{list(model.parameters())}, Cost: {cost:.3f}')

with torch.no_grad():
    model.eval()

    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        print(outputs)
        print(outputs >= torch.FloatTensor([0.5]).to(device))
        print('------------------------------------')

#128p 부터, 결과가 이상한편, binary.csv가 책에 있는게 아님