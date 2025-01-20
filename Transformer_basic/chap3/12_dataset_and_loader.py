# 89p
# 데이터셋 보통 DB의 테이블과 비슷
'''
Dataset 클래스를 사용하여 데이터 샘플을 정제, 그것들을 재정의(오버라이딩?) 하여 사용
__init__(self, data, *arg, **kwargs): 학습에 필요한 형태로 전처리
__getitem__(self, index): 학습을 진행할 때 사용되는 하나의 행을 불러오는 과정, __init__에서 처리한것을 가져오며 샘플과 정답을 반환
__len__(self): 학습에 사용된 전체 데이터 세트 개수 반환
'''

'''
DataLoader는 데이터세트에 저장된 데이터를 어떻게 불러와 활용할지 정의
batch_size, shuffle(데이터 순서변경), num_workers(데이터 로드 프로세스 수)
'''

# 이 둘을 활용하여 다중 선형 회귀를 구현한다.

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import torch_directml

dml = torch_directml.device()

train_x = torch.FloatTensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]).to(dml)
train_y = torch.FloatTensor([[0.1, 1.5], [1, 2.8], [1.9, 4.1], [2.8, 5.4], [3.7, 6.7], [4.6, 8]]).to(dml)

train_dataset = TensorDataset(train_x, train_y)  # Tensordataset은 초기화 값을 *arg로 받아서 여러 데이터 입력 가능
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                              drop_last=True)  # drop last는 배치하고 남은 자투리 제거, 데이터 섞기 o, 배치 크기 2

model = nn.Linear(2, 2, bias=True, device=dml)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(20000):
    cost = 0.0

    for batch in train_dataloader:
        x, y = batch
        output = model(x)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost/len(train_dataloader)

    if(epoch + 1) % 1000 == 0:
        print(f'Epoch : {epoch + 1:4d}, Model: {list(model.parameters())}, Cost: {cost:.4f}')


# 지금 실제로 한 일
# 데이터셋 제작
# 데이터 로더로 가져오는 옵션 설정
# 모델/오차함수/최적화설정
# 반복
# 반복 내부에서 배치로 쪼개서 반복, 배치는 데이터로더로 가져옴
# 모델에 넣고 오차구하고 가중치 갱신하고 cost에 loss 더하고, 배치 다 끝나면 배치 크기로 나눔