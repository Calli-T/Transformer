import torch
from torch import nn
from torch import optim
import torch_directml

dml = torch_directml.device()

x = torch.FloatTensor(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
     [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]]).to(dml)

y = torch.FloatTensor([[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
                       [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
                       [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]]).to(dml)

model = nn.Linear(1, 1, device=dml)  # bias는 True가 default, dtype=None/시작 2개의 값은 입력과 출력의 벡터 차원 크기
criterion = nn.MSELoss()  # 손실함수
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10000):
    output = model(x)
    cost = criterion(output, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch : {epoch + 1:4d}, Model: {list(model.parameters())}, Cost: {cost:.3f}')

# 선형모델은 모델선언->손실함수선언->최적화함수-> 반복
# 반복 내에서는 output/cost 실행후, zerograd backward step

