# 79p
import torch
from torch import optim

x = torch.FloatTensor(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
     [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]])

y = torch.FloatTensor([[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
                       [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
                       [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]])

# 하이퍼파라미터 설정, requires_grad는 파이토치 자동 미분 ㅋㅋㅋㅋ
weight = torch.zeros(1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)
learning_rate = 0.001

# 최적화 기법은 확률적 경사 하강법
# 매개변수는 가중치들, 학습률, 기타등등
optimizer = optim.SGD([weight, bias], lr=learning_rate)


for epoch in range(10000):
    # 가설과 손실 함수 선언
    hypothesis = x * weight + bias
    cost = torch.mean((hypothesis - y) ** 2)
    
    '''
    optimizer.zero_grad() 매개변수의 기술기를 9으로 초기화, weight += x로 저장되므로 미리 0으로 초기화해야함
    cost.backward()는 역전파, optimizer 변수에 포함된 매개변수들의 기울기가 새로 계산됨
    optimizer.step()는 계산한 매개변수들 반영
    '''
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 학습 기록 출력
    # .item()으로 텐서들 꺼내야하는듯?
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch : {epoch + 1:4d}, Weight: {weight.item():.3f}, Bias: {bias.item():.3f}, Cost: {cost:.3f}')