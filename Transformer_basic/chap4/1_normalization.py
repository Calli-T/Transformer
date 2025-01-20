import torch
from torch import nn

x = torch.FloatTensor(
    [
        [-0.6577, -0.5797, 0.6360],
        [0.7382, 0.2145, 1.523],
        [0.2432, 0.5662, 0.322]
    ]
)

print(nn.BatchNorm1d(3)(x))
print(nn.LayerNorm(3)(x))
print(nn.InstanceNorm1d(3)(x))
print(nn.GroupNorm(1, 3)(x))

# 채널이 포함된 데이터를 정규화할 땐 BatchNorm2d를 사용한다고 한다
# 2D는 4D 입력 데이터
# 3d는 5D 입력데이터
# 이게 무슨 소린지는 나중에 알아보도록 하자
# 저 3x3벡터도 뭐가 샘플이고 뭐가 차원이고 뭐가 채널이고가 없다
# 입력하는 정보는 feature수다
'''
x = torch.FloatTensor(
    [
        [-0.6577, -0.5797, 0.6360, -0.6577, -0.5797, 0.6360],
        [0.7382, 0.2145, 1.523, 0.7382, 0.2145, 1.523],
        [0.2432, 0.5662, 0.322, 0.2432, 0.5662, 0.322]
    ]
)
print(nn.BatchNorm1d(6)(x))
'''
# 이 예시를 보면 가장 작은 리스트 단위가 feature혹은 dimension인듯
