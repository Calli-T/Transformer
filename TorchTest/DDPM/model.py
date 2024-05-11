import torch
from torch import nn

from dataloader import *


# 긴 소수점 표기용
# torch.set_printoptions(precision=8)
def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = 0.02
    max_signal_rate = 0.95

    start_angle = torch.acos(torch.Tensor([max_signal_rate]))
    end_angle = torch.acos(torch.Tensor([min_signal_rate]))

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates


# --------------------blocks--------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding="same")  # bias=False)
        # self.acti_func = nn.ReLU()  # inplace=True) # inplace 옵션은 True일 때, 입력 텐서를 직접수정
        self.conv1_acti_func = nn.Hardswish()  # 토치에 swish 함수가 없더라, 얘는 같은 기반이지만 시그모이드를 계산비용 문제로 치워버리고, 조각별 선형 아날로그로 대체하였다
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")

        # expansion 생략됨
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                # GDL 서적쪽은 bias True(default), 트랜스포머 서적쪽은 False가 기본이었음
                # nn.BatchNorm2d(self.out_channels)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.conv1_acti_func(out)
        out = self.conv2(out)
        out += self.shortcut(x)

        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth):
        super().__init__()

        self.block_depth = block_depth
        self.skips = []

        # 잔차 블럭들 추가
        self.residuals = []
        self.residuals.append(ResidualBlock(in_channels, out_channels))
        for _ in range(block_depth - 1):
            self.residuals.append(ResidualBlock(out_channels, out_channels))
        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        for i in range(self.block_depth):
            x = self.residuals[i](x)
            self.skips.append(x)
        x = self.avgpool(x)

        return x

    def get_skips(self):
        return self.skips


class UpBlock(nn.Module):
    def __init__(self, out_channels, block_depth, skips):
        super().__init__()

        self.block_depth = block_depth
        self.skips = skips

        # 업샘플링
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # keras에서는 align옵션이 따로 있는지모름

        # 잔차 블럭들 추가 전에 concat 이후 채널 값을 맞춰줄 채널 개수 가져옴
        self.additional_channels = []
        for i in range(block_depth):
            self.additional_channels.append(skips[i].shape[1])  # NCHW format에 의거한 채널의 위치
        self.additional_channels.reverse()

        # 잔차 블럭들 추가
        self.residuals = []
        for idx, c in enumerate(self.additional_channels):
            self.residuals.append(ResidualBlock(out_channels + c, out_channels))

    def forward(self, x):
        x = self.upsampling(x)
        for i in range(self.block_depth):
            x = torch.concat([x, self.skips.pop()], dim=1)
            x = self.residuals[i](x)


# down block의 구현을 어떻게 할 것인가 생각해보자
# 파이토치의 클래스식 잔차블럭을 그대로 가져다 업 다운 블럭에 박을 수 있을것인가?
# d = DownBlock(3, 64, 2)
# u = UpBlock(64, 2, torch.tensor([[[[2]]], [[[2]]]]))
# --------------------U-Net--------------------

'''
train_dataloader = getDataLoader("./datasets")
test_block = ResidualBlock(3, 64)
for batch in train_dataloader:
    print(test_block.forward(batch))
'''

'''
# 오프셋 코사인 확산 스케줄 테스트용(전체 step 수 1000으로 설정)
T = 1000
diffusion_times = torch.FloatTensor([x / T for x in range(T)])

(offset_cosine_noise_rates, offset_cosine_signal_rates,
 ) = offset_cosine_diffusion_schedule(diffusion_times)

print(offset_cosine_noise_rates)
'''

# class DDPM(nn.Module):
#     def __init__(self):
#         super().__init__()
