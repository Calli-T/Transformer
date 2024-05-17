from torch import nn

from dataloader import *

# 긴 소수점 표기용
# torch.set_printoptions(precision=8)

from torchinfo import summary


# --------------------blocks--------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        '''
        momentum과 eps는 keras의 기본 세팅으로 해뒀다
        keras의 center/scale의 False는 정규화 식에서 곱해지는 수, 더해지는 수의 비활성화를 의미한다
        torch의 affine은 z = g(Wu + b) 즉 아핀 변환을 의미하는데, False로 두면 해당 매개변수가 비활성화 되는듯? FC나 Conv나 모두 Affine Transformation을 사용한다
        https://ban2aru.tistory.com/35
        https://stackoverflow.com/questions/73891401/what-does-the-affine-parameter-do-in-pytorch-nn-batchnorm2d
        https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization#symbolic_call
        저 링크들의 정보를 종합했을 때 affine을 끄면 keras의 center/scale을 모두 끄는 효과가 있다
        
        track_running_stats는 일단 꺼보고 생각함 -> 다시 켜봄
        '''
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.99, eps=0.001, affine=False, track_running_stats=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding="same")  # bias=False)
        # self.acti_func = nn.ReLU()  # inplace=True) # inplace 옵션은 True일 때, 입력 텐서를 직접수정
        self.conv1_acti_func = nn.SiLU()  #SiLU가 곧 swish이다 # 토치에 swish 함수가 없더라, 얘는 같은 기반이지만 시그모이드를 계산비용 문제로 치워버리고, 조각별 선형 아날로그로 대체하였다
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
                # GDL 서적쪽은 bias True(default), 트랜스포머 서적쪽은 False가 기본이었음
                # nn.BatchNorm2d(self.out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.bn1(x)
        out = self.conv1(x)
        out = self.conv1_acti_func(out)
        out = self.conv2(out)
        out += residual

        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth=2):
        super().__init__()

        self.block_depth = block_depth
        self.skips = []

        # 잔차 블럭들 추가
        self.residuals = []
        self.residuals.append(ResidualBlock(in_channels, out_channels).to(device))
        for _ in range(block_depth - 1):
            self.residuals.append(ResidualBlock(out_channels, out_channels).to(device))
        self.residuals = nn.ModuleList(self.residuals)  # nn.Sequential(*self.residuals)

        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        for i in range(self.block_depth):
            x = self.residuals[i](x)
            self.skips.append(x)

        # x = self.residuals(x)
        x = self.avgpool(x)

        return x

    def get_skips(self):
        return self.skips


class UpBlock(nn.Module):
    def __init__(self, concat_input_channels, out_channels, block_depth=2):
        super().__init__()

        self.block_depth = block_depth
        self.skips = []

        # self.out_channels = out_channels
        # self.additional_channels = []

        # 업샘플링
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # keras에서는 align옵션이 따로 있는지모름

        # 잔차블럭
        self.residuals = []

        for idx, c in enumerate(concat_input_channels):
            # self.residuals.append(ResidualBlock(out_channels + c, out_channels))
            self.residuals.append(ResidualBlock(c, out_channels).to(device))

        self.residuals = nn.ModuleList(self.residuals)

    def forward(self, x):
        x = self.upsampling(x)
        for i in range(self.block_depth):
            x = torch.concat([x, self.skips.pop()], dim=1)
            x = self.residuals[i](x)

        return x

    def set_skips(self, skips):
        self.skips = skips

        '''
        # legacy 코드는 forward 도중에 upblock 모양을 맞춰줬어야했었음
        # 잔차 블럭들 추가 전에 concat 이후 채널 값을 맞춰줄 채널 개수 가져옴
        for i in range(self.block_depth):
            # print(str(i) + " " + str(skips[i].shape[1]))
            self.additional_channels.append(skips[i].shape[1])  # NCHW format에 의거한 채널의 위치
        self.additional_channels.reverse()

        # 잔차 블럭들 추가
        for idx, c in enumerate(self.additional_channels):
            self.residuals.append(ResidualBlock(self.out_channels + c, self.out_channels))
        '''


# d = DownBlock(3, 64, 2)
# summary(d)
# u = UpBlock([224, 192], 96, 2).to(device)
# summary(u)


# down block의 구현을 어떻게 할 것인가 생각해보자
# 파이토치의 클래스식 잔차블럭을 그대로 가져다 업 다운 블럭에 박을 수 있을것인가?
# d = DownBlock(3, 64, 2)
# u = UpBlock(64, 2, torch.tensor([[[[2]]], [[[2]]]]))
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

# - 2 -
