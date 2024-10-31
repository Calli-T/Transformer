import torch
from torch import nn


# 긴 소수점 표기용
# torch.set_printoptions(precision=8)


# --------------------blocks--------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.99, eps=0.001, affine=False, track_running_stats=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding="same")
        self.conv1_acti_func = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
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
        self.residuals.append(ResidualBlock(in_channels, out_channels))
        for _ in range(block_depth - 1):
            self.residuals.append(ResidualBlock(out_channels, out_channels))
        self.residuals = nn.ModuleList(self.residuals)  # nn.Sequential(*self.residuals)

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
    def __init__(self, concat_input_channels, out_channels, block_depth=2):
        super().__init__()

        self.block_depth = block_depth
        self.skips = []

        # 업샘플링
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # keras에서는 align옵션이 따로 있는지모름

        # 잔차블럭
        self.residuals = []

        for idx, c in enumerate(concat_input_channels):
            self.residuals.append(ResidualBlock(c, out_channels))

        self.residuals = nn.ModuleList(self.residuals)

    def forward(self, x):
        x = self.upsampling(x)
        for i in range(self.block_depth):
            x = torch.concat([x, self.skips.pop()], dim=1)
            x = self.residuals[i](x)

        return x

    def set_skips(self, skips):
        self.skips = skips