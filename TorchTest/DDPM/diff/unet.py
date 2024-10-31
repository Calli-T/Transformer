import torch
from torch import nn

from math import log as ln

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

# --------------------U-Net--------------------
class UNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.device = hparams['device']
        self.img_size = hparams['IMAGE_SIZE']
        self.NOISE_EMBEDDING_SIZE = hparams['NOISE_EMBEDDING_SIZE']

        # 이하 레이어
        # 각 계층들, upblock의 resiblock은 최초 1회 downblock에 의해 정해진다
        self.upsampling = nn.Upsample(scale_factor=self.img_size, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding="same")

        self.down1 = DownBlock(64, 32, 2)
        self.down2 = DownBlock(32, 64, 2)
        self.down3 = DownBlock(64, 96, 2)
        self.skips_blocks = []

        self.residual1 = ResidualBlock(96, 128)
        self.residual2 = ResidualBlock(128, 128)

        self.up1 = UpBlock([224, 192], 96, 2)
        self.up2 = UpBlock([160, 128], 64, 2)
        self.up3 = UpBlock([96, 64], 32, 2)

        #
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding="same")
        self.conv2.weight = torch.nn.init.zeros_(self.conv2.weight)

    def forward(self, noise_variances, noisy_images):
        upsampled = self.upsampling(self.nchw_tensor_sinusoidal_embedding(noise_variances))
        convoluted_noisy = self.conv1(noisy_images)

        x = torch.concat([upsampled, convoluted_noisy], dim=1)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.residual1(x)
        x = self.residual2(x)

        self.up1.set_skips(self.down3.get_skips())
        self.up2.set_skips(self.down2.get_skips())
        self.up3.set_skips(self.down1.get_skips())

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        x = self.conv2(x)

        return x

    # 사인파 임베딩 함수
    def sinusoidal_embedding(self, var):
        # 주파수 최대 스케일링 계수 f = ln(1000)/(L-1)
        freq = torch.linspace(ln(1.0), ln(1000.0), self.NOISE_EMBEDDING_SIZE // 2)
        angular_speeds = 2.0 * torch.pi * freq
        embedding = torch.concat([torch.sin(angular_speeds * var), torch.cos(angular_speeds * var)],
                                 dim=2)
        return embedding.permute(2, 0, 1)  # NCHW 규격을 지키도록 다시 코딩해보자

    # 얘는 float list를 받습니다
    def nchw_tensor_sinusoidal_embedding(self, variances):
        if torch.is_tensor(variances):
            variances = variances.cpu().numpy()

        embeddings = []
        for var in variances:
            embeddings.append(self.sinusoidal_embedding(torch.tensor([[[var]]], dtype=torch.float32)))

        return torch.stack(embeddings, dim=0).to(self.device)


# print(nchw_tensor_sinusoidal_embedding(torch.rand(5)).shape)

'''
unet = UNet()
train_dataloader, _, _ = getDataLoader("./datasets")
for batch in train_dataloader:
    variances = [random.random() for _ in range(5)]
    print(unet.forward(variances, batch).shape)
    # print(variances)
    # print(unet.forward(variances, batch))
'''
# class DDPM(nn.Module):
#     def __init__(self):
#         super().__init__()

# 하다보니 텐서에 detach를 붙이는것이 권장된다는 경고가 떴다
# 사인파 임베딩이라 뭐...
# detach는 기존 텐서에서 기울기 전파가 안되는 텐서라고한다

# - 3 -
