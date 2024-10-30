from blocks import *

import random
import torch
from math import log as ln

# hyper
NOISE_EMBEDDING_SIZE = 32


# 사인파 임베딩 함수
def sinusoidal_embedding(var):
    # 주파수 최대 스케일링 계수 f = ln(1000)/(L-1)
    freq = torch.linspace(ln(1.0), ln(1000.0), NOISE_EMBEDDING_SIZE // 2)
    angular_speeds = 2.0 * torch.pi * freq
    embedding = torch.concat([torch.sin(angular_speeds * var), torch.cos(angular_speeds * var)],
                             dim=2)
    return embedding.permute(2, 0, 1)  # NCHW 규격을 지키도록 다시 코딩해보자


# 얘는 float list를 받습니다
def nchw_tensor_sinusoidal_embedding(variances):
    if torch.is_tensor(variances):
        variances = variances.cpu().numpy()

    embeddings = []
    for var in variances:
        # print(var.shape)
        embeddings.append(sinusoidal_embedding(torch.tensor([[[var]]], dtype=torch.float32)))
        # embeddings.append(sinusoidal_embedding(torch.tensor([[[var]]], dtype=torch.float32)))
    # print (torch.stack(embeddings, dim=0).shape)

    return torch.stack(embeddings, dim=0).to(device)


# --------------------U-Net--------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 이하 레이어
        # 각 계층들, upblock의 resiblock은 최초 1회 downblock에 의해 정해진다
        self.upsampling = nn.Upsample(scale_factor=IMAGE_SIZE, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding="same")

        self.down1 = DownBlock(64, 32, 2).to(device)
        self.down2 = DownBlock(32, 64, 2).to(device)
        self.down3 = DownBlock(64, 96, 2).to(device)
        self.skips_blocks = []

        self.residual1 = ResidualBlock(96, 128).to(device)
        self.residual2 = ResidualBlock(128, 128).to(device)

        self.up1 = UpBlock([224, 192], 96, 2).to(device)
        self.up2 = UpBlock([160, 128], 64, 2).to(device)
        self.up3 = UpBlock([96, 64], 32, 2).to(device)

        #
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding="same")
        self.conv2.weight = torch.nn.init.zeros_(self.conv2.weight)

    def forward(self, noise_variances, noisy_images):
        '''
        print("noise_variance: " + str(len(noise_variances)))
        print("noise sinusoidal embedding: " + str(nchw_tensor_sinusoidal_embedding(noise_variances).shape))
        print("noise upsampled: " + str(self.upsampling(nchw_tensor_sinusoidal_embedding(noise_variances)).shape))
        print("noisy images: " + str(noisy_images.shape))
        print("noise images feature map: " + str(self.conv(noisy_images).shape))
        '''

        upsampled = self.upsampling(nchw_tensor_sinusoidal_embedding(noise_variances))
        convoluted_noisy = self.conv1(noisy_images)
        # print(convoluted_noisy.shape)
        # print(upsampled.shape)
        x = torch.concat([upsampled, convoluted_noisy], dim=1)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        # self.skips_blocks.append(self.down1.get_skips())
        # self.skips_blocks.append(self.down2.get_skips())
        # self.skips_blocks.append(self.down3.get_skips())

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
