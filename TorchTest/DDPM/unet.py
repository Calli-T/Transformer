from blocks import *

# --------------------U-Net--------------------

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.skips_blocks = []
        self.down1 = DownBlock(64, 32, 2)
        self.down2 = DownBlock(32, 64, 2)
        self.down3 = DownBlock(64, 96, 2)

        self.residual1 = ResidualBlock(96, 128)
        self.residual2 = ResidualBlock(128, 128)

        self.up1 = UpBlock(128, 96)
        self.up2 = UpBlock(96, 64)
        self.up3 = UpBlock(64, 32)

    def forward(self, x):
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

        return x
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

# - 3 -
