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
unet = UNet()
train_dataloader = getDataLoader("./datasets")
for batch in train_dataloader:
    print(unet.forward(batch))
'''

# class DDPM(nn.Module):
#     def __init__(self):
#         super().__init__()

# - 3 -
