from hparams import hparams

print(hparams)

# - 1 -

from utils.dataloader import getDataLoader

train_dataloader = getDataLoader(hparams)
# print(hparams['std'].shape)


# - 2 -
'''
from diff.blocks import ResidualBlock, UpBlock, DownBlock

d = DownBlock(3, 64, 2).to(hparams['device'])
u = UpBlock([224, 192], 96, 2).to(hparams['device'])
print(d)
print(u)
'''

'''
# - 3 -

from diff.unet import UNet
from utils.dataloader import getDataLoader
import random

unet = UNet(hparams).to(hparams['device'])
print(unet)

train_dataloader = getDataLoader(hparams)
for batch in train_dataloader:
    variances = [random.random() for _ in range(len(batch))]
    print(unet.forward(variances, batch.to(hparams['device'])).shape)
'''
from ddpm import DDPM
import numpy as np
from utils import show_images
ddpm = DDPM(hparams, train_dataloader)
print(ddpm)

ddpm.load()
gallery = ddpm.generate(8, 10, None, True).permute(0, 2, 3, 1).to('cpu').detach().numpy()
summarized = gallery[0::8] # 뭔가 매핑으로 좀 더 깔끔하게 자르는게 가능할지도?
for i in range(7):
    summarized = np.concatenate((summarized, gallery[i+1::8]), axis=0)
show_images(summarized, 8, 11)
# ddpm.train()
