from hparams import hparams
from args2hparams import get_default

print(hparams)

# - DataLoader -

'''from utils.dataloader import getDataLoader

train_dataloader = getDataLoader(hparams)
print(hparams['mean'])'''

# - Blocks -
'''
from diff-legacy.blocks import ResidualBlock, UpBlock, DownBlock

d = DownBlock(3, 64, 2).to(hparams['device'])
u = UpBlock([224, 192], 96, 2).to(hparams['device'])
print(d)
print(u)
'''

'''
# - UNet -

from diff-legacy.unet import UNet
from utils.dataloader import getDataLoader
import random

unet = UNet(hparams).to(hparams['device'])
print(unet)

train_dataloader = getDataLoader(hparams)
for batch in train_dataloader:
    variances = [random.random() for _ in range(len(batch))]
    print(unet.forward(variances, batch.to(hparams['device'])).shape)
'''

# ---------------이하 테스트 코드

# training
'''from utils.dataloader import getDataLoader

train_dataloader = getDataLoader(hparams)
print(hparams['std'].shape)

from ddpm import DDPM
import numpy as np
from utils import show_images

ddpm = DDPM(hparams, train_dataloader)  # DDPM(hparams)  # , train_dataloader)
# print(ddpm.diffusion_schedule([x for x in range(1, 11)]))
# print([x for x in range(1, 11)])
ddpm.train()
'''
# ----

# 스케줄 값 확인용
'''ddpm = DDPM(hparams)
sche = ddpm.set_schedule(hparams['steps'])
sen = ""
for i in range(0, len(sche[0])):
    sen += (str((sche[0][i])) + " ")
    if i > 0:
        if i % 10 == 0:
            print(sen)
            sen = ""
        if i % 100 == 0:
            print(i)'''

# sampling
'''from ddpm import DDPM
from utils import show_images

ddpm = DDPM(hparams)
ddpm.load()
gallery = ddpm.p_sample_loop_ddpm(trace_diffusion=True).to('cpu').detach().numpy()
print(gallery.shape)
show_images(gallery, 11, 8)'''