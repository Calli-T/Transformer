from hparams import hparams

print(hparams)

# - 1 -

from utils.dataloader import getDataLoader

# train_dataloader = getDataLoader(hparams)
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

# ---------------이하 테스트 코드

# 학습하는 코드
from utils.dataloader import getDataLoader

train_dataloader = getDataLoader(hparams)
print(hparams['std'].shape)

from ddpm import DDPM
import numpy as np
from utils import show_images

'''ddpm = DDPM(hparams, train_dataloader)  # DDPM(hparams)  # , train_dataloader)
# print(ddpm.diffusion_schedule([x for x in range(1, 11)]))
# print([x for x in range(1, 11)])
ddpm.train()'''

# ----

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

'''# 견본 몇 개 떠주는 코드
ddpm = DDPM(hparams)
ddpm.load()
gallery = ddpm.p_sample_loop_ddpm(3).permute(0, 2, 3, 1).to('cpu').detach().numpy()
print(gallery.shape)
show_images(gallery, 1, 3)'''

'''# 구간 잘라서 확산 new 코드
ddpm = DDPM(hparams)
ddpm.load()
gallery = ddpm.p_sample_loop_ddpm(1, return_all_t=True).permute(0, 2, 3, 1).to('cpu').detach().numpy()
show_images(gallery[:10], 1, 10)
'''
# 구간 잘라서 확산 단계를 보여주는 코드, 나중에 쓸것
'''ddpm.load()
gallery = ddpm.generate(8, 10, None, True).permute(0, 2, 3, 1).to('cpu').detach().numpy()
summarized = gallery[0::8]  # 뭔가 매핑으로 좀 더 깔끔하게 자르는게 가능할지도?
for i in range(7):
    summarized = np.concatenate((summarized, gallery[i + 1::8]), axis=0)
show_images(summarized, 8, 11)'''

# ddpm.train()

'''# UNet 위치 임베딩 코드 수정용 테스트
from diff.unet import UNet

unet = UNet(hparams).to(hparams['device'])
print(unet.nchw_tensor_sinusoidal_embedding([0.34, 0.7]).shape)'''
