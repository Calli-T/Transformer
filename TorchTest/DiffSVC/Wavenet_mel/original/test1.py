from net import DiffNet
from temp_hparams import hparams
from temp_load_ckpt import load_ckpt

import torch
from torch import device, cuda

device = device('cuda' if cuda.is_available() else 'cpu')

wavenet_mel = DiffNet(hparams)
load_ckpt(wavenet_mel, 'SinChangSeop/model_ckpt_steps_30000.ckpt')
# print(wavenet_mel)
wavenet_mel.to(device)
'''
:param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
'''
temp_spec = torch.randn((384, 1, 128, 1)).to(device)
temp_step = torch.randint(0, 1000, (1, 1)).to(device)
temp_cond = torch.randn((1, 256, 1)).to(device)
print(wavenet_mel(temp_spec, temp_step, temp_cond))
