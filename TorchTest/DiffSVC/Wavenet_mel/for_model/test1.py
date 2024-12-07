from net import DiffNet
from temp_hparams import hparams
from temp_load_ckpt import load_ckpt

import torch
# from torch import device, cuda
#
# device = device('cuda' if cuda.is_available() else 'cpu')

wavenet_mel = DiffNet(hparams)
# load_ckpt(wavenet_mel, 'SinChangSeop/model_ckpt_steps_30000.ckpt')
wavenet_mel.load_state_dict(torch.load(hparams['wavenet_model_path'], map_location='cpu'))
# print(wavenet_mel)
wavenet_mel.to(hparams['device'])
'''
:param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1] -> [B, ] 실제로는 1축 텐서이니 앞의 ,1은 무시할것
        :param cond: [B, M, T] -> 이거 M이 mel-band 수가 아니고, hidden 값과 관련있는 것 같다
'''

temp_spec = torch.randn((2, 1, 128, 1)).to(hparams['device'])
temp_step = torch.randint(0, 1000, (2,)).to(hparams['device'])
temp_cond = torch.randn((2, 256, 1)).to(hparams['device'])
print(temp_spec.shape)
print(temp_step.shape)
print(temp_cond.shape)
print(wavenet_mel(temp_spec, temp_step, temp_cond).shape)
