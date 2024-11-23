from net import DiffNet
from temp_hparams import hparams
from temp_load_ckpt import load_ckpt

import torch
from torch import device, cuda

device = device('cuda' if cuda.is_available() else 'cpu')

wavenet_mel = DiffNet(hparams)
load_ckpt(wavenet_mel, 'SinChangSeop/model_ckpt_steps_30000.ckpt')
print(wavenet_mel)