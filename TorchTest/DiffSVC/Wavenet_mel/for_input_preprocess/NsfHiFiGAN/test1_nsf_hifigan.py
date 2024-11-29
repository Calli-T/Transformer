from nsf_hifigan import NsfHifiGAN
import torch

'''state_dict = torch.load('nsf_hifigan/model')['generator']
for key in state_dict.keys():
    print(key)'''

vocoder = NsfHifiGAN()
