import torch

state_dict = torch.load('SinChangSeop/model_ckpt_steps_30000.ckpt', map_location='cpu')["state_dict"]
for key in state_dict.keys():
    print(key)