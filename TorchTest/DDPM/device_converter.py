'''from torch_directml import device

device = device()'''
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)