'''from torch_directml import device

device = device()'''
from torch import device, cuda

device = device('cuda' if cuda.is_available() else 'cpu')
# print(device)
