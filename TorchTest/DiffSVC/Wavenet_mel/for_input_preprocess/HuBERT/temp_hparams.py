from torch import device as get_device
from torch import cuda

device = get_device("cuda" if cuda.is_available() else "cpu")
hparams = {
    "hubert_gpu": True,
    "device": device,
    "pt_path": 'hubert/hubert_soft.pt'
}
