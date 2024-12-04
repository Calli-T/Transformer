from args2hparams import get_hparams
from ddpm import DDPM
from utils.dataloader import getDataLoader

hparams = get_hparams()

train_dataloader = getDataLoader(hparams)
ddpm = DDPM(hparams, train_dataloader)
ddpm.load()
ddpm.train()
