from args2hparams import get_default
from ddpm import DDPM
from utils.dataloader import getDataLoader

hparams = get_default()

train_dataloader = getDataLoader(hparams)
ddpm = DDPM(hparams, train_dataloader)
ddpm.train()
