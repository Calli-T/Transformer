from args2hparams import get_hparams
from ddpm import DDPM
from utils import show_images

hparams = get_hparams()

ddpm = DDPM(hparams)
ddpm.load()
gallery = ddpm.p_sample_loop_ddpm(trace_diffusion=True).to('cpu').detach().numpy()
print(gallery.shape)
show_images(gallery, 11, hparams["BATCH_SIZE_SAMPLE"])
