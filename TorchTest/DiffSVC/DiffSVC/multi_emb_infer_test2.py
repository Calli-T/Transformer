from args2hparams import hparams
from diffusion.diffusion import GuassianDiffusion
from vocoder.NsfHiFiGAN.nsf_hifigan import NsfHifiGAN
from utils.gen_sound_file import after_infer

import os

vocoder = NsfHifiGAN(hparams)

dir_path = hparams['raw_wave_dir_path']
wav_fname_list = [os.path.join(dir_path, fname) for fname in os.listdir(hparams['raw_wave_dir_path'])]
diff = GuassianDiffusion(hparams, NsfHifiGAN.wav2spec)

outputs = diff.infer_batch(wav_fname_list)

after_infer(outputs, vocoder, hparams)
