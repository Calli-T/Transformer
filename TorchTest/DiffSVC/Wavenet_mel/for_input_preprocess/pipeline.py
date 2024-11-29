from NsfHiFiGAN import models, temp_hparams, nvSTFT, utils
from NsfHiFiGAN.nsf_hifigan import NsfHifiGAN

vocoder = NsfHifiGAN()
wav, mel = vocoder.wav2spec('../raw/L-O-V-E.wav')
print(wav.shape, mel.shape)
