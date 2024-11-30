from NsfHiFiGAN.nsf_hifigan import NsfHifiGAN
from temp_hparams import hparams, rel2abs

vocoder = NsfHifiGAN(hparams)
wav, mel = vocoder.wav2spec(rel2abs(hparams['relative_raw_wave_path']), hparams)
print(wav.shape, mel.shape)
