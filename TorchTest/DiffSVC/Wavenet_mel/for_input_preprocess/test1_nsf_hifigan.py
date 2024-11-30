from NsfHiFiGAN.nsf_hifigan import NsfHifiGAN
from temp_hparams import hparams

vocoder = NsfHifiGAN(hparams)
# wav, mel = VOCODERS[hparams['vocoder']].wav2spec(temp_dict['wav_fn'])
wav, mel = vocoder.wav2spec('./raw/L-O-V-E.wav', hparams)
print(wav.shape, mel.shape)
