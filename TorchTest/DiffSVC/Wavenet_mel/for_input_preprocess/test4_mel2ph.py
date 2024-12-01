from NsfHiFiGAN.nsf_hifigan import NsfHifiGAN
from mel2ph.mel2ph import get_align
from temp_hparams import hparams, rel2abs
from HuBERT.hubertinfer import Hubertencoder

vocoder = NsfHifiGAN(hparams)
wav, mel = vocoder.wav2spec(rel2abs(hparams['raw_wave_path']), hparams)

hubert = Hubertencoder(hparams)
hubert_encoded = hubert.encode(rel2abs(hparams['raw_wave_path']))

mel2ph = get_align(mel, hubert_encoded)

'''temp = ''
for idx, ph in enumerate(mel2ph):
    if idx % 10 == 0:
        print(temp)
        temp = ''
    temp += str(ph)
    temp += ' '
print(f'mel2ph.shape: {mel2ph.shape}')'''