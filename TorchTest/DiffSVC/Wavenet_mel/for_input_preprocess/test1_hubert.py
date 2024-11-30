from HuBERT.hubertinfer import Hubertencoder
from temp_hparams import hparams, rel2abs

hubert = Hubertencoder(hparams)
# print(hubert.hbt_model)
hubert_encoded = hubert.encode(rel2abs(hparams['relative_raw_wave_path']))
# hubert_encoded = hubert.encode('../raw/dancenote_origin.wav')
# hubert_encoded = hubert.encode('../raw/roundabout_vocals_cut.wav')
print(hubert_encoded.shape)
