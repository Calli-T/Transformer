from hubertinfer import Hubertencoder
from temp_hparams import hparams

hubert = Hubertencoder(hparams)
# print(hubert.hbt_model)
hubert_encoded = hubert.encode('../raw/L-O-V-E.wav')
# hubert_encoded = hubert.encode('../raw/dancenote_origin.wav')
# hubert_encoded = hubert.encode('../raw/roundabout_vocals_cut.wav')
print(hubert_encoded.shape)