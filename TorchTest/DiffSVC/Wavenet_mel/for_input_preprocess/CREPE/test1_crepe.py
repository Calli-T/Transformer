from wav2spec_stand_alone import wav2spec
from crepe import get_pitch_crepe
from temp_hparams import hparams

wav, mel = wav2spec('../raw/L-O-V-E.wav')
gt_f0, coarse_f0 = get_pitch_crepe(wav, mel, hparams)
print(gt_f0.shape, coarse_f0.shape)