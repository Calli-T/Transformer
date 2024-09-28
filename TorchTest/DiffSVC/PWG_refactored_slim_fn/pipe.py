import wav2mel
import mel2wav
import mel2wav_2

# 나중에 dict 형태로 살짝 변경해둡시다
# 또 save에는 return 없고, non-save에는 확실히 numpy array return 해주도록 분기 수정 ㄱㄱ
params = [
    "files_for_gen/sample/",
    "files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/config.yml",
    "files_for_gen/dump/sample/raw",
    "files_for_gen/dump/sample/norm/",
    "files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/stats.h5",
    "files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl",
    "files_for_gen/outputs/"
]

audio_mel = [*wav2mel.wav2mel(sample_path=params[0], for_config=params[1])]
print(len(audio_mel))

mel_norm = mel2wav.normalize(raw_path=params[2], for_stats=params[4], for_config=params[1],
                             for_dataset=audio_mel)
print(mel_norm[0].shape)

wave = mel2wav_2.mel2wav(model_path=params[5], for_dataset=[audio_mel[0], mel_norm])
print(wave[0].shape)
