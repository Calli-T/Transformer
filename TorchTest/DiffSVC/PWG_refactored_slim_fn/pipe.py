import wav2mel
import mel2wav
import mel2wav_2

# 파이프 1차 형태 멜인코딩-정상화-보코더
# 저 import 3개에는 코드 좀 정리 해둡시다


# 또 save에는 return 없고, non-save에는 확실히 numpy array return 해주도록 분기 수정 ㄱㄱ
params = {"origin_path": "files_for_gen/sample/",
          "mel_path": "files_for_gen/dump/sample/raw",
          "mel_norm_path": "files_for_gen/dump/sample/norm/",
          "output_path": "files_for_gen/outputs/",
          "model_path": "files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl",
          "stats_path": "files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/stats.h5",
          "config_path": "files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/config.yml", }

audio_mel = [*wav2mel.wav2mel(sample_path=params["origin_path"], for_config=params["config_path"])]
print(len(audio_mel))

mel_norm = mel2wav.normalize(raw_path=params["mel_path"], for_stats=params["stats_path"],
                             for_config=params["config_path"],
                             for_dataset=audio_mel)
print(mel_norm[0].shape)

wave = mel2wav_2.mel2wav(model_path=params["model_path"], for_dataset=[audio_mel[0], mel_norm])
mel2wav_2.mel2wav(model_path=params["model_path"], output_path=params["output_path"],
                  for_dataset=[audio_mel[0], mel_norm])
print(wave[0].shape)
