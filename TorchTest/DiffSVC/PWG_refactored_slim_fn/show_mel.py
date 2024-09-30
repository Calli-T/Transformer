import librosa
import numpy as np
from matplotlib import pyplot as plt
from wav2mel import *


def show_mel(mel, sr):
    mel = librosa.power_to_db(pow(10, mel), ref=np.max)
    print(mel)
    # Mel Filter Bank 시각화
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, sr=sr, x_axis='time',
                             y_axis='mel')  # librosa.power_to_db(pow(10, mel), ref=np.max) # -librosa.amplitude_to_db(mel,  ref=np.max)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()


def plot_waveform(y, sr, title="Waveform"):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


params = {"origin_path": "files_for_gen/sample/",
          "mel_path": "files_for_gen/dump/sample/raw",
          "mel_norm_path": "files_for_gen/dump/sample/norm/",
          "output_path": "files_for_gen/outputs/",
          "model_path": "files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl",
          "stats_path": "files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/stats.h5",
          "config_path": "files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/config.yml", }

audio_mel = [*wav2mel(sample_path=params["origin_path"], for_config=params["config_path"])]
print(audio_mel[2][0].T.shape)
show_mel(audio_mel[2][0].T, sr=24000)
