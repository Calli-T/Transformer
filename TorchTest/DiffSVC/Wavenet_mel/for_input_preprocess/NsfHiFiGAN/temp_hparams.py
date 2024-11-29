# nsf HiFiGAN에서 사용하는 hparams를 임시로 가져온것
# 기준은 신창섭 모델에서 사용하던 기준
hparams = {
    "vocoder": "nsf_hifigan.NsfHifiGAN",
    "vocoder_ckpt": "nsf_hifigan/model",
    "audio_sample_rate": 44100,
    "audio_num_mel_bins": 128,
    "fft_size": 2048,
    "win_size": 2048,
    "hop_size": 512,
    "use_nsf": True,
    "fmax": 16000,
    "fmin": 40,
}
