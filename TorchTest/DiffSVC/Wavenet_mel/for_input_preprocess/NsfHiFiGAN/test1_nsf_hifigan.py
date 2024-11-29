from nsf_hifigan import NsfHifiGAN

vocoder = NsfHifiGAN()
# wav, mel = VOCODERS[hparams['vocoder']].wav2spec(temp_dict['wav_fn'])
wav, mel = vocoder.wav2spec('../raw/L-O-V-E.wav')
print(wav.shape, mel.shape)
