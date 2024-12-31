from args2hparams import hparams
from diffusion.diffusion import GuassianDiffusion
from vocoder.NsfHiFiGAN.nsf_hifigan import NsfHifiGAN
from utils.gen_sound_file import after_infer

vocoder = NsfHifiGAN(hparams)

wav_fname_list = ['raw/scs_origin-026.wav',
                  'raw/scs_origin-023.wav']  # ["raw/dancenote_origin.wav", "raw/L-O-V-E.wav", "raw/yoon3.wav"]

diff = GuassianDiffusion(hparams, NsfHifiGAN.wav2spec)

outputs = diff.infer_batch(wav_fname_list)
print(outputs.keys())
# print(ret['mel_out'].shape)

after_infer(outputs, vocoder, hparams)
