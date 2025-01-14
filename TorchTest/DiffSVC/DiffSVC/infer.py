from args2hparams import hparams
from diffusion.diffusion import GuassianDiffusion
from vocoder.NsfHiFiGAN.nsf_hifigan import NsfHifiGAN
from utils.gen_sound_file import gen_wav_from_output
from utils.normalizer import denormalize_files, normalize_files_and_report, move_files

import os

# 음원 dB 정상화
if hparams['use_norm']:
    print('normalizing')
    dB_scale_list = normalize_files_and_report(hparams['raw_wave_dir_path'])
    # print(dB_scale_list)

#
diff = GuassianDiffusion(hparams, NsfHifiGAN.wav2spec)

# mel과 nsf-hifigan용 f0 생성
dir_path = hparams['raw_wave_dir_path']
wav_fname_list = [os.path.join(dir_path, fname) for fname in os.listdir(hparams['raw_wave_dir_path'])]
outputs = diff.infer(wav_fname_list)

# 음원 생성
vocoder = NsfHifiGAN(hparams)
gen_wav_from_output(outputs, vocoder, hparams)

# 음원 dB 역정상화
if hparams['use_norm']:
    print('denormalizing')
    denormalize_files(hparams['raw_wave_dir_path'], dB_scale_list)
    denormalize_files(hparams['denorm_dir_path'], dB_scale_list)
    move_files(hparams['denorm_dir_path'], hparams['result_dir_path'])

