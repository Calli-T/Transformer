from args2hparams import hparams
from diffusion.diffusion import GuassianDiffusion
from vocoder.NsfHiFiGAN.nsf_hifigan import NsfHifiGAN

import os

# 음원명 배치 크기로 자르기
wav_dir_name = os.path.join(hparams['train_dataset_path_output'], 'final')
wav_fname_list = sorted(os.listdir(wav_dir_name))
wav_fname_list = [os.path.join(wav_dir_name, fname) for fname in wav_fname_list]
split_to_batches = lambda original_list, BATCH_SIZE_TRAIN: [
    original_list[i:i + BATCH_SIZE_TRAIN] for i in range(0, len(original_list), BATCH_SIZE_TRAIN)
]
wav_fname_list = split_to_batches(wav_fname_list, hparams['BATCH_SIZE_TRAIN'])

diff = GuassianDiffusion(hparams, NsfHifiGAN.wav2spec)

raw_padded = diff.get_padded_np_conds(wav_fname_list[0])
pack_padded = diff.get_pack_padded_tensor_conds(raw_padded)
