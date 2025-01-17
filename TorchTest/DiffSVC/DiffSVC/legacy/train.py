from utils.sep_wav import separate_run
from args2hparams import hparams
from utils.path_utils import rel2abs

import os


# ----- dataset preprocessing -----
def exist_separated(_hparams):
    # 채널, 샘플링 레이트, 확장자 변경하고 음원 10~15초 사이로 자르는 전처리 함수
    # 위 4개의 전처리 해둔 작업물이 있으면 True 없으면 False를 반환
    sep_outputs_dir = os.path.join(_hparams['train_dataset_path_output'], 'final')
    if not os.path.isdir(sep_outputs_dir) or len(os.listdir(sep_outputs_dir)) <= 0:
        print('학습용 음성 wav 파일 미발견, 음원 분할 작업 시작')
        separate_run(_hparams)
        print('음원 분할 완료, 학습 시작')
        return False
    else:
        print('학습용 음성 wav 파일 발견, 해당 파일을 사용해 학습 시작')
        return True

# 얘는 모델 내부의 기능을 활용해서 만드는 것이니, 모델 안으로 집어넣자
'''
def exist_f0_npy(_hparams):
    # f0 파일이 있는지 확인
    f0_npy_dir = _hparams['train_dataset_path_f0']
    sep_outputs_dir = os.path.join(_hparams['train_dataset_path_output'], 'final')
    if os.path.isdir(sep_outputs_dir) and os.path.isdir(f0_npy_dir):
        if os.listdir(f0_npy_dir) == os.listdir(sep_outputs_dir):
            print('학습용 f0 파일 확인, 해당 파일을 사용해 학습 시작')
            return True
        else:
            return False
    else:
        return False
'''


exist_separated(hparams)
# exist_f0_npy(hparams)

from diffusion.diffusion import GuassianDiffusion
from vocoder.NsfHiFiGAN.nsf_hifigan import NsfHifiGAN

diff = GuassianDiffusion(hparams, NsfHifiGAN.wav2spec)  # , vocoder.wav2spec)
# diff.train()
diff.train()
