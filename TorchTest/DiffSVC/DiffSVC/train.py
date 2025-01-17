from utils.sep_wav import separate_run
from utils.get_spec_min_max import get_spec_min_max
from args2hparams import hparams
from diffusion.diffusion import GuassianDiffusion
from vocoder.NsfHiFiGAN.nsf_hifigan import NsfHifiGAN

import os


# ----- dataset preprocessing -----

# - separating raw waves
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


exist_separated(hparams)

# - get min&max of mel-spectrogram

spec_min, spec_max = get_spec_min_max(hparams, NsfHifiGAN.wav2spec)
hparams['spec_min'] = spec_min
hparams['spec_max'] = spec_max

# ----- train -----

diff = GuassianDiffusion(hparams, NsfHifiGAN.wav2spec)
diff.train()
