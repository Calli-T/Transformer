from torch import device, cuda
import os

# nsf HiFiGAN에서 사용하는 hparams를 임시로 가져온것
# 기준은 신창섭 모델에서 사용하던 기준
device = device('cuda' if cuda.is_available() else 'cpu')
hparams = {
    "project_name": "SinChangSeop",

    "raw_wave_path": "raw/L-O-V-E_[cut_6sec].wav",
    "raw_dir_path": "raw",

    # for vocoder, NsfHiFiGAN
    # "vocoder": "nsf_hifigan.NsfHifiGAN",
    "device": device,
    "vocoder_ckpt": "NsfHiFiGAN/nsf_hifigan/model",
    "audio_sample_rate": 44100,
    "audio_num_mel_bins": 128,
    "fft_size": 2048,
    "win_size": 2048,
    "hop_size": 512,
    "use_nsf": True,
    "fmax": 16000,
    "fmin": 40,

    # for self_regressive_phonetic, HuBERT
    "hubert_gpu": True,
    "pt_path": 'HuBERT/hubert/hubert_soft.pt',

    # for Pitch Extractor, CREPE
    "f0_bin": 256,
    "f0_max": 1100.0,
    "f0_min": 40.0,
    # "audio_sample_rate": 44100,
    # "hop_size": 512,

    # for condition integrate & preprocess
    "max_frames": 42000,
    "max_input_tokens": 60000,
    "pitch_norm": "log",
    "emb_model_path": "model_ckpt_steps_30000.ckpt",
    "hidden_size": 256
}


def rel2abs(rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)


def dir2list(raw_path):
    abs_raw_path = rel2abs(raw_path)
    name_list = os.listdir(abs_raw_path)
    return [os.path.join(abs_raw_path, name) for name in name_list]
