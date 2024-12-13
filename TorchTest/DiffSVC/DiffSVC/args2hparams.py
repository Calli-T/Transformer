from torch import device, cuda

# 신창섭 epoch 모델을 둘로 쪼갠 것 기준
device = device('cuda' if cuda.is_available() else 'cpu')
pt_epoch = 38100
project_name = "SinChangSeop"
hparams = {
    "project_name": project_name,

    "raw_wave_path": "raw/L-O-V-E_[cut_6sec].wav",
    # "raw_dir_path": "raw",

    # for vocoder, NsfHiFiGAN
    # "vocoder": "nsf_hifigan.NsfHifiGAN",
    "device": device,
    "vocoder_ckpt": "models/nsf_hifigan/model",
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
    "pt_path": 'models/hubert_soft.pt',

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
    "emb_model_path": "models/embedding_model_steps_38100.pt",  # "model_ckpt_steps_30000.ckpt",
    "hidden_size": 256,

    # for wavenet
    # "hidden_size": 256,
    "residual_layers": 20,
    "residual_channels": 384,
    "dilation_cycle_length": 4,
    # "audio_num_mel_bins": 128,
    # "device": device,
    "wavenet_model_path": f"models/wavenet_model_steps_{pt_epoch}.pt",

    # for diffusion
    "schedule_name": "linear",
    "steps": 1000,
}

'''
import argparse

from torch import device, cuda

device = device('cuda' if cuda.is_available() else 'cpu')


def get_parsed_dict():
    parser = argparse.ArgumentParser()
    keys_int = ['IMAGE_SIZE', 'in_channels', 'out_channels', 'num_res_blocks', 'num_channels', 'num_heads',
                'BATCH_SIZE_TRAIN', 'DATASET_REPETITION',
                'EPOCHS', 'save_interval', 'BATCH_SIZE_SAMPLE', 'steps']
    keys_float = ['dropout', 'LEARNING_RATE', 'WEIGHT_DECAY', 'EMA', ]
    keys_str = ['data_path', 'model_path', 'schedule_name', "attention_resolutions"]
    keys_bool = ['learn_sigma', 'use_scale_shift_norm', ]

    for key in keys_int:
        parser.add_argument(f'--{key}', default=None, type=int, help='')
    for key in keys_float:
        parser.add_argument(f'--{key}', default=None, type=float, help='')
    for key in keys_str:
        parser.add_argument(f'--{key}', default=None, type=str, help='')
    for key in keys_bool:
        parser.add_argument(f'--{key}', default=None, type=bool, help='')

    return vars(parser.parse_args())


def get_default():
    return {
        "device": device,
        # for model
        "IMAGE_SIZE": 256,
        "learn_sigma": True,
        "in_channels": 3,
        "out_channels": 6,
        "dropout": 0.0,
        "num_res_blocks": 2,
        "num_channels": 128,
        "num_heads": 1,
        "use_scale_shift_norm": False,

        "attention_resolutions": "16",
        "channel_mult": (1, 1, 2, 2, 4, 4),
        # for training
        "BATCH_SIZE_TRAIN": 2,
        "DATASET_REPETITION": 1,
        "data_path": "./datasets/mint_mini",
        "model_path": "./models/mint_mini",
        "LEARNING_RATE": 0.0001,
        "WEIGHT_DECAY": 0.00001,
        "EPOCHS": 200000,
        "save_interval": 2000,
        "EMA": 0.999,
        # for sampling
        "BATCH_SIZE_SAMPLE": 16,
        # for diffusion
        "schedule_name": "linear",
        "steps": 1000,
    }


def get_hparams():
    defaults = get_default()
    args_parsed = get_parsed_dict()

    for key in args_parsed:
        if args_parsed[key] is not None:
            defaults[key] = args_parsed[key]

    if defaults["IMAGE_SIZE"] == 256:
        defaults["channel_mult"] = (1, 1, 2, 2, 4, 4)
    elif defaults["IMAGE_SIZE"] == 64:
        defaults["channel_mult"] = (1, 2, 3, 4)
    elif defaults["IMAGE_SIZE"] == 32:
        defaults["channel_mult"] = (1, 2, 2, 2)

    attention_ds = []
    for res in defaults["attention_resolutions"].split(","):
        attention_ds.append(defaults["IMAGE_SIZE"] // int(res))

    defaults["attention_resolutions"] = tuple(attention_ds)

    return defaults
'''
