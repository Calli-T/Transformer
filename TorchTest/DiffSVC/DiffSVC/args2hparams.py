from torch import device, cuda

# 신창섭 epoch 모델을 둘로 쪼갠 것 기준
device = device('cuda' if cuda.is_available() else 'cpu')
pt_epoch = 84200
project_name = "SinChangSeop"
hparams = {
    "project_name": project_name,
    "model_pt_epoch": pt_epoch,  # 학습'된' epoch

    "raw_wave_path": "raw/apart [vocals]-24_48sec.wav",  # "raw/L-O-V-E_[cut_6sec].wav",
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
    "emb_model_path": f"models/singer/{project_name}/embedding_model_steps_{pt_epoch}.pt",  # "model_ckpt_steps_30000.ckpt",
    "hidden_size": 256,

    # for wavenet
    # "hidden_size": 256,
    "residual_layers": 20,
    "residual_channels": 384,
    "dilation_cycle_length": 4,
    # "audio_num_mel_bins": 128,
    # "device": device,
    "wavenet_model_path": f"models/singer/{project_name}/wavenet_model_steps_{pt_epoch}.pt",

    # for diffusion
    "schedule_name": "linear",
    "steps": 1000,  # 00,

    # for postprocess
    "spec_max": [0.5272776484489441, 0.9114222526550293, 1.1117855310440063, 1.0987094640731812, 1.1520036458969116,
                 1.205979347229004, 1.1478168964385986, 1.0361219644546509, 0.9603051543235779, 1.1451283693313599,
                 1.0149853229522705, 1.0380277633666992, 1.1308894157409668, 1.1211094856262207, 1.000557541847229,
                 1.0907360315322876, 1.0838499069213867, 0.9790574312210083, 1.089969515800476, 1.0722992420196533,
                 1.0278816223144531, 0.9898473620414734, 1.0580487251281738, 1.054341435432434, 1.015966534614563,
                 1.0645880699157715, 0.911625325679779, 1.0145788192749023, 0.9623145461082458, 0.9171468019485474,
                 0.8047866225242615, 0.7397677898406982, 0.8876128792762756, 0.7318429350852966, 0.6728546619415283,
                 0.6627925038337708, 0.8188887238502502, 0.5903922319412231, 0.6322281360626221, 0.5471858382225037,
                 0.5222923755645752, 0.4334467053413391, 0.5173780918121338, 0.46364495158195496, 0.4395354688167572,
                 0.5396381616592407, 0.42747896909713745, 0.5859876275062561, 0.40101614594459534, 0.3250364363193512,
                 0.400651216506958, 0.4415549337863922, 0.5018853545188904, 0.48584455251693726, 0.4867593050003052,
                 0.3564012944698334, 0.49589088559150696, 0.3893994688987732, 0.4077587425708771, 0.3942872881889343,
                 0.273531049489975, 0.3805701732635498, 0.2766077518463135, 0.22353535890579224, 0.38731473684310913,
                 0.4120328426361084, 0.3243582844734192, 0.388715535402298, 0.4885849356651306, 0.4638833701610565,
                 0.37133342027664185, 0.25582262873649597, 0.3552630543708801, 0.4902227818965912, 0.5880915522575378,
                 0.33124953508377075, 0.31935736536979675, 0.28546303510665894, 0.16602271795272827, 0.3434958755970001,
                 0.17701327800750732, 0.34745070338249207, 0.05982062965631485, 0.2704027593135834, 0.05331733450293541,
                 0.19360707700252533, 0.0936945453286171, 0.09670297801494598, 0.18013012409210205,
                 -0.05549187958240509, -0.0011373111046850681, -0.03177299723029137, -0.07811982184648514,
                 0.11156868189573288, 0.07773900777101517, 0.0034726052545011044, -0.004783670883625746,
                 0.010186700150370598, -0.02491397224366665, -0.07301700115203857, -0.19408515095710754,
                 -0.18510079383850098, -0.26655352115631104, -0.26560330390930176, -0.4981132745742798,
                 -0.5400726199150085, -0.40430617332458496, -0.3858506381511688, -0.5397453904151917,
                 -0.4821416139602661, -0.5218197703361511, -0.580097496509552, -0.7246339321136475, -0.700655996799469,
                 -0.9624589681625366, -0.9454383254051208, -0.9504799842834473, -1.0763174295425415,
                 -1.0317856073379517, -1.1064244508743286, -1.0824952125549316, -1.0807260274887085,
                 -1.1013997793197632, -1.2503222227096558, -1.394013524055481, -1.362666130065918, -1.492540717124939,
                 -1.6387512683868408],
    "spec_min": [-4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126, -4.99999475479126,
                 -4.99999475479126, -4.99999475479126, -4.99999475479126],
    "keep_bins": 128,
    "mel_vmax": 1.5,
    "mel_vmin": -6.0,
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
