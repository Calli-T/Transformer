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


'''args_dict = get_parsed_dict()
for arg in args_dict:
    print(f'{arg}: {args_dict[arg]}')'''


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
        "data_path": "./datasets/IDDPM_LSUN",
        "model_path": "./models/IDDPM_LSUN",
        "LEARNING_RATE": 0.0001,
        "WEIGHT_DECAY": 0.00001,
        "EPOCHS": 200000,
        "save_interval": 2000,
        "EMA": 0.999,
        # for sampling
        "BATCH_SIZE_SAMPLE": 1,
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


'''args_dict = get_hparams()
for arg in args_dict:
    print(f'{arg}: {args_dict[arg]}')'''

# print(get_default().keys())
'''
DIFFUSION_FLAGS list
diffusion_steps
noise_schedule
rescale_learned_sigmas # 이거 뭐에 쓰는 옵션이지?
rescale_timesteps # 이건 항상 true로 해두자
use_scale_shift_norm # model flasg랑 겹침
'''

'''
MODEL_FLAGS list
image_size
learn_sigma
dropout
num_channels
num_res_blocks
num_heads
use_scale_shift_norm
attention_resolutions
'''

'''
TRAIN_FLAGS list
lr
batch_size 128
schedule_sampler # 이게 대체 뭔...
'''

# print(get_default())
