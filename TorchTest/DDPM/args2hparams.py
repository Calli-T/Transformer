from torch import device, cuda

device = device('cuda' if cuda.is_available() else 'cpu')


def get_default():
    return {
        "device": device,
        # for model
        "IMAGE_SIZE": 256,
        "learn_sigma": True,
        "dropout": 0.0,
        "num_res_blocks": 2,
        "num_channels": 128,
        "num_heads": 1,
        "use_scale_shift_norm": False,

        "attention_resolutions": tuple([16]),
        "channel_mult": (1, 1, 2, 2, 4, 4),
        # for training
        "BATCH_SIZE_TRAIN": 2,
        "DATASET_REPETITION": 4,
        "data_path": "./datasets/sunflower",
        "model_path": "./models/sunflower",
        "LEARNING_RATE": 0.0001,
        "WEIGHT_DECAY": 0.00001,
        "EPOCHS": 200000,
        "save_interval": 4000,
        "EMA": 0.999,
        # for sampling
        "BATCH_SIZE_SAMPLE": 8,
        # for diffusion
        "schedule_name": "linear",
        "steps": 1000,
    }


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

print(get_default())
