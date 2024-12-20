from torch import device, cuda

device = device('cuda' if cuda.is_available() else 'cpu')

hparams = {
    "device": device,
    # for model and diffusion
    "IMAGE_SIZE": 256,
    "learn_sigma": True,
    # for training
    "BATCH_SIZE_TRAIN": 2,  # 64,
    "DATASET_REPETITION": 4,
    "data_path": "./datasets/sunflower",  # "data_path": "./datasets/IDDPM_LSUN",
    "model_path": "./models/sunflower",  # "model_path": "./models/IDDPM_LSUN",
    "LEARNING_RATE": 0.0001,
    "WEIGHT_DECAY": 0.00001,
    "EPOCHS": 200000,
    "save_interval": 4000,
    "EMA": 0.999,
    # for sampling
    "BATCH_SIZE_SAMPLE": 8,  # 16
    # for diffusion
    "schedule_name": "linear",  # "linear
    "steps": 1000,
}

'''
hparams = {
    "device": device,
    "IMAGE_SIZE": 64,
    "BATCH_SIZE": 64,
    "DATASET_REPETITION": 5,
    "data_path": "./datasets/IDDPM_LSUN",
    "model_path": "./models/IDDPM_LSUN",
    "NOISE_EMBEDDING_SIZE": 32,
    "mean": 0.0,
    "std": 1.0,
    "steps": 4000,
    "EMA": 0.999,
    "LEARNING_RATE": 0.001,
    "WEIGHT_DECAY": 0.0001,
    "EPOCHS": 500,
    "DDIM_STEPS": 10,
    "learn_sigma": True,
    "schedule_name": "cosine",  # "linear
}
'''
