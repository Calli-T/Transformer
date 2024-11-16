from torch import device, cuda

device = device('cuda' if cuda.is_available() else 'cpu')

hparams = {
    "device": device,
    "IMAGE_SIZE": 64,
    "BATCH_SIZE": 64,
    "DATASET_REPETITION": 5,
    "data_path": "./datasets/sunflower",
    "model_path": "./models/sunflower",
    "NOISE_EMBEDDING_SIZE": 32,
    "mean": 0.0,
    "std": 1.0,
    "steps": 4000,
    "EMA": 0.999,
    "LEARNING_RATE": 0.001,
    "WEIGHT_DECAY": 0.0001,
    "EPOCHS": 500,
    "DDIM_STEPS": 10,
}
