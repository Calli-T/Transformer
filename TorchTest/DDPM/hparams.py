from torch import device, cuda

device = device('cuda' if cuda.is_available() else 'cpu')

hparams = {
    "device": device,
    "IMAGE_SIZE": 64,
    "BATCH_SIZE": 64,
    "DATASET_REPETITION": 5,
    "data_path": "./datasets/flower",
    "model_path": "./models/flower",
    "NOISE_EMBEDDING_SIZE": 32,
    "mean": 0.0,
    "std": 1.0,
    "steps": 1000,
}
