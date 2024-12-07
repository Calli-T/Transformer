from torch import device, cuda
device = device('cuda' if cuda.is_available() else 'cpu')
epoch = 38100
project_name = "SinChangSeop"
# class DiffNet에서 사용하는 hparams를 임시로 가져온것
hparams = {
    "hidden_size": 256,
    "residual_layers": 20,
    "residual_channels": 384,
    "dilation_cycle_length": 4,
    "audio_num_mel_bins": 128,
    "device": device,
    "wavenet_model_path": f"{project_name}/wavenet_model_steps_{epoch}.pt",
}
