import torch
from transformers import HubertModel
import librosa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model이 '어딘가에' 다운로드 되었는데 찾아야한다
'''
pytorch_model.bin: 100%|██████████| 378M/378M [00:12<00:00, 29.7MB/s]
config.json: 100%|██████████| 1.39k/1.39k [00:00<00:00, 11.2MB/s]
이놈들 찾아서 위치 바꾸고 from_pretrained에 경로 넣어야한다!!
'''
model = HubertModel.from_pretrained("models/models--team-lucid--hubert-base-korean").to(device)
# facebook/hubert-large-ls960-ft
# facebook/hubert-base-ls960 <- 이 3개는 웹에서 다운 받는 방법
# team-lucid/?

# prophesier의 코드를 그대로 긴빠이 쳐온것, 나중에 바꿔봅시다
# 하는 작업은 stereo to mono, sr 16k upper to 16k
def get_units(raw_wav_path):
    wav, sr = librosa.load(raw_wav_path, sr=None)
    '''print(wav.shape)
    print(sr)'''
    assert (sr >= 16000)
    if len(wav.shape) > 1:
        wav = librosa.to_mono(wav)
    if sr != 16000:
        wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    else:
        wav16 = wav

    return wav16


wav = get_units("./raw/L-O-V-E.wav")
wav = torch.Tensor(wav).to(device).unsqueeze(0)
print(wav.shape)
wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
output = model(wav)
print(output.last_hidden_state.shape)

print(len(output[0][0]))
print(len(output[0][0][0]))
# print(model.encoder.proj)