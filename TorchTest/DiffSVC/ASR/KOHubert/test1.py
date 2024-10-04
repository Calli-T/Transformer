import torch
from transformers import HubertModel
import librosa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이거 허깅페이스 저장 위치에서 찾아내서 지우고, model load하는걸로 바꿉시다 나중에!!!!
model = HubertModel.from_pretrained("team-lucid/hubert-base-korean").to(device)


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

    # 원본 코드는 hubert까지 먹여서 줍니다, 우리는 wav만 읽어서 모노 16k로 변환만 하면됨
    '''dev = torch.device("cuda" if (dev == torch.device('cuda') and torch.cuda.is_available()) else "cpu")
        torch.cuda.is_available() and torch.cuda.empty_cache()
        with torch.inference_mode():
            units = hbt_soft.units(torch.FloatTensor(wav16.astype(float)).unsqueeze(0).unsqueeze(0).to(dev))
            return units'''

    return wav16


# wav = torch.ones(1, 16000)
wav = get_units("./wav_sample/dancenote_origin.wav")
wav = torch.Tensor(wav).to(device).unsqueeze(0)
wav = torch.split(wav, 16000, 1)


for wave in wav:
    print(wave.shape)
    outputs = model(wave)
    print(f"Output:  {outputs.last_hidden_state.shape}") # 24000나 320000등 다른 숫자를 써보고, 어느 차원을 기준으로 통합할지 생각해보자
    break

'''outputs = model(wav)
print(f"Input:   {wav.shape}")  # [1, 16000]
print(f"Output:  {outputs.last_hidden_state.shape}")  # [1, 49, 768]'''
