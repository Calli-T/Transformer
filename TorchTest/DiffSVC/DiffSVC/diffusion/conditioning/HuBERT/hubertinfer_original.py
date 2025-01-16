import torch
from transformers import HubertModel
import librosa


class HuBERTModel:
    def __init__(self, _hparams):
        self.hparams = _hparams
        self.device = self.hparams['device']
        model_path = self.hparams['hubert_original_path']

        self.model = HubertModel.from_pretrained(model_path).to(self.device)
        # hubert_encoded = self.hubert.encode(wav=wav, sr=self.hparams['audio_sample_rate'])

    def encode(self, wav, sr=None, is_single_wav=True):
        # 일단 wave는 하나씩 받아옵시다

        if sr is None:
            sr = self.hparams['sample_rate']

        if sr != 16000:
            wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        else:
            wav16 = wav

        torch.cuda.is_available() and torch.cuda.empty_cache()

        wav16 = torch.Tensor(wav).to(self.device).unsqueeze(0)
        wav16 = torch.nn.functional.pad(wav16, ((400 - 320) // 2, (400 - 320) // 2))

        output = self.model(wav16)
        output = output.last_hidden_state.detach().cpu().numpy()

        return output # torch.FloatTensor(output).to(self.device)


# prophesier의 코드를 그대로 긴빠이 쳐온것, 나중에 바꿔봅시다
# 하는 작업은 stereo to mono, sr 16k upper to 16k
'''def get_units(raw_wav_path):
    wav, sr = librosa.load(raw_wav_path, sr=None)
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
wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
output = model(wav)
print(output.last_hidden_state.shape)

print(len(output[0][0]))
print(len(output[0][0][0]))'''
# print(model.encoder.proj)
