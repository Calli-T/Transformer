_hparams = {
    "audio_sample_rate": 44100,
    "audio_num_mel_bins": 128,
    "fft_size": 2048,
    "win_size": 2048,
    "hop_size": 512,
    "fmax": 16000,
    "fmin": 40,
}

import soundfile as sf
import numpy as np
import torch
import librosa
from librosa.filters import mel as librosa_mel_fn


def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)  # than soundfile.
    except Exception as ex:
        print(f"'{full_path}' failed to load.\nException:")
        print(ex)
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 48000
        else:
            raise Exception(ex)

    if len(data.shape) > 1:
        data = data[:, 0]
        assert len(
            data) > 2  # check duration of audio file is > 2 samples (because otherwise the slice operation was on the wrong dimension)

    if np.issubdtype(data.dtype, np.integer):  # if audio data is type int
        max_mag = -np.iinfo(data.dtype).min  # maximum magnitude = min possible value of intXX
    else:  # if audio data is type fp32
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = (2 ** 31) + 1 if max_mag > (2 ** 15) else ((
                                                                     2 ** 15) + 1 if max_mag > 1.01 else 1.0)  # data should be either 16-bit INT, 32-bit INT or [-1 to 1] float32

    data = torch.FloatTensor(data.astype(np.float32)) / max_mag

    if (torch.isinf(data) | torch.isnan(
            data)).any() and return_empty_on_exception:  # resample will crash with inf/NaN inputs. return_empty_on_exception will return empty arr instead of except
        return [], sampling_rate or target_sr or 48000
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr

    return data, sampling_rate


class STFT():
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025,
                 clip_val=1e-5):
        self.target_sr = sr

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, center=False):
        sampling_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val

        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))

        if fmax not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
            self.hann_window[str(y.device)] = torch.hann_window(self.win_size).to(y.device)

        y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
                                    mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(y, n_fft, hop_length=hop_length, win_length=win_size, window=self.hann_window[str(y.device)],
                          center=center, pad_mode='reflect', normalized=False, onesided=True)
        # print(111,spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        # print(222,spec)
        spec = torch.matmul(self.mel_basis[str(fmax) + '_' + str(y.device)], spec)
        # print(333,spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        # print(444,spec)
        return spec

    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

# NsfHiFiGAN의 wav2spec을 해당 함수'만' 구현해둔것
def wav2spec(inp_path, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sampling_rate = _hparams['audio_sample_rate']
    num_mels = _hparams['audio_num_mel_bins']
    n_fft = _hparams['fft_size']
    win_size = _hparams['win_size']
    hop_size = _hparams['hop_size']
    fmin = _hparams['fmin']
    fmax = _hparams['fmax']
    stft = STFT(sampling_rate, num_mels, n_fft, win_size, hop_size, fmin, fmax)
    with torch.no_grad():
        wav_torch, _ = load_wav_to_torch(inp_path, target_sr=stft.target_sr)
        mel_torch = stft.get_mel(wav_torch.unsqueeze(0).to(device)).squeeze(0).T
        # log mel to log10 mel
        mel_torch = 0.434294 * mel_torch
        return wav_torch.cpu().numpy(), mel_torch.cpu().numpy()


'''wav, mel = wav2spec('../raw/L-O-V-E.wav')
print(wav.shape, mel.shape)'''