import torch
import numpy as np
import torchcrepe
import resampy

'''def f0_to_coarse(f0, _hparams):
    f0_bin = _hparams['f0_bin']
    f0_max = _hparams['f0_max']
    f0_min = _hparams['f0_min']
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse'''


def get_pitch_crepe(wav_data, mel, _hparams, threshold=0.05):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = _hparams['device']
    # crepe只支持16khz采样率，需要重采样
    wav16k = resampy.resample(wav_data, _hparams['audio_sample_rate'], 16000)
    wav16k_torch = torch.FloatTensor(wav16k).unsqueeze(0).to(device)

    # 频率范围
    f0_min = _hparams['f0_min']
    f0_max = _hparams['f0_max']

    # 重采样后按照hopsize=80,也就是5ms一帧分析f0
    f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, f0_min, f0_max, pad=True, model='full', batch_size=1024,
                                device=device, return_periodicity=True)

    # 滤波，去掉静音，设置uv阈值，参考原仓库readme
    pd = torchcrepe.filter.median(pd, 3)
    pd = torchcrepe.threshold.Silence(-60.)(pd, wav16k_torch, 16000, 80)
    f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    f0 = torchcrepe.filter.mean(f0, 3)

    # 将nan频率（uv部分）转换为0频率
    f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)

    '''
    np.savetxt('问棋-crepe.csv',np.array([0.005*np.arange(len(f0[0])),f0[0].cpu().numpy()]).transpose(),delimiter=',')
    '''

    # 去掉0频率，并线性插值
    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005 * nzindex.cpu().numpy()
    time_frame = np.arange(len(mel)) * _hparams['hop_size'] / _hparams['audio_sample_rate']
    if f0.shape[0] == 0:
        f0 = torch.FloatTensor(time_frame.shape[0]).fill_(0)
        print('f0 all zero!')
    else:
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
    # pitch_coarse = f0_to_coarse(f0, _hparams)
    return f0  # , pitch_coarse
