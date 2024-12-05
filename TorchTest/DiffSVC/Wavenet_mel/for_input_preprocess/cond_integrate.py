import numpy as np
import torch

from NsfHiFiGAN.nsf_hifigan import NsfHifiGAN
from temp_hparams import hparams, rel2abs, dir2list
from CREPE.crepe import get_pitch_crepe
from HuBERT.hubertinfer import Hubertencoder
from mel2ph.mel2ph import get_align


def load_cond_model(_hparams):
    return NsfHifiGAN(_hparams), Hubertencoder(_hparams)


def get_raw_cond(_vocoder, _hubert, _hparams, abs_raw_wav_path):
    wav, mel = _vocoder.wav2spec(abs_raw_wav_path, _hparams)
    f0, coarse_f0 = get_pitch_crepe(wav, mel, _hparams)
    hubert_encoded = _hubert.encode(rel2abs(_hparams['raw_wave_path']))
    mel2ph = get_align(mel, hubert_encoded)

    return {"name": abs_raw_wav_path,
            "wav": wav,
            "mel": mel,
            "f0": f0,
            "pitch": coarse_f0,
            "hubert": hubert_encoded,
            "mel2ph": mel2ph}


def norm_interp_f0(_f0, _hparams):
    # 기본 주파수를 로그스케일로 바꾸고, 보간하고, f0가 0인 곳을 uv 마스크로 만든다
    # 원본 코드랑 반대로, 반드시 numpy array만 받아주니 유의!
    uv = _f0 == 0

    _f0 = np.log2(_f0)

    if sum(uv) == len(_f0):
        _f0[uv] = 0
    elif sum(uv) > 0:
        _f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], _f0[~uv])

    return _f0, uv


def get_tensor_cond(item, _hparams):
    max_frames = hparams['max_frames']
    max_input_tokens = hparams['max_input_tokens']
    device = hparams['device']

    f0, _ = norm_interp_f0(item['f0'], _hparams)  # 저 _는 uv다

    item['mel'] = torch.Tensor(item['mel'][:max_frames]).to(device)
    item['mel2ph'] = torch.LongTensor(item['mel2ph'][:max_frames]).to(device)
    item['hubert'] = torch.Tensor(item['hubert'][:max_input_tokens]).to(device)
    item['f0'] = torch.Tensor(f0[:max_frames]).to(device)
    item['pitch'] = torch.LongTensor(item['pitch'][:max_frames]).to(device)

    return item


'''sample = get_tensor_cond(get_raw_cond(*load_cond_model(hparams), hparams, rel2abs(hparams['raw_wave_path'])), hparams)
print()
print(sample['mel'].shape)
print(sample['mel2ph'].shape)
print(sample['hubert'].shape)
print(sample['f0'].shape)
print(sample['pitch'].shape)'''


def condtion_integrate(item):
    def collate_1d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):

        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values) if max_len is None else max_len
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if shift_right:
                dst[1:] = src[:-1]
                dst[0] = shift_id
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res

    def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
        """Convert a list of 2d tensors into a padded 3d tensor."""
        size = max(v.size(0) for v in values) if max_len is None else max_len
        res = values[0].new(len(values), size, values[0].shape[1]).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if shift_right:
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res

    # item이 list라도 받아들일 수 있도록 개조해주자, 원본 코드는 list comprihension을 사용하였다
    # 원본 코드는 batch를 만들도록 되어있다 함수명 자체가 processed_input2batch
    # 원본 코드도 노래 하나씩만 넣는데 ??? filename list에 2개 넣어봐도 각각 따로한다
    item['hubert'] = collate_2d([item['hubert']], 0.0)
    item['f0'] = collate_1d([item['f0']], 0.0)
    item['pitch'] = collate_1d([item['pitch']])
    item['mel2ph'] = collate_1d([item['mel2ph']], 0.0)  # 이거 없는 것도 if로 처리하더라
    item['mel'] = collate_2d([item['mel']], 0.0)


sample = get_tensor_cond(get_raw_cond(*load_cond_model(hparams), hparams, rel2abs(hparams['raw_wave_path'])), hparams)
condtion_integrate(sample)
print(sample['mel'].shape)
print(sample['mel2ph'].shape)
print(sample['hubert'].shape)
print(sample['f0'].shape)
print(sample['pitch'].shape)

'''
# vocoder = NsfHifiGAN(hparams)
# wav, mel = vocoder.wav2spec(rel2abs(hparams['raw_wave_path']), hparams)
# print(wav.shape, mel.shape)

# from wav2spec_stand_alone import wav2spec
# from NsfHiFiGAN.nsf_hifigan import NsfHifiGAN
from CREPE.crepe import get_pitch_crepe

# from temp_hparams import hparams, rel2abs

# wav, mel = NsfHifiGAN.wav2spec(rel2abs(hparams['raw_wave_path']), hparams)
# gt_f0, coarse_f0 = get_pitch_crepe(wav, mel, hparams)
# print(wav.shape, mel.shape)
# print(gt_f0.shape, coarse_f0.shape)

from HuBERT.hubertinfer import Hubertencoder

# from temp_hparams import hparams, rel2abs

# hubert = Hubertencoder(hparams)
# hubert_encoded = hubert.encode(rel2abs(hparams['raw_wave_path']))
# print(hubert_encoded.shape)

from mel2ph.mel2ph import get_align

# mel2ph = get_align(mel, hubert_encoded)
'''

# def condtion_integrate(_mel, _f0, _coarse_f0, _hubert_encoded, _mel2ph):
#     def collate_1d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
#         """Convert a list of 1d tensors into a padded 2d tensor."""
#         size = max(v.size(0) for v in values) if max_len is None else max_len
#         res = values[0].new(len(values), size).fill_(pad_idx)
#
#         def copy_tensor(src, dst):
#             assert dst.numel() == src.numel()
#             if shift_right:
#                 dst[1:] = src[:-1]
#                 dst[0] = shift_id
#             else:
#                 dst.copy_(src)
#
#         for i, v in enumerate(values):
#             copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
#         return res
#
#     def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
#         """Convert a list of 2d tensors into a padded 3d tensor."""
#         size = max(v.size(0) for v in values) if max_len is None else max_len
#         res = values[0].new(len(values), size, values[0].shape[1]).fill_(pad_idx)
#
#         def copy_tensor(src, dst):
#             assert dst.numel() == src.numel()
#             if shift_right:
#                 dst[1:] = src[:-1]
#             else:
#                 dst.copy_(src)
#
#         for i, v in enumerate(values):
#             copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
#         return res
#
#     '''
#             Args:
#                 samples: one batch of processed_input
#             NOTE:
#                 the batch size is controlled by hparams['max_sentences']
#         '''
#     hubert_encoded_ = collate_2d(_hubert_encoded, 0.0)
#     f0_ = collate_1d(_f0, 0.0)
#     pitch_ = collate_1d(_coarse_f0)
#     mel2ph_ = collate_1d(_mel2ph, 0.0)  # 이거 없는 것도 if로 처리하더라
#     mel_ = collate_2d(_mel, 0.0)
#
#     return hubert_encoded_, f0_, pitch_, mel2ph_, mel_
#
# # 오류 있음
# vocoder, hubert = load_cond_model(hparams)
# wav, mel, f0, coarse_f0, hubert_encoded, mel2ph = cond_preprocess(vocoder, hubert, hparams)
# hubert_encoded, f0, pitch, mel2ph, mel = condtion_integrate(mel, f0, coarse_f0, hubert_encoded, mel2ph)
# print(hubert_encoded.shape)
# print(f0.shape)
# print(pitch.shape)
# print(mel2ph.shape)
# print(mel.shape)

'''
def get_batch_dict(raw_dir_path, _vocoder, _hubert, _hparams):
    names = dir2list(raw_dir_path)

    for name in names:
        print(name)
        wav, mel = _vocoder.wav2spec(name, _hparams)
        print(wav.shape, mel.shape)


# get_batch_dict(hparams['raw_dir_path'], *load_cond_model(hparams), hparams)
vocoder, hubert = load_cond_model(hparams)
wav, mel, f0, coarse_f0, hubert_encoded, mel2ph = cond_preprocess(vocoder, hubert, hparams)
print(mel.shape, hubert_encoded.shape, mel2ph.shape, f0.shape)
'''
