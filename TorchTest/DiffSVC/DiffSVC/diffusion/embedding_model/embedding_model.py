import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class ConditionEmbedding(nn.Module):
    def __init__(self, _hparams):
        super().__init__()
        self.hparams = _hparams
        self.hidden_size = self.hparams['hidden_size']
        self.out_dims = self.hparams['audio_num_mel_bins']

        self.padding_idx = 0
        self.pitch_embed = nn.Embedding(300, self.hidden_size, self.padding_idx)
        nn.init.normal_(self.pitch_embed.weight, mean=0, std=self.hidden_size ** -0.5)
        nn.init.constant_(self.pitch_embed.weight[self.padding_idx], 0)

        # 안쓰는 코드다
        # self.mel_out = nn.Linear(self.hidden_size, self.out_dims, bias=True)
        # nn.init.xavier_uniform_(self.mel_out.weight)
        # nn.init.constant_(self.mel_out.bias, 0)

    def add_pitch(self, f0, mel2ph, ret):
        pitch_padding = (mel2ph == 0)
        ret['f0_denorm'] = f0_denorm = denorm_f0(f0)
        if pitch_padding is not None:
            f0[pitch_padding] = 0
        pitch = f0_to_coarse(f0_denorm, self.hparams)
        pitch_embedding = self.pitch_embed(pitch)
        return pitch_embedding

    def forward(self, items_dict):
        ret = {}

        encoder_out = items_dict['hubert']
        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = items_dict['mel2ph'][..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)
        tgt_nonpadding = (items_dict['mel2ph'] > 0).float()[:, :, None]  # 그 값들 전부 0보단 클텐데?
        decoder_inp = decoder_inp + self.add_pitch(items_dict['f0'], items_dict['mel2ph'], ret)
        ret['decoder_inp'] = decoder_inp * tgt_nonpadding

        return ret


'''def load_cond_embedding_state(abs_cond_model_path):
    model_state_dict = torch.load(abs_cond_model_path, map_location='cpu')

    return model_state_dict'''


def denorm_f0(_f0):
    return 2 ** _f0


def f0_to_coarse(f0, hparams):
    f0_bin = hparams['f0_bin']
    f0_max = hparams['f0_max']
    f0_min = hparams['f0_min']
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse
