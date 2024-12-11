from .wavenet.net import DiffNet
from .embedding_model.embedding_model import ConditionEmbedding

import math
import numpy as np
import torch


class GuassianDiffusion:
    def set_schedule(self):
        t = self.hparams["steps"]

        betas = []
        if self.hparams["schedule_name"] == "cosine":
            for i in range(t):
                t1 = i / t
                t2 = (i + 1) / t
                alpha_bar_t1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
                alpha_bar_t2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2
                betas.append(min(1 - alpha_bar_t2 / alpha_bar_t1, 0.999))  # max_beta))
            betas = np.array(betas, dtype=np.float64)
        else:
            scale = 1000 / self.hparams["steps"]
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            betas = np.linspace(beta_start, beta_end, self.hparams["steps"], dtype=np.float64)

        alphas = 1.0 - betas
        self.alphas = alphas
        self.betas = betas

        self.alpha_bars = np.cumprod(alphas, axis=0)
        self.alpha_bars_prev = np.append(1., self.alpha_bars[:-1])

        # for what?
        self.sqrt_recip_alpha_bars = np.sqrt(1.0 / self.alpha_bars)
        self.sqrt_recipm1_alpha_bars = np.sqrt(1.0 / self.alpha_bars - 1)

        # for posterior distribution q(x_t-1 | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alpha_bars_prev) / (1. - self.alpha_bars)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(self.alpha_bars_prev) / (1. - self.alpha_bars)
        self.posterior_mean_coef2 = (1. - self.alpha_bars_prev) * np.sqrt(alphas) / (1. - self.alpha_bars)

    def __init__(self, _hparams):
        self.hparams = _hparams
        '''
        schedule의 register_buffer는 나중에 고려
        criterion, optimizer를 만들자
        q_sample과 샘플링 과정을 만들자, t의 변환은 당장은 필요없다
        p_sample과 학습 과정을 만들자
        embedding_model과 wavenet model을 장착하자
        norm/denorm을 구현해두자 ※ 고정값으로 해두고 opencpop으로 만들어진 값이므로 나중에 대체하자
        '''

        # trainable models
        self.embedding_model = ConditionEmbedding(self.hparams)
        self.embedding_model.load_state_dict((torch.load(self.hparams['emb_model_path'], map_location='cpu')))
        self.embedding_model.to(self.hparams['device'])
        self.wavenet = DiffNet(self.hparams)
        self.wavenet.load_state_dict(torch.load(self.hparams['wavenet_model_path'], map_location='cpu'))
        self.wavenet.to(self.hparams['device'])

        # pretrained models (for conditioning)

        # schedule
        self.alphas = None
        self.betas = None
        self.alpha_bars = None
        self.alpha_bars_prev = None
        self.posterior_variance = None
        self.posterior_log_variance_clipped = None
        self.posterior_mean_coef1 = None
        self.posterior_mean_coef2 = None
        self.sqrt_recip_alpha_bars = None
        self.sqrt_recipm1_alpha_bars = None
        self.set_schedule()
