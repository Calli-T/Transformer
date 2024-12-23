from .wavenet.net import DiffNet
from .embedding_model.embedding_model import ConditionEmbedding
from .conditioning.CREPE.crepe import get_pitch_crepe
from .conditioning.HuBERT.hubertinfer import Hubertencoder
from .conditioning import get_align

import math
import numpy as np
import torch
from tqdm import tqdm
import time
import os


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

    def __init__(self, _hparams, wav2spec=None):
        self.hparams = _hparams
        '''
        schedule의 register_buffer는 나중에 고려
        criterion, optimizer를 만들자
        q_sample과 샘플링 과정을 만들자, t의 변환은 당장은 필요없다
        p_sample과 학습 과정을 만들자
        embedding_model과 wavenet model을 장착하자
        norm/denorm을 구현해두자 ※ 고정값으로 해두고 opencpop으로 만들어진 값이므로 나중에 대체하자
        '''

        # for conditioning
        if wav2spec is not None:
            self.wav2spec = wav2spec
        else:
            from .conditioning import wav2spec as w2s
            self.wav2spec = w2s
        self.crepe = get_pitch_crepe
        self.hubert = Hubertencoder(self.hparams)
        self.get_align = get_align

        # trainable models
        self.embedding_model = ConditionEmbedding(self.hparams)
        self.embedding_model.load_state_dict((torch.load(self.hparams['emb_model_path'], map_location='cpu')))
        self.embedding_model.to(self.hparams['device'])
        self.wavenet = DiffNet(self.hparams)
        self.wavenet.load_state_dict(torch.load(self.hparams['wavenet_model_path'], map_location='cpu'))
        self.wavenet.to(self.hparams['device'])

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

        # for data scaling ([-1, 1] min-max normalize)
        spec_min = np.array(self.hparams['spec_min'])
        spec_max = np.array(self.hparams['spec_max'])
        self.spec_min = torch.FloatTensor(spec_min)[None, None, :self.hparams['keep_bins']].to(self.hparams['device'])
        self.spec_max = torch.FloatTensor(spec_max)[None, None, :self.hparams['keep_bins']].to(self.hparams['device'])

    def norm_interp_f0(self, _f0):
        # f0를 보간하고 정규화
        uv = _f0 == 0
        _f0 = np.log2(_f0)
        if sum(uv) == len(_f0):
            _f0[uv] = 0
        elif sum(uv) > 0:
            _f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], _f0[~uv])
        return _f0, uv

    def get_raw_cond(self, raw_wave_path, saved_f0=None):
        start_time = time.time()
        wav, mel = self.wav2spec(raw_wave_path, self.hparams)
        print(f'{time.time() - start_time:.4f}초')
        start_time = time.time()
        # print(wav.shape, mel.shape)
        # print(f"mel.shape: {mel.shape}")

        if saved_f0 is not None:
            f0 = saved_f0
        else:
            gt_f0 = self.crepe(wav, mel, self.hparams)
            f0, _ = self.norm_interp_f0(gt_f0)
        print(f'{time.time() - start_time:.4f}초')
        start_time = time.time()
        # print(f0.shape)

        hubert_encoded = self.hubert.encode(raw_wave_path)
        # print(hubert_encoded.shape)
        print(f'{time.time() - start_time:.4f}초')
        start_time = time.time()

        mel2ph = self.get_align(mel, hubert_encoded)
        # print(mel2ph.shape)
        print(f'{time.time() - start_time:.4f}초')

        return {"name": raw_wave_path,
                "wav": wav,
                "mel": mel,
                "f0": f0,
                "hubert": hubert_encoded,
                "mel2ph": mel2ph}

    def get_tensor_cond(self, item):
        max_frames = self.hparams['max_frames']
        max_input_tokens = self.hparams['max_input_tokens']
        device = self.hparams['device']

        tensor_cond = dict()
        tensor_cond['mel'] = torch.Tensor(item['mel'][:max_frames]).to(device)
        tensor_cond['mel2ph'] = torch.LongTensor(item['mel2ph'][:max_frames]).to(device)
        tensor_cond['hubert'] = torch.Tensor(item['hubert'][:max_input_tokens]).to(device)
        tensor_cond['f0'] = torch.Tensor(item['f0'][:max_frames]).to(device)

        return tensor_cond

    def get_collated_cond(self, item):
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

        collated_cond = dict()
        collated_cond['hubert'] = collate_2d([item['hubert']], 0.0)
        collated_cond['f0'] = collate_1d([item['f0']], 0.0)
        collated_cond['mel2ph'] = collate_1d([item['mel2ph']], 0.0)  # 이거 없는 것도 if로 처리하더라
        collated_cond['mel'] = collate_2d([item['mel']], 0.0)

        return collated_cond

    def get_cond(self, raw_wave_path, saved_f0=None):
        raw_cond = self.get_raw_cond(raw_wave_path, saved_f0)
        cond_tensor = self.get_tensor_cond(raw_cond)
        collated_tensor = self.get_collated_cond(cond_tensor)
        self.embedding_model.eval()
        embedding = self.embedding_model(collated_tensor)

        # for mel2wav
        embedding['raw_mel2ph'] = collated_tensor['mel2ph']  # raw_cond['mel2ph']

        # for train
        embedding['raw_gt_mel'] = raw_cond['mel']

        return embedding

    def predict_start_from_noise(self, x_t, t, noise):
        x_start = self.sqrt_recip_alpha_bars[t] * x_t - self.sqrt_recipm1_alpha_bars[t] * noise

        return x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (self.posterior_mean_coef1[t] * x_start
                          +
                          self.posterior_mean_coef2[t] * x_t)

        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]

        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised=True):
        step = torch.Tensor([t]).to(self.hparams['device'])
        epsilon_theta = self.wavenet(x, step, cond)
        x_recon = self.predict_start_from_noise(x, t, epsilon_theta)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance_clipped = self.q_posterior(x_recon, x, t)

        return model_mean, posterior_log_variance_clipped

    def p_sample(self, x, t, cond):
        model_mean, model_log_variance = self.p_mean_variance(x, t, cond, clip_denoised=True)
        # print(type(model_mean), type(model_log_variance))
        # 저 평균과 분산이 register_buffer의 주요 등록 대상이다

        z = torch.randn_like(x)
        if t > 0:
            x = model_mean + z * math.exp(0.5 * model_log_variance)
        else:
            x = model_mean
        # print(x.shape, t, cond.shape)

        return x

    def infer(self, raw_wave_path):
        with torch.no_grad():
            self.embedding_model.eval()
            self.wavenet.eval()

            ret = self.get_cond(raw_wave_path)
            cond = ret['decoder_inp'].transpose(1, 2)
            M = self.hparams['audio_num_mel_bins']
            T = cond.shape[2]
            B = 1
            device = self.hparams['device']

            x = torch.randn((B, 1, M, T)).to(device)
            for t in tqdm(reversed(range(0, self.hparams["steps"]))):
                x = self.p_sample(x, t, cond)

        x = x[:, 0].transpose(1, 2)
        ret['mel_out'] = self.denorm_spec(x) * ((ret['raw_mel2ph'] > 0).float()[:, :, None])
        try:
            ret['filename'] = raw_wave_path.split('/')[-1].split('.')[0]
        except:
            print("file name error")

        # print(f"mel_out.shape: {ret['mel_out'].shape}")

        return ret

    def norm_spec(self, x):
        T = x.shape[1]
        spec_min_expand = self.spec_min.expand(1, T, self.hparams['audio_num_mel_bins']).transpose(1, 2)
        spec_max_expand = self.spec_max.expand(1, T, self.hparams['audio_num_mel_bins']).transpose(1, 2)
        spec_min_expand = spec_min_expand.expand(self.hparams['batch_size_train'], -1, -1, -1)
        spec_max_expand = spec_max_expand.expand(self.hparams['batch_size_train'], -1, -1, -1)

        return (x - spec_min_expand) / (spec_max_expand - spec_min_expand) * 2 - 1
        # return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        print(self.spec_max.shape)
        '''print(x.shape, self.spec_min.shape, self.spec_max.shape)
        print(type(x), type(self.spec_min), type(self.spec_max))'''
        T = x.shape[1]
        spec_min_expand = self.spec_min.expand(1, T, self.hparams['audio_num_mel_bins'])
        spec_max_expand = self.spec_max.expand(1, T, self.hparams['audio_num_mel_bins'])
        return (x + 1) / 2 * (spec_max_expand - spec_min_expand) + spec_min_expand

    # ----- for train (or trainer class) -----

    def exist_f0_npy(self):
        # f0 파일이 있는지 확인
        f0_npy_dir = self.hparams['train_dataset_path_f0']
        sep_outputs_dir = os.path.join(self.hparams['train_dataset_path_output'], 'final')
        if os.path.isdir(sep_outputs_dir) and os.path.isdir(f0_npy_dir):
            if len(os.listdir(f0_npy_dir)) == len(os.listdir(sep_outputs_dir)):
                print('학습용 f0 파일 확인, 해당 파일을 사용해 학습 시작')
                return True
            else:
                return False
        else:
            return False

    def train(self):
        wav_path = os.path.join(self.hparams['train_dataset_path_output'], 'final')
        wav_list = os.listdir(wav_path)

        if not self.exist_f0_npy():
            for fname in wav_list:
                temp_path = os.path.join(wav_path, fname)

                print(f"음원 '{fname}' 기본 주파수 추출 작업 중")
                wav, mel = self.wav2spec(temp_path, self.hparams)
                gt_f0 = self.crepe(wav, mel, self.hparams)
                f0, _ = self.norm_interp_f0(gt_f0)

                save_path = os.path.join(self.hparams['train_dataset_path_f0'], fname + "_f0.npy")
                np.save(save_path, f0)

        # '일단은' 한 파일씩 학습함
        for fname in wav_list:
            print(f"'{fname}'파일 작업중")
            temp_path = os.path.join(wav_path, fname)
            save_path = os.path.join(self.hparams['train_dataset_path_f0'], fname + "_f0.npy")
            f0 = np.load(save_path)

            # - for model input -
            ret = self.get_cond(temp_path, f0)
            cond = ret['decoder_inp'].transpose(1, 2)
            gt_mel = torch.from_numpy(ret['raw_gt_mel']).to(self.hparams['device'])
            B1MT_input_mel = gt_mel.detach().unsqueeze(0).unsqueeze(1).transpose(2, 3)
            B1MT_input_mel = B1MT_input_mel.expand(self.hparams['batch_size_train'], -1, -1, -1)
            B1MT_input_mel = self.norm_spec(B1MT_input_mel)  # 정상화
            # print(cond.shape)
            # print(B1MT_input_mel.shape)

            # 잡음 먹이기
            noises = torch.randn(B1MT_input_mel.shape).to(self.hparams['device'])
            diffusion_times = np.random.randint(low=0, high=self.hparams["steps"],
                                                size=self.hparams['batch_size_train'])
            signal_rates = torch.Tensor(np.sqrt(self.alpha_bars[diffusion_times])).to(self.hparams['device'])
            noise_rates = torch.Tensor(np.sqrt(1. - self.alpha_bars[diffusion_times])).to(self.hparams['device'])
            noisy_images = torch.mul(signal_rates.view([-1, 1, 1, 1]), B1MT_input_mel) + torch.mul(
                noise_rates.view([-1, 1, 1, 1]), noises).to(self.hparams['device'])

            # 일단 임시로 diffusion_times는 배열이 아니라 원시 int 하나만 보낸다, 개조전임
            t = diffusion_times[0]
            # print(signal_rates, noise_rates)
            pred_noises = self.p_sample(B1MT_input_mel, t, cond)
            # print(pred_noises.shape)

            # for loss, comparison (origin)
            gt_mel = gt_mel.detach().expand(self.hparams['batch_size_train'], -1, -1)  # .transpose(1, 2)
            # print(gt_mel.shape)

            # print(cond.shape)
            '''
            M = self.hparams['audio_num_mel_bins']
            T = cond.shape[2]
            B = 1
            device = self.hparams['device']

            x = torch.randn((B, 1, M, T)).to(device)
            for t in tqdm(reversed(range(0, self.hparams["steps"]))):
                x = self.p_sample(x, t, cond)
            '''
            print()
