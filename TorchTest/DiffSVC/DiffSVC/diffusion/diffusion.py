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
from torch import nn, optim


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

    def load_most_epochs_model(self):
        model_path = self.hparams["model_path"]

        # set maximum epoch, 형식은 wavenet_model_epochs_96400.pt
        pt_list = os.listdir(model_path)
        if len(pt_list) != 0:
            epoch_maximum = max([int(fname.split('.')[0].split('_')[-1]) for fname in pt_list])

            print(f"epoch: {epoch_maximum}만큼 학습된 모델 load")
            self.hparams['model_pt_epoch'] = epoch_maximum

            # load maximum epoch model
            embedding_model_path = os.path.join(model_path, f"embedding_model_epochs_{epoch_maximum}.pt")
            wavenet_model_path = os.path.join(model_path, f"wavenet_model_epochs_{epoch_maximum}.pt")
            optimizer_path = os.path.join(model_path, f"optimizer_epochs_{epoch_maximum}.pt")
            embedding_model_pt = torch.load(embedding_model_path, map_location='cpu')
            wavenet_model_pt = torch.load(wavenet_model_path, map_location='cpu')
            optimizer_pt = torch.load(optimizer_path, map_location='cpu')
            self.embedding_model.load_state_dict(embedding_model_pt)
            self.wavenet.load_state_dict(wavenet_model_pt)
            self.optimizer.load_state_dict(optimizer_pt)
        else:
            self.hparams['model_pt_epoch'] = 0

    def __init__(self, _hparams, wav2spec=None):
        self.hparams = _hparams

        # for conditioning
        if wav2spec is not None:
            self.wav2spec = wav2spec
        else:
            from .conditioning import wav2spec as w2s
            self.wav2spec = w2s
        self.crepe = get_pitch_crepe
        self.hubert = Hubertencoder(self.hparams)
        self.get_align = get_align

        # for loss
        self.criterion = nn.MSELoss(reduction='none').to(self.hparams['device'])

        # trainable models
        self.embedding_model = ConditionEmbedding(self.hparams).to(self.hparams['device'])
        # self.embedding_model.load_state_dict((torch.load(self.hparams['emb_model_path'], map_location='cpu')))
        # self.embedding_model.to(self.hparams['device'])
        self.wavenet = DiffNet(self.hparams).to(self.hparams['device'])
        # self.wavenet.load_state_dict(torch.load(self.hparams['wavenet_model_path'], map_location='cpu'))
        # self.wavenet.to(self.hparams['device'])
        self.optimizer = optim.AdamW(list(self.embedding_model.parameters()) + list(self.wavenet.parameters()),
                                     lr=self.hparams["LEARNING_RATE"], weight_decay=self.hparams["WEIGHT_DECAY"])
        self.load_most_epochs_model()

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

    def get_padded_np_conds(self, raw_wave_dir_path, saved_f0=None):
        if isinstance(raw_wave_dir_path, list):
            wav_list = []
            mel_list = []
            mel_len_list = []
            for fname in raw_wave_dir_path:
                wav, mel = self.wav2spec(fname, self.hparams)
                wav_list.append(wav)
                mel_len_list.append(mel.shape[0])
                mel_list.append(mel)

            maximum_mel_len = max(mel_len_list)
            for idx, mel in enumerate(mel_list):
                m_len = mel_len_list[idx]
                mel_list[idx] = np.pad(mel, ((0, maximum_mel_len - m_len), (0, 0)))
            mel_list = np.array(mel_list)

            if saved_f0 is not None:
                f0_list = saved_f0
            else:
                f0_list = []
                for wav, mel in zip(wav_list, mel_list):
                    gt_f0 = self.crepe(wav, mel, self.hparams)
                    f0, _ = self.norm_interp_f0(gt_f0)
                    f0_list.append(f0)
                f0_list = np.array(f0_list)

            hubert_encoded_list = []
            hubert_encoded_len_list = []
            for fname in raw_wave_dir_path:
                hubert_encoded = self.hubert.encode(fname)
                hubert_encoded_list.append(hubert_encoded)
                hubert_encoded_len_list.append(len(hubert_encoded))

            maximum_hubert_len = max(hubert_encoded_len_list)
            for idx, hubert_encoded in enumerate(hubert_encoded_list):
                m_len = hubert_encoded_len_list[idx]
                hubert_encoded_list[idx] = np.pad(hubert_encoded, ((0, maximum_hubert_len - m_len), (0, 0)))
            hubert_encoded_list = np.array(hubert_encoded_list)

            mel2ph_list = []
            for mel, hubert_encoded in zip(mel_list, hubert_encoded_list):
                mel2ph_list.append(self.get_align(mel, hubert_encoded))
            mel2ph_list = np.array(mel2ph_list)

            return {"name": raw_wave_dir_path,
                    "wav": wav_list,
                    "mel": mel_list,
                    "mel_len": mel_len_list,
                    "f0": f0_list,
                    "hubert": hubert_encoded_list,
                    "hubert_len": hubert_encoded_len_list,
                    "mel2ph": mel2ph_list}

    def get_tensor_conds(self, item):
        device = self.hparams['device']

        tensor_cond = dict()
        tensor_cond['mel'] = torch.Tensor(item['mel']).to(device)
        tensor_cond['mel2ph'] = torch.LongTensor(item['mel2ph']).to(device)
        tensor_cond['hubert'] = torch.Tensor(item['hubert']).to(device)
        tensor_cond['f0'] = torch.Tensor(item['f0']).to(device)

        tensor_cond['mel_len'] = item['mel_len']
        tensor_cond['hubert_len'] = item['hubert_len']

        return tensor_cond

    def get_conds(self, raw_wave_dir_path, saved_f0=None):
        raw_conds = self.get_padded_np_conds(raw_wave_dir_path, saved_f0)
        conds_tensor = self.get_tensor_conds(raw_conds)
        embedding = self.embedding_model(conds_tensor)

        # for mel2wav
        embedding['raw_mel2ph'] = conds_tensor['mel2ph']  # raw_cond['mel2ph']

        # for train
        embedding['raw_gt_mel'] = conds_tensor['mel']

        # for padd_masking
        embedding['mel_len'] = raw_conds['mel_len']

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

    def infer_batch(self, raw_wave_dir_path):
        with torch.no_grad():
            self.embedding_model.eval()
            self.wavenet.eval()

            ret = self.get_conds(raw_wave_dir_path)
            cond = ret['decoder_inp'].transpose(1, 2)
            M = self.hparams['audio_num_mel_bins']
            T = cond.shape[2]
            B = cond.shape[0]
            device = self.hparams['device']

            x = torch.randn((B, 1, M, T)).to(device)
            for t in tqdm(reversed(range(0, self.hparams["steps"]))):
                x = self.p_sample(x, t, cond)

        x = x[:, 0].transpose(1, 2)
        ret['mel_out'] = self.denorm_spec(x) * ((ret['raw_mel2ph'] > 0).float()[:, :, None])

        fnames = []
        for raw_wave_path in raw_wave_dir_path:
            try:
                fnames.append(raw_wave_path.split('/')[-1].split('.')[0])
            except:
                print("file name error")
        ret['filename'] = fnames

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

    def save_and_remove(self):
        model_path = self.hparams["model_path"]
        self.hparams['model_pt_epoch'] += self.hparams['save_interval']

        # save models
        embedding_model_path = os.path.join(model_path, f"embedding_model_epochs_{self.hparams['model_pt_epoch']}.pt")
        wavenet_model_path = os.path.join(model_path, f"wavenet_model_epochs_{self.hparams['model_pt_epoch']}.pt")
        optimizer_path = os.path.join(model_path, f"optimizer_epochs_{self.hparams['model_pt_epoch']}.pt")
        torch.save(self.embedding_model.state_dict(), embedding_model_path)
        torch.save(self.wavenet.state_dict(), wavenet_model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # remove models
        pt_list = os.listdir(model_path)
        if len(pt_list) > self.hparams['number_of_savepoint'] * 3:
            epoch_minimum = min([int(fname.split('.')[0].split('_')[-1]) for fname in pt_list])
            # print(epoch_minimum)
            embedding_model_path = os.path.join(model_path, f"embedding_model_epochs_{epoch_minimum}.pt")
            wavenet_model_path = os.path.join(model_path, f"wavenet_model_epochs_{epoch_minimum}.pt")
            optimizer_path = os.path.join(model_path, f"optimizer_epochs_{epoch_minimum}.pt")
            os.remove(embedding_model_path)
            os.remove(wavenet_model_path)
            os.remove(optimizer_path)

    def train_batch(self):
        self.embedding_model.train()
        self.wavenet.train()

        # for train_data_path (need refactoring)
        wav_path = os.path.join(self.hparams['train_dataset_path_output'], 'final')
        wav_fname_list = sorted(os.listdir(wav_path))
        split_to_batches = lambda original_list, BATCH_SIZE_TRAIN: [
            original_list[i:i + BATCH_SIZE_TRAIN] for i in range(0, len(original_list), BATCH_SIZE_TRAIN)
        ]
        wav_fname_list = split_to_batches(wav_fname_list, self.hparams['BATCH_SIZE_TRAIN'])

        # gen f0 files
        if not self.exist_f0_npy():
            for wav_fname_sublist in tqdm(wav_fname_list):
                # wav_list = []
                # mel_list = []
                # mel_len_list = []

                f0_list = []
                for fname in wav_fname_sublist:
                    temp_path = os.path.join(wav_path, fname)
                    wav, mel = self.wav2spec(temp_path, self.hparams)

                    # print(f"음원 '{fname}' 기본 주파수 추출 작업 중")
                    gt_f0 = self.crepe(wav, mel, self.hparams)
                    f0, _ = self.norm_interp_f0(gt_f0)
                    f0_list.append(f0)

                    # wav_list.append(wav)
                    # mel_len_list.append(mel.shape[0])
                    # mel_list.append(mel)

                # maximum_mel_len = max(mel_len_list)
                # for idx, mel in enumerate(mel_list):
                #     m_len = mel_len_list[idx]
                #     mel_list[idx] = np.pad(mel, ((0, maximum_mel_len - m_len), (0, 0)))
                # mel_list = np.array(mel_list)

                # for wav, mel in zip(wav_list, mel_list):
                #     gt_f0 = self.crepe(wav, mel, self.hparams)
                #     f0, _ = self.norm_interp_f0(gt_f0)
                #     f0_list.append(f0)
                #     # print(f"음원 '{fname}' 기본 주파수 추출 작업 중")

                for f0, fname in zip(f0_list, wav_fname_sublist):
                    save_path = os.path.join(self.hparams['train_dataset_path_f0'], fname + "_f0.npy")
                    # print(f'음원 {fname}의 기본 주파수 {save_path}에 저장')
                    np.save(save_path, f0)

        for epoch in tqdm(range(self.hparams['train_target_epochs'])):
            cost = 0.0

            # train
            for wav_fname_sublist in tqdm(wav_fname_list):
                f0 = []
                temp_path = []

                for wav_fname in wav_fname_sublist:
                    # for saved_f0
                    save_path = os.path.join(self.hparams['train_dataset_path_f0'], wav_fname + "_f0.npy")
                    f0.append(np.load(save_path))

                    # paths
                    temp_path.append(os.path.join(wav_path, wav_fname))

                # pad f0
                max_f0_length = max([len(f_temp) for f_temp in f0])
                for idx, f_temp in enumerate(f0):
                    f0[idx] = np.pad(f_temp, (0, max_f0_length - len(f_temp)))
                f0 = np.array(f0)
                f0 = torch.from_numpy(f0).to(self.hparams['device'])

                # condition
                ret = self.get_conds(temp_path, f0)
                cond = ret['decoder_inp'].transpose(1, 2)
                gt_mel = ret['raw_gt_mel']
                B1MT_input_mel = gt_mel.unsqueeze(1).transpose(2, 3)
                B1MT_input_mel = self.norm_spec(B1MT_input_mel)
                B = cond.shape[0]

                # diffuse
                noises = torch.randn(B1MT_input_mel.shape).to(self.hparams['device'])
                diffusion_times = np.random.randint(low=0, high=self.hparams["steps"],
                                                    size=B)
                signal_rates = torch.Tensor(np.array(np.sqrt(self.alpha_bars[diffusion_times]))).to(
                    self.hparams['device'])
                noise_rates = torch.Tensor(np.array(np.sqrt(1. - self.alpha_bars[diffusion_times]))).to(
                    self.hparams['device'])
                noisy_images = torch.mul(signal_rates.view([-1, 1, 1, 1]), B1MT_input_mel) + torch.mul(
                    noise_rates.view([-1, 1, 1, 1]), noises).to(self.hparams['device'])
                # print(noisy_images.shape)

                # predict noise
                diffusion_times = torch.from_numpy(diffusion_times).to(self.hparams['device'])
                pred_noises = self.wavenet(noisy_images, diffusion_times, cond)

                # training
                loss = self.criterion(noises, pred_noises)
                # masking
                original_frame = ret['mel_len']
                max_seq = max(original_frame)
                mask = torch.zeros(B, 1, 1, max_seq, dtype=torch.bool).to(self.hparams['device'])
                for i, len_seg in enumerate(original_frame):
                    mask[i, :, :, :len_seg] = True
                masked_loss = loss * mask
                masked_loss = masked_loss.mean()
                # back propagation
                self.optimizer.zero_grad()
                masked_loss.backward()  # loss.backward()
                self.optimizer.step()
                cost += (masked_loss / B)

            print(f"Epoch: {epoch + 1}, Loss:{cost:.4f}")
            if (epoch + 1) % self.hparams['save_interval'] == 0:
                self.save_and_remove()
            # break


'''
    def train(self):
        self.embedding_model.train()
        self.wavenet.train()

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
        for epoch in tqdm(range(self.hparams['train_target_epochs'])):
            # '일단은' 한 파일씩 학습함
            cost = 0.0

            for fname in wav_list:
                # print(f"'{fname}'파일 작업중")
                temp_path = os.path.join(wav_path, fname)
                save_path = os.path.join(self.hparams['train_dataset_path_f0'], fname + "_f0.npy")
                f0 = np.load(save_path)

                # - for model input -
                ret = self.get_cond(temp_path, f0)
                cond = ret['decoder_inp'].transpose(1, 2)
                gt_mel = torch.from_numpy(ret['raw_gt_mel']).to(self.hparams['device'])
                B1MT_input_mel = gt_mel.unsqueeze(0).unsqueeze(1).transpose(2, 3)
                B1MT_input_mel = B1MT_input_mel.expand(self.hparams['batch_size_train'], -1, -1, -1)
                B1MT_input_mel = self.norm_spec(B1MT_input_mel)  # 정상화
                # print(cond.shape)
                # print(B1MT_input_mel.shape)

                # 잡음 먹이기
                noises = torch.randn(B1MT_input_mel.shape).to(self.hparams['device'])
                diffusion_times = np.random.randint(low=0, high=self.hparams["steps"],
                                                    size=self.hparams['batch_size_train'])
                t = int(diffusion_times[0])
                signal_rates = torch.Tensor(np.array(np.sqrt(self.alpha_bars[t]))).to(self.hparams['device'])
                noise_rates = torch.Tensor(np.array(np.sqrt(1. - self.alpha_bars[t]))).to(self.hparams['device'])
                noisy_images = torch.mul(signal_rates.view([-1, 1, 1, 1]), B1MT_input_mel) + torch.mul(
                    noise_rates.view([-1, 1, 1, 1]), noises).to(self.hparams['device'])

                # 일단 임시로 diffusion_times는 배열이 아니라 원시 int 하나만 보낸다, 개조전임

                # print(signal_rates, noise_rates)
                step = torch.Tensor([t]).to(self.hparams['device'])
                pred_noises = self.wavenet(noisy_images, step, cond)
                # print(pred_noises.shape)

                # for loss, comparison (origin)
                # print(gt_mel.shape)
                # print(gt_mel.shape, pred_noises.shape)
                print(pred_noises.shape)
                loss = self.criterion(noises, pred_noises)
                print(type(loss))
                print(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                cost += loss

                # print()

            print(f"Epoch: {epoch + 1}, Loss:{cost / len(wav_list):.4f}")
'''

'''
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

        return ret
'''

'''
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
'''
'''
    def get_raw_cond(self, raw_wave_path, saved_f0=None):
        wav, mel = self.wav2spec(raw_wave_path, self.hparams)

        if saved_f0 is not None:
            f0 = saved_f0
        else:
            gt_f0 = self.crepe(wav, mel, self.hparams)
            f0, _ = self.norm_interp_f0(gt_f0)

        hubert_encoded = self.hubert.encode(raw_wave_path)

        mel2ph = self.get_align(mel, hubert_encoded)

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
'''
