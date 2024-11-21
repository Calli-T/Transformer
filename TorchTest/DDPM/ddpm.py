from diff.unet import UNet

from tqdm import tqdm
from torch import optim, nn
import torch
from collections import OrderedDict
import numpy as np
import math


class DDPM:
    def __init__(self, hparams, train_dataloader=None):
        self.hparams = hparams

        # 레이어 정규화, 샘플 각각에 대해서 수행함
        self.normalizer = nn.LayerNorm([3, hparams['IMAGE_SIZE'], hparams['IMAGE_SIZE']]).to(
            self.hparams['device'])

        # model
        self.network = UNet(
            in_channels=3,
            model_channels=128,
            out_channels=6,
            num_res_blocks=2,
            attention_resolutions=tuple([16]),
            dropout=0.0,
            channel_mult=(1, 1, 2, 2, 4, 4),
            num_classes=None,
            use_checkpoint=False,
            num_heads=1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
        ).to(self.hparams['device'])
        self.ema_network = UNet(
            in_channels=3,
            model_channels=128,
            out_channels=6,
            num_res_blocks=2,
            attention_resolutions=tuple([16]),
            dropout=0.0,
            channel_mult=(1, 1, 2, 2, 4, 4),
            num_classes=None,
            use_checkpoint=False,
            num_heads=1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
        ).to(self.hparams['device'])

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

        self.loss = 0.0

        # L1 vs MSE
        self.criterion = nn.MSELoss().to(
            hparams['device'])  # nn.L1Loss().to(hparams['device']) # nn.L1Loss().to(device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self.hparams["LEARNING_RATE"],
                                     weight_decay=self.hparams["WEIGHT_DECAY"])

        self.train_dataloader = train_dataloader

    def pred_noise(self, x_t, t, training):
        # num_images = self.hparams["BATCH_SIZE_SAMPLE"]
        # training일 경우 t가 하나의 값이 아니므로 그걸 복제해서 쓰면안되고 각각의 t에 대해 스케일링을 싹 다 해줘야한다
        if training:
            return self.network.forward(x_t, self.t_embedding_scaling(t, is_training=True))
        else:
            return self.ema_network.forward(x_t, self.t_embedding_scaling(t, is_training=False))

    def predict_start_from_noise(self, x_t, t, noise):
        x_start = self.sqrt_recip_alpha_bars[t] * x_t - self.sqrt_recipm1_alpha_bars[t] * noise

        return torch.clip(x_start, -1., 1.)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (self.posterior_mean_coef1[t] * x_start
                          +
                          self.posterior_mean_coef2[t] * x_t)

        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]

        return posterior_mean, posterior_log_variance_clipped

    def learn_range_var(self, t, model_var_values):
        min_log = self.posterior_log_variance_clipped[t]
        max_log = np.log(self.betas[t])
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        # model_variance = torch.exp(model_log_variance)

        return model_log_variance

    def p_mean_variance(self, x_t, t, num_images):
        epsilon_theta = self.pred_noise(x_t, t, training=False)  # , num_images=num_images)

        if epsilon_theta.shape[1] == 6 or self.hparams["learn_sigma"]:
            epsilon_theta, model_var_values = self.output_split_channel(epsilon_theta)
            model_var_values = self.learn_range_var(t, model_var_values)

        x_recon = self.predict_start_from_noise(x_t, t, epsilon_theta)

        model_mean, posterior_log_variance_clipped = self.q_posterior(x_recon, x_t, t)
        if epsilon_theta.shape[1] == 6 or self.hparams["learn_sigma"]:
            posterior_log_variance_clipped = model_var_values

        return model_mean, posterior_log_variance_clipped

    def p_sample_loop_ddpm(self, batch_size_sample=1, trace_diffusion=False, initial_noise=None):
        '''
        :param batch_size_sample:
        :param trace_diffusion: if True, each sample returns 11 steps for tracing.
        :param initial_noise: x_T will be replaced with this param.
        :return:
        '''
        batch_size_sample = self.hparams["BATCH_SIZE_SAMPLE"]
        if initial_noise is None:
            x_t = torch.randn(batch_size_sample, 3, self.hparams['IMAGE_SIZE'], self.hparams['IMAGE_SIZE'])
            x_t = x_t.to(self.hparams['device'])
        else:
            x_t = initial_noise

        step_footprint = x_t.detach()
        trace_t = range(0, self.hparams["steps"] + 1)[::int(self.hparams["steps"] * 0.1)]

        with torch.no_grad():
            self.network.eval()
            self.ema_network.eval()

            # each cycle operates like func p_sample()
            for t in tqdm(reversed(range(0, self.hparams["steps"]))):
                model_mean, model_log_variance = self.p_mean_variance(x_t, t, batch_size_sample)
                z = torch.randn_like(x_t)

                if t > 0:
                    if self.hparams["learn_sigma"]:
                        x_t = model_mean + z * torch.exp(0.5 * model_log_variance)
                    else:
                        x_t = model_mean + z * math.exp(0.5 * model_log_variance)
                else:
                    x_t = model_mean

                if trace_diffusion and t in trace_t:
                    step_footprint = torch.concat([step_footprint, x_t.detach()], 0)

            self.network.train()
            self.ema_network.train()

        if trace_diffusion:
            return self.convert_output_to_hex(step_footprint)
        else:
            return self.convert_output_to_hex(x_t)

    def train_steps_t_big(self):
        cost = 0.0

        for batch in tqdm(self.train_dataloader):
            batch = batch.to(self.hparams['device'])
            images = self.normalizer(batch)
            noises = torch.randn(batch.shape).to(self.hparams['device'])

            # 배치만큼 신호비 alpha와 잡음비 beta를 가져옴
            diffusion_times = np.random.randint(low=0, high=self.hparams["steps"], size=len(batch))
            signal_rates = torch.Tensor(np.sqrt(self.alpha_bars[diffusion_times])).to(self.hparams['device'])
            noise_rates = torch.Tensor(np.sqrt(1. - self.alpha_bars[diffusion_times])).to(self.hparams['device'])

            # 정방향 확산 과정은 한 큐에!
            noisy_images = torch.mul(signal_rates.view([-1, 1, 1, 1]), images) + torch.mul(
                noise_rates.view([-1, 1, 1, 1]), noises).to(self.hparams['device'])

            # u-net을 통한 잡음 예측
            # pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
            pred_noises = self.pred_noise(noisy_images, diffusion_times,
                                          training=True)  #, num_images=self.hparams["BATCH_SIZE_SAMPLE"])
            if self.hparams["learn_sigma"]:
                pred_noises, _ = self.output_split_channel(pred_noises)
            loss = self.criterion(noises, pred_noises)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            cost += loss

        # ema 신경망에 카피
        with torch.no_grad():
            model_params = OrderedDict(self.network.named_parameters())
            shadow_params = OrderedDict(self.ema_network.named_parameters())

            for name, param in model_params.items():
                shadow_params[name].sub_((1.0 - self.hparams["EMA"]) * (shadow_params[name] - param))

        cost /= (len(self.train_dataloader))

        return cost

    def train(self):
        for epoch in range(self.hparams["EPOCHS"]):
            cost = self.train_steps_t_big()  # self.train_steps()
            print(f'Epoch: {epoch + 1:4d}, Loss: {cost:3f}')

            if (epoch + 1) % self.hparams["save_interval"] == 0:
                torch.save(self.network.state_dict(), f'{self.hparams["model_path"]}/unet-{epoch + 1}.pt')
                torch.save(self.ema_network.state_dict(), f'{self.hparams["model_path"]}/ema-unet-{epoch + 1}.pt')

        torch.save(self.network.state_dict(), f'{self.hparams["model_path"]}unet.pt')
        torch.save(self.ema_network.state_dict(), f'{self.hparams["model_path"]}/ema-unet.pt')

    def load(self):
        self.network.load_state_dict(
            torch.load(f'{self.hparams["model_path"]}/unet.pt', map_location=self.hparams['device']))
        self.ema_network.load_state_dict(
            torch.load(f'{self.hparams["model_path"]}/ema-unet.pt', map_location=self.hparams['device']))

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

        '''
        # ※ 후위 분산은 DDPM 3.2장에 보면 그냥 β로 써도 별 상관 없다더라
        # posterior_standard_deviation = np.sqrt(posterior_variance)
        # ※ 시작할 때 값이 분산 값이 0이여서 미니멈 걸었놨다더라, 왜 sqrt안쓰고 log -> exp(.5배)해서 빼내는지는 나중에 알아보자
        '''

        return self.alphas, self.betas, self.posterior_log_variance_clipped

    def output_split_channel(self, model_output):
        # split output to noise(original model output) & posterior variance
        model_output, model_var_values = torch.split(model_output, 3, dim=1)

        return model_output, model_var_values

    def t_embedding_scaling(self, t, is_training=False):
        # for timestep embedding

        if is_training:
            return torch.FloatTensor(
                [(step / self.hparams["steps"]) * 1000 for step in t]).to(self.hparams['device'])
        else:
            return torch.FloatTensor(
                [(t / self.hparams["steps"]) * 1000 for x in range(self.hparams["BATCH_SIZE_SAMPLE"])]).to(
                self.hparams['device'])

    def convert_output_to_hex(self, sample):
        # [-1, 1] to [0, 255]

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        return sample


'''
    def test_forward(self):
        init = torch.randn(16, 3, self.hparams['IMAGE_SIZE'], self.hparams['IMAGE_SIZE']).to(self.hparams['device'])
        test_t = self.timestep_scaling(3600, 16)

        return self.network(init, test_t)
'''

'''
Unconditional ImageNet-64 with the L_vlb objective and cosine noise schedule
        self.network = UNet(
            in_channels=3,
            model_channels=128,
            out_channels=6,
            num_res_blocks=3,
            attention_resolutions=tuple([4, 8]),
            dropout=0.0,
            channel_mult=(1, 2, 3, 4),
            num_classes=None,
            use_checkpoint=False,
            num_heads=4,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
        ).to(self.hparams['device'])
        self.ema_network = UNet(
            in_channels=3,
            model_channels=128,
            out_channels=6,
            num_res_blocks=3,
            attention_resolutions=tuple([4, 8]),
            dropout=0.0,
            channel_mult=(1, 2, 3, 4),
            num_classes=None,
            use_checkpoint=False,
            num_heads=4,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
        ).to(self.hparams['device'])
'''

'''
    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network

        pred_noises = network.forward(noise_rates ** 2, noisy_images)
        pred_images = torch.div((noisy_images - torch.mul(noise_rates.view([-1, 1, 1, 1]), pred_noises)),
                                signal_rates.view([-1, 1, 1, 1]))

        return pred_noises, pred_images
'''
