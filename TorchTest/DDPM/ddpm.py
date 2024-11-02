from diff.unet import UNet
from utils import show_images

import tqdm
from torch import optim, nn
import torch
from collections import OrderedDict
import numpy as np


class DDPM:
    def __init__(self, hparams, train_dataloader):
        self.hparams = hparams

        self.normalizer = nn.LayerNorm([3, hparams['IMAGE_SIZE'], hparams['IMAGE_SIZE']]).to(
            self.hparams['device'])  # 레이어 정규화, 샘플 각각에 대해서 수행함
        self.network = UNet(hparams).to(hparams['device'])
        self.ema_network = UNet(hparams).to(hparams['device'])

        self.loss = 0.0

        # hyper
        self.EMA = 0.999
        self.reverse_diffusion_steps = 20
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-4
        self.EPOCHS = 5000

        # train_set 평균 절대 오차/RMSprop사용 -> L1 loss/RMSprop사용
        self.criterion = nn.L1Loss().to(hparams['device'])  # nn.MSELoss().to(device)  # nn.L1Loss().to(device)
        self.ratio = 0.5
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self.LEARNING_RATE, weight_decay=self.WEIGHT_DECAY)

        self.train_dataloader = train_dataloader
        self.mean = torch.FloatTensor(hparams['mean']).to(self.hparams['device'])
        self.std = torch.FloatTensor(hparams['std']).to(self.hparams['device'])
        np.save(f'{hparams["model_path"]}/mean.npy', hparams['mean'])
        np.save(f'{hparams["model_path"]}/std.npy', hparams['std'])

    def denomalize(self, images):
        images = torch.clamp(self.mean + self.std * images, min=0, max=1)
        return images

    # denoise함수 테스트중
    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network

        pred_noises = network.forward(noise_rates ** 2, noisy_images)

        # 차원 혹은 차원 축 정렬
        pred_images = torch.div((noisy_images - torch.mul(noise_rates.view([-1, 1, 1, 1]), pred_noises)),
                                signal_rates.view([-1, 1, 1, 1]))

        return pred_noises, pred_images

    # 역방향 확산의 반복부터 다시 시작하면된다
    def p_sample_loop_ddim(self, initial_noise, diffusion_steps=None, return_all_t=False):
        if diffusion_steps is None:
            diffusion_steps = self.reverse_diffusion_steps

        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        step_footprint = initial_noise.detach()

        for step in range(diffusion_steps):
            diffusion_times = [1 - step * step_size for _ in range(num_images)]
            noise_rates, signal_rates = self.diffusion_schedule(
                torch.FloatTensor(diffusion_times)
            )
            noise_rates = noise_rates.to(self.hparams['device'])
            signal_rates = signal_rates.to(self.hparams['device'])

            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=True  # False
            )

            if return_all_t:
                step_footprint = torch.concat([step_footprint, pred_images.detach()], 0)

            # next_diffusion_times = diffusion_times - step_size
            next_diffusion_times = [x - step_size for x in diffusion_times]
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                torch.FloatTensor(next_diffusion_times)
            )
            next_noise_rates = next_noise_rates.to(self.hparams['device'])
            next_signal_rates = next_signal_rates.to(self.hparams['device'])

            current_images = (next_signal_rates.view([-1, 1, 1, 1]) * pred_images +
                              next_noise_rates.view([-1, 1, 1, 1]) * pred_noises)

        if return_all_t:
            return step_footprint
        else:
            return pred_images

    def p_sample_loop_ddpm(self, initial_noise, diffusion_steps=None, return_all_t=False):
        pass

    def generate(self, num_images, diffusion_steps, initial_noise=None, return_all_t=False):
        if initial_noise is None:
            initial_noise = torch.randn(num_images, 3, self.hparams['IMAGE_SIZE'], self.hparams['IMAGE_SIZE']).to(
                self.hparams['device'])

        with torch.no_grad():
            self.network.eval()
            self.ema_network.eval()

            generated_images = self.p_sample_loop_ddim(initial_noise, diffusion_steps, return_all_t=return_all_t)
            generated_images = self.denomalize(generated_images)

            self.network.train()
            self.ema_network.train()

        return generated_images


    # test 스탭이 따로 필요한지? train 함수를 완성하자
    def train_steps(self):
        cost = 0.0

        for batch in tqdm.tqdm(self.train_dataloader):
            batch = batch.to(self.hparams['device'])
            # 이미지를 정규화하고 무작위 잡음을 뽑아낸다
            images = self.normalizer(batch)
            noises = torch.randn(batch.shape).to(self.hparams['device'])

            # 이미지 수만큼 신호비와 잡음비를 뽑아낸다
            diffusion_times = torch.rand(len(batch))
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            noise_rates = noise_rates.to(self.hparams['device'])
            signal_rates = signal_rates.to(self.hparams['device'])

            # 정방향 확산 과정은 한 큐에!
            noisy_images = torch.mul(signal_rates.view([-1, 1, 1, 1]), images) + torch.mul(
                noise_rates.view([-1, 1, 1, 1]), noises).to(self.hparams['device'])

            # u-net을 통한 잡음 예측
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
            loss = self.criterion(noises, pred_noises)
            # print(self.loss)

            # 이거 제대로 되긴 하는가? self 달아 줘야 할 지도 모른다?
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            cost += loss

        # ema 신경망에 카피
        with torch.no_grad():
            model_params = OrderedDict(self.network.named_parameters())
            shadow_params = OrderedDict(self.ema_network.named_parameters())

            for name, param in model_params.items():
                shadow_params[name].sub_((1.0 - self.EMA) * (shadow_params[name] - param))

        cost /= (len(self.train_dataloader))  # * BATCH_SIZE)

        # print(f'Epoch: {self.EPOCHS:4d}, Cost: {cost:3f}')
        return cost

    def train(self):
        for epoch in range(self.EPOCHS):
            cost = self.train_steps()
            print(f'Epoch: {epoch + 1:4d}, Loss: {cost:3f}')

            if (epoch + 1) % 5 == 0:
                # 보여주기용 무작위 생성
                generates = self.generate(9, 20).permute(0, 2, 3, 1).to('cpu').detach().numpy()
                show_images(generates, 3, 3)

                torch.save(self.network.state_dict(), f'{self.hparams["model_path"]}/unet-{epoch + 1}.pt')
                torch.save(self.ema_network.state_dict(), f'{self.hparams["model_path"]}/ema-unet-{epoch + 1}.pt')

        torch.save(self.network.state_dict(), f'{self.hparams["model_path"]}unet.pt')
        torch.save(self.ema_network.state_dict(), f'{self.hparams["model_path"]}/ema-unet.pt')

    def load(self):
        self.network.load_state_dict(
            torch.load(f'{self.hparams["model_path"]}/unet.pt', map_location=self.hparams['device']))
        self.ema_network.load_state_dict(
            torch.load(f'{self.hparams["model_path"]}/ema-unet.pt', map_location=self.hparams['device']))
        self.mean = torch.tensor(np.load(f'{self.hparams["model_path"]}/mean.npy')).to(self.hparams['device'])
        self.std = torch.tensor(np.load(f'{self.hparams["model_path"]}/std.npy')).to(self.hparams['device'])

    # 확산 스케줄, 오프셋 코사인 사용
    def diffusion_schedule(self, diffusion_times):  # offset_cosine_diffusion_schedule(diffusion_times):
        min_signal_rate = 0.02
        max_signal_rate = 0.95

        start_angle = torch.acos(torch.Tensor([max_signal_rate]))
        end_angle = torch.acos(torch.Tensor([min_signal_rate]))

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)

        return noise_rates, signal_rates


'''
# 확산 단계
ddpm.load()
gallery = ddpm.generate(8, 10, None, True).permute(0, 2, 3, 1).to('cpu').detach().numpy()
summarized = gallery[0::8] # 뭔가 매핑으로 좀 더 깔끔하게 자르는게 가능할지도?
for i in range(7):
    summarized = np.concatenate((summarized, gallery[i+1::8]), axis=0)
show_images(summarized, 8, 11)
'''

'''
# 구면 선형 보간을 활용한 이미지num, step, initial
gallery = np.zeros((55, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
for i in range(5):
    outputs = None
    inits = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    for idx, ratio in enumerate(np.arange(0.0, 1.1, 0.1)):
        initial_noise = torch.unsqueeze(inits[0] * sin((pi / 2) * ratio) + inits[1] * cos((pi / 2) * ratio), 0)
        gallery[i * 11 + idx] = ddpm.generate(1, 100, initial_noise).permute(0, 2, 3, 1).to('cpu').detach().numpy()

show_images(gallery, 5, 11)
'''

'''
# 샘플 뜨기, detach/numpy/to cpu/permute 등은 처리과정, seed 고정은 역확산 횟수와 성능차이 확인용
torch.manual_seed(42)
sample = ddpm.generate(9, 100).permute(0, 2, 3, 1).to('cpu').detach().numpy()
print(sample.shape)
show_images(sample, 3, 3)
'''

'''
# ema test 코드
# https://www.zijianhu.com/post/pytorch/ema/

with torch.no_grad():
    model_params = OrderedDict(ddpm.network.named_parameters())
    shadow_params = OrderedDict(ddpm.ema_network.named_parameters())

    for param in shadow_params.values():
        print(param[0][0][0][0])
        break
    for name, param in model_params.items():
        shadow_params[name].sub_((1.0 - 0.999) * (shadow_params[name] - param))
    for param in shadow_params.values():
        print(param[0][0][0][0])
        break
'''

'''
# with torch.no_grad()는 딱히 영구 전역 적용은 아닌듯
x = torch.tensor([1.], requires_grad=True)
with torch.no_grad():
    y = x * 2
    print(y.requires_grad)
print(x.requires_grad)
'''

'''
# 차원이 다른 텐서의 곱 테스트
x = torch.FloatTensor([1.0, 2.0, 3.0, 4.0, 5.0])
y = torch.ones([5, 3, 64, 64], dtype=torch.float32)
z_div = torch.div(y, x.view(-1, 1, 1, 1))
z_mul = torch.mul(x.view(-1, 1, 1, 1), y)
z_mul_reverse = torch.mul(y, x.view(-1, 1, 1, 1))
print(z_mul_reverse)
'''

# 생성기 테스트 코드
# train_dataloader, mean_numpy, std_numpy = getDataLoader("./datasets")
# ddpm = DDPM(mean=mean_numpy, std=std_numpy)
# print(ddpm.generate(5, 20).shape)

'''
# 역방향 확산 테스트 코드
# ddpm.reverse_diffusion(torch.FloatTensor([[[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]]]), 20)
# 무작위 잡음의 생성은, [N, 3채널, 1, 1]를 생성하도록 함
reverse_test_random_tensor = torch.randn([5, 3, 64, 64])
# print(reverse_test_random_tensor.shape)
# print(reverse_test_random_tensor)
ddpm.reverse_diffusion(reverse_test_random_tensor, 20)
'''

# ddpm.denoise_test()


'''
# normalize_test와 denormalize코드 테스트용
images = ddpm.denomalize(ddpm.normalize_test()).detach().numpy()

plt.imshow(images[0].swapaxes(0, 1).swapaxes(1, 2))
plt.show()
'''

'''
# 확산 스케줄 뽑아내기
# diffusion_times = torch.FloatTensor([x / 20 for x in range(20)])
diffusion_times = 20
signal_rates, noise_rates = offset_cosine_diffusion_schedule(diffusion_times)
print(signal_rates.shape, noise_rates.shape)

'''
