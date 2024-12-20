import numpy as np
import torch

from unet import *

import tqdm
from torch import optim
from collections import OrderedDict
from math import sin, cos, pi


# 확산 스케줄, 오프셋 코사인 사용
def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = 0.02
    max_signal_rate = 0.95

    start_angle = torch.acos(torch.Tensor([max_signal_rate]))
    end_angle = torch.acos(torch.Tensor([min_signal_rate]))

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates


def show_images(images, h, w):
    plt.figure(figsize=(w, h))
    for idx, image in enumerate(images):
        plt.subplot(h, w, idx + 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    plt.show()


class DDIM:
    def __init__(self):

        self.normalizer = nn.LayerNorm([3, IMAGE_SIZE, IMAGE_SIZE]).to(device)  # 레이어 정규화, 샘플 각각에 대해서 수행함
        self.network = UNet().to(device)
        self.ema_network = UNet().to(device)
        self.diffusion_schedule = offset_cosine_diffusion_schedule

        self.loss = 0.0

        # hyper
        self.EMA = 0.999
        self.reverse_diffusion_steps = 20
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-4
        self.EPOCHS = 5000

        # train_set 평균 절대 오차/RMSprop사용 -> L1 loss/RMSprop사용
        self.criterion = nn.L1Loss().to(device)  # nn.MSELoss().to(device)  # nn.L1Loss().to(device)
        # self.criterion2 = nn.MSELoss().to(device)
        self.ratio = 0.5
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self.LEARNING_RATE, weight_decay=self.WEIGHT_DECAY)
        # self.loss = 0.0

        self.train_dataloader = None
        self.mean = None
        self.std = None

        # if torch.is_tensor(mean):
        #     self.mean = mean
        # else:
        #     self.mean = torch.tensor(mean)
        # if torch.is_tensor(std):
        #     self.std = std
        # else:
        #     self.std = torch.tensor(std)

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
        '''
        unsqueeze 3번보다는 view가 나은듯? reshape도 써보자
        print(noisy_images.shape)
        print(noise_rates.view([-1, 1, 1, 1]).shape)
        print(pred_noises.shape)
        print(signal_rates.view([-1, 1, 1, 1]).shape)
        '''
        # pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        # print(torch.mul(noise_rates.view([-1, 1, 1, 1]), pred_noises).shape)
        # print("muyaho")
        # print((noisy_images - torch.mul(noise_rates.view([-1, 1, 1, 1]), pred_noises)).shape)
        # print("muyaho")

        # print(torch.div((noisy_images - torch.mul(noise_rates.view([-1, 1, 1, 1]), pred_noises)), signal_rates.view([-1, 1, 1, 1])).shape)
        # pred_images = torch.div((noisy_images - torch.mul(noise_rates, pred_noises)), signal_rates)

        # 차원을 맞추기 위한 눈물겨운 %꼬쑈
        pred_images = torch.div((noisy_images - torch.mul(noise_rates.view([-1, 1, 1, 1]), pred_noises)),
                                signal_rates.view([-1, 1, 1, 1]))

        # print('--------------------------')
        # print(noise_rates.device)
        # print(noisy_images.device)

        return pred_noises, pred_images

    # 역방향 확산의 반복부터 다시 시작하면된다
    def reverse_diffusion(self, initial_noise, diffusion_steps=None, return_all_t=False):
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
            noise_rates = noise_rates.to(device)
            signal_rates = signal_rates.to(device)

            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=True  # False
            )
            # print(pred_noises.shape)
            # print(pred_images.shape)
            if return_all_t:
                step_footprint = torch.concat([step_footprint, pred_images.detach()], 0)

            # next_diffusion_times = diffusion_times - step_size
            next_diffusion_times = [x - step_size for x in diffusion_times]
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                torch.FloatTensor(next_diffusion_times)
            )
            next_noise_rates = next_noise_rates.to(device)
            next_signal_rates = next_signal_rates.to(device)

            '''
            print(next_signal_rates.shape)
            print(pred_images.shape)
            print(next_noise_rates.shape)
            print(pred_noises.shape)
            '''
            # current_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
            current_images = (next_signal_rates.view([-1, 1, 1, 1]) * pred_images +
                              next_noise_rates.view([-1, 1, 1, 1]) * pred_noises)

        if return_all_t:
            return step_footprint
        else:
            return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None, return_all_t=False):
        if initial_noise is None:
            initial_noise = torch.randn(num_images, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

        with torch.no_grad():
            self.network.eval()
            self.ema_network.eval()

            generated_images = self.reverse_diffusion(initial_noise, diffusion_steps, return_all_t=return_all_t)
            # print(generated_images.shape)
            generated_images = self.denomalize(generated_images)

            self.network.train()
            self.ema_network.train()
        # print(generated_images.shape)

        return generated_images

    # 데이터로더, 평균, 표준 편차 모두 여기서 세팅
    def set_datasets_from_path(self, path):
        train_dataloader, mean, std = getDataLoader(path)
        self.mean = torch.FloatTensor(mean).to(device)
        self.std = torch.FloatTensor(std).to(device)
        self.train_dataloader = train_dataloader

        np.save('models/flowers/mean.npy', self.mean.detach().cpu().numpy())
        np.save('models/flowers/std.npy', self.std.detach().cpu().numpy())

    # test 스탭이 따로 필요한지? train 함수를 완성하자
    def train_steps(self):
        cost = 0.0

        for batch in tqdm.tqdm(self.train_dataloader):
            batch = batch.to(device)
            # 이미지를 정규화하고 무작위 잡음을 뽑아낸다
            images = self.normalizer(batch)
            noises = torch.randn(batch.shape).to(device)
            # print(images.shape)
            # print(noises.shape)

            # 이미지 수만큼 신호비와 잡음비를 뽑아낸다
            diffusion_times = torch.rand(len(batch))
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            # print(noise_rates.shape)
            # print(signal_rates.shape)
            noise_rates = noise_rates.to(device)
            signal_rates = signal_rates.to(device)

            # 정방향 확산 과정은 한 큐에!
            noisy_images = torch.mul(signal_rates.view([-1, 1, 1, 1]), images) + torch.mul(
                noise_rates.view([-1, 1, 1, 1]), noises).to(device)
            # print(noisy_images.shape)

            # u-net을 통한 잡음 예측
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
            loss = self.criterion(noises, pred_noises)
            # print(self.loss)

            # 이거 제대로 되긴 하는가? self 달아 줘야 할 지도 모른다?
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(loss)
            cost += loss

        # ema 신경망에 카피
        with torch.no_grad():
            model_params = OrderedDict(ddim.network.named_parameters())
            shadow_params = OrderedDict(ddim.ema_network.named_parameters())

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

                torch.save(self.network.state_dict(), f'./models/unet-{epoch + 1}.pt')
                torch.save(self.ema_network.state_dict(), f'./models/ema-unet-{epoch + 1}.pt')

        torch.save(self.network.state_dict(), f'./models/unet.pt')
        torch.save(self.ema_network.state_dict(), f'./models/ema-unet.pt')

    def load(self, dir=None):
        if dir is None:
            self.network.load_state_dict(torch.load(f'./models/flowers/unet.pt', map_location=device))
            self.ema_network.load_state_dict(torch.load(f'./models/flowers/ema-unet.pt', map_location=device))
            self.mean = torch.tensor(np.load(f'./models/flowers/mean.npy')).to(device)
            self.std = torch.tensor(np.load(f'./models/flowers/std.npy')).to(device)
        else:
            self.network.load_state_dict(torch.load(f'{dir}/unet.pt', map_location=device))
            self.ema_network.load_state_dict(torch.load(f'{dir}/ema-unet.pt', map_location=device))
            self.mean = torch.tensor(np.load(f'{dir}/mean.npy')).to(device)
            self.std = torch.tensor(np.load(f'{dir}/std.npy')).to(device)

    '''
        def set_mean_and_std(self, mean, std):
            if torch.is_tensor(mean):
                self.mean = mean
            else:
                self.mean = torch.tensor(mean)
            if torch.is_tensor(std):
                self.std = std
            else:
                self.std = torch.tensor(std)
        '''

    '''
    def denoise_test(self):
        train_dataloader, mean, std = getDataLoader("./datasets")
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

        # 확산 스케줄 뽑아내기
        diffusion_times = torch.FloatTensor([0.5, 0.5, 0.5, 0.5, 0.5])
        signal_rates, noise_rates = self.diffusion_schedule(diffusion_times)
        # print(signal_rates)

        for batch in train_dataloader:
            _, __ = self.denoise(batch, noise_rates, signal_rates, training=True)
            print(_.shape)
            print(__.shape)
    '''

    # print(element.shape)
    '''
    def normalize_test(self):
        train_dataloader, mean, std = getDataLoader("./datasets")

        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)
        for batch in train_dataloader:
            image = batch[0].permute(1, 2, 0).detach().numpy()
            # plt.imshow(image)
            # plt.show()

            return self.normalizer(batch)
    '''

    # 정방향 확산 프로세스(이건 함수가 따로 필요없고 훈련 때 바로 생성하는것도 가능함)
    #
    # 그리고 pytorch의 손실함수 선택, 최적화 알고리즘적용, 오차 역전파등의 학습과정


ddim = DDIM()
ddim.set_datasets_from_path("./datasets/flower_t_[0, 1]")

# ddim.train()

'''# 확산 단계
ddim.load()
gallery = ddim.generate(8, 10, None, True).permute(0, 2, 3, 1).to('cpu').detach().numpy()
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
        gallery[i * 11 + idx] = ddim.generate(1, 100, initial_noise).permute(0, 2, 3, 1).to('cpu').detach().numpy()

show_images(gallery, 5, 11)
'''

ddim.load('models/flower_t_[0, 1]')
# 샘플 뜨기, detach/numpy/to cpu/permute 등은 처리과정, seed 고정은 역확산 횟수와 성능차이 확인용
torch.manual_seed(42)
sample = ddim.generate(9, 100).permute(0, 2, 3, 1).to('cpu').detach().numpy()
print(sample.shape)
show_images(sample, 3, 3)

'''
# ema test 코드
# https://www.zijianhu.com/post/pytorch/ema/

with torch.no_grad():
    model_params = OrderedDict(ddim.network.named_parameters())
    shadow_params = OrderedDict(ddim.ema_network.named_parameters())

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
# ddim = DDIM(mean=mean_numpy, std=std_numpy)
# print(ddim.generate(5, 20).shape)

'''
# 역방향 확산 테스트 코드
# ddim.reverse_diffusion(torch.FloatTensor([[[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]]]), 20)
# 무작위 잡음의 생성은, [N, 3채널, 1, 1]를 생성하도록 함
reverse_test_random_tensor = torch.randn([5, 3, 64, 64])
# print(reverse_test_random_tensor.shape)
# print(reverse_test_random_tensor)
ddim.reverse_diffusion(reverse_test_random_tensor, 20)
'''

# ddim.denoise_test()


'''
# normalize_test와 denormalize코드 테스트용
images = ddim.denomalize(ddim.normalize_test()).detach().numpy()

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
