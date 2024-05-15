import torch

from unet import *

# hyper
EMA = 0.999


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


class DDPM(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()

        self.normalizer = nn.LayerNorm([3, IMAGE_SIZE, IMAGE_SIZE])  # 레이어 정규화, 샘플 각각에 대해서 수행함
        self.network = UNet()
        self.ema_network = UNet()
        self.diffusion_schedule = offset_cosine_diffusion_schedule

        self.mean = mean
        self.std = std

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

        return pred_noises, pred_images

    # 역방향 확산의 반복부터 다시 시작하면된다
    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        for step in range(diffusion_steps):
            diffusion_times = [1 - step * step_size for _ in range(num_images)]
            noise_rates, signal_rates = self.diffusion_schedule(torch.FloatTensor(diffusion_times))

            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )
            # print(pred_noises.shape)
            # print(pred_images.shape)

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
            plt.imshow(image)
            plt.show()

            return self.normalizer(batch)
    '''

    # 정방향 확산 프로세스(이건 함수가 따로 필요없고 훈련 때 바로 생성하는것도 가능함)
    # 역방향 확산 프로세스(denoise함수 생성 완료, 그걸이용하는 reverse diffusion 함수 만들것)
    # 생성 함수
    # 그리고 pytorch의 손실함수 선택, 최적화 알고리즘적용, 오차 역전파등의 학습과정


ddpm = DDPM()

# ddpm.reverse_diffusion(torch.FloatTensor([[[[1]]], [[[1]]], [[[1]]], [[[1]]], [[[1]]]]), 20)
# 무작위 잡음의 생성은, [N, 3채널, 1, 1]를 생성하도록 함
reverse_test_random_tensor = torch.rand([5, 3, 64, 64])
# print(reverse_test_random_tensor.shape)
# print(reverse_test_random_tensor)
ddpm.reverse_diffusion(reverse_test_random_tensor, 20)

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
