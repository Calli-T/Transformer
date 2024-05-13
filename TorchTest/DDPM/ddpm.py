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
    def __init__(self):
        super().__init__()

        self.normalizer = nn.LayerNorm([3, IMAGE_SIZE, IMAGE_SIZE])  # 레이어 정규화, 샘플 각각에 대해서 수행함
        self.network = UNet()
        self.ema_network = UNet()
        self.diffusion_schedule = offset_cosine_diffusion_schedule

    def denomalize(self, images):
        pass

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        pass

    def normalize_test(self):
        train_dataloader, mean, std = getDataLoader("./datasets")
        for batch in train_dataloader:
            print(self.normalizer(batch).shape)

    # numpy로 만든 mean std로 denormalizer 만들기
    # 그리고 pytorch의 손실함수 선택, 최적화 알고리즘적용, 오차 역전파


ddpm = DDPM()
ddpm.normalize_test()
