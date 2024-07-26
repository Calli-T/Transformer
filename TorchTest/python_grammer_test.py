'''
a = [1, 2, 3, 4]
print(a[-1])
'''

'''
a = [1, 2, 3, 4]
print(list(enumerate(a))[::-1])
'''

'''a = [1, 2, 3, 4]
level = 0
for i in range(len(a) + 1):
    print(i)
    if level and i == 4:
        print("muyaho")
'''

'''import torch as th
print(th.arange(start=0, end=16, dtype=th.float32) / 16)'''

# import torch
#
#
# def offset_cosine_diffusion_schedule(diffusion_times):
#     min_signal_rate = 0.02
#     max_signal_rate = 0.95
#
#     start_angle = torch.acos(torch.Tensor([max_signal_rate]))
#     end_angle = torch.acos(torch.Tensor([min_signal_rate]))
#
#     diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
#
#     signal_rates = torch.cos(diffusion_angles)
#     noise_rates = torch.sin(diffusion_angles)
#
#     return noise_rates, signal_rates
#
#
# def linear_diff_sche(diffusion_times):
#     min_rate = 0.0001
#     max_rate = 0.02
#     betas = min_rate + diffusion_times * (max_rate - min_rate)
#     alphas = 1 - betas
#
#     alpha_bars = torch.cumprod(alphas, dim=0)
#     print(alpha_bars)
#     return alpha_bars, 1 - alpha_bars
#
#
# T = 1000
# diffusion_times = torch.FloatTensor([x / T for x in range(T)])
# # print(offset_cosine_diffusion_schedule(diffusion_times))
# sig_rates, noi_rates = linear_diff_sche(diffusion_times)
# print(sig_rates, noi_rates)

'''
import numpy as np
beta = np.array([0.2, 0.4, 0.6, 0.8])
alpha = 1-beta
alpha = np.cumprod(alpha)

alpha = alpha[:-1]
print(alpha)
alpha = np.append(1.0, alpha)
print(alpha)
alpha = alpha[1:]
print(np.append(alpha, 0.0))
# print(np.append(1.0, alpha[:-1]))

'''

'''
import numpy as np

a = np.ones([20])
p = a / np.sum(a)
print(p)
indices_np = np.random.choice(len(p), size=(64,), p=p)
print(indices_np)
bunmo = len(p) * p[indices_np]
print(bunmo)
'''

a = [[1, 2], [3, 4]]
import torch
a = torch.tensor(a)
print(a[[0, 1], [0, 1]])