import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import sqrt

img_path = "./outputs/openai-2024-06-03-22-22-17-509729/samples_64x32x32x3.npz"
img_path = "/tmp/openai-2024-07-16-10-51-03-359415/samples_16x256x256x3.npz"

imgs = np.load(img_path, mmap_mode='r')

# npz 인덱스 보기
# for k in imgs.files:
#     print(k)

imgs = imgs['arr_0']
print(imgs.shape[0])
print(imgs.shape)

# 아마도, 2^(even num)으로 나오면 상관없음
wh = int(np.sqrt(imgs.shape[0]))

plt.figure(figsize=(wh, wh))
for idx, image in enumerate(imgs):
    plt.subplot(wh, wh, idx + 1)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)#cv2.COLOR_BGR2RGB)
    plt.imshow(image)
plt.show()
