import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import sqrt

img_path = "./outputs/openai-2024-07-16-11-39-00-185773/samples_16x256x256x3.npz"

imgs = np.load(img_path, mmap_mode='r')

imgs = imgs['arr_0']
# print(imgs.shape[0])

wh = int(np.sqrt(imgs.shape[0]))

plt.figure(figsize=(wh, wh))
for idx, image in enumerate(imgs):
    plt.subplot(wh, wh, idx + 1)
    plt.imshow(image)
plt.show()
