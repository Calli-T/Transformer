from device_converter import device

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
import tqdm
from torch.utils.data import DataLoader

# hyper
IMAGE_SIZE = 64
BATCH_SIZE = 64 * 4
DATASET_REPETITIONS = 5


def getImgsFromDir(path):
    imglist = []

    file_list = os.listdir(path)

    for file_name in file_list:
        f_path_name = os.path.join(path, file_name)

        if os.path.isdir(f_path_name):
            imglist += getImgsFromDir(f_path_name)
        else:
            img = cv2.resize(cv2.imread(f_path_name), dsize=(IMAGE_SIZE, IMAGE_SIZE),
                             interpolation=cv2.INTER_LINEAR)
            imglist.append(img)

    return imglist


def preprocess(_imgs, _DATASET_REPETITIONS=1):
    # torch 텐서 순서는 (N,C,H,W), 읽어온건 (N,H,W,C) 따라서 swapaxes함수로 축 변경
    # 축 변경은 torch의 .permute로도 가능, 그게 더 편함
    # 시드 고정하고 확인 해보니 결과는 똑같음
    # float 32로 바꾸고 [0, 1]로 스케일링
    _imgs = np.float32(_imgs).swapaxes(3, 1).swapaxes(2, 3) / 255.0
    # _imgs = np.float32(_imgs) / 255.0

    # 그냥 따로 함수 만들지말고 mean std 반환까지 n회 반복 이전에 그대로!
    _mean = np.mean(_imgs, axis=0)
    _std = np.std(_imgs, axis=0)

    origin = np.copy(_imgs)
    for _ in range(_DATASET_REPETITIONS - 1):
        origin = np.concatenate((origin, _imgs), axis=0)

    return origin, _mean, _std


# 고정 시드
torch.manual_seed(42)


def getDataLoader(img_path):
    imgs, _mean, _std = preprocess(getImgsFromDir(img_path), DATASET_REPETITIONS)
    # imgs, _mean, _std = preprocess(getImgsFromDir('./fakesets'), DATASET_REPETITIONS)
    # imgs = preprocess(getImgsFromDir('./datasets'), DATASET_REPETITIONS)
    # print(imgs.shape)
    train_dataset = torch.FloatTensor(imgs)  # .permute(0, 3, 1, 2)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    # print(train_dataset.shape)
    # print(len(train_dataloader))
    # print(len(imgs))
    # print(imgs[:1])
    # print(imgs.shape)
    # img = cv2.resize(cv2.imread("./image_06734.jpg"), dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    return train_dataloader, _mean, _std


'''
_, mean, std = getDataLoader('./datasets')
print(mean.shape)
print(std.shape)
print(std)
'''

# def get_mean_std():


# - 1 -
