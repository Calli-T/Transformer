import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader

IMAGE_SIZE = 64
BATCH_SIZE = 64
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
    _imgs = np.float32(_imgs) / 255.0
    origin = np.copy(_imgs)
    for _ in range(_DATASET_REPETITIONS - 1):
        origin = np.concatenate((origin, _imgs), axis=0)

    return origin


def getDataLoader(img_path):
    imgs = preprocess(getImgsFromDir(img_path), DATASET_REPETITIONS)
    # imgs = preprocess(getImgsFromDir('./datasets'), DATASET_REPETITIONS)
    # imgs = preprocess(getImgsFromDir('./fakesets'), DATASET_REPETITIONS)
    # print(imgs.shape)
    train_dataset = torch.FloatTensor(imgs)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(train_dataset.shape)
    # print(len(train_dataloader))
    # print(len(imgs))
    # print(imgs[:1])
    # print(imgs.shape)
    # img = cv2.resize(cv2.imread("./image_06734.jpg"), dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    return train_dataloader
