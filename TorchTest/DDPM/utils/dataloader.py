import numpy as np
import torch
import cv2
import os
from torch.utils.data import DataLoader


def getImgsFromDir(img_path, img_size):
    imglist = []

    file_list = os.listdir(img_path)

    for file_name in file_list:
        f_path_name = os.path.join(img_path, file_name)

        if os.path.isdir(f_path_name):
            imglist += getImgsFromDir(f_path_name, img_size)
        else:
            try:
                img = cv2.resize(cv2.imread(f_path_name), dsize=(img_size, img_size),
                                 interpolation=cv2.INTER_LINEAR)
                imglist.append(img)
            except:
                print

    return imglist


def preprocess(_imgs, repeat):
    _imgs = np.float32(_imgs).swapaxes(3, 1).swapaxes(2, 3) / 255.0

    _mean = np.mean(_imgs, axis=0)
    _std = np.std(_imgs, axis=0)

    origin = np.copy(_imgs)
    for _ in range(repeat - 1):
        origin = np.concatenate((origin, _imgs), axis=0)

    return origin, _mean, _std


def getDataLoader(hparams):
    imgs, _mean, _std = preprocess(getImgsFromDir(hparams["data_path"], hparams["IMAGE_SIZE"]),
                                   hparams['DATASET_REPETITION'])
    # imgs, _mean, _std = preprocess(getImgsFromDir('./fakesets'), DATASET_REPETITIONS)

    train_dataset = torch.FloatTensor(imgs)  # .permute(0, 3, 1, 2)
    train_dataloader = DataLoader(train_dataset, batch_size=hparams['BATCH_SIZE'], shuffle=True, drop_last=False)

    hparams['mean'] = _mean
    hparams['std'] = _std

    return train_dataloader
