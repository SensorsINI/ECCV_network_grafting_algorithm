"""Data pipline for training.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import glob

import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from albumentations.augmentations.functional import get_random_crop_coords
from albumentations.augmentations.functional import get_center_crop_coords

from evtrans.yolo.utils.datasets import pad_to_square
from evtrans.yolo.utils.datasets import resize


class NMNISTDataset(Dataset):
    def __init__(self, folder_path, labels=None, is_train=True,
                 parsing="/*.*", no_resampling=True, event_only=True):
        self.files = sorted(glob.glob("{}".format(folder_path)+parsing))
        self.labels = labels
        self.is_train = is_train
        self.event_only = event_only

    def __getitem__(self, index):
        data_path = self.files[index]

        data = np.load(data_path)
        ev_vol = torch.tensor(data["ev_img"], dtype=torch.float)
        img = transforms.ToTensor()(data["img"])

        if self.event_only is True:
            return ev_vol, self.labels[index]
        else:
            return ev_vol, img

    def __len__(self):
        return len(self.files)


class ThermalYOLODataset(Dataset):
    def __init__(self, folder_path, is_train, img_size=416,
                 parsing="/*/*.*", shutdown_aps=False, no_resampling=True):
        self.files = sorted(glob.glob("{}".format(folder_path)+parsing))
        self.img_size = img_size
        self.is_train = is_train
        self.shutdown_aps = shutdown_aps
        self.no_resampling = no_resampling

    def __getitem__(self, index):
        data_path = self.files[index % len(self.files)]

        data = np.load(data_path)
        ev_vol = transforms.ToTensor()(data["ev_img"])
        img = transforms.ToTensor()(data["img"])

        if self.no_resampling is True:
            height, width = img.shape[1], img.shape[2]
            if self.is_train:
                h_start, w_start = random.random(), random.random()
                x1, y1, x2, y2 = get_random_crop_coords(
                    height, width, 224, 224, h_start, w_start)
            else:
                x1, y1, x2, y2 = get_center_crop_coords(
                    height, width, 224, 224)

            img = img[..., y1:y2, x1:x2]
            ev_vol = ev_vol[..., y1:y2, x1:x2]
        else:
            ev_vol, _ = pad_to_square(ev_vol, 0)
            img, _ = pad_to_square(img, 0)

            ev_vol = resize(ev_vol, self.img_size)
            img = resize(img, self.img_size)

        return ev_vol, img

    def __len__(self):
        return len(self.files)


class EVYOLODataset(Dataset):
    def __init__(self, folder_path, is_train, img_size=416,
                 parsing="/*/*.*", shutdown_aps=False, is_recurrent=True,
                 no_resampling=False, sample_limit=0):
        self.files = sorted(glob.glob("{}".format(folder_path)+parsing))
        self.img_size = img_size
        self.is_train = is_train
        self.shutdown_aps = shutdown_aps
        # indicate if the dataset is prepared for conv version
        self.is_recurrent = is_recurrent

        if sample_limit != 0:
            random.shuffle(self.files)
            self.files = self.files[:sample_limit]

        self.no_resampling = no_resampling

    def __getitem__(self, index):
        data_path = self.files[index % len(self.files)]

        data = np.load(data_path)
        ev_vol = torch.tensor(data["ev_img"], dtype=torch.float)
        img = transforms.ToTensor()(data["img"])

        # turn to a color image if a gray image
        if img.shape[0] == 1:
            img = torch.cat([img, img, img], dim=0)

        if self.no_resampling is True:
            height, width = img.shape[1], img.shape[2]
            if self.is_train:
                h_start, w_start = random.random(), random.random()
                x1, y1, x2, y2 = get_random_crop_coords(
                    height, width, 224, 224, h_start, w_start)
            else:
                x1, y1, x2, y2 = get_center_crop_coords(
                    height, width, 224, 224)

            img = img[..., y1:y2, x1:x2]
            ev_vol = ev_vol[..., y1:y2, x1:x2]
        else:
            ev_vol, _ = pad_to_square(ev_vol, 0)
            img, _ = pad_to_square(img, 0)

            ev_vol = resize(ev_vol, self.img_size)
            img = resize(img, self.img_size)

        # add a dimension
        if self.is_recurrent is True:
            ev_vol = ev_vol.unsqueeze(1)

        return ev_vol, img

    def __len__(self):
        return len(self.files)
