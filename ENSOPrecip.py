# ENSOPrecip dataset

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import netCDF4 as nc

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ENSOPrecipDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if type(index) == torch.Tensor:
            index = index.item()

        enso_index = np.loadtxt(os.path.join(self.root, "train", "enso_index.txt"), delimiter=',')
        enso_precip = np.load(os.path.join(self.root, "train", "precip.npy"))
        mask = enso_precip < 0

        sample = {"index": enso_index, "precip": enso_precip, "mask": mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    def __call__(self, sample):
        index = sample["index"]
        precip = sample["precip"]
        mask = sample["mask"]

        index_min = 24.58
        index_max = 29.42
        index = (index - (index_min + index_max) / 2.) / ((index_max - index_min) / 2.)

        precip_min = 0.0
        precip_max = 2222.5168
        precip = (precip - (precip_min + precip_max) / 2.) / ((precip_max - precip_min) / 2.)

        return {"index": index, "precip": precip, "mask": mask}

class ToTensor(object):
    def __call__(self, sample):
        index = sample["index"]
        precip = sample["precip"]
        mask = sample["mask"]

        # dimension raising
        # numpy shape: [N, ]
        # torch shape: [N, 1]
        index = index[None, :]
        precip = precip[None, :]
        mask = mask[None, :]
        return {"index": torch.from_numpy(index),
                "precip": torch.from_numpy(precip),
                "mask": torch.from_numpy(mask)}
