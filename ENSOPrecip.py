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

        enso_sst = np.load(os.path.join(self.root, "train", "AnomSST_1979-2020.npy"))
        enso_precip = np.load(os.path.join(self.root, "train", "AnomPrecip_1979-2020.npy"))

        sample = {"sst": enso_sst, "precip": enso_precip}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    def __call__(self, sample):
        sst = sample["sst"]
        precip = sample["precip"]

        sst_min = -14.20
        sst_max = 18.42
        sst = ((sst - (sst_min + sst_max) / 2.) / ((sst_max - sst_min) / 2.)).astype(np.float32)

        precip_min = -21.87
        precip_max = 88.38
        precip = ((precip - (precip_min + precip_max) / 2.) / ((precip_max - precip_min) / 2.)).astype(np.float32)

        return {"sst": sst, "precip": precip}

class ToTensor(object):
    def __call__(self, sample):
        sst = sample["sst"]
        precip = sample["precip"]

        # dimension raising
        # numpy shape: [N, ]
        # torch shape: [N, 1]
        sst = sst[None, :]
        precip = precip[None, :]
        return {"sst": torch.from_numpy(sst),
                "precip": torch.from_numpy(precip)}
