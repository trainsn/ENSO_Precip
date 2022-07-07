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

        self.enso_input_feat = np.load(os.path.join(self.root, "train", "input_feat.npy"))
        self.enso_precip = np.load(os.path.join(self.root, "train", "PRISM_ppt.npy"))
        self.timesteps = self.enso_input_feat.shape[1]
        self.recep_field = 22

    def __len__(self):
        return self.timesteps - self.recep_field + 1

    def __getitem__(self, index):
        if type(index) == torch.Tensor:
            index = index.item()

        input_feat = self.enso_input_feat[:, index : index + self.recep_field]
        precip = self.enso_precip[index + self.recep_field - 1 : index + self.recep_field]

        sample = {"index": index + self.recep_field - 1, "input_feat": input_feat, "precip": precip}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    def __call__(self, sample):
        index = sample["index"]
        input_feat = sample["input_feat"]
        precip = sample["precip"]

        input_feat_min = np.array([95827.602, 199.63177, 4644.373, 4644.373, -21.138626, -21.138626, 267.872]).reshape((-1, 1, 1, 1))
        input_feat_max = np.array([104431.23, 314.57242, 5970.5391, 5970.5391, 87.681274, 87.681274, 309.222]).reshape((-1, 1, 1, 1))
        input_feat = ((input_feat - (input_feat_min + input_feat_max) / 2.)
                      / ((input_feat_max - input_feat_min) / 2.)).astype(np.float32)

        precip_min = 0.
        precip_max = 1908.8616
        precip = ((precip - (precip_min + precip_max) / 2.) / ((precip_max - precip_min) / 2.)).astype(np.float32)

        return {"index": index, "input_feat": input_feat, "precip": precip}

class ToTensor(object):
    def __call__(self, sample):
        index = sample["index"]
        input_feat = sample["input_feat"]
        precip = sample["precip"]

        # dimension raising
        # numpy shape: [N, ]
        # torch shape: [N, 1]
        precip = precip[None, :]
        return {"index": index,
                "input_feat": torch.from_numpy(input_feat),
                "precip": torch.from_numpy(precip)}
