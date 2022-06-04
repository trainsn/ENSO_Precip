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

        enso_input_feat = np.load(os.path.join(self.root, "train", "input_feat.npy"))
        enso_precip = np.load(os.path.join(self.root, "train", "PRISM_ppt.npy"))

        sample = {"input_feat": enso_input_feat, "precip": enso_precip}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    def __call__(self, sample):
        input_feat = sample["input_feat"]
        precip = sample["precip"]

        input_feat_min = np.array([95827.602, 199.63177, 4644.373, 4644.373, -21.138626, -21.138626]).reshape((-1, 1, 1, 1))
        input_feat_max = np.array([104431.23, 314.57242, 5970.5391, 5970.5391, 87.681274, 87.681274]).reshape((-1, 1, 1, 1))
        input_feat = ((input_feat - (input_feat_min + input_feat_max) / 2.)
                      / ((input_feat_max - input_feat_min) / 2.)).astype(np.float32)

        precip_min = 0.
        precip_max = 1088.1039
        precip = ((precip - (precip_min + precip_max) / 2.) / ((precip_max - precip_min) / 2.)).astype(np.float32)

        return {"input_feat": input_feat, "precip": precip}

class ToTensor(object):
    def __call__(self, sample):
        input_feat = sample["input_feat"]
        precip = sample["precip"]

        # dimension raising
        # numpy shape: [N, ]
        # torch shape: [N, 1]
        precip = precip[None, :]
        return {"input_feat": torch.from_numpy(input_feat),
                "precip": torch.from_numpy(precip)}
