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
        self.enso_continent_mask = np.isnan(self.enso_input_feat[-1, 0])
        self.enso_precip = np.load(os.path.join(self.root, "train", "PRISM_ppt.npy"))
        self.num_variables = self.enso_input_feat.shape[0]
        self.timesteps = self.enso_input_feat.shape[1]
        self.recep_field = 22

    def __len__(self):
        return self.timesteps - self.recep_field + 1

    def __getitem__(self, index):
        if type(index) == torch.Tensor:
            index = index.item()

        input_feat = self.enso_input_feat[:, index : index + self.recep_field]
        precip = self.enso_precip[index + self.recep_field - 1 : index + self.recep_field]
        input_mask = self.enso_continent_mask[np.newaxis, np.newaxis, :]\
            .repeat(self.recep_field, axis=1).repeat(self.num_variables, axis=0)

        sample = {"index": index + self.recep_field - 1, "input_feat": input_feat,
                  "input_mask": input_mask, "precip": precip}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    def __call__(self, sample):
        index = sample["index"]
        input_feat = sample["input_feat"]
        input_mask = sample["input_mask"]
        precip = sample["precip"]

        input_feat_min = np.array([97790.586, 229.67299, 4825.4565, 9235.415, -18.268473, -23.363884, 268.30017]).reshape((-1, 1, 1, 1))
        input_feat_max = np.array([103410.69, 309.953,   5973.6914, 11126.077, 87.62169,  89.21173,   305.68158]).reshape((-1, 1, 1, 1))
        input_feat = ((input_feat - (input_feat_min + input_feat_max) / 2.)
                      / ((input_feat_max - input_feat_min) / 2.)).astype(np.float32)
        input_feat[input_mask] = 0.

        precip_min = 0.
        precip_max = 1759.4718
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
