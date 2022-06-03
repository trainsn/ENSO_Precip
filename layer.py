import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class ConvSkew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvSkew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.UT = nn.Conv3d(in_features, out_features, 1, 1, 0, bias=True)
        self.UNN = nn.Conv3d(in_features, out_features, 1, 1, 0, bias=True)
        self.UNZ = nn.Conv3d(in_features, out_features, 1, 1, 0, bias=True)
        self.UNP = nn.Conv3d(in_features, out_features, 1, 1, 0, bias=True)
        self.UZN = nn.Conv3d(in_features, out_features, 1, 1, 0, bias=True)
        self.UZZ = nn.Conv3d(in_features, out_features, 1, 1, 0, bias=True)
        self.UZP = nn.Conv3d(in_features, out_features, 1, 1, 0, bias=True)
        self.UPN = nn.Conv3d(in_features, out_features, 1, 1, 0, bias=True)
        self.UPZ = nn.Conv3d(in_features, out_features, 1, 1, 0, bias=True)
        self.UPP = nn.Conv3d(in_features, out_features, 1, 1, 0, bias=True)

        self.kernels = [self.UNN, self.UNZ, self.UNP, self.UZN, self.UZZ, self.UZP, self.UPN, self.UPZ, self.UPP]
        self.dirLat = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        self.dirLon = [0, 1, 2, 0, 1, 2, 0, 1, 2]

    def forward(self, input):
        dimT, dimLat, dimLon = input.shape[2], input.shape[3], input.shape[4]
        tmp = torch.zeros((input.shape[0], input.shape[1], dimT + 1, dimLat + 2, dimLon + 2), dtype=input.dtype).cuda()
        tmp[:, :, 1:, 1:dimLat+1, 1:dimLon+1] = input
        out = self.UT(tmp[:, :, :dimT, 1:dimLat+1, 1:dimLon+1])
        for i in range(9):
            out += self.kernels[i](tmp[:, :, 1:, self.dirLat[i]:dimLat+self.dirLat[i], self.dirLon[i]:dimLon+self.dirLon[i]])

        return out


def downsample(x):
    B, ch, ts, dimLat, dimLon = x.shape
    x = x.view((B, ch * ts, dimLat, dimLon))
    x = F.avg_pool2d(x, kernel_size=2)
    x = x.view(B, ch, ts, dimLat // 2, dimLon // 2)
    return x

def upsample(x):
    B, ch, ts, dimLat, dimLon = x.shape
    x = x.view((B, ch * ts, dimLat, dimLon))
    x = F.interpolate(x, scale_factor=2.)
    x = x.view(B, ch, ts, dimLat * 2, dimLon * 2)
    return x