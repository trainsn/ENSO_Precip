# Residual block architecture

import torch
import torch.nn as nn
from torch.nn import functional as F

from layer import *

import pdb

class BasicBlockEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, activation=F.relu, downsample=True):
        super(BasicBlockEncoder, self).__init__()

        self.activation = activation
        self.downsample = downsample
        self.conv_res = None
        if self.downsample or in_channels != out_channels:
            self.conv_res = nn.Conv3d(in_channels, out_channels,
                                        1, 1, 0, bias=True)

        self.bn0 = nn.BatchNorm3d(in_channels)
        self.conv0 = ConvSkew(in_channels, out_channels)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv1 = ConvSkew(out_channels, out_channels)

    def forward(self, x):
        residual = x
        if self.conv_res is not None:
            residual = self.conv_res(residual)
        if self.downsample:
            residual = downsample(residual)

        out = self.bn0(x)
        out = self.activation(out)
        out = self.conv0(out)

        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = downsample(out)

        return out + residual

class FirstBlockEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, activation=F.relu):
        super(FirstBlockEncoder, self).__init__()

        self.activation = activation
        self.conv_res = nn.Conv3d(in_channels, out_channels,
                                  1, 1, 0, bias=True)

        self.bn0 = nn.BatchNorm3d(in_channels)
        self.conv0 = ConvSkew(in_channels, out_channels)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv1 = ConvSkew(out_channels, out_channels)

    def forward(self, x):
        residual = self.conv_res(x)
        residual = downsample(residual)

        out = self.bn0(x)
        out = self.conv0(out)

        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv1(out)

        out = downsample(out)

        return out + residual

class BasicBlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, out_size, activation=F.relu, upsample=True):
        super(BasicBlockDecoder, self).__init__()

        self.out_size = out_size
        self.activation = activation
        self.upsample = upsample
        self.conv_res = None

        if self.upsample or in_channels != out_channels:
            self.conv_res = nn.Conv3d(in_channels, out_channels,
                                      1, 1, 0, bias=False)

        self.bn0 = nn.BatchNorm3d(in_channels)
        self.conv0 = ConvSkew(in_channels, out_channels)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv1 = ConvSkew(out_channels, out_channels)

    def forward(self, x):
        residual = x
        if self.upsample:
            residual = upsample(residual, self.out_size)
        if self.conv_res is not None:
            residual = self.conv_res(residual)

        out = self.bn0(x)
        out = self.activation(out)
        if self.upsample:
            out = upsample(out, self.out_size)
        out = self.conv0(out)

        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv1(out)

        return out + residual
