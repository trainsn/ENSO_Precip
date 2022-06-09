# Generator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from resblock import BasicBlockEncoder, FirstBlockEncoder, BasicBlockDecoder
from layer import ConvSkew

import pdb

class Generator(nn.Module):
    def __init__(self, ch=64):
        super(Generator, self).__init__()

        self.ch = ch

        self.BE0 = FirstBlockEncoder(6, 1 * ch)
        self.BE1 = BasicBlockEncoder(1 * ch, 2 * ch)
        self.BE2 = BasicBlockEncoder(2 * ch, 4 * ch)
        self.BE3 = BasicBlockEncoder(4 * ch, 8 * ch)
        self.BE4 = BasicBlockEncoder(8 * ch, 16 * ch)

        self.BD0 = BasicBlockDecoder(16 * ch, 8 * ch, [32, 60])
        self.BD1 = BasicBlockDecoder(8 * ch, 4 * ch, [64, 120])
        self.BD2 = BasicBlockDecoder(4 * ch, 2 * ch, [129, 240])
        self.BD3 = BasicBlockDecoder(2 * ch, 1 * ch, [258, 480])
        self.BD4 = BasicBlockDecoder(1 * ch, 1 * ch, [516, 960])
        self.BD5 = nn.Sequential(
            nn.BatchNorm3d(ch),
            nn.ReLU(),
            ConvSkew(ch, 1)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.BE0(x)
        x = x[:, :, 2:]
        x = self.BE1(x)
        x = x[:, :, 2:]
        x = self.BE2(x)
        x = x[:, :, 2:]
        x = self.BE3(x)
        x = x[:, :, 2:]
        x = self.BE4(x)
        x = x[:, :, 2:]
        x = self.BD0(x)
        x = x[:, :, 2:]
        x = self.BD1(x)
        x = x[:, :, 2:]
        x = self.BD2(x)
        x = x[:, :, 2:]
        x = self.BD3(x)
        x = x[:, :, 2:]
        x = self.BD4(x)
        x = x[:, :, 2:]
        x = self.BD5(x)
        x = x[:, :, 1:]
        x = self.tanh(x)

        return x
