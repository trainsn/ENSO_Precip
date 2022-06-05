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

        self.BD0 = BasicBlockDecoder(16 * ch, 8 * ch, [18, 36])
        self.BD1 = BasicBlockDecoder(8 * ch, 4 * ch, [36, 72])
        self.BD2 = BasicBlockDecoder(4 * ch, 2 * ch, [72, 144])
        self.BD3 = BasicBlockDecoder(2 * ch, 1 * ch, [145, 288])
        self.BD4 = nn.Sequential(
            nn.BatchNorm3d(ch),
            nn.ReLU(),
            ConvSkew(ch, 1)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.BE0(x)    # 145, 288
        x = self.BE1(x)   # 72, 144
        x = self.BE2(x)   # 36, 72
        x = self.BE3(x)   # 18, 36
        x = self.BE4(x)   # 9, 18
        x = self.BD0(x)  # 18, 36
        x = self.BD1(x)  # 36, 72
        x = self.BD2(x)  # 72, 144
        x = self.BD3(x)  # 145, 288
        x = self.BD4(x)
        x = self.tanh(x)

        return x
