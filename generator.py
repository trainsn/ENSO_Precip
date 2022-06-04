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
        e0 = self.BE0(x)    # 145, 288
        del x
        e1 = self.BE1(e0)   # 72, 144
        e2 = self.BE2(e1)   # 36, 72
        e3 = self.BE3(e2)   # 18, 36
        e4 = self.BE4(e3)   # 9, 18
        out = self.BD0(e4) + e3  # 18, 36
        del e4, e3
        out = self.BD1(out) + e2  # 36, 72
        del e2
        out = self.BD2(out) + e1  # 72, 144
        del e1
        out = self.BD3(out) + e0  # 145, 288
        del e0
        out = self.BD4(out)
        out = self.tanh(out)

        return out
