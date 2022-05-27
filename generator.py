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

        self.BE0 = FirstBlockEncoder(1, 1 * ch)
        self.BE1 = BasicBlockEncoder(1 * ch, 2 * ch)
        self.BE2 = BasicBlockEncoder(2 * ch, 4 * ch)
        self.BE3 = BasicBlockEncoder(4 * ch, 8 * ch)

        self.BD0 = BasicBlockDecoder(8 * ch, 4 * ch)
        self.BD1 = BasicBlockDecoder(4 * ch, 2 * ch)
        self.BD2 = BasicBlockDecoder(2 * ch, 1 * ch)
        self.BD3 = BasicBlockDecoder(1 * ch, 1 * ch)
        self.BD4 = nn.Sequential(
            nn.BatchNorm3d(ch),
            nn.ReLU(),
            ConvSkew(ch, 1)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.BE0(x)
        x = self.BE1(x)
        x = self.BE2(x)
        x = self.BE3(x)
        x = self.BD0(x)
        x = self.BD1(x)
        x = self.BD2(x)
        x = self.BD3(x)
        x = self.BD4(x)
        x = self.tanh(x)

        return x
