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

        self.BE0 = FirstBlockEncoder(7, 1 * ch).to("cuda:0")
        self.BE1 = BasicBlockEncoder(1 * ch, 2 * ch).to("cuda:1")
        self.BE2 = BasicBlockEncoder(2 * ch, 4 * ch).to("cuda:1")
        self.BE3 = BasicBlockEncoder(4 * ch, 8 * ch).to("cuda:1")
        self.BE4 = BasicBlockEncoder(8 * ch, 16 * ch).to("cuda:1")

        self.BD0 = BasicBlockDecoder(16 * ch, 8 * ch, [21, 40]).to("cuda:1")
        self.BD1 = BasicBlockDecoder(8 * ch, 4 * ch, [43, 80]).to("cuda:1")
        self.BD2 = BasicBlockDecoder(4 * ch, 2 * ch, [86, 160]).to("cuda:1")
        self.BD3 = BasicBlockDecoder(2 * ch, 1 * ch, [172, 320]).to("cuda:1")
        self.BD4 = BasicBlockDecoder(1 * ch, 1 * ch, [344, 640]).to("cuda:0")
        self.BD5 = nn.Sequential(
            nn.BatchNorm3d(ch),
            nn.ReLU(),
            ConvSkew(ch, 1)
        ).to("cuda:0")

    def forward(self, x):
        x = self.BE0(x)
        x = x[:, :, 2:].to("cuda:1")    # 20
        x = self.BE1(x)
        x = x[:, :, 2:]     # 18
        x = self.BE2(x)
        x = x[:, :, 2:]     # 16
        x = self.BE3(x)
        x = x[:, :, 2:]     # 14
        x = self.BE4(x)
        x = x[:, :, 2:]     # 12
        x = self.BD0(x)
        x = x[:, :, 2:]     # 10
        x = self.BD1(x)
        x = x[:, :, 2:]     # 8
        x = self.BD2(x)
        x = x[:, :, 2:]     # 6
        x = self.BD3(x)
        x = x[:, :, 2:].to("cuda:0")     # 4
        x = self.BD4(x)
        x = x[:, :, 2:]     # 2
        x = self.BD5(x)
        x = x[:, :, 1:]
        x = F.tanh(x)

        return x
