# Generator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from resblock import BasicBlockGenerator1D, BasicBlockGenerator3D

import pdb

class Generator(nn.Module):
    def __init__(self, ch=64):
        super(Generator, self).__init__()

        self.ch = ch

        self.BG0 = BasicBlockGenerator1D(1, 256, 220 * 16 * ch, 487, kernel_size=3, stride=1, padding=1)
        self.BG1 = BasicBlockGenerator3D(16 * ch, 8 * ch, [487, 20, 44], kernel_size=3, stride=1, padding=1)
        self.BG2 = BasicBlockGenerator3D(8 * ch, 4 * ch, [487, 39, 88], kernel_size=3, stride=1, padding=1)
        self.BG3 = BasicBlockGenerator3D(4 * ch, 2 * ch, [487, 78, 176], kernel_size=3, stride=1, padding=1)
        self.BG4 = BasicBlockGenerator3D(2 * ch, 1 * ch, [487, 155, 351], kernel_size=3, stride=1, padding=1)
        self.BG5 = nn.Sequential(
            nn.BatchNorm3d(ch),
            nn.ReLU(),
            nn.Conv3d(ch, 1, kernel_size=3, stride=1, padding=1),
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.BG0(x)
        x = x.view(x.shape[0], self.ch * 16, 487, 10, 22)
        x = self.BG1(x)
        x = self.BG2(x)
        x = self.BG3(x)
        x = self.BG4(x)
        x = self.BG5(x)
        x = self.tanh(x)

        return x
