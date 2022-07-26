# sensitivity file

from __future__ import absolute_import, division, print_function

import pdb

import os
import argparse
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from ENSOPrecip import *
from generator import Generator
from resblock import *

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")

    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")

    parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")

    parser.add_argument("--ch", type=int, default=64,
                        help="channel multiplier (default: 64)")

    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")

    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")

    parser.add_argument("--time-idx", required=True, type=int,
                        help="precip temporal index")
    parser.add_argument("--lat-idx", required=True, type=int,
                        help="precip latitude index")
    parser.add_argument("--lon-idx", required=True, type=int,
                        help="precip longitude index")
    parser.add_argument("--save", action="store_true", default=False,
                        help="save the npy file")

    return parser.parse_args()

# the main function
def main(args):
    # log hyperparameters
    print(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    def add_sn(m):
        for name, c in m.named_children():
            m.add_module(name, add_sn(c))
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            return nn.utils.spectral_norm(m, eps=1e-4)
        else:
            return m

    g_model = Generator(args.ch)
    if args.sn:
        g_model = add_sn(g_model)

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint["epoch"]
            g_model.load_state_dict(checkpoint["g_model_state_dict"])
            # g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
            print("=> loaded checkpoint {} (epoch {})"
                  .format(args.resume, checkpoint["epoch"]))

    g_model.train()  # In BatchNorm, we still want the mean and var calculated from the current instance

    enso_input_feat = np.load(os.path.join(args.root, "train", "input_feat.npy"))
    enso_precip = np.load(os.path.join(args.root, "train", "PRISM_ppt.npy"))
    recep_field = 22
    input_feat = enso_input_feat[:, args.time_idx - recep_field + 1: args.time_idx + 1]
    precip = enso_precip[args.time_idx]

    input_feat_min = np.array([95827.602, 199.63177, 4644.373, 4644.373, -21.138626, -21.138626, 267.872]).reshape((-1, 1, 1, 1))
    input_feat_max = np.array([104431.23, 314.57242, 5970.5391, 5970.5391, 87.681274, 87.681274, 309.222]).reshape((-1, 1, 1, 1))
    input_feat = ((input_feat - (input_feat_min + input_feat_max) / 2.)
                  / ((input_feat_max - input_feat_min) / 2.)).astype(np.float32)

    precip_min = 0.
    precip_max = 1908.8616
    precip = ((precip - (precip_min + precip_max) / 2.) / ((precip_max - precip_min) / 2.)).astype(np.float32)

    input_feat = Variable(torch.from_numpy(input_feat).unsqueeze(0), requires_grad=True).to("cuda:0")
    precip = torch.from_numpy(precip).to("cuda:0")
    precip_mask = torch.ones_like(precip)
    multiplier = 4
    precip_mask[int(44.5 * multiplier + 0.5) + args.lat_idx : int(44.5 * multiplier + 0.5) + args.lat_idx + 2,
           int(80 * multiplier + 0.5) + args.lon_idx : int(80 * multiplier + 0.5) + args.lon_idx + 2] = 0
    pdb.set_trace()
    precip_mask = (precip_mask + (precip < -1.)).bool()

    fake_precip = g_model(input_feat)
    grad = torch.autograd.grad(fake_precip[0, 0, 0][~precip_mask].norm(p=1), input_feat)
    print("{:d}:".format(args.time_idx))
    for j in range(grad[0].shape[2]):
        print("\t{:d}, {:.6f}".format(j, abs(grad[0][0, :, j]).sum().item()))
    del input_feat, precip, precip_mask, fake_precip

    if args.save:
        np.save(os.path.join(args.root, "train", "gradient", "grad_" + str(args.time_idx)), grad[0][0].cpu().numpy())

if __name__ == "__main__":
    main(parse_args())
