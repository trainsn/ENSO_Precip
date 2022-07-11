# main file

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

    # data loader
    train_dataset = ENSOPrecipDataset(
        root=args.root,
        transform=transforms.Compose([Normalize(), ToTensor()]))

    kwargs = {"num_workers": 4, "pin_memory": True}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, **kwargs)

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

    precip_min = 0.
    precip_max = 1908.8616
    for i, sample in enumerate(train_loader):
        index = sample["index"].item()
        input_feat = Variable(sample["input_feat"], requires_grad=True).to("cuda:0")
        precip = sample["precip"].to("cuda:0")
        precip_mask = precip < -1.

        fake_precip = g_model(input_feat)
        grad = torch.autograd.grad(fake_precip[~precip_mask].norm(p=1), input_feat)
        print("{:d}:".format(index))
        for j in range(grad[0].shape[2]):
            print("\t{:d}, {:.6f}".format(j, abs(grad[0][0, :, j]).mean().item()))
        del input_feat, precip, precip_mask, fake_precip

        if args.save:
            np.save(os.path.join(args.root, "train", "grad", "grad_" + str(index)), grad[0][0].cpu().numpy())

if __name__ == "__main__":
    main(parse_args())
