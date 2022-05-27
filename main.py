# main file

from __future__ import absolute_import, division, print_function

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

import sys
sys.path.append("/root/apex")
from apex import amp

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

    parser.add_argument("--lr", type=float, default=5e-5,
                        help="learning rate (default: 5e-5)")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="beta1 of Adam (default: 0.0)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--opt-level", default='O2',
                        help='amp opt_level, default="O2"')
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="number of epochs to train")

    parser.add_argument("--log-every", type=int, default=10,
                        help="log training status every given given number of epochs (default: 10)")
    parser.add_argument("--check-every", type=int, default=200,
                        help="save checkpoint every given number of epochs ")

    return parser.parse_args()

# the main function
def main(args):
    # log hyperparameters
    print(args)

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # data loader
    train_dataset = ENSOPrecipDataset(
        root=args.root,
        transform=transforms.Compose([Normalize(), ToTensor()]))

    kwargs = {"num_workers": 4, "pin_memory": True}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, **kwargs)

    # model
    def weights_init(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def add_sn(m):
        for name, c in m.named_children():
            m.add_module(name, add_sn(c))
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            return nn.utils.spectral_norm(m, eps=1e-4)
        else:
            return m

    g_model = Generator(args.ch)
    g_model.apply(weights_init)
    if args.sn:
        g_model = add_sn(g_model)
    g_model.to("cuda")

    l1_criterion = nn.L1Loss().cuda()

    # optimizer
    g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr,
                             betas=(args.beta1, args.beta2))
    g_model, g_optimizer = amp.initialize(g_model, g_optimizer, opt_level=args.opt_level)

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            g_model.load_state_dict(checkpoint["g_model_state_dict"])
            # g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
            # train_losses = checkpoint["train_losses"]
            print("=> loaded checkpoint {} (epoch {})"
                  .format(args.resume, checkpoint["epoch"]))

        # main loop
    for epoch in range(args.start_epoch, args.epochs):
        # training...
        g_model.train()
        train_l1_loss = 0.
        for i, sample in enumerate(train_loader):
            sst = sample["sst"].to("cuda:0")
            precip = sample["precip"].to("cuda:0")

            g_optimizer.zero_grad()
            fake_precip = g_model(sst)

            loss = 0.
            l1_loss = l1_criterion(precip, fake_precip)
            loss += l1_loss

            with amp.scale_loss(loss, g_optimizer, loss_id=0) as loss_scaled:
                loss_scaled.backward()
            g_optimizer.step()
            train_l1_loss += l1_loss.item()

        if epoch % args.log_every == 0:
            print("====> Epoch: {} Average L1_loss: {:.4f}".format(
                epoch, train_l1_loss / len(train_loader.dataset)))

        if (epoch + 1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch + 1))
            torch.save({"epoch": epoch + 1,
                        "g_model_state_dict": g_model.state_dict(),
                        "g_optimizer_state_dict": g_optimizer.state_dict(),
                        # "train_losses": train_losses
                        },
                       os.path.join(args.root, "model_" + str(epoch + 1) + ".pth.tar"))

            torch.save(g_model.state_dict(),
                       os.path.join(args.root, "model_" + str(epoch + 1) + ".pth"))

if __name__ == "__main__":
    main(parse_args())