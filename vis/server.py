# server file

from __future__ import absolute_import, division, print_function

import pdb

import os
import argparse
import math
import numpy as np
from tqdm import tqdm
from silx.math import colormap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("../")

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

    return parser.parse_args()

# global variables
g_model = None
enso_input_feat, input_feat = None, None
precip = None
enso_precip = None
recep_field = 22
input_feat_min = np.array([95827.602, 199.63177, 4644.373, 4644.373, -21.138626, -21.138626, 267.872]).reshape((-1, 1, 1, 1))
input_feat_max = np.array([104431.23, 314.57242, 5970.5391, 5970.5391, 87.681274, 87.681274, 309.222]).reshape((-1, 1, 1, 1))
precip_min = 0.
precip_max = 1908.8616
multiplier = 4
grad = None
time_idx, lat_idx, lon_idx = None, None, None

# the init function
def init(args):
    # log hyperparameters
    print(args)

    # define global variables
    global g_model
    global enso_input_feat
    global enso_precip
    global recep_field
    global input_feat_min, input_feat_max
    global precip_min, precip_max
    global multiplier

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

# import flask
from flask import Flask, jsonify, render_template, request, send_file
app = Flask(__name__)

# flask code
@app.route("/")
def main():
    return render_template("index.html")

@app.route("/forward_prop", methods=["POST"])
def forward_prop():
    global precip
    global enso_precip
    global input_feat
    global time_idx

    time_idx = int(request.form["idx"]) + recep_field - 1
    input_feat = enso_input_feat[:, time_idx - recep_field + 1: time_idx + 1]
    precip = enso_precip[time_idx]

    input_feat = ((input_feat - (input_feat_min + input_feat_max) / 2.)
                  / ((input_feat_max - input_feat_min) / 2.)).astype(np.float32)
    precip = ((precip - (precip_min + precip_max) / 2.) / ((precip_max - precip_min) / 2.)).astype(np.float32)

    input_feat = Variable(torch.from_numpy(input_feat).unsqueeze(0), requires_grad=True).to("cuda:0")
    precip = torch.from_numpy(precip).to("cuda:0")
    precip_mask = precip < -1.

    fake_precip = g_model(input_feat)[0, 0, 0]
    fake_precip[precip_mask] = precip[precip_mask]
    print(time_idx, (abs(fake_precip - precip).sum() / (~precip_mask).sum()).item(), abs(fake_precip - precip).max().item())

    fake_precip = fake_precip[int(44.5 * multiplier + 0.5):int(69.5 * multiplier + 0.5),
                  int(80 * multiplier + 0.5):int(138.5 * multiplier + 0.5)].unsqueeze(0).unsqueeze(0)
    fake_precip = F.interpolate(fake_precip, scale_factor = 2., mode="bilinear")
    fake_precip = fake_precip[0, 0].detach().cpu().numpy()

    viridis = np.array([[68, 68, 69, 69, 70, 70, 70, 70, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 71, 71, 71, 71, 71, 70, 70, 70, 70, 69, 69, 69, 68, 68, 68, 67, 67, 66, 66, 66, 65, 65, 64, 64, 63, 63, 62, 62, 62, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 56, 56, 55, 55, 54, 54, 53, 53, 52, 52, 51, 51, 50, 50, 49, 49, 49, 48, 48, 47, 47, 46, 46, 46, 45, 45, 44, 44, 44, 43, 43, 42, 42, 42, 41, 41, 41, 40, 40, 39, 39, 39, 38, 38, 38, 37, 37, 37, 36, 36, 35, 35, 35, 34, 34, 34, 33, 33, 33, 33, 32, 32, 32, 31, 31, 31, 31, 31, 31, 31, 30, 30, 30, 31, 31, 31, 31, 31, 31, 32, 32, 33, 33, 34, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 49, 50, 52, 53, 55, 56, 58, 59, 61, 63, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 99, 101, 103, 105, 108, 110, 112, 115, 117, 119, 122, 124, 127, 129, 132, 134, 137, 139, 142, 144, 147, 149, 152, 155, 157, 160, 162, 165, 168, 170, 173, 176, 178, 181, 184, 186, 189, 192, 194, 197, 200, 202, 205, 208, 210, 213, 216, 218, 221, 223, 226, 229, 231, 234, 236, 239, 241, 244, 246, 248, 251, 253],
                        [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 182, 183, 184, 185, 186, 187, 188, 188, 189, 190, 191, 192, 193, 193, 194, 195, 196, 197, 197, 198, 199, 200, 200, 201, 202, 203, 203, 204, 205, 205, 206, 207, 208, 208, 209, 209, 210, 211, 211, 212, 213, 213, 214, 214, 215, 215, 216, 216, 217, 217, 218, 218, 219, 219, 220, 220, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 229, 230, 230, 230, 231, 231],
                        [84, 86, 87, 89, 90, 92, 93, 94, 96, 97, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 125, 126, 126, 127, 128, 129, 129, 130, 131, 131, 132, 132, 133, 133, 134, 134, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 138, 139, 139, 139, 139, 140, 140, 140, 140, 140, 140, 141, 141, 141, 141, 141, 141, 141, 141, 141, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 141, 141, 141, 141, 141, 141, 141, 140, 140, 140, 140, 140, 139, 139, 139, 139, 138, 138, 138, 137, 137, 137, 136, 136, 136, 135, 135, 134, 134, 133, 133, 133, 132, 131, 131, 130, 130, 129, 129, 128, 127, 127, 126, 125, 124, 124, 123, 122, 121, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 101, 100, 99, 98, 96, 95, 94, 92, 91, 90, 88, 87, 86, 84, 83, 81, 80, 78, 77, 75, 73, 72, 70, 69, 67, 65, 64, 62, 60, 59, 57, 55, 54, 52, 50, 48, 47, 45, 43, 41, 40, 38, 37, 35, 33, 32, 31, 29, 28, 27, 26, 25, 25, 24, 24, 24, 25, 25, 26, 27, 28, 29, 30, 32, 33, 35, 37],
                        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]).T
    output_image = colormap.cmap(fake_precip, viridis, -1., -0.25)
    del fake_precip

    return jsonify({"image": output_image.flatten().tolist()})

@app.route("/backward_prop", methods=["POST"])
def backward_prop():
    global grad
    global time_idx

    lat_idx = int(request.form["lat_idx"])
    lon_idx = int(request.form["lon_idx"])
    assert(time_idx is not None)
    print(time_idx, lat_idx, lon_idx)

    precip_mask = torch.ones_like(precip)
    precip_mask[int(44.5 * multiplier + 0.5) + lat_idx: int(44.5 * multiplier + 0.5) + lat_idx + 2,
    int(80 * multiplier + 0.5) + lon_idx: int(80 * multiplier + 0.5) + lon_idx + 2] = 0
    precip_mask = (precip_mask + (precip < -1.)).bool()

    fake_precip = g_model(input_feat)[0, 0, 0]
    grad = None
    grad = torch.autograd.grad(fake_precip[~precip_mask].norm(p=1), input_feat)
    for j in range(grad[0].shape[2]):
        print("\t{:d}, {:.6f}".format(j, abs(grad[0][0, :, j]).sum().item()))
    return jsonify({"status": 0})

# @app.route("/retrieve_variable", methods=["POST"])
# def retrieve_variable():


if __name__ == "__main__":
    init(parse_args())
    app.run()
