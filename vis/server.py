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
        print("\t{:d}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".
              format(j, abs(grad[0][0, 0, j]).sum().item(), abs(grad[0][0, 1, j]).sum().item(), abs(grad[0][0, 2, j]).sum().item(),
                     abs(grad[0][0, 3, j]).sum().item(), abs(grad[0][0, 4, j]).sum().item(), abs(grad[0][0, 5, j]).sum().item(),
                     abs(grad[0][0, 6, j]).sum().item()))
    grad_stats = abs(grad[0]).sum((3, 4))
    return jsonify({"grad_stats": grad_stats.flatten().tolist()})

@app.route("/retrieve_variable_time", methods=["POST"])
def retrieve_variable_time():
    variable_idx = int(request.form["variable_idx"])
    variable_name = request.form["variable_name"]
    relative_month = recep_field - int(request.form["relamonth_idx"])
    print(variable_name, relative_month)

    retrieved_values = input_feat[0, variable_idx, relative_month].detach().cpu().numpy()
    retrieved_sensitivity = grad[0][0, variable_idx, relative_month].detach().cpu().numpy()

    kindlmann = np.array([[0, 5, 9, 13, 17, 20, 22, 25, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 38, 38, 37, 37, 36, 35, 34, 33, 32, 32, 31, 30, 29, 28, 27, 26, 25, 25, 24, 24, 20, 15, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 12, 15, 17, 18, 20, 22, 24, 26, 29, 32, 35, 38, 41, 45, 48, 52, 56, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 116, 120, 124, 128, 133, 137, 141, 145, 150, 154, 158, 163, 167, 171, 175, 180, 184, 188, 193, 197, 201, 205, 209, 214, 218, 223, 228, 233, 237, 243, 246, 247, 248, 249, 249, 250, 250, 250, 251, 251, 251, 251, 252, 252, 252, 252, 252, 252, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 12, 15, 19, 22, 25, 28, 32, 35, 38, 41, 43, 46, 49, 51, 53, 56, 58, 60, 62, 64, 66, 68, 70, 71, 73, 75, 76, 78, 80, 81, 83, 84, 86, 87, 88, 90, 91, 93, 94, 95, 96, 98, 99, 100, 102, 103, 104, 105, 107, 108, 109, 110, 112, 113, 114, 115, 117, 118, 119, 120, 122, 123, 124, 125, 127, 128, 129, 130, 132, 133, 134, 135, 137, 138, 139, 140, 142, 143, 144, 145, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 190, 191, 192, 193, 193, 194, 195, 196, 196, 197, 198, 198, 199, 199, 200, 200, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 205, 204, 205, 205, 206, 207, 207, 208, 209, 210, 211, 212, 214, 215, 216, 217, 218, 220, 221, 222, 224, 225, 226, 227, 229, 230, 232, 233, 234, 236, 237, 238, 240, 241, 242, 244, 245, 247, 248, 249, 251, 252, 254, 255],
                          [0, 4, 8, 13, 16, 20, 23, 26, 29, 32, 35, 38, 42, 45, 48, 51, 54, 57, 60, 63, 66, 68, 71, 74, 77, 80, 83, 86, 89, 93, 96, 99, 102, 106, 109, 112, 116, 119, 123, 126, 129, 132, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 170, 174, 175, 175, 175, 175, 175, 174, 173, 172, 170, 168, 166, 165, 163, 160, 158, 156, 154, 152, 149, 147, 145, 143, 141, 139, 137, 135, 133, 132, 130, 128, 127, 125, 123, 122, 120, 119, 118, 116, 115, 114, 113, 112, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 96, 95, 94, 92, 91, 89, 88, 86, 84, 83, 81, 79, 77, 75, 74, 72, 70, 67, 65, 63, 61, 59, 56, 54, 52, 49, 47, 44, 42, 39, 37, 34, 31, 29, 26, 23, 20, 18, 15, 12, 10, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 57, 86, 105, 119, 131, 141, 149, 157, 163, 169, 174, 179, 184, 188, 192, 195, 199, 202, 205, 208, 211, 213, 216, 218, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 246, 248, 250, 252, 253, 255],
                          [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]).T

    extendedKindlmann = np.array([[0, 5, 9, 13, 16, 19, 22, 24, 26, 27, 28, 29, 29, 30, 30, 30, 30, 29, 29, 28, 27, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 18, 17, 14, 8, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 8, 10, 11, 13, 15, 17, 20, 22, 26, 29, 32, 36, 40, 43, 47, 51, 55, 59, 63, 67, 71, 76, 80, 84, 88, 92, 97, 101, 105, 109, 113, 118, 122, 126, 130, 135, 139, 144, 149, 154, 160, 165, 171, 177, 183, 189, 196, 202, 209, 216, 222, 229, 236, 243, 244, 245, 245, 245, 246, 246, 246, 246, 246, 246, 247, 247, 247, 247, 247, 248, 248, 248, 248, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 248, 245, 242, 239, 236, 234, 232, 230, 229, 227, 226, 225, 224, 224, 223, 223, 222, 222, 222, 222, 223, 223, 223, 224, 224, 225, 226, 226, 227, 228, 229, 229, 230, 231, 231, 231, 232, 232, 232, 232, 233, 233, 233, 233, 233, 233, 234, 234, 234, 235, 235, 236, 237, 237, 238, 239, 240, 242, 243, 245, 248, 251, 255],
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 8, 11, 15, 18, 22, 25, 28, 31, 33, 36, 38, 40, 42, 44, 46, 48, 49, 51, 52, 54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 112, 113, 114, 115, 116, 116, 117, 118, 118, 119, 119, 120, 120, 121, 121, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 122, 122, 121, 120, 119, 118, 117, 115, 113, 111, 108, 105, 102, 98, 94, 90, 91, 92, 94, 96, 98, 99, 101, 103, 105, 107, 109, 111, 112, 114, 115, 117, 118, 120, 121, 122, 123, 125, 126, 127, 129, 130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 152, 155, 159, 162, 165, 168, 171, 173, 176, 178, 180, 182, 184, 186, 188, 190, 191, 193, 195, 196, 198, 199, 201, 202, 204, 205, 207, 208, 209, 211, 212, 213, 215, 216, 218, 219, 221, 222, 224, 225, 227, 228, 230, 231, 233, 234, 236, 237, 239, 240, 242, 243, 245, 246, 247, 249, 250, 251, 252, 253, 254, 255, 255],
                                  [0, 4, 9, 13, 17, 21, 24, 27, 30, 34, 38, 42, 46, 50, 53, 57, 61, 65, 68, 72, 75, 79, 82, 85, 88, 92, 95, 98, 101, 103, 106, 109, 111, 115, 119, 120, 120, 119, 118, 116, 114, 112, 109, 107, 104, 101, 99, 96, 94, 91, 89, 87, 85, 83, 81, 79, 77, 76, 74, 73, 71, 70, 69, 67, 66, 65, 64, 63, 61, 60, 58, 56, 55, 53, 51, 49, 47, 45, 43, 41, 39, 36, 34, 31, 29, 26, 24, 21, 18, 15, 13, 10, 8, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 27, 37, 46, 52, 58, 63, 67, 71, 74, 77, 79, 83, 87, 91, 96, 101, 106, 112, 118, 123, 129, 135, 141, 147, 153, 158, 164, 169, 175, 180, 186, 191, 196, 201, 206, 211, 216, 221, 225, 230, 235, 239, 243, 248, 250, 250, 250, 251, 251, 251, 251, 251, 251, 251, 251, 251, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 255, 255, 255, 255],
                                  [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]).T
    value_image = colormap.cmap(retrieved_values, kindlmann, retrieved_values.min(), retrieved_values.max())

    bent_cool_warm = np.array([[59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 95, 96, 97, 98, 100, 101, 102, 103, 105, 106, 107, 109, 110, 111, 113, 114, 115, 117, 118, 119, 121, 122, 123, 125, 126, 128, 129, 130, 132, 133, 135, 136, 138, 139, 140, 142, 143, 145, 146, 148, 149, 151, 152, 154, 155, 157, 159, 160, 162, 163, 165, 166, 168, 170, 171, 173, 174, 176, 178, 179, 181, 183, 184, 186, 188, 189, 191, 193, 194, 196, 198, 200, 201, 203, 205, 207, 208, 210, 212, 214, 215, 217, 219, 221, 223, 225, 226, 228, 230, 232, 234, 236, 238, 239, 241, 242, 242, 241, 241, 241, 241, 240, 240, 240, 239, 239, 239, 238, 238, 238, 237, 237, 237, 236, 236, 236, 235, 235, 235, 234, 234, 233, 233, 233, 232, 232, 232, 231, 231, 230, 230, 230, 229, 229, 228, 228, 228, 227, 227, 226, 226, 226, 225, 225, 224, 224, 223, 223, 223, 222, 222, 221, 221, 220, 220, 219, 219, 218, 218, 217, 217, 217, 216, 216, 215, 215, 214, 214, 213, 212, 212, 211, 211, 210, 210, 209, 209, 208, 208, 207, 207, 206, 205, 205, 204, 204, 203, 203, 202, 201, 201, 200, 200, 199, 198, 198, 197, 197, 196, 195, 195, 194, 193, 193, 192, 192, 191, 190, 190, 189, 188, 188, 187, 186, 186, 185, 184, 184, 183, 182, 181, 181, 180],
                               [76, 78, 79, 80, 82, 83, 85, 86, 87, 89, 90, 91, 93, 94, 95, 97, 98, 100, 101, 102, 104, 105, 106, 108, 109, 110, 112, 113, 114, 116, 117, 118, 120, 121, 122, 124, 125, 126, 128, 129, 130, 131, 133, 134, 135, 137, 138, 139, 141, 142, 143, 145, 146, 147, 149, 150, 151, 152, 154, 155, 156, 158, 159, 160, 162, 163, 164, 165, 167, 168, 169, 171, 172, 173, 175, 176, 177, 178, 180, 181, 182, 184, 185, 186, 187, 189, 190, 191, 193, 194, 195, 196, 198, 199, 200, 202, 203, 204, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 218, 219, 220, 222, 223, 224, 225, 227, 228, 229, 230, 232, 233, 234, 235, 237, 238, 239, 240, 242, 241, 240, 238, 237, 235, 234, 232, 231, 229, 228, 226, 225, 223, 222, 220, 219, 217, 216, 214, 213, 211, 210, 208, 207, 205, 204, 202, 201, 199, 197, 196, 194, 193, 191, 190, 188, 187, 185, 184, 182, 181, 179, 177, 176, 174, 173, 171, 170, 168, 167, 165, 163, 162, 160, 159, 157, 156, 154, 152, 151, 149, 148, 146, 144, 143, 141, 140, 138, 136, 135, 133, 132, 130, 128, 127, 125, 123, 122, 120, 119, 117, 115, 114, 112, 110, 108, 107, 105, 103, 102, 100, 98, 96, 95, 93, 91, 89, 87, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 55, 53, 50, 48, 45, 43, 40, 37, 34, 30, 27, 22, 18, 12, 4],
                               [192, 193, 194, 194, 195, 196, 196, 197, 197, 198, 199, 199, 200, 200, 201, 202, 202, 203, 203, 204, 204, 205, 206, 206, 207, 207, 208, 208, 209, 209, 210, 210, 211, 211, 212, 212, 213, 213, 214, 214, 214, 215, 215, 216, 216, 217, 217, 218, 218, 218, 219, 219, 220, 220, 220, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 230, 231, 231, 231, 232, 232, 232, 232, 233, 233, 233, 234, 234, 234, 234, 235, 235, 235, 236, 236, 236, 236, 237, 237, 237, 237, 238, 238, 238, 238, 239, 239, 239, 239, 240, 240, 240, 240, 241, 241, 241, 241, 242, 242, 242, 241, 239, 237, 235, 232, 230, 228, 226, 224, 222, 219, 217, 215, 213, 211, 209, 207, 205, 203, 201, 199, 196, 194, 192, 190, 188, 186, 184, 182, 180, 179, 177, 175, 173, 171, 169, 167, 165, 163, 161, 159, 158, 156, 154, 152, 150, 148, 147, 145, 143, 141, 140, 138, 136, 134, 133, 131, 129, 128, 126, 124, 123, 121, 119, 118, 116, 114, 113, 111, 110, 108, 107, 105, 104, 102, 101, 99, 98, 96, 95, 93, 92, 90, 89, 88, 86, 85, 83, 82, 81, 79, 78, 77, 75, 74, 73, 72, 70, 69, 68, 67, 65, 64, 63, 62, 61, 60, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38],
                               [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]).T

    threshold = max(abs(retrieved_sensitivity.min()), abs(retrieved_sensitivity.max()))
    sensitivity_image = colormap.cmap(retrieved_sensitivity, bent_cool_warm, -threshold, threshold)

    return jsonify({"value_image": value_image.flatten().tolist(),
                    "sensitivity_image": sensitivity_image.flatten().tolist()})

if __name__ == "__main__":
    init(parse_args())
    app.run()
