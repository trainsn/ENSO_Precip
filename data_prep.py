import os
import numpy as np
import netCDF4 as nc

import torch
import torch.nn.functional as F
import pdb

root = "/fs/project/PAS0027/enso_precip/train"
ts = 487
low_lat, low_lon = 145, 288
precip = (-9999. * np.ones((ts, low_lat, low_lon))).astype(np.float32)
for i in range(1981, 2022):
    for j in range(1, 13):
        if i == 2021 and j == 8:
            break
        f = nc.Dataset(os.path.join(root, "PRISM_ppt", "PRISM_ppt_" + str(i) + str(j).zfill(2) + ".nc"))
        high_res = np.array(f["Band1"][:])[600-578:1170-578, :-1620+3000]
        high_res = torch.from_numpy(high_res).unsqueeze(0).unsqueeze(0).cuda()
        low_res = F.interpolate(high_res, size=[20, 47], mode="bilinear")
        low_res = low_res.cpu().numpy().astype(np.float32)
        precip[(i - 1981) * 12 + j - 1, low_lat//2+20:low_lat//2+40, low_lon-101:low_lon-54] = low_res
        # print((low_res < 0).sum())
    print("finish year " + str(i))

np.save(os.path.join(root, "PRISM_ppt.npy"), precip)

high_lat, high_lon =  361, 576
high_feat = np.zeros((6, ts, high_lat, high_lon), dtype=np.float32)

f = nc.Dataset(os.path.join(root, "MERRA2_SLP_mo_80_22.nc"))
slp = f["SLP"][:][12:12+ts]
high_feat[0, :, :, :high_lon//2] = slp[:, :, high_lon//2:]   # east
high_feat[0, :, :, high_lon//2:] = slp[:, :, :high_lon//2]    # west
high_feat[0].tofile(os.path.join(root, 'MERRA2_SLP.raw'))

f = nc.Dataset(os.path.join(root, "MERRA2_T2_mo_80_22.nc"))
t2 = f["T2"][:][12:12+ts]
high_feat[1, :, :, :high_lon//2] = t2[:, :, high_lon//2:]   # east
high_feat[1, :, :, high_lon//2:] = t2[:, :, :high_lon//2]    # west
high_feat[1].tofile(os.path.join(root, 'MERRA2_T2.raw'))

f = nc.Dataset(os.path.join(root, "MERRA2_HGT_mo_80_22.nc"))
hgt = f["HGT"][:][12:12+ts, [16, 21]].transpose(1, 0, 2, 3)
high_feat[2:4, :, :, :high_lon//2] = hgt[:, :, :, high_lon//2:]   # east
high_feat[2:4, :, :, high_lon//2:] = hgt[:, :, :, :high_lon//2]    # west
high_feat[2].tofile(os.path.join(root, 'MERRA2_HGT_500hPa.raw'))
high_feat[3].tofile(os.path.join(root, 'MERRA2_HGT_250hPa.raw'))

f = nc.Dataset(os.path.join(root, "MERRA2_U_mo_80_22.nc"))
u = f["U"][:][12:12+ts, [21, 22]].transpose(1, 0, 2, 3)
high_feat[4:6, :, :, :high_lon//2] = u[:, :, :, high_lon//2:]   # east
high_feat[4:6, :, :, high_lon//2:] = u[:, :, :, :high_lon//2]    # west
high_feat[4].tofile(os.path.join(root, 'MERRA2_U_250hPa.raw'))
high_feat[5].tofile(os.path.join(root, 'MERRA2_U_200hPa.raw'))

low_feat = np.zeros((6, ts, low_lat, low_lon), dtype=np.float32)

for i in range(ts):
    high_res = torch.from_numpy(high_feat[:, i]).unsqueeze(0).cuda()
    low_res = F.interpolate(high_res, size=[low_lat, low_lon], mode="bilinear")
    low_feat[:, i] = low_res.cpu().numpy().astype(np.float32)

np.save(os.path.join(root, "input_feat.npy"), low_feat)
pdb.set_trace()