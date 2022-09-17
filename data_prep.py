import os
import numpy as np
import netCDF4 as nc

import torch
import torch.nn.functional as F
import pdb

root = "/fs/project/PAS0027/enso_precip/train"
ts = 487
multiplier = 4
low_lat, low_lon = 86 * multiplier, 160 * multiplier    # 20S ~ 66N, 155E ~ 45W
precip = (-9999. * np.ones((ts, low_lat, low_lon))).astype(np.float32)
for i in range(1981, 2022):
    for j in range(1, 13):
        if i == 2021 and j == 8:
            break
        f = nc.Dataset(os.path.join(root, "PRISM_ppt", "PRISM_ppt_" + str(i) + str(j).zfill(2) + ".nc"))
        high_res = np.array(f["Band1"][:])[10:-11, :-1]   # (24.5N~49.5N, 125W~66.5W) [600, 1404]
        high_res = torch.from_numpy(high_res).unsqueeze(0).unsqueeze(0).cuda()
        low_res = F.interpolate(high_res, scale_factor =[multiplier / 24., multiplier / 24.], mode="bilinear")
        low_res = low_res.cpu().numpy().astype(np.float32)
        precip[(i - 1981) * 12 + j - 1, int(44.5 * multiplier + 0.5):int(69.5 * multiplier + 0.5),
        int(80 * multiplier + 0.5):int(138.5 * multiplier + 0.5)]  = low_res
    print("finish year " + str(i))

np.save(os.path.join(root, "PRISM_ppt.npy"), precip)
precip[:, int(44.5 * multiplier + 0.5):int(69.5 * multiplier + 0.5), int(80 * multiplier + 0.5):int(138.5 * multiplier + 0.5)]\
    .tofile(os.path.join(root, "PRISM_ppt.raw"))

high_lat, high_lon =  361, 576
high_feat = np.zeros((7, ts, high_lat, high_lon), dtype=np.float32)
sub_lat_st, sub_lat_en = 70 * 2, 156 * 2   # 20S ~ 66N
sub_lon_st, sub_lon_en = int(155. / 0.625 + 0.5), int(315. / 0.625 + 0.5) # 155E ~ 45W

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

f = nc.Dataset(os.path.join(root, "MERRA2_SST_80_21.nc"))
sst = f["SST"][:][12:12+ts]
high_feat[6, :, :, :high_lon//2] = sst[:, :, high_lon//2:]   # east
high_feat[6, :, :, high_lon//2:] = sst[:, :, :high_lon//2]    # west
high_feat[6].tofile(os.path.join(root, 'MERRA2_SST.raw'))

low_feat = np.zeros((7, ts, low_lat, low_lon), dtype=np.float32)
for i in range(ts):
    high_res = torch.from_numpy(high_feat[:, i, sub_lat_st:sub_lat_en, sub_lon_st:sub_lon_en]).unsqueeze(0).cuda()
    low_res = F.interpolate(high_res, size=[low_lat, low_lon], mode="bilinear")
    low_feat[:, i] = low_res.cpu().numpy().astype(np.float32)

np.save(os.path.join(root, "input_feat.npy"), low_feat)
pdb.set_trace()
