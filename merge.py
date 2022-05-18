import os
import numpy as np
import netCDF4 as nc

import torch
import torch.nn.functional as F
import pdb

root = "/fs/project/PAS0027/enso_precip/train"
precip = np.zeros((487, 155, 351), dtype=np.float32)
for i in range(1981, 2022):
    for j in range(1, 13):
        if i == 2021 and j == 8:
            break
        f = nc.Dataset(os.path.join(root, "PRISM_ppt_" + str(i) + str(j).zfill(2) + ".nc"))
        high_res = np.array(f["Band1"][:])
        high_res = torch.from_numpy(high_res).unsqueeze(0).unsqueeze(0).cuda()
        low_res = F.interpolate(high_res, scale_factor=[0.25, 0.25], mode="bilinear")
        low_res = low_res.cpu().numpy().astype(np.float32)
        precip[(i - 1981) * 12 + j - 1] = low_res
        # print((low_res < 0).sum())
    print("finish year " + str(i))

np.save(os.path.join(root, "precip.npy"), precip)
