import os
import numpy as np
import netCDF4 as nc

import torch
import torch.nn.functional as F
import pdb

root = "/fs/project/PAS0027/enso_precip/train"

sst = nc.Dataset(os.path.join(root, 'AnomSST_1979-2020.nc'))
sst_high = sst.variables['ASST'][:].data
sst_high = torch.from_numpy(sst_high).unsqueeze(0).unsqueeze(0).cuda()
sst_low = F.interpolate(sst_high, size=[507, 128, 256], mode="trilinear")
sst_low = sst_low.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
sst_high_mask = sst.variables['ASST'][:].mask.astype(np.float32)
sst_high_mask = torch.from_numpy(sst_high_mask).unsqueeze(0).unsqueeze(0).cuda()
sst_low_mask = F.interpolate(sst_high_mask, size=[507, 128, 256], mode="trilinear")
sst_low_mask = sst_low_mask.squeeze(0).squeeze(0).cpu().numpy() > 0
sst_min = sst_low[~sst_low_mask].min()
sst_max = sst_low[~sst_low_mask].max()
sst_low[sst_low_mask] = (sst_min + sst_max) / 2.
np.save(os.path.join(root, 'AnomSST_1979-2020'), sst_low)

precip = nc.Dataset(os.path.join(root, 'AnomPrecip_1979-2020.nc'))
precip_high = precip.variables['ATP'][:].data
precip_high = torch.from_numpy(precip_high).unsqueeze(0).unsqueeze(0).cuda()
precip_low = F.interpolate(precip_high, size=[507, 128, 256], mode="trilinear")
precip_low = precip_low.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
np.save(os.path.join(root, 'AnomPrecip_1979-2020'), precip_low)