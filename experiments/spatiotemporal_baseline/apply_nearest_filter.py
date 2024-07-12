import torch
import numpy as np
import sys
import os
from time import time
from tqdm import tqdm

sys.path.insert(0, '../..')
from lib.data.borey.datasets import glorys
from lib.data.dataset_utils import ConcatI2IDataset
from lib.utils.interpolation import InvDistTree
from lib.config.cfg import cfg

start = time()
correction_field_file = 'day_correction_fields.npy'
hyperparameters = {'num_spatial_neighbours': 20,
                   'num_temporal_neighbours': 7,
                   'spatial_variance': 0.1,
                   'time_variance': 25}
print(hyperparameters)

ds_op = glorys.GlorysOperativeSalinityDataset(cfg.data.operative_folder)
ds_re = glorys.GlorysReanalysisSalinityDataset(cfg.data.reanalysis_folder, dst_grid=ds_op.src_grid, average_times=[0])
ds = ConcatI2IDataset(ds_op, ds_re)

lon, lat = ds_op.src_grid.grid.coords[0][0], ds_op.src_grid.grid.coords[0][1]
coords = np.stack([lon.flatten(), lat.flatten()]).T

print('Start calculating spatiotemporal weights')
interp = InvDistTree(x=coords, q=coords, n_near=hyperparameters['num_spatial_neighbours'],
                     sigma_squared=hyperparameters['spatial_variance'])
time_dist = np.abs(np.arange(-hyperparameters['num_temporal_neighbours'],
                             hyperparameters['num_temporal_neighbours'] + 0.1)) + 1
time_coefs = interp.calc_dist_coefs(time_dist, sigma_squared=hyperparameters['time_variance'])
spatial_coefs = interp.weights
raw_weights = torch.einsum('ab,c->abc', spatial_coefs, time_coefs)

print('Start applying mask')
mask = torch.from_numpy(ds.share_mask).flatten(-2, -1)
neighbours = interp.ix
neighbour_mask = mask[:, neighbours]
neighbour_mask[torch.where(~neighbour_mask[:, :, 0])] = False
masked_weights = torch.einsum('zNn,Nnt->zNnt', neighbour_mask, raw_weights)

print('Start calculating resulting correction')
field = torch.from_numpy(np.load(os.path.join(cfg.data.logs_path, 'spatiotemporal_baseline', correction_field_file)))
field = field.flatten(-2, -1)

res = torch.zeros([366, 34, 104437])
for t in tqdm(range(366)):
    slices = np.arange(t - hyperparameters['num_temporal_neighbours'],
                       t + hyperparameters['num_temporal_neighbours'] + 0.1).astype(int) % 366
    temp_res = (field[slices][:, :, neighbours] * torch.permute(masked_weights, (3, 0, 1, 2))).sum([0, -1])
    res[t] = temp_res

print(res.shape)
print(f'Time spent in total: {time() - start} sec.')
save_result = True
if save_result:
    np.save(os.path.join(cfg.data.logs_path, 'spatiotemporal_baseline',
                         correction_field_file[:-4] + '_nearestn_meaned.npy'), res.numpy())
