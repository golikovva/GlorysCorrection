from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import os

sys.path.insert(0, '../../')
from lib.data.split_train_test import split_dates, split_dates_by_dates
from lib.data.borey.datasets import glorys
from lib.data.dataset_utils import ConcatI2IDataset, Sampler, numpy_collate
from lib.config.cfg import cfg
from lib.model.corr_accumulator import AccumCorrector

mean_period = 'day'

ds_op = glorys.GlorysOperativeSalinityDataset(cfg.data.operative_folder)
ds_re = glorys.GlorysReanalysisSalinityDataset(cfg.data.reanalysis_folder, dst_grid=ds_op.src_grid, average_times=[0])

start_date, end_date = np.datetime64(cfg.data.start_date), np.datetime64(cfg.data.end_date)
val_end = train_end = np.datetime64(cfg.data.end_date) - np.timedelta64(365, 'D')

_, _, test_days = split_dates_by_dates(start_date, end_date, train_end, val_end)
print(len(test_days), 'test days len')
ds = ConcatI2IDataset(ds_op, ds_re)

train_sampler = Sampler(test_days, shuffle=False)
train_loader = DataLoader(ds, batch_size=1, num_workers=10, sampler=train_sampler,
                          collate_fn=numpy_collate)

loss_mask = ds.share_mask
print(loss_mask.shape)

path1 = os.path.join(cfg.data.output_path, 'day_so_correction_fields_nearestn_meaned.npy')
path2 = os.path.join(cfg.data.output_path, 'day_so_correction_fields_conv_meaned.npy')

model_nearest = AccumCorrector(path1)
model_convmean = AccumCorrector(path2)


losses_to_cat = ['operative', 'reanalysis', 'nearest', 'conv_mean', 'unet2d', 'unet3dl']

ops, res, corrs1, corrs2 = [], [], [], []
for operative, reanalysis, i in (pbar := tqdm(train_loader)):
    i = np.squeeze(i)
    print(i)
    dates = np.datetime64(cfg.data.start_date) + i * np.timedelta64(1, 'D')
    days_of_year = (dates - dates.astype('datetime64[Y]')).astype(int)
    print(days_of_year)
    print(operative.shape)
    ops.append(operative[:, loss_mask].mean())
    res.append(reanalysis[:, loss_mask].mean())
    corrs1.append(model_nearest(operative, days_of_year)[:, loss_mask].mean())
    corrs2.append(model_convmean(operative, days_of_year)[:, loss_mask].mean())

np.save(os.path.join(cfg.data.logs_path, 'spatiotemporal_baseline', 'ops'), np.stack(ops))
np.save(os.path.join(cfg.data.logs_path, 'spatiotemporal_baseline', 'res'), np.stack(res))
np.save(os.path.join(cfg.data.logs_path, 'spatiotemporal_baseline', 'model_nearest'), np.stack(corrs1))
np.save(os.path.join(cfg.data.logs_path, 'spatiotemporal_baseline', 'model_convmean'), np.stack(corrs2))
