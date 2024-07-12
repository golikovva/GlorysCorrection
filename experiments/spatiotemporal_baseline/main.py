import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import os
import esmpy

sys.path.insert(0, '../../')
from lib.data.split_train_test import split_dates, split_dates_by_dates
from lib.data.borey.datasets import glorys
from lib.data.dataset_utils import ConcatI2IDataset, Sampler, numpy_collate
from lib.config.cfg import cfg
from lib.model.corr_accumulator import CorrAccumulator


mg = esmpy.Manager()

mean_period = 'day'

ds_op = glorys.GlorysOperativeSalinityDataset(cfg.data.operative_folder)
ds_re = glorys.GlorysReanalysisSalinityDataset(cfg.data.reanalysis_folder, dst_grid=ds_op.src_grid, average_times=[0])

start_date, end_date = np.datetime64(cfg.data.start_date), np.datetime64(cfg.data.end_date)
val_end = train_end = np.datetime64(cfg.data.end_date) - np.timedelta64(365, 'D')

# train_days, val_days, test_days = split_dates(start_date, end_date, 1.0, 0., 0.)
train_days, val_days, test_days = split_dates_by_dates(start_date, end_date, train_end, val_end)
print(len(test_days), 'test days len')
print(len(train_days), 'train days len')
ds = ConcatI2IDataset(ds_op, ds_re)

train_sampler = Sampler(train_days, shuffle=False)
train_loader = DataLoader(ds, batch_size=10, num_workers=10, sampler=train_sampler,
                          collate_fn=numpy_collate)

model = CorrAccumulator(start_date, mean_period)

loss_mask = ds.share_mask


for operative, reanalysis, i in (pbar := tqdm(train_loader)):
    i = np.squeeze(i)
    corr = reanalysis - operative
    model.accumulate_corr(corr, i)
    if i[0] % 16 == 0:
        print(corr.max(axis=(1, 2, 3)))
        print(model.period_counts)
        print(model.period_means.mean(axis=(1, 2, 3)))

os.makedirs(os.path.join(cfg.data.logs_path, 'spatiotemporal_baseline'), exist_ok=True)
model.save_correction_fields(os.path.join(cfg.data.logs_path, 'spatiotemporal_baseline',
                                          f'{mean_period}_correction_fields.npy'))
