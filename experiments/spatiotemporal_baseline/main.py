import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import os
import esmpy

sys.path.insert(0, '../../')
from lib.data.split_train_test import split_dates
from lib.data.borey.datasets import glorys
from lib.data.dataset_utils import ConcatDataset, Sampler, numpy_collate
from lib.config.cfg import cfg
from lib.model.corr_accumulator import CorrAccumulator


mg = esmpy.Manager()

mean_period = 'day'

ds_op = glorys.GlorysOperativeSalinityDataset(cfg.data.operative_folder)
ds_re = glorys.GlorysReanalysisSalinityDataset(cfg.data.reanalysis_folder, dst_grid=ds_op.src_grid, average_times=[0])

start_date, end_date = np.datetime64(cfg.data.start_date), np.datetime64(cfg.data.end_date)
months = np.arange(start_date, end_date, np.timedelta64(1, 'M'), dtype='datetime64[M]')

train_days, val_days, test_days = split_dates(start_date, end_date, 1.0, 0., 0.)
ds = ConcatDataset(ds_op, ds_re)

train_sampler = Sampler(train_days, shuffle=False)
train_loader = DataLoader(ds, batch_size=10, num_workers=10, sampler=train_sampler,
                          collate_fn=numpy_collate)

model = CorrAccumulator(start_date, mean_period)

loss_mask = ds.share_mask

if cfg.run_config.run_mode == 'train':
    for operative, reanalysis, i in (pbar := tqdm(train_loader)):
        i = np.squeeze(i)
        corr = reanalysis - operative
        model.accumulate_corr(corr, i)
        if i[0] % 16 == 0:
            print(corr.max(axis=(1, 2, 3)))
            print(model.period_counts)
            print(model.period_means.mean(axis=(1, 2, 3)))

os.makedirs(os.path.join(cfg.data.logs_path, 'spatiotempor  al_baseline'), exist_ok=True)
model.save_correction_fields(os.path.join(cfg.data.logs_path, 'spatiotemporal_baseline',
                                          f'{mean_period}_correction_fields_1.npy'))
