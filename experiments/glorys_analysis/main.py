import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import sys
import os
sys.path.insert(0, '../../')
from lib.data.split_train_test import split_dates
from lib.data.borey.datasets import glorys
from lib.data.dataset_utils import ConcatDataset, Sampler
from lib.config.cfg import cfg
from lib.model.build_module import build_correction_model
from lib.utils.logger import Logger

logger = Logger(cfg.data.logs_path, cfg.model_type)

cfg['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_op = glorys.GlorysOperativeSalinityDataset(cfg.data.operative_folder)
ds_re = glorys.GlorysReanalysisSalinityDataset(cfg.data.reanalysis_folder, dst_grid=ds_op.src_grid, average_times=[0])

start_date, end_date = np.datetime64(cfg.data.start_date), np.datetime64(cfg.data.end_date)
months = np.arange(start_date, end_date, np.timedelta64(1, 'M'), dtype='datetime64[M]')

train_days, _, test_days = split_dates(start_date, end_date, 0.7, 0.0, 0.25)

train_sampler = Sampler(train_days, shuffle=False)
test_sampler = Sampler(test_days, shuffle=False)

train_loader = DataLoader(ConcatDataset(ds_op, ds_re), batch_size=10, num_workers=16, sampler=train_sampler)
test_loader = DataLoader(ConcatDataset(ds_op, ds_re), batch_size=1,  num_workers=0, sampler=test_sampler)

model = build_correction_model(cfg)
