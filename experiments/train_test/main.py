import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import sys
import os
sys.path.insert(0, '../../')
from lib.data.split_train_test import split_dates
from lib.data.borey.datasets import glorys
from lib.data.dataset_utils import Sampler
from lib.data.scaler import StandardScaler
from lib.config.cfg import cfg
from lib.model.build_module import build_correction_model, build_dataset
from lib.model.losses import MaskedLoss
from lib.utils.logger import Logger
from lib.pipeline.train import train
from lib.pipeline.test import test


print(f'Training {cfg.model_type} model')
logger = Logger(cfg.data.logs_path, cfg.model_type)
logger.save_configuration_yaml(cfg) if cfg.run_config.run_mode == 'train' else None
cfg['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_op = glorys.GlorysOperativeSalinityDataset(cfg.data.operative_folder)
ds_re = glorys.GlorysReanalysisSalinityDataset(cfg.data.reanalysis_folder, dst_grid=ds_op.src_grid, average_times=[0])
start_date, end_date = np.datetime64(cfg.data.start_date), np.datetime64(cfg.data.end_date)
months = np.arange(start_date, end_date, np.timedelta64(1, 'M'), dtype='datetime64[M]')

train_days, val_days, test_days = split_dates(start_date, end_date, 0.70, 0.05, 0.25)

train_sampler = Sampler(train_days, shuffle=True)
val_sampler = Sampler(val_days, shuffle=False)
test_sampler = Sampler(test_days, shuffle=False)
glorys_dataset = build_dataset(ds_op, ds_re, cfg=cfg)
train_loader = DataLoader(glorys_dataset, batch_size=cfg.train.batch_size, num_workers=16, sampler=train_sampler)
val_loader = DataLoader(glorys_dataset, batch_size=cfg.train.batch_size,  num_workers=10, sampler=val_sampler)
test_loader = DataLoader(glorys_dataset, batch_size=1,  num_workers=0, sampler=test_sampler)

model = build_correction_model(cfg)

loss_mask = torch.from_numpy(glorys_dataset.share_mask).to(cfg.device)
criterion = MaskedLoss(torch.nn.MSELoss, mask=loss_mask).to(cfg.device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=1e-5)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

scaler = StandardScaler()
scaler.apply_scaler_channel_params(torch.load(cfg.data.glorys_mean_path), torch.load(cfg.data.glorys_std_path))
if cfg.run_config.run_mode == 'train':
    print(f'Start training run No {logger.experiment_number} ')
    best_epoch, model = train(train_loader, val_loader, model, optimizer, scaler, criterion, scheduler, logger, cfg)
else:
    best_epoch = cfg.test_config.best_epoch
    print(f'Skip training, loading {best_epoch} epoch from {cfg.test_config.run_id} run')
state_dict = torch.load(os.path.join(logger.model_save_dir, f'model_{best_epoch}.pth'))
model.load_state_dict(state_dict)
test(model, scaler, test_loader, logger, cfg)
