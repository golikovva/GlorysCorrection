import sys
from random import randint

sys.path.insert(0, '../')
import torch
import os
from lib.utils import plot_utils
from lib.model.losses import RMSELoss, MaskedLoss
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd



def test(model, scaler, dataloader, logger, cfg, save_losses=False):
    for channel in ['stats', 'samples']:
        os.makedirs(os.path.join(logger.save_dir, 'plots', channel), exist_ok=True)
    with torch.no_grad():
        model.eval()
        print(len(dataloader), len(dataloader))
        days = dataloader.sampler.days.astype('datetime64[D]')
        losses_to_cat = ['mean_orig_mae', 'mean_corr_mae',
                         # 'level_orig_mae', 'level_corr_mae',
                         'mean_orig_rmse', 'mean_corr_rmse',
                         # 'level_orig_rmse', 'level_corr_rmse',
                         'mean_orig_so', 'mean_corr_so', 'mean_target_so']
        acc = LossesAccumulator(names=losses_to_cat)
        loss_mask = torch.from_numpy(dataloader.dataset.share_mask).to(cfg.device)
        mae = MaskedLoss(torch.nn.L1Loss, mask=loss_mask).to(cfg.device)
        rmse = MaskedLoss(RMSELoss, mask=loss_mask).to(cfg.device)
        losses = {'mae': mae, 'rmse': rmse}
        for j, (data, label, i) in (pbar := tqdm(enumerate(dataloader), total=len(days))):
            data = data.type(torch.float).to(cfg.device)
            label = label.type(torch.float).to(cfg.device)
            i = i.item()

            output = model(scaler.transform(data, dims=1))
            output = scaler.inverse_transform(output, means=scaler.means[0], stds=scaler.stddevs[0], dims=1)
            data = data[:, 0]

            for name in losses:
                # level_orig_loss = losses[name](data, label).mean([0, 2, 3])
                # level_corr_loss = losses[name](output, label).mean([0, 2, 3])
                mean_orig_loss = losses[name](data, label).mean().item()
                mean_corr_loss = losses[name](output, label).mean().item()
                acc.cat_accumulate_losses(names=[f'mean_orig_{name}', f'mean_corr_{name}',],
                                                 # f'level_orig_{name}', f'level_corr_{name}'],
                                          losses=[mean_orig_loss, mean_corr_loss,])
                                                  # level_orig_loss, level_corr_loss])

            mask_sum = loss_mask.sum()
            mean_orig_so = ((data*loss_mask).sum()/mask_sum).item()
            mean_corr_so = ((output*loss_mask).sum()/mask_sum).item()
            mean_target_so = ((label*loss_mask).sum()/mask_sum).item()

            acc.cat_accumulate_losses(names=['mean_orig_so', 'mean_corr_so', 'mean_target_so'],
                                      losses=[mean_orig_so, mean_corr_so, mean_target_so])
            if cfg.test_config.draw_plots:
                if i % 16 == 0:
                    era_metric = _metric(mean_orig_loss, mean_corr_loss)
                    simple_surface_plot = plot_utils.draw_simple_plots(torch.squeeze(data),
                                                                       torch.squeeze(output),
                                                                       torch.squeeze(label), 0,
                                                                       mean_orig_loss,
                                                                       mean_corr_loss,
                                                                       era_metric,
                                                                       date=days[j], mask=loss_mask)
                    plt.savefig(os.path.join(logger.save_dir, 'plots', 'samples', f'plot_{i}'))
                plt.close('all')
        acc.cat_losses(losses_to_cat)
        orig, corr, target = acc.data['mean_orig_so'], acc.data['mean_corr_so'], acc.data['mean_target_so']

        times = dataloader.sampler.days.astype('datetime64[D]')
        plot_utils.draw_so_means([orig, corr, target], ['orig', 'corr', 'target'], times)
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'stats', f'so_means'))
        loss = acc.data['mean_corr_mae'].mean()
        acc.save_data(os.path.join(logger.save_dir))
    return loss


def _metric(orig, corr):
    return (orig - corr) / orig


class LossesAccumulator:
    def __init__(self, names):
        self.data = {names[i]: [] for i in range(len(names))}

    def cat_accumulate_losses(self, names, losses):
        for i, name in enumerate(names):
            if type(losses[i]) is not torch.Tensor:
                losses[i] = torch.tensor([losses[i]])
            self.data[names[i]].append(losses[i].cpu())

    def sum_accumulate_losses(self, names, losses):
        for i in range(len(names)):
            if len(self.data[names[i]]) == 0:
                self.data[names[i]] = losses[i].cpu()
            else:
                self.data[names[i]] += losses[i].cpu()

    def cat_losses(self, names):
        for name in names:
            self.data[name] = torch.cat(self.data[name])

    def save_data(self, dir_path):
        for name in self.data.keys():
            torch.save(self.data[name], os.path.join(dir_path, f'{name}'))
