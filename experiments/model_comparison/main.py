from yaml import load, SafeLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
sys.path.insert(0, '../../')
from lib.utils.plot_utils import draw_so_means
from lib.config.cfg import cfg
from lib.data.split_train_test import split_dates_by_dates

start_date, end_date = np.datetime64(cfg.data.start_date), np.datetime64(cfg.data.end_date)
val_end = np.datetime64(cfg.data.end_date) - np.timedelta64(365, 'D')
train_end = val_end - np.timedelta64(60, 'D')
_, _, test_days = split_dates_by_dates(start_date, end_date, train_end, val_end)

with open('models.yaml', 'r') as namelist_file:
    namelist = load(namelist_file, Loader=SafeLoader)

res = {}
for model_name in namelist:
    res_path = namelist[model_name]
    if res_path.endswith('.npy'):
        res[model_name] = np.load(res_path)
    else:
        res[model_name] = torch.load(res_path).numpy()

maes = {}
for model_name in namelist:
    maes[model_name] = np.abs(res[model_name] - res['reanalysis']).mean().round(decimals=3)
mses = {}
for model_name in namelist:
    mses[model_name] = np.sqrt(np.square(res[model_name] - res['reanalysis']).mean()).round(decimals=3)

fig = draw_so_means(res.values(), res.keys(), test_days)

# data = [[ 66386, 174296,  75131, 577908,  32015]]
#
columns = ('MAE', 'RMSE')
rows = list(maes.keys())
data = list(zip(maes.values(), mses.values()))
#
the_table = plt.table(cellText=data,
                      rowLabels=rows,
                      colLabels=columns,
                      rowLoc='left',
                      bbox=[1.1, 0, 0.1, 1])

print(maes)
print(mses)
fig.savefig(os.path.join('/home/logs/model_comparison', f'so_means'))
