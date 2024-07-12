import scipy
import numpy as np
import sys
import os
from time import time

sys.path.insert(0, '../..')
from lib.model.conv_4d import conv4d_numba
from lib.config.cfg import cfg

start = time()
correction_field_file = 'day_correction_fields.npy'
hyperparameters = {'k_size': (15, 3, 7, 7),
                   'sigmas': (9, 0.5, 2, 2)}


def get_gauss_kernel(k, sigma):
    n = np.zeros(k)
    center = tuple(i // 2 for i in k)
    n[center] = 1
    kk = scipy.ndimage.gaussian_filter(n, sigma=sigma)
    return kk


kernel = get_gauss_kernel(hyperparameters['k_size'], hyperparameters['sigmas'])
field = np.load(os.path.join(cfg.data.logs_path, 'spatiotemporal_baseline', correction_field_file))

result = conv4d_numba(field, kernel)

print(f'Time spent in total: {time() - start} sec.')
save_result = True
if save_result:
    np.save(os.path.join(cfg.data.logs_path, 'spatiotemporal_baseline',
                         correction_field_file[:-4] + '_conv_meaned.npy'), result)
