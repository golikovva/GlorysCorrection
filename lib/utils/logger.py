import logging
import os
import re

import numpy as np
import torch
import pickle

from lib.config.cfg import cfg


class Logger:
    def __init__(self, base_log_dir=None, folder_name=None):
        if base_log_dir is None:
            base_log_dir = '/home/logs'
        if folder_name is None:
            folder_name = 'unknown'

        if not os.path.exists(os.path.join(base_log_dir, folder_name)):
            os.makedirs(os.path.join(base_log_dir, folder_name))
        self.folder_path = os.path.join(base_log_dir, folder_name)

        if cfg.run_config.run_mode == 'test':
            self.experiment_number = cfg.test_config.run_id
        else:
            self.experiment_number = self.get_experiment_number()

        self.save_dir = os.path.join(self.folder_path, f'misc_{self.experiment_number}')
        self.model_save_dir = os.path.join(self.save_dir, 'models')
        self.log_dir = os.path.join(self.save_dir, 'logs')

        os.makedirs(self.log_dir) if not os.path.exists(self.log_dir) else None
        os.makedirs(self.model_save_dir) if not os.path.exists(self.model_save_dir) else None

        self.logger = self.create_logger()
        self.logger.info(f"Testing the custom logger for module {__name__}...")

        self.train_loss = []
        self.loss_evolution = []
        self.best_epoch = -1
        self.mse = 0
        self.iters_counted = 0

    def create_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(f"{os.path.join(self.log_dir, __name__)}.log", mode='a')
        formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_experiment_number(self):
        numbers = set()
        for directory in os.listdir(self.folder_path):
            if re.match(r'(misc)_\d+$', directory):
                numbers.add(int(directory.split('_')[-1]))
        return max(numbers) + 1 if len(numbers) else 1

    def accumulate_stat(self, mse):
        self.mse += float(mse)
        self.iters_counted += 1

    def reset_stat(self):
        self.mse = 0
        self.iters_counted = 0

    def print_stat_readable(self, epoch=None, reset=True):
        if epoch:
            self.logger.info(f"Validation epoch {epoch} successful with val loss:")
        else:
            self.logger.info(f"Validation successful with val loss:")
        if self.mse > 0:
            mse = round(self.mse / self.iters_counted, 5)
            self.logger.info(f"    MSE + deltaMSE: {mse}")
            self.loss_evolution.append(mse)
        if reset:
            self.reset_stat()

    def get_stat(self):
        mse = self.mse / self.iters_counted
        return mse

    def save_configuration(self):
        cfg.run_id = self.experiment_number
        cfg.best_epoch = self.best_epoch
        measurements_file = os.path.join(self.save_dir, 'configuration.pkl')
        with open(measurements_file, 'wb') as f:
            pickle.dump(cfg, f)

    def save_configuration_yaml(self, cfg_dict):
        import yaml
        with open(os.path.join(self.log_dir, f'run_config_{self.experiment_number}.yaml'), 'w') as outfile:
            yaml.dump(cfg_dict, outfile, default_flow_style=False)

    def load_configuration(self, run_id):  # todo нет записи параметров по типу use_sp_t_e
        if os.path.exists(os.path.join(self.folder_path, f'misc_{run_id}')):
            self.experiment_number = run_id
            self.save_dir = os.path.join(self.folder_path, f'misc_{self.experiment_number}')
            self.model_save_dir = os.path.join(self.save_dir, 'models')
            self.log_dir = os.path.join(self.save_dir, 'logs')
            with open(os.path.join(self.save_dir, 'configuration.pkl'), 'rb') as f:
                loaded_cfg = pickle.load(f)
                cfg.run_config.best_epoch = loaded_cfg.run_config.best_epoch
                cfg.run_config.use_spatiotemporal_encoding = loaded_cfg.run_config.use_spatiotemporal_encoding

    def save_model(self, model_state_dict, epoch):
        loss = self.loss_evolution
        if len(loss) > 0 and loss.index(min(loss)) == len(loss) - 1:
            torch.save(model_state_dict, os.path.join(self.model_save_dir, f'model_{epoch}.pth'))
            old_model_path = os.path.join(self.model_save_dir, f'model_{self.best_epoch}.pth')
            if os.path.exists(old_model_path):
                os.remove(old_model_path)
            self.best_epoch = len(loss) - 1
        np.save(os.path.join(self.log_dir, 'val_loss'), np.stack(self.loss_evolution))
        np.save(os.path.join(self.log_dir, 'train_loss'), np.stack(self.train_loss))
        return self.best_epoch
