import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from lib.data.borey import Grid, BarentsKaraGrid
from lib.data.borey.datasets.base import Dataset


class ShapefileSicDataset(Dataset):
    def __init__(self, path, resolution=5.0, dst_grid=None, average_times=None, name=None):
        self.resolution = resolution
        super().__init__(path, dst_grid, average_times, name)

    def _create_grid(self):
        return BarentsKaraGrid(self.resolution)

    def _process_field(self, field):
        field *= 10  # convert from ice grades to percents
        field = field[None, :, :]  # (time, lat, lon)
        return field

    def _extract_data(self, file, load_fn=np.load):
        field = load_fn(file)
        field = self._process_field(field)
        field = field[:, None, :, :]  # (time, channel, lat, lon)
        return field

    @staticmethod
    def _parse_date(file):
        date_part = file.parent.name
        date = datetime.strptime(date_part, '%Y%m%d').date()
        return date

    @property
    def _files_template(self):
        return f'*/rasterized_S_{int(self.resolution)}km.npy'


class ShapefileDriftDataset(Dataset):
    def __init__(self, path, region, dst_grid=None, average_times=None, name=None):
        self.shp_df = None
        self.region = region
        super().__init__(path, dst_grid, average_times, name)

        with open(self.path / 'grids_mapping.pkl', 'rb') as f:
            self._grids_mapping = pickle.load(f)

    def _create_dates_dict(self):
        self.csv_df = pd.read_csv(self.path / 'ice_drift.csv').dropna()
        result = {
            datetime.strptime(date_str, '%Y-%m-%d').date():
                [self.csv_df[self.csv_df['date'] == date_str]]
            for date_str in self.csv_df['date'].unique()
        }
        print(f'parsed {len(result)} dates')
        return result

    def _create_grid(self):
        with open(self.path / 'grids.pkl', 'rb') as f:
            grids = pickle.load(f)
        lat, lon = grids[self.region]
        grid = Grid(lat, lon)
        return grid

    def _extract_data(self, file, load_fn=lambda x: x):
        df_date = load_fn(file)
        data = np.full((2, *self.src_grid.shape), np.nan)  # (channels, lat, lon)
        for _, row in df_date.iterrows():
            x = row['x0']
            y = row['y0']
            region, i, j = self._grids_mapping[(x, y)]
            if region == self.region:
                drift_norm = 51.444 * row['w_drift']  # nm/hr -> cm/s
                drift_angle = np.radians(90 - row['d_drift'])  # north -> east & degrees -> radians
                drift_direction = np.array([np.cos(drift_angle), np.sin(drift_angle)])
                data[:, i, j] = drift_norm * drift_direction
        data = data[None]  # (time, channels, lat, lon)
        return data

    def _process_field(self, field):
        raise NotImplementedError

    @staticmethod
    def _parse_date(file):
        raise NotImplementedError

    @property
    def _files_template(self):
        raise NotImplementedError


class ShapefileThickDataset(Dataset):
    def __init__(self, path, resolution=5.0, dst_grid=None, average_times=None, name=None):
        self.resolution = resolution
        super().__init__(path, dst_grid, average_times, name)

    def _create_grid(self):
        return BarentsKaraGrid(self.resolution)

    def _process_field(self, field):
        field = field[None, :, :]  # (time, lat, lon)
        return field

    def _extract_data(self, file, load_fn=np.load):
        field = load_fn(file)
        field = self._process_field(field)
        field = field[:, None, :, :]  # (time, channel, lat, lon)
        return field

    @staticmethod
    def _parse_date(file):
        date_part = file.parent.name
        date = datetime.strptime(date_part, '%Y%m%d').date()
        return date

    @property
    def _files_template(self):
        return f'*/rasterized_thick_{int(self.resolution)}km.npy'
