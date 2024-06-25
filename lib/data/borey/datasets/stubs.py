import datetime

import numpy as np

from lib.data.borey.datasets.base import Dataset


class ConstantDataset(Dataset):
    def __init__(self, grid, value=0.0, name=None,
                 start_date=datetime.date(2000, 1, 1),
                 end_date=datetime.date(2030, 1, 1)):
        self.src_grid = grid
        self.start_date = start_date
        self.end_date = end_date
        super().__init__(path='', dst_grid=None, average_times=None, name=name)

        if isinstance(value, (int, float)):
            self.value = np.array([value])
        else:
            self.value = np.array(value)

    def __getitem__(self, date):
        if not (self.start_date <= date < self.end_date):
            return None
        item = np.tile(self.value[:, None, None], self.src_grid.shape)
        return item

    def _create_dates_dict(self):
        result = {
            self.start_date + datetime.timedelta(days=days): None
            for days in range((self.end_date - self.start_date).days)
        }
        return result

    def _create_grid(self):
        return self.src_grid

    def _process_field(self, field):
        raise NotImplementedError

    def _extract_data(self, file, load_fn=None):
        raise NotImplementedError

    @staticmethod
    def _parse_date(file):
        raise NotImplementedError

    @property
    def _files_template(self):
        raise NotImplementedError


class FusionDataset(Dataset):
    def __init__(self, datasets, dst_grid=None, name=None):
        self.datasets = datasets
        super().__init__(path='', dst_grid=dst_grid, average_times=slice(None), name=name)

    def _create_dates_dict(self):
        result = {
            date: [(dataset, date) for dataset in self.datasets if date in dataset.dates_dict]
            for date in set(date for dataset in self.datasets for date in dataset.dates_dict)
        }
        print(f'parsed {len(result)} dates')
        return result

    def _create_grid(self):
        return self.datasets[0].grid  # all datasets are assumed to have the same grid

    def _extract_data(self, file, load_fn=lambda x: x):
        dataset, date = load_fn(file)
        data = dataset[date][None]  # (time, channels, lat, lon)
        return data

    def _process_field(self, field):
        raise NotImplementedError

    @staticmethod
    def _parse_date(file):
        raise NotImplementedError

    @property
    def _files_template(self):
        raise NotImplementedError
