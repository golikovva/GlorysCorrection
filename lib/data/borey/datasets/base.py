import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr

from lib.data.borey.interpolator import Interpolator
import gc

class Dataset(ABC):
    def __init__(self, path, dst_grid=None, average_times=None, name=None):
        super().__init__()
        self.path = Path(path)
        self.dst_grid = dst_grid
        self.average_times = average_times
        self.name = name
        if self.dst_grid is not None and self.average_times is None:
            raise ValueError('average_times must be specified when dst_grid is '
                             '(4d interpolation is not currently supported)')

        if self.name is not None:
            print(f'initializing {self.name} dataset')
        self.dates_dict = self._create_dates_dict()
        self.src_grid = self._create_grid()
        self.interpolator = self._create_interpolator()

    def _create_dates_dict(self):
        result = {}
        files = sorted(self.path.glob(self._files_template))
        for file in files:
            try:
                date = self._parse_date(file)
            except ValueError:
                print(f'skipping {file}')
                continue
            if date in result:
                result[date].append(file)
            else:
                result[date] = [file]
        print(f'parsed {len(result)} dates')
        return result

    def _create_interpolator(self):
        if self.dst_grid is None:
            return None
        interpolator = Interpolator(self.src_grid, self.dst_grid)
        print('initializing interpolator')
        interpolator.initialize()
        return interpolator

    def __getitem__(self, date):
        if date not in self.dates_dict:
            return None
        result = []
        files = self.dates_dict[date]
        for file in sorted(files):
            data = self._extract_data(file)  # (time, channel, lat, lon)
            assert len(data.shape) == 4, f'expected 4D data, got {data.shape}'
            result.append(data)
        result = np.concatenate(result, axis=0)  # assuming different files are for different times
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            result = np.nanmean(result[self.average_times], axis=0)  # if self.average_times is None, nothing happens
        if self.interpolator is not None:
            result = np.stack([self.interpolator(field) for field in result])
        return result

    def __len__(self):
        return len(self.dates_dict)

    def __lt__(self, other):
        if not isinstance(other, Dataset):
            # Don't attempt to compare against unrelated types
            return NotImplemented
        return self.name < other.name

    @property
    def grid(self):
        if self.dst_grid is not None:
            return self.dst_grid
        return self.src_grid

    @abstractmethod
    def _create_grid(self):
        raise NotImplementedError

    @abstractmethod
    def _process_field(self, field):
        raise NotImplementedError

    @abstractmethod
    def _extract_data(self, file, load_fn=None):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _parse_date(file):
        raise NotImplementedError

    @property
    @abstractmethod
    def _files_template(self):
        raise NotImplementedError


class ModelSicDataset(Dataset):
    def _process_field(self, field):
        field = np.nan_to_num(field, nan=0.0)  # (time, lat, lon)
        field[:, self.src_grid.land_mask()] = np.nan
        field = field * 100  # fraction -> percentage
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        ds = load_fn(file)
        field = ds.variables[self._sic_variable].values
        data = self._process_field(field)
        data = data[:, None, :, :]  # (time, channel, lat, lon)
        return data

    @property
    @abstractmethod
    def _sic_variable(self):
        raise NotImplementedError


class ModelDriftDataset(Dataset):
    def _process_field(self, field):
        field = field * 100  # m/s -> cm/s
        return field  # (time, lat, lon)

    def _extract_data(self, file, load_fn=xr.open_dataset):
        ds = load_fn(file)
        ufield = ds.variables[self._udrift_variable].values
        vfield = ds.variables[self._vdrift_variable].values
        data = np.stack([
            self._process_field(ufield),
            self._process_field(vfield),
        ], axis=1)  # (time, channel, lat, lon)
        return data

    @property
    @abstractmethod
    def _udrift_variable(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _vdrift_variable(self):
        raise NotImplementedError


class ModelThickDataset(Dataset):
    def _process_field(self, field):
        field = np.nan_to_num(field, nan=0.0)  # (time, lat, lon)
        field[:, self.src_grid.land_mask()] = np.nan
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        ds = load_fn(file)
        field = ds.variables[self._thick_variable].values
        data = self._process_field(field)
        data = data[:, None, :, :]  # (time, channel, lat, lon)
        return data

    @property
    @abstractmethod
    def _thick_variable(self):
        raise NotImplementedError


class ModelSalinityDataset(Dataset):
    def _process_field(self, field):
        return field  # (time, channel=depth, lat, lon)

    def _extract_data(self, file, load_fn=xr.open_dataset):
        with load_fn(file, cache=False) as ds:
            # ds = load_fn(file)
            field = ds.variables[self._salinity_variable].values
            data = self._process_field(field)
        # ds.close()
        return data  # (time, channel=depth, lat, lon)

    @property
    @abstractmethod
    def _salinity_variable(self):
        raise NotImplementedError


class ModelTemperatureDataset(Dataset):
    def _process_field(self, field):
        return field  # (time, channel=depth, lat, lon)

    def _extract_data(self, file, load_fn=xr.open_dataset):
        ds = load_fn(file)
        field = ds.variables[self._temp_variable].values
        data = self._process_field(field)
        return data  # (time, channel=depth, lat, lon)

    @property
    @abstractmethod
    def _temp_variable(self):
        raise NotImplementedError


class ModelEastCurrentDataset(Dataset):
    def _process_field(self, field):
        return field  # (time, channel=depth, lat, lon)

    def _extract_data(self, file, load_fn=xr.open_dataset):
        ds = load_fn(file)
        field = ds.variables[self._east_cur_variable].values
        data = self._process_field(field)
        return data  # (time, channel=depth, lat, lon)

    @property
    @abstractmethod
    def _east_cur_variable(self):
        raise NotImplementedError


class ModelNorthCurrentDataset(Dataset):
    def _process_field(self, field):
        return field  # (time, channel=depth, lat, lon)

    def _extract_data(self, file, load_fn=xr.open_dataset):
        ds = load_fn(file)
        field = ds.variables[self._north_cur_variable].values
        data = self._process_field(field)
        return data  # (time, channel=depth, lat, lon)

    @property
    @abstractmethod
    def _north_cur_variable(self):
        raise NotImplementedError


class ModelCurrentVelocityDataset(Dataset):
    def _process_field(self, field):
        return field  # (time, channel=depth, lat, lon)

    def _extract_data(self, file, load_fn=xr.open_dataset):
        ds = load_fn(file)
        ufield = ds.variables[self._east_cur_variable].values
        vfield = ds.variables[self._north_cur_variable].values
        data = np.stack([
            self._process_field(ufield),
            self._process_field(vfield),
        ], axis=0)  # (components, time, channel=depth, lat, lon)
        data = np.linalg.norm(data, axis=0)  # (time, channel=depth, lat, lon)
        return data

    @property
    @abstractmethod
    def _east_cur_variable(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _north_cur_variable(self):
        raise NotImplementedError
