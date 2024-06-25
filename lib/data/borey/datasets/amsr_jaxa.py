from abc import abstractmethod
from datetime import datetime

import numpy as np
import pyproj
import xarray as xr

from lib.data.borey import Grid
from lib.data.borey.datasets.base import Dataset


class AmsrJaxaHsiDataset(Dataset):
    def _create_grid(self):
        x = np.linspace(-3_850_000, 3_750_000, self._grid_shape[0])
        y = np.linspace(5_850_000, -5_350_000, self._grid_shape[1])
        xx, yy = np.meshgrid(x, y)

        stereo_crs = pyproj.CRS.from_epsg(3413)
        plate_caree_crs = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(stereo_crs, plate_caree_crs)
        lat, lon = transformer.transform(xx, yy)
        grid = Grid(lat, lon)
        return grid

    def _process_field(self, field):
        field = field.astype(np.float32)
        field = field.transpose(2, 0, 1)  # (time, lat, lon)
        field[field <= -32_767.0] = np.nan
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        ds = load_fn(file)
        entry = ds.variables['Geophysical Data']
        data = self._process_field(entry.values)
        data = data[:, None, :, :]  # (time, channel, lat, lon)
        data = data * float(entry.attrs['SCALE FACTOR'])
        return data

    @staticmethod
    def _parse_date(file):
        date_part = file.name.split('_')[1]
        date = datetime.strptime(date_part, '%Y%m%d').date()
        return date

    @property
    @abstractmethod
    def _grid_shape(self):
        raise NotImplementedError


class AmsrJaxaHsi25kmDataset(AmsrJaxaHsiDataset):
    @property
    def _files_template(self):
        return 'HSI/v100/L3/**/GW1AM2_*_01D_PNM?_L3RGHSILW1100100.h5'

    @property
    def _grid_shape(self):
        return 304, 448


class AmsrJaxaHsi10kmDataset(AmsrJaxaHsiDataset):
    @property
    def _files_template(self):
        return 'HSI/v100/L3/**/GW1AM2_*_01D_PNM?_L3RGHSIHW1100100.h5'

    @property
    def _grid_shape(self):
        return 760, 1120


class AmsrJaxaSimRDataset(Dataset):
    def _create_grid(self):
        grid_path = self.path / 'SIM_R/SIM_R_latlon.dat'
        raw_grid = np.fromfile(grid_path, dtype=np.float32).reshape(2, 448, 304)
        raw_grid[raw_grid == -32768.0] = -32767.0
        lat, lon = raw_grid
        lat = self._process_field(lat)[0]
        lon = self._process_field(lon)[0]
        grid = Grid(lat, lon)
        return grid

    @property
    def _files_template(self):
        return 'SIM_R/v101/L3/**/GW1AM2_*_01D_PNMB_L3RGSIMLR1101101.h5'

    # todo dont support python < 3.11
    # def _process_field(self, field):
    #     region_mask = (field != -32_767)
    #     i_range = np.arange(region_mask.shape[0])[region_mask.any(axis=1)]
    #     i_slice = slice(i_range.min(), i_range.max() + 1)
    #     j_range = np.arange(region_mask.shape[1])[region_mask.any(axis=0)]
    #     j_slice = slice(j_range.min(), j_range.max() + 1)
    #     region = (i_slice, j_slice)
    #     assert region_mask[*region].all(), 'masked region is not rectangular'
    #     field = field[*region]
    #     field = field[::2, ::2]
    #
    #     field = field.astype(np.float32)
    #     field[field == -32_768.0] = np.nan
    #     field = field[None, :, :]  # (time, lat, lon)
    #     return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        ds = load_fn(file)
        entry = ds.variables['Geophysical Data EN']
        ufield, vfield = entry.values.transpose(2, 0, 1)
        data = np.stack([
            self._process_field(ufield),
            self._process_field(vfield),
        ], axis=1)  # (time, channel, lat, lon)
        data = data * float(entry.attrs['SCALE FACTOR'])
        return data

    @staticmethod
    def _parse_date(file):
        date_part = file.name.split('_')[1]
        date = datetime.strptime(date_part, '%Y%m%d').date()
        return date


class AmsrJaxaSimYDataset(Dataset):
    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        ds = xr.open_dataset(grid_path)
        lat = ds.variables['lat'].values
        lon = ds.variables['lon'].values
        grid = Grid(lat, lon)
        return grid

    @property
    def _files_template(self):
        return 'SIM_Y/v100/L3/**/GW1AM2_*_01D_PNMB_L3RGSIMPY1100100.h5'

    def _process_field(self, field):
        field = field.astype(np.float32)
        field[field == -32_768.0] = np.nan
        field = field[None, :, :]  # (time, lat, lon)
        return field

    def _extract_data(self, file, load_fn=xr.open_dataset):
        ds = load_fn(file)
        ufield = ds.variables['ve'].values
        vfield = ds.variables['vn'].values
        data = np.stack([
            self._process_field(ufield),
            self._process_field(vfield),
        ], axis=1)  # (time, channel, lat, lon)
        return data

    @staticmethod
    def _parse_date(file):
        date_part = file.name.split('_')[1]
        date = datetime.strptime(date_part, '%Y%m%d').date()
        return date
