import warnings
from datetime import datetime

import numpy as np
import xarray as xr

from lib.data.borey.grid import Grid
from lib.data.borey.datasets.base import (
    Dataset,
    ModelCurrentVelocityDataset,
    ModelDriftDataset,
    ModelEastCurrentDataset,
    ModelNorthCurrentDataset,
    ModelSalinityDataset,
    ModelSicDataset,
    ModelTemperatureDataset,
    ModelThickDataset,
)


class GlorysDataset(Dataset):
    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        with xr.open_dataset(grid_path, cache=False) as ds:

            lat_vec = ds.coords['latitude'].values
            lon_vec = ds.coords['longitude'].values

            lat = np.tile(lat_vec, (lon_vec.size, 1)).T
            lon = np.tile(lon_vec, (lat_vec.size, 1))

            grid = Grid(lat, lon)
        return grid


class GlorysOperativeDataset(GlorysDataset):
    @property
    def _files_template(self):
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy_anfc_0.083deg_P1D-m.nc'

    @staticmethod
    def _parse_date(file):
        date_part = file.name.split('_')[1]
        date = datetime.strptime(date_part, '%Y-%m-%d').date()
        return date


class GlorysReanalysisDataset(GlorysDataset):
    def _create_dates_dict(self):
        result = {}
        for file in self.path.glob(self._files_template):
            # ds = xr.open_dataset(file)
            with xr.open_dataset(file, cache=False) as ds:
                ds_dates_dict = {
                    dt.astype('M8[D]').astype('O'): ds.sel(time=[dt])
                    for dt in ds.coords['time'].values
                }
                assert not (result.keys() & ds_dates_dict.keys()), 'files have overlapping dates'
                result.update(ds_dates_dict)
        print(f'parsed {len(result)} dates')
        return result

    def __getitem__(self, date):
        if date not in self.dates_dict:
            return None
        ds = self.dates_dict[date]
        item = self._extract_data(ds, load_fn=lambda x, cache: x)  # (time, channel, lat, lon)
        assert len(item.shape) == 4, f'expected 4D data, got {item.shape}'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            item = np.nanmean(item[self.average_times], axis=0)  # if self.times is None, nothing happens
        if self.interpolator is not None:
            item = np.stack([self.interpolator(field) for field in item])
        return item

    @property
    def _files_template(self):
        return 'cmems_mod_glo_phy_my*_0.083deg_P1D-m_*.nc'

    @staticmethod
    def _parse_date(file):
        raise NotImplementedError


class GlorysSicDataset(ModelSicDataset):
    @property
    def _sic_variable(self):
        return 'siconc'


class GlorysDriftDataset(ModelDriftDataset):
    def _extract_data(self, file, load_fn=xr.open_dataset):
        data = super(GlorysDriftDataset, self)._extract_data(file, load_fn)
        nan_mask = np.all(data == 0.0, axis=0)
        data[:, nan_mask] = np.nan
        return data

    @property
    def _udrift_variable(self):
        return 'usi'

    @property
    def _vdrift_variable(self):
        return 'vsi'


class GlorysThickDataset(ModelThickDataset):
    @property
    def _thick_variable(self):
        return 'sithick'


class GlorysSalinityDataset(ModelSalinityDataset):
    @property
    def _salinity_variable(self):
        return 'so'


class GlorysTemperatureDataset(ModelTemperatureDataset):
    @property
    def _temp_variable(self):
        return 'thetao'


class GlorysEastCurrentDataset(ModelEastCurrentDataset):
    @property
    def _east_cur_variable(self):
        return 'uo'


class GlorysNorthCurrentDataset(ModelNorthCurrentDataset):
    @property
    def _north_cur_variable(self):
        return 'vo'


class GlorysCurrentVelocityDataset(ModelCurrentVelocityDataset):
    @property
    def _east_cur_variable(self):
        return 'uo'

    @property
    def _north_cur_variable(self):
        return 'vo'


class GlorysOperativeSicDataset(GlorysOperativeDataset, GlorysSicDataset):
    pass


class GlorysReanalysisSicDataset(GlorysReanalysisDataset, GlorysSicDataset):
    pass


class GlorysOperativeDriftDataset(GlorysOperativeDataset, GlorysDriftDataset):
    pass


class GlorysReanalysisDriftDataset(GlorysReanalysisDataset, GlorysDriftDataset):
    pass


class GlorysOperativeThickDataset(GlorysOperativeDataset, GlorysThickDataset):
    pass


class GlorysReanalysisThickDataset(GlorysReanalysisDataset, GlorysThickDataset):
    pass


class GlorysOperativeSalinityDataset(GlorysOperativeDataset, GlorysSalinityDataset):
    @property
    def _files_template(self):
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m.nc'


class GlorysReanalysisSalinityDataset(GlorysReanalysisDataset, GlorysSalinityDataset):
    pass


class GlorysOperativeTemperatureDataset(GlorysOperativeDataset, GlorysTemperatureDataset):
    @property
    def _files_template(self):
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m.nc'


class GlorysReanalysisTemperatureDataset(GlorysReanalysisDataset, GlorysTemperatureDataset):
    pass


class GlorysOperativeEastCurrentDataset(GlorysOperativeDataset, GlorysEastCurrentDataset):
    @property
    def _files_template(self):
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m.nc'


class GlorysReanalysisEastCurrentDataset(GlorysReanalysisDataset, GlorysEastCurrentDataset):
    pass


class GlorysOperativeNorthCurrentDataset(GlorysOperativeDataset, GlorysNorthCurrentDataset):
    @property
    def _files_template(self):
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m.nc'


class GlorysReanalysisNorthCurrentDataset(GlorysReanalysisDataset, GlorysNorthCurrentDataset):
    pass


class GlorysOperativeCurrentVelocityDataset(GlorysOperativeDataset, GlorysCurrentVelocityDataset):
    @property
    def _files_template(self):
        return 'glorys_*/GLORYS_*_00_cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m.nc'


class GlorysReanalysisCurrentVelocityDataset(GlorysReanalysisDataset, GlorysCurrentVelocityDataset):
    pass
