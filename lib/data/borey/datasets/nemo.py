from datetime import datetime

import xarray as xr

from lib.data.borey import Grid
from lib.data.borey.datasets.base import (
    Dataset,
    ModelDriftDataset,
    ModelEastCurrentDataset,
    ModelNorthCurrentDataset,
    ModelSalinityDataset,
    ModelSicDataset,
    ModelTemperatureDataset,
    ModelThickDataset,
)


class NemoDataset(Dataset):
    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        ds = xr.open_dataset(grid_path)

        lat = ds.variables['nav_lat'].values
        lon = ds.variables['nav_lon'].values
        grid = Grid(lat, lon)
        return grid

    @property
    def _files_template(self):
        return 'run_*/BARKA12-TEST_*_forecast.1h_icemod.nc'

    @staticmethod
    def _parse_date(file):
        date_part = file.name.split('_')[1]
        date = datetime.strptime(date_part, 'y%Ym%md%d').date()
        return date


class NemoSicDataset(ModelSicDataset, NemoDataset):
    @property
    def _sic_variable(self):
        return 'siconc'


class NemoDriftDataset(ModelDriftDataset, NemoDataset):
    @property
    def _udrift_variable(self):
        return 'sivelu'

    @property
    def _vdrift_variable(self):
        return 'sivelv'


class NemoThickDataset(ModelThickDataset, NemoDataset):
    @property
    def _thick_variable(self):
        return 'sithic'


class NemoSalinityDataset(ModelSalinityDataset, NemoDataset):
    @property
    def _files_template(self):
        return 'run_*/BARKA12-TEST_*_forecast.1d_gridT.nc'

    @property
    def _salinity_variable(self):
        return 'vosaline'


class NemoTemperatureDataset(ModelTemperatureDataset, NemoDataset):
    @property
    def _files_template(self):
        return 'run_*/BARKA12-TEST_*_forecast.1d_gridT.nc'

    @property
    def _temp_variable(self):
        return 'votemper'


class NemoEastCurrentDataset(ModelEastCurrentDataset, NemoDataset):
    @property
    def _files_template(self):
        return 'run_*/BARKA12-TEST_*_forecast.1d_gridV.nc'

    @property
    def _east_cur_variable(self):
        return 'vozocrtx'


class NemoNorthCurrentDataset(ModelNorthCurrentDataset, NemoDataset):
    @property
    def _files_template(self):
        return 'run_*/BARKA12-TEST_*_forecast.1d_gridV.nc'

    @property
    def _north_cur_variable(self):
        return 'vomecrty'
