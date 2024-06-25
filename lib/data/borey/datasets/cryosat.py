from datetime import datetime

import xarray as xr

from lib.data.borey import Grid
from lib.data.borey.datasets.base import ModelThickDataset


class CryosatThickDataset(ModelThickDataset):
    def __init__(self, path, dst_grid=None, average_times=None, name=None):
        super().__init__(path, dst_grid, average_times, name)

    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f'loading grid from {grid_path}')
        ds = xr.open_dataset(grid_path)

        lat = ds.variables['lat'].values
        lon = ds.variables['lon'].values
        grid = Grid(lat, lon)
        return grid

    @staticmethod
    def _parse_date(file):
        # Extract dates from filename
        parts = file.name.split('_')
        start_date_str = parts[8]
        end_date_str = parts[9]

        # Convert string dates to datetime objects
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")

        # Calculate the middle date
        middle_date = start_date + (end_date - start_date) / 2
        return middle_date.date()

    @property
    def _files_template(self):
        return 'SEAICE_ARC_PHY_L4_NRT_011_014/esa_obs-si_arc_phy-sit_nrt_l4_multi_P1D-m_202207/**/*.nc'

    @property
    def _thick_variable(self):
        return 'analysis_sea_ice_thickness'
