import esmpy
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import scipy
import gc

class Interpolator:
    def __init__(self, src_grid, dst_grid):
        self.src_grid = src_grid
        self.dst_grid = dst_grid

        self.regrid = None
        self.dst_region = None

    def initialize(self):
        self.src_field = esmpy.Field(grid=self.src_grid.grid)
        self.src_field.data[:] = 0.0

        self.dst_field = esmpy.Field(grid=self.dst_grid.grid)
        self.dst_field.data[:] = np.nan

        src_scale = self.src_grid.cell_areas().mean()
        dst_scale = self.dst_grid.cell_areas().mean()
        if src_scale < dst_scale:
            regrid_method = esmpy.api.constants.RegridMethod.CONSERVE
            print('using conserve regrid method')
        else:
            regrid_method = esmpy.api.constants.RegridMethod.BILINEAR
            print('using bilinear regrid method')

        self.regrid = esmpy.Regrid(
            self.src_field,
            self.dst_field,
            regrid_method=regrid_method,
            unmapped_action=esmpy.api.constants.UnmappedAction.IGNORE
        )
        self.dst_region = ~np.isnan(self.dst_field.data)

    def __call__(self, field):
        assert self.regrid is not None, 'Interpolator must be initialized before use'

        # Обновляем данные в полях
        self.src_field.data[:] = field
        self.dst_field.data[:] = np.nan

        # Выполняем интерполяцию
        self.regrid(self.src_field, self.dst_field)
        interpolated_field = self.dst_field.data.copy()

        interpolated_field[~self.dst_region] = np.nan

        return interpolated_field
