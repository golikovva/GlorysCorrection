import esmpy
import global_land_mask as glm
import numpy as np
import pyproj
import scipy
import xarray as xr


class Grid:
    earth_radius = 6371.0

    def __init__(self, lat, lon):
        assert lat.shape == lon.shape, 'lat and lon must have the same shape'
        self.shape = lat.shape

        self.grid = esmpy.Grid(
            np.array(self.shape),
            staggerloc=[esmpy.StaggerLoc.CENTER, esmpy.StaggerLoc.CORNER],
            coord_sys=esmpy.CoordSys.SPH_DEG
        )

        self.grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)[:] = lon
        self.grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)[:] = lat

        lat_corners, lon_corners = self._estimate_cell_corners(lat, lon)
        self.grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)[:] = lon_corners
        self.grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)[:] = lat_corners

    @property
    def lat(self):
        return self.grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CENTER)[:]

    @property
    def lon(self):
        return self.grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CENTER)[:]

    @property
    def lat_corners(self):
        return self.grid.get_coords(1, staggerloc=esmpy.StaggerLoc.CORNER)[:]

    @property
    def lon_corners(self):
        return self.grid.get_coords(0, staggerloc=esmpy.StaggerLoc.CORNER)[:]

    def cell_areas(self):
        field = esmpy.Field(self.grid)
        field.get_area()
        areas = field.data * self.earth_radius ** 2
        return areas

    def land_mask(self):
        mask = glm.is_land(self.lat, self.lon)
        return mask

    def as_netcdf(self, title):
        # Create xarray DataArrays, optionally you can specify coordinates and other attributes
        lat = xr.DataArray(self.lat, dims=['y', 'x'], name='lat')
        lon = xr.DataArray(self.lon, dims=['y', 'x'], name='lon')

        # Combine into a Dataset
        ds = xr.Dataset({'lat': lat, 'lon': lon})

        # Add attributes to variables if needed
        ds['lat'].attrs['units'] = 'degrees'
        ds['lat'].attrs['description'] = 'Latitude'
        ds['lon'].attrs['units'] = 'degrees'
        ds['lon'].attrs['description'] = 'Longitude'

        # Add global attributes
        ds.attrs['title'] = title
        return ds

    @staticmethod
    def _estimate_cell_corners(lat, lon):
        stereo_crs = pyproj.CRS.from_epsg(3413)
        plate_caree_crs = pyproj.CRS.from_epsg(4326)

        transformer = pyproj.Transformer.from_crs(plate_caree_crs, stereo_crs)
        xx, yy = transformer.transform(lat, lon)

        xx_corners = Grid._estimate_array_corners(xx)
        yy_corners = Grid._estimate_array_corners(yy)

        transformer = pyproj.Transformer.from_crs(stereo_crs, plate_caree_crs)
        lat_corners, lon_corners = transformer.transform(xx_corners, yy_corners)
        return lat_corners, lon_corners

    @staticmethod
    def _estimate_array_corners(arr):
        assert len(arr.shape) == 2, 'array to extrapolate must be 2D'
        assert arr.shape[0] > 1 and arr.shape[1] > 1, \
            'array to extrapolate must have at least 2 elements in each dimension'

        extr_arr = np.pad(arr, 1)
        extr_arr[0, :] = 2 * extr_arr[1, :] - extr_arr[2, :]
        extr_arr[:, -1] = 2 * extr_arr[:, -2] - extr_arr[:, -3]
        extr_arr[-1, :] = 2 * extr_arr[-2, :] - extr_arr[-3, :]
        extr_arr[:, 0] = 2 * extr_arr[:, 1] - extr_arr[:, 2]

        corners = scipy.signal.convolve2d(extr_arr, 0.25 * np.ones((2, 2)), mode='valid')
        return corners


class BarentsKaraGrid(Grid):
    def __init__(self, resolution=1.0):
        x = np.linspace(1_200_000, 2_700_000, round(1500 / resolution))
        y = np.linspace(1_000_000, -800_000, round(1800 / resolution))
        xx, yy = np.meshgrid(x, y)

        stereo_crs = pyproj.CRS.from_epsg(3413)
        plate_caree_crs = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(stereo_crs, plate_caree_crs)
        lat, lon = transformer.transform(xx, yy)

        super().__init__(lat, lon)


class PseudoNemoGrid(Grid):
    def __init__(self, resolution=1.0):
        x = np.linspace(1_385_000, 2_335_000, round(950 / resolution))
        y = np.linspace(-1_565_000, -275_000, round(1290 / resolution))
        theta = np.radians(32.0)
        pregrid = np.meshgrid(x, y)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xx, yy = np.einsum('ij, jmn -> imn', rot, pregrid)

        stereo_crs = pyproj.CRS.from_epsg(3413)
        plate_caree_crs = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(stereo_crs, plate_caree_crs)
        lat, lon = transformer.transform(xx, yy)

        super().__init__(lat, lon)
