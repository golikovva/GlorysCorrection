import xarray as xr


class GlorysDataset():
    def __init__(self, file_names, var_name='so'):
        self.data = xr.open_dataset(file_names[0])
        self.var_name = var_name

    def __getitem__(self, i):
        data = self.data[i]

    def __len__(self):
        pass


class GlorysOperativeDataset(GlorysDataset):
    pass


class GlorysReanalysisDataset(GlorysDataset):
    pass
