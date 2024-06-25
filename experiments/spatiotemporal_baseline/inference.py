import os
import argparse
import xarray as xr
import numpy as np
from yaml import load, SafeLoader


class AccumCorrector:
    def __init__(self, corr_fields_path):
        self.corr_fields = np.load(corr_fields_path).reshape((366, 34, 181, 577))  # day, z, h, w

    def __call__(self, operative, day_of_year):
        return operative + self.corr_fields[day_of_year]


def run_correction(parameters):
    ds = xr.open_dataset(parameters['input']).copy(deep=True)
    model = AccumCorrector(parameters['weights'])

    if parameters['output'] is None:
        output_file = parameters['input'][:-3] + '_corrected.nc'
    elif parameters['output'] == parameters['input']:
        # os.remove(parameters['input'])
        output_file = parameters['input'][:-3] + '_corrected.nc'
    else:
        output_file = parameters['output']

    so_var = parameters['salinity_var']
    time_var = parameters['time_var']
    orig_so = ds[so_var].data
    time = ds[time_var].data
    day_of_year = (time.astype('datetime64[D]') - time.astype('datetime64[Y]')).astype(int)
    ds[so_var].data = model(orig_so, day_of_year)

    os.remove(output_file) if os.path.exists(output_file) else None
    ds.to_netcdf(output_file)
    return parameters['output']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Glorys salinity correction',
        description='Correct glorys data from the input glorys data and correction field',
    )
    parser.add_argument(
        'input',
        type=str,
        help='path to the file to correct',
    )
    parser.add_argument('namelist', type=str, help='path to the namelist file')
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='path to the output file',
    )
    parser.add_argument(
        '--weights',
        '-w',
        type=str,
        help='path to the correction weights file',
    )

    args_dict = vars(parser.parse_args())
    with open(args_dict['namelist'], 'r') as namelist_file:
        namelist = load(namelist_file, Loader=SafeLoader)
    parameters = {**args_dict, **namelist}

    saved_filename = run_correction(parameters)

