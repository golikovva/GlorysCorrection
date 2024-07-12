import os
import time
import argparse
import xarray as xr
import numpy as np
import torch
from yaml import load, SafeLoader
import sys
sys.path.insert(0, '../../')
from lib.data.dataset_utils import get_time_encoding
from lib.data.scaler import StandardScaler
from lib.model.build_module import build_inference_correction_model


def run_correction(parameters):
    start = time.time()
    ds = xr.open_dataset(parameters['input']).copy(deep=True)
    print('Loading model...')
    model = build_inference_correction_model(parameters)

    if parameters['output'] is None:
        output_file = parameters['input'][:-3] + '_corrected.nc'
    elif parameters['output'] == parameters['input']:
        # os.remove(parameters['input'])
        output_file = parameters['input'][:-3] + '_corrected.nc'
    else:
        output_file = parameters['output']
    print('Applying model...')
    ds[parameters['salinity_var']].data = apply_model(ds, model, parameters)
    os.remove(output_file) if os.path.exists(output_file) else None
    ds.to_netcdf(output_file)
    print(f'Time spent to apply correction: {round(time.time()-start, 2)} s')
    return parameters['output']


def apply_model(dataset, model, parameters):
    orig_so = dataset[parameters['salinity_var']].data
    time = dataset[parameters['time_var']].data
    if parameters['model_type'] == "spatiotemporal_baseline":
        day_of_year = (time.astype('datetime64[D]') - time.astype('datetime64[Y]')).astype(int)
        return model(orig_so, day_of_year)

    elif "unet" in parameters['model_type']:
        nan_mask = np.isnan(orig_so)
        orig_so = np.nan_to_num(orig_so)

        lon = dataset[parameters['longitude_var']].data
        lat = dataset[parameters['latitude_var']].data
        lon, lat = np.stack(np.meshgrid(lon, lat))  # 181, 577
        time_encoding = get_time_encoding(time, lon, lat, frequency=4)[:, None]  # 5 1 181 577
        time_encoding = np.broadcast_to(time_encoding, orig_so.shape).copy()  # 5 34 181 577
        lon = np.broadcast_to(lon, orig_so.shape).copy()  # 5 34 181 577
        lat = np.broadcast_to(lat, orig_so.shape).copy()  # 5 34 181 577
        model_input = torch.from_numpy(np.stack([orig_so.data, time_encoding, lat, lon], axis=1)).float()  # 5 4 34 h,w
        model_input = model_input.to(parameters['device'])
        scaler = StandardScaler()
        scaler.apply_scaler_channel_params(torch.load(parameters['means_path']).float(),
                                           torch.load(parameters['stds_path']).float())
        scaler.to(parameters['device'])
        with torch.no_grad():
            model_input = scaler.transform(model_input, dims=1)
            out = model(model_input)
            out = scaler.inverse_transform(out, means=scaler.means[0], stds=scaler.stddevs[0], dims=1)
            out = out.view(orig_so.shape).cpu().numpy()
        out[nan_mask] = np.nan
        return out

    else:
        raise NotImplementedError


if __name__ == '__main__':
    """
    example inference command:
    python inference.py /home/glorys_op/glorys_24010200/GLORYS_2024-01-02_00_cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m.nc \
    config.yaml \
    -o /home/logs/spatiotemporal_baseline/GLORYS_2024-01-02_00_phy-so_corrected_1.nc
    
    """
    parser = argparse.ArgumentParser(
        prog='Glorys salinity correction',
        description='Correct glorys data from the input glorys data and correction field',
    )
    parser.add_argument(
        'input',
        type=str,
        help='path to the file to correct',
    )
    parser.add_argument(
        'namelist',
        type=str,
        help='path to the namelist file'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='path to the output file',
    )

    args_dict = vars(parser.parse_args())
    with open(args_dict['namelist'], 'r') as namelist_file:
        namelist = load(namelist_file, Loader=SafeLoader)
    parameters = {**args_dict, **namelist}

    saved_filename = run_correction(parameters)

