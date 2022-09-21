import os
import yaml
import xarray as xr


def get_file_path(fldr_dir):
    """
    return files' path under target directory
    """
    fl_path_list = []
    for root, dirs, files in os.walk(fldr_dir):
        for flnm in files:
            fl_path = os.path.join(root, flnm)
            fl_path_list.append(fl_path)
    return fl_path_list


def concat_adms(fldr_dir):
    fl_path_list = get_file_path(fldr_dir)
    ds_init = xr.open_dataset(fl_path_list[0])
    LON = ds_init.longitude
    LAT = ds_init.latitude
    ds_concat = ds_init.drop_vars(['longitude', 'latitude'])
    for fl_path in fl_path_list[1:]:
        ds = xr.open_dataset(fl_path)
        ds = ds.drop_vars(['longitude', 'latitude'])
        ds_concat = xr.concat([ds_concat, ds], dim='datatime')
    ds_concat['longitude'] = LON
    ds_concat['latitude'] = LAT
    return ds_concat


config_path = 'config.yml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
today = arrow.now().floor('day')
before2day = today.shift(days=-2)
y, ym, ymd = get_ymd(today)
_y, _ym, _ymd = get_ymd(before2day)
fldr_dir = config['concat_fldr_dir'] % (_y, _ym, _ymd)
output_dir = config['concat_output_dir'] % (y, ym)
output_flnm = config['concat_output_flnm'] % ymd
print(f'Concatenating files in {fldr_dir} ...')
ds_concat = concat_adms(fldr_dir)
encoding = {var: {'zlib': True, 'complevel':9} for var in ds_concat.data_vars}
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
out_fl_path = os.path.join(output_dir, output_flnm)
ds_concat.to_netcdf(out_fl_path, encoding=encoding)
print(f'Concatenated file saved to {out_fl_path}')