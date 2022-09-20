# Auth: CJ
# 2022/09/20

import os
import yaml
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.basemap import Basemap

# set up
config_path = 'config.yml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
fl_dir = config['file_dir']
time_tag = config['time_tag']
time_frame = config['time_frame'] 
model_type = config['model_type'] 
pllt_list = config['pllt_list'] 
colorbar_max = config['colorbar_max'] 
shp_fl_dir = config['shape_file_dir'] 
output_root = config['outpur_root'] 
output_dir = os.path.join(output_root, time_tag)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
plt.rcParams['font.sans-serif'] = config['font'] 

# read dataset
dataset = xr.open_dataset(fl_dir)
# dataset = dataset.sel(time=YYYYMMDDHH)
lat = dataset.lat.values
lon = dataset.lon.values
lat_min = lat.min()
lat_max = lat.max()
lon_min = lon.min()
lon_max = lon.max()

# plot
for idx, pllt in enumerate(pllt_list):
    data = dataset[pllt]
    proj = ccrs.PlateCarree()  # create projection
    fig = plt.figure(figsize=(28,12))  

    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
    _map = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,)
    _map.readshapefile(shp_fl_dir, 'states', drawbounds=True, linewidth=1, color='black', default_encoding='gbk')

    t_str = time_tag + ' ' + time_frame + ' ' + pllt
    ax.set_title(t_str, fontsize=18)
    mesh = plt.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), 
                          cmap='jet', vmin=0, vmax=colorbar_max[idx],)
    plt.colorbar(mesh)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    flnm = f'{model_type}_{time_tag}_{pllt}.jpg'
    save_path = os.path.join(output_dir, flnm)
    print(f'Ploting {save_path} ...')
    plt.savefig(save_path) 
    plt.close()
print('All imgs plotted.')