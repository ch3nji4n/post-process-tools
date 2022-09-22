# Author: CJ
# 2022/07/08

import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import arrow
import time
from multiprocessing import Pool, cpu_count
import os
from functools import wraps 
import config as cfg
from pyaqi import calc_aqi_xr
from process_data import aq_hourly2hourly_xr, aq_hourly2daily_xr
from date_time import get_ymd


def timer(func):
    @wraps(func)
    def cal_time(*args, **kwargs):
        t1 = time.time()
        ret = func(*args, **kwargs)
        t2 = time.time()
        print(f'time cost of func: {func.__name__} is {t2 - t1:.6f} sec')
        return ret
    return cal_time


def interp_hourly(t, t_stamp, out_fl_path):
    """
    interp amds hourly then save 
    use global DATASET_ADMS_HOURLY, LON, LAT, OUTPUT_LON, OUTPUT_LAT
    :param t: time from adms
    :param t_stamp: str time stamp
    :param out_fl_path: str target save file path
    """
    print(f'interpolating adms - time: {t_stamp} ...')

    ds_t = DATASET_ADMS_HOURLY.sel(datatime=t) #using global ds
    var_list = ['PM10', 'PM2.5', 'NO2', 'SO2', 'O3_1hr', 'O3_8hr', 'CO', 'AQI']
    
    ds_interp_h = xr.Dataset()
    ds_interp_h = ds_interp_h.assign(lat=(('west_east', 'south_north'), OUTPUT_LAT))
    ds_interp_h = ds_interp_h.assign(lon=(('west_east', 'south_north'), OUTPUT_LON))

    for var in var_list:
        if var == 'CO':
            grid_zeros = np.zeros(shape=OUTPUT_LON.shape) 
            ds_interp_h = ds_interp_h.assign(CO=(('west_east', 'south_north'), grid_zeros))
        else:
            grid_interp = griddata((LON, LAT), ds_t[var], (OUTPUT_LON, OUTPUT_LAT), method='linear')

        if var == 'PM10':
            ds_interp_h = ds_interp_h.assign(PM10=(('west_east', 'south_north'), grid_interp))
        elif var == 'PM2.5':
            ds_interp_h = ds_interp_h.assign(PM25=(('west_east', 'south_north'), grid_interp))
        elif var == 'NO2':
            ds_interp_h = ds_interp_h.assign(NO2=(('west_east', 'south_north'), grid_interp))
        elif var == 'SO2':
            ds_interp_h = ds_interp_h.assign(SO2=(('west_east', 'south_north'), grid_interp))
        elif var == 'O3_1hr':
            ds_interp_h = ds_interp_h.assign(O3=(('west_east', 'south_north'), grid_interp))
        elif var == 'O3_8hr':
            ds_interp_h = ds_interp_h.assign(O3_8H=(('west_east', 'south_north'), grid_interp))
        elif var == 'AQI':
            ds_interp_h = ds_interp_h.assign(AQI=(('west_east', 'south_north'), grid_interp))

    #print(ds_interp_h)
    print('saving to', out_fl_path)
    encoding = {var: {'zlib': True, 'complevel':9} for var in ds_interp_h.data_vars}
    ds_interp_h.to_netcdf(out_fl_path, encoding=encoding)
    #ds_interp_h.to_netcdf(out_fl_path)


def interp_daily(t, out_fl_path):
    """
    interp amds daily then save 
    use global DATASET_ADMS_DAILY, LON, LAT, OUTPUT_LON, OUTPUT_LAT
    :param t: time from adms
    :param out_fl_path: str target save file path
    """
    print(f'interpolating adms - time: {t.values} ...')

    ds_t = DATASET_ADMS_DAILY.sel(datatime=t) 
    var_list = ['PM10', 'PM2.5', 'NO2', 'SO2', 'O3_1hr', 'O3_8hr', 'CO', 'AQI']
    
    ds_interp_dly = xr.Dataset()
    ds_interp_dly = ds_interp_dly.assign(lat=(('west_east', 'south_north'), OUTPUT_LAT))
    ds_interp_dly = ds_interp_dly.assign(lon=(('west_east', 'south_north'), OUTPUT_LON))

    for var in var_list:
        if var == 'CO':
            grid_zeros = np.zeros(shape=OUTPUT_LON.shape) 
            ds_interp_dly = ds_interp_dly.assign(CO=(('west_east', 'south_north'), grid_zeros))
        else:
            grid_interp = griddata((LON, LAT), ds_t[var], (OUTPUT_LON, OUTPUT_LAT), method='linear')

        if var == 'PM10':
            ds_interp_dly = ds_interp_dly.assign(PM10=(('west_east', 'south_north'), grid_interp))
        elif var == 'PM2.5':
            ds_interp_dly = ds_interp_dly.assign(PM25=(('west_east', 'south_north'), grid_interp))
        elif var == 'NO2':
            ds_interp_dly = ds_interp_dly.assign(NO2=(('west_east', 'south_north'), grid_interp))
        elif var == 'SO2':
            ds_interp_dly = ds_interp_dly.assign(SO2=(('west_east', 'south_north'), grid_interp))
        elif var == 'O3_1hr':
            ds_interp_dly = ds_interp_dly.assign(O3=(('west_east', 'south_north'), grid_interp))
        elif var == 'O3_8hr':
            ds_interp_dly = ds_interp_dly.assign(O3_8H=(('west_east', 'south_north'), grid_interp))
        elif var == 'AQI':
            ds_interp_dly = ds_interp_dly.assign(AQI=(('west_east', 'south_north'), grid_interp))

    #print(ds_interp_dly)
    print('saving to', out_fl_path)
    encoding = {var: {'zlib': True, 'complevel':9} for var in ds_interp_dly.data_vars}
    ds_interp_dly.to_netcdf(out_fl_path, encoding=encoding)
    #ds_interp_dly.to_netcdf(out_fl_path)


@timer
def cal_aqi(ds_pllts):
    """
    calculate aqi and return dataset with aqi
    :param ds: target adms dataset 
    """
    ds_pllts = aq_hourly2hourly_xr(ds_pllts, time_ind_name='datatime')
    print(ds_pllts)
    ds_aqi = calc_aqi_xr(ds_pllts, option='hourly')
    print(ds_aqi)
    ds_aqi = ds_aqi.rename({0: 'AQI'})
    ds_merge = xr.merge([ds_pllts, ds_aqi])
    return ds_merge 


def get_pllts(ds):
    pllt_list = ['PM10', 'PM2.5', 'NO2', 'SO2', 'O3']
    ds_pllts = ds[pllt_list[0]]
    for pllt in pllt_list[1:]:
        ds_pllt = ds[pllt] 
        ds_pllts = xr.merge([ds_pllts, ds_pllt])
    return ds_pllts


def err_call_back(err):
    """
    print error for Pool.apply_async()
    """
    print(f'error：{str(err)}')

  
def get_adms_fl_path(date, shift_days):
    """
    gen adms file path, if not exists, then find last 3 days files
    if can not find file within last 3 days, then exit
    """
    shifted_date, fldr_dir_adms, adms_fl_path = cfg.get_adms_dir(date, shift_days)
    if not os.path.exists(adms_fl_path):
        print(f'warning: adms file not exits({adms_fl_path}), finding last day file..')
        for i in [-1,-2,-3]:
            shift_days += i
            shifted_date, fldr_dir_adms, adms_fl_path = cfg.get_adms_dir(date, shift_days)
            if not os.path.exists(adms_fl_path):
                continue
            else:
                print(f'loading {adms_fl_path}')
                break
    if not os.path.exists(adms_fl_path):
        print(f'error: can not find {adms_fl_path}')
        exit()
    return adms_fl_path


def fix_adms(conc_xr):
    adms_coef = {'coef': {'PM10': 0.56, 'PM2.5': 0.4, 'O3': 0.76, 'NO2': 0.16, 'SO2': 0.04, 'CO': 1},
                 'intercept': {'PM10': 15, 'PM2.5': 8, 'O3': 20, 'NO2': 15, 'SO2': 6, 'CO': 0}}
    for var in conc_xr.data_vars:
        conc_xr[var] = conc_xr[var] * adms_coef['coef'][var] + adms_coef['intercept'][var]
    return conc_xr


if __name__ == '__main__':
    target_date = arrow.now()
    trg_y, trg_ym, trg_ymd = get_ymd(target_date)
    adms_fl_path = get_adms_fl_path(target_date, -2)
    out_fldr_dir = cfg.adms_interp_output_fldr_template % (trg_y, trg_ym, trg_ymd)   
    if not os.path.exists(out_fldr_dir):
        os.makedirs(out_fldr_dir)

    t1 = time.time()
    print('start to interpolate adms file...')
    print(f'loading {adms_fl_path}')

    global DATASET_ADMS_HOURLY, DATASET_ADMS_DAILY, LON, LAT, OUTPUT_LON, OUTPUT_LAT #global vars for multi processes 
    DATASET_ADMS = xr.open_dataset(adms_fl_path)
    LON = DATASET_ADMS['longitude']
    LAT = DATASET_ADMS['latitude']
    lon_min = 113.7
    lon_max = 114.7
    lat_min = 22.4
    lat_max = 22.9
    grid_interval = 0.001 #resolution
    lon_range = np.arange(lon_min, lon_max, grid_interval)
    lat_range = np.arange(lat_min, lat_max, grid_interval)
    OUTPUT_LON, OUTPUT_LAT = np.meshgrid(lon_range, lat_range)

    fcst_time = DATASET_ADMS['datatime']
    dt = target_date.naive 
    dt = np.datetime64(dt)
    fcst_time = fcst_time.where(fcst_time >= dt, drop=True)
    DATASET_ADMS = get_pllts(DATASET_ADMS) 
    DATASET_ADMS = fix_adms(DATASET_ADMS) 
    DATASET_ADMS_HOURLY = cal_aqi(DATASET_ADMS) # will concat O3_8hr and AQI

    # start to interpolate hourly adms 
    cpu_num = cpu_count() 
    cpu_num = min(cpu_num, len(fcst_time))
    p = Pool(cpu_num)

    for t in fcst_time:
        t_stamp = pd.Timestamp(t.values).strftime('%Y-%m-%d %X')
        if arrow.get(t_stamp).minute != 0:
            t_stamp = arrow.get(t_stamp).shift(hours=1).floor('hour')
        else:
            t_stamp = arrow.get(t_stamp)
        ymdh = t_stamp.format('YYYYMMDDHH')
        t_stamp_str = t_stamp.format('YYYY-MM-DD HH:mm:ss')        
        out_flnm = cfg.adms_interp_flnm_template % ymdh
        out_fl_path = os.path.join(out_fldr_dir, out_flnm)
        #interp_hourly(t.values, t_stamp_str, out_fl_path)
        p.apply_async(interp_hourly, 
                      args=(t.values, t_stamp_str, out_fl_path,),
                      error_callback = err_call_back)
    p.close()
    p.join()
    t2 = time.time()
    print('hourly interpolation time cost(s):', t2-t1)
    # hourly interplation end
     
    # start to interpolate daily adms
    t3 = time.time()
    DATASET_ADMS_DAILY = aq_hourly2daily_xr(DATASET_ADMS, time_ind_name='datatime')
    ds_daily_aqi = calc_aqi_xr(DATASET_ADMS_DAILY, option='daily')
    ds_daily_aqi = ds_daily_aqi.rename({0: 'AQI'})
    DATASET_ADMS_DAILY = xr.merge([DATASET_ADMS_DAILY, ds_daily_aqi]) 
    del DATASET_ADMS
    
    dates_daily = DATASET_ADMS_DAILY['datatime']
    dates_daily = dates_daily.where(dates_daily >= dt, drop=True)
    cpu_num = len(dates_daily.values)
    p = Pool(cpu_num)
    for t in dates_daily:
        dt = pd.Timestamp(t.values).strftime('%Y-%m-%d')
        ymd = arrow.get(dt).format('YYYYMMDD')
        out_flnm = cfg.adms_interp_daily_flnm_template % ymd
        out_fl_path = os.path.join(out_fldr_dir, out_flnm)
        p.apply_async(interp_daily, 
                      args=(t, out_fl_path,),
                      error_callback = err_call_back)
    p.close()
    p.join()
    t4 = time.time()
    print('daily interpolation time cost(s):', t4-t3)
    # daily interpolation end