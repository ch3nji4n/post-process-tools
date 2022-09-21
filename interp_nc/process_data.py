import arrow
import pandas as pd
import xarray as xr
import warnings
from date_time import gen_time_list

warnings.filterwarnings('ignore')


def aq_hourly2daily_xr(hourly_conc_xr, time_ind_name='time'):
    """
    calculate daily conc for calculating daily aqi in xr.dataset format
    :param hourly_conc_xr: xr.dataset, hourly concentration
    :param time_ind_name: str, name of the coords representing the time list
    :return: xr.dataset, daily concentration
    """
    # set format of time list, floor(second)
    format_hour_list = pd.to_datetime([arrow.get(pd.to_datetime(i).strftime('%Y-%m-%d %H:%M:%S')).floor('hour').format('YYYY-MM-DD HH:mm:SS') if pd.to_datetime(i).minute == 0 else arrow.get(pd.to_datetime(i).strftime('%Y-%m-%d %H:%M:%S')).shift(hours=1).floor('hour').format('YYYY-MM-DD HH:mm:SS')
                                       for i in hourly_conc_xr[time_ind_name].values])
    hourly_conc_xr[time_ind_name] = format_hour_list
    # make up whole time list
    hour_list_temp = gen_time_list('hour', pd.to_datetime(format_hour_list.values[0]).strftime('%Y-%m-%d 00:00:00'),
                                   pd.to_datetime(format_hour_list.values[-1]).strftime('%Y-%m-%d 23:00:00'),
                                   datestr_format='YYYY-MM-DD HH:mm:SS')
    hour_list = [t.format('YYYY-MM-DD HH:mm:ss') for t in hour_list_temp]
    hourly_conc_xr = hourly_conc_xr.reindex({time_ind_name: pd.to_datetime(hour_list)})
    hour_list_new = hourly_conc_xr[time_ind_name]
    # valid daily mean needs >=20-hour valid hourly data
    hour_rolling_mean = hourly_conc_xr.rolling({time_ind_name: 24}, min_periods=20).mean(dim=time_ind_name)
    hour_data_valid = hour_rolling_mean.loc[{time_ind_name: hour_list_new[hour_list_new.dt.hour == 23]}]
    daily_obs_xr = hour_data_valid.resample({time_ind_name: '1D'}).mean(dim=time_ind_name, skipna=False)

    if 'O3' in list(hourly_conc_xr.data_vars):
        # >=6-hour valid hourly data for each 8 hours
        O3_8hr = hourly_conc_xr.rolling({time_ind_name: 8}, min_periods=6).mean(dim=time_ind_name)
        # to calc O3 8-hour-average max, only use hourly data after 7:00
        O3_8hr_valid = O3_8hr.loc[{time_ind_name: hour_list_new[hour_list_new.dt.hour >= 7]}]
        O3_8hr_max = O3_8hr_valid.resample({time_ind_name: '1D'}).max(dim=time_ind_name, skipna=True)
        O3_8hr_max = O3_8hr_max.rename({'O3': 'O3_8hr'})
        hour_rolling_max = hourly_conc_xr.rolling({time_ind_name: 24}, min_periods=20).max(dim=time_ind_name)
        hour_data_max = hour_rolling_max.loc[{time_ind_name: hour_list_new[hour_list_new.dt.hour == 23]}]
        conc_max = hour_data_max.resample({time_ind_name: '1D'}).mean(dim=time_ind_name, skipna=False)
        conc_max = conc_max.rename({'O3': 'O3_1hr'})
        daily_obs_xr = xr.merge([daily_obs_xr, O3_8hr_max['O3_8hr'], conc_max['O3_1hr']])
        daily_obs_xr = daily_obs_xr.drop_vars('O3')
    return daily_obs_xr


def aq_hourly2hourly_xr(hourly_conc_xr, time_ind_name='time'):
    """
    calculate daily conc for calculating daily aqi in xr.dataset format
    :param hourly_conc_xr: xr.dataset, hourly concentration
    :param time_ind_name: str, name of the coords representing the time list
    :return: xr.dataset, daily concentration
    """
    new_hourly_conc_xr = hourly_conc_xr.copy()
    if 'O3' in list(new_hourly_conc_xr.data_vars):
        hour_list = new_hourly_conc_xr[time_ind_name]
        O3_8hr = new_hourly_conc_xr.rolling({time_ind_name: 8}, min_periods=6).mean(dim=time_ind_name)
        O3_8hr_valid = O3_8hr.loc[{time_ind_name: hour_list[hour_list.dt.hour >= 7]}]
        O3_8hr_valid = O3_8hr_valid.rename({'O3': 'O3_8hr'})
        new_hourly_conc_xr = xr.merge([new_hourly_conc_xr, O3_8hr_valid['O3_8hr']])
        new_hourly_conc_xr['O3_1hr'] = new_hourly_conc_xr['O3']
        new_hourly_conc_xr = new_hourly_conc_xr.drop_vars('O3')
    return new_hourly_conc_xr