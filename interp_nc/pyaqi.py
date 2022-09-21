# @Time: 2020/7/15
# @Author: Weilin, CJ

import numpy as np
import pandas as pd
from functools import wraps 
import time


def timer(func):
    @wraps(func)
    def cal_time(*args, **kwargs):
        t1 = time.time()
        ret = func(*args, **kwargs)
        t2 = time.time()
        print(f'time cost of func: {func.__name__} is {t2 - t1:.6f} sec')
        return ret
    return cal_time


class Criteria:
    def __init__(self, option):
        self.aqi_criteria = np.array([0, 50, 100, 150, 200, 300, 400, 500])
        if option == 'hourly':
            self.iaqi_criteria = {
                'PM10': [0, 50, 150, 250, 350, 420, 500, 600],  # PM10_24hr_average
                'PM2.5': [0, 35, 75, 115, 150, 250, 350, 500],  # PM2.5_24hr_average
                'O3_1hr': [0, 160, 200, 300, 400, 800, 1000, 1200],  # O3_1hr
                'O3_8hr': [0, 100, 160, 215, 265, 800, 1000, 1200],  # O3_8hr_moving_average
                'NO2': [0, 100, 200, 700, 1200, 2340, 3090, 3840],  # NO2_1hr
                'SO2': [0, 150, 500, 650, 800, 1600, 2100, 2620],  # SO2_1hr
                'CO': [0, 5, 10, 35, 60, 90, 120, 150],
                'AQI': [0, 50, 100, 150, 200, 300, 400, 500]}  # new add

        elif option == 'daily':
            self.iaqi_criteria = {
                'PM10': [0, 50, 150, 250, 350, 420, 500, 600],  # PM10_24hr_average
                'PM2.5': [0, 35, 75, 115, 150, 250, 350, 500],  # PM2.5_24hr_average
                'O3_1hr': [0, 160, 200, 300, 400, 800, 1000, 1200],  # O3_1hr_max
                'O3_8hr': [0, 100, 160, 215, 265, 800, 1000, 1200],  # O3_8hr_moving_average_max
                'NO2': [0, 40, 80, 180, 280, 565, 750, 940],  # NO2_24hr_average
                'SO2': [0, 50, 150, 475, 800, 1600, 2100, 2620],  # SO2_24hr_average
                'CO': [0, 2, 4, 14, 24, 36, 48, 60],
                'AQI': [0, 50, 100, 150, 200, 300, 400, 500]}  # new add
        else:
            raise ValueError('option should only be hourly or daily')
        # aqi_slope = (IAQI_high - IAQI_low)/(BP_high - BP_low)
        self.aqi_slope = pd.DataFrame((np.diff(self.aqi_criteria) /  # IAQI_high - IAQI_low
                                       np.diff(pd.DataFrame(self.iaqi_criteria).values.T)).T,  # BP_high - BP_low
                                      # 这里要先转置是因为np.diff()是右边减去左边，所以要先转置，得出差值，然后再转置回原先的shape再进行相除
                                      columns=self.iaqi_criteria.keys(), dtype=object)


class AQI:
    """
    Attribute of pollutant
    """

    def __init__(self, name):
        self.name = name

    def calc_iaqi(self, conc_data, option):
        """
        calculate hourly or daily iaqi of concentration data (one pollutant)

        :param conc_data: pd.DataFrame, pd.Series, np.ndarray, list, int or float,
                          concentration of one pollutant, whose length is flexible
        :param option: str, 'daily' for calculating hourly iaqi,
                            'hourly' for calculationg daily iaqi

        :return: iaqi, same type as conc_data, iaqi of the pollutant

        example:
        --------
             iaqi = calc_iaqi([50, 30, np.nan, 57, 234], 'hourly')
        """

        if option not in ['hourly', 'daily']:
            raise ValueError('Parameter "option" can only be "hourly" or "daily".')

        # check data type of conc_data，and change into <class: np.ndarray>
        if isinstance(conc_data, pd.DataFrame) or isinstance(conc_data, pd.Series):
            conc_values = conc_data.values
        elif isinstance(conc_data, np.ndarray) or isinstance(conc_data, list):
            conc_values = np.array(conc_data)
        elif isinstance(conc_data, int) or isinstance(conc_data, float):
            conc_values = np.array([conc_data])
        else:
            raise TypeError('Parameter "conc_data" in type %s is invalid in this function.' % type(conc_data))

        # check if conc_data is 1-d data
        if len(conc_values.shape) != 1 and min(conc_values.shape) > 1:
            raise ValueError('Parameter "conc_data" should be 1-dimensional')

        criteria = Criteria(option)
        iaqi_criteria = criteria.iaqi_criteria[self.name]
        iaqi_criteria = np.reshape(iaqi_criteria, [1, len(iaqi_criteria)])
        aqi_criteria = criteria.aqi_criteria

        conc_temp = np.reshape(conc_values, [len(conc_values), 1])
        temp = (iaqi_criteria <= conc_temp).tolist()
        beg_ind = np.array([temp[i].count(True) - 1 for i in range(len(conc_temp))])
        exceed_ind = np.where(beg_ind > len(criteria.aqi_slope[self.name]) - 1)[0]  # index of iaqi larger than 500
        invalid_ind = np.where(beg_ind < 0)[0]  # index of iaqi is nan
        valid_ind = np.where((beg_ind >= 0) & (beg_ind <= len(criteria.aqi_slope[self.name]) - 1))[0]

        iaqi = np.array(list(range(len(conc_values)))).astype(float)
        iaqi[exceed_ind] = 501  # if iaqi is larger than 500, iaqi can't be calculated
        iaqi[invalid_ind] = np.nan  # invalid conc data
        # calc iaqi function: IAQI = aqi_slope * (conc - BP_low) + IAQI_low
        iaqi[valid_ind] = (
                criteria.aqi_slope.loc[beg_ind[valid_ind], self.name].values *  # aqi_slope
                (conc_values[valid_ind].flatten() - iaqi_criteria.flatten()[beg_ind[valid_ind]]) +  # conc - BP_low
                aqi_criteria[beg_ind[valid_ind]]).tolist()  # IAQI_low
        iaqi = np.squeeze(np.ceil(iaqi))

        # use index of conc_data as the index of iaqi if possible
        if isinstance(conc_data, pd.DataFrame) or isinstance(conc_data, pd.Series):
            iaqi = pd.DataFrame(iaqi, index=conc_data.index, columns=[self.name])

        return iaqi


    def aqi_to_conc(self, iaqi, option):
        """
        calculate concentration from iaqi

        :param iaqi: pd.DataFrame, pd.Series, np.ndarray, list, int or float,
                     iaqi of one pollutant, whose length is flexible
        :param option: str, 'daily' for calculating hourly concentration,
                            'hourly' for calculating daily concentration

        :return: concentration, same type as iaqi
        """

        if option not in ['hourly', 'daily']:
            raise ValueError('Parameter "option" can only be "hourly" or "daily".')

        # check data type of conc_data，and change into <class: np.ndarray>
        if isinstance(iaqi, pd.DataFrame) or isinstance(iaqi, pd.Series):
            iaqi_values = iaqi.values
        elif isinstance(iaqi, np.ndarray) or isinstance(iaqi, list):
            iaqi_values = np.array(iaqi)
        elif isinstance(iaqi, int) or isinstance(iaqi, float):
            iaqi_values = np.array([iaqi])
        else:
            raise TypeError('Parameter "iaqi" in type %s is invalid in this function.' % type(iaqi))

        # check if conc_data is 1-d data
        if len(iaqi_values.shape) != 1 and min(iaqi_values.shape) > 1:
            raise ValueError('Parameter "iaqi" should be 1-dimensional')

        criteria = Criteria(option)
        iaqi_criteria = criteria.iaqi_criteria[self.name]
        iaqi_criteria = np.reshape(iaqi_criteria, [1, len(iaqi_criteria)])
        aqi_criteria = criteria.aqi_criteria

        iaqi_temp = np.reshape(iaqi_values, [len(iaqi_values), 1])
        temp = (aqi_criteria <= iaqi_temp).tolist()
        beg_ind = np.array([temp[i].count(True) - 1 for i in range(len(iaqi_temp))])
        exceed_ind = np.where(beg_ind > len(criteria.aqi_slope[self.name]) - 1)[0]  # index of iaqi larger than 500
        invalid_ind = np.where(beg_ind < 0)[0]  # index of iaqi is nan
        valid_ind = np.where((beg_ind >= 0) & (beg_ind <= len(criteria.aqi_slope[self.name]) - 1))[0]

        conc = np.array(list(range(len(iaqi_values)))).astype(float)
        conc[exceed_ind] = np.max(iaqi_criteria) + 1  # if iaqi is larger than 500, conc can't be calculated
        conc[invalid_ind] = np.nan  # invalid conc data
        # calculate valid concentration
        # function: conc = (IAQI - IAQI_low) / aqi_slope + BP_low
        conc[valid_ind] = (
                (iaqi_values[valid_ind].flatten() - aqi_criteria[beg_ind[valid_ind]]) /
                criteria.aqi_slope.loc[beg_ind[valid_ind], self.name].values +
                iaqi_criteria.flatten()[beg_ind[valid_ind]]
        ).tolist()  # IAQI_low
        conc = np.squeeze(conc)

        # use index of conc_data as the index of iaqi if possible
        if isinstance(iaqi, pd.DataFrame) or isinstance(iaqi, pd.Series):
            conc = pd.DataFrame(conc, index=iaqi.index, columns=[self.name], dtype=object)

        return conc


@timer
def calc_aqi(conc_data, option):
    iaqi_df = pd.DataFrame(index=conc_data.index, columns=conc_data.columns, dtype=object)
    for var_nm in conc_data.columns:
        aqi_demo = AQI(var_nm)
        iaqi_df[var_nm] = aqi_demo.calc_iaqi(conc_data[var_nm].values, option)
    iaqi_df = iaqi_df.dropna(how='all')  # 传入how=’all’滤除全为NaN的行， 若滤除列，则添加axis=1
    aqi = pd.DataFrame(iaqi_df.max(axis=1, skipna=True), dtype=object)  # calc aqi: maximum of iaqi
    # iaqi_df.rename(columns={'O3_8hr': 'O3', 'O3_1hr': 'O3'}, inplace=True)
    # pp = iaqi_df.idxmax(axis=1)  # only one index
    # return aqi, iaqi_df, pp
    return aqi, iaqi_df


@timer
def calc_aqi_xr(conc_xr, option):
    conc_df = conc_xr.to_dataframe()
    aqi, iaqi_df = calc_aqi(conc_df, option)
    aqi_xr = aqi.to_xarray()
    return aqi_xr


if __name__ == '__main__':
    aqi_demo = AQI('PM10')
    print(aqi_demo.calc_iaqi([65, 100, 55], 'daily'))
