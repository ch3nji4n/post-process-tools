import os


def get_adms_dir(dt, shift_days=0):
    """
    :param dt: base date in arrow format
    :return date: shifted date
    """
    date = dt.shift(days=shift_days)
    y = date.format('YYYY')
    ym = date.format('YYYYMM')
    ymd12 = date.format('YYYYMMDD12')
    fldr_dir_adms = adms_dir_template % (y, ym)
    flnm_adms = adms_flnm_template % ymd12
    adms_fl_path = os.path.join(fldr_dir_adms, flnm_adms)
    return date, fldr_dir_adms, adms_fl_path


adms_dir_template = '.../fcstout/%s/%s'
adms_flnm_template = '%s-adms-fcst.nc'
