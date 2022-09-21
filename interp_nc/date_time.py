import arrow


def gen_time_list(span_unit, datetime_beg, datetime_end, datestr_format='YYYY-MM-DD HH:mm:ss'):
    """
    :param span_unit: str, examples: 'hour', 'day' ...
    :param datetime_beg: arrow date structure or str, time list begin
    :param datetime_end: arrow date structure or str, time list end
    :param datestr_format: str, format of the input datestr if the input datetime_beg or datetime_end is string
    :return: list, a list of arrow date structure
    """
    # if datetime_beg or datetime_end is string, change it to arrow date structure
    if isinstance(datetime_beg, str):
        datetime_beg = arrow.get(datetime_beg, datestr_format)

    if isinstance(datetime_end, str):
        datetime_end = arrow.get(datetime_end, datestr_format)

    time_list = list(arrow.Arrow.range(span_unit, datetime_beg, datetime_end))
    return time_list