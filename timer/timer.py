import time
from functools import wraps


def timer(func):
    @wraps(func)
    def cal_time(*args, **kwargs):
        t1 = time.time()
        ret = func(*args, **kwargs)
        t2 = time.time()
        print(f'time cost of func: {func.__name__} is {t2 - t1:.6f} sec')
        return ret
    return cal_time