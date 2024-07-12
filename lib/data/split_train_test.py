import numpy as np
import math
from datetime import datetime


def split_dates(start_date, end_date, train_size, validation_size, test_size=None):
    days = np.arange(start_date, end_date, np.timedelta64(1, 'D'), dtype='datetime64[D]')  # .astype(datetime)
    train_end = math.ceil(len(days) * train_size)
    val_end = math.ceil(len(days) * (train_size + validation_size))
    train = days[:train_end]
    val = days[train_end:val_end]
    test = days[val_end:]
    return train, val, test


def split_dates_by_dates(start_date, end_date, train_end, validation_end):
    train_days = np.arange(start_date, train_end, np.timedelta64(1, 'D'), dtype='datetime64[D]')
    val_days = np.arange(train_end, validation_end, np.timedelta64(1, 'D'), dtype='datetime64[D]')
    test_days = np.arange(validation_end, end_date+np.timedelta64(1, 'D'), np.timedelta64(1, 'D'), dtype='datetime64[D]')
    return train_days, val_days, test_days
