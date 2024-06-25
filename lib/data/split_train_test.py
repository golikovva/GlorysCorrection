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
