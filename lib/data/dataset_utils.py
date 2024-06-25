from torch.utils.data import Dataset
from datetime import datetime
import numpy as np


class ConcatDataset(Dataset):
    def __init__(self, *datasets, start_date=None):
        self.datasets = datasets
        if start_date is None:
            start_date = np.datetime64('2020-11-01')
        self.days = np.arange(start_date, start_date + len(self) * np.timedelta64(1, 'D'),
                              np.timedelta64(1, 'D')).astype(datetime)
        self.set_mask()

    def set_mask(self):
        op, re = tuple(d[self.days[0]] for d in self.datasets)
        self.op_mask = np.isnan(op[0])
        self.re_mask = np.isnan(re[:34])
        self.imp_mask = (self.re_mask ^ self.op_mask) * self.re_mask  # where re is nan, op is valid
        self.share_mask = np.invert((self.re_mask + self.op_mask))
        return self.op_mask, self.re_mask

    def __len__(self):
        return min(len(d) for d in self.datasets)

    def __getitem__(self, item):
        raise NotImplementedError


class ConcatI2IDataset(ConcatDataset):
    def __init__(self, *datasets):
        super().__init__(*datasets)

    def __getitem__(self, i):
        op, re = tuple(d[i.astype(datetime)] for d in self.datasets)
        # op, re = op[0], re[:34]
        op = np.nan_to_num(op[0])
        re = np.nan_to_num(re[:34])
        re[self.imp_mask] = op[self.imp_mask]  # чтобы не учиться на нулях
        return op, re, np.where(self.days == i)[0]


class ConcatS2SDataset(ConcatDataset):
    def __init__(self, *datasets, start_date=None, sequence_len=1,  use_spatiotemporal_encoding=False):
        super().__init__(*datasets, start_date=start_date)
        self.seq_len = sequence_len

    def __getitem__(self, i):
        dates = np.arange(i, i + np.timedelta64(self.seq_len, 'D')).astype(datetime)
        ops, res = [], []
        for date in dates:
            op, re = tuple(d[date] for d in self.datasets)
            op = np.nan_to_num(op[0])
            re = np.nan_to_num(re[:34])
            re[self.imp_mask] = op[self.imp_mask]  # чтобы не учиться на нулях
            ops.append(op)
            res.append(re)
        ops = np.stack(ops)
        res = np.stack(res)
        return ops, res, np.where(self.days == i)[0]


class Sampler:
    def __init__(self, days, shuffle=False):
        self.days = days
        self.shuffle = shuffle

    def __len__(self):
        return len(self.days)

    def __iter__(self):
        ids = np.arange(len(self.days))
        if self.shuffle:
            np.random.shuffle(ids)
        for i in ids:
            yield self.days[i]


def numpy_collate(batch):
    transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
    return [np.array(samples) for samples in transposed]  # Backwards compatibility.


if __name__ == '__main__':
    start_date, end_date = np.datetime64('2019-01-01'), np.datetime64('2019-01-10')
    days = np.arange(start_date, end_date, np.timedelta64(1, 'D'), dtype='datetime64[D]').astype(datetime)
    sampler = Sampler(days, shuffle=True)
    for sample in sampler:
        print(sample)
