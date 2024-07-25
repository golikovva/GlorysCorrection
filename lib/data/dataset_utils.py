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
        re[self.op_mask] = 0  # чтобы не учить, там где не надо
        return op, re, np.where(self.days == i)[0]


class ConcatS2SDataset(ConcatDataset):
    def __init__(self, *datasets, start_date=None, sequence_len=1, use_spatial_encoding=False,
                 use_temporal_encoding=False, return_mask=False):
        super().__init__(*datasets, start_date=start_date)
        self.seq_len = sequence_len
        self.use_spatial_encoding = use_spatial_encoding
        self.use_temporal_encoding = use_temporal_encoding
        self.return_mask = return_mask

    def __getitem__(self, i):
        dates = np.arange(i, i + np.timedelta64(self.seq_len, 'D')).astype(datetime)
        ops, res, masks = [], [], []
        for date in dates:
            op, re = tuple(d[date] for d in self.datasets)
            if self.return_mask:
                masks.append(~np.isnan(op[0]) * 1.)
            op = np.nan_to_num(op[0])
            re = np.nan_to_num(re[:34])
            re[self.imp_mask] = op[self.imp_mask]  # чтобы не учиться на нулях
            re[self.op_mask] = 0
            ops.append(op)
            res.append(re)

        res = np.stack(res)
        if self.use_temporal_encoding:
            encoding = get_time_encoding(i, self.datasets[0].src_grid.lon, self.datasets[0].src_grid.lat)  # todo
            ops.append(np.broadcast_to(encoding, ops[0].shape).copy())
            masks.append(masks[0])
        if self.use_spatial_encoding:
            ops.append(np.broadcast_to(self.datasets[0].src_grid.lat, ops[0].shape).copy())
            ops.append(np.broadcast_to(self.datasets[0].src_grid.lon, ops[0].shape).copy())
            masks.extend([masks[0]] * 2)
        ops = np.stack(ops)

        if self.return_mask:
            masks = np.stack(masks)
            return (ops, masks), res, np.where(self.days == i)[0]
        return (ops,), res, np.where(self.days == i)[0]


def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x


def get_time_encoding(date, lon, lat, frequency=4):
    day = (date.astype('datetime64[D]') - date.astype('datetime64[Y]')).astype(int) / 365
    if day.ndim == 0:
        day = day[None]
    assert day.ndim == 1, f'Input dates should be 0 or 1 dimensional, got {day.ndim}D'
    d1 = abs(abs(0.5 - day) - 0.5) + 0.05
    d2 = abs(abs(0.25 - day) - 0.5) + 0.05
    # альтернативное преобразование через синус
    # s1 = (np.cos(day * 2 * np.pi + np.pi) + 1) / 4 + 0.05
    # s2 = (np.sin(day * 2 * np.pi) + 1) / 4 + 0.05
    day_encoded = (np.sin(frequency * np.einsum('i,jk->ijk', d1, lon))
                   + np.sin(frequency * np.einsum('i,jk->ijk', d2, lat))) / 2
    return day_encoded


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
