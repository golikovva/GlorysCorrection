import torch
import numpy as np


class CorrAccumulator:
    def __init__(self, start_date, time_mean_period):
        self.start_date = start_date
        self.period = time_mean_period
        self.period_range = self._get_range()
        self.period_means = np.zeros([self.period_range, 34, 181, 577])
        self.period_counts = np.zeros(self.period_range)

    def _get_month_from_i(self, days_from_start_date):
        dates = np.datetime64(self.start_date) + days_from_start_date * np.timedelta64(1, 'D')
        months = dates.astype('datetime64[M]').astype(int) % 12
        return months

    def _get_day_from_i(self, days_from_start_date):
        dates = np.datetime64(self.start_date) + days_from_start_date * np.timedelta64(1, 'D')
        days_of_year = (dates - dates.astype('datetime64[Y]')).astype(int)
        return days_of_year

    def _get_time_period_from_i(self, i):
        if self.period == 'month':
            return self._get_month_from_i(i)
        elif self.period == 'day':
            return self._get_day_from_i(i)

    def _get_range(self):
        if self.period == 'month':
            return 12
        elif self.period == 'day':
            return 366

    def accumulate_corr(self, corr, days_from_start_date):
        periods = self._get_time_period_from_i(days_from_start_date)
        ids_to_sort = periods.argsort()
        non_repeated_ids = np.where(np.diff(periods[ids_to_sort]))[0]
        non_repeated_slice = np.r_[0, non_repeated_ids + 1]
        data_summed_by_month = np.add.reduceat(corr[ids_to_sort], non_repeated_slice, axis=0)
        unique_periods, unique_counts = np.unique(periods, return_counts=True)
        self.period_means[unique_periods] += data_summed_by_month
        self.period_counts[unique_periods] += unique_counts

    def _get_mean_corrections(self):
        return self.period_means / np.expand_dims(self.period_counts, axis=tuple(range(1, self.period_means.ndim)))

    def save_correction_fields(self, path):
        res = self._get_mean_corrections()
        np.save(path, res)

    def __call__(self, data, days_from_start_date):
        periods = self._get_time_period_from_i(days_from_start_date)
        print(self.period_means[periods].shape)
        return data + self.period_means[periods]


class AccumCorrector:
    def __init__(self, corr_fields_path):
        self.corr_fields = np.load(corr_fields_path).reshape(366, 34, 181, 577)  # day, z, h, w

    def __call__(self, operative, day_of_year):
        return operative + self.corr_fields[day_of_year]

