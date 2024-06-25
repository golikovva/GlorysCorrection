import datetime
import pickle
import warnings

import numpy as np
from tqdm import tqdm


class ErrorField:
    def __init__(self, shape):
        self.shape = shape
        self.error_sum = np.zeros(shape)
        self.error_count = np.zeros(shape, dtype=np.int64)
        self.timeline = {}

    def accumulate(self, error_field, date=None):
        assert error_field.shape == self.shape, \
            f'error field shape mismatch: expected {self.shape}, got {error_field.shape}'
        self.error_sum += np.nan_to_num(error_field, nan=0.0)
        self.error_count += ~np.isnan(error_field)
        if date is not None:
            if date in self.timeline:
                raise RuntimeError(f'duplicate date {date} during error accumulation')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                self.timeline[date] = np.nanmean(error_field)

    @property
    def result(self):
        result = self.error_sum / self.error_count
        result[self.error_count == 0] = np.nan
        return result

    @property
    def mean(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            result = np.nanmean(list(self.timeline.values()))
        return result


class Validator:
    def __init__(self, datasets, metrics, start_date, end_date):
        self.datasets = datasets
        self.metrics = metrics
        self.start_date = start_date
        self.end_date = end_date
        self.result = None

    def run(self, show_errors=False):
        self.result = [[[
            ErrorField(dataset.grid.shape)
            for _ in self.datasets
        ] for dataset in self.datasets
        ] for _ in self.metrics
        ]

        dates = [
            self.start_date + datetime.timedelta(days=days)
            for days in range((self.end_date - self.start_date).days)
        ]
        iterator = tqdm(dates)
        for date in iterator:
            iterator.set_description(str(date))

            # iterator.set_postfix_str('extracting data')
            data = []
            for dataset in self.datasets:
                item = None
                if date in dataset.dates_dict:
                    try:
                        item = dataset[date]
                    except Exception as exc:
                        if show_errors:
                            print(f'skipping {date} in {dataset.name} due to a error during data extraction:')
                            print(exc)
                if item is None:
                    item = np.full((1, *dataset.grid.shape), np.nan)
                data.append(item)

            # iterator.set_postfix_str('computing metrics')
            for metric_idx, metric in enumerate(self.metrics):
                for first_idx, first_item in enumerate(data):
                    for second_idx, second_item in enumerate(data):
                        if first_item is not None and second_item is not None:
                            error_field = metric(first_item, second_item)
                            self.result[metric_idx][first_idx][second_idx].accumulate(error_field, date)

    def fit_over(self, baseline_name):
        if self.result is None:
            raise RuntimeError('run the validation first')
        baseline_idx = [dataset.name for dataset in self.datasets].index(baseline_name)
        result = np.array([[[
            self._normalize_metric(
                self.result[metric_idx][first_idx][second_idx].mean,
                self.result[metric_idx][baseline_idx][second_idx].mean
            )
            for second_idx in range(len(self.datasets))
            if second_idx != baseline_idx
        ] for first_idx in range(len(self.datasets))
            if first_idx != baseline_idx
        ] for metric_idx in range(len(self.metrics))
        ])
        return result

    def save_result(self, path):
        if self.result is None:
            raise RuntimeError('nothing to save, run the validation first')
        with open(path, 'wb') as f:
            pickle.dump(self.result, f)

    def load_result(self, path):
        with open(path, 'rb') as f:
            self.result = pickle.load(f)

    @staticmethod
    def _normalize_metric(value, baseline_value):
        return (value - baseline_value) / (100 - baseline_value) * 100
