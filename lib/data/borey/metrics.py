import numpy as np


def abs_error_fn(first, second, magnitude):
    diff = np.linalg.norm(first - second, axis=0)
    clipped_diff = diff.clip(max=magnitude)
    return (1 - clipped_diff / magnitude) * 100


def full_error_fn(first, second, sensitivity):
    first_norm = np.linalg.norm(first, axis=0)
    second_norm = np.linalg.norm(second, axis=0)
    diff = np.linalg.norm(first - second, axis=0)
    scale = (first_norm + second_norm).clip(min=sensitivity)
    return (1 - diff / scale) * 100


def norm_error_fn(first, second, sensitivity):
    first_norm = np.linalg.norm(first, axis=0)
    second_norm = np.linalg.norm(second, axis=0)
    diff = np.abs(first_norm - second_norm)
    scale = (first_norm + second_norm).clip(min=sensitivity)
    return (1 - diff / scale) * 100


def angle_error_fn(first, second, sensitivity):
    first_norm = np.linalg.norm(first, axis=0)
    first_normed = first / first_norm[None].clip(min=sensitivity)
    second_norm = np.linalg.norm(second, axis=0)
    second_normed = second / second_norm[None].clip(min=sensitivity)
    angle_cos = np.sum(first_normed * second_normed, axis=0).clip(min=-1, max=1)
    angle_cos[(first_norm < sensitivity) & (second_norm < sensitivity)] = np.nan
    return (1 - np.arccos(angle_cos) / np.pi) * 100


def _create_metric(name, error_fn, **params):
    def metric(first, second):
        return error_fn(first, second, **params)

    metric.name = name
    return metric


# sic metrics
sic_abs_metric = _create_metric('sic_abs', abs_error_fn, magnitude=100)  # 100% of ice concentration
sic_full_metric = _create_metric('sic_full', full_error_fn, sensitivity=0.1)  # 0.1% of ice concentration

# drift metrics
drift_abs_metric = _create_metric('drift_abs', abs_error_fn, magnitude=100)  # 100 cm/s drift speed
drift_angle_metric = _create_metric('dirft_angle', angle_error_fn, sensitivity=0.1)  # 0.1 cm/s drift speed
drift_full_metric = _create_metric('drift_full', full_error_fn, sensitivity=0.1)  # 0.1 cm/s drift speed
drift_norm_metric = _create_metric('drift_norm', norm_error_fn, sensitivity=0.1)  # 0.1 cm/s drift speed

# thick metrics
thick_abs_metric = _create_metric('thick_abs', abs_error_fn, magnitude=2.0)  # 2 m of ice thickness
thick_full_metric = _create_metric('thick_full', full_error_fn, sensitivity=0.1)  # 0.1 m of ice thickness
