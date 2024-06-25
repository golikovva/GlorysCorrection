import torch


class StandardScaler:
    def __init__(self):
        self.channel_means = None
        self.channel_stddevs = None

    def apply_scaler_channel_params(self, means, stds):
        self.channel_means = means
        self.channel_stddevs = stds

    @staticmethod
    def create_permutation(tensor_ndim, dims=None):
        permutation = list(range(tensor_ndim))
        if not hasattr(dims, '__iter__'):
            dims = [dims]
        dims_to_normalize = []
        for dim in dims:
            if dim is not None:
                permutation.remove(dim)
                dims_to_normalize.append(dim)
        permutation.extend(dims_to_normalize)
        return permutation

    def transform(self, tensor, means=None, stds=None, dims=None):
        if means is None:
            means = self.channel_means
        if stds is None:
            stds = self.channel_stddevs
        permutation = self.create_permutation(tensor.ndim, dims)
        out = (tensor.permute(permutation) - means) / stds  # returns an input copy
        return out.permute(*torch.argsort(torch.tensor(permutation)))

    def inverse_transform(self, tensor, means=None, stds=None, dims=None):
        if means is None:
            means = self.channel_means
        if stds is None:
            stds = self.channel_stddevs
        permutation = self.create_permutation(tensor.ndim, dims)
        out = (tensor.permute(permutation) * stds) + means  # returns an input copy
        return out.permute(*torch.argsort(torch.tensor(permutation)))


class SeasonalStandardScaler(StandardScaler):
    def seasonal_channel_transform(self, tensor, month, batch_dim, channels_dim):
        means = self.channel_means[month]
        stds = self.channel_stddevs[month]
        tensor = self.transform(tensor, means, stds, [batch_dim, channels_dim])
        return tensor

    def seasonal_channel_inverse_transform(self, tensor, month, batch_dim, channels_dim):
        means = self.channel_means[month]
        stds = self.channel_stddevs[month]
        tensor = self.inverse_transform(tensor, means, stds, [batch_dim, channels_dim])
        return tensor
