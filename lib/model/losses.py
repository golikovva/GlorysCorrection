import numpy as np
import torch
import torch.nn as nn


class MaskedLoss(nn.Module):
    def __init__(self, criterion, mask, eps=1e-6, reduction='mean'):
        super().__init__()
        self.base_criterion = criterion(reduction='none')
        self.mask = mask
        self.reduction = reduction

    def to(self, device):
        self.mask = self.mask.to(device)
        return self

    def forward(self, yhat, y):
        loss = self.base_criterion(torch.masked_select(yhat, self.mask), torch.masked_select(y, self.mask))
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
