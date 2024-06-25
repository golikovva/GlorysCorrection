import torch
import torch.nn as nn


class Corrector(nn.Module):
    def __init__(self, model, n_classes=34):
        super().__init__()
        self.n_classes = n_classes
        self.unet = model

    def forward(self, x_orig):
        x = x_orig
        unet_out = self.unet(x)
        o_input = torch.split(x_orig, self.n_classes, dim=-3)
        return o_input[0] + unet_out.view(*o_input[0].shape)
