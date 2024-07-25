import torch
import torch.nn as nn


class Corrector(nn.Module):
    def __init__(self, model, n_classes=34, classes_dim=-3):
        super().__init__()
        self.n_classes = n_classes
        self.unet = model
        self.classes_dim = classes_dim

    def forward(self, x_orig, *args):
        x = x_orig
        unet_out = self.unet(x, *args)
        o_input = torch.split(x_orig, self.n_classes, dim=self.classes_dim)
        return o_input[0] + unet_out.view(*o_input[0].shape)
