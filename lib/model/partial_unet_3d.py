import torch
import torch.nn as nn
import torch.nn.functional as F
from .partial_conv_3d import PartialConv3d


class PartUNet3DLight(nn.Module):
    def __init__(self, n_channels, n_classes, return_mask=False, bilinear=False, factor=2):
        super(PartUNet3DLight, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.return_mask = return_mask
        bi_factor = 2 if bilinear else 1

        self.inc = (InpaintingDoubleConv3D(n_channels, 128 // factor))
        self.down1 = (InpaintingDownDoubleConv3D(128 // factor, 256 // factor))
        # self.down2 = (Down3D(256 // factor, 256 // factor))
        self.down3 = (InpaintingDownDoubleConv3D(256 // factor, 512 // factor))
        self.down4 = (InpaintingDownDoubleConv3D(512 // factor, 1024 // factor // 2))

        self.up1 = (InpaintingUp3D(1024 // factor, 512 // factor // 2))
        # self.up2 = (Up3D(512 // factor, 256 // factor // bi_factor, bilinear))
        self.up3 = (InpaintingUp3D(512 // factor, 256 // factor // 2))
        self.up4 = (InpaintingUp3D(256 // factor, 128 // factor))
        self.outc = (InpaintingOutConv3D(128 // factor, n_classes))

    def forward(self, x, mask):
        x1, mask1 = self.inc(x, mask)
        x2, mask2 = self.down1(x1, mask1)
        # x3 = self.down2(x2)
        x4, mask4 = self.down3(x2, mask2)
        x5, mask5 = self.down4(x4, mask4)
        x, mask = self.up1(x5, mask5, x4, mask4)
        # x = self.up2(x, x3)
        x, mask = self.up3(x, mask, x2, mask2)
        x, mask = self.up4(x, mask, x1, mask1)
        logits, mask = self.outc(x, mask)
        if self.return_mask:
            return logits, mask
        return logits


class InpaintingDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = PartialConv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, multi_channel=True,
                                   return_mask=True)
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = PartialConv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, multi_channel=True,
                                   return_mask=True)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x, mask_in):
        x, mask = self.conv1(x, mask_in=mask_in)
        x = self.relu(self.bn1(x))
        x, mask = self.conv2(x, mask_in=mask)
        x = self.relu(self.bn2(x))
        return x, mask


class InpaintingDownDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = PartialConv3d(in_channels, mid_channels, kernel_size=7, padding=2, stride=2, bias=False,
                                   multi_channel=True,
                                   return_mask=True)
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = PartialConv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, multi_channel=True,
                                   return_mask=True)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x, mask_in):
        x, mask = self.conv1(x, mask_in=mask_in)
        x = self.relu(self.bn1(x))
        x, mask = self.conv2(x, mask_in=mask)
        x = self.relu(self.bn2(x))
        return x, mask


class InpaintingDown3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv = InpaintingDoubleConv3D(in_channels, out_channels)

    def forward(self, x, mask_in):
        x = self.maxpool(x)
        mask_in = self.maxpool(mask_in)
        x, mask = self.conv(x, mask_in=mask_in)
        return x, mask


class InpaintingUp3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = InpaintingDoubleConv3D(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, mask1, x2, mask2):
        x1 = self.up(x1)
        mask1 = self.up(mask1)
        # input is CZHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        mask1 = F.pad(mask1, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        mask = torch.cat([mask2, mask1], dim=1)
        return self.conv(x, mask)


class InpaintingOutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InpaintingOutConv3D, self).__init__()
        self.conv = PartialConv3d(in_channels, out_channels, kernel_size=1, multi_channel=True,
                                  return_mask=True)

    def forward(self, x, mask):
        return self.conv(x, mask)
