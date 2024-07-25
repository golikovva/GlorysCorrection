import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3DLight(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, factor=2):
        super(UNet3DLight, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        bi_factor = 2 if bilinear else 1

        self.inc = (DoubleConv3D(n_channels, 128 // factor))
        self.down1 = (Down3D(128 // factor, 256 // factor))
        # self.down2 = (Down3D(256 // factor, 256 // factor))
        self.down3 = (Down3D(256 // factor, 512 // factor))
        self.down4 = (Down3D(512 // factor, 1024 // factor // bi_factor))

        self.up1 = (Up3D(1024 // factor, 512 // factor // bi_factor, bilinear))
        # self.up2 = (Up3D(512 // factor, 256 // factor // bi_factor, bilinear))
        self.up3 = (Up3D(512 // factor, 256 // factor // bi_factor, bilinear))
        self.up4 = (Up3D(256 // factor, 128 // factor, bilinear))
        self.outc = (OutConv3D(128 // factor, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # x3 = self.down2(x2)
        x4 = self.down3(x2)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, factor=2):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv3D(n_channels, 64 // factor))
        self.down1 = (Down3D(64 // factor, 128 // factor))
        self.down2 = (Down3D(128 // factor, 256 // factor))
        self.down3 = (Down3D(256 // factor, 512 // factor))
        bi_factor = 2 if bilinear else 1
        self.down4 = (Down3D(512 // factor, 1024 // factor // bi_factor))
        self.up1 = (Up3D(1024 // factor, 512 // factor // bi_factor, bilinear))
        self.up2 = (Up3D(512 // factor, 256 // factor // bi_factor, bilinear))
        self.up3 = (Up3D(256 // factor, 128 // factor // bi_factor, bilinear))
        self.up4 = (Up3D(128 // factor, 64 // factor, bilinear))
        self.outc = (OutConv3D(64 // factor, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CZHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
