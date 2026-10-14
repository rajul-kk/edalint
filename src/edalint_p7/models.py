"""Torch ports of the CircuitNet authors' MAVI (IR drop) and GPDL (congestion) networks, without mmcv.

Source: github.com/circuitnet/CircuitNet commit 41ade1d7e913a418b4495b6b13630b26b0ff7279, routability_ir_drop_prediction/models/
mavi.py (63-199) and gpdl.py (67-171). Layer order, channel widths, padding, bias flags and initialisation follow the originals; the
state_dict key names are identical (checked by scripts/p7_dev_smoke.py against the original files, which need mmcv stubs to import).
Differences: no checkpoint-loading helper; mmcv's kaiming_init is written out (kaiming_normal_, fan_out, relu, bias 0).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ MAVI (mavi.py)
class DoubleConv3d(nn.Module):
    def __init__(self, cin, cout, cmid=None):
        super().__init__()
        cmid = cmid or cout
        self.double_conv = nn.Sequential(
            nn.Conv3d(cin, cmid, kernel_size=3, padding=(0, 1, 1), bias=False), nn.BatchNorm3d(cmid), nn.ReLU(inplace=True),
            nn.Conv3d(cmid, cout, kernel_size=3, padding=(0, 1, 1), bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv2d(nn.Module):
    def __init__(self, cin, cout, cmid=None):
        super().__init__()
        cmid = cmid or cout
        self.double_conv = nn.Sequential(
            nn.Conv2d(cin, cmid, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(cmid), nn.ReLU(inplace=True),
            nn.Conv2d(cmid, cout, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)), DoubleConv3d(cin, cout))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, cin, cout, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2d(cin, cout, cin // 2)
        else:
            self.up = nn.ConvTranspose2d(cin, cin // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2d(cin, cout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MAVI(nn.Module):
    """Input (B, 1, 24, 256, 256) -> output (B, 256, 256). The first `out_channels` depth slices (power_i, power_s, power_sca, power_all)
    are multiplied by the 4 predicted maps and summed (mavi.py 165-179)."""

    def __init__(self, in_channels=1, out_channels=4, bilinear=False):
        super().__init__()
        self.in_channels, self.out_channels, self.bilinear = in_channels, out_channels, bilinear
        self.inc = DoubleConv3d(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)
        self.init_weights()

    def forward(self, x):
        x_in = x[:, :, :self.out_channels, :, :]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4.mean(dim=2), x3.mean(dim=2))
        x = self.up2(x, x2.mean(dim=2))
        x = self.up3(x, x1.mean(dim=2))
        logits = x_in.squeeze(1) * self.outc(x)
        return torch.sum(logits, dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ------------------------------------------------------------------ GPDL (gpdl.py)
def generation_init_weights(module):
    def init_func(m):
        name = m.__class__.__name__
        if hasattr(m, 'weight') and (name.find('Conv') != -1 or name.find('Linear') != -1):
            if m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)
    module.apply(init_func)


class _conv(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, bias=True):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=bias), nn.InstanceNorm2d(cout, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cout, cout, kernel_size=k, stride=s, padding=p, bias=bias), nn.InstanceNorm2d(cout, affine=True), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.main(x)


class _upconv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.main = nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.InstanceNorm2d(cout, affine=True), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.main(x)


class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=32):
        super().__init__()
        self.c1 = _conv(in_dim, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = _conv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Sequential(nn.Conv2d(64, out_dim, 3, 1, 1), nn.BatchNorm2d(out_dim), nn.Tanh())

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.pool1(h1)
        h3 = self.c2(h2)
        h4 = self.pool2(h3)
        return self.c3(h4), h2


class Decoder(nn.Module):
    def __init__(self, out_dim=2, in_dim=32):
        super().__init__()
        self.conv1 = _conv(in_dim, 32)
        self.upc1 = _upconv(32, 16)
        self.conv2 = _conv(16, 16)
        self.upc2 = _upconv(32 + 16, 4)
        self.conv3 = nn.Sequential(nn.Conv2d(4, out_dim, 3, 1, 1), nn.Sigmoid())

    def forward(self, inp):
        feature, skip = inp
        d1 = self.conv1(feature)
        d2 = self.upc1(d1)
        d3 = self.conv2(d2)
        d4 = self.upc2(torch.cat([d3, skip], dim=1))
        return self.conv3(d4)


class GPDL(nn.Module):
    """Input (B, 3, 256, 256) -> output (B, out_channels, 256, 256) in (0, 1) (sigmoid). Authors' config: out_channels = 1."""

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = Encoder(in_dim=in_channels)
        self.decoder = Decoder(out_dim=out_channels)
        generation_init_weights(self)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def l1_loss100(pred, target):
    """utils/losses.py L1Loss: loss_weight 100 x mean absolute error (MAVI)."""
    return 100.0 * F.l1_loss(pred, target)


def mse_loss100(pred, target):
    """utils/losses.py MSELoss: loss_weight 100 x mean squared error (GPDL, RouteNet)."""
    return 100.0 * F.mse_loss(pred, target)
