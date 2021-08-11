"""
credit: https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
"""

from htorch.layers import QConv2d
from htorch.functions import QModReLU
import torch
import torch.nn as nn

from madgrad import MADGRAD
from crf import dense_crf_wrapper

# constants
import configparser

config = configparser.SafeConfigParser()
config.read("hTorch/experiments/constants.cfg")
LEARNING_RATE = config.getfloat("training", "learning_rate")


def set_ops(quaternion):
    global conv, act, factor
    conv = QConv2d if quaternion else nn.Conv2d
    act = nn.ReLU
    factor = 4 if quaternion else 1


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        conv(in_channels, out_channels, 3, padding=1), 
        nn.BatchNorm2d(out_channels),
        act(),
        conv(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        act()
    )


class UNet(nn.Module):
    def __init__(self, quaternion=True, n_class=10):
        super().__init__()

        set_ops(quaternion)
        channels = 128
        self.dconv_down1 = double_conv(8 // factor, channels // factor)

        self.dconv_down2 = double_conv(channels // factor, channels * 2 // factor)
        self.dconv_down3 = double_conv(channels * 2 // factor, channels * 4 // factor)
        self.dconv_down4 = double_conv(channels * 4 // factor, channels * 8 // factor)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv((channels * 4 + channels * 8) // factor, channels * 4 // factor)
        self.dconv_up2 = double_conv((channels * 2 + channels * 4) // factor, channels * 2 // factor)
        self.dconv_up1 = double_conv((channels * 2 + channels) // factor, channels // factor)

        self.conv_last = nn.Conv2d(channels, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out
