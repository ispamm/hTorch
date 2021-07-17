"""
credit: https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
"""

import pytorch_lightning as pl
from htorch.layers import QConv2d
from htorch.functions import QModReLU
import torch
import torch.nn as nn

from ..madgrad import MADGRAD
from ..loss import FocalTverskyLoss
from ..utils import f1_score
from ..crf import dense_crf_wrapper

# constants
import configparser

config = configparser.SafeConfigParser()
config.read("../constants.cfg")
LEARNING_RATE = config.getfloat("training", "learning_rate")

def set_ops(quaternion):
    global conv, act, factor
    conv = QConv2d if quaternion else nn.Conv2d
    act = QModReLU if quaternion else nn.ReLU
    factor = 4 if quaternion else 1

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        conv(in_channels, out_channels, 3, padding=1),
        act(),
        conv(out_channels, out_channels, 3, padding=1),
        act()
    )


class UNet(pl.LightningModule):
    def __init__(self, quaternion=True, n_class=10):
        super().__init__()

        set_ops(quaternion)

        self.dconv_down1 = double_conv(8 // factor, 64 // factor)

        self.dconv_down2 = double_conv(64 // factor, 128 // factor)
        self.dconv_down3 = double_conv(128 // factor, 256 // factor)
        self.dconv_down4 = double_conv(256 // factor, 512 // factor)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv((256 + 512) // factor, 256 // factor)
        self.dconv_up2 = double_conv((128 + 256) // factor, 128 // factor)
        self.dconv_up1 = double_conv((128 + 64) // factor, 64 // factor)

        self.conv_last = nn.Conv2d(64, n_class, 1)


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

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def focal_tversky_loss(self, x, y):
        loss = FocalTverskyLoss()(x, y)
        return loss

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self.forward(inputs)

        probs = torch.sigmoid(outputs).data.cpu().numpy()
        crf = np.stack(list(map(dense_crf_wrapper, zip(inputs.cpu().numpy(), probs))))
        crf = np.ascontiguousarray(crf)
        f1_crf = f1_score(torch.from_numpy(crf).to(self.device), labels)

        loss =  self.focal_tversky_loss(outputs.float(), labels.float())
        f1 = f1_score(outputs, labels)

        self.log('train_loss', loss)
        self.log('train_f1', f1)
        self.log('train_f1_crf', f1_crf)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self.forward(inputs)


        probs = torch.sigmoid(outputs).data.cpu().numpy()
        crf = np.stack(list(map(dense_crf_wrapper, zip(inputs.cpu().numpy(), probs))))
        crf = np.ascontiguousarray(crf)
        f1_crf = f1_score(torch.from_numpy(crf).to(self.device), labels)

        loss = self.focal_tversky_loss(outputs.float(), labels.float())
        f1 = f1_score(outputs, labels)

        self.log('val_loss', loss)
        self.log('val_f1_crf', f1_crf)
        self.log('val_f1', f1)

