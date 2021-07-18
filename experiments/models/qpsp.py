import pytorch_lightning as pl
from htorch.layers import QConv2d
from htorch.functions import QModReLU
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..madgrad import MADGRAD
from .qresnet import resnet50, resnet101, resnet152
from ..loss import FocalTverskyLoss
from ..utils import f1_score
from ..crf import dense_crf_wrapper

# constants
import configparser

config = configparser.SafeConfigParser()
config.read("hTorch/experiments/constants.cfg")
LEARNING_RATE = config.getfloat("training", "learning_rate")
ALPHA_AUX = config.getfloat("psp", "alpha_aux")
LAYERS = config.getint("psp", "layers")
DROPOUT = config.getint("psp", "dropout")

def set_ops(quaternion):
    global conv, act, factor
    conv = QConv2d if quaternion else nn.Conv2d
    act = QModReLU if quaternion else nn.ReLU
    factor = 4 if quaternion else 1


class PPM(torch.nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        self.act = act

        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                conv(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim * 4),
                self.act()

            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(pl.LightningModule):
    def __init__(self, quaternion=True, layers=LAYERS, bins=(1, 2, 3, 6), dropout=DROPOUT, classes=10, zoom_factor=8,
                 use_ppm=True,
                 pretrained=False, training=True):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm

        set_ops(quaternion)
        self.act = act

        if layers == 50:
            resnet = resnet50(pretrained=pretrained, quaternion=quaternion)
        elif layers == 101:
            resnet = resnet101(pretrained=pretrained, quaternion=quaternion)
        else:
            resnet = resnet152(pretrained=pretrained, quaternion=quaternion)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.conv2, resnet.bn2, resnet.conv3, resnet.bn3,
                                    resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048 // factor
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            conv(fea_dim, 512 // factor, kernel_size=5, padding=1, bias=False),
            nn.BatchNorm2d(512),
            self.act(),

            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                conv(1024 // factor, 256 // factor, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                self.act(),

                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, kernel_size=1)
            )

    def forward(self, x, y=None):

        x_size = x.size()
        h = int(x_size[2] / 8 * self.zoom_factor)
        w = int(x_size[3] / 8 * self.zoom_factor)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        if self.use_ppm:
            x = self.ppm(x)

        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        if self.training:

            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.focal_tversky_loss(x, y)
            aux_loss = self.focal_tversky_loss(aux, y)
            return x, main_loss, aux_loss
        else:
            return x

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def focal_tversky_loss(self, x, y):
        loss = FocalTverskyLoss()(x, y)
        return loss

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs, main_loss, aux_loss = self.forward(inputs, labels)

        probs = torch.sigmoid(outputs).data.cpu().numpy()
        crf = np.stack(list(map(dense_crf_wrapper, zip(inputs.cpu().numpy(), probs))))
        crf = np.ascontiguousarray(crf)
        f1_crf = f1_score(torch.from_numpy(crf).to(self.device), labels)

        loss = main_loss + ALPHA_AUX * aux_loss
        f1 = f1_score(outputs, labels)

        self.log('train_loss', loss)
        self.log('train_f1', f1)
        self.log('train_f1_crf', f1_crf)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self.forward(inputs, labels)

        probs = torch.sigmoid(outputs).data.cpu().numpy()
        crf = np.stack(list(map(dense_crf_wrapper, zip(inputs.cpu().numpy(), probs))))
        crf = np.ascontiguousarray(crf)
        f1_crf = f1_score(torch.from_numpy(crf).to(self.device), labels)

        loss = self.focal_tversky_loss(outputs.float(), labels.float())
        f1 = f1_score(outputs, labels)

        self.log('val_loss', loss)
        self.log('val_f1_crf', f1_crf)
        self.log('val_f1', f1)

