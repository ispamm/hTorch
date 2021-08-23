import torch
import torch.nn as nn
import torch.nn.functional as F

from htorch.quaternion import *
from htorch.layers import QConv2d
from htorch.functions import QModReLU


def set_ops(quaternion):
    global conv, act, factor
    conv = QConv2d if quaternion else nn.Conv2d
    act = nn.ReLU
    factor = 4 if quaternion else 1


def conv3x3(in_planes, out_planes, stride=1):
    return conv(in_planes, out_planes, kernel_size=3, stride=stride,
                padding=1, bias=False)


class ChannelBlock(nn.Module):
    """
    Channel Block for learning channels interactions (from
    Deep Complex Networks)
    """

    def __init__(self, inplanes, planes, stride=1):
        super(ChannelBlock, self).__init__()

        self.act = act

        self.bn1 = nn.BatchNorm2d(inplanes * factor)
        self.conv1 = conv(inplanes, planes, kernel_size=1, stride=stride,
                          padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(planes * factor)
        self.conv2 = conv(planes, planes, kernel_size=1, stride=stride,
                          padding=0, bias=False)

        self.stride = stride

    def forward(self, x):
        out = self.act(self.bn1(x))
        out = self.conv1(out)

        out = self.act(self.bn2(out))
        out = self.conv2(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.act = act

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes * 4)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.act(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.act = act()
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * factor)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                          padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * factor)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * factor * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.act(self.bn1(out))

        out = self.conv2(out)
        out = self.act(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, deep_base=True):
        super(ResNet, self).__init__()

        self.act = act

        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64 // factor
            self.conv1 = conv(8 // factor, 64 // factor, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.inplanes = 128 // factor
            self.conv1 = conv3x3(8 // factor, 64 // factor, stride=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = conv3x3(64 // factor, 64 // factor)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = conv3x3(64 // factor, 128 // factor)
            self.bn3 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 // factor, layers[0])
        self.layer2 = self._make_layer(block, 128 // factor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256 // factor, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512 // factor, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 // factor * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(self.inplanes, planes * block.expansion,
                     kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * factor * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.act(self.bn2(self.conv2(x)))
            x = self.act(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, quaternion=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    set_ops(quaternion)

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, quaternion=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    set_ops(quaternion)

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, quaternion=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    set_ops(quaternion)

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_path = './initmodel/resnet50_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    return model


def resnet101(pretrained=False, quaternion=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    set_ops(quaternion)

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model_path = './initmodel/resnet101_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    return model


def resnet152(pretrained=False, quaternion=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    set_ops(quaternion)

    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model_path = './initmodel/resnet152_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    return model
