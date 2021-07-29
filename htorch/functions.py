from scipy.stats import chi
import numpy as np
import torch
import torch.nn.functional as F
from math import factorial
from .quaternion import QuaternionTensor
import math
from copy import copy

Q = QuaternionTensor


def initialize_linear(in_channels, out_channels, init_mode="he"):
    """
    Initializes quaternion weight parameter for linear.
    It can be shown that the variance for the magnitude is given
    as 4sigma from a chi-distribution with 4 dof's.
    The phase is uniformly initialized in [-pi, pi].
    The basis vectors are randomly initialized and normalized based on their norm.
    The whole initialization is performed considering the polar form of the quaternion.

    @type in_channels: int
    @type out_channels: int
    @type init_mode: str
    """

    if init_mode == "he":
        scale = 1 / np.sqrt(in_channels * 8)
    elif init_mode in ["xavier", "glorot"]:
        scale = 1 / np.sqrt((in_channels + out_channels) * 8)

    size_real = [in_channels, out_channels]
    size_img = [in_channels, out_channels * 3]

    img_mat = torch.Tensor(*size_img).uniform_(-1, 1)
    mat = Q(torch.cat([torch.zeros(size_real), img_mat], 1))
    mat /= mat.norm()

    phase = torch.Tensor(*size_real).uniform_(-np.pi, np.pi)
    magnitude = torch.from_numpy(chi.rvs(4, loc=0, scale=scale, size=size_real)).float()

    r = magnitude * torch.cos(phase)
    factor = magnitude * torch.sin(phase)

    mat *= factor
    mat += r
    return mat


def initialize_conv(in_channels, out_channels, kernel_size=[2, 2], init_mode="he"):
    """
    Initializes quaternion weight parameter for convolution.
    
    @type in_channels: int
    @type out_channels: int
    @type kernel_size: int/list/tuple
    @type init_mode: str
    """

    prod = np.prod(kernel_size)
    if init_mode == "he":
        scale = 1 / np.sqrt(in_channels * prod * 8)
    elif init_mode in ["xavier", "glorot"]:
        scale = 1 / np.sqrt((in_channels + out_channels) * prod * 8)

    if type(kernel_size) == int:
        window = [kernel_size, kernel_size]
    elif type(kernel_size) == tuple:
        window = list(kernel_size)
    elif type(kernel_size) == list:
        window = kernel_size

    size_real = [in_channels, out_channels] + window
    size_img = [size_real[0]] + [size_real[1] * 3] + size_real[2:]

    img_mat = torch.Tensor(*size_img).uniform_(-1, 1)
    mat = Q(torch.cat([torch.zeros(size_real), img_mat], 1))
    mat /= mat.norm()

    phase = torch.Tensor(*size_real).uniform_(-np.pi, np.pi)
    magnitude = torch.from_numpy(chi.rvs(4, loc=0, scale=scale, size=size_real)).float()

    r = magnitude * torch.cos(phase)
    factor = magnitude * torch.sin(phase)

    mat *= factor
    mat += r

    return mat


###############################################  
#                                             #
# autograds yet to be efficiently implemented #
#                                             #
###############################################

class QConvAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding,
                dilation, groups, type):

        output = getattr(F, type)(input, weight, bias,
                                  stride, padding, dilation, groups)

        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output, stride, padding, dilation, groups
            )

        if ctx.needs_input_grad[1]:
            input[:, input.size(1) // 4:] = float("nan")
            grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output, stride, padding, dilation, groups
            )

            a, _b, _c, _d = torch.chunk(grad_weight_r[:, :weight.size(1) // 4], 4, 0)
            grad_weight = torch.cat([torch.cat([a, _b, _c, _d], 1),
                                     torch.cat([-_b, a, _d, -_c], 1),
                                     torch.cat([-_c, -_d, a, _b], 1),
                                     torch.cat([-_d, _c, -_b, a], 1)], 0)

        if ctx.needs_input_grad[2] and bias is not None:
            size = list(range(len(weight.shape)))
            grad_bias = grad_output.sum([size[0]] + size[2:]).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class QTransposeConvAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding,
                output_padding, dilation, groups, device, type):

        ctx.save_for_backward(input, weight.a, bias)
        if bias is not None:
            bias = bias.to(device)

        output = getattr(F, type)(input, weight.torch().to(device), bias,
                                  stride, padding, output_padding, dilation, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        grad_input_r = grad_input_bias = None

        input, weight_r, bias, = ctx.saved_tensors
        grad_output_r = grad_output[:, :grad_output.size(1) // 4]

        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            with torch.enable_grad():
                grad_input_r = torch.autograd.grad(output, weight_r, grad_output_r)

            a, _b, _c, _d = torch.chunk(grad_input_r, 0)
            grad_input = torch.cat([torch.cat([a, _b, _c, _d], dim=1),
                                    torch.cat([-_b, a, _d, -_c], dim=1),
                                    torch.cat([-_c, -_d, a, _b], dim=1),
                                    torch.cat([-_d, _c, -_b, a], dim=1)], dim=0)

        if ctx.needs_input_grad[2] and bias is not None:
            grad_input_bias = grad_output.sum(0)

        return grad_input_r, grad_input_bias


class QLinearAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, device, type):

        ctx.save_for_backward(input, weight.a, bias)
        if bias is not None:
            bias = bias.to(device)
        output = getattr(F, type)(input, weight.torch().to(device), bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        grad_input_r = grad_input_bias = None

        input, weight_r, bias, = ctx.saved_tensors
        grad_output_r = grad_output[:, :grad_output.size(1) // 4]

        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            with torch.enable_grad():
                grad_input_r = torch.autograd.grad(output, weight_r, grad_output_r)

            a, _b, _c, _d = torch.chunk(grad_input_r, 0)
            grad_input = torch.cat([torch.cat([a, _b, _c, _d], dim=1),
                                    torch.cat([-_b, a, _d, -_c], dim=1),
                                    torch.cat([-_c, -_d, a, _b], dim=1),
                                    torch.cat([-_d, _c, -_b, a], dim=1)], dim=0)

        if ctx.needs_input_grad[2] and bias is not None:
            grad_input_bias = grad_output.sum(0)

        return grad_input_r, grad_input_bias


class QModReLU(torch.nn.Module):
    """
    Quaternion ModeReLU
    """

    def __init__(self, bias=0):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.Tensor([bias]))

    def forward(self, x):
        norm = x.norm().to(x.device)
        return F.relu(norm + self.bias) * (x / norm)
