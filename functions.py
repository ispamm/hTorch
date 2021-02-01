from scipy.stats import chi
import numpy as np
import torch
import torch.nn.functional as F
from math import factorial
from quaternion import Quaternion
import math
from copy import deepcopy

Q = Quaternion

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
        scale = 1 / np.sqrt(in_channels * 2)
    elif init_mode in ["xavier","glorot"]:
        scale = 1 / np.sqrt((in_channels + out_channels) * 2)
        
    in_channels //= 4
    out_channels //= 4
        
    size_real = [in_channels, out_channels]
    size_img = [in_channels, out_channels * 3]

    img_mat = torch.Tensor(*size_img).uniform_(-1, 1)
    mat = Q(torch.cat([torch.zeros(size_real), img_mat], 1))
    mat /= torch.cat([mat.norm]*4, 1)
    
    phase = torch.Tensor(*size_real).uniform_(-np.pi, np.pi)
    magnitude = torch.from_numpy(chi.rvs(4, loc=0, scale=scale, size=size_real))

    r = magnitude * torch.cos(phase)
    factor = magnitude * torch.sin(phase)
    
    mat *= torch.cat([factor]*4, 1)
    mat += r.float()

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
        scale = 1 / np.sqrt(in_channels * prod * 2)
    elif init_mode in ["xavier","glorot"]:
        scale = 1 / np.sqrt((in_channels + out_channels) * prod * 2)
        
    in_channels //= 4
    out_channels //= 4

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
    mat /= torch.cat([mat.norm]*4, 1)
    
    phase = torch.Tensor(*size_real).uniform_(-np.pi, np.pi)
    magnitude = torch.from_numpy(chi.rvs(4, loc=0, scale=scale, size=size_real))

    r = magnitude * torch.cos(phase)
    factor = magnitude * torch.sin(phase)
    
    mat *= torch.cat([factor]*4, 1)
    mat += r.float()

    return mat