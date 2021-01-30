import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import re
import sys
from functions import *

def to_hermitian(weight):
    """
    applies hermitian (conjugate + tranpose) of a weight matrix
    
    @type weight: torch.Tensor
    """
    
    r, i, j, k = torch.chunk(weight, 4, 1)
    return get_real_matrix(r, -i, -j, -k).permute(1, 0, 2, 3)


def apply_quaternion_gradient(model):
    """
    hooks real-valued gradients and transforms them into one for 
    quaternion gradient descent
    
    @type model: nn.Module
    """
    
    for name, parameter in zip(model.children(), model.parameters()):
        if name in ["Linear","Conv1d", "Conv2d","Conv3d",
                    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]:        
            parameter.register_hook(lambda grad: 4*to_hermitian(grad))
    
    return model

def convert_to_quaternion(Net: nn.Module, spinor=False):
    """
    converts a real_valued initialized Network to a quaternion one
    """
    last_module = len([mod for mod in Net.children()])
    for n, (name, layer) in enumerate(Net.named_children()):
        
        layer_name = re.match("^\w+", str(layer)).group()
        if layer_name == "Linear" and n != last_module-1:
            
            params = re.findall("(?<==)\w+", str(layer))
            in_features, out_features, bias = int(params[0]), int(params[1]), bool(params[2])
            
            assert in_features % 4 == 0, "number of in_channels must be divisible by 4"
            assert out_features % 4 == 0, "number of out_channels must be divisible by 4"
            
            quaternion_weight = initialize_linear(in_features, out_features)
            r, i, j, k = quaternion_weight.chunk()
            
            if spinor:
                weights = get_rot_matrix(r, i, j, k)
            else:
                weights = get_real_matrix(r, i, j, k)
            
            getattr(Net, name).weight = nn.Parameter(weights.permute(1,0))
            
            if getattr(Net, name).bias != None:
                getattr(Net, name).bias.data.zero_()
        
        if layer_name in ["Conv1d", "Conv2d","Conv3d",
                          "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"] and n != last_module-1:
            
            params = re.findall("(?<!\w)\d+(?<=\w)", str(layer))
            in_features, out_features, kernel_size, stride = \
            int(params[0]), int(params[1]), (int(params[2]),int(params[3])) , (int(params[4]),int(params[5]))
            
            assert in_features % 4 == 0, "number of in_channels must be divisible by 4"
            assert out_features % 4 == 0, "number of out_channels must be divisible by 4"
            
            quaternion_weight = initialize_conv(in_features, out_features, kernel_size)
            r, i, j, k = quaternion_weight.chunk()
            
            if spinor:
                weights = get_rot_matrix(r, i, j, k)
            else:
                weights = get_real_matrix(r, i, j, k)
            
            getattr(Net, name).weight = nn.Parameter(weights.permute(1, 0, 2, 3))
            
            if getattr(Net, name).bias != None:
                getattr(Net, name).bias.data.zero_()
    
    return Net
                