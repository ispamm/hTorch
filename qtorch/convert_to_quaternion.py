import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import re
import sys
from functions import *

# does not find an application yet
def apply_quaternion_gradient(model, layers):
    """
    hooks real-valued gradients and transforms them into one for 
    quaternion gradient descent
    
    @type model: nn.Module
    """
   
    for n, ((_, layer), parameter) in enumerate(zip(model.named_children(), model.parameters())):
        
        layer_name = re.match("^\w+", str(layer)).group()
        if layer_name in layers and len(parameter.shape) > 1 and n != 1: 
            parameter.register_hook(to_conj)
    
    return model

def convert_to_quaternion(Net, spinor=False):
    """
    converts a real_valued initialized Network to a quaternion one
    
    @type Net: nn.Module
    @type spinor: bool
    """
    last_module = len([mod for mod in Net.children()])
    layers = ["Linear","Conv1d", "Conv2d","Conv3d",
                    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]
    for n, (name, layer) in enumerate(Net.named_children()):
        
        layer_name = re.match("^\w+", str(layer)).group()
        if layer_name == "Linear" and n != last_module-1:
            
            params = re.findall("(?<==)\w+", str(layer))
            in_features, out_features, bias = int(params[0]), int(params[1]), bool(params[2])
            
            assert in_features % 4 == 0, "number of in_channels must be divisible by 4"
            assert out_features % 4 == 0, "number of out_channels must be divisible by 4"
            
            quaternion_weight = initialize_linear(in_features, out_features)
            
            if spinor:
                weight = quaternion_weight._real_rot_repr
            else:
                weight = quaternion_weight._real_repr
            
            getattr(Net, name).weight = nn.Parameter(weight)
            
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
            
            if spinor:
                weight = quaternion_weight._real_rot_repr
            else:
                weight = quaternion_weight._real_repr
            
            getattr(Net, name).weight = nn.Parameter(weight)
            
            if getattr(Net, name).bias != None:
                getattr(Net, name).bias.data.zero_()
    
    return Net
                