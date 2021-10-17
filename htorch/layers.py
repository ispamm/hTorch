import torch
import torch.nn as nn
import torch.nn.functional as F
from .functions import *
from .quaternion import QuaternionTensor
from torch.nn import init

Q = QuaternionTensor


#########################################################################################
#                                                                                       #
#                                QUATERNION LAYERS                                      #
#                                                                                       #
#########################################################################################


class QuaternionToReal(nn.Module):
    """
    Casts to real by its norm
    """
    
    def __init__(self, in_channels):
        super(QuaternionToReal, self).__init__()
        self.in_channels = in_channels
    
    def forward(self, x, quat_format=False):
        
        if quat_format:
            norm = x.norm()
            if len(norm.shape) == 1:
                out = Q(torch.cat([norm,*[torch.zeros_like(norm)]*3], 0))
            else:
                out = Q(torch.cat([norm,*[torch.zeros_like(norm)]*3], 1))
        else:
            out = x.norm()
            
        return out


class QConv1d(nn.Module):
    """
    Quaternion convolution 1d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, bias=True, dilation=1, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.spinor = spinor

        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)

        r, i, j, k = quaternion_weight.chunk()
        
        self.r_weight     = nn.Parameter(r)
        self.i_weight     = nn.Parameter(i)
        self.j_weight     = nn.Parameter(j)
        self.k_weight     = nn.Parameter(k)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
        if x.dim() == 5:
            x = torch.cat([*x.chunk()], 2).squeeze()

        weight = torch.cat([torch.cat([self.r_weight, -self.i_weight, -self.j_weight,  -self.k_weight], dim=0),
                            torch.cat([self.i_weight,  self.r_weight, -self.k_weight,   self.j_weight], dim=0),
                            torch.cat([self.j_weight,  self.k_weight,  self.r_weight,  -self.i_weight], dim=0),
                            torch.cat([self.k_weight, -self.j_weight,  self.i_weight,   self.r_weight], dim=0)], dim = 1)
    
        return Q(F.conv1d(x, weight.transpose(1,0), self.bias, self.stride,
                                  self.padding, self.dilation, self.groups))


    
class QConv2d(nn.Module):
    """
    Quaternion convolution 2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, bias=True, dilation=1, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QConv2d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.spinor = spinor
        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)
        
        r, i, j, k = quaternion_weight.chunk()
        
        self.r_weight     = nn.Parameter(r)
        self.i_weight     = nn.Parameter(i)
        self.j_weight     = nn.Parameter(j)
        self.k_weight     = nn.Parameter(k)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
        if x.dim() == 5:
            x = torch.cat([*x.chunk()], 2).squeeze()

        weight = torch.cat([torch.cat([self.r_weight, -self.i_weight, -self.j_weight,  -self.k_weight], dim=0),
                            torch.cat([self.i_weight,  self.r_weight, -self.k_weight,   self.j_weight], dim=0),
                            torch.cat([self.j_weight,  self.k_weight,  self.r_weight,  -self.i_weight], dim=0),
                            torch.cat([self.k_weight, -self.j_weight,  self.i_weight,   self.r_weight], dim=0)], dim = 1)
    
        return Q(F.conv2d(x, weight.transpose(1,0), self.bias, self.stride,
                                  self.padding, self.dilation, self.groups))


class QConv3d(nn.Module):
    """
    Quaternion convolution 3d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, bias=True, dilation=1, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QConv3d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.spinor = spinor

        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)

        r, i, j, k = quaternion_weight.chunk()
        
        self.r_weight     = nn.Parameter(r)
        self.i_weight     = nn.Parameter(i)
        self.j_weight     = nn.Parameter(j)
        self.k_weight     = nn.Parameter(k)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
        if x.dim() == 5:
            x = torch.cat([*x.chunk()], 2).squeeze()

        weight = torch.cat([torch.cat([self.r_weight, -self.i_weight, -self.j_weight,  -self.k_weight], dim=0),
                            torch.cat([self.i_weight,  self.r_weight, -self.k_weight,   self.j_weight], dim=0),
                            torch.cat([self.j_weight,  self.k_weight,  self.r_weight,  -self.i_weight], dim=0),
                            torch.cat([self.k_weight, -self.j_weight,  self.i_weight,   self.r_weight], dim=0)], dim = 1)
    
        return Q(F.conv2d(x, weight.transpose(1,0), self.bias, self.stride,
                                  self.padding, self.dilation, self.groups))


class QLinear(nn.Module):
    """
    Quaternion linear
    """

    def __init__(self, in_channels, out_channels, bias=True, spinor=False):
        super(QLinear, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.bias = bias
        self.spinor = spinor

        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_linear(self.in_channels, self.out_channels)
        r, i, j, k = quaternion_weight.chunk()
        
        self.r_weight     = nn.Parameter(r)
        self.i_weight     = nn.Parameter(i)
        self.j_weight     = nn.Parameter(j)
        self.k_weight     = nn.Parameter(k)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
      
        weight = torch.cat([torch.cat([self.r_weight, -self.i_weight, -self.j_weight,  -self.k_weight], dim=0),
                            torch.cat([self.i_weight,  self.r_weight, -self.k_weight,   self.j_weight], dim=0),
                            torch.cat([self.j_weight,  self.k_weight,  self.r_weight,  -self.i_weight], dim=0),
                            torch.cat([self.k_weight, -self.j_weight,  self.i_weight,   self.r_weight], dim=0)], dim = 1)
    
        return Q(F.linear(x, weight.t(), self.bias))


class QConvTranspose1d(nn.Module):
    """
    Quaternion transpose convolution 1d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type output_padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QConvTranspose1d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.spinor = spinor

        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)
        
        r, i, j, k = quaternion_weight.chunk()
        
        self.r_weight     = nn.Parameter(r)
        self.i_weight     = nn.Parameter(i)
        self.j_weight     = nn.Parameter(j)
        self.k_weight     = nn.Parameter(k)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
        if x.dim() == 5:
            x = torch.cat([*x.chunk()], 2).squeeze()

        weight = torch.cat([torch.cat([self.r_weight, -self.i_weight, -self.j_weight,  -self.k_weight], dim=0),
                            torch.cat([self.i_weight,  self.r_weight, -self.k_weight,   self.j_weight], dim=0),
                            torch.cat([self.j_weight,  self.k_weight,  self.r_weight,  -self.i_weight], dim=0),
                            torch.cat([self.k_weight, -self.j_weight,  self.i_weight,   self.r_weight], dim=0)], dim = 1)
    
        return Q(F.conv_transpose1d(x, weight.transpose(1,0), self.bias, self.stride,
                                  self.padding, self.output_padding, self.dilation, self.groups))





class QConvTranspose2d(nn.Module):
    """
    Quaternion transpose convolution 2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, dilation=1,  bias=True, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type output_padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QConvTranspose2d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.spinor = spinor

        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)
        
        r, i, j, k = quaternion_weight.chunk()
        
        self.r_weight     = nn.Parameter(r)
        self.i_weight     = nn.Parameter(i)
        self.j_weight     = nn.Parameter(j)
        self.k_weight     = nn.Parameter(k)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
        if x.dim() == 5:
            x = torch.cat([*x.chunk()], 2).squeeze()

        weight = torch.cat([torch.cat([self.r_weight, -self.i_weight, -self.j_weight,  -self.k_weight], dim=0),
                            torch.cat([self.i_weight,  self.r_weight, -self.k_weight,   self.j_weight], dim=0),
                            torch.cat([self.j_weight,  self.k_weight,  self.r_weight,  -self.i_weight], dim=0),
                            torch.cat([self.k_weight, -self.j_weight,  self.i_weight,   self.r_weight], dim=0)], dim = 1)
    
        return Q(F.conv_transpose2d(x, weight.transpose(1,0), self.bias, self.stride,
                                  self.padding, self.output_padding, self.dilation, self.groups))


class QConvTranspose3d(nn.Module):
    """
    Quaternion transpose convolution 3d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type output_padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QConvTranspose3d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.spinor = spinor

        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)
        
        r, i, j, k = quaternion_weight.chunk()
        
        self.r_weight     = nn.Parameter(r)
        self.i_weight     = nn.Parameter(i)
        self.j_weight     = nn.Parameter(j)
        self.k_weight     = nn.Parameter(k)
            
        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
        if x.dim() == 5:
            x = torch.cat([*x.chunk()], 2).squeeze()

        weight = torch.cat([torch.cat([self.r_weight, -self.i_weight, -self.j_weight,  -self.k_weight], dim=0),
                            torch.cat([self.i_weight,  self.r_weight, -self.k_weight,   self.j_weight], dim=0),
                            torch.cat([self.j_weight,  self.k_weight,  self.r_weight,  -self.i_weight], dim=0),
                            torch.cat([self.k_weight, -self.j_weight,  self.i_weight,   self.r_weight], dim=0)], dim = 1)
    
        return Q(F.conv_transpose3d(x, weight.transpose(1,0), self.bias, self.stride,
                                  self.padding, self.output_padding, self.dilation, self.groups))

# reference https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8632910
class QMaxPool2d(nn.Module):
    """
    Quaternion max pooling 2d
    """

    def __init__(self, kernel_size, stride, padding=0):
        """
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        """
        super(QMaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)

    def forward(self, x):
        c, idx = self.pool(x.norm())
        idx = torch.cat([idx]*4, 1)
        flat = x.flatten(start_dim=2)
        output = flat.gather(dim=2, index=idx.flatten(start_dim=2)).view_as(idx)

        return output

class QBatchNorm2d(nn.Module):
    """
    Quaternion batch normalization 2d
    please check whitendxd in cplx module at https://github.com/ivannz
    """

    def __init__(self,
                 in_channels,
                 affine=True,
                 training=True,
                 eps=1e-5,
                 momentum=0.9,
                 track_running_stats=True):
        """
        @type in_channels: int
        @type affine: bool
        @type training: bool
        @type eps: float
        @type momentum: float
        @type track_running_stats: bool
        """
        super(QBatchNorm2d, self).__init__()
        self.in_channels = in_channels

        self.affine = affine
        self.training = training
        self.track_running_stats = track_running_stats
        self.register_buffer('eye', torch.diag(torch.cat([torch.Tensor([eps])] * 4)).unsqueeze(0))

        if self.affine:
            self.weight = torch.nn.Parameter(torch.zeros(4, 4, in_channels))
            self.bias = torch.nn.Parameter(torch.zeros(4, in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(4, in_channels))
            self.register_buffer('running_cov', torch.zeros(in_channels, 4, 4))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_cov', None)

        self.momentum = momentum

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_cov.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[0, 0], 0.5)
            init.constant_(self.weight[1, 1], 0.5)
            init.constant_(self.weight[2, 2], 0.5)
            init.constant_(self.weight[3, 3], 0.5)

    def forward(self, x):
        x = torch.stack(torch.chunk(x, 4, 1), 1).permute(1, 0, 2, 3, 4)
        axes, d = (1, *range(3, x.dim())), x.shape[0]
        shape = 1, x.shape[2], *([1] * (x.dim() - 3))

        if self.training:
            mean = x.mean(dim=axes)
            if self.running_mean is not None:
                with torch.no_grad():
                    self.running_mean = self.momentum * self.running_mean + \
                                        (1.0 - self.momentum) * mean
        else:
            mean = self.running_mean

        x = x - mean.reshape(d, *shape)

        if self.training:
            perm = x.permute(2, 0, *axes).flatten(2, -1)
            cov = torch.matmul(perm, perm.transpose(-1, -2)) / perm.shape[-1]

            if self.running_cov is not None:
                with torch.no_grad():
                    self.running_cov = self.momentum * self.running_cov + \
                                       (1.0 - self.momentum) * cov

        else:
            cov = self.running_cov

        ell = torch.cholesky(cov + self.eye, upper=True)
        soln = torch.triangular_solve(
            x.unsqueeze(-1).permute(*range(1, x.dim()), 0, -1),
            ell.reshape(*shape, d, d)
        )

        wht = soln.solution.squeeze(-1)
        z = torch.stack(torch.unbind(wht, dim=-1), dim=0)

        if self.affine:
            weight = self.weight.view(4, 4, *shape)
            scaled = torch.stack([
                z[0] * weight[0, 0] + z[1] * weight[0, 1] + z[2] * weight[0, 2] + z[3] * weight[0, 3],
                z[0] * weight[1, 0] + z[1] * weight[1, 1] + z[2] * weight[1, 2] + z[3] * weight[1, 3],
                z[0] * weight[2, 0] + z[1] * weight[2, 1] + z[2] * weight[2, 2] + z[3] * weight[2, 3],
                z[0] * weight[3, 0] + z[1] * weight[3, 1] + z[2] * weight[3, 2] + z[3] * weight[3, 3],
            ], dim=0)
            z = scaled + self.bias.reshape(4, *shape)

        z = torch.cat(torch.chunk(z, 4, 0), 2).squeeze()

        return Q(z)
    
#########################################################################################
#                                                                                       #
#               QUATERNION AUTOGRAD LAYERS (highly experimental)                        #
#                                                                                       #
#########################################################################################
   
    
class QAutogradConv1d(nn.Module):
    """
    Quaternion autograd convolution 1d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, bias=True, dilation=1, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QAutogradConv1d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.spinor = spinor
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)
        
        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight._real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight._real_repr)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return QConvAutograd.apply(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups, self.dummy_param.device, "conv1d")
    
 
    
class QAutogradConv2d(nn.Module):
    """
    Quaternion autograd convolution 2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, bias=True, dilation=1, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QAutogradConv2d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.spinor = spinor
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)
        
        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight._real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight._real_repr)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return QConvAutograd.apply(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups, "conv2d")
    
    
class QAutogradConv3d(nn.Module):
    """
    Quaternion autograd convolution 3d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, bias=True, dilation=1, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QAutogradConv3d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation
        self.spinor = spinor
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)
        
        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight._real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight._real_repr)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return QConvAutograd.apply(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups, self.dummy_param.device, "conv3d")
    


class QAutogradLinear(nn.Module):
    """
    Quaternion autograd linear
    """

    def __init__(self, in_channels, out_channels, bias=False, spinor=False):
        super(QAutogradLinear, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.bias = bias
        self.spinor = spinor
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_linear(self.in_channels, self.out_channels)

        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight._real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight._real_repr)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return QLinearAutograd.apply(x, self.weight, self.bias,
                                   self.dummy_param.device, "linear")



class QAutogradConvTranspose1d(nn.Module):
    """
    Quaternion autograd transpose convolution 1d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type output_padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QAutogradConvTranspose1d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.spinor = spinor
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)
        
        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight._real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight._real_repr)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return QTransposeConvAutograd(x, self.weight, self.bias, self.stride,self.padding, 
                                      self.output_padding, self.groups, self.dilation, 
                                      self.dummy_param.device, "conv_transpose1d")



class QAutogradConvTranspose2d(nn.Module):
    """
    Quaternion autograd transpose convolution 2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type output_padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QAutogradConvTranspose2d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.spinor = spinor
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)
        
        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight._real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight._real_repr)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return QTransposeConvAutograd(x, self.weight, self.bias, self.stride,self.padding, 
                                      self.output_padding, self.groups, self.dilation, 
                                      self.dummy_param.device, "conv_transpose2d")



class QAutogradConvTranspose3d(nn.Module):
    """
    Quaternion autograd transpose convolution 3d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, spinor=False):
        """
        @type in_channels: int
        @type out_channels: int
        @type kernel_size: int/tuple/list
        @type stride: int/tuple/list
        @type padding: int
        @type output_padding: int
        @type groups: int
        @type bias: bool
        @type dilation: int
        @type spinor: bool
        """
        super(QAutogradConvTranspose3d, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.spinor = spinor
        self.dummy_param = nn.Parameter(torch.empty(0))

        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_conv(self.in_channels, self.out_channels,
                              kernel_size=self.kernel_size)
        
        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight._real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight._real_repr)

        if self.bias:
            bias = torch.zeros(self.out_channels*4)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return QTransposeConvAutograd(x, self.weight, self.bias, self.stride,self.padding, 
                                      self.output_padding, self.groups, self.dilation, 
                                      self.dummy_param.device, "conv_transpose3d")
