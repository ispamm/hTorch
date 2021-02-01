import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import *
from quaternion import Q
from torch.nn import init

Q = Quaternion

#########################################################################################
#                                                                                       #
#                                AUTOGRAD FUNCTIONS                                     #
#                                                                                       #
#########################################################################################


class QConvAutograd(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, 
                dilation, groups, device, type):

        ctx.save_for_backward(input, weight.a, bias)
        if bias is not None:
            bias = bias.to(device)
            
        output = getattr(F, type)(input, weight.torch().to(device), bias,
                                  stride, padding, dilation, groups)       
        return output
       
    @staticmethod
    def backward(ctx, grad_output):
        
        grad_input_r = grad_input_bias = None
        
        input, weight_r, bias, = ctx.saved_tensors
        grad_output_r = grad_output[:,:grad_output.size(1)//4]
        
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            with torch.enable_grad():
                
                grad_input_r = torch.autograd.grad(output, weight_r, grad_output_r)
            
            a, _b, _c, _d = torch.chunk(grad_input_r, 0)
            grad_input = torch.cat([torch.cat([  a,   _b,   _c,   _d], dim=1),
                                    torch.cat([-_b,    a,   _d,  -_c], dim=1),
                                    torch.cat([-_c,  -_d,    a,   _b], dim=1),
                                    torch.cat([-_d,   _c,  -_b,    a], dim=1)], dim=0)
        
        if ctx.needs_input_grad[2] and bias is not None:
            
            grad_input_bias = grad_output.sum(0)
                
        return grad_input_r, grad_input_bias
    

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
        grad_output_r = grad_output[:,:grad_output.size(1)//4]
        
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            with torch.enable_grad():
                
                grad_input_r = torch.autograd.grad(output, weight_r, grad_output_r)
            
            a, _b, _c, _d = torch.chunk(grad_input_r, 0)
            grad_input = torch.cat([torch.cat([  a,   _b,   _c,   _d], dim=1),
                                    torch.cat([-_b,    a,   _d,  -_c], dim=1),
                                    torch.cat([-_c,  -_d,    a,   _b], dim=1),
                                    torch.cat([-_d,   _c,  -_b,    a], dim=1)], dim=0)
        
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
        grad_output_r = grad_output[:,:grad_output.size(1)//4]
        
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            with torch.enable_grad():
                
                grad_input_r = torch.autograd.grad(output, weight_r, grad_output_r)
            
            a, _b, _c, _d = torch.chunk(grad_input_r, 0)
            grad_input = torch.cat([torch.cat([  a,   _b,   _c,   _d], dim=1),
                                    torch.cat([-_b,    a,   _d,  -_c], dim=1),
                                    torch.cat([-_c,  -_d,    a,   _b], dim=1),
                                    torch.cat([-_d,   _c,  -_b,    a], dim=1)], dim=0)
        
        if ctx.needs_input_grad[2] and bias is not None:
            
            grad_input_bias = grad_output.sum(0)
                
        return grad_input_r, grad_input_bias

#########################################################################################
#                                                                                       #
#                                QUATERNION LAYERS                                      #
#                                                                                       #
#########################################################################################


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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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

        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight.real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight.real_repr)
        
        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return F.conv1d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


    
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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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
        
        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight.real_rot_repr.torch())
        else:
            self.weight = nn.Parameter(quaternion_weight.real_repr.torch())

        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = bias.cuda()
        else:
            self.bias = None

    def forward(self, x):
        
        return F.conv2d(x, self.weight, self.bias, self.stride,
                                     self.padding, self.dilation, self.groups)


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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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

        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight.real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight.real_repr)

        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return F.conv3d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class QLinear(nn.Module):
    """
    Quaternion linear
    """

    def __init__(self, in_channels, out_channels, bias=False, spinor=False):
        super(QLinear, self).__init__()

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
        self.out_channels = out_channels

        self.bias = bias
        self.spinor = spinor

        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_linear(self.in_channels, self.out_channels)

        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight.real_rot_repr.torch())
        else:
            self.weight = nn.Parameter(quaternion_weight.real_repr.torch())

        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return F.linear(x, self.weight, self.bias)


class QBatchNorm2d(nn.Module):
    """
    Quaternion batch normalization 2d
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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        self.affine = affine
        self.training = training
        self.register_buffer("I", torch.Tensor([1]))
        self.eps = eps
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = torch.nn.Parameter(torch.zeros(4, 4, in_channels // 4))
            self.bias = torch.nn.Parameter(torch.zeros(4, in_channels // 4))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(4, in_channels // 4))
            self.register_buffer('running_var', torch.zeros(4, 4, in_channels // 4))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.momentum = 0.1

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var[0, 0].fill_(1)
            self.running_var[1, 1].fill_(1)
            self.running_var[2, 2].fill_(1)
            self.running_var[3, 3].fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight[0, 0])
            init.ones_(self.weight[1, 1])
            init.ones_(self.weight[2, 2])
            init.ones_(self.weight[3, 3])
            init.zeros_(self.bias)

    def forward(self, x):
        assert self.in_channels == x.size(1), "channels should be the same"
        x = torch.stack(torch.chunk(x, 4, 1), 1).permute(1, 0, 2, 3, 4)
        axes, d = (1, *range(3, x.dim())), x.shape[0]
        shape = 1, x.shape[2], *([1] * (x.dim() - 3))

        if self.training:
            mean = x.mean(dim=axes)
            if self.running_mean is not None:
                with torch.no_grad():
                    self.running_mean += self.momentum * (mean - self.running_mean)
        else:
            mean = self.running_mean
        x = x - mean.reshape(d, *shape)

        if self.training:
            perm = x.permute(2, 0, *axes).flatten(2, -1)
            cov = torch.matmul(perm, perm.transpose(-1, -2)) / perm.shape[-1]
            if self.running_var is not None:
                with torch.no_grad():
                    self.running_var += self.momentum * (cov.permute(1, 2, 0) - self.running_var)

        else:
            cov = self.running_var.permute(2, 0, 1)

        eye = self.eps * torch.diag(torch.cat([self.I] * cov.size(1))).unsqueeze(0)
        ell = torch.cholesky(cov + eye, upper=True)
        soln = torch.triangular_solve(
            x.unsqueeze(-1).permute(*range(1, x.dim()), 0, -1),
            ell.reshape(*shape, d, d))

        soln = soln.solution.squeeze(-1)
        z = torch.stack(torch.unbind(soln, dim=-1), dim=0)

        if self.affine:
            shape = 1, z.shape[2], *([1] * (x.dim() - 3))

            weight = self.weight.reshape(4, 4, *shape)
            scaled = torch.stack([
                z[0] * weight[0, 0] + z[1] * weight[0, 1] + z[2] * weight[0, 2] + z[3] * weight[0, 3],
                z[0] * weight[1, 0] + z[1] * weight[1, 1] + z[2] * weight[1, 2] + z[3] * weight[1, 3],
                z[0] * weight[2, 0] + z[1] * weight[2, 1] + z[2] * weight[2, 2] + z[3] * weight[2, 3],
                z[0] * weight[3, 0] + z[1] * weight[3, 1] + z[2] * weight[3, 2] + z[3] * weight[3, 3],
            ], dim=0)
            z = scaled + self.bias.reshape(4, *shape)

        z = torch.cat(torch.chunk(z, 4, 0), 2).squeeze()

        return z


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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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
        
        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight.real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight.real_repr)

        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return F.conv_transpose1d(x, self.weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)




class QConvTranspose2d(nn.Module):
    """
    Quaternion transpose convolution 2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, dilation=1, spinor=False):
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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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
        
        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight.real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight.real_repr)

        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return F.conv_transpose2d(x, self.weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)


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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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
        
        if self.spinor:
            self.weight = nn.Parameter(quaternion_weight.real_rot_repr)
        else:
            self.weight = nn.Parameter(quaternion_weight.real_repr)

        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return F.conv_transpose3d(x, self.weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)

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
        x = Q(x)
        c, idx = self.pool(x.norm)
        idx = bcast(idx, n=4)
        flat = x.flatten(start_dim=2)
        output = flat.gather(dim=2, index=idx.flatten(start_dim=2)).view_as(idx)

        return output
    
    
#########################################################################################
#                                                                                       #
#                            QUATERNION AUTOGRAD LAYERS                                 #
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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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
            self.weight = quaternion_weight.real_rot_repr
        else:
            self.weight = quaternion_weight.real_repr

        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = bias
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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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
            self.weight = quaternion_weight.real_rot_repr
        else:
            self.weight = quaternion_weight.real_repr

        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = bias
        else:
            self.bias = None

    def forward(self, x):

        return QConvAutograd.apply(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups, self.dummy_param.device, "conv2d")
    
    
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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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
            self.weight = quaternion_weight.real_rot_repr
        else:
            self.weight = quaternion_weight.real_repr

        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = bias
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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
        self.out_channels = out_channels

        self.bias = bias
        self.spinor = spinor

        self.reset_parameters()

    def reset_parameters(self):
        
        quaternion_weight = initialize_linear(self.in_channels, self.out_channels)

        if self.spinor:
            self.weight = quaternion_weight.real_rot_repr
        else:
            self.weight = quaternion_weight.real_repr

        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return QConvAutograd.apply(x, self.weight, self.bias,
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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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
        
        if self.spinor:
            self.weight = quaternion_weight.real_rot_repr
        else:
            self.weight = quaternion_weight.real_repr

        if self.bias:
            bias = torch.zeros(self.out_channels)
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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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
        
        if self.spinor:
            self.weight = quaternion_weight.real_rot_repr
        else:
            self.weight = quaternion_weight.real_repr

        if self.bias:
            bias = torch.zeros(self.out_channels)
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

        assert in_channels % 4 == 0, "number of in_channels should be a multiple of 4"
        self.in_channels = in_channels

        assert out_channels % 4 == 0, "number of out_channels should be a multiple of 4"
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
        
        if self.spinor:
            self.weight = quaternion_weight.real_rot_repr
        else:
            self.weight = quaternion_weight.real_repr

        if self.bias:
            bias = torch.zeros(self.out_channels)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):

        return QTransposeConvAutograd(x, self.weight, self.bias, self.stride,self.padding, 
                                      self.output_padding, self.groups, self.dilation, 
                                      self.dummy_param.device, "conv_transpose3d")

