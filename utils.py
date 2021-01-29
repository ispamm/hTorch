import torch

def chunk(x):
    if len(x.shape) == 1:
        out = torch.chunk(x,4,0)
    else:
        out = torch.chunk(x,4,1)
    return out


def bcast(*args, dim=1, n=4):
    if isinstance(args[0],torch.Tensor):
        if len(args) == 1:
            out = torch.cat([args[0]]*n, dim)
        else:
            out = torch.cat([*args], dim)
    else:

        out = args

    return out

def bstack(*args, n=4, dim=0):
    """
    Stacks tensor n times along dim axis
    or if args is multiple, concatenates given tesnsors
    """

    if len(args) == 1:
        out = torch.stack([args[0]]*n, dim)
    elif len(args) == 4:
        out = torch.stack([*args], dim)


    return out

def to_hermitian(weight):
    
    r, i, j, k = torch.chunk(weight, 4, 1)
    return get_real_matrix(r, -i, -j, -k).permute(1, 0, 2, 3)


def apply_quaternion_gradient(model):
    
    for name, parameter in zip(model.children(), model.parameters()):
        if name in ["Linear","Conv1d", "Conv2d","Conv3d",
                    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]:        
            parameter.register_hook(lambda grad: 4*to_hermitian(grad))
    
    return model
