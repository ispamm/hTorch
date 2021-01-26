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
