import torch
import numpy as np
import warnings
import functools


HANDLEdFUNCTIONS = {}

def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLEdFUNCTIONS[torch_function] = func
        return func
    return decorator


# ----------------------------------- getters ---------------------------------------------

@implements(torch.Tensor.shape.__get__)
def shape(self):
    """
    Gets standard shape
    """
    return torch.Tensor.shape.__get__(self.q)

@implements(torch.Tensor.T.__get__)
def T(self):
    return self.q.t()

@implements(torch.Tensor.requires_grad.__get__)
def get_requires_grad(self):
    return self.q.requires_grad

@implements(torch.Tensor.grad.__get__)
def get_grad(self):
    return self.q.grad

@implements(torch.Tensor.__getitem__)
def get_item(self, idx):
    return torch.Tensor.__getitem__(self.q, idx)

# ----------------------------------- setters -------------------------------------------------

@implements(torch.Tensor.requires_grad.__set__)
def set_requires_grad(self, requires_grad):
    self.q.requires_grad = requires_grad

@implements(torch.Tensor.grad.__set__)
def set_grad(self, grad):
    self.q.grad = grad
    
# ----------------------------------- general ------------------------------------------------

@implements(torch.Tensor.dim)
def dim(self):
    return self.q.dim()


@implements(torch.Tensor.t)
def t(self):
    return self.q.t()

@implements(torch.chunk)
def chunk(input, *args, **kwargs):
    return torch.chunk(input.q, *args, **kwargs)

@implements(torch.Tensor.cpu)
def cpu(self):
    return self.q.cpu()

@implements(torch.Tensor.cuda)
def cuda(self):
    return self.q.cuda()

@implements(torch.Tensor.reshape)
def reshape(self, *args, **kwargs):
    return self.q.reshape(*args, **kwargs)

@implements(torch.Tensor.view)
def view(self, *args, **kwargs):
    return self.q.view(*args, **kwargs)

@implements(torch.allclose)
def allclose(input1, input2, *args, **kwargs):
    if all(isinstance(i, QuaternionTensor) for i in [input1, input2]):
        return torch.allclose(input1.q, input2.q)
    else:
        return torch.allclose(input1.q, input2)

@implements(torch.cat)
def cat(input, *args, **kwargs):
    return torch.cat(input, *args, **kwargs)

@implements(torch.stack)
def stack(input, *args, **kwargs):
    return torch.cat(input, *args, **kwargs)

@implements(torch.Tensor.neg)
def neg(input):
    return input.__class__(-input.q)

# -----------------------------activation functions ------------------------------------------

@implements(torch.nn.functional.relu)
def relu(input, *args, **kwargs):
    return torch.nn.functional.relu(input.q, *args, **kwargs)


# ----------------------------------- conj ---------------------------------------------------

@implements(torch.Tensor.conj)
def conj(self):
    """
    Quaternion conjugate
    """
    if len(self.shape) > 1:
        con = torch.cat([self.a, -self.b, -self.c, -self.d], 1)
    else:
        con = [self.a, -self.b, -self.c, -self.d]

    return self.__class__(con, False)

@implements(torch.conj)
def conj(input):
    return input.conj()

# ----------------------------------- inverse ------------------------------------------------

@implements(torch.Tensor.inverse)
def inverse(self):
    """
    Quaternion inverse
    """
    if len(self.shape) > 1:
        inverse = self.conj() / self.sq_norm()
    else:
        inverse = self.conj() / self.sq_norm()
    return inverse

@implements(torch.inverse)
def inverse(input):
    return input.inverse()

# ----------------------------------- min ------------------------------------------------

@implements(torch.Tensor.min)
def min(self):
    return torch.min(self.q)

@implements(torch.min)
def min(input):
    return input.min()

# ----------------------------------- max ------------------------------------------------

@implements(torch.Tensor.max)
def max(self):
    return torch.max(self.q)

@implements(torch.max)
def max(input):
    return input.max()

# ----------------------------------- sum ------------------------------------------------

@implements(torch.Tensor.sum)
def sum(self, *args, **kwargs):
    return torch.sum(self.q, *args, **kwargs)

@implements(torch.sum)
def sum(input, *args, **kwargs):
    return input.sum(*args, **kwargs)

# ----------------------------------- mean ------------------------------------------------

@implements(torch.Tensor.mean)
def mean(self, *args, **kwargs):
    return torch.mean(self.q, *args, **kwargs)

@implements(torch.mean)
def mean(input, *args, **kwargs):
    return input.mean(*args, **kwargs)

# ----------------------------------- norm ------------------------------------------------

@implements(torch.Tensor.norm)
def norm(self, *args, **kwargs):
    """
    Quaternion (non-squared) norm.
    """
    return torch.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2)

@implements(torch.linalg.norm)
def norm(input, *args, **kwargs):
    return input.norm(*args, **kwargs)

# ----------------------------------- add ------------------------------------------------

@implements(torch.Tensor.add)
def add(self, other):
    """
    Standard addition but only adds the other tensor
    to the real part if it has 1/4 of the channels.
    """
    if np.prod(self.shape) != 4:
        if isinstance(other, torch.Tensor):
            if other.__class__.__name__ == "QuaternionTensor":
                other = other.q
            if other.dim() > 1 and self.dim() > 1:
                if other.shape[1] * 4 == self.shape[1]:
                    out = torch.cat([self.a + other, self.b, self.c, self.d], 1)
                    if self.real_tensor:
                        out = real_repr(out)
                else:
                    out = self.q + other                    
            else:
                out = self.q + other
        else:
            out = self.q + other
                            
    else:
        print(self.shape)
        if other.__class__.__name__ == "QuaternionTensor":
            out = self.q + other.q        
        elif isinstance(other, int):
            out = [self.a + other, self.b, self.c, self.d]
        elif len(other) == 1:
            out = [self.a + other, self.b, self.c, self.d]
        else:
            out = self.q + other
        

    return self.__class__(out)

@implements(torch.add)
def add(input1, input2):
    return input1.add(input2)

# ----------------------------------- mul ------------------------------------------------

@implements(torch.Tensor.mul)
def mul(self, other):
    """
    Product of two quaternions, called "Hamilton product".
    Using the basis product's rules and the distributive rule 
    for two quaternions q1 = a1 + b1*i + c1*j + d1*k and
    q2 = a2 + b2*i + c2*j + d2*k we get:

    anew = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    bnew = (a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2) i
    cnew = (a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2) j
    dnew = (a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2) k

    It broadcastes the other C/4 tensor to C channels
    to compute the standard Kronecker product.
    """

    if isinstance(other, QuaternionTensor):
        a2, b2, c2, d2 = other.chunk()
        r = self.a * a2 - self.b * b2 - self.c * c2 - self.d * d2
        i = self.a * b2 + self.b * a2 + self.c * d2 - self.d * c2
        j = self.a * c2 - self.b * d2 + self.c * a2 + self.d * b2
        k = self.a * d2 + self.b * c2 - self.c * b2 + self.d * a2

        if self.dim() > 1 or other.dim() > 1:
            out = torch.cat([r, i, j, k], 1)
        else:
            out = [r, i, j, k]

    elif isinstance(other, torch.Tensor):
        if other.dim() > 1 and other.dim() > 1:
            if other.shape[1] * 4 == self.shape[1]:
                out = self.q * torch.cat([other]*4, 1)
            else:
                out = self.q * other
        else:
            if other.dim() == 1 and other.shape[0] == self.shape[0]:
                out = self.q * torch.stack([other]*self.shape[1], 1)
            else:
                out = self.q * other
    else:
        out = self.q * other

    return self.__class__(out)

@implements(torch.mul)
def mul(input1, input2):
    return input1.mul(input2)


# ----------------------------------- matmul ---------------------------------------------

@implements(torch.Tensor.matmul)
def matmul(self, other):
    return torch.matmul(self.q, other)

@implements(torch.matmul)
def matmul(input1, input2):
    return input1.matmul(input2)

# ----------------------------------- div ------------------------------------------------

@implements(torch.Tensor.div)
def true_div(self, other):
    """
    Quaternion division q1 * (q2)^-1.
    It broadcastes the other C/4 tensor to C channels
    to compute the standard elementwise division.
    """
    real_tensor = False

    if isinstance(other, QuaternionTensor):
        out = self * other.inverse()

    elif isinstance(other, torch.Tensor):
        if other.dim() > 1 and self.dim() > 1:
            if other.shape[1] * 4 == self.shape[1]:
                out = self.q / torch.cat([other]*4, 1)
            else:
                out = self.q / other
        else:
            if other.dim() == 1 and other.shape[0] == self.shape[0]:
                out = self.q / torch.stack([other]*self.shape[1], 1)
            else:
                out = self.q / other          
    else:
        out = self.q / other  

    return self.__class__(out)

@implements(torch.div)
def div(input1, input2):
    return input1.div(input2)

# ----------------------------------- pow ------------------------------------------------

@implements(torch.Tensor.pow)
def pow(self, n):
    """
    Quaternion power.
    """
    n = float(n)
    if self.quat_ops:
        v = self.v
        if self.dim() > 1:
            out = v / v.norm() * torch.sin(n * self.theta())
            out += torch.cos(n * self.theta())
            out *= (self.norm() ** n)

        else:
            out = v / v.norm() * torch.sin(n * self.theta())
            out += torch.cos(n * self.theta())
            out *= (self.norm() ** n)

        return self.__class__(out)
    else:
        return torch.pow(self.q, n)

@implements(torch.pow)
def div(input1, input2):
    return input1.pow(input2)

# ----------------------------------- exp ------------------------------------------------

@implements(torch.Tensor.exp)
def exp(self):
        """
        Quaternion exponential.
        """
        if self.quat_ops:
            v = self.v
            a = self.a
            v_norm = v.norm()
            exp = torch.exp(a)
            real = exp * torch.cos(v_norm)
            if self.dim() > 1:
                vector = exp * (v / v_norm) * torch.sin(v_norm)
                out = real + vector
            else:
                vector = exp * (v / v_norm) * torch.sin(v_norm)
                out = torch.cat([
                    real, vector[1].unsqueeze(0), vector[2].unsqueeze(0), vector[3].unsqueeze(0)
                ], 0)
                
            return self.__class__(out)
        
        else:
            return torch.exp(self.q)

@implements(torch.exp)
def exp(input):
    return input.exp()

# ----------------------------------- log ------------------------------------------------


@implements(torch.Tensor.log)
def log(self):
    """
    Quaternion logarithm.
    """
    if self.quat_ops:
        v = self.v
        a = self.a
        v_norm = v.norm()
        q_norm = self.q.norm()

        real = torch.log(q_norm)

        if self.dim() > 1:
            vector = (v / v_norm) * self.theta()
            out = real + vector
        else:
            vector = (v / v_norm) * self.theta()
            out = [real.unsqueeze(0), vector[1].unsqueeze(0),
                   vector[2].unsqueeze(0), vector[3].unsqueeze(0)]

        return self.__class__(out)
    else:
        return torch.log(self.q)

@implements(torch.log)
def log(input):
    return input.log()

# ----------------------------------- layers ------------------------------------------------

@implements(torch.conv1d)
def conv1d(input, *args, **kwargs):
    return torch.conv1d(input.q, *args, **kwargs)

@implements(torch.conv2d)
def conv2d(input, *args, **kwargs):
    return torch.conv2d(input.q, *args, **kwargs)

@implements(torch.conv3d)
def conv3d(input, *args, **kwargs):
    return torch.conv3d(input.q, *args, **kwargs)

@implements(torch.conv_transpose1d)
def conv_transpose1d(input, *args, **kwargs):
    return torch.conv_transpose1d(input.q, *args, **kwargs)

@implements(torch.conv_transpose2d)
def conv_transpose2d(input, *args, **kwargs):
    return torch.conv_transpose2d(input.q, *args, **kwargs)

@implements(torch.conv_transpose3d)
def conv_transpose3d(input, *args, **kwargs):
    return torch.conv_transpose3d(input.q, *args, **kwargs)

@implements(torch.nn.functional.batch_norm)
def bn(input, *args, **kwargs):
    return torch.nn.functional.batch_norm(input.q, *args, **kwargs)

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ---------------------------- QuaternionTensor ------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

def get_parts(q):
    """
    Divides input tensor in real and imaginary parts.
    
    @type q: torch.Tensor
    @type dim: int
    """
    
    a, b, c, d = torch.chunk(q, 4, 1)

    return a.transpose(1, 0),\
           b.transpose(1, 0),\
           c.transpose(1, 0),\
           d.transpose(1, 0)

def check_q_type(q):
    """
    Readies the tensor for the QuaternionTensor class
    
    @type q: torch.Tensor/list/tuple
    """
    if isinstance(q, (tuple, list)):
        
        if all(isinstance(i, torch.Tensor) for i in q) == True and len(q) != 0:
            assert len(q) == 4, "Quaternion must have 4 elements."
            if all(i.dim() == 1 for i in q):
                q = torch.cat(q, 0)
            else:
                q = torch.stack(q, 1)
        else:
            q = torch.Tensor(q)
    return q

def real_repr(q):
    """
    Gets the real representation of the tensor.
    
    @type q: torch.Tensor
    """
    a, b, c, d = get_parts(q)
    
    if all((i.dim() == 1) for i in [a, b, c, d]) == True:
        a = a.view(1, 1)
        b = b.view(1, 1)
        c = c.view(1, 1)
        d = d.view(1, 1)

    return torch.cat([torch.cat([a, -b, -c, -d], dim=1),
                      torch.cat([b,  a, -d,  c], dim=1),
                      torch.cat([c,  d,  a, -b], dim=1),
                      torch.cat([d, -c,  b,  a], dim=1)], dim=0)

def real_rot_repr(q):
    """
    Gets the real rotation representation of the tensor.
    
    @type q: torch.Tensor
    """
    a, b, c, d = get_parts(q)
    
    if all((i.dim() == 1) for i in [a, b, c, d]) == True:
        a = a.view(1, 1)
        b = b.view(1, 1)
        c = c.view(1, 1)
        d = d.view(1, 1)

    row1 = torch.cat([torch.zeros_like(b)] * 4, dim=1)
    row2 = torch.cat([torch.zeros_like(b),
                      1 - 2 * (c ** 2 + d ** 2),
                      2 * (b * c - d * a),
                      2 * (b * d + c * a)], dim=1)
    row3 = torch.cat([torch.zeros_like(b),
                      2 * (b * c + d * a),
                      1 - 2 * (b ** 2 + d ** 2),
                      2 * (c * d - b * a)], dim=1)
    row4 = torch.cat([torch.zeros_like(b), 
                      2 * (b * d - c * a), 
                      2 * (c * d + b * a), 
                      1 - 2 * (b ** 2 + c ** 2)], dim=1)
    
    return torch.cat([row1, row2, row3, row4], dim=0)


class QuaternionTensor(torch.Tensor):
    """
    The class contains the common quaternion operations suited
    for the Pytorch framework subclassing its 
    "torch.Tensor". Thus it can be fed in any
    Pytorch function that accepts a torch.Tensor or can be
    used independently for other, more general, applications.    
    """
    
    @staticmethod 
    def __new__(cls, q=[], real_tensor=False, quat_ops=True, *args, **kwargs):
        
        q = check_q_type(q)
        if real_tensor:
            q = real_repr(q)
        cls.q = q
        cls.device = q.device
        return super().__new__(cls, q.cpu(), *args, **kwargs) 
    
    def __init__(self, q=[], real_tensor=False, quat_ops=True):
        super().__init__()
        """
        Init accepts incoming quaternion and immediately
        transforms it into its real represention when 
        "real_tensor" is set to True.
        
        @type q: torch.Tensor/list/tuple
        @type real_tensor: bool
        """
        
        self.real_tensor = real_tensor
        self.quat_ops = quat_ops
        q = check_q_type(q)
        
        if len(q) != 0:
            if real_tensor:
                q = real_repr(q)
            self.q = q
            self.q.grad = None

    @property
    def a(self):
        if self.dim() == 1:
            out = self.q[:self.shape[0]//4]
        else:
            out = self.q[:, :self.shape[1]//4]
        out.quat_ops = False
        return out

    @property
    def b(self):
        if self.dim() == 1:
            step = self.shape[0]//4
            out = self.q[step:step*2]
        else:
            step = self.shape[1]//4
            out = self.q[:, step:step*2]
            
        out.quat_ops = False
        return out

    @property
    def c(self):
        if self.dim() == 1:
            step = self.shape[0]//4
            out = self.q[step*2:step*3]
        else:
            step = self.shape[1]//4
            out = self.q[:, step*2:step*3]

        out.quat_ops = False
        return out
    
    @property
    def d(self):
        if self.dim() == 1:
            step = self.shape[0]//4
            out = self.q[step*3:]
        else:
            step = self.shape[1]//4
            out = self.q[:, step*3:]

        out.quat_ops = False
        return out
    
    # overrides pytorch operations
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLEdFUNCTIONS or not all(
            issubclass(t, QuaternionTensor)
            for t in types
        ):
            return NotImplemented
        return HANDLEdFUNCTIONS[func](*args, **kwargs)

    def torch(self):
        """
        Casts to standard pytorch
        """
        return self.q

    def rand(self, size):
        """
        Creates random QuaternionTensor
        in the interval [0,1)
        """
        return self.__class__(torch.rand(size))
    
    def sq_norm(self):
        """
        Quaternion squared norm.
        """
        return self.q.norm() ** 2

    @property
    def _real_repr(self):
        """
        Real representation of the quaternion.
        """
        return real_repr(self.q)

    @property
    def _real_rot_repr(self):
        """
        Real rotation representation.
        """
        return real_rot_repr(self.q)
    
    @property
    def v(self):
        """
        Vector part of the quaternion: 0 + b*i + c*j + d*k. 
        """
        if self.dim() > 1:
            out = torch.cat([torch.zeros_like(self.b), self.b, self.c, self.d], 1)
        else:
            out = [torch.zeros_like(self.b), self.b, self.c, self.d]
        
        return self.__class__(out)


    def theta(self):
        """
        Angle of the quaternion.
        """
        return torch.acos(self.a / self.norm())

    @property
    def qshape(self):
        """
        Quaternion shape
        """
        if self.real_tensor:
            return self.a.shape
        else:
            return self.shape

    def clone(self): 
        """
        General Pytorch cloning.
        """
        return self.__class__(self.q.clone())

    def to(self, device):
        """
        Sends tensor to CPU or GPU.
        """
        new_obj = QuaternionTensor(self.q)
        tempTensor = super().to(device)
        new_obj.data = tempTensor.data
        new_obj.device = device
        new_obj.requires_grad = tempTensor.requires_grad

        return new_obj 
            
    def chunk(self):
        return self.a,self.b, self.c, self.d
    
    def __add__(self, other):
        return torch.add(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __iadd_(self, other):
        return self.__class__(self + other)
    
    def __sub__(self, other):
        return torch.add(self, -other)

    def __rsub__(self, other):
        return -self.__add__(-other)
    
    def __isub__(self, other):
        return self.__class__(self - other)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __rmul__(self, other):

        if isinstance(other, torch.Tensor):
            if other.dim() > 1 and self.dim() > 1:
                if other.shape[1] * 4 == self.shape[1]:
                    out = torch.cat([other]*4, 1) * self.q
                else:
                    out = other * self.q
            else:
                if other.dim() == 1 and other.shape[0] == self.shape[0]:
                    out = other * self.q
                else:
                    out = other * self.q
        else:
            out = other * self.q

        return self.__class__(out)

    def __imul__(self, other):
        return self.__class__(self * other)
    
    def __matmul__(self, other):
        return torch.matmul(self, other)
    
    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def __truediv__(self, other):
        return torch.div(self, other)

    def __rtruediv__(self, other):
            
        if isinstance(other, torch.Tensor):
            if other.dim() > 1 and self.dim() > 1: 
                if other.shape[1] * 4 == self.shape[1]:
                    out = torch.cat([other]*4, 1) / self.q
                else:
                    out = other / self.q
            else:
                if other.dim() == 1 and other.shape[0] == self.shape[0]:
                    out = torch.stack([other]*self.shape[1], 1) / self.q
                else:
                    out = other / self.q
        else:
            out = other / self.q

        return self.__class__(out)
    
    def __itruediv__(self, other):
        return self.__class__(self / other)
    
    def __pow__(self, n):
        return torch.pow(self, n)
    
    def __len__(self):
        return len(self.q)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"real part: {self.a}\n" +\
               f"imaginary part (i): {self.b}\n" +\
               f"imaginary part (j): {self.c}\n" +\
               f"imaginary part (k): {self.d}"