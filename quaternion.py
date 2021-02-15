import torch
import numpy as np
import warnings

def get_parts(q):
    """
    Divides input tensor in real and imaginary parts.
    
    @type q: torch.Tensor
    """
    if len(q.shape) == 1:
        a, b, c, d = torch.chunk(q, 4, 0)
    else:
        a, b, c, d = torch.chunk(q, 4, 1)

    return a, b, c, d

def check_q_type(q):
    """
    Readies the tensor for the QuaternionTensor class
    formatting it in a manageable way.
    
    @type q: torch.Tensor/list/tuple
    """
    if isinstance(q, (tuple, list)):
        if all(isinstance(i, torch.Tensor) for i in q) == True:
            if all(len(i.shape) == 1 for i in q):
                q = torch.cat(q, 0)
            else:
                q = torch.cat(q, 1)
        else:
            q = torch.Tensor(q)
    return q

def real_repr(q):
    """
    Gets the real representation of the tensor.
    
    @type q: torch.Tensor
    """
    a, b, c, d = get_parts(q)
    
    if all((len(i.shape) == 1) for i in [a, b, c, d]) == True:
        a = a.view(1, 1)
        b = b.view(1, 1)
        c = c.view(1, 1)
        d = d.view(1, 1)
        
    a = a.transpose(1,0)
    b = b.transpose(1,0)
    c = c.transpose(1,0)
    d = d.transpose(1,0)

    weight = torch.cat([torch.cat([a, -b, -c, -d], dim=1),
                        torch.cat([b,  a, -d,  c], dim=1),
                        torch.cat([c,  d,  a, -b], dim=1),
                        torch.cat([d, -c,  b,  a], dim=1)], dim=0)
    return weight, a, b, c, d

def real_rot_repr(q):
    """
    Gets the real rotation representation of the tensor.
    
    @type q: torch.Tensor
    """
    a, b, c, d = get_parts(q)
    
    if all((len(i.shape) == 1) for i in [a, b, c, d]) == True:
        a = a.view(1, 1)
        b = b.view(1, 1)
        c = c.view(1, 1)
        d = d.view(1, 1)
        
    a = a.transpose(1,0)
    b = b.transpose(1,0)
    c = c.transpose(1,0)
    d = d.transpose(1,0)


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
    
    weight = torch.cat([row1, row2, row3, row4], dim=0)
    return weight, a, b, c, d


class QuaternionTensor(torch.Tensor):
    """
    The class contains the common quaternion operations suited
    for the Pytorch framework subclassing its 
    "torch.Tensor". Thus it can be fed in any
    Pytorch function that accepts a torch.Tensor or can be
    used independently for other, more general, applications.    
    """
    
    @staticmethod 
    def __new__(cls, q, real_tensor=False, *args, **kwargs):
        
        q = check_q_type(q)
        if real_tensor:
            q, _, _, _, _ = real_repr(q)
        cls.q = q
        return super().__new__(cls, q, *args, **kwargs) 

    def __init__(self, q, real_tensor=False):
        super().__init__()
        """
        Init accepts incoming quaternion and immediately
        transforms it into its real represention when 
        "real_tensor" is set to True.
        
        @type q: torch.Tensor/list/tuple
        @type real_tensor: bool
        """
        
        self.real_tensor = real_tensor
        q = check_q_type(q)
        if real_tensor:
            q, self.a, self.b, self.c, self.d = real_repr(q)
        else:
            self.a, self.b, self.c, self.d = get_parts(q)
        
        self.q = q
        self.grad = None

    @property
    def i_mul(self):
        """
        Quaternion multiplication by 0 + 1i + 0j + 0k.
        """
        return torch.cat([self.a, self.b, -self.c, -self.d], 1)

    @property
    def j_mul(self):
        """
        Quaternion multiplication by 0 + 0i + 1j + 0k.
        """
        return torch.cat([self.a, -self.b, self.c, -self.d], 1)

    @property
    def k_mul(self):
        """
        Quaternion multiplication by 0 + 0i + 0j + 1k.
        """
        return torch.cat([self.a, -self.b, -self.c, self.d], 1)

    @property
    def min(self):
        """
        Minimum of quaternion.
        """
        return torch.min(self)

    @property
    def sq_norm(self):
        """
        Quaternion squared norm.
        """
        return self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2

    @property
    def inv(self):
        """
        Quaternion inverse.
        """
        if len(self.shape) > 1:
            inverse = self.conj / self.sq_norm
        else:
            inverse = self.conj / self.sq_norm

        return self.__class__(inverse, False)
    
    @property
    def T(self):
        """
        General transpose.
        """
        zipped = torch.cat([self.a, self.b, self.c, self.d], 1)
        return zipped

    @property
    def _real_repr(self):
        """
        Real representation of the quaternion.
        """
        q, _, _, _, _ = real_repr(self.q)
        return q
    
    @property
    def _real_rot_repr(self):
        """
        Real rotation representation.
        """
        q, _, _, _, _ = real_rot_repr(self.q)
        return q
    
    @property
    def v(self):
        """
        Vector part of the quaternion: 0 + b*i + c*j + d*k. 
        """
        if len(self.shape) > 1:
            vec = torch.cat([torch.zeros_like(self.b), self.b, self.c, self.d], 1)
        else:
            vec = [0, self.b, self.c, self.d]
        return self.__class__(vec, False)

    @property
    def theta(self):
        """
        Angle of the quaternion.
        """
        return torch.acos(self.a / self.norm)

    @property
    def conj(self):
        """
        Quaternion conjugate: a - b*i - c*j - d*k.
        """
        if len(self.shape) > 1:
            con = torch.cat([self.a, -self.b, -self.c, -self.d], 1)
        else:
            con = [self.a, -self.b, -self.c, -self.d]

        return self.__class__(con, False)

    @property
    def norm(self):
        """
        Quaternion (non-squared) norm.
        """
        return torch.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2 + 1e-10)
    
    def clone(self, *args, **kwargs): 
        """
        General Pytorch cloning.
        """
        return QuaternionTensor(super().clone(*args, **kwargs))

    def to(self, *args, **kwargs):
        """
        Sends tensor to CPU or GPU.
        """
        new_obj = QuaternionTensor([], self.q)
        tempTensor = super().to(*args, **kwargs)
        new_obj.data = tempTensor.data
        new_obj.requires_grad = tempTensor.requires_grad
        
        return new_obj 
    

    def exp(self):
        """
        Quaternion exponential.
        """
        v = self.v
        a = self.a
        v_norm = v.norm
        exp = torch.clamp(torch.exp(a), 0, 1e10)
        real = exp * torch.cos(v_norm)
        if len(self.shape) > 1:
            vector = exp * (v.view(4,1) / v_norm) * torch.sin(v_norm)
            out = real + vector
        else:
            vector = exp * (v.view(4,1) / v_norm) * torch.sin(v_norm)
            out = [real, vector[1].unsqueeze(0), vector[2].unsqueeze(0), vector[3].unsqueeze(0)]

        return self.__class__(out, False)

    def log(self):
        """
        Quaternion logarithm.
        """
        v = self.v
        a = self.a
        v_norm = v.norm
        q_norm = self.norm

        real = torch.log(q_norm)

        if len(self.shape) > 1:
            vector = (v / v_norm) * self.theta
            out = real + vector
        else:
            vector = (v / v_norm) * self.theta
            out = [real, vector[1].unsqueeze(0), vector[2].unsqueeze(0), vector[3].unsqueeze(0)]

        return self.__class__(out, False)

    def chunk(self):
        """
        Returns the single parts as torch.Tensor's.
        """
        return torch.Tensor(self.a),\
               torch.Tensor(self.b),\
               torch.Tensor(self.c),\
               torch.Tensor(self.d)
                
    def __add__(self, other):
        """
        Standard addition but only adds the other tensor
        to the real part if it has 1/4 of the channels.
        """
        real_tensor = False
        
        if len(other.shape) > 1 and other.shape[1] * 4 == self.shape[1]:
            warnings.warn('Considering the N x C//4 x [...] tensor as a real (a + 0i + 0j + 0k) quaternion')
            a = self.a + other
            out = torch.cat([a, self.b, self.c, self.d], 1)
            real_tensor = True
        else:
            out = self.q + other

        return self.__class__(out, real_tensor)

    def __radd__(self, other):
        
        real_tensor = False

        if len(other.shape) > 1 and other.shape[1] * 4 == self.shape[1]:
            a = other + self.a
            out = torch.cat([a, self.b, self.c, self.d], 1)
            real_tensor = True
        else:
            out = other + self.q
        
        return self.__class__(out, real_tensor)

    def __iadd__(self, other):
        add = self + other
        return self.__class__(add.q, False)

    def __sub__(self, other):
        """
        Standard difference but only subtracts the other tensor
        to the real part if it has 1/4 of the channels.
        """
        real_tensor = False
        
        if len(other.shape) > 1 and other.shape[1] * 4 == self.shape[1]:
            warnings.warn('Considering the N x C//4 x [...] tensor as a real (a + 0i + 0j + 0k) quaternion')
            a = self.a - other
            out = torch.cat([a, self.b, self.c, self.d], 1)
            real_tensor = True
        else:
            out = self.q - other
            
        return self.__class__(out, real_tensor)

    def __rsub__(self, other):
        
        real_tensor = False
        
        if len(other.shape) > 1 and other.shape[1] * 4 == self.shape[1]:
            a = other - self.a
            out = torch.cat([a, self.b, self.c, self.d], 1)
            real_tensor = True
        else:
            out = other - self.q
        
        return self.__class__(out, real_tensor)

    def __isub__(self, other):
        sub = self - other
        return self.__class__(sub.q, False)

    def __mul__(self, other):
        """
        Product of two quaternions, called "Hamilton product".
        Using the basis product's rules and the distributive rule 
        for two quaternions q1 = a1 + b1*i + c1*j + d1*k and
        q2 = a2 + b2*i + c2*j + d2*k we get:

        a_new = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        b_new = (a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2) i
        c_new = (a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2) j
        d_new = (a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2) k
        
        It broadcastes the other C/4 tensor to C channels
        to compute the standard Kronecker product.
        """
        real_tensor = False
        
        if isinstance(other, QuaternionTensor):
            a2, b2, c2, d2 = other.chunk()
            r = self.a * a2 - self.b * b2 - self.c * c2 - self.d * d2
            i = self.a * b2 + self.b * a2 + self.c * d2 - self.d * c2
            j = self.a * c2 - self.b * d2 + self.c * a2 + self.d * b2
            k = self.a * d2 + self.b * c2 - self.c * b2 + self.d * a2

            if len(self.shape) > 1 or len(other.shape) > 1:
                out = torch.cat([r, i, j, k], 1)
            else:
                out = [r, i, j, k]
                
        elif len(other.shape) > 1 and other.shape[1] * 4 == self.shape[1]:
            warnings.warn('Considering the N x C//4 x [...] tensor as a real (a + 0i + 0j + 0k) quaternion')
            out = torch.cat([other]*4, 1) * self.q
            real_tensor = True
        else:
            out = self.q * other

        return self.__class__(out, real_tensor)

    def __rmul__(self, other):

        real_tensor = False
        
        if len(other.shape) > 1 and other.shape[1] * 4 == self.shape[1]:
            out = torch.cat([other]*4, 1) * self.q
            real_tensor = True
        else:
            out = other * self.q

        return self.__class__(out, real_tensor)

    def __imul__(self, other):
        mul = self * other
        return self.__class__(mul.q, False)

    def __truediv__(self, other):
        """
        Quaternion division q1 * (q2)^-1.
        It broadcastes the other C/4 tensor to C channels
        to compute the standard elementwise division.
        """
        real_tensor = True
            
        if isinstance(other, QuaternionTensor):
            out = self * other.inv
        
        elif len(other.shape) > 1 and other.shape[1] * 4 == self.shape[1]:
            warnings.warn('Considering the N x C//4 x [...] tensor as a real (a + 0i + 0j + 0k) quaternion')
            out = self.q / torch.cat([other.q]*4)
            real_tensor = True
        else:
            out = self.q / other            
            
        return self.__class__(out, real_tensor)

    def __rtruediv__(self, other):
        
        real_tensor = False
        
        if len(other.shape) > 1 and other.shape[1] * 4 == self.shape[1]:
            out = torch.cat([other.q]*4) / self.q
            real_tensor = True
        else:
            out = other / self.q
            
        return self.__class__(out, real_tensor)

    def __itruediv__(self, other):
        div = self / other
        return self.__class__(div.q, False)

    def __pow__(self, n):
        """
        Quaternion power.
        """
        n = float(n)
        v = self.v

        if len(self.shape) > 1:
            out = v / v.norm * torch.sin(n * self.theta)
            out.a += torch.cos(n * self.theta)
            out *= (self.norm ** n)

        else:
            out = v / v.norm * torch.sin(n * self.theta)
            out.a += torch.cos(n * self.theta)
            out *= (self.norm ** n)

        return self.__class__(out, False)