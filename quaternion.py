"""
Quaternion class
"""

import torch
import numpy
from utils import *
import torch.nn as nn
import torch.nn.functional as F


class Quaternion:

    def __init__(self, q):

        if isinstance(q, numpy.ndarray):
            q = list(q)

        if isinstance(q, Quaternion):
            self.q = q.q
            a, b, c, d = chunk(self.q)
            self.shape = self.q.shape
            self.a = a
            self.b = b
            self.c = c
            self.d = d

        elif isinstance(q, torch.Tensor) and len(q.shape) > 1:
            q = q.float()
            self.q = q
            a, b, c, d = chunk(self.q)
            self.shape = self.q.shape
            self.a = a
            self.b = b
            self.c = c
            self.d = d

        elif isinstance(q, (tuple, list)):
            q = torch.Tensor(q)
            self.q = q
            self.shape = self.q.shape
            self.a = q[0]
            self.b = q[1]
            self.c = q[2]
            self.d = q[3]

        elif isinstance(q, torch.Tensor) and len(q.shape) == 1:
            self.q = q.float()
            self.shape = self.q.shape
            self.a = q[0]
            self.b = q[1]
            self.c = q[2]
            self.d = q[3]

    @property
    def i_mul(self):
        return self.__class__(bcast(self.a, self.b, -self.c, -self.d))

    @property
    def j_mul(self):
        return self.__class__(bcast(self.a, -self.b, self.c, -self.d))

    @property
    def k_mul(self):
        return self.__class__(bcast(self.a, -self.b, -self.c, self.d))

    @property
    def min(self):
        return torch.min(self.q)

    @property
    def sq_norm(self):
        return self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2

    @property
    def inv(self):

        if len(self.shape) > 1:
            inverse = self.conj / self.sq_norm
        else:
            inverse = self.conj / self.sq_norm

        return inverse

    @property
    def v(self):

        if len(self.shape) > 1:
            vec = self.__class__(bcast(torch.zeros_like(self.b), self.b, self.c, self.d))
        else:
            vec = self.__class__([0, self.b, self.c, self.d])
        return vec

    @property
    def theta(self):
        return torch.acos(self.a / self.norm)

    @property
    def conj(self):

        if len(self.shape) > 1:
            con = self.__class__(bcast(self.a, -self.b, -self.c, -self.d))
        else:
            con = self.__class__([self.a, -self.b, -self.c, -self.d])

        return con

    @property
    def norm(self):
        return torch.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2 + 1e-10)

    def flatten(self, start_dim):
        return self.q.flatten(start_dim)

    def squeeze(self, dim=None):
        return self.__class__(torch.squeeze(self.q, dim))

    def unsqueeze(self, dim):
        return self.__class__(torch.unsqueeze(self.q, dim))

    def reshape(self, shape):
        return self.__class__(torch.reshape(self.q, shape))

    def exp(self):
        v = self.v
        a = self.a
        v_norm = v.norm
        exp = torch.clamp(torch.exp(a), 0, 1e10)

        real = exp * torch.cos(v_norm)
        if len(self.shape) > 1:
            vector = exp * (v / v_norm) * torch.sin(v_norm)
            out = real + vector
        else:
            vector = exp * (v / v_norm) * torch.sin(v_norm)
            out = [real, vector[1], vector[2], vector[3]]

        return self.__class__(out)

    def log(self):
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
            out = [real, vector[1], vector[2], vector[3]]

        return self.__class__(out)

    def chunk(self):
        return self.a, self.b, self.c, self.d

    def __neg__(self):
        return self.__class__(-self.q)

    def __getitem__(self, n):
        return self.q[n]

    def __repr__(self):
        return str(self.__class__.__name__) + "\n" + str(self.q)

    def __add__(self, other):

        if isinstance(other, Quaternion):
            out = self.q + other.q

        elif isinstance(other, (int, float)):
            out = self.q + other

        elif isinstance(other, torch.Tensor):
            if sum(other.shape) in [0, 1]:
                out = self.q + other
            elif other.shape[1] * 4 == self.shape[1]:
                a = self.a + other
                out = bcast(a, self.b, self.c, self.d)
            elif other.shape == self.shape:
                out = self.q + other
            else:
                raise ValueError()

        else:
            raise ValueError()

        return self.__class__(out)

    def __radd__(self, other):

        if isinstance(other, Quaternion):
            out = other.q + self.q

        if isinstance(other, (int, float)):
            out = other + self.q

        elif isinstance(other, torch.Tensor):
            if sum(other.shape) in [0, 1] or other.shape == self.shape:
                out = other + self.q
            elif other.shape[1] * 4 == self.shape[1]:
                a = other + self.a
                out = bcast(a, self.b, self.c, self.d)
            else:
                raise ValueError("cannot broadcast shapes")

        else:
            raise ValueError()

        return self.__class__(out)

    def __iadd__(self, other):
        add = self + other
        self.q = add.q
        self.a, self.b, self.c, self.d = add.chunk()
        return self

    def __sub__(self, other):

        if isinstance(other, Quaternion):
            out = self.q - other.q

        elif isinstance(other, (int, float)):
            out = self.q - other

        elif isinstance(other, torch.Tensor):
            if sum(other.shape) in [0, 1] or other.shape == self.shape:
                out = self.q - other
            elif other.shape[1] * 4 == self.shape[1]:
                a = self.a - other
                out = bcast(a, self.b, self.c, self.d)
            else:
                raise ValueError("cannot broadcast shapes")

        else:
            raise ValueError()

        return self.__class__(out)

    def __rsub__(self, other):

        if isinstance(other, (int, float)):
            out = other - self.q

        elif isinstance(other, torch.Tensor):
            if sum(other.shape) in [0, 1] or other.shape == self.shape:
                out = other - self.q
            elif other.shape[1] * 4 == self.shape[1]:
                a = other - self.a
                out = bcast(a, self.b, self.c, self.d)
            else:
                raise ValueError("cannot broadcast shapes")

        else:
            raise ValueError()

        return self.__class__(out)

    def __isub__(self, other):
        sub = self - other
        self.q = sub.q
        self.a, self.b, self.c, self.d = sub.chunk()
        return self

    def __mul__(self, other):
        """
        Product of two quaternions, called "Hailton product".
        Using the basis product's rules and the distributive rule 
        for two quaternions b1 = a1 + b1 i + c1 j + d1 k and
        c1 = a2 + b2 i + c2 j + d2 k we get:

        a_new = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        b_new = (a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2) i
        c_new = (a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2) j
        d_new = (a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2) k
        """

        if isinstance(other, Quaternion):
            a2, b2, c2, d2 = other.chunk()

            r = self.a * a2 - self.b * b2 - self.c * c2 - self.d * d2
            i = self.a * b2 + self.b * a2 + self.c * d2 - self.d * c2
            j = self.a * c2 - self.b * d2 + self.c * a2 + self.d * b2
            k = self.a * d2 + self.b * c2 - self.c * b2 + self.d * a2

            if len(self.shape) > 1 or len(other.shape) > 1:
                out = bcast(r, i, j, k)
            else:
                out = [r, i, j, k]

        elif isinstance(other, (int, float)):
            out = self.q * other

        elif isinstance(other, torch.Tensor):
            if sum(other.shape) in [0, 1] or other.shape == self.shape:
                out = other * self.q
            elif other.shape[1] * 4 == self.shape[1]:
                out = bcast(other) * self.q
            else:
                raise ValueError("cannot broadcast shapes")


        else:
            raise ValueError()

        return self.__class__(out)

    def __rmul__(self, other):

        if isinstance(other, (int, float)):
            out = self.q * other

        elif isinstance(other, torch.Tensor):
            if sum(other.shape) in [0, 1] or other.shape == self.shape:
                out = other * self.q
            elif other.shape[1] * 4 == self.shape[1]:
                out = bcast(other) * self.q
            else:
                raise ValueError("cannot broadcast shapes")

        else:
            raise ValueError()

        return self.__class__(out)

    def __imul__(self, other):
        mul = self * other
        self.q = mul.q
        self.a, self.b, self.c, self.d = mul.chunk()
        return self

    def __truediv__(self, other):

        if isinstance(other, Quaternion):
            out = self * other.inv

        elif isinstance(other, (int, float)):
            out = self.q / other

        elif isinstance(other, torch.Tensor):
            if sum(other.shape) in [0, 1] or other.shape == self.shape:
                out = self.q / other
            elif other.shape[1] * 4 == self.shape[1]:
                out = self.q / bcast(other)
            else:
                raise ValueError("cannot broadcast shapes")
            out = self.__class__(out)
        else:
            raise ValueError()

        return out

    def __rtruediv__(self, other):

        if isinstance(other, Quaternion):
            out = other * self.inv

        elif isinstance(other, (int, float)):
            out = other / self.q

        elif isinstance(other, torch.Tensor):
            if sum(other.shape) in [0, 1] or other.shape == self.shape:
                out = other / self.q
            elif other.shape[1] * 4 == self.shape[1]:
                out = bcast(other) / self.q
            else:
                raise ValueError("cannot broadcast shapes")
            out = self.__class__(out)

        else:
            raise ValueError()

        return out

    def __itruediv__(self, other):
        div = self / other
        self.q = div.q
        self.a, self.b, self.c, self.d = div.chunk()
        return self

    def __pow__(self, n):
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

        return out