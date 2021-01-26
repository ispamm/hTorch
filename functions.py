"""
Code for quaternion neural networks

Python 3.8.0
Pytorch 1.4



Based (mostly) on the following works:

    [1] Chiheb Trabelsi et al.: Deep Complex Networks 
        https://arxiv.org/abs/1705.09792.pdf
        
    [2] Chase J. Gaudet and Anthony S. Maida: Deep Quaternion Networks
        https://arxiv.org/pdf/1712.04604.pdf
        
    [3] Hao Zhang et al.: Deep Quaternion Features for Privacy Protection
        https://arxiv.org/pdf/2003.08365.pdf     
        
    [4] Dongpo Xu et al.: Quaternion Derivatives: The GHR Calculus
        https://arxiv.org/pdf/1409.8168.pdf    
        
    [5] HyeonSeok Lee and Hyo Seon Park: A Generalization Method of Partitioned Activation Function for Complex Number
        https://arxiv.org/pdf/1802.02987.pdf  
        
    [6] Nitzan Guberman: On Complex Valued Convolutional Neural Networks
        https://arxiv.org/abs/1602.09046.pdf
        
    [7] Titouan Parcollet et al.: Quaternion Convolutional Neural Networks for End-to-End Automatic Speech Recognition
        https://arxiv.org/pdf/1806.07789.pdf

and on the code of T. Parcollet: https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks
"""

from scipy.stats import chi
import numpy as np
import torch
import torch.nn.functional as F
from math import factorial
from quaternion import Quaternion
from utils import *
import math
from copy import deepcopy

Q = Quaternion


def initialize_linear(in_channels, out_channels):
    """
    Quaternion initialization
    It can be shown [7] that the variance for the magnitude is given
    as 4sigma from a chi-distribution with 4 dof's.
    The phase is uniformly initialized in [-pi, pi].
    The basis vectors are randomly initialized and normalized based on their norm.
    The whole initialization is performed considering the polar form of the quaternion

    :type in_channels: int
    :type out_channels: int
    """
    in_channels //= 4
    out_channels //= 4

    # LeCun criterion
    scale = 1 / np.sqrt(in_channels * 2)
    size_real = [in_channels, out_channels]
    size_img = [in_channels, out_channels * 3]

    img_mat = torch.Tensor(*size_img).uniform_(-1, 1)
    mat = Q(torch.cat([torch.zeros(size_real), img_mat], 1))

    mat /= bcast(mat.norm)
    phase = torch.Tensor(*size_real).uniform_(-np.pi, np.pi)
    magnitude = torch.from_numpy(chi.rvs(4, loc=0, scale=scale, size=size_real))

    r = magnitude * torch.cos(phase)
    factor = magnitude * torch.sin(phase)
    mat *= bcast(factor)
    mat += r.float()

    return mat


def initialize_conv(in_channels, out_channels, kernel_size=[2, 2]):
    """
    Quaternion initialization
    It can be shown [7] that the variance for the magnitude is given
    as 4sigma from a chi-distribution with 4 dof's.
    The phase is uniformly initialized in [-pi, pi].
    The basis vectors are randomly initialized and normalized based on their norm.
    The whole initialization is performed considering the polar form of the quaternion

    :type in_channels: int
    :type out_channels: int
    :type kernel_size: int/list/tuple
    """
    in_channels //= 4
    out_channels //= 4

    if type(kernel_size) == int:
        window = [kernel_size, kernel_size]
    elif type(kernel_size) == tuple:
        window = list(kernel_size)
    elif type(kernel_size) == list:
        window = kernel_size

    prod = window[0] * window[1]
    features_in, features_out = in_channels * prod, out_channels * prod

    # LeCun criterion
    scale = 1 / np.sqrt(features_in * 2)
    size = [in_channels, out_channels] + window
    size_img = [size[0]] + [size[1] * 3] + size[2:]

    img_mat = torch.Tensor(*size_img).uniform_(-1, 1)
    mat = Q(torch.cat([torch.zeros(size), img_mat], 1))

    mat /= bcast(mat.norm)
    phase = torch.Tensor(*size).uniform_(-np.pi, np.pi)
    magnitude = chi.rvs(4, loc=0, scale=scale, size=size)
    magnitude = torch.from_numpy(magnitude)

    r = magnitude * torch.cos(phase)
    factor = magnitude * torch.sin(phase)
    mat *= bcast(factor)
    mat += r.float()

    return mat


def initialize_bn():
    """
    Batch norm beta and gamma initialization.
    gamma is a 4x4 symmetric matrix so it only has 10 learnable params and 
    the diagonoal elements are set to 1/2 to preserve the variance [2].
    beta is initialized at 0
    """

    gamma = torch.zeros((10))
    gamma[0] = 0.5
    gamma[2] = 0.5
    gamma[5] = 0.5
    gamma[9] = 0.5
    beta = torch.zeros((4))

    return gamma, beta


def get_real_matrix(r, i, j, k):
    """
    Quaternion weight matrix.
    Quaternion weights can be seen as a real matrix obtained
    by the Hamilton product
    """

    weights = torch.cat([torch.cat([r, -i, -j, -k], dim=0),
                         torch.cat([i, r, -k, j], dim=0),
                         torch.cat([j, k, r, -i], dim=0),
                         torch.cat([k, -j, i, r], dim=0)], dim=1)
    return weights


def get_rot_matrix(r, i, j, k):
    """
    Quaternion rotation matrix.
    Quaternion rotation can be written as Rq where R is the rotation matrix:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """

    row1 = torch.cat([torch.zeros_like(i)] * 4, 0)
    row2 = torch.cat([torch.zeros_like(i), 1 - 2 * (j ** 2 + k ** 2), 2 * (i * j - k * r), 2 * (i * k + j * r)], 0)
    row3 = torch.cat([torch.zeros_like(i), 2 * (i * j + k * r), 1 - 2 * (i ** 2 + k ** 2), 2 * (j * k - i * r)], 0)
    row4 = torch.cat([torch.zeros_like(i), 2 * (i * k - j * r), 2 * (j * k + i * r), 1 - 2 * (i ** 2 + j ** 2)], 0)

    return torch.cat([row1, row2, row3, row4], 1)


def Nrelu(x, C):
    """
    Modified ReLU [3]
    """

    norm = torch.norm(x)
    div = norm / np.max([norm, C])

    return div * x


def Qsplit_relu(x):
    """
    Quaternion split ReLU
    """

    r, i, j, k = torch.chunk(x, 4, 1)
    r, i, j, k = F.relu(r), F.relu(i), F.relu(j), F.relu(k)

    return torch.cat([r, i, j, k], 1)


class Qrelu(torch.autograd.Function):
    """
    Quaternion approximated ReLU
    """

    @staticmethod
    def forward(ctx, input, lambd, alpha):
        ctx.save_for_backward(input)
        ctx.lambd = lambd
        ctx.alpha = alpha

        q = Q(input)
        theta = q.theta
        theta_vec = Q(bcast(torch.zeros_like(theta), theta, theta, theta))
        output = (lambd * alpha) / 2 * (1 - theta_vec.exp()) * (q.exp() - 1) + lambd / 2 * (1 + theta_vec.exp()) * q
        # q = Q(input)
        # q_pos = q - q.min

        # v = q_pos.v
        # n = (v.norm / (2*np.pi) - 0.5).round()
        # rescale = (1/4 + 2 * n) * np.pi
        # v = v/v.norm*rescale
        # l = v + q.a
        # h = -2*alpha*l

        # g = h.exp()
        # output = input/(1+g).q
        # print("theta_vec\n",theta_vec.exp().q,"q_exp\n", q.exp().q,"q\n", q.q)
        # print(output.q.mean())
        return output.q

    @staticmethod
    def backward(ctx, output):
        grad = None

        input, = ctx.saved_tensors
        lambd = ctx.lambd
        alpha = ctx.alpha

        q = Q(input)
        theta_vec = Q(bcast(torch.zeros_like(q.theta), q.theta, q.theta, q.theta))

        if ctx.needs_input_grad[0]:
            grad = output.clone()

            # d_exp = torch.zeros_like(q.q)

            # diff = 1e5
            # flag = False

            # eps = 1e-5
            # for i in range(100):

            #   fact = math.factorial(i)

            #   exp_q = []

            #   for j in range(i):

            #     q_ij = q**(i-j)
            #     q_j1 = q**(j-1)

            #     exp_q.append((q_ij*(q_j1)).q)

            #   if len(exp_q):
            #     res = torch.sum(torch.stack(exp_q),0)/fact

            #     old_sum = deepcopy(d_exp)

            #     d_exp += res

            #     diff = torch.norm(torch.sum(old_sum - d_exp))

            #     if diff < eps:
            #       break

            #     if i == 99:
            #       print("Failed to converge")

            # grad = (lambd*alpha)/2 * (1-theta_vec.exp()) * (d_exp) + \
            # lambd/2 * (1+theta_vec.exp())
            v_norm = q.v.norm
            exp_grad = 0.5 * (q.exp() + torch.exp(q.a) / v_norm * torch.sin(v_norm))
            grad = (lambd * alpha) / 2 * (1 - theta_vec.exp()) * exp_grad + lambd / 2 * (1 + theta_vec.exp())
            # q_pos = q - q.min

            # v = q_pos.v
            # n = (v.norm/(2*np.pi)-0.5).round()
            # c = (1/4 + 2*n)*np.pi
            # v_rescaled = v/v.norm*c            
            # l = v_rescaled + q.a
            # h = -2*a*l
            # g = h.exp()
            # f = input/(1+g)
            # dl = 1/4 + 2*c/v.norm
            # #print("dl",dl.mean())
            # grad_list = []
            # for omega in range(4):

            #     omega_q = np.array([0, 0, 0, 0])
            #     omega_q[omega] = 1
            #     omega_basis = Q(omega_q)

            #     if omega != 0:
            #         omega_q_inv = - omega_q
            #     else:
            #         omega_q_inv = omega_q
            #     omega_basis_inv = Q(omega_q_inv)
            #     df = -f * (f * omega_basis).a * omega_basis_inv
            #     #print("df",df.q.mean())
            #     mu_list = []
            #     for mu in range(4):

            #         mu_q = np.array([0, 0, 0, 0])
            #         mu_q[mu] = 1
            #         mu_basis = Q(mu_q)

            #         if mu != 0:
            #             mu_q_inv = - mu_q
            #         else:
            #             mu_q_inv = mu_q
            #         mu_basis_inv = Q(mu_q_inv)
            #         omega_inv_mu = omega_basis_inv * mu_basis
            #         mu_inv_omega = mu_basis_inv * omega_basis

            #         #print(h.norm.mean())
            #         # h_mu_inv_omega = (h * omega_inv_mu)
            #         # exp_real_h_mu_inv_omega = torch.clamp(torch.exp(h_mu_inv_omega.a),0,1e8)
            #         #print ("g",g.q.mean(),"hreal", exp_real_h_mu_inv_omega.mean(), "h",h.q.mean())

            #         # dg = (g - exp_real_h_mu_inv_omega*
            #             #   torch.cos(h_mu_inv_omega.norm*
            #             #   torch.sin(h_mu_inv_omega.theta)))\
            #             #   /(h - h_mu_inv_omega.a) * mu_inv_omega

            #         h_omega_inv_mu = (h * omega_inv_mu)
            #         d_g = torch.zeros_like(h_omega_inv_mu.q)

            #         diff = 1e5
            #         flag = False

            #         eps = 1e-5
            #         for i in range(100):

            #           fact = math.factorial(i)

            #           exp_q = []

            #           for j in range(i):

            #             q_ij = h_omega_inv_mu**(i-j)
            #             q_j1 = h_omega_inv_mu**(j-1)

            #             exp_q.append((q_ij*q_j1).q)

            #           if len(exp_q):
            #             res = torch.sum(torch.stack(exp_q),0)/fact

            #             old_sum = deepcopy(d_g)

            #             d_g += res

            #             diff = torch.norm(torch.sum(old_sum - d_g))

            #             if diff < eps:
            #               break

            #             if i == 99:
            #               print("Failed to converge")

            #         dg = - (a*2) * (omega_basis * dg * omega_basis_inv)
            #         #print("dg",dg.q.mean())
            #         dg_dl = dg * dl
            #         mu_list.append(dg_dl)

            #     sum_dg_dl = mu_list[0] + mu_list[1] + mu_list[2] + mu_list[3]
            #     df_dg_dl = df * sum_dg_dl

            #     grad_list.append(df_dg_dl)

            # grad = grad_list[0] + grad_list[1] + grad_list[2] + grad_list[3]
        # print("mean",grad.q.mean())
        # grad = torch.clamp(grad.q,-1e8,1e8)
        # print(grad.q.mean())
        return grad.q, None, None, None
