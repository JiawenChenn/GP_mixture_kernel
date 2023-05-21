#!/usr/bin/env python3

# this code contains the mixture kernel class 

from typing import Optional

from linear_operator import to_linear_operator
from linear_operator.operators import KroneckerProductLinearOperator


import math
import torch
import torch.nn.functional as F
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import scipy

from gpytorch.kernels.kernel import Kernel

# calculate the matern kernel
def matern_kernel_same(x1,x2,nu,alpha):
    # alpha is a list here
    distance = torch.cdist(x1,x2).unsqueeze(-1) * alpha.reshape(1,1,-1)
    exp_component = torch.exp(-math.sqrt(nu * 2) * distance)
    if nu == 0.5:
        constant_component = 1
    elif nu == 1.5:
        constant_component = math.sqrt(3) * distance + 1
    elif nu == 2.5:
        constant_component = math.sqrt(5) * distance + 1 + (5.0 / 3.0 * distance**2)
    return constant_component * exp_component

# calculate the matern kernel
def matern_kernel_different(x1,x2,nu,alpha):
    # alpha is a number here
    distance = torch.cdist(x1,x2).unsqueeze(-1) * alpha.reshape(1,1,-1)
    exp_component = torch.exp(-math.sqrt(nu * 2) * distance)
    if nu == 0.5:
        constant_component = 1
    elif nu == 1.5:
        constant_component = math.sqrt(3) * distance + 1
    elif nu == 2.5:
        constant_component = math.sqrt(5) * distance + 1 + (5.0 / 3.0 * distance**2)
    return constant_component * exp_component

# this class define the mixture kernel with Matern of same smoothness(1/2)
# and Matern with different smoothness(1/2,3/2,5/2)
class Mixed_kernel_easy(Kernel):
    def __init__(
        self,
        data_covar_module_type: str,
        num_tasks: int,
        **kwargs,
    ):
        has_lengthscale = False
        super(Mixed_kernel_easy, self).__init__(**kwargs)
        self.num_tasks = num_tasks
        self.data_covar_module_type = data_covar_module_type
        # same smoothness kernel
        if data_covar_module_type == "matern_same_smoothness":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
            num_kernel = 3
            self.num_kernel = num_kernel
            self.register_parameter("mixed_weight",torch.nn.Parameter(torch.tensor((1.,1.,1.),device=device)))
            # three Matern kernel with nu=1/2
            self.matern1=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
            self.matern2=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
            self.matern3=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        # different smoothness kernel
        elif data_covar_module_type == "matern_different_smoothness":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
            num_kernel = 3
            self.num_kernel = num_kernel
             # three Matern kernel with nu=1/2,3/2,5/2
            self.register_parameter("mixed_weight",torch.nn.Parameter(torch.tensor((1.,1.,1.),device=device)))
            self.matern1=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
            self.matern2=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
            self.matern3=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        else:
            raise NotImplementedError

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # using clamp to gurantee that the mixture weight is greater than 1e-3
        mixed_weight = torch.clamp(self.mixed_weight, min=1e-3)
        mixed_weight = mixed_weight/(mixed_weight.sum())     
        # the mixture kernel is sum of three Matern kernels
        matern_sum_same = mixed_weight[0]*self.matern1(x1,x2)+mixed_weight[1]*self.matern2(x1,x2)+mixed_weight[2]*self.matern3(x1,x2)
        res = to_linear_operator(matern_sum_same)
        return res.diagonal(dim1=-1, dim2=-2) if diag else res
    
    # this function is used to calculate the true kernel given alpha, sigma and weight
    def calculate_true_cov(self, X, mixed_weight, alpha, sigma):
        if self.data_covar_module_type == "matern_same_smoothness":
            # three Matern kernels with nu=1/2, alpha, weight, sigma are length=3 tensors
            matern_sum_same = matern_kernel_same(X,X,0.5,alpha)
            mixed_weight = torch.clamp(mixed_weight, min=1e-3)
            mixed_weight = mixed_weight/(mixed_weight.sum())     
            matern_sum_same = matern_sum_same * (sigma.reshape(1,1,-1)) * (mixed_weight.reshape(1,1,-1))
            matern_sum_same = matern_sum_same.sum(axis=2)
            return matern_sum_same
        elif self.data_covar_module_type == "matern_different_smoothness":
            # three Matern kernels with nu=1/2,3/2,5/2, alpha, weight, sigma are length=3 tensors
            mixed_weight = torch.clamp(mixed_weight, min=1e-3)
            mixed_weight = mixed_weight/(mixed_weight.sum())
            m1=matern_kernel_different(X,X,0.5,alpha[0])
            m2=matern_kernel_different(X,X,1.5,alpha[1])
            m3=matern_kernel_different(X,X,2.5,alpha[2])  
            matern_sum_diff = m1 * sigma[0] * mixed_weight[0] + m2 * sigma[1] * mixed_weight[1]+m3 * sigma[2] * mixed_weight[2]
            matern_sum_diff = matern_sum_diff.sum(axis=2)
            return matern_sum_diff
        else:
            raise NotImplementedError

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks

# this class define the separable kernel with Matern 1/2
class A_matern12(Kernel):
    def __init__(
        self,
        num_tasks: int,
        **kwargs,
    ):
        has_lengthscale = False
        super(A_matern12, self).__init__(**kwargs)
        self.num_tasks = num_tasks
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        # set parameters for the A matrix
        # to gurantee that A is positive definite, we use the matrix exponential of q_upper
        self.register_parameter("q_upper",torch.nn.Parameter(torch.tensor((0.,0.,0.),device=device)))
        self.matern1=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # define log(A) using q_upper
        q_upper_matrix = torch.zeros((2, 2), device=self.q_upper.device)
        q_upper_matrix[0, 0] = self.q_upper[0]
        q_upper_matrix[0, 1] = self.q_upper[1]
        q_upper_matrix[1, 0] = self.q_upper[1]
        q_upper_matrix[1, 1] = self.q_upper[2]
        A = torch.matrix_exp(q_upper_matrix)
        # the final kernel is the kronecker product of A and Matern 1/2
        res = torch.kron(A, self.matern1(x1,x2).to_dense())
        res = to_linear_operator(res)
        return res.diagonal(dim1=-1, dim2=-2) if diag else res
    
    def calculate_true_cov(self, X, A, alpha, sigma):
        # calculate the true kernel
        m1=matern_kernel_different(X,X,0.5,alpha).squeeze(2)
        cov = torch.kron(torch.Tensor(A), m1 * sigma)
        return cov

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks


# this class define the mixture kernel with Matern of different smoothness
# that will be used in GPLVM (differnt lengthscale for different dimension)
class Mixed_kernel_GPLVM(Kernel):
    def __init__(
        self,
        data_covar_module_type: str,
        num_tasks: int,
        ard_num_dims: int,
        **kwargs,
    ):
        has_lengthscale = False
        super(Mixed_kernel_GPLVM, self).__init__(**kwargs)
        self.num_tasks = num_tasks
        self.data_covar_module_type = data_covar_module_type
        if data_covar_module_type == "matern_different_smoothness":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
            num_kernel = 3
            self.num_kernel = num_kernel
            self.register_parameter("mixed_weight",torch.nn.Parameter(torch.tensor((1.,1.,1.),device=device)))
            self.matern1=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=ard_num_dims))
            self.matern2=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5,ard_num_dims=ard_num_dims))
            self.matern3=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=ard_num_dims))
        else:
            raise NotImplementedError

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        mixed_weight = torch.clamp(self.mixed_weight, min=1e-3)
        mixed_weight = mixed_weight/(mixed_weight.sum())     
        matern_sum_same = mixed_weight[0]*self.matern1(x1,x2)+mixed_weight[1]*self.matern2(x1,x2)+mixed_weight[2]*self.matern3(x1,x2)
        res = to_linear_operator(matern_sum_same)
        return res.diagonal(dim1=-1, dim2=-2) if diag else res

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks