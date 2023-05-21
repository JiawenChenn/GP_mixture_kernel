import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import importlib
import sys
from numpy.linalg import eig
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.linalg import logm

sys.path.append("../../")
import mixture_kernel as mixture_kernel
importlib.reload(mixture_kernel)


plt.clf()

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# set the true value
p=2
A01 = 1.
A = np.array([[5.,A01],[A01,5.]])
alpha = torch.ones(1)
sigma = torch.ones(1)*10

lr=0.001

# define function to perform simulation using different sample size
def simulation3(n,p,lr=lr,epoch=2000,A01=A01):
    jitter = torch.tensor(0.5)
    # generate X from uniform distribution with random shift
    X = torch.Tensor(np.array([np.linspace(-10, 10,n) for i in range(p)])).reshape(n,p)
    shift = dist.Uniform(-1/(5*n), 1/(5*n)).sample(sample_shape=(n,p))
    X = X + shift
    # calculate true kernel
    kernel_true = mixture_kernel.A_matern12(num_tasks=1)
    A = np.array([[5.,A01],[A01,5.]])
    cov = kernel_true.calculate_true_cov(X = X, A = A, alpha = alpha, sigma=sigma)+jitter.expand(2*n).diag()
    try:
        # generate Y using MVN
        scale_tril = torch.linalg.cholesky(cov)
        true_distribution = MultivariateNormal(torch.zeros(X.shape[0]*2),scale_tril=scale_tril)
        Y = true_distribution.sample()
    except ValueError:
        return np.nan
    except torch._C._LinAlgError:
        print(np.linalg.eigvals(cov))
        print("scale_tril not complete")
        return np.nan
    #
    #
    class A_matern12_model(gpytorch.models.GP):
        def __init__(self):
            super(A_matern12_model, self).__init__()
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = mixture_kernel.A_matern12(num_tasks=2)
        def forward(self, x):
            jitter = torch.tensor(0.5)
            if torch.cuda.is_available():
                jitter = jitter.cuda()
            mean_x = self.mean_module(torch.cat((X,X)))
            covar_x = self.covar_module(x)+jitter.expand(2*n).diag()
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    model = A_matern12_model()
    #
    if torch.cuda.is_available():
        model = model.cuda()

    model.train()

    # Parameter initialization
    model.covar_module.matern1.base_kernel.raw_lengthscale = torch.nn.Parameter(torch.tensor(-2.2522,device=device).reshape(1,1))
    model.covar_module.matern1.raw_outputscale = torch.nn.Parameter(torch.tensor(0.5413,device=device))
    model.covar_module.q_upper = torch.nn.Parameter(torch.tensor((0.,0.,0.),device=device))

    if torch.cuda.is_available():
        X = X.cuda()
        Y = Y.cuda()

    # Use the SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    training_iterations=epoch
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model.forward(X)
        loss = -output.log_prob(Y)
        loss.backward()
        optimizer.step()
    #
    model.eval()
    model.cpu()
    # parameter estimation
    q_upper = model.covar_module.q_upper
    q_upper_matrix = torch.Tensor([[q_upper[0], q_upper[1]], [q_upper[1], q_upper[2]]])
    A = torch.matrix_exp(q_upper_matrix)

    return torch.cat((A[0,0].unsqueeze(0),A[0,1].unsqueeze(0),A[1,1].unsqueeze(0),1/model.covar_module.matern1.base_kernel.lengthscale.squeeze(0),model.covar_module.matern1.outputscale.unsqueeze(0))).detach().cpu()


torch.manual_seed(0)

# take the input as sample size
sample_size=int(sys.argv[1])

# simulation
# 100 replicates

sim0 = [simulation3(n=sample_size,p=p) for i in range(100)]
sim0 = torch.cat(sim0).reshape(-1,5)
np.savetxt('./result/s3_sim'+str(sample_size)+'.txt', sim0.numpy())

