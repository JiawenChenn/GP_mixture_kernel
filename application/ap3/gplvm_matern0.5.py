# this code is adopted from 
# https://docs.gpytorch.ai/en/stable/examples/045_GPLVM/Gaussian_Process_Latent_Variable_Models_with_Stochastic_Variational_Inference.html
# check the website for more explanation
import matplotlib.pylab as plt
import torch
import os
import numpy as np
from pathlib import Path

# If you are running this notebook interactively
wdir = Path(os.path.abspath('')).parent.parent
os.chdir(wdir)

# gpytorch imports
import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior

import h5py
import seaborn as sns
import sys


# Setting manual seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


dat_path = './Test_3_Pollen.h5'
dat_file = h5py.File(dat_path,'r')
y_train = dat_file['in_X'][:]
Y = torch.Tensor(y_train)
labels=torch.Tensor(dat_file['true_labs'][:])

from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from matplotlib import pyplot as plt
from tqdm.notebook import trange
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal


def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

# define the bGPLVM model
class bGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False):
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        #
        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)
        #
        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        #
        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
        #
        # Initialise X with PCA or randn
        if pca == True:
             X_init = _init_pca(Y, latent_dim) # Initialise X to PCA
        else:
             X_init = torch.nn.Parameter(torch.randn(n, latent_dim))
        #
        # LatentVariable (c)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)
        #
        super().__init__(X, q_f)
        #
        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        #######################################################
        ###### set the covariance kernel to be Matern 1/2 #####
        #######################################################
        self.covar_module = ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=latent_dim))
        #
    def forward(self, X):
        jitter = torch.tensor(1.)
        identity_matrix = torch.eye(X.shape[1])*jitter
        identity_tensor = torch.stack([identity_matrix] * X.shape[0], dim=0)
        if torch.cuda.is_available():
            identity_tensor = identity_tensor.cuda()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)+identity_tensor
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
        #
    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

# take an input
latent_dim = int(sys.argv[1])

N = len(Y)
data_dim = Y.shape[1]
n_inducing = 25
pca = True

# Model
model = bGPLVM(N, data_dim, latent_dim, n_inducing, pca=pca)

# Likelihood
likelihood = GaussianLikelihood(batch_shape=model.batch_shape)

if torch.cuda.is_available():
    Y = Y.cuda()
    model = model.cuda()
    likelihood = likelihood.cuda()


# Declaring the objective to be optimised along with optimiser
# (see models/latent_variable.py for how the additional loss terms are accounted for)
mll = VariationalELBO(likelihood, model, num_data=len(Y))

optimizer = torch.optim.SGD([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
], lr=0.001)

loss_list = []
iteration = 20000
batch_size = 100
for i in range(iteration):
    batch_index = model._get_batch_idx(batch_size)
    optimizer.zero_grad()
    sample = model.sample_latent_variable()  
    sample_batch = sample[batch_index]
    output_batch = model(sample_batch)
    loss = -mll(output_batch, Y[batch_index].T).sum()
    loss_list.append(loss.item())
    loss.backward()
    print(i)
    optimizer.step()

# save parameters
parameter = torch.cat((1/model.covar_module.base_kernel.lengthscale.squeeze(0),model.covar_module.outputscale.unsqueeze(0))).detach().cpu()
parameter = np.array(parameter)
np.savetxt('./result/matern0.5/model_latent'+str(latent_dim)+'_para.txt',parameter)

# save the latent embedding
X = model.X.q_mu.detach().cpu().numpy()
np.savetxt('./result/matern0.5/X_latent'+str(latent_dim)+'.txt', X)

if latent_dim==2:
    plt.clf()
    plt.figure(figsize=(5,5))
    scatter_plot = sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette="tab10")
    plt.savefig("./gplvm_matern0.5.png")