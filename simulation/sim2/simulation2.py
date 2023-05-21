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


sys.path.append("../../")
import mixture_kernel as mixture_kernel
#importlib.reload(mixture_kernel)


plt.clf()
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def inverse_softplus(y):
    y=torch.Tensor([y])
    return (y.exp()-1).log()

L=3
p=2
mixed_weight = torch.Tensor((0.1,0.3,0.6))
mixed_weight = torch.clamp(mixed_weight, min=1e-3)
mixed_weight=mixed_weight / (mixed_weight.sum())
alpha = torch.Tensor((4.,2.,1.))
sigma = torch.Tensor((16.,4.,1.))

# define function to perform simulation using different sample size
def simulation2(n,p,L=3,lr=0.005,epoch=1000):
    jitter = torch.tensor(1e-01)
    # generate X from uniform distribution with random shift
    X = torch.Tensor(np.array([np.linspace(-10, 10,n) for i in range(p)])).reshape(n,p)
    shift = dist.Uniform(-1/(5*n), 1/(5*n)).sample(sample_shape=(n,p))
    X = X + shift
    # calculate true kernel
    kernel_true = mixture_kernel.Mixed_kernel_easy(data_covar_module_type="matern_different_smoothness",num_tasks=1,num_kernel=L)
    cov = kernel_true.calculate_true_cov(X = X, mixed_weight = mixed_weight, alpha = alpha,
                                                sigma=sigma)+jitter.expand(n).diag()
    try:
        # generate Y using MVN
        scale_tril = torch.linalg.cholesky(cov)
        true_distribution = MultivariateNormal(torch.zeros(X.shape[0]),scale_tril=scale_tril)
        Y = true_distribution.sample()
    except ValueError:
        print("Value Error")
        return np.nan
    except torch._C._LinAlgError:
        print(np.linalg.eigvals(cov))
        print("scale_tril not complete")
        return np.nan
    #
    #
    class MixtureGPModel(gpytorch.models.GP):
        def __init__(self):
            super(MixtureGPModel, self).__init__()
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = mixture_kernel.Mixed_kernel_easy(data_covar_module_type="matern_different_smoothness",num_tasks=1)
        def forward(self, x):
            jitter = torch.tensor(1e-01)
            jitter = jitter.cuda()
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)+jitter.expand(n).diag()
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    model = MixtureGPModel()
    #
    # initialize parameter
    model.covar_module.mixed_weight=torch.nn.Parameter(torch.tensor((0.2,0.3,0.5),device=device))
    model.covar_module.matern1.base_kernel.raw_lengthscale = torch.nn.Parameter(torch.tensor(1.,device=device).reshape(1,1))
    model.covar_module.matern2.base_kernel.raw_lengthscale = torch.nn.Parameter(torch.tensor(2.,device=device).reshape(1,1))
    model.covar_module.matern3.base_kernel.raw_lengthscale = torch.nn.Parameter(torch.tensor(3.,device=device).reshape(1,1))
    model.covar_module.matern1.raw_outputscale = torch.nn.Parameter(torch.tensor(5.,device=device))
    model.covar_module.matern2.raw_outputscale = torch.nn.Parameter(torch.tensor(10.,device=device))
    model.covar_module.matern3.raw_outputscale = torch.nn.Parameter(torch.tensor(15.,device=device))
    #
    model = model.cuda()
    model.train()
    # Use the SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #
    # "Loss" for GPs - the marginal log likelihood
    #
    X = X.cuda()
    Y = Y.cuda()
    training_iterations=epoch
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model.forward(X)
        loss = -output.log_prob(Y)
        loss.backward()
        optimizer.step()

    output = model.forward(X)
    estimated_mle = output.log_prob(Y)
    true_mle = true_distribution.log_prob(Y.cpu())
    # parameter estimation
    predict_weight=model.covar_module.mixed_weight
    predict_weight=torch.clamp(predict_weight, min=1e-3)
    predict_weight=predict_weight/(predict_weight.sum())
    #
    return torch.Tensor((predict_weight[0],predict_weight[1],predict_weight[2],
                         1/model.covar_module.matern1.base_kernel.lengthscale,1/model.covar_module.matern2.base_kernel.lengthscale,1/model.covar_module.matern3.base_kernel.lengthscale,
                         model.covar_module.matern1.outputscale,model.covar_module.matern2.outputscale,model.covar_module.matern3.outputscale,estimated_mle,true_mle)).detach().cpu()

# set data dimension
L=3
p=2
lr = 0.005

# simulation
# 100 replicates

torch.manual_seed(0)

sim20 = [simulation2(n=20,p=p,L=L,lr=lr,epoch=1000) for i in range(100)]
sim20=torch.cat(sim20).reshape(-1,11)
np.savetxt('./result/s2_sim20.txt', sim20.numpy())

sim50 = [simulation2(n=50,p=p,L=L,lr=lr,epoch=1000) for i in range(100)]
sim50=torch.cat(sim50).reshape(-1,11)
np.savetxt('./result/s2_sim50.txt', sim50.numpy())

sim100 = [simulation2(n=100,p=p,L=L,lr=lr,epoch=1000) for i in range(100)]
sim100 =torch.cat(sim100).reshape(-1,11)
np.savetxt('./result/s2_sim100.txt', sim100.numpy())

sim500 = [simulation2(n=500,p=p,L=L,lr=lr,epoch=1000) for i in range(100)]
sim500=torch.cat(sim500).reshape(-1,11)
np.savetxt('./result/s2_sim500.txt', sim500.numpy())



# figure

sim20= np.loadtxt('./result/s2_sim20.txt')
sim50= np.loadtxt('./result/s2_sim50.txt')
sim100= np.loadtxt('./result/s2_sim100.txt')
sim500= np.loadtxt('./result/s2_sim500.txt')

plt.clf()

def figure_param(data):
    data = np.array(data)
    weight1 = data[:,0]
    weight2 = data[:,1]
    weight3 = data[:,2]
    dalpha1 = data[:,3]
    dalpha2 = data[:,4]
    dalpha3 = data[:,5]
    sigma1  = data[:,6]
    sigma2  = data[:,7]
    sigma3  = data[:,8]
    return weight1*dalpha1*sigma1, weight1, sigma1, dalpha1, weight2, sigma2, dalpha2,weight3, sigma3, dalpha3

# main figure
plt.clf()
fig, axs = plt.subplots(1, 4, figsize=(18,5))
labels = ['20','50','100','500']

sim_A=[figure_param(sim20)[0],figure_param(sim50)[0],figure_param(sim100)[0],figure_param(sim500)[0]]
axs[0].boxplot(sim_A, labels=labels,vert=True)
axs[0].axhline(y=mixed_weight[0]* alpha[0]*sigma[0], color='r',alpha=0.5, linestyle='dotted')
axs[0].set_title(r'$\hat{w_1}\hat{\sigma_1^2}\hat{\alpha_1}^{2\nu_1}$',fontsize=20)
axs[0].set_xlabel('Sample size',fontsize=15)

sim_A=[figure_param(sim20)[1],figure_param(sim50)[1],figure_param(sim100)[1],figure_param(sim500)[1]]
axs[1].boxplot(sim_A, labels=labels,vert=True)
axs[1].axhline(y=mixed_weight[0], color='r',alpha=0.5, linestyle='dotted')
axs[1].set_title(r'$\hat{w_1}$',fontsize=20)
axs[1].set_xlabel('Sample size',fontsize=15)

sim_A=[figure_param(sim20)[2],figure_param(sim50)[2],figure_param(sim100)[2],figure_param(sim500)[2]]
axs[2].boxplot(sim_A, labels=labels,vert=True)
axs[2].axhline(y=sigma[0], color='r',alpha=0.5, linestyle='dotted')
axs[2].set_title(r'$\hat{\sigma_1}^2$',fontsize=20)
axs[2].set_xlabel('Sample size',fontsize=15)

sim_A=[figure_param(sim20)[3],figure_param(sim50)[3],figure_param(sim100)[3],figure_param(sim500)[3]]
axs[3].boxplot(sim_A, labels=labels,vert=True)
axs[3].axhline(y=alpha[0], color='r',alpha=0.5, linestyle='dotted')
axs[3].set_title(r'$\hat{\alpha_1}$',fontsize=20)
axs[3].set_xlabel('Sample size',fontsize=15)

axs[0].text(-0.1, 1.1, 'A', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')
axs[1].text(-0.1, 1.1, 'B', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')
axs[2].text(-0.1, 1.1, 'C', transform=axs[2].transAxes, fontsize=16, fontweight='bold', va='top')
axs[3].text(-0.1, 1.1, 'D', transform=axs[3].transAxes, fontsize=16, fontweight='bold', va='top')

plt.tight_layout()

plt.savefig('./simulation2_main.png',dpi=300)

##### all parameter 
plt.clf()
fig, axs = plt.subplots(3, 3, figsize=(18,18))
labels = ['20', '50', '100', '500']


sim_A=[figure_param(sim20)[1],figure_param(sim50)[1],figure_param(sim100)[1],figure_param(sim500)[1]]
axs[0,0].boxplot(sim_A, labels=labels,vert=True)
axs[0,0].axhline(y=mixed_weight[0], color='r',alpha=0.5, linestyle='dotted')
axs[0,0].set_ylabel(r'$\hat{w_1}$')
axs[0,0].set_xlabel('Sample size')

sim_A=[figure_param(sim20)[2],figure_param(sim50)[2],figure_param(sim100)[2],figure_param(sim500)[2]]
axs[0,1].boxplot(sim_A, labels=labels,vert=True)
axs[0,1].axhline(y=sigma[0], color='r',alpha=0.5, linestyle='dotted')
axs[0,1].set_ylabel(r'$\hat{\sigma_1}^2$')
axs[0,1].set_xlabel('Sample size')


sim_A=[figure_param(sim20)[3],figure_param(sim50)[3],figure_param(sim100)[3],figure_param(sim500)[3]]
axs[0,2].boxplot(sim_A, labels=labels,vert=True)
axs[0,2].axhline(y=alpha[0], color='r',alpha=0.5, linestyle='dotted')
axs[0,2].set_ylabel(r'$\hat{\alpha_1}$')
axs[0,2].set_xlabel('Sample size')

sim_A=[figure_param(sim20)[4],figure_param(sim50)[4],figure_param(sim100)[4],figure_param(sim500)[4]]
axs[1,0].boxplot(sim_A, labels=labels,vert=True)
axs[1,0].axhline(y=mixed_weight[1], color='r',alpha=0.5, linestyle='dotted')
axs[1,0].set_ylabel(r'$\hat{w_2}$')
axs[1,0].set_xlabel('Sample size')

sim_A=[figure_param(sim20)[5],figure_param(sim50)[5],figure_param(sim100)[5],figure_param(sim500)[5]]
axs[1,1].boxplot(sim_A, labels=labels,vert=True)
axs[1,1].axhline(y=sigma[1], color='r',alpha=0.5, linestyle='dotted')
axs[1,1].set_ylabel(r'$\hat{\sigma_2}^2$')
axs[1,1].set_xlabel('Sample size')


sim_A=[figure_param(sim20)[6],figure_param(sim50)[6],figure_param(sim100)[6],figure_param(sim500)[6]]
axs[1,2].boxplot(sim_A, labels=labels,vert=True)
axs[1,2].axhline(y=alpha[1], color='r',alpha=0.5, linestyle='dotted')
axs[1,2].set_ylabel(r'$\hat{\alpha_2}$')
axs[1,2].set_xlabel('Sample size')

sim_A=[figure_param(sim20)[7],figure_param(sim50)[7],figure_param(sim100)[7],figure_param(sim500)[7]]
axs[2,0].boxplot(sim_A, labels=labels,vert=True)
axs[2,0].axhline(y=mixed_weight[2], color='r',alpha=0.5, linestyle='dotted')
axs[2,0].set_ylabel(r'$\hat{w_3}$')
axs[2,0].set_xlabel('Sample size')

sim_A=[figure_param(sim20)[8],figure_param(sim50)[8],figure_param(sim100)[8],figure_param(sim500)[8]]
axs[2,1].boxplot(sim_A, labels=labels,vert=True)
axs[2,1].axhline(y=sigma[2], color='r',alpha=0.5, linestyle='dotted')
axs[2,1].set_ylabel(r'$\hat{\sigma_3}^2$')
axs[2,1].set_xlabel('Sample size')


sim_A=[figure_param(sim20)[9],figure_param(sim50)[9],figure_param(sim100)[9],figure_param(sim500)[9]]
axs[2,2].boxplot(sim_A, labels=labels,vert=True)
axs[2,2].axhline(y=alpha[2], color='r',alpha=0.5, linestyle='dotted')
axs[2,2].set_ylabel(r'$\hat{\alpha_3}$')
axs[2,2].set_xlabel('Sample size')

plt.tight_layout()

plt.savefig("./simulation2_all_para.png",dpi=300)

