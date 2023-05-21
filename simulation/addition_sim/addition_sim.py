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

def add_simulation(n,p,L=3):
    jitter = torch.tensor(1e-01)
    # generate X from uniform distribution with random shift
    X = torch.Tensor(np.array([np.linspace(-10, 10,n) for i in range(p)])).reshape(n,p)
    shift = dist.Uniform(-1/(10*n), 1/(10*n)).sample(sample_shape=(n,p))
    X = X + shift
    # calculate true kernel
    kernel_true = mixture_kernel.Mixed_kernel_easy(data_covar_module_type="matern_same_smoothness",num_tasks=1,num_kernel=L)
    cov = kernel_true.calculate_true_cov(X = X, mixed_weight = torch.Tensor((0.2,0.3,0.5)), alpha = torch.Tensor((2.0,1.0,4.0)),
                                                sigma=torch.Tensor((16.,4.,1.)))+jitter.expand(n).diag()
    try:
        # generate Y using MVN
        scale_tril = torch.linalg.cholesky(cov)
        true_distribution = MultivariateNormal(torch.zeros(X.shape[0]),scale_tril=scale_tril)
        Y = true_distribution.sample()
    except ValueError:
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
            self.covar_module = mixture_kernel.Mixed_kernel_easy(data_covar_module_type="matern_same_smoothness",num_tasks=1)
        def forward(self, x):
            jitter = torch.tensor(1e-01)
            jitter = jitter.cuda()
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)+jitter.expand(n).diag()
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    model = MixtureGPModel()
    model = model.cuda()
    model.train()

    # Parameter initialization
    model.covar_module.matern1.base_kernel.raw_lengthscale = torch.nn.Parameter(torch.tensor(-1.2587,device=device).reshape(1,1))
    model.covar_module.matern2.base_kernel.raw_lengthscale = torch.nn.Parameter(torch.tensor(-0.4328,device=device).reshape(1,1))
    model.covar_module.matern3.base_kernel.raw_lengthscale = torch.nn.Parameter(torch.tensor(0.5413,device=device).reshape(1,1))
    model.covar_module.matern1.raw_outputscale = torch.nn.Parameter(torch.tensor(16.,device=device))
    model.covar_module.matern2.raw_outputscale = torch.nn.Parameter(torch.tensor(3.9815,device=device))
    model.covar_module.matern3.raw_outputscale = torch.nn.Parameter(torch.tensor(0.5413,device=device))

    X = X.cuda()
    Y = Y.cuda()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    training_iterations=1000
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model.forward(X)
        loss = -output.log_prob(Y)
        loss.backward()
        optimizer.step()
    #
    output = model.forward(X)
    estimated_mle = output.log_prob(Y)
    true_mle = true_distribution.log_prob(Y.cpu())
    # parameter estimation
    predict_weight=model.covar_module.mixed_weight
    predict_weight=torch.clamp(predict_weight, min=1e-3)
    predict_weight=predict_weight/(predict_weight.sum())
    temp1=predict_weight[0]*(1/model.covar_module.matern1.base_kernel.lengthscale)*model.covar_module.matern1.outputscale
    temp2=predict_weight[1]*(1/model.covar_module.matern2.base_kernel.lengthscale)*model.covar_module.matern2.outputscale
    temp3=predict_weight[2]*(1/model.covar_module.matern3.base_kernel.lengthscale)*model.covar_module.matern3.outputscale
    #
    return torch.Tensor(((temp1+temp2+temp3),predict_weight[0],predict_weight[1],predict_weight[2],
                        1/model.covar_module.matern1.base_kernel.lengthscale,1/model.covar_module.matern2.base_kernel.lengthscale,1/model.covar_module.matern3.base_kernel.lengthscale,
                        model.covar_module.matern1.outputscale,model.covar_module.matern2.outputscale,model.covar_module.matern3.outputscale,estimated_mle,true_mle)).detach().cpu()

# set data dimension
L=3
p=2

torch.manual_seed(0)
sim20 = [add_simulation(n=20,p=p,L=L) for i in range(100)]
sim50 = [add_simulation(n=50,p=p,L=L) for i in range(100)]
sim100 = [add_simulation(n=100,p=p,L=L) for i in range(100)]
sim500 = [add_simulation(n=500,p=p,L=L) for i in range(100)]


sim20=torch.cat(sim20).reshape(-1,12)
sim50=torch.cat(sim50).reshape(-1,12)
sim100=torch.cat(sim100).reshape(-1,12)
sim500=torch.cat(sim500).reshape(-1,12)

np.savetxt('./result/add_sim_sim20.txt', sim20.numpy())
np.savetxt('./result/add_sim_sim50.txt', sim50.numpy())
np.savetxt('./result/add_sim_sim100.txt', sim100.numpy())
np.savetxt('./result/add_sim_sim500.txt', sim500.numpy())


##### figure #####

mixed_weight = torch.Tensor((0.2,0.3,0.5))
mixed_weight = F.relu(mixed_weight)
mixed_weight=mixed_weight/(mixed_weight.sum())
alpha = torch.Tensor((2.0,1.0,4.0))
sigma=torch.Tensor((16.,4.,1.))

sim20 = np.loadtxt('./result/add_sim_sim20.txt')
sim50 = np.loadtxt('./result/add_sim_sim50.txt')
sim100= np.loadtxt('./result/add_sim_sim100.txt')
sim500= np.loadtxt('./result/add_sim_sim500.txt')

def figure_param(data):
    data = np.array(data)
    sum_all = data[:,0]
    weight1 = data[:,1]
    weight2 = data[:,2]
    weight3 = data[:,3]
    dalpha1 = data[:,4]
    dalpha2 = data[:,5]
    dalpha3 = data[:,6]
    sigma1  = data[:,7]
    sigma2  = data[:,8]
    sigma3  = data[:,9]
    return sum_all, weight1, sigma1, dalpha1, weight2, sigma2, dalpha2,weight3, sigma3, dalpha3

##### main figure #####
plt.clf()
fig, axs = plt.subplots(1, 4, figsize=(18,5))
labels = ['20', '50', '100', '500']

sim_A=[figure_param(sim20)[0],figure_param(sim50)[0],figure_param(sim100)[0],figure_param(sim500)[0]]
axs[0].boxplot(sim_A, labels=labels,vert=True)
axs[0].axhline(y=mixed_weight[0]* alpha[0]*sigma[0]+mixed_weight[1]* alpha[1]*sigma[1]+mixed_weight[2]* alpha[2]*sigma[2], color='r',alpha=0.5, linestyle='dotted')
axs[0].set_ylabel(r'$\sum_{l=1}^L \hat{w_l}\hat{\sigma_l^2}\hat{\alpha_l}^{2\nu}$') # nu = 1/2
axs[0].set_xlabel('Sample size')
axs[0].set_ylim(0,80)

sim_A=[figure_param(sim20)[1],figure_param(sim50)[1],figure_param(sim100)[1],figure_param(sim500)[1]]
axs[1].boxplot(sim_A, labels=labels,vert=True)
axs[1].axhline(y=mixed_weight[0], color='r',alpha=0.5, linestyle='dotted')
axs[1].set_ylabel(r'$\hat{w_1}$')
axs[1].set_xlabel('Sample size')

sim_A=[figure_param(sim20)[2],figure_param(sim50)[2],figure_param(sim100)[2],figure_param(sim500)[2]]
axs[2].boxplot(sim_A, labels=labels,vert=True)
axs[2].axhline(y=sigma[0], color='r',alpha=0.5, linestyle='dotted')
axs[2].set_ylabel(r'$\hat{\sigma_1}^2$')
axs[2].set_xlabel('Sample size')

sim_A=[figure_param(sim20)[3],figure_param(sim50)[3],figure_param(sim100)[3],figure_param(sim500)[3]]
axs[3].boxplot(sim_A, labels=labels,vert=True)
axs[3].axhline(y=alpha[0], color='r',alpha=0.5, linestyle='dotted')
axs[3].set_ylabel(r'$\hat{\alpha_1}$')
axs[3].set_xlabel('Sample size')
axs[3].set_ylim(0,25)

axs[0].text(-0.1, 1.1, 'A', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')
axs[1].text(-0.1, 1.1, 'B', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')
axs[2].text(-0.1, 1.1, 'C', transform=axs[2].transAxes, fontsize=16, fontweight='bold', va='top')
axs[3].text(-0.1, 1.1, 'D', transform=axs[3].transAxes, fontsize=16, fontweight='bold', va='top')

plt.tight_layout()

plt.savefig("./add_sim_main.png",dpi=300)

##### all parameter #####

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
axs[0,2].set_ylim(0,20)

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
axs[1,2].set_ylim(0,20)

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

plt.savefig("./add_sim_all_para.png",dpi=300)



