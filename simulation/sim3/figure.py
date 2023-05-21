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

p=2
A01 = 1.
A = np.array([[5.,A01],[A01,5.]])
alpha = torch.ones(1)
sigma = torch.ones(1)*10
lr=0.001

plt.clf()


sim50 = np.loadtxt('./result/s3_sim50.txt')
sim100= np.loadtxt('./result/s3_sim100.txt')
sim200= np.loadtxt('./result/s3_sim200.txt')
sim400= np.loadtxt('./result/s3_sim400.txt')

def figure_param(data):
    data = np.array(data)
    A00 = data[:,0]
    A01 = data[:,1]
    A11 = data[:,2]
    dalpha1 = data[:,3]
    sigma1 = data[:,4]
    return A00*dalpha1*sigma1,A01*dalpha1*sigma1,A11*dalpha1*sigma1,dalpha1,sigma1, A00,A01,A11


plt.clf()
fig, axs = plt.subplots(1, 5, figsize=(25,5,))
labels = ['50', '100', '200','400']

# boxplot for $\hat{A_{1,1}}\hat{\sigma}^2\hat{\alpha}^{2\nu}$
sim_A=[figure_param(sim50)[0],figure_param(sim100)[0],figure_param(sim200)[0],figure_param(sim400)[0]]
axs[0].boxplot(sim_A, labels=labels,vert=True)
axs[0].axhline(y=A[0,0]*sigma*alpha, color='r',alpha=0.5, linestyle='dotted')
axs[0].set_title(r'$\hat{A_{1,1}}\hat{\sigma}^2\hat{\alpha}^{2\nu}$',fontsize=18)
axs[0].set_xlabel('Sample size')

# boxplot for $\hat{A_{1,2}}\hat{\sigma}^2\hat{\alpha}^{2\nu}$
sim_A=[figure_param(sim50)[1],figure_param(sim100)[1],figure_param(sim200)[1],figure_param(sim400)[1]]
axs[1].boxplot(sim_A, labels=labels,vert=True)
axs[1].axhline(y=A[0,1]*sigma*alpha, color='r',alpha=0.5, linestyle='dotted')
axs[1].set_title(r'$\hat{A_{1,2}}\hat{\sigma}^2\hat{\alpha}^{2\nu}$',fontsize=18)
axs[1].set_xlabel('Sample size')

# boxplot for $\hat{A_{2,2}}\hat{\sigma}^2\hat{\alpha}^{2\nu}$
sim_A=[figure_param(sim50)[2],figure_param(sim100)[2],figure_param(sim200)[2],figure_param(sim400)[2]]
axs[2].boxplot(sim_A, labels=labels,vert=True)
axs[2].axhline(y=A[1,1]*sigma*alpha, color='r',alpha=0.5, linestyle='dotted')
axs[2].set_title(r'$\hat{A_{2,2}}\hat{\sigma}^2\hat{\alpha}^{2\nu}$',fontsize=18)
axs[2].set_xlabel('Sample size')

# boxplot for $\hat{\alpha}$
sim_A=[figure_param(sim50)[3],figure_param(sim100)[3],figure_param(sim200)[3],figure_param(sim400)[3]]
axs[3].boxplot(sim_A, labels=labels,vert=True)
axs[3].axhline(y=alpha, color='r',alpha=0.5, linestyle='dotted')
axs[3].set_title(r'$\hat{\alpha}$',fontsize=18)
axs[3].set_xlabel('Sample size')

# boxplot for $\hat{\sigma}^2$
sim_A=[figure_param(sim50)[4],figure_param(sim100)[4],figure_param(sim200)[4],figure_param(sim400)[4]]
axs[4].boxplot(sim_A, labels=labels,vert=True)
axs[4].axhline(y=sigma, color='r',alpha=0.5, linestyle='dotted')
axs[4].set_title(r'$\hat{\sigma}^2$',fontsize=18)
axs[4].set_xlabel('Sample size')


axs[0].text(-0.1, 1.1, 'A', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')
axs[1].text(-0.1, 1.1, 'B', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')
axs[2].text(-0.1, 1.1, 'C', transform=axs[2].transAxes, fontsize=16, fontweight='bold', va='top')
axs[3].text(-0.1, 1.1, 'D', transform=axs[3].transAxes, fontsize=16, fontweight='bold', va='top')
axs[4].text(-0.1, 1.1, 'E', transform=axs[4].transAxes, fontsize=16, fontweight='bold', va='top')

axs[0].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='x', labelsize=14)
axs[2].tick_params(axis='x', labelsize=14)
axs[3].tick_params(axis='x', labelsize=14)
axs[4].tick_params(axis='x', labelsize=14)

plt.tight_layout()

plt.savefig('./simulation3_main.png',dpi=100)
