import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import torch
import math
import gpytorch
import os
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

sys.path.append("../..")
import mixture_kernel as mixture_kernel
importlib.reload(mixture_kernel)

# Load the image
# https://en.wikipedia.org/wiki/Tread_plate#/media/File:Diamond_Plate.jpg

def boston(split_ratio=0.1,file_name=1):
    # Load the CSV file and extract the last two columns
    data = pd.read_csv("./ap2_boston_housing/housing.csv",header=None)
    data = data.iloc[:, -2:]

    # Convert the data to PyTorch tensors
    data_tensor = torch.tensor(data.values, dtype=torch.float32)

    # Split the data into training and testing sets
    num_rows = data_tensor.size()[0]
    num_train = int(num_rows * split_ratio)
    #num_train = num_rows - num_test

    # Shuffle the data randomly
    shuffled_data = data_tensor[torch.randperm(num_rows)]

    # Split the shuffled data into training and testing sets
    train_data = shuffled_data[:num_train]
    # keep only 50 training samples
    test_data = shuffled_data[num_train:(num_train+50)]

    # Split the training and testing data into inputs and outputs
    train_x = train_data[:, 0].unsqueeze(1).cuda()
    train_y = train_data[:, 1].cuda()
    test_x = test_data[:, 0].unsqueeze(1).cuda()
    test_y = test_data[:, 1].cuda()

    ######################################################
    ################ Mixture kernel model ################
    ######################################################

    class MixtureGPModel(gpytorch.models.GP):
        def __init__(self):
            super(MixtureGPModel, self).__init__()
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = mixture_kernel.Mixed_kernel_easy(data_covar_module_type="matern_different_smoothness",num_tasks=1)
        def forward(self, x):
            jitter = torch.tensor(1e-01)
            jitter = jitter.cuda()
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)+jitter.expand(x.shape[0]).diag()
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    model = MixtureGPModel()
    model = model.cuda()
    model.train()
    # Use the SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-05)  # Includes GaussianLikelihood parameters


    training_iterations=40000
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model.forward(train_x)
        loss = -output.log_prob(train_y)
        loss.backward()
        #print('Iter %d/%d - Loss: %.3f ' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
    

    predict_weight=model.covar_module.mixed_weight
    predict_weight=torch.clamp(predict_weight, min=1e-3)
    predict_weight=predict_weight/(predict_weight.sum())

    parameter = torch.Tensor((predict_weight[0],predict_weight[1],predict_weight[2],
                            1/model.covar_module.matern1.base_kernel.lengthscale,1/model.covar_module.matern2.base_kernel.lengthscale,1/model.covar_module.matern3.base_kernel.lengthscale,
                            model.covar_module.matern1.outputscale,model.covar_module.matern2.outputscale,model.covar_module.matern3.outputscale)).detach().cpu()

    np.savetxt('./boston_result/boston_data'+str(file_name)+'_split'+str(split_ratio)+'_model_parameter.txt', parameter.numpy())
    # prediction
    model.eval()
    jitter = torch.tensor(1e-02)
    jitter = jitter.cuda()
    model_train_train=model.covar_module(train_x,train_x)+jitter.expand(train_x.shape[0]).diag()
    model_test_train=model.covar_module(test_x,train_x)
    pred = (model_test_train @ torch.linalg.inv(torch.Tensor(model_train_train.numpy())).t().cuda() @ train_y).detach()
    pred = torch.cat([test_x.cpu(),test_y.unsqueeze(1).cpu(),pred.unsqueeze(1).cpu()],dim=1)
    np.savetxt('./boston_result/boston_data'+str(file_name)+'_split'+str(split_ratio)+'_model_pred.txt', pred.numpy())

    ######################################################
    ################  Matern 1/2 model    ################
    ######################################################

    class singleGPModel(gpytorch.models.GP):
        def __init__(self):
            super(singleGPModel, self).__init__()
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        def forward(self, x):
            jitter = torch.tensor(1e-01)
            jitter = jitter.cuda()
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)+jitter.expand(x.shape[0]).diag()
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    model_single = singleGPModel()
    model_single = model_single.cuda()
    model_single.train()
    optimizer = torch.optim.SGD(model_single.parameters(), lr=1e-05)

    training_iterations=40000
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model_single.forward(train_x)
        loss = -output.log_prob(train_y)
        loss.backward()
        optimizer.step()

    parameter = torch.Tensor((1/model_single.covar_module.base_kernel.lengthscale,model_single.covar_module.outputscale)).detach().cpu()
    np.savetxt('./boston_result/boston_data'+str(file_name)+'_split'+str(split_ratio)+'_model_single_parameter.txt', parameter.numpy())
    # prediction
    model_single.eval()
    jitter = torch.tensor(1e-01)
    jitter = jitter.cuda()
    model_train_train=model_single.covar_module(train_x,train_x)+jitter.expand(train_x.shape[0]).diag()
    model_test_train=model_single.covar_module(test_x,train_x)
    pred_single = (model_test_train @ torch.linalg.inv(torch.Tensor(model_train_train.numpy())).t().cuda() @ train_y).detach()
    pred_single = torch.cat([test_x.cpu(),test_y.unsqueeze(1).cpu(),pred_single.unsqueeze(1).cpu()],dim=1)
    np.savetxt('./boston_result/boston_data'+str(file_name)+'_split'+str(split_ratio)+'_model_single_pred.txt', pred_single.numpy())
    
    return np.nan

torch.manual_seed(0)
for i in range(10):
    for j in range(1, 10):
        boston(split_ratio=j/10,file_name=i)

# figure

def mse_diff(split_ratio):
    diff = []
    for file_name in range(10):
        mixture_pred = np.loadtxt('./ap2_boston_housing/boston_result/boston_data'+str(file_name)+'_split'+str(split_ratio)+'_model_pred.txt')
        single_pred = np.loadtxt('./ap2_boston_housing/boston_result/boston_data'+str(file_name)+'_split'+str(split_ratio)+'_model_single_pred.txt')
        diff.append(((mixture_pred[:,2]-mixture_pred[:,1])**2).sum()/50-((single_pred[:,2]-single_pred[:,1])**2).sum()/50)
    return diff

def diff(file_name):
    mixture_pred = np.loadtxt('./ap2_boston_housing/boston_result/boston_data'+str(file_name)+'model_pred.txt')
    single_pred = np.loadtxt('./ap2_boston_housing/boston_result/boston_data'+str(file_name)+'model_single_pred.txt')
    return mixture_pred[:,2]-single_pred[:,2]

file_name=0
split_ratio=0.9
mixture_pred = np.loadtxt('./ap2_boston_housing/boston_result/boston_data'+str(file_name)+'_split'+str(split_ratio)+'_model_pred.txt')
single_pred = np.loadtxt('./ap2_boston_housing/boston_result/boston_data'+str(file_name)+'_split'+str(split_ratio)+'_model_single_pred.txt')


plt.clf()
fig, ax = plt.subplots(1, 2, figsize=(10,5), sharex=False, sharey=False)

# Replication 1 scatter plot
ax[0].set_title('Replication 1 (training: 455)')
ax[0].scatter(mixture_pred[:,2], single_pred[:,2])
ax[0].set_xlabel('Mixed kernel prediction')
ax[0].set_ylabel('Mat\'ern 1/2 prediction')
ax[0].plot([0,50], [0,50], 'r--',label=r"$y=x$")
ax[0].legend()

# box plot for different training sample size
mse_diff_all=[mse_diff(j/10) for j in range(1,10)]
labels = [str(int(i*506)) for i in np.arange(0.1,1,0.1)]
ax[1].boxplot(mse_diff_all, labels=labels,vert=True)
ax[1].axhline(y=0, color='r',alpha=0.5, linestyle='dotted')
ax[1].set_ylabel('MSE difference')
ax[1].set_xlabel('Training sample size')
ax[1].set_ylim(-2,2)


ax[0].text(-0.1, 1.1, 'A', transform=ax[0].transAxes, fontsize=16, fontweight='bold', va='top')
ax[1].text(-0.1, 1.1, 'B', transform=ax[1].transAxes, fontsize=16, fontweight='bold', va='top')

plt.tight_layout()

plt.savefig("./ap2_final.png",dpi=300)


