import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import torch
import math
import gpytorch
import os
import torch.nn.functional as F
from datetime import datetime
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


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def co2_scipy(seed,traing_prop):
    torch.manual_seed(seed)
    data = pd.read_csv("./co2_mm_mlo.csv",header=None)
    # 10 year training data
    data = data[ (data[3] > 0) & (data[2] >= 1960) & (data[2] < 2020)]
    data_tensor = torch.tensor(data.values, dtype=torch.float64)
    num_rows = data_tensor.shape[0]
    shuffled_data = data_tensor[torch.randperm(num_rows)]
    shuffled_data[:,3] = shuffled_data[:,3] - shuffled_data[:,3].mean()
    num_train = int(num_rows*traing_prop)
    train_data = shuffled_data[:num_train]
    test_data = shuffled_data[num_train:num_rows]
    #
    # Split the training and testing data into inputs and outputs
    train_x = train_data[:, 2].unsqueeze(1)#.cuda()
    train_y = train_data[:, 3]#.cuda()
    test_x = test_data[:, 2].unsqueeze(1)#.cuda()
    test_y = test_data[:, 3]#.cuda()
    #
    #
    from sklearn.gaussian_process.kernels import WhiteKernel,ConstantKernel,Matern
    noise_kernel = WhiteKernel(noise_level=0.5,noise_level_bounds=(1e-10, 1e1))
    #
    matern1_2 = 10 * Matern(length_scale=0.25,nu=0.5)
    matern3_2 = 500.   * Matern(length_scale=10.,nu=1.5)
    #
    co2_kernel = (
        matern1_2  + noise_kernel + matern3_2
    )
    #
    from sklearn.gaussian_process import GaussianProcessRegressor
    #
    gaussian_process = GaussianProcessRegressor(kernel=co2_kernel, normalize_y=False,alpha=1e-12)
    gaussian_process.fit(train_x.numpy(), train_y.numpy())
    gaussian_process.kernel_
    gaussian_process.log_marginal_likelihood()
    mean_y_pred, std_y_pred = gaussian_process.predict(test_x.numpy(), return_std=True)
    #mean_y_pred += test_y
    rmse_mix_first2 = (((mean_y_pred-test_y.numpy())**2).mean())**0.5
    ######
    noise_kernel = WhiteKernel(noise_level=0.5,noise_level_bounds=(1e-10, 1e1))
    matern1_2 = 10 * Matern(length_scale=0.25,nu=0.5)
    #
    co2_kernel = (
        matern1_2  + noise_kernel# + matern3_2
    )
    #
    from sklearn.gaussian_process import GaussianProcessRegressor
    #
    gaussian_process = GaussianProcessRegressor(kernel=co2_kernel, normalize_y=False,alpha=1e-12)
    gaussian_process.fit(train_x.numpy(), train_y.numpy())
    gaussian_process.kernel_
    gaussian_process.log_marginal_likelihood()
    mean_y_pred, std_y_pred = gaussian_process.predict(test_x.numpy(), return_std=True)
    #mean_y_pred += test_y
    rmse_matern1_2 = (((mean_y_pred-test_y.numpy())**2).mean())**0.5
    #####
    noise_kernel = WhiteKernel(noise_level=0.5,noise_level_bounds=(1e-10, 1e1))
    matern3_2 = 500.   * Matern(length_scale=10.,nu=1.5)
    #
    co2_kernel = (
        matern3_2  + noise_kernel
    )
    #
    from sklearn.gaussian_process import GaussianProcessRegressor
    #
    gaussian_process = GaussianProcessRegressor(kernel=co2_kernel, normalize_y=False,alpha=1e-12)
    gaussian_process.fit(train_x.numpy(), train_y.numpy())
    gaussian_process.kernel_
    gaussian_process.log_marginal_likelihood()
    mean_y_pred, std_y_pred = gaussian_process.predict(test_x.numpy(), return_std=True)
    #mean_y_pred += test_y
    rmse_matern3_2 = (((mean_y_pred-test_y.numpy())**2).mean())**0.5
    ######
    noise_kernel = WhiteKernel(noise_level=0.5,noise_level_bounds=(1e-10, 1e1))
    matern3_2 = 500.   * Matern(length_scale=10.,nu=1.5)
    matern5_2 = 500.   * Matern(length_scale=10.,nu=2.5)
    #
    co2_kernel = (
        matern3_2  + matern5_2 + noise_kernel
    )
    #
    from sklearn.gaussian_process import GaussianProcessRegressor
    #
    gaussian_process = GaussianProcessRegressor(kernel=co2_kernel, normalize_y=False,alpha=1e-12)
    gaussian_process.fit(train_x.numpy(), train_y.numpy())
    gaussian_process.kernel_
    gaussian_process.log_marginal_likelihood()
    mean_y_pred, std_y_pred = gaussian_process.predict(test_x.numpy(), return_std=True)
    #mean_y_pred += test_y
    rmse_mix_second2 = (((mean_y_pred-test_y.numpy())**2).mean())**0.5
    ##########
    noise_kernel = WhiteKernel(noise_level=0.5,noise_level_bounds=(1e-10, 1e1))
    matern1_2 = 10 * Matern(length_scale=0.25,nu=0.5)
    matern3_2 = 500.   * Matern(length_scale=10.,nu=1.5)
    matern5_2 = 500.   * Matern(length_scale=10.,nu=2.5)
    #
    co2_kernel = (
        matern1_2 + matern3_2  + matern5_2 + noise_kernel
    )
    #
    from sklearn.gaussian_process import GaussianProcessRegressor
    #
    gaussian_process = GaussianProcessRegressor(kernel=co2_kernel, normalize_y=False,alpha=1e-12)
    gaussian_process.fit(train_x.numpy(), train_y.numpy())
    gaussian_process.kernel_
    gaussian_process.log_marginal_likelihood()
    mean_y_pred, std_y_pred = gaussian_process.predict(test_x.numpy(), return_std=True)
    #mean_y_pred += test_y
    rmse_mix3 = (((mean_y_pred-test_y.numpy())**2).mean())**0.5
    ############
    noise_kernel = WhiteKernel(noise_level=0.5,noise_level_bounds=(1e-10, 1e1))
    matern5_2 = 500.   * Matern(length_scale=10.,nu=2.5)
    #
    co2_kernel = (
       matern5_2 + noise_kernel
    )
    #
    from sklearn.gaussian_process import GaussianProcessRegressor
    #
    gaussian_process = GaussianProcessRegressor(kernel=co2_kernel, normalize_y=False,alpha=1e-12)
    gaussian_process.fit(train_x.numpy(), train_y.numpy())
    gaussian_process.kernel_
    gaussian_process.log_marginal_likelihood()
    mean_y_pred, std_y_pred = gaussian_process.predict(test_x.numpy(), return_std=True)
    #mean_y_pred += test_y
    rmse_matern5_2 = (((mean_y_pred-test_y.numpy())**2).mean())**0.5
    return rmse_mix_first2,rmse_matern1_2,rmse_matern3_2,rmse_mix_second2,rmse_mix3,rmse_matern5_2


for traing_prop in range(5,100,5):
    result=[co2_scipy(seed,traing_prop/100) for seed in range(10)]
    result = np.array(result)
    np.savetxt('./traing_prop_'+str(traing_prop)+'.txt',result)

