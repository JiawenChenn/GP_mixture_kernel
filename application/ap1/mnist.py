import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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

sys.path.append("../../")
import mixture_kernel as mixture_kernel
importlib.reload(mixture_kernel)

# Load the image

def data_processing(test_width=20):
    img = Image.open('mnist_0.png')
    img = img.convert('L')  # convert to grayscale
    img = img.resize((100, 100))
    # Crop the center test_width x test_width area
    # Crop the center test_width x test_width area
    w, h = img.size
    left = (w - test_width) // 3
    top = (h - test_width) // 2
    right = left + test_width
    bottom = top + test_width
    img_crop = img.crop((left, top, right, bottom))
    # Create the training and test datasets
    train_data = []
    test_data = []
    for y in range(h):
        for x in range(w):
            if left <= x < right and top <= y < bottom:
                # Add to test dataset
                test_data.append((x, y, img.getpixel((x, y))))
            else:
                # Add to training dataset
                train_data.append((x, y, img.getpixel((x, y))))
    # Convert to NumPy arrays
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    # Plot the images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    # Plot the training image
    train_img = np.zeros((h, w))
    train_img[train_data[:, 1], train_data[:, 0]] = train_data[:, 2]
    ax[0].imshow(train_img, cmap='gray')
    ax[0].set_title('Training Image')
    # Plot the testing image
    test_img = np.zeros((h, w))
    test_img[test_data[:, 1], test_data[:, 0]] = test_data[:, 2]
    ax[1].imshow(test_img, cmap='gray')
    ax[1].set_title('Testing Image')
    plt.savefig("./image_data.png")
    np.savetxt('./train_data.txt', train_data)
    np.savetxt('./test_data.txt', test_data)

#data_processing()
train_data= np.loadtxt('./train_data.txt')
test_data= np.loadtxt('./test_data.txt')
train_data = torch.Tensor(train_data)
test_data = torch.Tensor(test_data)

class MixtureGPModel(gpytorch.models.GP):
    def __init__(self):
        super(MixtureGPModel, self).__init__()
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = mixture_kernel.Mixed_kernel_easy(data_covar_module_type="matern_different_smoothness",num_tasks=1)

    def forward(self, x):
        jitter = torch.tensor(1e-02)
        jitter = jitter.cuda()
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)+jitter.expand(x.shape[0]).diag()
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

model = MixtureGPModel()
#
model = model.cuda()
model.train()
# Use the adam optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-05)  # Includes GaussianLikelihood parameters
#
# "Loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
#
train_x = train_data[:,:2]
train_y = train_data[:,2]

train_x = train_x.cuda()
train_y = train_y.cuda()

training_iterations=40000
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model.forward(train_x)
    loss = -output.log_prob(train_y)
    loss.backward()
    #print('Iter %d/%d - Loss: %.3f ' % (i + 1, training_iterations, loss.item()))
    optimizer.step()
    #print(i)

predict_weight=model.covar_module.mixed_weight
predict_weight=torch.clamp(predict_weight, min=1e-3)
predict_weight=predict_weight/(predict_weight.sum())

parameter = torch.Tensor((predict_weight[0],predict_weight[1],predict_weight[2],
                        1/model.covar_module.matern1.base_kernel.lengthscale,1/model.covar_module.matern2.base_kernel.lengthscale,1/model.covar_module.matern3.base_kernel.lengthscale,
                        model.covar_module.matern1.outputscale,model.covar_module.matern2.outputscale,model.covar_module.matern3.outputscale)).detach().cpu()

np.savetxt('./model_parameter.txt', parameter.numpy())


# prediction

test_x = test_data[:,:2]
test_y = test_data[:,2]

test_x = test_x.cuda()
test_y = test_y.cuda()

model.eval()
jitter = torch.tensor(1e-02)
jitter = jitter.cuda()
model_train_train=model.covar_module(train_x,train_x)+jitter.expand(train_x.shape[0]).diag()
model_test_train=model.covar_module(test_x,train_x)
pred = model_test_train @ torch.linalg.inv(torch.Tensor(model_train_train.numpy())).t().cuda() @ train_y


class singleGPModel(gpytorch.models.GP):
    def __init__(self):
        super(singleGPModel, self).__init__()
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))

    def forward(self, x):
        jitter = torch.tensor(1e-02)
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
    #print('Iter %d/%d - Loss: %.3f ' % (i + 1, training_iterations, loss.item()))
    optimizer.step()
    #print(i)


parameter = torch.Tensor((1/model_single.covar_module.base_kernel.lengthscale,model_single.covar_module.outputscale)).detach().cpu()

np.savetxt('./model_single_parameter.txt', parameter.numpy())


model_single.eval()
jitter = torch.tensor(1e-02)
jitter = jitter.cuda()
model_train_train=model_single.covar_module(train_x,train_x)+jitter.expand(train_x.shape[0]).diag()
model_test_train=model_single.covar_module(test_x,train_x)
pred_single = model_test_train @ torch.linalg.inv(torch.Tensor(model_train_train.numpy())).t().cuda() @ train_y
