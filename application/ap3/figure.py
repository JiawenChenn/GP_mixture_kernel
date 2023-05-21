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
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score


# Setting manual seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


dat_path = './Test_3_Pollen.h5'
dat_file = h5py.File(dat_path,'r')
y_train = dat_file['in_X'][:]
Y = torch.Tensor(y_train)
labels=torch.Tensor(dat_file['true_labs'][:])

def ari(latent_dim):
    single  = np.loadtxt('./result/matern0.5/X_latent'+str(latent_dim)+'.txt')
    mixture =np.loadtxt('./result/mixture/X_latent'+str(latent_dim)+'.txt')
    print((torch.Tensor(single)-torch.Tensor(mixture)).abs().max())
    # single
    kmeans = KMeans(n_clusters=11, random_state=42,n_init=10)  # Initialize k-means algorithm with 11 clusters
    kmeans.fit(single)  # Fit the k-means algorithm to your data
    single_predicted_labels = kmeans.labels_  # Retrieve the cluster assignments
    single_ari = adjusted_rand_score(labels, single_predicted_labels)
    # mixture
    kmeans = KMeans(n_clusters=11, random_state=42,n_init=10)  # Initialize k-means algorithm with 11 clusters
    kmeans.fit(mixture)  # Fit the k-means algorithm to your data
    mix_predicted_labels = kmeans.labels_  # Retrieve the cluster assignments
    mixture_ari = adjusted_rand_score(labels, mix_predicted_labels)
    return single_ari,mixture_ari

ari_all=[ari(i) for i in range(2,9)]
ari_all = np.array(ari_all)


plt.clf()
fig, axs = plt.subplots(1, 3, figsize=(15,5))

latent_dim=2
single  = np.loadtxt('./result/matern0.5/X_latent'+str(latent_dim)+'.txt')
mixture =np.loadtxt('./result/mixture/X_latent'+str(latent_dim)+'.txt')

# latent embedding for mixture kernel
mixture_dim2 = sns.scatterplot(x=mixture[:,0], y=mixture[:,1], hue=labels, palette="tab10", ax=axs[0])
mixture_dim2.get_legend().remove()
axs[0].set_title('Mixture kernel')

# latent embedding for Matern 1/2
single_dim2 = sns.scatterplot(x=single[:,0], y=single[:,1], hue=labels, palette="tab10", ax=axs[1])
single_dim2.get_legend().remove()
axs[1].set_title('Mat\'ern 1/2')

# scatter plot for ARI
X = np.arange(2, 9)
Y1 = ari_all[:, 0] # single kernel ARI
Y2 = ari_all[:, 1] # mixture kernel ARI
color1 = "royalblue"
color2 = "darkorange"
sns.lineplot(x=X, y=Y1, label="Mat\'ern 1/2", color=color1,ax=axs[2])
sns.scatterplot(x=X, y=Y1, color=color1, ax=axs[2],legend=False)
sns.lineplot(x=X, y=Y2, label="Mixture kernel", color=color2,ax=axs[2])
sns.scatterplot(x=X, y=Y2, color=color2, ax=axs[2],legend=False)
axs[2].set_ylabel('ARI')

axs[0].text(-0.1, 1.1, 'A', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')
axs[1].text(-0.1, 1.1, 'B', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')
axs[2].text(-0.1, 1.1, 'C', transform=axs[2].transAxes, fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('./ap3_all.png',dpi=300)
