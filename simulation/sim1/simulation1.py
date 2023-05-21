import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# Define the Matern kernels with different length scales
kernel1 = ConstantKernel(3) * Matern(length_scale=1.0, nu=0.5)
kernel2 = ConstantKernel(3) * Matern(length_scale=1.0, nu=1.5)
kernel3 = ConstantKernel(3) * Matern(length_scale=1.0, nu=2.5)

# Define the mixture of Matern kernels
mixture_kernel = 0.03*kernel1 + 0.33*kernel2 +0.63*kernel3
exp_kernel = kernel1
matern3_kernel = kernel2
matern5_kernel = kernel3
# Create a GaussianProcessRegressor with the mixture kernel
gp = GaussianProcessRegressor(kernel=mixture_kernel, random_state=42)
gp_exp = GaussianProcessRegressor(kernel=exp_kernel, random_state=42)
gp_matern3 = GaussianProcessRegressor(kernel=matern3_kernel, random_state=42)
gp_matern5 = GaussianProcessRegressor(kernel=matern5_kernel, random_state=42)
# Generate sample input data (reshape for scikit-learn compatibility)
X = np.linspace(0, 10, 200).reshape(-1, 1)

# Fit the Gaussian process to the input data (this is a dummy fit, as we don't have observations yet)
# gp.fit(X, np.zeros_like(X))
# gp_exp.fit(X, np.zeros_like(X))
# gp_matern.fit(X, np.zeros_like(X))
# Simulate y values from the mixture of Matern kernels
n_sample = 3
y = gp.sample_y(X, n_samples=3, random_state=42).reshape(-1,n_sample)
y_exp = gp_exp.sample_y(X, n_samples=3, random_state=42).reshape(-1,n_sample)
y_matern3 = gp_matern3.sample_y(X, n_samples=3, random_state=42).reshape(-1,n_sample)
y_matern5 = gp_matern5.sample_y(X, n_samples=3, random_state=42).reshape(-1,n_sample)
print(y)

# Plot the single simulated data
plt.plot(X, y, label='mixture', linestyle='-', linewidth=2)
plt.plot(X, y_exp, label='matern1/2', linestyle='--', linewidth=2)
plt.plot(X, y_matern3, label='matern3/2', linestyle='-.', linewidth=2)
plt.plot(X, y_matern5, label='matern5/2', linestyle='-.', linewidth=2)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simulated data from a mixture of Matern kernels")
plt.legend()
#plt.show()


plt.clf()
# Create a figure and two subplots
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(5, 10))

# Define a list of colors (you can use other colors or use colormap)
colors = ['r', 'g', 'b']

# Access the subplots in the 2x2 grid using their indices
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]
ax4 = axs[3]

# Plot each column of y versus x with different colors in the first subplot
for i in range(y.shape[1]):
    ax1.plot(X, y[:, i], color=colors[i], label=f'Sample {i} vs x')

# Add a legend, labels, and title to the first subplot
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Mixture kernel')

# Plot each column of y1 versus x with different colors in the second subplot
for i in range(y_exp.shape[1]):
    ax2.plot(X, y_exp[:, i], color=colors[i], label=f'Sample {i} vs x')

# Add a legend, labels, and title to the second subplot
ax2.legend()
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title(r'$Mat\'ern$ 1/2 kernel')

# Plot each column of y1 versus x with different colors in the second subplot
for i in range(y_matern3.shape[1]):
    ax3.plot(X, y_matern3[:, i], color=colors[i], label=f'Sample {i} vs x')

# Add a legend, labels, and title to the second subplot
ax3.legend()
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title(r'$Mat\'ern$ 3/2 kernel')

# Plot each column of y1 versus x with different colors in the second subplot
for i in range(y_matern5.shape[1]):
    ax4.plot(X, y_matern5[:, i], color=colors[i], label=f'Sample {i} vs x')

# Add a legend, labels, and title to the second subplot
ax4.legend()
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title(r'$Mat\'ern$ 5/2 kernel')

ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')
ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')
ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top')
ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top')

# Add some space between subplots
plt.tight_layout()

# Show the plot
plt.savefig("./simulation1.png",dpi=300)