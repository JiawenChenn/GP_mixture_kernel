# On the Identifiability and Interpretability of Gaussian Process Models

This repository houses the complete codebase and estimated parameters for **On the Identifiability and Interpretability of Gaussian Process Models** ([NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dea2b4f9012686bcc1f59a62bcd28158-Abstract-Conference.html))”

## Repository Structure

### 1. Simulation

This directory contains the code necessary for running the simulations in our study.

- **sim1**
  - `simulation1.py` - Python script for simulation 1 and generating Figure 1

- **sim2**
  - `Simulation2.py` - Python script for simulation 2 and generating Figure 2 and Figure S1

- **sim3**
  - `simulation3.py` - Python script for simulation 3 (Run as "python simulation3.py 20" for n=20 simulation)
  - `figure.py` - Python script for generating Figure 3 and Figure S2

- **additional_sim4**
  - `additional_sim4.py` - Python script for simulation 4 and generating Figure S3 and Figure S4

### 2. Application

This directory contains the code, datasets, and output for the three applications of our study.

- **application 1 (MNIST)**
  - `mnist_0.png` - Image downloaded from MNIST dataset.
  - `mnist.py` - Python script for running application 1.
  
- **application 2 (Mauna Loa CO2)**
  - `co2_scipy.py` - Python script for replicating application 2.


### 3. Mixture_kernel.py
This code defines the class for the mixture kernel used in simulations and applications. You need to import kernels from this file.

## Python Dependencies

Ensure your environment is set up with the following packages:

- python = "^3.8"
- torch = "1.11"
- gpytorch = "^1.9.1"
- matplotlib = "^3.7.0"
- plotly = "^5.13.0"
- pandas = "^1.5.3"
- scanpy = "^1.9.2"
- numpy = "1.23.4"
- imageio = "^2.26.0"
- pyro-ppl = "^1.8.4"
- Pillow = "^9.5.0"
- Edward = "^1.3.5"

All codes have been executed on Tesla V100-SXM2 GPUs or CPUs.

## Citation

```
@article{chen2024identifiability,
  title={On the Identifiability and Interpretability of Gaussian Process Models},
  author={Chen, Jiawen and Mu, Wancen and Li, Yun and Li, Didong},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
