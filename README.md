# On the Identifiability and Interpretability of Gaussian Process Models

This repository houses the complete codebase and estimated parameters for the research study titled “On the Identifiability and Interpretability of Gaussian Process Models”. 

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

- **ap1**
  - `diamond_plate.png` - Image employed in application 1 downloaded from [Wikipedia](https://en.wikipedia.org/wiki/Tread_plate#/media/File:Diamond_Plate.jpg).
  - `ap1_image_texture.py` - Python script for running application 1 and generating Figure 4
  
- **ap2**
  - `housing.csv` - Boston housing dataset used in application 2. This dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices).
  - `Boston_result` - Parameter estimates from application 2
  - `ap2_boston_housing.py` - Python script for replicating application 2 and generating Figure 5

- **ap3**
  - `result` - Parameter estimates and latent embeddings learned using GPLVM with mixture kernel and Matern 1/2 kernel
  - `gplvm_matern0.5.py` - Python script for GPLVM with Matern 1/2 kernel
  - `gplvm_mixture.py` - Python script for GPLVM with mixture kernel
  - `figure.py` - Python script for generating Figure 6
  - The GPLVM codes are adopted from [gpytorch examples](https://docs.gpytorch.ai/en/latest/examples/045_GPLVM/Gaussian_Process_Latent_Variable_Models_with_Stochastic_Variational_Inference.html).

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
