import torch
import matplotlib.pyplot as plt
from tristans_method_0 import FeedForwardNN, NTK, Train, Trist_train
import numpy as np
import torch.nn as nn
from tqdm import tqdm


# Generate data using noisy Gaussian wave function
def noisy_gaussian_wave_function(x, k=10, sigma=0.3, A=1, alpha=0.1):
    epsilon = torch.normal(mean=0.0, std=1.0, size=x.shape)  # Generate noise (epsilon) from N(0,1)
    return A * torch.exp(-(x**2) / (2 * sigma**2)) * torch.cos(k * x + epsilon * alpha)





if __name__ == "__main__":
    # Generate input data
    x_vals = torch.linspace(-1, 1, 1000).view(-1, 1)
    psi_vals = noisy_gaussian_wave_function(x_vals).view(-1, 1)

    # Define network parameters
    input_dim = x_vals.shape[1]
    output_dim = psi_vals.shape[1]
    hidden_layers = [256, 64, 8]

    # Initialize the model
    model = FeedForwardNN(x_vals, psi_vals, hidden_layers,activation_func=nn.ReLU())
   

    
    ntk = NTK(x_vals, x_vals, model)
    kernel_matrix, eigenvalues = ntk.compute_ntk()
    eigen = eigenvalues.detach().numpy()

    eigen = eigen[::-1]
    trainer = Trist_train(x_vals,psi_vals,model,eigen,opt=1,epochs=10000,decay=False)
    losses, predictions = trainer.T_train_model()
    print(1/(10*eigen[0]))
    