import torch
import json
import numpy as np
from tristans_method_0 import FeedForwardNN, NTK, Trist_train
import torch.nn as nn
from tqdm import tqdm

# Generate data using noisy Gaussian wave function
def noisy_gaussian_wave_function(x, k=10, sigma=0.3, A=1, alpha=0.1):
    epsilon = torch.normal(mean=0.0, std=1.0, size=x.shape)  # Generate noise (epsilon) from N(0,1)
    return A * torch.exp(-(x**2) / (2 * sigma**2)) * torch.cos(k * x + epsilon * alpha)

# Function to run tests with random seeds
def run_test(num_runs, Eigenvalues=True, Loss=True, NTK_train=True, fixed_lr=0.001):
    results = {
        'etas': []
    }
    
    with tqdm(total=num_runs, desc="Running tests", unit="run") as pbar:
        for seed in range(num_runs):
            torch.manual_seed(seed)  # Set random seed for reproducibility
            
            # Generate input data
            x_vals = torch.linspace(-1, 1, 1000).view(-1, 1)
            psi_vals = noisy_gaussian_wave_function(x_vals).view(-1, 1)
            
            # Define network parameters
            input_dim = x_vals.shape[1]
            output_dim = psi_vals.shape[1]
            hidden_layers = [256, 64, 8]
            
            # Initialize the model
            model = FeedForwardNN(x_vals, psi_vals, hidden_layers, activation_func=nn.ReLU())
            
            # Compute NTK eigenvalues
            ntk = NTK(x_vals, x_vals, model)
            kernel_matrix, eigenvalues = ntk.compute_ntk()
            eigen = eigenvalues.detach().numpy()
            eigen = eigen[::-1]

            # Training with custom learning rate schedule and saving etas
            epochs = 10000
            lambda_max = eigen[0]
            etas = []
            k = 1
            s = 0
            eta = 10 / lambda_max

            for epoch in range(epochs):
                if eta >= 1 / lambda_max:
                    eta = (10 / lambda_max) - (1 * epoch / epochs) * (10 / lambda_max - 1 / (10 * lambda_max))
                    s = epoch
                elif eta < 1 / lambda_max:
                    eta = (1 / lambda_max) - (0.1 * (epoch - s) / (epochs - s)) * (1 / lambda_max - 1 / (10 * lambda_max))
                    if k == 1:
                        print(f'\n Tristan \n')
                    k += 1
                etas.append(eta)

            results['etas'].append(etas)
            
            # Update progress bar
            pbar.update(1)
    
    # Save etas to JSON file
    with open('etas_100seeds.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    print("Etas saved to etas.json")

if __name__ == "__main__":
    run_test(100, Eigenvalues=False, Loss=True, NTK_train=False, fixed_lr=0.1)
