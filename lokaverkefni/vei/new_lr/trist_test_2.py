import torch
import json
import numpy as np
from tristans_method_0 import FeedForwardNN, NTK, Train, Trist_train
import torch.nn as nn
from tqdm import tqdm
import time

# Generate data using noisy Gaussian wave function
def noisy_gaussian_wave_function(x, k=10, sigma=0.3, A=1, alpha=0.1):
    epsilon = torch.normal(mean=0.0, std=1.0, size=x.shape)  # Generate noise (epsilon) from N(0,1)
    return A * torch.exp(-(x**2) / (2 * sigma**2)) * torch.cos(k * x + epsilon * alpha)

# Function to run tests with random seeds
def run_test(num_runs, Eigenvalues=True, Loss=True, NTK_train=True, fixed_lr=0.001):
    results = {
        'eigenvalues': [],
        'losses': []
    }
    
    start_time = time.time()
    
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
            
            if NTK_train:
                if Eigenvalues:
                    ntk = NTK(x_vals, x_vals, model)
                    kernel_matrix, eigenvalues = ntk.compute_ntk()
                    eigen = eigenvalues.detach().numpy()
                    eigen = eigen[::-1]
                    results['eigenvalues'].append(eigen.tolist())
                
                trainer = Trist_train(x_vals, psi_vals, model, eigen, opt=1, epochs=10000, decay=False, progress_bar=False)
                losses, predictions = trainer.T_train_model()
                if Loss:
                    results['losses'].append([loss.item() for loss in losses])
            else:
                trainer = Train(x_vals, psi_vals, model, opt=1, epochs=10000, learning_rate=fixed_lr, progress_bar=False)
                losses, predictions = trainer.train_model()
                if Loss:
                    results['losses'].append([loss.item() for loss in losses])
            
            # Update progress bar
            pbar.update(1)
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (seed + 1)) * num_runs
            pbar.set_postfix({"Elapsed Time": f"{elapsed_time:.2f}s", "ETA": f"{estimated_total_time - elapsed_time:.2f}s"})
    
    # Remove empty lists if the corresponding value is not calculated
    if not Eigenvalues or not NTK_train:
        results.pop('eigenvalues', None)
    
    # Save results to JSON file
    with open('0.0001_100seeds.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    print("Results saved to NTK_gaussian_100seeds.json")

if __name__ == "__main__":
    run_test(100, Eigenvalues=False, Loss=True, NTK_train=False, fixed_lr=0.0001)
