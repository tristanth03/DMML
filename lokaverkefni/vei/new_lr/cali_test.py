import torch
import json
import numpy as np
from klasar import FeedForwardNN, NTK, Train
import torch.nn as nn
from tqdm import tqdm
import time
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Load and preprocess the California housing data
def load_data():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Scaling the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert the data to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X_tensor, y_tensor

# Function to run tests with random seeds
def run_test(num_runs, Eigenvalues=True, Loss=True, NTK_train=True, fixed_lr=0.001):
    results = {
        'eigenvalues': [],
        'losses': []
    }
    
    start_time = time.time()
    
    # Load data
    X_tensor, y_tensor = load_data()
    
    with tqdm(total=num_runs, desc="Running tests", unit="run") as pbar:
        for seed in range(num_runs):
            torch.manual_seed(seed)  # Set random seed for reproducibility
            
            # Define network parameters
            input_dim = X_tensor.shape[1]
            output_dim = y_tensor.shape[1]
            hidden_layers = [256, 64]
            
            # Initialize the model
            model = FeedForwardNN(X_tensor, y_tensor, hidden_layers, activation_func=nn.ReLU())
            
            if NTK_train:
                if Eigenvalues:
                    ntk = NTK(X_tensor, X_tensor, model)
                    kernel_matrix, eigenvalues = ntk.compute_ntk()
                    eigen = eigenvalues.detach().numpy()
                    eigen = eigen[::-1]
                    results['eigenvalues'].append(eigen.tolist())
                
                eta = 1 / eigen[-1] if Eigenvalues else fixed_lr
                trainer = Train(X_tensor, y_tensor, model, opt=1, epochs=10000, learning_rate=eta, progress_bar=False)
                losses, predictions = trainer.train_model()
                if Loss:
                    results['losses'].append([loss.item() for loss in losses])
            else:
                trainer = Train(X_tensor, y_tensor, model, opt=1, epochs=10000, learning_rate=fixed_lr, progress_bar=False)
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
    with open('california_housing_results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    print("Results saved to california_housing_results.json")

if __name__ == "__main__":
    run_test(100, Eigenvalues=True, Loss=True, NTK_train=True, fixed_lr=0.1)
