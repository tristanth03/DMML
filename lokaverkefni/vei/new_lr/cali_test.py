import torch
import json
import numpy as np
from tristans_method_0 import FeedForwardNN, NTK
import torch.nn as nn
from tqdm import tqdm
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the California housing data
def load_data(seed=42):
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Scaling the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.8, random_state=seed)
    
    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

# Function to run tests with random seeds
def run_test(num_runs, Eigenvalues=True, NTK_train=True):
    results = {
        'eigenvalues': []
    }
    
    start_time = time.time()
    
    with tqdm(total=num_runs, desc="Running tests", unit="run") as pbar:
        for seed in range(num_runs):
            torch.manual_seed(seed)  # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Load data with specific seed
            X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_data(seed=seed)
            
            # Define network parameters
            input_dim = X_train_tensor.shape[1]
            output_dim = y_train_tensor.shape[1]
            hidden_layers = [256, 64, 8]
            
            # Initialize the model
            model = FeedForwardNN(X_train_tensor, y_train_tensor, hidden_layers, activation_func=nn.ReLU())
            
            if NTK_train and Eigenvalues:
                ntk = NTK(X_train_tensor, X_train_tensor, model)
                kernel_matrix, eigenvalues = ntk.compute_ntk()
                eigen = eigenvalues.detach().numpy()
                eigen = eigen[::-1]
                results['eigenvalues'].append(eigen.tolist())
            
            # Update progress bar
            pbar.update(1)
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (seed + 1)) * num_runs
            pbar.set_postfix({"Elapsed Time": f"{elapsed_time:.2f}s", "ETA": f"{estimated_total_time - elapsed_time:.2f}s"})
    
    # Remove empty lists if the corresponding value is not calculated
    if not Eigenvalues or not NTK_train:
        results.pop('eigenvalues', None)
    
    # Save results to JSON file
    with open('cali_eigens_20p.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    print("Results saved to california_housing_results.json")

if __name__ == "__main__":
    run_test(100, Eigenvalues=True, NTK_train=True)
