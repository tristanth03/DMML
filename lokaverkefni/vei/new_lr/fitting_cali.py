import torch
import json
import numpy as np
from tristans_method_0 import FeedForwardNN, NTK, Trist_train
import torch.nn as nn
from tqdm import tqdm
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and preprocess the California housing data
def load_data(seed=1234):
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Scaling the input features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed)
    
    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler_X, scaler_y

# Function to train the model and return the data to save
def train_model(seed=42, epochs=1000, Eigenvalues=True, NTK_train=True):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data with specific seed
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler_X, scaler_y = load_data(seed=seed)
    
    # Define network parameters
    input_dim = X_train_tensor.shape[1]
    output_dim = y_train_tensor.shape[1]
    hidden_layers = [256, 64, 8]
    
    # Initialize the model
    model = FeedForwardNN(X_train_tensor, y_train_tensor, hidden_layers, activation_func=nn.ReLU())
    
    # Compute NTK eigenvalues if required
    if NTK_train and Eigenvalues:
        ntk = NTK(X_train_tensor, X_train_tensor, model)
        kernel_matrix, eigenvalues = ntk.compute_ntk()
        eigen = eigenvalues.detach().numpy()[::-1]
    else:
        eigen = None
    
    # Train the model using Trist_train
    trainer = Trist_train(X_train_tensor, y_train_tensor, model, eigen, opt=1, epochs=epochs, progress_bar=True, decay=False)
    losses, predictions = trainer.T_train_model()
    
    # Unscale predictions for reporting purposes
    y_train_unscaled = scaler_y.inverse_transform(y_train_tensor.detach().numpy()) * 100000
    y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.detach().numpy()) * 100000
    train_predictions_unscaled = scaler_y.inverse_transform(np.array([p.detach().numpy() for p in predictions]).reshape(-1, 1)) * 100000
    test_predictions_unscaled = scaler_y.inverse_transform(model(X_test_tensor).detach().numpy()) * 100000

    # Evaluate test loss on original scaled values (not inverse-scaled)
    model.eval()
    criterion = nn.MSELoss()
    test_loss = criterion(
        model(X_test_tensor),
        y_test_tensor
    )
    
    # Save data
    data_to_save = {
        "train_losses": [float(loss) for loss in losses],
        "train_predictions": train_predictions_unscaled.flatten().tolist(),
        "test_predictions": test_predictions_unscaled.flatten().tolist(),
        "test_loss": float(test_loss),
        "eigenvalues": eigen.tolist() if eigen is not None else None
    }
    
    return data_to_save

if __name__ == "__main__":
    seed = 1234
    epochs = 100000
    results = train_model(seed=seed, epochs=epochs, Eigenvalues=True, NTK_train=True)
    
    # Save results to JSON file
    with open('cali_mega_train_NoDecay.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    print("Results saved to cali_mega_train_NoDecay.json")
