import torch
import matplotlib.pyplot as plt
from tristans_method_0 import FeedForwardNN, NTK, Train, Trist_train
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

torch.manual_seed(1234)

# Generate data using noisy Gaussian wave function
def noisy_gaussian_wave_function(x, k=10, sigma=0.3, A=1, alpha=0.1):
    epsilon = torch.normal(mean=0.0, std=1.0, size=x.shape)  # Generate noise (epsilon) from N(0,1)
    return A * torch.exp(-(x**2) / (2 * sigma**2)) * torch.cos(k * x + epsilon * alpha)

if __name__ == "__main__":
    # Generate input data
    x_vals = torch.linspace(-1, 1, 1000).view(-1, 1)
    psi_vals = noisy_gaussian_wave_function(x_vals).view(-1, 1)

    # Convert data to numpy arrays for splitting
    x_vals_np = x_vals.numpy()
    psi_vals_np = psi_vals.numpy()

    # Split data into training and testing sets (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(x_vals_np, psi_vals_np, test_size=0.2, random_state=1234)

    # Convert back to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Define network parameters
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    hidden_layers = [256, 64, 8]

    # Initialize the model
    model = FeedForwardNN(X_train, y_train, hidden_layers, activation_func=nn.ReLU())

    # Compute NTK using training data
    ntk = NTK(X_train, X_train, model)
    kernel_matrix, eigenvalues = ntk.compute_ntk()
    eigen = eigenvalues.detach().numpy()

    eigen = eigen[::-1]

    # Train the model using training data
    trainer = Trist_train(X_train, y_train, model, eigen, opt=1, epochs=100000, decay=True)
    losses, predictions = trainer.T_train_model()

    # Evaluate the model on the testing data
    model.eval()
    loss_function = nn.MSELoss()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = loss_function(test_predictions, y_test).item()

    # Save training results
    # Convert predictions and test predictions to lists before adding to data_to_save
    data_to_save = {
        "train_losses": [float(loss) for loss in losses],  # Convert losses to float for JSON compatibility
        "train_predictions": [prediction.tolist() for prediction in predictions],  # Convert each tensor to list
        "test_predictions": test_predictions.tolist(),  # Convert tensor to list for JSON compatibility
        "test_loss": float(test_loss)  # Convert loss to float for JSON compatibility
    }


    with open("decay_megafit_100k.json", "w") as f:
        json.dump(data_to_save, f)

    print("Predictions, losses, and test loss saved to mega_train_gaussian.json")
