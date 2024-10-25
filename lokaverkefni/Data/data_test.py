import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

torch.manual_seed(12345)
# Define a general feedforward neural network function with L hidden layers
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FeedForwardNN, self).__init__()
        layers = []
        layer_input_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = nn.Linear(layer_input_dim, hidden_dim)
            # Normalize weights by dividing by sqrt of input dimension size
            nn.init.normal_(layer.weight, mean=0, std=1.0 / torch.sqrt(torch.tensor(layer_input_dim, dtype=torch.float32)))
            layers.append(layer)
            layers.append(nn.ReLU())
            layer_input_dim = hidden_dim
        final_layer = nn.Linear(layer_input_dim, output_dim)
        nn.init.normal_(final_layer.weight, mean=0, std=1.0 / torch.sqrt(torch.tensor(layer_input_dim, dtype=torch.float32)))
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Generate noisy Gaussian wave packet data
def noisy_gaussian_wave_function(x, k=10, sigma=0.3, A=1, alpha=0.1):
    epsilon = torch.normal(mean=0.0, std=1.0, size=x.shape)  # Generate noise (epsilon) from N(0,1)
    return A * torch.exp(-(x**2) / (2 * sigma**2)) * torch.cos(k * x + epsilon * alpha)

# Neural Tangent Kernel computation using Torch
def ntk_kernel(model, x1, x2):
    # Compute the Jacobians
    x1 = x1.unsqueeze(0)  # Add batch dimension
    x2 = x2.unsqueeze(0)  # Add batch dimension
    
    y1 = model(x1)
    y2 = model(x2)

    jacobian1 = autograd.functional.jacobian(lambda inp: model(inp).squeeze(), x1, create_graph=True)
    jacobian2 = autograd.functional.jacobian(lambda inp: model(inp).squeeze(), x2, create_graph=True)

    # Flatten the Jacobians
    jacobian1_flat = torch.cat([j.view(-1) for j in jacobian1], dim=0)
    jacobian2_flat = torch.cat([j.view(-1) for j in jacobian2], dim=0)

    # Compute the NTK
    kernel_value = torch.dot(jacobian1_flat, jacobian2_flat)
    return kernel_value

# Compute NTK for multiple data points
def ntk_kernel_matrix(model, X):
    n = X.shape[0]
    jacobians = []
    for i in tqdm(range(n), desc="Calculating Jacobians"):
        x = X[i].unsqueeze(0)
        jacobian = autograd.functional.jacobian(lambda inp: model(inp).squeeze(), x, create_graph=True)
        jacobian_flat = torch.cat([j.view(-1) for j in jacobian], dim=0)
        jacobians.append(jacobian_flat)
    jacobians = torch.stack(jacobians)

    # Compute the NTK kernel matrix using torch.matmul for efficiency
    kernel_matrix = torch.matmul(jacobians, jacobians.T)
    return kernel_matrix

# Example usage with data generation and model training
def train_model(x, targets, model, learning_rate, num_epochs, opt, problem_type):
    if problem_type == "Regression":
        criterion = nn.MSELoss()
    if opt == "VanillaGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    L = []
    for epoch in tqdm(range(num_epochs), desc="Training Model"):
        model.train()
        optimizer.zero_grad()
        y = model(x)  # Entire dataset
        loss = criterion(y, targets)
        loss.backward()
        optimizer.step()
        L.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.16f}")
    return L, model(x)

if __name__ == "__main__":
    # Generate data using noisy Gaussian wave function
    x_vals = torch.linspace(-1, 1, 1000).view(-1, 1)  # Ensure x_vals is 2D
    psi_vals = noisy_gaussian_wave_function(x_vals).view(-1, 1)  # Ensure psi_vals is 2D

    # Define input and network dimensions
    input_dim = x_vals.shape[1]
    hidden_dims = [256,64]  # Two hidden layers with 256 neurons each
    output_dim = psi_vals.shape[1]

    # Initialize model
    model = FeedForwardNN(input_dim, hidden_dims, output_dim)

    # Compute the NTK kernel matrix
    kernel_matrix = ntk_kernel_matrix(model, x_vals)
    print("NTK Kernel Matrix:\n", kernel_matrix)

    # Compute the eigenvalues of the NTK kernel matrix
    eigenvalues = torch.linalg.eigvals(kernel_matrix).real.to(dtype=torch.float32)
    eigenvalues = torch.abs(eigenvalues)
    eigenvalues = torch.sort(eigenvalues)[0]
    print("Eigenvalues of NTK Kernel Matrix:\n", eigenvalues)

    # Train the model
    learning_rate = 1/eigenvalues[-1]
    num_epochs = 5000
    L, y_hat = train_model(x_vals, psi_vals, model, learning_rate, num_epochs, opt='VanillaGD', problem_type='Regression')

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(L, '.', markersize=2)
    plt.title("Loss", fontsize=16)
    plt.xlabel(r"Epoch ($\tau$)", fontsize=16)
    plt.ylabel(r"$\mathcal{L}$", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.yscale('log')
    plt.show()

    # Plot fit
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals.numpy(), psi_vals.numpy(), '.', markersize=3, label="Data")
    plt.plot(x_vals.numpy(), y_hat.detach().numpy(),'.', markersize=3, label="Fit")
    plt.legend()
    plt.title(f"Fit vs Real (fixed learning rate: {learning_rate})", fontsize=16)
    plt.xlabel("x", fontsize=16)
    plt.ylabel(fr"$y(x)$", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12, frameon=True, loc='upper right')
    plt.show()
