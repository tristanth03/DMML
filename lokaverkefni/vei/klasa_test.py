import torch
import matplotlib.pyplot as plt
from klasar import FeedForwardNN, NTK, Train

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
    hidden_layers = [128, 64]

    # Initialize the model
    model = FeedForwardNN(x_vals, psi_vals, hidden_layers)
   

    ntk = NTK(x_vals, x_vals, model)
    kernel_matrix, eigenvalues = ntk.compute_ntk()
    eigen = eigenvalues.detach().numpy()
    eta = 1/eigen[-1]    
    trainer = Train(x_vals, psi_vals, model, opt=1, epochs=1000, learning_rate=eta)
    losses, predictions = trainer.train_model()

    # Plot the loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses.numpy(), label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.show()

    # Plot the fit
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals.numpy(), psi_vals.numpy(), '.', label="True Data", markersize=3)
    plt.plot(x_vals.numpy(), predictions.detach().numpy(), '-', label="Model Predictions", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Model Fit vs True Data")
    plt.grid(True)
    plt.show()


