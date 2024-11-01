import torch
import matplotlib.pyplot as plt
from klasar import FeedForwardNN, NTK, Train
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

    # Split data into training and testing sets
    x_train, x_test, psi_train, psi_test = train_test_split(x_vals, psi_vals, test_size=0.2, random_state=1234)

    # Define network parameters
    input_dim = x_vals.shape[1]
    output_dim = psi_vals.shape[1]
    hidden_layers = [128, 64]

    # Initialize the model
    model = FeedForwardNN(x_train, psi_train, hidden_layers)

    ntk = NTK(x_train, x_train, model)
    kernel_matrix, eigenvalues = ntk.compute_ntk()
    eigen = eigenvalues.detach().numpy()
    eta = 0.0001
    trainer = Train(x_train, psi_train, model, opt=1, epochs=10000, learning_rate=eta)
    losses, predictions_train = trainer.train_model()

    # Get predictions for test data
    predictions_test = model(x_test)

    # Plot the eigenvalues
    plt.figure(figsize=(10, 6))
    plt.plot(eigen, '.', label="Eigenvalues")
    plt.xlabel("i")
    plt.ylabel(fr"$\lambda_i$")
    plt.yscale('log')
    plt.legend()
    plt.title("Eigenvalues")
    plt.grid(True)
    plt.show()

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
    plt.plot(x_train.numpy(), psi_train.numpy(), '.', label="Training Data", markersize=2)
    plt.plot(x_test.numpy(), psi_test.numpy(), '.', label="Testing Data", markersize=2)
    plt.plot(x_train.numpy(), predictions_train.detach().numpy(), '.', markersize=3, label="Model Predictions (Train)",)
    plt.plot(x_test.numpy(), predictions_test.detach().numpy(), '.',markersize=3, label="Model Predictions (Test)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='upper left')
    plt.title("Model Fit vs True Data (Train and Test)")
    plt.grid(True)

    # Add parameters of the Gaussian wave packet in a text box
    param_text = (r" $A = 1$"
                  r" $\sigma = 0.3$"
                  r" $k = 10$"
                  r" $\alpha = 0.1$")
    plt.text(0.95, 0.95, param_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white"))

    # Add number of datapoints in a text box at the bottom
    plt.text(0.05, 0.05, f"N = {len(x_vals)}", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white"))

    plt.show()
