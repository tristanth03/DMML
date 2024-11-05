import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from klasar import FeedForwardNN, NTK, Train
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
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
    hidden_layers = [256, 64, 8]

    # Initialize the model
    model = FeedForwardNN(x_train, psi_train, hidden_layers)

    # Compute NTK and set learning rate
    ntk = NTK(x_train, x_train, model)
    kernel_matrix, eigenvalues = ntk.compute_ntk()
    eigen = eigenvalues.detach().numpy()
    eta = 0.01
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

    # Define custom markers for the legend
    train_actual_marker = mlines.Line2D([], [], color='blue', marker='.', linestyle='None', markersize=6, label='Train Actuals')
    test_actual_marker = mlines.Line2D([], [], color='orange', marker='.', linestyle='None', markersize=6, label='Test Actuals')
    train_prediction_marker = mlines.Line2D([], [], color='green', marker='.', linestyle='None', markersize=4, label='Train Predictions')
    test_prediction_marker = mlines.Line2D([], [], color='red', marker='.', linestyle='None', markersize=4, label='Test Predictions')

    # Plot the fit
    plt.figure(figsize=(10, 6))
    plt.plot(x_train.numpy(), psi_train.numpy(), '.', color='blue', markersize=2, label="Train Actuals")
    plt.plot(x_test.numpy(), psi_test.numpy(), '.', color='orange', markersize=2, label="Test Actuals")
    plt.plot(x_train.numpy(), predictions_train.detach().numpy(), '.', color='green', markersize=3, label="Train Predictions")
    plt.plot(x_test.numpy(), predictions_test.detach().numpy(), '.', color='red', markersize=3, label="Test Predictions")
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.legend(handles=[train_actual_marker,  test_actual_marker, train_prediction_marker, test_prediction_marker],
               loc='upper left', fontsize=12)
    plt.title("Model Fit vs True Data (Train and Test)", fontsize=20)
    plt.grid(True)

    # Add parameters of the Gaussian wave packet in a text box
    param_text = (r" $A = 1$"
                  r" $\sigma = 0.3$"
                  r" $k = 10$"
                  r" $\alpha = 0.1$")
    plt.text(0.95, 0.95, param_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white"))

    # Add number of datapoints in a text box at the bottom
    plt.text(0.05, 0.05, f"N = {len(x_vals)}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white"))

    plt.show()
