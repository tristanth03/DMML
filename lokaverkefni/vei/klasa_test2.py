import torch
import matplotlib.pyplot as plt
import numpy as np
from klasar import FeedForwardNN, NTK, Train
from sklearn.model_selection import train_test_split

# Generate data using noisy Gaussian wave function
def noisy_gaussian_wave_function(x, k=10, sigma=0.3, A=1, alpha=0.1):
    epsilon = torch.normal(mean=0.0, std=1.0, size=x.shape)  # Generate noise (epsilon) from N(0,1)
    return A * torch.exp(-(x**2) / (2 * sigma**2)) * torch.cos(k * x + epsilon * alpha)

class ExperimentWithRandomSeeds:
    def __init__(self, x_vals, psi_vals, hidden_layers, num_seeds=100, epochs=10000, test_size=0.2):
        self.x_vals = x_vals
        self.psi_vals = psi_vals
        self.hidden_layers = hidden_layers
        self.num_seeds = num_seeds
        self.epochs = epochs
        self.test_size = test_size

    def run_experiments(self):
        all_eigenvalues = []
        all_losses = []
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            x_train, x_test, psi_train, psi_test = train_test_split(self.x_vals, self.psi_vals, test_size=self.test_size, random_state=seed)

            # Initialize the model
            model = FeedForwardNN(x_train, psi_train, self.hidden_layers)
            
            # Compute NTK kernel matrix
            ntk = NTK(x_train, x_train, model)
            kernel_matrix, eigenvalues = ntk.compute_ntk()
            eigen = eigenvalues.detach().numpy()
            all_eigenvalues.append(eigen)

            # Train the model
            eta = 1 / eigen[-1]
            trainer = Train(x_train, psi_train, model, opt=1, epochs=self.epochs, learning_rate=eta)
            losses, _ = trainer.train_model()
            all_losses.append(losses.numpy())

        self.plot_results(all_eigenvalues, all_losses)

    def plot_results(self, all_eigenvalues, all_losses):
        # Plot the eigenvalues
        plt.figure(figsize=(10, 6))
        for i, eigen in enumerate(all_eigenvalues):
            plt.plot(np.arange(1, len(eigen) + 1), eigen, color='gray', alpha=0.5)
        plt.plot(np.arange(1, len(all_eigenvalues[0]) + 1), np.mean(all_eigenvalues, axis=0), color='blue', linewidth=2)
        plt.xlabel("i")
        plt.ylabel(r"$\lambda_i$")
        plt.yscale('log')
        plt.title(f"Eigenvalues for {self.num_seeds} Different Random Seeds")
        plt.grid(True)
        plt.show()

        # Plot the loss
        plt.figure(figsize=(10, 6))
        for i, losses in enumerate(all_losses):
            plt.plot(np.arange(1, len(losses) + 1), losses, color='gray', alpha=0.5)
        plt.plot(np.arange(1, len(all_losses[0]) + 1), np.mean(all_losses, axis=0), color='blue', linewidth=2)
        plt.xlabel(fr"Epochs $(\tau)$")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.title(f"Training Loss over Epochs for {self.num_seeds} Different Random Seeds")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Generate input data
    x_vals = torch.linspace(-1, 1, 1000).view(-1, 1)
    psi_vals = noisy_gaussian_wave_function(x_vals).view(-1, 1)

    # Define network parameters
    hidden_layers = [128, 64]

    # Run experiment with multiple random seeds
    experiment = ExperimentWithRandomSeeds(x_vals, psi_vals, hidden_layers, num_seeds=100, epochs=10000)
    experiment.run_experiments()
