import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file with eigenvalues and losses
data_file_path = '0.0001_100seeds.json'  # Replace with your JSON file path
with open(data_file_path, 'r') as f:
    data = json.load(f)

# Extract eigenvalue and loss lists for each random seed
all_eigenvalues = []
all_losses = []

if 'eigenvalues' in data:
    all_eigenvalues = [np.array(eigenvalues) for eigenvalues in data['eigenvalues']]

if 'losses' in data:
    all_losses = [np.array(losses) for losses in data['losses']]

# Plotting eigenvalues
if all_eigenvalues:
    plt.figure(figsize=(10, 6))
    for eigenvalues in all_eigenvalues:
        plt.plot(range(len(eigenvalues)), eigenvalues, color='grey', alpha=0.5)

    # Plot the median eigenvalues across all seeds in blue
    median_eigenvalues = np.median(all_eigenvalues, axis=0)
    plt.plot(range(len(median_eigenvalues)), median_eigenvalues, color='blue', linewidth=2)

    # Set logarithmic scale for y-axis
    plt.yscale('log')

    # Add labels and title with current font size
    plt.xlabel('i', fontsize='medium')
    plt.ylabel(r'$\lambda$', fontsize='medium')
    plt.title(f'Eigenvalues for {len(all_eigenvalues)} Different Random Seeds', fontsize='medium')
    plt.grid(True)
    plt.show()

# Plotting losses
if all_losses:
    plt.figure(figsize=(10, 6))
    for losses in all_losses:
        plt.plot(range(len(losses)), losses, color='grey', alpha=0.5)

    # Plot the median losses across all seeds in blue
    median_losses = np.median(all_losses, axis=0)
    plt.plot(range(len(median_losses)), median_losses, color='blue', linewidth=2)

    # Set logarithmic scale for y-axis
    plt.yscale('log')

    # Add labels and title with current font size
    plt.xlabel(r'Epochs $(\tau)$', fontsize='medium')
    plt.ylabel('Loss', fontsize='medium')
    plt.title(f'Training Loss over Epochs for {len(all_losses)} Different Random Seeds', fontsize='medium')
    plt.grid(True)
    plt.show()
