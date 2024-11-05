import json
import matplotlib.pyplot as plt
import numpy as np

# Function to load data from multiple JSON files
def load_data(json_files):
    all_eigenvalues = []
    all_losses = []
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        eigenvalues = []
        losses = []

        if 'eigenvalues' in data:
            eigenvalues = [np.array(eigen) for eigen in data['eigenvalues']]
            all_eigenvalues.append(eigenvalues)
        
        if 'losses' in data:
            losses = [np.array(loss) for loss in data['losses'] if np.sum(loss) < 1e8]
            all_losses.append(losses)

        yield eigenvalues, losses

# Specify the list of JSON files
json_files = ['0.0001_100seeds.json', '0.1_100seeds.json', 'NTK_gaussian_100seeds.json', 'NTK_decay_100seeds.json']  # Replace with your list of JSON files

# Define a list of colors for different files
colors = ['blue', 'green', 'orange', 'purple']

# Plot for each file individually
all_median_losses = []
for file_index, (eigenvalues, losses) in enumerate(load_data(json_files)):
    # Plotting eigenvalues for each file
    if eigenvalues:
        plt.figure(figsize=(10, 6))
        for eigen in eigenvalues:
            plt.plot(range(len(eigen)), eigen, color='grey', alpha=0.5)
        
        # Calculate the median across seeds at each epoch
        median_eigenvalues = np.median(np.array(eigenvalues), axis=0)
        plt.plot(range(len(median_eigenvalues)), median_eigenvalues, color=colors[file_index], linewidth=2, label='Median Eigenvalues')

        # Set logarithmic scale for y-axis
        plt.yscale('log')

        # Add labels and title
        plt.xlabel('i', fontsize='medium')
        plt.ylabel(r'$\lambda$', fontsize='medium')
        plt.title(f'Eigenvalues for File {json_files[file_index]}', fontsize='medium')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Plotting losses for each file
    if losses:
        plt.figure(figsize=(10, 6))
        for loss in losses:
            plt.plot(range(len(loss)), loss, color='grey', alpha=0.5)

        # Calculate the median losses across filtered seeds at each epoch
        median_loss = np.median(np.array(losses), axis=0)
        all_median_losses.append((median_loss, json_files[file_index]))
        plt.plot(range(len(median_loss)), median_loss, color=colors[file_index], linewidth=2, label='Median Loss')

        # Set logarithmic scale for y-axis
        plt.yscale('log')

        # Add labels and title
        plt.xlabel(r'Epochs $(\tau)$', fontsize='medium')
        plt.ylabel('Loss', fontsize='medium')
        plt.title(f'Training Loss over Epochs for File {json_files[file_index]}', fontsize='medium')
        plt.grid(True)
        plt.legend()
        plt.show()

# Plotting median losses across all files
if all_median_losses:
    plt.figure(figsize=(10, 6))
    for index, (median_loss, filename) in enumerate(all_median_losses):
        plt.plot(range(len(median_loss)), median_loss, color=colors[index], alpha=0.7, linewidth=2, label=f'Median Loss - {filename}')

    # Calculate the median of the medians across all files
    overall_median_loss = np.median(np.array([median_loss for median_loss, _ in all_median_losses]), axis=0)
    plt.plot(range(len(overall_median_loss)), overall_median_loss, color='red', linewidth=2, label='Overall Median Loss')

    # Set logarithmic scale for y-axis
    plt.yscale('log')

    # Add labels and title
    plt.xlabel(r'Epochs $(\tau)$', fontsize='medium')
    plt.ylabel('Loss', fontsize='medium')
    plt.title(f'Median Training Loss over Epochs for All Files', fontsize='medium')
    plt.grid(True)
    plt.legend()
    plt.show()
