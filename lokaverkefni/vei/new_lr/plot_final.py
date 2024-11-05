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
            eigenvalues = [np.array(eigen[::-1]) for eigen in data['eigenvalues']]
            all_eigenvalues.append(eigenvalues)
        
        if 'losses' in data:
            losses = [np.array(loss) for loss in data['losses'] if np.sum(loss) < 1e8]
            all_losses.append(losses)

        yield eigenvalues, losses

# Specify the list of JSON files
json_files = ['0.0001_100seeds.json','0.01_100seeds.json', 'NTK_gaussian_100seeds.json', 'NTK_decay_100seeds.json']
# Define a list of colors for different files
colors = ['blue', 'green', 'orange', 'red']
# Define labels for each file
labels = [r'$\eta=0.0001$', r'$\eta=0.01$', r'$\eta=1/\lambda_{\text{max}}$', 'TS-decay']

# List to store median losses for each file
median_losses_per_file = []

# Plot for each file individually
for file_index, (eigenvalues, losses) in enumerate(load_data(json_files)):
    # Plotting eigenvalues for each file
    if eigenvalues:
        plt.figure(figsize=(10, 6))
        for eigen in eigenvalues:
            plt.plot(range(len(eigen)), eigen, color='grey', alpha=0.5)
        
        # Calculate the median across seeds at each epoch
        median_eigenvalues = np.median(np.array(eigenvalues), axis=0)
        plt.plot(range(len(median_eigenvalues)), median_eigenvalues, color='black', linewidth=2, label='Median Eigenvalues')

        # Set logarithmic scale for y-axis
        plt.yscale('log')

        # Add labels and title
        plt.xlabel('i', fontsize=18)
        plt.ylabel(r'$\lambda$', fontsize=18)
        plt.title(f'Eigenvalues', fontsize=20)
        plt.grid(True)
        plt.legend(fontsize=18)
        
        plt.show()

    # Plotting losses for each file
    if losses:
        plt.figure(figsize=(10, 6))
        for loss in losses:
            plt.plot(range(len(loss)), loss, color='grey', alpha=0.5)

        # Calculate the median losses across filtered seeds at each epoch
        median_loss = np.median(np.array(losses), axis=0)
        median_losses_per_file.append(median_loss)
        plt.plot(range(len(median_loss)), median_loss, color=colors[file_index], linewidth=2, label='Median Loss')

        # Set logarithmic scale for y-axis
        plt.yscale('log')

        # Add labels and title
        plt.xlabel(r'Epochs $(\tau)$', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.title(f'Training Loss over Epochs for {labels[file_index]}', fontsize=20)
        plt.grid(True)
        plt.legend(fontsize=18)
 
        plt.show()

# Plotting the median of each loss for each file
plt.figure(figsize=(10, 6))
for file_index, median_loss in enumerate(median_losses_per_file):
    plt.plot(range(len(median_loss)), median_loss, color=colors[file_index], linewidth=2, label=labels[file_index])

# Set logarithmic scale for y-axis
plt.yscale('log')

# Add labels and title
plt.xlabel(r'Epochs $(\tau)$', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.title('Median Training Loss over Epochs', fontsize=20)
plt.grid(True)
plt.legend(fontsize=18)

plt.show()
