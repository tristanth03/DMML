import json
import matplotlib.pyplot as plt
import numpy as np

# Function to load data from JSON files
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    eigenvalues = np.array(data.get('eigenvalues', []))
    losses = np.array(data.get('train_losses', []))
    test_loss = data.get('test_loss', None)
    
    return eigenvalues, losses, test_loss

# Define file paths and labels
eigen_file = 'cali_mega_train_NoDecay.json'  # File containing only eigenvalues
loss_files = ['cali_mega_train_NoDecay.json', 'cali_mega_train.json']  # Files containing losses
labels = [r'$\eta = 1/\lambda_{\text{max}}$', 'TS-decay']
colors = ['blue', 'red']

# Plot eigenvalues from the eigenvalues file
eigenvalues, _, _ = load_data(eigen_file)
if eigenvalues.size > 0:
    plt.figure(figsize=(10, 6))
    
    # Check if eigenvalues is a 2D array to calculate the median across epochs
    if eigenvalues.ndim > 1:
        median_eigenvalues = np.median(eigenvalues, axis=0)
        plt.plot(range(len(median_eigenvalues)), median_eigenvalues, color='black', linewidth=2, label='Median Eigenvalues')
    else:
        # Plot directly if eigenvalues is a single list
        plt.plot(range(len(eigenvalues)), eigenvalues[::-1], color='black', linewidth=2, label='Eigenvalues')

    plt.yscale('log')
    plt.xlabel('i', fontsize=18)
    plt.ylabel(r'$\lambda$', fontsize=18)
    plt.title('Eigenvalues', fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.show()

# List to store median losses for each loss file and test losses
median_losses_per_file = []
test_losses_per_file = []

# Plot losses for each file in loss_files
for file_index, file_path in enumerate(loss_files):
    _, losses, test_loss = load_data(file_path)
    test_losses_per_file.append(test_loss)
    
    if losses.size > 0:
        plt.figure(figsize=(10, 6))
        
        # Plot all losses if losses is 2D
        if losses.ndim > 1:
            plt.plot(range(losses.shape[1]), losses.T, color='grey', alpha=0.5)
            # Calculate and plot median losses
            median_loss = np.median(losses, axis=0)
        else:
            # If losses is 1D, use it directly
            median_loss = losses

        median_losses_per_file.append(median_loss)
        plt.plot(range(len(median_loss)), median_loss, color=colors[file_index], linewidth=2, label=labels[file_index])

        plt.yscale('log')
        plt.xlabel(r'Epochs $(\tau)$', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.title(f'Training Loss over Epochs for {labels[file_index]}', fontsize=20)
        plt.grid(True)
        plt.legend(fontsize=18)
        plt.show()

# Plot median of each loss for all files
plt.figure(figsize=(10, 6))
for file_index, median_loss in enumerate(median_losses_per_file):
    plt.plot(range(len(median_loss)), median_loss, color=colors[file_index], linewidth=2, label=labels[file_index])

plt.yscale('log')
plt.xlabel(r'Epochs $(\tau)$', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.title('Training Loss over Epochs', fontsize=20)
plt.grid(True)
plt.legend(fontsize=18)
plt.show()

# Print final loss for each label and test losses
for file_index, median_loss in enumerate(median_losses_per_file):
    final_loss = median_loss[-1] if len(median_loss) > 1 else median_loss
    print(f'Final training loss for {labels[file_index]}: {final_loss:.6f}')
    print(f'Test loss for {labels[file_index]}: {test_losses_per_file[file_index]:.6f}')
