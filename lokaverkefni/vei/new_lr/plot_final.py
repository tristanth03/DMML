import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file with eigenvalues and losses
data_file_path = '/mnt/data/your_json_file.json'  # Replace with your JSON file path
with open(data_file_path, 'r') as f:
    data = json.load(f)

# Assuming 'data' is structured as:
# {
#     "random_seed_1": {
#         "eigenvalues": [list of eigenvalues],
#         "losses": [list of losses]
#     },
#     "random_seed_2": {
#         "eigenvalues": [list of eigenvalues],
#         "losses": [list of losses]
#     },
#     ...
# }

# Extract eigenvalue and loss lists for each random seed
all_eigenvalues = [np.array(seed_data['eigenvalues']) for seed_data in data.values()]
all_losses = [np.array(seed_data['losses']) for seed_data in data.values()]

# Plotting eigenvalues
plt.figure(figsize=(10, 6))
for eigenvalues in all_eigenvalues:
    plt.plot(range(len(eigenvalues)), eigenvalues, color='grey', alpha=0.5)

# Plot the average eigenvalues across all seeds in blue
average_eigenvalues = np.mean(all_eigenvalues, axis=0)
plt.plot(range(len(average_eigenvalues)), average_eigenvalues, color='blue', linewidth=2)

# Set logarithmic scale for y-axis
plt.yscale('log')

# Add labels and title with current font size
plt.xlabel('i', fontsize='medium')
plt.ylabel(r'$\lambda$', fontsize='medium')
plt.title(f'Eigenvalues for {len(data)} Different Random Seeds', fontsize='medium')
plt.grid(True)
plt.show()

# Plotting losses
plt.figure(figsize=(10, 6))
for losses in all_losses:
    plt.plot(range(len(losses)), losses, color='grey', alpha=0.5)

# Plot the average losses across all seeds in blue
average_losses = np.mean(all_losses, axis=0)
plt.plot(range(len(average_losses)), average_losses, color='blue', linewidth=2)

# Set logarithmic scale for y-axis
plt.yscale('log')

# Add labels and title with current font size
plt.xlabel(r'Epochs $(\tau)$', fontsize='medium')
plt.ylabel('Loss', fontsize='medium')
plt.title(f'Training Loss over Epochs for {len(data)} Different Random Seeds', fontsize='medium')
plt.grid(True)
plt.show()
