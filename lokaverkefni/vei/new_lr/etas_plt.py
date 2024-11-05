import json
import matplotlib.pyplot as plt
import numpy as np

# Function to plot eta values from JSON file
def plot_etas(json_file):
    # Load eta values from JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)
        etas = data['etas']
    
    # Plot eta values for each run in gray with transparency
    plt.figure(figsize=(10, 6))
    for eta_values in etas:
        plt.plot(eta_values, color='grey', alpha=0.5)
    
    # Plot the median eta values across all runs in blue
    median_etas = np.median(etas, axis=0)
    plt.plot(median_etas, color='blue', linewidth=2)
    
    # Set logarithmic scale for y-axis
    plt.yscale('log')
    
    # Add labels and title with current font size
    plt.xlabel('Epoch', fontsize='medium')
    plt.ylabel('Eta Value', fontsize='medium')
    plt.title(f'Eta Values Over Epochs for {len(etas)} Different Random Seeds', fontsize='medium')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_etas('etas_100seeds.json')
