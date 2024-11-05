import json
import matplotlib.pyplot as plt
import numpy as np

# Function to mimic eta decay and find the change epoch
def find_change_epoch(lambda_max, epochs=10000):
    # Initialize eta values list
    eta_values = []
    s = 0
    k = 0

    for epoch in range(epochs):
        if epoch == 0:
            eta = 10 / lambda_max
        elif eta >= 1 / lambda_max:
            eta = (10 / lambda_max) - (1 * epoch / epochs) * (10 / lambda_max - 1 / (10 * lambda_max))
            s = epoch
        elif eta < 1 / lambda_max:
            eta = (1 / lambda_max) - (0.1 * (epoch - s) / (epochs - s)) * (1 / lambda_max - 1 / (10 * lambda_max))
            if k == 1:
                print(f'\n Tristan \n')
            k += 1

        eta_values.append(eta)

    # Convert eta_values to numpy array
    eta_values = np.array(eta_values)

    # Find the epoch where eta is approximately equal to 1/lambda_max
    change_epoch = np.where(np.isclose(eta_values, 1 / lambda_max, rtol=1e-2))[0]
    return eta_values, change_epoch[0] if len(change_epoch) > 0 else None

# Function to plot eta values
def plot_etas(json_file):
    # Load eta values from JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)
        etas = data['etas']
    
    # Plot eta values for each run in gray with transparency
    plt.figure(figsize=(10, 6))
    # for eta_values in etas:
    #     plt.plot(eta_values, color='grey', alpha=0.5)
    
    # Plot the median eta values across all runs in blue
    median_etas = np.median(etas, axis=0)
    plt.plot(median_etas, color='blue', linewidth=2)
    
    # Set logarithmic scale for y-axis
    # plt.yscale('log')
    # Mimic eta decay and find the change point for plotting
    lambda_max = np.max(median_etas)
    _, change_epoch = find_change_epoch(lambda_max)

    if change_epoch is not None:
        # Add red dashed vertical line at the change point
        plt.axvline(x=change_epoch, color='red', linestyle='--', linewidth=1, label=r'$1/\lambda_{\text{max}}$')
        # Add label for the change point
        plt.text(change_epoch, median_etas[change_epoch], r'$1/\lambda_{\text{max}}$', color='red', fontsize=12, verticalalignment='bottom')
    
    # Add labels for specific points
    plt.text(0, median_etas[0], r'$10/\lambda_{\text{max}}$', color='black', fontsize=12, verticalalignment='bottom')
    plt.text(len(median_etas) - 1, median_etas[-1], r'$91/100\lambda_{\text{max}}$', color='black', fontsize=12, verticalalignment='top')
    
    # Add labels and title with current font size
    plt.xlabel(fr'Epoch ($\tau$)', fontsize=18)
    plt.ylabel(fr'$\eta_{{\tau}}$', fontsize=18)
    plt.title(fr'$\eta$ Values Over Epochs', fontsize=20)
   
    plt.xlim(-1000,12000)


    plt.yticks([])
    plt.xticks([])
    plt.show()

# Plot eta values
if __name__ == "__main__":
    plot_etas('etas_100seeds.json')