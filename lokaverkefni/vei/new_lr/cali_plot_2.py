import torch
import json
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load and preprocess the California housing data
def load_data(seed=1234):
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Scaling the input features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed)
    
    return y_train, y_test, scaler_y

# Load the JSON data
def load_json_data(filepath):
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    return data

# Plot actuals and predictions
def plot_actuals_and_predictions(data):
    # Load the actual values
    y_train, y_test, scaler_y = load_data(seed=1234)
    
    # Unscale actual values
    y_train_unscaled = scaler_y.inverse_transform(y_train).flatten() * 100000
    y_test_unscaled = scaler_y.inverse_transform(y_test).flatten() * 100000

    # Extract predictions from JSON
    train_predictions = np.array(data.get('train_predictions', [])).flatten()
    test_predictions = np.array(data.get('test_predictions', [])).flatten()

    # Create subplots for PDFs
    fig, axs_pdf = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle('Train and Test Actuals and Predictions PDF', fontsize=20)

    # Calculate and plot PDF for train actuals and predictions
    train_actual_kde = gaussian_kde(y_train_unscaled)
    train_predictions_kde = gaussian_kde(train_predictions)
    x_range = np.linspace(min(y_train_unscaled.min(), train_predictions.min()), max(y_train_unscaled.max(), train_predictions.max()), 1000)
    axs_pdf[0].plot(x_range, train_actual_kde(x_range), color='green', label="Train Actuals PDF")
    axs_pdf[0].plot(x_range, train_predictions_kde(x_range), color='blue', label="Train Predictions PDF")
    axs_pdf[0].legend(fontsize=16)
    axs_pdf[0].grid(True)

    # Format ticks for x-axis
    
    axs_pdf[0].set_xticklabels([f'{int(tick/1000)}k' for tick in axs_pdf[0].get_xticks()])

    # Calculate and plot PDF for test actuals and predictions
    test_actual_kde = gaussian_kde(y_test_unscaled)
    test_predictions_kde = gaussian_kde(test_predictions)
    x_range = np.linspace(min(y_test_unscaled.min(), test_predictions.min()), max(y_test_unscaled.max(), test_predictions.max()), 1000)
    axs_pdf[1].plot(x_range, test_actual_kde(x_range), color='orange', label="Test Actuals PDF")
    axs_pdf[1].plot(x_range, test_predictions_kde(x_range), color='red', label="Test Predictions PDF")
    axs_pdf[1].legend(fontsize=16)
    axs_pdf[1].grid(True)

    # Format ticks for x-axis
    
    axs_pdf[1].set_xticklabels([f'{int(tick/1000)}k' for tick in axs_pdf[1].get_xticks()])

    # Set axis labels
    axs_pdf[0].set_ylabel('Density', fontsize=16)
    axs_pdf[1].set_ylabel('Density', fontsize=16)
    axs_pdf[1].set_xlabel('Price [$]', fontsize=16)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the data from the saved JSON file
    data = load_json_data('cali_mega_train_NoDecay.json')
    
    # Plot the actuals and predictions
    plot_actuals_and_predictions(data)
