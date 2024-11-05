import torch
import json
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

    # Sort train and test actuals and predictions separately
    sorted_train_indices = np.argsort(y_train_unscaled)
    sorted_train_actuals = y_train_unscaled[sorted_train_indices]
    sorted_train_predictions = train_predictions[sorted_train_indices]

    sorted_test_indices = np.argsort(y_test_unscaled)
    sorted_test_actuals = y_test_unscaled[sorted_test_indices]
    sorted_test_predictions = test_predictions[sorted_test_indices]

    # Create subplots for train and test data
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot sorted train actuals and predictions
    axs[0].plot(range(len(sorted_train_actuals)), sorted_train_actuals, '.', markersize=2, color='green', label="Train Actuals")
    axs[0].plot(range(len(sorted_train_predictions)), sorted_train_predictions, '.', markersize=0.5, color='blue', label="Train Predictions")
    axs[0].set_ylabel('Price [$]', fontsize=16)
    axs[0].set_title('Sorted Train/Test Actuals and Predictions',fontsize=20)
    axs[0].grid(True)
    axs[0].set_yticklabels([f'{int(tick/1000)}k' for tick in axs[0].get_yticks()])

    # Create custom legend markers with bigger size
    train_actual_marker = mlines.Line2D([], [], color='green', marker='.', linestyle='None', markersize=6, label='Train Actuals')
    train_prediction_marker = mlines.Line2D([], [], color='blue', marker='.', linestyle='None', markersize=4, label='Train Predictions')
    axs[0].legend(handles=[train_actual_marker, train_prediction_marker], fontsize=16)

    # Plot sorted test actuals and predictions
    axs[1].plot(range(len(sorted_test_actuals)), sorted_test_actuals, '.', markersize=2, color='orange', label="Test Actuals")
    axs[1].plot(range(len(sorted_test_predictions)), sorted_test_predictions, '.', markersize=0.5, color='red', label="Test Predictions")
    axs[1].set_xlabel('Sorted Data Points', fontsize=16)
    axs[1].set_ylabel('Price [$]', fontsize=16)
    axs[1].grid(True)
    axs[1].set_yticklabels([f'{int(tick/1000)}k' for tick in axs[1].get_yticks()])

    # Create custom legend markers with bigger size
    test_actual_marker = mlines.Line2D([], [], color='orange', marker='.', linestyle='None', markersize=6, label='Test Actuals')
    test_prediction_marker = mlines.Line2D([], [], color='red', marker='.', linestyle='None', markersize=4, label='Test Predictions')
    axs[1].legend(handles=[test_actual_marker, test_prediction_marker], fontsize=16)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the data from the saved JSON file
    data = load_json_data('cali_mega_train.json')
    
    # Plot the actuals and predictions
    plot_actuals_and_predictions(data)
