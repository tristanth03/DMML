import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
np.random.seed(1234)

def noisy_gaussian_wave_function(x, k=10, sigma=0.3, A=1, alpha=0.1):
    epsilon = np.random.normal(0.0, 1.0, size=x.shape)  # Generate noise (epsilon) from N(0,1)
    return A * np.exp(-(x**2) / (2 * sigma**2)) * np.cos(k * x + epsilon * alpha)

# Load data from the JSON file
with open("decay_megafit_100k.json", "r") as f:
    data = json.load(f)

# Extract data from the JSON object and flatten predictions
train_losses = data["train_losses"]
train_predictions = np.array([item[0] for item in data["train_predictions"]]).flatten()
test_predictions = np.array([item[0] for item in data["test_predictions"]]).flatten()

# Generate input data (same as the original script)
x_vals = np.linspace(-1, 1, 1000).reshape(-1, 1)
psi_vals = noisy_gaussian_wave_function(x_vals).reshape(-1, 1)

# Split data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(x_vals, psi_vals, test_size=0.2, random_state=1234)

# Define custom markers for legend
train_actual_marker = mlines.Line2D([], [], color='blue', marker='.', linestyle='None', markersize=6, label='Train Actuals')
test_actual_marker = mlines.Line2D([], [], color='orange', marker='.', linestyle='None', markersize=6, label='Test Actuals')
train_prediction_marker = mlines.Line2D([], [], color='green', marker='.', linestyle='None', markersize=4, label='Train Predictions')
test_prediction_marker = mlines.Line2D([], [], color='red', marker='.', linestyle='None', markersize=4, label='Test Predictions')

# Plot the fit
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, '.', color='blue', markersize=2, label="Train Actuals")
plt.plot(X_test, y_test, '.', color='orange', markersize=2, label="Test Actuals")
plt.plot(X_train, train_predictions, '.', color='green', markersize=3, label="Train Predictions")
plt.plot(X_test, test_predictions, '.', color='red', markersize=3, label="Test Predictions")
plt.xlabel("x", fontsize=16)
plt.ylabel("y", fontsize=16)
plt.legend(handles=[train_actual_marker, test_actual_marker, train_prediction_marker, test_prediction_marker], loc='upper left', fontsize=12)
plt.title("Model Fit vs True Data (Train and Test)", fontsize=20)
plt.grid(True)

# Add parameters of the Gaussian wave packet in a text box
param_text = (r" $A = 1$"
              r" $\sigma = 0.3$"
              r" $k = 10$"
              r" $\alpha = 0.1$")
plt.text(0.95, 0.95, param_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white"))

# Add number of datapoints in a text box at the bottom
plt.text(0.05, 0.05, f"N = {len(x_vals)}", transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='bottom', horizontalalignment='left',
         bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white"))

plt.show()

print("Test Loss:", data["test_loss"])
print("Final Training Loss:", data["train_losses"][-1])
