import json
import numpy as np
import matplotlib.pyplot as plt
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

# Plot the fit
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, '.', label="Training Data", markersize=2)
plt.plot(X_test, y_test, '.', label="Testing Data", markersize=2)
plt.plot(X_train, train_predictions, '.', markersize=3, label="Model Predictions (Train)")
plt.plot(X_test, test_predictions, '.', markersize=3, label="Model Predictions (Test)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper left')
plt.title("Model Fit vs True Data (Train and Test)")
plt.grid(True)

# Add parameters of the Gaussian wave packet in a text box
param_text = (r" $A = 1$"
              r" $\sigma = 0.3$"
              r" $k = 10$"
              r" $\alpha = 0.1$")
plt.text(0.95, 0.95, param_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white"))

# Add number of datapoints in a text box at the bottom
plt.text(0.05, 0.05, f"N = {len(x_vals)}", transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='bottom', horizontalalignment='left',
         bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white"))

plt.show()

print(data["test_loss"])
print(data["train_losses"][-1])