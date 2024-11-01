import torch
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from klasar import FeedForwardNN, NTK, Train

torch.manual_seed(1234)
# Step 1: Load the dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Step 2: Preprocess the data (optional but recommended)
# It's common to scale the input features before training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshape to match the output
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define network parameters
input_dim = X_train_tensor.shape[1]
output_dim = y_train_tensor.shape[1]
hidden_layers = [256, 64]

# Initialize the model
model = FeedForwardNN(X_train_tensor, y_train_tensor, hidden_layers)

# Compute NTK eigenvalues
ntk = NTK(X_train_tensor, X_train_tensor, model)
kernel_matrix, eigenvalues = ntk.compute_ntk()
eigen = eigenvalues.detach().numpy()
eta = 1 / eigen[-1]

# Train the model
trainer = Train(X_train_tensor, y_train_tensor, model, opt=1, epochs=10000, learning_rate=eta)
losses, predictions = trainer.train_model()

# Plot the eigenvalues
plt.figure(figsize=(10, 6))
plt.plot(eigen, '.', label="Eigenvalues")
plt.xlabel("i")
plt.ylabel(fr"$\lambda_i$")
plt.yscale('log')
plt.legend()
plt.title("Eigenvalues")
plt.grid(True)
plt.show()

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(losses.numpy(), label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale('log')
plt.legend()
plt.title("Training Loss over Epochs for California Housing Dataset")
plt.grid(True)
plt.show()

# Plot the accuracy (optional)
with torch.no_grad():
    predictions_test = model(X_test_tensor)
    mse_loss = torch.nn.MSELoss()(predictions_test, y_test_tensor).item()

plt.figure(figsize=(10, 6))
plt.plot(predictions_test.numpy(), '.', label="Predicted Prices", markersize=3)
plt.plot(y_test_tensor.numpy(), '.', label="True Prices", markersize=3)
plt.xlabel("Data Points")
plt.ylabel("Price")
plt.legend(loc='upper left')
plt.title(f"Model Predictions vs True Prices (MSE Loss: {mse_loss:.3f})")
plt.grid(True)
plt.show()
