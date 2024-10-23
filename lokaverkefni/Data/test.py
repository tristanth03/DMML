import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Check the tensor shapes
print(X_train_tensor.shape)  # e.g., torch.Size([16512, 8])
print(y_train_tensor.shape)  # e.g., torch.Size([16512, 1])
