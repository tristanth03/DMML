import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Assuming X and T are your training data and targets (tensors)
X = torch.randn((100, 10))  # Example data with 100 samples, 10 features
T = torch.randint(0, 2, (100,))  # Example binary targets (0 or 1)

# Define the DenseNTK layer class
class DenseNTK(nn.Module):
    """
    Class for the layers of a FFNN
    I) Follows the torch.nn module (structure and syntax)
    II
    a) Initialize weights by ATS_NTK method (i.e. scaled by 1/sqrt(D))
    b) Correct the forward pass
    """

    def __init__(
        self,
        in_features,  # input dim of layer
        out_features,  # output dim of layer
        activation=torch.nn.Identity(),  # no activation by default
        bias=True,  # bias added by default
        initialization_=None):
        
        super(DenseNTK, self).__init__()
        # Initialize weights using ATS_NTK initialization
        if initialization_:
            self.weight = nn.Parameter(initialization_(in_features, out_features))
        else:
            # Default initialization with N(0, 1)/sqrt(in)
            self.weight = nn.Parameter(torch.randn(out_features, in_features) / (in_features ** 0.5))

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

        self.activation = activation 
        
    def forward(self, x):
        
        output = F.linear(x,self.weight, self.bias)
        output = self.activation(output)

        return output

# Define the Feedforward Neural Network class
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FFNN, self).__init__()
        self.fc1 = DenseNTK(input_size, hidden_size1, activation=nn.ReLU())
        self.fc2 = DenseNTK(hidden_size1, hidden_size2, activation=nn.ReLU())
        self.fc3 = DenseNTK(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Define model parameters
input_size = X.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = 1  # Assuming binary classification

# Instantiate the model, define loss function and optimizer
model = FFNN(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.MSELoss()  # For binary classification
optimizer = optim.SGD(model.parameters(), lr=0.001)


print(model(X))
# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs.squeeze(), T.float())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss for every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Training complete!")

plt.plot(model(X).detach().numpy())
plt.show()