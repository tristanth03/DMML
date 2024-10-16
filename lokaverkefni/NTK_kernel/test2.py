import torch 
import torch.nn as nn
import torch.linalg
import torch.nn.functional as F
import torch.optim as optim

"""
Data needs to be on the following form:

X = torch( [ [x1], [x2], ... [xN] ])
with x_n being of dim D

"""

class DenseNTK(nn.Module):
    def __init__(self, in_features, out_features, activation=torch.nn.Identity(), bias=True, init_fn=None):
        super(DenseNTK, self).__init__()
        # Initialize weights using custom initialization function or default
        if init_fn:
            self.weight = nn.Parameter(init_fn(out_features, in_features))
        else:
            # Default initialization as per Flux.randn32 with N(0, 1)/sqrt(in)
            self.weight = nn.Parameter(torch.randn(out_features, in_features) / (in_features ** 0.5))

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

        # Activation function
        self.activation = activation

    def forward(self, x):
        # Weight normalization by sqrt(number of input features) applied at forward time
        weight_normalized = self.weight / (self.weight.size(1) ** 0.5)  # Normalize during the forward pass
        output = F.linear(x, weight_normalized, self.bias)
        output = self.activation(output)
        return output



def remove_last_bias(model, jacobian):
    """Removes last bias of model in jacobian, similar to the Julia function"""
    # Get the size of the last bias parameter
    params = list(model.parameters())
    last_bias_size = params[-1].numel()  # Get the number of elements in the last bias parameter
    
    # Remove the columns corresponding to the last bias from the Jacobian
    jacobian = jacobian[:, :-last_bias_size]

    return jacobian

def jacobian_torch(model, x, show_progress=False):
    if show_progress:
        import time
        start_time = time.time()
        print("\nComputing Jacobian with Torch...")

    # Ensure the model is in evaluation mode
    model.eval()

    # Forward pass through the model
    y = model(x)

    # Flatten output to make the computation easier if needed
    y_flat = y.view(-1)

    # Store the gradients for each parameter
    jacobian_list = []

    # Iterate over each element of the output
    for i in range(y_flat.size(0)):
        # Zero the gradients
        model.zero_grad()

        # Compute the gradient of the i-th output element with respect to all model parameters
        y_flat[i].backward(retain_graph=True)

        # Extract the gradient for each parameter and flatten it
        grads = []
        for param in model.parameters():
            grads.append(param.grad.view(-1))
        
        # Concatenate all gradients for the current output element
        jacobian_list.append(torch.cat(grads))

    # Stack all Jacobians together to form a matrix
    jacobian_matrix = torch.stack(jacobian_list).double()

    # Remove the contribution of the last bias parameter from the Jacobian
    jacobian_matrix = remove_last_bias(model, jacobian_matrix)

    if show_progress:
        print(f"\nTime taken: {time.time() - start_time} seconds")

   
    return jacobian_matrix













if __name__ == "__main__":
    torch.manual_seed(1234)
    x1 = [1,-1]
    x2 = [2,0]
    x3 = [3,-3]
    x4 = [4,-4]

    X = torch.tensor([x1,x2,x3,x4],dtype=torch.float64)
    T = torch.exp(X)

    N,D = X.shape
    K = T.shape[1]
    
  
    M1 = 10_000
    model = nn.Sequential(
        DenseNTK(D, M1, activation=nn.ReLU(), bias=True),
        DenseNTK(M1, K, activation=nn.Identity(), bias=True)
        ).double() # double for precision measure

    Df = jacobian_torch(model,X,False)
    kernel = torch.matmul(Df,Df.T)

    
    eig = torch.linalg.eigvals(kernel).real.to(dtype=torch.float64) # eigenvalues are real
    eta = 1/eig[1].item()

    
    
        # Number of epochs
    num_epochs = 10000

    # Define loss function (Mean Squared Error)
    criterion = nn.MSELoss()

    # Define the optimizer (Stochastic Gradient Descent, equivalent to normal GD)
    optimizer = optim.SGD(model.parameters(), lr=eta)  # Learning rate for normal GD

    # Training loop with Full Batch Gradient Descent
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: compute model output for the entire dataset
        predictions = model(X)

        # Compute loss between predictions and true values (on the entire dataset)
        loss = criterion(predictions, T)

        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Print progress occasionally
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.16f}")

    # Evaluation after training (optional)
    model.eval()
    with torch.no_grad():
        final_predictions = model(X)
        final_loss = criterion(final_predictions, T)
        print(f"Final Loss after Training: {final_loss.item():.f}")


    # print(list(model.named_parameters()))

