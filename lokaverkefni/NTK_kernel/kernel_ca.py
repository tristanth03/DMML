import torch
import torch.nn as nn
import torch.autograd
import numpy as np

# Define a function that checks the dimensions of input data
def check_dim(x):
    """This function checks the appropriate dimensions of input data"""
    if x.ndim == 2:  # Matrix
        return x.shape[1]  # Number of columns (width) of the matrix
    elif x.ndim == 1:  # Vector
        return 1  # Returns 1 if it's a vector
    else:
        raise TypeError(f"Input data type: {type(x)} is neither a matrix or column vector")

# Define a function that removes the last bias in the Jacobian matrix
def remove_last_bias(model, jacobian):
    """Removes last bias of model in Jacobian, because of 'frozen parameter'"""
    last_bias_params = list(model.parameters())[-1].numel()
    return jacobian[:, :-last_bias_params]

# Convert model to lambda function for easier differentiation
def model_to_lambda(model):
    """Convert a PyTorch model to a lambda function for easier differentiation."""
    def model_fn(x, params):
        # Set model parameters to provided values
        idx = 0
        for param in model.parameters():
            num_params = param.numel()
            param.data = params[idx:idx + num_params].view(param.shape).data
            idx += num_params
        return model(x)
    return model_fn

# Compute Jacobian using torch.autograd
def jacobian_autograd(model, x, show_progress):
    N = check_dim(x)
    params = torch.cat([p.view(-1) for p in model.parameters()])
    l = len(model(x[:, 0].view(1, -1)))
    k = len(params)
    model_fn = model_to_lambda(model)

    D = torch.zeros((N * l, k))
    for i in range(N):
        if show_progress:
            print(f"Processing sample {i + 1}/{N}")
        # Forward pass for input `x[:, i]`
        y = model_fn(x[:, i].view(1, -1), params)
        # Compute gradients w.r.t parameters
        grads = torch.autograd.grad(outputs=y, inputs=params, retain_graph=True, allow_unused=True)
        grads = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p).view(-1) for g, p in zip(grads, model.parameters())])
        D[i * l:(i + 1) * l, :] = grads

    D = remove_last_bias(model, D)
    return D

# Kernel function to compute the final result
def kernel(model, x, show_progress=False):
    jacobian = jacobian_autograd(model, x, show_progress)

    if show_progress:
        print("Kernel computation")
    K = jacobian @ jacobian.T
    return K