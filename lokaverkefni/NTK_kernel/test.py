
"""Ath gögn þurfa vera á forminu
    torch.tensor([ [x1],...,[xN] ])    
    þ.s. hvert x_n er D langt
"""


import torch
import torch.nn as nn
import torch.autograd
import numpy as np

# Define a function that checks the dimensions of input data
def check_dim(x):
    """This function checks the appropriate dimensions of input data"""
    if x.ndim == 2:  # Matrix
        return x.shape[0]  # Number of columns (width) of the matrix
    elif x.ndim == 1:  # Vector
        return 1  # Returns 1 if it's a vector
    else:
        raise TypeError(f"Input data type: {type(x)} is neither a matrix or column vector")



# test case
x = torch.tensor([[1, -1], [2, 0], [3, -3], [4, -4]], dtype=torch.float64)
t = torch.exp(x)


D = x.shape[1]
M1 = 10_000
K = t.shape[1]

print(t)
model = nn.Sequential(
        nn.Linear(D,M1),
        nn.ReLU(),
        nn.Linear(M1,K)
).double() # float64 precision


print(model(x))

# def remove_last_bias(model,Jacobian):
#     """
#     """







