from kernel_layer import DenseNTK
from kernel_layer import KernelMatrix

import torch 
import torch.nn as nn
import torch.linalg
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt


def train_data_I():
    x1 = [1,-1]
    x2 = [2,0]
    x3 = [3,-3]
    x4 = [4,-4]

    X = torch.tensor([x1,x2,x3,x4],dtype=torch.float64)
    T = torch.exp(X) # Targets

    N,D = X.shape
    K = T.shape[1]
    
    return X,N,D,T,K

def train_data_II():
      
    N = 200  # Number of points
    D = 8 # Number of dimensions

    points = [[random.uniform(-1, 1) for _ in range(D)] for _ in range(N)]
    X = torch.tensor(points,dtype=torch.float32)
    T = torch.exp(X) # Targets
    K = 1
   
    return X,N,D,T,K


def ffnn_model(D,K):
    M = [128,64,32,16] # number of nodes per hidden layer


    model = nn.Sequential(
        DenseNTK(D, M[0], activation=nn.ReLU(), bias=True),
        DenseNTK(M[0], M[1], activation=nn.ReLU(), bias=True),
        DenseNTK(M[1], M[2], activation=nn.ReLU(), bias=True),
        DenseNTK(M[2], M[3], activation=nn.ReLU(), bias=True),
        DenseNTK(M[3], K, activation=nn.Identity(), bias=True)
    ) # Double precision for extra precision

    return model 
def train_model(
    x,targets,
    model,
    learning_rate,num_epochs,
    opt,problem_type):

    if problem_type == "Regression":
        criterion = nn.MSELoss()
    if opt == "VanillaGD":
        # node this is stochastic gradient descent with the entire batch
        # no momentum nor randomness thus the same as Vanilla GD
        optimizer = optim.SGD(model.parameters(),lr=learning_rate) 
    
    L = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y = model(x) # entire dataset
        loss = criterion(y,targets)
        loss.backward()
        # Parameter update (per epoch whole dataset)
        optimizer.step() 

        L.append(loss)  # makes training a lot slower but ...
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.16f}")
    return L,model(x)


def plot_loss(L):
    plt.figure(figsize=(10, 6))
    plt.plot(L,'.',markersize=2)
    plt.title("Loss", fontsize=16)
    plt.xlabel(r"Epoch ($\tau$)", fontsize=16)
    plt.ylabel(r"$\mathcal{L}$", fontsize=16)
    plt.xticks(fontsize=14)  # Adjust the font size for x-axis ticks
    plt.yticks(fontsize=14)  # Adjust the font size for y-axis ticks
    plt.yscale('log')

    plt.show()

if __name__ == "__main__":
    torch.manual_seed(1234)
    X,N,D,T,K = train_data_II()

    NTK_ = KernelMatrix()
    model = ffnn_model(D,K)
    Df = NTK_.jacobian_torch(model, X, show_progress=True)

    kernel = torch.matmul(Df,Df.T)
    print(kernel)
    # eigenvalues are real (for the kernel matrix)
    eig = torch.linalg.eigvals(kernel).real.to(dtype=torch.float64) 
    eta = 0.0001


    num_epochs = 1000
    L,y= train_model(X,T,model,eta,num_epochs,opt="VanillaGD",problem_type="Regression")
    Lo = [t.item() for t in L]
    y_hat = [t.item() for t in y]

    plot_loss(Lo)
    plt.plot(y_hat)
    plt.plot(T[:,1],'.')
    plt.show()


