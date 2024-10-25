import psutil
import torch
import torch
import matplotlib.pyplot as plt
from kernel_layer import KernelMatrix
from kernel_layer import DenseNTK
import torch.nn as nn
import torch.optim as optim
import numpy as np


torch.manual_seed(1234)

num_threads = psutil.cpu_count(logical=True)
torch.set_num_threads(16)
current_threads = torch.get_num_threads()



def noisy_gaussian_wave_function(x, k=1, sigma=1, A=1,alpha=0.2):
    """Generates a Gaussian wave packet with added noise epsilon ~ N(0,1)."""


    epsilon = torch.normal(mean=0.0, std=1.0, size=x.shape)  # Generate noise (epsilon) from N(0,1)
    
    return A * torch.exp(-(x**2) / (2 * sigma**2)) * torch.cos(k * x + epsilon *alpha)

def ffnn_model(D, K, M,activation=nn.ReLU()):
    # Initialize the layers list
    layers = []


    # Add the first layer with input size D and first hidden layer size M[0]
    layers.append(DenseNTK(D, M[0], activation=activation, bias=True))

    # Add the remaining hidden layers dynamically
    for i in range(1, len(M)):
        layers.append(DenseNTK(M[i-1], M[i], activation=activation, bias=True))

    # Add the final output layer with the last hidden layer size and output size K
    layers.append(DenseNTK(M[-1], K, activation=nn.Identity(), bias=True))

    # Create the model using nn.Sequential
    model = nn.Sequential(*layers)

    return model


def test_case(plot=False):
        # Parameters
    A = 1
    sigma = 0.3
    k = 20
    alpha = 0.1

    # Step 2: Define the range of x values
    x_vals = torch.linspace(-1, 1, 1000)

    # Step 3: Generate the wave function with noise
    psi_vals = noisy_gaussian_wave_function(x_vals, k=k, sigma=sigma, A=A,alpha=alpha)

    # Step 4: Convert to numpy for plotting (since matplotlib works with numpy arrays)
    x_vals_np = x_vals.numpy()
    psi_vals_np = psi_vals.numpy()

    N = len(x_vals)
    # Step 5: Plot the noisy wave function

    if plot == True:
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals_np, psi_vals_np, '.', label='Noisy Wave Function')
        plt.title("Noisy Gaussian Wave Packet",fontsize=16)
        plt.xlabel("x",fontsize=16)
        plt.ylabel(fr"$\Psi$(x)",fontsize=16)
        plt.xticks(fontsize=14)  # Adjust the font size for x-axis ticks
        plt.yticks(fontsize=14)  # Adjust the font size for y-axis ticks

        plt.grid(True)

        # Step 6: Add a small box with parameters
        param_text = f"$A = {A}$\n$\\sigma = {sigma}$\n$k = {k}$\n$\\alpha = {alpha}$"
        param_text2 = f"N={N}"
        plt.text(0.9, 0.95, param_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="lightgray"))
        plt.text(0.05, 0.95, param_text2, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="lightgray"))
        plt.show()


        return  psi_vals , x_vals
    else:

        return  psi_vals , x_vals
    

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


def plot_eig(eig):
    plt.figure(figsize=(10, 6))
    plt.plot(eig, '.',markersize=2, label='Eiganvalues')
    plt.title(f"Eigenvalues",fontsize=16)
    plt.xlabel("i",fontsize=16)
    plt.ylabel(fr"$\lambda_i$",fontsize=16)
    plt.xticks(fontsize=14)  # Adjust the font size for x-axis ticks
    plt.yticks(fontsize=14)  # Adjust the font size for y-axis ticks


    plt.yscale('log')

   
    plt.show()


 

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
    

def plot_fit(x,y_hat,t):
    
    plt.figure(figsize=(10, 6))
    plt.plot(x,t,'.',markersize=3,label="Data")
    plt.plot(x,y_hat,linewidth=0.2,label="Fit")
    plt.legend()
    N = len(x_vals)
    plt.title(f"Fit vs Real (fixed learning rate: 0.0001)",fontsize=16)
    plt.xlabel("x",fontsize=16)
    plt.ylabel(fr"$y(x)$",fontsize=16)
    plt.xticks(fontsize=14)  # Adjust the font size for x-axis ticks
    plt.yticks(fontsize=14)  # Adjust the font size for y-axis ticks
    plt.grid(True)
    param_text2 = f"N={N}"

    plt.text(0.05, 0.95, param_text2, transform=plt.gca().transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="lightgray"))
    
    plt.legend(fontsize=12,frameon=True,loc='upper right')
    
    plt.show()
    
if __name__ == "__main__":

    psi_vals , x_vals = test_case(False)
    D = x_vals.shape[0]
    K = psi_vals.shape[0]
    M = [256,64,8]

    model = ffnn_model(D,K,M,activation=nn.LeakyReLU)

    NTK_ = KernelMatrix()
    model = ffnn_model(D,K,M)
    Df = NTK_.jacobian_torch(model, x_vals, show_progress=True)
    kernel = torch.mm(Df,Df.T)

    eig = torch.linalg.eigvals(kernel).real.to(dtype=torch.float32)
    eig = torch.abs(eig)
    eig = torch.sort(eig)[0]


    eta = 0.001

    
    L,y_ = train_model(x_vals,psi_vals,model,eta,10000,opt='VanillaGD',problem_type='Regression')
    Lo = [t.item() for t in L]
    y_hat = [t.item() for t in y_]
    eigens = [t.item() for t in eig]

 
    
    # plot_fit(x_vals,y_hat,psi_vals)


    # plot_loss(Lo)


    # plot_eig(eig)


    

    import json

    with open('eigen_wave_lr_0.0001.json','w') as file:
        json.dump(eigens,file)

    with open('loss_wave_lr_0.0001.json','w') as file:
        json.dump(Lo,file)

    with open('fit_wave_lr_0.0001.json','w') as file:
        json.dump(y_hat,file)





 