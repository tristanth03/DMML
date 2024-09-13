# Author: Tristan Thordarson
# Date: 12.09.2024
# Project: 03
# Acknowledgements: https://pytorch.org/docs/stable/generated/torch.Tensor.item.html


import torch
import matplotlib.pyplot as plt

from tools import load_regression_iris
from tools import split_train_test
from scipy.stats import multivariate_normal


def mvn_basis(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float
) -> torch.Tensor:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    M,D = mu.shape
    N = features.shape[0]
    sigma_k = var*torch.eye(M)
    basis_func = torch.zeros(N,M)
    for i in range(M):
        phi_k = multivariate_normal(mu[i,:],sigma_k[i,i])
        basis_func[:,i] = torch.asarray(phi_k.pdf(features))

    return basis_func


def _plot_mvn():
    '''Assuming same example'''


    X, t = load_regression_iris()
    N, D = X.shape
    M, var = 10, 10
    mu = torch.zeros((M, D))
    plt.figure(figsize=(10,5))
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, var)
    for i in range(fi.shape[1]):
        plt.plot(fi[:,i])
    plt.xlabel("N")
    plt.ylabel("Numerical value")
    plt.title("Output of M-many basis functions")
    plt.show()

def max_likelihood_linreg(
    fi: torch.Tensor,
    targets: torch.Tensor,
    lamda: float
) -> torch.Tensor:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''

    # the solution for the weigths is found on p.137, eq.4.27 (Bishop)
    # (lambda*identity+phi^T*phi)^-1*phi^T*t
    N,M = fi.shape 
    inner = torch.inverse(lamda*torch.eye(M)+torch.matmul(fi.t(),fi))
    outer = torch.matmul(fi.t(),targets)
    w = torch.matmul(inner,outer)
    return w


def linear_model(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float,
    w: torch.Tensor
) -> torch.Tensor:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''

    # y = phi*w , p.136, ch 4.1.4, (Bishop)
    
    fi = mvn_basis(features,mu,var)
    y_hat = torch.matmul(fi,w)

    return y_hat

def plot_loss():
    '''Plotting the MSE loss for train and test datasets with differents lambda values'''
    torch.manual_seed(1234) # Some random seed for the data shuffling part in split_train_test
    X, t = load_regression_iris()
    N, D = X.shape
    M, var = 10, 10
    mu = torch.zeros((M, D))
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)
 
    (train_features, train_targets), (test_features, test_targets) = split_train_test(X,t,0.9) # I fixed the tools.py torch.random.permutation->torch.randperm

    fi_train = mvn_basis(train_features,mu,var)
    fi_test = mvn_basis(test_features,mu,var)

    train_mse = []
    test_mse = []
    lambda_values = torch.logspace(-12, 20, 20)
    mse_loss_train = torch.nn.MSELoss()
    mse_loss_test = torch.nn.MSELoss()
    for lamda in lambda_values:
        wml_train = max_likelihood_linreg(fi_train, train_targets, lamda)
        wml_test = max_likelihood_linreg(fi_test, test_targets, lamda)
        pred_train = linear_model(train_features,mu,var,wml_train)
        pred_test = linear_model(test_features,mu,var,wml_test)
        
        train_mse.append(mse_loss_train(pred_train,train_targets).item())
        test_mse.append(mse_loss_test(pred_test,test_targets).item())
        

    plt.plot(lambda_values,train_mse,'--o',label='Train')
    plt.plot(lambda_values,test_mse,'--o',label='Test')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel(fr'$\lambda$')
    plt.ylabel(r"$\mathcal{L}_{MSE}$")
    plt.title(fr'Comparison of the MSE loss for different $\lambda$ values')
    plt.show()

def plot_pred_vs_target():
    X, t = load_regression_iris()
    N, D = X.shape
    M, var = 10, 10
    mu = torch.zeros((M, D))
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, var) 
    lamda = 0.001
    wml = max_likelihood_linreg(fi,t,lamda)
    pred = linear_model(X,mu,var,wml)
    mse_loss = torch.nn.MSELoss()
    mse = mse_loss(pred,t).item()
    plt.plot(t,'.',label='Target values')   
    plt.plot(pred,'.',label='Predicted values')
    plt.title(fr'Prediction and target values, $\lambda$ = {lamda}; MSE = {round(mse,2)}')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    

    _plot_mvn()

    plot_pred_vs_target()
    
    plot_loss()
    