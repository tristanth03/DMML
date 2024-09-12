# Author: Tristan Thordarson
# Date: 12.09.2024
# Project: 03
# Acknowledgements: 
#

import torch
import matplotlib.pyplot as plt

from tools import load_regression_iris
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


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    X, t = load_regression_iris()
    N, D = X.shape
    M, var = 10, 10
    mu = torch.zeros((M, D))
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, var) 

    # _plot_mvn()
    
    mse = []
    lambda_values = torch.logspace(-10, 20, 20)
    for lamda in lambda_values:
        wml = max_likelihood_linreg(fi, t, lamda)
        pred = linear_model(X,mu,var,wml)
        mse_loss = torch.nn.MSELoss()
        mse.append(mse_loss(pred,t))
        # plt.plot(pred,'*')

    plt.plot(lambda_values,mse,'-o')
    plt.xscale('log')
    plt.show()
    # plt.plot(t,'.')   
    # plt.show()

    # bias variance tradeoff pælingar, W vs \lambda og ....
    # leiða út jöfnu (lausn!!!)

    