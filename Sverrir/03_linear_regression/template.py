# Author: 
# Date:
# Project: 
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
    
    N, D = features.shape
    M, _ = mu.shape 
    fi = torch.zeros(N, M)
    cov_matrix = var * torch.eye(D)
    for j in range(M):
         mvn = multivariate_normal(mu[j], cov_matrix)
         fi[:, j] = torch.tensor(mvn.pdf(features))
    return fi
    


def _plot_mvn():
    X, t = load_regression_iris()
    N, D = X.shape
    M, var = 10, 10
    mu = torch.zeros((M, D))
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, var) 
    for x in range(M): 
        plt.plot(fi[:,x])
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
    # I am using formula 4.27 on page 118 in the book.
    gram_mat = torch.matmul(fi.T, fi)
    lamda_mat = lamda * torch.eye(gram_mat.shape[0])
    inner_mat = gram_mat+lamda_mat
    outer_mat = torch.matmul(fi.T, targets)
    weights = torch.matmul(torch.inverse(inner_mat),outer_mat)
    return weights
    

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
    fi = mvn_basis(features, mu, var)
    y_hat = torch.matmul(fi,w)
    return y_hat


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    X, t = load_regression_iris()
    N,D = X.shape
    M, var = 10, 10
    mu = torch.zeros((M, D))
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, var)
    #print(fi)
    _plot_mvn()
    wml = max_likelihood_linreg(fi, t, 0.001)
    prediction = linear_model(X, mu, var, wml)
    mse = torch.mean((t- prediction)**2)
    
    plt.plot(t,'.',label='Target')
    plt.plot(prediction,'.',label='Predicted')
    plt.legend()
   
    plt.show()

