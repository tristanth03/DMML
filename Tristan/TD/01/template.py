# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import matplotlib.pyplot as plt
import numpy as np

from tools import scatter_2d_data, bar_per_axis


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    std: np.float64
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    # Þarf að laga input
    cov = std**2*np.identity(k)
    p = np.random.multivariate_normal(mean,cov,n)

    return p

def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    # mu_2 = np.mean(x,0)
    # n_2 = x.shape[0]
    # new_mu = (mu*(n-n_2)+mu_2*n_2)/n
    mu_ml = mu+1/n*(x-mu)

    return mu_ml

def _plot_sequence_estimate():
    '''
    
    '''
    data = gen_data(100,2,np.array([0,0]),3) # Set this as the data
    estimates = [np.array([0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i],data[i],i+1))
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')  
    plt.legend(loc='upper center')
    plt.xlabel("N")
    plt.ylabel(r"$\mu_{ML}$")
    plt.show()


def _square_error(y, y_hat):
    """
    """
    se = []
    for i in range(len(y)):
        se.append(np.power(y[i]-y_hat[i],2))
    mse = np.mean(se,0)
    return mse


def _plot_mean_square_error():
    data = gen_data(100,2,np.array([0,0]),3) # Set this as the data
    estimates = [np.array([0, 0])]
    means = [data[0,:]]
    mse = [_square_error(means,estimates)]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i],data[i],i+1))
        means.append(np.mean(data[i],1))
        mse.append(_square_error(means[i],estimates[i]))
    plt.plot(mse)
    plt.show()

    


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: np.float64
) -> np.ndarray:
    # remove this if you don't go for the independent section
    pass


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    pass


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    np.random.seed(1234)
    # p = gen_data(2,3,np.array([0,1,-1]),1.3)
    # print(p)
    # X = gen_data(300,2,np.array([-1,2]),np.sqrt(4))
    # mean = np.mean(X,0)
    # new_x = gen_data(1, 2, np.array([0, 0]), 1)
    # p = update_sequence_mean(mean, new_x, X.shape[0]+1)
    # print(p)

    _plot_mean_square_error()

    # scatter_2d_data(X)
    # bar_per_axis(X)
    