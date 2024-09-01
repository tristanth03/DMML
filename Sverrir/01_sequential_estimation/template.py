# Author: Sverrir Hákonarson
# Date: 26.01.2024
# Project: 
# Acknowledgements: 
#

import matplotlib.pyplot as plt
import numpy as np
#import tools
from tools import scatter_2d_data, bar_per_axis


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    sigma: np.float64
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    cov = np.identity(k)*sigma**2
    return np.random.multivariate_normal(mean, cov, n)


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return mu + (1/(n))*(x-mu)

def _plot_sequence_estimate():
    data = gen_data(100, 2, np.array([0, 0]), 3)
    estimates = [np.array([0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i], i+1))
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')

    plt.legend(loc='upper center')
    plt.show()
    


def _square_error(y, y_hat):
    se = [] 
    for i in range(len(y)):
        se.append(np.power(y[i]-y_hat[i],2)) 
    mse = np.mean(se, axis=0)
    return mse


def _plot_mean_square_error():
    data = gen_data(100, 2, np.array([0, 0]), 3)
    estimate_mean = [np.array([0, 0])]
    actual_mean = []
    mses = []
    for i in range(data.shape[0]):
        estimate_mean.append(update_sequence_mean(estimate_mean[i], data[i], i+1))
        actual_mean.append(np.mean(data[:i+1],0))
        mses.append(_square_error(actual_mean[i], estimate_mean[i]))

    plt.plot([mse for mse in mses])
    plt.show()

    #errors = _square_error()


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
    # Section 2
    # data = gen_data(300, 2, np.array([-1,2]), np.sqrt(4))
    # scatter_2d_data(data)
    # bar_per_axis(data) 

    # Section 3
    np.random.seed(1234)
    # X = gen_data(300, 2, np.array([-1,2]), np.sqrt(4))
    # mean = np.mean(X, 0)
    # new_x = gen_data(1, 2, np.array([0, 0]), 1)
    # print(update_sequence_mean(mean, new_x, X.shape[0]+1))
    #_plot_sequence_estimate()
    #_plot_sequence_estimate()
    #_plot_mean_square_error()
    

