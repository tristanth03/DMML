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
    mu_ml = np.zeros((n+1,1))
    mu_ml[0] = mu
    for i in range(n):
        mu_ml[i] = mu_ml[i-1]+1/n*(x[i]-mu_ml[i-1])

    return mu_ml

    


def _plot_sequence_estimate():
    data = None # Set this as the data
    estimates = [np.array([0, 0])]
    for i in range(data.shape[0]):
        """
            your code here
        """
    plt.plot([e[0] for e in estimates], label='First dimension')
    """
        your code here
    """
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    pass


def _plot_mean_square_error():
    pass


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
    X = gen_data(10_000_000,2,np.array([-1,2]),np.sqrt(4))
    mean = np.mean(X)
    new_x = gen_data(1, 2, np.array([0, 0]), 1)
    new_x = gen_data(1, 2, np.array([0, 0]), 1)
    p = update_sequence_mean(mean, new_x, X.shape[0]+1)
    print(p)

    # scatter_2d_data(X)
    # bar_per_axis(X)
    