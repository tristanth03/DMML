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
    mu_2 = np.mean(x,0)
    n_2 = x.shape[0]
    new_mu = (mu*(n-n_2)+mu_2*n_2)/n
    return new_mu

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
    # Section 2
    # data = gen_data(300, 2, np.array([-1,2]), np.sqrt(4))
    # scatter_2d_data(data)
    # bar_per_axis(data) 

    # Section 3
    np.random.seed(1234)
    X = gen_data(300, 2, np.array([-1,2]), np.sqrt(4))
    mean = np.mean(X, 0)
    new_x = gen_data(1, 2, np.array([0, 0]), 1)
    print(update_sequence_mean(mean, new_x, X.shape[0]+1))
    

