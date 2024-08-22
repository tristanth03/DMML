import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: np.float64, mu: np.float64) -> np.ndarray:
    # Part 1.1
    a = 1/np.sqrt(2*np.pi*sigma**2)
    b = -np.power((x-mu),2)/(2*sigma**2)
    return a*np.exp(b)
    # print(normal(0, 1, 0))
    # normal(3, 1, 5)
    # normal(np.array([-1,0,1]), 1, 0)

def plot_normal(sigma: np.float64, mu:np.float64, x_start: np.float64, x_end: np.float64):
    # Part 1.2
    x_range = np.linspace(x_start,x_end,500)
    plt.plot(x_range, normal(x_range, sigma, mu), label = f'mu = {mu}:sigma = {sigma}')


def _plot_three_normals():
    plot_normal(0.5, 0, -2, 2)
    plot_normal(0.25, 1, -2, 2)
    plot_normal(1, 1.5, -2, 2)
    plt.legend(loc = 'upper left')
    plt.show()

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    mus = np.array(mus)
    sigmas = np.array(sigmas)
    arr = []
    for num in x:
        a = weights/np.sqrt(2*np.pi*np.power(sigmas,2))
        b = -(np.power(num-mus,2))/(2*np.power(sigmas,2))
        new = np.dot(a, np.exp(b))
        arr = np.append(arr, new)
    return np.array(arr)
#normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3])

def _compare_components_and_mixture():
    # Part 2.2  
    mus = [0, -0.5, 1.5]
    sigmas = [0.5, 1.5, 0.25]
    weights = [1/3, 1/3, 1/3]
    arr = normal_mixture(np.linspace(-5, 5, 500), sigmas, mus, weights)
    plot_normal(0.5, 0, -5, 5)
    plot_normal(1.5, -0.5, -5, 5)
    plot_normal(0.25, 1.5, -5, 5)
    plt.plot(np.linspace(-5,5,500), arr, label = f'mix')
    plt.legend(loc = 'upper left')
    plt.show()
_compare_components_and_mixture()


def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    pass
    # Part 3.1

def _plot_mixture_and_samples():
    pass
    # Part 3.2

if __name__ == '__main__':
    pass
    # select your function to test here and do `python3 template.py`


    
