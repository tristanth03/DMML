import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: np.float64, mu: np.float64) -> np.ndarray:
    # Part 1.1
    a = 1/(np.sqrt(2*np.pi*(sigma**2)))
    b = np.exp(-np.power((x-mu),2)/(2*(sigma**2)))
    return a*b


def plot_normal(sigma: np.float64, mu:np.float64, x_start: np.float64, x_end: np.float64):
    # Part 1.2
    n = 500
    x = np.linspace(x_start,x_end,n)
    y = normal(x,sigma,mu)
    p = plt.plot(x,y)
    return p

    
# def _plot_three_normals():
#     # Part 1.2

# def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
#     # Part 2.1

# def _compare_components_and_mixture():
#     # Part 2.2

# def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
#     # Part 3.1

# def _plot_mixture_and_samples():
#     # Part 3.2

# if __name__ == '__main__':
#     # select your function to test here and do `python3 template.py