import numpy as np
# import matplotlib.pyplot as plt


# def normal(x: np.ndarray, sigma: np.float64, mu: np.float64) -> np.ndarray:
#     # Part 1.1
#     a = 1/(np.sqrt(2*np.pi*(sigma**2)))
#     b = np.exp(-np.power((x-mu),2)/(2*(sigma**2)))
#     return a*b


# def plot_normal(sigma: np.float64, mu:np.float64, x_start: np.float64, x_end: np.float64):
#     # Part 1.2
#     n = 500
#     x = np.linspace(x_start,x_end,n)
#     y = normal(x,sigma,mu)
#     p = plt.plot(x,y,label=fr'$\sigma$ = {sigma}, $\mu$ = {mu}')
#     return p

    
# def _plot_three_normals():
#     # Part 1.2
#     plot_normal(0.5,0,-5,5)
#     plot_normal(0.25,1,-5,5)
#     plot_normal(1,1.5,-5,5)
#     plt.legend(loc='upper left')
    
    
    

# def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
#     # Part 2.1
#     p = 0
#     for i in range(0,len(sigmas)):
#         a = weights[i]/np.sqrt(2*np.pi*(sigmas[i]**2))
#         b = np.exp(-np.power((x-mus[i]),2)/(2*(sigmas[i]**2)))
#         p += a*b
#     return p

# def _compare_components_and_mixture():
#     # Part 2.2
#     _plot_three_normals()
#     x_ = np.linspace(-5,5,500)
#     p_multi = normal_mixture(x_,[0.5,1.5,0.25],[0,-0.5,1.5],[1/3,1/3,1/3])
#     plt.plot(x_,p_multi,label="mix")
#     plt.legend(loc='upper left')
#     plt.show()

# def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
#     # Part 3.1
#     pass



# # def _plot_mixture_and_samples():
# #     # Part 3.2

# if __name__ == '__main__':
#     np.random.multinomial(20, [1/6.]*6)

    