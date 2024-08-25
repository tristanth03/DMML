import numpy as np

def sample_gaussian_mixture(sigmas, mus, weights, n_samples):
    # Step 1: Determine the number of samples from each Gaussian component
    component_counts = np.random.multinomial(n_samples, weights)
    
    # Step 2: Sample from each Gaussian component
    samples = []
    for sigma, mu, count in zip(sigmas, mus, component_counts):
        samples.append(np.random.normal(mu, sigma, count))
    

    return samples

# Test the function with the provided example
np.random.seed(0)
print(sample_gaussian_mixture([0.1, 1], [-1, 1], [0.9, 0.1], 3))
