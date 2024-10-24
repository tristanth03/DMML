include("DenseNTK.jl")
include("FastNTKMethods.jl")
using Flux,ProgressMeter,LinearAlgebra,Zygote,Random
Random.seed!(1234)


# Function to generate a Gaussian wave packet with added noise epsilon ~ N(0,1)
function noisy_gaussian_wave_function(x;k=1, sigma=1, A=1, alpha=0.2)
    epsilon = randn(size(x))  # Generate noise (epsilon) from N(0,1)
    
    return A * exp.(-(x.^2) / (2 * sigma^2)) .* cos.(k * x .+ epsilon .* alpha)
end

# Parameters
A = 1
sigma = 0.2
k = 20
alpha = 0.1

# Step 2: Define the range of x values
x_vals = range(-5, stop=5, length=500)

# Step 3: Generate the wave function with noise
psi_vals = noisy_gaussian_wave_function(x_vals, k=k, sigma=sigma, A=A, alpha=alpha)


x = x_vals


activation = relu; N1=10_000; InDim = 1; OutDim = 1
model = Chain(DenseNTK(InDim=>N1,activation),DenseNTK(N1=>OutDim))|>f64

K = kernel(model,x)
eigen(K).values
