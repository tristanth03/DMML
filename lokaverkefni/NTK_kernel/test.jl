include("DenseNTK.jl")
include("FastNTKMethods.jl")
using Flux,ProgressMeter,LinearAlgebra


x = zeros(2,4)
x[1,:] = [1,2,3,4]
x[2,:] = [-1,0,-3,-4] # input test




y = zeros(2,4)
y[1,:] = exp.(x[1,:])
y[2,:] = exp.(x[2,:])

activation = relu; N1=10_000; InDim = size(x)[1]; OutDim = 2
model = Chain(DenseNTK(InDim=>N1,activation),DenseNTK(N1=>OutDim))|>f64

display(model(x))
display("xxx")
println(InDim)
K = kernel(model,x,true)
display(K)

