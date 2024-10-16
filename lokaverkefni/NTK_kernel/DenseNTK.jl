using Flux, ReverseDiff, Tracker

struct DenseNTK
    weight  # weight: initialized to N(0,1)/sqrt(in)
    bias  # bias: initialized to N(0,1)
    σ  # activation function
end

function DenseNTK((in, out)::Pair, σ=identity; bias=true, init=Flux.randn32)
    weight = init(out, in)
    bias = ( bias==true ? vec(Flux.randn32(out)) : Flux.create_bias(weight, bias, out) )
    DenseNTK(weight, bias, σ)
end

function DenseNTK(in::Integer, out::Integer, σ=identity; bias=true, init=Flux.randn32)
    weight = init(out, in)
    bias = ( bias==true ? vec(Flux.randn32(out)) : Flux.create_bias(weight, bias, out) )
    DenseNTK(weight, bias, σ)
end

function (m::DenseNTK)(x::Vector)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.
    σ = NNlib.fast_act(m.σ, x)  # replaces tanh => tanh_fast, etc
    return σ.((m.weight/sqrt(size(m.weight)[2]))*x .+ m.bias)
end

function (m::DenseNTK)(x::Array)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.
    
    σ = NNlib.fast_act(m.σ, x)  # replaces tanh => tanh_fast, etc
    return σ.((m.weight/sqrt(size(m.weight)[2]))*x .+ m.bias)
end

### FOR DIFFERENT DIFF PACKAGES
function (m::DenseNTK)(x::ReverseDiff.TrackedArray)
    # Extend methods for handling ReverseDiff.TrackedArray
    σ = NNlib.fast_act(m.σ, x)
    return σ.((m.weight/sqrt(size(m.weight)[2]))*x .+ m.bias)
end

function (m::DenseNTK)(x::Tracker.TrackedVector)
    # Extend methods for handling ReverseDiff.TrackedArray
    σ = NNlib.fast_act(m.σ, x)
    return σ.((m.weight/sqrt(size(m.weight)[2]))*x .+ m.bias)
end

Flux.@functor DenseNTK