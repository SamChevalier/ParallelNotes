using Pkg
Pkg.activate(".")

using CUDA
x = randn(1000)
xc = CuArray(x)