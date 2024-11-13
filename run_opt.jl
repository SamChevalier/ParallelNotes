using JuMP, Ipopt
using LinearAlgebra

function opt(M,A,b)
    nv = length(b)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:nv])
    @constraint(model, A*x .<= b)
    @objective(model, Min, x'*M*x)
    optimize!(model)
    return objective_value(model)
end

function solve_opts(M,A,b;serial=true)
    n_opt = length(M)
    if serial == true
        for ii in 1:n_opt
            println("solving problem ", ii)
            opt(M[ii],A[ii],b[ii])
        end
    else
        Threads.@threads for ii in 1:n_opt
            println("solving problem ", ii)
            opt(M[ii],A[ii],b[ii])
        end
    end
end

num_problems = 50
Mn = [rand(500,500) for ii in 1:num_problems]
for ii in 1:num_problems
    Mn[ii] = Mn[ii]'*Mn[ii]
end

An = [randn(500,500) for ii in 1:num_problems]
bn = [randn(500) for ii in 1:num_problems]

#obj_vals = solve_opts(Mn,An,bn; serial=true)

LinearAlgebra.BLAS.set_num_threads(1)
obj_vals = solve_opts(Mn,An,bn; serial=false)