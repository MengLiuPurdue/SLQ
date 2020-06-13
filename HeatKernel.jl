#= David's old Heat Kernel code from the WAW2015 tutorial, updated for Julia 1.0 =#

module HeatKernel

using SparseArrays

"""
This computes a vector
    exp(-(I-P)) v
where P is a column stochastic matrix
"""
function stochastic_heat_kernel_series!(
    x::Vector{T}, y::Vector{T}, z::Vector{T},
    P, t::T, v, eps::T,
    maxiter::Int) where T

    iexpt = exp(-t)
    _applyv(y,v,0.,1.) # iteration number 0
    # scale by iexpt
    @simd for i=1:length(x)
        @inbounds x[i] = iexpt*y[i]
    end

    eps_exp_t = eps*exp(t)
    err = exp(t)-1.
    coeff = 1.

    for k=1:maxiter
        A_mul_B!(z,P,y)       # compute z = P*y
        coeff = coeff*t/k
        @simd for i=1:length(x)
            @inbounds x[i] = x[i] + (iexpt*coeff)*z[i]
        end
        y,z = z,y # swap

        err = err - coeff
        if err < eps_exp_t
            break
        end
    end
    x
end



function single_seed_stochastic_heat_kernel_series(P,t::Float64,seed::Int64)
    n = size(P,1)
    v = sparsevec([seed],[1.],n)
    x = zeros(n)
    y = zeros(n)
    z = zeros(n)
    return stochastic_heat_kernel_series!(x, y, z, P, t, v, 1.e-8, 10000)
end

function _hk_taylor_degree(t::Float64, eps::Float64, maxdeg::Int)
    eps_exp_t = eps*exp(t)
    err = exp(t)-1.0
    coeff = 1.0
    k::Int = 0
    while err > eps_exp_t && k < maxdeg
        k += 1
        coeff = coeff*t/k
        err = err - coeff
    end
    return max(k,1)
end

function _hk_psis(N::Int,t::Float64,eps::Float64)
    psis = zeros(Float64,N)
    psis[N] = 1.0
    for k=N-1:-1:1
        psis[k] = psis[k+1]*t/(k+1.0) + 1.0
    end

    pushcoeffs = zeros(Float64,N+1)

    pushcoeffs[1] = (exp(t)*eps/N)/psis[1]
    for k=2:N
        pushcoeffs[k] = pushcoeffs[k-1]*psis[k-1]/psis[k]
    end

    return psis, pushcoeffs
end


function hk_relax(A::SparseMatrixCSC{T,Int},
    seeds::Array{Int}, t::Float64, eps::Float64, maxdeg::Int, maxpush::Int) where T
    colptr = A.colptr
    rowval = A.rowval
    n = size(A,1)

    N = _hk_taylor_degree(t,eps,maxdeg)
    psis, pushcoeffs = _hk_psis(N,t,eps)
    # @show psis
    # @show pushcoeffs

    exp_eps_t = exp(t)*eps

    x = Dict{Int,Float64}()     # Store x, r as dictionaries
    r = Dict{Tuple{Int,Int},Float64}()     # initialize residual
    Q = Tuple{Int,Int}[]        # initialize queue
    npush = 0.

    # TODO handle a generic seed
    for seed in seeds
        r[(seed,0)] = 1.
        push!(Q,(seed,0))
    end

    npush = 1

    while length(Q) > 0 && npush <= maxpush
        v,j = popfirst!(Q)
        rvj = r[(v,j)]

        r[(v,j)] = 0.
        x[v] = get(x,v,0.) + rvj

        dv = Float64(colptr[v+1]-colptr[v]) # get the degree
        update = t*rvj/(j+1.)
        mass = update/dv

        for nzi in colptr[v]:(colptr[v+1] - 1)

            u = rowval[nzi]
            next = (u,j+1)
            if j+1 == N
                x[u] += mass
                continue
            end
            rnext = get(r,next,0.)
            thresh = dv*pushcoeffs[j+1]
            if rnext < thresh && (rnext + mass) >= thresh
                push!(Q, (u,j+1))
            end
            r[next] = rnext + mass
        end
        npush += colptr[v+1]-colptr[v]
    end
    return x
end


function local_sweep_cut(A::SparseMatrixCSC{T,Int}, x::Dict{Int,V}) where {T,V}
    colptr = A.colptr
    rowval = A.rowval
    n = size(A,1)
    Gvol = A.colptr[n+1]

    sx = sort(collect(x), by=x->x[2], rev=true)
    S = Set{Int64}()
    volS = 0.
    cutS = 0.
    bestcond = 1.
    beststats = (1,1)
    bestset = Set{Int64}()
    for p in sx
        if length(S) == n-1
            break
        end
        u = p[1] # get the vertex
        volS += colptr[u+1] - colptr[u]
        for nzi in colptr[u]:(colptr[u+1] - 1)
            v = rowval[nzi]
            if v in S
                cutS -= 1.
            else
                cutS += 1.
            end
        end
        push!(S,u)
        if cutS/min(volS,Gvol-volS) < bestcond
            bestcond = cutS/min(volS,Gvol-volS)
            bestset = Set(S)
            beststats = (cutS,min(volS,Gvol-volS))
        end
    end
    return bestset, bestcond, beststats
end

function degree_normalized_sweep_cut!(A::SparseMatrixCSC{T,Int},
        x::Dict{Int,V}) where {T,V}
    colptr = A.colptr
    rowval = A.rowval

    for u in keys(x)
        x[u] = x[u]/(colptr[u+1] - colptr[u])
    end

    return local_sweep_cut(A,x)
end

function hk_relax_solution(A::SparseMatrixCSC{T,Int},
        t::Float64, seeds::Array{Int}, eps::Float64) where {T}
    maxdeg = 100000
    maxpush = 10^9
    return hk_relax(A,seeds,t,eps,maxdeg,maxpush)
end


function hk_grow(A::SparseMatrixCSC{T,Int}, seeds::Array{Int}) where T
    epsvals = [1.e-4,1.e-3,5.e-3,1.e-2]
    tvals = [10. 20. 40. 80.]

    @assert length(epsvals) == length(tvals)

    ntrials = length(tvals)

    hkvec = hk_relax_solution(A,tvals[1],seeds,epsvals[1])
    bestset,bestcond,beststats = degree_normalized_sweep_cut!(A,hkvec)

    for i=2:ntrials
        hkvec = hk_relax_solution(A,tvals[i],seeds,epsvals[i])
        set,cond,stats = degree_normalized_sweep_cut!(A,hkvec)
        if cond < bestcond
            bestset = set
            bestcond = cond
            beststats = stats
        end
    end

    return bestset, bestcond, beststats
end

function hk_grow_one(A::SparseMatrixCSC{T,Int},
        seed::Int, t::Float64, eps::Float64) where {T}
    hkvec = hk_relax_solution(A,t,[seed],eps)
    return degree_normalized_sweep_cut!(A,hkvec)
end

end # end HeatKernel module

using Test
include("common.jl")
@testset "HeatKernel" begin
    @testset "grid-graph" begin
        A = grid_graph(10,12)[1]
        @test_nowarn HeatKernel.hk_grow_one(A, 1, 2.5, 1e-4)
        @test_nowarn HeatKernel.hk_grow(A, [1])
    end

    @testset "four-clusters" begin
        using MatrixNetworks
        A = Float64.(MatrixNetworks.readSMAT("data/four_clusters.smat"))

        S = HeatKernel.hk_grow(A, [1])[1]
        @test S == Set([7,4,9,2,3,8,5,6,1])

        # This was from our diffusion tutorial results
        hkvec = HeatKernel.hk_relax_solution(A,5.0,[1],1e-4)
        @test hkvec[1] ≈ 1.935901229830458803e+01
        @test hkvec[2] ≈ 9.405980777381179436e+00
    end

        #= ipynb code
        "hkvec = DiffusionAlgorithms.hk_relax_solution(A,5.,1,1.e-4)\n",
         "@printf \"hkvec[1] = %.18e\\n\" hkvec[1]\n",
         "@printf \" true[1] = %.18e\\n\" 19.359526487166217\n",
         "@printf \"hkvec[1] = %.18e\\n\" hkvec[2]\n",
         "@printf \" true[1] = %.18e\\n\" 9.408867012443888\n"

         The output was

         "hkvec[1] = 1.935901229830458803e+01\n",
      " true[1] = 1.935952648716621738e+01\n",
      "hkvec[1] = 9.405980777381179436e+00\n",
      " true[1] = 9.408867012443888456e+00\n"

      which is now what we use above... 
        =#


end
