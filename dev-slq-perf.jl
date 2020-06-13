## Checkout the performance of slq_diffusion
# it seems much slower than ACL when run with the 2 norm loss and
# I want to see why!
include("SLQ.jl")
##
include("common.jl")
G = grid_graph_axis(256,256)[1]
##
include("PageRank.jl")
##
@time x = PageRank.acl_diffusion(SLQ.graph(G), [1], 0.01, 0.001)
@time y = SLQ.slq_diffusion(SLQ.graph(G), [1], 0.01, 0.001, 0.5, SLQ.TwoNormLoss{Float64}();
  max_iters=100000)
## when I started...
#   0.002004 seconds (78 allocations: 1.639 MiB)
#  0.047797 seconds (2.43 M allocations: 41.039 MiB)
## after I finished...
# 0.001944 seconds (77 allocations: 1.639 MiB)
# 0.002090 seconds (32 allocations: 2.502 MiB)
## The deal was that the SLQ.Graph type wasn't parameterized for the sparse
# matrix integer type. So it couldn't solve all the type inference problems
# on that, ugh. So annoying.
# So I had to change
#=
struct GraphAndDegrees{
        T<: Union{Float32,Float64,Int32,Int64},}   # T is the type of edges,
  A::SparseMatrixCSC{T}
  deg::Vector{T}
end

to

struct GraphAndDegrees{
        T<: Union{Float32,Float64,Int32,Int64},
        Ti <: Union{Int,Int32,Int64}}   # T is the type of edges,
  A::SparseMatrixCSC{T,Ti}
  deg::Vector{T}
end
=#
# it seems like tehre is
## Old working notes!
# so it seems like there is some per-iteration allocation. Let's see if we
# can figure out where!
@code_warntype SLQ.slq_diffusion(SLQ.graph(G), [1], 0.01, 0.001, 0.5, SLQ.TwoNormLoss{Float64}();
  max_iters=100000)
  # Not so helpful... let's try calling things directly.
##
using DataStructures
include("SLQ.jl")
function my_slq_diffusion(G::SLQ.GraphAndDegrees,S,gamma::T,kappa::T,rho::T,L::SLQ.EdgeLoss{T},
        max_iters::Int,epsilon::T) where {T <: Real}

    A = G.A
    n = size(A,1)
    x = zeros(n)
    r = zeros(n)

    max_deg = SLQ._max_nz_degree(A)

    buf_x = zeros(max_deg)
    buf_vals = zeros(max_deg)
    Q = CircularDeque{Int}(n)
    #
    for i in S
        r[i] = G.deg[i]
        push!(Q,i)
    end
    seedset = Set(S)

    iter = 0
    #thd1 = (sum(G.deg[S])/sum(G.deg))^(1/(q-1))
    thd1 = SLQ.minval(sum(G.deg[S])/sum(G.deg), L)
    thd2 = thd1
    while length(Q) > 0 && iter < max_iters
        i = popfirst!(Q)
        dxi = SLQ.dxi_solver(G,x,kappa,epsilon,gamma,r,seedset,rho,i,L,buf_x,buf_vals,thd1,thd2)
        thd2 = dxi
        SLQ.residual_update!(G,x,dxi,i,seedset,r,gamma,Q,kappa,L)
        x[i] += dxi
        iter += 1
    end
    if iter == max_iters && length(Q) > 0
        @warn "reached maximum iterations"
    end
    return x,r,iter
end
 y = my_slq_diffusion(SLQ.graph(G), [1], 0.01, 0.001, 0.5, SLQ.TwoNormLoss{Float64}(),
  100000, 1e-8)
@time y = my_slq_diffusion(SLQ.graph(G), [1], 0.01, 0.001, 0.5, SLQ.TwoNormLoss{Float64}(),
  100000, 1e-8)
##
@code_warntype  my_slq_diffusion(SLQ.graph(G), [1], 0.01, 0.001, 0.5, SLQ.TwoNormLoss{Float64}(),
  100000, 1e-8)
##
@code_warntype SLQ._max_nz_degree(G)
