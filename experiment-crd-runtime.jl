include("PageRank.jl")
include("SLQ.jl")
include("CRDlgc.jl") 
include("common.jl")

n = 1000
A = sbm(1,0.1,0.01,5,n)
G = SLQ.graph(A)
S = collect(1:round(Int,0.01*n))
L = SLQ.QHuberLoss(1.5, 0.0)
time_sllp = @elapsed (x,r,iter) = SLQ.slq_diffusion(G, S, 0.1, 0.005, 0.5, L, max_iters=100000,epsilon=1.0e-8)
cluster,time_crd = CRDlgc.crd(G,S)
