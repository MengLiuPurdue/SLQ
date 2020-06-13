module CRDlgc
using PyCall, LinearAlgebra
function __init__()
    include("setup-lgc.jl") # this will setup paths
    py"""
    import numpy as np
    import scipy.sparse as sp
    import time
    import os
    import localgraphclustering as lgc

    # h is the maximum flow that an edge can handle. h=3 seems to give good sparsity
    def crd(indptr,indices,data,n,S,U,h,w,iterations):
        Asp = sp.csr_matrix((data,indices,indptr), shape=(n,n))
        g = lgc.GraphLocal().from_sparse_adjacency(Asp)
        t1 = time.time()
        (cluster,conductance) = lgc.flow_clustering(g,S,method="crd",U=U,h=h,w=w,iterations=iterations)
        t2 = time.time()
        return cluster,conductance,t2-t1
    """
    # remove our addition...
    pop!(PyVector(pyimport("sys")."path"))
end

# h is the maximum flow that an edge can handle. h=3 seems to give good sparsity
function crd(G,S;U=3,h=3,w=2,iterations=20)
    A = G.A
    indptr = A.colptr.-1
    indices = A.rowval.-1
    S = S.-1
    data = A.nzval
    n = size(A,1)
    (cluster,conductance,time_crd) = py"crd"(indptr,indices,data,n,S,U,h,w,iterations)
    return cluster.+1,conductance,time_crd
end

end # end module

include("common.jl")
using Test
@testset "CRDlgc" begin
    A,xy = grid_graph(20,20)
    G = (A=A, ) # create a named tuple... which is a light struct
    @test_nowarn S, conductance, dt = CRDlgc.crd(G, [1,2])
end
