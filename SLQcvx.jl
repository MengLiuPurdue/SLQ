module SLQcvx
using PyCall, LinearAlgebra
function __init__()
    cvx = pyimport_conda("cvxpy","cvxpy","conda-forge")
    py"""
    import cvxpy as cp
    import numpy as np
    import scipy.sparse as sp
    import time
    import os

    # TODO Refactor these into a single function
    # But basicaly, don't ever use this function except for testing.
    def SLQcvx_dense(indptr,indices,data,n,S,deg,q,gamma,kappa,solver):
        x = cp.Variable(n)
        B = np.zeros((len(data),n))
        for i in range(n):
            for k in range(indptr[i],indptr[i+1]):
                j = indices[k]
                B[k,i] = 1
                B[k,j] = -1
        eS,w = np.zeros(n),gamma*deg
        eS[S] = 1
        t1 = time.time()
        objective = cp.Minimize(1/q*cp.sum(cp.multiply(data,cp.power(cp.abs(B@x),q)))+1/q*cp.sum(cp.multiply(w,cp.power(cp.abs(eS-x),q)))+gamma*kappa*cp.sum(cp.multiply(deg,x)))
        # objective = cp.Minimize(1/q*obj+gamma*kappa*cp.sum(cp.multiply(deg,x)))
        constraints = [np.min(x)>=0]
        prob = cp.Problem(objective,constraints)
        result = prob.solve(solver=solver)
        t2 = time.time()
        return x.value,t2-t1

    def make_incidence(n, indptr, indices):
        src = []
        dst = []
        vals = []
        for i in range(n):
            for k in range(indptr[i],indptr[i+1]):
                j = indices[k]
                # B[k,i] = 1
                src.append(k)
                dst.append(i)
                vals.append(1.0)
                # B[k,j] = -1
                src.append(k)
                dst.append(j)
                vals.append(-1.0)
        return sp.csc_matrix((vals, (src,dst)), shape=(len(indices), n))

    def SLQcvx_sparse(indptr,indices,data,n,S,deg,q,gamma,kappa,solver):
        x = cp.Variable(n)
        B = (make_incidence(n, indptr, indices))
        eS,w = np.zeros(n),gamma*deg
        eS[S] = 1
        t1 = time.time()
        objective = cp.Minimize(
                            1/q*cp.sum(cp.multiply(data,cp.power(cp.abs(B@x),q)))+
                            1/q*cp.sum(cp.multiply(w,cp.power(cp.abs(eS-x),q)))+
                            gamma*kappa*cp.sum(cp.multiply(deg,x)))
        # objective = cp.Minimize(1/q*obj+gamma*kappa*cp.sum(cp.multiply(deg,x)))
        constraints = [np.min(x)>=0]
        prob = cp.Problem(objective,constraints)
        result = prob.solve(solver=solver)
        t2 = time.time()
        return x.value,t2-t1
    """
end

function slq_cvx_dense(G,S,q,gamma,kappa;solver="SCS")
    A = G.A
    Au = triu(A)
    deg = G.deg
    indptr = Au.colptr.-1
    indices = Au.rowval.-1
    S = S.-1
    data = Au.nzval
    n = size(A,1)
    (x_cvx,time_cvx) = py"SLQcvx_dense"(indptr,indices,data,n,S,deg,q,gamma,kappa,solver)
    return x_cvx,time_cvx
end


function slq_cvx(G,S,q,gamma,kappa;solver="SCS")
    A = G.A
    Au = triu(A)
    deg = G.deg
    indptr = Au.colptr.-1
    indices = Au.rowval.-1
    S = S.-1
    data = Au.nzval
    n = size(A,1)
    (x_cvx,time_cvx) = py"SLQcvx_sparse"(indptr,indices,data,n,S,deg,q,gamma,kappa,solver)
    return x_cvx,time_cvx
end

end # end module

include("common.jl")
using Test
@testset "SLQcvx" begin
    A,xy = two_cliques(5,5)
    G = (A=A, deg=vec(sum(A,dims=2)))
    x = SLQcvx.slq_cvx_dense(G, [2], 1.5, 0.1, 0.1, solver="ECOS")[1]
    y = SLQcvx.slq_cvx(G, [2], 1.5, 0.1, 0.1, solver="ECOS")[1]
    @test x ≈ y

    A,xy = grid_graph(8,12)
    G = (A=A, deg=vec(sum(A,dims=2)))
    x = SLQcvx.slq_cvx_dense(G, [2], 2, 0.1, 0.1, solver="ECOS")[1]
    y = SLQcvx.slq_cvx(G, [2], 2, 0.1, 0.1, solver="ECOS")[1]
    @test x ≈ y
end
