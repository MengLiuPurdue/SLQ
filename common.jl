using SparseArrays, LightGraphs
using PyCall


function grid_graph_axis(m::Int, n::Int; distance::Int=1)
  N = m*n
  imap = reshape(1:N, m, n)
  ei = zeros(Int,0)
  ej = zeros(Int,0)
  for i=1:m
    for j=1:n
      # we only put in axis-aligned edges (adapted from some old code...)
      for di = -distance:distance
        dj = 0
        if i+di >= 1 && i+di <= m && j+dj >= 1 && j+dj <= n
          src = imap[i,j]
          dst = imap[i+di,j+dj]
          if (src != dst)
            push!(ei,src)
            push!(ej,dst)
          end
        end
      end
      for dj = -distance:distance
        di = 0
        if i+di >= 1 && i+di <= m && j+dj >= 1 && j+dj <= n
          src = imap[i,j]
          dst = imap[i+di,j+dj]
          if (src != dst)
            push!(ei,src)
            push!(ej,dst)
          end
        end
      end
    end
  end
  xy = zeros(N,2)
  for i=1:m
      for j=1:n
          I = imap[i,j]
          xy[I,1] = i
          xy[I,2] = j
      end
  end
  return sparse(ei,ej,1,N,N), xy, imap
end

function grid_graph(m::Int, n::Int; distance::Int=1)
  N = m*n
  imap = reshape(1:N, m, n)
  ei = zeros(Int,0)
  ej = zeros(Int,0)
  for i=1:m
    for j=1:n
      for di = -distance:distance
        for dj = -distance:distance
          if i+di >= 1 && i+di <= m && j+dj >= 1 && j+dj <= n
            src = imap[i,j]
            dst = imap[i+di,j+dj]
            if (src != dst)
              push!(ei,src)
              push!(ej,dst)
            end
          end
        end
      end
    end
  end
  xy = zeros(N,2)
  for i=1:m
      for j=1:n
          I = imap[i,j]
          xy[I,1] = i
          xy[I,2] = j
      end
  end
  return sparse(ei,ej,1,N,N), xy
end

##
using SparseArrays, LinearAlgebra
function two_cliques(m::Int, n::Int)
  N = m+n
  A = [sparse(ones(m,m)) spzeros(m,n); spzeros(n,m) sparse(ones(n,n))]
  A = A .- Diagonal(A)
  xy = zeros(N,2)
  for i=1:m
    I = i
    xy[I,1] = cos(2*pi*(i-1)/m)-1.5
    xy[I,2] = sin(2*pi*(i-1)/m)
  end
  A[1,end] = 1
  A[end,1] = 1
  for i=1:n
    I = i+m
    xy[I,1] = cos(2*pi*(i)/n+pi)+1.5
    xy[I,2] = sin(2*pi*(i)/n)
  end
  return A,xy
end

##

function sbm(random_seed,p,q,ncls,cls_size)
    G = stochastic_block_model(round(Int,cls_size*p),round(Int,cls_size*q),[cls_size for i in 1:ncls],seed=random_seed);
    G = LightGraphs.LinAlg.adjacency_matrix(G)
    return G
end

function readSMAT(filename::AbstractString)
  f = open(filename)
  header = readline(f)
  headerparts = split(header)
  nedges = parse(Int,headerparts[3])
  ei = zeros(Int64,nedges)
  ej = zeros(Int64, nedges)
  ev = zeros(Float64, nedges)
  @inbounds for i = 1:nedges
      curline = readline(f)
      parts = split(curline)
      ei[i] = parse(Int, parts[1])+1
      ej[i] = parse(Int, parts[2])+1
      ev[i] = parse(Float64, parts[3])
  end
  close(f)
  A = sparse(ei, ej, ev,
             parse(Int,headerparts[1]),
             parse(Int,headerparts[2])
             )
  return A
end

function compute_pr_rc(prediction,truth)
  pr = 1-length(setdiff(prediction,truth))/length(prediction)
  rc = 1-length(setdiff(truth,prediction))/length(truth)
  return pr,rc
end


module LFR
using PyCall, LinearAlgebra, SparseArrays
function __init__()
    pyimport_conda("networkx", "networkx")
    py"""
    import networkx as nx
    import numpy as np

    # h is the maximum flow that an edge can handle. h=3 seems to give good sparsity
    def lfr(n,mu,seed,tau1=2,tau2=2,average_degree=10,max_degree=50,min_community=200,max_community=500):
        g = nx.LFR_benchmark_graph(n,tau1,tau2,mu,average_degree=average_degree,max_degree=max_degree,min_community=min_community,max_community=max_community,seed=seed)
        A = nx.to_scipy_sparse_matrix(g)
        communities = list({frozenset(g.nodes[v]['community']) for v in g})
        communities = [list(community) for community in communities]
        return A.indptr+1,A.indices+1,A.data,communities
    """
end

function create_LFR(n,mu,seed;tau1=2,tau2=2,average_degree=10,max_degree=50,min_community=200,max_community=500)
  indptr,indices,data,communities = py"lfr"(n,mu,seed,tau1=tau1,tau2=tau2,average_degree=average_degree,max_degree=max_degree,min_community=min_community,max_community=max_community)
  communities = [community.+1 for community in communities]
  return SparseMatrixCSC(n,n,indptr,indices,data),communities
end

end
