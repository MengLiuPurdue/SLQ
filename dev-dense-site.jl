## This script is for developing the initial dense-site ideas
# This ended up becomgin the experiment-grid-vis as I realized the grid
# graph would do fine.
using Plots, MatrixNetworks, SparseArrays, Statistics
function mycontour!(x,y,z;nlevels=10,threshhold=1e-12,kwargs...)
    nzset = z .> threshhold
    levels = quantile(z[nzset], range(0, 1-1/nlevels, length=nlevels))
    pushfirst!(levels, 0.0)
    contour!(x,y,z;levels=levels,kwargs...)
end
function myscatter!(xy, x; threshhold=1e-12,kwargs...)
    nzset = x .> threshhold
    scatter!(xy[nzset,1], xy[nzset,2], marker_z=log10.(x[nzset]);kwargs...)
end

##
function dense_site_graph(m::Integer, n::Integer, k::Integer, p::Real, q::Real;
  distance::Integer=1)
  N = m*n*k
  imap = reshape(1:N, m, n, k)
  ei = zeros(Int,0)
  ej = zeros(Int,0)
  for i=1:m
    for j=1:n
      # generate the little dense ER graph
      A = rand(k,k) .<= p
      for t1=1:k
        for t2=t1+1:k
          if A[t1,t2]
            src = imap[i,j,t1]
            dst = imap[i,j,t2]
            push!(ei,src)
            push!(ej,dst)
          end
        end
      end
      # generate site-to-site
      for di = -distance:distance
        for dj = -0:0
          if i+di >= 1 && i+di <= m && j+dj >= 1 && j+dj <= n
            if !(di ==0 && dj == 0)
              for ti=1:k
                for tj=1:k
                  if rand() .<= q
                    src = imap[i,j,ti]
                    dst = imap[i+di,j+dj,tj]
                    push!(ei,src)
                    push!(ej,dst)
                  end
                end
              end
            end
          end
        end
      end

      # generate site-to-site
      for di = 0:0
        for dj = -distance:distance
          if i+di >= 1 && i+di <= m && j+dj >= 1 && j+dj <= n
            if !(di ==0 && dj == 0)
              for ti=1:k
                for tj=1:k
                  if rand() .<= q
                    src = imap[i,j,ti]
                    dst = imap[i+di,j+dj,tj]
                    push!(ei,src)
                    push!(ej,dst)
                  end
                end
              end
            end
          end
        end
      end
    end
  end
  xy = zeros(N,2)
  for i=1:m
    for j=1:n
      for t=1:k
        I = imap[i,j,t]
        xy[I,1] = i-0.25+0.5*rand()
        xy[I,2] = j-0.25+0.5*rand()
      end
    end
  end
  A = sparse(ei,ej,1,N,N)
  A = max.(A,A')
  return sparse(ei,ej,1,N,N), xy, imap
end
m,n,k,p,q = 46, 32, 1, 1.0, 1.0
A,xy,imap = dense_site_graph(m,n,k,p,q)
##
x = personalized_pagerank(A,0.85, Set(imap[m ÷ 2 ,n ÷ 2,1:k]))
scatter(xy[:,1], xy[:,2], markersize=2, markerstrokewidth=0, label="")
myscatter!(xy, x; label="", markersize=2, markerstrokewidth=0)
Z = reshape(x, m, n, k)
mycontour!(1:m,1:n,vec(sum(Z,dims=3)), color=1, linewidth=2)
##
include("PageRank.jl")
##
x = PageRank.acl_diffusion((A=A, deg=vec(sum(A,dims=2))),
  vec(imap[m ÷ 2 ,n ÷ 2,1:k]),0.001, 0.01)
scatter(xy[:,1], xy[:,2], markersize=2, markerstrokewidth=0, label="", alpha=0.1)
myscatter!(xy, x; label="", markersize=2, markerstrokewidth=0)
Z = reshape(x, m, n, k)
mycontour!(1:46,1:32,vec(sum(Z,dims=3)), color=1, linewidth=2, nlevels=5)
##
include("SLQcvx.jl")
##
x = SLQcvx.slq_cvx(SLQ.graph(A),
  vec(imap[23,16,1:10]),
  2.0, 0.1, 0.1)[1]

##
include("SLQ.jl")
##
x = SLQ.slq_diffusion(SLQ.graph(A),
  vec(imap[m ÷ 2, n ÷ 2, 1:k]),
  0.01, 0.01, 0.5, # gamma, kappa, rho
  SLQ.loss_type(2.0, 0.0); max_iters=1000000)[1]
scatter(xy[:,1], xy[:,2], markersize=2, markerstrokewidth=0, label="", alpha=0.1)
myscatter!(xy, x; label="", markersize=2, markerstrokewidth=0)
Z = reshape(x, m, n, k)
mycontour!(1:46,1:32,vec(sum(Z,dims=3)), color=1, linewidth=2, nlevels=5)

##
x = SLQ.slq_diffusion(SLQ.graph(A),
  vec(imap[m ÷ 2, n ÷ 2, 1:k]),
  0.001, 0.001, 0.5, # gamma, kappa, rho
  SLQ.loss_type(5.0, 0.0); max_iters=1000000)[1]
scatter(xy[:,1], xy[:,2], markersize=2, markerstrokewidth=0, label="", alpha=0.1)
myscatter!(xy, x; label="", markersize=2, markerstrokewidth=0)
Z = reshape(x, m, n, k)
mycontour!(1:46,1:32,vec(sum(Z,dims=3)), color=1, linewidth=2, nlevels=5)

##
x = SLQ.slq_diffusion(SLQ.graph(A),
  vec(imap[m ÷ 2, n ÷ 2, 1:k]),
  0.01, 0.01, 0.5, # gamma, kappa, rho
  SLQ.loss_type(1.4, 0.0); max_iters=1000000)[1]
scatter(xy[:,1], xy[:,2], markersize=2, markerstrokewidth=0, label="", alpha=0.1)
myscatter!(xy, x; label="", markersize=2, markerstrokewidth=0)
Z = reshape(x, m, n, k)
mycontour!(1:46,1:32,vec(sum(Z,dims=3)), color=1, linewidth=2, nlevels=5)
