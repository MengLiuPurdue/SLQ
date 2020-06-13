## Visualize various diffusions on a grid.
include("SLQ.jl")
include("SLQcvx.jl")
include("PageRank.jl")
using MatrixNetworks
##
using Plots
gr()
##
include("common.jl")
m, n = 50, 50
A, xy, imap = grid_graph_axis(m, n)
##
using Plots, Measures, Statistics
function draw_graph(A::SparseMatrixCSC, xy; kwargs...)
    ei,ej = findnz(triu(A,1))[1:2]
    # find the line segments
    lx = zeros(0)
    ly = zeros(0)
    for nz=1:length(ei)
        src = ei[nz]
        dst = ej[nz]
        push!(lx, xy[src,1])
        push!(lx, xy[dst,1])
        push!(lx, Inf)

        push!(ly, xy[src,2])
        push!(ly, xy[dst,2])
        push!(ly, Inf)
    end
    plot(lx,ly;
        kwargs...)
end
function mycontour!(x,y,z;nlevels=10,threshhold=1e-8,kwargs...)
    nzset = z .> threshhold
    levels = quantile(z[nzset], range(0, 1-1/nlevels, length=nlevels))
    pushfirst!(levels, 0.0)
    contour!(x,y,z;levels=levels,colorbar=false,kwargs...)
end
function myscatter!(xy, x; threshhold=1e-8,kwargs...)
    nzset = x .> threshhold
    scatter!(xy[nzset,1], xy[nzset,2],
        marker_z=log10.(x[nzset]), markerstrokecolor=:white,;kwargs...)
end
draw_graph(A,xy;linewidth=0.5, color=:lightgrey, markersize=1, markerstrokewidth=0,
    markercolor=:grey, legend=false, marker=:dot, size=(250,250),
    framestyle=:none,background=RGBA{Float64}(1.0,1.0,1.0,0.0))
##
using Measures
draw_graph(A,xy;
    linewidth=0.5, color=1, markersize=1.5, markerstrokewidth=0,
    markercolor=:grey, markerstrokecolor=:white, framestyle=:none,
    margin=-20mm, legend=false, marker=:dot, size=(150,150),
    xlims=(4.5, 45.5), ylims=(4.5, 45.5))
mydrawgraph(A,xy) = draw_graph(A,xy;
    linewidth=0.5, color=1, markersize=1.25, markerstrokewidth=0,
    markercolor=:lightblue, markerstrokecolor=:white, framestyle=:none,
    margin=-20mm, legend=false, marker=:dot, size=(150,150), dpi=300,
    xlims=(4.5, 45.5), ylims=(4.5, 45.5))

savefig("figures/grid-graph.pdf")
##
##
seed = imap[m ÷ 2 ,n ÷ 2]
## Show PageRank first
x = personalized_pagerank(A,0.85, seed)
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5)
savefig("figures/grid-graph-ppr.pdf")
## Next up in ACL
x = PageRank.acl_diffusion(SLQ.graph(A), [seed], 0.1, 0.0001)
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5)
savefig("figures/grid-graph-acl-01-00001.pdf")

##
x = PageRank.acl_diffusion(SLQ.graph(A), [seed], 0.001, 0.001)
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5)
savefig("figures/grid-graph-acl-0001-0001.pdf")
## Next up is SLQ, can we run CVX?
@time x = SLQcvx.slq_cvx(SLQ.graph(A), [seed], 2.0, 0.1, 0.0001, solver="ECOS")[1]
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5)
savefig("figures/grid-graph-slqcvx-2-01-00001.pdf")

## Next up is SLQ, can we run CVX?
@time x = SLQcvx.slq_cvx(SLQ.graph(A), [seed], 2.0, 0.001, 0.001, solver="ECOS")[1]
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5)
savefig("figures/grid-graph-slqcvx-2-0001-0001.pdf")
## Next up is SLQ, can we run CVX?
@time x = SLQcvx.slq_cvx(SLQ.graph(A), [seed], 5.0, 0.001, 0.0001, solver="ECOS")[1]
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=5e-7)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=5e-7)
savefig("figures/grid-graph-slqcvx-5-0001-0001.pdf")
##
@time x = SLQcvx.slq_cvx(SLQ.graph(A), [seed], 5.0, 0.00001, 0.0001, solver="ECOS")[1]
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=5e-7)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=5e-7)
savefig("figures/grid-graph-slqcvx-5-000001-00001.pdf")
##
@time x = SLQcvx.slq_cvx(SLQ.graph(A), [seed], 1.5, 0.05, 0.000001, solver="ECOS")[1]
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=5e-7)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=5e-7)
savefig("figures/grid-graph-slqcvx-15-005-0000001.pdf")
##
@time x = SLQcvx.slq_cvx(SLQ.graph(A), [seed], 1.5, 0.01, 0.0005, solver="ECOS")[1]
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=5e-7)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=5e-7)
savefig("figures/grid-graph-slqcvx-15-001-00005.pdf")
##
@time x = SLQcvx.slq_cvx(SLQ.graph(A), [seed], 1.5, 0.01, 0.00015, solver="ECOS")[1]
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=5e-7)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=5e-7)
savefig("figures/grid-graph-slqcvx-15-001-000015.pdf")
## This one didn't work :( ) but this using gamma=0.1 did

#@time x = SLQcvx.slq_cvx(SLQ.graph(A), [seed], 4/3.0, 0.05, 0.000001, solver="ECOS")[1]
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=5e-7)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=5e-7)

## Try smaller q...
# Can't use CVX here...
# These figures need the adjsuted threshold because the true solution is close to 0
@time x = SLQ.slq_diffusion(SLQ.graph(A), [seed], 0.001, 0.001, 0.99,
    SLQ.QHuberLoss(1.25, 0.0); max_iters=100000000)[1]
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=minimum(x[x.>0]))
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=minimum(x[x.>0]))
savefig("figures/grid-graph-slq-125-0001-0001-099.pdf")

##
# These figures need the adjsuted threshold because the true solution is close to 0
@time x = SLQ.slq_diffusion(SLQ.graph(A), [seed], 0.001, 0.001, 0.99,
    SLQ.QHuberLoss(1.1, 0.0); max_iters=100000000)[1]
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2,
    markerstrokewidth=0, c=:heat, threshhold=minimum(x[x.>0]))
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=minimum(x[x.>0]))
#savefig("figures/grid-graph-slq-125-0001-0001-099.pdf")

## Try CRD
include("CRDlgc.jl")
x = zeros(size(A,1))
@time x[CRDlgc.crd(SLQ.graph(A), [seed])[1]] .= 1
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:black, threshhold=0)

##
x = zeros(size(A,1))
@time x[CRDlgc.crd(SLQ.graph(A), [seed];  U=60, h=60, w=5)[1]] .= 1
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:black, threshhold=0)
savefig("figures/grid-graph-crd-60-60-5.pdf")
##
x = zeros(size(A,1))
@time x[CRDlgc.crd(SLQ.graph(A), [seed];  U=10, h=10, w=2)[1]] .= 1
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:black, threshhold=0)
savefig("figures/grid-graph-crd-10-10-2.pdf")
##
x = zeros(size(A,1))
@time x[CRDlgc.crd(SLQ.graph(A), [seed];  U=90, h=90, w=90)[1]] .= 1
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:black, threshhold=0)
#savefig("figures/grid-graph-crd-10-10-2.pdf")

##
x = zeros(size(A,1))
@time x[CRDlgc.crd(SLQ.graph(A), [seed];  U=1000, h=1000, w=11)[1]] .= 1
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:black, threshhold=0)
savefig("figures/grid-graph-crd-1000-1000-11.pdf")
##
function nonlinear_diffusion(M, h, niter, v, p)
  n = size(M,1)
  d = vec(sum(M,dims=2))
  u = zeros(n)
  u .+= v
  u ./= sum(u) # normalize
  for i=1:niter
    gu = u.^p
    u = u - h*(gu - M*gu./d)
    u = max.(u, 0) # truncate to positive
  end
  return u
end
##
x = nonlinear_diffusion(sparse(A), 0.002, 1000, sparsevec([seed],1.0,size(A,1)), 0.5 )
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=minimum(x[x.>0]))
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=5*minimum(x[x.>0]))
##
x = nonlinear_diffusion(sparse(A), 0.002, 35000, sparsevec([seed],1.0,size(A,1)), 1.5 )
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=minimum(x[x.>0]))
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=5*minimum(x[x.>0]))
savefig("figures/grid-graph-p-diff-15-0002-35000.pdf")
##
x = nonlinear_diffusion(sparse(A), 0.001, 7500, sparsevec([seed],1.0,size(A,1)), 1.5 )
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=minimum(x[x.>0]))
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=5*minimum(x[x.>0]))
savefig("figures/grid-graph-p-diff-15-0001-7500.pdf")
##
x = nonlinear_diffusion(sparse(A), 0.0002, 100, sparsevec([seed],1.0,size(A,1)), 0.6 )
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=1e-9)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=1e-9)

##
x = nonlinear_diffusion(sparse(A), 0.0002, 10000, sparsevec([seed],1.0,size(A,1)), 5.0 )
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=minimum(x[x.>0]))
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=5*minimum(x[x.>0]))
##
x = nonlinear_diffusion(sparse(A), 0.00001, 500, sparsevec([seed],1.0,size(A,1)), 0.25 )
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=minimum(x[x.>0]))
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=5*minimum(x[x.>0]))

## Try nonlinear p-Laplacian
function nonlinear_Lp_diffusion(M, h, niter, v, p)
  # Form the incidence matrix
  ei, ej=findnz(triu(M,1))[1:2]
  B = sparse([1:length(ei); 1:length(ei)],
    [ei; ej], [ones(length(ei)); -ones(length(ei))],
    length(ei), size(M,1))
  n = size(M,1)
  d = vec(sum(M,dims=2))
  u = zeros(n)
  u .+= v
  u ./= sum(u) # normalize
  for i=1:niter
      # du = L*(D^-1) u
      du = B*(u./d)
      u = u .- h*(B'*(abs.(du).^(p-1).*sign.(du)))
      u = max.(u, 0) # truncate to positive
  end
  return u./d
end
##
x = nonlinear_Lp_diffusion(sparse(A), 0.001, 1000, sparsevec([seed],1.0,size(A,1)), 1.5 )
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=1.e-4)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=1.e-4)
savefig("figures/grid-graph-Lp-diff-15-0001-1000.pdf")
##
x = nonlinear_Lp_diffusion(sparse(A), 0.0001, 7500, sparsevec([seed],1.0,size(A,1)), 1.5 )
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=1.e-5)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=1.e-5)
savefig("figures/grid-graph-Lp-diff-15-00001-7500.pdf")

##
##
x = nonlinear_Lp_diffusion(sparse(A), 0.001, 100000, sparsevec([seed],1.0,size(A,1)), 2.5 )
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=1.e-8)
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=1.e-8)
#savefig("figures/grid-graph-Lp-diff-1.5-0001-1000.pdf")

## Compare to heat-kernel
x = seeded_stochastic_heat_kernel(A, 6.5, sparsevec([seed],1.0,size(A,1)), 1.0e-5)
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=minimum(x[x.>0]))
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=minimum(x[x.>0]))
savefig("figures/grid-graph-hk-65-1e-5.pdf")
##
x = seeded_stochastic_heat_kernel(A, 3.5, sparsevec([seed],1.0,size(A,1)), 1.0e-9)
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=minimum(x[x.>0]))
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=minimum(x[x.>0]))
savefig("figures/grid-graph-hk-35-1e-9.pdf")
##
x = seeded_stochastic_heat_kernel(A, 10.0, sparsevec([seed],1.0,size(A,1)), 3.0e-3)
mydrawgraph(A,xy)
myscatter!(xy, x; label="", markersize=2.1, markerstrokewidth=0, c=:heat, threshhold=minimum(x[x.>0]))
mycontour!(1:m,1:n,x, color=1, linewidth=2, nlevels=5, threshhold=minimum(x[x.>0]))
savefig("figures/grid-graph-hk-100-3e-3.pdf")
