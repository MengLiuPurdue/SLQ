##
include("common.jl")
include("PageRank.jl")
include("SLQ.jl")
include("SLQcvx.jl")

##
gm,gn = 46, 32
A,xy = grid_graph(gm, gn)
seed = 32*16+15

##
using Plots
function myscatter!(xy, x; threshhold=1e-12,kwargs...)
    nzset = x .> threshhold
    scatter!(xy[nzset,1], xy[nzset,2], marker_z=log10.(x[nzset]);kwargs...)

end
## Show the Seed
scatter(xy[:,1],xy[:,2], label="", markerstrokewidth=0, markersize=2)
scatter!([xy[seed,1]], [xy[seed,2]], markersize=12, color=:darkred, label="")

##
scatter(xy[:,1],xy[:,2], label="", markerstrokewidth=0, markersize=2)
scatter!([xy[seed,1]], [xy[seed,2]], markersize=12, color=:darkred, label="")
x1, r1, iters = SLQ.slq_diffusion(SLQ.graph(A),[seed],
    0.1, # gamma, the aug-graph regularization
    0.001, # kappa, the sparsity regularizaiton
    0.99, # rho, KKT approx
    SLQ.TwoNormLoss{Float64}(),
    ; max_iters=1000000,
    )
@show iters
myscatter!(xy, x1; label="")
##
scatter(xy[:,1],xy[:,2], label="", markerstrokewidth=0, markersize=2)
scatter!([xy[seed,1]], [xy[seed,2]], markersize=12, color=:darkred, label="")
xt, r1, iters = SLQ.slq_diffusion(SLQ.graph(A),[seed],
    0.1, # gamma, the aug-graph regularization
    0.001, # kappa, the sparsity regularizaiton
    0.5, # rho, KKT approx
    SLQ.TwoNormLoss{Float64}(),
    ; max_iters=1000000,
    )
@show iters
myscatter!(xy, xt; label="")
contour(1:32, 1:46, xt, nlevels=15)

##
scatter(xy[:,1],xy[:,2], label="", markerstrokewidth=0, markersize=2)
scatter!([xy[seed,1]], [xy[seed,2]], markersize=12, color=:darkred, label="")
xacl  = PageRank.acl_diffusion(SLQ.graph(A),[seed],
    0.1, # gamma, the aug-graph regularization
    0.001, # kappa, the sparsity regularizaiton
    )
myscatter!(xy,xacl;label="")
##
contour(1:32, 1:46, xacl, nlevels=15)
##
scatter(xy[:,1],xy[:,2], label="", markerstrokewidth=0, markersize=2)
scatter!([xy[seed,1]], [xy[seed,2]], markersize=12, color=:darkred, label="")
xpr  = PageRank.pr_diffusion(SLQ.graph(A),[seed],
    0.1, # gamma, the aug-graph regularization
    )
myscatter!(xy,xpr;label="")
##
scatter(xy[:,1],xy[:,2], label="", markerstrokewidth=0, markersize=2)
scatter!([xy[seed,1]], [xy[seed,2]], markersize=12, color=:darkred, label="")
#=
xt, r1, iters = SLQ.slq_diffusion(SLQ.graph(A),[seed],
    0.1, # gamma, the aug-graph regularization
    0.0001, # kappa, the sparsity regularizaiton
    0.999, # rho, KKT approx
    SLQ.QHuberLoss{Float64}(1.2,0.0),
    ; max_iters=1000000,
    )
    =#
# y = SLQcvx.slq_cvx(G, [2], 1.5, 0.1, 0.1, solver="ECOS")[1]
xt, dt = SLQcvx.slq_cvx(SLQ.graph(A), [seed],
    1.5, # q
    0.1, #gamma
    0.1; solver="SCS") # kappa
@show iters
myscatter!(xy, xt; label="")
# figure out contour
function mycontour!(x,y,z;nlevels=10,threshhold=1e-12,kwargs...)
    nzset = z .> threshhold
    levels = quantile(z[nzset], range(0, 1-1/nlevels, length=nlevels))
    pushfirst!(levels, 0.0)
    contour!(x,y,z;levels=levels,kwargs...)
end
mycontour!(1:gm, 1:gn, xt, nlevels=5, linewidth=2, color=1)
plot!(aspect_ratio=:equal)
