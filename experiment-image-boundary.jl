## Experiment on boundaries in images.

## Setup
using Images, LinearAlgebra, SparseArrays
_default_diff_fn(c1::CT1,c2::CT2) where {CT1<:Union{Colorant,Real}, CT2<:Union{Colorant,Real}} = sum(abs2,(c1)-Images.accum(CT2)(c2))
function image2graph(im::AbstractArray, r::Real, sigmap2::Real, sigmax2::Real)
  # see https://julialang.org/blog/2016/02/iteration/
  # for some of the syntax help (in the boxcar filter)
  d = ceil(Int,sqrt(r))
  R = CartesianIndices(im) # get the iterator of CartesianIndices for im
  imap = LinearIndices(R)  # converge CartesianIndices to Linear Indices for the graph
  Ifirst, Ilast = first(R), last(R) # get the image bounds
  Id = d*oneunit(Ifirst)
  # allocate sparse array inputs, we could create this directly, but is' more
  # error prone
  ei = zeros(Int, 0)
  ej = zeros(Int, 0)
  ev = zeros(0)
  for I in CartesianIndices(im)
    # find adjacent pixel neighbors at distance d, respecting bounds
    Ilower = max(Ifirst,I-Id)
    Iupper = min(Ilast,I+Id)
    src = imap[I]
    pi = im[I]
    for J in Ilower:Iupper
      dst = imap[J]
      if src == dst # skip self-loops
        continue
      end
      dx2 = norm(Tuple(I-J))^2
      if dx2 <= r
        pj = im[J]
        dp2 = _default_diff_fn(pi,pj)
        w = exp(-dp2/sigmap2-dx2/sigmax2)
        push!(ei, src)
        push!(ej, dst)
        push!(ev, w)
      end
    end
  end
  return sparse(ei,ej,ev,length(im),length(im)), imap
end


using Test
@testset "image2graph" begin
  using TestImages
  im = testimage("cameraman")
  @test_nowarn image2graph(im, 80, maximum(size(im))/10, 100)
end

include("PageRank.jl")
include("SLQ.jl")

"""
Run a diffusion algorithm that takes a parameter kappa for a sparsity
  level and record the first sparsity level where we find a node.
  This assumes that kappas are sorted
  """
function diffuse_vals(kappas, F)
  @assert(issorted(kappas))
  x = F(first(kappas))
  x[x .> 0] .= 1
  for i = 2:length(kappas)
    y = F(kappas[i])
    x[y .> 0] .= i
  end
  return x
end

using MappedArrays, IndirectArrays, Plots
function color_me(A, cmap)
    # remake cmap with alpha values
    n = length(cmap)
    # add alpha values to cmap
    cmap = map( i -> RGBA(cmap[i].r, cmap[i].g, cmap[i].b, ceil(Int,(i-1)/(n-1))), 1:n)
    Amin,Amax = extrema(A)
    f = s->clamp(round(Int, (n-1)*((s-Amin)/(Amax-Amin)))+1, 1, n)  # safely convert 0-1 to 1:n
    Ai = mappedarray(f, A)       # like f.(A) but does not allocate significant memory
    X = IndirectArray(Ai, cmap)      # colormap array
end
mygrad = cgrad(:magma)
mygrad = mygrad[128:end]

## Adjustable setup
using Plots, Measures
function myplot(x,A,seed)
  heatmap(A, framestyle=:none, margin=-20mm, size=(400,400))
  heatmap!(color_me(reshape(x, size(A)...), mygrad))
  sy = CartesianIndices(A)[seed][1]
  sx = CartesianIndices(A)[seed][2]
  scatter!([sx],[sy],markersize=6,color=1,label="")
  xlims!(175,375)
  ylims!(175,375)
end
##
using TestImages
@time A = testimage("house")
@time G,imap = image2graph(A, 40, 0.001, maximum(size(A))/10)
seed = imap[300,243]
kappas = reverse([0.005,0.0025,0.001])
##
seedvec = zeros(size(G,1))
seedvec[seed] = 1
myplot(seedvec,A,seed)
##
savefig("figures/boundary-seed.png")
##
## We are going to use this solution as the boundary
xsoln = SLQ.slq_diffusion(SLQ.graph(G), [seed], 0.001,
          0.001, 0.5, SLQ.loss_type(1.1,0.0); max_iters=1000000)[1]
myplot(xsoln .> 0, A, seed)
##
function find_boundary(Y; d::Integer = 1)
  # find the boundary of Y, that is, non-zero pixels where one of the
  # adjacent pixels are different
  B = similar(BitArray, axes(A)) # boundary
  fill!(B, false)
  R = CartesianIndices(Y) # get the iterator of CartesianIndices for im
  Ifirst, Ilast = first(R), last(R) # get the image bounds
  Id = d*oneunit(Ifirst)
  for I in R
    Ilower = max(Ifirst,I-Id)
    Iupper = min(Ilast,I+Id)
    for J in Ilower:Iupper
      if Y[J] != Y[I]
        B[I] = true
      end
    end
  end
  return B
end
B = 2*Int.(find_boundary(reshape(xsoln .> 0, size(A)...)))
B[seed] = 5
myplot(vec(B),A,seed)
savefig("figures/boundary-seed-outline.png")
##
z = diffuse_vals(kappas,
  kappa ->  PageRank.acl_diffusion(SLQ.graph(G), [seed], 0.001, kappa; maxpush=100000))
##
myplot(z,A,seed)
##
savefig("figures/boundary-acl-0001-100k.png")
##
z = diffuse_vals(kappas,
  kappa ->  PageRank.acl_diffusion(SLQ.graph(G), [seed], 0.001, kappa; maxpush=1000000))

##
myplot(z,A,seed)
##
savefig("figures/boundary-acl-0001-1000k.png")
##
##
z = diffuse_vals(kappas,
  kappa ->  PageRank.acl_diffusion(SLQ.graph(G), [seed], 0.001, kappa))

##
myplot(z,A,seed)
##
savefig("figures/boundary-acl-0001-full.png")

##
@time w = diffuse_vals(kappas,
  kappa ->  SLQ.slq_diffusion(SLQ.graph(G), [seed], 0.001, kappa, 0.5, SLQ.loss_type(1.1,0.0);
    max_iters=100000)[1])
##
myplot(w,A,seed)
##
savefig("figures/boundary-slq-11-0001-100k.png")

##
@time w = diffuse_vals(kappas,
  kappa ->  SLQ.slq_diffusion(SLQ.graph(G), [seed], 0.001, kappa, 0.5, SLQ.loss_type(1.1,0.0);
    max_iters=1000000)[1])
##
myplot(w,A,seed)
##
savefig("figures/boundary-slq-11-0001-1000k.png")

##
@time w = diffuse_vals(kappas,
  kappa ->  SLQ.slq_diffusion(SLQ.graph(G), [seed], 0.001, kappa, 0.5, SLQ.loss_type(1.1,0.0);
    max_iters=1000000000)[1])
## Eek, this still take 2 hours to run and still isn't done.   I'll have
# to run it over night!
##
myplot(w,A,seed)
##
savefig("figures/boundary-slq-11-0001-full.png")




## Here are some other figures.

##
@time w = diffuse_vals(kappas,
  kappa ->  SLQ.slq_diffusion(SLQ.graph(G), [seed], 0.001, kappa, 0.5, SLQ.loss_type(1.2,0.0);
    max_iters=100000000)[1])
##
myplot(w,A,seed)
##
savefig("figures/boundary-slq-12-0001-full.png")

## Extra test with simplelocal
include("FlowSeed-1.0.jl")
function simplelocal(A::SparseMatrixCSC,S,delta::Real)
  inSc = ones(size(A,1))
  inSc[S] .= 0
  Sc = setdiff(1:size(A,1),S)
  RinS = zeros(length(S))
  pS = zeros(length(S))
  cluster,cond = FlowSeed(A,S,delta,pS,RinS)
end
S = [seed]
cluster,cond = simplelocal(G,S,0.0)
## Okay, this verifies that we are unable to grow this seed, good!
