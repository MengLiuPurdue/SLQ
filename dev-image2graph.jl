## Develop the image2graph function based on what was in the
# flow-based clustering paper with Meng, Kimon, etc.
# and then develop the experiemtns we needed.
# This has now been finalized in experiment-image-boundary.jl
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

##
using TestImages
@time A = testimage("house")
@time G,imap = image2graph(A, 80, 0.001, maximum(size(A))/10)
seed = imap[300,243]
using MatrixNetworks
@time x = seeded_pagerank(G,0.95,seed)

## Try two, we want to look at these diffusions at different sparsity levels.
include("PageRank.jl")
include("SLQ.jl")
##
x = PageRank.acl_diffusion(SLQ.graph(G), [seed], 0.01, 0.01)

##
@time y = SLQ.slq_diffusion(SLQ.graph(G), [seed], 0.01, 0.01, 0.5,
      SLQ.loss_type(1.25,0.0))[1]

##
using Plots, Measures
function shownzs(x,A;color=colorant"orange")
  X = RGBA.(RGB.(reshape(x .> 0,size(A)...)))
  V = channelview(X)
  V[4,:,:] .= @view V[1,:,:] # copy the alpha value from the binary value
  (@view V[1,:,:]) .*= red(color) # change the color
  (@view V[2,:,:]) .*= green(color) # change the color
  (@view V[3,:,:]) .*= blue(color) # change the color
  return X
end

heatmap(A, framestyle=:none, margin=-20mm)
heatmap!(shownzs(x,A))
heatmap!(shownzs(y,A;color=colorant"green"))
##
heatmap(A, framestyle=:none, margin=-20mm)
for (i,kappa)=enumerate(reverse([0.001,0.0005,0.0001]))
  x = PageRank.acl_diffusion(SLQ.graph(G), [seed], 0.01, kappa)
  heatmap!(shownzs(x,A;color=palette(:default)[i]))
end
plot!()
## Here's a better try at that plot
function diffuse_vals(kappas, F)
  x = F(first(kappas))
  x[x .> 0] .= 1
  for i = 2:length(kappas)
    y = F(kappas[i])
    x[y .> 0] .= i
  end
  return x
end
z = diffuse_vals(reverse([0.005,0.0025,0.001,0.0005]),
  kappa ->  PageRank.acl_diffusion(SLQ.graph(G), [seed], 0.01, kappa))
##
RGB.(reshape(z, size(A)...))
#heatmap(A, framestyle=:none, margin=-20mm)
#heatmap(shownsz())
## Needs colors!
# https://discourse.julialang.org/t/how-to-convert-a-matrix-to-an-rgb-image-using-images-jl/7265/7
using MappedArrays, IndirectArrays

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
heatmap(A, framestyle=:none, margin=-20mm)
mygrad = cgrad(:magma)
mygrad = mygrad[128:end]
heatmap!(color_me(reshape(z, size(A)...), mygrad))
##
w = diffuse_vals(reverse([0.005,0.0025,0.001]),
  kappa ->  SLQ.slq_diffusion(SLQ.graph(G), [seed], 0.01, kappa, 0.5, SLQ.loss_type(1.4,0.0);
    max_iters=100000)[1])
##
heatmap(A, framestyle=:none, margin=-20mm)
mygrad = cgrad(:magma)
mygrad = mygrad[128:end]
heatmap!(color_me(reshape(w, size(A)...), mygrad))

##
using TestImages
@time A = testimage("house")
@time G,imap = image2graph(A, 80, 0.001, maximum(size(A))/10)
using MatrixNetworks
x = seeded_pagerank(G,0.95,imap[300,243])

## newest
# using
# https://discourse.julialang.org/t/how-to-convert-a-matrix-to-an-rgb-image-using-images-jl/7265/7
using MappedArrays, IndirectArrays
function color_me(A, cmap)
    # remake cmap with alpha values
    n = length(cmap)
    # add alpha values to cmap
    cmap = map( i -> RGBA(cmap[i].r, cmap[i].g, cmap[i].b, i/n), 1:n)
    Amin,Amax = extrema(A)
    f = s->clamp(round(Int, (n-1)*((s-Amin)/(Amax-Amin)))+1, 1, n)  # safely convert 0-1 to 1:n
    Ai = mappedarray(f, A)       # like f.(A) but does not allocate significant memory
    X = IndirectArray(Ai, cmap)      # colormap array
end

(A .+ color_me(reshape(log10.(x), size(A)...), cgrad(:thermal)))
## Try contour instead
using Statistics, Measures
function mycontour!(Z;nlevels=10,threshhold=1e-8,kwargs...)
    y = 1:size(Z,1)
    x = 1:size(Z,1)
    z = vec(Z')
    nzset = z .> threshhold
    #levels = quantile(z[nzset], range(0, 1-1/nlevels, length=nlevels))
    zmin,zmax = extrema(z[nzset])
    levels = 10.0.^(range(log10(zmin), log10(zmax), length=nlevels))
    @show levels
    #pushfirst!(levels, 0.0)
      contour!(x,y,z;levels=levels,colorbar=false,kwargs...)
end
plot()
heatmap!(A, framestyle=:none, margin=-20mm)
mycontour!(reshape((x), size(A)...); nlevels=3, threshhold=1e-8, color=:orange)

## Compare to SLQ
include("SLQ.jl")
##
@time x = SLQ.slq_diffusion(SLQ.graph(G),[seed],
  0.01,0.001,0.5,SLQ.loss_type(2.0,0.0); max_iters=100000)[1]

plot()
heatmap!(A, framestyle=:none, margin=-20mm)
mycontour!(reshape((x), size(A)...); nlevels=3, threshhold=0, color=:orange)

##
color_me(reshape((x).^(1/10), size(A)...), cgrad(:thermal))
## Compare to ACL
include("PageRank.jl")
@time x = PageRank.acl_diffusion(SLQ.graph(G),[seed],0.01,0.001)
color_me(reshape((x).^(1/10), size(A)...), cgrad(:thermal))
##
include("SLQ.jl")
@time x = SLQ.slq_diffusion(SLQ.graph(G),[seed],
  0.01,0.0001,0.5,SLQ.loss_type(1.5,0.01); max_iters=100000)[1]


#plot()
#heatmap!(A, framestyle=:none, margin=-20mm)
#mycontour!(reshape((x), size(A)...); nlevels=3, threshhold=0, color=:orange)
## Not the best figiure.
color_me(reshape((x).^(1/10), size(A)...), cgrad(:thermal))

## older, using an
# now using an alternative from
# https://discourse.julialang.org/t/how-to-convert-a-matrix-to-an-rgb-image-using-images-jl/7265/7
using PerceptualColourMaps
function show_vectorc(x,A)
  c1 = cmap("L3")

  c = similar(c1,0)
  for i=1:length(c1)
    push!(c, RGBA(c3[i].r, c3[i].g, c3[i].b, c1[i].r))
  end
  X = applycolormap(Matrix(reshape(log10.(x),size(A)...)), c)
  @show size(X)
  return 0.9.*colorview(RGB, permutedims(X, (3, 1, 2))) + 0.1.*A
end
show_vectorc(x,A)
## Older
using Plots
function show_vector(x,A)
  X = sqrt.(reshape(x, size(A)))
  X = X./maximum(X)
  return 20.0*oneunit(A[1])*X .+ 0.5.*A
end
show_vector(x,A)
