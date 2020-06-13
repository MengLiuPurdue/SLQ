## Helpful figure
using Plots, Measures
# just debug
include("SLQ.jl")
##
# show the plot of the loss and the gradient
L = SLQ.loss_type(1.2, 0.3)
xx = range(-1, 1; length=1000)
plot(xx, map(x -> SLQ.loss_function(x, L), xx), framestyle=:zerolines, ticks=nothing, label="",
  size=(75,75),dpi=600, linewidth=2.0, thickness_scaling = 0.5)
##
function berq(x::Real, q::Real, delta::Real)
  if abs(x) < delta
      return (1/q)*delta^(2-q)*abs(x)^q
  else
      return (1/2)*x^2 + ((1/q) - 1/2)*delta^2
  end

end
delta = 0.3
myloss(x) = berq(x, 1.2, delta)
xx = range(-1, 1; length=1000)
plot(xx, map(x -> myloss(x), xx), framestyle=:zerolines, ticks=nothing, label="",
  size=(75,75),dpi=600, linewidth=2.0, margin=-20mm)
scatter!([-delta, delta], [myloss(-delta), myloss(delta)], label="", markerstrokewidth=0, color=:black,
  markersize=2, marker=:circle)
##
savefig("figures/loss-berq-12-03.pdf")
##
delta = 0.3
myloss(x) = berq(x, 1.2, delta)
xx = range(-1, 1; length=1000)
xx1 = range(-delta, delta; length=100)
plot(xx, map(x -> myloss(x), xx), framestyle=:zerolines, ticks=nothing, label="",
  size=(75,75),dpi=600, linewidth=2.0, margin=-20mm)
plot!(xx1, map(x -> myloss(x), xx1), framestyle=:zerolines, ticks=nothing, label="",
  size=(75,75),dpi=600, linewidth=2.0, margin=-20mm)
scatter!([-delta, delta], [myloss(-delta), myloss(delta)], label="", markerstrokewidth=0, color=:black,
  markersize=2, marker=:circle)
##
savefig("figures/loss-berq-12-03-colors.pdf")  
##
delta = 0.3
L = SLQ.loss_type(1.2, delta)
myloss(x) = SLQ.loss_function(x, L)
plot(xx, map(x -> myloss(x), xx), framestyle=:zerolines, ticks=nothing, label="",
  size=(75,75),dpi=600, linewidth=2.0, margin=-20mm)
scatter!([-delta, delta], [myloss(-delta), myloss(delta)], label="", markerstrokewidth=0, color=:black,
    markersize=2, marker=:circle)
##
savefig("figures/loss-huber-12-03.pdf")

##
delta = 0.3
L = SLQ.loss_type(1.2, delta)
myloss(x) = SLQ.loss_function(x, L)
plot(xx, map(x -> myloss(x), xx), framestyle=:zerolines, ticks=nothing, label="",
  size=(75,75),dpi=600, linewidth=2.0, margin=-20mm)
plot!(xx1, map(x -> myloss(x), xx1), framestyle=:zerolines, ticks=nothing, label="",
    size=(75,75),dpi=600, linewidth=2.0, margin=-20mm)

scatter!([-delta, delta], [myloss(-delta), myloss(delta)], label="", markerstrokewidth=0, color=:black,
    markersize=2, marker=:circle)

##
savefig("figures/loss-huber-12-03-colors.pdf")

## Show color information
@show palette(:default)[1]
@show palette(:default)[2]
