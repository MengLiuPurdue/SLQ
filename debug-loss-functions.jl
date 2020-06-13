using Plots
# just debug
include("SLQ.jl")
##
# show the plot of the loss and the gradient
L = SLQ.loss_type(1.2, 0.3)
xx = range(-3, 3; length=1000)
plot(xx, map(x -> SLQ.loss_function(x, L), xx))
plot!(xx, map(x -> SLQ.loss_gradient(x, L), xx))
