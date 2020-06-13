## Check the runtime of cvx sparse vs. cvx dense
include("SLQcvx.jl")
for j=[10,25,50,75]
  A = grid_graph(j,j)[1]
  x, dts = SLQcvx.slq_cvx(SLQ.graph(A), [1],  1.5, 0.1, 0.1)
  y, dtd = SLQcvx.slq_cvx_dense(SLQ.graph(A), [1], 1.5, 0.1, 0.1)
  @show j^2, dts, dtd
  @assert x ≈ y
end
# this shows the sparse routine is about 2x faster...
