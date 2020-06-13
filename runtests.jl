using Test
include("PageRank.jl")
include("SLQcvx.jl") # this includes SLQ.jl
include("common.jl")
# TODO, add more cases
@testset "cvx-vs-slq" begin
  rho = 0.999999
  delta = 0.0
  maxiter = 1000000
  for q in [2.0,1.5,2.5]
    for gamma in [1.0,0.5,0.1,1.5]
      for kappa in [0.05, 0.5]
        @testset "Testing q=$q, gamma=$gamma, kappa=$kappa" begin
          if q == 1.2
            maxiter = maxiter*100
          end
          L = SLQ.QHuberLoss(q, delta)

          A,xy = two_cliques(20,20)
          G = SLQ.graph(A)
          S = [1,2]
          x = SLQcvx.slq_cvx(G, S, q, gamma, kappa, solver="ECOS")[1]
          (y,r,iter) = SLQ.slq_diffusion(G, S, gamma, kappa, rho, L;
              max_iters=maxiter,epsilon=1.0e-8)
          @test iter < maxiter
          @test maximum(abs.(x-y)) <= 1.0e-4
          @test maximum(r - kappa*G.deg) <= 0

          A,xy = grid_graph(20,20)
          G = SLQ.graph(A)
          S = [10*20+10,10*20+11]
          x = SLQcvx.slq_cvx(G, S, q, gamma, kappa, solver="ECOS")[1]
          (y,r,iter) = SLQ.slq_diffusion(G, S, gamma, kappa, rho, L;
              max_iters=maxiter,epsilon=1.0e-8)
          @test iter < maxiter
          @test maximum(abs.(x-y)) <= 1.0e-4
          @test maximum(r - kappa*G.deg) <= 0
        end
      end
    end
  end
end
