using LightGraphs
using DelimitedFiles
using SparseArrays
using MAT
using Random
using JLD2,FileIO
using PyCall
using PyPlot
using MatrixNetworks

include("PageRank.jl")
include("SLQcvx.jl") # this includes SLQ.jl
include("common.jl")
include("CRDlgc.jl")
include("HeatKernel.jl")
include("FlowSeed-1.0.jl")
include("GCN.jl")

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


function eval_crd_facebook(datasets,ntrials;U=3,h=3,w=2,eval_labels=[2008.0])
    start_id = 0
    records = Dict()
    for file in datasets
        records[file] = Dict()
        vars = matread(file)
        A = vars["A"]
        A = round.(Int,A)
        A,p = largest_component(A)
        G = SLQ.graph(A)
        labels = vars["local_info"]
        for label in eval_labels
            records[file][label] = []
            truth = findall(labels[p,6] .== label)
            n = length(truth)
            for seed in start_id:(start_id+ntrials-1)
                S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.01*n))]]
                cluster_crd,cond_crd,time_crd = CRDlgc.crd(G,S,U=U,h=h,w=w)
                pr_crd,rc_crd = compute_pr_rc(cluster_crd,truth)
                @show file,label,[seed,cond_crd,pr_crd,rc_crd],2*pr_crd*rc_crd/(pr_crd+rc_crd)
                push!(records[file][label],[seed,cond_crd,pr_crd,rc_crd])
            end
            start_id += ntrials
        end
    end
    return records
end

function eval_acl_facebook(datasets,ntrials;eval_labels = [2008.0,2009.0])
    start_id = 0
    records = Dict()
    for file in datasets
        records[file] = Dict()
        vars = matread(file)
        A = vars["A"]
        A = round.(Int,A)
        A,p = largest_component(A)
        G = SLQ.graph(A)
        labels = vars["local_info"]
        for label in eval_labels
            records[file][label] = []
            truth = findall(labels[p,6] .== label)
            n = length(truth)
            for seed in start_id:(start_id+ntrials-1)
                S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.01*n))]]
                x_acl = PageRank.degnorm_acl_diffusion(G,S,0.05,0.005)
                cluster_acl,cond_acl = PageRank.round_to_cluster(G,x_acl)
                pr_acl,rc_acl = compute_pr_rc(cluster_acl,truth)
                push!(records[file][label],[seed,cond_acl,pr_acl,rc_acl])
                @show file,label,[seed,cond_acl,pr_acl,rc_acl]
            end
            start_id += ntrials
        end
    end
    return records
end

function eval_slq_facebook(datasets,ntrials;delta=0.0,eval_labels=[2008.0],q=1.2)
    start_id = 0
    records = Dict()
    for file in datasets
        records[file] = Dict()
        vars = matread(file)
        A = vars["A"]
        A = round.(Int,A)
        A,p = largest_component(A)
        G = SLQ.graph(A)
        labels = vars["local_info"]
        for label in eval_labels
            records[file][label] = []
            truth = findall(labels[p,6] .== label)
            n = length(truth)
            for seed in start_id:(start_id+ntrials-1)
                S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.01*n))]]
                L = SLQ.QHuberLoss(q, delta)
                (x_slq,r,iter) = SLQ.slq_diffusion(G, S, 0.05, 0.005, 0.5, L, max_iters=100000,epsilon=1.0e-8)
                if iter < 100000
                    cluster_slq,cond_slq = PageRank.round_to_cluster(G,x_slq)
                    pr_slq,rc_slq = compute_pr_rc(cluster_slq,truth)
                    push!(records[file][label],[seed,cond_slq,pr_slq,rc_slq])
                    @show file,label,q,[seed,cond_slq,pr_slq,rc_slq],2*pr_slq*rc_slq/(pr_slq+rc_slq)
                else
                    cluster_slq,cond_slq = PageRank.round_to_cluster(G,x_slq)
                    pr_slq,rc_slq = compute_pr_rc(cluster_slq,truth)
                    cond_slq = 1.0
                    push!(records[file][label],[seed,cond_slq,pr_slq,rc_slq])
                    @show file,label,q,[seed,cond_slq,pr_slq,rc_slq],2*pr_slq*rc_slq/(pr_slq+rc_slq)
                end
            end
            start_id += ntrials
        end
    end
    return records
end

function eval_nonlinear_diffusion_facebook(datasets,ntrials;q=1.5,eval_labels = [2008.0,2009.0])
    start_id = 0
    records = Dict()
    for file in datasets
        records[file] = Dict()
        vars = matread(file)
        A = vars["A"]
        A = round.(Int,A)
        A,p = largest_component(A)
        G = SLQ.graph(A)
        labels = vars["local_info"]
        for label in eval_labels
            records[file][label] = []
            truth = findall(labels[p,6] .== label)
            n = length(truth)
            for seed in start_id:(start_id+ntrials-1)
                S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.01*n))]]
                x = nonlinear_diffusion(A, 0.002, 10000, sparsevec(S,1.0,size(A,1)), q)
                cluster,cond = PageRank.round_to_cluster(G,x)
                pr,rc = compute_pr_rc(cluster,truth)
                push!(records[file][label],[seed,cond,pr,rc])
                @show file,label,[seed,cond,pr,rc]
            end
            start_id += ntrials
        end
    end
    return records
end

function eval_nonlinear_Lp_diffusion_facebook(datasets,ntrials)
    start_id = 0
    records = Dict()
    for file in datasets
        records[file] = Dict()
        vars = matread(file)
        A = vars["A"]
        A = round.(Int,A)
        A,p = largest_component(A)
        G = SLQ.graph(A)
        labels = vars["local_info"]
        eval_labels = [2008.0,2009.0]
        for label in eval_labels
            records[file][label] = []
            truth = findall(labels[p,6] .== label)
            n = length(truth)
            for seed in start_id:(start_id+ntrials-1)
                S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.01*n))]]
                x = nonlinear_Lp_diffusion(A, 0.002, 5000, sparsevec(S,1.0,size(A,1)), 1.5)
                cluster,cond = PageRank.round_to_cluster(G,x)
                pr,rc = compute_pr_rc(cluster,truth)
                push!(records[file][label],[seed,cond,pr,rc])
                @show file,label,[seed,cond,pr,rc]
            end
            start_id += ntrials
        end
    end
    return records
end

function eval_hk_facebook(datasets,ntrials;eval_labels=[2008.0,2009.0])
    start_id = 0
    records = Dict()
    for file in datasets
        records[file] = Dict()
        vars = matread(file)
        A = vars["A"]
        A = round.(Int,A)
        A,p = largest_component(A)
        G = SLQ.graph(A)
        labels = vars["local_info"]
        for label in eval_labels
            records[file][label] = []
            truth = findall(labels[p,6] .== label)
            n = length(truth)
            for seed in start_id:(start_id+ntrials-1)
                S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.01*n))]]
                cluster,cond,time = HeatKernel.hk_grow(A,S)
                pr,rc = compute_pr_rc(cluster,truth)
                @show file,label,[seed,cond,pr,rc],2*pr*rc/(pr+rc)
                push!(records[file][label],[seed,cond,pr,rc])
            end
            start_id += ntrials
        end
    end
    return records
end


function eval_sl_facebook(datasets,ntrials,epsilon;eval_labels=[2008.0,2009.0])
    start_id = 0
    records = Dict()
    for file in datasets
        records[file] = Dict()
        vars = matread(file)
        A = vars["A"]
        A = round.(Int,A)
        A,p = largest_component(A)
        A = A.*1.0
        G = SLQ.graph(A)
        labels = vars["local_info"]
        for label in eval_labels
            records[file][label] = []
            truth = findall(labels[p,6] .== label)
            n = length(truth)
            for seed in start_id:(start_id+ntrials-1)
                Srand = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.01*n))]]
                S = neighborhood(A,Srand,1) 
                Sn = setdiff(S,Srand)
                inSc = ones(size(A,1))
                inSc[S] .= 0
                Sc = setdiff(1:size(A,1),S)
                RinS = zeros(length(S))
                for r = 1:length(S)
                    if in(S[r],Srand)
                        RinS[r] = 1
                    end
                end
                volS = sum(G.deg[S])
                volA = sum(G.deg)
                pS = zeros(length(S))
                cluster,cond = FlowSeed(A,S,epsilon,pS,RinS)
                pr,rc = compute_pr_rc(cluster,truth)
                @show file,label,[seed,cond,pr,rc],2*pr*rc/(pr+rc)
                push!(records[file][label],[seed,cond,pr,rc])
            end
            start_id += ntrials
        end
    end
    return records
end



function eval_gcn_facebook(datasets,ntrials;eval_labels=[2008.0,2009.0])
    start_id = 0
    records = Dict()
    for file in datasets
        records[file] = Dict()
        vars = matread(file)
        A = vars["A"]
        A = round.(Int,A)
        A,p = largest_component(A)
        G = SLQ.graph(A)
        labels = vars["local_info"][p,:]
        for label in eval_labels
            records[file][label] = []
            truth = findall(labels[:,6] .== label)
            n = length(truth)
            for seed in start_id:(start_id+ntrials-1)
                S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.01*n))]]
                nontruth = setdiff(1:size(A,1),truth)
                nonS = nontruth[randperm(MersenneTwister(seed),length(nontruth))[1:max(1,round(Int,0.01*n))]]
                features = copy(labels)
                class_labels = zeros(size(A,1))
                class_labels[truth] .= 1
                cluster,cond,time = GCN.gcn(G,S,nonS,features,class_labels)
                pr,rc = compute_pr_rc(cluster,truth)
                @show file,label,[seed,cond,pr,rc],2*pr*rc/(pr+rc)
                push!(records[file][label],[seed,cond,pr,rc])
            end
            start_id += ntrials
        end
    end
    return records
end



ntrials = 50
datasets = ["UCLA26.mat","Brown11.mat","Duke14.mat","UPenn7.mat","Yale4.mat","Stanford3.mat","MIT8.mat","Cornell5.mat"]
#datasets = ["Colgate88.mat","Johns Hopkins55.mat","Yale4.mat","Stanford3.mat"]

time_crd_5 = @elapsed records_crd_5 = eval_crd_facebook(datasets,ntrials,h=5,eval_labels=[2008.0,2009.0])
save("results/records-crd-h-5.jld2",records_crd_5)

time_crd_3 = @elapsed records_crd_3 = eval_crd_facebook(datasets,ntrials,h=3,eval_labels=[2008.0,2009.0])
save("results/records-crd-h-3.jld2",records_crd_3)

time_acl = @elapsed records_acl = eval_acl_facebook(datasets,ntrials)
save("results/records-acl.jld2",records_acl)

time_slq_0 = @elapsed records_slq_0 = eval_slq_facebook(datasets,ntrials,delta=0.0,q=1.2,eval_labels=[2008.0,2009.0])
save("results/records-slq-0.jld2",records_slq_0)

time_slq_1 = @elapsed records_slq_1 = eval_slq_facebook(datasets,ntrials,delta=1.0e-5,q=1.2,eval_labels=[2008.0,2009.0])
save("results/records-slq-1.jld2",records_slq_1)

records_nonlinear_diffusion = eval_nonlinear_diffusion_facebook(datasets,ntrials)
save("results/records_nonlinear_diffusion-1.5.jld2",records_nonlinear_diffusion)

records_nonlinear_diffusion = eval_nonlinear_diffusion_facebook(datasets,ntrials,q=1.2)
save("results/records_nonlinear_diffusion-1.2.jld2",records_nonlinear_diffusion)

time_nonlinear_diffusion = @elapsed records_nonlinear_diffusion = eval_nonlinear_diffusion_facebook(datasets,ntrials,q=0.5)
save("results/records-nld.jld2",records_nonlinear_diffusion)

records_nonlinear_Lp_diffusion = eval_nonlinear_Lp_diffusion_facebook(datasets,ntrials)
save("results/records_nonlinear_Lp_diffusion.jld2",records_nonlinear_Lp_diffusion)

time_hk = @elapsed records_hk = eval_hk_facebook(datasets,ntrials)
save("results/records-hk.jld2",records_hk)

time_sl_5 = @elapsed records_sl_5 = eval_sl_facebook(datasets,ntrials,0.5)
save("results/records-sl-5.jld2",records_sl_5)

time_sl_4 = @elapsed records_sl_4 = eval_sl_facebook(datasets,ntrials,0.4)
save("results/records_sl_4.jld2",records_sl_4)

time_sl_3 = @elapsed records_sl_3 = eval_sl_facebook(datasets,ntrials,0.3)
save("results/records_sl_3.jld2",records_sl_3)

time_gcn = @elapsed records_gcn = eval_gcn_facebook(datasets,ntrials)
save("results/records-gcn.jld2",records_gcn)


include("GCN.jl")
file = "UCLA26.mat"
vars = matread(file)
A = vars["A"]
A = round.(Int,A)
A,p = largest_component(A)
G = SLQ.graph(A)
labels = vars["local_info"][p,:]
label = 2008
truth = findall(labels[:,6] .== label)
n = length(truth)
seed = 6
S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.01*n))]]
nontruth = setdiff(1:size(A,1),truth)
nonS = nontruth[randperm(MersenneTwister(seed),length(nontruth))[1:max(1,round(Int,0.01*n))]]
features = labels
labels = zeros(size(A,1))
labels[truth] .= 1
cluster,cond,time = GCN.gcn(G,S,nonS,features,labels)
