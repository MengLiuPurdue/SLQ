using JLD2,FileIO
using Distributed
addprocs(100);

@everywhere using LightGraphs
@everywhere cd("/homes/liu1740/Research/spectral_flow/SLQ/")

@everywhere include("PageRank.jl")
@everywhere include("SLQcvx.jl") # this includes SLQ.jl
@everywhere include("common.jl")
@everywhere include("CRDlgc.jl")

@everywhere function sparsity_runtime_worker(gamma,rho,kappa,pout,ncls,use_cvx,jobs,results)
    while true
        q,delta,n,seed,pin = take!(jobs)
        if q == -1
            break
        end
        A = sbm(seed,pin,pout,ncls,n)
        G = SLQ.graph(A)
        S = collect(1:round(Int,0.01*n))
        L = SLQ.QHuberLoss(q, delta)
        time_slq = @elapsed (x,r,iter) = SLQ.slq_diffusion(G, S, gamma, kappa, rho, L, max_iters=100000,epsilon=1.0e-8)
        nonzeros_slq = sum(abs.(x.>0))
        obj_slq = SLQ.objective(G,S,x,kappa,gamma,L)
        time_acl = @elapsed x_acl = PageRank.acl_diffusion(G, S, gamma, kappa)
        nonzeros_acl = sum(abs.(x_acl.>0))
        if use_cvx
            time_cvx = @elapsed x_cvx = SLQcvx.slq_cvx(G, S, q, gamma, kappa, solver="ECOS")[1]
            obj_cvx = SLQ.objective(G,S,x_cvx,kappa,gamma,L)
        else
            (x_cvx,time_cvx,obj_cvx) = (-1,-1,-1)
        end
        put!(results, (String(join((q,delta,Int(n),pin),",")),[time_slq,time_cvx,obj_slq,obj_cvx,1.0*nonzeros_slq,time_acl,nonzeros_acl*1.0]))
    end
end

@everywhere function sparsity_runtime_worker_crd(pout,ncls,jobs,results)
    while true
        n,seed,pin = take!(jobs)
        if n == -1
            break
        end
        A = sbm(seed,pin,pout,ncls,n)
        G = SLQ.graph(A)
        S = collect(1:round(Int,0.01*n))
        cluster,cond_crd,time_crd = CRDlgc.crd(G,S)
        nonzeros_crd = length(cluster)
        put!(results, (String(join((pin,Int(n)),",")),[time_crd,nonzeros_crd*1.0]))
    end
end

function make_jobs(jobs,q_list,n_list,delta_list,ntrials,pin_list)
    seed = 1
    records = Dict()
    for q in q_list
        for n in n_list
            for delta in delta_list
                for pin in pin_list
                    records[String(join((q,1.0*delta,Int(n),pin),","))] = []
                    for i = 1:ntrials
                        put!(jobs,(q,delta,n,seed,pin))
                        seed += 1
                    end
                end
            end
        end
    end
    for i = 1:length(workers())
        put!(jobs,(-1,-1,-1,-1,-1))
    end
    return records
end

function make_jobs_crd(jobs,n_list,ntrials,pin_list)
    seed = 1
    records = Dict()
    for pin in pin_list
        for n in n_list
            records[String(join((pin,Int(n)),","))] = []
            for i = 1:ntrials
                put!(jobs,(n,seed,pin))
                seed += 1
            end
        end
    end
    for i = 1:length(workers())
        put!(jobs,(-1,-1,-1))
    end
    return records
end

function sparsity_runtime_parallel(q_list,n_list,delta_list,ncls,gamma,rho,kappa,ntrials,pin_list,pout;use_cvx=true)
    nexps = length(q_list)*length(n_list)*length(delta_list)*length(pin_list)*ntrials
    jobs = RemoteChannel(()->Channel{Tuple{Float64,Float64,Int64,Int64,Float64}}(nexps+length(workers())))
    results = RemoteChannel(()->Channel{Tuple{String,Array{Float64,1}}}(nexps))
    records = make_jobs(jobs,q_list,n_list,delta_list,ntrials,pin_list)
    #make_jobs(jobs,q_list,n_list,delta_list,ntrials,records)
    for p in workers()
        remote_do(sparsity_runtime_worker,p,gamma,rho,kappa,pout,ncls,use_cvx,jobs,results)
    end
    while nexps > 0 # wait for all jobs to finish
       input,output = take!(results)
       push!(records[input],output)
       nexps = nexps - 1
       println("$nexps jobs left.")
    end
    return records
end

function sparsity_runtime_parallel_crd(n_list,ncls,ntrials,pin_list,pout)
    nexps = length(pin_list)*length(n_list)*ntrials
    jobs = RemoteChannel(()->Channel{Tuple{Int64,Int64,Float64}}(nexps+length(workers())))
    results = RemoteChannel(()->Channel{Tuple{String,Array{Float64,1}}}(nexps))
    records = make_jobs_crd(jobs,n_list,ntrials,pin_list)
    #make_jobs(jobs,q_list,n_list,delta_list,ntrials,records)
    for p in workers()
        remote_do(sparsity_runtime_worker_crd,p,pout,ncls,jobs,results)
    end
    while nexps > 0 # wait for all jobs to finish
       input,output = take!(results)
       push!(records[input],output)
       nexps = nexps - 1
       println("$nexps jobs left.")
    end
    return records
end

q_list = [1.6,1.4,1.2]
delta_list = [0.0,1.0e-5,1.0e-4]
pin_list = [0.1,0.08,0.06]
pout = 0.01
ncls = 5
rho = 0.5
kappa = 0.005
ntrials = 20
gamma = 0.1

n_list= [500,600,700,800,900,1000,2000,3000,4000,6000,8000,10000]
records_slq = sparsity_runtime_parallel(q_list,n_list,delta_list,ncls,gamma,rho,kappa,ntrials,pin_list,pout,use_cvx=false)

records_crd = sparsity_runtime_parallel_crd(n_list,ncls,ntrials,pin_list,pout)

records = merge(records_slq,records_crd)

save("sparsity_runtime.jld2",records)

using Statistics
using PyCall
@pyimport matplotlib.pyplot as plt
fig, plt_axes = plt.subplots(1,3,figsize=(18,3))
for (k,ax) = enumerate(plt_axes)
    tmp = Dict()
    # tmp["cvx"] = Dict()
    tmp["slq"] = Dict()
    tmp["crd"] = Dict()
    tmp["acl"] = Dict()

    pin = pin_list[k]
    for n in n_list
        # tmp["cvx"][n] = []
        tmp["slq"][n] = []
        tmp["acl"][n] = []
        for trial in records[join((1.4,0.0,n,pin),",")]
            time_slq,time_cvx,obj_slq,obj_cvx,nonzeros_slq,time_acl,nonzeros_acl = trial
            # push!(tmp["cvx"][n],time_cvx)
            push!(tmp["slq"][n],time_slq)
            push!(tmp["acl"][n],time_acl)
        end
    end

    for n in n_list
        tmp["crd"][n] = []
        for trial in records_crd[join((pin,n),",")]
            time_crd,nonzeros_crd = trial
            # push!(tmp["cvx"][n],time_cvx)
            push!(tmp["crd"][n],time_crd)
        end
    end
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    i = 1
    for key in keys(tmp)
        times = []
        error_20 = []
        error_80 = []
        for n in n_list
            push!(times,mean(tmp[key][n]))
            push!(error_20,quantile(tmp[key][n],0.2))
            push!(error_80,quantile(tmp[key][n],0.8))
        end
        ax.plot(n_list,times,color=colors[i])
        ax.fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[i])
        i += 1
    end

    ax.set_xlabel("cluster size",fontsize=15)
    if k == 1
        ax.set_ylabel("seconds",fontsize=15)
    end
    if k == 2
        ax.legend(["ACL","CRD", "SLQ"],fancybox=true,shadow=true,loc=(0.0,1.2),ncol=3,fontsize=15)
    end
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_yscale("log")
    for tick in ax.xaxis.get_major_ticks()
        tick.label.set_fontsize(15)
    end
    for tick in ax.yaxis.get_major_ticks()
        tick.label.set_fontsize(15)
    end
    ax.tick_params(axis="x", which="major", length=7, width=1)
    ax.tick_params(axis="y", which="major", length=7, width=1)
    ax.tick_params(axis="y", which="minor", length=3, width=1)
    ax.set_ylim((1.0e-4,1.0e3))
    if k != 1
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="both", which="both", length=0)
    end
    t = @sprintf "in probability is: %.2f" pin
    ax.set_title(t, fontsize=15)
end
fig.savefig("running_time.pdf",format="pdf",bbox_inches="tight")

using Printf

@everywhere function pr_rc(cluster,truth)
    pr = 1-length(setdiff(cluster,Set(truth)))/length(cluster)
    rc = 1-length(setdiff(Set(truth),cluster))/length(truth)
    return pr,rc
end

@everywhere function pr_rc_cond_worker(gamma,rho,kappa,pout,ncls,use_crd,jobs,results)
    while true
        delta,n,seed,pin = take!(jobs)
        if n == -1
            break
        end
        A = sbm(seed,pin,pout,ncls,n)
        G = SLQ.graph(A)
        S = collect(1:round(Int,0.01*n))
        bestset = []
        bestcond = 1.0
        q_best = 0
        for q_test in [1.2,1.3,1.4,1.5,1.6]
            L = SLQ.QHuberLoss(q_test, delta)
            time_slq = @elapsed (x,r,iter) = SLQ.slq_diffusion(G, S, gamma, kappa, rho, L, max_iters=100000,epsilon=1.0e-8)
            nnz_dict = Dict{Int,Float64}()
            for (i,v) in enumerate(x)
                if v > 0
                    nnz_dict[i] = v
                end
            end
            currset, currcond, currstats = PageRank.weighted_local_sweep_cut(A, nnz_dict, G.deg, sum(G.deg))
            if currcond < bestcond
                bestcond = currcond
                bestset = currset
                q_best = q_test
            end
        end
        cond_slq = bestcond
        pr_slq,rc_slq = pr_rc(bestset,1:n)
        if use_crd
            cluster,cond_crd,time_crd = CRDlgc.crd(G,S)
            pr_crd,rc_crd = pr_rc(Array{Int,1}(cluster),1:n)
        else
            (cond_crd,pr_crd,rc_crd) = (-1,-1,-1)
        end
        x_acl = PageRank.acl_diffusion(G, S, gamma, kappa)
        nnz_dict = Dict{Int,Float64}()
        for (i,v) in enumerate(x_acl)
            if v > 0
                nnz_dict[i] = v
            end
        end
        set_acl, cond_acl, stats_acl = PageRank.weighted_local_sweep_cut(A, nnz_dict, G.deg, sum(G.deg))
        pr_acl,rc_acl = pr_rc(set_acl,1:n)
        put!(results, (join((pin,delta,Int(n)),","),[pr_acl,rc_acl,cond_acl,pr_crd,rc_crd,cond_acl,pr_slq,rc_slq,cond_slq,q_best]))
    end
end


function make_jobs_pr_rc(jobs,n_list,delta_list,ntrials,pin_list)
    seed = 1
    records = Dict()
    for n in n_list
        for delta in delta_list
            for pin in pin_list
                records[String(join((pin,1.0*delta,Int(n)),","))] = []
                for i = 1:ntrials
                    put!(jobs,(delta,n,seed,pin))
                    seed += 1
                end
            end
        end
    end
    for i = 1:length(workers())
        put!(jobs,(-1,-1,-1,-1))
    end
    return records
end

function pr_rc_cond_runtime_parallel(n_list,delta_list,ncls,gamma,rho,kappa,ntrials,pin_list,pout;use_crd=true)
    nexps = length(pin_list)*length(n_list)*length(delta_list)*ntrials
    jobs = RemoteChannel(()->Channel{Tuple{Float64,Int64,Int64,Float64}}(nexps+length(workers())))
    results = RemoteChannel(()->Channel{Tuple{String,Array{Float64,1}}}(nexps))
    records = make_jobs_pr_rc(jobs,n_list,delta_list,ntrials,pin_list)
    #make_jobs(jobs,q_list,n_list,delta_list,ntrials,records)
    for p in workers()
        remote_do(pr_rc_cond_worker,p,gamma,rho,kappa,pout,ncls,use_crd,jobs,results)
    end
    while nexps > 0 # wait for all jobs to finish
       input,output = take!(results)
       push!(records[input],output)
       nexps = nexps - 1
       @show input,output
       println("$nexps jobs left.")
    end
    return records
end

n_list = [5000,10000]
delta_list = [0.0,1.0e-5,1.0e-4]
pin_list = [0.08,0.06,0.05,0.04,0.03]
pout = 0.01
ncls = 5
rho = 0.5
kappa = 0.005
ntrials = 20
gamma = 0.1

records_accuracy = pr_rc_cond_runtime_parallel(n_list,delta_list,ncls,gamma,rho,kappa,ntrials,pin_list,pout,use_crd=true)

save("pr_rc_cond.jld2",records_accuracy)
#
n = 5000
A = sbm(1,0.04,0.01,5,n)
G = SLQ.graph(A)
S = collect(1:round(Int,0.01*n))
L = SLQ.QHuberLoss(1.2, 0.0)
time_sllp = @elapsed (x,r,iter) = SLQ.slq_diffusion(G, S, 0.1, 0.005, 0.5, L, max_iters=100000,epsilon=1.0e-8)
cluster,cond_crd,time_crd = CRDlgc.crd(G,S)
@elapsed x_acl = PageRank.acl_diffusion(G,S,0.1,0.005)

nonzeros(x_acl.>0)

nnz_dict = Dict{Int,Float64}()
for (i,v) in enumerate(x_acl)
    if v > 0
        nnz_dict[i] = v
    end
end
bestset, bestcond, beststats = PageRank.weighted_local_sweep_cut(A, nnz_dict, G.deg, sum(G.deg))

bestcond

nnz_dict = Dict{Int,Float64}()
for (i,v) in enumerate(x)
    if v > 0
        nnz_dict[i] = v
    end
end
bestset, bestcond, beststats = PageRank.weighted_local_sweep_cut(A, nnz_dict, G.deg, sum(G.deg))
bestcond
length(bestset)

1-length(setdiff(bestset,Set(1:n)))/length(bestset)
1-length(setdiff(Array{Int,1}(cluster),Set(1:n)))/length(cluster)

#
# bestcond
# length(cluster)
# cond_crd
#
# using PyCall
# push!(PyVector(pyimport("sys")["path"]), "/homes/liu1740/Research/LocalGraphClustering/")
# @pyimport localgraphclustering as lgc
# @pyimport scipy.sparse as sp
#
# Asp = sp.csr_matrix((A.nzval,A.rowval.-1,A.colptr.-1), shape=size(A))
# g = lgc.GraphLocal().from_sparse_adjacency(Asp)
# (cluster,conductance) = lgc.flow_clustering(g,S.-1,method="crd",h=3)
