using Distributed
addprocs(100);

@everywhere using LightGraphs
@everywhere using DelimitedFiles
@everywhere using SparseArrays
@everywhere using MAT
@everywhere using Random
@everywhere using JLD2,FileIO
@everywhere using MatrixNetworks
@everywhere using Statistics


@everywhere include("PageRank.jl")
@everywhere include("SLQcvx.jl") # this includes SLQ.jl
@everywhere include("common.jl")
@everywhere include("FlowSeed-1.0.jl")
@everywhere include("HeatKernel.jl")
@everywhere include("CRDlgc.jl")
@everywhere include("GCN.jl")

@everywhere global offset = Dict("dblp"=>1,"liveJournal"=>10,"amazon"=>10)
@everywhere global dataset = "dblp"

@everywhere function nonlinear_diffusion(M, h, niter, v, p)
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

@everywhere function worker_slq(jobs,results,vars)
    A = vars["A"]
    A = Int.(A)
    C = vars["C"]
    csize = []
    for i = 1:size(C,2)
        push!(csize,nnz(C[:,i]))
    end
    G = SLQ.graph(A)
    c34 = round(Int,maximum(csize)^(3/4))
    cids = sortperm(abs.(csize .- c34))[1:600]
    k_list = collect(1:offset[dataset]:round(Int,c34/2))
    q_kappa_map = Dict()
    q_kappa_map[1.5] = 0.02
    q_kappa_map[4.0] = 0.001
    q_kappa_map[8.0] = 0.00001
    q_kappa_map[10.0] = 0.000001
    while true
        input = take!(jobs)
        q,index = input
        if q == -1
            break
        end
        cid = cids[index]
        truth = C[:,cid].nzind
        n = length(truth)
        seed = 1
        delta = 0.0
        S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.1*n))]]
        L = SLQ.QHuberLoss(q, delta)
        kappa = q_kappa_map[q]
        curr_time = @elapsed (x_slq_degnorm,r,iter) = SLQ.slq_diffusion(G, S, 0.05, kappa, 0.9, L, max_iters=10000000,epsilon=1.0e-8)
        x_slq = (x_slq_degnorm.^(q-1)).*G.deg
        sorted_ids = sortperm(-1*x_slq)
        sorted_ids_degnorm = sortperm(-1*x_slq_degnorm)
        curr_col = zeros(length(k_list))
        curr_col_degnorm = zeros(length(k_list))
        for (i,k) in enumerate(k_list)
            cluster_slq = sorted_ids[1:k]
            pr_slq,rc_slq = compute_pr_rc(cluster_slq,truth)
            cluster_slq_degnorm = sorted_ids_degnorm[1:k]
            pr_slq_degnorm,rc_slq_degnorm = compute_pr_rc(cluster_slq_degnorm,truth)
            curr_col[i] = rc_slq
            curr_col_degnorm[i] = rc_slq_degnorm
        end
        put!(results,(q,index,curr_col,curr_col_degnorm,curr_time))
    end
end




@everywhere function worker_acl(jobs,results,vars)
    A = vars["A"]
    A = Int.(A)
    C = vars["C"]
    csize = []
    for i = 1:size(C,2)
        push!(csize,nnz(C[:,i]))
    end
    G = SLQ.graph(A)
    c34 = round(Int,maximum(csize)^(3/4))
    cids = sortperm(abs.(csize .- c34))[1:600]
    k_list = collect(1:offset[dataset]:round(Int,c34/2))
    while true
        input = take!(jobs)
        index = input
        if index == -1
            break
        end
        cid = cids[index]
        truth = C[:,cid].nzind
        n = length(truth)
        seed = 1
        S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.1*n))]]
        curr_time = @elapsed x_acl = nonlinear_diffusion(G,S,0.05,0.002)
        x_acl_degnorm = x_acl./G.deg
        sorted_ids = sortperm(-1*x_acl)
        sorted_ids_degnorm = sortperm(-1*x_acl_degnorm)
        curr_col = zeros(length(k_list))
        curr_col_degnorm = zeros(length(k_list))
        for (i,k) in enumerate(k_list)
            cluster_acl = sorted_ids[1:k]
            pr_acl,rc_acl = compute_pr_rc(cluster_acl,truth)
            cluster_acl_degnorm = sorted_ids_degnorm[1:k]
            pr_acl_degnorm,rc_acl_degnorm = compute_pr_rc(cluster_acl_degnorm,truth)
            curr_col[i] = rc_acl
            curr_col_degnorm[i] = rc_acl_degnorm
        end
        put!(results,(index,curr_col,curr_col_degnorm,curr_time))
    end
end


@everywhere function worker_nld(jobs,results,vars)
    A = vars["A"]
    A = Int.(A)
    C = vars["C"]
    csize = []
    for i = 1:size(C,2)
        push!(csize,nnz(C[:,i]))
    end
    G = SLQ.graph(A)
    c34 = round(Int,maximum(csize)^(3/4))
    cids = sortperm(abs.(csize .- c34))[1:600]
    k_list = collect(1:offset[dataset]:round(Int,c34/2))
    while true
        input = take!(jobs)
        index = input
        if index == -1
            break
        end
        cid = cids[index]
        truth = C[:,cid].nzind
        n = length(truth)
        seed = 1
        S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.1*n))]]
        curr_time = @elapsed x_nld_degnorm = nonlinear_diffusion(A, 0.002, 1000, sparsevec(S,1.0,size(A,1)), 1.5)
        x_nld = x_nld_degnorm.*G.deg
        sorted_ids = sortperm(-1*x_nld)
        sorted_ids_degnorm = sortperm(-1*x_nld_degnorm)
        curr_col = zeros(length(k_list))
        curr_col_degnorm = zeros(length(k_list))
        for (i,k) in enumerate(k_list)
            cluster_nld = sorted_ids[1:k]
            pr_nld,rc_nld = compute_pr_rc(cluster_nld,truth)
            cluster_nld_degnorm = sorted_ids_degnorm[1:k]
            pr_nld_degnorm,rc_nld_degnorm = compute_pr_rc(cluster_nld_degnorm,truth)
            curr_col[i] = rc_nld
            curr_col_degnorm[i] = rc_nld_degnorm
        end
        put!(results,(index,curr_col,curr_col_degnorm,curr_time))
    end
end

@everywhere function worker_hk(jobs,results,vars)
    A = vars["A"]
    A = Int.(A)
    C = vars["C"]
    csize = []
    for i = 1:size(C,2)
        push!(csize,nnz(C[:,i]))
    end
    G = SLQ.graph(A)
    c34 = round(Int,maximum(csize)^(3/4))
    cids = sortperm(abs.(csize .- c34))[1:600]
    k_list = collect(1:offset[dataset]:round(Int,c34/2))
    while true
        input = take!(jobs)
        index = input
        if index == -1
            break
        end
        cid = cids[index]
        truth = C[:,cid].nzind
        n = length(truth)
        seed = 1
        S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.1*n))]]
        curr_time = @elapsed x_hk_degnorm_dict = HeatKernel.hk_grow(G.A,S)[4]
        x_hk_degnorm = zeros(size(A,1))
        for key in keys(x_hk_degnorm_dict)
            x_hk_degnorm[key] = x_hk_degnorm_dict[key]
        end
        x_hk = x_hk_degnorm.*G.deg
        sorted_ids_degnorm = sortperm(-1*x_hk_degnorm)
        curr_col_degnorm = zeros(length(k_list))
        sorted_ids = sortperm(-1*x_hk)
        curr_col = zeros(length(k_list))
        for (i,k) in enumerate(k_list)
            cluster_hk_degnorm = sorted_ids_degnorm[1:k]
            pr_hk_degnorm,rc_hk_degnorm = compute_pr_rc(cluster_hk_degnorm,truth)
            curr_col_degnorm[i] = rc_hk_degnorm
            cluster_hk = sorted_ids[1:k]
            pr_hk,rc_hk = compute_pr_rc(cluster_hk,truth)
            curr_col[i] = rc_hk
        end
        put!(results,(index,curr_col,curr_col_degnorm,curr_time))
    end
end



@everywhere function worker_crd(jobs,results,vars,h)
    A = vars["A"]
    A = Int.(A)
    C = vars["C"]
    csize = []
    for i = 1:size(C,2)
        push!(csize,nnz(C[:,i]))
    end
    G = SLQ.graph(A)
    c34 = round(Int,maximum(csize)^(3/4))
    cids = sortperm(abs.(csize .- c34))[1:600]
    k_list = collect(1:offset[dataset]:round(Int,c34/2))
    while true
        input = take!(jobs)
        index = input
        if index == -1
            break
        end
        cid = cids[index]
        truth = C[:,cid].nzind
        n = length(truth)
        seed = 1
        S = truth[randperm(MersenneTwister(seed),n)[1:max(1,round(Int,0.1*n))]]
        curr_time = @elapsed cluster = CRDlgc.crd(G,S,h=h)[1]
        x_crd_degnorm = zeros(size(A,1))
        x_crd_degnorm[cluster] .= 1
        sorted_ids_degnorm = sortperm(-1*x_crd_degnorm)
        curr_col_degnorm = zeros(length(k_list))
        for (i,k) in enumerate(k_list)
            cluster_crd_degnorm = sorted_ids_degnorm[1:k]
            pr_crd_degnorm,rc_crd_degnorm = compute_pr_rc(cluster_crd_degnorm,truth)
            curr_col_degnorm[i] = rc_crd_degnorm
        end
        put!(results,(index,curr_col_degnorm,curr_time))
    end
end



function make_jobs_slq(q_list,jobs)
    for q in q_list
        for i in 1:600
            put!(jobs,(q,i))
        end
    end
    for i in 1:length(workers())
        put!(jobs,(-1.0,-1))
    end
end

function make_jobs_acl(jobs)
    for i in 1:600
        put!(jobs,i)
    end
    for i in 1:length(workers())
        put!(jobs,-1)
    end
end

function make_jobs_nld(jobs)
    for i in 1:600
        put!(jobs,i)
    end
    for i in 1:length(workers())
        put!(jobs,-1)
    end
end

function make_jobs_hk(jobs)
    for i in 1:600
        put!(jobs,i)
    end
    for i in 1:length(workers())
        put!(jobs,-1)
    end
end

function make_jobs_crd(jobs)
    for i in 1:600
        put!(jobs,i)
    end
    for i in 1:length(workers())
        put!(jobs,-1)
    end
end




function huge_graph_parallel_slq(q_list)
    #vars = matread("liveJournal.mat")
    vars = matread("$dataset.mat")
    A = vars["A"]
    A = Int.(A)
    C = vars["C"]
    csize = []
    for i = 1:size(C,2)
        push!(csize,nnz(C[:,i]))
    end
    G = SLQ.graph(A)
    c34 = round(Int,maximum(csize)^(3/4))
    cids = sortperm(abs.(csize .- c34))[1:600]
    k_list = collect(1:offset[dataset]:round(Int,c34/2))
    nexps = length(q_list)*600
    jobs = RemoteChannel(()->Channel{Tuple{Float64,Int64}}(nexps+length(workers())))
    records = Dict(string(q)=>zeros(length(k_list),600) for q in q_list)
    records["time"] = zeros(length(q_list),1)
    records_degnorm = Dict(string(q)=>zeros(length(k_list),600) for q in q_list)
    records_degnorm["time"] = zeros(length(q_list),1)
    results = RemoteChannel(()->Channel{Tuple{Float64,Int64,Array{Float64,1},Array{Float64,1},Float64}}(nexps))
    make_jobs_slq(q_list,jobs)
    for p in workers()
        remote_do(worker_slq,p,jobs,results,vars)
    end
    while nexps > 0 # wait for all jobs to finish
       q,index,curr_col,curr_col_degnorm,curr_time = take!(results)
       records[string(q)][:,index] = curr_col
       records_degnorm[string(q)][:,index] = curr_col_degnorm
       nexps = nexps - 1
       records["time"][findall(x->x==q,q_list)[1]] += curr_time
       records_degnorm["time"][findall(x->x==q,q_list)[1]] += curr_time
       println("$nexps jobs left.")
    end
    return records,records_degnorm
end




function huge_graph_parallel_acl()
    #vars = matread("liveJournal.mat")
    vars = matread("$dataset.mat")
    A = vars["A"]
    A = Int.(A)
    C = vars["C"]
    csize = []
    for i = 1:size(C,2)
        push!(csize,nnz(C[:,i]))
    end
    G = SLQ.graph(A)
    c34 = round(Int,maximum(csize)^(3/4))
    cids = sortperm(abs.(csize .- c34))[1:600]
    k_list = collect(1:offset[dataset]:round(Int,c34/2))
    nexps = 600
    jobs = RemoteChannel(()->Channel{Int64}(nexps+length(workers())))
    records = Dict("acl"=>zeros(length(k_list),600))
    records["time"] = zeros(1,1)
    records_degnorm = Dict("acl"=>zeros(length(k_list),600))
    records_degnorm["time"] = zeros(1,1)
    results = RemoteChannel(()->Channel{Tuple{Int64,Array{Float64,1},Array{Float64,1},Float64}}(nexps))
    make_jobs_acl(jobs)
    for p in workers()
        remote_do(worker_acl,p,jobs,results,vars)
    end
    while nexps > 0 # wait for all jobs to finish
       index,curr_col,curr_col_degnorm,curr_time = take!(results)
       records["acl"][:,index] = curr_col
       records_degnorm["acl"][:,index] = curr_col_degnorm
       nexps = nexps - 1
       records["time"][1] += curr_time
       records_degnorm["time"][1] += curr_time
       println("$nexps jobs left.")
    end
    return records,records_degnorm
end


function huge_graph_parallel_nld()
    #vars = matread("liveJournal.mat")
    vars = matread("$dataset.mat")
    A = vars["A"]
    A = Int.(A)
    C = vars["C"]
    csize = []
    for i = 1:size(C,2)
        push!(csize,nnz(C[:,i]))
    end
    G = SLQ.graph(A)
    c34 = round(Int,maximum(csize)^(3/4))
    cids = sortperm(abs.(csize .- c34))[1:600]
    k_list = collect(1:offset[dataset]:round(Int,c34/2))
    nexps = 600
    jobs = RemoteChannel(()->Channel{Int64}(nexps+length(workers())))
    records = Dict("nld"=>zeros(length(k_list),600))
    records["time"] = zeros(1,1)
    records_degnorm = Dict("nld"=>zeros(length(k_list),600))
    records_degnorm["time"] = zeros(1,1)
    results = RemoteChannel(()->Channel{Tuple{Int64,Array{Float64,1},Array{Float64,1},Float64}}(nexps))
    make_jobs_nld(jobs)
    for p in workers()
        remote_do(worker_nld,p,jobs,results,vars)
    end
    while nexps > 0 # wait for all jobs to finish
       index,curr_col,curr_col_degnorm,curr_time = take!(results)
       records["nld"][:,index] = curr_col
       records_degnorm["nld"][:,index] = curr_col_degnorm
       nexps = nexps - 1
       records["time"][1] += curr_time
       records_degnorm["time"][1] += curr_time
       println("$nexps jobs left.")
    end
    return records,records_degnorm
end

function huge_graph_parallel_hk()
    #vars = matread("liveJournal.mat")
    vars = matread("$dataset.mat")
    A = vars["A"]
    A = Int.(A)
    C = vars["C"]
    csize = []
    for i = 1:size(C,2)
        push!(csize,nnz(C[:,i]))
    end
    G = SLQ.graph(A)
    c34 = round(Int,maximum(csize)^(3/4))
    cids = sortperm(abs.(csize .- c34))[1:600]
    k_list = collect(1:offset[dataset]:round(Int,c34/2))
    nexps = 600
    jobs = RemoteChannel(()->Channel{Int64}(nexps+length(workers())))
    records = Dict("hk"=>zeros(length(k_list),600))
    records["time"] = zeros(1,1)
    records_degnorm = Dict("hk"=>zeros(length(k_list),600))
    records_degnorm["time"] = zeros(1,1)
    results = RemoteChannel(()->Channel{Tuple{Int64,Array{Float64,1},Array{Float64,1},Float64}}(nexps))
    make_jobs_hk(jobs)
    for p in workers()
        remote_do(worker_hk,p,jobs,results,vars)
    end
    while nexps > 0 # wait for all jobs to finish
       index,curr_col,curr_col_degnorm,curr_time = take!(results)
       records["hk"][:,index] = curr_col
       records_degnorm["hk"][:,index] = curr_col_degnorm
       nexps = nexps - 1
       records_degnorm["time"][1] += curr_time
       records["time"][1] += curr_time
       println("$nexps jobs left.")
    end
    return records,records_degnorm
end


function huge_graph_parallel_crd(;h=5)
    #vars = matread("liveJournal.mat")
    vars = matread("$dataset.mat")
    A = vars["A"]
    A = Int.(A)
    C = vars["C"]
    csize = []
    for i = 1:size(C,2)
        push!(csize,nnz(C[:,i]))
    end
    G = SLQ.graph(A)
    c34 = round(Int,maximum(csize)^(3/4))
    cids = sortperm(abs.(csize .- c34))[1:600]
    k_list = collect(1:offset[dataset]:round(Int,c34/2))
    nexps = 600
    jobs = RemoteChannel(()->Channel{Int64}(nexps+length(workers())))
    records = Dict("crd"=>zeros(length(k_list),600))
    records["time"] = zeros(1,1)
    records_degnorm = Dict("crd"=>zeros(length(k_list),600))
    records_degnorm["time"] = zeros(1,1)
    results = RemoteChannel(()->Channel{Tuple{Int64,Array{Float64,1},Float64}}(nexps))
    make_jobs_crd(jobs)
    for p in workers()
        remote_do(worker_crd,p,jobs,results,vars,h)
    end
    while nexps > 0 # wait for all jobs to finish
       index,curr_col_degnorm,curr_time = take!(results)
       records_degnorm["crd"][:,index] = curr_col_degnorm
       nexps = nexps - 1
       records_degnorm["time"][1] += curr_time
       println("$nexps jobs left.")
    end
    return records_degnorm
end

q_list = [1.5,4.0,8.0]
records,records_degnorm = huge_graph_parallel_slq(q_list)

save("$dataset-slq.jld2",records)
save("$dataset-slq-degnorm.jld2",records_degnorm)

records,records_degnorm = huge_graph_parallel_acl()
save("$dataset-acl.jld2",records)
save("$dataset-acl-degnorm.jld2",records_degnorm)

records,records_degnorm = huge_graph_parallel_hk()
save("$dataset-hk.jld2",records)
save("$dataset-hk-degnorm.jld2",records_degnorm)

records_degnorm = huge_graph_parallel_crd()
save("$dataset-crd-h-5-degnorm.jld2",records_degnorm)

records_degnorm = huge_graph_parallel_crd(h=3)
save("$dataset-crd-h-3-degnorm.jld2",records_degnorm)


records,records_degnorm = huge_graph_parallel_nld()
save("$dataset-nld.jld2",records)
save("$dataset-nld-degnorm.jld2",records_degnorm)

@everywhere global dataset = "liveJournal"
records,records_degnorm = huge_graph_parallel_nld()
save("$dataset-nld.jld2",records)
save("$dataset-nld-degnorm.jld2",records_degnorm)