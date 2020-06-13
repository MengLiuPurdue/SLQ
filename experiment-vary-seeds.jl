using LightGraphs
using DelimitedFiles
using SparseArrays
using MAT
using Random
using JLD2,FileIO
using PyCall
using PyPlot
using MatrixNetworks
using Statistics

include("PageRank.jl")
include("SLQcvx.jl") # this includes SLQ.jl
include("common.jl")
include("CRDlgc.jl")
include("HeatKernel.jl")
include("FlowSeed-1.0.jl")

file = "MIT8.mat"
vars = matread(file)
A = vars["A"]
A = round.(Int,A)
A,p = largest_component(A)
G = SLQ.graph(A)
labels = vars["local_info"]
label = 2008
truth = findall(labels[p,6] .== label)
n = length(truth)

slq_f1_median = []
slq_f1_error20 = []
slq_f1_error80 = []
slq_cond_median = []
slq_cond_error20 = []
slq_cond_error80 = []

seed = 1
for nseeds in 1:60
    S = truth[randperm(MersenneTwister(seed),n)[1:nseeds]]
    q = 1.2
    delta = 0.0
    L = SLQ.QHuberLoss(q, delta)
    global kappa_min = 0.001
    global kappa_max = 0.1
    kappa_mid = (kappa_min+kappa_max)/2
    (x,r,iter) = SLQ.slq_diffusion(G, S, 0.05, kappa_mid, 0.5, L, max_iters=100000,epsilon=1.0e-8)
    while kappa_max - kappa_min > 0.001
        nonzero_entries = sum(x.>0)
        if nonzero_entries > size(A,1)/5
            global kappa_min = kappa_mid
        else
            global kappa_max = kappa_mid
        end
        kappa_mid = (kappa_min+kappa_max)/2
        (x,r,iter) = SLQ.slq_diffusion(G, S, 0.05, kappa_mid, 0.5, L, max_iters=100000,epsilon=1.0e-8)
    end
    kappa_mid = (kappa_min+kappa_max)/2
    conds = []
    f1s = []
    @show kappa_mid
    for seed = 1:50
        S = truth[randperm(MersenneTwister(seed),n)[1:nseeds]]
        (x,r,iter) = SLQ.slq_diffusion(G, S, 0.05, kappa_mid, 0.5, L, max_iters=100000,epsilon=1.0e-8)
        cluster,cond = PageRank.round_to_cluster(G,x)
        pr,rc = compute_pr_rc(cluster,truth)
        push!(conds,cond)
        push!(f1s,rc*pr*2/(rc+pr))
    end
    f1_median = median(f1s)
    f1_error20 = quantile(f1s,0.2)
    f1_error80 = quantile(f1s,0.8)
    push!(slq_f1_median,f1_median)
    push!(slq_f1_error20,f1_error20)
    push!(slq_f1_error80,f1_error80)
    cond_median = median(conds)
    cond_error20 = quantile(conds,0.2)
    cond_error80 = quantile(conds,0.8)
    push!(slq_cond_median,cond_median)
    push!(slq_cond_error20,cond_error20)
    push!(slq_cond_error80,cond_error80)
    @show nseeds,round(0.1*n),f1_median,f1_error20,f1_error80,cond_median,cond_error20,cond_error80
end

records = Dict("f1_median"=>slq_f1_median,"f1_error20"=>slq_f1_error20,"f1_error80"=>slq_f1_error80,"cond_median"=>slq_cond_median,"cond_error20"=>slq_cond_error20,"cond_error80"=>slq_cond_error80)
save("slq-varying-seeds.jld2",records)


acl_f1_median = []
acl_f1_error20 = []
acl_f1_error80 = []
acl_cond_median = []
acl_cond_error20 = []
acl_cond_error80 = []

seed = 1
for nseeds in 1:60
    S = truth[randperm(MersenneTwister(seed),n)[1:nseeds]]
    global kappa_min = 0.001
    global kappa_max = 0.1
    kappa_mid = (kappa_min+kappa_max)/2
    x = PageRank.degnorm_acl_diffusion(G, S, 0.05, kappa_mid)
    while kappa_max - kappa_min > 0.001
        nonzero_entries = sum(x.>0)
        if nonzero_entries > size(A,1)/5
            global kappa_min = kappa_mid
        else
            global kappa_max = kappa_mid
        end
        kappa_mid = (kappa_min+kappa_max)/2
        x = PageRank.degnorm_acl_diffusion(G, S, 0.05, kappa_mid)
    end
    kappa_mid = (kappa_min+kappa_max)/2
    conds = []
    f1s = []
    @show kappa_mid
    for seed = 1:50
        S = truth[randperm(MersenneTwister(seed),n)[1:nseeds]]
        x = PageRank.degnorm_acl_diffusion(G, S, 0.05, kappa_mid)
        cluster,cond = PageRank.round_to_cluster(G,x)
        pr,rc = compute_pr_rc(cluster,truth)
        push!(conds,cond)
        push!(f1s,rc*pr*2/(rc+pr))
    end
    f1_median = median(f1s)
    f1_error20 = quantile(f1s,0.2)
    f1_error80 = quantile(f1s,0.8)
    push!(acl_f1_median,f1_median)
    push!(acl_f1_error20,f1_error20)
    push!(acl_f1_error80,f1_error80)
    cond_median = median(conds)
    cond_error20 = quantile(conds,0.2)
    cond_error80 = quantile(conds,0.8)
    push!(acl_cond_median,cond_median)
    push!(acl_cond_error20,cond_error20)
    push!(acl_cond_error80,cond_error80)
    @show nseeds,round(0.1*n),f1_median,f1_error20,f1_error80
end

records = Dict("f1_median"=>acl_f1_median,"f1_error20"=>acl_f1_error20,"f1_error80"=>acl_f1_error80,"cond_median"=>acl_cond_median,"cond_error20"=>acl_cond_error20,"cond_error80"=>acl_cond_error80)
save("acl-varying-seeds.jld2",records)


crd_f1_median = []
crd_f1_error20 = []
crd_f1_error80 = []
crd_cond_median = []
crd_cond_error20 = []
crd_cond_error80 = []

seed = 1
for nseeds in 1:60
    conds = []
    f1s = []
    for seed = 1:50
        try
            S = truth[randperm(MersenneTwister(seed),n)[1:nseeds]]
            cluster,cond,time = CRDlgc.crd(G,S,U=3,h=3,w=2)
            pr,rc = compute_pr_rc(cluster,truth)
            push!(conds,cond)
            if rc+pr > 0
                push!(f1s,rc*pr*2/(rc+pr))
            else
                push!(f1s,0.0)
            end
            @show seed,cond,f1s[end]
        catch
            continue
        end
    end
    f1_median = median(f1s)
    f1_error20 = quantile(f1s,0.2)
    f1_error80 = quantile(f1s,0.8)
    push!(crd_f1_median,f1_median)
    push!(crd_f1_error20,f1_error20)
    push!(crd_f1_error80,f1_error80)
    cond_median = median(conds)
    cond_error20 = quantile(conds,0.2)
    cond_error80 = quantile(conds,0.8)
    push!(crd_cond_median,cond_median)
    push!(crd_cond_error20,cond_error20)
    push!(crd_cond_error80,cond_error80)
    @show nseeds,round(0.1*n),f1_median,f1_error20,f1_error80,cond_median,cond_error20,cond_error80
end


records = Dict("f1_median"=>crd_f1_median,"f1_error20"=>crd_f1_error20,"f1_error80"=>crd_f1_error80,"cond_median"=>crd_cond_median,"cond_error20"=>crd_cond_error20,"cond_error80"=>crd_cond_error80)
save("crd-h-3-varying-seeds.jld2",records)





crd_f1_median = []
crd_f1_error20 = []
crd_f1_error80 = []
crd_cond_median = []
crd_cond_error20 = []
crd_cond_error80 = []

seed = 1
for nseeds in 10:60
    conds = []
    f1s = []
    for seed = 1:50
        try
            S = truth[randperm(MersenneTwister(seed),n)[1:nseeds]]
            cluster,cond,time = CRDlgc.crd(G,S,U=3,h=5,w=2)
            pr,rc = compute_pr_rc(cluster,truth)
            push!(conds,cond)
            if rc+pr > 0
                push!(f1s,rc*pr*2/(rc+pr))
            else
                push!(f1s,0.0)
            end
            @show seed,cond,f1s[end]
        catch
            continue
        end
    end
    f1_median = median(f1s)
    f1_error20 = quantile(f1s,0.2)
    f1_error80 = quantile(f1s,0.8)
    push!(crd_f1_median,f1_median)
    push!(crd_f1_error20,f1_error20)
    push!(crd_f1_error80,f1_error80)
    cond_median = median(conds)
    cond_error20 = quantile(conds,0.2)
    cond_error80 = quantile(conds,0.8)
    push!(crd_cond_median,cond_median)
    push!(crd_cond_error20,cond_error20)
    push!(crd_cond_error80,cond_error80)
    @show nseeds,round(0.1*n),f1_median,f1_error20,f1_error80,cond_median,cond_error20,cond_error80
end
records = Dict("f1_median"=>crd_f1_median,"f1_error20"=>crd_f1_error20,"f1_error80"=>crd_f1_error80,"cond_median"=>crd_cond_median,"cond_error20"=>crd_cond_error20,"cond_error80"=>crd_cond_error80)
save("crd-h-5-varying-seeds.jld2",records)






sl_f1_median = []
sl_f1_error20 = []
sl_f1_error80 = []
sl_cond_median = []
sl_cond_error20 = []
sl_cond_error80 = []

A = A.*1.0
seed = 1
for nseeds in 1:10
    S = truth[randperm(MersenneTwister(seed),n)[1:nseeds]]
    global eps_min = 0.4
    global eps_max = 5.0
    eps_mid = (eps_min+eps_max)/2
    Srand = truth[randperm(MersenneTwister(seed),n)[1:nseeds]]
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
    cluster,cond = FlowSeed(A,S,0.5,pS,RinS)
    while eps_max - eps_min > 0.1
        nonzero_entries = length(cluster)
        if nonzero_entries > size(A,1)/5
            global eps_min = eps_mid
        else
            global eps_max = eps_mid
        end
        eps_mid = (eps_min+eps_max)/2
        cluster,cond = FlowSeed(A,S,eps_mid,pS,RinS)
    end
    eps_mid = (eps_min+eps_max)/2
    conds = []
    f1s = []
    for seed = 1:50
        Srand = truth[randperm(MersenneTwister(seed),n)[1:nseeds]]
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
        cluster,cond = FlowSeed(A,S,eps_mid,pS,RinS)
        pr,rc = compute_pr_rc(cluster,truth)
        push!(conds,cond)
        push!(f1s,rc*pr*2/(rc+pr))
    end
    f1_median = median(f1s)
    f1_error20 = quantile(f1s,0.2)
    f1_error80 = quantile(f1s,0.8)
    push!(sl_f1_median,f1_median)
    push!(sl_f1_error20,f1_error20)
    push!(sl_f1_error80,f1_error80)
    cond_median = median(conds)
    cond_error20 = quantile(conds,0.2)
    cond_error80 = quantile(conds,0.8)
    push!(sl_cond_median,cond_median)
    push!(sl_cond_error20,cond_error20)
    push!(sl_cond_error80,cond_error80)
    @show nseeds,round(0.1*n),f1_median,f1_error20,f1_error80,cond_median,cond_error20,cond_error80
end

records = Dict("f1_median"=>sl_f1_median,"f1_error20"=>sl_f1_error20,"f1_error80"=>sl_f1_error80,"cond_median"=>sl_cond_median,"cond_error20"=>sl_cond_error20,"cond_error80"=>sl_cond_error80)
save("sl-varying-seeds.jld2",records)



hk_f1_median = []
hk_f1_error20 = []
hk_f1_error80 = []
hk_cond_median = []
hk_cond_error20 = []
hk_cond_error80 = []


for nseeds in 1:60
    conds = []
    f1s = []
    for seed = 1:50
        S = truth[randperm(MersenneTwister(seed),n)[1:nseeds]]
        cluster,cond,time = HeatKernel.hk_grow(A,S)
        pr,rc = compute_pr_rc(cluster,truth)
        push!(conds,cond)
        if rc+pr > 0
            push!(f1s,rc*pr*2/(rc+pr))
        end
    end
    f1_median = median(f1s)
    f1_error20 = quantile(f1s,0.2)
    f1_error80 = quantile(f1s,0.8)
    push!(hk_f1_median,f1_median)
    push!(hk_f1_error20,f1_error20)
    push!(hk_f1_error80,f1_error80)
    cond_median = median(conds)
    cond_error20 = quantile(conds,0.2)
    cond_error80 = quantile(conds,0.8)
    push!(hk_cond_median,cond_median)
    push!(hk_cond_error20,cond_error20)
    push!(hk_cond_error80,cond_error80)
    @show nseeds,round(0.1*n),f1_median,f1_error20,f1_error80,cond_median,cond_error20,cond_error80
end
records = Dict("f1_median"=>hk_f1_median,"f1_error20"=>hk_f1_error20,"f1_error80"=>hk_f1_error80,"cond_median"=>hk_cond_median,"cond_error20"=>hk_cond_error20,"cond_error80"=>hk_cond_error80)
save("hk-varying-seeds.jld2",records)



records_crd = load("crd-h-3-varying-seeds.jld2")
records_sl = load("sl-varying-seeds.jld2")
records_hk = load("hk-varying-seeds.jld2")
records_acl = load("acl-varying-seeds.jld2")
records_slq = load("slq-varying-seeds.jld2")
crd_f1_median,crd_f1_error20,crd_f1_error80 = records_crd["f1_median"],records_crd["f1_error20"],records_crd["f1_error80"]
sl_f1_median,sl_f1_error20,sl_f1_error80 = records_sl["f1_median"],records_sl["f1_error20"],records_sl["f1_error80"]
slq_f1_median,slq_f1_error20,slq_f1_error80 = records_slq["f1_median"],records_slq["f1_error20"],records_slq["f1_error80"]
acl_f1_median,acl_f1_error20,acl_f1_error80 = records_acl["f1_median"],records_acl["f1_error20"],records_acl["f1_error80"]
hk_f1_median,hk_f1_error20,hk_f1_error80 = records_hk["f1_median"],records_hk["f1_error20"],records_hk["f1_error80"]




@pyimport matplotlib.patches as mpatches
fig,ax = subplots(1,1,figsize=(5,3))
seed_list = collect(1:60)
colors = []
p = ax.plot(seed_list,slq_f1_median)
ax.fill_between(seed_list, slq_f1_error20, slq_f1_error80, alpha=0.3, color=p[1].get_color())
ax.text(62,slq_f1_median[end]+0.02,"SLQ",fontsize=15)
ax.plot([60,61.5],[slq_f1_median[end],slq_f1_median[end]+0.03],color="k")
push!(colors,p[1].get_color())
p = ax.plot(seed_list,acl_f1_median)
ax.fill_between(seed_list, acl_f1_error20, acl_f1_error80, alpha=0.3, color=p[1].get_color())
ax.text(62,acl_f1_median[end]-0.03,"ACL",fontsize=15)
ax.plot([60,61.5],[acl_f1_median[end],acl_f1_median[end]-0.01],color="k")
push!(colors,p[1].get_color())
p = ax.plot(seed_list,crd_f1_median)
ax.fill_between(seed_list, crd_f1_error20, crd_f1_error80, alpha=0.3, color=p[1].get_color())
ax.text(62,crd_f1_median[end]+0.02,"CRD-3",fontsize=15)
ax.plot([60,61.5],[crd_f1_median[end],crd_f1_median[end]+0.03],color="k")
push!(colors,p[1].get_color())
p = ax.plot(seed_list,sl_f1_median)
ax.fill_between(seed_list, sl_f1_error20, sl_f1_error80, alpha=0.3, color=p[1].get_color())
ax.text(62,sl_f1_median[end],"FS",fontsize=15)
ax.plot([60,61.5],[sl_f1_median[end],sl_f1_median[end]+0.01],color="k")
push!(colors,p[1].get_color())
p = ax.plot(seed_list,hk_f1_median)
ax.fill_between(seed_list, hk_f1_error20, hk_f1_error80, alpha=0.3, color=p[1].get_color())
ax.text(62,hk_f1_median[end]-0.03,"HK",fontsize=15)
ax.plot([60,61.5],[hk_f1_median[end],hk_f1_median[end]-0.01],color="k")
push!(colors,p[1].get_color())
ax.set_ylim(0.0,0.75)
ax.set_xlim(1.0,62)
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)
for tick in ax.xaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
for tick in ax.yaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
# handles = [mpatches.Patch(color=color) for color in colors]
# lgd = ax.legend(labels = ["SLQ","ACL","CRD-3","SL","HK"],handles=handles,fancybox=true,shadow=true,loc=9,bbox_to_anchor=(0.475, 1.4),ncol=3,fontsize=15)
fig.savefig("figures/varying-seeds.pdf",format="pdf", bbox_inches="tight")
# ax.set_xscale("log")


