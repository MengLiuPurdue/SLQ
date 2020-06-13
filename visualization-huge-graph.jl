using LightGraphs
using DelimitedFiles
using SparseArrays
using MAT
using Random
using JLD2,FileIO
using PyCall
using PyPlot
using Statistics
@pyimport matplotlib.patches as mpatches

include("PageRank.jl")
include("SLQcvx.jl") # this includes SLQ.jl
include("common.jl")
include("CRDlgc.jl")
include("LocalClusteringObjs.jl")



dataset = "dblp"
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
k_list = collect(1:1:round(Int,c34/2))
rc_slq_records = load("results/$dataset-slq.jld2")
rc_slq_records_degnorm = load("results/$dataset-slq-degnorm.jld2")
rc_acl_all = load("results/$dataset-acl.jld2","acl")
rc_acl_all_degnorm = load("results/$dataset-acl-degnorm.jld2","acl")

fig,ax = subplots(1,1,figsize=(5,4))
q_list = [1.5,4.0,8.0]
offset = [-0.04,-0.02,0.001]
offset_degnorm = [-0.065,-0.04,-0.02]
for (i,q) in enumerate(q_list)
    rc_slq_all = rc_slq_records[string(q)]
    rc_slq_all_degnorm = rc_slq_records_degnorm[string(q)]
    med_slq = vec(median(rc_slq_all,dims=2))
    med_slq_degnorm = vec(median(rc_slq_all_degnorm,dims=2))
    p = ax.plot(k_list,med_slq)
    stderror = vec(std(rc_slq_all,dims=2))./sqrt(600)
    ax.fill_between(k_list, med_slq-2*stderror, med_slq+2*stderror, alpha=0.3, color=p[1].get_color())
    p = ax.plot(k_list,med_slq_degnorm)
    stderror = vec(std(rc_slq_all_degnorm,dims=2))./sqrt(600)
    ax.fill_between(k_list, med_slq_degnorm-2*stderror, med_slq_degnorm+2*stderror, alpha=0.3, color=p[1].get_color())
    ax.text(420,med_slq[end]+offset[i],"SLQ (q=$q)",fontsize=14)
    ax.plot([405,418],[med_slq[end],med_slq[end]+offset[i]+0.01],color="k")
    ax.plot([405,418],[med_slq_degnorm[end],med_slq_degnorm[end]+offset_degnorm[i]+0.02],color="k")
    ax.text(420,med_slq_degnorm[end]+offset_degnorm[i]+0.015,"SLQ normalized (q=$q)",fontsize=14)
    @show med_slq[end]
end

for tick in ax.xaxis.get_major_ticks()
    tick.label.set_fontsize(18)
end
for tick in ax.yaxis.get_major_ticks()
    tick.label.set_fontsize(18)
end

med_acl = vec(median(rc_acl_all,dims=2))
med_acl_degnorm = vec(median(rc_acl_all_degnorm,dims=2))
p = ax.plot(k_list,med_acl)
stderror = vec(std(rc_acl_all,dims=2))./sqrt(600)
ax.fill_between(k_list, med_acl-2*stderror, med_acl+2*stderror, alpha=0.3, color=p[1].get_color())
p = ax.plot(k_list,med_acl_degnorm)
stderror = vec(std(rc_acl_all_degnorm,dims=2))./sqrt(600)
ax.fill_between(k_list, med_acl_degnorm-2*stderror, med_acl_degnorm+2*stderror, alpha=0.3, color=p[1].get_color())
ax.text(420,med_acl[end]-0.04,"ACL",fontsize=14)
ax.plot([405,418],[med_acl[end],med_acl[end]-0.03],color="k")
ax.plot([405,418],[med_acl_degnorm[end],med_acl_degnorm[end]-0.02],color="k")
ax.text(420,med_acl_degnorm[end]-0.035,"ACL normalized",fontsize=14)
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)
fig.savefig("figures/600-communities-$dataset.pdf",format="pdf", bbox_inches="tight")



dataset = "liveJournal"
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
k_list = collect(1:10:round(Int,c34/2))
rc_slq_records = load("results/$dataset-slq.jld2")
rc_slq_records_degnorm = load("results/$dataset-slq-degnorm.jld2")
rc_acl_all = load("results/$dataset-acl.jld2","acl")
rc_acl_all_degnorm = load("results/$dataset-acl-degnorm.jld2","acl")

fig,ax = subplots(1,1,figsize=(5,4))
q_list = [1.5,8.0,4.0]
q_list_degnorm = [1.5,4.0,8.0]
offset = [0.01,-0.01,0.01]
offset_degnorm = [-0.03,-0.04,-0.025]
for (i,q) in enumerate(q_list)
    rc_slq_all = rc_slq_records[string(q)]
    med_slq = vec(median(rc_slq_all,dims=2))
    p = ax.plot(k_list,med_slq)
    stderror = vec(std(rc_slq_all,dims=2))./sqrt(600)
    ax.fill_between(k_list, med_slq-2*stderror, med_slq+2*stderror, alpha=0.3, color=p[1].get_color())
    ax.text(4520,med_slq[end]+offset[i],"SLQ (q=$q)",fontsize=14)
    ax.plot([4370,4470],[med_slq[end],med_slq[end]+offset[i]+0.01],color="k")
    @show med_slq[end]
    q = q_list_degnorm[i]
    rc_slq_all_degnorm = rc_slq_records_degnorm[string(q)]
    med_slq_degnorm = vec(median(rc_slq_all_degnorm,dims=2))
    p = ax.plot(k_list,med_slq_degnorm)
    stderror = vec(std(rc_slq_all_degnorm,dims=2))./sqrt(600)
    ax.fill_between(k_list, med_slq_degnorm-2*stderror, med_slq_degnorm+2*stderror, alpha=0.3, color=p[1].get_color())
    ax.plot([4370,4470],[med_slq_degnorm[end],med_slq_degnorm[end]+offset_degnorm[i]+0.025],color="k")
    ax.text(4520,med_slq_degnorm[end]+offset_degnorm[i]+0.015,"SLQ normalized (q=$q)",fontsize=14)
    @show med_slq_degnorm[end]
end


for tick in ax.xaxis.get_major_ticks()
    tick.label.set_fontsize(18)
end
for tick in ax.yaxis.get_major_ticks()
    tick.label.set_fontsize(18)
end

med_acl = vec(median(rc_acl_all,dims=2))
med_acl_degnorm = vec(median(rc_acl_all_degnorm,dims=2))
p = ax.plot(k_list,med_acl)
stderror = vec(std(rc_acl_all,dims=2))./sqrt(600)
ax.fill_between(k_list, med_acl-2*stderror, med_acl+2*stderror, alpha=0.3, color=p[1].get_color())
p = ax.plot(k_list,med_acl_degnorm)
stderror = vec(std(rc_acl_all_degnorm,dims=2))./sqrt(600)
ax.fill_between(k_list, med_acl_degnorm-2*stderror, med_acl_degnorm+2*stderror, alpha=0.3, color=p[1].get_color())
ax.text(4520,med_acl[end]-0.03,"ACL",fontsize=14)
ax.plot([4370,4470],[med_acl[end],med_acl[end]-0.02],color="k")
ax.plot([4370,4470],[med_acl_degnorm[end],med_acl_degnorm[end]-0.01],color="k")
ax.text(4520,med_acl_degnorm[end]-0.04,"ACL normalized",fontsize=1)
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

fig.savefig("figures/600-communities-$dataset.pdf",format="pdf", bbox_inches="tight")
