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

colors = ["#1f77b4", "#ff7f0e", "#d62728", "purple","brown","#2ca02c","grey"]
records_crd_3 = load("results/sparsity_runtime_crd_h_3.jld2")
records_crd_5 = load("results/sparsity_runtime_crd_h_5.jld2")
records_acl = load("results/sparsity_runtime_acl.jld2")
records_slq = load("results/sparsity_runtime_slq.jld2")
records_hk = load("results/sparsity_runtime_hk.jld2")

fig,plt_axes = subplots(1,3,figsize=(15,3))
plt_axes[1].set_yscale("log")
plt_axes[1].set_xscale("log")

n_list= [4000,5000,6000,8000,10000,20000,40000,60000,80000,100000,150000,200000]


running_time = []
error_20 = []
error_80 = []
for (i,n) in enumerate(n_list)
    key = join((1.2,0.0,n*1.0,0.3),",")
    record = records_slq[key]
    push!(running_time,median([tmp[1] for tmp in record]))
    push!(error_20,quantile([tmp[1] for tmp in record],0.2))
    push!(error_80,quantile([tmp[1] for tmp in record],0.8))
end
plt_axes[1].plot(n_list,running_time,color=colors[1],zorder=10)
plt_axes[1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[1],zorder=10)


running_time = []
error_20 = []
error_80 = []
for (i,n) in enumerate(n_list)
    key = join((1.4,0.0,n*1.0,0.3),",")
    record = records_slq[key]
    push!(running_time,median([tmp[1] for tmp in record]))
    push!(error_20,quantile([tmp[1] for tmp in record],0.2))
    push!(error_80,quantile([tmp[1] for tmp in record],0.8))
end
plt_axes[1].plot(n_list,running_time,color=colors[2])
plt_axes[1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[2])



running_time = []
error_20 = []
error_80 = []
for (i,n) in enumerate(n_list)
    key = join((1.6,0.0,n*1.0,0.3),",")
    record = records_slq[key]
    push!(running_time,median([tmp[1] for tmp in record]))
    push!(error_20,quantile([tmp[1] for tmp in record],0.2))
    push!(error_80,quantile([tmp[1] for tmp in record],0.8))
end
plt_axes[1].plot(n_list,running_time,color=colors[3])
plt_axes[1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[3])


running_time = []
error_20 = []
error_80 = []
for (i,n) in enumerate(n_list)
    key = join((n,0.3),",")
    record = records_crd_3[key]
    push!(running_time,median([tmp[1] for tmp in record]))
    push!(error_20,quantile([tmp[1] for tmp in record],0.2))
    push!(error_80,quantile([tmp[1] for tmp in record],0.8))
end
plt_axes[1].plot(n_list,running_time,color=colors[4])
plt_axes[1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[4])


running_time = []
error_20 = []
error_80 = []
for (i,n) in enumerate(n_list)
    key = join((n,0.3),",")
    record = records_crd_5[key]
    push!(running_time,median([tmp[1] for tmp in record]))
    push!(error_20,quantile([tmp[1] for tmp in record],0.2))
    push!(error_80,quantile([tmp[1] for tmp in record],0.8))
end
plt_axes[1].plot(n_list,running_time,color=colors[5])
plt_axes[1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[5])


running_time = []
error_20 = []
error_80 = []
for (i,n) in enumerate(n_list)
    key = join((n,0.3),",")
    record = records_acl[key]
    push!(running_time,median([tmp[1] for tmp in record]))
    push!(error_20,quantile([tmp[1] for tmp in record],0.2))
    push!(error_80,quantile([tmp[1] for tmp in record],0.8))
end
plt_axes[1].plot(n_list,running_time,color=colors[6])
plt_axes[1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[6])


running_time = []
error_20 = []
error_80 = []
for (i,n) in enumerate(n_list)
    key = join((n,0.3),",")
    record = records_hk[key]
    push!(running_time,median([tmp[1] for tmp in record]))
    push!(error_20,quantile([tmp[1] for tmp in record],0.2))
    push!(error_80,quantile([tmp[1] for tmp in record],0.8))
end
plt_axes[1].plot(n_list,running_time,color=colors[7])
plt_axes[1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[7])










mu_list = [0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5]
n = 10000
records_crd_3 = load("results/LFR_f1_cond_crd_h_3.jld2")
records_crd_5 = load("results/LFR_f1_cond_crd_h_5.jld2")
records_acl = load("results/LFR_f1_cond_acl.jld2")
records_slq = load("results/LFR_f1_cond_slq.jld2")
records_hk = load("results/LFR_f1_cond_hk.jld2")

f1 = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((1.2,0.0,n*1.0,mu),",")
    record = records_slq[key]
    push!(f1,median([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record]))
    push!(error_20,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.2))
    push!(error_80,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.8))
end
plt_axes[2].plot(mu_list,f1,color=colors[1],zorder=10)
plt_axes[2].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[1],zorder=10)


f1 = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((1.4,0.0,n*1.0,mu),",")
    record = records_slq[key]
    push!(f1,median([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record]))
    push!(error_20,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.2))
    push!(error_80,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.8))
end
plt_axes[2].plot(mu_list,f1,color=colors[2])
plt_axes[2].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[2])



f1 = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((1.6,0.0,n*1.0,mu),",")
    record = records_slq[key]
    push!(f1,median([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record]))
    push!(error_20,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.2))
    push!(error_80,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.8))
end
plt_axes[2].plot(mu_list,f1,color=colors[3])
plt_axes[2].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[3])



f1 = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((n,mu),",")
    record = records_crd_3[key]
    push!(f1,median([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record]))
    push!(error_20,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.2))
    push!(error_80,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.8))
end
plt_axes[2].plot(mu_list,f1,color=colors[4])
plt_axes[2].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[4])



f1 = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((n,mu),",")
    record = records_crd_5[key]
    push!(f1,median([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record]))
    push!(error_20,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.2))
    push!(error_80,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.8))
end
plt_axes[2].plot(mu_list,f1,color=colors[5])
plt_axes[2].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[5])



f1 = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((n,mu),",")
    record = records_acl[key]
    push!(f1,median([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record]))
    push!(error_20,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.2))
    push!(error_80,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.8))
end
plt_axes[2].plot(mu_list,f1,color=colors[6])
plt_axes[2].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[6])



f1 = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((n,mu),",")
    record = records_hk[key]
    push!(f1,median([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record]))
    push!(error_20,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.2))
    push!(error_80,quantile([tmp[4]*tmp[5]*2/(tmp[4]+tmp[5]) for tmp in record],0.8))
end
plt_axes[2].plot(mu_list,f1,color=colors[7])
plt_axes[2].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[7])













cond = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((1.2,0.0,n*1.0,mu),",")
    record = records_slq[key]
    push!(cond,median([tmp[3] for tmp in record]))
    push!(error_20,quantile([tmp[3] for tmp in record],0.2))
    push!(error_80,quantile([tmp[3] for tmp in record],0.8))
end
plt_axes[3].plot(mu_list,cond,color=colors[1],zorder=10)
plt_axes[3].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[1],zorder=10)


cond = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((1.4,0.0,n*1.0,mu),",")
    record = records_slq[key]
    push!(cond,median([tmp[3] for tmp in record]))
    push!(error_20,quantile([tmp[3] for tmp in record],0.2))
    push!(error_80,quantile([tmp[3] for tmp in record],0.8))
end
plt_axes[3].plot(mu_list,cond,color=colors[2])
plt_axes[3].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[2])



cond = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((1.6,0.0,n*1.0,mu),",")
    record = records_slq[key]
    push!(cond,median([tmp[3] for tmp in record]))
    push!(error_20,quantile([tmp[3] for tmp in record],0.2))
    push!(error_80,quantile([tmp[3] for tmp in record],0.8))
end
plt_axes[3].plot(mu_list,cond,color=colors[3])
plt_axes[3].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[3])



cond = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((n,mu),",")
    record = records_crd_3[key]
    push!(cond,median([tmp[3] for tmp in record]))
    push!(error_20,quantile([tmp[3] for tmp in record],0.2))
    push!(error_80,quantile([tmp[3] for tmp in record],0.8))
end
plt_axes[3].plot(mu_list,cond,color=colors[4])
plt_axes[3].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[4])



cond = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((n,mu),",")
    record = records_crd_5[key]
    push!(cond,median([tmp[3] for tmp in record]))
    push!(error_20,quantile([tmp[3] for tmp in record],0.2))
    push!(error_80,quantile([tmp[3] for tmp in record],0.8))
end
plt_axes[3].plot(mu_list,cond,color=colors[5])
plt_axes[3].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[5])



cond = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((n,mu),",")
    record = records_acl[key]
    push!(cond,median([tmp[3] for tmp in record]))
    push!(error_20,quantile([tmp[3] for tmp in record],0.2))
    push!(error_80,quantile([tmp[3] for tmp in record],0.8))
end
plt_axes[3].plot(mu_list,cond,color=colors[6])
plt_axes[3].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[6])



cond = []
error_20 = []
error_80 = []
for (i,mu) in enumerate(mu_list)
    key = join((n,mu),",")
    record = records_hk[key]
    push!(cond,median([tmp[3] for tmp in record]))
    push!(error_20,quantile([tmp[3] for tmp in record],0.2))
    push!(error_80,quantile([tmp[3] for tmp in record],0.8))
end
plt_axes[3].plot(mu_list,cond,color=colors[7])
plt_axes[3].fill_between(mu_list, error_20, error_80, alpha=0.3, color=colors[7])

plt_axes[1].set_xlabel("nodes",fontsize=18)
plt_axes[1].xaxis.set_label_coords(1.05, -0.025)
plt_axes[2].set_xlabel("\$\\mu\$",fontsize=18)
plt_axes[2].xaxis.set_label_coords(1.1, -0.025)
plt_axes[3].set_xlabel("\$\\mu\$",fontsize=18)
plt_axes[3].xaxis.set_label_coords(1.1, -0.025)
plt_axes[1].set_title("Running time (seconds)",fontsize=18)
plt_axes[2].set_title("F1 score",fontsize=18)
plt_axes[3].set_title("conductance",fontsize=18)

for ax in plt_axes
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
end

for ax in plt_axes
    for tick in ax.xaxis.get_major_ticks()
        tick.label.set_fontsize(18)
    end
    for tick in ax.yaxis.get_major_ticks()
        tick.label.set_fontsize(18)
    end
end


handles = [mpatches.Patch(color=color) for color in colors]
lgd = plt_axes[2].legend(labels = ["SLQ (q=1.2)","SLQ (q=1.4)","SLQ (q=1.6)","CRD (h=3)","CRD (h=5)","ACL","heat kernel"],
            handles=handles,fancybox=true,shadow=true,loc=9,bbox_to_anchor=(0.5, 1.6),ncol=4,fontsize=18)
fig.savefig("figures/running_time.pdf",format="pdf",bbox_extra_artists=(lgd,), bbox_inches="tight")
# ## CRD
# running_time = []
# error_20 = []
# error_80 = []
# for (i,n) in enumerate(n_list)
#     key = join((n,0.3),",")
#     record = records_crd[key]
#     push!(running_time,median([tmp[1] for tmp in record]))
#     push!(error_20,quantile([tmp[1] for tmp in record],0.2))
#     push!(error_80,quantile([tmp[1] for tmp in record],0.8))
# end
# plt_axes[1].plot(n_list,running_time,color=colors[1])
# plt_axes[1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[1])
# record = records_crd[join((nsmall,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([1],median(f1_score),color=colors[1],s=60,marker="s")
# record = records_crd[join((nlarge,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([1.5],median(f1_score),color=colors[1],s=60)

# record = records_crd[join((nsmall,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([1],median(cond),color=colors[1],s=60,marker="s")
# record = records_crd[join((nlarge,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([1.5],median(cond),color=colors[1],s=60)


# ## ACL
# running_time = []
# error_20 = []
# error_80 = []
# for (i,n) in enumerate(n_list)
#     key = join((n,0.3),",")
#     record = records_acl[key]
#     push!(running_time,median([tmp[1] for tmp in record]))
#     push!(error_20,quantile([tmp[1] for tmp in record],0.2))
#     push!(error_80,quantile([tmp[1] for tmp in record],0.8))
# end
# plt_axes[1,1].plot(n_list,running_time,color=colors[2])
# plt_axes[1,1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[2])
# record = records_acl[join((nsmall,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([2],median(f1_score),color=colors[2],s=60,marker="s")
# record = records_acl[join((nlarge,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([2.5],median(f1_score),color=colors[2],s=60)

# record = records_acl[join((nsmall,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([2],median(cond),color=colors[2],s=60,marker="s")
# record = records_acl[join((nlarge,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([2.5],median(cond),color=colors[2],s=60)


# ## slq-1.2
# running_time = []
# error_20 = []
# error_80 = []
# for (i,n) in enumerate(n_list)
#     key = join((1.1,0.0,n*1.0,0.3),",")
#     record = records_slq[key]
#     push!(running_time,median([tmp[1] for tmp in record]))
#     push!(error_20,quantile([tmp[1] for tmp in record],0.2))
#     push!(error_80,quantile([tmp[1] for tmp in record],0.8))
# end
# plt_axes[1,1].plot(n_list,running_time,color=colors[3])
# plt_axes[1,1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[3])
# record = records_slq[join((1.1,0.0,nsmall*1.0,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([3],median(f1_score),color=colors[3],s=60,marker="s")
# record = records_slq[join((1.1,0.0,nlarge*1.0,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([3.5],median(f1_score),color=colors[3],s=60)

# record = records_slq[join((1.1,0.0,nsmall*1.0,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([3],median(cond),color=colors[3],s=60,marker="s")
# record = records_slq[join((1.1,0.0,nlarge*1.0,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([3.5],median(cond),color=colors[3],s=60)

# ## slq-1.4
# running_time = []
# error_20 = []
# error_80 = []
# for (i,n) in enumerate(n_list)
#     key = join((1.4,0.0,n*1.0,0.3),",")
#     record = records_slq[key]
#     push!(running_time,median([tmp[1] for tmp in record]))
#     push!(error_20,quantile([tmp[1] for tmp in record],0.2))
#     push!(error_80,quantile([tmp[1] for tmp in record],0.8))
# end
# plt_axes[1,1].plot(n_list,running_time,color=colors[4])
# plt_axes[1,1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[4])
# record = records_slq[join((1.4,0.0,nsmall*1.0,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([4],median(f1_score),color=colors[4],s=60,marker="s")
# record = records_slq[join((1.4,0.0,nlarge*1.0,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([4.5],median(f1_score),color=colors[4],s=60)

# record = records_slq[join((1.4,0.0,nsmall*1.0,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([4],median(cond),color=colors[4],s=60,marker="s")
# record = records_slq[join((1.4,0.0,nlarge*1.0,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([4.5],median(cond),color=colors[4],s=60)

# ## slq-1.6
# running_time = []
# error_20 = []
# error_80 = []
# for (i,n) in enumerate(n_list)
#     key = join((1.6,0.0,n*1.0,0.3),",")
#     record = records_slq[key]
#     push!(running_time,median([tmp[1] for tmp in record]))
#     push!(error_20,quantile([tmp[1] for tmp in record],0.2))
#     push!(error_80,quantile([tmp[1] for tmp in record],0.8))
# end
# plt_axes[1,1].plot(n_list,running_time,color=colors[5])
# plt_axes[1,1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[5])
# record = records_slq[join((1.6,0.0,nsmall*1.0,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([5],median(f1_score),color=colors[5],s=60,marker="s")
# record = records_slq[join((1.6,0.0,nlarge*1.0,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([5.5],median(f1_score),color=colors[5],s=60)

# record = records_slq[join((1.6,0.0,nsmall*1.0,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([5],median(cond),color=colors[5],s=60,marker="s")
# record = records_slq[join((1.6,0.0,nlarge*1.0,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([5.5],median(cond),color=colors[5],s=60)

# ## hk
# running_time = []
# error_20 = []
# error_80 = []
# for (i,n) in enumerate(n_list)
#     key = join((n,0.3),",")
#     record = records_hk[key]
#     push!(running_time,median([tmp[1] for tmp in record]))
#     push!(error_20,quantile([tmp[1] for tmp in record],0.2))
#     push!(error_80,quantile([tmp[1] for tmp in record],0.8))
# end
# plt_axes[1,1].plot(n_list,running_time,color=colors[6])
# plt_axes[1,1].fill_between(n_list, error_20, error_80, alpha=0.3, color=colors[6])
# record = records_hk[join((nsmall,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([6],median(f1_score),color=colors[6],s=60,marker="s")
# record = records_hk[join((nlarge,0.3),",")]
# f1_score = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(f1_score,2*tmp[4]*tmp[5]/(tmp[4]+tmp[5]))
#     end
# end
# plt_axes[2].scatter([6.5],median(f1_score),color=colors[6],s=60)

# record = records_acl[join((nsmall,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([6],median(cond),color=colors[6],s=60,marker="s")
# record = records_hk[join((nlarge,0.3),",")]
# cond = []
# for tmp in record
#     if tmp[4]*tmp[5] != 0
#         push!(cond,tmp[3])
#     end
# end
# plt_axes[3].scatter([6.5],median(cond),color=colors[6],s=60)

# handles = [mpatches.Patch(color=color) for color in colors]
# lgd = plt_axes[2].legend(labels = ["CRD","ACL","SLQ (q=1.2)","SLQ (q=1.4)","SLQ (q=1.6)","heat kernel"],
#             handles=handles,fancybox=true,shadow=true,loc=9,bbox_to_anchor=(0.5, 1.45),ncol=6,fontsize=18)
# for ax in plt_axes[2:end]
#     ax.spines["top"].set_visible(false)
#     ax.spines["right"].set_visible(false)
#     # ax.spines["left"].set_visible(false)
#     ax.spines["bottom"].set_visible(false)
#     ax.xaxis.set_ticklabels([])
#     ax.tick_params(axis="x", which="both", length=0)
# end
# ax = plt_axes[1]
# ax.spines["top"].set_visible(false)
# ax.spines["right"].set_visible(false)
# plt_axes[1].set_title("Running time (seconds)",fontsize=18)
# plt_axes[2].set_title("F1 score",fontsize=18)
# plt_axes[3].set_title("conductance",fontsize=18)
# for ax in plt_axes
#     for tick in ax.xaxis.get_major_ticks()
#         tick.label.set_fontsize(18)
#     end
#     for tick in ax.yaxis.get_major_ticks()
#         tick.label.set_fontsize(18)
#     end
# end

# fig.savefig("figures/running_time.pdf",format="pdf",bbox_extra_artists=(lgd,), bbox_inches="tight")
