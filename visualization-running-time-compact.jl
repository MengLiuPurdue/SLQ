using DelimitedFiles
using SparseArrays
using MAT
using Random
using JLD2,FileIO
using PyCall
using PyPlot
using Statistics
@pyimport matplotlib.patches as mpatches

#=
include("PageRank.jl")
include("SLQcvx.jl") # this includes SLQ.jl
include("common.jl")
include("CRDlgc.jl")
include("LocalClusteringObjs.jl")
=#

#colors = ["#1f77b4", "#ff7f0e", "#d62728", "purple","brown","#2ca02c","grey"]
#colors = ["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0","#f0027f","#bf5b17"]
colors = ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d"]
records_crd_3 = load("results/sparsity_runtime_crd_h_3.jld2")
records_crd_5 = load("results/sparsity_runtime_crd_h_5.jld2")
records_acl = load("results/sparsity_runtime_acl.jld2")
records_slq = load("results/sparsity_runtime_slq.jld2")
records_hk = load("results/sparsity_runtime_hk.jld2")

fig,plt_axes = subplots(1,3,figsize=(15,2))
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

fontsize=14

plt_axes[1].set_xlabel("nodes",fontsize=fontsize, color="gray")
plt_axes[1].xaxis.set_label_coords(1.05, -0.025)
plt_axes[2].set_xlabel("\$\\mu\$",fontsize=fontsize, color="gray")
plt_axes[2].xaxis.set_label_coords(1.1, -0.025)
plt_axes[3].set_xlabel("\$\\mu\$",fontsize=fontsize, color="gray")
plt_axes[3].xaxis.set_label_coords(1.1, -0.025)
plt_axes[1].set_title("Running time (seconds)",fontsize=fontsize)
plt_axes[2].set_title("F1 score",fontsize=fontsize)
plt_axes[3].set_title("conductance",fontsize=fontsize)

for ax in plt_axes
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
end


function fancy_axis(ax)
    # use a gray background
    ax.set_facecolor("#E6E6E6")
    ax.set_axisbelow(true)

    # draw solid white grid lines
    ax.grid(color="w", linestyle="solid")

    # hide axis spines
    for spine in ax.spines
        spine[2].set_visible(false)
    end

    # hide top and right ticks
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    # lighten ticks and labels
    ax.tick_params(colors="gray", direction="out")
    for tick in ax.get_xticklabels()
        tick.set_color("gray")
    end
    for tick in ax.get_yticklabels()
        tick.set_color("gray")
    end
end


for ax in plt_axes
    fancy_axis(ax)
    for tick in ax.xaxis.get_major_ticks()
        tick.label.set_fontsize(fontsize)
    end
    for tick in ax.yaxis.get_major_ticks()
        tick.label.set_fontsize(fontsize)
    end
end



handles = [mpatches.Patch(color=color) for color in colors]
lgd = plt_axes[2].legend(labels = ["SLQ (q=1.2)","SLQ (q=1.4)","SLQ (q=1.6)","CRD (h=3)","CRD (h=5)","ACL","heat kernel"],
            handles=handles,frameon=false,shadow=false,loc=9,bbox_to_anchor=(0.5, 1.4),ncol=9,fontsize=12)
fig.savefig("figures/running_time_small.pdf",format="pdf",bbox_extra_artists=(lgd,), bbox_inches="tight")
