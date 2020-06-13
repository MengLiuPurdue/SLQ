using LightGraphs
using DelimitedFiles
using SparseArrays
using MAT
using Random
using JLD2,FileIO
using PyCall
using PyPlot
using Statistics

include("PageRank.jl")
include("SLQcvx.jl") # this includes SLQ.jl
include("common.jl")
include("CRDlgc.jl")
include("LocalClusteringObjs.jl")

#datasets = ["Colgate88.mat","Rice31.mat","Johns Hopkins55.mat","Simmons81.mat"]

records_crd_3 = load("results/records-crd-h-3.jld2")
records_crd_5 = load("results/records-crd-h-5.jld2")
records_acl = load("results/records-acl.jld2")
records_slq_0 = load("results/records-slq-0.jld2")
records_slq_1 = load("results/records-slq-1.jld2")
records_hk = load("results/records-hk.jld2")
records_nonlinear_diffusion = load("results/records-nld.jld2")
records_sl = load("results/records-sl-5.jld2")
records_gcn = load("results/records-gcn.jld2")


datasets = ["UCLA26.mat","Brown11.mat","Duke14.mat","UPenn7.mat","Yale4.mat","Stanford3.mat","MIT8.mat","Cornell5.mat"]
records = [records_crd_3,records_acl,records_slq_0,records_slq_1,records_hk,records_nonlinear_diffusion,records_sl]
years = [2008,2009]
ntrials = 50
nmethods = length(records)
slq_ids = [3,4]

color = "#F2AA4CFF"


for file in datasets
    for year in years
        fig,ax = subplots(1,1,figsize=(4,2))
        f1 = []
        for tmp in records_slq_0[file][year]
            if tmp[2] < 1
                push!(f1,2*tmp[3]*tmp[4]/(tmp[3]+tmp[4]))
            end
        end
        patches = ax.violinplot(f1,vert=false,showextrema=false)
        patch = patches["bodies"][1]
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(0.6)
        ax.plot([minimum(f1),maximum(f1)],[1,1],color="k")
        ax.set_xlim(minimum(f1),maximum(f1))
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.xaxis.set_ticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        fig.tight_layout(pad=0)
        name = split(file,".")[1]
        year = Int(year)
        val = round(median(f1),digits=1)
        println("$name $year $val")
        fig.savefig("figures/$name-$year-slq-0.pdf",bbox_inches="tight",pad_inches=0)
    end
end


for file in datasets
    for year in years
        fig,ax = subplots(1,1,figsize=(4,2))
        f1 = []
        for tmp in records_slq_1[file][year]
            if tmp[2] < 1
                push!(f1,2*tmp[3]*tmp[4]/(tmp[3]+tmp[4]))
            end
        end
        patches = ax.violinplot(f1,vert=false,showextrema=false)
        patch = patches["bodies"][1]
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(0.6)
        ax.plot([minimum(f1),maximum(f1)],[1,1],color="k")
        ax.set_xlim(minimum(f1),maximum(f1))
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.xaxis.set_ticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        fig.tight_layout(pad=0)
        name = split(file,".")[1]
        year = Int(year)
        val = round(median(f1),digits=1)
        println("$name $year $val")
        fig.savefig("figures/$name-$year-slq-1.pdf",bbox_inches="tight",pad_inches=0)
    end
end




for file in datasets
    for year in years
        fig,ax = subplots(1,1,figsize=(4,2))
        f1 = []
        for tmp in records_crd_3[file][year]
            if tmp[2] < 1 && tmp[3]*tmp[4] > 0
                push!(f1,2*tmp[3]*tmp[4]/(tmp[3]+tmp[4]))
            end
        end
        patches = ax.violinplot(f1,vert=false,showextrema=false)
        patch = patches["bodies"][1]
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(0.6)
        ax.plot([minimum(f1),maximum(f1)],[1,1],color="k")
        ax.set_xlim(minimum(f1),maximum(f1))
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.xaxis.set_ticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        fig.tight_layout(pad=0)
        name = split(file,".")[1]
        year = Int(year)
        val = round(median(f1),digits=1)
        println("$name $year $val")
        fig.savefig("figures/$name-$year-crd-h-3.pdf",bbox_inches="tight",pad_inches=0)
    end
end





for file in datasets
    for year in years
        fig,ax = subplots(1,1,figsize=(4,2))
        f1 = []
        for tmp in records_crd_5[file][year]
            if tmp[2] < 1 && tmp[3]*tmp[4] > 0
                push!(f1,2*tmp[3]*tmp[4]/(tmp[3]+tmp[4]))
            end
        end
        patches = ax.violinplot(f1,vert=false,showextrema=false)
        patch = patches["bodies"][1]
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(0.6)
        ax.plot([minimum(f1),maximum(f1)],[1,1],color="k")
        ax.set_xlim(minimum(f1),maximum(f1))
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.xaxis.set_ticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        fig.tight_layout(pad=0)
        name = split(file,".")[1]
        year = Int(year)
        val = round(median(f1),digits=1)
        println("$name $year $val")
        fig.savefig("figures/$name-$year-crd-h-5.pdf",bbox_inches="tight",pad_inches=0)
    end
end






for file in datasets
    for year in years
        fig,ax = subplots(1,1,figsize=(4,2))
        f1 = []
        for tmp in records_acl[file][year]
            if tmp[2] < 1 && tmp[3]*tmp[4] > 0
                push!(f1,2*tmp[3]*tmp[4]/(tmp[3]+tmp[4]))
            end
        end
        patches = ax.violinplot(f1,vert=false,showextrema=false)
        patch = patches["bodies"][1]
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(0.6)
        ax.plot([minimum(f1),maximum(f1)],[1,1],color="k")
        ax.set_xlim(minimum(f1),maximum(f1))
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.xaxis.set_ticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        fig.tight_layout(pad=0)
        name = split(file,".")[1]
        year = Int(year)
        val = round(median(f1),digits=1)
        println("$name $year $val")
        fig.savefig("figures/$name-$year-acl.pdf",bbox_inches="tight",pad_inches=0)
    end
end






for file in datasets
    for year in years
        fig,ax = subplots(1,1,figsize=(4,2))
        f1 = []
        for tmp in records_hk[file][year]
            if tmp[2] < 1 && tmp[3]*tmp[4] > 0
                push!(f1,2*tmp[3]*tmp[4]/(tmp[3]+tmp[4]))
            end
        end
        patches = ax.violinplot(f1,vert=false,showextrema=false)
        patch = patches["bodies"][1]
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(0.6)
        ax.plot([minimum(f1),maximum(f1)],[1,1],color="k")
        ax.set_xlim(minimum(f1),maximum(f1))
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.xaxis.set_ticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        fig.tight_layout(pad=0)
        name = split(file,".")[1]
        year = Int(year)
        val = round(median(f1),digits=1)
        println("$name $year $val")
        fig.savefig("figures/$name-$year-hk.pdf",bbox_inches="tight",pad_inches=0)
    end
end












for file in datasets
    for year in years
        fig,ax = subplots(1,1,figsize=(4,2))
        f1 = []
        for tmp in records_nonlinear_diffusion[file][year]
            if tmp[2] < 1 && tmp[3]*tmp[4] > 0
                push!(f1,2*tmp[3]*tmp[4]/(tmp[3]+tmp[4]))
            end
        end
        patches = ax.violinplot(f1,vert=false,showextrema=false)
        patch = patches["bodies"][1]
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(0.6)
        ax.plot([minimum(f1),maximum(f1)],[1,1],color="k")
        ax.set_xlim(minimum(f1),maximum(f1))
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.xaxis.set_ticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        fig.tight_layout(pad=0)
        name = split(file,".")[1]
        year = Int(year)
        val = round(median(f1),digits=1)
        println("$name $year $val")
        fig.savefig("figures/$name-$year-nld.pdf",bbox_inches="tight",pad_inches=0)
    end
end







for file in datasets
    for year in years
        fig,ax = subplots(1,1,figsize=(4,2))
        f1 = []
        for tmp in records_sl[file][year]
            if tmp[2] < 1 && tmp[3]*tmp[4] > 0
                push!(f1,2*tmp[3]*tmp[4]/(tmp[3]+tmp[4]))
            end
        end
        patches = ax.violinplot(f1,vert=false,showextrema=false)
        patch = patches["bodies"][1]
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(0.6)
        ax.plot([minimum(f1),maximum(f1)],[1,1],color="k")
        ax.set_xlim(minimum(f1),maximum(f1))
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.xaxis.set_ticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        fig.tight_layout(pad=0)
        name = split(file,".")[1]
        year = Int(year)
        val = round(median(f1),digits=1)
        println("$name $year $val")
        fig.savefig("figures/$name-$year-sl-5.pdf",bbox_inches="tight",pad_inches=0)
    end
end



for file in datasets
    for year in years
        fig,ax = subplots(1,1,figsize=(4,2))
        f1 = []
        for tmp in records_gcn[file][year]
            if tmp[2] < 1 && tmp[3]*tmp[4] > 0
                push!(f1,2*tmp[3]*tmp[4]/(tmp[3]+tmp[4]))
            end
        end
        patches = ax.violinplot(f1,vert=false,showextrema=false)
        patch = patches["bodies"][1]
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(0.6)
        ax.plot([minimum(f1),maximum(f1)],[1,1],color="k")
        ax.set_xlim(minimum(f1),maximum(f1))
        ax.spines["top"].set_visible(false)
        ax.spines["right"].set_visible(false)
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", which="both", length=0)
        ax.xaxis.set_ticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        fig.tight_layout(pad=0)
        name = split(file,".")[1]
        year = Int(year)
        val = round(median(f1),digits=1)
        println("$name $year $val")
        fig.savefig("figures/$name-$year-gcn.pdf",bbox_inches="tight",pad_inches=0)
    end
end



# fig,plt_axes = subplots(length(years),length(datasets[1]),figsize=(20,5),gridspec_kw=Dict("wspace"=>0.1))
# for (j,row_dataset) in enumerate(datasets)
#     for (k,dataset) in enumerate(row_dataset)
#         file = dataset
#         ax = plt_axes[j,k]
#         ax.spines["top"].set_visible(false)
#         ax.spines["right"].set_visible(false)
#         # ax.spines["left"].set_visible(false)
#         ax.spines["bottom"].set_visible(false)
#         for (y,year) in enumerate(years)
#             tmp = zeros(ntrials,length(records))
#             invalid_i = []
#             for i in 1:ntrials
#                 for (f,record) in enumerate(records)
#                     pr,rc = record[file][year*1.0][i][3],record[file][year*1.0][i][4]
#                     tmp[i,f] = 2*pr*rc/(pr+rc)
#                     if 2*pr*rc/(pr+rc) == 1 || pr*rc == 0
#                         push!(invalid_i,i)
#                     end
#                 end
#             end
#             colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "purple","brown"]
#             valid_i = setdiff(1:ntrials,invalid_i)
#             tmp = tmp[valid_i,:]
#             pos = 2*collect(1:length(records)).+0.6*(y-1)
#             bp = ax.violinplot(tmp,positions=pos,showextrema=false)
#             ax.vlines(pos,minimum(tmp,dims=1),maximum(tmp,dims=1))
#             ax.xaxis.set_ticklabels([])
#             if k != 1
#                 ax.yaxis.set_ticklabels([])
#                 ax.tick_params(axis="y", which="both", length=0)
#                 ax.spines["left"].set_visible(false)
#             end
#             ax.tick_params(axis="x", which="both", length=0)
#             for (l,patch) in enumerate(bp["bodies"])
#                 patch.set_facecolor(colors[l])
#                 patch.set_edgecolor("k")
#                 patch.set_alpha(0.6)
#             end
#             if k == 1 && j == 2
#                 plt_axes[1,2].legend(labels = ["CRD","ACL","SLQ (\$\\delta=0\$)","SLQ (\$\\delta=10^{-5}\$)","heat kernel","nonlin-diff"],
#                 handles=bp["bodies"],fancybox=true,shadow=true,loc=(-1.0,1.05),ncol=6,fontsize=18)
#             end
#         end
#         for tick in ax.xaxis.get_major_ticks()
#             tick.label.set_fontsize(18)
#         end
#         for tick in ax.yaxis.get_major_ticks()
#             tick.label.set_fontsize(18)
#         end
#         ax.set_ylim(0,1)
#     end
# end
# plt_axes[1,1].set_xlabel("UCLA",fontsize=20)
# plt_axes[1,2].set_xlabel("Brown",fontsize=20)
# plt_axes[1,3].set_xlabel("Duke",fontsize=20)
# plt_axes[1,4].set_xlabel("UPenn",fontsize=20)
# plt_axes[2,1].set_xlabel("Yale",fontsize=20)
# plt_axes[2,2].set_xlabel("Stanford",fontsize=20)
# plt_axes[2,3].set_xlabel("MIT",fontsize=20)
# plt_axes[2,4].set_xlabel("Cornell",fontsize=20)
# fig.savefig("figures/facebook_f1_score_h_5.pdf",format="pdf",bbox_inches="tight")
# gcf()

# # save("records_nonlinear_diffusion-1.5.jld2",records_nonlinear_diffusion)

# # save("records_nonlinear_diffusion-1.2.jld2",records_nonlinear_diffusion)

# # save("records_nonlinear_Lp_diffusion.jld2",records_nonlinear_Lp_diffusion)

# minimum(rand(3,3),dims=1)