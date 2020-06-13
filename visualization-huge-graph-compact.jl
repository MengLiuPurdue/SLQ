using DelimitedFiles
using SparseArrays
using MAT
using Random
using JLD2,FileIO
using PyCall
using PyPlot
using Statistics
@pyimport matplotlib.patches as mpatches


smallfont = 10
bigfont = 13
dataset = "dblp"
#=
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
=#
rc_slq_records = load("results/$dataset-slq.jld2")
rc_slq_records_degnorm = load("results/$dataset-slq-degnorm.jld2")
rc_acl_all = load("results/$dataset-acl.jld2","acl")
rc_acl_all_degnorm = load("results/$dataset-acl-degnorm.jld2","acl")

fig,ax = subplots(1,1,figsize=(3,1.45))
q_list = [1.5,4.0,8.0]
offset = [-0.05,-0.01,0.02]
offset_degnorm = [-0.1,-0.07,-0.04]
for (i,q) in enumerate(q_list)
    rc_slq_all = rc_slq_records[string(q)]
    rc_slq_all_degnorm = rc_slq_records_degnorm[string(q)]
    med_slq = vec(median(rc_slq_all,dims=2))
    med_slq_degnorm = vec(median(rc_slq_all_degnorm,dims=2))
    k_list = collect(1:length(med_slq))
    # plot unnormalized
    p = ax.plot(k_list,med_slq)
    stderror = vec(std(rc_slq_all,dims=2))./sqrt(600)
    ax.fill_between(k_list, med_slq-2*stderror, med_slq+2*stderror, alpha=0.3, color=p[1].get_color())
    ax.text(420,med_slq[end]+offset[i],"SLQ (q=$q)",fontsize=smallfont, color=p[1].get_color())
    ax.plot([405,418],[med_slq[end],med_slq[end]+offset[i]+0.01],color=p[1].get_color())

    p = ax.plot(k_list,med_slq_degnorm)
    stderror = vec(std(rc_slq_all_degnorm,dims=2))./sqrt(600)
    ax.fill_between(k_list, med_slq_degnorm-2*stderror, med_slq_degnorm+2*stderror, alpha=0.3, color=p[1].get_color())
    ax.plot([405,418],[med_slq_degnorm[end],med_slq_degnorm[end]+offset_degnorm[i]+0.015],color=p[1].get_color())
    ax.text(420,med_slq_degnorm[end]+offset_degnorm[i],"SLQ-DN (q=$q)",color=p[1].get_color(),fontsize=smallfont)
    @show med_slq[end]
end

for tick in ax.xaxis.get_major_ticks()
    tick.label.set_fontsize(bigfont)
end
for tick in ax.yaxis.get_major_ticks()
    tick.label.set_fontsize(bigfont)
end

med_acl = vec(median(rc_acl_all,dims=2))
k_list = collect(1:length(med_acl))
med_acl_degnorm = vec(median(rc_acl_all_degnorm,dims=2))
p = ax.plot(k_list,med_acl)
stderror = vec(std(rc_acl_all,dims=2))./sqrt(600)

acl_offset = 0.04
ax.fill_between(k_list, med_acl-2*stderror, med_acl+2*stderror, alpha=0.3, color=p[1].get_color())
ax.text(420,med_acl[end]-acl_offset,"ACL",fontsize=smallfont,color=p[1].get_color())
ax.plot([405,418],[med_acl[end],med_acl[end]-acl_offset+0.01],color=p[1].get_color())

p = ax.plot(k_list,med_acl_degnorm)
stderror = vec(std(rc_acl_all_degnorm,dims=2))./sqrt(600)
ax.fill_between(k_list, med_acl_degnorm-2*stderror, med_acl_degnorm+2*stderror, alpha=0.3, color=p[1].get_color())

ax.plot([405,418],[med_acl_degnorm[end],med_acl_degnorm[end]-0.04],color=p[1].get_color())
ax.text(420,med_acl_degnorm[end]-0.08,"ACL-DN",fontsize=smallfont,color=p[1].get_color())
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)
fig.savefig("figures/600-communities-$dataset.pdf",format="pdf", bbox_inches="tight")



##
dataset = "liveJournal"
#=
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
=#
rc_slq_records = load("results/$dataset-slq.jld2")
rc_slq_records_degnorm = load("results/$dataset-slq-degnorm.jld2")
rc_acl_all = load("results/$dataset-acl.jld2","acl")
rc_acl_all_degnorm = load("results/$dataset-acl-degnorm.jld2","acl")

med_acl = vec(median(rc_acl_all,dims=2))
k_list = collect(1:10:10*length(med_acl))


fig,ax = subplots(1,1,figsize=(3,1.45))
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
    ax.text(4520,med_slq[end]+offset[i],"SLQ (q=$q)",fontsize=smallfont,color=p[1].get_color())
    ax.plot([4370,4470],[med_slq[end],med_slq[end]+offset[i]+0.01],color=color=p[1].get_color())

    @show med_slq[end]
    q = q_list_degnorm[i]
    rc_slq_all_degnorm = rc_slq_records_degnorm[string(q)]
    med_slq_degnorm = vec(median(rc_slq_all_degnorm,dims=2))
    p = ax.plot(k_list,med_slq_degnorm)
    stderror = vec(std(rc_slq_all_degnorm,dims=2))./sqrt(600)
    ax.fill_between(k_list, med_slq_degnorm-2*stderror, med_slq_degnorm+2*stderror, alpha=0.3, color=p[1].get_color())
    ax.plot([4370,4470],[med_slq_degnorm[end],med_slq_degnorm[end]+offset_degnorm[i]+0.025],color=p[1].get_color())
    ax.text(4520,med_slq_degnorm[end]+offset_degnorm[i]+0.015,"SLQ-DN (q=$q)",fontsize=smallfont,color=p[1].get_color())
    @show med_slq_degnorm[end]
end


for tick in ax.xaxis.get_major_ticks()
    tick.label.set_fontsize(bigfont)
end
for tick in ax.yaxis.get_major_ticks()
    tick.label.set_fontsize(bigfont)
end

med_acl = vec(median(rc_acl_all,dims=2))
med_acl_degnorm = vec(median(rc_acl_all_degnorm,dims=2))
p = ax.plot(k_list,med_acl)
stderror = vec(std(rc_acl_all,dims=2))./sqrt(600)
ax.fill_between(k_list, med_acl-2*stderror, med_acl+2*stderror, alpha=0.3, color=p[1].get_color())
ax.text(4520,med_acl[end]-0.03,"ACL",fontsize=smallfont,color=p[1].get_color())
ax.plot([4370,4470],[med_acl[end],med_acl[end]-0.02],color=p[1].get_color())

p = ax.plot(k_list,med_acl_degnorm)
stderror = vec(std(rc_acl_all_degnorm,dims=2))./sqrt(600)
ax.fill_between(k_list, med_acl_degnorm-2*stderror, med_acl_degnorm+2*stderror, alpha=0.3, color=p[1].get_color())


ax.plot([4370,4470],[med_acl_degnorm[end],med_acl_degnorm[end]-0.01],color=p[1].get_color())
ax.text(4520,med_acl_degnorm[end]-0.04,"ACL-DN",fontsize=smallfont,color=p[1].get_color())
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

fig.savefig("figures/600-communities-$dataset.pdf",format="pdf", bbox_inches="tight")
