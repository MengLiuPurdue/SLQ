using JLD2,FileIO
using PyPlot
using Statistics


#datasets = ["Colgate88.mat","Rice31.mat","Johns Hopkins55.mat","Simmons81.mat"]

records_crd_3 = load("results/records_crd_h_3.jld2")
records_crd_5 = load("results/records_crd_h_5.jld2")
records_acl = load("results/records_acl.jld2")
records_slq_0 = load("results/records_slq_0_q_1.2.jld2")
records_slq_1 = load("results/records_slq_1_q_1.2.jld2")
records_hk = load("results/records_hk.jld2")
records_nonlinear_diffusion = load("results/records_nonlinear_diffusion-0.5.jld2")
records_sl = load("results/records_sl_5.jld2")
records_gcn = load("results/records-gcn.jld2")

years = [2008,2009]
datasets = ["UCLA26.mat","Brown11.mat","Duke14.mat","UPenn7.mat","Yale4.mat","Stanford3.mat","MIT8.mat","Cornell5.mat"]
colors = ["#F2AA4CFF" for i =1:100]

function myplot(f1)
    fig,ax = subplots(1,1,figsize=(4,2))
    patches = ax.violinplot(f1,vert=false,showextrema=false)
    patch = patches["bodies"][1]
    patch.set_facecolor(colors[1])
    patch.set_edgecolor("k")
    patch.set_alpha(0.6)
    ax.plot([minimum(f1),maximum(f1)],[1,1],color="k")

    # some simple heuristics for the scale
    xmin,xmax = extrema(f1)
    xmin,xmax = 0,1
    ax.set_xlim(xmin,xmax)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.yaxis.set_ticklabels([])
    ax.tick_params(axis="y", which="both", length=0)
    ax.xaxis.set_ticklabels([])
    ax.tick_params(axis="x", which="both", length=0)
    fig.tight_layout(pad=0)
    return fig
end

function get_f1(records)
    f1 = []
    for tmp in records
        if tmp[2] < 1 && tmp[3]*tmp[4] > 0
            push!(f1,2*tmp[3]*tmp[4]/(tmp[3]+tmp[4]))
        end
    end
    return f1
end

for (records,method) in [
     (records_slq_0, "slq-0"),
     (records_slq_1, "slq-1"),
     (records_crd_3, "crd-h-3"),
     (records_crd_5, "crd-h-5"),
     (records_acl, "acl"),
     (records_hk, "hk"),
     (records_nonlinear_diffusion, "nld"),
     (records_sl, "sl-5"),
     (records_gcn, "gcn")
    ]

    for file in datasets
        for year in years
            f1 = get_f1(records[file][year])
            fig = myplot(f1)
            name = split(file,".")[1]
            year = Int(year)
            fig.savefig("figures/$name-$year-$method.pdf",bbox_inches="tight",pad_inches=0)
        end
    end
end
