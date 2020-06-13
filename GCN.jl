# currently, this assumes pygcn has been downloaded and installed from
# https://github.com/tkipf/pygcn

module GCN
using PyCall, LinearAlgebra
include("FlowSeed-1.0.jl")
function __init__()
    push!(PyVector(pyimport("sys")."path"),"")
    py"""
    import numpy as np
    import scipy.sparse as sp
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    import time

    from pygcn.utils import load_data, accuracy
    from pygcn.models import GCN

    import torch
    torch.set_num_threads(4)
    
    def test():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))
    
    def prediction(output, labels):
        preds = output.max(1)[1].type_as(labels)
        return preds

    def runGCN(indptr,indices,data,n,seeds,nonseeds,features,labels,num_epoch=200):
        labels = torch.LongTensor(labels)
        features = torch.FloatTensor(features.astype(np.float))
        adj = sp.csr_matrix((data,indices,indptr), shape=(n,n), dtype=np.int64)
        idx_train = seeds[0:int(0.8*len(seeds))].tolist()+nonseeds[0:int(0.8*len(nonseeds))].tolist()
        idx_val = seeds[int(0.8*len(seeds))::].tolist()+nonseeds[int(0.8*len(nonseeds))::].tolist()
        idx_test = range(adj.shape[0])
        idx_train = torch.tensor(idx_train)
        idx_val = torch.tensor(idx_val)
        idx_test = torch.tensor(idx_test)
        adj = adj.tocoo()
        values = adj.data
        indices = np.vstack((adj.row, adj.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj.shape
        adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        model = GCN(nfeat=features.shape[1],nhid=5,nclass=2,dropout=0.5)
        optimizer = optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)               
        t1 = time.time()
        for epoch in range(num_epoch):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj) 
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(loss_train.item()),
                    'acc_train: {:.4f}'.format(acc_train.item()),
                    'loss_val: {:.4f}'.format(loss_val.item()),
                    'acc_val: {:.4f}'.format(acc_val.item()),
                    'time: {:.4f}s'.format(time.time() - t1))
        output = model(features, adj)
        pred = prediction(output,labels)
        pred = pred.numpy()
        cluster = np.nonzero(pred)[0]
        t2 = time.time()
        return cluster+1,t2-t1
    """
    pop!(PyVector(pyimport("sys")."path"))
end

function gcn(G,seeds,nonseeds,features,labels)
    A = G.A
    indptr = A.colptr.-1
    indices = A.rowval.-1
    seeds = seeds.-1
    nonseeds = nonseeds.-1
    data = A.nzval
    n = size(A,1)
    (cluster,time_gcn) = py"runGCN"(indptr,indices,data,n,seeds,nonseeds,features,labels)
    if length(cluster) == 0
        conductance = 1.0
    else
        _,_,_,conductance = set_stats(A.*1.0,cluster,sum(G.deg)*1.0)
    end
    return cluster,conductance,time_gcn
end

end # end module