# SLQ
code for SLQ project, see our arXiv paper (will update a link once it's up)

To run our code, simply `include("SLQ.jl")` This has minimal dependencies. Then
to run the code on an Erdos-Renyi graph, run

    using SparseArrays
    # make an Erdos Renyi graph
    A = triu(sprand(100,100,8/100),1)
    A = max.(A,A') # symmetrize
    fill!(A.nzval, 1) # set all values to 1. 
	G = SLQ.graph(A) # convert an adjacency matrix into a graph
	SLQ.slq_diffusion(SLQ.graph(A), 
		[1], # seed set
		 0.1, # value of gamma (regularization on seed) 
		 0.1, # value of kappa (sparsity regularization)
		 0.5, # value of rho (KKT apprx-val)
        SLQ.loss_type(1.4,0.0) # the loss-type, this is a 1.4-norm without huber)


# SLQ via CVX
We need cvxpy. This can be installed in Julia's conda-forge environment. We try to do this
when you `include("SLQcvx.jl")`. CVX does not support the q-huber penalties. This
should just work. 

# Additional experiemtns with other dependencies 
We need localgraphclustering for comparisons with CRD. 

### Install localgraphclustering
On my mac, with a homebrew install of Python, I just ran

    pip3 install localgraphclustering --user

And then everything should just work.     This will install localgraphclustering
for the system python3. But then we use PyCall conda and just point it at
the needed directory. Try include("CRD.jl").




