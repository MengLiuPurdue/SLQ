# SLQ
code for SLQ project


## Setup for macOS
We need cvxpy and localgraphclustering installed in order to run
the comparison algorithms in CVX and CRD.

### Install CVXpy
This is installed via conda-forge, which should just work...

### Install localgraphclustering
On my mac, with a homebrew install of Python, I just ran

    pip3 install localgraphclustering --user

And then everything should just work.     This will install localgraphclustering
for the system python3. But then we use PyCall conda and just point it at
the needed directory. 
