using PyCall
if haskey(ENV, "USER") && ENV["USER"] == "liu1740"
  push!(PyVector(pyimport("sys")."path"),
    "/homes/liu1740/Research/LocalGraphClustering/")
elseif haskey(ENV, "USER") && ENV["USER"] == "mengliu"
  push!(PyVector(pyimport("sys")."path"),
    "/Users/mengliu/OneDrive/Mengs_Files/Research/LocalGraphClustering/")
else
  if Sys.isapple()
    push!(PyVector(pyimport("sys")."path"),
      "$(homedir())/Library/Python/3.7/lib/python/site-packages")
  end
end
