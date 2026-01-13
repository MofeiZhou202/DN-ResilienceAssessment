using XLSX
using DataFrames

path = joinpath(@__DIR__, "..", "data", "topology_reconfiguration_results.xlsx")
df = DataFrame(XLSX.readtable(path, "cluster_summary"))
println(names(df))
