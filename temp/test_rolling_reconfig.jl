# 测试修复后的滚动拓扑重构数据加载
using Pkg
Pkg.activate(".")

println("Loading workflows module...")
include(joinpath(@__DIR__, "..", "src", "workflows.jl"))
using .Workflows

# 数据文件路径
case_path = joinpath(@__DIR__, "..", "data", "ac_dc_real_case.xlsx")

println("\n====== Testing Rolling Topology Reconfiguration Data Loading ======\n")

# 加载电力系统数据
println("1. Loading power system data...")
julia_case = Workflows.load_julia_power_data(case_path)

# 转换为JPC_tp格式
println("2. Converting to JPC_tp format...")
jpc, C_sparse = Workflows.JuliaPowerCase2Jpc_tp(julia_case)

# 输出关键数据以验证
println("\n====== Key Data Verification ======\n")

println("--- Counts ---")
println("nb_ac = ", jpc[:nb_ac])
println("nb_dc = ", jpc[:nb_dc])
println("ng = ", jpc[:ng])
println("nmg = ", jpc[:nmg])
println("nl_ac = ", jpc[:nl_ac])
println("nl_dc = ", jpc[:nl_dc])
println("nl_vsc = ", jpc[:nl_vsc])

println("\n--- Key Values (matching reference) ---")
println("Pgmax = ", jpc[:Pgmax], " kW")
println("Pvscmax = ", jpc[:Pvscmax], " kW")
println("eta (VSC efficiency) = ", jpc[:η])

println("\n--- Matrix Sizes ---")
println("Cg size = ", size(C_sparse[:Cg]))
println("Cd size = ", size(C_sparse[:Cd]))
println("Cmg size = ", size(C_sparse[:Cmg]))
println("Cft_vsc size = ", size(C_sparse[:Cft_vsc]))

println("\n====== Test Complete ======")
println("\nExpected reference values:")
println("  Pgmax = [170219.65, 178979.691] kW")
println("  Pvscmax = [3000.0 x 7]")
println("  eta = [0.9 x 7]")
println("  nmg = 10")
