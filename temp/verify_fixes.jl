# 验证修复后的API Server数据读取
using Pkg
Pkg.activate(".")

# 加载api_server实现
include(joinpath(@__DIR__, "..", "src", "workflows.jl"))
using .Workflows

# 数据文件路径
case_path = joinpath(@__DIR__, "..", "data", "ac_dc_real_case.xlsx")

println("Loading API Server implementation data...")
julia_case = Workflows.load_julia_power_data(case_path)
api_jpc, api_C_sparse = Workflows.JuliaPowerCase2Jpc_tp(julia_case)

println()
println("====== API Server Data Verification ======")
println()

# Check key parameters
println("--- Node and Branch Counts ---")
println("nb_ac = ", api_jpc[:nb_ac])
println("nb_dc = ", api_jpc[:nb_dc])
println("ng = ", api_jpc[:ng])
println("nl_ac = ", api_jpc[:nl_ac])
println("nl_dc = ", api_jpc[:nl_dc])
println("nl_vsc = ", api_jpc[:nl_vsc])
println("nmg = ", api_jpc[:nmg])

println()
println("--- Generator Parameters ---")
Pgmax = api_jpc[:Pgmax]
Qgmax = api_jpc[:Qgmax]
println("Pgmax (kW) = ", Pgmax)
println("Qgmax (kVar) = ", Qgmax)

println()
println("--- Load Parameters ---")
Pd = api_jpc[:Pd]
Qd = api_jpc[:Qd]
println("Pd (first 10, kVA) = ", Pd[1:min(10, length(Pd))])
println("Qd (first 10, kVar) = ", Qd[1:min(10, length(Qd))])
println("Total loads = ", length(Pd))

println()
println("--- Microgrid Parameters ---")
Pmgmax = api_jpc[:Pmgmax]
Qmgmax = api_jpc[:Qmgmax]
println("Pmgmax (kW) = ", Pmgmax)
println("Qmgmax (kVar) = ", Qmgmax)

println()
println("--- VSC Parameters ---")
Pvscmax = api_jpc[:Pvscmax]
eta = api_jpc[:η]
println("Pvscmax (kW) = ", Pvscmax)
println("eta = ", eta)

println()
println("--- Cd Matrix ---")
Cd = api_C_sparse[:Cd]
println("Cd size = ", size(Cd))

println()
println("===== Reference Values =====")
println("Reference: Pgmax=[170219.65, 178979.691] kW")
println("Reference: Pd=[100, 300, 300, ...] kVA (17 total = 16 AC + 1 DC)")
println("Reference: Pvscmax=[3000, 3000, 3000, 3000, 3000, 3000, 3000] kW")
println("Reference: eta=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]")
println("Reference: Cd size=(17, 30)")
