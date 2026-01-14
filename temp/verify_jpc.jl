# 验证数据加载
project_root = dirname(@__DIR__)
cd(project_root)

include(joinpath(project_root, "src", "workflows.jl"))
using .Workflows

case = Workflows.load_julia_power_data("data/ac_dc_real_case.xlsx")
jpc_tp, C = Workflows.JuliaPowerCase2Jpc_tp(case)

println("微网数量: ", jpc_tp[:nmg])
println("Pmgmax: ", jpc_tp[:Pmgmax] .* jpc_tp.baseMVA)
println("Cmg 类型: ", typeof(C[:Cmg]))
println("Cft_ac 类型: ", typeof(C[:Cft_ac]))

# 检查矩阵大小
println("\n矩阵维度:")
println("  Cft_ac: ", size(C[:Cft_ac]))
println("  Cft_dc: ", size(C[:Cft_dc]))
println("  Cft_vsc: ", size(C[:Cft_vsc]))
println("  Cg: ", size(C[:Cg]))
println("  Cmg: ", size(C[:Cmg]))
println("  Cd: ", size(C[:Cd]))
