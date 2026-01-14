# 测试 load_hybrid_case 函数
project_root = dirname(@__DIR__)
cd(project_root)

using Pkg
Pkg.activate(project_root)

include(joinpath(project_root, "src", "workflows.jl"))
using .Workflows

# 定义 MESS 配置
mess_configs = [
    Workflows.MESSConfig("MESS-1", 5, 1500.0, 1500.0, 4500.0, 2500.0, 92.0, 90.0),
]

# 加载数据
case_path = joinpath(project_root, "data", "ac_dc_real_case.xlsx")
case = Workflows.load_hybrid_case(case_path, mess_configs)

println("\n" * "=" ^ 60)
println("HybridGridCase 数据验证:")
println("=" ^ 60)
println("基本参数:")
println("  - nb: $(case.nb)")
println("  - nb_ac: $(case.nb_ac)")
println("  - nb_dc: $(case.nb_dc)")
println("  - nl_ac: $(case.nl_ac)")
println("  - nl_dc: $(case.nl_dc)")
println("  - nl_vsc: $(case.nl_vsc)")
println("  - ng: $(case.ng)")
println("  - nmg: $(case.nmg)")

println("\n微网数据:")
println("  - Pmgmax: $(case.Pmgmax)")
println("  - Qmgmax: $(case.Qmgmax)")
println("  - mg_nodes: $(case.microgrid_nodes)")

println("\n负荷数据:")
println("  - 总负荷 Pd: $(sum(case.Pd)) MW")
println("  - 总负荷 Qd: $(sum(case.Qd)) MVar")

println("\n矩阵类型验证:")
println("  - Cft_ac: $(typeof(case.Cft_ac))")
println("  - Cmg: $(typeof(case.Cmg))")
println("  - Cd: $(typeof(case.Cd))")
