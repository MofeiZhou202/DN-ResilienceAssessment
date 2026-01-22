# 全面对比参考实现和api_server实现的数据读取结果

using Pkg
Pkg.activate(".")

using DataFrames
using XLSX
using SparseArrays

# 加载参考实现
include(joinpath(@__DIR__, "..", "Topology_Reconfiguration", "src", "juliapowercase2jpc_tp.jl"))

# 加载api_server实现
include(joinpath(@__DIR__, "..", "src", "workflows.jl"))
using .Workflows

# 数据文件路径
project_root = joinpath(@__DIR__, "..")
case_path = joinpath(project_root, "data", "ac_dc_real_case.xlsx")

println("=" ^ 80)
println("对比参考实现和api_server实现的数据读取结果")
println("=" ^ 80)

# 1. 加载参考实现的数据
println("\n[1] 加载参考实现数据...")
ref_jpc = load_case(case_path)

# 2. 加载api_server实现的数据
println("\n[2] 加载api_server实现数据...")
julia_case = Workflows.load_julia_power_data(case_path)
api_jpc, api_C_sparse = Workflows.JuliaPowerCase2Jpc_tp(julia_case)

# 3. 对比关键参数
println("\n" * "=" ^ 80)
println("关键参数对比")
println("=" ^ 80)

# 节点数量
println("\n--- 节点数量 ---")
println("参考: nb_ac=$(ref_jpc[:nb_ac]), nb_dc=$(ref_jpc[:nb_dc]), nb=$(ref_jpc[:nb])")
println("API:  nb_ac=$(api_jpc[:nb_ac]), nb_dc=$(api_jpc[:nb_dc]), nb=$(api_jpc[:nb_ac] + api_jpc[:nb_dc])")

# 线路数量
println("\n--- 线路数量 ---")
println("参考: nl_ac=$(ref_jpc[:nl_ac]), nl_dc=$(ref_jpc[:nl_dc]), nl_vsc=$(ref_jpc[:nl_vsc])")
println("API:  nl_ac=$(api_jpc[:nl_ac]), nl_dc=$(api_jpc[:nl_dc]), nl_vsc=$(api_jpc[:nl_vsc])")

# 发电机数量
println("\n--- 发电机数量 ---")
println("参考: ng=$(ref_jpc[:ng])")
println("API:  ng=$(api_jpc[:ng])")

# 负荷数量
println("\n--- 负荷数量 ---")
println("参考: nd=$(ref_jpc[:nd])")
println("API:  nd=$(api_jpc[:nd])")

# 微电网数量
println("\n--- 微电网数量 ---")
println("参考: nmg=$(ref_jpc[:nmg])")
println("API:  nmg=$(api_jpc[:nmg])")

# 发电机最大出力
println("\n--- 发电机最大出力 Pgmax ---")
println("参考: $(ref_jpc[:Pgmax])")
api_Pgmax = api_jpc[:Pgmax] .* api_jpc.baseMVA
println("API:  $(api_Pgmax)")

# 微电网最大出力
println("\n--- 微电网最大出力 Pmgmax ---")
println("参考: $(ref_jpc[:Pmgmax])")
api_Pmgmax = api_jpc[:Pmgmax] .* api_jpc.baseMVA
println("API:  $(api_Pmgmax)")

# 负荷功率
println("\n--- 负荷功率 Pd ---")
println("参考: $(ref_jpc[:Pd])")
api_Pd = api_jpc[:Pd] .* api_jpc.baseMVA
println("API:  $(api_Pd)")

# VSC最大功率
println("\n--- VSC最大功率 Pvscmax ---")
println("参考: $(ref_jpc[:Pvscmax])")
api_Pvscmax = api_jpc[:Pvscmax] .* api_jpc.baseMVA
println("API:  $(api_Pvscmax)")

# VSC效率
println("\n--- VSC效率 η ---")
println("参考: $(ref_jpc[:η])")
api_η = api_jpc[:η]
println("API:  $(api_η)")

# 线路电阻
println("\n--- 线路电阻 R ---")
println("参考 (前10): $(ref_jpc[:R][1:min(10, length(ref_jpc[:R]))])")
api_R = api_jpc[:R]
println("API  (前10): $(api_R[1:min(10, length(api_R))])")

# 线路电抗
println("\n--- 线路电抗 X ---")
println("参考 (前10): $(ref_jpc[:X][1:min(10, length(ref_jpc[:X]))])")
api_X = api_jpc[:X]
println("API  (前10): $(api_X[1:min(10, length(api_X))])")

# 开关状态
println("\n--- 开关状态 r (switch_flag) ---")
println("参考: $(ref_jpc[:r])")
# API没有直接的r字段，需要检查

# 线路初始状态
println("\n--- 线路初始状态 α_pre ---")
println("参考: $(ref_jpc[:α_pre])")
api_α_pre = api_jpc["α_pre"]
println("API:  $(api_α_pre)")

# 电压限制
println("\n--- 电压上限 VMAX ---")
println("参考: $(ref_jpc[:VMAX])")
api_VMAX = api_jpc[:VMAX]
println("API:  $(api_VMAX)")

println("\n--- 电压下限 VMIN ---")
println("参考: $(ref_jpc[:VMIN])")
api_VMIN = api_jpc[:VMIN]
println("API:  $(api_VMIN)")

# 连接矩阵对比
println("\n--- 连接矩阵大小 ---")
println("参考: Cft_ac=$(size(ref_jpc[:Cft_ac])), Cft_dc=$(size(ref_jpc[:Cft_dc])), Cft_vsc=$(size(ref_jpc[:Cft_vsc]))")
println("API:  Cft_ac=$(size(api_C_sparse[:Cft_ac])), Cft_dc=$(size(api_C_sparse[:Cft_dc])), Cft_vsc=$(size(api_C_sparse[:Cft_vsc]))")

println("\n参考 Cmg 大小: $(size(ref_jpc[:Cmg]))")
println("API  Cmg 大小: $(size(api_C_sparse[:Cmg]))")

println("\n参考 Cg 大小: $(size(ref_jpc[:Cg]))")
println("API  Cg 大小: $(size(api_C_sparse[:Cg]))")

println("\n参考 Cd 大小: $(size(ref_jpc[:Cd]))")
println("API  Cd 大小: $(size(api_C_sparse[:Cd]))")

# 检查差异并输出建议
println("\n" * "=" ^ 80)
println("差异分析")
println("=" ^ 80)

# 检查nmg
if ref_jpc[:nmg] != api_jpc[:nmg]
    println("❌ 微电网数量不一致: 参考=$(ref_jpc[:nmg]), API=$(api_jpc[:nmg])")
else
    println("✓ 微电网数量一致: $(ref_jpc[:nmg])")
end

# 检查ng
if ref_jpc[:ng] != api_jpc[:ng]
    println("❌ 发电机数量不一致: 参考=$(ref_jpc[:ng]), API=$(api_jpc[:ng])")
else
    println("✓ 发电机数量一致: $(ref_jpc[:ng])")
end

# 检查nd
if ref_jpc[:nd] != api_jpc[:nd]
    println("❌ 负荷数量不一致: 参考=$(ref_jpc[:nd]), API=$(api_jpc[:nd])")
else
    println("✓ 负荷数量一致: $(ref_jpc[:nd])")
end

# 检查nl_vsc
if ref_jpc[:nl_vsc] != api_jpc[:nl_vsc]
    println("❌ VSC数量不一致: 参考=$(ref_jpc[:nl_vsc]), API=$(api_jpc[:nl_vsc])")
else
    println("✓ VSC数量一致: $(ref_jpc[:nl_vsc])")
end
