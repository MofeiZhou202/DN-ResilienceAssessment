# 验证微电网数据读取修复
# 期望输出：10个微电网（与参考实现 Topology_Reconfiguration/test/test_ac_dc_tp.jl 一致）

using Pkg
Pkg.activate(".")

# 加载必要的模块
include("../src/workflows.jl")
using .Workflows

# 数据文件路径
case_path = joinpath(@__DIR__, "..", "data", "ac_dc_real_case.xlsx")

# 使用 ETAPImporter 加载数据
println("=" ^ 60)
println("正在加载数据...")
julia_case = Workflows.load_julia_power_data(case_path)

# 检查 case.pvarray 中的数据
println("\n" * "=" ^ 60)
println("ETAPImporter 加载结果:")
println("pvarray 数量: $(length(julia_case.pvarray))")
println("ACPVSystems 数量: $(length(julia_case.ACPVSystems))")

println("\npvarray 详细信息:")
for (i, pv) in enumerate(julia_case.pvarray)
    println("  $i: name=$(pv.name), bus_id=$(pv.bus_id), p_rated_mw=$(pv.p_rated_mw)")
end

# 转换为 JPC_tp 格式
println("\n" * "=" ^ 60)
println("正在转换为 JPC_tp 格式...")
jpc_tp, C_sparse = Workflows.JuliaPowerCase2Jpc_tp(julia_case)

# 检查微电网数据
println("\n" * "=" ^ 60)
println("JPC_tp 转换结果:")
println("jpc_tp.pv 矩阵大小: $(size(jpc_tp.pv))")
println("jpc_tp.pv_acsystem 矩阵大小: $(size(jpc_tp.pv_acsystem))")

# 获取微电网参数
nmg = jpc_tp[:nmg]
Pmgmax = jpc_tp[:Pmgmax]
Qmgmax = jpc_tp[:Qmgmax]
Cmg = C_sparse[:Cmg]

println("\n微电网数量 (nmg): $nmg")
println("微电网最大有功 (Pmgmax, 标幺值): $Pmgmax")
println("微电网最大有功 (Pmgmax, MW): $(Pmgmax .* jpc_tp.baseMVA)")
println("微电网最大无功 (Qmgmax, 标幺值): $Qmgmax")
println("Cmg 矩阵大小: $(size(Cmg))")

# 检查 Cmg 矩阵的母线连接
println("\n微电网连接的母线:")
for mg_idx in 1:nmg
    bus_idx = findfirst(x -> x == 1, Cmg[mg_idx, :])
    if !isnothing(bus_idx)
        println("  微电网 $mg_idx -> 母线 $bus_idx")
    else
        println("  微电网 $mg_idx -> 未连接到任何母线")
    end
end

# 验证结果
println("\n" * "=" ^ 60)
println("验证结果:")
expected_nmg = 10
if nmg == expected_nmg
    println("✓ 微电网数量正确: $nmg (期望: $expected_nmg)")
else
    println("✗ 微电网数量错误: $nmg (期望: $expected_nmg)")
end

# 与参考实现的输出比较
# 参考实现的 Pmgmax 应该是: [4.2, 0.3, 0.02, 0.09, 0.03, 0.01, 0.02, 4.2, 0.002, 0.002]
# 注意顺序可能不同，取决于 Excel 文件中的行顺序
println("\n与参考实现比较:")
println("参考实现 Pmgmax (MW): [4.2, 0.3, 0.02, 0.09, 0.03, 0.01, 0.02, 4.2, 0.002, 0.002]")
println("当前实现 Pmgmax (MW): $(Pmgmax .* jpc_tp.baseMVA)")
