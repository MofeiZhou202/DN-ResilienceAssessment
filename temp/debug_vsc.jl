# 调试VSC/逆变器数据
# 切换到项目根目录
project_root = dirname(@__DIR__)
cd(project_root)
println("当前工作目录: ", pwd())

using Pkg
Pkg.activate(project_root)

# 加载所需模块
include(joinpath(project_root, "src", "workflows.jl"))
using .Workflows

# 读取数据 - 使用 Workflows 模块中的函数
case = Workflows.load_julia_power_data("data/ac_dc_real_case.xlsx")

println("=== 逆变器/VSC 数据检查 ===")
println("逆变器数量: ", length(case.converters))

if length(case.converters) > 0
    println("\n--- 逆变器详细信息 ---")
    for (i, inv) in enumerate(case.converters)
        println("\n逆变器 $i:")
        println("  AC母线ID: ", inv.ac_bus_id)
        println("  DC母线ID: ", inv.dc_bus_id)
        println("  状态: ", inv.status)
        println("  P_AC (MW): ", inv.p_ac_mw)
        println("  Q_AC (MVar): ", inv.q_ac_mvar)
        println("  效率: ", inv.efficiency)
        println("  模式: ", inv.mode)
    end
end

# 转换为JPC_tp格式
jpc_tp, _ = Workflows.JuliaPowerCase2Jpc_tp(case)

println("\n=== JPC_tp 中的 converter 矩阵 ===")
println("converter 矩阵大小: ", size(jpc_tp.converter))
if size(jpc_tp.converter, 1) > 0
    println("converter 矩阵内容:")
    display(jpc_tp.converter)
end

println("\n=== 网络拓扑检查 ===")
jpc = Workflows.get_jpc_tp_data(jpc_tp)
println("nl_vsc (VSC数量): ", jpc[:nl_vsc])
println("Pvscmax (VSC最大功率标幺值): ", jpc[:Pvscmax])
println("baseMVA: ", jpc_tp.baseMVA)
println("Pvscmax实际值 (MW): ", jpc[:Pvscmax] .* jpc_tp.baseMVA)

# 检查Cft_vsc关联矩阵
println("\n=== VSC关联矩阵 Cft_vsc ===")
println("Cft_vsc 大小: ", size(jpc[:Cft_vsc]))

# 检查VSC效率
println("\n=== VSC效率 (η) ===")
println("η: ", jpc[:η])
