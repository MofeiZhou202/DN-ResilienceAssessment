# 设置工作目录
project_root = joinpath(@__DIR__, "..")
cd(project_root)
include(joinpath(project_root, "src/workflows.jl"))
using .Workflows: load_data_unified

# 测试数据加载
println("正在加载数据...")
jpc = load_data_unified("data/ac_dc_real_case.xlsx")

println("\n=== 数据加载结果 ===")
println("光伏阵列数量: ", length(jpc.pvarray))
println("AC光伏系统数量: ", length(jpc.ACPVSystems))

# 打印光伏阵列详情
if length(jpc.pvarray) > 0
    println("\n光伏阵列详情:")
    for pv in jpc.pvarray
        println("  - $(pv.name): bus=$(pv.bus_id), p_rated=$(pv.p_rated_mw) MW")
    end
end
