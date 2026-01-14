# 验证修复后的数据加载和供电率
println("="^60)
println("测试脚本：验证数据加载和供电率")
println("="^60)

using Pkg
Pkg.activate(".")

# 切换到项目根目录
project_root = dirname(@__DIR__)
cd(project_root)
println("项目目录: $project_root")

# 按正确顺序加载依赖
include(joinpath(project_root, "src/utils/idx.jl"))
include(joinpath(project_root, "src/utils/ComponentStructs.jl"))  # 先加载组件定义
include(joinpath(project_root, "src/utils/Types.jl"))
include(joinpath(project_root, "src/utils/ETAPImporter.jl"))
include(joinpath(project_root, "src/utils/juliapowercase2jpc_tp.jl"))

# 加载数据
xlsx_path = "data/ac_dc_real_case.xlsx"
println("\n1. 加载 JuliaPowerCase...")
case = load_julia_power_data(xlsx_path)

println("\n2. 转换为 JPC_tp...")
result = JuliaPowerCase2Jpc_tp(case)
jpc_tp = result isa Tuple ? result[1] : result

# 打印负荷数据
println("\n" * "="^60)
println("负荷数据统计")
println("="^60)
baseMVA = jpc_tp.baseMVA
println("baseMVA = $baseMVA")

if size(jpc_tp.loadAC, 1) > 0
    pd_ac = jpc_tp.loadAC[:, 4]
    println("\n交流负荷 (loadAC):")
    println("  数量: $(size(jpc_tp.loadAC, 1))")
    println("  有功功率范围: $(minimum(pd_ac)) ~ $(maximum(pd_ac)) MW")
    println("  有功功率总和: $(sum(pd_ac)) MW = $(sum(pd_ac)*1000) kW")
end

if size(jpc_tp.loadDC, 1) > 0
    pd_dc = jpc_tp.loadDC[:, 4]
    println("\n直流负荷 (loadDC):")
    println("  数量: $(size(jpc_tp.loadDC, 1))")
    println("  有功功率范围: $(minimum(pd_dc)) ~ $(maximum(pd_dc)) MW")
    println("  有功功率总和: $(sum(pd_dc)) MW = $(sum(pd_dc)*1000) kW")
end

# 打印发电数据
println("\n" * "="^60)
println("发电数据统计")
println("="^60)

if size(jpc_tp.genAC, 1) > 0
    pg_max = jpc_tp.genAC[:, 9]  # PMAX 列
    println("\n发电机 (genAC):")
    println("  数量: $(size(jpc_tp.genAC, 1))")
    println("  最大出力范围: $(minimum(pg_max)) ~ $(maximum(pg_max)) MW")
    println("  最大出力总和: $(sum(pg_max)) MW = $(sum(pg_max)*1000) kW")
end

# 打印光伏数据
println("\n" * "="^60)
println("光伏数据统计")
println("="^60)

if size(jpc_tp.pv, 1) > 0
    pv_power = jpc_tp.pv[:, 3]  # p_max 列
    println("\n直流光伏 (pv / DC PV):")
    println("  数量: $(size(jpc_tp.pv, 1))")
    println("  功率范围: $(minimum(pv_power)) ~ $(maximum(pv_power)) MW")
    println("  功率总和: $(sum(pv_power)) MW = $(sum(pv_power)*1000) kW")
    for i in 1:size(jpc_tp.pv, 1)
        println("    PV $i: $(jpc_tp.pv[i, 3]) MW = $(jpc_tp.pv[i, 3]*1000) kW")
    end
end

if size(jpc_tp.pv_acsystem, 1) > 0
    ac_pv_power = jpc_tp.pv_acsystem[:, 10]  # p_mw 列
    println("\n交流光伏系统 (pv_acsystem / AC PV):")
    println("  数量: $(size(jpc_tp.pv_acsystem, 1))")
    println("  功率范围: $(minimum(ac_pv_power)) ~ $(maximum(ac_pv_power)) MW")
    println("  功率总和: $(sum(ac_pv_power)) MW = $(sum(ac_pv_power)*1000) kW")
    for i in 1:size(jpc_tp.pv_acsystem, 1)
        println("    AC PV $i: $(jpc_tp.pv_acsystem[i, 10]) MW = $(jpc_tp.pv_acsystem[i, 10]*1000) kW")
    end
end

# 计算总供给和总需求
println("\n" * "="^60)
println("供需平衡分析")
println("="^60)

total_load = 0.0
if size(jpc_tp.loadAC, 1) > 0
    total_load += sum(jpc_tp.loadAC[:, 4])
end
if size(jpc_tp.loadDC, 1) > 0
    total_load += sum(jpc_tp.loadDC[:, 4])
end

total_gen = 0.0
if size(jpc_tp.genAC, 1) > 0
    total_gen += sum(jpc_tp.genAC[:, 9])
end
if size(jpc_tp.pv, 1) > 0
    total_gen += sum(jpc_tp.pv[:, 3])
end
if size(jpc_tp.pv_acsystem, 1) > 0
    total_gen += sum(jpc_tp.pv_acsystem[:, 10])
end

println("\n总负荷: $(total_load) MW = $(total_load*1000) kW")
println("总发电能力: $(total_gen) MW = $(total_gen*1000) kW")
println("供需比例: $(total_gen > 0 ? round(total_gen/total_load*100, digits=1) : 0)%")

if total_gen >= total_load
    println("\n✓ 发电能力充足，能够满足负荷需求")
else
    println("\n✗ 发电能力不足，需要外部电网支持")
    println("  缺口: $(total_load - total_gen) MW = $((total_load - total_gen)*1000) kW")
end

println("\n" * "="^60)
println("测试完成")
println("="^60)
