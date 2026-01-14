# 调试脚本：检查数据加载结果

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# 加载工作流模块
include(joinpath(@__DIR__, "..", "src", "workflows.jl"))
using .Workflows

# 加载数据
case_file = joinpath(@__DIR__, "..", "data", "ac_dc_real_case.xlsx")
println("正在加载电网数据: $case_file")
case_jpc = Workflows.load_case(case_file)

baseMVA = case_jpc["baseMVA"]
println("=" ^ 60)
println("baseMVA = ", baseMVA)
println("=" ^ 60)

# 打印负荷信息
println("\n=== AC负荷信息 ===")
loadAC = case_jpc["loadAC"]
println("负荷数量: ", size(loadAC, 1))
if size(loadAC, 1) > 0
    println("loadAC矩阵 (列: [index, bus_id, status, pd_mw, qd_mvar, const_z, const_i, const_p, type]):")
    for i in 1:min(size(loadAC, 1), 10)
        println("  负荷 $i: bus=$(loadAC[i, 2]), P=$(loadAC[i, 4]) MW, Q=$(loadAC[i, 5]) MVar")
    end
    if size(loadAC, 1) > 10
        println("  ... 共 $(size(loadAC, 1)) 个负荷")
    end
    total_load_p = sum(loadAC[:, 4])
    total_load_q = sum(loadAC[:, 5])
    println("总有功负荷: $(total_load_p) MW")
    println("总无功负荷: $(total_load_q) MVar")
end

# 打印DC负荷信息
println("\n=== DC负荷信息 ===")
loadDC = case_jpc["loadDC"]
println("DC负荷数量: ", size(loadDC, 1))
if size(loadDC, 1) > 0
    for i in 1:min(size(loadDC, 1), 10)
        println("  DC负荷 $i: bus=$(loadDC[i, 2]), P=$(loadDC[i, 4]) MW")
    end
    total_dcload_p = sum(loadDC[:, 4])
    println("总DC有功负荷: $(total_dcload_p) MW")
end

# 打印发电机信息
println("\n=== 发电机信息 ===")
genAC = case_jpc["genAC"]
println("发电机数量: ", size(genAC, 1))
if size(genAC, 1) > 0
    println("genAC矩阵 (列9是PMAX):")
    for i in 1:size(genAC, 1)
        println("  发电机 $i: bus=$(genAC[i, 1]), Pmax=$(genAC[i, 9]) MW, Pmin=$(genAC[i, 10]) MW")
    end
    total_gen_cap = sum(genAC[:, 9])
    println("总发电容量: $(total_gen_cap) MW")
end

# 打印线路信息
println("\n=== 线路信息 ===")
branchAC = case_jpc["branchAC"]
println("AC线路数量: ", size(branchAC, 1))
if size(branchAC, 1) > 0
    println("branchAC矩阵 (列6是RATE_A容量):")
    for i in 1:min(size(branchAC, 1), 10)
        println("  线路 $i: $(branchAC[i, 1])->$(branchAC[i, 2]), 容量=$(branchAC[i, 6]) MVA, R=$(branchAC[i, 3]), X=$(branchAC[i, 4])")
    end
    if size(branchAC, 1) > 10
        println("  ... 共 $(size(branchAC, 1)) 条线路")
    end
end

branchDC = case_jpc["branchDC"]
println("DC线路数量: ", size(branchDC, 1))

# 打印VSC信息
println("\n=== VSC换流器信息 ===")
converter = case_jpc["converter"]
println("VSC数量: ", size(converter, 1))
if size(converter, 1) > 0
    for i in 1:size(converter, 1)
        println("  VSC $i: AC_bus=$(converter[i, 1]), DC_bus=$(converter[i, 2]), 效率=$(converter[i, 7])")
    end
end

# 打印标幺化后的关键参数
println("\n=== 标幺化后的参数 (除以baseMVA=$(baseMVA)) ===")
Pd = case_jpc["Pd"]
Pgmax = case_jpc["Pgmax"]
Smax = case_jpc["Smax"]

println("Pd (标幺值): ", round.(Pd, digits=4))
println("Pd (实际MW): ", round.(Pd .* baseMVA, digits=2))
println("Pgmax (标幺值): ", round.(Pgmax, digits=4))
println("Pgmax (实际MW): ", round.(Pgmax .* baseMVA, digits=2))
println("Smax (标幺值，前10条线路): ", round.(Smax[1:min(length(Smax), 10)], digits=4))
println("Smax (实际MVA，前10条线路): ", round.(Smax[1:min(length(Smax), 10)] .* baseMVA, digits=2))

# 检查供需平衡
println("\n=== 供需平衡检查 ===")
total_demand = sum(Pd) * baseMVA
total_capacity = sum(Pgmax) * baseMVA
println("总负荷需求: $(round(total_demand, digits=2)) MW")
println("总发电容量: $(round(total_capacity, digits=2)) MW")
println("供需比: $(round(total_capacity/total_demand * 100, digits=1))%")
if total_capacity < total_demand
    println("⚠️ 警告: 发电容量不足以满足全部负荷!")
end
