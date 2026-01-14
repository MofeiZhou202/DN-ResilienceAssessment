# 调试VSC/逆变器数据 - 详细版本
project_root = dirname(@__DIR__)
cd(project_root)

using Pkg
Pkg.activate(project_root)

include(joinpath(project_root, "src", "workflows.jl"))
using .Workflows

# 读取数据
case = Workflows.load_julia_power_data("data/ac_dc_real_case.xlsx")
jpc_tp, C_sparse = Workflows.JuliaPowerCase2Jpc_tp(case)

# 基本参数
baseMVA = jpc_tp.baseMVA
nb_ac = size(jpc_tp.busAC, 1)
nb_dc = size(jpc_tp.busDC, 1)
nb = nb_ac + nb_dc
nl_ac = size(jpc_tp.branchAC, 1)
nl_dc = size(jpc_tp.branchDC, 1)
nl_vsc = size(jpc_tp.converter, 1)
ng = size(jpc_tp.genAC, 1)

println("=" ^ 60)
println("系统基本信息:")
println("=" ^ 60)
println("baseMVA = $baseMVA")
println("AC母线数量 (nb_ac): $nb_ac")
println("DC母线数量 (nb_dc): $nb_dc")
println("总母线数 (nb): $nb")
println("AC线路数量 (nl_ac): $nl_ac")
println("DC线路数量 (nl_dc): $nl_dc")
println("VSC数量 (nl_vsc): $nl_vsc")
println("发电机数量 (ng): $ng")

println("\n" * "=" ^ 60)
println("VSC (逆变器) 详细信息:")
println("=" ^ 60)
for i in 1:nl_vsc
    acbus = Int(jpc_tp.converter[i, 1])
    dcbus = Int(jpc_tp.converter[i, 2])
    status = Int(jpc_tp.converter[i, 3])
    p_ac = jpc_tp.converter[i, 4]
    q_ac = jpc_tp.converter[i, 5]
    p_dc = jpc_tp.converter[i, 6]
    eff = jpc_tp.converter[i, 7]
    pmax = jpc_tp.converter[i, 8]
    
    println("\nVSC $i:")
    println("  AC母线: $acbus (AC母线范围 1-$nb_ac)")
    println("  DC母线: $dcbus (DC母线范围 $(nb_ac+1)-$nb)")
    println("  状态: $status $(status==1 ? "(在线)" : "(离线)")")
    println("  当前P_AC: $p_ac MW")
    println("  当前Q_AC: $q_ac MVar")
    println("  效率: $eff")
    println("  PMAX (第8列): $pmax MW")
end

# 检查 Pvscmax
println("\n" * "=" ^ 60)
println("Pvscmax (优化中的最大功率限制):")
println("=" ^ 60)
# 模拟 Types.jl 中的计算
Pvscmax_pu = fill(100.0 / baseMVA, nl_vsc)
Pvscmax_mw = Pvscmax_pu .* baseMVA
println("Pvscmax (标幺值): ", Pvscmax_pu)
println("Pvscmax (MW): ", Pvscmax_mw)

println("\n" * "=" ^ 60)
println("Cft_vsc 关联矩阵:")
println("=" ^ 60)
Cft_vsc = C_sparse[:Cft_vsc]
println("Cft_vsc 大小: $(size(Cft_vsc)) (VSC数 × 总母线数)")
println("\nCft_vsc 矩阵 (每行一个VSC, 列是母线):")
println("格式: VSC i: AC母线j 系数 = +1, DC母线k 系数 = -1")
for i in 1:nl_vsc
    row = Cft_vsc[i, :]
    acbus = findfirst(x -> x == 1, row)
    dcbus = findfirst(x -> x == -1, row)
    println("VSC $i: AC母线 $acbus (+1), DC母线 $dcbus (-1)")
end

println("\n" * "=" ^ 60)
println("负荷信息:")
println("=" ^ 60)
println("AC负荷数量: ", size(jpc_tp.loadAC, 1))
local total_ac_load_p = 0.0
local total_ac_load_q = 0.0
if size(jpc_tp.loadAC, 1) > 0
    for i in 1:size(jpc_tp.loadAC, 1)
        bus = Int(jpc_tp.loadAC[i, 2])
        P = jpc_tp.loadAC[i, 3]
        Q = jpc_tp.loadAC[i, 4]
        global total_ac_load_p += P
        global total_ac_load_q += Q
        println("  AC负荷 $i: 母线=$bus, P=$(round(P, digits=2)) MW, Q=$(round(Q, digits=2)) MVar")
    end
end
println("  【AC侧总负荷: P=$(round(total_ac_load_p, digits=2)) MW, Q=$(round(total_ac_load_q, digits=2)) MVar】")

println("\nDC负荷数量: ", size(jpc_tp.loadDC, 1))
local total_dc_load = 0.0
if size(jpc_tp.loadDC, 1) > 0
    for i in 1:size(jpc_tp.loadDC, 1)
        bus = Int(jpc_tp.loadDC[i, 2])
        P = jpc_tp.loadDC[i, 3]
        global total_dc_load += P
        println("  DC负荷 $i: 母线=$(bus + nb_ac), P=$(round(P, digits=2)) MW")
    end
end
println("  【DC侧总负荷: P=$(round(total_dc_load, digits=2)) MW】")

println("\n" * "=" ^ 60)
println("发电机信息:")
println("=" ^ 60)
local total_gen_pmax = 0.0
if ng > 0
    for i in 1:ng
        bus = Int(jpc_tp.genAC[i, 1])
        Pmax = jpc_tp.genAC[i, 2]
        Pmin = jpc_tp.genAC[i, 3]
        global total_gen_pmax += Pmax
        println("发电机 $i: 母线=$bus, Pmax=$(round(Pmax, digits=2)) MW, Pmin=$(round(Pmin, digits=2)) MW")
    end
end
println("  【发电机总容量: Pmax=$(round(total_gen_pmax, digits=2)) MW】")

println("\n" * "=" ^ 60)
println("α_pre (初始开关状态) 中的VSC状态:")
println("=" ^ 60)
α_pre = jpc_tp[:α_pre]
local vsc_status = []
if length(α_pre) >= nl_ac + nl_dc + nl_vsc
    vsc_status = α_pre[nl_ac+nl_dc+1 : nl_ac+nl_dc+nl_vsc]
    for i in 1:nl_vsc
        println("VSC $i: α = $(vsc_status[i]) $(vsc_status[i]==1 ? "(在线可用)" : "(离线不可用)")")
    end
    println("在线VSC数量: ", sum(vsc_status .== 1))
else
    println("α_pre 长度不足，当前长度: $(length(α_pre)), 需要: $(nl_ac+nl_dc+nl_vsc)")
end

println("\n" * "=" ^ 60)
println("分析:")
println("=" ^ 60)
println("1. AC侧总负荷: $(round(total_ac_load_p, digits=2)) MW")
println("2. DC侧总负荷: $(round(total_dc_load, digits=2)) MW")
println("3. 发电机总容量: $(round(total_gen_pmax, digits=2)) MW")
num_online_vsc = length(vsc_status) > 0 ? sum(vsc_status .== 1) : 0
println("4. 在线VSC最大传输能力: $(num_online_vsc * 100) MW (每个100MW)")

if total_gen_pmax >= total_ac_load_p + total_dc_load
    println("\n结论: 发电机容量 ($(round(total_gen_pmax, digits=2)) MW) >= 总负荷 ($(round(total_ac_load_p + total_dc_load, digits=2)) MW)")
    println("       如果发电机全在AC侧，可能不需要VSC传输功率到DC侧！")
end

# 检查发电机连接的母线是AC还是DC
println("\n" * "=" ^ 60)
println("发电机所在母线类型:")
println("=" ^ 60)
for i in 1:ng
    bus = Int(jpc_tp.genAC[i, 1])
    if bus <= nb_ac
        println("发电机 $i 连接在 AC母线 $bus")
    else
        println("发电机 $i 连接在 DC母线 $bus")
    end
end

# 检查DC母线是否有发电来源
println("\n" * "=" ^ 60)
println("关键检查 - DC侧电源:")
println("=" ^ 60)
println("检查DC侧（母线26-30）是否有发电/储能/PV接入：")

# 检查genAC中有没有发电机在DC侧
dc_gen_count = 0
for i in 1:ng
    bus = Int(jpc_tp.genAC[i, 1])
    if bus > nb_ac
        dc_gen_count += 1
        println("  发电机 $i 在DC母线 $bus")
    end
end
if dc_gen_count == 0
    println("  【无】DC侧没有发电机")
end

# 检查PV
pv_count = size(jpc_tp.pv, 1)
println("\nDC侧光伏数量: $pv_count")
if pv_count > 0
    for i in 1:pv_count
        bus = Int(jpc_tp.pv[i, 2])
        pmax = jpc_tp.pv[i, 3]
        println("  DC光伏 $i: 母线=$bus, Pmax=$pmax MW")
    end
end

# 检查储能
bess_count = size(jpc_tp.bess, 1)
println("\n储能数量: $bess_count")
if bess_count > 0
    for i in 1:bess_count
        bus = Int(jpc_tp.bess[i, 2])
        pmax = jpc_tp.bess[i, 3]
        println("  储能 $i: 母线=$bus, Pmax=$pmax MW")
    end
end

println("\n" * "=" ^ 60)
println("总结:")
println("=" ^ 60)
println("如果DC侧有负荷但没有本地发电，")
println("VSC必须从AC侧向DC侧传输功率来供电。")
println("这时 pinv (逆变功率) 应该 > 0。")
println("\n但如果优化结果显示 prec=pinv=0，")
println("可能的原因：")
println("  1. DC侧负荷为0")
println("  2. DC侧有足够的本地发电")
println("  3. VSC的β(开关状态)约束导致无法使用")
println("  4. 约束设置有问题")
