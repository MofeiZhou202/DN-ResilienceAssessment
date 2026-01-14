# 调试VSC/逆变器数据 - 简化版本
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

println("\n" * "=" ^ 60)
println("负荷信息:")
println("=" ^ 60)

# AC 负荷
total_ac_p = 0.0
total_ac_q = 0.0
println("AC负荷:")
for i in 1:size(jpc_tp.loadAC, 1)
    bus = Int(jpc_tp.loadAC[i, 2])
    P = jpc_tp.loadAC[i, 3]
    Q = jpc_tp.loadAC[i, 4]
    total_ac_p += P
    total_ac_q += Q
    println("  负荷 $i: 母线=$bus, P=$(round(P, digits=2)) MW, Q=$(round(Q, digits=2)) MVar")
end
println("  【AC侧总负荷: P=$(round(total_ac_p, digits=2)) MW, Q=$(round(total_ac_q, digits=2)) MVar】")

# DC 负荷
total_dc_p = 0.0
println("\nDC负荷:")
for i in 1:size(jpc_tp.loadDC, 1)
    bus = Int(jpc_tp.loadDC[i, 2])
    P = jpc_tp.loadDC[i, 3]
    total_dc_p += P
    println("  负荷 $i: 母线=$(bus + nb_ac), P=$(round(P, digits=2)) MW")
end
println("  【DC侧总负荷: P=$(round(total_dc_p, digits=2)) MW】")

println("\n" * "=" ^ 60)
println("发电机信息:")
println("=" ^ 60)
total_gen_pmax = 0.0
for i in 1:ng
    bus = Int(jpc_tp.genAC[i, 1])
    Pmax = jpc_tp.genAC[i, 2]
    Pmin = jpc_tp.genAC[i, 3]
    total_gen_pmax += Pmax
    bus_type = bus <= nb_ac ? "AC" : "DC"
    println("发电机 $i: 母线=$bus ($bus_type), Pmax=$(round(Pmax, digits=2)) MW, Pmin=$(round(Pmin, digits=2)) MW")
end
println("【发电机总容量: Pmax=$(round(total_gen_pmax, digits=2)) MW】")

println("\n" * "=" ^ 60)
println("α_pre (初始开关状态) 中的VSC状态:")
println("=" ^ 60)
α_pre = jpc_tp[:α_pre]
vsc_online_count = 0
if length(α_pre) >= nl_ac + nl_dc + nl_vsc
    vsc_status = α_pre[nl_ac+nl_dc+1 : nl_ac+nl_dc+nl_vsc]
    for i in 1:nl_vsc
        status_str = vsc_status[i] == 1 ? "(在线可用)" : "(离线不可用)"
        println("VSC $i: α = $(vsc_status[i]) $status_str")
        if vsc_status[i] == 1
            vsc_online_count += 1
        end
    end
    println("在线VSC数量: $vsc_online_count")
end

println("\n" * "=" ^ 60)
println("分析结论:")
println("=" ^ 60)
println("1. AC侧总负荷: $(round(total_ac_p, digits=2)) MW")
println("2. DC侧总负荷: $(round(total_dc_p, digits=2)) MW")
println("3. 总负荷: $(round(total_ac_p + total_dc_p, digits=2)) MW")
println("4. 发电机总容量: $(round(total_gen_pmax, digits=2)) MW")
println("5. 在线VSC最大传输能力: $(vsc_online_count * 100) MW (每个100MW)")

println("\n" * "=" ^ 60)
println("关键检查 - DC侧是否需要VSC供电:")
println("=" ^ 60)

# 检查DC侧是否有发电
dc_gen_pmax = 0.0
for i in 1:ng
    bus = Int(jpc_tp.genAC[i, 1])
    if bus > nb_ac
        dc_gen_pmax += jpc_tp.genAC[i, 2]
    end
end
println("DC侧发电机容量: $(round(dc_gen_pmax, digits=2)) MW")

# 检查DC侧光伏
dc_pv_pmax = 0.0
if size(jpc_tp.pv, 1) > 0
    for i in 1:size(jpc_tp.pv, 1)
        pmax = jpc_tp.pv[i, 3]
        dc_pv_pmax += pmax
        println("DC光伏 $i: Pmax=$(round(pmax, digits=2)) MW")
    end
end
println("DC侧光伏容量: $(round(dc_pv_pmax, digits=2)) MW")

dc_total_gen = dc_gen_pmax + dc_pv_pmax
println("\nDC侧总发电能力: $(round(dc_total_gen, digits=2)) MW")
println("DC侧总负荷: $(round(total_dc_p, digits=2)) MW")

if total_dc_p > dc_total_gen
    deficit = total_dc_p - dc_total_gen
    println("\n【结论】DC侧发电不足，缺口: $(round(deficit, digits=2)) MW")
    println("VSC 应该从AC侧向DC侧传输 $(round(deficit, digits=2)) MW 功率 (pinv)")
else
    surplus = dc_total_gen - total_dc_p
    println("\n【结论】DC侧发电充足，盈余: $(round(surplus, digits=2)) MW")
    println("如果有DC侧向AC侧传输需求，VSC 可以传输 $(round(surplus, digits=2)) MW 功率 (prec)")
end

# 检查DC母线上连接的设备
println("\n" * "=" ^ 60)
println("DC母线详情 (母线编号 $(nb_ac+1) - $nb):")
println("=" ^ 60)

for dc_bus_idx in 1:nb_dc
    actual_bus = nb_ac + dc_bus_idx
    println("\nDC母线 $actual_bus (原编号 $dc_bus_idx):")
    
    # 检查连接的VSC
    vsc_list = []
    for i in 1:nl_vsc
        if Int(jpc_tp.converter[i, 2]) == actual_bus
            status = Int(jpc_tp.converter[i, 3])
            push!(vsc_list, (i, status))
        end
    end
    if !isempty(vsc_list)
        for (vsc_idx, status) in vsc_list
            status_str = status == 1 ? "在线" : "离线"
            println("  - 连接 VSC $vsc_idx ($status_str)")
        end
    else
        println("  - 无VSC连接")
    end
    
    # 检查连接的负荷
    for i in 1:size(jpc_tp.loadDC, 1)
        if Int(jpc_tp.loadDC[i, 2]) + nb_ac == actual_bus
            println("  - 负荷: P=$(round(jpc_tp.loadDC[i, 3], digits=2)) MW")
        end
    end
    
    # 检查连接的光伏
    for i in 1:size(jpc_tp.pv, 1)
        if Int(jpc_tp.pv[i, 2]) == actual_bus
            println("  - 光伏: Pmax=$(round(jpc_tp.pv[i, 3], digits=2)) MW")
        end
    end
end
