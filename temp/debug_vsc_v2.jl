# 调试VSC/逆变器数据 - 函数版本
project_root = dirname(@__DIR__)
cd(project_root)

using Pkg
Pkg.activate(project_root)

include(joinpath(project_root, "src", "workflows.jl"))
using .Workflows

function main()
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
        P = jpc_tp.loadAC[i, 4]  # 第4列是有功负荷
        Q = jpc_tp.loadAC[i, 5]  # 第5列是无功负荷
        total_ac_p += P
        total_ac_q += Q
        println("  负荷 $i: 母线=$bus, P=$(round(P, digits=2)) MW, Q=$(round(Q, digits=2)) MVar")
    end
    println("  【AC侧总负荷: P=$(round(total_ac_p, digits=2)) MW, Q=$(round(total_ac_q, digits=2)) MVar】")

    # DC 负荷
    total_dc_p = 0.0
    println("\nDC负荷:")
    for i in 1:size(jpc_tp.loadDC, 1)
        bus = Int(jpc_tp.loadDC[i, 2])  # 已经包含了 nb_ac 偏移
        P = jpc_tp.loadDC[i, 4]  # 注意：第4列是有功负荷
        total_dc_p += P
        println("  负荷 $i: 母线=$bus, P=$(round(P, digits=2)) MW")
    end
    println("  【DC侧总负荷: P=$(round(total_dc_p, digits=2)) MW】")

    println("\n" * "=" ^ 60)
    println("发电机信息:")
    println("=" ^ 60)
    total_gen_pmax = 0.0
    for i in 1:ng
        bus = Int(jpc_tp.genAC[i, 1])
        Pg = jpc_tp.genAC[i, 2]   # 当前有功输出
        Pmax = jpc_tp.genAC[i, 9] # 最大有功功率
        Pmin = jpc_tp.genAC[i, 10] # 最小有功功率
        total_gen_pmax += Pmax
        bus_type = bus <= nb_ac ? "AC" : "DC"
        println("发电机 $i: 母线=$bus ($bus_type), Pg=$(round(Pg, digits=2)) MW, Pmax=$(round(Pmax, digits=2)) MW, Pmin=$(round(Pmin, digits=2)) MW")
    end
    println("【发电机总容量: Pmax=$(round(total_gen_pmax, digits=2)) MW】")

    println("\n" * "=" ^ 60)
    println("α_pre (初始开关状态) 中的VSC状态:")
    println("=" ^ 60)
    α_pre = jpc_tp[:α_pre]
    vsc_online_count = 0
    vsc_status = zeros(nl_vsc)
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
            dc_gen_pmax += jpc_tp.genAC[i, 9]  # 第9列是PMAX
        end
    end
    println("DC侧发电机容量: $(round(dc_gen_pmax, digits=2)) MW")

    # 检查DC侧光伏 (从jpc_tp.pv矩阵) - 第3列是p_max
    dc_pv_pmax = 0.0
    if size(jpc_tp.pv, 1) > 0
        println("\nDC光伏 (jpc_tp.pv矩阵):")
        for i in 1:size(jpc_tp.pv, 1)
            pmax = jpc_tp.pv[i, 3]  # 第3列是计算得出的p_max
            dc_pv_pmax += pmax
            println("  DC光伏 $i: 母线=$(Int(jpc_tp.pv[i, 2])), Pmax=$(round(pmax, digits=4)) MW")
        end
        println("【DC侧光伏总容量: $(round(dc_pv_pmax, digits=4)) MW】")
    else
        println("\n【警告】没有找到DC光伏数据 (jpc_tp.pv 为空)")
    end

    # 检查AC侧光伏系统 (从jpc_tp.pv_acsystem矩阵) - 第10列是p_mw
    ac_pv_pmax = 0.0
    if size(jpc_tp.pv_acsystem, 1) > 0
        println("\nAC光伏系统 (jpc_tp.pv_acsystem矩阵):")
        for i in 1:size(jpc_tp.pv_acsystem, 1)
            pmax = jpc_tp.pv_acsystem[i, 10]  # 第10列是p_mw
            ac_pv_pmax += pmax
            println("  AC光伏 $i: 母线=$(Int(jpc_tp.pv_acsystem[i, 2])), Pmax=$(round(pmax, digits=4)) MW")
        end
        println("【AC侧光伏总容量: $(round(ac_pv_pmax, digits=4)) MW】")
    else
        println("\n【警告】没有找到AC光伏数据 (jpc_tp.pv_acsystem 为空)")
    end

    # 检查 Pmgmax (从 jpc_tp 字典获取，这是优化模型使用的数据)
    println("\n" * "-" ^ 40)
    println("优化模型使用的微网数据 (jpc_tp[:Pmgmax]):")
    println("-" ^ 40)
    Pmgmax = jpc_tp[:Pmgmax]
    nmg = jpc_tp[:nmg]
    println("微网数量 (nmg): $nmg")
    println("Pmgmax 向量长度: $(length(Pmgmax))")
    mg_total = sum(Pmgmax) * baseMVA  # 转换为实际值
    for i in 1:length(Pmgmax)
        println("  微网 $i: Pmgmax=$(round(Pmgmax[i] * baseMVA, digits=4)) MW (p.u.=$(round(Pmgmax[i], digits=4)))")
    end
    println("【微网总容量: $(round(mg_total, digits=4)) MW】")
    
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
        if surplus > 0
            println("如果有DC侧向AC侧传输需求，VSC 可以传输 $(round(surplus, digits=2)) MW 功率 (prec)")
        end
    end

    # 检查DC母线上连接的设备
    println("\n" * "=" ^ 60)
    println("DC母线详情 (母线编号 $(nb_ac+1) - $nb):")
    println("=" ^ 60)

    for dc_bus_idx in 1:nb_dc
        actual_bus = nb_ac + dc_bus_idx
        println("\nDC母线 $actual_bus (原编号 $dc_bus_idx):")
        
        # 检查连接的VSC
        for i in 1:nl_vsc
            if Int(jpc_tp.converter[i, 2]) == actual_bus
                status = Int(jpc_tp.converter[i, 3])
                status_str = status == 1 ? "在线" : "离线"
                println("  - 连接 VSC $i ($status_str)")
            end
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
    
    # 最终建议
    println("\n" * "=" ^ 60)
    println("最终分析:")
    println("=" ^ 60)
    println("如果优化结果中 prec = pinv = 0，可能原因:")
    println("1. DC侧没有负荷 (当前DC负荷: $(round(total_dc_p, digits=2)) MW)")
    println("2. DC侧本地发电足够 (当前DC发电: $(round(dc_total_gen, digits=2)) MW)")
    println("3. 所有VSC都离线 (当前在线: $vsc_online_count)")
    println("4. VSC连接的DC母线与有负荷的DC母线之间没有线路连接")
end

main()
