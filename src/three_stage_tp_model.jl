using LinearAlgebra

include(joinpath(@__DIR__, "..", "solvers", "mixed_integer_linear_programming.jl"))

struct FaultIsolationIndices
    # Binary variables
    z_start::Int      # 故障区域指示变量
    β_start::Int      # 线路开关状态变量
    γ_start::Int      # 节点通电状态变量
    
    # Power variables
    pls_start::Int    # 节点有功负荷削减
    qls_start::Int    # 节点无功负荷削减
    pij_start::Int    # 线路有功功率
    qij_start::Int    # 线路无功功率
    pg_start::Int     # 发电机出力
    qg_start::Int     # 发电机无功功率
    pm_start::Int     # 分布式电源有功功率
    qm_start::Int     # 分布式电源无功功率
    prec_start::Int   # VSC交流传入直流侧
    pinv_start::Int   # VSC直流传入交流侧

    # voltage variables
    v_start::Int      
    
    # Total number of variables
    n_vars::Int
    
    function FaultIsolationIndices(nl::Int, nl_vsc::Int, nb::Int, ng::Int, nd::Int, nmg::Int)
        z_start = 1
        β_start = z_start + nb
        γ_start = β_start + nl + nl_vsc
        pls_start = γ_start + nl_vsc
        qls_start = pls_start + nb
        pij_start = qls_start + nb
        qij_start = pij_start + nl
        pg_start = qij_start + nl
        qg_start = pg_start + ng
        pm_start = qg_start + ng
        qm_start = pm_start + nmg
        prec_start = qm_start + nmg
        pinv_start = prec_start + nl_vsc
        v_start = pinv_start + nl_vsc
        n_vars = v_start + nb
        
        new(z_start, β_start, γ_start,
        pls_start, qls_start,
        pij_start, qij_start, 
        pg_start, qg_start, pm_start, qm_start,
        prec_start, pinv_start, 
        v_start,
        n_vars)
    end
end

struct PostFaultReconfigIndices
    # Binary variables
    β2_start::Int     # 第二阶段线路开关状态
    Is2_start::Int    # 第二阶段孤岛指示变量
    
    # Virtual Power Flow variables
    Fij2_start::Int   # 第二阶段虚拟潮流
    Fij_vsc2_start::Int # 第二阶段VSC虚拟潮流（如果有）
    Fg2_start::Int    # 第二阶段电源
    Fvg2_start::Int   # 第二阶段虚拟电源
    
    # Power variables
    pls2_start::Int   # 第二阶段节点有功负荷削减
    qls2_start::Int   # 第二阶段节点无功负荷削减
    pij2_start::Int   # 第二阶段线路有功功率
    qij2_start::Int   # 第二阶段线路无功功率
    pg2_start::Int    # 第二阶段发电机有功功率
    qg2_start::Int    # 第二阶段发电机无功功率
    pm2_start::Int    # 第二阶段分布式电源有功功率
    qm2_start::Int    # 第二阶段分布式电源无功功率
    prec2_start::Int  # 第二阶段VSC发电机有功功率
    pinv2_start::Int  # 第二阶段VSC逆变器有功功率

    # voltage variables
    v2_start::Int     # 第二阶段节点电压
    
    # Total number of variables
    n_vars::Int
    
    function PostFaultReconfigIndices(nl::Int, nl_vsc::Int, nb::Int, ng::Int, nd::Int, nmg::Int)
        β2_start = 1
        Is2_start = β2_start + nl + nl_vsc
        Fij2_start = Is2_start + nl + nl_vsc
        Fij_vsc2_start = Fij2_start + nl
        Fg2_start = Fij_vsc2_start + nl_vsc
        Fvg2_start = Fg2_start + ng
        pls2_start = Fvg2_start + nb - ng
        qls2_start = pls2_start + nb
        pij2_start = qls2_start + nb
        qij2_start = pij2_start + nl
        pg2_start = qij2_start + nl
        qg2_start = pg2_start + ng
        pm2_start = qg2_start + ng
        qm2_start = pm2_start + nmg
        prec2_start = qm2_start + nmg
        pinv2_start = prec2_start + nl_vsc
        v2_start = pinv2_start + nl_vsc
        n_vars = v2_start + nb
        
        new(β2_start, Is2_start,
        Fij2_start, Fij_vsc2_start, Fg2_start, Fvg2_start, 
        pls2_start, qls2_start,
        pij2_start, qij2_start, pg2_start, qg2_start, pm2_start, qm2_start,
        prec2_start, pinv2_start,
        v2_start,
        n_vars)
    end
end

struct PostRepairReconfigIndices
    # Binary variables
    β3_start::Int     # 第三阶段线路开关状态
    Is3_start::Int    # 孤岛指示变量
    
    # Flow variables
    Fij3_start::Int   # 第三阶段虚拟潮流
    Fij_vsc3_start::Int # 第三阶段VSC虚拟潮流（如果有）
    Fg3_start::Int    # 第三阶段电源
    Fvg3_start::Int   # 第三阶段虚拟电源
    
    # Power variables
    pls3_start::Int   # 第三阶段节点有功负荷削减
    qls3_start::Int   # 第三阶段节点无功负荷削减
    pij3_start::Int   # 第三阶段线路有功功率
    qij3_start::Int   # 第三阶段线路无功功率
    pg3_start::Int    # 第三阶段发电机出力
    qg3_start::Int    # 第三阶段发电机无功功率
    pm3_start::Int    # 第三阶段分布式电源有功功率
    qm3_start::Int    # 第三阶段分布式电源无功功率
    prec3_start::Int  # 第三阶段VSC发电机出力
    pinv3_start::Int  # 第三阶段VSC逆变器出力

    # voltage variables
    v3_start::Int     # 第三阶段节点电压
    
    # Total number of variables
    n_vars::Int
    
    function PostRepairReconfigIndices(nl::Int, nl_vsc::Int, nb::Int, ng::Int, nd::Int, nmg::Int)
        β3_start = 1
        Is3_start = β3_start + nl + nl_vsc
        Fij3_start = Is3_start + nl + nl_vsc
        Fij_vsc3_start = Fij3_start + nl
        Fg3_start = Fij_vsc3_start + nl_vsc
        Fvg3_start = Fg3_start + ng
        pls3_start = Fvg3_start + nb
        qls3_start = pls3_start + nb
        pij3_start = qls3_start + nb
        qij3_start = pij3_start + nl
        pg3_start = qij3_start + nl
        qg3_start = pg3_start + ng
        pm3_start = qg3_start + ng
        qm3_start = pm3_start + nmg
        prec3_start = qm3_start + nmg
        pinv3_start = prec3_start + nl_vsc
        v3_start = pinv3_start + nl_vsc
        n_vars = v3_start + nb
        
        new(β3_start, Is3_start, 
        Fij3_start, Fij_vsc3_start, Fg3_start, Fvg3_start, 
        pls3_start, qls3_start,
        pij3_start, qij3_start, pg3_start, qg3_start, pm3_start, qm3_start,
        prec3_start, pinv3_start,
        v3_start, 
        n_vars)
    end
end

# Helper functions
function get_range(start::Int, length::Int)
    return start:start+length-1
end

function get_variable_values(x::Vector, idx, var_name::Symbol, dim::Int)
    start_idx = getfield(idx, Symbol(var_name, "_start"))
    return x[get_range(start_idx, dim)]
end

"""
第一阶段：故障隔离阶段 (0 - τ_SW)
"""
function solve_fault_isolation(jpc, fault_lines)   
    # 提取网络参数
    Cft_ac = jpc[:Cft_ac]  # AC lines
    Cft_dc = jpc[:Cft_dc]  # DC lines
    Cft = vcat(Cft_ac, Cft_dc)  # merge AC and DC lines
    Cft_vsc = jpc[:Cft_vsc]  # VSC lines
    Cg = jpc[:Cg]
    Cd = jpc[:Cd]
    Cmg = jpc[:Cmg]
    Pd = jpc[:Pd]
    Qd = jpc[:Qd]

    Pgmax = jpc[:Pgmax]
    Qgmax = jpc[:Qgmax]
    Pmgmax = jpc[:Pmgmax]
    Qmgmax = jpc[:Qmgmax]
    Pvscmax = jpc[:Pvscmax]
    Smax = jpc[:Smax]

    R = jpc[:R]
    X = jpc[:X]
    VMAX = jpc[:VMAX]
    VMIN = jpc[:VMIN]
    bigM = jpc[:bigM]
    η = jpc[:η]  # VSC efficiency
    c_load = jpc[:c_load]
    c_vsc = jpc[:c_vsc]

    α = jpc[:α_pre]
    r = jpc[:r]
    
    nb_ac = jpc[:nb_ac]
    nb_dc = jpc[:nb_dc]
    nb = nb_ac + nb_dc
    ng = jpc[:ng]
    nl_ac = jpc[:nl_ac]
    nl_dc = jpc[:nl_dc]
    nl = nl_ac + nl_dc
    nl_vsc = jpc[:nl_vsc]
    nmg = jpc[:nmg]
    nd = jpc[:nd]

    fault_infos = Vector{NamedTuple{(:line, :nodes), Tuple{Int, Vector{Int}}}}(undef, length(fault_lines))
    total_fault_nodes = 0
    for (i, fault_line) in enumerate(fault_lines)
        if fault_line <= nl
            connected_nodes = findall(x -> x != 0, Cft[fault_line, :])
        else
            connected_nodes = findall(x -> x != 0, Cft_vsc[fault_line - nl, :])
        end
        fault_infos[i] = (line = fault_line, nodes = connected_nodes)
        total_fault_nodes += length(connected_nodes)
    end

    idx = FaultIsolationIndices(nl, nl_vsc, nb, ng, nd, nmg)
    
    # Set up bounds
    lb = fill(-Inf, idx.n_vars)
    ub = fill(Inf, idx.n_vars)
    # Variables for common condition
    # Bounds for z
    lb[get_range(idx.z_start, nb)] .= 0
    ub[get_range(idx.z_start, nb)] .= 1
    # Bounds for β
    lb[get_range(idx.β_start, nl + nl_vsc)] .= 0
    ub[get_range(idx.β_start, nl + nl_vsc)] .= 1
    # Bounds for γ
    lb[get_range(idx.γ_start, nl_vsc)] .= 0
    ub[get_range(idx.γ_start, nl_vsc)] .= 1
    # Bounds for pls
    lb[get_range(idx.pls_start, nb)] .= 0
    ub[get_range(idx.pls_start, nb)] .= Cd' * Pd
    # Bounds for qls
    lb[get_range(idx.qls_start, nb)] .= 0
    ub[get_range(idx.qls_start, nb)] .= Cd' * Qd
    # Bounds for pij
    lb[get_range(idx.pij_start, nl)] .= -Smax
    ub[get_range(idx.pij_start, nl)] .= Smax
    # Bounds for qij
    lb[get_range(idx.qij_start, nl)] .= -Smax
    ub[get_range(idx.qij_start, nl)] .= Smax
    # Bounds for pg
    lb[get_range(idx.pg_start, ng)] .= 0
    ub[get_range(idx.pg_start, ng)] .= Pgmax
    # Bounds for qg
    lb[get_range(idx.qg_start, ng)] .= 0
    ub[get_range(idx.qg_start, ng)] .= Qgmax
    # Bounds for pmg
    lb[get_range(idx.pm_start, nmg)] .= 0
    ub[get_range(idx.pm_start, nmg)] .= Pmgmax
    # Bounds for qmg
    lb[get_range(idx.qm_start, nmg)] .= 0
    ub[get_range(idx.qm_start, nmg)] .= Qmgmax
    # Bounds for prec
    lb[get_range(idx.prec_start, nl_vsc)] .= 0
    ub[get_range(idx.prec_start, nl_vsc)] .= Pvscmax
    # Bounds for pinv
    lb[get_range(idx.pinv_start, nl_vsc)] .= 0
    ub[get_range(idx.pinv_start, nl_vsc)] .= Pvscmax
    # Bounds for v
    lb[get_range(idx.v_start, nb)] .= VMIN
    ub[get_range(idx.v_start, nb)] .= VMAX

    # Objective function
    # min - sum(β) + sum(z) - sum(γ) + c*sum(pls)
    c = zeros(idx.n_vars)
    c[get_range(idx.z_start, nb)] .= 1  # 负荷削减项权重
    c[get_range(idx.β_start, nl + nl_vsc)] .= -1  # 开关状态项权重
    c[get_range(idx.γ_start, nl_vsc)] .= -1  # VSC配置状态项权重
    c[get_range(idx.pls_start, nb)] .= c_load  # 负荷削减项权重

    # Calculate number of constraints
    n_eq = 2*nb     # Power balance constraints
    nf = length(fault_infos)
    n_common = nl_vsc + 1
    n_fault_topology = nf * (4*nl + 4*nl_vsc) + total_fault_nodes
    n_tail = 4*nl + 4*nl_vsc + 2*nb
    n_ineq = n_common + n_fault_topology + n_tail

    Aeq = spzeros(n_eq, idx.n_vars)
    beq = zeros(n_eq)
    A = spzeros(n_ineq, idx.n_vars)
    b = zeros(n_ineq)
    eq_row = 1
    ineq_row = 1

    ## Constraints for common condition
    # 直接得到正常情况下的拓扑α
    # (0) γ >= α  ————nl_vsc
    # @constraint(model, γ .>= α);
    for i = 1:nl_vsc
        A[ineq_row, idx.γ_start+i-1] = -1
        b[ineq_row] = -α[i+nl]
        ineq_row += 1
    end

    # (0) sum(γ) >= 2  ————1
    A[ineq_row, get_range(idx.γ_start, nl_vsc)] .= -1  # 对所有 γ[i] 加权系数为 1
    b[ineq_row] = -2  # 右侧常数为 2
    ineq_row += 1

    ## Topology constraints
    for fault in fault_infos
        connected_nodes = fault.nodes
        fault_line = fault.line
        # (1) 故障线路传播约束: βij[fault] <= zi、zj————2
        for j in connected_nodes
            A[ineq_row, idx.z_start + j - 1] = -1  # zj
            A[ineq_row, idx.β_start + fault_line - 1] = 1  # +βij
            ineq_row += 1
        end

        # (2) 故障节点传播约束: zj - (1-βij) <= zi <= zj + (1-βij)————2*(nl+nl_vsc)
        for j in 1:nl
            row_j = Cft[j, :]  # 第j条线路的连接关系
            # zi <= zj + (1 - βij)  <=>  (zi - zj) + βij <= 1
            A[ineq_row, get_range(idx.z_start, nb)] = row_j   # zi - zj
            A[ineq_row, idx.β_start + j - 1] = 1              # +βij
            b[ineq_row] = 1
            ineq_row += 1
            # zj - (1 - βij) <= zi  <=>  (zj - zi) + βij <= 1
            A[ineq_row, get_range(idx.z_start, nb)] = -row_j  # -zi + zj
            A[ineq_row, idx.β_start + j - 1] = 1              # +βij
            b[ineq_row] = 1
            ineq_row += 1
        end
        for j in 1:nl_vsc
            row_j = Cft_vsc[j, :]  # 第j条VSC线路的连接关系
            # zi <= zj + (1 - βij)  <=>  (zi - zj) + βij <= 1
            A[ineq_row, get_range(idx.z_start, nb)] = row_j   # zi - zj
            A[ineq_row, idx.β_start + nl + j - 1] = 1         # +βij
            b[ineq_row] = 1
            ineq_row += 1
            # zj - (1 - βij) <= zi  <=>  (zj - zi) + βij <= 1
            A[ineq_row, get_range(idx.z_start, nb)] = -row_j  # -zi + zj
            A[ineq_row, idx.β_start + nl + j - 1] = 1         # +βij
            b[ineq_row] = 1
            ineq_row += 1
        end

        # (3) 开关隔离约束：αij*(1-rij) <= βij <= αij————2(nl + nl_vsc)
        for j in 1:nl+nl_vsc
            # αij*(1-rij) <= βij
            A[ineq_row, idx.β_start + j - 1] = -1
            b[ineq_row] = -α[j] * (1 - r[j])  # 左侧常数
            ineq_row += 1
            # βij <= αij
            A[ineq_row, idx.β_start + j - 1] = 1
            b[ineq_row] = α[j]  # 右侧常数
            ineq_row += 1
        end
    end
    # (4) power balance  ————2*nb
    for i = 1:nb_ac
        # 交流节点
        Aeq[eq_row, get_range(idx.pij_start, nl)] .= Cft'[i, :]
        Aeq[eq_row, get_range(idx.pg_start, ng)] .= -Cg'[i, :]
        Aeq[eq_row, get_range(idx.pm_start, nmg)] .= -Cmg'[i, :]
        Aeq[eq_row, idx.pls_start + i - 1] = -1
        # 对于交流节点， prec不用乘损耗系数，pinv需要乘损耗系数
        Aeq[eq_row, get_range(idx.prec_start, nl_vsc)] .= Cft_vsc'[i, :]
        Aeq[eq_row, get_range(idx.pinv_start, nl_vsc)] .= -Cft_vsc'[i, :] .* η
        beq[eq_row] = -dot(Cd'[i, :], Pd)
        eq_row += 1
    end
    for i = nb_ac+1:nb
        # 直流节点
        Aeq[eq_row, get_range(idx.pij_start, nl)] .= Cft'[i, :]
        Aeq[eq_row, get_range(idx.pg_start, ng)] .= -Cg'[i, :]
        Aeq[eq_row, get_range(idx.pm_start, nmg)] .= -Cmg'[i, :]
        Aeq[eq_row, idx.pls_start + i - 1] = -1
        # 对于直流节点， prec需要乘损耗系数，pinv不用乘损耗系数
        Aeq[eq_row, get_range(idx.prec_start, nl_vsc)] .= Cft_vsc'[i, :] .* η
        Aeq[eq_row, get_range(idx.pinv_start, nl_vsc)] .= -Cft_vsc'[i, :]
        beq[eq_row] = -dot(Cd'[i, :], Pd)
        eq_row += 1
    end
    # @constraint(model, Cft'*qij .== Cg'*qg - Cd'*(Qd - qls) + Cmg'*qm);
    Aeq[eq_row:eq_row+nb-1, get_range(idx.qij_start, nl)] = Cft'
    Aeq[eq_row:eq_row+nb-1, get_range(idx.qg_start, ng)] = -Cg'
    Aeq[eq_row:eq_row+nb-1, get_range(idx.qm_start, nmg)] = -Cmg'
    for i = 1:nb
        Aeq[eq_row + i - 1, idx.qls_start + i - 1] = -1  # qls
    end
    beq[eq_row:eq_row+nb-1] = -Cd'*Qd
    eq_row += nb

    #（5）-Smax.*βij <= pij/qij .<= Smax.*βij  ————4*nl
    # AC
    for i = 1:nl_ac
        # pij .<= Smax.*βij
        A[ineq_row, idx.pij_start+i-1] = 1
        A[ineq_row, idx.β_start+i-1] = -Smax[i]
        ineq_row += 1
        # pij .>= -Smax.*βij
        A[ineq_row, idx.pij_start+i-1] = -1
        A[ineq_row, idx.β_start+i-1] = -Smax[i]
        ineq_row += 1
        # qij .<= Smax.*βij
        A[ineq_row, idx.qij_start+i-1] = 1
        A[ineq_row, idx.β_start+i-1] = -Smax[i]
        ineq_row += 1
        # qij .>= -Smax.*βij
        A[ineq_row, idx.qij_start+i-1] = -1
        A[ineq_row, idx.β_start+i-1] = -Smax[i]
        ineq_row += 1
    end
    # DC
    for i = (nl_ac+1):nl
        # pij .<= bigM.*βij
        A[ineq_row, idx.pij_start+i-1] = 1
        A[ineq_row, idx.β_start+i-1] = -bigM
        ineq_row += 1
        # pij .>= -bigM.*βij
        A[ineq_row, idx.pij_start+i-1] = -1
        A[ineq_row, idx.β_start+i-1] = -bigM
        ineq_row += 1
        # qij .<= bigM.*βij
        A[ineq_row, idx.qij_start+i-1] = 1
        A[ineq_row, idx.β_start+i-1] = -bigM
        ineq_row += 1
        # qij .>= -bigM.*βij
        A[ineq_row, idx.qij_start+i-1] = -1
        A[ineq_row, idx.β_start+i-1] = -bigM
        ineq_row += 1
    end

    # (6) VSC power constraints ————4*nl_vsc
    for i = 1:nl_vsc
        # 0 <= pij_rec .<= Pvscmax.*βij
        A[ineq_row, idx.prec_start+i-1] = 1
        A[ineq_row, idx.β_start+nl+i-1] = -Pvscmax[i]
        ineq_row += 1
        A[ineq_row, idx.prec_start+i-1] = -1
        b[ineq_row] = 0
        ineq_row += 1
        # 0 <= pij_inv .<= Pvscmax.*βij
        A[ineq_row, idx.pinv_start+i-1] = 1
        A[ineq_row, idx.β_start+nl+i-1] = -Pvscmax[i]
        ineq_row += 1
        A[ineq_row, idx.pinv_start+i-1] = -1
        b[ineq_row] = 0
        ineq_row += 1
    end

    # (7) 故障节点负荷削减约束：-M*(1-z) <= pls - pd ≤ M*(1-z)————2*nb
    for i = 1:nb
        # pls - pd <= M*(1-z)
        A[ineq_row, idx.pls_start+i-1] = 1
        A[ineq_row, idx.z_start+i-1] = bigM
        b[ineq_row] = bigM + dot(Cd'[i, :], Pd)
        ineq_row += 1
        # pls - pd >= -M*(1-z)
        A[ineq_row, idx.pls_start+i-1] = -1
        A[ineq_row, idx.z_start+i-1] = bigM
        b[ineq_row] = bigM - dot(Cd'[i, :], Pd)
        ineq_row += 1
    end

    # 设置变量类型
    vtypes = fill('C', idx.n_vars)
    vtypes[get_range(idx.z_start, nb)] .= 'B'
    vtypes[get_range(idx.β_start, nl+nl_vsc)] .= 'B'
    vtypes[get_range(idx.γ_start, nl_vsc)] .= 'B'
    
    # 求解MILP
    results = solve_milp_sparse(c, A, b, Aeq, beq, lb, ub, vtypes)
    
    # 解码解
    x = results[:x]
    z = get_variable_values(x, idx, :z, nb)
    β = get_variable_values(x, idx, :β, nl + nl_vsc)
    γ = get_variable_values(x, idx, :γ, nl_vsc)
    pls = get_variable_values(x, idx, :pls, nb)
    pij = get_variable_values(x, idx, :pij, nl)
    pg = get_variable_values(x, idx, :pg, ng)
    pm = get_variable_values(x, idx, :pm, nmg)
    prec = get_variable_values(x, idx, :prec, nl_vsc)
    pinv = get_variable_values(x, idx, :pinv, nl_vsc)
    
    # 计算负荷削减
    Pls = sum(pls)
    
    # 获取baseMVA用于转换为实际MW值
    baseMVA = jpc["baseMVA"]
    
    println("=== 第一阶段：故障隔离结果 ===")
    println("节点是否处于故障区域", z)
    println("线路开关状态 (β): ", β)
    println("VSC配置状态 (γ): ", γ)
    println("节点有功负荷削减 (pls): ", round.(pls .* baseMVA, digits=2), " MW")
    println("线路有功功率 (pij): ", round.(pij .* baseMVA, digits=2), " MW")
    println("发电机出力 (pg): ", round.(pg .* baseMVA, digits=2), " MW")
    println("微电网发电机出力 (pm): ", round.(pm .* baseMVA, digits=2), " MW")
    println("VSC交流传入直流功率 (prec): ", round.(prec .* baseMVA, digits=2), " MW")
    println("VSC直流传入交流功率 (pinv): ", round.(pinv .* baseMVA, digits=2), " MW")
    println("节点有功负荷削减总和 (Pls): ", round(Pls * baseMVA, digits=2), " MW")

    return Dict(
        :z => z, :β1 => β, :γ => γ, 
        :pls1 => pls, :Pls1 => Pls,
        :objective => results[:objval], :status => results[:status_name], :runtime => results[:runtime]
    )
end

"""
第二阶段：故障后切除阶段 (τ_SW - τ_TP)
"""
function solve_post_fault_reconfig(jpc, stage1_result)
    # 提取网络参数
    Cft_ac = jpc[:Cft_ac]  # AC lines
    Cft_dc = jpc[:Cft_dc]  # DC lines
    Cft = vcat(Cft_ac, Cft_dc)  # merge AC and DC lines
    Cft_vsc = jpc[:Cft_vsc]  # VSC lines
    Cg = jpc[:Cg]
    Cd = jpc[:Cd]
    Cdf = jpc[:Cdf]
    Cmg = jpc[:Cmg]
    Fd = jpc[:Fd]
    Pd = jpc[:Pd]
    Qd = jpc[:Qd]

    Pgmax = jpc[:Pgmax]
    Qgmax = jpc[:Qgmax]
    Pmgmax = jpc[:Pmgmax]
    Qmgmax = jpc[:Qmgmax]
    Pvscmax = jpc[:Pvscmax]
    Smax = jpc[:Smax]

    R = jpc[:R]
    X = jpc[:X]
    VMAX = jpc[:VMAX]
    VMIN = jpc[:VMIN]
    bigM = jpc[:bigM]
    η = jpc[:η]  # VSC efficiency

    α = jpc[:α_pre]
    r = jpc[:r]

    # 第一阶段结果
    z = stage1_result[:z]
    β1 = stage1_result[:β1]
    γ = stage1_result[:γ]
   
    nb_ac = jpc[:nb_ac]
    nb_dc = jpc[:nb_dc]
    nb = nb_ac + nb_dc
    ng = jpc[:ng]
    nl_ac = jpc[:nl_ac]
    nl_dc = jpc[:nl_dc]
    nl = nl_ac + nl_dc
    nl_vsc = jpc[:nl_vsc]
    nmg = jpc[:nmg]
    nd = jpc[:nd]
    
    # 创建变量索引
    idx = PostFaultReconfigIndices(nl, nl_vsc, nb, ng, nd, nmg)

    # 设置变量边界
    lb = fill(-Inf, idx.n_vars)
    ub = fill(Inf, idx.n_vars)
    # β2变量边界
    lb[get_range(idx.β2_start, nl+nl_vsc)] .= 0
    ub[get_range(idx.β2_start, nl+nl_vsc)] .= 1
    # Is2变量边界
    lb[get_range(idx.Is2_start, nl+nl_vsc)] .= 0
    ub[get_range(idx.Is2_start, nl+nl_vsc)] .= 1
    # Fij2变量边界
    lb[get_range(idx.Fij2_start, nl)] .= -nb
    ub[get_range(idx.Fij2_start, nl)] .= nb
    # Fij_vsc2变量边界
    lb[get_range(idx.Fij_vsc2_start, nl_vsc)] .= -nb
    ub[get_range(idx.Fij_vsc2_start, nl_vsc)] .= nb
    # Fg2变量边界
    lb[get_range(idx.Fg2_start, ng)] .= 0
    ub[get_range(idx.Fg2_start, ng)] .= nb
    # Fvg2变量边界
    lb[get_range(idx.Fvg2_start, nb-ng)] .= 0
    ub[get_range(idx.Fvg2_start, nb-ng)] .= 1
    # pls2变量边界
    lb[get_range(idx.pls2_start, nb)] .= 0
    ub[get_range(idx.pls2_start, nb)] .= Cd' * Pd  # 负荷削减上限
    # qls2变量边界
    lb[get_range(idx.qls2_start, nb)] .= 0
    ub[get_range(idx.qls2_start, nb)] .= Cd' * Qd  # 无功负荷削减上限
    # pij2变量边界
    lb[get_range(idx.pij2_start, nl)] .= -Smax
    ub[get_range(idx.pij2_start, nl)] .= Smax
    # qij2变量边界
    lb[get_range(idx.qij2_start, nl)] .= -Smax
    ub[get_range(idx.qij2_start, nl)] .= Smax
    # pg2变量边界
    lb[get_range(idx.pg2_start, ng)] .= 0
    ub[get_range(idx.pg2_start, ng)] .= Pgmax
    # qg2变量边界
    lb[get_range(idx.qg2_start, ng)] .= 0
    ub[get_range(idx.qg2_start, ng)] .= Qgmax
    # pm2变量边界
    lb[get_range(idx.pm2_start, nmg)] .= 0
    ub[get_range(idx.pm2_start, nmg)] .= Pmgmax
    # qm2变量边界
    lb[get_range(idx.qm2_start, nmg)] .= 0
    ub[get_range(idx.qm2_start, nmg)] .= Qmgmax
    # prec2变量边界
    lb[get_range(idx.prec2_start, nl_vsc)] .= 0
    ub[get_range(idx.prec2_start, nl_vsc)] .= Pvscmax
    # pinv2变量边界
    lb[get_range(idx.pinv2_start, nl_vsc)] .= 0
    ub[get_range(idx.pinv2_start, nl_vsc)] .= Pvscmax
    # v2变量边界
    lb[get_range(idx.v2_start, nb)] .= VMIN
    ub[get_range(idx.v2_start, nb)] .= VMAX
    
    # 目标函数：min sum(Fvg2) + sum(pls2)
    c = zeros(idx.n_vars)
    c[get_range(idx.Fvg2_start, nb-ng)] .= 1  # 虚拟电源权重
    c[get_range(idx.pls2_start, nb)] .= 1  # 负荷削减权重
    c[idx.β2_start + 28 - 1] = 2  # 强制最小化第28条线路的功率
    
    # 计算约束数量
    n_eq = 3*nb + 1  # 虚拟潮流约束 + 功率平衡约束
    n_ineq = 14*nl + 15*nl_vsc + 2*nb  # 各种不等式约束
    
    Aeq = spzeros(n_eq, idx.n_vars)
    beq = zeros(n_eq)
    A = spzeros(n_ineq, idx.n_vars)
    b = zeros(n_ineq)
    
    eq_row = 1
    ineq_row = 1
    
    ## Topology constraints
    # (0) γ >= β2  ————nl_vsc
    # @constraint(model, γ .>= β2);
    for i = 1:nl_vsc
        A[ineq_row, idx.β2_start+nl+i-1] = 1
        b[ineq_row] = γ[i]
        ineq_row += 1
    end

    # (1) 开关状态约束：1 - β2 >= α - β1————nl + nl_vsc
    for i = 1:nl+nl_vsc
        A[ineq_row, idx.β2_start + i - 1] = 1
        b[ineq_row] = 1 - (α[i] - β1[i])
        ineq_row += 1
    end

    # (2) 开关状态约束：zi*(1-zj) + zj*(1-zi) <= 1 - β2————nl + nl_vsc
    for i = 1:nl
        j = findall(x -> x != 0, Cft[i, :])
        A[ineq_row, idx.β2_start + i - 1] = 1
        b[ineq_row] = 1 - z[j[1]]*(1 - z[j[2]]) - z[j[2]]*(1 - z[j[1]])
        ineq_row += 1
    end
    for i = 1:nl_vsc
        j = findall(x -> x != 0, Cft_vsc[i, :])
        A[ineq_row, idx.β2_start + nl + i - 1] = 1
        b[ineq_row] = 1 - z[j[1]]*(1 - z[j[2]]) - z[j[2]]*(1 - z[j[1]])
        ineq_row += 1
    end

    # (2) 开关状态约束：β1 - r <= β2 <= β1 + r————2*(nl + nl_vsc)
    for i = 1:nl+nl_vsc
        # β1 - r <= β2
        A[ineq_row, idx.β2_start + i - 1] = -1
        b[ineq_row] = -(β1[i] - r[i])
        ineq_row += 1
        # β2 <= β1 + r
        A[ineq_row, idx.β2_start + i - 1] = -1
        b[ineq_row] = β1[i] + r[i]
        ineq_row += 1
    end

    # (3) 虚拟潮流约束：sum(Fij2) = Fg2i - Fd2i + Fvg2i————nb
    Aeq[eq_row:eq_row+nb-1, get_range(idx.Fij2_start, nl)] = Cft'
    Aeq[eq_row:eq_row+nb-1, get_range(idx.Fij_vsc2_start, nl_vsc)] = Cft_vsc'
    Aeq[eq_row:eq_row+nb-1, get_range(idx.Fg2_start, ng)] = -Cg'
    Aeq[eq_row:eq_row+nb-1, get_range(idx.Fvg2_start, nb-ng)] = -Cdf'  # 虚拟电源Fvg2i
    beq[eq_row:eq_row+nb-1] = -Cdf' * Fd
    eq_row += nb 

    # (4) -Smax.*β2 <= Fij2/Fij_vsc2 <= Smax.*β2 ————2*(nl+nl_vsc)
    # @constraint(model, Fij2 .<= nb.*β2);
    for i = 1:nl
        A[ineq_row, idx.Fij2_start+i-1] = 1
        A[ineq_row, idx.β2_start+i-1] = -nb
        ineq_row += 1
    end
    # 对于 VSC 配置线路的 Fij_vsc2 .<= nb.*β2
    for i = 1:nl_vsc
        A[ineq_row, idx.Fij_vsc2_start+i-1] = 1
        A[ineq_row, idx.β2_start+nl+i-1] = -nb
        ineq_row += 1
    end
    # @constraint(model, Fij2.>= -nb.*β2);
    for i = 1:nl
        A[ineq_row, idx.Fij2_start+i-1] = -1
        A[ineq_row, idx.β2_start+i-1] = -nb
        ineq_row += 1
    end
    # 对于 VSC 配置线路的 Fij_vsc2 .>= -nb.*β2
    for i = 1:nl_vsc
        A[ineq_row, idx.Fij_vsc2_start+i-1] = -1
        A[ineq_row, idx.β2_start+nl+i-1] = -nb
        ineq_row += 1
    end

    # (5) 辐射状拓扑约束：sum(β2) = nb - ng - sum(Fvg2) + sum(Is2)————1
    Aeq[eq_row, get_range(idx.β2_start, nl+nl_vsc)] .= 1
    Aeq[eq_row, get_range(idx.Fvg2_start, nb-ng)] .= 1  # Fvg2i
    Aeq[eq_row, get_range(idx.Is2_start, nl+nl_vsc)] .= -1  # Is2i
    beq[eq_row] = nb - ng 
    eq_row += 1

    # (6) 故障区域约束：M*(1-z) <= pls2 - pd ≤ M*(1-z)————2*nb
    for i = 1:nb 
        # pls2 - pd <= M*(1-z)
        A[ineq_row, idx.pls2_start + i - 1] = 1
        b[ineq_row] = bigM *(1 - z[i]) + dot(Cd'[i, :], Pd)
        ineq_row += 1
        # pls2 - pd >= -M*(1-z)
        A[ineq_row, idx.pls2_start + i - 1] = -1
        b[ineq_row] = bigM *(1 - z[i]) - dot(Cd'[i, :], Pd)
        ineq_row += 1
    end

    # (7) 孤岛存在验证约束：Is2ij = Fvg2i *Fvg2j *β2ij————4*(nl+nl_vsc)
    for i in 1:nl
        ij = findall(x -> x != 0, Cft[i, :])
        # Is2ij <= Fvg2i
        A[ineq_row, idx.Is2_start + i - 1] = 1
        A[ineq_row, idx.Fvg2_start + ij[1]] = -1  # Fvg2i
        ineq_row += 1
        # Is2ij <= Fvg2j
        A[ineq_row, idx.Is2_start + i - 1] = 1
        A[ineq_row, idx.Fvg2_start + ij[2]] = -1  # Fvg2j
        ineq_row += 1
        # Is2ij <= β2ij
        A[ineq_row, idx.Is2_start + i - 1] = 1
        A[ineq_row, idx.β2_start + i - 1] = -1  # β2ij
        ineq_row += 1
        # Is2ij >= Fvg2i + Fvg2j + β2ij - 2
        A[ineq_row, idx.Is2_start + i - 1] = -1
        A[ineq_row, idx.Fvg2_start + ij[1]] = 1  # Fvg2i
        A[ineq_row, idx.Fvg2_start + ij[2]] = 1  # Fvg2j
        A[ineq_row, idx.β2_start + i - 1] = 1  # β2ij
        b[ineq_row] = 2  # 右侧常数
        ineq_row += 1
    end
    for i in nl+1:nl+nl_vsc
        ij = findall(x -> x != 0, Cft_vsc[i-nl, :])
        # Is2ij <= Fvg2i
        A[ineq_row, idx.Is2_start + i - 1] = 1
        A[ineq_row, idx.Fvg2_start + ij[1]] = -1  # Fvg2i
        ineq_row += 1
        # Is2ij <= Fvg2j
        A[ineq_row, idx.Is2_start + i - 1] = 1
        A[ineq_row, idx.Fvg2_start + ij[2]] = -1  # Fvg2j
        ineq_row += 1
        # Is2ij <= β2ij
        A[ineq_row, idx.Is2_start + i - 1] = 1
        A[ineq_row, idx.β2_start + i - nl] = -1  # β2ij
        ineq_row += 1
        # Is2ij >= Fvg2i + Fvg2j + β2ij - 2
        A[ineq_row, idx.Is2_start + i - 1] = -1
        A[ineq_row, idx.Fvg2_start + ij[1]] = 1  # Fvg2i
        A[ineq_row, idx.Fvg2_start + ij[2]] = 1  # Fvg2j
        A[ineq_row, idx.β2_start + i - nl] = 1  # β2ij
        b[ineq_row] = 2  # 右侧常数
        ineq_row += 1
    end

    # (8) power balance  ————2*nb
    for i = 1:nb_ac
        # 交流节点
        Aeq[eq_row, get_range(idx.pij2_start, nl)] .= Cft'[i, :]
        Aeq[eq_row, get_range(idx.pg2_start, ng)] .= -Cg'[i, :]
        Aeq[eq_row, get_range(idx.pm2_start, nmg)] .= -Cmg'[i, :]
        Aeq[eq_row, idx.pls2_start + i - 1] = -1
        # 对于交流节点， prec不用乘损耗系数，pinv需要乘损耗系数
        Aeq[eq_row, get_range(idx.prec2_start, nl_vsc)] .= Cft_vsc'[i, :]
        Aeq[eq_row, get_range(idx.pinv2_start, nl_vsc)] .= -Cft_vsc'[i, :] .* η
        beq[eq_row] = -dot(Cd'[i, :], Pd)
        eq_row += 1
    end
    for i = nb_ac+1:nb
        # 直流节点
        Aeq[eq_row, get_range(idx.pij2_start, nl)] .= Cft'[i, :]
        Aeq[eq_row, get_range(idx.pg2_start, ng)] .= -Cg'[i, :]
        Aeq[eq_row, get_range(idx.pm2_start, nmg)] .= -Cmg'[i, :]
        Aeq[eq_row, idx.pls2_start + i - 1] = -1
        # 对于直流节点， prec需要乘损耗系数，pinv不用乘损耗系数
        Aeq[eq_row, get_range(idx.prec2_start, nl_vsc)] .= Cft_vsc'[i, :] .* η
        Aeq[eq_row, get_range(idx.pinv2_start, nl_vsc)] .= -Cft_vsc'[i, :]
        beq[eq_row] = -dot(Cd'[i, :], Pd)
        eq_row += 1
    end
    # @constraint(model, Cft'*qij .== Cg'*qg - Cd'*(Qd - qls) + Cmg'*qm);
    Aeq[eq_row:eq_row+nb-1, get_range(idx.qij2_start, nl)] = Cft'
    Aeq[eq_row:eq_row+nb-1, get_range(idx.qg2_start, ng)] = -Cg'
    Aeq[eq_row:eq_row+nb-1, get_range(idx.qm2_start, nmg)] = -Cmg'
    for i = 1:nb
        Aeq[eq_row + i - 1, idx.qls2_start + i - 1] = -1
    end
    beq[eq_row:eq_row+nb-1] = -Cd'*Qd
    eq_row += nb

    #（9）-Smax.*β2 <= pij2/qij2 .<= Smax.*β2  ————4*nl
    # AC
    for i = 1:nl_ac
        # pij2 .<= Smax.*βij2
        A[ineq_row, idx.pij2_start+i-1] = 1
        A[ineq_row, idx.β2_start+i-1] = -Smax[i]
        ineq_row += 1
        # pij2 .>= -Smax.*βij2
        A[ineq_row, idx.pij2_start+i-1] = -1
        A[ineq_row, idx.β2_start+i-1] = -Smax[i]
        ineq_row += 1
        # qij2 .<= Smax.*βij2
        A[ineq_row, idx.qij2_start+i-1] = 1
        A[ineq_row, idx.β2_start+i-1] = -Smax[i]
        ineq_row += 1
        # qij2 .>= -Smax.*βij2
        A[ineq_row, idx.qij2_start+i-1] = -1
        A[ineq_row, idx.β2_start+i-1] = -Smax[i]
        ineq_row += 1
    end
    # DC
    for i = (nl_ac+1):nl
        # pij2 .<= bigM.*βij2
        A[ineq_row, idx.pij2_start+i-1] = 1
        A[ineq_row, idx.β2_start+i-1] = -bigM
        ineq_row += 1
        # pij2 .>= -bigM.*βij2
        A[ineq_row, idx.pij2_start+i-1] = -1
        A[ineq_row, idx.β2_start+i-1] = -bigM
        ineq_row += 1
        # qij2 .<= bigM.*βij2
        A[ineq_row, idx.qij2_start+i-1] = 1
        A[ineq_row, idx.β2_start+i-1] = -bigM
        ineq_row += 1
        # qij2 .>= -bigM.*βij2
        A[ineq_row, idx.qij2_start+i-1] = -1
        A[ineq_row, idx.β2_start+i-1] = -bigM
        ineq_row += 1
    end

    # (10) VSC power constraints ————4*nl_vsc
    for i = 1:nl_vsc
        # 0 <= prec2 .<= Pvscmax.*βij2
        A[ineq_row, idx.prec2_start+i-1] = 1
        A[ineq_row, idx.β2_start+nl+i-1] = -Pvscmax[i]
        ineq_row += 1
        A[ineq_row, idx.prec2_start+i-1] = -1
        b[ineq_row] = 0
        ineq_row += 1
        # 0 <= pinv2 .<= Pvscmax.*βij2
        A[ineq_row, idx.pinv2_start+i-1] = 1
        A[ineq_row, idx.β2_start+nl+i-1] = -Pvscmax[i]
        ineq_row += 1
        A[ineq_row, idx.pinv2_start+i-1] = -1
        b[ineq_row] = 0
        ineq_row += 1
    end
    
    # 设置变量类型
    vtypes = fill('C', idx.n_vars)
    vtypes[get_range(idx.β2_start, nl+nl_vsc)] .= 'B'
    vtypes[get_range(idx.Is2_start, nl+nl_vsc)] .= 'B'
    vtypes[get_range(idx.Fvg2_start, nb-ng)] .= 'B'

    # 求解MILP
    results = solve_milp_sparse(c, A, b, Aeq, beq, lb, ub, vtypes)
    
    # 解码解
    x = results[:x]
    β2 = get_variable_values(x, idx, :β2, nl+nl_vsc)
    pls2 = get_variable_values(x, idx, :pls2, nb)
    pij2 = get_variable_values(x, idx, :pij2, nl)
    pg2 = get_variable_values(x, idx, :pg2, ng)
    pm2 = get_variable_values(x, idx, :pm2, nmg)
    prec2 = get_variable_values(x, idx, :prec2, nl_vsc)
    pinv2 = get_variable_values(x, idx, :pinv2, nl_vsc)

    # 计算负荷削减
    Pls2 = sum(pls2)
    
    # 获取baseMVA用于转换为实际MW值
    baseMVA = jpc["baseMVA"]
    
    println("=== 第二阶段：故障后切除结果 ===")
    println("线路开关状态 (β2): ", β2)
    println("节点有功负荷削减 (pls2): ", round.(pls2 .* baseMVA, digits=2), " MW")
    println("线路功率 (pij2): ", round.(pij2 .* baseMVA, digits=2), " MW")
    println("发电机出力 (pg2): ", round.(pg2 .* baseMVA, digits=2), " MW")
    println("微电网发电机出力 (pm2): ", round.(pm2 .* baseMVA, digits=2), " MW")
    println("VSC交流传入直流功率 (prec2): ", round.(prec2 .* baseMVA, digits=2), " MW")
    println("VSC直流传入交流功率 (pinv2): ", round.(pinv2 .* baseMVA, digits=2), " MW")
    println("节点有功负荷削减总和 (Pls2): ", round(Pls2 * baseMVA, digits=2), " MW")
    
    return Dict(
        :β2 => β2, 
        :pls2 => pls2, :Pls2 => Pls2, 
        :status => results[:status_name], :runtime => results[:runtime]
    )
end

"""
第三阶段：故障切除后重构阶段 (τ_TP - τ_RP)
"""
function solve_post_repair_reconfig(jpc, fault_lines, stage1_result, stage2_result)
    # 提取网络参数
    Cft_ac = jpc[:Cft_ac]  # AC lines
    Cft_dc = jpc[:Cft_dc]  # DC lines
    Cft = vcat(Cft_ac, Cft_dc)  # merge AC and DC lines
    Cft_vsc = jpc[:Cft_vsc]  # VSC lines
    Cg = jpc[:Cg]
    Cd = jpc[:Cd]
    Cdf = jpc[:Cdf]
    Cmg = jpc[:Cmg]
    Fd = jpc[:Fd]
    Pd = jpc[:Pd]
    Qd = jpc[:Qd]

    Pgmax = jpc[:Pgmax]
    Qgmax = jpc[:Qgmax]
    Pmgmax = jpc[:Pmgmax]
    Qmgmax = jpc[:Qmgmax]
    Pvscmax = jpc[:Pvscmax]
    Smax = jpc[:Smax]

    R = jpc[:R]
    X = jpc[:X]
    VMAX = jpc[:VMAX]
    VMIN = jpc[:VMIN]
    bigM = jpc[:bigM]
    η = jpc[:η]  # VSC efficiency

    # 第二阶段结果
    β2 = stage2_result[:β2]
    γ = stage1_result[:γ]

    nb_ac = jpc[:nb_ac]
    nb_dc = jpc[:nb_dc]
    nb = nb_ac + nb_dc
    ng = jpc[:ng]
    nl_ac = jpc[:nl_ac]
    nl_dc = jpc[:nl_dc]
    nl = nl_ac + nl_dc
    nl_vsc = jpc[:nl_vsc]
    nmg = jpc[:nmg]
    nd = jpc[:nd]
    
    # 创建变量索引
    idx = PostRepairReconfigIndices(nl, nl_vsc, nb, ng, nd, nmg)
    
    # 设置变量边界
    lb = fill(-Inf, idx.n_vars)
    ub = fill(Inf, idx.n_vars)
    # β3变量边界
    lb[get_range(idx.β3_start, nl + nl_vsc)] .= 0
    ub[get_range(idx.β3_start, nl + nl_vsc)] .= 1
    # Is3变量边界
    lb[idx.Is3_start] = 0
    ub[idx.Is3_start] = 1   
    # Fij3变量边界
    lb[get_range(idx.Fij3_start, nl)] .= -nb
    ub[get_range(idx.Fij3_start, nl)] .= nb
    # Fij_vsc3变量边界
    lb[get_range(idx.Fij_vsc3_start, nl_vsc)] .= -nb
    ub[get_range(idx.Fij_vsc3_start, nl_vsc)] .= nb
    # Fg3变量边界
    lb[get_range(idx.Fg3_start, ng)] .= 0
    ub[get_range(idx.Fg3_start, ng)] .= nb
    # Fvg3变量边界
    lb[get_range(idx.Fvg3_start, nb)] .= 0
    ub[get_range(idx.Fvg3_start, nb)] .= 1
    # pls3变量边界
    lb[get_range(idx.pls3_start, nb)] .= 0
    ub[get_range(idx.pls3_start, nb)] .= Cd' * Pd  # 负荷削减上限
    # qls3变量边界
    lb[get_range(idx.qls3_start, nb)] .= 0
    ub[get_range(idx.qls3_start, nb)] .= Cd' * Qd  # 无功负荷削减上限
    # pij3变量边界
    lb[get_range(idx.pij3_start, nl)] .= -Smax
    ub[get_range(idx.pij3_start, nl)] .= Smax
    # qij3变量边界
    lb[get_range(idx.qij3_start, nl)] .= -Smax
    ub[get_range(idx.qij3_start, nl)] .= Smax
    # pg3变量边界
    lb[get_range(idx.pg3_start, ng)] .= 0
    ub[get_range(idx.pg3_start, ng)] = Pgmax
    # qg3变量边界
    lb[get_range(idx.qg3_start, ng)] .= 0
    ub[get_range(idx.qg3_start, ng)] .= Qgmax
    # pm3变量边界
    lb[get_range(idx.pm3_start, nmg)] .= 0
    ub[get_range(idx.pm3_start, nmg)] .= Pmgmax
    # qm3变量边界
    lb[get_range(idx.qm3_start, nmg)] .= 0
    ub[get_range(idx.qm3_start, nmg)] .= Qmgmax
    # prec3变量边界
    lb[get_range(idx.prec3_start, nl_vsc)] .= 0
    ub[get_range(idx.prec3_start, nl_vsc)] .= Pvscmax
    # pinv3变量边界
    lb[get_range(idx.pinv3_start, nl_vsc)] .= 0
    ub[get_range(idx.pinv3_start, nl_vsc)] .= Pvscmax
    # v3变量边界
    lb[get_range(idx.v3_start, nb)] .= VMIN
    ub[get_range(idx.v3_start, nb)] .= VMAX
    
    # 目标函数：min sum(Fvg3) + sum(pls3) + pij3[28]
    c = zeros(idx.n_vars)
    c[get_range(idx.Fvg3_start, nb)] .= 1
    c[get_range(idx.pls3_start, nb)] .= 1
    c[idx.β3_start + 28 - 1] = 2  # 强制最小化第28条线路的功率
    
    # 计算约束数量
    n_eq = 3*nb + length(fault_lines) + 1  # include fault-specific eq and virtual flow + power balance
    n_ineq = 10*nl + 11*nl_vsc   # 各种不等式约束
    
    Aeq = spzeros(n_eq, idx.n_vars)
    beq = zeros(n_eq)
    A = spzeros(n_ineq, idx.n_vars)
    b = zeros(n_ineq)
    
    eq_row = 1
    ineq_row = 1
    
    ## Topology constraints
    # (0) γ >= β3  ————nl_vsc
    # @constraint(model, γ .>= β3);
    for i = 1:nl_vsc
        A[ineq_row, idx.β3_start+nl+i-1] = 1
        b[ineq_row] = γ[i]
        ineq_row += 1
    end
    
    # (1) 开关状态约束：β3[fault] = 0————1
    for i in fault_lines
        Aeq[eq_row, idx.β3_start + i - 1] = 1
        beq[eq_row] = 0
        eq_row += 1
    end

    # # (2) 开关状态约束：
    # # β3[non-fault] >= β2[non-fault]————nl + nl_vsc - 1
    # for i = 1:nl + nl_vsc
    #     if !(i in fault_lines)
    #         # β3 >= β2
    #         A[ineq_row, idx.β3_start + i - 1] = -1
    #         b[ineq_row] = -β2[i]
    #         ineq_row += 1
    #     end
    # end

    # (3) 虚拟潮流约束：sum(Fij3) = Fg3i - Fd3i + Fvg3i————nb
    Aeq[eq_row:eq_row+nb-1, get_range(idx.Fij3_start, nl)] = Cft'
    Aeq[eq_row:eq_row+nb-1, get_range(idx.Fij_vsc3_start, nl_vsc)] = Cft_vsc'
    Aeq[eq_row:eq_row+nb-1, get_range(idx.Fg3_start, ng)] = -Cg'
    for i in 1:nb
        Aeq[eq_row + i - 1, idx.Fvg3_start + i - 1] = -1
    end
    beq[eq_row:eq_row+nb-1] = -Cdf' * Fd
    eq_row += nb 

    # (4) -Smax.*β3 <= Fij3/Fij_vsc3 <= Smax.*β3 ————2*(nl+nl_vsc)
    # @constraint(model, Fij3 .<= nb.*β3);
    for i = 1:nl
        A[ineq_row, idx.Fij3_start+i-1] = 1
        A[ineq_row, idx.β3_start+i-1] = -nb
        ineq_row += 1
    end
    # 对于 VSC 配置线路的 Fij_vsc3 .<= nb.*β3
    for i = 1:nl_vsc
        A[ineq_row, idx.Fij_vsc3_start+i-1] = 1
        A[ineq_row, idx.β3_start+nl+i-1] = -nb
        ineq_row += 1
    end
    # @constraint(model, Fij3.>= -nb.*β3);
    for i = 1:nl
        A[ineq_row, idx.Fij3_start+i-1] = -1
        A[ineq_row, idx.β3_start+i-1] = -nb
        ineq_row += 1
    end
    # 对于 VSC 配置线路的 Fij_vsc3 .>= -nb.*β3
    for i = 1:nl_vsc
        A[ineq_row, idx.Fij_vsc3_start+i-1] = -1
        A[ineq_row, idx.β3_start+nl+i-1] = -nb
        ineq_row += 1
    end

    # (5) 孤岛存在验证约束：Is3ij = Fvg3i *Fvg3j *β3ij————4*(nl+nl_vsc)
    for i in 1:nl
        ij = findall(x -> x != 0, Cft[i, :])
        # Is3ij <= Fvg3i
        A[ineq_row, idx.Is3_start + i - 1] = 1
        A[ineq_row, idx.Fvg3_start + ij[1]] = -1  # Fvg3i
        ineq_row += 1
        # Is3ij <= Fvg3j
        A[ineq_row, idx.Is3_start + i - 1] = 1
        A[ineq_row, idx.Fvg3_start + ij[2]] = -1  # Fvg3j
        ineq_row += 1
        # Is3ij <= β3ij
        A[ineq_row, idx.Is3_start + i - 1] = 1
        A[ineq_row, idx.β3_start + i - 1] = -1  # β3ij
        ineq_row += 1
        # Is3ij >= Fvg3i + Fvg3j + β3ij - 2
        A[ineq_row, idx.Is3_start + i - 1] = -1
        A[ineq_row, idx.Fvg3_start + ij[1]] = 1  # Fvg3i
        A[ineq_row, idx.Fvg3_start + ij[2]] = 1  # Fvg3j
        A[ineq_row, idx.β3_start + i - 1] = 1  # β3ij
        b[ineq_row] = 2  # 右侧常数
        ineq_row += 1
    end
    for i in nl+1:nl+nl_vsc
        ij = findall(x -> x != 0, Cft_vsc[i-nl, :])
        # Is3ij <= Fvg3i
        A[ineq_row, idx.Is3_start + i - 1] = 1
        A[ineq_row, idx.Fvg3_start + ij[1]] = -1  # Fvg3i
        ineq_row += 1
        # Is3ij <= Fvg3j
        A[ineq_row, idx.Is3_start + i - 1] = 1
        A[ineq_row, idx.Fvg3_start + ij[2]] = -1  # Fvg3j
        ineq_row += 1
        # Is3ij <= β3ij
        A[ineq_row, idx.Is3_start + i - 1] = 1
        A[ineq_row, idx.β3_start + i - nl] = -1  # β3ij
        ineq_row += 1
        # Is3ij >= Fvg3i + Fvg3j + β3ij - 2
        A[ineq_row, idx.Is3_start + i - 1] = -1
        A[ineq_row, idx.Fvg3_start + ij[1]] = 1  # Fvg3i
        A[ineq_row, idx.Fvg3_start + ij[2]] = 1  # Fvg3j
        A[ineq_row, idx.β3_start + i - nl] = 1  # β3ij
        b[ineq_row] = 2  # 右侧常数
        ineq_row += 1
    end

    # (6) 辐射状拓扑约束：sum(β3) = nb - ng - sum(Fvg3) + sum(Is3)————1
    Aeq[eq_row, get_range(idx.β3_start, nl+nl_vsc)] .= 1
    Aeq[eq_row, get_range(idx.Fvg3_start, nb)] .= 1  # Fvg3i
    Aeq[eq_row, get_range(idx.Is3_start, nl+nl_vsc)] .= -1  # Is_3
    beq[eq_row] = nb - ng
    eq_row += 1
    
    # (7) power balance  ————2*nb
    for i = 1:nb_ac
        # 交流节点
        Aeq[eq_row, get_range(idx.pij3_start, nl)] .= Cft'[i, :]
        Aeq[eq_row, get_range(idx.pg3_start, ng)] .= -Cg'[i, :]
        Aeq[eq_row, get_range(idx.pm3_start, nmg)] .= -Cmg'[i, :]
        Aeq[eq_row, idx.pls3_start + i - 1] = -1
        # 对于交流节点， prec不用乘损耗系数，pinv需要乘损耗系数
        Aeq[eq_row, get_range(idx.prec3_start, nl_vsc)] .= Cft_vsc'[i, :]
        Aeq[eq_row, get_range(idx.pinv3_start, nl_vsc)] .= -Cft_vsc'[i, :] .* η
        beq[eq_row] = -dot(Cd'[i, :], Pd)
        eq_row += 1
    end
    for i = nb_ac+1:nb
        # 直流节点
        Aeq[eq_row, get_range(idx.pij3_start, nl)] .= Cft'[i, :]
        Aeq[eq_row, get_range(idx.pg3_start, ng)] .= -Cg'[i, :]
        Aeq[eq_row, get_range(idx.pm3_start, nmg)] .= -Cmg'[i, :]
        Aeq[eq_row, idx.pls3_start + i - 1] = -1
        # 对于直流节点， prec需要乘损耗系数，pinv不用乘损耗系数
        Aeq[eq_row, get_range(idx.prec3_start, nl_vsc)] .= Cft_vsc'[i, :] .* η
        Aeq[eq_row, get_range(idx.pinv3_start, nl_vsc)] .= -Cft_vsc'[i, :]
        beq[eq_row] = -dot(Cd'[i, :], Pd)
        eq_row += 1
    end
    # @constraint(model, Cft'*qij3 .== Cg'*qg3 - Cd'*(Qd - qls3) + Cmg'*qm3);
    Aeq[eq_row:eq_row+nb-1, get_range(idx.qij3_start, nl)] = Cft'
    Aeq[eq_row:eq_row+nb-1, get_range(idx.qg3_start, ng)] = -Cg'
    Aeq[eq_row:eq_row+nb-1, get_range(idx.qm3_start, nmg)] = -Cmg'
    for i = 1:nb
        Aeq[eq_row + i - 1, idx.qls3_start + i - 1] = -1
    end
    beq[eq_row:eq_row+nb-1] = -Cd'*Qd
    eq_row += nb

    #（8）-Smax.*β3 <= pij3/qij3 .<= Smax.*β3  ————4*nl
    # AC
    for i = 1:nl_ac
        # pij3 .<= Smax.*βij3
        A[ineq_row, idx.pij3_start+i-1] = 1
        A[ineq_row, idx.β3_start+i-1] = -Smax[i]
        ineq_row += 1
        # pij3 .>= -Smax.*βij3
        A[ineq_row, idx.pij3_start+i-1] = -1
        A[ineq_row, idx.β3_start+i-1] = -Smax[i]
        ineq_row += 1
        # qij3 .<= Smax.*βij3
        A[ineq_row, idx.qij3_start+i-1] = 1
        A[ineq_row, idx.β3_start+i-1] = -Smax[i]
        ineq_row += 1
        # qij3 .>= -Smax.*βij3
        A[ineq_row, idx.qij3_start+i-1] = -1
        A[ineq_row, idx.β3_start+i-1] = -Smax[i]
        ineq_row += 1
    end
    # DC
    for i = (nl_ac+1):nl
        # pij3 .<= bigM.*βij3
        A[ineq_row, idx.pij3_start+i-1] = 1
        A[ineq_row, idx.β3_start+i-1] = -bigM
        ineq_row += 1
        # pij3 .>= -bigM.*βij3
        A[ineq_row, idx.pij3_start+i-1] = -1
        A[ineq_row, idx.β3_start+i-1] = -bigM
        ineq_row += 1
        # qij3 .<= bigM.*βij3
        A[ineq_row, idx.qij3_start+i-1] = 1
        A[ineq_row, idx.β3_start+i-1] = -bigM
        ineq_row += 1
        # qij3 .>= -bigM.*βij3
        A[ineq_row, idx.qij3_start+i-1] = -1
        A[ineq_row, idx.β3_start+i-1] = -bigM
        ineq_row += 1
    end

    # (9) VSC power constraints ————4*nl_vsc
    for i = 1:nl_vsc
        # 0 <= prec3 .<= Pvscmax.*βij3
        A[ineq_row, idx.prec3_start+i-1] = 1
        A[ineq_row, idx.β3_start+nl+i-1] = -Pvscmax[i]
        ineq_row += 1
        A[ineq_row, idx.prec3_start+i-1] = -1
        b[ineq_row] = 0
        ineq_row += 1
        # 0 <= pinv3 .<= Pvscmax.*βij3
        A[ineq_row, idx.pinv3_start+i-1] = 1
        A[ineq_row, idx.β3_start+nl+i-1] = -Pvscmax[i]
        ineq_row += 1
        A[ineq_row, idx.pinv3_start+i-1] = -1
        b[ineq_row] = 0
        ineq_row += 1
    end
    
    # 设置变量类型
    vtypes = fill('C', idx.n_vars)
    vtypes[get_range(idx.β3_start, nl+nl_vsc)] .= 'B'
    vtypes[get_range(idx.Is3_start, nl+nl_vsc)] .= 'B'
    vtypes[get_range(idx.Fvg3_start, nb)] .= 'B'
    
    # 求解MILP
    results = solve_milp_sparse(c, A, b, Aeq, beq, lb, ub, vtypes)
    
    # 解码解
    x = results[:x]
    β3 = get_variable_values(x, idx, :β3, nl+nl_vsc)
    Is3 = get_variable_values(x, idx, :Is3, nl+nl_vsc)
    pls3 = get_variable_values(x, idx, :pls3, nb)
    pij3 = get_variable_values(x, idx, :pij3, nl)
    pg3 = get_variable_values(x, idx, :pg3, ng)
    pm3 = get_variable_values(x, idx, :pm3, nmg)
    prec3 = get_variable_values(x, idx, :prec3, nl_vsc)
    pinv3 = get_variable_values(x, idx, :pinv3, nl_vsc)

    # 计算负荷削减
    Pls3 = sum(pls3)
    
    # 获取baseMVA用于转换为实际MW值
    baseMVA = jpc["baseMVA"]
    
    println("=== 第三阶段：故障修复后重构结果 ===")
    println("线路开关状态 (β3): ", β3)
    println("孤岛指示 (Is3): ", Is3)
    println("节点有功负荷削减 (pls3): ", round.(pls3 .* baseMVA, digits=2), " MW")
    println("线路功率 (pij3): ", round.(pij3 .* baseMVA, digits=2), " MW")
    println("发电机出力 (pg3): ", round.(pg3 .* baseMVA, digits=2), " MW")
    println("微电网发电机出力 (pm3): ", round.(pm3 .* baseMVA, digits=2), " MW")
    println("VSC交流传入直流功率 (prec3): ", round.(prec3 .* baseMVA, digits=2), " MW")
    println("VSC直流传入交流功率 (pinv3): ", round.(pinv3 .* baseMVA, digits=2), " MW")
    println("节点有功负荷削减 (Pls3): ", round(Pls3 * baseMVA, digits=2), " MW")
    
    return Dict(
        :β3 => β3, :Is3 => Is3, 
        :pls3 => pls3, :Pls3 => Pls3,
        :objective => results[:objval],
        :status => results[:status_name], :runtime => results[:runtime]
    )
end

"""
改进版第一阶段：故障隔离阶段 (τ_FI - τ_TP)
"""

function solve_fault_isolation_improve(jpc, fault_lines)   
    # 提取网络参数
    Cft_ac = jpc[:Cft_ac]  # AC lines
    Cft_dc = jpc[:Cft_dc]  # DC lines
    Cft = vcat(Cft_ac, Cft_dc)  # merge AC and DC lines
    Cft_vsc = jpc[:Cft_vsc]  # VSC lines
    Cg = jpc[:Cg]
    Cd = jpc[:Cd]
    Cmg = jpc[:Cmg]
    Pd = jpc[:Pd]
    Qd = jpc[:Qd]

    Pgmax = jpc[:Pgmax]
    Qgmax = jpc[:Qgmax]
    Pmgmax = jpc[:Pmgmax]
    Qmgmax = jpc[:Qmgmax]
    Pvscmax = jpc[:Pvscmax]
    Smax = jpc[:Smax]

    R = jpc[:R]
    X = jpc[:X]
    VMAX = jpc[:VMAX]
    VMIN = jpc[:VMIN]
    bigM = jpc[:bigM]
    η = jpc[:η]  # VSC efficiency
    c_load = jpc[:c_load]
    c_vsc = jpc[:c_vsc]

    α = jpc[:α_pre]
    r = jpc[:r]
    
    nb_ac = jpc[:nb_ac]
    nb_dc = jpc[:nb_dc]
    nb = nb_ac + nb_dc
    ng = jpc[:ng]
    nl_ac = jpc[:nl_ac]
    nl_dc = jpc[:nl_dc]
    nl = nl_ac + nl_dc
    nl_vsc = jpc[:nl_vsc]
    nmg = jpc[:nmg]
    nd = jpc[:nd]

    idx = FaultIsolationIndices(nl, nl_vsc, nb, ng, nd, nmg)
    
    # Set up bounds
    lb = fill(-Inf, idx.n_vars)
    ub = fill(Inf, idx.n_vars)
    # Variables for common condition
    # Bounds for z
    lb[get_range(idx.z_start, nb)] .= 0
    ub[get_range(idx.z_start, nb)] .= 1
    # Bounds for β
    lb[get_range(idx.β_start, nl + nl_vsc)] .= 0
    ub[get_range(idx.β_start, nl + nl_vsc)] .= 1
    # Bounds for γ
    lb[get_range(idx.γ_start, nl_vsc)] .= 0
    ub[get_range(idx.γ_start, nl_vsc)] .= 1
    # Bounds for pls
    lb[get_range(idx.pls_start, nb)] .= 0
    ub[get_range(idx.pls_start, nb)] .= Cd' * Pd
    # Bounds for qls
    lb[get_range(idx.qls_start, nb)] .= 0
    ub[get_range(idx.qls_start, nb)] .= Cd' * Qd
    # Bounds for pij
    lb[get_range(idx.pij_start, nl)] .= -Smax
    ub[get_range(idx.pij_start, nl)] .= Smax
    # Bounds for qij
    lb[get_range(idx.qij_start, nl)] .= -Smax
    ub[get_range(idx.qij_start, nl)] .= Smax
    # Bounds for pg
    lb[get_range(idx.pg_start, ng)] .= 0
    ub[get_range(idx.pg_start, ng)] .= Pgmax
    # Bounds for qg
    lb[get_range(idx.qg_start, ng)] .= 0
    ub[get_range(idx.qg_start, ng)] .= Qgmax
    # Bounds for pmg
    lb[get_range(idx.pm_start, nmg)] .= 0
    ub[get_range(idx.pm_start, nmg)] .= Pmgmax
    # Bounds for qmg
    lb[get_range(idx.qm_start, nmg)] .= 0
    ub[get_range(idx.qm_start, nmg)] .= Qmgmax
    # Bounds for prec
    lb[get_range(idx.prec_start, nl_vsc)] .= 0
    ub[get_range(idx.prec_start, nl_vsc)] .= Pvscmax
    # Bounds for pinv
    lb[get_range(idx.pinv_start, nl_vsc)] .= 0
    ub[get_range(idx.pinv_start, nl_vsc)] .= Pvscmax
    # Bounds for v
    lb[get_range(idx.v_start, nb)] .= VMIN
    ub[get_range(idx.v_start, nb)] .= VMAX

    # Objective function
    # min - sum(β) + sum(z) - sum(γ) + c*sum(pls)
    c = zeros(idx.n_vars)
    c[get_range(idx.z_start, nb)] .= 1  # 负荷削减项权重
    c[get_range(idx.β_start, nl + nl_vsc)] .= -1  # 开关状态项权重
    c[get_range(idx.γ_start, nl_vsc)] .= -1  # VSC配置状态项权重
    c[get_range(idx.pls_start, nb)] .= c_load  # 负荷削减项权重

    # Calculate number of constraints
    n_eq = 2*nb     # Power balance constraints

    if fault_lines <= 11
        # 故障发生在上方交流线路
        n_ineq = 8*nl + 9*nl_vsc + 2*nb + 4
    elseif fault_lines <= 23
        # 故障发生在下方交流线路
        n_ineq = 8*nl + 9*nl_vsc + 2*nb + 4
    elseif fault_lines <= nl_ac
        # 故障发生在断开的交流线路
        n_ineq = 8*nl + 9*nl_vsc + 2*nb + 3
    elseif fault_lines <= nl
        # 故障发生在 DC 线路
        n_ineq = 8*nl + 9*nl_vsc + 2*nb + 3
    else
        # 故障发生在 VSC 线路
        n_ineq = 8*nl + 9*nl_vsc + 2*nb + 3
    end

    Aeq = spzeros(n_eq, idx.n_vars)
    beq = zeros(n_eq)
    A = spzeros(n_ineq, idx.n_vars)
    b = zeros(n_ineq)
    eq_row = 1
    ineq_row = 1

    ## Constraints for common condition
    # 直接得到正常情况下的拓扑α
    # (0) γ >= α  ————nl_vsc
    # @constraint(model, γ .>= α);
    for i = 1:nl_vsc
        A[ineq_row, idx.γ_start+i-1] = -1
        b[ineq_row] = -α[i+nl]
        ineq_row += 1
    end

    # (0) sum(γ) >= 2  ————1
    A[ineq_row, get_range(idx.γ_start, nl_vsc)] .= -1  # 对所有 γ[i] 加权系数为 1
    b[ineq_row] = -2  # 右侧常数为 2
    ineq_row += 1

    ## Topology constraints
    for fault_line in fault_lines
        if fault_line <= 11
            # (0) 变电站隔离约束: βij[1] <= 0————1
            A[ineq_row, idx.β_start] = 1  # +βij
            ineq_row += 1
            connected_nodes = findall(x -> x != 0, Cft[fault_line, :])
        elseif fault_line <= 23
            # (0) 变电站隔离约束: βij[12] <= 0————1
            A[ineq_row, idx.β_start + 12 - 1] = 1  # +βij
            ineq_row += 1
            connected_nodes = findall(x -> x != 0, Cft[fault_line, :])
        elseif fault_line <= nl_ac
        elseif fault_line <= nl
            connected_nodes = findall(x -> x != 0, Cft[fault_line, :])
        else
            connected_nodes = findall(x -> x != 0, Cft_vsc[fault_line-nl, :])
        end
        # (1) 故障线路传播约束: βij[fault] <= zi、zj————2
        for j in connected_nodes
            A[ineq_row, idx.z_start + j - 1] = -1  # zj
            A[ineq_row, idx.β_start + fault_line - 1] = 1  # +βij
            ineq_row += 1
        end

        # (2) 故障节点传播约束: zj - (1-βij) <= zi <= zj + (1-βij)————2*(nl+nl_vsc)
        for j in 1:nl
            row_j = Cft[j, :]  # 第j条线路的连接关系
            # zi <= zj + (1 - βij)  <=>  (zi - zj) + βij <= 1
            A[ineq_row, get_range(idx.z_start, nb)] = row_j   # zi - zj
            A[ineq_row, idx.β_start + j - 1] = 1              # +βij
            b[ineq_row] = 1
            ineq_row += 1
            # zj - (1 - βij) <= zi  <=>  (zj - zi) + βij <= 1
            A[ineq_row, get_range(idx.z_start, nb)] = -row_j  # -zi + zj
            A[ineq_row, idx.β_start + j - 1] = 1              # +βij
            b[ineq_row] = 1
            ineq_row += 1
        end
        for j in 1:nl_vsc
            row_j = Cft_vsc[j, :]  # 第j条VSC线路的连接关系
            # zi <= zj + (1 - βij)  <=>  (zi - zj) + βij <= 1
            A[ineq_row, get_range(idx.z_start, nb)] = row_j   # zi - zj
            A[ineq_row, idx.β_start + nl + j - 1] = 1         # +βij
            b[ineq_row] = 1
            ineq_row += 1
            # zj - (1 - βij) <= zi  <=>  (zj - zi) + βij <= 1
            A[ineq_row, get_range(idx.z_start, nb)] = -row_j  # -zi + zj
            A[ineq_row, idx.β_start + nl + j - 1] = 1         # +βij
            b[ineq_row] = 1
            ineq_row += 1
        end

        # (3) 开关隔离约束：αij*(1-rij) <= βij <= αij————2(nl + nl_vsc)
        for j in 1:nl+nl_vsc
            # αij*(1-rij) <= βij
            A[ineq_row, idx.β_start + j - 1] = -1
            b[ineq_row] = -α[j] * (1 - r[j])  # 左侧常数
            ineq_row += 1
            # βij <= αij
            A[ineq_row, idx.β_start + j - 1] = 1
            b[ineq_row] = α[j]  # 右侧常数
            ineq_row += 1
        end
        
        # (4) power balance  ————2*nb
        for i = 1:nb_ac
            # 交流节点
            Aeq[eq_row, get_range(idx.pij_start, nl)] .= Cft'[i, :]
            Aeq[eq_row, get_range(idx.pg_start, ng)] .= -Cg'[i, :]
            Aeq[eq_row, get_range(idx.pm_start, nmg)] .= -Cmg'[i, :]
            Aeq[eq_row, idx.pls_start + i - 1] = -1
            # 对于交流节点， prec不用乘损耗系数，pinv需要乘损耗系数
            Aeq[eq_row, get_range(idx.prec_start, nl_vsc)] .= Cft_vsc'[i, :]
            Aeq[eq_row, get_range(idx.pinv_start, nl_vsc)] .= -Cft_vsc'[i, :] .* η
            beq[eq_row] = -dot(Cd'[i, :], Pd)
            eq_row += 1
        end
        for i = nb_ac+1:nb
            # 直流节点
            Aeq[eq_row, get_range(idx.pij_start, nl)] .= Cft'[i, :]
            Aeq[eq_row, get_range(idx.pg_start, ng)] .= -Cg'[i, :]
            Aeq[eq_row, get_range(idx.pm_start, nmg)] .= -Cmg'[i, :]
            Aeq[eq_row, idx.pls_start + i - 1] = -1
            # 对于直流节点， prec需要乘损耗系数，pinv不用乘损耗系数
            Aeq[eq_row, get_range(idx.prec_start, nl_vsc)] .= Cft_vsc'[i, :] .* η
            Aeq[eq_row, get_range(idx.pinv_start, nl_vsc)] .= -Cft_vsc'[i, :]
            beq[eq_row] = -dot(Cd'[i, :], Pd)
            eq_row += 1
        end
        # @constraint(model, Cft'*qij .== Cg'*qg - Cd'*(Qd - qls) + Cmg'*qm);
        Aeq[eq_row:eq_row+nb-1, get_range(idx.qij_start, nl)] = Cft'
        Aeq[eq_row:eq_row+nb-1, get_range(idx.qg_start, ng)] = -Cg'
        Aeq[eq_row:eq_row+nb-1, get_range(idx.qm_start, nmg)] = -Cmg'
        for i = 1:nb
            Aeq[eq_row + i - 1, idx.qls_start + i - 1] = -1  # qls
        end
        beq[eq_row:eq_row+nb-1] = -Cd'*Qd
        eq_row += nb
    end

    #（5）-Smax.*βij <= pij/qij .<= Smax.*βij  ————4*nl
    # AC
    for i = 1:nl_ac
        # pij .<= Smax.*βij
        A[ineq_row, idx.pij_start+i-1] = 1
        A[ineq_row, idx.β_start+i-1] = -Smax[i]
        ineq_row += 1
        # pij .>= -Smax.*βij
        A[ineq_row, idx.pij_start+i-1] = -1
        A[ineq_row, idx.β_start+i-1] = -Smax[i]
        ineq_row += 1
        # qij .<= Smax.*βij
        A[ineq_row, idx.qij_start+i-1] = 1
        A[ineq_row, idx.β_start+i-1] = -Smax[i]
        ineq_row += 1
        # qij .>= -Smax.*βij
        A[ineq_row, idx.qij_start+i-1] = -1
        A[ineq_row, idx.β_start+i-1] = -Smax[i]
        ineq_row += 1
    end
    # DC
    for i = (nl_ac+1):nl
        # pij .<= bigM.*βij
        A[ineq_row, idx.pij_start+i-1] = 1
        A[ineq_row, idx.β_start+i-1] = -bigM
        ineq_row += 1
        # pij .>= -bigM.*βij
        A[ineq_row, idx.pij_start+i-1] = -1
        A[ineq_row, idx.β_start+i-1] = -bigM
        ineq_row += 1
        # qij .<= bigM.*βij
        A[ineq_row, idx.qij_start+i-1] = 1
        A[ineq_row, idx.β_start+i-1] = -bigM
        ineq_row += 1
        # qij .>= -bigM.*βij
        A[ineq_row, idx.qij_start+i-1] = -1
        A[ineq_row, idx.β_start+i-1] = -bigM
        ineq_row += 1
    end

    # (6) VSC power constraints ————4*nl_vsc
    for i = 1:nl_vsc
        # 0 <= pij_rec .<= Pvscmax.*βij
        A[ineq_row, idx.prec_start+i-1] = 1
        A[ineq_row, idx.β_start+nl+i-1] = -Pvscmax[i]
        ineq_row += 1
        A[ineq_row, idx.prec_start+i-1] = -1
        b[ineq_row] = 0
        ineq_row += 1
        # 0 <= pij_inv .<= Pvscmax.*βij
        A[ineq_row, idx.pinv_start+i-1] = 1
        A[ineq_row, idx.β_start+nl+i-1] = -Pvscmax[i]
        ineq_row += 1
        A[ineq_row, idx.pinv_start+i-1] = -1
        b[ineq_row] = 0
        ineq_row += 1
    end

    # (7) 故障节点负荷削减约束：-M*(1-z) <= pls - pd ≤ M*(1-z)————2*nb
    for i = 1:nb
        # pls - pd <= M*(1-z)
        A[ineq_row, idx.pls_start+i-1] = 1
        A[ineq_row, idx.z_start+i-1] = bigM
        b[ineq_row] = bigM + dot(Cd'[i, :], Pd)
        ineq_row += 1
        # pls - pd >= -M*(1-z)
        A[ineq_row, idx.pls_start+i-1] = -1
        A[ineq_row, idx.z_start+i-1] = bigM
        b[ineq_row] = bigM - dot(Cd'[i, :], Pd)
        ineq_row += 1
    end

    # 设置变量类型
    vtypes = fill('C', idx.n_vars)
    vtypes[get_range(idx.z_start, nb)] .= 'B'
    vtypes[get_range(idx.β_start, nl+nl_vsc)] .= 'B'
    vtypes[get_range(idx.γ_start, nl_vsc)] .= 'B'
    
    # 求解MILP
    results = solve_milp_sparse(c, A, b, Aeq, beq, lb, ub, vtypes)
    
    # 解码解
    x = results[:x]
    z = get_variable_values(x, idx, :z, nb)
    β = get_variable_values(x, idx, :β, nl + nl_vsc)
    γ = get_variable_values(x, idx, :γ, nl_vsc)
    pls = get_variable_values(x, idx, :pls, nb)
    pij = get_variable_values(x, idx, :pij, nl)
    pg = get_variable_values(x, idx, :pg, ng)
    pm = get_variable_values(x, idx, :pm, nmg)
    prec = get_variable_values(x, idx, :prec, nl_vsc)
    pinv = get_variable_values(x, idx, :pinv, nl_vsc)
    
    # 计算负荷削减
    Pls = sum(pls)
    
    # 获取baseMVA用于转换为实际MW值
    baseMVA = jpc["baseMVA"]
    
    println("=== 第一阶段：故障隔离结果 ===")
    println("节点是否处于故障区域", z)
    println("线路开关状态 (β): ", β)
    println("VSC配置状态 (γ): ", γ)
    println("节点有功负荷削减 (pls): ", round.(pls .* baseMVA, digits=2), " MW")
    println("线路有功功率 (pij): ", round.(pij .* baseMVA, digits=2), " MW")
    println("发电机出力 (pg): ", round.(pg .* baseMVA, digits=2), " MW")
    println("微电网发电机出力 (pm): ", round.(pm .* baseMVA, digits=2), " MW")
    println("VSC交流传入直流功率 (prec): ", round.(prec .* baseMVA, digits=2), " MW")
    println("VSC直流传入交流功率 (pinv): ", round.(pinv .* baseMVA, digits=2), " MW")
    println("节点有功负荷削减总和 (Pls): ", round(Pls * baseMVA, digits=2), " MW")

    return Dict(
        :z => z, :β1 => β, :γ => γ, 
        :pls1 => pls, :Pls1 => Pls,
        :objective => results[:objval], :status => results[:status_name], :runtime => results[:runtime]
    )
end



"""
完整的三阶段故障恢复求解器
"""
function solve_three_stage_fault_recovery(jpc, fault_lines)
    
    println("=== 开始三阶段故障恢复求解 ===")
    println("故障线路: ", fault_lines)
    
    # 第一阶段：故障隔离
    println("\n--- 第一阶段：故障隔离 ---")
    stage1_result = solve_fault_isolation(jpc, fault_lines)
    
    # 第二阶段：故障后重构
    println("\n--- 第二阶段：故障后重构 ---")
    stage2_result = solve_post_fault_reconfig(jpc, stage1_result)
    
    # 第三阶段：故障修复后重构
    println("\n--- 第三阶段：故障修复后重构 ---")
    stage3_result = solve_post_repair_reconfig(jpc, fault_lines, stage1_result, stage2_result)

    Pls = stage1_result[:Pls1] + stage2_result[:Pls2] + stage3_result[:Pls3]
    
    # 汇总结果
    total_results = Dict(
        :stage1 => stage1_result,
        :stage2 => stage2_result,
        :stage3 => stage3_result,
        :fault_lines => fault_lines,
        :total_load_shedding => Pls
    )
    return total_results
end