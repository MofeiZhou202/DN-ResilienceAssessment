# ========== JPC_tp 矩阵列索引常量 ==========

# busAC/busDC 矩阵列索引（15列）
const BUS_ID = 1          # 节点ID
const BUS_TYPE = 2        # 节点类型
const PD = 3              # 有功负荷（MW）
const QD = 4              # 无功负荷（MVAR）
const GS = 5              # 有功发电（MW）
const BS = 6              # 无功发电（MVAR）
const AREA_ID = 7         # 区域编号
const VM = 8              # 电压幅值（p.u.）
const VA = 9              # 电压相角（度）
const BASE_KV = 10        # 基准电压（kV）
const ZONE_ID = 11        # 区域编号
const V_MAX = 12          # 最大电压幅值（p.u.）
const V_MIN = 13          # 最小电压幅值（p.u.）

# branchAC/branchDC 矩阵列索引（14列）
const F_BUS = 1           # 首端母线ID
const T_BUS = 2           # 末端母线ID
const BR_R = 3            # 电阻（p.u.）
const BR_X = 4            # 电抗（p.u.）
const BR_B = 5            # 电纳（p.u.）
const RATE_A = 6          # 长期额定容量（MVA/MW）
const RATE_B = 7          # 短期额定容量（MVA/MW）
const RATE_C = 8          # 应急额定容量（MVA/MW）
const TAP = 9             # 变比
const SHIFT = 10          # 相移角（度）
const BR_STATUS = 11      # 状态（1=投入, 0=退出）
const ANGMIN = 12         # 最小相角（度）
const ANGMAX = 13         # 最大相角（度）

# genAC/genDC 矩阵列索引
const GEN_BUS = 1         # 连接母线ID
const PG = 2              # 有功出力（MW）
const QG = 3              # 无功出力（MVAR）
const QMAX = 4            # 最大无功出力（MVAR）
const QMIN = 5            # 最小无功出力（MVAR）
const GEN_STATUS = 6      # 状态（1=投入, 0=退出）

# loadAC/loadDC 矩阵列索引
const LOAD_BUS = 1        # 连接母线ID
const LOAD_PD = 2         # 有功负荷（MW）
const LOAD_QD = 3         # 无功负荷（MVAR）
const LOAD_STATUS = 4     # 状态（1=投入, 0=退出）

# converter 矩阵列索引
const VSC_FROM_BUS = 1    # 首端母线ID
const VSC_TO_BUS = 2      # 末端母线ID
const VSC_R = 3           # 电阻（p.u.）
const VSC_X = 4           # 电抗（p.u.）
const VSC_P_DC = 5        # 直流功率（MW）
const VSC_Q_AC = 6        # 交流无功（MVAR）

# converter 矩阵列索引（从 idx.jl）
const CONV_ACBUS = 1      # 交流母线ID
const CONV_DCBUS = 2      # 直流母线ID
const CONV_INSERVICE = 3  # 投运状态
const CONV_P_AC = 4       # 交流有功（MW）
const CONV_Q_AC = 5       # 交流无功（MVAR）
const CONV_P_DC = 6       # 直流功率（MW）
const CONV_EFF = 7        # 效率
const CONV_MODE = 8       # 控制模式
const CONV_DROOP_KP = 9   # Droop系数

function JuliaPowerCase2Jpc_tp(case::JuliaPowerCase)
    # 1. 创建JPC_tp对象
    jpc_tp = JPC_tp()
    
    # 2. 设置基本参数
    jpc_tp.baseMVA = case.baseMVA
    
    # 3. 设置节点数据
    JPC_tp_buses_process(case, jpc_tp)

    # 4. 设置直流节点数据
    JPC_tp_dcbuses_process(case, jpc_tp)
    
    # 5. 设置线路数据
    JPC_tp_branches_process(case, jpc_tp)
    
    # 6. 设置直流线路数据
    JPC_tp_dcbranches_process(case, jpc_tp)

    # 7. 设置发电机数据
    JPC_tp_gens_process(case, jpc_tp)

    # 8. 设置直流发电机数据
    # JPC_tp_battery_gens_process(case, jpc_tp)

    # 9. 设置soc电池数据
    # JPC_tp_battery_soc_process(case, jpc_tp)
    
    # 10. 设置负荷数据
    JPC_tp_loads_process(case, jpc_tp)

    # 11. 设置直流负荷数据
    JPC_tp_dcloads_process(case, jpc_tp)

    # 12. 设置PV阵列数据 
    JPC_tp_pv_process(case, jpc_tp)

    # 13. 设置换流器数据
    JPC_tp_inverters_process(case, jpc_tp)

    # 14. 设置交流光伏系统数据
    JPC_tp_ac_pv_system_process(case, jpc_tp)

    # 15. 设置稀疏矩阵
    C_sparse = sparse_matrix(jpc_tp)
    
    # 16. 设置拓扑信息
    nl_ac = size(jpc_tp.branchAC, 1)
    nl_dc = size(jpc_tp.branchDC, 1)
    nl_vsc = size(jpc_tp.converter, 1)
    
    # 提取线路状态（假设 branchAC 的 BR_STATUS 列是第 11 列）
    α = Float64[]
    if nl_ac > 0
        α = Float64.(jpc_tp.branchAC[:, 11])  # BR_STATUS 列
    end
    if nl_dc > 0
        append!(α, Float64.(jpc_tp.branchDC[:, 11]))
    end
    if nl_vsc > 0
        # VSC 状态，从 converter 矩阵的 CONV_INSERVICE 列（第3列）获取
        append!(α, Float64.(jpc_tp.converter[:, CONV_INSERVICE]))
    end
    
    jpc_tp["α_pre"] = α
    jpc_tp["nl_ac"] = nl_ac
    jpc_tp["nl_dc"] = nl_dc
    jpc_tp["nl_vsc"] = nl_vsc

    return jpc_tp, C_sparse

end

function JPC_tp_buses_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
    # 获取节点数据并深拷贝防止误操作
    buses = deepcopy(case.busesAC)
    
    # 创建一个空矩阵，行数为节点数，列数为13
    num_buses = length(buses)
    bus_matrix = zeros(num_buses, 15)
    
    for (i, bus) in enumerate(buses)
        # 设置电压初始值（根据序号）
        vm = 1.0
        va = 0.0
        
        # 填充矩阵的每一行
        bus_matrix[i, :] = [
            bus.index,      # 节点ID
            1.0,             # 节点类型(全部设为PQ节点)
            0.0,             # PD (MW) 有功负荷（MW）
            0.0,             # QD (MVAR) 无功负荷（MVAR）
            0.0,             # GS (MW) 有功发电（MW）
            0.0,             # BS (MVAR) 无功发电（MVAR）
            bus.area_id,     # 区域编号
            vm,              # 节点电压幅值（p.u.）
            va,              # 节点电压相角（度）
            bus.vn_kv,       # 节点电压基准（kV）
            bus.zone_id,     # 区域编号
            bus.v_max_pu,   # 最大电压幅值（p.u.）
            bus.v_min_pu,   # 最小电压幅值（p.u.）
            1.0,             # 碳排放区域编号（默认1）
            1.0,             # 碳排放区域编号（默认1）
        ]
    end
    
    # 所有序号的结果都存储到busAC字段
    jpc_tp.busAC = bus_matrix
    
    return jpc_tp
end

function JPC_tp_dcbuses_process(case, jpc_tp)
    # 处理直流节点数据，转换为JPC_tp格式
    dcbuses = deepcopy(case.busesDC)
    
    # 创建一个空矩阵，行数为节点数，列数为13
    num_dcbuses = length(dcbuses)
    num_acbuses = size(case.busesAC, 1)
    dcbus_matrix = zeros(num_dcbuses, 15)
    
    for (i, dcbus) in enumerate(dcbuses)
        # 设置电压初始值（根据序号）
        vm = 1.0
        va = 0.0
        
        # 填充矩阵的每一行
        dcbus_matrix[i, :] = [
            dcbus.index + num_acbuses,      # 节点ID
            1.0,               # 节点类型(全部设为PQ节点)
            0.0,               # PD (MW) 有功负荷（MW）
            0.0,               # QD (MVAR) 无功负荷（MVAR）
            0.0,               # GS (MW) 有功发电（MW）
            0.0,               # BS (MVAR) 无功发电（MVAR）
            dcbus.area_id,     # 区域编号
            vm,                # 节点电压幅值（p.u.）
            va,                # 节点电压相角（度）
            dcbus.vn_kv,       # 节点电压基准（kV）
            dcbus.zone_id,     # 区域编号
            dcbus.v_max_pu,   # 最大电压幅值（p.u.）
            dcbus.v_min_pu,   # 最小电压幅值（p.u.）
            1.0,               # 碳排放区域编号（默认1）
            1.0,               # 碳排放区域编号（默认1）
        ]
    end
    
    # 所有序号的结果都存储到busDC字段
    jpc_tp.busDC = dcbus_matrix

    # jpc_tp = JPC_tp_battery_bus_process(case, jpc_tp)
    
    return jpc_tp
end

# function JPC_tp_battery_bus_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
#     # 处理电池节点数据，转换为JPC_tp格式并合并到busDC
#     batteries = deepcopy(case.storageetap)
    
#     # 获取当前busDC的数据
#     busDC = jpc_tp.busDC
#     current_size = size(busDC, 1)
    
#     # 创建一个矩阵来存储电池节点数据
#     num_batteries = length(batteries)
#     battery_matrix = zeros(num_batteries, size(busDC, 2))
    
#     for (i, battery) in enumerate(batteries)
#         # 设置电压初始值
#         vm = 1.0
#         va = 0.0
#         vn_kv = battery.voc
#         # 创建虚拟节点数据
#         battery_row = zeros(1, size(busDC, 2))
#         battery_row[1, :] = [
#             current_size + i,    # 为电池分配新的节点ID
#             2.0,                 # 节点类型(全部设为slack节点)
#             0.0,                 # PD (MW) 有功负荷（MW）
#             0.0,                 # QD (MVAR) 无功负荷（MVAR）
#             0.0,                 # GS (MW) 有功发电（MW）
#             0.0,                 # BS (MVAR) 无功发电（MVAR）
#             1.0,                 # 区域编号
#             vm,                  # 节点电压幅值（p.u.）
#             va,                  # 节点电压相角（度）
#             vn_kv,               # 节点电压基准（kV）
#             1.0,                 # 区域编号
#             1.05,                 # 最大电压幅值（p.u.）
#             0.95,                  # 最小电压幅值（p.u.）
#             1.0,                 # 碳排放区域编号（默认1）
#             1.0,                 # 碳排放区域编号（默认1）
#         ]
        
#         # 将电池节点数据存入矩阵
#         battery_matrix[i, :] = battery_row
#     end
    
#     # 将电池虚拟节点合并到busDC中
#     jpc_tp.busDC = vcat(busDC, battery_matrix)
    
#     # # 同时保存原始电池数据到jpc_tp.battery字段，以便后续处理
#     # jpc_tp.battery = battery_matrix

#     return jpc_tp
# end

function JPC_tp_branches_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
    # if sequence == 1||sequence == 2
        # 处理线路数据，转换为JPC_tp格式
        calculate_line_parameters(case::JuliaPowerCase, jpc_tp)
        # 处理变压器数据，转换为JPC_tp格式
        calculate_transformer2w_parameters(case::JuliaPowerCase, jpc_tp)
        # 处理三相变压器数据，转换为JPC_tp格式
    # else
    #     # 处理支路数据，转换为JPC_tp格式
    #     calculate_branch_JPC_tp_zero(case::JuliaPowerCase, jpc_tp)
    # end
end

function JPC_tp_dcbranches_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
    # 处理直流线路数据，转换为JPC_tp格式
    nbr = length(case.branchesDC)
    branch = zeros(nbr, 14)
    dclines = case.branchesDC
    num_acbuses = size(case.busesAC, 1)

    for (i, dcline) in enumerate(dclines)
        # 获取起始和终止母线编号
        from_bus_idx = dcline.from_bus 
        to_bus_idx = dcline.to_bus 
        
        # 直接使用 LineDC 结构体中已有的标幺值参数
        r_pu = dcline.r_pu
        x_pu = 0.0
        
        # 填充branchDC矩阵
        branch[i, F_BUS] = from_bus_idx + num_acbuses
        branch[i, T_BUS] = to_bus_idx + num_acbuses
        branch[i, BR_R] = r_pu
        branch[i, BR_X] = x_pu
        
        # 设置额定容量（如果Excel中容量太小，使用默认最小容量）
        const_min_dcline_capacity_mw = 500.0  # 默认最小DC线路容量500 MW
        actual_dc_rate_a = dcline.rate_a_mw < const_min_dcline_capacity_mw ? const_min_dcline_capacity_mw : dcline.rate_a_mw
        actual_dc_rate_b = dcline.rate_b_mw < const_min_dcline_capacity_mw ? const_min_dcline_capacity_mw : dcline.rate_b_mw
        actual_dc_rate_c = dcline.rate_c_mw < const_min_dcline_capacity_mw ? const_min_dcline_capacity_mw : dcline.rate_c_mw
        branch[i, RATE_A] = actual_dc_rate_a
        branch[i, RATE_B] = actual_dc_rate_b
        branch[i, RATE_C] = actual_dc_rate_c
        
        # 设置支路状态
        branch[i, BR_STATUS] = dcline.status
        
        # 设置相角限制
        branch[i, ANGMIN] = -360.0
        branch[i, ANGMAX] = 360.0
        
        # 设置变比和相移角（默认值）
        branch[i, TAP] = 1.0
        branch[i, SHIFT] = 0.0
        
        # 设置电纳为0（直流线路没有电纳）
        branch[i, BR_B] = 0.0
    end
    # 将直流线路数据添加到JPC_tp结构体
    if isempty(jpc_tp.branchDC)
        jpc_tp.branchDC = branch
    else
        jpc_tp.branchDC = [jpc_tp.branchDC; branch]
    end
    
    # 处理储能虚拟连接
    # jpc_tp = JPC_tp_battery_branch_process(case, jpc_tp)

    return jpc_tp
end

# function JPC_tp_battery_branch_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
#     # 处理电池虚拟连接，创建从电池虚拟节点到实际连接节点的支路
#     batteries = deepcopy(case.storageetap)
#     num_batteries = length(batteries)
    
#     # 如果没有电池，直接返回
#     if num_batteries == 0
#         return jpc_tp
#     end
    
#     # 创建电池虚拟支路矩阵，与branchDC结构相同
#     battery_branches = zeros(num_batteries, 14)
    
#     # 获取当前busDC的大小，用于确定虚拟节点的编号
#     busDC_size = size(jpc_tp.busDC, 1) - num_batteries
    
#     for (i, battery) in enumerate(batteries)
#         # 获取电池连接的实际节点编号
#         actual_bus = battery.bus
        
#         # 计算电池虚拟节点编号（基于之前在JPC_tp_battery_process中的编号规则）
#         virtual_bus = busDC_size + i
        
#         # 获取节点的基准电压(kV)
#         basekv = 0.0
#         for j in 1:size(jpc_tp.busDC, 1)
#             if jpc_tp.busDC[j, 1] == actual_bus
#                 basekv = jpc_tp.busDC[j, BASE_KV]
#                 break
#             end
#         end
        
#         # 计算基准阻抗
#         baseR = (basekv^2) / case.baseMVA
        
#         # 计算标幺值阻抗（使用电池的内阻）
#         # r_pu = battery.ra / baseR
#         # r_pu = 0.0242/baseR  # 假设电池内阻为0.0242Ω
#         r_pu = 0.0252115/baseR  # 假设电池内阻为0.0249Ω
#         x_pu = 0  # 直流系统无感抗，设置为一个非常小的值
        
#         # 填充虚拟支路矩阵
#         battery_branches[i, F_BUS] = virtual_bus       # 虚拟节点
#         battery_branches[i, T_BUS] = actual_bus        # 实际连接节点
#         battery_branches[i, BR_R] = r_pu               # 标幺值电阻
#         battery_branches[i, BR_X] = x_pu               # 标幺值电抗
        
#         # 设置额定容量（基于电池参数计算）
#         # 假设电池额定容量可以从电池参数计算得到
#         rated_capacity = battery.package * battery.voc  # 简化计算，实际可能需要更复杂的公式
#         battery_branches[i, RATE_A] = rated_capacity
        
#         # 设置支路状态
#         battery_branches[i, BR_STATUS] = battery.in_service ? 1.0 : 0.0
        
#         # 设置相角限制（直流系统中通常不受限制）
#         battery_branches[i, ANGMIN] = -360.0
#         battery_branches[i, ANGMAX] = 360.0
#     end
    
#     # 将电池虚拟支路添加到branchDC中
#     if isempty(jpc_tp.branchDC)
#         jpc_tp.branchDC = battery_branches
#     else
#         jpc_tp.branchDC = [jpc_tp.branchDC; battery_branches]
#     end
    
#     return jpc_tp
# end

# function JPC_tp_battery_soc_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
#     # 处理电池SOC数据，转换为JPC_tp格式
#     batteries = deepcopy(case.storages)
#     num_batteries = length(batteries)
    
#     # 如果没有电池，直接返回
#     if num_batteries == 0
#         return jpc_tp
#     end
    
#     # 创建电池SOC矩阵
#     battery_soc = zeros(num_batteries, 8)  
    
#     for (i, battery) in enumerate(batteries)
#         battery_soc[i, 1] = battery.bus  # 电池连接的母线ID
#         battery_soc[i, 2] = battery.power_capacity_mw   # 电池的SOC值（标幺值）
#         battery_soc[i, 3] = battery.energy_capacity_mwh  # 电池的有功功率（MW）
#         battery_soc[i, 4] = battery.soc_init  # 电池的无功功率（MVAR）
#         battery_soc[i, 5] = battery.min_soc  # 电池的最大有功功率（MW）
#         battery_soc[i, 6] = battery.max_soc  # 电池的最小有功功率（MW）
#         battery_soc[i, 7] = battery.efficiency  # 电池的最大无功功率（MVAR）
#         battery_soc[i, 8] = battery.in_service ? 1.0 : 0.0  # 电池是否在服务中（1.0表示在服务，0.0表示不在服务）
#     end
#     for (i, battery) in enumerate(batteries)
#         # 获取电池连接的实际节点编号
#         bus_id = battery.bus
#         # 在JPC_tp的busDC中查找对应的节点
#         bus_index = findfirst(x -> x[1] == bus_id, jpc_tp.busDC[:, 1])
#         jpc_tp.busDC[bus_index, PD] -= 0.0
#         loadDC = zeros(1, 8)  # 创建一个空的负荷矩阵
#         nd = size(jpc_tp.busDC, 1)
#         loadDC[1, 1] = nd + 1  # 设置负荷对应的母线ID
#         loadDC[1, 2] = bus_index
#         loadDC[1, 3] = 1 # inservice
#         loadDC[1, 4] = 0.0
#         loadDC[1, 5] = 0.0
#         loadDC[1, 6] = 0.0  
#         loadDC[1, 7] = 0.0
#         loadDC[1, 8] = 1.0
#         # 将负荷数据添加到JPC_tp的负荷矩阵中
#         if isempty(jpc_tp.loadDC)
#             jpc_tp.loadDC = loadDC
#         else
#             jpc_tp.loadDC = [jpc_tp.loadDC; loadDC]
#         end
#     end
#     # 将电池SOC数据添加到JPC_tp结构体
#     jpc_tp.storage = battery_soc
    
#     return jpc_tp
# end


function calculate_line_parameters(case::JuliaPowerCase, jpc_tp::JPC_tp)
    # 处理线路数据，转换为JPC_tp格式
    nbr = length(case.branchesAC)
    branch = zeros(nbr, 14)
    lines = case.branchesAC

    for (i, line) in enumerate(lines)
        # 获取起始和终止母线编号
        from_bus_idx = line.from_bus
        to_bus_idx = line.to_bus
        
        # 直接使用 Line 结构体中已有的标幺值参数
        r_pu = line.r_pu
        x_pu = line.x_pu
        b_pu = line.b_pu
        
        # 填充branchAC矩阵
        branch[i, F_BUS] = from_bus_idx
        branch[i, T_BUS] = to_bus_idx
        branch[i, BR_R] = r_pu
        branch[i, BR_X] = x_pu
        branch[i, BR_B] = b_pu
        
        # 设置额定容量（如果Excel中容量太小，使用默认最小容量）
        const_min_line_capacity_mva = 500.0  # 默认最小线路容量500 MVA
        actual_rate_a = line.rate_a_mva < const_min_line_capacity_mva ? const_min_line_capacity_mva : line.rate_a_mva
        actual_rate_b = line.rate_b_mva < const_min_line_capacity_mva ? const_min_line_capacity_mva : line.rate_b_mva
        actual_rate_c = line.rate_c_mva < const_min_line_capacity_mva ? const_min_line_capacity_mva : line.rate_c_mva
        branch[i, RATE_A] = actual_rate_a
        branch[i, RATE_B] = actual_rate_b
        branch[i, RATE_C] = actual_rate_c
        
        # 设置支路状态
        branch[i, BR_STATUS] = line.status
        
        # 设置相角限制
        branch[i, ANGMIN] = -360.0
        branch[i, ANGMAX] = 360.0
        
        # 设置变比和相移角（默认值）
        branch[i, TAP] = 1.0
        branch[i, SHIFT] = 0.0
    end

    jpc_tp.branchAC = branch
end

function calculate_transformer2w_parameters(case::JuliaPowerCase, jpc_tp::JPC_tp)
    # 处理变压器数据，转换为JPC_tp格式
    transformers = case.transformers_2w_etap
    nbr = length(transformers)
    
    if nbr == 0
        return  # 如果没有变压器，直接返回
    end
    
    # 创建变压器分支矩阵
    branch = zeros(nbr, 14)
    
    for (i, transformer) in enumerate(transformers)
        # 获取高压侧和低压侧母线编号
        hv_bus_idx = transformer.hv_bus
        lv_bus_idx = transformer.lv_bus
        
        # 获取高压侧母线的基准电压(kV)
        hv_basekv = jpc_tp.busAC[hv_bus_idx, BASE_KV]
        
        # 计算阻抗参数
        # 变压器阻抗百分比转换为标幺值
        z_pu = transformer.z_percent
        x_r_ratio = transformer.x_r
        
        # 计算电阻和电抗（考虑基准功率转换）
        s_ratio = transformer.sn_mva / case.baseMVA
        z_pu = z_pu / s_ratio  # 转换到系统基准
        
        r_pu = z_pu / sqrt(1 + x_r_ratio^2)
        x_pu = r_pu * x_r_ratio
        
        # 考虑并联变压器
        parallel = transformer.parallel
        if parallel > 1
            r_pu = r_pu / parallel
            x_pu = x_pu / parallel
        end
        
        # 填充分支矩阵
        branch[i, F_BUS] = hv_bus_idx
        branch[i, T_BUS] = lv_bus_idx
        branch[i, BR_R] = r_pu
        branch[i, BR_X] = x_pu
        branch[i, BR_B] = 0.0  # 变压器通常没有并联电纳
        
        # 设置变比和相移
        branch[i, TAP] = 1.0  # 默认变比为1.0
        branch[i, SHIFT] = 0.0  # 默认相移角度为0.0
        
        # 设置额定容量
        branch[i, RATE_A] = case.baseMVA 
        
        # 设置支路状态
        branch[i, BR_STATUS] = transformer.in_service ? 1.0 : 0.0
        
        # 设置相角限制
        branch[i, ANGMIN] = -360.0
        branch[i, ANGMAX] = 360.0
    end
    
    # 将变压器分支数据添加到JPC_tp结构体
    if isempty(jpc_tp.branchAC)
        jpc_tp.branchAC = branch
    else
        jpc_tp.branchAC = [jpc_tp.branchAC; branch]
    end
end

function JPC_tp_gens_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
    # 统计各类发电设备数量
    n_gen = length(case.gensAC)
    n_sgen = length(case.sgensAC)
    n_ext = length(case.ext_grids)
    
    # 计算总发电设备数量
    total_gens = n_gen + n_sgen + n_ext
    
    if total_gens == 0
        return  # 如果没有发电设备，直接返回
    end
    
    # 创建发电机矩阵，行数为发电设备数量，列数为27
    gen_data = zeros(total_gens, 27)
    
    # 处理外部电网(通常作为平衡节点/参考节点)
    for (i, ext) in enumerate(case.ext_grids)
        if ext.status == 0
            continue
        end
        
        bus_idx = ext.bus_id
        
        # 填充发电机数据
        gen_data[i, :] = [
            bus_idx,        # 发电机连接的母线编号
            0.0,            # 有功功率输出(MW)
            0.0,            # 无功功率输出(MVAr)
            9999.0,         # 最大无功功率输出(MVAr)
            -9999.0,        # 最小无功功率输出(MVAr)
            ext.vm_pu,      # 电压幅值设定值(p.u.)
            case.baseMVA,   # 发电机基准容量(MVA)
            1.0,            # 发电机状态(1=运行, 0=停运)
            9999.0,         # 最大有功功率输出(MW)
            -9999.0,        # 最小有功功率输出(MW)
            0.0,            # PQ能力曲线低端有功功率输出(MW)
            0.0,            # PQ能力曲线高端有功功率输出(MW)
            0.0,            # PC1处最小无功功率输出(MVAr)
            0.0,            # PC1处最大无功功率输出(MVAr)
            0.0,            # PC2处最小无功功率输出(MVAr)
            0.0,            # PC2处最大无功功率输出(MVAr)
            0.0,            # AGC调节速率(MW/min)
            0.0,            # 10分钟备用调节速率(MW)
            0.0,            # 30分钟备用调节速率(MW)
            0.0,            # 无功功率调节速率(MVAr/min)
            1.0,            # 区域参与因子
            2.0,            # 发电机模型(2=多项式成本模型)
            0.0,            # 启动成本(美元)
            0.0,            # 关机成本(美元)
            3.0,            # 多项式成本函数系数数量
            0.0,             # 成本函数参数(后续需要扩展)
            0.0,            # 碳排放默认值
        ]
        
    end
    
    # 处理常规发电机(通常作为PV节点)
    offset = n_ext
    for (i, gen) in enumerate(case.gensAC)
        if gen.status == 0
            continue
        end
        
        idx = i + offset
        bus_idx = gen.bus
        
        # 计算无功功率(如果没有直接给出)
        q_mvar = 0.0
        if hasfield(typeof(gen), :q_mvar)
            q_mvar = gen.q_mvar
        else
            # 根据功率因数计算无功功率
            p_mw = gen.p_mw * gen.scaling
            if gen.cos_phi > 0 && p_mw > 0
                q_mvar = p_mw * tan(acos(gen.cos_phi))
            end
        end
        
        # 基准容量
        mbase = gen.sn_mva > 0 ? gen.sn_mva : case.baseMVA
        
        # 爬坡率参数
        ramp_agc = hasfield(typeof(gen), :ramp_up_rate_mw_per_min) ? 
                   gen.ramp_up_rate_mw_per_min : 
                   (gen.max_p_mw - gen.min_p_mw) / 10
        ramp_10 = hasfield(typeof(gen), :ramp_up_rate_mw_per_min) ? 
                  gen.ramp_up_rate_mw_per_min * 10 : 
                  gen.max_p_mw - gen.min_p_mw
        ramp_30 = hasfield(typeof(gen), :ramp_up_rate_mw_per_min) ? 
                  gen.ramp_up_rate_mw_per_min * 30 : 
                  gen.max_p_mw - gen.min_p_mw
        
        # 填充发电机数据
        gen_data[idx, :] = [
            bus_idx,                               # 发电机连接的母线编号
            gen.p_mw * gen.scaling,                # 有功功率输出(MW)
            q_mvar,                                # 无功功率输出(MVAr)
            gen.max_q_mvar,                        # 最大无功功率输出(MVAr)
            gen.min_q_mvar,                        # 最小无功功率输出(MVAr)
            gen.vm_pu,                             # 电压幅值设定值(p.u.)
            mbase,                                 # 发电机基准容量(MVA)
            1.0,                                   # 发电机状态(1=运行, 0=停运)
            gen.max_p_mw,                          # 最大有功功率输出(MW)
            gen.min_p_mw,                          # 最小有功功率输出(MW)
            gen.min_p_mw,                          # PQ能力曲线低端有功功率输出(MW)
            gen.max_p_mw,                          # PQ能力曲线高端有功功率输出(MW)
            gen.min_q_mvar,                        # PC1处最小无功功率输出(MVAr)
            gen.max_q_mvar,                        # PC1处最大无功功率输出(MVAr)
            gen.min_q_mvar,                        # PC2处最小无功功率输出(MVAr)
            gen.max_q_mvar,                        # PC2处最大无功功率输出(MVAr)
            ramp_agc,                              # AGC调节速率(MW/min)
            ramp_10,                               # 10分钟备用调节速率(MW)
            ramp_30,                               # 30分钟备用调节速率(MW)
            (gen.max_q_mvar - gen.min_q_mvar)/10,  # 无功功率调节速率(MVAr/min)
            1.0,                                   # 区域参与因子
            2.0,                                   # 发电机模型(2=多项式成本模型)
            0.0,                                   # 启动成本(美元)
            0.0,                                   # 关机成本(美元)
            3.0,                                   # 多项式成本函数系数数量
            0.0,                                   # 成本函数参数(后续需要扩展)
            0.0                                    # 碳排放占位
        ]
        
        # 如果母线尚未设置为参考节点，则设置为PV节点
        if jpc_tp.busAC[bus_idx, 2] != 3  # 3表示REF节点
            jpc_tp.busAC[bus_idx, 2] = 2  # 2表示PV节点
        end
    end
    
    # 处理静态发电机(通常作为PQ节点，但如果有电压控制能力，也可以是PV节点)
    offset = n_ext + n_gen
    for (i, sgen) in enumerate(case.sgensAC)
        if !sgen.in_service
            continue
        end
        
        idx = i + offset
        bus_idx = sgen.bus
        
        # 填充发电机数据
        gen_data[idx, :] = [
            bus_idx,                                # 发电机连接的母线编号
            sgen.p_mw * sgen.scaling,               # 有功功率输出(MW)
            sgen.q_mvar * sgen.scaling,             # 无功功率输出(MVAr)
            sgen.max_q_mvar,                        # 最大无功功率输出(MVAr)
            sgen.min_q_mvar,                        # 最小无功功率输出(MVAr)
            1.0,                                    # 电压幅值设定值(p.u.)
            case.baseMVA,                           # 发电机基准容量(MVA)
            1.0,                                    # 发电机状态(1=运行, 0=停运)
            sgen.max_p_mw,                          # 最大有功功率输出(MW)
            sgen.min_p_mw,                          # 最小有功功率输出(MW)
            sgen.min_p_mw,                          # PQ能力曲线低端有功功率输出(MW)
            sgen.max_p_mw,                          # PQ能力曲线高端有功功率输出(MW)
            sgen.min_q_mvar,                        # PC1处最小无功功率输出(MVAr)
            sgen.max_q_mvar,                        # PC1处最大无功功率输出(MVAr)
            sgen.min_q_mvar,                        # PC2处最小无功功率输出(MVAr)
            sgen.max_q_mvar,                        # PC2处最大无功功率输出(MVAr)
            (sgen.max_p_mw - sgen.min_p_mw) / 10,   # AGC调节速率(MW/min)
            sgen.max_p_mw - sgen.min_p_mw,          # 10分钟备用调节速率(MW)
            sgen.max_p_mw - sgen.min_p_mw,          # 30分钟备用调节速率(MW)
            (sgen.max_q_mvar - sgen.min_q_mvar)/10, # 无功功率调节速率(MVAr/min)
            1.0,                                    # 区域参与因子
            2.0,                                    # 发电机模型(2=多项式成本模型)
            0.0,                                    # 启动成本(美元)
            0.0,                                    # 关机成本(美元)
            3.0,                                    # 多项式成本函数系数数量
            0.0,                                    # 成本函数参数(后续需要扩展)
            0.0                                     # 碳排放占位
        ]
        
        # 如果静态发电机可控且母线尚未设置为REF或PV节点，则可能设置为PV节点
        if sgen.controllable && jpc_tp.busAC[bus_idx, 2] == 1  # 1表示PQ节点
            jpc_tp.busAC[bus_idx, 2] = 2  # 2表示PV节点
        end
    end
    
    # 移除未使用的行(对应未投运的发电设备)
    active_rows = findall(x -> x > 0, gen_data[:, 8])  # 第8列是GEN_STATUS
    gen_data = gen_data[active_rows, :]
    
    # 将发电机数据存储到JPC_tp结构体
    jpc_tp.genAC = gen_data
    
end

# function JPC_tp_battery_gens_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
#     # 为电池虚拟节点创建虚拟发电机
#     batteries = deepcopy(case.storageetap)
#     num_batteries = length(batteries)
    
#     # 如果没有电池，直接返回
#     if num_batteries == 0
#         return jpc_tp
#     end
    
#     # 获取当前busDC的大小，用于确定虚拟节点的编号
#     busDC_size = size(jpc_tp.busDC, 1) - num_batteries
    
#     # 创建电池虚拟发电机矩阵
#     # genDC矩阵通常包含以下列：
#     # [GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...]
#     # 具体列数应与你的genDC结构一致
#     num_gen_cols = size(jpc_tp.genAC, 2)
#     if num_gen_cols == 0  # 如果genDC为空，设置一个默认列数
#         num_gen_cols = 10
#     end
    
#     battery_gens = zeros(num_batteries, num_gen_cols)
#      # 创建storage矩阵
#     num_storage_cols = 5  # 根据idx_ess函数定义的列数
#     storage_matrix = zeros(num_batteries, num_storage_cols)
    
#     for (i, battery) in enumerate(batteries)
#         # 计算电池虚拟节点编号
#         virtual_bus = busDC_size + i
        
#         # 计算电池的功率容量（基于电池参数）
#         # 这里使用简化计算，实际应根据电池特性进行更准确的计算
#         power_capacity = battery.package * battery.voc
        
#         # 填充虚拟发电机矩阵
#         battery_gens[i, 1] = virtual_bus       # GEN_BUS: 发电机连接的节点编号
#         battery_gens[i, 2] = 0.0               # PG: 初始有功功率输出(MW)，初始设为0
#         battery_gens[i, 3] = 0.0               # QG: 初始无功功率输出(MVAR)，直流系统通常为0
        
#         # 设置无功功率限制（直流系统通常不考虑）
#         battery_gens[i, 4] = 0.0               # QMAX: 最大无功功率输出
#         battery_gens[i, 5] = 0.0               # QMIN: 最小无功功率输出
        
#         # 设置电压和基准功率
#         battery_gens[i, 6] = 1.0               # VG: 电压设定值(p.u.)
#         battery_gens[i, 7] = case.baseMVA      # MBASE: 发电机基准功率(MVA)
        
#         # 设置发电机状态
#         battery_gens[i, 8] = battery.in_service ? 1.0 : 0.0  # GEN_STATUS: 发电机状态
        
#         # 设置有功功率限制（充电为负，放电为正）
#         battery_gens[i, 9] = power_capacity    # PMAX: 最大有功功率输出(MW)，放电功率
#         battery_gens[i, 10] = -power_capacity  # PMIN: 最小有功功率输出(MW)，充电功率

#          # 填充storage矩阵
#         storage_matrix[i, ESS_BUS] = virtual_bus               # ESS_BUS: 连接的节点编号
#         # storage_matrix[i, ESS_POWER_CAPACITY] = power_capacity # ESS_POWER_CAPACITY: 功率容量(MW)
#         # storage_matrix[i, ESS_ENERGY_CAPACITY] = 0
#         # storage_matrix[i, ESS_AREA] = 1                        # ESS_AREA: 区域编号，默认为1
        
#         # 如果genDC有更多列，根据需要设置其他参数
#         if num_gen_cols > 10
#             # 例如，设置爬坡率限制、成本系数等
#             # 这里需要根据你的系统具体需求进行设置
#             for j in 11:num_gen_cols
#                 battery_gens[i, j] = 0.0  # 默认设为0
#             end
#         end
#     end
    
#     # 将电池虚拟发电机添加到genDC中
#     if isempty(jpc_tp.genDC)
#         jpc_tp.genDC = battery_gens
#     else
#         jpc_tp.genDC = [jpc_tp.genDC; battery_gens]
#     end
    
#     # 将存储设备信息添加到storage中
#     if !isdefined(jpc_tp, :storageetap) || isempty(jpc_tp.storageetap)
#         jpc_tp.storageetap = storage_matrix
#     else
#         jpc_tp.storageetap = [jpc_tp.storageetap; storage_matrix]
#     end

#     return jpc_tp
# end


function JPC_tp_loads_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
    # 处理负荷数据，转换为JPC_tp格式并更新busAC的PD和QD
    
    # 默认负荷值（当Excel中没有给出负荷值时使用）
    # 配电网级别：使用 kW 量级的默认值
    const_default_p_mw = 0.1    # 默认有功负荷 100 kW = 0.1 MW
    const_default_q_mvar = 0.01 # 默认无功负荷 10 kVar = 0.01 MVar
    
    # 过滤出投运的负荷 (使用 status 字段)
    in_service_loads = filter(load -> load.status == 1, case.loadsAC)
    
    # 如果没有投运的负荷，直接返回
    if isempty(in_service_loads)
        return
    end
    
    # 创建一个空矩阵，行数为负荷数，列数为8
    num_loads = length(in_service_loads)
    load_matrix = zeros(num_loads, 9)
    
    # 创建一个字典，用于累加连接到同一母线的负荷
    bus_load_sum = Dict{Int, Vector{Float64}}()
    
    for (i, load) in enumerate(in_service_loads)
        # 使用 pd_mw 和 qd_mvar 字段
        # 如果Excel中没有给出负荷值（为0或值太小，小于等于1），则使用默认值
        actual_p_mw = abs(load.pd_mw) <= 1.0 ? const_default_p_mw : load.pd_mw
        actual_q_mvar = abs(load.qd_mvar) <= 1.0 ? const_default_q_mvar : load.qd_mvar
        
        # 填充负荷矩阵的每一行
        load_matrix[i, :] = [
            i,              # 负荷连接的母线编号
            load.bus_id,                     # 负荷编号
            1.0,                   # 负荷状态(1=投运)
            actual_p_mw,           # 有功负荷(MW)
            actual_q_mvar,         # 无功负荷(MVAr)
            0.0,  # 恒阻抗负荷百分比 (默认0)
            0.0,  # 恒电流负荷百分比 (默认0)
            1.0,   # 恒功率负荷百分比 (默认1)
            0,  # 负荷类型标识
        ]
    end
    
    # 将负荷数据存储到JPC_tp结构体
    jpc_tp.loadAC = load_matrix
    
end

function JPC_tp_dcloads_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
    # 处理直流负荷数据，转换为JPC_tp格式并更新busDC的PD和QD
    num_acbuses = size(jpc_tp.busAC, 1)
    
    # 默认负荷值（当Excel中没有给出负荷值或值太小时使用）
    # 配电网级别：使用 kW 量级的默认值
    const_default_p_mw = 0.1    # 默认有功负荷 100 kW = 0.1 MW
    
    # 过滤出投运的直流负荷 (使用 status 字段)
    in_service_dcloads = filter(dcload -> dcload.status == 1, case.loadsDC)
    
    # 如果没有投运的直流负荷，直接返回
    if isempty(in_service_dcloads)
        return
    end
    
    # 创建一个空矩阵，行数为直流负荷数，列数为8
    num_dcloads = length(in_service_dcloads)
    dcload_matrix = zeros(num_dcloads, 9)
    
    for (i, dcload) in enumerate(in_service_dcloads)
        # 如果Excel中没有给出负荷值（为0或值太小，小于等于1MW），则使用默认值
        actual_p_mw = abs(dcload.pd_mw) <= 1.0 ? const_default_p_mw : dcload.pd_mw
        
        # 填充直流负荷矩阵的每一行 (使用 pd_mw 和 bus_id 字段)
        dcload_matrix[i, :] = [
            dcload.index,              # 直流负荷编号
            dcload.bus_id + num_acbuses,     # 直流负荷连接的母线编号
            1.0,            # 直流负荷状态(1=投运)
            actual_p_mw,    # 有功负荷(MW)
            0.0,            # 无功负荷(MVAr)
            0.0,           # 恒阻抗百分比(默认0)
            0.0,           # 恒电流百分比(默认0)
            1.0,            # 恒功率百分比(默认1)
            0.0,            # 负荷类型标识(默认0)
        ]
        
    end
    
    # 将直流负荷数据存储到JPC_tp结构体
    jpc_tp.loadDC = dcload_matrix

    return jpc_tp
end

function JPC_tp_pv_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
    # 处理光伏发电机数据，转换为JPC_tp格式并更新busAC的PD和QD
    num_acbuses = size(jpc_tp.busAC, 1)
    
    # 过滤出投运的光伏发电机 (使用 status 字段)
    in_service_pvs = filter(pv -> pv.status == 1, case.pvarray)
    
    # 如果没有投运的光伏发电机，直接返回
    if isempty(in_service_pvs)
        return
    end
    
    # 创建一个空矩阵，行数为光伏发电机数，列数为10
    num_pvs = length(in_service_pvs)
    pv_matrix = zeros(num_pvs, 10)
    
    for (i, pv) in enumerate(in_service_pvs)
        # 使用 PVArray 结构体中的字段
        # voc_v, vmpp_v, isc_a, impp_a, irradiance_w_m2, area_m2, p_rated_mw
        Voc = pv.voc_v
        Vmpp = pv.vmpp_v
        Isc = pv.isc_a * (pv.irradiance_w_m2 / 1000.0)
        Impp = pv.impp_a * (pv.irradiance_w_m2 / 1000.0)

        # 优先使用 Excel 中的额定功率 (PVAPower)，如果没有则计算
        p_max = pv.p_rated_mw > 0.0 ? pv.p_rated_mw : (Vmpp * Impp / 1000000.0)
        
        # 填充光伏发电机矩阵的每一行
        pv_matrix[i, :] = [
            i,              # 光伏发电机编号
            pv.bus_id + num_acbuses,         # 光伏发电机连接的母线编号
            p_max,          # 光伏发电机有功出力(MW)
            0,              # 光伏发电机无功出力(MVar)
            Vmpp,           # 光伏发电机额定电压(V)
            Isc,            # 光伏发电机短路电流(A)
            Impp,           # 光伏发电机额定电流(A)
            pv.irradiance_w_m2,  # 光伏发电机辐照度(W/m²)
            pv.area_m2,            # area
            1.0,            # 光伏发电机状态(1=投运)
        ]
        
        # # 更新busAC矩阵中的PD和QD字段
        # bus_row = findfirst(x -> x == pv.bus, jpc_tp.busAC[:, 1])
        
        # if !isnothing(bus_row)
        #     jpc_tp.busAC[bus_row, PD] += pv_matrix[i, 4]  # PD - 有功负荷(MW)
        #     jpc_tp.busAC[bus_row, QD] += pv_matrix[i, 5]  # QD - 无功负荷(MVAr)
        # end
    end
    
    # 将光伏发电机数据存储到JPC_tp结构体
    jpc_tp.pv = pv_matrix

    return jpc_tp
    
end

function JPC_tp_ac_pv_system_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
    # 处理交流侧光伏系统数据，根据控制模式转换为发电机或负荷
    
    # 过滤出投运的交流侧光伏系统 (使用 status 字段)
    in_service_ac_pvs = filter(ac_pv -> ac_pv.status == 1, case.ACPVSystems)
    
    # 如果没有投运的交流侧光伏系统，直接返回
    if isempty(in_service_ac_pvs)
        return jpc_tp
    end
    
    # 创建一个空矩阵，行数为交流侧光伏系统数，列数为13
    num_ac_pvs = length(in_service_ac_pvs)
    ac_pv_matrix = zeros(num_ac_pvs, 15)
    for (i, ac_pv) in enumerate(in_service_ac_pvs)
        # 使用 ACPVSystem 结构体中的字段
        Vmpp = ac_pv.vmpp_v
        Voc = ac_pv.voc_v
        Isc = ac_pv.isc_a * (ac_pv.irradiance_w_m2 / 1000.0)
        Impp = ac_pv.impp_a * (ac_pv.irradiance_w_m2 / 1000.0)

        p_max = Vmpp * Impp / 1000000.0 * ac_pv.inverter_efficiency # 最大有功输出(MW)

        # inverter_mode: 1 = PQ控制, 其他 = PV控制
        mode = ac_pv.inverter_mode
        
        # 填充交流侧光伏系统矩阵的每一行
        ac_pv_matrix[i, :] = [
            i,                 # 交流光伏系统编号
            ac_pv.bus_id,         # 连接母线编号
            Voc,               # 光伏系统额定电压(V)
            Vmpp,              # 光伏系统额定电压(V)
            Isc,               # 光伏系统短路电流(A)
            Impp,              # 光伏系统额定电流(A)
            ac_pv.irradiance_w_m2,  # 光伏系统辐照度(W/m²)
            1.0 - ac_pv.inverter_efficiency,  # 损耗百分比
            mode,              # 控制模式
            ac_pv.inverter_pac_mw,           # 有功出力(MW)
            ac_pv.inverter_qac_mvar,         # 无功出力(MVAr)
            ac_pv.inverter_qac_max_mvar,     # 无功上限(MVAr)
            ac_pv.inverter_qac_min_mvar,     # 无功下限(MVAr)
            1,            # 区域编号
            ac_pv.status == 1 ? 1.0 : 0.0  # 光伏系统状态(1=投运, 0=停运)
        ]
    end
    # 将交流侧光伏系统数据存储到JPC_tp结构体
    jpc_tp.pv_acsystem = ac_pv_matrix
    
    return jpc_tp
end



function JPC_tp_inverters_process(case::JuliaPowerCase, jpc_tp::JPC_tp)
    # 处理逆变器数据，转换为JPC_tp格式并更新busAC和busDC的负荷

    # 如果没有投运的逆变器，直接返回
    if isempty(case.converters)
        return jpc_tp
    end
    num_acbuses = size(jpc_tp.busAC, 1)
    
    # 获取当前负荷数量，用于新增负荷的编号
    nld_ac = size(jpc_tp.loadAC, 1)  # 交流侧负荷数量
    nld_dc = size(jpc_tp.loadDC, 1)  # 直流侧负荷数量
    
    # 创建用于存储需要新增的负荷记录
    # 使用矩阵而不是数组来存储新负荷
    num_cols_ac = size(jpc_tp.loadAC, 2)
    num_cols_dc = size(jpc_tp.loadDC, 2)
    
    # 计算需要添加的最大可能负荷数（每个逆变器最多添加一个负荷）
    max_new_loads = length(case.converters)
    new_loads_ac = zeros(0, num_cols_ac)  # 创建一个空矩阵，行数为0，列数与loadAC相同
    new_loads_dc = zeros(0, num_cols_dc)  # 创建一个空矩阵，行数为0，列数与loadDC相同

    #创建converter空矩阵
    converters = zeros(0, 18)
    
    # 跟踪新增负荷的数量
    new_ac_load_count = 0
    new_dc_load_count = 0

   for (i, inverter) in enumerate(case.converters)
        #为converter矩阵添加连接关系
        converter = zeros(1, 18)  # 创建一行
        
        # 逆变器的工作模式 (使用 mode 字段，为整数类型)
        mode = inverter.mode
        if mode == 1  # δs_Us模式
            converter[1, CONV_MODE] = 1.0
        elseif mode == 0  # Ps_Qs模式
            converter[1, CONV_MODE] = 0.0
        elseif mode == 3  # Ps_Us模式
            converter[1, CONV_MODE] = 3.0
        elseif mode == 4  # Udc_Qs模式
            converter[1, CONV_MODE] = 4.0
        elseif mode == 5  # Udc_Us模式
            converter[1, CONV_MODE] = 5.0
        elseif mode == 6  # Droop_Udc_Qs模式
            converter[1, CONV_MODE] = 6.0
        elseif mode == 7  # Droop_Udc_Us模式
            converter[1, CONV_MODE] = 7.0
        else
            @warn "逆变器 $i 的控制模式 $mode 未知或不支持，默认启用 Ps_Qs"
            converter[1, CONV_MODE] = 0.0  # 设置为默认值
        end

        # 计算交流侧功率 (使用 p_ac_mw 和 q_ac_mvar 字段)
        p_ac = -inverter.p_ac_mw 
        q_ac = -inverter.q_ac_mvar 
        
        # 计算直流侧功率（考虑效率）(使用 efficiency 字段)
        efficiency = inverter.efficiency
        
        if p_ac <= 0  # 交流侧输出功率，直流侧输入功率
            p_dc = -p_ac / efficiency  # 负值，表示直流侧消耗的功率
        else  # 交流侧输入功率，直流侧输出功率
            p_dc = -p_ac * efficiency  # 正值，表示直流侧输出的功率
        end

        converter[1,CONV_ACBUS] = inverter.ac_bus_id
        converter[1,CONV_DCBUS] = inverter.dc_bus_id + num_acbuses
        converter[1,CONV_INSERVICE] = inverter.status == 1 ? 1.0 : 0.0
        converter[1,CONV_P_AC] = inverter.p_ac_mw *10
        converter[1,CONV_Q_AC] = inverter.q_ac_mvar *10
        converter[1,CONV_P_DC] = p_dc
        converter[1,CONV_EFF] = efficiency
        converter[1,CONV_DROOP_KP] = 0.0  # 默认droop系数
        converters = vcat(converters, converter)
    end

    jpc_tp.converter = converters

    return jpc_tp
end


function JPC_tp_energy_router_process(case::JuliaPowerCase,jpc_tp::JPC_tp)
    # 处理能量路由器
    # 这里假设case中有energy_router字段，包含所有能量路由器的信息
    if length(case.energyrouter) == 0
        return jpc_tp  # 如果没有能量路由器，直接返回原始JPC_tp
    end
    
    energy_routers = case.energyrouter
    for router in energy_routers
       # 提取出该能量路由器中包含的Energy Router Converter
        prim_converters = router.prime_converter
        second_converters = router.second_converter
        for prim_converter in prim_converters
            # 处理主转换器
            prime_conv = JPC_tp_energy_router_converter_process(prim_converter, jpc_tp, "primary")
        end
        for second_converter in second_converters
            # 处理次级转换器
            second_conv = JPC_tp_energy_router_converter_process(second_converter, jpc_tp, "secondary")
        end
        
    end
    
    return jpc_tp
end

function JPC_tp_energy_router_converter_process(prim_converter, jpc_tp, type)
    if type == "primary"
        prime_conv = zeros(0,18)
        # 处理主转换器
        bus_ac = prim_converter.bus_ac
        bus_dc = prim_converter.bus_dc
        p_mw = prim_converter.p_mw
        q_mvar = prim_converter.q_mvar
        loss_percent = prim_converter.loss_percent
        mode = prim_converter.control_mode
        inservice = prim_converter.in_service

        
    else

    end
end


function sparse_matrix(jpc)
    nb_ac = size(jpc.busAC ,1)
    nb_dc = size(jpc.busDC ,1)
    nb = nb_ac + nb_dc
    ng = size(jpc.genAC ,1)
    
    Cft_ac = build_incidence_matrix(jpc.branchAC[:, 1:2], nb)
    Cft_dc = build_incidence_matrix(jpc.branchDC[:, 1:2], nb)
    Cft_vsc = build_incidence_matrix(jpc.converter[:, 1:2], nb)
    Cg = ng > 0 ? zeros(Int, ng, nb) : zeros(Int, 0, nb)
    for gen_idx in 1:ng
        Cg[gen_idx, Int(jpc.genAC[gen_idx, 1])] = 1
    end
    Cd_ac = length(jpc.loadAC) > 0 ? zeros(Int, size(jpc.loadAC, 1), nb) : zeros(Int, 0, nb)
    for load_idx in 1:size(jpc.loadAC, 1)
        Cd_ac[load_idx, Int(jpc.loadAC[load_idx, 2])] = 1
    end
    Cd_dc = length(jpc.loadDC) > 0 ? zeros(Int, size(jpc.loadDC, 1), nb) : zeros(Int, 0, nb)
    for load_idx in 1:size(jpc.loadDC, 1)
        Cd_dc[load_idx, Int(jpc.loadDC[load_idx, 2])] = 1
    end
    Cd = vcat(Cd_ac, Cd_dc)
    gen_bus_indices = ng > 0 ? Int.(round.(jpc.genAC[:, 1])) : Int[]
    non_gen_buses = setdiff(1:nb, gen_bus_indices)
    num_non_gen = length(non_gen_buses)
    Cdf = num_non_gen > 0 ? zeros(Int, num_non_gen, nb) : zeros(Int, 0, nb)
    for (i, bus) in enumerate(non_gen_buses)
        Cdf[i, Int(bus)] = 1
    end
    Cmg_ac = size(jpc.pv_acsystem, 1) > 0 ? zeros(Int, size(jpc.pv_acsystem, 1), nb) : zeros(Int, 0, nb)
    for mg_index in 1:size(jpc.pv_acsystem, 1)
        Cmg_ac[mg_index, Int(jpc.pv_acsystem[mg_index, 2])] = 1
    end
    Cmg_dc = size(jpc.pv, 1) > 0 ? zeros(Int, size(jpc.pv, 1), nb) : zeros(Int, 0, nb)
    for mg_index in 1:size(jpc.pv, 1)
        Cmg_dc[mg_index, Int(jpc.pv[mg_index, 2])] = 1
    end
    Cmg = vcat(Cmg_ac, Cmg_dc)

    C_sparse = Dict(
        :Cft_ac => Cft_ac,
        :Cft_dc => Cft_dc,
        :Cft_vsc => Cft_vsc,
        :Cg => Cg,
        :Cd => Cd,
        :Cdf => Cdf,
        :Cmg => Cmg
    )

    return C_sparse

end

function build_incidence_matrix(connections, num_nodes)
    # 筛选指定类型的线路
    num_lines = size(connections, 1)

    # 初始化行索引、列索引和值
    row_indices = Int[]
    col_indices = Int[]
    values = Int[]

    # 填充行、列和值
    for line_idx in 1:num_lines
        from = connections[line_idx, 1]
        to   = connections[line_idx, 2]

        push!(row_indices, line_idx)
        push!(col_indices, from)
        push!(values, 1)  # 起点

        push!(row_indices, line_idx)
        push!(col_indices, to)
        push!(values, -1) # 终点
    end

    # 构造稀疏矩阵
    return sparse(row_indices, col_indices, values, num_lines, num_nodes)
end