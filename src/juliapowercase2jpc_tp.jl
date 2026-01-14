function build_incidence_matrix(connections, num_nodes, line_type)
    # 筛选指定类型的线路
    filtered_connections = filter(x -> x[3] == line_type, connections)
    num_lines = length(filtered_connections)

    # 初始化行索引、列索引和值
    row_indices = Int[]
    col_indices = Int[]
    values = Int[]

    # 填充行、列和值
    for (line_idx, (from, to, _)) in enumerate(filtered_connections)
        push!(row_indices, line_idx)
        push!(col_indices, from)
        push!(values, 1)  # 起点为 1

        push!(row_indices, line_idx)
        push!(col_indices, to)
        push!(values, -1)  # 终点为 -1
    end

    # 构造稀疏矩阵
    return sparse(row_indices, col_indices, values, num_lines, num_nodes)
end


function load_case(filepath)
    name_to_index = Dict{String, Int}()
    # ================== 节点处理 ==================
    bus_df = DataFrame(XLSX.readtable(filepath, "bus"))

    bus_ac = []  # 交流节点索引
    VMAX_ac = []  # 交流节点电压上限
    VMIN_ac = []  # 交流节点电压下限
    
    for row in eachrow(bus_df)
        # 使用symbol或string索引而不是属性访问
        push!(VMAX_ac, row[:VMaxLimit]/100)  # 或者使用 row["VMaxLimit"]
        push!(VMIN_ac, row[:VMinLimit]/100)
        name_to_index[row[:ID]] = row[:Index]
        push!(bus_ac, row[:Index])
    end

    num_ac_nodes = length(bus_ac)
    dcbus_df = DataFrame(XLSX.readtable(filepath, "dcbus"))
    bus_dc = []  # 直流节点索引
    VMAX_dc = []  # 直流节点电压上限
    VMIN_dc = []  # 直流节点电压下限
    
    for (i, row) in enumerate(eachrow(dcbus_df))
        new_index = num_ac_nodes + i  # 新索引从 num_ac_nodes+1 开始
        push!(VMAX_dc, 1.05)  # 直流节点电压上限
        push!(VMIN_dc, 0.9)   # 直流节点电压下限
        name_to_index[row[:ID]] = new_index  # 使用新索引
        push!(bus_dc, new_index)  # 存储新索引
    end

    VMAX = vcat(VMAX_ac, VMAX_dc)  # 合并交流和直流节点电压上限
    VMIN = vcat(VMIN_ac, VMIN_dc)  # 合并交流和直流节点电压下限
    buses = vcat(bus_ac, bus_dc)  # 合并节点索引
    
    # println("节点电压上限: ", VMAX)
    # println("节点电压下限: ", VMIN)
    # println("节点索引: ", buses)
    
    # ================== 线路处理 ==================
    all_connections = []  # 存储所有线路的连接信息 (from, to, type)
    R_ac = []  # 交流线路的电阻
    X_ac = []  # 交流线路的电抗
    R_dc = []  # 直流线路的电阻
    X_dc = []  # 直流线路的电抗
    α_ac = []  # 交流线路是否在运行
    α_dc = []  # 直流线路是否在运行

    line_df = DataFrame(XLSX.readtable(filepath, "cable"))
    for row in eachrow(line_df)
        # 检查起点和终点是否存在于 name_to_index 中
        if !haskey(name_to_index, row.FromBus) || !haskey(name_to_index, row.ToBus)
            error("交流线路的起点或终点节点不存在: $(row.FromBus) -> $(row.ToBus)")
        end
        # 提取线路阻抗参数
        push!(R_ac, row.RPosValue)  # 电阻
        push!(X_ac, row.XPosValue)  # 电抗
        # 存储线路连接信息
        push!(all_connections, (name_to_index[row.FromBus], name_to_index[row.ToBus], "AC"))
        # 将字符串"true"/"false"转换为整数1/0
        push!(α_ac, row.InService == "true" ? 1 : 0) 
    end

    dcline_df = DataFrame(XLSX.readtable(filepath, "dcimpedance"))
    for row in eachrow(dcline_df)
        # 检查起点和终点是否存在于 name_to_index 中
        if !haskey(name_to_index, row.FromBus) || !haskey(name_to_index, row.ToBus)
            error("直流线路的起点或终点节点不存在: $(row.FromBus) -> $(row.ToBus)")
        end
        # 提取直流线路阻抗参数
        push!(R_dc, row.RValue)  # 电阻
        push!(X_dc, 0)  # 电抗
        # 存储线路连接信息
        push!(all_connections, (name_to_index[row.FromBus], name_to_index[row.ToBus], "DC"))
        # 将字符串"true"/"false"转换为整数1/0
        push!(α_dc, row.InService == "true" ? 1 : 0)  # 或 Int(row.InService == "true")
    end 
    
    R = vcat(R_ac, R_dc)  # 合并交流和直流线路电阻
    X = vcat(X_ac, X_dc)  # 合并交流和直流线路电抗

    # println(all_connections)
    # println("线路电阻: ", R)
    # println("线路电抗: ", X)
    # println("线路是否在运行: ", α)

    # 处理换流器(converter)
    converter_df = DataFrame(XLSX.readtable(filepath, "inverter"))
    Pvscmax = []  # 换流器最大有功
    η = []  # 换流器效率 (假设为常数)
    α_vsc = []
    for row in eachrow(converter_df)
        # 验证节点存在性
        if !haskey(name_to_index, row.BusID)
            error("换流器关联的交流节点不存在: $(row.BusID)")
        end
        if !haskey(name_to_index, row.CZNetwork)
            error("换流器关联的直流节点不存在: $(row.CZNetwork)")
        end
        # 换流器最大有功
        push!(Pvscmax, row.DckW)  # 有功功率
        # 换流器效率
        push!(η, row.DcPercentEFF/100)  # 效率系数
        
        push!(all_connections,
            (name_to_index[row.BusID], name_to_index[row.CZNetwork], "CONVERTER")
        )
        # 将字符串"true"/"false"转换为整数1/0
        push!(α_vsc, row.InService == "true" ? 1 : 0)  
    end    

    # println("换流器最大有功: ", Pvscmax)
    # println("换流器效率: ", η)
    # println("换流器是否在运行: ", α_vsc)

    # 构建传输矩阵
    num_nodes = length(name_to_index)
    # println("节点数量: ", num_nodes)
    Cft_ac = build_incidence_matrix(all_connections, num_nodes, "AC")
    Cft_dc = build_incidence_matrix(all_connections, num_nodes, "DC")
    # println("Cft_ac = ")
    # display(Matrix(Cft_ac))
    # println("Cft_dc = ")
    # display(Matrix(Cft_dc))
    Cft_vsc = build_incidence_matrix(all_connections, num_nodes, "CONVERTER")
    # println("Cft_vsc = ")
    # display(Matrix(Cft_vsc))

    # 合并所有线路的 in service 状态
    α = vcat(α_ac, α_dc, α_vsc)
    # println("线路是否在运行: ", α)

    # ================== 发电机处理 ==================
    gen_index = Tuple{Int, String}[]  # (index, type)
    gen_df = DataFrame(XLSX.readtable(filepath, "util"))
    Pgmax = []  # 发电机最大出力
    Qgmax = []  # 发电机最大无功出力
    gen = zeros(Int, num_nodes)
    GEN_BUS = []  # 发电机连接节点
    for row in eachrow(gen_df)
        # 验证发电机关联的节点存在性
        if !haskey(name_to_index, row.Bus)
            error("发电机关联的节点不存在: $(row.Bus)")
        end
        # 发电机最大出力
        push!(Pgmax, row.OpMW *1000)  # 有功功率
        push!(Qgmax, row.OpMvar *1000)  # 无功功率

        # 这里可以添加更多发电机处理逻辑
        push!(gen_index,
            (name_to_index[row.Bus], "GEN")
        )
    end
    GEN_BUS = [idx for (idx, _) in gen_index]  # 只保留节点编号
    # println("发电机连接节点: ", GEN_BUS)
    # 构建连接矩阵
    Cg = zeros(Int, nrow(gen_df), num_nodes)  # 发电机连接矩阵 (1×num_nodes)
    for (i, (idx, _)) in enumerate(gen_index)
        Cg[i, idx] = 1
    end
    # println("Cg = ")
    # display(Cg)

    # ================== 微网处理 ==================
    nmg_index = Tuple{Int, String}[]  # (index, type)
    nmg_df = DataFrame(XLSX.readtable(filepath, "pvarray"))
    Pmgmax = []  # 微网有功
    Qmgmax = []  # 微网无功
    Pmgmin = []  # 微网最小有功
    Qmgmin = []  # 微网最小无功
    mg_buses = []  # 微网节点索引
    for row in eachrow(nmg_df)
        # 验证微网关联的节点存在性
        if !haskey(name_to_index, row.Bus)
            error("微网关联的节点不存在: $(row.Bus)")
        end
        # 微网值
        push!(Pmgmax, row.PVAPower)    # 有功功率
        push!(Qmgmax, row.PVAPower/5)  # 无功功率
        push!(Pmgmin, 0)               # 有功功率
        push!(Qmgmin, 0)               # 无功功率
        
        # 微网索引
        push!(nmg_index,
            (name_to_index[row.Bus], "NMG")
        )
        push!(mg_buses, name_to_index[row.Bus])  # 添加微网节点索引
    end
    Cmg = zeros(Int, nrow(nmg_df), num_nodes)  # 微网连接矩阵 (n_mg×num_nodes)
    for (i, (idx, _)) in enumerate(nmg_index)
        Cmg[i, idx] = 1
    end
    # println("Cmg = ")
    # display(Cmg)
    # println("Pmgmax = ")
    # display(Pmgmax)
    # println("Qmgmax = ")
    # display(Qmgmax)
    # println("Pmgmin = ")
    # display(Pmgmin)
    # println("Qmgmin = ")
    # display(Qmgmin)
    Fmgmax = fill(1, nrow(nmg_df))                  # 微网虚拟功率

    # ================== 负荷处理 ==================
    load_index = Tuple{Int, String}[]  # (index, type)
    load_df = DataFrame(XLSX.readtable(filepath, "lumpedload"))
    Pd = []  
    Qd = []  
    const_load_p = 100.0
    const_load_q = const_load_p * 0.1
    for row in eachrow(load_df)
        # 验证负荷关联的节点存在性
        if !haskey(name_to_index, row.Bus)
            error("负荷关联的节点不存在: $(row.Bus)")
        end

        # 负荷值
        push!(Pd, const_load_p)      # 有功功率
        push!(Qd, const_load_q)      # 无功功率
        
        # 负荷索引
        push!(load_index,
            (name_to_index[row.Bus], "LOAD")
        )
    end

    load_dc_df = DataFrame(XLSX.readtable(filepath, "dclumpload"))
    Pd_dc = []  
    Qd_dc = [] 
    for row in eachrow(load_dc_df)
        # 验证负荷关联的节点存在性
        if !haskey(name_to_index, row.Bus)
            error("负荷关联的节点不存在: $(row.Bus)")
        end

        # 负荷值
        push!(Pd_dc, const_load_p)     # 有功功率，与交流负荷保持一致
        push!(Qd_dc, const_load_q)     # 无功功率
        
        # 负荷索引
        push!(load_index,
            (name_to_index[row.Bus], "LOAD")
        )
    end

    # 合并交流和直流负荷
    Pd = vcat(Pd, Pd_dc)  # 合并交流和直流
    Qd = vcat(Qd, Qd_dc)  # 合并交流和直流

    # Cd: 只对有负荷的节点，16*28
    Cd = zeros(Int, length(load_index), num_nodes)
    for (i, (node_idx, _)) in enumerate(load_index)
        Cd[i, node_idx] = 1
    end

    # Cdf: 所有非电源节点，26*28
    all_nodes = collect(1:num_nodes)
    non_gen_nodes = [i for i in all_nodes if !(i in GEN_BUS)]
    Cdf = zeros(Int, length(non_gen_nodes), num_nodes)
    for (i, node_idx) in enumerate(non_gen_nodes)
        Cdf[i, node_idx] = 1
    end

    # Fd: 负荷节点，26*1
    Fd = ones(length(non_gen_nodes))
    # println("Cd = ")
    # display(Cd)
    # println("Pd = ") 
    # display(Pd)
    # println("Qd = ")
    # display(Qd)

    # # ================== 断路器处理 ==================
    switch_df = DataFrame(XLSX.readtable(filepath, "hvcb"))
    switch_lines = [(row.FromElement, row.ToElement) for row in eachrow(switch_df)]
    switch_flag = Int[]  # 1表示有开关，0表示没有

    # AC线路
    for row in eachrow(line_df)
        if (row.FromBus, row.ToBus) in switch_lines
            push!(switch_flag, 1)
        else
            push!(switch_flag, 0)
        end
    end

    # DC线路
    for row in eachrow(dcline_df)
        if (row.FromBus, row.ToBus) in switch_lines
            push!(switch_flag, 1)
        else
            push!(switch_flag, 0)
        end
    end

    # VSC全部视为有开关
    for row in eachrow(converter_df)
        push!(switch_flag, 1)
    end

    # 转为1×(nl+nl_vsc)矩阵
    switch_flag_mat = reshape(switch_flag, 1, :)

    # println("开关状态: ", switch_flag_mat)

    Smax = fill(4000, nrow(line_df) + nrow(dcline_df))  # 线路容量
    Smax[27] = 200
    Smax[28] = 20000

    jpc = Dict(
        :Cft_ac => Cft_ac,
        :Cft_dc => Cft_dc,
        :Cft_vsc => Cft_vsc,
        :Cg => Cg,
        :Cd => Cd,
        :Cdf => Cdf,
        :Cmg => Cmg,
        :Pd => Pd,
        :Qd => Qd,
        :Fd => Fd,
        :Pgmax => Pgmax,                   # 发电机最大出力
        :Qgmax => Qgmax,                   # 发电机最大无功出力
        :Pmgmax => Pmgmax,                 # 微网最大有功
        :Pmgmin => Pmgmin,                 # 微网最小有功
        :Qmgmax => Qmgmax,                 # 微网最大无功
        :Qmgmin => Qmgmin,                 # 微网最小无功
        :Fmgmax => Fmgmax,                 # 微网虚拟功率
        :Pvscmax => Pvscmax,               # 换流器最大有功
        :R => R,                           # 线路电阻
        :X => X,                           # 线路电抗
        :Smax => Smax,                     # 线路容量
        :α_pre => α,                       # 线路状态
        :r => switch_flag_mat,             # 线路开关状态
        :VMAX => VMAX,                     # 电压上限
        :VMIN => VMIN,                     # 电压下限
        :gen => Cg,                        # 发电机数据
        :GEN_BUS => GEN_BUS,               # 发电机连接节点
        :mg_buses => mg_buses,             # 微网节点
        :η => η,                           # 效率系数
        :bigM => 1000,                     # 大数约束
        :c_vsc => 1e6,                     # VSC权重
        :c_load => 50,                     # 负荷权重
        :c_sg => 5,                        # 发电权重
        :τ_reclosing => 1/3600,            # 断路器重合闸时间(小时)
        :τ_switch => 1/60,                 # 故障时间(小时)
        :τ_tripping => 1/30,               # 切除时间(小时)
        :τ_repair => 1,                    # 恢复时间(小时).
        :NC => fill(1, nrow(load_df)),     # 每个节点的客户数（示例值）
        :λ_line => fill(0.1, nrow(line_df) + nrow(dcline_df) + nrow(converter_df)),
        # 每条线路的故障率（示例值）
        :ω => 10.0,                         # EENS的单位成本（$/kWh）
        :nb_ac => length(bus_ac),  # 交流节点数
        :nb_dc => length(bus_dc),  # 直流节点数
        :nb => num_nodes,  # 总节点数
        :ng => nrow(gen_df),  # 发电机数
        :nl_ac => nrow(line_df),  # 交流线路数
        :nl_dc => nrow(dcline_df),  # 直流线路数
        :nl => nrow(line_df) + nrow(dcline_df),  # 总线路数
        :nl_vsc => nrow(converter_df),  # VSC 数量
        :nmg => nrow(nmg_df),  # 假设微网数量为 1（根据实际情况调整）
        :nd => nrow(load_df)  # 负荷数量
    )

    return jpc
end