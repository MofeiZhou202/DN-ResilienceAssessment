"""
    convert_jpc_to_function(jpc::JPC, case_name::String="case_converted")

将JPC结构体转换为返回字典的函数格式。
输出的函数将包含所有非空矩阵数据，并附带详细注释。

参数:
- jpc: JPC结构体实例
- case_name: 输出函数的名称，默认为"case_converted"

返回:
- 字符串形式的Julia函数代码
"""
function convert_jpc_to_function(jpc, case_name::String="case_converted")
    # 开始构建输出函数
    output = """
    function $(case_name)()
        # 创建电力系统案例字典
        jpc = Dict{String, Any}();
        
        # 基本信息
        jpc["version"] = "$(jpc.version)";
        jpc["baseMVA"] = $(jpc.baseMVA);
        jpc["success"] = $(jpc.success);
        jpc["iterationsAC"] = $(jpc.iterationsAC);
        jpc["iterationsDC"] = $(jpc.iterationsDC);
        
    """
    
    # 添加AC网络组件
    if size(jpc.busAC, 1) > 0
        output *= """
        # AC Bus Data
        # 列含义: [BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN, ...]
        # BUS_I: 母线编号
        # BUS_TYPE: 母线类型 (1-PQ, 2-PV, 3-参考节点, 4-孤立节点)
        # PD, QD: 有功/无功负荷 (MW, MVAr)
        # VM, VA: 电压幅值 (p.u.) 和相角 (度)
        jpc["busAC"] = [
            $(format_matrix(jpc.busAC))
        ];
        
        """
    else
        output *= """
        # AC Bus Data - 空矩阵
        jpc["busAC"] = Array{Float64}(undef, 0, $(size(jpc.busAC, 2)));
        
        """
    end
    
    if size(jpc.genAC, 1) > 0
        output *= """
        # AC Generator Data
        # 列含义: [GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...]
        # GEN_BUS: 发电机所在母线编号
        # PG, QG: 有功/无功出力 (MW, MVAr)
        # GEN_STATUS: 1-在运行, 0-停运
        jpc["genAC"] = [
            $(format_matrix(jpc.genAC))
        ];
        
        """
    else
        output *= """
        # AC Generator Data - 空矩阵
        jpc["genAC"] = Array{Float64}(undef, 0, $(size(jpc.genAC, 2)));
        
        """
    end
    
    if size(jpc.branchAC, 1) > 0
        output *= """
        # AC Branch Data
        # 列含义: [F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ...]
        # F_BUS, T_BUS: 支路起点和终点母线编号
        # BR_R, BR_X: 支路电阻和电抗 (p.u.)
        # BR_STATUS: 1-投入运行, 0-退出运行
        jpc["branchAC"] = [
            $(format_matrix(jpc.branchAC))
        ];
        
        """
    else
        output *= """
        # AC Branch Data - 空矩阵
        jpc["branchAC"] = Array{Float64}(undef, 0, $(size(jpc.branchAC, 2)));
        
        """
    end
    
    if size(jpc.loadAC, 1) > 0
        output *= """
        # AC Load Data
        # 列含义: [LOAD_I, LOAD_CND, LOAD_STATUS, LOAD_PD, LOAD_QD, LOADZ_PERCENT, LOADI_PERCENT, LOADP_PERCENT]
        # LOAD_I: 负荷编号
        # LOAD_CND: 负荷所在母线编号
        # LOAD_STATUS: 负荷状态 (1-投入, 0-退出)
        # LOAD_PD, LOAD_QD: 有功/无功负荷 (MW, MVAr)
        jpc["loadAC"] = [
            $(format_matrix(jpc.loadAC))
        ];
        
        """
    else
        output *= """
        # AC Load Data - 空矩阵
        jpc["loadAC"] = Array{Float64}(undef, 0, $(size(jpc.loadAC, 2)));
        
        """
    end
    
    # 添加灵活负荷数据
    if size(jpc.loadAC_flex, 1) > 0
        output *= """
        # Flexible AC Load Data
        # 包含灵活负荷的详细参数
        jpc["loadAC_flex"] = [
            $(format_matrix(jpc.loadAC_flex))
        ];
        
        """
    else
        output *= """
        # Flexible AC Load Data - 空矩阵
        jpc["loadAC_flex"] = Array{Float64}(undef, 0, $(size(jpc.loadAC_flex, 2)));
        
        """
    end
    
    # 添加非对称负荷数据
    if size(jpc.loadAC_asymm, 1) > 0
        output *= """
        # Asymmetric AC Load Data
        # 包含非对称负荷的详细参数
        jpc["loadAC_asymm"] = [
            $(format_matrix(jpc.loadAC_asymm))
        ];
        
        """
    else
        output *= """
        # Asymmetric AC Load Data - 空矩阵
        jpc["loadAC_asymm"] = Array{Float64}(undef, 0, $(size(jpc.loadAC_asymm, 2)));
        
        """
    end
    
    # 添加三相支路数据
    if size(jpc.branch3ph, 1) > 0
        output *= """
        # Three-Phase Branch Data
        # 包含三相支路的详细参数
        jpc["branch3ph"] = [
            $(format_matrix(jpc.branch3ph))
        ];
        
        """
    else
        output *= """
        # Three-Phase Branch Data - 空矩阵
        jpc["branch3ph"] = Array{Float64}(undef, 0, $(size(jpc.branch3ph, 2)));
        
        """
    end
    
    # 添加DC网络组件
    if size(jpc.busDC, 1) > 0
        output *= """
        # DC Bus Data
        # 列含义类似于AC母线，但适用于DC网络
        # BUS_TYPE: 1-P节点, 2-参考节点, 3-孤立节点
        jpc["busDC"] = [
            $(format_matrix(jpc.busDC))
        ];
        
        """
    else
        output *= """
        # DC Bus Data - 空矩阵
        jpc["busDC"] = Array{Float64}(undef, 0, $(size(jpc.busDC, 2)));
        
        """
    end
    
    if size(jpc.branchDC, 1) > 0
        output *= """
        # DC Branch Data
        # 列含义类似于AC支路，但适用于DC网络
        jpc["branchDC"] = [
            $(format_matrix(jpc.branchDC))
        ];
        
        """
    else
        output *= """
        # DC Branch Data - 空矩阵
        jpc["branchDC"] = Array{Float64}(undef, 0, $(size(jpc.branchDC, 2)));
        
        """
    end
    
    if size(jpc.genDC, 1) > 0
        output *= """
        # DC Generator Data
        # 列含义类似于AC发电机，但适用于DC网络
        jpc["genDC"] = [
            $(format_matrix(jpc.genDC))
        ];
        
        """
    else
        output *= """
        # DC Generator Data - 空矩阵
        jpc["genDC"] = Array{Float64}(undef, 0, $(size(jpc.genDC, 2)));
        
        """
    end
    
    if size(jpc.loadDC, 1) > 0
        output *= """
        # DC Load Data
        # 列含义类似于AC负荷，但适用于DC网络
        jpc["loadDC"] = [
            $(format_matrix(jpc.loadDC))
        ];
        
        """
    else
        output *= """
        # DC Load Data - 空矩阵
        jpc["loadDC"] = Array{Float64}(undef, 0, $(size(jpc.loadDC, 2)));
        
        """
    end
    
    # 添加分布式能源
    if size(jpc.sgenAC, 1) > 0
        output *= """
        # AC Solar/PV Generation Data
        # 包含AC连接的太阳能/光伏发电数据
        jpc["sgenAC"] = [
            $(format_matrix(jpc.sgenAC))
        ];
        
        """
    else
        output *= """
        # AC Solar/PV Generation Data - 空矩阵
        jpc["sgenAC"] = Array{Float64}(undef, 0, $(size(jpc.sgenAC, 2)));
        
        """
    end
    
    if size(jpc.storageetap, 1) > 0
        output *= """
        # ETAP Energy Storage System Data
        # 包含ETAP格式的储能系统数据
        jpc["storageetap"] = [
            $(format_matrix(jpc.storageetap))
        ];
        
        """
    else
        output *= """
        # ETAP Energy Storage System Data - 空矩阵
        jpc["storageetap"] = Array{Float64}(undef, 0, $(size(jpc.storageetap, 2)));
        
        """
    end
    
    if size(jpc.storage, 1) > 0
        output *= """
        # Energy Storage System Data
        # 列含义: [ESS_BUS, ESS_POWER_CAPACITY, ESS_ENERGY_CAPACITY, ESS_SOC_INIT, ESS_SOC_MIN, ESS_SOC_MAX, ESS_EFFICIENCY, ESS_STATUS]
        # ESS_BUS: 储能系统所在母线编号
        # ESS_POWER_CAPACITY: 功率容量 (MW)
        # ESS_ENERGY_CAPACITY: 能量容量 (MWh)
        # ESS_SOC_INIT/MIN/MAX: 初始/最小/最大荷电状态
        # ESS_STATUS: 1-投入运行, 0-退出运行
        jpc["storage"] = [
            $(format_matrix(jpc.storage))
        ];
        
        """
    else
        output *= """
        # Energy Storage System Data - 空矩阵
        jpc["storage"] = Array{Float64}(undef, 0, $(size(jpc.storage, 2)));
        
        """
    end
    
    if size(jpc.sgenDC, 1) > 0
        output *= """
        # DC-connected PV Data
        # 包含DC连接的光伏发电数据
        jpc["sgenDC"] = [
            $(format_matrix(jpc.sgenDC))
        ];
        
        """
    else
        output *= """
        # DC-connected PV Data - 空矩阵
        jpc["sgenDC"] = Array{Float64}(undef, 0, $(size(jpc.sgenDC, 2)));
        
        """
    end
    
    if size(jpc.pv, 1) > 0
        output *= """
        # PV Array Data
        # 列含义: [PV_ID, PV_BUS, PV_VOC, PV_VMPP, PV_ISC, PV_IMPP, PV_IRRADIANCE, PV_AREA, PV_IN_SERVICE]
        # PV_ID: 光伏阵列ID
        # PV_BUS: 光伏阵列所在母线编号
        # PV_VOC, PV_VMPP: 开路电压和最大功率点电压
        # PV_ISC, PV_IMPP: 短路电流和最大功率点电流
        # PV_IRRADIANCE: 光照强度 (W/m²)
        # PV_IN_SERVICE: 1-投入运行, 0-退出运行
        jpc["pv"] = [
            $(format_matrix(jpc.pv))
        ];
        
        """
    else
        output *= """
        # PV Array Data - 空矩阵
        jpc["pv"] = Array{Float64}(undef, 0, $(size(jpc.pv, 2)));
        
        """
    end
    
    if size(jpc.pv_acsystem, 1) > 0
        output *= """
        # AC PV System Data
        # 包含带逆变器的AC光伏系统数据
        # 列含义包括光伏参数和逆变器参数
        jpc["pv_acsystem"] = [
            $(format_matrix(jpc.pv_acsystem))
        ];
        
        """
    else
        output *= """
        # AC PV System Data - 空矩阵
        jpc["pv_acsystem"] = Array{Float64}(undef, 0, $(size(jpc.pv_acsystem, 2)));
        
        """
    end
    
    # 添加特殊组件
    if size(jpc.converter, 1) > 0
        output *= """
        # AC/DC Converter Data
        # 列含义: [CONV_ACBUS, CONV_DCBUS, CONV_INSERVICE, CONV_P_AC, CONV_Q_AC, CONV_P_DC, CONV_EFF, CONV_MODE, CONV_DROOP_KP]
        # CONV_ACBUS, CONV_DCBUS: 变流器AC侧和DC侧母线编号
        # CONV_INSERVICE: 1-投入运行, 0-退出运行
        # CONV_P_AC, CONV_Q_AC: AC侧有功/无功功率 (MW, MVAr)
        # CONV_P_DC: DC侧功率 (MW)
        # CONV_EFF: 变流器效率
        # CONV_MODE: 控制模式 (1-7不同控制策略)
        jpc["converter"] = [
            $(format_matrix(jpc.converter))
        ];
        
        """
    else
        output *= """
        # AC/DC Converter Data - 空矩阵
        jpc["converter"] = Array{Float64}(undef, 0, $(size(jpc.converter, 2)));
        
        """
    end
    
    if size(jpc.ext_grid, 1) > 0
        output *= """
        # External Grid Data
        # 包含外部电网的详细参数
        jpc["ext_grid"] = [
            $(format_matrix(jpc.ext_grid))
        ];
        
        """
    else
        output *= """
        # External Grid Data - 空矩阵
        jpc["ext_grid"] = Array{Float64}(undef, 0, $(size(jpc.ext_grid, 2)));
        
        """
    end
    
    if size(jpc.hvcb, 1) > 0
        output *= """
        # High Voltage Circuit Breaker Data
        # 列含义: [HVCB_ID, HVCB_FROM_ELEMENT, HVCB_TO_ELEMENT, HVCB_INSERVICE, HVCB_STATUS]
        # HVCB_ID: 断路器ID
        # HVCB_FROM_ELEMENT, HVCB_TO_ELEMENT: 断路器连接的元件编号
        # HVCB_INSERVICE: 1-可用, 0-不可用
        # HVCB_STATUS: 1-闭合, 0-断开
        jpc["hvcb"] = [
            $(format_matrix(jpc.hvcb))
        ];
        
        """
    else
        output *= """
        # High Voltage Circuit Breaker Data - 空矩阵
        jpc["hvcb"] = Array{Float64}(undef, 0, $(size(jpc.hvcb, 2)));
        
        """
    end
    
    if size(jpc.microgrid, 1) > 0
        output *= """
        # Microgrid Data
        # 列含义: [MG_ID, MG_CAPACITY, MG_PEAK_LOAD, MG_DURATION, MG_AREA]
        # MG_ID: 微电网ID
        # MG_CAPACITY: 微电网容量 (MW)
        # MG_PEAK_LOAD: 微电网峰值负荷 (MW)
        # MG_DURATION: 微电网持续时间 (小时)
        # MG_AREA: 微电网所属区域
        jpc["microgrid"] = [
            $(format_matrix(jpc.microgrid))
        ];
        
        """
    else
        output *= """
        # Microgrid Data - 空矩阵
        jpc["microgrid"] = Array{Float64}(undef, 0, $(size(jpc.microgrid, 2)));
        
        """
    end
    
    # 结束函数
    output *= """
        return jpc
    end
    """
    
    return output
end

"""
    format_matrix(matrix::Array{Float64,2})

格式化二维浮点数矩阵为易读的字符串表示。
每行一个数组元素，保持适当的缩进。

参数:
- matrix: 要格式化的二维浮点数矩阵

返回:
- 格式化后的矩阵字符串表示
"""
function format_matrix(matrix::Array{Float64,2})
    if size(matrix, 1) == 0
        return ""
    end
    
    rows = []
    for i in 1:size(matrix, 1)
        row_str = join([@sprintf("%.6g", matrix[i,j]) for j in 1:size(matrix, 2)], ", ")
        push!(rows, "        " * row_str)
    end
    
    return join(rows, ";\n")
end

# example usage
# 假设我们有一个JPC对象
# jpc = JPC("2.0", 100.0, true, 5, 0)
# # 添加一些数据到jpc...

# # 转换为函数
# function_str = convert_jpc_to_function(jpc, "case_14_bus")

# # 将函数写入文件
# open("case_14_bus.jl", "w") do f
#     write(f, function_str)
# end

# # 或者从字符串解析JPC对象
# jpc_str = "JPC(\"2.0\", 100.0, true, 5, 0, ...)"
# jpc = parse_jpc_from_string(jpc_str)
