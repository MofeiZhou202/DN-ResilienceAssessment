using SparseArrays

"""
   Definition of the Microgrid Planning Problem structure.
"""
mutable struct MicrogridPlanningProblem
    # 常数
    bigM::Float64
    
    # 规划参数
    nPV::Int
    nBESS::Int
    DeltaPV::Float64
    DeltaBESS::Float64
    
    # 技术参数
    soc0::Float64
    soc_max::Float64
    soc_min::Float64
    eta_ug::Float64
    eff_ch::Float64
    eff_dc::Float64
    
    # 经济参数
    pv_unit_cost::Float64
    ess_unit_cost_e::Float64
    ess_unit_cost_p::Float64
    ug_unit_cost::Float64
    carbon_permit_price::Float64
    carbon_emission_coeff::Float64
    npv::Float64

    # 可靠性参数
    ug_prob::Float64
    mg_prob::Float64
    power_requirements::Float64
    energy_requirements::Float64
    pv_power_factor::Float64
    pv_energy_factor::Float64
    
    # 碳排放限制
    carbon_limit::Float64
    # 不确定性参数
    ns::Int
    ws::Vector{Float64}
    T::Int
    ppv_un::Int
    pl::Int
    fl::Int
    ug_status::Int
    ele_price::Int
    carbon_price::Int
    carbon_emission_ug::Int
    nu::Int
    NU::Int
    
    # 索引
    IPV::Int
    IESS::Int
    IPESS::Int
    PUG::Int
    CARBON_PERMIT::Int
    CARBON::Int
    NX::Int
    
    # 第一阶段约束
    lx::Vector{Float64}
    ux::Vector{Float64}
    vtypex::Vector{Char}
    Aeq::SparseMatrixCSC{Float64, Int}
    beq::Vector{Float64}
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    
    # 目标函数
    c::Vector{Float64}

    # 第二阶段索引
    pug::Int
    pug_sold::Int
    ppv::Int
    pess_ch::Int
    pess_dc::Int
    eess::Int
    iess_ch::Int
    iess_dc::Int
    wpv::Int
    wpess_dc::Int
    wpess_ch::Int
    plc::Int
    pf::Int
    ny::Int
    NY::Int
    # 构造函数
    function MicrogridPlanningProblem(data::Dict{Symbol, Any})
        # 创建实例
        self = new()
    
        # 初始化常数
        self.bigM = 1e6

        # 读取规划参数
        planning_params = data[:planning_params]
        self.nPV = planning_params[1, "光伏单元数"]
        self.nBESS = planning_params[1, "储能单元数"]
        self.DeltaPV = planning_params[1, "光伏单元容量"]
        self.DeltaBESS = planning_params[1, "储能单元容量"]
    
        # 读取技术参数
        tech_params = data[:tech_params]
        self.soc0 = tech_params[1, "储能单元初始SOC"]
        self.soc_max = tech_params[1, "储能单元最大SOC"]
        self.soc_min = tech_params[1, "储能单元最小SOC"]
        self.eta_ug = tech_params[1, "电网效率"]
        self.eff_ch = tech_params[1, "储能单元充电效率"]
        self.eff_dc = tech_params[1, "储能单元放电效率"]
    
        # 读取经济参数
        econ_params = data[:econ_params]
        self.pv_unit_cost = econ_params[1, "光伏单元成本(元/kW)"]
        self.ess_unit_cost_e = econ_params[1, "储能单元能量部分成本(元/kWh)"]
        self.ess_unit_cost_p = econ_params[1, "储能单元功率部分成本(元/kW)"]
        self.ug_unit_cost = econ_params[1, "容量成本(元/年/kW)"]
        self.carbon_permit_price = econ_params[1, "碳排放权单价(元/kWh)"]
        self.carbon_emission_coeff = econ_params[1, "碳排放系数（kg/kWh）"]
        self.npv = capital_recovery_factor(econ_params[1, "贴现率"], econ_params[1, "总规划年数"])

        # 读取可靠性参数
        reliability_params = data[:reliability_params]
        self.ug_prob = reliability_params[1, "供电可靠率"]
        self.mg_prob = reliability_params[1, "微网可靠率"]
    
        # 定义不确定性参数（这部分需要从场景生成函数获取）
        # 暂时使用占位符
        self.ns = 0
        self.ws = Float64[]
        self.T = planning_params[1, "时段数"]
        self.ppv_un = 0  # PV output in per unit
        self.pl = self.ppv_un + 1  # Active load consumption
        self.fl = self.pl + 1  # Flexible load consumption
        self.ug_status = self.fl + 1  # Utility grid operating status
        self.ele_price = self.ug_status + 1  # Electricity price
        self.carbon_price = self.ele_price + 1  # Carbon price
        self.carbon_emission_ug = self.carbon_price + 1  # Carbon price
        self.nu = self.carbon_emission_ug + 1 # Uncertainties of loads
        self.NU = self.nu * self.T  # Total uncertainties
    
        # 返回实例
        return self
    end
end

"""
    Definition of the power flow problem structure.
    Aligned with JPC structure for consistency.
"""
mutable struct JuliaPowerCase
    version::String
    baseMVA::Float32
    basef::Float32
    
    # AC Network Components
    busesAC::Vector{Bus}                # Matches busAC in JPC
    branchesAC::Vector{Line}            # Matches branchAC in JPC
    loadsAC::Vector{Load}               # Matches loadAC in JPC
    loadsAC_flex::Vector{FlexLoad}      # Matches loadAC_flex in JPC
    loadsAC_asymm::Vector{AsymmetricLoad}  # Matches loadAC_asymm in JPC
    # branches3ph::Vector{ThreePhaseBranch}  # Matches branch3ph in JPC
    gensAC::Vector{Generator}            # Matches genAC in JPC
    
    # DC Network Components
    busesDC::Vector{BusDC}              # Matches busDC in JPC
    branchesDC::Vector{LineDC}          # Matches branchDC in JPC
    loadsDC::Vector{LoadDC}          # Matches loadDC in JPC
    
    # Distributed Energy Resources
    sgensAC::Vector{StaticGenerator}     # Matches sgenAC in JPC
    storages::Vector{Storage}            # Matches storage in JPC
    storageetap::Vector{Storageetap}      # Matches storageetap in JPC
    sgensDC::Vector{StaticGeneratorDC}   # Matches sgenDC in JPC
    pvarray::Vector{PVArray}          # Matches pv_array in JPC
    ACPVSystems::Vector{ACPVSystem}  # Matches AC PV systems in JPC
    
    # Special Components
    energyrouter::Vector{EnergyRouterCore}  # Matches converter in JPC
    converters::Vector{Converter}        # Matches converter in JPC
    ext_grids::Vector{ExternalGrid}      # Matches ext_grid in JPC
    hvcbs::Vector{HighVoltageCircuitBreaker}  # Matches hvcb in JPC
    microgrids::Vector{Microgrid}        # Matches microgrid in JPC
    
    # Original components without direct JPC equivalents (preserved)
    transformers_2w::Vector{Transformer2W}
    transformers_3w::Vector{Transformer3W}
    transformers_2w_etap::Vector{Transformer2Wetap}
    charging_stations::Vector{ChargingStation}
    chargers::Vector{Charger}
    ev_aggregators::Vector{EVAggregator}
    virtual_power_plants::Vector{VirtualPowerPlant}
    carbon_time_series::Vector{CarbonTimeSeries}
    equipment_carbon::Vector{EquipmentCarbon}
    
    # Lookup dictionaries
    bus_name_to_id::Dict{String, Int}
    busdc_name_to_id::Dict{String, Int}
    zone_to_id::Dict{String, Int}
    area_to_id::Dict{String, Int}
    
    # Constructor
    function JuliaPowerCase(version::String = "2.0", baseMVA::Float32 = 100.0f0, basef::Float32 = 50.0f0)
        return new(
            version, baseMVA, basef,
            # AC Network Components
            Bus[], Line[], Load[], FlexLoad[], AsymmetricLoad[], Generator[],EnergyRouterCore[],
            # DC Network Components
            BusDC[], LineDC[],LoadDC[],
            # Distributed Energy Resources
            StaticGenerator[], Storage[], Storageetap[], StaticGeneratorDC[], PVArray[], ACPVSystem[],
            # Special Components
            Converter[], ExternalGrid[], HighVoltageCircuitBreaker[], Microgrid[],
            # Original components without direct JPC equivalents
            Transformer2W[], Transformer3W[],Transformer2Wetap[], ChargingStation[], Charger[], 
            EVAggregator[], VirtualPowerPlant[], CarbonTimeSeries[], EquipmentCarbon[],
            # Lookup dictionaries
            Dict{String, Int}(), Dict{String, Int}(), Dict{String, Int}(), Dict{String, Int}(),
        )
    end
end

"""
    Definition of the JPC structure as one matrix format.
    This structure represents a power system case in a one matrix format,
    including buses, generators, branches, loads, and other power system components.
    Each matrix is sized according to the number of attributes defined in idx.jl.
"""
mutable struct JPC
    version::String
    baseMVA::Float64
    success::Bool
    iterationsAC::Int
    iterationsDC::Int
    
    # AC Network Components
    busAC::Array{Float64,2}         # Bus data (indexed by idx_bus(), dimensions: n_bus × N_BUS_ATTR)
    genAC::Array{Float64,2}         # Generator data (indexed by idx_gen(), dimensions: n_gen × N_GEN_ATTR)
    branchAC::Array{Float64,2}      # Branch data (indexed by idx_brch(), dimensions: n_branch × N_BRANCH_ATTR)
    loadAC::Array{Float64,2}        # Load data (indexed by idx_ld(), dimensions: n_load × N_LOAD_ATTR)
    loadAC_flex::Array{Float64,2}   # Flexible load data (indexed by idx_ld_flex(), dimensions: n_flex_load × N_FLEX_LOAD_ATTR)
    loadAC_asymm::Array{Float64,2}  # Asymmetric load data (indexed by idx_ld_asymmetric(), dimensions: n_asym_load × N_ASYM_LOAD_ATTR)
    branch3ph::Array{Float64,2}     # Three-phase branch data (indexed by idx_3ph_brch(), dimensions: n_3ph_branch × N_3PH_BRANCH_ATTR)
    
    # DC Network Components
    busDC::Array{Float64,2}         # DC bus data (indexed by idx_dcbus(), dimensions: n_dcbus × N_DCBUS_ATTR)
    branchDC::Array{Float64,2}      # DC branch data (indexed by idx_brch(), dimensions: n_dc_branch × N_BRANCH_ATTR)
    genDC::Array{Float64,2}         # DC generator data (indexed by idx_gen(), dimensions: n_dc_gen × N_GEN_ATTR)
    loadDC::Array{Float64,2}        # DC load data (indexed by idx_ld(), dimensions: n_dc_load × N_LOAD_ATTR)
    
    # Distributed Energy Resources
    sgenAC::Array{Float64,2}        # Solar/PV generation data (indexed by idx_pv(), dimensions: n_pv × N_PV_ATTR)
    storageetap::Array{Float64,2}  # Energy storage system data (indexed by idx_storageetap(), dimensions: n_storageetap × N_STORAGEETAP_ATTR)
    storage::Array{Float64,2}       # Energy storage system data (indexed by idx_ess(), dimensions: n_ess × N_ESS_ATTR)
    sgenDC::Array{Float64,2}        # DC-connected PV data (indexed by idx_pv(), dimensions: n_dc_pv × N_PV_ATTR)
    pv::Array{Float64,2}          # PV array data (indexed by idx_pv_array(), dimensions: n_pv_array × N_PV_ARRAY_ATTR)
    pv_acsystem::Array{Float64,2}  # AC PV systems data (indexed by idx_ac_pv_system(), dimensions: n_ac_pv_system × N_AC_PV_SYSTEM_ATTR)
    
    # Special Components
    converter::Array{Float64,2}     # AC/DC converter data (indexed by idx_conv(), dimensions: n_conv × N_CONV_ATTR)
    energyrouterCore::Array{Float64,2}  # Energy router core data (indexed by idx_energyrouter(), dimensions: n_energyrouter × N_ENERGYROUTER_ATTR)
    energyrouterConverter::Array{Float64,2}  # Energy router converter data (indexed by idx_energyrouter_converter(), dimensions: n_energyrouter_converter × N_ENERGYROUTER_CONVERTER_ATTR)
    ext_grid::Array{Float64,2}      # External grid data (indexed by idx_ext_grid(), dimensions: n_ext_grid × N_EXT_GRID_ATTR)
    hvcb::Array{Float64,2}          # High voltage circuit breaker data (indexed by idx_hvcb(), dimensions: n_hvcb × N_HVCB_ATTR)
    microgrid::Array{Float64,2}     # Microgrid data (indexed by idx_microgrid(), dimensions: n_mg × N_MG_ATTR)
    
    # Constructor with default empty arrays
    function JPC(version::String = "2.0", baseMVA::Float64 = 100.0, success::Bool = false, iterationsAC::Int = 0, iterationsDC::Int = 0)
        # Initialize with empty arrays - they will be properly sized when data is loaded
        new(version, baseMVA, success, iterationsAC, iterationsDC,
            Array{Float64}(undef, 0, 22),  # busAC (N_BUS_ATTR = 22)
            Array{Float64}(undef, 0, 32),  # genAC (N_GEN_ATTR = 32)
            Array{Float64}(undef, 0, 28),  # branchAC (N_BRANCH_ATTR = 28)
            Array{Float64}(undef, 0, 8),   # loadAC (N_LOAD_ATTR = 8)
            Array{Float64}(undef, 0, 25),  # loadAC_flex (N_FLEX_LOAD_ATTR = 25)
            Array{Float64}(undef, 0, 12),  # loadAC_asymm (N_ASYM_LOAD_ATTR = 12)
            Array{Float64}(undef, 0, 30),  # branch3ph (N_3PH_BRANCH_ATTR = 30)
            Array{Float64}(undef, 0, 21),  # busDC (N_DCBUS_ATTR = 21)
            Array{Float64}(undef, 0, 28),  # branchDC (uses same structure as branchAC)
            Array{Float64}(undef, 0, 32),  # genDC (N_GEN_ATTR = 32)
            Array{Float64}(undef, 0, 8),   # loadDC (N_LOAD_ATTR = 8)
            Array{Float64}(undef, 0, 3),   # sgenAC (N_PV_ATTR = 3)
            Array{Float64}(undef, 0, 15),  # storageetap (N_STORAGEETAP_ATTR = 15)
            Array{Float64}(undef, 0, 15),  # storage (N_ESS_ATTR = 15)
            Array{Float64}(undef, 0, 3),   # sgenDC (uses same structure as sgenAC)
            Array{Float64}(undef, 0, 9),   # pv (N_PV_ARRAY_ATTR = 10)
            Array{Float64}(undef, 0, 15),  # pv_acsystem (N_AC_PV_SYSTEM_ATTR = 18)
            Array{Float64}(undef, 0, 18),  # converter (N_CONV_ATTR = 18)
            Array{Float64}(undef, 0, 18),  # energyrouterCore (N_ENERGYROUTER_ATTR = 18)
            Array{Float64}(undef, 0, 18),  # energyrouterConverter (N_ENERGYROUTER_CONVERTER_ATTR = 18)
            Array{Float64}(undef, 0, 13),  # ext_grid (N_EXT_GRID_ATTR = 13)
            Array{Float64}(undef, 0, 5),   # hvcb (N_HVCB_ATTR = 5)
            Array{Float64}(undef, 0, 5)    # microgrid (N_MG_ATTR = 5)
        )
    end
end

import Base: getindex, setindex!

function getindex(jpc::JPC, key::String)
    # 根据字符串键返回对应的字段
    if key == "version"
        return jpc.version
    elseif key == "baseMVA"
        return jpc.baseMVA
    elseif key == "success"
        return jpc.success
    elseif key == "iterationsAC"
        return jpc.iterationsAC
    elseif key == "iterationsDC"
        return jpc.iterationsDC
    elseif key == "busAC"
        return jpc.busAC
    elseif key == "genAC"
        return jpc.genAC
    elseif key == "branchAC"
        return jpc.branchAC
    elseif key == "loadAC"
        return jpc.loadAC
    elseif key == "loadAC_flex"
        return jpc.loadAC_flex
    elseif key == "loadAC_asymm"
        return jpc.loadAC_asymm
    elseif key == "branch3ph"
        return jpc.branch3ph
    elseif key == "busDC"
        return jpc.busDC
    elseif key == "branchDC"
        return jpc.branchDC
    elseif key == "genDC"
        return jpc.genDC
    elseif key == "loadDC"
        return jpc.loadDC
    elseif key == "α_pre" || key == "alpha_pre"
        return jpc.α_pre
    elseif key == "nl_ac"
        return jpc.nl_ac
    elseif key == "nl_dc"
        return jpc.nl_dc
    elseif key == "nl_vsc"
        return jpc.nl_vsc
    elseif key == "sgenAC"
        return jpc.sgenAC
    elseif key == "storageetap"
        return jpc.storageetap
    elseif key == "storage"
        return jpc.storage
    elseif key == "sgenDC"
        return jpc.sgenDC
    elseif key == "pv"
        return jpc.pv
    elseif key == "pv_acsystem"
        return jpc.pv_acsystem
    elseif key == "converter"
        return jpc.converter
    elseif key == "ext_grid"
        return jpc.ext_grid
    elseif key == "hvcb"
        return jpc.hvcb
    elseif key == "microgrid"
        return jpc.microgrid
    else
        error("JPC 结构体中不存在键: $key")
    end
end

function setindex!(jpc::JPC, value, key::String)
    # 根据字符串键设置对应的字段
    if key == "version"
        jpc.version = value
    elseif key == "baseMVA"
        jpc.baseMVA = value
    elseif key == "success"
        jpc.success = value
    elseif key == "iterationsAC"
        jpc.iterationsAC = value
    elseif key == "iterationsDC"
        jpc.iterationsDC = value
    elseif key == "busAC"
        jpc.busAC = value
    elseif key == "genAC"
        jpc.genAC = value
    elseif key == "branchAC"
        jpc.branchAC = value
    elseif key == "loadAC"
        jpc.loadAC = value
    elseif key == "loadAC_flex"
        jpc.loadAC_flex = value
    elseif key == "loadAC_asymm"
        jpc.loadAC_asymm = value
    elseif key == "branch3ph"
        jpc.branch3ph = value
    elseif key == "busDC"
        jpc.busDC = value
    elseif key == "branchDC"
        jpc.branchDC = value
    elseif key == "genDC"
        jpc.genDC = value
    elseif key == "loadDC"
        jpc.loadDC = value
    elseif key == "sgenAC"
        jpc.sgenAC = value
    elseif key == "storageetap"
        jpc.storageetap = value
    elseif key == "storage"
        jpc.storage = value
    elseif key == "sgenDC"
        jpc.sgenDC = value
    elseif key == "pv"
        jpc.pv = value
    elseif key == "pv_acsystem"
        jpc.pv_acsystem = value
    elseif key == "converter"
        jpc.converter = value
    elseif key == "ext_grid"
        jpc.ext_grid = value
    elseif key == "hvcb"
        jpc.hvcb = value
    elseif key == "microgrid"
        jpc.microgrid = value
    else
        error("JPC 结构体中不存在键: $key")
    end
end

mutable struct JPC_3ph
    version::String
    baseMVA::Float32
    basef::Float32
    mode::String
    success::Bool  # 三项潮流是否成功
    iterations::Int  # 三项潮流迭代次数

    
    # AC Network Components - 矩阵版本的JuliaPowerCase组件
    busAC_0::Array{Float64,2}  # 对应case.bus的矩阵表示
    busAC_1::Array{Float64,2}  # 对应case.bus的矩阵表示
    busAC_2::Array{Float64,2}  # 对应case.bus的矩阵表示

    branchAC_0::Array{Float64,2}  # 对应case.line的矩阵表示
    branchAC_1::Array{Float64,2}  # 对应case.line的矩阵表示
    branchAC_2::Array{Float64,2}  # 对应case.line的矩阵表示

    loadAC_0::Array{Float64,2}  # 对应case.load的矩阵表示
    loadAC_1::Array{Float64,2}  # 对应case.load的矩阵表示
    loadAC_2::Array{Float64,2}  # 对应case.load的矩阵表示

    genAC_0::Array{Float64,2}  # 对应case.gen的矩阵表示
    genAC_1::Array{Float64,2}  # 对应case.gen的矩阵表示
    genAC_2::Array{Float64,2}  # 对应case.gen的矩阵表示

    storageAC::Array{Float64,2}  # 对应case.storage的矩阵表示
    
    # DC Network Components
    busDC::Array{Float64,2}  # 对应case.bus_dc的矩阵表示
    branchDC::Array{Float64,2}  # 对应case.line_dc的矩阵表示
    loadDC::Array{Float64,2}  # 对应case.load_dc的矩阵表示
    genDC::Array{Float64,2}  # 对应case.gen_dc的矩阵表示
    storageDC::Array{Float64,2}  # 对应case.storage_dc的矩阵表示
    
    # Special Components
    ext_grid::Array{Float64,2}  # 对应case.ext_grid的矩阵表示
    switche::Array{Float64,2}  # 对应case.switch的矩阵表示

    # three phase power flow results
    res_bus_3ph::Array{Float64,2}  # 三相潮流结果的总线数据
    res_loadsAC_3ph::Array{Float64,2}  # 三相潮流结果的负荷数据
    res_ext_grid_3ph::Array{Float64,2}  # 三相潮流结果的外部电网数据
    
    # Lookup dictionaries - 保持不变
    bus_name_to_id::Dict{String, Int}
    zone_to_id::Dict{String, Int}
    area_to_id::Dict{String, Int}
    
    # Constructor
    function JPC_3ph(version::String = "2.0", baseMVA::Float32 = 100.0f0, basef::Float32 = 50.0f0, mode::String = "etap",
                     success::Bool = false, iterations::Int = 0)
        # 初始化矩阵 - 这里使用空矩阵，实际数据加载时会填充
        return new(
            version, baseMVA, basef, mode,success, iterations,
            # 初始化空矩阵 - 列数对应原始结构的字段数
            Array{Float64}(undef, 0, 22),  # busAC_0 (N_BUS_ATTR = 22)
            Array{Float64}(undef, 0, 22),  # busAC_1 (N_BUS_ATTR = 22)
            Array{Float64}(undef, 0, 22),  # busAC_2 (N_BUS_ATTR = 22)

            Array{Float64}(undef, 0, 28),  # branchAC_0 (N_BRANCH_ATTR = 28)
            Array{Float64}(undef, 0, 28),  # branchAC_1 (N_BRANCH_ATTR = 28)
            Array{Float64}(undef, 0, 28),  # branchAC_2 (N_BRANCH_ATTR = 28)
            Array{Float64}(undef, 0, 8),   # loadAC_0 (N_LOAD_ATTR = 8)
            Array{Float64}(undef, 0, 8),   # loadAC_1 (N_LOAD_ATTR = 8)
            Array{Float64}(undef, 0, 8),   # loadAC_2 (N_LOAD_ATTR = 8)

            Array{Float64}(undef, 0, 32),  # genAC_0 (N_GEN_ATTR = 32)
            Array{Float64}(undef, 0, 32),  # genAC_1 (N_GEN_ATTR = 32)
            Array{Float64}(undef, 0, 32),  # genAC_2 (N_GEN_ATTR = 32)

            Array{Float64}(undef, 0, 15),  # storageAC (N_ESS_ATTR = 15)

            Array{Float64}(undef, 0, 21),  # busDC (N_DCBUS_ATTR = 21)
            Array{Float64}(undef, 0, 28),  # branchDC (uses same structure as branchAC)
            Array{Float64}(undef, 0, 8),   # loadDC (N_LOAD_ATTR = 8)
            Array{Float64}(undef, 0, 32),  # genDC (N_GEN_ATTR = 32)

            Array{Float64}(undef, 0, 15),  # storageDC (N_ESS_ATTR = 15)
            Array{Float64}(undef, 0, 13),  # ext_grid (N_EXT_GRID_ATTR = 13)
            Array{Float64}(undef, 0, 5),   # hvcb (N_HVCB_ATTR = 5)

            Array{Float64}(undef, 0, 15),   # res_bus_3ph (N_RES_BUS_3PH_ATTR = 8)
            Array{Float64}(undef, 0, 18),  # res_loadsAC_3ph (N_RES_LOADS_AC_3PH_ATTR = 18)
            Array{Float64}(undef, 0, 13),  # res_ext_grid_3ph (N_RES_EXT_GRID_3PH_ATTR = 13)

            Dict{String, Int}(),  # bus_name_to_id
            Dict{String, Int}(),  # zone_to_id
            Dict{String, Int}(),  # area_to_id
        )
    end
end

function getindex(jpc::JPC_3ph, key::String)
    # 根据字符串键返回对应的字段
    if key == "version"
        return jpc.version
    elseif key == "baseMVA"
        return jpc.baseMVA
    elseif key == "basef"
        return jpc.basef
    elseif key == "mode"
        return jpc.mode
    elseif key == "success"
        return jpc.success
    elseif key == "iterations"
        return jpc.iterations
    elseif key == "busAC_0"
        return jpc.busAC_0
    elseif key == "busAC_1"
        return jpc.busAC_1
    elseif key == "busAC_2"
        return jpc.busAC_2
    elseif key == "branchAC_0"
        return jpc.branchAC_0
    elseif key == "branchAC_1"
        return jpc.branchAC_1
    elseif key == "branchAC_2"
        return jpc.branchAC_2
    elseif key == "loadAC_0"
        return jpc.loadAC_0
    elseif key == "loadAC_1"
        return jpc.loadAC_1
    elseif key == "loadAC_2"
        return jpc.loadAC_2
    elseif key == "genAC_0"
        return jpc.genAC_0
    elseif key == "genAC_1"
        return jpc.genAC_1
    elseif key == "genAC_2"
        return jpc.genAC_2
    elseif key == "storageAC"
        return jpc.storageAC
    elseif key == "busDC"
        return jpc.busDC
    elseif key == "branchDC"
        return jpc.branchDC
    elseif key == "loadDC"
        return jpc.loadDC
    elseif key == "genDC"
        return jpc.genDC
    elseif key == "storageDC"
        return jpc.storageDC
    elseif key == "ext_grids"
        return jpc.ext_grids
    elseif key == "switches"
        return jpc.switches
    elseif key == "res_bus_3ph"
        return jpc.res_bus_3ph
    elseif key == "res_loadsAC_3ph"
        return jpc.res_loadsAC_3ph
    elseif key == "res_ext_grid_3ph"
        return jpc.res_ext_grid_3ph
    else
        error("JPC 3ph 结构体中不存在键: $key")
    end
end

function setindex!(jpc::JPC_3ph, value, key::String)
    # 根据字符串键设置对应的字段
    if key == "version"
        jpc.version = value
    elseif key == "baseMVA"
        jpc.baseMVA = value
    elseif key == "basef"
        jpc.basef = value
    elseif key == "mode"
        jpc.mode = value
    elseif key == "success"
        jpc.success = value
    elseif key == "iterations"
        jpc.iterations = value
    elseif key == "busAC_0"
        jpc.busAC_0 = value
    elseif key == "busAC_1"
        jpc.busAC_1 = value
    elseif key == "busAC_2"
        jpc.busAC_2 = value
    elseif key == "branchAC_0"
        jpc.branchAC_0 = value
    elseif key == "branchAC_1"
        jpc.branchAC_1 = value
    elseif key == "branchAC_2"
        jpc.branchAC_2 = value
    elseif key == "loadAC_0"
        jpc.loadAC_0 = value
    elseif key == "loadAC_1"
        jpc.loadAC_1 = value
    elseif key == "loadAC_2"
        jpc.loadAC_2 = value
    elseif key == "genAC_0"
        jpc.genAC_0 = value
    elseif key == "genAC_1"
        jpc.genAC_1 = value
    elseif key == "genAC_2"
        jpc.genAC_2 = value
    elseif key == "storageAC"
        jpc.storageAC = value
    elseif key == "busDC"
        jpc.busDC = value
    elseif key == "branchDC"
        jpc.branchDC = value
    elseif key == "loadDC"
        jpc.loadDC = value
    elseif key == "genDC"
        jpc.genDC = value
    elseif key == "storageDC"
        jpc.storageDC = value
    elseif key == "ext_grid"
        jpc.ext_grid = value
    elseif key == "switches"
        jpc.switches = value
    elseif key == "res_bus_3ph"
        jpc.res_bus_3ph = value
    elseif key == "res_loadsAC_3ph"
        jpc.res_loadsAC_3ph = value
    elseif key == "res_ext_grid_3ph"
        jpc.res_ext_grid_3ph = value
    else
        error("JPC 3ph 结构体中不存在键: $key")
    end
end

"""
    Definition of the JPC_3ph structure as one matrix format.
    This structure represents a three-phase power system case in a one matrix format,
    including buses, generators, branches, loads, and other power system components.
    Each matrix is sized according to the number of attributes defined in idx.jl.
"""


"""
    Definition of the JPC_tp structure as one matrix format.
    This structure represents a power system case in a one matrix format,
    including buses, generators, branches, loads, and other power system components.
    Each matrix is sized according to the number of attributes defined in idx.jl.
"""
mutable struct JPC_tp
    version::String
    baseMVA::Float64
    
    # AC Network Components
    busAC::Array{Float64,2}         # Bus data (indexed by idx_bus(), dimensions: n_bus × N_BUS_ATTR)
    genAC::Array{Float64,2}         # Generator data (indexed by idx_gen(), dimensions: n_gen × N_GEN_ATTR)
    branchAC::Array{Float64,2}      # Branch data (indexed by idx_brch(), dimensions: n_branch × N_BRANCH_ATTR)
    loadAC::Array{Float64,2}        # Load data (indexed by idx_ld(), dimensions: n_load × N_LOAD_ATTR)
    
    # DC Network Components
    busDC::Array{Float64,2}         # DC bus data (indexed by idx_dcbus(), dimensions: n_dcbus × N_DCBUS_ATTR)
    branchDC::Array{Float64,2}      # DC branch data (indexed by idx_brch(), dimensions: n_dc_branch × N_BRANCH_ATTR)
    genDC::Array{Float64,2}         # DC generator data (indexed by idx_gen(), dimensions: n_dc_gen × N_GEN_ATTR)
    loadDC::Array{Float64,2}        # DC load data (indexed by idx_ld(), dimensions: n_dc_load × N_LOAD_ATTR)
    
    # Topology information (for reconfiguration)
    α_pre::Vector{Float64}         # Initial line status (1 = closed, 0 = open)
    nl_ac::Int                       # Number of AC lines
    nl_dc::Int                       # Number of DC lines
    nl_vsc::Int                      # Number of VSC converters
    
    # Distributed Energy Resources
    sgenAC::Array{Float64,2}        # Solar/PV generation data (indexed by idx_pv(), dimensions: n_pv × N_PV_ATTR)
    storageetap::Array{Float64,2}  # Energy storage system data (indexed by idx_storageetap(), dimensions: n_storageetap × N_STORAGEETAP_ATTR)
    storage::Array{Float64,2}       # Energy storage system data (indexed by idx_ess(), dimensions: n_ess × N_ESS_ATTR)
    sgenDC::Array{Float64,2}        # DC-connected PV data (indexed by idx_pv(), dimensions: n_dc_pv × N_PV_ATTR)
    pv::Array{Float64,2}          # PV array data (indexed by idx_pv_array(), dimensions: n_pv_array × N_PV_ARRAY_ATTR)
    pv_acsystem::Array{Float64,2}  # AC PV systems data (indexed by idx_ac_pv_system(), dimensions: n_ac_pv_system × N_AC_PV_SYSTEM_ATTR)
    
    # Special Components
    converter::Array{Float64,2}     # AC/DC converter data (indexed by idx_conv(), dimensions: n_conv × N_CONV_ATTR)
    ext_grid::Array{Float64,2}      # External grid data (indexed by idx_ext_grid(), dimensions: n_ext_grid × N_EXT_GRID_ATTR)
    hvcb::Array{Float64,2}          # High voltage circuit breaker data (indexed by idx_hvcb(), dimensions: n_hvcb × N_HVCB_ATTR)
    microgrid::Array{Float64,2}     # Microgrid data (indexed by idx_microgrid(), dimensions: n_mg × N_MG_ATTR)
    
    # Connectivity Matrix
    Cft::Dict{Symbol, AbstractMatrix{Int64}}  # Connectivity matrix for topology

    # Constructor
    function JPC_tp(version::String = "2.0", baseMVA::Float64 = 100.0)
        # Initialize with empty arrays - they will be properly sized when data is loaded
        new(version, baseMVA,
            Array{Float64}(undef, 0, 22),  # busAC (N_BUS_ATTR = 22)
            Array{Float64}(undef, 0, 32),  # genAC (N_GEN_ATTR = 32)
            Array{Float64}(undef, 0, 28),  # branchAC (N_BRANCH_ATTR = 28)
            Array{Float64}(undef, 0, 8),   # loadAC (N_LOAD_ATTR = 8)
            Array{Float64}(undef, 0, 21),  # busDC (N_DCBUS_ATTR = 21)
            Array{Float64}(undef, 0, 28),  # branchDC (uses same structure as branchAC)
            Array{Float64}(undef, 0, 32),  # genDC (N_GEN_ATTR = 32)
            Array{Float64}(undef, 0, 8),   # loadDC (N_LOAD_ATTR = 8)
            Float64[],                      # α_pre (initial line status)
            0,                             # nl_ac (number of AC lines)
            0,                             # nl_dc (number of DC lines)
            0,                             # nl_vsc (number of VSC converters)
            Array{Float64}(undef, 0, 3),   # sgenAC (N_PV_ATTR = 3)
            Array{Float64}(undef, 0, 15),  # storageetap (N_STORAGEETAP_ATTR = 15)
            Array{Float64}(undef, 0, 15),  # storage (N_ESS_ATTR = 15)
            Array{Float64}(undef, 0, 3),   # sgenDC (uses same structure as sgenAC)
            Array{Float64}(undef, 0, 9),   # pv (N_PV_ARRAY_ATTR = 10)
            Array{Float64}(undef, 0, 15),  # pv_acsystem (N_AC_PV_SYSTEM_ATTR = 18)
            Array{Float64}(undef, 0, 18),  # converter (N_CONV_ATTR = 18)
            Array{Float64}(undef, 0, 13),  # ext_grid (N_EXT_GRID_ATTR = 13)
            Array{Float64}(undef, 0, 5),   # hvcb (N_HVCB_ATTR = 5)
            Array{Float64}(undef, 0, 5),   # microgrid (N_MG_ATTR = 5)
            Dict{Symbol, AbstractMatrix{Int64}}()  # Cft (connectivity matrix)
        )
    end

end

# 支持 Symbol 类型的键访问
function getindex(jpc::JPC_tp, key::Symbol)
    return getindex(jpc, String(key))
end

function setindex!(jpc::JPC_tp, value, key::Symbol)
    setindex!(jpc, value, String(key))
end

function getindex(jpc::JPC_tp, key::String)
    # 根据字符串键返回对应的字段
    if key == "version"
        return jpc.version
    elseif key == "baseMVA"
        return jpc.baseMVA
    elseif key == "busAC"
        return jpc.busAC
    elseif key == "genAC"
        return jpc.genAC
    elseif key == "branchAC"
        return jpc.branchAC
    elseif key == "loadAC"
        return jpc.loadAC
    elseif key == "busDC"
        return jpc.busDC
    elseif key == "branchDC"
        return jpc.branchDC
    elseif key == "genDC"
        return jpc.genDC
    elseif key == "loadDC"
        return jpc.loadDC
    elseif key == "sgenAC"
        return jpc.sgenAC
    elseif key == "storage"
        return jpc.storage
    elseif key == "sgenDC"
        return jpc.sgenDC
    elseif key == "pv"
        return jpc.pv
    elseif key == "pv_acsystem"
        return jpc.pv_acsystem
    elseif key == "converter"
        return jpc.converter
    elseif key == "ext_grid"
        return jpc.ext_grid
    elseif key == "hvcb"
        return jpc.hvcb
    elseif key == "microgrid"
        return jpc.microgrid
    elseif key == "Cft"
        return jpc.Cft
    elseif key == "α_pre" || key == "alpha_pre"
        return jpc.α_pre
    elseif key == "nl_ac"
        return jpc.nl_ac
    elseif key == "nl_dc"
        return jpc.nl_dc
    elseif key == "nl_vsc"
        return jpc.nl_vsc
    elseif key == "nl"
        return jpc.nl_ac + jpc.nl_dc  # 总线路数 = AC线路数 + DC线路数
    elseif key == "nb_ac"
        return size(jpc.busAC, 1)
    elseif key == "nb_dc"
        return size(jpc.busDC, 1)
    elseif key == "nb"
        return size(jpc.busAC, 1) + size(jpc.busDC, 1)
    elseif key == "ng"
        return size(jpc.genAC, 1)
    elseif key == "nd"
        return size(jpc.loadAC, 1) + size(jpc.loadDC, 1)
    elseif key == "nmg"
        return size(jpc.pv_acsystem, 1) + size(jpc.pv, 1)
    # 从 Cft 字典获取子键
    elseif key == "Cft_ac" && haskey(jpc.Cft, :Cft_ac)
        return jpc.Cft[:Cft_ac]
    elseif key == "Cft_dc" && haskey(jpc.Cft, :Cft_dc)
        return jpc.Cft[:Cft_dc]
    elseif key == "Cft_vsc" && haskey(jpc.Cft, :Cft_vsc)
        return jpc.Cft[:Cft_vsc]
    elseif key == "Cg" && haskey(jpc.Cft, :Cg)
        return jpc.Cft[:Cg]
    elseif key == "Cd" && haskey(jpc.Cft, :Cd)
        return jpc.Cft[:Cd]
    elseif key == "Cdf" && haskey(jpc.Cft, :Cdf)
        return jpc.Cft[:Cdf]
    elseif key == "Cmg" && haskey(jpc.Cft, :Cmg)
        return jpc.Cft[:Cmg]
    # 计算派生的电力系统参数 (标幺化处理，除以baseMVA)
    elseif key == "Pd"
        # 有功负荷向量 (标幺值)
        baseMVA = jpc.baseMVA
        if size(jpc.loadAC, 1) > 0 || size(jpc.loadDC, 1) > 0
            pd_ac = size(jpc.loadAC, 1) > 0 ? jpc.loadAC[:, 4] ./ baseMVA : Float64[]
            pd_dc = size(jpc.loadDC, 1) > 0 ? jpc.loadDC[:, 4] ./ baseMVA : Float64[]
            return vcat(pd_ac, pd_dc)
        else
            return Float64[]
        end
    elseif key == "Qd"
        # 无功负荷向量 (标幺值)
        baseMVA = jpc.baseMVA
        if size(jpc.loadAC, 1) > 0 || size(jpc.loadDC, 1) > 0
            qd_ac = size(jpc.loadAC, 1) > 0 ? jpc.loadAC[:, 5] ./ baseMVA : Float64[]
            qd_dc = size(jpc.loadDC, 1) > 0 ? zeros(size(jpc.loadDC, 1)) : Float64[]
            return vcat(qd_ac, qd_dc)
        else
            return Float64[]
        end
    elseif key == "Pgmax"
        # 发电机最大有功 (标幺值)
        baseMVA = jpc.baseMVA
        if size(jpc.genAC, 1) > 0
            return jpc.genAC[:, 9] ./ baseMVA  # PMAX 列
        else
            return Float64[]
        end
    elseif key == "Qgmax"
        # 发电机最大无功 (标幺值)
        baseMVA = jpc.baseMVA
        if size(jpc.genAC, 1) > 0
            return jpc.genAC[:, 4] ./ baseMVA  # QMAX 列
        else
            return Float64[]
        end
    elseif key == "Pmgmax"
        # 微网最大有功 (标幺值)
        baseMVA = jpc.baseMVA
        pmg_ac = size(jpc.pv_acsystem, 1) > 0 ? jpc.pv_acsystem[:, 10] ./ baseMVA : Float64[]  # p_mw 列
        pmg_dc = size(jpc.pv, 1) > 0 ? jpc.pv[:, 3] ./ baseMVA : Float64[]  # p_max 列
        return vcat(pmg_ac, pmg_dc)
    elseif key == "Qmgmax"
        # 微网最大无功 (标幺值)
        baseMVA = jpc.baseMVA
        qmg_ac = size(jpc.pv_acsystem, 1) > 0 ? jpc.pv_acsystem[:, 12] ./ baseMVA : Float64[]  # q_max 列
        qmg_dc = size(jpc.pv, 1) > 0 ? zeros(size(jpc.pv, 1)) : Float64[]
        return vcat(qmg_ac, qmg_dc)
    elseif key == "Pvscmax"
        # VSC 最大功率 (标幺值)
        baseMVA = jpc.baseMVA
        if size(jpc.converter, 1) > 0
            return fill(100.0 / baseMVA, size(jpc.converter, 1))  # 默认100MVA转标幺
        else
            return Float64[]
        end
    elseif key == "Smax"
        # 线路容量限制 (标幺值)
        baseMVA = jpc.baseMVA
        smax_ac = size(jpc.branchAC, 1) > 0 ? jpc.branchAC[:, 6] ./ baseMVA : Float64[]  # RATE_A 列
        smax_dc = size(jpc.branchDC, 1) > 0 ? jpc.branchDC[:, 6] ./ baseMVA : Float64[]
        return vcat(smax_ac, smax_dc)
    elseif key == "R"
        # 线路电阻
        r_ac = size(jpc.branchAC, 1) > 0 ? jpc.branchAC[:, 3] : Float64[]  # BR_R 列
        r_dc = size(jpc.branchDC, 1) > 0 ? jpc.branchDC[:, 3] : Float64[]
        return vcat(r_ac, r_dc)
    elseif key == "X"
        # 线路电抗
        x_ac = size(jpc.branchAC, 1) > 0 ? jpc.branchAC[:, 4] : Float64[]  # BR_X 列
        x_dc = size(jpc.branchDC, 1) > 0 ? jpc.branchDC[:, 4] : Float64[]
        return vcat(x_ac, x_dc)
    elseif key == "VMAX"
        # 电压上限
        vmax_ac = size(jpc.busAC, 1) > 0 ? jpc.busAC[:, 12] : Float64[]  # V_MAX 列
        vmax_dc = size(jpc.busDC, 1) > 0 ? fill(1.1, size(jpc.busDC, 1)) : Float64[]
        return vcat(vmax_ac, vmax_dc)
    elseif key == "VMIN"
        # 电压下限
        vmin_ac = size(jpc.busAC, 1) > 0 ? jpc.busAC[:, 13] : Float64[]  # V_MIN 列
        vmin_dc = size(jpc.busDC, 1) > 0 ? fill(0.9, size(jpc.busDC, 1)) : Float64[]
        return vcat(vmin_ac, vmin_dc)
    elseif key == "bigM"
        return 1000.0  # 大M常数
    elseif key == "η"
        # VSC 效率
        if size(jpc.converter, 1) > 0
            return jpc.converter[:, 7]  # CONV_EFF 列
        else
            return Float64[]
        end
    elseif key == "c_load"
        return 100.0  # 负荷切除成本系数
    elseif key == "c_vsc"
        return 1.0  # VSC 切换成本系数
    elseif key == "r"
        # 线路修复率（默认全部为1）
        nl_total = jpc.nl_ac + jpc.nl_dc + jpc.nl_vsc
        return ones(nl_total)
    elseif key == "Fd"
        # 虚拟负荷需求向量 (非发电机节点的需求，用于虚拟潮流约束)
        # Fd 是 (nb-ng) 维向量，每个元素代表非发电机节点的虚拟需求
        nb = size(jpc.busAC, 1) + size(jpc.busDC, 1)
        ng = size(jpc.genAC, 1)
        return ones(nb - ng)  # 虚拟潮流模型中设为1
    else
        error("JPC_tp 结构体中不存在键: $key")
    end
    
end

function setindex!(jpc::JPC_tp, value, key::String)
    # 根据字符串键设置对应的字段
    if key == "version"
        jpc.version = value
    elseif key == "baseMVA"
        jpc.baseMVA = value
    elseif key == "busAC"
        jpc.busAC = value
    elseif key == "genAC"
        jpc.genAC = value
    elseif key == "branchAC"
        jpc.branchAC = value
    elseif key == "loadAC"
        jpc.loadAC = value
    elseif key == "busDC"
        jpc.busDC = value
    elseif key == "branchDC"
        jpc.branchDC = value
    elseif key == "genDC"
        jpc.genDC = value
    elseif key == "loadDC"
        jpc.loadDC = value
    elseif key == "sgenAC"
        jpc.sgenAC = value
    elseif key == "storage"
        jpc.storage = value
    elseif key == "sgenDC"
        jpc.sgenDC = value
    elseif key == "pv"
        jpc.pv = value
    elseif key == "pv_acsystem"
        jpc.pv_acsystem = value
    elseif key == "converter"
        jpc.converter = value
    elseif key == "ext_grid"
        jpc.ext_grid = value
    elseif key == "hvcb"
        jpc.hvcb = value
    elseif key == "microgrid"
        jpc.microgrid = value
    elseif key == "α_pre" || key == "alpha_pre"
        jpc.α_pre = value
    elseif key == "nl_ac"
        jpc.nl_ac = value
    elseif key == "nl_dc"
        jpc.nl_dc = value
    elseif key == "nl_vsc"
        jpc.nl_vsc = value
    elseif key == "Cft"
        jpc.Cft = value
    else
        error("JPC_tp 结构体中不存在键: $key")
    end
    
end