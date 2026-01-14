"""
    基本电力系统组件结构体定义
    这些结构体用于表示电力系统的各个组件
"""

# ========== 交流母线 ==========
"""
    交流母线
"""
mutable struct Bus
    index::Int                    # 母线ID
    name::String                  # 母线名称
    bus_type::Int                 # 母线类型 (1=PQ, 2=PV, 3=REF, 4=NONE)
    area_id::Int                  # 区域ID
    zone_id::Int                  # 分区ID
    vn_kv::Float64                # 额定电压 (kV)
    base_kv::Float64              # 基准电压 (kV)
    v_max_pu::Float64             # 最大电压幅值 (p.u.)
    v_min_pu::Float64             # 最小电压幅值 (p.u.)
    vm_pu::Float64                # 电压幅值 (p.u.)
    va_deg::Float64               # 电压相角 (度)
    
    # 构造函数
    function Bus(; 
        index=0, 
        name="", 
        bus_type=1, 
        area_id=1, 
        zone_id=1, 
        vn_kv=10.0, 
        base_kv=10.0, 
        v_max_pu=1.05, 
        v_min_pu=0.95, 
        vm_pu=1.0, 
        va_deg=0.0)
        return new(index, name, bus_type, area_id, zone_id, vn_kv, base_kv, v_max_pu, v_min_pu, vm_pu, va_deg)
    end
end

# ========== 直流母线 ==========
"""
    直流母线
"""
mutable struct BusDC
    index::Int                    # 母线ID
    name::String                  # 母线名称
    bus_type::Int                 # 母线类型
    area_id::Int                  # 区域ID
    zone_id::Int                  # 分区ID
    vn_kv::Float64                # 额定电压 (kV)
    base_kv::Float64              # 基准电压 (kV)
    v_max_pu::Float64             # 最大电压幅值 (p.u.)
    v_min_pu::Float64             # 最小电压幅值 (p.u.)
    vm_pu::Float64                # 电压幅值 (p.u.)
    va_deg::Float64               # 电压相角 (度)
    
    # 构造函数
    function BusDC(;
        index=0,
        name="",
        bus_type=1,
        area_id=1,
        zone_id=1,
        vn_kv=0.75,
        base_kv=0.75,
        v_max_pu=1.05,
        v_min_pu=0.95,
        vm_pu=1.0,
        va_deg=0.0)
        return new(index, name, bus_type, area_id, zone_id, vn_kv, base_kv, v_max_pu, v_min_pu, vm_pu, va_deg)
    end
end

# ========== 交流线路 ==========
"""
    交流线路
"""
mutable struct Line
    index::Int                    # 线路ID
    name::String                  # 线路名称
    from_bus::Int                 # 首端母线ID
    to_bus::Int                   # 末端母线ID
    r_pu::Float64                 # 电阻 (p.u.)
    x_pu::Float64                 # 电抗 (p.u.)
    b_pu::Float64                 # 电纳 (p.u.)
    rate_a_mva::Float64          # 长期额定容量 (MVA)
    rate_b_mva::Float64          # 短期额定容量 (MVA)
    rate_c_mva::Float64          # 应急额定容量 (MVA)
    tap::Float64                  # 变比
    shift_deg::Float64            # 相移角 (度)
    status::Int                   # 状态 (1=投入, 0=退出)
    
    # 构造函数
    function Line(;
        index=0,
        name="",
        from_bus=0,
        to_bus=0,
        r_pu=0.0,
        x_pu=0.0,
        b_pu=0.0,
        rate_a_mva=100.0,
        rate_b_mva=100.0,
        rate_c_mva=100.0,
        tap=1.0,
        shift_deg=0.0,
        status=1)
        return new(index, name, from_bus, to_bus, r_pu, x_pu, b_pu, rate_a_mva, rate_b_mva, rate_c_mva, tap, shift_deg, status)
    end
end

# ========== 直流线路 ==========
"""
    直流线路
"""
mutable struct LineDC
    index::Int                    # 线路ID
    name::String                  # 线路名称
    from_bus::Int                 # 首端母线ID
    to_bus::Int                   # 末端母线ID
    r_pu::Float64                 # 电阻 (p.u.)
    rate_a_mw::Float64            # 长期额定容量 (MW)
    rate_b_mw::Float64            # 短期额定容量 (MW)
    rate_c_mw::Float64            # 应急额定容量 (MW)
    status::Int                   # 状态 (1=投入, 0=退出)
    
    # 构造函数
    function LineDC(;
        index=0,
        name="",
        from_bus=0,
        to_bus=0,
        r_pu=0.0,
        rate_a_mw=100.0,
        rate_b_mw=100.0,
        rate_c_mw=100.0,
        status=1)
        return new(index, name, from_bus, to_bus, r_pu, rate_a_mw, rate_b_mw, rate_c_mw, status)
    end
end

# ========== 交流负荷 ==========
"""
    交流负荷
"""
mutable struct Load
    index::Int                    # 负荷ID
    name::String                  # 负荷名称
    bus_id::Int                   # 连接母线ID
    pd_mw::Float64                # 有功功率 (MW)
    qd_mvar::Float64              # 无功功率 (MVAR)
    status::Int                   # 状态 (1=投入, 0=退出)
    
    # 构造函数
    function Load(;
        index=0,
        name="",
        bus_id=0,
        pd_mw=0.0,
        qd_mvar=0.0,
        status=1)
        return new(index, name, bus_id, pd_mw, qd_mvar, status)
    end
end

# ========== 柔性负荷 ==========
"""
    柔性负荷
"""
mutable struct FlexLoad
    index::Int
    name::String
    bus_id::Int
    pd_mw::Float64
    qd_mvar::Float64
    pd_max_mw::Float64
    pd_min_mw::Float64
    status::Int
    
    function FlexLoad(;
        index=0,
        name="",
        bus_id=0,
        pd_mw=0.0,
        qd_mvar=0.0,
        pd_max_mw=0.0,
        pd_min_mw=0.0,
        status=1)
        return new(index, name, bus_id, pd_mw, qd_mvar, pd_max_mw, pd_min_mw, status)
    end
end

# ========== 不对称负荷 ==========
"""
    不对称负荷
"""
mutable struct AsymmetricLoad
    index::Int
    name::String
    bus_id::Int
    pd_a_mw::Float64
    pd_b_mw::Float64
    pd_c_mw::Float64
    qd_a_mvar::Float64
    qd_b_mvar::Float64
    qd_c_mvar::Float64
    status::Int
    
    function AsymmetricLoad(;
        index=0,
        name="",
        bus_id=0,
        pd_a_mw=0.0,
        pd_b_mw=0.0,
        pd_c_mw=0.0,
        qd_a_mvar=0.0,
        qd_b_mvar=0.0,
        qd_c_mvar=0.0,
        status=1)
        return new(index, name, bus_id, pd_a_mw, pd_b_mw, pd_c_mw, qd_a_mvar, qd_b_mvar, qd_c_mvar, status)
    end
end

# ========== 直流负荷 ==========
"""
    直流负荷
"""
mutable struct LoadDC
    index::Int
    name::String
    bus_id::Int
    pd_mw::Float64
    status::Int
    
    function LoadDC(;
        index=0,
        name="",
        bus_id=0,
        pd_mw=0.0,
        status=1)
        return new(index, name, bus_id, pd_mw, status)
    end
end

# ========== 发电机 ==========
"""
    发电机
"""
mutable struct Generator
    index::Int
    name::String
    bus_id::Int
    pg_mw::Float64
    qg_mvar::Float64
    qg_max_mvar::Float64
    qg_min_mvar::Float64
    vg_pu::Float64
    mbase_mva::Float64
    status::Int
    pg_max_mw::Float64
    pg_min_mw::Float64
    
    function Generator(;
        index=0,
        name="",
        bus_id=0,
        pg_mw=0.0,
        qg_mvar=0.0,
        qg_max_mvar=10.0,
        qg_min_mvar=-10.0,
        vg_pu=1.0,
        mbase_mva=100.0,
        status=1,
        pg_max_mw=100.0,
        pg_min_mw=0.0)
        return new(index, name, bus_id, pg_mw, qg_mvar, qg_max_mvar, qg_min_mvar, vg_pu, mbase_mva, status, pg_max_mw, pg_min_mw)
    end
end

# ========== 静态发电机 ==========
"""
    静态发电机（如光伏）
"""
mutable struct StaticGenerator
    index::Int
    name::String
    bus_id::Int
    pg_mw::Float64
    qg_mvar::Float64
    status::Int
    
    function StaticGenerator(;
        index=0,
        name="",
        bus_id=0,
        pg_mw=0.0,
        qg_mvar=0.0,
        status=1)
        return new(index, name, bus_id, pg_mw, qg_mvar, status)
    end
end

# ========== 直流静态发电机 ==========
"""
    直流静态发电机
"""
mutable struct StaticGeneratorDC
    index::Int
    name::String
    bus_id::Int
    pg_mw::Float64
    status::Int
    
    function StaticGeneratorDC(;
        index=0,
        name="",
        bus_id=0,
        pg_mw=0.0,
        status=1)
        return new(index, name, bus_id, pg_mw, status)
    end
end

# ========== 储能系统 ==========
"""
    储能系统
"""
mutable struct Storage
    index::Int
    name::String
    bus_id::Int
    energy_capacity_kwh::Float64
    power_capacity_mw::Float64
    soc_init::Float64
    soc_min::Float64
    soc_max::Float64
    efficiency::Float64
    status::Int
    
    function Storage(;
        index=0,
        name="",
        bus_id=0,
        energy_capacity_kwh=100.0,
        power_capacity_mw=1.0,
        soc_init=0.5,
        soc_min=0.1,
        soc_max=0.9,
        efficiency=0.9,
        status=1)
        return new(index, name, bus_id, energy_capacity_kwh, power_capacity_mw, soc_init, soc_min, soc_max, efficiency, status)
    end
end

# ========== ETAP 储能系统 ==========
"""
    ETAP 储能系统
"""
mutable struct Storageetap
    index::Int
    name::String
    bus_id::Int
    energy_capacity_kwh::Float64
    power_capacity_mw::Float64
    soc_init::Float64
    soc_min::Float64
    soc_max::Float64
    efficiency::Float64
    voc_kv::Float64
    status::Int
    
    function Storageetap(;
        index=0,
        name="",
        bus_id=0,
        energy_capacity_kwh=100.0,
        power_capacity_mw=1.0,
        soc_init=0.5,
        soc_min=0.1,
        soc_max=0.9,
        efficiency=0.9,
        voc_kv=0.75,
        status=1)
        return new(index, name, bus_id, energy_capacity_kwh, power_capacity_mw, soc_init, soc_min, soc_max, efficiency, voc_kv, status)
    end
end

# ========== 光伏阵列 ==========
"""
    光伏阵列
"""
mutable struct PVArray
    index::Int
    name::String
    bus_id::Int
    voc_v::Float64
    vmpp_v::Float64
    isc_a::Float64
    impp_a::Float64
    irradiance_w_m2::Float64
    area_m2::Float64
    status::Int
    p_rated_mw::Float64  # 额定有功功率 (MW)，来自 Excel 的 PVAPower 列
    
    function PVArray(;
        index=0,
        name="",
        bus_id=0,
        voc_v=1000.0,
        vmpp_v=800.0,
        isc_a=10.0,
        impp_a=9.0,
        irradiance_w_m2=1000.0,
        area_m2=10.0,
        status=1,
        p_rated_mw=0.0)
        return new(index, name, bus_id, voc_v, vmpp_v, isc_a, impp_a, irradiance_w_m2, area_m2, status, p_rated_mw)
    end
end

# ========== 交流光伏系统 ==========
"""
    交流光伏系统
"""
mutable struct ACPVSystem
    index::Int
    name::String
    bus_id::Int
    voc_v::Float64
    vmpp_v::Float64
    isc_a::Float64
    impp_a::Float64
    irradiance_w_m2::Float64
    area_m2::Float64
    inverter_efficiency::Float64
    inverter_mode::Int
    inverter_pac_mw::Float64
    inverter_qac_mvar::Float64
    inverter_qac_max_mvar::Float64
    inverter_qac_min_mvar::Float64
    status::Int
    
    function ACPVSystem(;
        index=0,
        name="",
        bus_id=0,
        voc_v=1000.0,
        vmpp_v=800.0,
        isc_a=10.0,
        impp_a=9.0,
        irradiance_w_m2=1000.0,
        area_m2=10.0,
        inverter_efficiency=0.97,
        inverter_mode=1,
        inverter_pac_mw=0.0,
        inverter_qac_mvar=0.0,
        inverter_qac_max_mvar=1.0,
        inverter_qac_min_mvar=-1.0,
        status=1)
        return new(index, name, bus_id, voc_v, vmpp_v, isc_a, impp_a, irradiance_w_m2, area_m2, inverter_efficiency, inverter_mode, inverter_pac_mw, inverter_qac_mvar, inverter_qac_max_mvar, inverter_qac_min_mvar, status)
    end
end

# ========== 换流器 ==========
"""
    换流器
"""
mutable struct Converter
    index::Int
    name::String
    ac_bus_id::Int
    dc_bus_id::Int
    p_ac_mw::Float64
    q_ac_mvar::Float64
    p_dc_mw::Float64
    efficiency::Float64
    mode::Int
    status::Int
    
    function Converter(;
        index=0,
        name="",
        ac_bus_id=0,
        dc_bus_id=0,
        p_ac_mw=0.0,
        q_ac_mvar=0.0,
        p_dc_mw=0.0,
        efficiency=0.97,
        mode=1,
        status=1)
        return new(index, name, ac_bus_id, dc_bus_id, p_ac_mw, q_ac_mvar, p_dc_mw, efficiency, mode, status)
    end
end

# ========== 外部电网 ==========
"""
    外部电网
"""
mutable struct ExternalGrid
    index::Int
    name::String
    bus_id::Int
    vm_pu::Float64
    va_deg::Float64
    status::Int
    
    function ExternalGrid(;
        index=0,
        name="",
        bus_id=0,
        vm_pu=1.0,
        va_deg=0.0,
        status=1)
        return new(index, name, bus_id, vm_pu, va_deg, status)
    end
end

# ========== 高压断路器 ==========
"""
    高压断路器
"""
mutable struct HighVoltageCircuitBreaker
    index::Int
    name::String
    from_element::String
    to_element::String
    status::Int
    
    function HighVoltageCircuitBreaker(;
        index=0,
        name="",
        from_element="",
        to_element="",
        status=1)
        return new(index, name, from_element, to_element, status)
    end
end

# ========== 微电网 ==========
"""
    微电网
"""
mutable struct Microgrid
    index::Int
    name::String
    capacity_mw::Float64
    peak_load_mw::Float64
    duration_h::Float64
    area_id::Int
    
    function Microgrid(;
        index=0,
        name="",
        capacity_mw=0.0,
        peak_load_mw=0.0,
        duration_h=0.0,
        area_id=1)
        return new(index, name, capacity_mw, peak_load_mw, duration_h, area_id)
    end
end

# ========== 能量路由器核心 ==========
"""
    能量路由器核心
"""
mutable struct EnergyRouterCore
    index::Int
    name::String
    prime_bus_id::Int
    second_bus_id::Int
    loss_percent::Float64
    max_p_mw::Float64
    min_p_mw::Float64
    status::Int
    
    function EnergyRouterCore(;
        index=0,
        name="",
        prime_bus_id=0,
        second_bus_id=0,
        loss_percent=1.0,
        max_p_mw=100.0,
        min_p_mw=0.0,
        status=1)
        return new(index, name, prime_bus_id, second_bus_id, loss_percent, max_p_mw, min_p_mw, status)
    end
end

# ========== 变压器2绕组 ==========
"""
    变压器2绕组
"""
mutable struct Transformer2W
    index::Int
    name::String
    from_bus::Int
    to_bus::Int
    r_pu::Float64
    x_pu::Float64
    rate_a_mva::Float64
    tap::Float64
    status::Int
    
    function Transformer2W(;
        index=0,
        name="",
        from_bus=0,
        to_bus=0,
        r_pu=0.0,
        x_pu=0.0,
        rate_a_mva=100.0,
        tap=1.0,
        status=1)
        return new(index, name, from_bus, to_bus, r_pu, x_pu, rate_a_mva, tap, status)
    end
end

# ========== 变压器3绕组 ==========
"""
    变压器3绕组
"""
mutable struct Transformer3W
    index::Int
    name::String
    from_bus::Int
    to_bus::Int
    tertiary_bus::Int
    r12_pu::Float64
    x12_pu::Float64
    r13_pu::Float64
    x13_pu::Float64
    r23_pu::Float64
    x23_pu::Float64
    rate_a_mva::Float64
    status::Int
    
    function Transformer3W(;
        index=0,
        name="",
        from_bus=0,
        to_bus=0,
        tertiary_bus=0,
        r12_pu=0.0,
        x12_pu=0.0,
        r13_pu=0.0,
        x13_pu=0.0,
        r23_pu=0.0,
        x23_pu=0.0,
        rate_a_mva=100.0,
        status=1)
        return new(index, name, from_bus, to_bus, tertiary_bus, r12_pu, x12_pu, r13_pu, x13_pu, r23_pu, x23_pu, rate_a_mva, status)
    end
end

# ========== ETAP 变压器2绕组 ==========
"""
    ETAP 变压器2绕组
"""
mutable struct Transformer2Wetap
    index::Int
    name::String
    from_bus::Int
    to_bus::Int
    r_pu::Float64
    x_pu::Float64
    rate_a_mva::Float64
    tap::Float64
    status::Int
    
    function Transformer2Wetap(;
        index=0,
        name="",
        from_bus=0,
        to_bus=0,
        r_pu=0.0,
        x_pu=0.0,
        rate_a_mva=100.0,
        tap=1.0,
        status=1)
        return new(index, name, from_bus, to_bus, r_pu, x_pu, rate_a_mva, tap, status)
    end
end

# ========== 充电站 ==========
"""
    充电站
"""
mutable struct ChargingStation
    index::Int
    name::String
    bus_id::Int
    capacity_mw::Float64
    status::Int
    
    function ChargingStation(;
        index=0,
        name="",
        bus_id=0,
        capacity_mw=0.0,
        status=1)
        return new(index, name, bus_id, capacity_mw, status)
    end
end

# ========== 充电器 ==========
"""
    充电器
"""
mutable struct Charger
    index::Int
    name::String
    bus_id::Int
    power_mw::Float64
    status::Int
    
    function Charger(;
        index=0,
        name="",
        bus_id=0,
        power_mw=0.0,
        status=1)
        return new(index, name, bus_id, power_mw, status)
    end
end

# ========== 电动汽车聚合器 ==========
"""
    电动汽车聚合器
"""
mutable struct EVAggregator
    index::Int
    name::String
    bus_id::Int
    capacity_mw::Float64
    flex_capacity_mw::Float64
    status::Int
    
    function EVAggregator(;
        index=0,
        name="",
        bus_id=0,
        capacity_mw=0.0,
        flex_capacity_mw=0.0,
        status=1)
        return new(index, name, bus_id, capacity_mw, flex_capacity_mw, status)
    end
end

# ========== 虚拟电厂 ==========
"""
    虚拟电厂
"""
mutable struct VirtualPowerPlant
    index::Int
    name::String
    bus_id::Int
    capacity_mw::Float64
    status::Int
    
    function VirtualPowerPlant(;
        index=0,
        name="",
        bus_id=0,
        capacity_mw=0.0,
        status=1)
        return new(index, name, bus_id, capacity_mw, status)
    end
end

# ========== 碳排放时间序列 ==========
"""
    碳排放时间序列
"""
mutable struct CarbonTimeSeries
    index::Int
    name::String
    carbon_emissions::Vector{Float64}
    
    function CarbonTimeSeries(;
        index=0,
        name="",
        carbon_emissions=Float64[])
        return new(index, name, carbon_emissions)
    end
end

# ========== 设备碳排放 ==========
"""
    设备碳排放
"""
mutable struct EquipmentCarbon
    index::Int
    name::String
    carbon_emission_kg_kwh::Float64
    
    function EquipmentCarbon(;
        index=0,
        name="",
        carbon_emission_kg_kwh=0.0)
        return new(index, name, carbon_emission_kg_kwh)
    end
end