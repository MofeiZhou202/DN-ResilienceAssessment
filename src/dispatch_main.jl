using DataFrames
using XLSX
using SparseArrays
using LinearAlgebra
using JuMP
using Gurobi
using Printf
using Dates
using Base.Filesystem: basename

# 注意：utils 文件已在 workflows.jl 中统一加载，这里不再重复加载
include(joinpath(@__DIR__, "transportation_network.jl"))
using .TransportationNetwork: transportation_network, TransportNetworkData

const MOI = JuMP.MOI
const VariableRef = JuMP.VariableRef

const DEFAULT_CASE_XLSX = joinpath(@__DIR__, "..", "data", "ac_dc_real_case.xlsx")
const DEFAULT_TOPOLOGY_XLSX = joinpath(@__DIR__, "..", "data", "topology_reconfiguration_results.xlsx")
const DEFAULT_MC_XLSX = joinpath(@__DIR__, "..", "data", "mc_simulation_results_k100_clusters.xlsx")
const DEFAULT_HOURS = 48
const TIME_STEP_HOURS = 1.0
const MESS_TRAVEL_ENERGY_LOSS_PER_HOUR = 0.0

const NO_POWER_MAX_CONSECUTIVE_HOURS = 2
const NO_POWER_WINDOW = NO_POWER_MAX_CONSECUTIVE_HOURS + 1
const NO_POWER_SOFT_PENALTY = 1.0e5
const NO_POWER_REL_TOL = 0.02
const NO_POWER_ABS_TOL = 1e-3

const TRANSPORT_NODE_TO_GRID = Dict(
    1 => 5,
    2 => 10,
    3 => 15,
    4 => 20,
    5 => 25,
    6 => 30,
)

struct TrafficArc
    origin::Int
    destination::Int
    travel_time::Int
end

struct MessMobilitySchedule
    u_vars::Vector{Matrix{VariableRef}}
    travel_vars::Dict{Int, Matrix{VariableRef}}
    travel_actions::Dict{Int, Vector{TrafficArc}}
    departure_lookup::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}
    arrival_lookup::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}
    transit_lookup::Dict{Int, Vector{Tuple{Int, Int}}}
    nb_transport::Int
end

struct MESSConfig
    name::String
    node::Int
    charge_max::Float64
    discharge_max::Float64
    energy_max::Float64
    soc_initial::Float64
    eta_charge::Float64
    eta_discharge::Float64
end

struct HybridGridCase
    nb::Int
    nb_ac::Int
    nb_dc::Int
    nl_ac::Int
    nl_dc::Int
    nl_vsc::Int
    ng::Int
    nmg::Int
    nmess::Int
    Cft_ac::SparseMatrixCSC{Float64, Int}
    Cft_dc::SparseMatrixCSC{Float64, Int}
    Cft_vsc::SparseMatrixCSC{Float64, Int}
    Cg::SparseMatrixCSC{Float64, Int}
    Cmg::SparseMatrixCSC{Float64, Int}
    Cd::SparseMatrixCSC{Float64, Int}
    Pd::Vector{Float64}
    Qd::Vector{Float64}
    Pgmax::Vector{Float64}
    Qgmax::Vector{Float64}
    Pmgmax::Vector{Float64}
    Pmgmin::Vector{Float64}
    Qmgmax::Vector{Float64}
    Qmgmin::Vector{Float64}
    Pvscmax::Vector{Float64}
    eta_vsc::Vector{Float64}
    Smax_ac::Vector{Float64}
    Smax_dc::Vector{Float64}
    alpha_ac::Vector{Float64}
    alpha_dc::Vector{Float64}
    alpha_vsc::Vector{Float64}
    switch_flag::Vector{Float64}
    VMAX::Vector{Float64}
    VMIN::Vector{Float64}
    R::Vector{Float64}
    X::Vector{Float64}
    bigM::Float64
    c_load::Float64
    c_sg::Float64
    c_mg::Float64
    c_vsc::Float64
    load_per_node::Vector{Float64}
    generator_nodes::Vector{Int}
    microgrid_nodes::Vector{Int}
    mess_nodes::Vector{Int}
    mess_connect::SparseMatrixCSC{Float64, Int}
    mess_charge_max::Vector{Float64}
    mess_discharge_max::Vector{Float64}
    mess_energy_max::Vector{Float64}
    mess_soc_initial::Vector{Float64}
    mess_eta_charge::Vector{Float64}
    mess_eta_discharge::Vector{Float64}
    line_types::Vector{String}
    transport_node_to_grid::Dict{Int, Int}
    transport_grid_to_node::Dict{Int, Int}
    mess_transport_initial::Vector{Int}
    mess_names::Vector{String}
    Cft_ac_rows::Vector{Vector{Tuple{Int, Float64}}}
    Cft_dc_rows::Vector{Vector{Tuple{Int, Float64}}}
    Cft_vsc_rows::Vector{Vector{Tuple{Int, Float64}}}
end

# 默认 MESS 配置 (单位: kW, kWh)
# 配电网级别：典型移动储能系统容量
const DEFAULT_MESS = [
    MESSConfig("MESS-1", 5, 1500.0, 1500.0, 4500.0, 2500.0, 92.0, 90.0),   # 1500 kW, 4500 kWh
    MESSConfig("MESS-2", 10, 1000.0, 1000.0, 4000.0, 2000.0, 92.0, 90.0),  # 1000 kW, 4000 kWh
    MESSConfig("MESS-3", 30, 500.0, 500.0, 3500.0, 1500.0, 92.0, 90.0),    # 500 kW, 3500 kWh
]

_normalize_label(value) = strip(string(value))

function _parse_binary(value)
    if value isa Bool
        return value ? 1 : 0
    elseif value isa Integer
        return value != 0 ? 1 : 0
    elseif value isa AbstractString
        lowered = lowercase(strip(value))
        return lowered in ("1", "true", "yes", "y") ? 1 : 0
    elseif value isa Real
        return value != 0 ? 1 : 0
    else
        return 0
    end
end

function _row_get(row::DataFrameRow, sym::Symbol, default)
    if !(sym in propertynames(row))
        return default
    end
    value = row[sym]
    if value === missing || value === nothing
        return default
    end
    return value
end

function _get_label(row::DataFrameRow, name::AbstractString)
    return _normalize_label(_row_get(row, Symbol(name), ""))
end

function _get_float(row::DataFrameRow, name::AbstractString, default::Float64=0.0)
    value = _row_get(row, Symbol(name), default)
    if value === default
        return default
    elseif value isa Real
        return float(value)
    elseif value isa AbstractString
        cleaned = strip(value)
        if isempty(cleaned)
            return default
        end
        parsed = try
            parse(Float64, cleaned)
        catch
            default
        end
        return parsed
    else
        return default
    end
end

function _get_int(row::DataFrameRow, name::AbstractString, default::Int=0)
    value = _row_get(row, Symbol(name), default)
    if value === default
        return default
    elseif value isa Integer
        return Int(value)
    elseif value isa Real
        return Int(round(value))
    elseif value isa AbstractString
        cleaned = strip(value)
        if isempty(cleaned)
            return default
        end
        parsed = try
            parse(Int, cleaned)
        catch
            default
        end
        return parsed
    else
        return default
    end
end

_normalized_column_name(name) = replace(lowercase(String(name)), r"\s+" => "")

function _find_column(df::DataFrame, candidates::Vector{String})
    normalized_candidates = map(_normalized_column_name, candidates)
    for name in names(df)
        if _normalized_column_name(name) in normalized_candidates
            return name
        end
    end
    error("未能在表格中找到列: $(join(candidates, ", "))")
end

function _build_incidence_matrix(edges::Vector{Tuple{Int, Int}}, num_nodes::Int)
    n_edges = length(edges)
    if n_edges == 0
        return spzeros(Float64, 0, num_nodes)
    end
    total_entries = 2 * n_edges
    I = Vector{Int}(undef, total_entries)
    J = Vector{Int}(undef, total_entries)
    V = Vector{Float64}(undef, total_entries)
    cursor = 1
    for (row_idx, (frm, to)) in enumerate(edges)
        I[cursor] = row_idx
        J[cursor] = frm
        V[cursor] = 1.0
        cursor += 1
        I[cursor] = row_idx
        J[cursor] = to
        V[cursor] = -1.0
        cursor += 1
    end
    return sparse(I, J, V, n_edges, num_nodes)
end

function _build_connection_matrix(nodes::Vector{Int}, num_nodes::Int)
    n = length(nodes)
    if n == 0
        return spzeros(Float64, 0, num_nodes)
    end
    I = collect(1:n)
    J = nodes
    V = ones(Float64, n)
    return sparse(I, J, V, n, num_nodes)
end

function _max_consecutive_true(values::AbstractVector{<:Real}; threshold::Float64=0.5)
    best = 0
    current = 0
    for value in values
        if value > threshold
            current += 1
            best = max(best, current)
        else
            current = 0
        end
    end
    return best
end

function _row_entries(mat::SparseMatrixCSC{Float64, Int})
    rows = [Vector{Tuple{Int, Float64}}() for _ in 1:size(mat, 1)]
    for col in 1:size(mat, 2)
        for nz in mat.colptr[col]:(mat.colptr[col+1]-1)
            row_idx = mat.rowval[nz]
            value = mat.nzval[nz]
            push!(rows[row_idx], (col, value))
        end
    end
    return rows
end

function _build_transport_mapping(nb::Int)
    node_map = Dict{Int, Int}()
    grid_map = Dict{Int, Int}()
    for (transport_label, grid_label) in TRANSPORT_NODE_TO_GRID
        grid_index = grid_label
        if grid_index < 1 || grid_index > nb
            error("TRANSPORT_NODE_TO_GRID references grid node $(grid_label) which exceeds case node count $(nb).")
        end
        node_map[transport_label] = grid_index
        grid_map[grid_index] = transport_label
    end
    if isempty(node_map)
        error("Transport node mapping must define at least one node.")
    end
    return node_map, grid_map
end

function _resolve_mess_transport_initial(mess_nodes::Vector{Int}, grid_to_transport::Dict{Int, Int})
    if isempty(mess_nodes)
        return Int[]
    end
    initial_nodes = Int[]
    for grid_idx in mess_nodes
        transport_node = get(grid_to_transport, grid_idx, nothing)
        if transport_node === nothing
            error("无法为电网节点 $(grid_idx) 找到映射的交通节点，更新 TRANSPORT_NODE_TO_GRID。")
        end
        push!(initial_nodes, transport_node)
    end
    return initial_nodes
end

function load_hybrid_case(case_path::AbstractString, mess_configs::Vector{MESSConfig})
    if !isfile(case_path)
        error("未找到电网算例文件 $(case_path)")
    end

    # 使用统一的数据加载流程（ETAPImporter + JuliaPowerCase2Jpc_tp）
    println("正在使用统一数据流程加载电网数据...")
    julia_case = load_julia_power_data(case_path)
    jpc_tp, C_sparse = JuliaPowerCase2Jpc_tp(julia_case)
    println("数据加载完成: AC节点=$(jpc_tp[:nb_ac]), DC节点=$(jpc_tp[:nb_dc])")

    # 从 jpc_tp 获取基本参数
    nb_ac = jpc_tp[:nb_ac]
    nb_dc = jpc_tp[:nb_dc]
    nb = nb_ac + nb_dc
    nl_ac = jpc_tp[:nl_ac]
    nl_dc = jpc_tp[:nl_dc]
    nl_vsc = jpc_tp[:nl_vsc]
    ng = jpc_tp[:ng]
    baseMVA = jpc_tp.baseMVA

    # 获取稀疏矩阵
    Cft_ac = C_sparse[:Cft_ac]
    Cft_dc = C_sparse[:Cft_dc]
    Cft_vsc = C_sparse[:Cft_vsc]
    Cg = C_sparse[:Cg]
    Cmg = C_sparse[:Cmg]
    Cd = C_sparse[:Cd]

    # 获取负荷数据（转换为 kW）
    Pd = jpc_tp[:Pd] .* baseMVA .* 1000  # 标幺值 -> MW -> kW
    Qd = jpc_tp[:Qd] .* baseMVA .* 1000  # 标幺值 -> MW -> kW
    
    # 获取发电机数据（转换为 kW）
    Pgmax = jpc_tp[:Pgmax] .* baseMVA .* 1000  # MW -> kW
    Qgmax = jpc_tp[:Qgmax] .* baseMVA .* 1000  # MW -> kW
    
    # 获取微网/光伏数据（转换为 kW）
    Pmgmax = jpc_tp[:Pmgmax] .* baseMVA .* 1000  # MW -> kW
    Qmgmax = jpc_tp[:Qmgmax] .* baseMVA .* 1000  # MVar -> kVar
    nmg = jpc_tp[:nmg]
    
    # 获取 VSC 数据（转换为 kW）
    Pvscmax = jpc_tp[:Pvscmax] .* baseMVA .* 1000  # MW -> kW
    eta_vsc = fill(0.95, nl_vsc)  # 默认效率 95%
    if size(jpc_tp.converter, 1) > 0 && size(jpc_tp.converter, 2) >= 9
        for i in 1:nl_vsc
            eta_vsc[i] = jpc_tp.converter[i, 9] > 0 ? jpc_tp.converter[i, 9] : 0.95
        end
    end
    
    # 获取线路参数
    R_ac = nl_ac > 0 ? jpc_tp.branchAC[:, 3] : Float64[]  # R 列
    X_ac = nl_ac > 0 ? jpc_tp.branchAC[:, 4] : Float64[]  # X 列
    R_dc = nl_dc > 0 ? jpc_tp.branchDC[:, 3] : Float64[]
    
    # 获取线路状态（alpha）
    alpha_ac = nl_ac > 0 ? Float64.(jpc_tp.branchAC[:, 11]) : Float64[]  # BR_STATUS 列
    alpha_dc = nl_dc > 0 ? Float64.(jpc_tp.branchDC[:, 11]) : Float64[]
    alpha_vsc = nl_vsc > 0 ? Float64.(jpc_tp.converter[:, 3]) : Float64[]  # CONV_INSERVICE 列
    
    # 获取电压限制
    VMAX_ac = nb_ac > 0 ? jpc_tp.busAC[:, 12] : Float64[]  # VMAX 列
    VMIN_ac = nb_ac > 0 ? jpc_tp.busAC[:, 13] : Float64[]  # VMIN 列
    VMAX_dc = fill(1.05, nb_dc)
    VMIN_dc = fill(0.90, nb_dc)
    
    # 获取线路容量
    Smax_ac = nl_ac > 0 ? jpc_tp.branchAC[:, 6] : Float64[]  # RATE_A 列
    Smax_dc = nl_dc > 0 ? jpc_tp.branchDC[:, 6] : Float64[]
    
    # 读取开关信息（这部分仍需从 Excel 读取，因为 jpc_tp 没有存储）
    switch_df = DataFrame(XLSX.readtable(case_path, "hvcb"))
    switch_pairs = Set{Tuple{String, String}}()
    for row in eachrow(switch_df)
        a = _get_label(row, "FromElement")
        b = _get_label(row, "ToElement")
        if isempty(a) || isempty(b)
            continue
        end
        push!(switch_pairs, (a, b))
        push!(switch_pairs, (b, a))
    end
    
    # 构建开关标志向量（需要根据母线名字判断）
    # 由于 jpc_tp 没有存储母线名字，暂时设置所有线路都可切换
    switch_ac = ones(Float64, nl_ac)
    switch_dc = ones(Float64, nl_dc)
    switch_vec = vcat(switch_ac, switch_dc, ones(Float64, nl_vsc))
    
    # 计算每个节点的负荷
    load_per_node = vec(transpose(Cd) * Pd)
    
    # 获取发电机和微网节点
    gen_nodes = ng > 0 ? Int.(round.(jpc_tp.genAC[:, 1])) : Int[]
    
    # 微网节点需要从 Cmg 矩阵反推
    mg_nodes = Int[]
    for i in 1:nmg
        node_idx = findfirst(x -> x == 1, Cmg[i, :])
        if node_idx !== nothing
            push!(mg_nodes, node_idx)
        end
    end
    
    # MESS 配置处理
    mess_nodes = Int[]
    mess_charge_max = Float64[]
    mess_discharge_max = Float64[]
    mess_energy_max = Float64[]
    mess_soc_initial = Float64[]
    mess_eta_charge = Float64[]
    mess_eta_discharge = Float64[]
    mess_names = String[]

    for config in mess_configs
        node_idx = config.node
        if node_idx < 1 || node_idx > nb
            error("MESS $(config.name) 的接入节点 $(config.node) 超出节点范围")
        end
        push!(mess_nodes, node_idx)
        push!(mess_charge_max, config.charge_max)
        push!(mess_discharge_max, config.discharge_max)
        push!(mess_energy_max, config.energy_max)
        push!(mess_soc_initial, min(config.soc_initial, config.energy_max))
        eta_c = config.eta_charge > 1 ? config.eta_charge / 100.0 : config.eta_charge
        eta_d = config.eta_discharge > 1 ? config.eta_discharge / 100.0 : config.eta_discharge
        push!(mess_eta_charge, eta_c)
        push!(mess_eta_discharge, eta_d)
        push!(mess_names, config.name)
    end

    mess_connect = _build_connection_matrix(mess_nodes, nb)
    
    line_types = vcat(fill("AC", nl_ac), fill("DC", nl_dc), fill("VSC", nl_vsc))

    transport_node_to_grid_map, transport_grid_to_node_map = _build_transport_mapping(nb)
    mess_transport_initial_nodes = _resolve_mess_transport_initial(copy(mess_nodes), transport_grid_to_node_map)

    # 转换矩阵类型为 SparseMatrixCSC{Float64, Int}
    Cft_ac_f = sparse(Float64.(Cft_ac))
    Cft_dc_f = sparse(Float64.(Cft_dc))
    Cft_vsc_f = sparse(Float64.(Cft_vsc))
    Cg_f = sparse(Float64.(Cg))
    Cmg_f = sparse(Float64.(Cmg))
    Cd_f = sparse(Float64.(Cd))

    case = HybridGridCase(
        nb,
        nb_ac,
        nb_dc,
        nl_ac,
        nl_dc,
        nl_vsc,
        ng,
        nmg,
        length(mess_nodes),
        Cft_ac_f,
        Cft_dc_f,
        Cft_vsc_f,
        Cg_f,
        Cmg_f,
        Cd_f,
        Pd,
        Qd,
        Pgmax,
        Qgmax,
        Pmgmax,
        zeros(nmg),
        Qmgmax,
        zeros(nmg),
        Pvscmax,
        collect(eta_vsc),
        Smax_ac,
        Smax_dc,
        collect(alpha_ac),
        collect(alpha_dc),
        collect(alpha_vsc),
        switch_vec,
        vcat(VMAX_ac, VMAX_dc),
        vcat(VMIN_ac, VMIN_dc),
        vcat(R_ac, R_dc),
        vcat(X_ac, zeros(length(R_dc))),
        1000.0,
        1000.0,
        5.0,
        2.0,
        1.0,
        load_per_node,
        collect(gen_nodes),
        collect(mg_nodes),
        collect(mess_nodes),
        mess_connect,
        collect(mess_charge_max),
        collect(mess_discharge_max),
        collect(mess_energy_max),
        collect(mess_soc_initial),
        collect(mess_eta_charge),
        collect(mess_eta_discharge),
        line_types,
        transport_node_to_grid_map,
        transport_grid_to_node_map,
        mess_transport_initial_nodes,
        mess_names,
        _row_entries(Cft_ac_f),
        _row_entries(Cft_dc_f),
        _row_entries(Cft_vsc_f),
    )

    println("HybridGridCase 构建完成:")
    println("  - 节点: AC=$nb_ac, DC=$nb_dc, 总计=$nb")
    println("  - 线路: AC=$nl_ac, DC=$nl_dc, VSC=$nl_vsc")
    println("  - 发电机: $ng, 微网: $nmg, MESS: $(length(mess_nodes))")
    println("  - 总负荷: $(round(sum(Pd), digits=1)) kW")
    println("  - 微网总容量: $(round(sum(Pmgmax), digits=1)) kW")

    return case
end

function schedule_departure_grid(case::HybridGridCase, transport_node::Int)
    grid_idx = get(case.transport_node_to_grid, transport_node, nothing)
    if grid_idx === nothing
        return "Node?"
    end
    return "Node" * string(grid_idx)
end

schedule_transport_grid_label(case::HybridGridCase, transport_node::Int) = schedule_departure_grid(case, transport_node)

function _setup_mess_mobility(model::Model, case::HybridGridCase, hours::Int, scenario_label::String)
    transport = transportation_network()
    nb_transport = size(transport.bus, 1)
    arcs = TrafficArc[]
    for row_idx in 1:size(transport.branch, 1)
        origin = Int(round(transport.branch[row_idx, 1]))
        destination = Int(round(transport.branch[row_idx, 2]))
        travel_time = max(Int(round(transport.branch[row_idx, 3])), 1)
        if 1 <= origin <= nb_transport && 1 <= destination <= nb_transport
            push!(arcs, TrafficArc(origin, destination, travel_time))
            push!(arcs, TrafficArc(destination, origin, travel_time))
        end
    end

    travel_actions = Dict{Int, Vector{TrafficArc}}()
    for start_time in 0:(hours - 1)
        actions = Vector{TrafficArc}()
        for arc in arcs
            if start_time + arc.travel_time <= hours
                push!(actions, arc)
            end
        end
        if !isempty(actions)
            travel_actions[start_time] = actions
        end
    end

    u_vars = Vector{Matrix{VariableRef}}(undef, hours + 1)
    for t in 0:hours
        u_vars[t + 1] = @variable(model, [m=1:case.nmess, node=1:nb_transport], Bin, base_name="mess_loc[$(scenario_label),$(t)]")
    end

    for (m_idx, start_node) in enumerate(case.mess_transport_initial)
        for node_idx in 1:nb_transport
            value = node_idx == start_node ? 1.0 : 0.0
            fix(u_vars[1][m_idx, node_idx], value; force=true)
        end
    end

    travel_vars = Dict{Int, Matrix{VariableRef}}()
    for (start_time, actions) in travel_actions
        travel_vars[start_time] = @variable(model, [m=1:case.nmess, idx=1:length(actions)], Bin, base_name="mess_travel[$(scenario_label),$(start_time)]")
    end

    departure_lookup = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    arrival_lookup = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    transit_lookup = Dict{Int, Vector{Tuple{Int, Int}}}()

    for (start_time, actions) in travel_actions
        var = travel_vars[start_time]
        for (action_idx, action) in enumerate(actions)
            push!(get!(departure_lookup, (start_time, action.origin), Tuple{Int, Int}[]), (start_time, action_idx))
            arrival_time = start_time + action.travel_time
            if arrival_time <= hours
                push!(get!(arrival_lookup, (arrival_time, action.destination), Tuple{Int, Int}[]), (start_time, action_idx))
            end
            last_active = min(start_time + action.travel_time, hours) - 1
            if start_time + 1 <= last_active
                for active_time in (start_time + 1):last_active
                    push!(get!(transit_lookup, active_time, Tuple{Int, Int}[]), (start_time, action_idx))
                end
            end
            for m_idx in 1:case.nmess
                @constraint(model, var[m_idx, action_idx] <= u_vars[start_time + 1][m_idx, action.origin])
            end
        end
    end

    for t in 0:(hours - 1)
        for node_idx in 1:nb_transport
            departures = get(departure_lookup, (t, node_idx), Tuple{Int, Int}[])
            arrivals = get(arrival_lookup, (t + 1, node_idx), Tuple{Int, Int}[])
            for m_idx in 1:case.nmess
                expr = u_vars[t + 2][m_idx, node_idx] - u_vars[t + 1][m_idx, node_idx]
                for (start_time, action_idx) in departures
                    expr += travel_vars[start_time][m_idx, action_idx]
                end
                for (start_time, action_idx) in arrivals
                    expr -= travel_vars[start_time][m_idx, action_idx]
                end
                @constraint(model, expr == 0)
            end
        end
    end

    return MessMobilitySchedule(u_vars, travel_vars, travel_actions, departure_lookup, arrival_lookup, transit_lookup, nb_transport)
end

function _collect_mobility_log(case::HybridGridCase, schedule::MessMobilitySchedule, hours::Int)
    if case.nmess == 0
        return String[]
    end
    per_mess_paths = [String[] for _ in 1:case.nmess]
    for t in 0:(hours - 1)
        for m_idx in 1:case.nmess
            traveling_action = nothing
            for (start_time, action_index) in get(schedule.transit_lookup, t, Tuple{Int, Int}[])
                val = value(schedule.travel_vars[start_time][m_idx, action_index])
                if val > 0.5
                    traveling_action = schedule.travel_actions[start_time][action_index]
                    break
                end
            end
            entry = if traveling_action !== nothing
                origin_idx = get(case.transport_node_to_grid, traveling_action.origin, nothing)
                destination_idx = get(case.transport_node_to_grid, traveling_action.destination, nothing)
                string(origin_idx === nothing ? "?" : string(origin_idx), "->", destination_idx === nothing ? "?" : string(destination_idx))
            else
                loc_vals = [value(schedule.u_vars[t + 1][m_idx, node]) for node in 1:schedule.nb_transport]
                if any(v -> v > 0.5, loc_vals)
                    transport_node = argmax(loc_vals)
                    grid_idx = get(case.transport_node_to_grid, transport_node, nothing)
                    grid_idx === nothing ? "?" : string(grid_idx)
                else
                    "Unknown"
                end
            end
            push!(per_mess_paths[m_idx], entry)
        end
    end
    log_entries = String[]
    for m_idx in 1:case.nmess
        label = m_idx <= length(case.mess_names) ? case.mess_names[m_idx] : "MESS-" * string(m_idx)
        path_text = join(per_mess_paths[m_idx], ", ")
        push!(log_entries, string(label, " 轨迹: [", path_text, "] (共", length(per_mess_paths[m_idx]), "个元素)"))
    end
    return log_entries
end

function _collect_mess_vectors(case::HybridGridCase, schedule::MessMobilitySchedule, hours::Int, mess_chg_vars::Vector, mess_dis_vars::Vector)
    if case.nmess == 0
        return Dict{String, Dict{String, Vector}}()
    end
    result = Dict{String, Dict{String, Vector}}()
    for m_idx in 1:case.nmess
        name = m_idx <= length(case.mess_names) ? case.mess_names[m_idx] : "MESS-" * string(m_idx)
        location_vector = Int[]
        power_vector = Float64[]
        for t in 0:(hours - 1)
            traveling = false
            for (start_time, action_idx) in get(schedule.transit_lookup, t, Tuple{Int, Int}[])
                if value(schedule.travel_vars[start_time][m_idx, action_idx]) > 0.5
                    traveling = true
                    break
                end
            end
            if traveling
                push!(location_vector, 0)
                push!(power_vector, 0.0)
            else
                loc_vals = [value(schedule.u_vars[t + 1][m_idx, node]) for node in 1:schedule.nb_transport]
                if any(v -> v > 0.5, loc_vals)
                    transport_node = argmax(loc_vals)
                    grid_idx = get(case.transport_node_to_grid, transport_node, nothing)
                    push!(location_vector, grid_idx === nothing ? -1 : grid_idx)
                else
                    push!(location_vector, -1)
                end
                if t + 1 <= length(mess_chg_vars) && t + 1 <= length(mess_dis_vars)
                    total_charge = sum(value(mess_chg_vars[t + 1][m_idx, node]) for node in 1:schedule.nb_transport)
                    total_discharge = sum(value(mess_dis_vars[t + 1][m_idx, node]) for node in 1:schedule.nb_transport)
                    net_power = total_discharge - total_charge
                    push!(power_vector, round(net_power; digits=2))
                else
                    push!(power_vector, 0.0)
                end
            end
        end
        result[name] = Dict("location" => location_vector, "power" => power_vector)
    end
    return result
end

function load_topology_status(topology_path::AbstractString, total_lines::Int, hours::Int; fallback_path::Union{Nothing, AbstractString}=nothing)
    actual_path = topology_path
    use_fallback = false
    if !isfile(topology_path)
        if fallback_path !== nothing && isfile(fallback_path)
            println("[警告] 未找到 $(basename(topology_path))，使用备用文件 $(basename(fallback_path))")
            actual_path = fallback_path
            use_fallback = true
        else
            error("未找到网络拓扑状态文件 $(topology_path)")
        end
    end
    try
        XLSX.openxlsx(actual_path) do ws
            nothing
        end
    catch err
        if fallback_path !== nothing && !use_fallback && isfile(fallback_path)
            println("[警告] $(basename(topology_path)) 无法读取 ($(err))，使用备用文件 $(basename(fallback_path))")
            actual_path = fallback_path
        else
            rethrow(err)
        end
    end

    outage_df = DataFrame(XLSX.readtable(actual_path, "cluster_representatives"; infer_eltypes=true))
    if ncol(outage_df) < 4
        error("cluster_representatives 表结构不正确")
    end
    outage_matrix = Matrix{Float64}(outage_df[:, 5:end])
    if size(outage_matrix, 2) != hours
        error("拓扑文件时间维度为 $(size(outage_matrix, 2))，需为 $(hours)")
    end
    rows = size(outage_matrix, 1)
    if rows % total_lines != 0
        error("拓扑状态行数与线路数不匹配，无法重构场景矩阵")
    end
    scenario_count = rows ÷ total_lines
    matrix = round.(Int, outage_matrix)
    if any(x -> x != 0 && x != 1, matrix)
        error("拓扑状态矩阵应为 0/1 值")
    end
    reshaped = reshape(matrix, total_lines, scenario_count, hours)
    outages = permutedims(reshaped, (2, 3, 1))
    status = 1 .- Float64.(outages)

    summary_df = DataFrame(XLSX.readtable(actual_path, "cluster_summary"))
    sample_col = _find_column(summary_df, ["sample_fraction", "probability", "weight"])
    weights = Float64.(summary_df[!, sample_col])
    if length(weights) != scenario_count
        error("概率向量长度与场景数不一致")
    end
    weight_sum = sum(weights)
    if weight_sum <= 0
        error("概率向量总和必须为正")
    end
    weights ./= weight_sum
    labels = Vector{String}(undef, size(summary_df, 1))
    cluster_col = try
        _find_column(summary_df, ["cluster_id", "cluster"])
    catch
        nothing
    end
    samples_col = try
        _find_column(summary_df, ["sample_count", "samples"])
    catch
        nothing
    end
    for (idx, row) in enumerate(eachrow(summary_df))
        parts = String[]
        if cluster_col !== nothing
            value = row[cluster_col]
            if value !== missing
                push!(parts, "Cluster " * string(value))
            end
        end
        if samples_col !== nothing
            value = row[samples_col]
            if value !== missing
                push!(parts, "samples " * string(value))
            end
        end
        labels[idx] = isempty(parts) ? "Unnamed scenario" : join(parts, " / ")
    end
    return status, weights, labels
end

function optimize_hybrid_dispatch(case::HybridGridCase, statuses, weights::Union{Nothing, AbstractVector}=nothing)
    status_arr = Array{Float64}(statuses)
    if ndims(status_arr) == 2
        status_arr = reshape(status_arr, 1, size(status_arr, 1), size(status_arr, 2))
    elseif ndims(status_arr) != 3
        error("拓扑状态数据必须为2维或3维数组")
    end

    scenario_count, hours, total_lines = size(status_arr)
    expected_lines = case.nl_ac + case.nl_dc + case.nl_vsc
    if total_lines != expected_lines
        error("拓扑状态维度与线路数不一致")
    end

    weights_arr = if weights === nothing
        fill(1.0 / scenario_count, scenario_count)
    else
        w = collect(weights)
        if length(w) != scenario_count
            error("概率向量长度必须与场景数一致")
        end
        total_weight = sum(w)
        if total_weight <= 0
            error("概率向量总和必须为正")
        end
        w ./ total_weight
    end

    ac_range = 1:case.nl_ac
    dc_range = (case.nl_ac + 1):(case.nl_ac + case.nl_dc)
    vsc_range = (case.nl_ac + case.nl_dc + 1):expected_lines

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1)
    set_optimizer_attribute(model, "LogToConsole", 1)
    set_optimizer_attribute(model, "TimeLimit", 360)
    set_optimizer_attribute(model, "MIPGap", 0.01)

    objective_expr = JuMP.AffExpr(0.0)
    scenario_records = Vector{Dict{Symbol, Any}}()

    for scenario_idx in 1:scenario_count
        scenario_status = clamp.(status_arr[scenario_idx, :, :], 0.0, 1.0)
        scenario_weight = weights_arr[scenario_idx]
        status_ac = case.nl_ac > 0 ? scenario_status[:, ac_range] .* reshape(case.alpha_ac, (1, case.nl_ac)) : zeros(Float64, hours, 0)
        status_dc = case.nl_dc > 0 ? scenario_status[:, dc_range] .* reshape(case.alpha_dc, (1, case.nl_dc)) : zeros(Float64, hours, 0)
        status_vsc = case.nl_vsc > 0 ? scenario_status[:, vsc_range] .* reshape(case.alpha_vsc, (1, case.nl_vsc)) : zeros(Float64, hours, 0)

        scenario_mobility::Union{Nothing, MessMobilitySchedule} = nothing
        scenario_soc_vars = Any[]
        scenario_mess_chg_vars = Any[]
        scenario_mess_dis_vars = Any[]
        scenario_shed_vars = Any[]
        scenario_pg_vars = Any[]
        scenario_pm_vars = Any[]
        scenario_prec_vars = Any[]
        scenario_pinv_vars = Any[]
        scenario_no_power_vars = Any[]
        scenario_no_power_violation_vars = Any[]

        if case.nmess > 0
            scenario_mobility = _setup_mess_mobility(model, case, hours, "s$(scenario_idx)")
            soc_init = @variable(model, [m=1:case.nmess], lower_bound=0.0, upper_bound=case.mess_energy_max[m], base_name="soc[$(scenario_idx),0]")
            for m_idx in 1:case.nmess
                @constraint(model, soc_init[m_idx] == case.mess_soc_initial[m_idx])
            end
            push!(scenario_soc_vars, soc_init)
        end

        for t in 0:(hours - 1)
            ac_var = nothing
            dc_var = nothing
            prec_var = nothing
            pinv_var = nothing

            if case.nl_ac > 0
                ac_var = @variable(model, [l=1:case.nl_ac], base_name="p_ac[$(scenario_idx),$(t)]")
                for l in 1:case.nl_ac
                    ub = case.Smax_ac[l] * status_ac[t + 1, l]
                    set_upper_bound(ac_var[l], ub)
                    set_lower_bound(ac_var[l], -ub)
                end
            end

            if case.nl_dc > 0
                dc_var = @variable(model, [l=1:case.nl_dc], base_name="p_dc[$(scenario_idx),$(t)]")
                for l in 1:case.nl_dc
                    ub = case.Smax_dc[l] * status_dc[t + 1, l]
                    set_upper_bound(dc_var[l], ub)
                    set_lower_bound(dc_var[l], -ub)
                end
            end

            pg_var = nothing
            if case.ng > 0
                pg_var = @variable(model, [g=1:case.ng], lower_bound=0.0, upper_bound=case.Pgmax[g], base_name="pg[$(scenario_idx),$(t)]")
                push!(scenario_pg_vars, pg_var)
                for g in 1:case.ng
                    add_to_expression!(objective_expr, scenario_weight * case.c_sg, pg_var[g])
                end
            end

            pm_var = nothing
            if case.nmg > 0
                pm_var = @variable(model, [mg=1:case.nmg], lower_bound=case.Pmgmin[mg], upper_bound=case.Pmgmax[mg], base_name="pm[$(scenario_idx),$(t)]")
                push!(scenario_pm_vars, pm_var)
                for mg in 1:case.nmg
                    add_to_expression!(objective_expr, scenario_weight * case.c_mg, pm_var[mg])
                end
            end

            if case.nl_vsc > 0
                prec_var = @variable(model, [v=1:case.nl_vsc], lower_bound=0.0, base_name="prec[$(scenario_idx),$(t)]")
                pinv_var = @variable(model, [v=1:case.nl_vsc], lower_bound=0.0, base_name="pinv[$(scenario_idx),$(t)]")
                for v in 1:case.nl_vsc
                    ub = case.Pvscmax[v] * status_vsc[t + 1, v]
                    set_upper_bound(prec_var[v], ub)
                    set_upper_bound(pinv_var[v], ub)
                    add_to_expression!(objective_expr, scenario_weight * case.c_vsc, prec_var[v])
                    add_to_expression!(objective_expr, scenario_weight * case.c_vsc, pinv_var[v])
                end
                push!(scenario_prec_vars, prec_var)
                push!(scenario_pinv_vars, pinv_var)
            end

            mess_chg_node = nothing
            mess_dis_node = nothing
            charge_state = nothing
            discharge_state = nothing
            total_charge_exprs = Any[]
            total_discharge_exprs = Any[]
            in_transit_exprs = Any[]

            if case.nmess > 0
                nb_transport = scenario_mobility.nb_transport
                mess_chg_node = @variable(model, [m=1:case.nmess, node=1:nb_transport], lower_bound=0.0, upper_bound=case.mess_charge_max[m], base_name="mess_chg[$(scenario_idx),$(t)]")
                mess_dis_node = @variable(model, [m=1:case.nmess, node=1:nb_transport], lower_bound=0.0, upper_bound=case.mess_discharge_max[m], base_name="mess_dis[$(scenario_idx),$(t)]")
                charge_state = @variable(model, [m=1:case.nmess], Bin, base_name="mess_charge_state[$(scenario_idx),$(t)]")
                discharge_state = @variable(model, [m=1:case.nmess], Bin, base_name="mess_discharge_state[$(scenario_idx),$(t)]")
                push!(scenario_mess_chg_vars, mess_chg_node)
                push!(scenario_mess_dis_vars, mess_dis_node)

                for m_idx in 1:case.nmess
                    location_expr = sum(scenario_mobility.u_vars[t + 1][m_idx, node] for node in 1:nb_transport)
                    transit_terms = [scenario_mobility.travel_vars[start_time][m_idx, action_idx] for (start_time, action_idx) in get(scenario_mobility.transit_lookup, t, Tuple{Int, Int}[])]
                    in_transit = isempty(transit_terms) ? 0.0 : sum(transit_terms)
                    push!(in_transit_exprs, in_transit)
                    @constraint(model, location_expr + in_transit == 1.0)
                    @constraint(model, charge_state[m_idx] <= location_expr)
                    @constraint(model, discharge_state[m_idx] <= location_expr)
                    @constraint(model, charge_state[m_idx] + discharge_state[m_idx] + in_transit <= 1.0)
                    total_charge = sum(mess_chg_node[m_idx, node] for node in 1:nb_transport)
                    total_discharge = sum(mess_dis_node[m_idx, node] for node in 1:nb_transport)
                    push!(total_charge_exprs, total_charge)
                    push!(total_discharge_exprs, total_discharge)
                    @constraint(model, total_charge <= case.mess_charge_max[m_idx] * charge_state[m_idx])
                    @constraint(model, total_discharge <= case.mess_discharge_max[m_idx] * discharge_state[m_idx])
                    for node in 1:nb_transport
                        @constraint(model, mess_chg_node[m_idx, node] <= case.mess_charge_max[m_idx] * scenario_mobility.u_vars[t + 1][m_idx, node])
                        @constraint(model, mess_dis_node[m_idx, node] <= case.mess_discharge_max[m_idx] * scenario_mobility.u_vars[t + 1][m_idx, node])
                    end
                end
            end

            shed_var = @variable(model, [node=1:case.nb], lower_bound=0.0, upper_bound=case.load_per_node[node] + 1e-6, base_name="shed[$(scenario_idx),$(t)]")
            push!(scenario_shed_vars, shed_var)
            for node in 1:case.nb
                add_to_expression!(objective_expr, scenario_weight * case.c_load, shed_var[node])
            end

            no_power_var = @variable(model, [node=1:case.nb], Bin, base_name="no_power[$(scenario_idx),$(t)]")
            push!(scenario_no_power_vars, no_power_var)
            for node in 1:case.nb
                load_value = case.load_per_node[node]
                if load_value <= NO_POWER_ABS_TOL
                    fix(no_power_var[node], 0.0; force=true)
                    continue
                end
                tolerance = max(NO_POWER_ABS_TOL, NO_POWER_REL_TOL * load_value)
                powered_upper = max(load_value - tolerance, 0.0)
                outage_lower = max(load_value - NO_POWER_ABS_TOL, 0.0)
                bigM = load_value + NO_POWER_ABS_TOL
                @constraint(model, shed_var[node] <= powered_upper + bigM * no_power_var[node])
                if outage_lower > 0
                    @constraint(model, shed_var[node] >= outage_lower - bigM * (1.0 - no_power_var[node]))
                end
            end

            node_expr = [JuMP.AffExpr(0.0) for _ in 1:case.nb]

            if case.nl_ac > 0 && ac_var !== nothing
                for (line_idx, entries) in enumerate(case.Cft_ac_rows)
                    for (node_idx, coeff) in entries
                        add_to_expression!(node_expr[node_idx], coeff, ac_var[line_idx])
                    end
                end
            end

            if case.nl_dc > 0 && dc_var !== nothing
                for (line_idx, entries) in enumerate(case.Cft_dc_rows)
                    for (node_idx, coeff) in entries
                        add_to_expression!(node_expr[node_idx], coeff, dc_var[line_idx])
                    end
                end
            end

            if case.nl_vsc > 0 && prec_var !== nothing && pinv_var !== nothing
                for (v_idx, entries) in enumerate(case.Cft_vsc_rows)
                    for (node_idx, coeff) in entries
                        if node_idx <= case.nb_ac
                            add_to_expression!(node_expr[node_idx], coeff, prec_var[v_idx])
                            add_to_expression!(node_expr[node_idx], -coeff * case.eta_vsc[v_idx], pinv_var[v_idx])
                        else
                            add_to_expression!(node_expr[node_idx], coeff * case.eta_vsc[v_idx], prec_var[v_idx])
                            add_to_expression!(node_expr[node_idx], -coeff, pinv_var[v_idx])
                        end
                    end
                end
            end

            if case.ng > 0 && pg_var !== nothing
                for (g_idx, node_idx) in enumerate(case.generator_nodes)
                    add_to_expression!(node_expr[node_idx], 1.0, pg_var[g_idx])
                end
            end

            if case.nmg > 0 && pm_var !== nothing
                for (mg_idx, node_idx) in enumerate(case.microgrid_nodes)
                    add_to_expression!(node_expr[node_idx], 1.0, pm_var[mg_idx])
                end
            end

            if case.nmess > 0 && mess_chg_node !== nothing && mess_dis_node !== nothing
                for (transport_node, grid_node) in case.transport_node_to_grid
                    for m_idx in 1:case.nmess
                        add_to_expression!(node_expr[grid_node], 1.0, mess_dis_node[m_idx, transport_node])
                        add_to_expression!(node_expr[grid_node], -1.0, mess_chg_node[m_idx, transport_node])
                    end
                end
            end

            for node_idx in 1:case.nb
                add_to_expression!(node_expr[node_idx], 1.0, shed_var[node_idx])
                add_to_expression!(node_expr[node_idx], -case.load_per_node[node_idx])
                @constraint(model, node_expr[node_idx] == 0.0)
            end

            if case.nmess > 0
                prev_soc = scenario_soc_vars[end]
                next_soc = @variable(model, [m=1:case.nmess], lower_bound=0.0, upper_bound=case.mess_energy_max[m], base_name="soc[$(scenario_idx),$(t+1)]")
                for m_idx in 1:case.nmess
                    total_charge = total_charge_exprs[m_idx]
                    total_discharge = total_discharge_exprs[m_idx]
                    in_transit = in_transit_exprs[m_idx]
                    @constraint(model, next_soc[m_idx] == prev_soc[m_idx] + case.mess_eta_charge[m_idx] * total_charge * TIME_STEP_HOURS - (total_discharge / case.mess_eta_discharge[m_idx]) * TIME_STEP_HOURS - MESS_TRAVEL_ENERGY_LOSS_PER_HOUR * in_transit * TIME_STEP_HOURS)
                end
                push!(scenario_soc_vars, next_soc)
            end
        end

        if case.nmess > 0
            final_soc = scenario_soc_vars[end]
            for m_idx in 1:case.nmess
                min_terminal = case.mess_soc_initial[m_idx] * 0.5
                max_terminal = min(case.mess_soc_initial[m_idx] * 1.5, case.mess_energy_max[m_idx])
                @constraint(model, final_soc[m_idx] >= min_terminal)
                @constraint(model, final_soc[m_idx] <= max_terminal)
            end
        end

        if !isempty(scenario_no_power_vars) && hours >= NO_POWER_WINDOW
            for node in 1:case.nb
                for window_start in 1:(hours - NO_POWER_WINDOW + 1)
                    viol = @variable(model, lower_bound=0.0, base_name="no_power_violation[$(scenario_idx),$(node),$(window_start)]")
                    window_sum = JuMP.AffExpr(0.0)
                    for offset in 0:(NO_POWER_WINDOW - 1)
                        add_to_expression!(window_sum, 1.0, scenario_no_power_vars[window_start + offset][node])
                    end
                    @constraint(model, window_sum <= NO_POWER_MAX_CONSECUTIVE_HOURS + viol)
                    add_to_expression!(objective_expr, scenario_weight * NO_POWER_SOFT_PENALTY, viol)
                    push!(scenario_no_power_violation_vars, viol)
                end
            end
        end

        push!(scenario_records, Dict(
            :weight => scenario_weight,
            :hours => hours,
            :mobility_schedule => scenario_mobility,
            :mess_chg_vars => scenario_mess_chg_vars,
            :mess_dis_vars => scenario_mess_dis_vars,
            :shed_vars => scenario_shed_vars,
            :pg_vars => scenario_pg_vars,
            :pm_vars => scenario_pm_vars,
            :prec_vars => scenario_prec_vars,
            :pinv_vars => scenario_pinv_vars,
            :soc_vars => scenario_soc_vars,
            :no_power_vars => scenario_no_power_vars,
            :no_power_violation_vars => scenario_no_power_violation_vars,
        ))
    end

    @objective(model, Min, objective_expr)
    optimize!(model)

    status = termination_status(model)
    if status == MOI.OPTIMAL
        nothing
    elseif status == MOI.TIME_LIMIT && JuMP.has_values(model)
        gap_value = objective_bound(model)
        println("Warning: Gurobi hit TIME_LIMIT but produced a feasible solution. Best bound: $(gap_value)")
    else
        error("调度模型求解失败，状态 $(status)")
    end

    scenario_results = Vector{Dict{String, Any}}()
    for record in scenario_records
        hours = record[:hours]
        shed_total = sum(sum(value.(var)) for var in record[:shed_vars])
        load_demand_total = sum(case.load_per_node) * hours
        generation_total = isempty(record[:pg_vars]) ? 0.0 : sum(sum(value.(var)) for var in record[:pg_vars])
        microgrid_total = isempty(record[:pm_vars]) ? 0.0 : sum(sum(value.(var)) for var in record[:pm_vars])
        prec_total = isempty(record[:prec_vars]) ? 0.0 : sum(sum(value.(var)) for var in record[:prec_vars])
        pinv_total = isempty(record[:pinv_vars]) ? 0.0 : sum(sum(value.(var)) for var in record[:pinv_vars])
        soc_terminal = if case.nmess > 0 && !isempty(record[:soc_vars])
            value.(record[:soc_vars][end])
        else
            zeros(Float64, 0)
        end
        mobility_log = String[]
        mess_vectors = Dict{String, Dict{String, Vector}}()
        if case.nmess > 0 && record[:mobility_schedule] !== nothing
            mobility_log = _collect_mobility_log(case, record[:mobility_schedule], hours)
            mess_vectors = _collect_mess_vectors(case, record[:mobility_schedule], hours, record[:mess_chg_vars], record[:mess_dis_vars])
        end
        no_power_violations = Dict{Int, Int}()
        if haskey(record, :no_power_vars) && !isempty(record[:no_power_vars])
            num_steps = length(record[:no_power_vars])
            for node in 1:case.nb
                values = [value(record[:no_power_vars][t][node]) for t in 1:num_steps]
                max_run = _max_consecutive_true(values)
                if max_run > NO_POWER_MAX_CONSECUTIVE_HOURS
                    no_power_violations[node] = max_run
                end
            end
        end
        pg_cost = case.c_sg * generation_total
        pm_cost = case.c_mg * microgrid_total
        vsc_cost = case.c_vsc * (prec_total + pinv_total)
        load_cost = case.c_load * shed_total
        scenario_objective = pg_cost + pm_cost + vsc_cost + load_cost
        served = load_demand_total - shed_total
        served_ratio = load_demand_total > 0 ? served / load_demand_total : 1.0
        push!(scenario_results, Dict(
            "probability" => record[:weight],
            "load_demand_total" => load_demand_total,
            "load_shed_total" => shed_total,
            "generation_total" => generation_total,
            "microgrid_total" => microgrid_total,
            "mess_soc_terminal" => soc_terminal,
            "mobility_log" => mobility_log,
            "mess_vectors" => mess_vectors,
            "served_ratio" => served_ratio,
            "objective" => scenario_objective,
            "no_power_violations" => no_power_violations,
        ))
    end

    expected_load_shed = sum(detail["probability"] * detail["load_shed_total"] for detail in scenario_results)
    expected_supply_ratio = sum(detail["probability"] * detail["served_ratio"] for detail in scenario_results)

    result = Dict(
        "objective" => objective_value(model),
        "expected_load_shed_total" => expected_load_shed,
        "expected_supply_ratio" => expected_supply_ratio,
        "scenario_results" => scenario_results,
    )

    return model, result
end

function run_mess_dispatch_julia(; case_path::AbstractString=DEFAULT_CASE_XLSX,
        topology_path::AbstractString=DEFAULT_TOPOLOGY_XLSX,
        fallback_topology::AbstractString=DEFAULT_MC_XLSX,
        hours::Int=DEFAULT_HOURS,
        mess_configs::Vector{MESSConfig}=DEFAULT_MESS)
    println("\n" * "="^60)
    println("混合配电网 + MESS 协同调度 (Julia)")
    println("="^60)

    case = load_hybrid_case(case_path, mess_configs)
    total_lines = case.nl_ac + case.nl_dc + case.nl_vsc
    statuses, weights, labels = load_topology_status(topology_path, total_lines, hours; fallback_path=fallback_topology)
    if isempty(statuses)
        error("未加载到任何拓扑场景")
    end

    model, result = optimize_hybrid_dispatch(case, statuses, weights)
    scenario_details = result["scenario_results"]
    total_scenarios = length(scenario_details)
    node_violation_prob = Dict{Int, Float64}()
    node_violation_breakdown = Dict{Int, Dict{Int, Float64}}()

    for (idx, detail) in enumerate(scenario_details)
        label = idx <= length(labels) ? labels[idx] : "Scenario $(idx)"
        prob = detail["probability"]
        println("\n" * "="^60)
        println("场景 $(idx)/$(total_scenarios) ($(label))")
        println("="^60)
        @printf("场景概率: %.4f\n", prob)
        @printf("调度模型目标值: %.4f\n", detail["objective"])
        @printf("总负荷需求: %.4f kW·h\n", detail["load_demand_total"])
        @printf("削减负荷: %.4f kW·h (供电率 %.2f%%)\n", detail["load_shed_total"], detail["served_ratio"] * 100)

        if case.nmess > 0
            soc_terminal = detail["mess_soc_terminal"]
            soc_text = join([@sprintf("%.2f", val) for val in soc_terminal], ", ")
            println("MESS 末端 SOC: [$(soc_text)] kW·h")
            println("\n--- MESS 向量化结果 ($(hours)时段) ---")
            mess_vectors = detail["mess_vectors"]
            for (mess_name, vectors) in mess_vectors
                location = vectors["location"]
                power = vectors["power"]
                println()
                println("$(mess_name) Location: $(location)")
                println("$(mess_name) Power (kW): $(power)")
                unique_locs = sort(unique(filter(x -> x > 0, location)))
                if length(unique_locs) > 1
                    println("  → ✅ $(mess_name) 发生了移动! 访问了节点: $(unique_locs)")
                else
                    println("  → ⚠️ $(mess_name) 未移动，始终在节点 $(unique_locs)")
                end
            end
        end

        violations = detail["no_power_violations"]
        if !isempty(violations)
            println("\n⚠️ 场景 $(idx) 中连续断电超过 $(NO_POWER_MAX_CONSECUTIVE_HOURS) 小时的节点：")
            for node_id in sort(collect(keys(violations)))
                duration = violations[node_id]
                println("  - 节点 $(node_id): 最长 $(duration) 小时无电")
                scenario_prob = prob
                stats = get!(node_violation_breakdown, node_id, Dict{Int, Float64}())
                stats[duration] = get(stats, duration, 0.0) + scenario_prob
                node_violation_prob[node_id] = get(node_violation_prob, node_id, 0.0) + scenario_prob
            end
        else
            println("\n场景 $(idx) 中所有节点均满足“任意节点连续断电不超过$(NO_POWER_MAX_CONSECUTIVE_HOURS)小时”的软约束。")
        end
    end

    println("\n" * "="^60)
    println("全局连续断电超标节点（汇总全部场景）")
    println("="^60)
    if isempty(node_violation_prob)
        println("  - 所有节点在全部场景中均满足连续断电约束。")
    else
        for node_id in sort(collect(keys(node_violation_prob)))
            stats = node_violation_breakdown[node_id]
            longest = maximum(keys(stats))
            total_prob = node_violation_prob[node_id]
            @printf("  - 节点 %d: 最长连续 %d 小时，累计超标概率 %.4f\n", node_id, longest, total_prob)
        end
    end

    println("\n" * "="^60)
    println("节点连续断电概率统计（汇总全部场景）")
    println("="^60)
    for node_id in 1:case.nb
        total_prob = get(node_violation_prob, node_id, 0.0)
        compliance_prob = max(0.0, 1.0 - total_prob)
        stats = get(node_violation_breakdown, node_id, nothing)
        if stats === nothing
            @printf("  - 节点 %d: 始终满足约束 (概率 %.4f)\n", node_id, 1.0)
        else
            @printf("  - 节点 %d: 超标概率 %.4f, 满足概率 %.4f\n", node_id, total_prob, compliance_prob)
            for duration in sort(collect(keys(stats)))
                @printf("      · 最长 %d 小时: 场景概率 %.4f\n", duration, stats[duration])
            end
        end
    end

    println("\n" * "="^60)
    println("总结")
    println("="^60)
    @printf("期望调度目标: %.4f\n", result["objective"])
    @printf("期望削减负荷: %.4f kW·h\n", result["expected_load_shed_total"])
    @printf("期望供电率: %.2f%%\n", result["expected_supply_ratio"] * 100)

    return model, result
end

function main()
    run_mess_dispatch_julia()
end
