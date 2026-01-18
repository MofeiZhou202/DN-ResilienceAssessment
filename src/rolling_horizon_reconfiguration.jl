using DataFrames
using JSON
using XLSX
using Logging
using Dates

# 注意：utils 文件已在 workflows.jl 中统一加载，这里不再重复加载
include(joinpath(@__DIR__, "three_stage_tp_model.jl"))

const DEFAULT_CASE_FILE = joinpath(@__DIR__, "..", "data", "ac_dc_real_case.xlsx")
const DEFAULT_FAULT_FILE = joinpath(@__DIR__, "..", "data", "mc_simulation_results_k100_clusters.xlsx")
const DEFAULT_STAGE_FILE = joinpath(@__DIR__, "..", "data", "scenario_phase_classification.xlsx")
const DEFAULT_OUTPUT_FILE = joinpath(@__DIR__, "..", "data", "topology_reconfiguration_results.xlsx")
const DEFAULT_FAULT_SHEET = "cluster_representatives"
const DEFAULT_STAGE_SHEET = "StageDetails"
const EXPECTED_AC_LINES = 26
const EXPECTED_DC_LINES = 2
const EXPECTED_VSC_LINES = 7
const EXPECTED_TOTAL_LINES = EXPECTED_AC_LINES + EXPECTED_DC_LINES + EXPECTED_VSC_LINES
const SIMULATION_STEPS = 48
const NORMALLY_OPEN_LINE_INDICES = Set([24, 25, 26, 29, 31])
const DEFAULT_LINES_PER_SCENARIO = EXPECTED_TOTAL_LINES

const STAGE_NAMES = Dict(
    0 => "阶段0 - 基准拓扑",
    1 => "阶段1 - 故障聚集",
    2 => "阶段2 - 前期恢复",
    3 => "阶段3 - 恢复后"
)

function stage_label(stage::Int)
    return get(STAGE_NAMES, stage, "阶段未知")
end

"""
    load_case(filepath::String)

使用新的数据加载流程：先通过 ETAPImporter 加载 Excel 数据，然后转换为 JPC_tp 格式。

参数:
- `filepath::String`: Excel 文件路径

返回:
- JPC_tp 结构体，包含转换后的电力系统数据
"""
function load_case(filepath::String)
    println("正在加载电网数据: $filepath")
    
    # 第一步：使用 ETAPImporter 加载 Excel 数据到 JuliaPowerCase 结构体
    julia_case = load_julia_power_data(filepath)
    
    # 第二步：将 JuliaPowerCase 转换为 JPC_tp 格式
    jpc_tp, C_sparse = JuliaPowerCase2Jpc_tp(julia_case)
    
    # 将稀疏矩阵添加到 jpc_tp 中（兼容原有逻辑）
    jpc_tp["Cft"] = C_sparse
    
    println("电网数据加载完成")
    return jpc_tp
end

function baseline_topology_result(jpc)
    β0 = Int.(clamp01.(round.(jpc[:α_pre])))
    return Dict(:β0 => β0, :status => "Baseline", :objective => missing)
end

function column_letter_to_index(letter::AbstractString)
    letter = uppercase(letter)
    result = 0
    for ch in letter
        if ch < 'A' || ch > 'Z'
            error("Invalid column letter: $letter")
        end
        result = result * 26 + (Int(ch) - Int('A') + 1)
    end
    return result
end

function parse_cell_reference(ref::AbstractString)
    letter_part = match(r"[A-Z]+", ref)
    number_part = match(r"\d+", ref)
    if letter_part === nothing || number_part === nothing
        error("Cannot parse cell reference: $ref")
    end
    column = column_letter_to_index(letter_part.match)
    row = parse(Int, number_part.match)
    return column, row
end

clamp01(value) = clamp(value, 0, 1)

function normalize_cell_value(cell)
    if cell === nothing
        return 0
    end
    value = cell isa XLSX.Cell ? cell.value : cell
    if value === nothing
        return 0
    end
    if value isa Number
        return clamp01(Int(round(value)))
    elseif value isa Bool
        return value ? 1 : 0
    else
        cleaned = strip(string(value))
        if isempty(cleaned)
            return 0
        end
        parsed = try
            parse(Int, cleaned)
        catch
            0
        end
        return clamp01(parsed)
    end
end

function validate_line_indexing!(jpc)
    total_lines = jpc[:nl] + jpc[:nl_vsc]
    if total_lines != EXPECTED_TOTAL_LINES
        error("基础模型的线路总数($total_lines)不等于期望的 $EXPECTED_TOTAL_LINES 条线路，请检查 ac_dc_real_case.xlsx 的 AC/DC/VSC 排列顺序。")
    end
    if jpc[:nl_ac] != EXPECTED_AC_LINES || jpc[:nl_dc] != EXPECTED_DC_LINES || jpc[:nl_vsc] != EXPECTED_VSC_LINES
        error("线路类别数量不匹配：AC=" * string(jpc[:nl_ac]) * ", DC=" * string(jpc[:nl_dc]) * ", VSC=" * string(jpc[:nl_vsc]) * "。期望的顺序是 26 AC、2 DC、7 VSC。")
    end
    α_pre = jpc[:α_pre]
    if length(α_pre) != total_lines
        error("α_pre 向量长度 $(length(α_pre)) 与线路数量 $total_lines 不一致。请核对基础数据。")
    end
    zero_positions = Set(findall(x -> x == 0, α_pre))
    missing_normally_open = setdiff(NORMALLY_OPEN_LINE_INDICES, zero_positions)
    if !isempty(missing_normally_open)
        error("基础状态下应该常开（Normally Open）的线路索引 $missing_normally_open 未在 α_pre 中被标为 0。实际常开索引: $(sort(collect(zero_positions))).")
    end
    @info "线路索引顺序通过校验" total_lines=total_lines
    return total_lines
end

function ensure_initial_line_state!(fault_matrix)
    if size(fault_matrix, 1) < EXPECTED_TOTAL_LINES
        error("故障轨迹矩阵的行数 $(size(fault_matrix, 1)) 小于期望的 $(EXPECTED_TOTAL_LINES) 条线路。")
    end
    initial = fault_matrix[1:EXPECTED_TOTAL_LINES, 1]
    zero_positions = Set(findall(x -> x == 0, initial))
    missing = setdiff(NORMALLY_OPEN_LINE_INDICES, zero_positions)
    if !isempty(missing)
        error("故障序列第一列并未保持应当常开的线路 $missing 处于 0 状态，可能行顺序错误或数据对应错误。0 值位置: $(sort(collect(zero_positions))).")
    end
    @info "初始状态第一时间步已验证常开线路" open_lines=sort(collect(zero_positions))
end

function build_line_column_symbols()
    ac_cols = [Symbol("AC_Line_$(i)") for i in 1:EXPECTED_AC_LINES]
    dc_cols = [Symbol("DC_Line_$(i)") for i in 1:EXPECTED_DC_LINES]
    vsc_cols = [Symbol("VSC_Line_$(i)") for i in 1:EXPECTED_VSC_LINES]
    return vcat(ac_cols, dc_cols, vsc_cols)
end

function summarize_line_status(line_columns, row, status)
    matched = String[]
    for col in line_columns
        val = row[col]
        if val == status
            push!(matched, string(col))
        end
    end
    return isempty(matched) ? "" : join(matched, ", ")
end

function read_fault_matrix(input_path::AbstractString, sheet_name::AbstractString;
        start_row::Int = 2,
        start_col::Int = column_letter_to_index("E"),
        stop_col::Int = column_letter_to_index("AZ"),
        lines_per_scenario::Int = DEFAULT_LINES_PER_SCENARIO)
    matrix = nothing
    scenario_count = 0
    XLSX.openxlsx(input_path) do workbook
        if !(sheet_name in XLSX.sheetnames(workbook))
            error("Sheet $(sheet_name) not found in $(input_path)")
        end
        sheet = XLSX.getsheet(workbook, sheet_name)
        dims = sheet.dimension
        dim_str = dims === nothing ? "" : string(dims)
        if isempty(strip(dim_str))
            error("Sheet $(sheet_name) appears empty")
        end
        _, end_ref = split(dim_str, ":")
        actual_end_col, actual_end_row = parse_cell_reference(end_ref)
        end_col = min(stop_col, actual_end_col)
        if end_col < start_col
            error("Computed column range is invalid: $start_col - $end_col")
        end
        available_rows = actual_end_row - start_row + 1
        if available_rows < lines_per_scenario
            error("Not enough rows after row $start_row for $lines_per_scenario lines per scenario")
        end
        usable_rows = lines_per_scenario * fld(available_rows, lines_per_scenario)
        if usable_rows == 0
            error("Unable to split rows evenly into scenarios with $lines_per_scenario lines per scenario")
        end
        scenario_count = usable_rows ÷ lines_per_scenario
        n_cols = end_col - start_col + 1
        matrix = Matrix{Int}(undef, usable_rows, n_cols)
        for offset_row in 1:usable_rows
            excel_row = start_row + offset_row - 1
            for offset_col in 1:n_cols
                excel_col = start_col + offset_col - 1
                cell = sheet[excel_row, excel_col]
                matrix[offset_row, offset_col] = normalize_cell_value(cell)
            end
        end
    end
    return matrix, scenario_count
end

function safe_int(value)
    if value === nothing
        return missing
    end
    if value isa Missing
        return missing
    end
    if value isa Number
        return Int(round(value))
    end
    cleaned = strip(string(value))
    if isempty(cleaned)
        return missing
    end
    try
        return parse(Int, cleaned)
    catch
        return missing
    end
end

function normalized_column_name(name)
    return replace(lowercase(string(name)), r"\s+" => "")
end

function find_column_name(df::DataFrame, candidates::Vector{String})
    candidates_norm = map(x -> normalized_column_name(x), candidates)
    for name in names(df)
        if normalized_column_name(name) in candidates_norm
            return name
        end
    end
    error("No matching column found for candidates $(candidates)")
end

function build_stage_schedule(stage_file::AbstractString, stage_sheet::AbstractString,
        scenario_count::Int, n_steps::Int)
    df = DataFrame(XLSX.readtable(stage_file, stage_sheet))
    scenario_col = find_column_name(df, ["Scenario", "scenario"])
    time_col = find_column_name(df, ["TimeStep", "Time", "Timesteps", "时间"])
    stage_col = find_column_name(df, ["Stage", "stage"])
    schedule = Dict{Int, Vector{Int}}()
    for scenario in 1:scenario_count
        schedule[scenario] = fill(3, n_steps)
    end
    for row in eachrow(df)
        scenario = safe_int(row[scenario_col])
        time_step = safe_int(row[time_col])
        stage_value = safe_int(row[stage_col])
        if scenario isa Int && time_step isa Int && stage_value isa Int
            if 1 <= scenario <= scenario_count && 1 <= time_step <= n_steps
                schedule[scenario][time_step] = clamp(stage_value, 0, 3)
            end
        end
    end
    return schedule
end

function write_dataframe_sheet(workbook, sheet_name::String, df::DataFrame)
    sheet = XLSX.addsheet!(workbook, sheet_name)
    for (col_idx, col_name) in enumerate(names(df))
        XLSX.setdata!(sheet, 1, col_idx, string(col_name))
    end
    for row_idx in 1:nrow(df)
        for col_idx in 1:ncol(df)
            XLSX.setdata!(sheet, row_idx + 1, col_idx, df[row_idx, col_idx])
        end
    end
    return sheet
end

function copy_named_sheet!(workbook, source_path::AbstractString, sheet_name::AbstractString)
    XLSX.openxlsx(source_path) do source_workbook
        sheet_names = XLSX.sheetnames(source_workbook)
        if !(sheet_name in sheet_names)
            error("输入文件 $(source_path) 中未找到名为 $(sheet_name) 的工作表，无法复制。")
        end
        source_sheet = XLSX.getsheet(source_workbook, sheet_name)
        dest_sheet = XLSX.addsheet!(workbook, sheet_name)
        dims = source_sheet.dimension
        if dims === nothing
            return dest_sheet
        end
        dim_str = strip(string(dims))
        if isempty(dim_str)
            return dest_sheet
        end
        refs = split(dim_str, ":")
        start_ref = refs[1]
        end_ref = refs[end]
        start_col, start_row = parse_cell_reference(start_ref)
        end_col, end_row = parse_cell_reference(end_ref)
        for row in start_row:end_row
            for col in start_col:end_col
                cell = source_sheet[row, col]
                value = cell isa XLSX.Cell ? cell.value : cell
                if value !== nothing
                    XLSX.setdata!(dest_sheet, row - start_row + 1, col - start_col + 1, value)
                end
            end
        end
        return dest_sheet
    end
end

function write_reconfiguration_report(output_path::AbstractString, template_path::AbstractString,
    template_sheet::AbstractString, results_matrix::Matrix{Int}, raw_df::DataFrame;
    cluster_summary_source::Union{Nothing, AbstractString}=nothing,
    cluster_summary_sheet::AbstractString="cluster_summary")
    template_df = DataFrame(XLSX.readtable(template_path, template_sheet))
    expected_rows = size(results_matrix, 1)
    if nrow(template_df) != expected_rows
        error("模板行数 $(nrow(template_df)) 与结果行数 $expected_rows 不匹配，请检查数据源。")
    end
    data_columns = names(template_df)[5:end]
    if length(data_columns) != size(results_matrix, 2)
        error("模板数据列数 $(length(data_columns)) 与时间步数 $(size(results_matrix, 2)) 不一致。")
    end
    for (col_idx, col_name) in enumerate(data_columns)
        template_df[!, col_name] = results_matrix[:, col_idx]
    end
    XLSX.openxlsx(output_path, mode = "w") do workbook
        write_dataframe_sheet(workbook, template_sheet, template_df)
        write_dataframe_sheet(workbook, "RollingDecisionsOriginal", raw_df)
        if cluster_summary_source !== nothing
            copy_named_sheet!(workbook, cluster_summary_source, cluster_summary_sheet)
        end
    end
end

function run_rolling_reconfiguration(; case_file::AbstractString = DEFAULT_CASE_FILE,
        fault_file::AbstractString = DEFAULT_FAULT_FILE,
        stage_file::AbstractString = DEFAULT_STAGE_FILE,
        output_file::AbstractString = DEFAULT_OUTPUT_FILE,
        fault_sheet::AbstractString = DEFAULT_FAULT_SHEET,
        stage_sheet::AbstractString = DEFAULT_STAGE_SHEET,
        lines_per_scenario::Int = DEFAULT_LINES_PER_SCENARIO)
    println("正在加载拓扑数据和故障轨迹...")
    jpc = load_case(case_file)
    total_lines = validate_line_indexing!(jpc)
    fault_matrix, scenario_count = read_fault_matrix(fault_file, fault_sheet;
        lines_per_scenario=lines_per_scenario)
    if size(fault_matrix, 2) < SIMULATION_STEPS
        error("故障轨迹只有 $(size(fault_matrix, 2)) 个时间步，但滚动仿真需要 $SIMULATION_STEPS 小时的数据。")
    end
    fault_matrix = fault_matrix[:, 1:SIMULATION_STEPS]
    ensure_initial_line_state!(fault_matrix)
    n_steps = SIMULATION_STEPS
    stage_schedule = build_stage_schedule(stage_file, stage_sheet, scenario_count, n_steps)
    baseline_result = baseline_topology_result(jpc)
    line_columns = build_line_column_symbols()
    if length(line_columns) != total_lines
        error("构建的列标签数量 $(length(line_columns)) 与基础模型的线路数量 $total_lines 不一致。")
    end
    if lines_per_scenario != total_lines
        error("lines_per_scenario=$lines_per_scenario 与基础模型的线路数 $total_lines 不一致。")
    end
    scenario_col = Int[]
    time_col = Int[]
    stage_col = Int[]
    stage_label_col = String[]
    fault_count_col = Int[]
    objective_col = Vector{Union{Missing, Float64}}()
    status_col = Vector{Union{Missing, String}}()
    line_buffers = [Int[] for _ in 1:total_lines]
    status_tensor = Array{Int}(undef, lines_per_scenario, scenario_count, n_steps)

    for scenario in 1:scenario_count
        println("处理中: 场景 $scenario / $scenario_count")
        block_start = (scenario - 1) * lines_per_scenario + 1
        block_end = scenario * lines_per_scenario
        scenario_block = fault_matrix[block_start:block_end, :]
        last_stage1 = nothing
        last_stage2 = nothing
        for t in 1:n_steps
            step_vector = scenario_block[:, t]
            fault_lines = findall(x -> x == 1, step_vector)
            stage = clamp(stage_schedule[scenario][t], 0, 3)
            println("  运行日志 -> 场景 $scenario, 时间步 $t, 阶段 $(stage_label(stage)), 故障数量 $(length(fault_lines))")
            stage_result = nothing
            stage_obj = missing
            stage_status = missing
            if stage == 0
                stage_result = baseline_result
                stage_status = get(stage_result, :status, missing)
            elseif stage == 1
                last_stage1 = solve_fault_isolation(jpc, fault_lines)
                stage_result = last_stage1
                stage_obj = get(last_stage1, :objective, missing)
                stage_status = get(last_stage1, :status, missing)
            elseif stage == 2
                if last_stage1 === nothing
                    last_stage1 = solve_fault_isolation(jpc, fault_lines)
                end
                last_stage2 = solve_post_fault_reconfig(jpc, last_stage1)
                stage_result = last_stage2
                stage_status = get(last_stage2, :status, missing)
            elseif stage == 3
                if last_stage1 === nothing
                    last_stage1 = solve_fault_isolation(jpc, fault_lines)
                end
                if last_stage2 === nothing
                    last_stage2 = solve_post_fault_reconfig(jpc, last_stage1)
                end
                stage_result = solve_post_repair_reconfig(jpc, fault_lines, last_stage1, last_stage2)
                stage_obj = get(stage_result, :objective, missing)
                stage_status = get(stage_result, :status, missing)
            else
                @warn "未识别的阶段" stage=stage
            end
            decision_beta = stage_result === nothing ? zeros(Int, total_lines) : begin
                beta_key = stage == 0 ? :β0 : stage == 1 ? :β1 : stage == 2 ? :β2 : :β3
                beta_vector = get(stage_result, beta_key, zeros(total_lines))
                Int.(clamp01.(round.(beta_vector)))
            end
            if length(decision_beta) != total_lines
                @warn "Beta vector length mismatch" total_length=total_lines beta_length=length(decision_beta)
                decision_beta = resize!(Vector{Int}(decision_beta), total_lines)
            end
            push!(scenario_col, scenario)
            push!(time_col, t)
            push!(stage_col, stage)
            push!(stage_label_col, stage_label(stage))
            push!(fault_count_col, sum(step_vector))
            push!(objective_col, stage_obj)
            push!(status_col, stage_status)
            for (i, value) in enumerate(decision_beta)
                push!(line_buffers[i], value)
                status_tensor[i, scenario, t] = value
            end
        end
    end

    results_df = DataFrame(
        Scenario = scenario_col,
        TimeStep = time_col,
        Stage = stage_col,
        StageLabel = stage_label_col,
        FaultCount = fault_count_col,
        Objective = objective_col,
        Status = status_col
    )
    for (i, col_name) in enumerate(line_columns)
        results_df[!, col_name] = line_buffers[i]
    end
    results_df[!, :ClosedLines] = map(eachrow(results_df)) do row
        summarize_line_status(line_columns, row, 1)
    end
    results_df[!, :OpenLines] = map(eachrow(results_df)) do row
        summarize_line_status(line_columns, row, 0)
    end
    results_matrix = Matrix{Int}(undef, scenario_count * lines_per_scenario, n_steps)
    for scenario in 1:scenario_count
        for line in 1:lines_per_scenario
            row_idx = (scenario - 1) * lines_per_scenario + line
            results_matrix[row_idx, :] = status_tensor[line, scenario, :]
        end
    end
    # Retain original status tensor internally but flip values for the exported report
    flipped_matrix = 1 .- results_matrix
    write_reconfiguration_report(output_file, fault_file, fault_sheet, flipped_matrix, results_df;
        cluster_summary_source=fault_file,
        cluster_summary_sheet="cluster_summary")
    color_stage_font_via_python(output_file, stage_schedule, scenario_count, lines_per_scenario, n_steps)
    println("滚动拓扑重构完成，结果已写入：", output_file)
    return results_df
end

function color_stage_font_via_python(output_path::AbstractString,
        stage_schedule::Dict{Int, Vector{Int}},
        scenario_count::Int,
        lines_per_scenario::Int,
        n_steps::Int)
    stage_matrix = Vector{Vector{Int}}()
    for scenario in 1:scenario_count
        if !haskey(stage_schedule, scenario)
            error("阶段调度中缺少场景 $scenario 的信息，无法标色。")
        end
        push!(stage_matrix, stage_schedule[scenario])
    end
    for row in stage_matrix
        if length(row) != n_steps
            error("场景的阶段向量长度不等于 $(n_steps)，请检查阶段表。")
        end
    end
    payload = Dict(
        "stages" => stage_matrix,
        "lines_per_scenario" => lines_per_scenario,
        "n_steps" => n_steps,
        "colors" => Dict(
            "0" => "000000",
            "1" => "FF0000",
            "2" => "0000FF",
            "3" => "00B050"
        )
    )
    json_path = tempname() * ".json"
    open(json_path, "w") do io
        JSON.print(io, payload)
    end
    try
        _run_stage_coloring_script(output_path, json_path)
    finally
        isfile(json_path) && rm(json_path, force=true)
    end
end

function _run_stage_coloring_script(output_path::AbstractString, json_path::AbstractString)
    safe_output_path = replace(output_path, '\\' => "\\\\")
    safe_output_path = replace(safe_output_path, '"' => "\\\"")
    safe_json_path = replace(json_path, '\\' => "\\\\")
    safe_json_path = replace(safe_json_path, '"' => "\\\"")
    script = """
import json
import subprocess
import sys
from pathlib import Path

try:
    import openpyxl
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    import openpyxl

output_path = Path(r"$safe_output_path")
json_path = Path(r"$safe_json_path")

with open(json_path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

wb = openpyxl.load_workbook(output_path)
sheet = wb["cluster_representatives"]

lines_per_scenario = int(data["lines_per_scenario"])
n_steps = int(data["n_steps"])
colors = data["colors"]
stages = data["stages"]

for scenario_index, stage_row in enumerate(stages):
    base_row = 2 + scenario_index * lines_per_scenario
    for line_offset in range(lines_per_scenario):
        excel_row = base_row + line_offset
        for step_index, stage_value in enumerate(stage_row):
            excel_col = 5 + step_index
            color = colors.get(str(stage_value), "000000")
            cell = sheet.cell(row=excel_row, column=excel_col)
            cell.font = openpyxl.styles.Font(color=color)
wb.save(output_path)
"""
    run(`python -c $script`)
end

function parse_cli_args()
    opts = Dict(
        :case => DEFAULT_CASE_FILE,
        :fault => DEFAULT_FAULT_FILE,
        :stage => DEFAULT_STAGE_FILE,
        :output => DEFAULT_OUTPUT_FILE,
        :fault_sheet => DEFAULT_FAULT_SHEET,
        :stage_sheet => DEFAULT_STAGE_SHEET,
        :lines_per_scenario => DEFAULT_LINES_PER_SCENARIO
    )
    for raw in ARGS
        if !startswith(raw, "--") || !occursin('=', raw)
            error("参数必须采用 --key=value 形式: $raw")
        end
        key, value = split(raw[3:end], "=", limit=2)
        key_sym = Symbol(replace(lowercase(key), '-' => '_'))
        if key_sym == :case
            opts[:case] = value
        elseif key_sym == :fault
            opts[:fault] = value
        elseif key_sym == :stage
            opts[:stage] = value
        elseif key_sym == :output
            opts[:output] = value
        elseif key_sym == :fault_sheet
            opts[:fault_sheet] = value
        elseif key_sym == :stage_sheet
            opts[:stage_sheet] = value
        elseif key_sym == :lines_per_scenario
            opts[:lines_per_scenario] = parse(Int, value)
        else
            error("未知参数: --$key")
        end
    end
    return opts
end

function main()
    opts = parse_cli_args()
    run_rolling_reconfiguration(
        case_file=opts[:case],
        fault_file=opts[:fault],
        stage_file=opts[:stage],
        output_file=opts[:output],
        fault_sheet=opts[:fault_sheet],
        stage_sheet=opts[:stage_sheet],
        lines_per_scenario=opts[:lines_per_scenario]
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
