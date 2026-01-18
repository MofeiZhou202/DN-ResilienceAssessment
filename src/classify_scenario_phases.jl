using DataFrames
using XLSX

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

function clamp01(value)
    return clamp(value, 0, 1)
end

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

function read_fault_matrix(input_path::AbstractString, sheet_name::AbstractString;
        start_row::Int = 2,
        start_col::Int = column_letter_to_index("E"),
        stop_col::Int = column_letter_to_index("AZ"),
        lines_per_scenario::Int = 35)
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

const STAGE_NAMES = Dict(
    0 => "阶段0 - 基准拓扑",
    1 => "阶段1 - 故障聚集",
    2 => "阶段2 - 前期恢复",
    3 => "阶段3 - 恢复后"
)

function stage_label(stage::Int)
    return get(STAGE_NAMES, stage, "阶段未知")
end

function classify_scenario_block(block::Matrix{Int}, minutes_per_step::Real, stage2_minutes::Real)
    n_steps = size(block, 2)
    stage_assignment = fill(0, n_steps)
    total_faults = sum(block, dims=1)[:]
    prev_state = falses(size(block, 1))
    new_faults = zeros(Int, n_steps)
    last_new_fault = missing
    for t in 1:n_steps
        current = block[:, t] .== 1
        new_at_t = count(current .& .!prev_state)
        new_faults[t] = new_at_t
        if new_at_t > 0
            last_new_fault = t
        end
        prev_state .= current
    end
    first_fault = findfirst(>(0), total_faults)
    stage1_start = first_fault === nothing || last_new_fault === missing ? nothing : first_fault
    stage1_end = first_fault === nothing || last_new_fault === missing ? nothing : last_new_fault
    if stage1_start !== nothing && stage1_end !== nothing
        for t in stage1_start:stage1_end
            stage_assignment[t] = 1
        end
    end
    stage2_range_start = stage1_end === nothing ? missing : stage1_end + 1
    stage2_steps = stage2_minutes <= 0 ? 0 : max(1, ceil(Int, stage2_minutes / minutes_per_step))
    stage2_range_end = missing
    if stage2_steps > 0 && stage2_range_start !== missing
        if stage2_range_start <= n_steps
            stage2_range_end = min(n_steps, stage2_range_start + stage2_steps - 1)
            for t in stage2_range_start:stage2_range_end
                stage_assignment[t] = 2
            end
        else
            stage2_range_start = missing
        end
    else
        stage2_range_start = missing
    end
    stage3_start = missing
    if stage2_range_end !== missing
        stage3_start = min(n_steps, stage2_range_end + 1)
    elseif stage1_end !== nothing
        stage3_start = min(n_steps, stage1_end + 1)
    end
    if stage3_start !== missing && stage3_start <= n_steps
        for t in stage3_start:n_steps
            stage_assignment[t] = 3
        end
    end

    return (
        stage=stage_assignment,
        total_faults=total_faults,
        new_faults=new_faults,
        stage1_start=stage1_start,
        stage1_end=stage1_end,
        stage2_start=stage2_range_start,
        stage2_end=stage2_range_end,
        stage3_start=stage3_start,
        n_steps=n_steps,
        first_fault=first_fault,
        last_new_fault=stage1_end === nothing ? missing : stage1_end
    )
end

function build_phase_tables(fault_matrix::Matrix{Int}, scenario_count::Int;
    lines_per_scenario::Int = 35,
    minutes_per_step::Real = 60,
    stage2_minutes::Real = 120)
    detail_rows = Vector{NamedTuple{(:Scenario, :TimeStep, :Stage, :StageLabel, :Minutes,
        :FaultCount, :NewFaultCount, :Stage1Start, :Stage1End, :Stage2Start, :Stage2End,
        :FirstFaultTimeStep, :LastNewFaultTimeStep)}}()
    summary_rows = Vector{NamedTuple{(:Scenario, :FirstFaultTimeStep, :LastNewFaultTimeStep,
        :Stage1DurationMinutes, :Stage2DurationMinutes, :Stage3StartTimeStep, :Stage3StartMinutes,
        :TotalNewFaults)}}()

    for scenario in 1:scenario_count
        start_idx = (scenario - 1) * lines_per_scenario + 1
        end_idx = scenario * lines_per_scenario
        block = fault_matrix[start_idx:end_idx, :]
        result = classify_scenario_block(block, minutes_per_step, stage2_minutes)
        stage3_start = result.stage3_start
        stage1_duration = result.stage1_start === nothing ? 0 : (result.stage1_end - result.stage1_start + 1) * minutes_per_step
        stage2_duration = result.stage2_start === missing || result.stage2_end === missing ? 0 :
            (result.stage2_end - result.stage2_start + 1) * minutes_per_step
        for t in 1:result.n_steps
            push!(detail_rows, (
                Scenario=scenario,
                TimeStep=t,
                Stage=result.stage[t],
                StageLabel=stage_label(result.stage[t]),
                Minutes=(t - 1) * minutes_per_step,
                FaultCount=result.total_faults[t],
                NewFaultCount=result.new_faults[t],
                Stage1Start=result.stage1_start === nothing ? missing : result.stage1_start,
                Stage1End=result.stage1_end === nothing ? missing : result.stage1_end,
                Stage2Start=result.stage2_start === missing ? missing : result.stage2_start,
                Stage2End=result.stage2_end === missing ? missing : result.stage2_end,
                FirstFaultTimeStep=result.first_fault === nothing ? missing : result.first_fault,
                LastNewFaultTimeStep=result.last_new_fault
            ))
        end
        push!(summary_rows, (
            Scenario=scenario,
            FirstFaultTimeStep=result.first_fault === nothing ? missing : result.first_fault,
            LastNewFaultTimeStep=result.last_new_fault,
            Stage1DurationMinutes=stage1_duration,
            Stage2DurationMinutes=stage2_duration,
            Stage3StartTimeStep=stage3_start,
            Stage3StartMinutes=stage3_start === missing ? missing : (stage3_start - 1) * minutes_per_step,
            TotalNewFaults=sum(result.new_faults)
        ))
    end

    detail_df = DataFrame(detail_rows)
    summary_df = DataFrame(summary_rows)
    return detail_df, summary_df
end

function write_dataframe_sheet(workbook, sheet_name::String, df::DataFrame)
    sheet = XLSX.addsheet!(workbook, sheet_name)
    for (col_idx, col_name) in enumerate(names(df))
        XLSX.setdata!(sheet, 1, col_idx, string(col_name))
    end
    for row_idx in 1:nrow(df)
        for col_idx in 1:ncol(df)
            # Ensure missing values are preserved
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

function write_phase_report(output_path::AbstractString, detail_df::DataFrame, summary_df::DataFrame;
        cluster_summary_source::Union{Nothing, AbstractString}=nothing,
        cluster_summary_sheet::AbstractString="cluster_summary")
    XLSX.openxlsx(output_path, mode = "w") do workbook
        write_dataframe_sheet(workbook, "StageDetails", detail_df)
        write_dataframe_sheet(workbook, "ScenarioSummary", summary_df)
        if cluster_summary_source !== nothing
            copy_named_sheet!(workbook, cluster_summary_source, cluster_summary_sheet)
        end
    end
end

function classify_phases(;input_path::AbstractString,
    output_path::AbstractString,
    lines_per_scenario::Int = 35,
    minutes_per_step::Real = 60,
    stage2_minutes::Real = 120,
    sheet_name::AbstractString = "cluster_representatives")
    fault_matrix, scenario_count = read_fault_matrix(input_path, sheet_name;
        lines_per_scenario=lines_per_scenario)
    detail_df, summary_df = build_phase_tables(fault_matrix, scenario_count;
        lines_per_scenario=lines_per_scenario,
        minutes_per_step=minutes_per_step,
        stage2_minutes=stage2_minutes)
    write_phase_report(output_path, detail_df, summary_df;
        cluster_summary_source=input_path,
        cluster_summary_sheet="cluster_summary")
    return detail_df, summary_df
end

function parse_cli_args()
    defaults = Dict(
        :input => joinpath(@__DIR__, "..", "data", "mc_simulation_results_k100_clusters.xlsx"),
        :output => joinpath(@__DIR__, "..", "data", "scenario_phase_classification.xlsx"),
        :lines_per_scenario => 35,
        :minutes_per_step => 60,
        :stage2_minutes => 120,
        :sheet => "cluster_representatives"
    )
    for raw in ARGS
        if !startswith(raw, "--") || !occursin('=', raw)
            error("Arguments must use the form --key=value")
        end
        key, value = split(raw[3:end], "=", limit=2)
        key_sym = Symbol(replace(key, '-' => '_'))
        if key_sym == :input
            defaults[:input] = value
        elseif key_sym == :output
            defaults[:output] = value
        elseif key_sym == :lines_per_scenario
            defaults[:lines_per_scenario] = parse(Int, value)
        elseif key_sym == :minutes_per_step
            defaults[:minutes_per_step] = parse(Float64, value)
        elseif key_sym == :stage2_minutes
            defaults[:stage2_minutes] = parse(Float64, value)
        elseif key_sym == :sheet
            defaults[:sheet] = value
        else
            error("Unknown argument: --$(key)")
        end
    end
    return defaults
end

function main()
    opts = parse_cli_args()
    detail_df, summary_df = classify_phases(
        input_path = opts[:input],
        output_path = opts[:output],
        lines_per_scenario = opts[:lines_per_scenario],
        minutes_per_step = opts[:minutes_per_step],
        stage2_minutes = opts[:stage2_minutes],
        sheet_name = opts[:sheet]
    )
    println("Wrote stage detail sheet with $(nrow(detail_df)) rows and summary sheet with $(nrow(summary_df)) rows to $(opts[:output])")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
