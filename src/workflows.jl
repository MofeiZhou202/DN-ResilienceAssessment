module Workflows

using DataFrames
using XLSX
using PyCall
using Dates

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))

# 首先加载所有工具文件（只加载一次，避免重复定义）
include(joinpath(@__DIR__, "utils", "idx.jl"))
include(joinpath(@__DIR__, "utils", "ComponentStructs.jl"))
include(joinpath(@__DIR__, "utils", "Types.jl"))
include(joinpath(@__DIR__, "utils", "ETAPImporter.jl"))
include(joinpath(@__DIR__, "utils", "juliapowercase2jpc_tp.jl"))

include(joinpath(@__DIR__, "classify_scenario_phases.jl"))
include(joinpath(@__DIR__, "rolling_horizon_reconfiguration.jl"))
include(joinpath(@__DIR__, "dispatch_main.jl"))

export run_classify_phases, run_rolling_reconfig, run_mess_dispatch,
       run_typhoon_workflow, run_full_pipeline, show_menu, show_help

function _read_user_input(prompt::String)
    print(prompt)
    flush(stdout)
    try
        return readline()
    catch
        return ""
    end
end

function _normalize_user_path(path::String)
    if isempty(path)
        return path
    end
    expanded = expanduser(path)
    return normpath(isabspath(expanded) ? expanded : abspath(expanded))
end

function _resolve_path_from_user(label::String, default_value::String)
    prompt = isempty(default_value) ? string(label, ": ") : string(label, "（默认: ", default_value, "）: ")
    response = strip(_read_user_input(prompt))
    if isempty(response)
        println("→ 使用默认路径: $default_value")
        return default_value
    end
    normalized = _normalize_user_path(response)
    println("→ 使用路径: $normalized")
    return normalized
end

function _resolve_path_arg(value::Union{Nothing, String}, label::String, default_value::String)
    if isnothing(value) || isempty(value)
        return _resolve_path_from_user(label, default_value)
    end
    normalized = _normalize_user_path(String(value))
    println("→ 使用路径: $normalized")
    return normalized
end

function run_classify_phases(; 
    input_path::Union{Nothing, String} = nothing,
    output_path::Union{Nothing, String} = nothing,
    lines_per_scenario::Int = 35,
    minutes_per_step::Real = 60,
    stage2_minutes::Real = 120,
    sheet_name::String = "cluster_representatives")
    println("\n" * "="^60)
    println("场景阶段分类")
    println("="^60)

    default_input = joinpath(ROOT_DIR, "data", "mc_simulation_results_k100_clusters.xlsx")
    default_output = joinpath(ROOT_DIR, "data", "scenario_phase_classification.xlsx")
    selected_input = _resolve_path_arg(input_path, "请输入场景阶段分类输入Excel路径", default_input)
    selected_output = _resolve_path_arg(output_path, "请输入分类结果输出Excel路径", default_output)

    detail_df, summary_df = classify_phases(
        input_path = selected_input,
        output_path = selected_output,
        lines_per_scenario = lines_per_scenario,
        minutes_per_step = minutes_per_step,
        stage2_minutes = stage2_minutes,
        sheet_name = sheet_name,
    )

    println("✓ 场景阶段分类完成")
    println("  详细表: $(nrow(detail_df)) 行")
    println("  汇总表: $(nrow(summary_df)) 行")
    println("  输出文件: $(selected_output)")

    return detail_df, summary_df
end

function run_rolling_reconfig(;
    case_file::Union{Nothing, String} = nothing,
    fault_file::Union{Nothing, String} = nothing,
    stage_file::Union{Nothing, String} = nothing,
    output_file::Union{Nothing, String} = nothing,
    fault_sheet::String = "cluster_representatives",
    stage_sheet::String = "StageDetails",
    lines_per_scenario::Int = 35)
    println("\n" * "="^60)
    println("滚动拓扑重构")
    println("="^60)

    default_case = joinpath(ROOT_DIR, "data", "ac_dc_real_case.xlsx")
    default_fault = joinpath(ROOT_DIR, "data", "mc_simulation_results_k100_clusters.xlsx")
    default_stage = joinpath(ROOT_DIR, "data", "scenario_phase_classification.xlsx")
    default_output = joinpath(ROOT_DIR, "data", "topology_reconfiguration_results.xlsx")

    selected_case = _resolve_path_arg(case_file, "请输入滚动拓扑重构算例输入路径", default_case)
    selected_fault = _resolve_path_arg(fault_file, "请输入Monte Carlo故障结果文件路径", default_fault)
    selected_stage = _resolve_path_arg(stage_file, "请输入场景阶段分类结果文件路径", default_stage)
    selected_output = _resolve_path_arg(output_file, "请输入拓扑重构结果输出路径", default_output)

    results_df = run_rolling_reconfiguration(
        case_file = selected_case,
        fault_file = selected_fault,
        stage_file = selected_stage,
        output_file = selected_output,
        fault_sheet = fault_sheet,
        stage_sheet = stage_sheet,
        lines_per_scenario = lines_per_scenario,
    )

    println("✓ 滚动拓扑重构完成")
    println("  输出文件: $(selected_output)")

    return results_df
end

function run_mess_dispatch(; 
    case_path::Union{Nothing, String} = nothing,
    topology_path::Union{Nothing, String} = nothing,
    fallback_topology::Union{Nothing, String} = nothing)
    default_case = DEFAULT_CASE_XLSX
    default_topology = DEFAULT_TOPOLOGY_XLSX
    default_fallback = DEFAULT_MC_XLSX

    selected_case = _resolve_path_arg(case_path, "请输入混合配电网算例输入路径", default_case)
    selected_topology = _resolve_path_arg(topology_path, "请输入拓扑重构结果文件路径", default_topology)
    selected_fallback = _resolve_path_arg(fallback_topology, "请输入拓扑缺失回退文件路径", default_fallback)

    run_mess_dispatch_julia(
        case_path = selected_case,
        topology_path = selected_topology,
        fallback_topology = selected_fallback,
    )
    println("✓ MESS协同调度完成")
end

function run_typhoon_workflow(; command::Union{Nothing, String} = nothing)
    println("\n" * "="^60)
    println("台风场景生成工作流")
    println("="^60)

    selected_command = if isnothing(command)
        strip(_read_user_input("如需传入台风工作流子命令（可包含自定义文件路径），请输入（直接回车进入交互菜单）: "))
    else
        String(command)
    end

    if isempty(selected_command)
        py"""
        import sys
        import os

        project_root = $ROOT_DIR
        os.chdir(project_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from app import main
        main()
        """
    else
        py"""
        import sys
        import os

        project_root = $ROOT_DIR
        os.chdir(project_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from app import main
        main($selected_command.split())
        """
    end
end

function show_menu()
    println("\n" * "="^60)
    println("DN-ResilienceAssessment 配电网韧性评估系统")
    println("="^60)
    println()
    println("请选择要运行的功能：")
    println()
    println("  1. 场景阶段分类")
    println("  2. 滚动拓扑重构")
    println("  3. 混合配电网+MESS协同调度")
    println("  4. 台风场景生成工作流")
    println("  5. 完整流程（1→2→3）")
    println("  q. 退出")
    println()
    print("请输入选项 [1-5/q]: ")
end

function run_full_pipeline()
    println("\n" * "="^60)
    println("运行完整流程")
    println("="^60)

    println("\n[步骤 1/3] 场景阶段分类...")
    run_classify_phases()

    println("\n[步骤 2/3] 滚动拓扑重构...")
    run_rolling_reconfig()

    println("\n[步骤 3/3] MESS协同调度...")
    run_mess_dispatch()

    println("\n" * "="^60)
    println("✓ 完整流程执行完毕")
    println("="^60)
end

function show_help()
    println("""
使用方法:
    julia main.jl                    # 交互式菜单
    julia main.jl --classify         # 场景阶段分类
    julia main.jl --reconfig         # 滚动拓扑重构
    julia main.jl --dispatch         # MESS协同调度
    julia main.jl --typhoon          # 台风场景生成（交互式）
    julia main.jl --typhoon typhoon  # 台风场景生成（指定子命令）
    julia main.jl --full             # 完整流程
    julia main.jl --help             # 显示帮助
    """)
end

end # module
