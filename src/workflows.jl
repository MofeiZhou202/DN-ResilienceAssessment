module Workflows

using DataFrames
using XLSX
using PyCall
using Dates
using Random
using Clustering
using Distances
using Statistics

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
       run_typhoon_workflow, run_full_pipeline, run_resilience_assessment,
       show_menu, show_help

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
    fallback_topology::Union{Nothing, String} = nothing,
    output_file::Union{Nothing, String} = nothing)
    # ✅ 修复：直接使用路径，不依赖dispatch_main.jl中未导出的常量
    default_case = joinpath(ROOT_DIR, "data", "ac_dc_real_case.xlsx")
    default_topology = joinpath(ROOT_DIR, "data", "topology_reconfiguration_results.xlsx")
    default_fallback = joinpath(ROOT_DIR, "data", "mc_simulation_results_k100_clusters.xlsx")
    default_output = joinpath(ROOT_DIR, "data", "mess_dispatch_report.xlsx")

    selected_case = _resolve_path_arg(case_path, "请输入混合配电网算例输入路径", default_case)
    selected_topology = _resolve_path_arg(topology_path, "请输入拓扑重构结果文件路径", default_topology)
    selected_fallback = if isnothing(fallback_topology) || isempty(String(fallback_topology))
        println("→ 使用默认拓扑缺失回退文件: $(default_fallback)")
        default_fallback
    else
        _resolve_path_arg(fallback_topology, "请输入拓扑缺失回退文件路径", default_fallback)
    end
    selected_output = _resolve_path_arg(output_file, "请输入调度结果Excel输出路径", default_output)

    run_mess_dispatch_julia(
        case_path = selected_case,
        topology_path = selected_topology,
        fallback_topology = selected_fallback,
        output_file = selected_output,
    )
    println("✓ MESS协同调度完成")
    println("  输出文件: $(selected_output)")
end

function run_typhoon_workflow(; command::Union{Nothing, String} = nothing,
    tower_excel::Union{Nothing, String} = nothing,
    final_output::Union{Nothing, String} = nothing)
    println("\n" * "="^60)
    println("台风场景生成工作流")
    println("="^60)

    selected_command = isnothing(command) ? "" : strip(String(command))
    if !isempty(selected_command)
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
        return
    end

    default_tower = joinpath(ROOT_DIR, "data", "TowerSeg.xlsx")
    selected_tower = _resolve_path_arg(tower_excel, "请输入TowerSeg Excel路径", default_tower)

    default_output = joinpath(ROOT_DIR, "data", "mc_simulation_results_k100_clusters.xlsx")
    selected_output = if isnothing(final_output) || isempty(String(final_output))
        println("→ 使用默认聚类结果输出路径: $(default_output)")
        default_output
    else
        normalized = _normalize_user_path(String(final_output))
        println("→ 使用聚类结果输出路径: $(normalized)")
        normalized
    end

    println("→ 正在执行一键台风评估流程，请稍候...")

    py"""
    import sys
    import os

    project_root = $ROOT_DIR
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from app import main
    main(["one-click", "--tower-excel", $selected_tower, "--final-output", $selected_output])
    """
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
    println("  6. 完整弹性评估（4+5融合，一键执行）★推荐★")
    println("  q. 退出")
    println()
    print("请输入选项 [1-6/q]: ")
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

"""
完整弹性评估流程 - 融合功能4和功能5

用户只需提供两个文件：
1. tower_seg_file: TowerSeg.xlsx - 配电网塔杆分段结构文件
2. case_file: ac_dc_real_case.xlsx - 混合交直流配电网算例文件

执行流程：
Step 1: 台风场景生成 (功能4) - 生成 mc_simulation_results_k100_clusters.xlsx
Step 2: 场景阶段分类 - 生成 scenario_phase_classification.xlsx
Step 3: 滚动拓扑重构 - 生成 topology_reconfiguration_results.xlsx
Step 4: MESS协同调度 - 生成 mess_dispatch_results.xlsx

参数：
- tower_seg_file: TowerSeg.xlsx 文件路径
- case_file: ac_dc_real_case.xlsx 文件路径
- output_dir: (可选) 输出目录路径，默认为 data/

返回：
- Dict 包含所有输出文件路径
"""
function run_resilience_assessment(;
    tower_seg_file::Union{Nothing, String} = nothing,
    case_file::Union{Nothing, String} = nothing,
    output_dir::Union{Nothing, String} = nothing)
    
    println("\n" * "="^60)
    println("完整弹性评估流程 (功能4 + 功能5)")
    println("="^60)
    
    # 解析输入路径
    default_tower = joinpath(ROOT_DIR, "data", "TowerSeg.xlsx")
    default_case = joinpath(ROOT_DIR, "data", "ac_dc_real_case.xlsx")
    default_output_dir = joinpath(ROOT_DIR, "data")
    
    selected_tower = _resolve_path_arg(tower_seg_file, "请输入TowerSeg Excel路径", default_tower)
    selected_case = _resolve_path_arg(case_file, "请输入ac_dc_real_case Excel路径", default_case)
    
    selected_output_dir = if isnothing(output_dir) || isempty(String(output_dir))
        println("→ 使用默认输出目录: $(default_output_dir)")
        default_output_dir
    else
        normalized = _normalize_user_path(String(output_dir))
        println("→ 使用输出目录: $(normalized)")
        normalized
    end
    
    # 确保输出目录存在
    if !isdir(selected_output_dir)
        mkpath(selected_output_dir)
    end
    
    # 定义输出文件路径
    cluster_output = joinpath(selected_output_dir, "mc_simulation_results_k100_clusters.xlsx")
    phase_output = joinpath(selected_output_dir, "scenario_phase_classification.xlsx")
    topology_output = joinpath(selected_output_dir, "topology_reconfiguration_results.xlsx")
    dispatch_output = joinpath(selected_output_dir, "mess_dispatch_results.xlsx")
    
    println("\n输入文件:")
    println("  - TowerSeg: $(selected_tower)")
    println("  - Case: $(selected_case)")
    println("\n输出文件:")
    println("  - 聚类结果: $(cluster_output)")
    println("  - 阶段分类: $(phase_output)")
    println("  - 拓扑重构: $(topology_output)")
    println("  - 调度结果: $(dispatch_output)")
    
    # ===========================================================
    # Step 1: 台风场景生成 (功能4)
    # ===========================================================
    println("\n" * "-"^60)
    println("[Step 1/4] 执行台风场景生成...")
    println("-"^60)
    
    py"""
    import sys
    import os

    project_root = $ROOT_DIR
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from app import main
    main(["one-click", "--tower-excel", $selected_tower, "--final-output", $cluster_output])
    """
    
    if !isfile(cluster_output)
        error("台风场景生成失败：聚类结果文件未生成")
    end
    println("✓ Step 1 完成: 聚类结果已生成")
    
    # ===========================================================
    # Step 2: 场景阶段分类
    # ===========================================================
    println("\n" * "-"^60)
    println("[Step 2/4] 执行场景阶段分类...")
    println("-"^60)
    
    detail_df, summary_df = classify_phases(
        input_path = cluster_output,
        output_path = phase_output,
    )
    println("✓ Step 2 完成: 场景阶段分类已完成")
    println("  详细表: $(nrow(detail_df)) 行")
    println("  汇总表: $(nrow(summary_df)) 行")
    
    # ===========================================================
    # Step 3: 滚动拓扑重构
    # ===========================================================
    println("\n" * "-"^60)
    println("[Step 3/4] 执行滚动拓扑重构...")
    println("-"^60)
    
    results_df = run_rolling_reconfiguration(
        case_file = selected_case,
        fault_file = cluster_output,
        stage_file = phase_output,
        output_file = topology_output,
    )
    println("✓ Step 3 完成: 滚动拓扑重构已完成")
    
    # ===========================================================
    # Step 4: MESS协同调度
    # ===========================================================
    println("\n" * "-"^60)
    println("[Step 4/4] 执行MESS协同调度...")
    println("-"^60)
    
    run_mess_dispatch_julia(
        case_path = selected_case,
        topology_path = topology_output,
        fallback_topology = cluster_output,
        output_file = dispatch_output,
    )
    println("✓ Step 4 完成: MESS协同调度已完成")
    
    # ===========================================================
    # 完成
    # ===========================================================
    println("\n" * "="^60)
    println("✓ 完整弹性评估流程执行完毕")
    println("="^60)
    println("\n最终输出文件:")
    println("  → $(dispatch_output)")
    
    return Dict(
        "tower_seg_file" => selected_tower,
        "case_file" => selected_case,
        "cluster_output" => cluster_output,
        "phase_output" => phase_output,
        "topology_output" => topology_output,
        "dispatch_output" => dispatch_output,
    )
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
    julia main.jl --full             # 完整流程（1→2→3）
    julia main.jl --resilience       # 完整弹性评估（4+5融合）★推荐★
    julia main.jl --help             # 显示帮助

完整弹性评估说明:
    --resilience 选项将台风场景生成(功能4)和完整流程(功能5)融合为一键执行。
    用户只需提供两个文件：
    1. TowerSeg.xlsx - 配电网塔杆分段结构文件
    2. ac_dc_real_case.xlsx - 混合交直流配电网算例文件
    
    执行后将自动生成：
    - mc_simulation_results_k100_clusters.xlsx (聚类结果)
    - scenario_phase_classification.xlsx (阶段分类)
    - topology_reconfiguration_results.xlsx (拓扑重构)
    - mess_dispatch_results.xlsx (调度结果) ← 最终输出
    """)
end

end # module
