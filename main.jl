"""
DN-ResilienceAssessment 配电网韧性评估系统 - Julia主入口

功能模块：
1. 场景阶段分类 (classify_scenario_phases.jl)
2. 滚动拓扑重构 (rolling_horizon_reconfiguration.jl)  
3. 混合配电网+MESS协同调度 (通过PyCall调用Python)
4. 台风场景生成工作流 (通过PyCall调用app.py)

使用方法：
    julia main.jl                    # 交互式菜单
    julia main.jl --classify         # 场景阶段分类
    julia main.jl --reconfig         # 滚动拓扑重构
    julia main.jl --dispatch         # MESS协同调度
    julia main.jl --typhoon          # 台风场景生成菜单
"""

using Pkg

# 确保在正确的项目环境中
const PROJECT_DIR = @__DIR__
Pkg.activate(PROJECT_DIR)

# 检查并安装PyCall
if !haskey(Pkg.project().dependencies, "PyCall")
    println("正在安装 PyCall...")
    Pkg.add("PyCall")
end

using PyCall

# 设置Python路径（使用系统Python）
ENV["PYTHON"] = ""  # 使用默认Python

# 导入Python的sys模块来添加路径
py"""
import sys
import os
project_root = $PROJECT_DIR
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)
"""

println("正在加载工作流模块...")
include(joinpath(PROJECT_DIR, "src", "workflows.jl"))
using .Workflows
println("模块加载完成")

"""
主函数
"""
function main()
    # 解析命令行参数
    if length(ARGS) > 0
        arg = ARGS[1]
        if arg == "--classify"
            run_classify_phases()
        elseif arg == "--reconfig"
            run_rolling_reconfig()
        elseif arg == "--dispatch"
            run_mess_dispatch()
        elseif arg == "--typhoon"
            command = length(ARGS) > 1 ? join(ARGS[2:end], " ") : ""
            run_typhoon_workflow(command = command)
        elseif arg == "--full"
            run_full_pipeline()
        elseif arg == "--help" || arg == "-h"
            show_help()
        else
            println("未知参数: $arg")
            println("使用 --help 查看帮助")
        end
        return
    end
    
    # 交互式菜单
    show_menu()
    choice = readline()
    choice = strip(choice)
    if choice == "1"
        run_classify_phases()
    elseif choice == "2"
        run_rolling_reconfig()
    elseif choice == "3"
        run_mess_dispatch()
    elseif choice == "4"
        run_typhoon_workflow()
    elseif choice == "5"
        run_full_pipeline()
    elseif choice == "q" || choice == "Q" || choice == "quit" || choice == "exit"
        println("\n再见！")
    else
        println("\n无效选项，请重新选择")
    end
end

# 如果直接运行此文件
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
