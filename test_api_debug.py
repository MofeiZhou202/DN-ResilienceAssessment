"""
调试脚本：测试PyJulia调用run_mess_dispatch时的行为
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 确保 Julia 使用项目内的环境
os.environ.setdefault("JULIA_PROJECT", str(PROJECT_ROOT))

print("正在初始化 Julia...")
from julia.api import Julia
from julia import Main

Julia(compiled_modules=False)
Main.include(str(PROJECT_ROOT / "src" / "workflows.jl"))
Workflows = Main.Workflows

print("Julia 初始化完成")

# 测试参数
case_path = str(PROJECT_ROOT / "data" / "ac_dc_real_case.xlsx")
topology_path = str(PROJECT_ROOT / "data" / "topology_reconfiguration_results.xlsx")
fallback_topology = str(PROJECT_ROOT / "data" / "mc_simulation_results_k100_clusters.xlsx")
output_file = str(PROJECT_ROOT / "data" / "test_mess_dispatch_output.xlsx")

print("\n" + "="*60)
print("测试 run_mess_dispatch 调用")
print("="*60)
print(f"case_path: {case_path}")
print(f"topology_path: {topology_path}")
print(f"fallback_topology: {fallback_topology}")
print(f"output_file: {output_file}")
print()

# 首先测试一下 _is_interactive_mode 的返回值
print("\n检查交互模式状态:")
Main.eval("""
println("isinteractive() = ", isinteractive())
println("isa(stdin, Base.TTY) = ", isa(stdin, Base.TTY))
""")

print("\n开始调用 run_mess_dispatch...")
try:
    Workflows.run_mess_dispatch(
        case_path=case_path,
        topology_path=topology_path,
        fallback_topology=fallback_topology,
        output_file=output_file
    )
    print("\n✓ run_mess_dispatch 调用成功完成")
except Exception as e:
    print(f"\n✗ 调用失败: {e}")
    import traceback
    traceback.print_exc()
