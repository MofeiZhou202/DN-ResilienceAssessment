"""
配电网弹性推理系统 - 完整端到端流程 V2
===========================================
正确的推理流程：
1. 从拓扑重构结果获取每个场景每小时的线路状态（100场景×48小时=4800行）
2. 从调度结果获取每个场景每小时的弹性指标（失负荷量、供电率）
3. 合并数据，建立"(场景,时间,线路状态)→弹性指标"映射
4. 执行推理分析（XGBoost代理模型 + SHAP归因）

使用方法:
    python run_inference_pipeline_v2.py --dispatch data/mess_dispatch_with_scenarios.xlsx

必需的输入文件:
    - topology_reconfiguration_results.xlsx (每小时线路状态，4800行)
    - mess_dispatch_with_scenarios.xlsx (每小时弹性指标，HourlyDetails工作表)

数据结构说明:
    拓扑数据(RollingDecisionsOriginal):
        - Scenario: 场景ID (1-100)
        - TimeStep: 时间步 (0-47)
        - AC_Line_X: 线路状态 (0=断开, 1=联通)
        
    弹性指标(HourlyDetails):
        - Scenario_ID: 场景ID (1-100)
        - TimeStep: 时间步 (1-48)
        - Load_Shed: 当前小时失负荷量
        - Supply_Rate: 当前小时供电率
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class NumpyEncoder(json.JSONEncoder):
    """JSON编码器，支持numpy类型的序列化"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class InferenceDataBuilder:
    """
    推理数据构建器 V2
    
    将拓扑状态数据（每小时线路状态）和弹性指标数据（每小时失负荷）合并，
    生成4800行的完整推理数据集。
    
    数据流:
    ┌─────────────────────┐     ┌─────────────────────┐
    │ topology_results    │     │ dispatch_results    │
    │ (4800行, 线路状态)   │     │ (4800行, 弹性指标)   │
    └─────────┬───────────┘     └──────────┬──────────┘
              │                            │
              │  按 (Scenario, TimeStep) 合并
              │                            │
              └─────────────┬──────────────┘
                            ▼
                   ┌────────────────────┐
                   │ merged_data        │
                   │ (4800行, 线路+指标) │
                   └────────────────────┘
                            │
                            ▼
                   ┌────────────────────┐
                   │ ResilienceInference│
                   │ (XGBoost + SHAP)   │
                   └────────────────────┘
    """
    
    def __init__(self):
        self.topology_data = None
        self.resilience_data = None
        self.merged_data = None
        self.line_columns = []
        self.metadata = {}
    
    def load_topology_data(
        self,
        topology_path: str,
        sheet_name: str = "RollingDecisionsOriginal"
    ) -> pd.DataFrame:
        """
        加载拓扑重构结果（每个场景每小时的线路状态）
        
        输入文件结构:
            | Scenario | TimeStep | AC_Line_1 | AC_Line_2 | ... |
            |----------|----------|-----------|-----------|-----|
            | 1        | 0        | 1         | 1         | ... |
            | 1        | 1        | 1         | 0         | ... |
            | ...      | ...      | ...       | ...       | ... |
            | 100      | 47       | 1         | 1         | ... |
        
        Returns:
            DataFrame with shape (4800, n_cols)
        """
        print(f"[DataBuilder] 加载拓扑数据: {topology_path}")
        
        xl = pd.ExcelFile(topology_path)
        
        # 尝试不同工作表
        target_sheets = [sheet_name, "RollingDecisionsOriginal", "Sheet1"]
        for sheet in target_sheets:
            if sheet in xl.sheet_names:
                self.topology_data = pd.read_excel(xl, sheet_name=sheet)
                if len(self.topology_data) > 0 and 'Scenario' in self.topology_data.columns:
                    print(f"  使用工作表: {sheet}")
                    break
        
        if self.topology_data is None or len(self.topology_data) == 0:
            raise ValueError(f"无法从文件加载有效的拓扑数据")
        
        # 识别线路列
        all_columns = self.topology_data.columns.tolist()
        self.line_columns = [
            col for col in all_columns
            if col.startswith(('AC_Line_', 'DC_Line_', 'VSC_Line_'))
        ]
        
        n_scenarios = self.topology_data['Scenario'].nunique()
        n_timesteps = len(self.topology_data) // n_scenarios if n_scenarios > 0 else 0
        
        print(f"  数据形状: {self.topology_data.shape}")
        print(f"  场景数: {n_scenarios}")
        print(f"  每场景时间步: {n_timesteps}")
        print(f"  线路数: {len(self.line_columns)}")
        print(f"  总数据点: {len(self.topology_data)} (应为 {n_scenarios}×{n_timesteps}={n_scenarios * n_timesteps})")
        
        self.metadata['n_scenarios'] = n_scenarios
        self.metadata['n_timesteps'] = n_timesteps
        self.metadata['n_lines'] = len(self.line_columns)
        
        return self.topology_data
    
    def load_resilience_data(
        self,
        dispatch_results_path: str,
        sheet_name: str = "HourlyDetails"
    ) -> pd.DataFrame:
        """
        加载调度结果中的每小时弹性指标
        
        输入文件结构 (HourlyDetails工作表):
            | Scenario_ID | TimeStep | Load_Shed | Supply_Rate | ... |
            |-------------|----------|-----------|-------------|-----|
            | 1           | 1        | 100.5     | 0.95        | ... |
            | 1           | 2        | 80.2      | 0.97        | ... |
            | ...         | ...      | ...       | ...         | ... |
            | 100         | 48       | 0.0       | 1.0         | ... |
        
        Returns:
            DataFrame with shape (4800, n_cols)
        """
        print(f"\n[DataBuilder] 加载弹性指标数据: {dispatch_results_path}")
        
        xl = pd.ExcelFile(dispatch_results_path)
        print(f"  可用工作表: {xl.sheet_names}")
        
        # 优先使用 HourlyDetails（每小时数据）
        if sheet_name in xl.sheet_names:
            self.resilience_data = pd.read_excel(xl, sheet_name=sheet_name)
            print(f"  使用工作表: {sheet_name} (每小时详细数据)")
        elif "ScenarioDetails" in xl.sheet_names:
            # 后备：使用场景汇总数据（需要扩展到每小时）
            print(f"  [警告] 未找到'{sheet_name}'工作表")
            print(f"         使用'ScenarioDetails'（场景汇总数据，将扩展到每小时）")
            self.resilience_data = pd.read_excel(xl, sheet_name="ScenarioDetails")
            self._expand_scenario_to_hourly()
        else:
            raise ValueError(
                f"\n调度结果文件中缺少'{sheet_name}'或'ScenarioDetails'工作表！\n"
                f"这意味着弹性指标未被保存。\n"
                f"请重新运行MESS调度以生成完整数据。\n"
                f"可用工作表: {xl.sheet_names}"
            )
        
        if len(self.resilience_data) == 0:
            raise ValueError("弹性指标数据为空")
        
        print(f"  数据形状: {self.resilience_data.shape}")
        print(f"  列: {list(self.resilience_data.columns)}")
        
        return self.resilience_data
    
    def _expand_scenario_to_hourly(self):
        """
        将场景级别的汇总数据扩展到每小时（后备方案）
        
        注意：这只是为了兼容旧数据，新数据应该直接使用HourlyDetails
        """
        if self.resilience_data is None:
            return
        
        n_timesteps = self.metadata.get('n_timesteps', 48)
        expanded_rows = []
        
        for _, row in self.resilience_data.iterrows():
            scenario_id = row.get('Scenario_ID', row.name + 1)
            prob = row.get('Probability', 0.01)
            load_demand = row.get('Load_Demand_Total', 0) / n_timesteps
            load_shed = row.get('Load_Shed_Total', 0) / n_timesteps
            supply_rate = row.get('Supply_Rate', 1.0)
            
            for t in range(1, n_timesteps + 1):
                expanded_rows.append({
                    'Scenario_ID': scenario_id,
                    'TimeStep': t,
                    'Probability': prob,
                    'Load_Demand': load_demand,
                    'Load_Shed': load_shed,
                    'Supply_Rate': supply_rate,
                })
        
        self.resilience_data = pd.DataFrame(expanded_rows)
        print(f"  [扩展] 从场景汇总扩展到每小时: {len(self.resilience_data)} 行")
    
    def merge_data(self) -> pd.DataFrame:
        """
        合并拓扑状态和弹性指标数据
        
        按 (Scenario, TimeStep) 进行精确匹配合并
        
        Returns:
            合并后的数据，每行一个 (场景, 时间步) 组合，共4800行
        """
        if self.topology_data is None or self.resilience_data is None:
            raise ValueError("请先加载拓扑数据和弹性指标数据")
        
        print(f"\n[DataBuilder] 合并数据")
        
        # 准备拓扑数据（确保TimeStep列名一致）
        topo_df = self.topology_data.copy()
        if 'TimeStep' not in topo_df.columns and 'Step' in topo_df.columns:
            topo_df['TimeStep'] = topo_df['Step']
        
        # 准备弹性数据
        resil_df = self.resilience_data.copy()
        
        # 检查TimeStep的对齐：
        # 拓扑数据 TimeStep: 0-47
        # 弹性数据 TimeStep: 1-48
        # 需要对齐
        topo_timesteps = sorted(topo_df['TimeStep'].unique())
        resil_timesteps = sorted(resil_df['TimeStep'].unique())
        print(f"  拓扑TimeStep范围: {min(topo_timesteps)} - {max(topo_timesteps)}")
        print(f"  弹性TimeStep范围: {min(resil_timesteps)} - {max(resil_timesteps)}")
        
        # 如果拓扑是0-based，弹性是1-based，则对齐
        if min(topo_timesteps) == 0 and min(resil_timesteps) == 1:
            print(f"  对齐TimeStep: 拓扑+1 以匹配弹性数据")
            topo_df['TimeStep'] = topo_df['TimeStep'] + 1
        
        # 重命名列以便合并
        topo_df = topo_df.rename(columns={'Scenario': 'Scenario_ID'})
        
        # 只保留需要的列
        topo_cols = ['Scenario_ID', 'TimeStep'] + self.line_columns
        if 'FaultCount' in topo_df.columns:
            topo_cols.append('FaultCount')
        topo_df = topo_df[topo_cols]
        
        # 合并
        self.merged_data = pd.merge(
            topo_df,
            resil_df,
            on=['Scenario_ID', 'TimeStep'],
            how='inner'
        )
        
        # 重命名线路列为_Status后缀
        rename_map = {col: f"{col}_Status" for col in self.line_columns}
        self.merged_data = self.merged_data.rename(columns=rename_map)
        
        # 重命名弹性指标列以保持一致
        if 'Load_Shed' in self.merged_data.columns:
            self.merged_data = self.merged_data.rename(columns={
                'Load_Shed': 'Total_Load_Loss',
                'Load_Demand': 'Load_Demand_Hourly',
            })
        
        print(f"  合并后数据形状: {self.merged_data.shape}")
        print(f"  平均供电率: {self.merged_data['Supply_Rate'].mean():.4f}")
        print(f"  平均每小时失负荷: {self.merged_data['Total_Load_Loss'].mean():.2f}")
        print(f"  最大每小时失负荷: {self.merged_data['Total_Load_Loss'].max():.2f}")
        
        # 显示节点复电超时信息（如果有）
        if 'Nodes_Over_2h' in self.merged_data.columns:
            over_2h_samples = (self.merged_data['Nodes_Over_2h'] > 0).sum()
            print(f"  含复电超2h样本数: {over_2h_samples} ({over_2h_samples/len(self.merged_data)*100:.1f}%)")
            print(f"  平均超2h节点数: {self.merged_data['Nodes_Over_2h'].mean():.2f}")
        
        return self.merged_data
    
    def save_data(self, output_path: str) -> str:
        """保存合并后的数据"""
        if self.merged_data is None:
            raise ValueError("无数据可保存")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self.merged_data.to_excel(writer, sheet_name='InferenceData', index=False)
            
            # 添加元数据工作表
            meta_df = pd.DataFrame([self.metadata])
            meta_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        print(f"\n[DataBuilder] 数据已保存至: {output_path}")
        return str(output_path)


def find_latest_results(data_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    查找最新的弹性评估结果文件
    
    Returns:
        (topology_path, dispatch_path) 元组
    """
    topology_path = None
    dispatch_path = None
    
    # 优先查找 auto_eval_runs 目录
    auto_eval_dir = data_dir / "auto_eval_runs"
    if auto_eval_dir.exists():
        runs = sorted(auto_eval_dir.glob("dn_resilience_*"), reverse=True)
        for run_dir in runs:
            topo_file = run_dir / "topology_reconfiguration_results.xlsx"
            if topo_file.exists():
                topology_path = topo_file
                break
    
    # 如果没找到，查找data目录直接的文件
    if topology_path is None:
        direct_topo = data_dir / "topology_reconfiguration_results.xlsx"
        if direct_topo.exists():
            topology_path = direct_topo
    
    # 查找调度结果（优先带HourlyDetails的文件）
    # 按优先级排序：hourly文件 > for_inference > with_scenarios
    dispatch_candidates = [
        "mess_dispatch_hourly.xlsx",  # 最优先：包含每小时详细数据
        "mess_dispatch_for_inference.xlsx",
        "mess_dispatch_with_scenarios.xlsx",
    ]
    for dispatch_name in dispatch_candidates:
        dispatch_file = data_dir / dispatch_name
        if dispatch_file.exists():
            # 检查是否包含HourlyDetails工作表
            try:
                xl = pd.ExcelFile(dispatch_file)
                if "HourlyDetails" in xl.sheet_names:
                    dispatch_path = dispatch_file
                    print(f"  [文件选择] 找到包含HourlyDetails的文件: {dispatch_name}")
                    break
                elif dispatch_path is None:
                    dispatch_path = dispatch_file  # 后备选择
            except:
                pass
    
    return topology_path, dispatch_path


def _create_counterfactual_mc_data(
    original_mc_path: str,
    target_lines: List[str],
    output_dir: Path
) -> str:
    """
    创建反事实MC数据文件
    
    在MC数据中把指定线路的48小时状态全部设为0（正常），
    然后可以用这个反事实数据重新跑完整流程来验证加固效果。
    
    MC数据结构说明:
    - cluster_representatives工作表: 100场景×35线路=3500行
    - row_in_sample: 线路编号(1-35)，对应AC_Line_1到AC_Line_35
    - Col_01到Col_48: 48小时的线路状态（0=正常，1=故障）
    
    Args:
        original_mc_path: 原始MC数据文件路径
        target_lines: 要加固的线路列表（如 ['AC_Line_20_Status', 'AC_Line_6_Status']）
        output_dir: 输出目录
    
    Returns:
        反事实MC数据文件路径
    """
    print(f"      [创建反事实MC数据]")
    print(f"        原始文件: {original_mc_path}")
    print(f"        加固线路: {target_lines}")
    
    # 读取原始MC数据的所有工作表
    xl = pd.ExcelFile(original_mc_path)
    all_sheets = {}
    for sheet_name in xl.sheet_names:
        all_sheets[sheet_name] = pd.read_excel(xl, sheet_name=sheet_name)
    
    # 修改 cluster_representatives 工作表
    target_sheet = 'cluster_representatives'
    if target_sheet not in all_sheets:
        raise ValueError(f"工作表 {target_sheet} 不存在于MC数据文件中")
    
    df = all_sheets[target_sheet].copy()
    time_cols = [f'Col_{i:02d}' for i in range(1, 49)]  # Col_01 到 Col_48
    
    # 解析线路编号并修改状态
    modified_lines = []
    for target_line in target_lines:
        # 从 AC_Line_20_Status 提取编号 20
        line_name = target_line.replace('_Status', '')  # AC_Line_20
        try:
            line_num = int(line_name.split('_')[-1])  # 20
        except ValueError:
            print(f"        - 跳过无效线路名: {target_line}")
            continue
        
        # 找到这条线路的所有行（row_in_sample == line_num）
        line_mask = df['row_in_sample'] == line_num
        n_rows = line_mask.sum()
        
        if n_rows == 0:
            print(f"        - 未找到线路 {line_name} (row_in_sample={line_num})")
            continue
        
        # 统计原始故障数
        original_faults = df.loc[line_mask, time_cols].sum().sum()
        
        # 将这条线路的48小时状态全部设为0（正常）
        df.loc[line_mask, time_cols] = 0
        
        modified_lines.append({
            'line': line_name,
            'line_num': line_num,
            'rows_affected': n_rows,
            'faults_removed': int(original_faults)
        })
        print(f"        - {line_name}: 消除 {int(original_faults)} 个故障点（影响 {n_rows} 个场景）")
    
    if not modified_lines:
        raise ValueError("没有有效的线路被修改")
    
    # 更新工作表
    all_sheets[target_sheet] = df
    
    # 生成输出文件名
    lines_str = '_'.join([m['line'].replace('AC_Line_', 'L') for m in modified_lines[:3]])
    if len(modified_lines) > 3:
        lines_str += f'_etc{len(modified_lines)}'
    cf_filename = f"counterfactual_mc_{lines_str}_reinforced.xlsx"
    cf_path = output_dir / cf_filename
    
    # 保存反事实MC数据文件
    with pd.ExcelWriter(cf_path, engine='openpyxl') as writer:
        for sheet_name, sheet_df in all_sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"        - 反事实MC数据已保存: {cf_path}")
    print(f"        - 共修改 {len(modified_lines)} 条线路")
    
    return str(cf_path), modified_lines


def _create_counterfactual_topology(
    original_topology_path: str,
    target_line: str,
    line_status: int,
    output_dir: Path
) -> str:
    """
    创建反事实拓扑文件（兼容旧版本）
    
    把目标线路的48小时状态全部设置为指定值
    
    Args:
        original_topology_path: 原始拓扑文件路径
        target_line: 目标线路名称（如 AC_Line_20_Status）
        line_status: 设置的状态值（0=故障, 1=正常）
        output_dir: 输出目录
    
    Returns:
        反事实拓扑文件路径
    """
    import shutil
    
    # 读取原始拓扑的所有工作表
    xl = pd.ExcelFile(original_topology_path)
    all_sheets = {}
    for sheet_name in xl.sheet_names:
        all_sheets[sheet_name] = pd.read_excel(xl, sheet_name=sheet_name)
    
    # 找到要修改的工作表
    target_sheet = 'RollingDecisionsOriginal'
    if target_sheet not in all_sheets:
        raise ValueError(f"工作表 {target_sheet} 不存在于拓扑文件中")
    
    df = all_sheets[target_sheet]
    
    # 转换列名：去掉 "_Status" 后缀（推理管道添加的）
    # AC_Line_20_Status -> AC_Line_20
    original_col_name = target_line.replace('_Status', '')
    
    if original_col_name not in df.columns:
        # 也尝试原始名称
        if target_line not in df.columns:
            raise ValueError(f"线路 {target_line} (或 {original_col_name}) 不存在于拓扑文件中。可用列: {[c for c in df.columns if 'Line' in c][:10]}...")
        original_col_name = target_line
    
    # 记录原始状态
    original_values = df[original_col_name].copy()
    n_changed = (original_values != line_status).sum()
    
    # 修改目标线路的所有48小时状态
    df[original_col_name] = line_status
    all_sheets[target_sheet] = df
    
    # 保存反事实拓扑文件（包含所有工作表）
    cf_filename = f"counterfactual_{original_col_name}_status{line_status}.xlsx"
    cf_path = output_dir / cf_filename
    
    with pd.ExcelWriter(cf_path, engine='openpyxl') as writer:
        for sheet_name, sheet_df in all_sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"      - 反事实拓扑已生成: {cf_path}")
    print(f"      - 修改内容: {original_col_name} 全部设置为 {line_status}")
    print(f"      - 修改行数: {n_changed} / {len(df)}")
    print(f"      - 保留工作表: {list(all_sheets.keys())}")
    
    return str(cf_path)


def _run_full_pipeline_via_julia(
    mc_data_path: str,
    case_path: str,
    output_dir: Path,
    project_root: Path = None,
    timeout: int = 1800
) -> Optional[Dict]:
    """
    通过Julia本地代码运行完整的弹性评估流程
    
    流程: 阶段划分 → 拓扑重构 → MESS调度
    
    Args:
        mc_data_path: MC数据文件路径（可以是原始或反事实数据）
        case_path: 配电网算例文件路径
        output_dir: 输出目录
        project_root: Julia项目根目录
        timeout: 超时时间(秒)
    
    Returns:
        包含输出文件路径和弹性指标的字典，失败时返回None
    """
    import subprocess
    
    if project_root is None:
        project_root = Path(__file__).resolve().parent
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义输出文件路径
    phase_output = str(output_dir / "scenario_phase_classification.xlsx")
    topology_output = str(output_dir / "topology_reconfiguration_results.xlsx")
    dispatch_output = str(output_dir / "mess_dispatch_results.xlsx")
    
    # 转换为正斜杠路径（Julia/Windows兼容）
    project_root_str = str(project_root).replace('\\', '/')
    mc_data_path_str = mc_data_path.replace('\\', '/')
    case_path_str = case_path.replace('\\', '/')
    phase_output_str = phase_output.replace('\\', '/')
    topology_output_str = topology_output.replace('\\', '/')
    dispatch_output_str = dispatch_output.replace('\\', '/')
    
    # 创建Julia脚本，执行完整流程
    julia_script = f'''
# 设置项目根目录
const PROJECT_ROOT = raw"{project_root_str}"

# 加载main.jl来设置环境
include(joinpath(PROJECT_ROOT, "main.jl"))

println("=" ^ 60)
println("反事实弹性评估 - 完整流程")
println("=" ^ 60)

# Step 1: 场景阶段分类
println("\\n[Step 1/3] 场景阶段分类...")
Workflows.run_classify_phases(
    input_path = raw"{mc_data_path_str}",
    output_path = raw"{phase_output_str}"
)

# Step 2: 滚动拓扑重构
println("\\n[Step 2/3] 滚动拓扑重构...")
Workflows.run_rolling_reconfig(
    case_file = raw"{case_path_str}",
    fault_file = raw"{mc_data_path_str}",
    stage_file = raw"{phase_output_str}",
    output_file = raw"{topology_output_str}"
)

# Step 3: MESS协同调度
println("\\n[Step 3/3] MESS协同调度...")
Workflows.run_mess_dispatch(
    case_path = raw"{case_path_str}",
    topology_path = raw"{topology_output_str}",
    fallback_topology = raw"{mc_data_path_str}",
    output_file = raw"{dispatch_output_str}"
)

println("\\n" * "=" ^ 60)
println("✓ 反事实弹性评估完成!")
println("=" ^ 60)
'''
    
    # 写入临时脚本
    script_path = project_root / "temp" / "_julia_counterfactual_pipeline.jl"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(julia_script)
    
    try:
        print(f"      - 正在通过Julia运行完整弹性评估流程...")
        print(f"      - MC数据: {Path(mc_data_path).name}")
        print(f"      - 预计耗时: 5-15分钟")
        print(f"      - 实时日志:")
        
        # 使用Popen实现实时日志输出
        import sys
        process = subprocess.Popen(
            ['julia', '--project=.', str(script_path)],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1  # 行缓冲
        )
        
        # 收集输出用于后续检查
        output_lines = []
        step_keywords = ['Step 1', 'Step 2', 'Step 3', '阶段分类', '拓扑重构', 'MESS', '调度', '完成', '✓', 'ERROR', 'error', 'Warning']
        
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            # 只打印关键进度信息，避免刷屏
            if any(kw in line for kw in step_keywords) and line.strip():
                print(f"        {line}")
        
        process.wait(timeout=timeout)
        stdout = '\n'.join(output_lines)
        
        # 检查是否成功
        success_indicators = ["反事实弹性评估完成", "MESS协同调度完成", "协同调度完成"]
        is_success = any(ind in stdout for ind in success_indicators) or process.returncode == 0
        
        if is_success and Path(dispatch_output).exists():
            print(f"      - Julia完整流程计算完成")
            
            # 读取弹性指标
            key_metrics_path = Path(dispatch_output).with_name(
                Path(dispatch_output).stem + "_key_metrics.json"
            )
            
            metrics = {}
            if key_metrics_path.exists():
                with open(key_metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
            
            return {
                'status': 'success',
                'phase_output': phase_output,
                'topology_output': topology_output,
                'dispatch_output': dispatch_output,
                'metrics': metrics
            }
        else:
            print(f"      - Julia计算失败 (returncode={process.returncode})")
            # 从输出中查找错误信息
            err_lines = [l for l in output_lines if 'ERROR' in l or 'error' in l.lower()][:3]
            for err in err_lines:
                print(f"      - 错误: {err[:200]}")
            return None
            
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"      - Julia计算超时 ({timeout}秒)")
        return None
    except Exception as e:
        print(f"      - Julia调用失败: {e}")
        return None


def _run_julia_dispatch_cli(
    topology_path: str,
    case_path: str,
    output_file: str,
    project_root: Path = None,
    timeout: int = 600
) -> Optional[Dict]:
    """
    通过命令行直接调用Julia计算弹性指标（使用main.jl）
    
    Args:
        topology_path: 拓扑文件路径
        case_path: 配电网数据文件路径
        output_file: 输出文件路径
        project_root: Julia项目根目录
        timeout: 超时时间(秒)
    
    Returns:
        计算结果或None（失败时）
    """
    import subprocess
    
    if project_root is None:
        project_root = Path(__file__).resolve().parent
    
    # 转换为正斜杠路径
    project_root_str = str(project_root).replace('\\', '/')
    case_path_str = case_path.replace('\\', '/')
    topology_path_str = topology_path.replace('\\', '/')
    output_file_str = output_file.replace('\\', '/')
    
    # 创建简单的Julia脚本，直接加载main.jl
    julia_script = f'''
# 设置项目根目录
const PROJECT_ROOT = raw"{project_root_str}"

# 加载main.jl来设置环境
include(joinpath(PROJECT_ROOT, "main.jl"))

# 调用MESS调度
Workflows.run_mess_dispatch(
    case_path = raw"{case_path_str}",
    topology_path = raw"{topology_path_str}",
    output_file = raw"{output_file_str}"
)
println("Julia计算完成!")
'''
    
    # 写入临时脚本
    script_path = project_root / "temp" / "_julia_dispatch_temp.jl"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(julia_script)
    
    try:
        print(f"      - 正在通过Julia本地计算...")
        # 使用bytes模式避免编码问题
        result = subprocess.run(
            ['julia', '--project=.', str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            timeout=timeout
        )
        
        # 解码输出
        stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
        stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
        
        if result.returncode == 0 or "Julia计算完成" in stdout or "MESS协同调度完成" in stdout:
            print(f"      - Julia本地计算完成")
            return {'status': 'success', 'method': 'cli'}
        else:
            print(f"      - Julia本地计算失败 (returncode={result.returncode})")
            if stderr:
                # 只显示关键错误信息
                err_lines = [l for l in stderr.split('\n') if 'ERROR' in l or 'error' in l.lower()]
                if err_lines:
                    print(f"      - 错误: {err_lines[0][:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"      - Julia计算超时 ({timeout}秒)")
        return None
    except Exception as e:
        print(f"      - Julia命令行调用失败: {e}")
        return None


def _run_julia_dispatch(
    topology_path: str,
    case_path: str,
    output_file: str,
    api_url: str = "http://localhost:5000",
    timeout: int = 600,
    use_cli_fallback: bool = True
) -> Optional[Dict]:
    """
    调用Julia计算弹性指标（优先使用API，失败时回退到命令行）
    
    Args:
        topology_path: 拓扑文件路径
        case_path: 配电网数据文件路径
        output_file: 输出文件路径
        api_url: API服务地址
        timeout: 超时时间(秒)
        use_cli_fallback: API失败时是否回退到命令行模式
    
    Returns:
        计算结果或None（失败时）
    """
    import requests
    
    # 首先尝试API方式
    api_success = False
    try:
        health_resp = requests.get(f"{api_url}/api/health", timeout=5)
        if health_resp.status_code == 200:
            payload = {
                "case_path": case_path,
                "topology_path": topology_path,
                "output_file": output_file
            }
            
            print(f"      - 正在通过API调用Julia计算...")
            response = requests.post(
                f"{api_url}/api/mess-dispatch",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    api_success = True
                    return result
                else:
                    print(f"      - API计算失败: {result.get('message', '未知错误')}")
            else:
                print(f"      - API返回错误: {response.status_code}")
    except Exception as e:
        print(f"      - API调用失败: {e}")
    
    # API失败，尝试命令行方式
    if not api_success and use_cli_fallback:
        print(f"      - 回退到命令行模式...")
        return _run_julia_dispatch_cli(
            topology_path=topology_path,
            case_path=case_path,
            output_file=output_file,
            timeout=timeout
        )
    
    return None


def _extract_metrics_from_dispatch(dispatch_path: str) -> Dict:
    """
    从调度结果文件提取弹性指标
    
    支持两种格式:
    1. Excel文件 (.xlsx) - 从HourlyDetails工作表读取
    2. JSON文件 (_key_metrics.json) - 直接读取汇总指标
    
    Returns:
        {'mean_loss': float, 'mean_over2h': float, 'total_loss': float}
    """
    dispatch_path = Path(dispatch_path)
    
    # 首先尝试读取对应的key_metrics.json文件
    key_metrics_path = dispatch_path.with_name(
        dispatch_path.stem + "_key_metrics.json"
    )
    
    if key_metrics_path.exists():
        with open(key_metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 从JSON获取期望失负荷和超标节点数
        total_loss = data.get('expected_load_shed_total', 0)
        violations = data.get('violations', [])
        n_violations = len(violations) if isinstance(violations, list) else 0
        
        # 计算平均值（假设100场景×48小时=4800样本）
        n_samples = 4800  # 100 scenarios × 48 hours
        mean_loss = total_loss / 100  # 每场景平均
        
        return {
            'mean_loss': mean_loss,
            'total_loss': total_loss,
            'mean_over2h': n_violations,  # 使用超标节点数作为近似
            'n_samples': n_samples,
            'supply_ratio': data.get('expected_supply_ratio', 0),
            'source': 'key_metrics.json'
        }
    
    # 回退到Excel文件读取
    df = pd.read_excel(dispatch_path, sheet_name='HourlyDetails')
    
    loss_col = 'Load_Shed' if 'Load_Shed' in df.columns else 'Total_Load_Loss'
    over2h_col = 'Nodes_Over_2h'
    
    return {
        'mean_loss': df[loss_col].mean(),
        'total_loss': df[loss_col].sum(),
        'mean_over2h': df[over2h_col].mean() if over2h_col in df.columns else 0,
        'n_samples': len(df),
        'source': 'excel'
    }


def _verify_counterfactual_with_julia(
    original_mc_path: str,
    original_dispatch_path: str,
    case_path: str,
    target_lines: List[str],
    predicted_loss_improvement: float,
    predicted_over2h_improvement: float,
    output_dir: Path,
    api_url: str = "http://localhost:5000"
) -> Dict:
    """
    使用Julia本地代码运行完整流程验证反事实推演的预测准确性
    
    验证方法（使用完整弹性评估流程）：
    1. 在MC数据中把目标线路48h状态全部设置为0（正常，模拟加固）
    2. 生成反事实MC数据文件
    3. 调用Julia main.jl完整流程：阶段划分→拓扑重构→MESS调度
    4. 对比原始结果和加固后结果
    
    改善率定义：
    - 原始状态：某些时刻线路正常，某些时刻故障
    - 加固后状态：线路48h全部正常（在MC数据中状态=0）
    - 改善 = (原始损失 - 加固后损失) / 原始损失
    - 这表示"加固后能减少多少损失"
    
    Args:
        original_mc_path: 原始MC数据文件路径
        original_dispatch_path: 原始调度结果文件路径
        case_path: 配电网数据文件路径
        target_lines: 要加固的目标线路列表
        predicted_loss_improvement: 模型预测的失负荷改善率
        predicted_over2h_improvement: 模型预测的复电超时改善率
        output_dir: 输出目录
        api_url: API服务地址（已弃用，保留兼容性）
    
    Returns:
        验证结果字典
    """
    print(f"\n    验证线路: {target_lines}")
    
    project_root = Path(__file__).resolve().parent
    
    # Step 1: 提取原始弹性指标
    print(f"      [Step 1] 读取原始弹性指标...")
    
    # 查找正确的原始key_metrics文件
    candidate_files = [
        project_root / "data" / "mess_dispatch_results_key_metrics.json",
        project_root / "data" / "mess_dispatch_report_key_metrics.json",
        Path(original_dispatch_path).with_name(
            Path(original_dispatch_path).stem + "_key_metrics.json"
        )
    ]
    
    original_metrics = None
    for candidate in candidate_files:
        if candidate.exists():
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                total_loss = data.get('expected_load_shed_total', 0)
                supply_ratio = data.get('expected_supply_ratio', 0)
                violations = data.get('violations', [])
                original_metrics = {
                    'total_loss': total_loss,
                    'mean_loss': total_loss / 100,  # 100场景
                    'mean_over2h': len(violations) if isinstance(violations, list) else 0,
                    'supply_ratio': supply_ratio,
                    'source': str(candidate)
                }
                print(f"        使用指标文件: {candidate.name}")
                break
            except Exception as e:
                continue
    
    if original_metrics is None:
        original_metrics = _extract_metrics_from_dispatch(original_dispatch_path)
    
    print(f"        原始期望失负荷: {original_metrics['total_loss']:.2f} kW·h")
    if 'supply_ratio' in original_metrics:
        print(f"        原始供电率: {original_metrics['supply_ratio']:.2%}")
    
    # Step 2: 创建反事实MC数据（在MC数据中把目标线路设为恒正常）
    print(f"      [Step 2] 创建反事实MC数据（线路加固后48h恒为正常）...")
    
    # 确保target_lines是列表
    if isinstance(target_lines, str):
        target_lines = [target_lines]
    
    try:
        cf_mc_path, modified_info = _create_counterfactual_mc_data(
            original_mc_path=original_mc_path,
            target_lines=target_lines,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"        创建反事实MC数据失败: {e}")
        return {'status': 'failed', 'error': str(e)}
    
    # Step 3: 调用Julia完整流程计算反事实弹性
    print(f"      [Step 3] 调用Julia完整流程计算反事实弹性...")
    print(f"        流程: 阶段划分 → 拓扑重构 → MESS调度")
    
    cf_output_dir = output_dir / "counterfactual_pipeline"
    cf_output_dir.mkdir(parents=True, exist_ok=True)
    
    result = _run_full_pipeline_via_julia(
        mc_data_path=cf_mc_path,
        case_path=case_path,
        output_dir=cf_output_dir,
        project_root=project_root
    )
    
    if result is None or result.get('status') != 'success':
        print(f"        Julia完整流程计算失败，无法验证")
        return {
            'status': 'failed',
            'error': 'Julia完整流程计算失败',
            'counterfactual_mc_path': cf_mc_path,
            'modified_lines': modified_info
        }
    
    # Step 4: 提取加固后弹性指标
    print(f"      [Step 4] 提取加固后弹性指标...")
    
    cf_dispatch_output = result['dispatch_output']
    cf_metrics = None
    
    # 优先从key_metrics.json读取
    cf_key_metrics_path = Path(cf_dispatch_output).with_name(
        Path(cf_dispatch_output).stem + "_key_metrics.json"
    )
    if cf_key_metrics_path.exists():
        try:
            with open(cf_key_metrics_path, 'r', encoding='utf-8') as f:
                cf_data = json.load(f)
            cf_metrics = {
                'total_loss': cf_data.get('expected_load_shed_total', 0),
                'mean_loss': cf_data.get('expected_load_shed_total', 0) / 100,
                'mean_over2h': len(cf_data.get('violations', [])),
                'supply_ratio': cf_data.get('expected_supply_ratio', 0),
                'source': str(cf_key_metrics_path)
            }
        except Exception as e:
            print(f"        读取key_metrics失败: {e}")
    
    if cf_metrics is None:
        cf_metrics = _extract_metrics_from_dispatch(cf_dispatch_output)
    
    print(f"        加固后期望失负荷: {cf_metrics['total_loss']:.2f} kW·h")
    if 'supply_ratio' in cf_metrics:
        print(f"        加固后供电率: {cf_metrics['supply_ratio']:.2%}")
    
    # Step 5: 计算实际改善率
    # 改善率 = (原始损失 - 加固后损失) / 原始损失
    actual_loss_improvement = (original_metrics['total_loss'] - cf_metrics['total_loss']) / (original_metrics['total_loss'] + 1e-8)
    actual_over2h_improvement = (original_metrics['mean_over2h'] - cf_metrics['mean_over2h']) / (original_metrics['mean_over2h'] + 1e-8) if original_metrics['mean_over2h'] > 0 else 0
    
    # 应用物理约束：改善率不能为负（如果加固后反而更差，视为0改善）
    actual_loss_improvement = max(0, actual_loss_improvement)
    actual_over2h_improvement = max(0, actual_over2h_improvement)
    
    # 计算预测误差
    loss_error = abs(predicted_loss_improvement - actual_loss_improvement)
    over2h_error = abs(predicted_over2h_improvement - actual_over2h_improvement)
    
    # 计算供电率提升
    supply_improvement = 0
    if 'supply_ratio' in original_metrics and 'supply_ratio' in cf_metrics:
        supply_improvement = cf_metrics['supply_ratio'] - original_metrics['supply_ratio']
    
    print(f"\n      [Julia完整流程验证结果]")
    print(f"        加固线路: {[m['line'] for m in modified_info]}")
    print(f"        失负荷:")
    print(f"          原始期望值: {original_metrics['total_loss']:.2f} kW·h")
    print(f"          加固后期望值: {cf_metrics['total_loss']:.2f} kW·h")
    print(f"          实际改善率: {actual_loss_improvement:.2%}")
    print(f"          预测改善率: {predicted_loss_improvement:.2%}")
    print(f"          预测误差: {loss_error:.2%}")
    
    if supply_improvement != 0:
        print(f"        供电率:")
        print(f"          原始: {original_metrics['supply_ratio']:.2%}")
        print(f"          加固后: {cf_metrics['supply_ratio']:.2%}")
        print(f"          提升: {supply_improvement:.2%}")
    
    return {
        'target_lines': [info['line'] for info in modified_info],
        'status': 'validated',
        'method': 'julia_full_pipeline',
        'loss_original': float(original_metrics['total_loss']),
        'loss_counterfactual': float(cf_metrics['total_loss']),
        'loss_improvement_actual': float(actual_loss_improvement),
        'loss_improvement_predicted': float(predicted_loss_improvement),
        'loss_prediction_error': float(loss_error),
        'supply_ratio_original': float(original_metrics.get('supply_ratio', 0)),
        'supply_ratio_counterfactual': float(cf_metrics.get('supply_ratio', 0)),
        'supply_ratio_improvement': float(supply_improvement),
        'over2h_original': float(original_metrics['mean_over2h']),
        'over2h_counterfactual': float(cf_metrics['mean_over2h']),
        'over2h_improvement_actual': float(actual_over2h_improvement),
        'over2h_improvement_predicted': float(predicted_over2h_improvement),
        'over2h_prediction_error': float(over2h_error),
        'counterfactual_mc_path': cf_mc_path,
        'counterfactual_pipeline_output': {
            'phase_output': result['phase_output'],
            'topology_output': result['topology_output'],
            'dispatch_output': result['dispatch_output']
        },
        'modified_lines_detail': modified_info
    }


def _verify_counterfactual_with_data(
    merged_data: pd.DataFrame,
    target_line: str,
    predicted_loss_improvement: float,
    predicted_over2h_improvement: float
) -> Dict:
    """
    使用现有数据验证反事实推演的预测准确性（回退方案）
    
    方法：比较该线路正常时 vs 故障时的实际数据
    - 实际改善 = (故障时均值 - 正常时均值) / 故障时均值
    - 预测误差 = |预测改善 - 实际改善|
    
    Args:
        merged_data: 合并后的推理数据
        target_line: 要加固的目标线路
        predicted_loss_improvement: 模型预测的失负荷改善率
        predicted_over2h_improvement: 模型预测的复电超时改善率
    
    Returns:
        验证结果字典
    """
    print(f"\n      [回退] 使用数据统计验证...")
    
    # 获取线路状态列
    line_col = target_line  # 已经是 AC_Line_X_Status 格式
    
    if line_col not in merged_data.columns:
        print(f"        - 警告: 未找到线路列 {line_col}")
        return None
    
    # 分组：线路正常(=1) vs 故障(=0)
    normal_mask = merged_data[line_col] == 1
    fault_mask = merged_data[line_col] == 0
    
    n_normal = normal_mask.sum()
    n_fault = fault_mask.sum()
    
    print(f"        - 正常样本数: {n_normal}")
    print(f"        - 故障样本数: {n_fault}")
    
    if n_fault == 0 or n_normal == 0:
        print(f"        - 样本不足，无法验证")
        return None
    
    # 获取失负荷列名
    loss_col = 'Load_Shed' if 'Load_Shed' in merged_data.columns else 'Total_Load_Loss'
    over2h_col = 'Nodes_Over_2h'
    
    # 计算实际均值
    loss_fault = merged_data.loc[fault_mask, loss_col].mean()
    loss_normal = merged_data.loc[normal_mask, loss_col].mean()
    
    over2h_fault = merged_data.loc[fault_mask, over2h_col].mean() if over2h_col in merged_data.columns else 0
    over2h_normal = merged_data.loc[normal_mask, over2h_col].mean() if over2h_col in merged_data.columns else 0
    
    # 计算实际改善率
    actual_loss_improvement = (loss_fault - loss_normal) / (loss_fault + 1e-8)
    actual_over2h_improvement = (over2h_fault - over2h_normal) / (over2h_fault + 1e-8) if over2h_fault > 0 else 0
    
    # 计算预测误差
    loss_error = abs(predicted_loss_improvement - actual_loss_improvement)
    over2h_error = abs(predicted_over2h_improvement - actual_over2h_improvement)
    
    print(f"\n        [数据统计验证结果]")
    print(f"          失负荷: 预测 {predicted_loss_improvement:.1%} vs 实际 {actual_loss_improvement:.1%} (误差 {loss_error:.1%})")
    print(f"          复电超时: 预测 {predicted_over2h_improvement:.1%} vs 实际 {actual_over2h_improvement:.1%} (误差 {over2h_error:.1%})")
    
    return {
        'target_line': target_line,
        'method': 'data_statistics',
        'n_normal_samples': int(n_normal),
        'n_fault_samples': int(n_fault),
        'loss_fault': float(loss_fault),
        'loss_normal': float(loss_normal),
        'loss_improvement_actual': float(actual_loss_improvement),
        'loss_improvement_predicted': float(predicted_loss_improvement),
        'loss_prediction_error': float(loss_error),
        'over2h_fault': float(over2h_fault),
        'over2h_normal': float(over2h_normal),
        'over2h_improvement_actual': float(actual_over2h_improvement),
        'over2h_improvement_predicted': float(predicted_over2h_improvement),
        'over2h_prediction_error': float(over2h_error),
        'status': 'validated'
    }


def run_inference_with_real_data(
    topology_path: Optional[str] = None,
    dispatch_path: Optional[str] = None,
    case_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    top_n_diagnosis: int = 5,
    top_n_prescriptions: int = 3,
    verify_with_julia: bool = True
) -> Dict:
    """
    使用真实弹性评估数据运行推理分析
    
    完整流程：
    1. 加载拓扑重构数据（100场景×48小时=4800行线路状态）
    2. 加载MESS调度数据（100场景×48小时=4800行弹性指标）
    3. 按(Scenario, TimeStep)合并为推理数据集
    4. 训练XGBoost代理模型：线路状态 → 失负荷量
    5. 使用SHAP进行归因分析，找出薄弱线路
    6. 生成反事实处方建议
    
    数据说明：
    - 输入X: 每行是一个(场景,时间步)组合的线路状态向量
    - 输出y: 对应的失负荷量或供电率
    - 样本数: 4800 (100场景 × 48小时)
    """
    print("=" * 70)
    print("配电网弹性推理系统 V2 - 基于真实弹性评估数据")
    print("=" * 70)
    print("\n数据结构说明:")
    print("  - 每个场景: 48小时时间序列")
    print("  - 总场景数: 100个")
    print("  - 总数据点: 4800 (100×48)")
    print("  - 模型输入: 线路状态向量 (0=断开, 1=联通)")
    print("  - 模型输出: 当前小时失负荷量/供电率")
    
    data_dir = PROJECT_ROOT / "data"
    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配电网数据文件路径（用于Julia验证）
    if case_path:
        case_file = Path(case_path)
    else:
        case_file = data_dir / "ac_dc_real_case.xlsx"
    
    # 查找输入文件
    if topology_path:
        topo_path = Path(topology_path)
    else:
        topo_path, _ = find_latest_results(data_dir)
    
    if dispatch_path:
        disp_path = Path(dispatch_path)
    else:
        _, disp_path = find_latest_results(data_dir)
    
    if topo_path is None:
        raise FileNotFoundError(
            "未找到拓扑重构结果文件！\n"
            "请先运行弹性评估: python api_server.py 然后调用 /api/resilience-assessment"
        )
    
    if disp_path is None:
        raise FileNotFoundError(
            "未找到调度结果文件！\n"
            "请先运行MESS调度: python api_server.py 然后调用 /api/mess-dispatch"
        )
    
    print(f"\n[Step 1] 输入文件")
    print(f"  拓扑数据: {topo_path}")
    print(f"  弹性指标: {disp_path}")
    
    # 构建推理数据
    print("\n" + "-" * 50)
    print("[Step 2] 构建推理数据 (合并拓扑+弹性)")
    print("-" * 50)
    
    builder = InferenceDataBuilder()
    builder.load_topology_data(str(topo_path))
    
    try:
        builder.load_resilience_data(str(disp_path))
    except ValueError as e:
        print(f"\n[错误] {e}")
        print("\n" + "=" * 70)
        print("解决方案")
        print("=" * 70)
        print("当前调度结果文件缺少每小时的详细弹性指标。")
        print("请重新运行MESS调度来生成完整数据（包含HourlyDetails工作表）：")
        print("")
        print("方法: 使用API")
        print("  1. 启动API服务: python api_server.py")
        print("  2. 调用MESS调度API: POST /api/mess-dispatch")
        print("=" * 70)
        raise
    
    merged_data = builder.merge_data()
    
    # 保存合并后的数据
    inference_data_path = output_dir / "inference_data_real.xlsx"
    builder.save_data(str(inference_data_path))
    
    # 运行推理分析
    print("\n" + "-" * 50)
    print("[Step 3] 综合推理分析 (多目标学习)")
    print("-" * 50)
    print("  代理模型: XGBoost (多目标)")
    print("  归因方法: SHAP TreeExplainer")
    print("  学习目标: 失负荷量 + 复电超时节点数 (综合)")
    
    try:
        import xgboost as xgb
        import shap
    except ImportError as e:
        print(f"[ERROR] 无法导入模块: {e}")
        print("请确保已安装依赖: pip install xgboost shap")
        raise
    
    # 识别所有线路状态列（AC_Line_, DC_Line_, VSC_Line_）
    all_line_cols = [col for col in merged_data.columns if col.endswith('_Status')]
    print(f"  特征数量: {len(all_line_cols)} 条线路")
    print(f"  样本数量: {len(merged_data)} (100场景×48小时)")
    
    # 显示线路类型分布
    ac_lines = [c for c in all_line_cols if c.startswith('AC_Line_')]
    dc_lines = [c for c in all_line_cols if c.startswith('DC_Line_')]
    vsc_lines = [c for c in all_line_cols if c.startswith('VSC_Line_')]
    print(f"    - AC线路: {len(ac_lines)}条")
    print(f"    - DC线路: {len(dc_lines)}条")
    print(f"    - VSC线路: {len(vsc_lines)}条")
    
    # 准备特征矩阵
    X = merged_data[all_line_cols].values
    
    # 检查是否有复电超时数据
    has_over2h = 'Nodes_Over_2h' in merged_data.columns
    
    # ===== 综合目标学习 =====
    print("\n  [综合学习] 线路状态 → (失负荷量, 复电超时)")
    
    # 目标1: 失负荷量
    y_loss = merged_data['Total_Load_Loss'].values
    
    # 目标2: 复电超时节点数（如果有）
    y_over2h = merged_data['Nodes_Over_2h'].values if has_over2h else np.zeros(len(merged_data))
    
    # 创建综合目标：归一化后加权组合
    # y_combined = α * norm(y_loss) + (1-α) * norm(y_over2h)
    loss_weight = 0.6  # 失负荷权重
    over2h_weight = 0.4  # 复电超时权重
    
    # 归一化
    y_loss_norm = (y_loss - y_loss.min()) / (y_loss.max() - y_loss.min() + 1e-8)
    y_over2h_norm = (y_over2h - y_over2h.min()) / (y_over2h.max() - y_over2h.min() + 1e-8) if y_over2h.max() > 0 else np.zeros_like(y_over2h)
    
    # 综合目标
    y_combined = loss_weight * y_loss_norm + over2h_weight * y_over2h_norm
    
    print(f"    - 失负荷范围: {y_loss.min():.2f} ~ {y_loss.max():.2f}")
    print(f"    - 超2h节点范围: {y_over2h.min()} ~ {y_over2h.max()}")
    print(f"    - 综合目标权重: 失负荷={loss_weight}, 复电超时={over2h_weight}")
    
    # 训练综合模型
    print("\n  [模型训练] XGBoost回归器...")
    model_combined = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model_combined.fit(X, y_combined)
    
    # 评估模型
    y_pred = model_combined.predict(X)
    mse = np.mean((y_combined - y_pred) ** 2)
    ss_res = np.sum((y_combined - y_pred) ** 2)
    ss_tot = np.sum((y_combined - y_combined.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"    - MSE: {mse:.6f}")
    print(f"    - R²: {r2:.4f}")
    
    # 同时训练单目标模型（用于对比和详细分析）
    model_loss = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
    model_loss.fit(X, y_loss)
    
    model_over2h = None
    if has_over2h and y_over2h.max() > 0:
        model_over2h = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
        model_over2h.fit(X, y_over2h)
    
    # ===== SHAP归因分析 =====
    print("\n  [SHAP归因] 计算综合敏感性...")
    
    # 综合模型的SHAP值
    explainer_combined = shap.TreeExplainer(model_combined)
    shap_values_combined = explainer_combined.shap_values(X)
    
    # 单目标SHAP值（用于详细分析）
    explainer_loss = shap.TreeExplainer(model_loss)
    shap_values_loss = explainer_loss.shap_values(X)
    
    shap_values_over2h = None
    if model_over2h is not None:
        explainer_over2h = shap.TreeExplainer(model_over2h)
        shap_values_over2h = explainer_over2h.shap_values(X)
    
    # 计算敏感性（全局特征重要性）
    sensitivity_combined = np.mean(np.abs(shap_values_combined), axis=0)
    sensitivity_loss = np.mean(np.abs(shap_values_loss), axis=0)
    sensitivity_over2h = np.mean(np.abs(shap_values_over2h), axis=0) if shap_values_over2h is not None else np.zeros(len(all_line_cols))
    
    # 创建敏感性排名DataFrame
    sensitivity_df = pd.DataFrame({
        'line': all_line_cols,
        'combined_sensitivity': sensitivity_combined,
        'loss_sensitivity': sensitivity_loss,
        'over2h_sensitivity': sensitivity_over2h
    }).sort_values('combined_sensitivity', ascending=False)
    
    # Top N薄弱线路
    top_lines = sensitivity_df.head(top_n_diagnosis)['line'].tolist()
    
    print(f"    - 综合敏感性Top {top_n_diagnosis}:")
    for i, row in sensitivity_df.head(top_n_diagnosis).iterrows():
        print(f"      {sensitivity_df.index.get_loc(i)+1}. {row['line']}: 综合={row['combined_sensitivity']:.4f}, 失负荷={row['loss_sensitivity']:.4f}, 超时={row['over2h_sensitivity']:.4f}")
    
    # ===== 统一反事实推演 =====
    print(f"\n  [反事实推演] 分析Top {top_n_prescriptions}线路加固效果...")
    print("    注: 应用物理约束 - 加固线路不应导致恶化")
    
    prescriptions = []
    for line in top_lines[:top_n_prescriptions]:
        line_idx = all_line_cols.index(line)
        
        # 找出该线路故障的样本
        fault_mask = X[:, line_idx] == 0
        if fault_mask.sum() == 0:
            print(f"    {line}: 无故障样本，跳过")
            continue
        
        X_fault = X[fault_mask].copy()
        y_loss_fault = y_loss[fault_mask]
        y_over2h_fault = y_over2h[fault_mask]
        y_combined_fault = y_combined[fault_mask]
        
        print(f"\n    分析线路: {line}")
        print(f"      - 故障样本数: {fault_mask.sum()}")
        
        # 构造反事实：假设该线路修复
        X_counterfactual = X_fault.copy()
        X_counterfactual[:, line_idx] = 1
        
        # 预测原始和反事实结果
        # 综合目标
        pred_combined_original = model_combined.predict(X_fault)
        pred_combined_cf = model_combined.predict(X_counterfactual)
        
        # 失负荷
        pred_loss_original = model_loss.predict(X_fault)
        pred_loss_cf = model_loss.predict(X_counterfactual)
        
        # 复电超时
        if model_over2h is not None:
            pred_over2h_original = model_over2h.predict(X_fault)
            pred_over2h_cf = model_over2h.predict(X_counterfactual)
        else:
            pred_over2h_original = np.zeros(len(X_fault))
            pred_over2h_cf = np.zeros(len(X_fault))
        
        # 应用物理约束：加固不应导致恶化
        # 如果预测反事实比原始更差，则将反事实限制为不超过原始值
        pred_loss_cf_constrained = np.minimum(pred_loss_cf, pred_loss_original)
        pred_over2h_cf_constrained = np.minimum(pred_over2h_cf, pred_over2h_original)
        
        # 计算改善（使用物理约束后的值）
        combined_improvement = (pred_combined_original.mean() - pred_combined_cf.mean()) / (pred_combined_original.mean() + 1e-8)
        combined_improvement = max(0, combined_improvement)  # 综合改善也不能为负
        
        loss_improvement = (pred_loss_original.mean() - pred_loss_cf_constrained.mean()) / (pred_loss_original.mean() + 1e-8)
        loss_improvement = max(0, loss_improvement)  # 物理约束：加固不应使失负荷恶化
        
        over2h_improvement_raw = (pred_over2h_original.mean() - pred_over2h_cf.mean()) / (pred_over2h_original.mean() + 1e-8) if pred_over2h_original.mean() > 0 else 0
        over2h_improvement = max(0, over2h_improvement_raw)  # 物理约束：加固不应使复电时间恶化
        
        # 标记是否应用了物理约束
        loss_constrained = pred_loss_cf.mean() > pred_loss_original.mean()
        over2h_constrained = pred_over2h_cf.mean() > pred_over2h_original.mean()
        
        print(f"      - 综合目标改善: {combined_improvement:.2%}")
        print(f"      - 失负荷改善: {loss_improvement:.2%} ({pred_loss_original.mean():.2f} → {pred_loss_cf_constrained.mean():.2f})")
        if loss_constrained:
            print(f"        [注意] 模型预测失负荷恶化 ({pred_loss_cf.mean():.2f})，已应用物理约束")
        print(f"      - 复电超时改善: {over2h_improvement:.2%} ({pred_over2h_original.mean():.2f} → {pred_over2h_cf_constrained.mean():.2f})")
        if over2h_constrained:
            print(f"        [注意] 模型预测复电超时恶化 ({pred_over2h_cf.mean():.2f})，已应用物理约束")
        
        # 生成建议
        if combined_improvement > 0.3:
            recommendation = f"强烈建议加固{line}。综合改善{combined_improvement*100:.1f}%，同时降低失负荷{loss_improvement*100:.1f}%和减少复电超时{over2h_improvement*100:.1f}%。"
        elif combined_improvement > 0.15:
            recommendation = f"建议加固{line}。综合改善{combined_improvement*100:.1f}%。"
        elif combined_improvement > 0:
            recommendation = f"可考虑加固{line}，综合改善{combined_improvement*100:.1f}%。"
        else:
            recommendation = f"加固{line}效果有限，建议优先考虑其他线路。"
        
        prescriptions.append({
            'target_line': line,
            'affected_samples': int(fault_mask.sum()),
            'combined_improvement': float(combined_improvement),
            'loss_original': float(pred_loss_original.mean()),
            'loss_counterfactual': float(pred_loss_cf_constrained.mean()),
            'loss_counterfactual_raw': float(pred_loss_cf.mean()),
            'loss_improvement': float(loss_improvement),
            'loss_constrained': bool(loss_constrained),
            'over2h_original': float(pred_over2h_original.mean()),
            'over2h_counterfactual': float(pred_over2h_cf_constrained.mean()),
            'over2h_counterfactual_raw': float(pred_over2h_cf.mean()),
            'over2h_improvement': float(over2h_improvement),
            'over2h_constrained': bool(over2h_constrained),
            'recommendation': recommendation
        })
    
    # ===== 反事实验证（使用Julia完整流程，失败则回退到数据统计）=====
    verification_results = None
    if verify_with_julia and prescriptions and len(prescriptions) > 0:
        print(f"\n  [反事实验证] 使用Julia完整流程验证模型预测...")
        print(f"    方法: 在MC数据中把目标线路48h状态全部设为0（正常），重新运行完整弹性评估流程")
        
        # 获取要验证的线路（可以是多条）
        lines_to_verify = [p['target_line'] for p in prescriptions[:1]]  # 默认只验证最重要的一条
        
        # 查找MC数据路径
        mc_data_path = data_dir / "mc_simulation_results_k100_clusters.xlsx"
        if not mc_data_path.exists():
            print(f"    - 警告: 未找到MC数据文件 {mc_data_path}")
            print(f"    - 跳过Julia完整流程验证，使用数据统计验证")
            mc_data_path = None
        
        if mc_data_path:
            # 尝试Julia完整流程验证
            verification_results = _verify_counterfactual_with_julia(
                original_mc_path=str(mc_data_path),
                original_dispatch_path=str(disp_path),
                case_path=str(case_file),
                target_lines=lines_to_verify,
                predicted_loss_improvement=prescriptions[0]['loss_improvement'],
                predicted_over2h_improvement=prescriptions[0]['over2h_improvement'],
                output_dir=output_dir
            )
        
        # 如果Julia失败，回退到数据统计验证
        if verification_results is None or verification_results.get('status') != 'validated':
            print(f"\n    Julia验证失败，回退到数据统计验证...")
            verification_results = _verify_counterfactual_with_data(
                merged_data=merged_data,
                target_line=prescriptions[0]['target_line'],
                predicted_loss_improvement=prescriptions[0]['loss_improvement'],
                predicted_over2h_improvement=prescriptions[0]['over2h_improvement']
            )
        
        if verification_results and verification_results.get('status') == 'validated':
            prescriptions[0]['verification'] = verification_results
    elif not verify_with_julia and prescriptions:
        print(f"\n  [反事实验证] 已跳过验证 (使用 --verify 启用)")
    
    # ===== 统计汇总 =====
    epsr = merged_data['Supply_Rate'].mean() if 'Supply_Rate' in merged_data.columns else 1 - y_loss.mean() / (y_loss.max() + 1e-8)
    
    analysis_result = {
        'statistics': {
            'epsr': float(epsr),
            'mean_load_loss': float(y_loss.mean()),
            'std_load_loss': float(y_loss.std()),
            'mean_over2h_nodes': float(y_over2h.mean()),
            'max_over2h_nodes': int(y_over2h.max()),
            'samples_with_over2h': int((y_over2h > 0).sum()),
            'sample_count': len(merged_data),
        },
        'model_performance': {
            'combined_r2': float(r2),
            'combined_mse': float(mse),
        },
        'sensitivity_ranking': sensitivity_df.head(15).to_dict('records'),
        'top_vulnerable_lines': top_lines,
        'prescriptions': prescriptions,
    }
    
    # 保存报告
    print("\n" + "-" * 50)
    print("[Step 4] 保存分析报告")
    print("-" * 50)
    
    # 生成Markdown报告
    md_content = _generate_comprehensive_report(analysis_result, len(all_line_cols))
    
    md_report_path = output_dir / "inference_report_real.md"
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  Markdown报告: {md_report_path}")
    
    # JSON报告
    json_report_path = output_dir / "inference_report_real.json"
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"  JSON报告: {json_report_path}")
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("综合分析完成！")
    print("=" * 70)
    
    print("\n【关键发现】")
    print(f"  期望供电率: {epsr:.2%}")
    print(f"  平均失负荷: {y_loss.mean():.2f}")
    print(f"  平均超2h节点数: {y_over2h.mean():.2f}")
    print(f"  含超时样本: {(y_over2h > 0).sum()} / {len(merged_data)} ({(y_over2h > 0).mean()*100:.1f}%)")
    print(f"  综合薄弱线路: {', '.join(top_lines)}")
    
    if prescriptions:
        best = prescriptions[0]
        print(f"\n【首选加固建议】")
        print(f"  目标线路: {best['target_line']}")
        print(f"  综合改善: {best['combined_improvement']:.1%}")
        print(f"  失负荷改善: {best['loss_improvement']:.1%}")
        print(f"  复电超时改善: {best['over2h_improvement']:.1%}")
        
        # 显示物理约束应用情况
        if best.get('loss_constrained') or best.get('over2h_constrained'):
            constrained = []
            if best.get('loss_constrained'):
                constrained.append("失负荷")
            if best.get('over2h_constrained'):
                constrained.append("复电超时")
            print(f"  物理约束: 已对{', '.join(constrained)}应用约束")
        
        # 显示验证结果
        if 'verification' in best and best['verification']:
            v = best['verification']
            if v.get('status') == 'validated':
                if v.get('method') == 'data_statistics':
                    print(f"\n【数据验证】对比正常({v['n_normal_samples']}样本) vs 故障({v['n_fault_samples']}样本)")
                elif v.get('method') == 'julia_full_pipeline':
                    target_lines_str = ', '.join(v.get('target_lines', []))
                    print(f"\n【Julia完整流程验证】加固线路: {target_lines_str}")
                    print(f"  原始期望失负荷: {v['loss_original']:.2f} kW·h → 加固后: {v['loss_counterfactual']:.2f} kW·h")
                    print(f"  原始供电率: {v['supply_ratio_original']:.2%} → 加固后: {v['supply_ratio_counterfactual']:.2%} (+{v['supply_ratio_improvement']:.2%})")
                else:
                    print(f"\n【Julia验证】原始 vs 加固后弹性对比")
                print(f"  失负荷: 预测{v['loss_improvement_predicted']:.1%} vs 实际{v['loss_improvement_actual']:.1%} (误差{v['loss_prediction_error']:.1%})")
                print(f"  复电超时: 预测{v['over2h_improvement_predicted']:.1%} vs 实际{v['over2h_improvement_actual']:.1%} (误差{v['over2h_prediction_error']:.1%})")
            else:
                print(f"\n【验证】失败: {v.get('error', '未知错误')}")
    
    return {
        "status": "success",
        "inference_data_path": str(inference_data_path),
        "markdown_report_path": str(md_report_path),
        "json_report_path": str(json_report_path),
        "report": analysis_result,
    }


def _generate_comprehensive_report(result: Dict, n_lines: int) -> str:
    """生成综合分析的Markdown报告"""
    stats = result['statistics']
    lines = [
        "# 配电网弹性综合推理分析报告",
        "",
        "## 一、统计评估结果",
        "",
        f"- **期望供电率 (EPSR)**: {stats['epsr']:.4f} ({stats['epsr']*100:.2f}%)",
        f"- **平均失负荷量**: {stats['mean_load_loss']:.4f}",
        f"- **失负荷标准差**: {stats['std_load_loss']:.4f}",
        f"- **平均超2h断电节点数**: {stats['mean_over2h_nodes']:.4f}",
        f"- **最大超2h断电节点数**: {stats['max_over2h_nodes']}",
        f"- **含超时样本数**: {stats['samples_with_over2h']} / {stats['sample_count']} ({stats['samples_with_over2h']/stats['sample_count']*100:.1f}%)",
        f"- **分析线路数**: {n_lines}",
        "",
        "## 二、综合敏感性分析",
        "",
        "### 学习目标",
        "",
        "综合目标 = 0.6×归一化(失负荷量) + 0.4×归一化(复电超时节点数)",
        "",
        f"### 模型性能",
        "",
        f"- **综合模型R²**: {result['model_performance']['combined_r2']:.4f}",
        "",
        "### 线路敏感性排名（综合目标）",
        "",
        "| 排名 | 线路 | 综合敏感性 | 失负荷敏感性 | 复电超时敏感性 |",
        "|------|------|-----------|-------------|---------------|",
    ]
    
    for i, row in enumerate(result['sensitivity_ranking'][:10]):
        lines.append(f"| {i+1} | {row['line']} | {row['combined_sensitivity']:.6f} | {row['loss_sensitivity']:.6f} | {row['over2h_sensitivity']:.6f} |")
    
    lines.extend([
        "",
        f"### Top {len(result['top_vulnerable_lines'])} 关键薄弱线路",
        "",
    ])
    for i, line in enumerate(result['top_vulnerable_lines'], 1):
        lines.append(f"{i}. {line}")
    
    lines.extend([
        "",
        "## 三、反事实策略推演",
        "",
    ])
    
    for i, p in enumerate(result['prescriptions'], 1):
        constrained_notes = []
        if p.get('loss_constrained', False):
            constrained_notes.append("失负荷")
        if p.get('over2h_constrained', False):
            constrained_notes.append("复电超时")
        
        lines.extend([
            f"### 策略 {i}: 加固 {p['target_line']}",
            "",
            f"- **受影响故障样本数**: {p['affected_samples']}",
            f"- **综合改善率**: {p['combined_improvement']:.2%}",
        ])
        
        if constrained_notes:
            lines.append(f"- **物理约束**: 已对{', '.join(constrained_notes)}应用物理约束（加固不应导致恶化）")
        
        lines.extend([
            "",
            "| 指标 | 原始值 | 加固后预测值 | 改善率 |",
            "|------|--------|-------------|--------|",
            f"| 失负荷量 | {p['loss_original']:.4f} | {p['loss_counterfactual']:.4f} | {p['loss_improvement']:.2%} |",
            f"| 超2h节点数 | {p['over2h_original']:.4f} | {p['over2h_counterfactual']:.4f} | {p['over2h_improvement']:.2%} |",
            "",
            f"**建议**: {p['recommendation']}",
            "",
        ])
        
        # 如果有验证结果，添加验证部分
        if 'verification' in p and p['verification']:
            v = p['verification']
            if v.get('status') == 'validated':
                if v.get('method') == 'data_statistics':
                    # 数据统计验证
                    lines.extend([
                        "#### 数据统计验证结果",
                        "",
                        f"验证方法: 对比线路正常时({v['n_normal_samples']}样本) vs 故障时({v['n_fault_samples']}样本)的实际数据",
                        "",
                        "| 指标 | 故障时均值 | 正常时均值 | 实际改善 | 预测改善 | 预测误差 |",
                        "|------|-----------|-----------|---------|---------|---------|",
                        f"| 失负荷量 | {v['loss_fault']:.2f} | {v['loss_normal']:.2f} | {v['loss_improvement_actual']:.2%} | {v['loss_improvement_predicted']:.2%} | {v['loss_prediction_error']:.2%} |",
                        f"| 超2h节点数 | {v['over2h_fault']:.2f} | {v['over2h_normal']:.2f} | {v['over2h_improvement_actual']:.2%} | {v['over2h_improvement_predicted']:.2%} | {v['over2h_prediction_error']:.2%} |",
                        "",
                    ])
                elif v.get('method') == 'julia_full_pipeline':
                    # Julia完整流程验证
                    target_lines_str = ', '.join(v.get('target_lines', []))
                    lines.extend([
                        "#### Julia完整流程验证结果",
                        "",
                        f"验证方法: 在MC数据中将线路 {target_lines_str} 的48h状态全部设为0（正常），重新运行完整弹性评估流程（阶段划分→拓扑重构→MESS调度）",
                        "",
                        "| 指标 | 原始值 | 加固后值 | 实际改善 | 预测改善 | 预测误差 |",
                        "|------|--------|---------|---------|---------|---------|",
                        f"| 期望失负荷(kW·h) | {v['loss_original']:.2f} | {v['loss_counterfactual']:.2f} | {v['loss_improvement_actual']:.2%} | {v['loss_improvement_predicted']:.2%} | {v['loss_prediction_error']:.2%} |",
                        f"| 供电率 | {v['supply_ratio_original']:.2%} | {v['supply_ratio_counterfactual']:.2%} | +{v['supply_ratio_improvement']:.2%} | - | - |",
                        f"| 超2h节点数 | {v['over2h_original']:.0f} | {v['over2h_counterfactual']:.0f} | {v['over2h_improvement_actual']:.2%} | {v['over2h_improvement_predicted']:.2%} | {v['over2h_prediction_error']:.2%} |",
                        "",
                    ])
                else:
                    # 兼容旧版Julia验证
                    lines.extend([
                        "#### Julia计算验证结果",
                        "",
                        "验证方法: 将目标线路48h状态全部设置为正常，用Julia重新计算弹性指标",
                        "",
                        "| 指标 | 原始均值 | 加固后均值 | 实际改善 | 预测改善 | 预测误差 |",
                        "|------|---------|-----------|---------|---------|---------|",
                        f"| 失负荷量 | {v['loss_original']:.2f} | {v['loss_counterfactual']:.2f} | {v['loss_improvement_actual']:.2%} | {v['loss_improvement_predicted']:.2%} | {v['loss_prediction_error']:.2%} |",
                        f"| 超2h节点数 | {v['over2h_original']:.2f} | {v['over2h_counterfactual']:.2f} | {v['over2h_improvement_actual']:.2%} | {v['over2h_improvement_predicted']:.2%} | {v['over2h_prediction_error']:.2%} |",
                        "",
                    ])
            else:
                # 验证失败
                lines.extend([
                    "#### 验证",
                    "",
                    f"- 验证失败: {v.get('error', '未知错误')}",
                    "",
                ])
        else:
            # 未进行验证
            lines.extend([
                "#### 验证",
                "",
                "- 使用 --verify 参数启用验证",
                "",
            ])
    
    lines.extend([
        "---",
        "",
        "## 总结",
        "",
        f"系统整体期望供电率为 {stats['epsr']*100:.2f}%，",
        f"平均每小时失负荷 {stats['mean_load_loss']:.2f}，",
        f"约 {stats['samples_with_over2h']/stats['sample_count']*100:.1f}% 的时间步存在节点复电超过2小时的情况。",
        "",
        f"**综合分析识别的关键薄弱线路**: {', '.join(result['top_vulnerable_lines'])}",
        "",
    ])
    
    if result['prescriptions']:
        best = result['prescriptions'][0]
        lines.append(f"**优先加固建议**: {best['target_line']}，预计综合改善 {best['combined_improvement']*100:.1f}%，")
        lines.append(f"可同时降低失负荷 {best['loss_improvement']*100:.1f}% 和减少复电超时 {best['over2h_improvement']*100:.1f}%。")
    
    return "\n".join(lines)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="配电网弹性推理分析 V2 - 使用真实弹性数据")
    parser.add_argument("--topology", "-t", type=str, default=None, help="拓扑重构结果文件")
    parser.add_argument("--dispatch", "-d", type=str, default=None, help="调度结果文件")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出目录")
    parser.add_argument("--top-n", type=int, default=5, help="Top N薄弱线路")
    parser.add_argument("--verify", action="store_true", help="使用Julia弹性评估验证反事实推演")
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="跳过Julia验证")
    parser.set_defaults(verify=True)  # 默认不验证（验证较慢）
    
    args = parser.parse_args()
    
    try:
        result = run_inference_with_real_data(
            topology_path=args.topology,
            dispatch_path=args.dispatch,
            output_dir=args.output,
            top_n_diagnosis=args.top_n,
            top_n_prescriptions=min(3, args.top_n),
            verify_with_julia=args.verify,
        )
        print(f"\n输出文件:")
        print(f"  {result['inference_data_path']}")
        print(f"  {result['markdown_report_path']}")
        print(f"  {result['json_report_path']}")
        
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n[ERROR] 缺少依赖: {e}")
        print("请安装: pip install xgboost shap")
        sys.exit(1)


if __name__ == "__main__":
    main()
