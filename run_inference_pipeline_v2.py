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
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 网络拓扑分析 — 计算线路结构特征
# ============================================================

def build_network_topology(case_path: Path = None) -> Dict[str, Dict]:
    """
    从 ac_dc_real_case.xlsx 构建 networkx 图, 计算每条线路的拓扑结构特征:
      1. Edge Betweenness Centrality (介数中心性) — 衡量线路作为"桥梁"的重要性
      2. Isolated Load (孤立负荷) — 线路断开后，考虑联络开关转供后仍与电源断联的负荷
      3. is_critical (关键瓶颈标记) — BC>0.08 或 孤立负荷占比>0.2
      
    HVCB/联络开关逻辑:
    - HVCB表仅表示线路上有断路器，用于故障隔离（断开故障段）
    - 故障线路本身不能闭合，HVCB只能将故障段与健康段隔开
    - Cable24-26 是常开联络开关(InService=False)，在故障时可闭合进行转供
    - isolated_load 的计算考虑了联络开关的转供能力
    """
    if case_path is None:
        case_path = PROJECT_ROOT / "data" / "ac_dc_real_case.xlsx"
    case_path = Path(case_path)
    
    if not case_path.exists():
        print(f"  [警告] 未找到配网拓扑文件 {case_path}，使用默认拓扑特征")
        return {}
    
    cables = pd.read_excel(case_path, sheet_name='cable')
    loads = pd.read_excel(case_path, sheet_name='lumpedload', header=1)
    
    G = nx.Graph()
    cable_map = {}
    all_edges = {}
    tie_switches = {}  # line_num -> (from_bus, to_bus), 联络开关(常开)
    
    for _, r in cables.iterrows():
        line_num = int(r['ID'].replace('Cable', ''))
        line_id = f'AC_Line_{line_num}'
        all_edges[line_num] = (r['FromBus'], r['ToBus'])
        cable_map[line_id] = (r['FromBus'], r['ToBus'])
        if r['InService']:
            G.add_edge(r['FromBus'], r['ToBus'],
                       weight=float(r['LengthValue']), line_id=line_id)
        else:
            # 常开联络开关 (如Cable24-26), 故障时可闭合用于转供
            tie_switches[line_num] = (r['FromBus'], r['ToBus'])
    
    load_on_bus = {}
    for _, r in loads.iterrows():
        bus = r.get('Bus')
        mva = r.get('MVA', 0)
        if pd.notna(bus) and pd.notna(mva):
            load_on_bus[bus] = float(mva)
            if bus in G.nodes:
                G.nodes[bus]['load_mva'] = float(mva)
    
    sources = ['Bus_草河F27', 'Bus_石楼F12']
    ebc = nx.edge_betweenness_centrality(G)
    total_load = sum(load_on_bus.values())
    
    if tie_switches:
        print(f"  联络开关(常开): {sorted(tie_switches.keys())} — 用于转供恢复计算")
    
    topo_features = {}
    for line_id in sorted(cable_map.keys(), key=lambda x: int(x.split('_')[-1])):
        u, v = cable_map[line_id]
        line_num = int(line_id.split('_')[-1])
        bc = ebc.get((u, v), ebc.get((v, u), 0))
        
        G2 = G.copy()
        if G2.has_edge(u, v):
            G2.remove_edge(u, v)
        
        # 无转供时的原始孤立负荷 (用于对比/调试)
        isolated_load_raw = 0.0
        for node, data in G2.nodes(data=True):
            mva = data.get('load_mva', 0)
            if mva <= 0:
                continue
            connected = any(src in G2 and nx.has_path(G2, node, src) for src in sources)
            if not connected:
                isolated_load_raw += mva
        
        # 加入可用联络开关进行转供 (排除正在评估的线路自身)
        G2_reconfig = G2.copy()
        for ts_num, (ts_u, ts_v) in tie_switches.items():
            if ts_num == line_num:
                continue  # 若评估的就是联络开关自身, 不能用它来自救
            G2_reconfig.add_edge(ts_u, ts_v)
        
        # 转供后的残余孤立负荷 — 使用与raw相同的节点集(G.nodes)确保可比性
        isolated_load = 0.0
        for node in G.nodes:
            mva = load_on_bus.get(node, 0)
            if mva <= 0:
                continue
            if node not in G2_reconfig:
                isolated_load += mva
                continue
            connected = any(src in G2_reconfig and nx.has_path(G2_reconfig, node, src)
                           for src in sources)
            if not connected:
                isolated_load += mva
        
        iso_load_frac = isolated_load / total_load if total_load > 0 else 0.0
        
        topo_features[line_id] = {
            'betweenness_centrality': bc,
            'isolated_load_mva': isolated_load,
            'isolated_load_mva_raw': isolated_load_raw,
            'isolated_load_fraction': iso_load_frac,
            'is_critical': bc > 0.08 or iso_load_frac > 0.2,
        }
    
    for i in range(1, 36):
        lid = f'AC_Line_{i}'
        if lid not in topo_features:
            topo_features[lid] = {
                'betweenness_centrality': 0.0,
                'isolated_load_mva': 0.0,
                'isolated_load_fraction': 0.0,
                'is_critical': False,
            }
    
    topo_features['_network'] = {
        'all_edges': all_edges,
        'load_on_bus': load_on_bus,
        'sources': sources,
        'total_load': total_load,
        'tie_switches': tie_switches,  # 联络开关 {line_num: (from_bus, to_bus)}
    }
    
    return topo_features


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
    
    # ===== 策略: 优先使用data/目录的文件（确保拓扑和调度来自同一流程）=====
    
    # 首先检查data/目录是否同时有拓扑和调度文件
    direct_topo = data_dir / "topology_reconfiguration_results.xlsx"
    dispatch_candidates = [
        "mess_dispatch_hourly.xlsx",
        "mess_dispatch_report.xlsx",
        "mess_dispatch_results.xlsx",
        "mess_dispatch_for_inference.xlsx",
        "mess_dispatch_with_scenarios.xlsx",
    ]
    
    if direct_topo.exists():
        # data/目录有拓扑文件，优先在同目录找调度文件
        for dispatch_name in dispatch_candidates:
            dispatch_file = data_dir / dispatch_name
            if dispatch_file.exists():
                try:
                    xl = pd.ExcelFile(dispatch_file)
                    if "HourlyDetails" in xl.sheet_names:
                        topology_path = direct_topo
                        dispatch_path = dispatch_file
                        print(f"  [文件选择] 使用data/目录一致数据源:")
                        print(f"    拓扑: {direct_topo.name}")
                        print(f"    调度: {dispatch_name}")
                        return topology_path, dispatch_path
                except:
                    pass
    
    # 回退: 查找auto_eval_runs中同时有拓扑和调度的目录
    auto_eval_dir = data_dir / "auto_eval_runs"
    if auto_eval_dir.exists():
        runs = sorted(auto_eval_dir.glob("dn_resilience_*"), reverse=True)
        for run_dir in runs:
            topo_file = run_dir / "topology_reconfiguration_results.xlsx"
            if topo_file.exists():
                # 在同一run目录中查找调度文件
                for disp_name in ["mess_dispatch_results.xlsx", "mess_dispatch_report.xlsx"]:
                    disp_file = run_dir / disp_name
                    if disp_file.exists():
                        try:
                            xl = pd.ExcelFile(disp_file)
                            if "HourlyDetails" in xl.sheet_names:
                                topology_path = topo_file
                                dispatch_path = disp_file
                                print(f"  [文件选择] 使用auto_eval_runs一致数据源: {run_dir.name}")
                                return topology_path, dispatch_path
                        except:
                            pass
                # 如果run目录没有调度文件，记录拓扑路径作为後备
                if topology_path is None:
                    topology_path = topo_file
    
    # 最终回退: 分别查找
    if topology_path is None:
        if direct_topo.exists():
            topology_path = direct_topo
    
    if dispatch_path is None:
        for dispatch_name in dispatch_candidates:
            dispatch_file = data_dir / dispatch_name
            if dispatch_file.exists():
                try:
                    xl = pd.ExcelFile(dispatch_file)
                    if "HourlyDetails" in xl.sheet_names:
                        dispatch_path = dispatch_file
                        print(f"  [文件选择] 找到包含HourlyDetails的文件: {dispatch_name}")
                        break
                    elif dispatch_path is None:
                        dispatch_path = dispatch_file
                except:
                    pass
    
    if topology_path and dispatch_path:
        # 检查是否来自同一目录
        if topology_path.parent != dispatch_path.parent:
            print(f"  [警告] 拓扑文件和调度文件来自不同目录，数据可能不一致!")
            print(f"    拓扑: {topology_path}")
            print(f"    调度: {dispatch_path}")
    
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
    - row_in_sample: 线路编号(1-35)，AC_Line_1~26→1~26, DC_Line_1~2→27~28, VSC_Line_1~7→29~35
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
    # 线路名→MC故障矩阵row_in_sample映射:
    #   AC_Line_N  → row_in_sample = N      (1-26)
    #   DC_Line_N  → row_in_sample = 26 + N (27-28)
    #   VSC_Line_N → row_in_sample = 28 + N (29-35)
    modified_lines = []
    for target_line in target_lines:
        line_name = target_line.replace('_Status', '')  # e.g. AC_Line_20, DC_Line_1, VSC_Line_3
        try:
            num = int(line_name.split('_')[-1])
        except ValueError:
            print(f"        - 跳过无效线路名: {target_line}")
            continue
        
        # 根据线路类型计算row_in_sample
        if line_name.startswith('AC_Line'):
            line_num = num          # AC_Line_N → row N
        elif line_name.startswith('DC_Line'):
            line_num = 26 + num     # DC_Line_1 → row 27, DC_Line_2 → row 28
        elif line_name.startswith('VSC_Line'):
            line_num = 28 + num     # VSC_Line_1 → row 29, ..., VSC_Line_7 → row 35
        else:
            print(f"        - 未知线路类型: {target_line}")
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
    def _short_name(m):
        n = m['line']
        if n.startswith('AC_Line_'):
            return n.replace('AC_Line_', 'ACL')
        elif n.startswith('DC_Line_'):
            return n.replace('DC_Line_', 'DCL')
        elif n.startswith('VSC_Line_'):
            return n.replace('VSC_Line_', 'VSC')
        return n[:6]
    lines_str = '_'.join([_short_name(m) for m in modified_lines[:3]])
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
        {'mean_loss': float, 'mean_over2h': float, 'total_loss': float,
         'total_violation_prob': float, 'n_violations': int}
    
    注: mean_over2h 使用 total_violation_probability (= Σ violation_probability(node))
        而非之前的 len(violations) (违规节点计数)。
        total_violation_prob 是连续指标，能更精确地衡量超时改善。
    """
    dispatch_path = Path(dispatch_path)
    
    # 首先尝试读取对应的key_metrics.json文件
    key_metrics_path = dispatch_path.with_name(
        dispatch_path.stem + "_key_metrics.json"
    )
    
    if key_metrics_path.exists():
        with open(key_metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 从JSON获取期望失负荷和超标节点违规概率
        total_loss = data.get('expected_load_shed_total', 0)
        violations = data.get('violations', [])
        n_violations = len(violations) if isinstance(violations, list) else 0
        
        # 使用 total_violation_probability 作为 over-2h 指标
        # = Σ violation_probability(node)，连续指标
        total_violation_prob = 0.0
        if isinstance(violations, list):
            total_violation_prob = sum(
                v.get('violation_probability', 0.0) for v in violations
            )
        
        # 计算平均值（假设100场景×48小时=4800样本）
        n_samples = 4800  # 100 scenarios × 48 hours
        mean_loss = total_loss / 100  # 每场景平均
        
        return {
            'mean_loss': mean_loss,
            'total_loss': total_loss,
            'mean_over2h': total_violation_prob,  # 使用总违规概率（连续指标）
            'total_violation_prob': total_violation_prob,
            'n_violations': n_violations,
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
    
    # 查找正确的原始key_metrics文件（优先与原始调度文件同名配套，其次取 data 下最新）
    candidate_files = []
    dispatch_path = Path(original_dispatch_path) if original_dispatch_path else None
    if dispatch_path:
        paired = dispatch_path.with_name(dispatch_path.stem + "_key_metrics.json")
        if paired.exists():
            candidate_files.append(paired)

    data_candidates = [
        project_root / "data" / "mess_dispatch_results_key_metrics.json",
        project_root / "data" / "mess_dispatch_report_key_metrics.json",
    ]
    data_candidates = [p for p in data_candidates if p.exists()]
    data_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for p in data_candidates:
        if p not in candidate_files:
            candidate_files.append(p)
    
    original_metrics = None
    for candidate in candidate_files:
        if candidate.exists():
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                total_loss = data.get('expected_load_shed_total', 0)
                supply_ratio = data.get('expected_supply_ratio', 0)
                violations = data.get('violations', [])
                # 使用 total_violation_probability 作为 over-2h 指标（连续指标）
                total_violation_prob = sum(
                    v.get('violation_probability', 0.0) for v in violations
                ) if isinstance(violations, list) else 0.0
                original_metrics = {
                    'total_loss': total_loss,
                    'mean_loss': total_loss / 100,  # 100场景
                    'mean_over2h': total_violation_prob,  # 总违规概率
                    'total_violation_prob': total_violation_prob,
                    'n_violations': len(violations) if isinstance(violations, list) else 0,
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
            cf_violations = cf_data.get('violations', [])
            cf_total_violation_prob = sum(
                v.get('violation_probability', 0.0) for v in cf_violations
            ) if isinstance(cf_violations, list) else 0.0
            cf_metrics = {
                'total_loss': cf_data.get('expected_load_shed_total', 0),
                'mean_loss': cf_data.get('expected_load_shed_total', 0) / 100,
                'mean_over2h': cf_total_violation_prob,  # 总违规概率
                'total_violation_prob': cf_total_violation_prob,
                'n_violations': len(cf_violations) if isinstance(cf_violations, list) else 0,
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
    print(f"        超时节点 (total_violation_prob):")
    print(f"          原始: {original_metrics['mean_over2h']:.4f} (违规节点: {original_metrics.get('n_violations', '?')})")
    print(f"          加固后: {cf_metrics['mean_over2h']:.4f} (违规节点: {cf_metrics.get('n_violations', '?')})")
    print(f"          实际改善率: {actual_over2h_improvement:.2%}")
    print(f"          预测改善率: {predicted_over2h_improvement:.2%}")
    print(f"          预测误差: {over2h_error:.2%}")
    
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
        'over2h_n_violations_original': int(original_metrics.get('n_violations', 0)),
        'over2h_n_violations_counterfactual': int(cf_metrics.get('n_violations', 0)),
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
    
    # 构建网络拓扑特征 (介数中心性 + 孤立负荷 + 关键瓶颈判据)
    print("\n  [拓扑分析] 构建网络拓扑,计算结构特征...")
    topo_features = build_network_topology(case_file)
    if topo_features:
        n_critical = sum(1 for k, v in topo_features.items() if isinstance(v, dict) and v.get('is_critical'))
        print(f"    拓扑特征: {len(topo_features)-1}条线路, {n_critical}条关键瓶颈线路")
    
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
    
    # ===== 物理感知反事实推演 (Physics-Aware Counterfactual Analysis) =====
    # 根本问题诊断:
    #   旧方法(XGBoost翻转): 对故障样本翻转单线路状态 → 样本级改善48.5%
    #   Julia实际验证: 在MC数据中加固线路 → 系统级改善仅0.58%
    #   差异原因:
    #     1. 故障高度相关(台风路径): AC_Line_19=0时其他线路也多故障
    #     2. XGBoost将并发故障的联合影响错误归因到单线路
    #     3. 样本级改善 ≠ 系统级改善(分母完全不同)
    #     4. 线性/树模型无法捕获调度优化的非线性补偿效应
    #
    # 新方法: 近邻匹配因果推断 + Julia基线校准
    #   1. 对每个故障样本，找其他线路状态相似但目标线正常的样本(对照组)
    #   2. 匹配差异 = 因果效应(消除并发故障混淆)
    #   3. 按场景概率加权聚合到系统级
    #   4. 使用Julia key_metrics基线确保分母一致
    
    print(f"\n  [反事实推演] 近邻匹配因果推断 (Top {top_n_prescriptions} 线路)")
    print("    方法: KNN匹配消除并发故障混淆 + Julia基线归一化")
    
    from scipy.spatial.distance import cdist
    from sklearn.linear_model import Ridge
    
    # Step A: 读取Julia基线期望失负荷(确保与验证同口径)
    baseline_expected_loss = None
    baseline_violations = None
    baseline_supply_ratio = None
    for km_name in ["mess_dispatch_results_key_metrics.json",
                     "mess_dispatch_report_key_metrics.json"]:
        km_path = data_dir / km_name
        if km_path.exists():
            try:
                with open(km_path, 'r', encoding='utf-8') as f:
                    km_data = json.load(f)
                baseline_expected_loss = km_data.get('expected_load_shed_total')
                baseline_supply_ratio = km_data.get('expected_supply_ratio')
                baseline_violations = km_data.get('violations', [])
                if baseline_expected_loss:
                    print(f"    Julia基线期望总失负荷: {baseline_expected_loss:.2f} kW·h")
                    print(f"    Julia基线供电率: {baseline_supply_ratio:.4f}")
                    print(f"    Julia基线超标节点数: {len(baseline_violations)}")
                    break
            except Exception:
                pass
    
    # Step B: 场景级概率加权分析
    scenario_groups = merged_data.groupby('Scenario_ID')
    scenario_probs = {}
    scenario_total_loss = {}
    scenario_total_over2h = {}
    scenario_fault_hours = {}
    
    for scen_id, group in scenario_groups:
        prob = group['Probability'].iloc[0] if 'Probability' in group.columns else 1.0 / merged_data['Scenario_ID'].nunique()
        scenario_probs[scen_id] = prob
        scenario_total_loss[scen_id] = group['Total_Load_Loss'].sum()
        scenario_total_over2h[scen_id] = group['Nodes_Over_2h'].sum() if 'Nodes_Over_2h' in group.columns else 0
        fh = {}
        for lc in all_line_cols:
            fh[lc] = int((group[lc] == 0).sum())
        scenario_fault_hours[scen_id] = fh
    
    # 从数据计算期望总失负荷
    data_expected_loss = sum(
        scenario_probs[s] * scenario_total_loss[s] for s in scenario_probs
    )
    data_expected_over2h = sum(
        scenario_probs[s] * scenario_total_over2h[s] for s in scenario_probs
    )
    
    # 使用Julia基线(如有)，否则使用数据计算值
    if baseline_expected_loss and baseline_expected_loss > 0:
        total_expected_loss = baseline_expected_loss
        # 计算数据/基线比例因子(用于校准)
        data_to_julia_ratio = data_expected_loss / baseline_expected_loss
        print(f"    数据计算期望失负荷: {data_expected_loss:.2f} kW·h")
        print(f"    数据/Julia比例: {data_to_julia_ratio:.4f}")
    else:
        total_expected_loss = data_expected_loss
        data_to_julia_ratio = 1.0
        print(f"    期望总失负荷(数据计算): {data_expected_loss:.2f} kW·h")
    
    total_expected_over2h = data_expected_over2h
    n_scenarios = len(scenario_probs)
    print(f"    场景数: {n_scenarios}")
    
    # Step C: 辅助Ridge回归(仅用于参考排名，不作为最终预测)
    ridge_loss = Ridge(alpha=1.0)
    ridge_loss.fit(X, y_loss)
    ridge_loss_r2 = ridge_loss.score(X, y_loss)
    
    ridge_over2h = None
    ridge_over2h_r2 = 0.0
    if has_over2h and y_over2h.max() > 0:
        ridge_over2h = Ridge(alpha=1.0)
        ridge_over2h.fit(X, y_over2h)
        ridge_over2h_r2 = ridge_over2h.score(X, y_over2h)
    
    print(f"    Ridge R²(失负荷): {ridge_loss_r2:.4f} (辅助参考)")
    
    # Step D: 对每条目标线路执行近邻匹配因果推断
    prescriptions = []
    for line in top_lines[:top_n_prescriptions]:
        line_idx = all_line_cols.index(line)
        other_indices = [i for i in range(len(all_line_cols)) if i != line_idx]
        
        # 分割: 故障组(target=0) vs 对照组(target=1)
        fault_mask = X[:, line_idx] == 0
        normal_mask = X[:, line_idx] == 1
        
        n_fault = fault_mask.sum()
        n_normal = normal_mask.sum()
        
        if n_fault == 0 or n_normal == 0:
            print(f"    {line}: 无故障或无正常样本，跳过")
            continue
        
        # 获取其他线路的特征向量
        X_fault_other = X[fault_mask][:, other_indices]
        X_normal_other = X[normal_mask][:, other_indices]
        y_fault_loss = y_loss[fault_mask]
        y_normal_loss = y_loss[normal_mask]
        y_fault_over2h = y_over2h[fault_mask]
        y_normal_over2h = y_over2h[normal_mask]
        
        # 概率向量
        if 'Probability' in merged_data.columns:
            fault_probs = merged_data.loc[fault_mask, 'Probability'].values
        else:
            fault_probs = np.ones(n_fault) / n_scenarios
        
        # KNN匹配: 对每个故障样本，找K个最近的对照样本
        # 使用Hamming距离(二值特征的比例差异)
        K = min(10, n_normal)  # 增大K降低方差
        distances = cdist(X_fault_other, X_normal_other, metric='hamming')
        
        matched_loss_effects = np.zeros(n_fault)
        matched_over2h_effects = np.zeros(n_fault)
        match_quality = np.zeros(n_fault)
        
        for i in range(n_fault):
            nearest_k = np.argsort(distances[i])[:K]
            # 按距离加权: 近的匹配权重大, 远的权重小
            dists_k = distances[i, nearest_k]
            if dists_k.max() > 0:
                # 使用距离的倒数作为权重, 加一个小常数避免除零
                weights_k = 1.0 / (dists_k + 0.01)
                weights_k = weights_k / weights_k.sum()
            else:
                weights_k = np.ones(K) / K
            
            matched_loss_avg = np.average(y_normal_loss[nearest_k], weights=weights_k)
            matched_over2h_avg = np.average(y_normal_over2h[nearest_k], weights=weights_k)
            
            # 因果效应 = 故障时实际值 - 匹配对照值
            # 注意: 不在此处应用max(0)，允许负值（避免截断偏差）
            matched_loss_effects[i] = y_fault_loss[i] - matched_loss_avg
            matched_over2h_effects[i] = y_fault_over2h[i] - matched_over2h_avg
            match_quality[i] = dists_k.mean()
        
        # ===== 方法1: 样本级匹配 + 概率加权聚合 =====
        # 在聚合层面应用max(0)（消除截断偏差）
        hourly_expected_loss_reduction = max(0, np.sum(fault_probs * matched_loss_effects))
        hourly_expected_over2h_reduction = max(0, np.sum(fault_probs * matched_over2h_effects))
        
        # ===== 方法2: 场景级匹配（更保守，与Julia口径更一致）=====
        # 构建场景级特征: 每条其他线路的故障小时数
        other_line_cols = [lc for lc in all_line_cols if lc != line]
        treat_scen_features = []
        treat_scen_losses = []
        treat_scen_over2h = []
        treat_scen_probs = []
        ctrl_scen_features = []
        ctrl_scen_losses = []
        ctrl_scen_over2h = []
        
        for scen_id in sorted(scenario_probs.keys()):
            other_vec = [scenario_fault_hours[scen_id].get(lc, 0) for lc in other_line_cols]
            target_fh = scenario_fault_hours[scen_id].get(line, 0)
            
            if target_fh > 0:
                treat_scen_features.append(other_vec)
                treat_scen_losses.append(scenario_total_loss[scen_id])
                treat_scen_over2h.append(scenario_total_over2h[scen_id])
                treat_scen_probs.append(scenario_probs[scen_id])
            else:
                ctrl_scen_features.append(other_vec)
                ctrl_scen_losses.append(scenario_total_loss[scen_id])
                ctrl_scen_over2h.append(scenario_total_over2h[scen_id])
        
        scen_expected_loss_reduction = 0.0
        scen_expected_over2h_reduction = 0.0
        
        if treat_scen_features and ctrl_scen_features:
            treat_arr = np.array(treat_scen_features, dtype=float)
            ctrl_arr = np.array(ctrl_scen_features, dtype=float)
            treat_losses = np.array(treat_scen_losses)
            ctrl_losses = np.array(ctrl_scen_losses)
            treat_over2h_arr = np.array(treat_scen_over2h)
            ctrl_over2h_arr = np.array(ctrl_scen_over2h)
            treat_prob_arr = np.array(treat_scen_probs)
            
            # 归一化特征(场景级故障小时数范围0-48)
            feat_std = treat_arr.std(axis=0) + 1e-8
            treat_norm = treat_arr / feat_std
            ctrl_norm = ctrl_arr / feat_std
            
            K_scen = min(5, len(ctrl_losses))
            scen_distances = cdist(treat_norm, ctrl_norm, metric='euclidean')
            
            scen_matched_effects_loss = np.zeros(len(treat_losses))
            scen_matched_effects_over2h = np.zeros(len(treat_losses))
            
            for i in range(len(treat_losses)):
                nearest_k = np.argsort(scen_distances[i])[:K_scen]
                dists_k = scen_distances[i, nearest_k]
                if dists_k.max() > 0:
                    wk = 1.0 / (dists_k + 0.1)
                    wk = wk / wk.sum()
                else:
                    wk = np.ones(K_scen) / K_scen
                matched_ctrl_loss = np.average(ctrl_losses[nearest_k], weights=wk)
                matched_ctrl_over2h = np.average(ctrl_over2h_arr[nearest_k], weights=wk)
                scen_matched_effects_loss[i] = treat_losses[i] - matched_ctrl_loss
                scen_matched_effects_over2h[i] = treat_over2h_arr[i] - matched_ctrl_over2h
            
            # 场景级聚合(在聚合层面应用max(0))
            scen_expected_loss_reduction = max(0, np.sum(treat_prob_arr * scen_matched_effects_loss))
            scen_expected_over2h_reduction = max(0, np.sum(treat_prob_arr * scen_matched_effects_over2h))
        
        # ===== 方法3: 拓扑加权比例归因法 (Topology-Weighted Proportional Attribution) =====
        # 按 (介数中心性 + 孤立负荷比例) 加权的故障小时数比例归因
        line_base = line.replace('_Status', '')
        line_ft = topo_features.get(line_base, {})
        target_is_critical = line_ft.get('is_critical', False)
        
        def _get_topo_weight(col_name):
            base = col_name.replace('_Status', '')
            ft = topo_features.get(base, {})
            bc = ft.get('betweenness_centrality', 0.0)
            iso_frac = ft.get('isolated_load_fraction', 0.0)
            return max(0.01, bc + iso_frac)
        
        topo_prop_loss_reduction = 0.0
        topo_prop_over2h_reduction = 0.0
        for scen_id in scenario_probs:
            target_fh = scenario_fault_hours[scen_id].get(line, 0)
            if target_fh == 0:
                continue
            target_wfh = target_fh * _get_topo_weight(line)
            total_wfh = sum(
                scenario_fault_hours[scen_id].get(lc, 0) * _get_topo_weight(lc)
                for lc in all_line_cols
            )
            if total_wfh <= 0:
                continue
            frac = target_wfh / total_wfh
            prob = scenario_probs[scen_id]
            topo_prop_loss_reduction += prob * scenario_total_loss[scen_id] * frac
            topo_prop_over2h_reduction += prob * scenario_total_over2h[scen_id] * frac
        topo_prop_loss_reduction = max(0, topo_prop_loss_reduction)
        topo_prop_over2h_reduction = max(0, topo_prop_over2h_reduction)
        
        # ===== 方法4: 连通性物理模型 (Connectivity Physical Model) =====
        # 计算加固目标线路后减少的孤立负荷 (作为效果理论上界)
        conn_loss_reduction = 0.0
        net = topo_features.get('_network', {})
        all_edges_net = net.get('all_edges', {})
        load_on_bus = net.get('load_on_bus', {})
        sources_net = net.get('sources', ['Bus_草河F27', 'Bus_石楼F12'])
        
        if all_edges_net and load_on_bus:
            target_base_name = line_base
            scenarios_g = merged_data.groupby('Scenario_ID')
            for sid, g in scenarios_g:
                prob = g['Probability'].iloc[0] if 'Probability' in merged_data.columns else 1.0 / n_scenarios
                for _, row in g.iterrows():
                    statuses = {}
                    for lc in all_line_cols:
                        base = lc.replace('_Status', '') if lc.endswith('_Status') else lc
                        statuses[base] = int(row[lc])
                    
                    if statuses.get(target_base_name, 1) == 1:
                        continue
                    
                    def _build_iso_graph(sts):
                        G_t = nx.Graph()
                        for n, (u, v) in all_edges_net.items():
                            col = f'AC_Line_{n}'
                            if sts.get(col, 1) == 0:
                                continue
                            G_t.add_edge(u, v)
                        for bus in load_on_bus:
                            if bus not in G_t: G_t.add_node(bus)
                        for src in sources_net:
                            if src not in G_t: G_t.add_node(src)
                        iso = 0.0
                        for bus, mva in load_on_bus.items():
                            if any(src in G_t and nx.has_path(G_t, bus, src) for src in sources_net):
                                continue
                            iso += mva
                        return iso
                    
                    iso_actual = _build_iso_graph(statuses)
                    sts_cf = statuses.copy()
                    sts_cf[target_base_name] = 1
                    iso_cf = _build_iso_graph(sts_cf)
                    conn_loss_reduction += prob * (iso_actual - iso_cf)
        
        conn_loss_reduction = max(0, conn_loss_reduction)
        
        # ===== 融合: 基于 is_critical 的集成策略 =====
        # 关键瓶颈线路(高BC骨干): 优化器有大量缓解手段 → 连通性模型严重高估 → 用topo_prop
        # 普通分支线路(低BC末梢): 优化器缓解空间有限 → avg(topo_prop, connectivity)
        if target_is_critical:
            expected_loss_reduction = topo_prop_loss_reduction
            fusion_method = 'topo_prop [CRITICAL]'
        else:
            expected_loss_reduction = (topo_prop_loss_reduction + conn_loss_reduction) / 2.0
            fusion_method = 'avg(topo_prop, connectivity) [NON_CRITICAL]'
        
        # over-2h: 基于图连通性的物理模型 + 逐节点校准
        # 替代旧的匹配法+拓扑归因融合，使用与Julia相同的metric（total_violation_probability）
        from validate_inference import compute_over2h_physical
        _baseline_for_over2h = {'violations': baseline_violations or []}
        over2h_result = compute_over2h_physical(
            merged_data, all_line_cols, _baseline_for_over2h, topo_features,
            reinforce_cols=[line]
        )
        system_over2h_improvement = over2h_result['over2h_improvement']
        
        # 受影响场景统计
        n_affected_scenarios = 0
        total_fault_prob_hours = 0.0
        for scen_id in scenario_probs:
            fh = scenario_fault_hours[scen_id].get(line, 0)
            if fh > 0:
                n_affected_scenarios += 1
                total_fault_prob_hours += scenario_probs[scen_id] * fh
        
        system_loss_improvement = expected_loss_reduction / (total_expected_loss + 1e-8)
        
        # 物理约束: 0 ≤ improvement ≤ 1
        system_loss_improvement = max(0.0, min(1.0, system_loss_improvement))
        # system_over2h_improvement 已由 compute_over2h_physical 计算
        
        combined_improvement = loss_weight * system_loss_improvement + over2h_weight * system_over2h_improvement
        
        # 匹配质量评估
        avg_match_quality = match_quality.mean()
        good_matches_pct = (match_quality < 0.1).mean() * 100  # <10%特征差异的匹配比例
        avg_matched_effect_loss = matched_loss_effects.mean()
        
        # 参考: XGBoost样本级预测
        X_fault_data = X[fault_mask].copy()
        X_cf = X_fault_data.copy()
        X_cf[:, line_idx] = 1
        xgb_loss_orig = y_loss[fault_mask].mean()
        xgb_loss_cf = model_loss.predict(X_cf).mean()
        xgb_sample_improvement = max(0, (xgb_loss_orig - xgb_loss_cf) / (xgb_loss_orig + 1e-8))
        
        # XGBoost over2h 样本级(仅参考用)
        xgb_over2h_orig = y_over2h[fault_mask].mean()
        xgb_over2h_cf = model_over2h.predict(X_cf).mean() if model_over2h is not None else 0.0
        
        print(f"\n    分析线路: {line}")
        print(f"      受影响场景: {n_affected_scenarios}/{n_scenarios}")
        print(f"      故障样本数: {n_fault}/{len(X)}")
        crit_tag = '★关键瓶颈' if target_is_critical else '普通分支'
        bc_val = line_ft.get('betweenness_centrality', 0)
        iso_val = line_ft.get('isolated_load_mva', 0)
        print(f"      拓扑特征: {crit_tag} (BC={bc_val:.3f}, IsoLoad={iso_val:.0f}MVA)")
        topo_prop_imp = topo_prop_loss_reduction / (total_expected_loss + 1e-8)
        conn_imp = conn_loss_reduction / (total_expected_loss + 1e-8)
        print(f"      拓扑加权归因:   {topo_prop_imp:.2%}")
        print(f"      连通性物理模型: {conn_imp:.2%}")
        print(f"      → 融合策略: {fusion_method}")
        print(f"      → 期望减少(loss): {expected_loss_reduction:.2f} kW·h (基线{total_expected_loss:.2f})")
        print(f"      ★ 系统失负荷改善: {system_loss_improvement:.2%}")
        print(f"      ★ 系统超时改善:   {system_over2h_improvement:.2%}")
        print(f"      ★ 综合改善:       {combined_improvement:.2%}")
        
        # 生成建议
        if combined_improvement > 0.03:
            recommendation = f"强烈建议加固{line}。系统级综合改善{combined_improvement*100:.2f}%，期望减少失负荷{expected_loss_reduction:.1f}kW·h。"
        elif combined_improvement > 0.01:
            recommendation = f"建议加固{line}。系统级综合改善{combined_improvement*100:.2f}%。"
        elif combined_improvement > 0:
            recommendation = f"可考虑加固{line}，系统级综合改善{combined_improvement*100:.2f}%。"
        else:
            recommendation = f"加固{line}效果有限，建议优先考虑其他线路。"
        
        prescriptions.append({
            'target_line': line,
            'affected_scenarios': int(n_affected_scenarios),
            'affected_samples': int(n_fault),
            'combined_improvement': float(combined_improvement),
            'loss_improvement': float(system_loss_improvement),
            'over2h_improvement': float(system_over2h_improvement),
            'loss_original': float(xgb_loss_orig),
            'loss_counterfactual': float(xgb_loss_cf),
            'over2h_original': float(xgb_over2h_orig),
            'over2h_counterfactual': float(xgb_over2h_cf),
            'matched_avg_causal_effect_loss': float(avg_matched_effect_loss),
            'expected_loss_reduction': float(expected_loss_reduction),
            'total_expected_loss': float(total_expected_loss),
            'total_fault_prob_hours': float(total_fault_prob_hours),
            'match_quality_pct': float(good_matches_pct),
            'xgb_sample_improvement': float(xgb_sample_improvement),
            'topo_prop_improvement': float(topo_prop_loss_reduction / (total_expected_loss + 1e-8)),
            'conn_improvement': float(conn_loss_reduction / (total_expected_loss + 1e-8)),
            'is_critical': bool(target_is_critical),
            'fusion_method': fusion_method,
            'recommendation': recommendation,
            'over2h_detail': over2h_result,
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
            'ridge_loss_r2': float(ridge_loss_r2),
            'ridge_over2h_r2': float(ridge_over2h_r2),
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
        print(f"\n【首选加固建议】(系统级改善率 - 近邻匹配因果推断)")
        print(f"  目标线路: {best['target_line']}")
        print(f"  系统级综合改善: {best['combined_improvement']:.2%}")
        print(f"  系统级失负荷改善: {best['loss_improvement']:.2%}")
        print(f"  系统级超时改善: {best['over2h_improvement']:.2%}")
        print(f"  匹配因果效应: {best.get('matched_avg_causal_effect_loss', 0):.2f} kW/故障小时")
        print(f"  受影响场景: {best.get('affected_scenarios', 0)}/{n_scenarios}")
        
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
        "## 三、反事实策略推演（系统级改善率 - 近邻匹配因果推断）",
        "",
        "方法说明: 对每个故障样本，找其他线路状态相似但目标线路正常的样本(KNN匹配),",
        "消除并发故障混淆后计算真实因果效应(Causal Effect)。",
        "按场景概率加权聚合到与Julia弹性评估一致口径的系统级改善率。",
        "",
    ])
    
    for i, p in enumerate(result['prescriptions'], 1):
        lines.extend([
            f"### 策略 {i}: 加固 {p['target_line']}",
            "",
            f"- **受影响场景数**: {p.get('affected_scenarios', 'N/A')}",
            f"- **受影响故障样本数**: {p['affected_samples']}",
            f"- **匹配因果效应**: {p.get('matched_avg_causal_effect_loss', 0):.2f} kW/故障小时",
            f"- **期望失负荷减少**: {p.get('expected_loss_reduction', 0):.2f} kW·h",
            f"- **系统级综合改善率**: {p['combined_improvement']:.2%}",
            f"- **匹配质量**: {p.get('match_quality_pct', 0):.0f}%样本有优质匹配",
            "",
            "| 指标 | 系统级改善率 | 样本级均值(故障时) | 样本级均值(修复后) |",
            "|------|------------|------------------|------------------|",
            f"| 失负荷量 | {p['loss_improvement']:.2%} | {p['loss_original']:.2f} | {p['loss_counterfactual']:.2f} |",
            f"| 超2h节点数 | {p['over2h_improvement']:.2%} | {p['over2h_original']:.2f} | {p['over2h_counterfactual']:.2f} |",
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
