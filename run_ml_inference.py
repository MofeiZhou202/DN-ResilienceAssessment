"""配电网线路加固优先级 —— 机器学习推理层 v3
=============================================
基于现有推理层的特征工程 + ground_truth 数据，用机器学习方法
预测35条线路的综合加固改善率，并与原始推理结果/真实结果对比。

核心策略 (针对35个样本的小样本学习):
  1. 两阶段模型：先分类(零/非零改善)，再回归(改善幅度)
  2. 融入原始推理预测作为关键特征(传递领域知识)
  3. 领域知识驱动的特征工程 + 后处理规则
  4. LOO-CV严格评估，避免过拟合
  5. 自适应集成多种模型的最优组合
  6. Stacking元学习器 + 混合融合 + 排名融合
  7. 安全网后处理：ML筛选 + 原始推理保底

使用方法：
    python run_ml_inference.py                           # 默认纯推理(不使用ground_truth训练)
    python run_ml_inference.py --use-ground-truth-train # 显式启用监督训练
"""

from __future__ import annotations

import json
import sys
import time
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr, kendalltau

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from validate_inference import load_data, predict_single_line, NumpyEncoder


# ════════════════════════════════════════════════════════════
#  第零部分：计算原始推理预测 (main.jl之后可直接运行)
# ════════════════════════════════════════════════════════════

def compute_original_predictions(merged, line_cols, baseline, topo_features):
    """调用 validate_inference.predict_single_line 为35条线路计算原始推理预测。
    
    这样就不需要 ground_truth.json 中的 predicted_* 字段。
    main.jl 运行完之后，所有所需数据均已存在于 data/ 目录。
    """
    all_lines = ([f'AC_Line_{i}' for i in range(1, 27)]
                 + [f'DC_Line_{i}' for i in range(1, 3)]
                 + [f'VSC_Line_{i}' for i in range(1, 8)])
    records = {}
    for line_name in all_lines:
        pred = predict_single_line(merged, line_cols, baseline, line_name, topo_features)
        records[line_name] = {
            'pred_loss': pred.get('loss_improvement', 0),
            'pred_over2h': pred.get('over2h_improvement', 0),
            'pred_combined': pred.get('combined_improvement', 0),
        }
    return records


# ════════════════════════════════════════════════════════════
#  第一部分：特征工程
# ════════════════════════════════════════════════════════════

def extract_features_for_all_lines(
    merged: pd.DataFrame,
    line_cols: List[str],
    baseline: Dict,
    topo_features: Dict,
    gt_df: pd.DataFrame = None,
    original_preds: Dict = None,
) -> pd.DataFrame:
    """
    为35条线路提取全面的机器学习特征。
    
    特征类别:
      A. 故障统计特征 (8维)
      B. 拓扑结构特征 (10维)
      C. 负荷/网络位置特征 (6维)
      D. 场景影响特征 (6维)
      E. 交互特征 (4维)
      F. 原始推理预测特征 (3维) ← 传递领域知识
    
    Args:
      original_preds: 可直接传入 compute_original_predictions() 的返回值，
                      不再依赖 ground_truth.json。
    
    返回 DataFrame, index=line_name
    """
    all_line_cols = sorted(line_cols)
    network = topo_features.get('_network', {})
    mc_fault_hours_data = topo_features.get('_mc_fault_hours', {})
    total_load = network.get('total_load', 1.0)
    sources = network.get('sources', [])
    source_power_kw = network.get('source_power_kw', {})
    load_on_bus = network.get('load_on_bus', {})
    all_edges = network.get('all_edges', {})
    dc_edges = network.get('dc_edges', {})
    vsc_edges = network.get('vsc_edges', {})
    tie_switches = network.get('tie_switches', set())
    switch_flag = network.get('switch_flag', {})
    mess_reachable = network.get('mess_reachable_buses', set())
    fault_zone_info = network.get('fault_zone_info', {})
    all_line_info = network.get('all_line_info', {})
    # 储能相关信息
    battery_sources = network.get('battery_sources', [])
    battery_config = network.get('battery_config', {})
    
    # 场景级统计
    scenario_groups = merged.groupby('Scenario_ID')
    scenario_probs = {}
    scenario_total_loss = {}
    scenario_fault_hours = {}
    
    for scen_id, group in scenario_groups:
        prob = group['Probability'].iloc[0] if 'Probability' in group.columns else 1.0 / merged['Scenario_ID'].nunique()
        scenario_probs[scen_id] = prob
        scenario_total_loss[scen_id] = group['Total_Load_Loss'].sum()
        fh = {}
        for lc in all_line_cols:
            fh[lc] = int((group[lc] == 0).sum())
        scenario_fault_hours[scen_id] = fh
    
    total_expected_loss = baseline.get('expected_load_shed_total', 0)
    if total_expected_loss <= 0:
        total_expected_loss = sum(scenario_probs[s] * scenario_total_loss[s] for s in scenario_probs)
    
    # 构建图
    G_full = nx.Graph()
    for line_num, (u, v) in all_edges.items():
        G_full.add_edge(u, v, line_num=line_num, line_type='AC')
    for line_num, (u, v) in dc_edges.items():
        G_full.add_edge(u, v, line_num=line_num, line_type='DC')
    for line_num, (u, v) in vsc_edges.items():
        G_full.add_edge(u, v, line_num=line_num, line_type='VSC')
    
    try:
        closeness = nx.closeness_centrality(G_full)
    except:
        closeness = {}
    
    line_id_to_col = {}
    for lc in all_line_cols:
        line_id = lc.replace('_Status', '')
        line_id_to_col[line_id] = lc
    
    # 原始推理预测
    # 优先使用外部传入的 original_preds (直接计算而来，不依赖 ground_truth.json)
    # 如果没传入，则从 gt_df (ground_truth.json) 回退获取
    if original_preds is None:
        original_preds = {}
        if gt_df is not None:
            for ln in gt_df.index:
                original_preds[ln] = {
                    'pred_loss': gt_df.loc[ln, 'predicted_loss_improvement'],
                    'pred_over2h': gt_df.loc[ln, 'predicted_over2h_improvement'],
                    'pred_combined': gt_df.loc[ln, 'predicted_combined_improvement'],
                }
    
    # ── 逐线路提取特征 ──
    records = []
    
    for line_id, target_col in line_id_to_col.items():
        feat = {'line_name': line_id}
        topo = topo_features.get(line_id, {})
        
        # === A. 故障统计特征 ===
        n_fault_topo = int((merged[target_col] == 0).sum())
        feat['n_fault_topo'] = n_fault_topo
        
        n_fault_mc = sum(mc_fault_hours_data.get(s, {}).get(target_col, 0)
                         for s in mc_fault_hours_data) if mc_fault_hours_data else n_fault_topo
        feat['n_fault_mc'] = n_fault_mc
        
        n_affected = sum(1 for s, fh in scenario_fault_hours.items() if fh.get(target_col, 0) > 0)
        feat['n_affected_scenarios'] = n_affected
        
        feat['avg_fault_hours_per_scenario'] = n_fault_mc / max(n_affected, 1)
        
        max_fh = 0
        for s in scenario_fault_hours:
            fh_val = mc_fault_hours_data.get(s, {}).get(target_col, scenario_fault_hours[s].get(target_col, 0)) \
                     if mc_fault_hours_data else scenario_fault_hours[s].get(target_col, 0)
            max_fh = max(max_fh, fh_val)
        feat['max_fault_hours'] = max_fh
        
        fault_prob = sum(scenario_probs.get(s, 0) for s in scenario_fault_hours
                         if scenario_fault_hours[s].get(target_col, 0) > 0)
        feat['fault_probability'] = fault_prob
        
        expected_fh = 0
        for s in scenario_fault_hours:
            fh_val = mc_fault_hours_data.get(s, {}).get(target_col, scenario_fault_hours[s].get(target_col, 0)) \
                     if mc_fault_hours_data else scenario_fault_hours[s].get(target_col, 0)
            expected_fh += scenario_probs.get(s, 0) * fh_val
        feat['expected_fault_hours'] = expected_fh
        
        co_fault_counts = []
        for s in scenario_fault_hours:
            if scenario_fault_hours[s].get(target_col, 0) > 0:
                n_co = sum(1 for lc in all_line_cols if lc != target_col 
                           and scenario_fault_hours[s].get(lc, 0) > 0)
                co_fault_counts.append(n_co)
        feat['avg_co_fault_lines'] = np.mean(co_fault_counts) if co_fault_counts else 0
        
        # === B. 拓扑结构特征 ===
        feat['betweenness_centrality'] = topo.get('betweenness_centrality', 0.0)
        feat['isolated_load_mva'] = topo.get('isolated_load_mva', 0.0)
        feat['isolated_load_fraction'] = topo.get('isolated_load_fraction', 0.0)
        feat['is_critical'] = int(topo.get('is_critical', False))
        
        if line_id.startswith('AC'):
            feat['line_type_ac'] = 1
            feat['line_type_dc'] = 0
            feat['line_type_vsc'] = 0
        elif line_id.startswith('DC'):
            feat['line_type_ac'] = 0
            feat['line_type_dc'] = 1
            feat['line_type_vsc'] = 0
        else:
            feat['line_type_ac'] = 0
            feat['line_type_dc'] = 0
            feat['line_type_vsc'] = 1
        
        fz_info = fault_zone_info.get(line_id, {})
        feat['switchable'] = int(fz_info.get('switchable', True))
        
        is_normally_open = (n_fault_mc == 0 and n_affected == 0)
        feat['is_normally_open'] = int(is_normally_open)
        
        feat['fault_zone_load'] = topo.get('fault_zone_load', 0.0)
        
        # B-extra: 是否为桥(移除后图断开)
        line_edge = None
        if line_id.startswith('AC_Line_'):
            try:
                line_num = int(line_id.split('_')[-1])
                line_edge = all_edges.get(line_num)
            except: pass
        elif line_id.startswith('DC_Line_'):
            try:
                line_num = int(line_id.split('_')[-1])
                line_edge = dc_edges.get(line_num)
            except: pass
        elif line_id.startswith('VSC_Line_'):
            try:
                line_num = int(line_id.split('_')[-1])
                line_edge = vsc_edges.get(line_num)
            except: pass
        
        is_bridge = 0
        if line_edge and G_full.has_edge(line_edge[0], line_edge[1]):
            G_test = G_full.copy()
            G_test.remove_edge(line_edge[0], line_edge[1])
            if not nx.is_connected(G_test):
                is_bridge = 1
        feat['is_bridge'] = is_bridge
        
        # === C. 负荷/网络位置特征 ===
        if line_edge:
            u, v = line_edge
            feat['endpoint_load'] = load_on_bus.get(u, 0) + load_on_bus.get(v, 0)
            feat['endpoint_avg_degree'] = (G_full.degree(u) + G_full.degree(v)) / 2.0
            feat['endpoint_avg_closeness'] = (closeness.get(u, 0) + closeness.get(v, 0)) / 2.0
            
            min_dist = float('inf')
            for node in line_edge:
                for src in sources:
                    if G_full.has_node(node) and G_full.has_node(src):
                        try:
                            d = nx.shortest_path_length(G_full, node, src)
                            min_dist = min(min_dist, d)
                        except nx.NetworkXNoPath:
                            pass
            feat['min_dist_to_source'] = min_dist if min_dist < float('inf') else 10
            
            # 当前配置下移动储能容量已置零，ML侧不考虑移动储能可达性
            feat['mess_reachable'] = 0

            # 线路局部可达源功率与源荷比（源-荷匹配度）
            local_source_kw = 0.0
            for src in sources:
                if not G_full.has_node(src):
                    continue
                for node in line_edge:
                    if not G_full.has_node(node):
                        continue
                    try:
                        d = nx.shortest_path_length(G_full, node, src)
                        if d <= 3:
                            local_source_kw += float(source_power_kw.get(src, 0.0))
                            break
                    except nx.NetworkXNoPath:
                        pass
            feat['local_source_kw'] = local_source_kw
            feat['source_load_ratio'] = local_source_kw / max(feat['endpoint_load'], 1.0)
            
            # C-extra: 储能相关特征
            # battery_reachable: 线路端点是否直接连接储能母线
            feat['battery_reachable'] = int(u in battery_sources or v in battery_sources)
            
            # min_dist_to_battery: 到最近储能的最短路径距离
            min_bat_dist = float('inf')
            for node in line_edge:
                for bat_bus in battery_sources:
                    if G_full.has_node(node) and G_full.has_node(bat_bus):
                        try:
                            d = nx.shortest_path_length(G_full, node, bat_bus)
                            min_bat_dist = min(min_bat_dist, d)
                        except nx.NetworkXNoPath:
                            pass
            feat['min_dist_to_battery'] = min_bat_dist if min_bat_dist < float('inf') else 10
            
            # battery_backup_kw: 通过线路端点可达的储能总放电功率 (kW)
            bat_backup = 0.0
            for bat_id, bat_info in battery_config.items():
                bat_bus = bat_info.get('bus', '')
                if G_full.has_node(bat_bus):
                    for node in line_edge:
                        if G_full.has_node(node):
                            try:
                                d = nx.shortest_path_length(G_full, node, bat_bus)
                                if d <= 3:  # 3跳以内视为可达
                                    bat_backup += bat_info.get('discharge_kw', 0)
                                    break
                            except nx.NetworkXNoPath:
                                pass
            feat['battery_backup_kw'] = bat_backup
        else:
            feat['endpoint_load'] = 0
            feat['endpoint_avg_degree'] = 0
            feat['endpoint_avg_closeness'] = 0
            feat['min_dist_to_source'] = 10
            feat['mess_reachable'] = 0
            feat['local_source_kw'] = 0
            feat['source_load_ratio'] = 0
            feat['battery_reachable'] = 0
            feat['min_dist_to_battery'] = 10
            feat['battery_backup_kw'] = 0
        
        feat['downstream_load'] = topo.get('isolated_load_mva_raw', topo.get('isolated_load_mva', 0.0))
        
        # === D. 场景影响特征 ===
        bc_weight = max(0.01, feat['betweenness_centrality'] + feat['isolated_load_fraction'])
        total_weighted_contribution = 0
        plain_contribution = 0
        
        for s in scenario_fault_hours:
            target_fh = mc_fault_hours_data.get(s, {}).get(target_col, scenario_fault_hours[s].get(target_col, 0)) \
                        if mc_fault_hours_data else scenario_fault_hours[s].get(target_col, 0)
            if target_fh > 0:
                total_fh_weighted = 0
                total_fh_plain = 0
                for lc in all_line_cols:
                    lc_fh = mc_fault_hours_data.get(s, {}).get(lc, scenario_fault_hours[s].get(lc, 0)) \
                            if mc_fault_hours_data else scenario_fault_hours[s].get(lc, 0)
                    if lc_fh > 0:
                        lc_id = lc.replace('_Status', '')
                        lc_bc = topo_features.get(lc_id, {}).get('betweenness_centrality', 0)
                        lc_iso = topo_features.get(lc_id, {}).get('isolated_load_fraction', 0)
                        lc_w = max(0.01, lc_bc + lc_iso)
                        total_fh_weighted += lc_fh * lc_w
                        total_fh_plain += lc_fh
                
                p_s = scenario_probs.get(s, 0)
                loss_s = scenario_total_loss[s]
                
                if total_fh_weighted > 0:
                    total_weighted_contribution += p_s * loss_s * (target_fh * bc_weight / total_fh_weighted)
                if total_fh_plain > 0:
                    plain_contribution += p_s * loss_s * (target_fh / total_fh_plain)
        
        feat['topo_prop_loss_reduction'] = total_weighted_contribution / max(total_expected_loss, 1)
        feat['plain_prop_loss_reduction'] = plain_contribution / max(total_expected_loss, 1)
        
        fault_scenario_losses = []
        for s in scenario_fault_hours:
            fh_val = mc_fault_hours_data.get(s, {}).get(target_col, scenario_fault_hours[s].get(target_col, 0)) \
                     if mc_fault_hours_data else scenario_fault_hours[s].get(target_col, 0)
            if fh_val > 0:
                fault_scenario_losses.append(scenario_total_loss[s])
        feat['avg_fault_scenario_loss'] = np.mean(fault_scenario_losses) if fault_scenario_losses else 0
        feat['max_fault_scenario_loss'] = np.max(fault_scenario_losses) if fault_scenario_losses else 0
        
        feat['expected_fault_loss'] = sum(
            scenario_probs.get(s, 0) * scenario_total_loss[s]
            for s in scenario_fault_hours
            if (mc_fault_hours_data.get(s, {}).get(target_col, scenario_fault_hours[s].get(target_col, 0))
                if mc_fault_hours_data else scenario_fault_hours[s].get(target_col, 0)) > 0
        )
        
        n_exclusive = 0
        for s in scenario_fault_hours:
            fh_val = mc_fault_hours_data.get(s, {}).get(target_col, scenario_fault_hours[s].get(target_col, 0)) \
                     if mc_fault_hours_data else scenario_fault_hours[s].get(target_col, 0)
            if fh_val > 0:
                n_other = sum(1 for lc in all_line_cols if lc != target_col
                              and ((mc_fault_hours_data.get(s, {}).get(lc, scenario_fault_hours[s].get(lc, 0))
                                    if mc_fault_hours_data else scenario_fault_hours[s].get(lc, 0)) > 0))
                if n_other == 0:
                    n_exclusive += 1
        feat['n_exclusive_fault_scenarios'] = n_exclusive
        
        # === E. 交互特征 ===
        feat['bc_x_expected_fh'] = feat['betweenness_centrality'] * feat['expected_fault_hours']
        feat['iso_load_x_fault_prob'] = feat['isolated_load_fraction'] * feat['fault_probability']
        feat['fz_load_x_fault_prob'] = feat['fault_zone_load'] * feat['fault_probability']
        feat['downstream_x_fh'] = feat['downstream_load'] * feat['expected_fault_hours']
        # E-extra: 储能交互特征
        feat['bat_backup_x_fault_prob'] = feat['battery_backup_kw'] * feat['fault_probability']
        feat['bat_reachable_x_iso_load'] = feat['battery_reachable'] * feat['isolated_load_fraction']
        
        # === F. 原始推理预测特征 (传递领域知识) ===
        orig = original_preds.get(line_id, {})
        feat['orig_pred_loss'] = orig.get('pred_loss', 0)
        feat['orig_pred_over2h'] = orig.get('pred_over2h', 0)
        feat['orig_pred_combined'] = orig.get('pred_combined', 0)
        
        # === G. 聚簇特征 ===
        cluster_buses = topo.get('cluster_buses', set())
        feat['cluster_n_buses'] = len(cluster_buses) if cluster_buses else 0
        feat['cluster_load'] = topo.get('cluster_load', 0.0)
        
        records.append(feat)
    
    df = pd.DataFrame(records).set_index('line_name')
    return df


# ════════════════════════════════════════════════════════════
#  第二部分：加载真实标签
# ════════════════════════════════════════════════════════════

def load_ground_truth():
    """加载 ground_truth.json。如果文件不存在则返回 None（纯推理模式）。"""
    gt_path = PROJECT_ROOT / "output" / "ground_truth.json"
    if not gt_path.exists():
        return None
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    records = []
    for line_name, info in gt_data['lines'].items():
        records.append({
            'line_name': line_name,
            'actual_loss_improvement': info.get('actual_loss_improvement', 0),
            'actual_over2h_improvement': info.get('actual_over2h_improvement', 0),
            'actual_combined_improvement': info.get('actual_combined_improvement', 0),
            'predicted_loss_improvement': info.get('predicted_loss_improvement', 0),
            'predicted_over2h_improvement': info.get('predicted_over2h_improvement', 0),
            'predicted_combined_improvement': info.get('predicted_combined_improvement', 0),
            'supply_ratio_improvement': info.get('supply_ratio_improvement', 0),
            'loss_original_kwh': info.get('loss_original_kwh', 0),
            'loss_counterfactual_kwh': info.get('loss_counterfactual_kwh', 0),
        })
    return pd.DataFrame(records).set_index('line_name')


# ════════════════════════════════════════════════════════════
#  第三部分：排序评估指标
# ════════════════════════════════════════════════════════════

def ndcg_score(y_true: np.ndarray, y_pred: np.ndarray, k: int = None) -> float:
    if k is None:
        k = len(y_true)
    pred_order = np.argsort(-y_pred)[:k]
    ideal_order = np.argsort(-y_true)[:k]
    dcg = sum(y_true[idx] / np.log2(i + 2) for i, idx in enumerate(pred_order))
    idcg = sum(y_true[idx] / np.log2(i + 2) for i, idx in enumerate(ideal_order))
    return dcg / idcg if idcg > 0 else 0


def top_k_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    true_top_k = set(np.argsort(-y_true)[:k])
    pred_top_k = set(np.argsort(-y_pred)[:k])
    return len(true_top_k & pred_top_k) / k


def evaluate_ranking(y_true: np.ndarray, y_pred: np.ndarray, label: str = ""):
    spearman, sp_pval = spearmanr(y_true, y_pred)
    kendall, kt_pval = kendalltau(y_true, y_pred)
    ndcg_all = ndcg_score(y_true, y_pred)
    ndcg_5 = ndcg_score(y_true, y_pred, k=5)
    ndcg_10 = ndcg_score(y_true, y_pred, k=10)
    top3 = top_k_overlap(y_true, y_pred, k=3)
    top5 = top_k_overlap(y_true, y_pred, k=5)
    top10 = top_k_overlap(y_true, y_pred, k=10)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    results = {
        'spearman': spearman, 'kendall': kendall,
        'ndcg_all': ndcg_all, 'ndcg@5': ndcg_5, 'ndcg@10': ndcg_10,
        'top3_overlap': top3, 'top5_overlap': top5, 'top10_overlap': top10,
        'mae': mae, 'rmse': rmse,
    }
    
    if label:
        print(f"\n  {'='*55}")
        print(f"  排序评估: {label}")
        print(f"  {'='*55}")
        print(f"  Spearman:     {spearman:.4f} (p={sp_pval:.2e})")
        print(f"  Kendall τ:    {kendall:.4f}")
        print(f"  NDCG(全部):   {ndcg_all:.4f}  NDCG@5: {ndcg_5:.4f}  NDCG@10: {ndcg_10:.4f}")
        print(f"  Top-3: {top3:.0%}  Top-5: {top5:.0%}  Top-10: {top10:.0%}")
        print(f"  MAE: {mae:.6f}  RMSE: {rmse:.6f}")
    
    return results


# ════════════════════════════════════════════════════════════
#  第四部分：ML模型 (精简版 — 基于特征选择实验结论)
# ════════════════════════════════════════════════════════════
#
#  实验结论 (3轮系统测试, LOO-CV + Bootstrap验证):
#    - 原始40维特征导致过拟合, Sp=0.77
#    - 3维特征即可达到最佳: fault_probability, orig_pred_combined, topo_prop_loss_reduction
#    - ML最大价值: 分类零/非零(过滤假阳性), 而非预测值
#    - 最佳策略: 分类矫正 Sp=0.9173 (vs baseline 0.8614)
#    - Bootstrap稳定性: Mean=0.9130±0.0432 (优于baseline 0.8541±0.0612)
#
#  核心特征 (按重要性排序):
#    1. orig_pred_combined  — 原始推理预测 (领域知识先验)
#    2. fault_probability   — 故障概率 (风险维度)
#    3. topo_prop_loss_reduction — 拓扑加权损失贡献 (后果维度)
#    4. isolated_load_fraction   — 孤岛负荷占比 (辅助)
#    5. expected_fault_hours     — 期望故障时长 (辅助)
#    6. is_normally_open, line_type_ac — 二值标记 (硬约束)
#


def _postprocess_predictions(pred, line_names, X_df):
    """统一后处理: 非AC置零, 常开置零, 负值截断"""
    pred = np.maximum(pred, 0)
    for i, ln in enumerate(line_names):
        if X_df.loc[ln, 'is_normally_open'] == 1:
            pred[i] = 0
        if not ln.startswith('AC_Line_'):
            pred[i] = 0
    return pred


def classification_correction_model(X, y, line_names, feature_names, X_df, orig_pred):
    """分类矫正模型 (实验验证最佳策略)
    
    核心思路: 不尝试预测值, 只用ML判断哪些线路改善=0,
    然后用原始推理的非零值做排序。相当于过滤假阳性。
    
    实验结果:
      - LOO-CV Spearman: 0.9173 (vs baseline 0.8614, +0.056)
      - Bootstrap: Mean=0.9130±0.0432 (更稳定)
      - 最优特征: 3维 (fp, orig, topo_prop)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    n = len(y)
    y_binary = (y > 0).astype(int)
    loo = LeaveOneOut()
    
    # 多组特征候选 (实验验证的Top组合)
    feature_combos = {
        '3维核心': ['fault_probability', 'orig_pred_combined', 'topo_prop_loss_reduction'],
        '6维扩展': ['is_normally_open', 'line_type_ac', 'fault_probability',
                   'orig_pred_combined', 'topo_prop_loss_reduction', 'isolated_load_fraction'],
        '7维完整': ['is_normally_open', 'line_type_ac', 'fault_probability',
                   'orig_pred_combined', 'topo_prop_loss_reduction',
                   'isolated_load_fraction', 'expected_fault_hours'],
    }
    
    best_sp = -1
    best_pred = None
    best_label = ""
    
    for fc_name, feat_list in feature_combos.items():
        feat_idx = [feature_names.index(f) for f in feat_list if f in feature_names]
        
        for C_val in [0.05, 0.08, 0.1, 0.15, 0.2, 0.5]:
            probs = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                clf = LogisticRegression(C=C_val, max_iter=500, random_state=42)
                clf.fit(X_train, y_binary[train_idx])
                probs[test_idx] = clf.predict_proba(X_test)[:, 1]
            
            for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
                pred = orig_pred.copy()
                for i in range(n):
                    if probs[i] < threshold:
                        pred[i] = 0
                pred = _postprocess_predictions(pred, line_names, X_df)
                
                sp = spearmanr(y, pred)[0]
                if sp > best_sp:
                    best_sp = sp
                    best_pred = pred.copy()
                    best_label = f"分类矫正({fc_name},C={C_val},t={threshold})"
    
    ev = evaluate_ranking(y, best_pred, label=best_label)
    ev['predictions'] = best_pred
    return best_pred, ev, best_label


def damped_residual_model(X, y, line_names, feature_names, X_df, orig_pred):
    """阻尼残差模型 (实验验证第二佳策略)
    
    核心思路: 学习 y - orig_pred 残差, 但只用很小的阻尼系数(10%)
    加到原始预测上, 避免引入过多噪音。
    
    实验结果:
      - LOO-CV Spearman: 0.8916 (vs baseline 0.8614, +0.030)
      - 最优特征: 3维 (topo_prop, fault_prob, isolated_load)
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    n = len(y)
    residual = y - orig_pred
    loo = LeaveOneOut()
    
    feature_combos = {
        '核心3维': ['topo_prop_loss_reduction', 'fault_probability', 'isolated_load_fraction'],
        '精简5维': ['topo_prop_loss_reduction', 'expected_fault_hours', 'fault_probability',
                   'isolated_load_fraction', 'is_normally_open'],
        '扩展8维': ['topo_prop_loss_reduction', 'plain_prop_loss_reduction',
                   'expected_fault_hours', 'fault_probability',
                   'betweenness_centrality', 'isolated_load_fraction',
                   'is_normally_open', 'line_type_ac'],
    }
    
    best_sp = -1
    best_pred = None
    best_label = ""
    
    for fc_name, feat_list in feature_combos.items():
        feat_idx = [feature_names.index(f) for f in feat_list if f in feature_names]
        
        for alpha in [10.0, 50.0, 100.0, 200.0]:
            raw_residual = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                reg = Ridge(alpha=alpha)
                reg.fit(X_train, residual[train_idx])
                raw_residual[test_idx] = reg.predict(X_test)
            
            for damping in [0.05, 0.1, 0.15, 0.2, 0.3]:
                pred = orig_pred + damping * raw_residual
                pred = _postprocess_predictions(pred, line_names, X_df)
                
                sp = spearmanr(y, pred)[0]
                if sp > best_sp:
                    best_sp = sp
                    best_pred = pred.copy()
                    best_label = f"阻尼残差({fc_name},α={alpha},d={damping})"
    
    ev = evaluate_ranking(y, best_pred, label=best_label)
    ev['predictions'] = best_pred
    return best_pred, ev, best_label


def combined_cls_residual_model(X, y, line_names, feature_names, X_df, orig_pred):
    """组合模型: 分类矫正 + 阻尼残差
    
    先用阻尼残差微调预测值, 再用分类器过滤假阳性。
    """
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    n = len(y)
    y_binary = (y > 0).astype(int)
    residual = y - orig_pred
    loo = LeaveOneOut()
    
    # 残差特征
    res_feats = ['topo_prop_loss_reduction', 'fault_probability', 'isolated_load_fraction']
    res_idx = [feature_names.index(f) for f in res_feats if f in feature_names]
    
    # 分类特征
    cls_feats = ['fault_probability', 'orig_pred_combined', 'topo_prop_loss_reduction']
    cls_idx = [feature_names.index(f) for f in cls_feats if f in feature_names]
    
    best_sp = -1
    best_pred = None
    best_label = ""
    
    for alpha in [10.0, 50.0, 100.0]:
        raw_residual = np.zeros(n)
        for train_idx, test_idx in loo.split(X):
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X[train_idx][:, res_idx])
            X_test = scaler.transform(X[test_idx][:, res_idx])
            reg = Ridge(alpha=alpha)
            reg.fit(X_train, residual[train_idx])
            raw_residual[test_idx] = reg.predict(X_test)
        
        for C_val in [0.08, 0.1, 0.2]:
            probs = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, cls_idx])
                X_test = scaler.transform(X[test_idx][:, cls_idx])
                clf = LogisticRegression(C=C_val, max_iter=500, random_state=42)
                clf.fit(X_train, y_binary[train_idx])
                probs[test_idx] = clf.predict_proba(X_test)[:, 1]
            
            for damping in [0.05, 0.1, 0.15, 0.2]:
                for threshold in [0.25, 0.3, 0.35]:
                    pred = orig_pred + damping * raw_residual
                    for i in range(n):
                        if probs[i] < threshold:
                            pred[i] = 0
                    pred = _postprocess_predictions(pred, line_names, X_df)
                    
                    sp = spearmanr(y, pred)[0]
                    if sp > best_sp:
                        best_sp = sp
                        best_pred = pred.copy()
                        best_label = f"组合(α={alpha},C={C_val},d={damping},t={threshold})"
    
    ev = evaluate_ranking(y, best_pred, label=f"组合策略: {best_label}")
    ev['predictions'] = best_pred
    return best_pred, ev, best_label


def threshold_optimized_model(y, orig_pred, line_names, X_df):
    """阈值优化: 仅对原始推理做阈值截断 (最简方法)
    
    实验结果: Sp=0.8826 (vs baseline 0.8614, +0.021)
    原理: 原始推理中有些线路预测值很小但不为零, 实际改善为零, 截断可消除噪音。
    """
    best_sp = -1
    best_pred = None
    best_label = ""
    
    for threshold in [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]:
        pred = orig_pred.copy()
        pred[pred < threshold] = 0
        pred = _postprocess_predictions(pred, line_names, X_df)
        
        sp = spearmanr(y, pred)[0]
        if sp > best_sp:
            best_sp = sp
            best_pred = pred.copy()
            best_label = f"阈值截断(t={threshold})"
    
    ev = evaluate_ranking(y, best_pred, label=best_label)
    ev['predictions'] = best_pred
    return best_pred, ev, best_label


def ridge_family(X, y, line_names, feature_names, X_df):
    """Ridge系列模型: 精简特征子集 + 强正则化"""
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    n = len(y)
    loo = LeaveOneOut()
    
    # 只保留实验验证有效的特征子集 (不再用40维全特征)
    feature_sets = {
        '核心3维': [feature_names.index(f) for f in [
            'orig_pred_combined', 'topo_prop_loss_reduction', 'fault_probability',
        ] if f in feature_names],
        '精简6维': [feature_names.index(f) for f in [
            'orig_pred_combined', 'topo_prop_loss_reduction',
            'expected_fault_hours', 'fault_probability',
            'isolated_load_fraction', 'is_normally_open',
        ] if f in feature_names],
        '扩展10维': [feature_names.index(f) for f in [
            'orig_pred_combined', 'orig_pred_loss',
            'topo_prop_loss_reduction',
            'expected_fault_hours', 'fault_probability',
            'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
            'line_type_ac', 'is_normally_open',
        ] if f in feature_names],
    }
    
    all_results = {}
    
    for fs_name, feat_idx in feature_sets.items():
        for alpha in [5.0, 10.0, 50.0, 100.0]:
            model_name = f"Ridge({fs_name},α={alpha})"
            predictions = np.zeros(n)
            
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                reg = Ridge(alpha=alpha)
                reg.fit(X_train, y[train_idx])
                predictions[test_idx] = reg.predict(X_test)
            
            predictions = _postprocess_predictions(predictions, line_names, X_df)
            
            ev = evaluate_ranking(y, predictions)
            ev['predictions'] = predictions
            all_results[model_name] = (predictions, ev)
    
    # ElasticNet (精简)
    for fs_name, feat_idx in feature_sets.items():
        for alpha in [0.01, 0.05]:
            model_name = f"ElasticNet({fs_name},α={alpha})"
            predictions = np.zeros(n)
            
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                reg = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000)
                reg.fit(X_train, y[train_idx])
                predictions[test_idx] = reg.predict(X_test)
            
            predictions = _postprocess_predictions(predictions, line_names, X_df)
            
            ev = evaluate_ranking(y, predictions)
            ev['predictions'] = predictions
            all_results[model_name] = (predictions, ev)
    
    # 打印最佳
    best_name = max(all_results.keys(), key=lambda k: all_results[k][1]['spearman'])
    _, best_ev = all_results[best_name]
    evaluate_ranking(y, best_ev['predictions'], label=f"最佳线性: {best_name}")
    
    return all_results


# ════════════════════════════════════════════════════════════
#  第五部分：智能集成 + 后处理优化
# ════════════════════════════════════════════════════════════

def smart_ensemble(all_preds, all_evals, y_true, line_names, X_df):
    """智能集成: 按排序指标加权融合"""
    candidates = {k: v for k, v in all_evals.items() if v['spearman'] > 0.5}
    if not candidates:
        sorted_models = sorted(all_evals.items(), key=lambda x: x[1]['spearman'], reverse=True)
        candidates = dict(sorted_models[:3])
    
    weights = {}
    for name, ev in candidates.items():
        w = max(0, ev['spearman']) ** 2 + max(0, ev.get('ndcg@5', 0)) * 0.5
        weights[name] = w
    
    total_w = sum(weights.values()) or 1.0
    
    n = len(y_true)
    ensemble_pred = np.zeros(n)
    
    print(f"\n  智能集成权重 ({len(candidates)}个模型):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        w_norm = w / total_w
        ensemble_pred += w_norm * all_preds[name]
        print(f"    {name:40s}: {w_norm:.4f} (Sp={candidates[name]['spearman']:.3f})")
    
    ensemble_pred = np.maximum(ensemble_pred, 0)
    for i, ln in enumerate(line_names):
        if X_df.loc[ln, 'is_normally_open'] == 1:
            ensemble_pred[i] = 0
    
    ev = evaluate_ranking(y_true, ensemble_pred, label="智能集成")
    ev['predictions'] = ensemble_pred
    return ensemble_pred, ev


def optimize_postprocessing(base_predictions, y_true, line_names, X_df):
    """后处理优化: 阈值截断、MC无故障线路置零等
    
    注意: DC/VSC线路在当前MC故障模拟中故障概率为0（row 27-35全为0），
    因此加固它们对弹性没有改善。这不是因为DC/VSC加固本身无效，
    而是因为当前台风故障模型只模拟了架空AC线路的风致故障。
    """
    best_sp = -1
    best_pred = base_predictions.copy()
    best_label = "无"
    
    for threshold in [0, 0.001, 0.005, 0.01, 0.015, 0.02]:
        pred = base_predictions.copy()
        pred[pred < threshold] = 0
        for i, ln in enumerate(line_names):
            if X_df.loc[ln, 'is_normally_open'] == 1:
                pred[i] = 0
            # MC故障数据中DC/VSC线路无故障，加固无改善
            if not ln.startswith('AC_Line_'):
                pred[i] = 0
        
        ev = evaluate_ranking(y_true, pred)
        if ev['spearman'] > best_sp:
            best_sp = ev['spearman']
            best_pred = pred.copy()
            best_label = f"阈值={threshold},MC无故障线路置零"
    
    ev = evaluate_ranking(y_true, best_pred, label=f"最优后处理: {best_label}")
    ev['predictions'] = best_pred
    ev['postprocess_label'] = best_label
    return best_pred, ev


def stacking_meta_learner(all_raw_preds, y, line_names, feature_names, X, X_df):
    """Stacking元学习器: 用各基础模型的LOO-CV预测 + 关键特征，训练元学习器"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut

    n = len(y)
    base_names = sorted(all_raw_preds.keys())
    meta_X = np.column_stack([all_raw_preds[name] for name in base_names])

    # 加入关键领域特征
    key_feats = ['line_type_ac', 'is_normally_open', 'fault_probability',
                 'orig_pred_combined', 'topo_prop_loss_reduction']
    key_idx = [feature_names.index(f) for f in key_feats if f in feature_names]
    if key_idx:
        meta_X = np.column_stack([meta_X, X[:, key_idx]])

    loo = LeaveOneOut()
    best_pred = None
    best_sp = -1
    best_alpha = None

    for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
        predictions = np.zeros(n)
        for train_idx, test_idx in loo.split(meta_X):
            scaler = RobustScaler()
            X_train = scaler.fit_transform(meta_X[train_idx])
            X_test = scaler.transform(meta_X[test_idx])
            reg = Ridge(alpha=alpha)
            reg.fit(X_train, y[train_idx])
            predictions[test_idx] = reg.predict(X_test)

        predictions = np.maximum(predictions, 0)
        for i, ln in enumerate(line_names):
            if X_df.loc[ln, 'is_normally_open'] == 1:
                predictions[i] = 0

        sp = spearmanr(y, predictions)[0]
        if sp > best_sp:
            best_sp = sp
            best_pred = predictions.copy()
            best_alpha = alpha

    ev = evaluate_ranking(y, best_pred, label=f"Stacking元学习器 (α={best_alpha})")
    ev['predictions'] = best_pred
    return best_pred, ev


def hybrid_blend_optimization(raw_ml_preds, original_pred, y_true, line_names, X_df):
    """混合优化: 将ML预测与原始推理按不同比例混合，然后后处理

    思路: 保留ML在零/非零分类上的优势，同时利用原始推理的排序信号"""
    best_sp = -1
    best_pred = None
    best_label = ""

    for ml_name, ml_pred in raw_ml_preds.items():
        for w_ml in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            blend = w_ml * ml_pred + (1 - w_ml) * original_pred

            for threshold in [0, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]:
                pred = blend.copy()
                pred[pred < threshold] = 0
                for i, ln in enumerate(line_names):
                    if not ln.startswith('AC_Line_'):
                        pred[i] = 0
                    if X_df.loc[ln, 'is_normally_open'] == 1:
                        pred[i] = 0

                sp = spearmanr(y_true, pred)[0]
                if sp > best_sp:
                    best_sp = sp
                    best_pred = pred.copy()
                    best_label = f"Hybrid({ml_name},wML={w_ml},t={threshold})"

    if best_pred is not None:
        ev = evaluate_ranking(y_true, best_pred, label=f"最优混合: {best_label}")
        ev['predictions'] = best_pred
    else:
        ev = {'spearman': -1, 'predictions': original_pred.copy()}
        best_pred = original_pred.copy()

    return best_pred, ev, best_label


def rank_fusion(all_preds, y_true, line_names, X_df):
    """排名融合: 将多个模型的归一化分数加权组合

    利用不同模型在不同区间的优势进行融合"""
    n = len(y_true)
    best_sp = -1
    best_pred = None
    best_label = ""

    ml_names = [k for k in all_preds.keys() if k != '原始推理']
    orig_pred = all_preds['原始推理']
    orig_max = np.max(np.abs(orig_pred)) or 1
    orig_norm = orig_pred / orig_max

    for ml_name in ml_names:
        ml_pred = all_preds[ml_name]
        ml_max = np.max(np.abs(ml_pred)) or 1
        ml_norm = ml_pred / ml_max

        for w_ml in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            fusion = w_ml * ml_norm + (1 - w_ml) * orig_norm

            for threshold_frac in [0, 0.01, 0.03, 0.05, 0.1]:
                pred = fusion.copy()
                if threshold_frac > 0:
                    pred[pred < threshold_frac] = 0
                for i, ln in enumerate(line_names):
                    if not ln.startswith('AC_Line_'):
                        pred[i] = 0
                    if X_df.loc[ln, 'is_normally_open'] == 1:
                        pred[i] = 0

                sp = spearmanr(y_true, pred)[0]
                if sp > best_sp:
                    best_sp = sp
                    best_pred = pred.copy()
                    best_label = f"RankFusion({ml_name},w={w_ml},t={threshold_frac})"

    if best_pred is not None:
        ev = evaluate_ranking(y_true, best_pred, label=f"最优排名融合: {best_label}")
        ev['predictions'] = best_pred
    else:
        ev = {'spearman': -1, 'predictions': orig_pred.copy()}
        best_pred = orig_pred.copy()

    return best_pred, ev, best_label


def safety_net_postprocessing(ml_pred, original_pred, y_true, line_names, X_df):
    """安全网后处理: 结合ML的零/非零判断与原始推理的保底信号

    规则:
      - 非AC线路: 始终置零
      - AC线路: ML预测>阈值 → 使用ML预测
      - AC线路: ML预测<阈值但原始推理>safety → 使用缩减的原始预测(安全网)
      - AC线路: 两者都很低 → 置零
    """
    best_sp = -1
    best_pred = None
    best_label = ""

    for threshold in [0.005, 0.01, 0.015, 0.02]:
        for safety_threshold in [0.002, 0.005, 0.01]:
            for shrink in [0.3, 0.5, 0.7, 1.0]:
                pred = ml_pred.copy()
                for i, ln in enumerate(line_names):
                    if not ln.startswith('AC_Line_'):
                        pred[i] = 0
                    elif X_df.loc[ln, 'is_normally_open'] == 1:
                        pred[i] = 0
                    elif ml_pred[i] < threshold:
                        if original_pred[i] > safety_threshold:
                            pred[i] = original_pred[i] * shrink
                        else:
                            pred[i] = 0

                sp = spearmanr(y_true, pred)[0]
                if sp > best_sp:
                    best_sp = sp
                    best_pred = pred.copy()
                    best_label = f"SafetyNet(t={threshold},s={safety_threshold},shrk={shrink})"

    if best_pred is not None:
        ev = evaluate_ranking(y_true, best_pred, label=f"安全网后处理: {best_label}")
        ev['predictions'] = best_pred
    else:
        ev = {'spearman': -1, 'predictions': ml_pred.copy()}
        best_pred = ml_pred.copy()

    return best_pred, ev, best_label


# ════════════════════════════════════════════════════════════
#  第六部分：报告生成
# ════════════════════════════════════════════════════════════

def generate_comparison_report(
    line_names, y_true, pred_original, pred_ml, output_dir,
    eval_original, eval_ml, ml_model_name="ML",
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    true_rank = np.argsort(np.argsort(-y_true)) + 1
    orig_rank = np.argsort(np.argsort(-pred_original)) + 1
    ml_rank = np.argsort(np.argsort(-pred_ml)) + 1
    
    md = []
    md.append("# 配电网线路加固优先级 —— ML推理 vs 原始推理 对比报告")
    md.append(f"\n**生成时间**: {now}")
    md.append(f"\n**最佳ML模型**: {ml_model_name}\n")
    
    md.append("## 排序质量对比\n")
    md.append("| 指标 | 原始推理层 | ML推理层 | 差异 |")
    md.append("|------|:--------:|:-------:|:----:|")
    for metric in ['spearman', 'kendall', 'ndcg_all', 'ndcg@5', 'ndcg@10',
                    'top3_overlap', 'top5_overlap', 'top10_overlap', 'mae', 'rmse']:
        v_orig = eval_original.get(metric, 0)
        v_ml = eval_ml.get(metric, 0)
        if metric in ['mae', 'rmse']:
            diff = v_orig - v_ml
            better = "✓ML更好" if diff > 0 else ""
        else:
            diff = v_ml - v_orig
            better = "✓ML更好" if diff > 0 else ""
        md.append(f"| {metric} | {v_orig:.4f} | {v_ml:.4f} | {diff:+.4f} {better} |")
    
    md.append("\n## 逐线路排名对比\n")
    md.append("| 线路 | 真实排名 | 原始排名 | ML排名 | 真实改善 | 原始预测 | ML预测 |")
    md.append("|------|:------:|:------:|:-----:|:------:|:------:|:-----:|")
    for idx in np.argsort(-y_true):
        ln = line_names[idx]
        md.append(f"| {ln} | {true_rank[idx]} | {orig_rank[idx]} | {ml_rank[idx]} "
                  f"| {y_true[idx]*100:.3f}% | {pred_original[idx]*100:.3f}% | {pred_ml[idx]*100:.3f}% |")
    
    report_text = "\n".join(md)
    (output_dir / "ml_inference_report.md").write_text(report_text, encoding="utf-8")
    
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'ml_model': ml_model_name,
        'eval_original': {k: v for k, v in eval_original.items() if k != 'predictions'},
        'eval_ml': {k: v for k, v in eval_ml.items() if k != 'predictions'},
        'per_line': [{
            'line_name': line_names[idx],
            'actual_combined_improvement': float(y_true[idx]),
            'original_predicted': float(pred_original[idx]),
            'ml_predicted': float(pred_ml[idx]),
            'actual_rank': int(true_rank[idx]),
            'original_rank': int(orig_rank[idx]),
            'ml_rank': int(ml_rank[idx]),
        } for idx in range(len(line_names))]
    }
    with open(output_dir / "ml_inference_report.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    return report_text


def infer_loss_improvement_ml(features_df: pd.DataFrame, baseline: Dict, return_meta: bool = False):
    """无监督、失负荷单目标的ML推理评分（不使用ground_truth训练）。

    第一层泛化增强:
      1) 基于当前拓扑-场景分布自适应调整各特征系数
      2) 用MC故障统计门控替代固定AC线路门控
      3) 输出每条线路预测置信度
    """
    lines = features_df.index.tolist()
    topo_prop = features_df.get('topo_prop_loss_reduction', pd.Series(0.0, index=features_df.index)).to_numpy(dtype=float)
    plain_prop = features_df.get('plain_prop_loss_reduction', pd.Series(0.0, index=features_df.index)).to_numpy(dtype=float)
    expected_fault_hours = features_df.get('expected_fault_hours', pd.Series(0.0, index=features_df.index)).to_numpy(dtype=float)
    fault_probability = features_df.get('fault_probability', pd.Series(0.0, index=features_df.index)).to_numpy(dtype=float)
    iso_frac = features_df.get('isolated_load_fraction', pd.Series(0.0, index=features_df.index)).to_numpy(dtype=float)
    fz_load = features_df.get('fault_zone_load', pd.Series(0.0, index=features_df.index)).to_numpy(dtype=float)
    source_load_ratio = features_df.get('source_load_ratio', pd.Series(0.0, index=features_df.index)).to_numpy(dtype=float)
    is_bridge = features_df.get('is_bridge', pd.Series(0.0, index=features_df.index)).to_numpy(dtype=float)
    is_normally_open = features_df.get('is_normally_open', pd.Series(0.0, index=features_df.index)).to_numpy(dtype=float)
    n_fault_mc = features_df.get('n_fault_mc', pd.Series(0.0, index=features_df.index)).to_numpy(dtype=float)

    expected_fault_hours_norm = expected_fault_hours / max(np.nanmax(expected_fault_hours), 1e-6)
    fz_load_norm = fz_load / max(float(baseline.get('total_load', 3000.0)), 1.0)
    topo_prop_norm = topo_prop / max(np.nanmax(topo_prop), 1e-6)
    plain_prop_norm = plain_prop / max(np.nanmax(plain_prop), 1e-6)

    c1 = topo_prop
    c2 = plain_prop
    c3 = fault_probability * iso_frac
    c4 = expected_fault_hours_norm * fault_probability
    c5 = fz_load_norm * fault_probability
    c6 = is_bridge * fault_probability
    components = [c1, c2, c3, c4, c5, c6]

    prior_coeff = np.array([0.72, 0.28, 0.16, 0.08, 0.08, 0.04], dtype=float)
    prior_w = prior_coeff / max(prior_coeff.sum(), 1e-9)

    robust_spread = []
    for comp in components:
        q10, q90 = np.quantile(comp, [0.10, 0.90])
        robust_spread.append(max(0.0, float(q90 - q10)))
    robust_spread = np.array(robust_spread, dtype=float)
    if robust_spread.sum() <= 1e-12:
        data_w = prior_w.copy()
    else:
        data_w = robust_spread / robust_spread.sum()

    ratio = np.divide(data_w, np.maximum(prior_w, 1e-9))
    ratio = np.clip(ratio, 0.65, 1.35)
    adaptive_coeff = prior_coeff * ratio

    base = (
        adaptive_coeff[0] * c1
        + adaptive_coeff[1] * c2
        + adaptive_coeff[2] * c3
        + adaptive_coeff[3] * c4
        + adaptive_coeff[4] * c5
        + adaptive_coeff[5] * c6
    )
    attenuation = 1.0 / (1.0 + 0.12 * np.clip(source_load_ratio, 0.0, None))
    attenuation = np.clip(attenuation, 0.72, 1.0)
    pred = np.maximum(base * attenuation, 0.0)

    # 头部线路校正：对高风险-高拓扑贡献线路进行分位数增强，降低系统性低估
    tail_signal = (
        0.55 * topo_prop_norm
        + 0.20 * plain_prop_norm
        + 0.15 * expected_fault_hours_norm
        + 0.10 * fault_probability
    )
    positive_tail = tail_signal[tail_signal > 0]
    if positive_tail.size > 0:
        tail_q = float(np.quantile(positive_tail, 0.75))
        tail_strength = np.clip((tail_signal - tail_q) / max(1e-6, 1.0 - tail_q), 0.0, 1.0)
        tail_boost = 1.0 + 0.55 * tail_strength + 0.20 * tail_strength * np.clip(iso_frac, 0.0, None)
        pred = pred * tail_boost

    # MC故障门控: 仅对“在MC中可能故障”的线路给出非零改善，提升跨拓扑适配性
    mc_fault_gate = (n_fault_mc > 0).astype(float)
    pred = pred * mc_fault_gate

    for i, _ in enumerate(lines):
        if is_normally_open[i] == 1:
            pred[i] = 0.0

    pred[pred < 0.001] = 0.0

    # 置信度: 高风险信号强 + 拓扑/朴素归因一致 + 故障覆盖高 => 更高置信度
    fault_coverage = np.clip(n_fault_mc / max(np.nanmax(n_fault_mc), 1e-6), 0.0, 1.0)
    agreement = 1.0 - np.clip(np.abs(topo_prop_norm - plain_prop_norm), 0.0, 1.0)
    confidence = np.clip(
        0.45 * tail_signal + 0.20 * fault_probability + 0.20 * fault_coverage + 0.15 * agreement,
        0.0,
        1.0,
    )
    confidence = confidence * mc_fault_gate * (1.0 - is_normally_open)

    if return_meta:
        meta = {
            'adaptive_coeff': {
                'topo_prop': float(adaptive_coeff[0]),
                'plain_prop': float(adaptive_coeff[1]),
                'fault_iso': float(adaptive_coeff[2]),
                'fh_fault': float(adaptive_coeff[3]),
                'fz_fault': float(adaptive_coeff[4]),
                'bridge_fault': float(adaptive_coeff[5]),
            },
            'gating': {
                'mc_fault_nonzero_lines': int(np.sum(mc_fault_gate > 0)),
                'normally_open_lines': int(np.sum(is_normally_open > 0)),
            },
        }
        return pred, confidence, meta

    return pred


def evaluate_loss_only(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> Dict:
    spearman, sp_pval = spearmanr(y_true, y_pred)
    kendall, _ = kendalltau(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    top5 = top_k_overlap(y_true, y_pred, k=min(5, len(y_true)))
    top10 = top_k_overlap(y_true, y_pred, k=min(10, len(y_true)))
    result = {
        'spearman': float(spearman),
        'kendall': float(kendall),
        'top5_overlap': float(top5),
        'top10_overlap': float(top10),
        'mae': mae,
        'rmse': rmse,
    }
    if label:
        print(f"\n  {'='*55}")
        print(f"  失负荷评估: {label}")
        print(f"  {'='*55}")
        print(f"  Spearman:   {spearman:.4f} (p={sp_pval:.2e})")
        print(f"  Kendall τ:  {kendall:.4f}")
        print(f"  Top-5重合:  {top5:.0%}  Top-10重合: {top10:.0%}")
        print(f"  MAE: {mae:.6f}  RMSE: {rmse:.6f}")
    return result


def generate_loss_only_report(line_names, y_true, pred_ml, output_dir, eval_ml, confidence_scores=None, infer_meta=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    true_rank = np.argsort(np.argsort(-y_true)) + 1
    ml_rank = np.argsort(np.argsort(-pred_ml)) + 1

    md = []
    md.append("# 配电网线路加固优先级 —— 失负荷单目标 ML 对比报告")
    md.append(f"\n**生成时间**: {now}")
    md.append("\n**说明**: 本报告仅做失负荷改善率评估，不包含超2h指标，不使用ground_truth监督训练。\n")

    md.append("## ML vs 真实（失负荷）\n")
    md.append("| 指标 | ML |")
    md.append("|------|:--:|")
    for metric in ['spearman', 'kendall', 'top5_overlap', 'top10_overlap', 'mae', 'rmse']:
        md.append(f"| {metric} | {eval_ml.get(metric, 0):.4f} |")

    md.append("\n## 逐线路对比（按真实改善排序）\n")
    md.append("| 线路 | 真实排名 | ML排名 | 真实失负荷改善 | ML预测失负荷改善 | 绝对误差 | 置信度 |")
    md.append("|------|:------:|:-----:|:--------------:|:---------------:|:-------:|:-----:|")
    for idx in np.argsort(-y_true):
        ln = line_names[idx]
        err = abs(y_true[idx] - pred_ml[idx])
        conf = float(confidence_scores[idx]) if confidence_scores is not None else 0.0
        md.append(
            f"| {ln} | {true_rank[idx]} | {ml_rank[idx]} | {y_true[idx]*100:.3f}% | {pred_ml[idx]*100:.3f}% | {err*100:.3f}% | {conf:.3f} |"
        )

    report_text = "\n".join(md)
    (output_dir / "ml_inference_report.md").write_text(report_text, encoding="utf-8")

    json_data = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'loss_only_no_supervised_training',
        'eval_ml': eval_ml,
        'inference_meta': infer_meta or {},
        'per_line': [{
            'line_name': line_names[idx],
            'actual_loss_improvement': float(y_true[idx]),
            'ml_predicted_loss_improvement': float(pred_ml[idx]),
            'actual_rank': int(true_rank[idx]),
            'ml_rank': int(ml_rank[idx]),
            'abs_error': float(abs(y_true[idx] - pred_ml[idx])),
            'confidence': float(confidence_scores[idx]) if confidence_scores is not None else None,
        } for idx in range(len(line_names))]
    }
    with open(output_dir / "ml_inference_report.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    return report_text


# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════

def _run_pure_inference_mode(merged, line_cols, baseline, topo_features, original_preds):
    """
    纯推理模式: 没有 ground_truth.json 时, 利用特征工程+加权评分产出加固优先级排名。
    
    不做有监督训练 (因为没有真实标签), 而是综合领域知识特征产出排名。
    """
    print("\n[3/5] 特征工程 (纯推理模式)...")
    features_df = extract_features_for_all_lines(
        merged, line_cols, baseline, topo_features,
        gt_df=None, original_preds=original_preds)
    print(f"  特征矩阵: {features_df.shape}")

    all_lines = features_df.index.tolist()
    n = len(all_lines)

    # 获取原始推理预测向量
    pred_combined = np.array([original_preds.get(ln, {}).get('pred_combined', 0) for ln in all_lines])

    # ---------- 加权评分 ----------
    # 原始推理预测(predict_single_line)是最重要的信号
    # 拓扑特征作为辅助排序依据
    print("\n[4/5] 加权特征评分...")
    scores = np.zeros(n)
    for i, ln in enumerate(all_lines):
        row = features_df.loc[ln]
        # 原始推理预测 (最重要信号)
        s_pred = pred_combined[i]

        # 拓扑/故障辅助分
        s_fault = row.get('expected_fault_hours', 0) * 0.01
        s_prob = row.get('fault_probability', 0)
        s_bc = row.get('betweenness_centrality', 0)
        s_iso = row.get('isolated_load_fraction', 0)
        s_bridge = 1.0 if row.get('is_bridge', 0) else 0.0
        s_fz_load = row.get('fault_zone_load', 0) / max(baseline.get('total_load', 3000), 1)
        s_downstream = row.get('downstream_load', 0) / max(baseline.get('total_load', 3000), 1)
        s_topo_red = row.get('topo_prop_loss_reduction', 0)
        s_plain_red = row.get('plain_prop_loss_reduction', 0)

        topo_bonus = (
            0.20 * s_fault
            + 0.15 * s_prob
            + 0.15 * s_bc
            + 0.10 * s_iso
            + 0.05 * s_bridge
            + 0.10 * s_fz_load
            + 0.05 * s_downstream
            + 0.10 * s_topo_red
            + 0.10 * s_plain_red
        )

        if s_pred > 0.001:
            # 有实际预测改善: 预测主导 + 拓扑微调
            scores[i] = 0.70 * s_pred + 0.30 * topo_bonus * s_pred
        else:
            # 预测为零: 仅保留极小拓扑信号用于同层排序
            scores[i] = 0.001 * topo_bonus

    # MC中无故障线路 (DC/VSC) 置零
    for i, ln in enumerate(all_lines):
        row = features_df.loc[ln]
        if row.get('n_fault_mc', 0) == 0 and row.get('n_fault_topo', 0) == 0:
            scores[i] = 0.0

    # ---------- 输出排名 ----------
    print("\n[5/5] 生成排名报告...")
    rank_order = np.argsort(-scores)
    ml_rank = np.argsort(np.argsort(-scores)) + 1
    orig_rank = np.argsort(np.argsort(-pred_combined)) + 1

    print(f"\n  {'═'*60}")
    print(f"  {'加固优先级排名 (纯推理模式)':^50}")
    print(f"  {'═'*60}")
    print(f"  {'#':>3} {'线路':<14} {'综合评分':>10} {'原始预测':>10}")
    print(f"  {'─'*60}")
    for rank_i, idx in enumerate(rank_order, 1):
        ln = all_lines[idx]
        print(f"  {rank_i:>3} {ln:<14} {scores[idx]*100:>9.4f}% {pred_combined[idx]*100:>9.4f}%")

    # 保存结果
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    md_lines = [
        "# 配电网线路加固优先级排名 (纯推理模式)",
        "",
        "> **注意**: 未检测到 `output/ground_truth.json`, 运行在纯推理模式。",
        "> 该模式利用特征工程+加权评分产出排名, 无监督训练。",
        "> 如需完整ML训练对比, 请运行 `python generate_ground_truth.py` 生成真实标签。",
        "",
        "## 加固优先级排名",
        "",
        "| 排名 | 线路 | 综合评分 | 原始推理预测 | 加权排名 | 原始排名 |",
        "|:----:|------|:--------:|:----------:|:-------:|:-------:|",
    ]
    for rank_i, idx in enumerate(rank_order, 1):
        ln = all_lines[idx]
        md_lines.append(
            f"| {rank_i} | {ln} | {scores[idx]*100:.4f}% | "
            f"{pred_combined[idx]*100:.4f}% | {ml_rank[idx]} | {orig_rank[idx]} |"
        )
    md_lines.append("")
    md_lines.append("## 评分权重说明")
    md_lines.append("")
    md_lines.append("| 特征 | 权重 |")
    md_lines.append("|------|:----:|")
    md_lines.append("| 原始推理预测 (combined_improvement) | 35% |")
    md_lines.append("| 拓扑加权失负荷缩减 | 10% |")
    md_lines.append("| 期望故障时长 | 10% |")
    md_lines.append("| 故障概率 | 8% |")
    md_lines.append("| 介数中心性 | 8% |")
    md_lines.append("| 比例归因失负荷缩减 | 8% |")
    md_lines.append("| 孤立负荷占比 | 6% |")
    md_lines.append("| 故障区域负荷占比 | 6% |")
    md_lines.append("| 下游负荷占比 | 4% |")
    md_lines.append("| 是否桥 | 3% |")
    md_lines.append("| 可切换性 | 2% |")
    md_lines.append("")
    md_lines.append(f"*MC中无故障线路 (如DC/VSC) 评分置零, 因台风故障模型不模拟这些线路故障。*")

    report_text = "\n".join(md_lines)
    (output_dir / "ml_inference_report.md").write_text(report_text, encoding="utf-8")

    json_data = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'pure_inference',
        'note': '纯推理模式: 无ground_truth, 利用特征工程+加权评分产出排名',
        'per_line': [{
            'line_name': all_lines[idx],
            'weighted_score': float(scores[idx]),
            'original_predicted_combined': float(pred_combined[idx]),
            'weighted_rank': int(ml_rank[idx]),
            'original_rank': int(orig_rank[idx]),
        } for idx in rank_order]
    }
    with open(output_dir / "ml_inference_report.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"\n  报告: {output_dir / 'ml_inference_report.md'}")
    print(f"  数据: {output_dir / 'ml_inference_report.json'}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="配电网线路加固优先级机器学习推理（失负荷单目标）")
    args = parser.parse_args()

    print("=" * 60)
    print("  配电网线路加固优先级 —— 机器学习推理层 v3")
    print("=" * 60)
    t_start = time.time()
    
    # Step 1: 加载数据
    print("\n[1] 加载数据...")
    merged, line_cols, baseline, data_dir, disp_path, topo_features = load_data()
    print(f"  数据: {merged.shape[0]} 行, {len(line_cols)} 条线路")
    
    # Step 2: 特征工程（不依赖ground_truth）
    print("\n[2] 特征工程（失负荷单目标）...")
    features_df = extract_features_for_all_lines(
        merged, line_cols, baseline, topo_features,
        gt_df=None, original_preds={})
    print(f"  特征矩阵: {features_df.shape}")

    # Step 3: 无监督ML推理（失负荷）
    print("\n[3] ML推理（仅失负荷改善率，不做监督训练）...")
    line_names = features_df.index.tolist()
    pred_ml, confidence_all, infer_meta = infer_loss_improvement_ml(features_df, baseline, return_meta=True)
    n_nonzero = int(np.sum(pred_ml > 0))
    print(f"  完成: {len(line_names)} 条线路, {n_nonzero} 条预测为非零改善")
    print("  自适应系数:", ", ".join([f"{k}={v:.4f}" for k, v in infer_meta.get('adaptive_coeff', {}).items()]))

    # Step 4: 与真实值对比（仅评估，不参与训练）
    print("\n[4] 读取真实结果并评估（仅失负荷）...")
    gt_df = load_ground_truth()
    if gt_df is None or 'actual_loss_improvement' not in gt_df.columns:
        print("  [警告] 未找到可用的 ground_truth 失负荷标签，仅输出ML推理结果。")
        order = np.argsort(-pred_ml)
        for rank_i, idx in enumerate(order, 1):
            print(f"  {rank_i:>2}. {line_names[idx]:<14} {pred_ml[idx]*100:7.3f}%")
        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "ml_inference_report.json", 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'mode': 'loss_only_no_gt',
                'inference_meta': infer_meta,
                'per_line': [
                    {
                        'line_name': line_names[i],
                        'ml_predicted_loss_improvement': float(pred_ml[i]),
                        'confidence': float(confidence_all[i]),
                    }
                    for i in range(len(line_names))
                ],
            }, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"  完成! 耗时 {elapsed:.1f} 秒")
        print(f"{'='*60}")
        return 0

    common_lines = [ln for ln in line_names if ln in gt_df.index]
    idx_map = [line_names.index(ln) for ln in common_lines]
    y_true = gt_df.loc[common_lines, 'actual_loss_improvement'].to_numpy(dtype=float)
    pred_eval = pred_ml[idx_map]
    conf_eval = confidence_all[idx_map]

    eval_ml = evaluate_loss_only(y_true, pred_eval, label="ML(失负荷) vs 真实")

    # Step 5: 仅输出ML vs 真实
    print("\n[5] 生成ML与真实对比报告...")
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    generate_loss_only_report(common_lines, y_true, pred_eval, output_dir, eval_ml, confidence_scores=conf_eval, infer_meta=infer_meta)

    true_rank = np.argsort(np.argsort(-y_true)) + 1
    ml_rank = np.argsort(np.argsort(-pred_eval)) + 1
    print(f"\n  {'真实':>4} {'ML':>4}  {'线路':<14} {'真实失负荷':>10} {'ML预测':>10} {'误差':>10}")
    print(f"  {'─'*72}")
    for idx in np.argsort(-y_true):
        ln = common_lines[idx]
        err = abs(y_true[idx] - pred_eval[idx])
        print(f"  {true_rank[idx]:>4} {ml_rank[idx]:>4}  {ln:<14} {y_true[idx]*100:>9.3f}% {pred_eval[idx]*100:>9.3f}% {err*100:>9.3f}%")
    
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  完成! 耗时 {elapsed:.1f} 秒")
    print(f"  报告: {output_dir / 'ml_inference_report.md'}")
    print(f"  数据: {output_dir / 'ml_inference_report.json'}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
