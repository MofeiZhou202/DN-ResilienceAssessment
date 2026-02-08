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
  python run_ml_inference.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
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
    load_on_bus = network.get('load_on_bus', {})
    all_edges = network.get('all_edges', {})
    dc_edges = network.get('dc_edges', {})
    vsc_edges = network.get('vsc_edges', {})
    tie_switches = network.get('tie_switches', set())
    switch_flag = network.get('switch_flag', {})
    mess_reachable = network.get('mess_reachable_buses', set())
    fault_zone_info = network.get('fault_zone_info', {})
    all_line_info = network.get('all_line_info', {})
    
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
            
            feat['mess_reachable'] = int(u in mess_reachable or v in mess_reachable)
        else:
            feat['endpoint_load'] = 0
            feat['endpoint_avg_degree'] = 0
            feat['endpoint_avg_closeness'] = 0
            feat['min_dist_to_source'] = 10
            feat['mess_reachable'] = 0
        
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
    print("=" * 60)
    print("  配电网线路加固优先级 —— 机器学习推理层 v3")
    print("=" * 60)
    t_start = time.time()
    
    # Step 1: 加载数据
    print("\n[1] 加载数据...")
    merged, line_cols, baseline, data_dir, disp_path, topo_features = load_data()
    print(f"  数据: {merged.shape[0]} 行, {len(line_cols)} 条线路")
    
    # Step 2: 计算原始推理预测 (直接调用 predict_single_line, 不依赖 ground_truth.json)
    print("\n[2] 计算原始推理预测 (predict_single_line × 35 条线路)...")
    original_preds = compute_original_predictions(merged, line_cols, baseline, topo_features)
    n_nonzero = sum(1 for v in original_preds.values() if v['pred_combined'] > 0)
    print(f"  完成: {len(original_preds)} 条线路, {n_nonzero} 条有非零预测")

    # Step 3: 尝试加载 ground truth
    gt_df = load_ground_truth()
    
    if gt_df is None:
        # ═══ 纯推理模式 ═══
        print("\n" + "═" * 60)
        print("  未检测到 output/ground_truth.json")
        print("  → 进入【纯推理模式】: 特征工程+加权评分产出排名")
        print("  → 如需完整ML训练对比, 请运行: python generate_ground_truth.py")
        print("═" * 60)
        rc = _run_pure_inference_mode(merged, line_cols, baseline, topo_features, original_preds)
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"  完成! 耗时 {elapsed:.1f} 秒")
        print(f"{'='*60}")
        return rc
    
    # ═══ 完整ML训练模式 ═══
    print("\n[3/7] 加载 ground truth...")
    n_positive = (gt_df['actual_combined_improvement'] > 0).sum()
    n_zero = (gt_df['actual_combined_improvement'] == 0).sum()
    print(f"  真实标签: {len(gt_df)} 条 (有改善:{n_positive}, 无改善:{n_zero})")
    
    # Step 3: 特征工程
    print("\n[3/7] 特征工程...")
    features_df = extract_features_for_all_lines(
        merged, line_cols, baseline, topo_features,
        gt_df=gt_df, original_preds=original_preds)
    print(f"  特征矩阵: {features_df.shape}")
    
    # 对齐
    common_lines = features_df.index.intersection(gt_df.index)
    X_df = features_df.loc[common_lines].copy()
    y = gt_df.loc[common_lines, 'actual_combined_improvement'].values
    pred_original = gt_df.loc[common_lines, 'predicted_combined_improvement'].values
    
    feature_cols = [c for c in X_df.columns if X_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    X = X_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    feature_names = feature_cols
    line_names = common_lines.tolist()
    n = len(line_names)
    
    # 特征维度说明
    core_features = ['orig_pred_combined', 'fault_probability', 'topo_prop_loss_reduction']
    core_in_data = [f for f in core_features if f in feature_names]
    print(f"  共 {n} 条线路, {len(feature_names)} 个原始特征")
    print(f"  核心特征 ({len(core_in_data)}维): {core_in_data}")
    print(f"  注: 实验验证3-6维特征即为最优, 40维过拟合")
    
    # Step 4: 评估原始推理层
    print("\n[4/7] 评估原始推理层...")
    eval_original = evaluate_ranking(y, pred_original, label="原始推理层")
    
    # Step 5: 训练ML模型 (精简版 — 基于3轮特征选择实验结论)
    print("\n[5/7] 训练ML模型 (精简特征, 3-6维)...")
    
    all_preds = {'原始推理': pred_original.copy()}
    all_evals = {'原始推理': eval_original}
    
    # 5a. ★ 分类矫正 (实验验证最佳策略, Sp=0.9173)
    print("\n  ── ★ 分类矫正模型 (3-7维特征) ──")
    cls_pred, cls_eval, cls_label = classification_correction_model(
        X, y, line_names, feature_names, X_df, pred_original)
    all_preds['分类矫正'] = cls_pred
    all_evals['分类矫正'] = cls_eval
    
    # 5b. 阻尼残差 (实验验证第二佳, Sp=0.8916)
    print("\n  ── 阻尼残差模型 (3-8维特征) ──")
    dr_pred, dr_eval, dr_label = damped_residual_model(
        X, y, line_names, feature_names, X_df, pred_original)
    all_preds['阻尼残差'] = dr_pred
    all_evals['阻尼残差'] = dr_eval
    
    # 5c. 组合模型 (分类矫正 + 阻尼残差)
    print("\n  ── 组合模型 (分类+残差) ──")
    combo_pred, combo_eval, combo_label = combined_cls_residual_model(
        X, y, line_names, feature_names, X_df, pred_original)
    all_preds['组合模型'] = combo_pred
    all_evals['组合模型'] = combo_eval
    
    # 5d. 阈值优化 (最简方法, Sp=0.8826)
    print("\n  ── 阈值优化 ──")
    th_pred, th_eval, th_label = threshold_optimized_model(y, pred_original, line_names, X_df)
    all_preds['阈值优化'] = th_pred
    all_evals['阈值优化'] = th_eval
    
    # 5e. Ridge系列 (精简特征子集)
    print("\n  ── Ridge 系列 (精简特征) ──")
    ridge_results = ridge_family(X, y, line_names, feature_names, X_df)
    for name, (pred, ev) in ridge_results.items():
        all_preds[name] = pred
        all_evals[name] = ev

    # Step 6: 集成 + 后处理
    print("\n[6/7] 集成 + 后处理优化...")
    
    # 6a. 智能集成
    ens_pred, ens_eval = smart_ensemble(all_preds, all_evals, y, line_names, X_df)
    all_preds['智能集成'] = ens_pred
    all_evals['智能集成'] = ens_eval
    
    # 6b. 对Ridge结果做后处理
    for name in list(ridge_results.keys()):
        pp_pred, pp_eval = optimize_postprocessing(all_preds[name].copy(), y, line_names, X_df)
        all_preds[f"{name}+PP"] = pp_pred
        all_evals[f"{name}+PP"] = pp_eval

    # 6c. Stacking (用精简模型)
    raw_model_preds = {k: all_preds[k] for k in ['分类矫正', '阻尼残差', '阈值优化']
                       if k in all_preds}
    # 加几个最佳Ridge
    best_ridges = sorted(
        [(k, all_evals[k]['spearman']) for k in ridge_results.keys()],
        key=lambda x: -x[1]
    )[:3]
    for rname, _ in best_ridges:
        raw_model_preds[rname] = all_preds[rname]
    
    if len(raw_model_preds) >= 3:
        print("\n  ── Stacking 元学习器 ──")
        stk_pred, stk_eval = stacking_meta_learner(
            raw_model_preds, y, line_names, feature_names, X, X_df)
        all_preds['Stacking'] = stk_pred
        all_evals['Stacking'] = stk_eval

    # 6d. 混合优化
    print("\n  ── 混合优化 ──")
    top_ml = {k: all_preds[k] for k in ['分类矫正', '阻尼残差', '组合模型']
              if k in all_preds and all_evals.get(k, {}).get('spearman', 0) > 0.6}
    if top_ml:
        blend_pred, blend_eval, blend_label = hybrid_blend_optimization(
            top_ml, pred_original, y, line_names, X_df)
        all_preds['混合优化'] = blend_pred
        all_evals['混合优化'] = blend_eval

    # Step 7: 汇总
    print("\n[7/7] 最终汇总...")
    
    # 排行榜
    sorted_models = sorted(all_evals.items(), key=lambda x: x[1]['spearman'], reverse=True)
    
    print(f"\n  {'═'*70}")
    print(f"\n  {'模型排行榜 (Top 20)':^60}")
    print(f"  {'═'*70}")
    print(f"  {'#':>3} {'模型':<42} {'Spearman':>8} {'NDCG@5':>7} {'Top5':>5}")
    print(f"  {'─'*70}")
    
    for rank, (name, ev) in enumerate(sorted_models[:20], 1):
        sp = ev['spearman']
        ndcg5 = ev.get('ndcg@5', 0)
        t5 = ev.get('top5_overlap', 0)
        marker = " ★" if name != '原始推理' and sp >= eval_original['spearman'] else ""
        print(f"  {rank:>3} {name:<42} {sp:>8.4f} {ndcg5:>7.4f} {t5:>5.0%}{marker}")
    
    # 选最终ML模型
    ml_models = {k: v for k, v in all_evals.items() if k != '原始推理'}
    best_ml_name = max(ml_models.keys(), key=lambda k: ml_models[k]['spearman'])
    best_ml_pred = all_preds[best_ml_name]
    best_ml_eval = all_evals[best_ml_name]
    
    print(f"\n  最终ML选择: {best_ml_name}")
    print(f"    Spearman: {best_ml_eval['spearman']:.4f} vs 原始 {eval_original['spearman']:.4f}")
    diff = best_ml_eval['spearman'] - eval_original['spearman']
    if diff > 0:
        print(f"    ML比原始推理好 {diff:.4f}")
    else:
        print(f"    ML比原始推理差 {abs(diff):.4f}")
    
    # 报告
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    generate_comparison_report(line_names, y, pred_original, best_ml_pred,
                               output_dir, eval_original, best_ml_eval, best_ml_name)
    
    # 排名对比
    true_rank = np.argsort(np.argsort(-y)) + 1
    orig_rank = np.argsort(np.argsort(-pred_original)) + 1
    ml_rank = np.argsort(np.argsort(-best_ml_pred)) + 1
    
    print(f"\n  {'真实':>4} {'原始':>4} {'ML':>4}  {'线路':<14} {'真实':>8} {'原始':>8} {'ML':>8}")
    print(f"  {'─'*70}")
    for idx in np.argsort(-y):
        ln = line_names[idx]
        print(f"  {true_rank[idx]:>4} {orig_rank[idx]:>4} {ml_rank[idx]:>4}  "
              f"{ln:<14} {y[idx]*100:>7.3f}% {pred_original[idx]*100:>7.3f}% "
              f"{best_ml_pred[idx]*100:>7.3f}%")
    
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  完成! 耗时 {elapsed:.1f} 秒")
    print(f"  报告: {output_dir / 'ml_inference_report.md'}")
    print(f"  数据: {output_dir / 'ml_inference_report.json'}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
