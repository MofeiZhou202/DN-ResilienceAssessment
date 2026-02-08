"""特征选择实验: 系统测试不同特征子集对ML预测准确度的影响
================================================================
针对35条线路的小样本问题，通过LOO-CV严格评估不同特征组合的效果。

使用: python test_feature_selection.py
"""

from __future__ import annotations
import json, sys, time, warnings, itertools
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from validate_inference import load_data, predict_single_line, NumpyEncoder
from run_ml_inference import (
    extract_features_for_all_lines,
    load_ground_truth,
    compute_original_predictions,
    ndcg_score,
    top_k_overlap,
)


# ═══════════════════════════════════════════════
#  评估函数
# ═══════════════════════════════════════════════

def quick_eval(y_true, y_pred):
    """快速评估，返回关键指标字典"""
    sp, sp_p = spearmanr(y_true, y_pred)
    kt, _ = kendalltau(y_true, y_pred)
    ndcg5 = ndcg_score(y_true, y_pred, k=5)
    ndcg10 = ndcg_score(y_true, y_pred, k=10)
    t3 = top_k_overlap(y_true, y_pred, k=3)
    t5 = top_k_overlap(y_true, y_pred, k=5)
    t10 = top_k_overlap(y_true, y_pred, k=10)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {
        'spearman': sp, 'kendall': kt,
        'ndcg@5': ndcg5, 'ndcg@10': ndcg10,
        'top3': t3, 'top5': t5, 'top10': t10,
        'mae': mae, 'rmse': rmse,
    }


# ═══════════════════════════════════════════════
#  LOO-CV 测试框架
# ═══════════════════════════════════════════════

def loo_ridge_test(X, y, feature_idx, line_names, X_df, alpha=10.0):
    """用指定特征子集做LOO-CV Ridge回归，返回评估结果"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut

    n = len(y)
    X_sel = X[:, feature_idx]
    loo = LeaveOneOut()
    predictions = np.zeros(n)

    for train_idx, test_idx in loo.split(X_sel):
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_sel[train_idx])
        X_test = scaler.transform(X_sel[test_idx])
        reg = Ridge(alpha=alpha)
        reg.fit(X_train, y[train_idx])
        predictions[test_idx] = reg.predict(X_test)

    predictions = np.maximum(predictions, 0)
    for i, ln in enumerate(line_names):
        if X_df.loc[ln, 'is_normally_open'] == 1:
            predictions[i] = 0
        if not ln.startswith('AC_Line_'):
            predictions[i] = 0

    return predictions, quick_eval(y, predictions)


def loo_two_stage_test(X, y, feature_idx_cls, feature_idx_reg, line_names, X_df,
                       alpha_reg=5.0, C_cls=0.5):
    """两阶段模型: 分类+回归"""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut

    n = len(y)
    y_binary = (y > 0).astype(int)
    loo = LeaveOneOut()
    predictions = np.zeros(n)

    for train_idx, test_idx in loo.split(X):
        # Stage 1: classification
        scaler_cls = RobustScaler()
        X_cls_train = scaler_cls.fit_transform(X[train_idx][:, feature_idx_cls])
        X_cls_test = scaler_cls.transform(X[test_idx][:, feature_idx_cls])
        clf = LogisticRegression(C=C_cls, penalty='l2', max_iter=500, random_state=42)
        clf.fit(X_cls_train, y_binary[train_idx])
        prob_nonzero = clf.predict_proba(X_cls_test)[:, 1]

        # Stage 2: regression on positives
        pos_mask = y_binary[train_idx] == 1
        if pos_mask.sum() >= 3:
            scaler_reg = RobustScaler()
            X_reg_train = scaler_reg.fit_transform(X[train_idx[pos_mask]][:, feature_idx_reg])
            X_reg_test = scaler_reg.transform(X[test_idx][:, feature_idx_reg])
            reg = Ridge(alpha=alpha_reg)
            reg.fit(X_reg_train, y[train_idx[pos_mask]])
            magnitude = reg.predict(X_reg_test)
        else:
            magnitude = np.array([0.0])

        predictions[test_idx] = prob_nonzero * np.maximum(magnitude, 0)

    predictions = np.maximum(predictions, 0)
    for i, ln in enumerate(line_names):
        if X_df.loc[ln, 'is_normally_open'] == 1:
            predictions[i] = 0
        if not ln.startswith('AC_Line_'):
            predictions[i] = 0

    return predictions, quick_eval(y, predictions)


def loo_elasticnet_test(X, y, feature_idx, line_names, X_df, alpha=0.01, l1_ratio=0.5):
    """ElasticNet LOO-CV"""
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut

    n = len(y)
    X_sel = X[:, feature_idx]
    loo = LeaveOneOut()
    predictions = np.zeros(n)

    for train_idx, test_idx in loo.split(X_sel):
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_sel[train_idx])
        X_test = scaler.transform(X_sel[test_idx])
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        reg.fit(X_train, y[train_idx])
        predictions[test_idx] = reg.predict(X_test)

    predictions = np.maximum(predictions, 0)
    for i, ln in enumerate(line_names):
        if X_df.loc[ln, 'is_normally_open'] == 1:
            predictions[i] = 0
        if not ln.startswith('AC_Line_'):
            predictions[i] = 0

    return predictions, quick_eval(y, predictions)


# ═══════════════════════════════════════════════
#  特征子集定义
# ═══════════════════════════════════════════════

def define_feature_subsets(feature_names: List[str]) -> Dict[str, List[str]]:
    """定义多组特征子集进行对比测试"""
    
    def safe_list(names):
        return [f for f in names if f in feature_names]
    
    subsets = {}
    
    # ── 0. 全部特征 (baseline) ──
    subsets['S0_全部特征(40维)'] = list(feature_names)
    
    # ── 1. AI建议的"黄金9维" ──
    subsets['S1_黄金9维'] = safe_list([
        'orig_pred_combined',
        'expected_fault_hours', 'fault_probability',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'line_type_ac', 'is_normally_open',
        'topo_prop_loss_reduction',
    ])
    
    # ── 2. 黄金9维 + 少量补充 (12维) ──
    subsets['S2_黄金扩展12维'] = safe_list([
        'orig_pred_combined', 'orig_pred_loss',
        'expected_fault_hours', 'fault_probability',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'line_type_ac', 'is_normally_open',
        'topo_prop_loss_reduction',
        'fault_zone_load', 'n_affected_scenarios',
    ])
    
    # ── 3. 原始代码的 top15 ──
    subsets['S3_原top15'] = safe_list([
        'orig_pred_combined', 'orig_pred_loss',
        'topo_prop_loss_reduction', 'plain_prop_loss_reduction',
        'expected_fault_hours', 'fault_probability',
        'betweenness_centrality', 'isolated_load_fraction',
        'fault_zone_load', 'downstream_load',
        'n_affected_scenarios', 'avg_fault_scenario_loss',
        'is_normally_open', 'is_critical', 'n_fault_mc',
    ])
    
    # ── 4. 原始代码的 minimal ──
    subsets['S4_原minimal6维'] = safe_list([
        'orig_pred_combined', 'topo_prop_loss_reduction',
        'expected_fault_hours', 'fault_probability',
        'isolated_load_fraction', 'is_normally_open',
    ])
    
    # ── 5. 纯原始推理(1维, lower bound) ──
    subsets['S5_纯原始推理1维'] = safe_list([
        'orig_pred_combined',
    ])
    
    # ── 6. 原始推理3维 ──
    subsets['S6_原始推理3维'] = safe_list([
        'orig_pred_combined', 'orig_pred_loss', 'orig_pred_over2h',
    ])
    
    # ── 7. 去掉交互特征 + 去掉噪音特征 ──
    subsets['S7_去噪去交互'] = safe_list([
        'orig_pred_combined', 'orig_pred_loss', 'orig_pred_over2h',
        'expected_fault_hours', 'fault_probability', 'n_affected_scenarios',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'line_type_ac', 'is_normally_open', 'is_critical',
        'fault_zone_load',
        'topo_prop_loss_reduction', 'plain_prop_loss_reduction',
        'avg_fault_scenario_loss', 'expected_fault_loss',
    ])
    
    # ── 8. 只保留"会不会坏"+"坏了多严重" ──
    subsets['S8_风险×后果'] = safe_list([
        'expected_fault_hours', 'fault_probability', 'n_affected_scenarios',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'topo_prop_loss_reduction',
        'is_normally_open', 'line_type_ac',
    ])
    
    # ── 9. 黄金9维 + 原始推理3维 (11维) ──
    subsets['S9_黄金+原始3维'] = safe_list([
        'orig_pred_combined', 'orig_pred_loss', 'orig_pred_over2h',
        'expected_fault_hours', 'fault_probability',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'line_type_ac', 'is_normally_open',
        'topo_prop_loss_reduction',
    ])
    
    # ── 10. 精简到极致(5维): 推理+风险+后果 ──
    subsets['S10_极简5维'] = safe_list([
        'orig_pred_combined',
        'expected_fault_hours', 'fault_probability',
        'isolated_load_fraction',
        'topo_prop_loss_reduction',
    ])
    
    # ── 11. 不含原始推理的纯特征 ──
    subsets['S11_纯特征无推理'] = safe_list([
        'expected_fault_hours', 'fault_probability', 'n_affected_scenarios',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'line_type_ac', 'is_normally_open', 'is_critical',
        'fault_zone_load', 'downstream_load',
        'topo_prop_loss_reduction', 'plain_prop_loss_reduction',
    ])
    
    # ── 12. 黄金+场景损失 ──
    subsets['S12_黄金+场景损失'] = safe_list([
        'orig_pred_combined',
        'expected_fault_hours', 'fault_probability',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'line_type_ac', 'is_normally_open',
        'topo_prop_loss_reduction',
        'avg_fault_scenario_loss', 'expected_fault_loss',
    ])
    
    # ── 13. 只保留故障+拓扑 (无推理, 无场景) ──
    subsets['S13_故障+拓扑'] = safe_list([
        'expected_fault_hours', 'fault_probability',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'is_normally_open', 'line_type_ac',
        'fault_zone_load',
    ])
    
    # ── 14. 原始推理 + topo_prop (2维核心) ──
    subsets['S14_推理+topo_prop'] = safe_list([
        'orig_pred_combined', 'topo_prop_loss_reduction',
    ])
    
    # ── 15. 中等精简: 去除共线性后的15维 ──
    subsets['S15_去共线性15维'] = safe_list([
        'orig_pred_combined', 'orig_pred_loss',
        'expected_fault_hours', 'fault_probability',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'line_type_ac', 'is_normally_open', 'is_critical',
        'fault_zone_load',
        'topo_prop_loss_reduction',
        'min_dist_to_source',
        'n_exclusive_fault_scenarios',
        'avg_fault_scenario_loss',
    ])
    
    # ── 16. 黄金9维 + downstream_load ──
    subsets['S16_黄金+下游负荷'] = safe_list([
        'orig_pred_combined',
        'expected_fault_hours', 'fault_probability',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'line_type_ac', 'is_normally_open',
        'topo_prop_loss_reduction',
        'downstream_load',
    ])
    
    # ── 17. 推理+损失归因(精准) ──
    subsets['S17_推理+损失归因'] = safe_list([
        'orig_pred_combined', 'orig_pred_loss', 'orig_pred_over2h',
        'topo_prop_loss_reduction', 'plain_prop_loss_reduction',
        'is_normally_open', 'line_type_ac',
    ])
    
    # ── 18. 黄金+独占场景 ──
    subsets['S18_黄金+独占场景'] = safe_list([
        'orig_pred_combined',
        'expected_fault_hours', 'fault_probability',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'line_type_ac', 'is_normally_open',
        'topo_prop_loss_reduction',
        'n_exclusive_fault_scenarios',
    ])
    
    # ── 19. 推理+故障概率+负荷(4维) ──
    subsets['S19_最精简4维'] = safe_list([
        'orig_pred_combined',
        'fault_probability',
        'isolated_load_fraction',
        'topo_prop_loss_reduction',
    ])
    
    # ── 20. 黄金9维 + fault_zone_load + downstream ──
    subsets['S20_黄金+负荷增强'] = safe_list([
        'orig_pred_combined',
        'expected_fault_hours', 'fault_probability',
        'betweenness_centrality', 'isolated_load_fraction', 'is_bridge',
        'line_type_ac', 'is_normally_open',
        'topo_prop_loss_reduction',
        'fault_zone_load', 'downstream_load',
    ])
    
    return subsets


# ═══════════════════════════════════════════════
#  主测试流程
# ═══════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  特征选择实验: 系统测试不同特征子集")
    print("=" * 70)
    t_start = time.time()

    # 1. 加载数据
    print("\n[1] 加载数据...")
    merged, line_cols, baseline, data_dir, disp_path, topo_features = load_data()
    
    print("\n[2] 计算原始推理预测...")
    original_preds = compute_original_predictions(merged, line_cols, baseline, topo_features)
    
    gt_df = load_ground_truth()
    if gt_df is None:
        print("  ✗ 未找到 ground_truth.json, 无法进行实验")
        return 1
    
    # 2. 特征工程
    print("\n[3] 提取全量特征...")
    features_df = extract_features_for_all_lines(
        merged, line_cols, baseline, topo_features,
        gt_df=gt_df, original_preds=original_preds)
    
    common_lines = features_df.index.intersection(gt_df.index)
    X_df = features_df.loc[common_lines].copy()
    y = gt_df.loc[common_lines, 'actual_combined_improvement'].values
    pred_original = np.array([original_preds.get(ln, {}).get('pred_combined', 0) for ln in common_lines])
    
    feature_cols = [c for c in X_df.columns if X_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    X = X_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    feature_names = feature_cols
    line_names = common_lines.tolist()
    n = len(line_names)
    
    print(f"  {n} 条线路, {len(feature_names)} 个特征")
    print(f"  特征列表: {feature_names}")
    
    # 3. 评估原始推理baseline
    print("\n[4] 原始推理 baseline...")
    eval_orig = quick_eval(y, pred_original)
    print(f"  原始推理 Spearman={eval_orig['spearman']:.4f}, NDCG@5={eval_orig['ndcg@5']:.4f}, Top5={eval_orig['top5']:.0%}")
    
    # 4. 定义并测试所有特征子集
    subsets = define_feature_subsets(feature_names)
    
    print(f"\n[5] 测试 {len(subsets)} 个特征子集 × 多种模型...")
    print("=" * 90)
    
    all_results = []
    
    for subset_name, feat_list in subsets.items():
        feat_idx = [feature_names.index(f) for f in feat_list]
        n_feat = len(feat_idx)
        
        # 测试多个alpha, 多种模型
        # Ridge with multiple alphas
        best_ridge_sp = -1
        best_ridge_alpha = None
        best_ridge_ev = None
        for alpha in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
            _, ev = loo_ridge_test(X, y, feat_idx, line_names, X_df, alpha=alpha)
            if ev['spearman'] > best_ridge_sp:
                best_ridge_sp = ev['spearman']
                best_ridge_alpha = alpha
                best_ridge_ev = ev
        
        # ElasticNet
        best_en_sp = -1
        best_en_ev = None
        best_en_params = None
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.1]:
            for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
                _, ev = loo_elasticnet_test(X, y, feat_idx, line_names, X_df, alpha=alpha, l1_ratio=l1)
                if ev['spearman'] > best_en_sp:
                    best_en_sp = ev['spearman']
                    best_en_ev = ev
                    best_en_params = (alpha, l1)
        
        # 取最好
        if best_ridge_sp >= best_en_sp:
            best_ev = best_ridge_ev
            best_model = f"Ridge(α={best_ridge_alpha})"
        else:
            best_ev = best_en_ev
            best_model = f"EN(α={best_en_params[0]},l1={best_en_params[1]})"
        
        all_results.append({
            'subset': subset_name,
            'n_features': n_feat,
            'best_model': best_model,
            'spearman': best_ev['spearman'],
            'kendall': best_ev['kendall'],
            'ndcg@5': best_ev['ndcg@5'],
            'ndcg@10': best_ev['ndcg@10'],
            'top3': best_ev['top3'],
            'top5': best_ev['top5'],
            'top10': best_ev['top10'],
            'mae': best_ev['mae'],
            'rmse': best_ev['rmse'],
            'features': feat_list,
        })
        
        sp_diff = best_ev['spearman'] - eval_orig['spearman']
        marker = " ★" if sp_diff > 0 else ""
        print(f"  {subset_name:<30} {n_feat:>2}维 {best_model:<22} "
              f"Sp={best_ev['spearman']:.4f}({sp_diff:+.4f}) "
              f"NDCG@5={best_ev['ndcg@5']:.4f} Top5={best_ev['top5']:.0%}{marker}")
    
    # 5. 对Top特征子集再测两阶段模型
    print("\n" + "=" * 90)
    print("  对Top子集额外测试两阶段模型")
    print("=" * 90)
    
    # 按spearman排序取top8
    sorted_results = sorted(all_results, key=lambda x: x['spearman'], reverse=True)
    top_subsets = sorted_results[:8]
    
    two_stage_results = []
    for res in top_subsets:
        feat_list = res['features']
        feat_idx = [feature_names.index(f) for f in feat_list]
        
        # 分分类和回归特征
        cls_feats_all = [f for f in feat_list if f in [
            'is_normally_open', 'line_type_ac', 'n_fault_mc', 'n_fault_topo',
            'orig_pred_combined', 'fault_probability',
            'topo_prop_loss_reduction', 'downstream_load', 'isolated_load_fraction',
            'expected_fault_hours', 'n_affected_scenarios', 'is_bridge',
        ]]
        reg_feats_all = feat_list  # 回归用全部特征
        
        if len(cls_feats_all) < 3:
            cls_feats_all = feat_list[:min(6, len(feat_list))]
        
        cls_idx = [feature_names.index(f) for f in cls_feats_all]
        reg_idx = feat_idx
        
        best_ts_sp = -1
        best_ts_ev = None
        best_ts_params = None
        for alpha in [1.0, 5.0, 10.0, 20.0, 50.0]:
            for C in [0.1, 0.5, 1.0, 2.0]:
                _, ev = loo_two_stage_test(X, y, cls_idx, reg_idx, line_names, X_df,
                                            alpha_reg=alpha, C_cls=C)
                if ev['spearman'] > best_ts_sp:
                    best_ts_sp = ev['spearman']
                    best_ts_ev = ev
                    best_ts_params = (alpha, C)
        
        sp_diff = best_ts_ev['spearman'] - eval_orig['spearman']
        marker = " ★" if sp_diff > 0 else ""
        print(f"  {res['subset']:<30} 2Stage(α={best_ts_params[0]},C={best_ts_params[1]}) "
              f"Sp={best_ts_ev['spearman']:.4f}({sp_diff:+.4f}) "
              f"NDCG@5={best_ts_ev['ndcg@5']:.4f} Top5={best_ts_ev['top5']:.0%}{marker}")
        
        two_stage_results.append({
            'subset': res['subset'],
            'best_2stage_sp': best_ts_ev['spearman'],
            'best_2stage_params': best_ts_params,
            'best_2stage_ev': best_ts_ev,
        })
    
    # 6. 汇总排行
    print("\n" + "=" * 90)
    print("  ★ 最终排行榜 (按Spearman排序)")
    print("=" * 90)
    
    # 合并所有结果
    final_results = []
    final_results.append({
        'name': '原始推理(baseline)',
        'spearman': eval_orig['spearman'],
        'kendall': eval_orig['kendall'],
        'ndcg@5': eval_orig['ndcg@5'],
        'top5': eval_orig['top5'],
        'n_feat': '-',
        'model': '规则推理',
    })
    
    for res in all_results:
        final_results.append({
            'name': res['subset'],
            'spearman': res['spearman'],
            'kendall': res['kendall'],
            'ndcg@5': res['ndcg@5'],
            'top5': res['top5'],
            'n_feat': res['n_features'],
            'model': res['best_model'],
        })
    
    for ts_res in two_stage_results:
        final_results.append({
            'name': f"{ts_res['subset']}+2Stage",
            'spearman': ts_res['best_2stage_sp'],
            'kendall': ts_res['best_2stage_ev']['kendall'],
            'ndcg@5': ts_res['best_2stage_ev']['ndcg@5'],
            'top5': ts_res['best_2stage_ev']['top5'],
            'n_feat': '-',
            'model': f"2Stage(α={ts_res['best_2stage_params'][0]},C={ts_res['best_2stage_params'][1]})",
        })
    
    final_sorted = sorted(final_results, key=lambda x: x['spearman'], reverse=True)
    
    print(f"\n  {'#':>3} {'名称':<40} {'特征':>4} {'模型':<24} {'Spearman':>8} {'Kendall':>8} {'NDCG@5':>7} {'Top5':>5}")
    print(f"  {'─'*110}")
    for rank, res in enumerate(final_sorted, 1):
        marker = " ◄" if res['name'] == '原始推理(baseline)' else ""
        print(f"  {rank:>3} {res['name']:<40} {str(res['n_feat']):>4} {res['model']:<24} "
              f"{res['spearman']:>8.4f} {res['kendall']:>8.4f} {res['ndcg@5']:>7.4f} {res['top5']:>5.0%}{marker}")
    
    # 7. 给出推荐
    print("\n" + "=" * 90)
    print("  ★ 推荐分析")
    print("=" * 90)
    
    # 找到最佳非baseline
    best = final_sorted[0]
    if best['name'] == '原始推理(baseline)':
        best = final_sorted[1] if len(final_sorted) > 1 else best
    
    best_subset_res = max(all_results, key=lambda x: x['spearman'])
    print(f"\n  最佳特征子集: {best_subset_res['subset']}")
    print(f"  特征维数: {best_subset_res['n_features']}")
    print(f"  Spearman: {best_subset_res['spearman']:.4f} (vs 原始 {eval_orig['spearman']:.4f}, Δ={best_subset_res['spearman']-eval_orig['spearman']:+.4f})")
    print(f"  最佳模型: {best_subset_res['best_model']}")
    print(f"  特征列表: {best_subset_res['features']}")
    
    elapsed = time.time() - t_start
    print(f"\n  实验耗时: {elapsed:.1f} 秒")
    
    # 保存结果
    output_path = PROJECT_ROOT / "output" / "feature_selection_results.json"
    save_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'baseline_spearman': eval_orig['spearman'],
        'results': [{k: v for k, v in r.items()} for r in all_results],
        'ranking': [{'rank': i+1, **{k: (v if not isinstance(v, np.floating) else float(v)) for k, v in r.items()}} for i, r in enumerate(final_sorted)],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  结果保存: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
