"""特征选择实验 第二轮: 更智能的建模策略
====================================================
Round 1 发现: 减少特征有帮助(40维→3维, Sp 0.77→0.80), 但都不及原始推理(0.86)。
原因: ML在35个样本上引入的方差 > 修正的偏差。

本轮策略:
  1. 残差学习: 学习 y - orig_pred 的误差，而不是直接学y
  2. 单调校准: 对orig_pred做保序回归(Isotonic Regression)
  3. 弹性融合: 动态调权 ML/原始推理比例
  4. 排序保护: 约束ML不要破坏原始推理中already-correct的排名
  5. 特征交互更精准: 只用物理有意义的交互特征
"""

from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
from typing import Dict, List
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


def quick_eval(y_true, y_pred, label=""):
    sp, sp_p = spearmanr(y_true, y_pred)
    kt, _ = kendalltau(y_true, y_pred)
    ndcg5 = ndcg_score(y_true, y_pred, k=5)
    ndcg10 = ndcg_score(y_true, y_pred, k=10)
    t3 = top_k_overlap(y_true, y_pred, k=3)
    t5 = top_k_overlap(y_true, y_pred, k=5)
    t10 = top_k_overlap(y_true, y_pred, k=10)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ev = {'spearman': sp, 'kendall': kt, 'ndcg@5': ndcg5, 'ndcg@10': ndcg10,
          'top3': t3, 'top5': t5, 'top10': t10, 'mae': mae, 'rmse': rmse}
    if label:
        print(f"    {label:<40} Sp={sp:.4f} NDCG@5={ndcg5:.4f} Top5={t5:.0%} Top10={t10:.0%}")
    return ev


def postprocess(pred, line_names, X_df):
    """统一后处理"""
    pred = np.maximum(pred, 0)
    for i, ln in enumerate(line_names):
        if X_df.loc[ln, 'is_normally_open'] == 1:
            pred[i] = 0
        if not ln.startswith('AC_Line_'):
            pred[i] = 0
    return pred


# ═══════════════════════════════════════════════
#  策略1: 残差学习 (学习 y - orig_pred)
# ═══════════════════════════════════════════════

def residual_learning(X, y, feature_names, line_names, X_df, orig_pred):
    """学习残差 = y - orig_pred, 然后 final = orig_pred + residual"""
    from sklearn.linear_model import Ridge, ElasticNet, Lasso
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    print("\n  ── 策略1: 残差学习 ──")
    n = len(y)
    residual = y - orig_pred  # 目标变为残差
    loo = LeaveOneOut()
    
    # 测试不同特征子集
    feature_sets = {
        '全部': list(range(len(feature_names))),
        '少量': [feature_names.index(f) for f in [
            'topo_prop_loss_reduction', 'plain_prop_loss_reduction',
            'expected_fault_hours', 'fault_probability',
            'betweenness_centrality', 'isolated_load_fraction',
            'is_normally_open', 'line_type_ac',
        ] if f in feature_names],
        '精简': [feature_names.index(f) for f in [
            'topo_prop_loss_reduction',
            'expected_fault_hours', 'fault_probability',
            'isolated_load_fraction', 'is_normally_open',
        ] if f in feature_names],
        '核心': [feature_names.index(f) for f in [
            'topo_prop_loss_reduction', 'fault_probability',
            'isolated_load_fraction',
        ] if f in feature_names],
    }
    
    results = {}
    for fs_name, feat_idx in feature_sets.items():
        for alpha in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]:
            predictions = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                reg = Ridge(alpha=alpha)
                reg.fit(X_train, residual[train_idx])
                pred_residual = reg.predict(X_test)
                predictions[test_idx] = orig_pred[test_idx] + pred_residual
            
            predictions = postprocess(predictions, line_names, X_df)
            ev = quick_eval(y, predictions)
            key = f"残差Ridge({fs_name},α={alpha})"
            results[key] = (predictions, ev)
    
        # ElasticNet on residuals
        for alpha in [0.001, 0.005, 0.01, 0.05]:
            for l1 in [0.3, 0.5, 0.7, 0.9]:
                predictions = np.zeros(n)
                for train_idx, test_idx in loo.split(X):
                    scaler = RobustScaler()
                    X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                    X_test = scaler.transform(X[test_idx][:, feat_idx])
                    reg = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000)
                    reg.fit(X_train, residual[train_idx])
                    pred_residual = reg.predict(X_test)
                    predictions[test_idx] = orig_pred[test_idx] + pred_residual
                
                predictions = postprocess(predictions, line_names, X_df)
                ev = quick_eval(y, predictions)
                key = f"残差EN({fs_name},α={alpha},l1={l1})"
                results[key] = (predictions, ev)
    
    # Lasso残差 (更强的特征选择)
    for fs_name, feat_idx in feature_sets.items():
        for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
            predictions = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                reg = Lasso(alpha=alpha, max_iter=5000)
                reg.fit(X_train, residual[train_idx])
                pred_residual = reg.predict(X_test)
                predictions[test_idx] = orig_pred[test_idx] + pred_residual
            
            predictions = postprocess(predictions, line_names, X_df)
            ev = quick_eval(y, predictions)
            key = f"残差Lasso({fs_name},α={alpha})"
            results[key] = (predictions, ev)
    
    # 打印top5
    sorted_results = sorted(results.items(), key=lambda x: x[1][1]['spearman'], reverse=True)
    for name, (_, ev) in sorted_results[:5]:
        quick_eval(y, _, label=name)
    
    return results


# ═══════════════════════════════════════════════
#  策略2: 保序回归校准 (Isotonic Regression)
# ═══════════════════════════════════════════════

def isotonic_calibration(y, orig_pred, line_names, X_df):
    """对原始推理预测做保序回归校准"""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import LeaveOneOut
    
    print("\n  ── 策略2: 保序回归校准 ──")
    n = len(y)
    loo = LeaveOneOut()
    
    results = {}
    
    # 纯保序回归
    predictions = np.zeros(n)
    for train_idx, test_idx in loo.split(orig_pred.reshape(-1, 1)):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(orig_pred[train_idx], y[train_idx])
        predictions[test_idx] = iso.predict(orig_pred[test_idx])
    
    predictions = postprocess(predictions, line_names, X_df)
    ev = quick_eval(y, predictions, label="保序回归(orig_pred)")
    results['保序回归'] = (predictions, ev)
    
    return results


# ═══════════════════════════════════════════════
#  策略3: 阻尼残差学习 (限制ML修正幅度)
# ═══════════════════════════════════════════════

def damped_residual(X, y, feature_names, line_names, X_df, orig_pred):
    """阻尼残差: final = orig + damping * ML_residual"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    print("\n  ── 策略3: 阻尼残差学习 ──")
    n = len(y)
    residual = y - orig_pred
    loo = LeaveOneOut()
    
    feature_sets = {
        '精简': [feature_names.index(f) for f in [
            'topo_prop_loss_reduction',
            'expected_fault_hours', 'fault_probability',
            'isolated_load_fraction', 'is_normally_open',
        ] if f in feature_names],
        '核心': [feature_names.index(f) for f in [
            'topo_prop_loss_reduction', 'fault_probability',
            'isolated_load_fraction',
        ] if f in feature_names],
        '少量': [feature_names.index(f) for f in [
            'topo_prop_loss_reduction', 'plain_prop_loss_reduction',
            'expected_fault_hours', 'fault_probability',
            'betweenness_centrality', 'isolated_load_fraction',
            'is_normally_open', 'line_type_ac',
        ] if f in feature_names],
    }
    
    results = {}
    for fs_name, feat_idx in feature_sets.items():
        for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
            # 先训练残差模型
            raw_residual_pred = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                reg = Ridge(alpha=alpha)
                reg.fit(X_train, residual[train_idx])
                raw_residual_pred[test_idx] = reg.predict(X_test)
            
            # 然后用不同damping系数
            for damping in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
                pred = orig_pred + damping * raw_residual_pred
                pred = postprocess(pred, line_names, X_df)
                ev = quick_eval(y, pred)
                key = f"阻尼残差({fs_name},α={alpha},d={damping})"
                results[key] = (pred, ev)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][1]['spearman'], reverse=True)
    for name, (_, ev) in sorted_results[:5]:
        quick_eval(y, _, label=name)
    
    return results


# ═══════════════════════════════════════════════
#  策略4: 零/非零分类器矫正 + 原始推理
# ═══════════════════════════════════════════════

def classification_correction(X, y, feature_names, line_names, X_df, orig_pred):
    """只用ML判断哪些线路改善=0，然后用原始推理的非零值做排序"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    print("\n  ── 策略4: 分类矫正 (只判零/非零) ──")
    n = len(y)
    y_binary = (y > 0).astype(int)
    loo = LeaveOneOut()
    
    feature_sets = {
        '少量': [feature_names.index(f) for f in [
            'is_normally_open', 'line_type_ac', 'fault_probability',
            'orig_pred_combined', 'topo_prop_loss_reduction',
            'isolated_load_fraction', 'expected_fault_hours',
        ] if f in feature_names],
        '极简': [feature_names.index(f) for f in [
            'is_normally_open', 'line_type_ac', 'fault_probability',
            'orig_pred_combined',
        ] if f in feature_names],
        '中等': [feature_names.index(f) for f in [
            'is_normally_open', 'line_type_ac', 'fault_probability',
            'orig_pred_combined', 'topo_prop_loss_reduction',
            'isolated_load_fraction', 'expected_fault_hours',
            'betweenness_centrality', 'is_bridge',
        ] if f in feature_names],
    }
    
    results = {}
    for fs_name, feat_idx in feature_sets.items():
        for C_val in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
            probs = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                clf = LogisticRegression(C=C_val, max_iter=500, random_state=42)
                clf.fit(X_train, y_binary[train_idx])
                probs[test_idx] = clf.predict_proba(X_test)[:, 1]
            
            # 用分类概率过滤原始推理
            for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
                pred = orig_pred.copy()
                for i in range(n):
                    if probs[i] < threshold:
                        pred[i] = 0
                pred = postprocess(pred, line_names, X_df)
                ev = quick_eval(y, pred)
                key = f"分类矫正({fs_name},C={C_val},t={threshold})"
                results[key] = (pred, ev)
            
            # 用概率加权原始推理
            pred = orig_pred * probs
            pred = postprocess(pred, line_names, X_df)
            ev = quick_eval(y, pred)
            key = f"概率加权({fs_name},C={C_val})"
            results[key] = (pred, ev)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][1]['spearman'], reverse=True)
    for name, (_, ev) in sorted_results[:5]:
        quick_eval(y, _, label=name)
    
    return results


# ═══════════════════════════════════════════════
#  策略5: 简单加权融合 (不训练, 只混合信号)
# ═══════════════════════════════════════════════

def weighted_signal_fusion(X, y, feature_names, line_names, X_df, orig_pred):
    """直接用 orig_pred * (1 + w * feature) 做微调，不需要训练"""
    
    print("\n  ── 策略5: 无训练信号融合 ──")
    n = len(y)
    
    results = {}
    
    # 获取关键特征
    topo_idx = feature_names.index('topo_prop_loss_reduction') if 'topo_prop_loss_reduction' in feature_names else None
    fp_idx = feature_names.index('fault_probability') if 'fault_probability' in feature_names else None
    efh_idx = feature_names.index('expected_fault_hours') if 'expected_fault_hours' in feature_names else None
    bc_idx = feature_names.index('betweenness_centrality') if 'betweenness_centrality' in feature_names else None
    iso_idx = feature_names.index('isolated_load_fraction') if 'isolated_load_fraction' in feature_names else None
    
    for w_topo in [0, 0.1, 0.2, 0.3, 0.5]:
        for w_fp in [0, 0.05, 0.1, 0.2]:
            for w_bc in [0, 0.05, 0.1]:
                boost = np.ones(n)
                if topo_idx is not None and w_topo > 0:
                    topo_vals = X[:, topo_idx]
                    topo_norm = topo_vals / (np.max(topo_vals) + 1e-10)
                    boost += w_topo * topo_norm
                if fp_idx is not None and w_fp > 0:
                    fp_vals = X[:, fp_idx]
                    fp_norm = fp_vals / (np.max(fp_vals) + 1e-10)
                    boost += w_fp * fp_norm
                if bc_idx is not None and w_bc > 0:
                    bc_vals = X[:, bc_idx]
                    bc_norm = bc_vals / (np.max(bc_vals) + 1e-10)
                    boost += w_bc * bc_norm
                
                pred = orig_pred * boost
                pred = postprocess(pred, line_names, X_df)
                ev = quick_eval(y, pred)
                key = f"信号融合(topo={w_topo},fp={w_fp},bc={w_bc})"
                results[key] = (pred, ev)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][1]['spearman'], reverse=True)
    for name, (_, ev) in sorted_results[:5]:
        quick_eval(y, _, label=name)
    
    return results


# ═══════════════════════════════════════════════
#  策略6: 排名学习 (直接学rank而非值)
# ═══════════════════════════════════════════════

def rank_correction(X, y, feature_names, line_names, X_df, orig_pred):
    """学习排名位置的校正，而非值的校正"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    print("\n  ── 策略6: 排名校正学习 ──")
    n = len(y)
    
    # 将y和orig_pred转换为归一化排名
    y_rank = np.argsort(np.argsort(-y)).astype(float) / n  # 0=最高, 1=最低
    orig_rank = np.argsort(np.argsort(-orig_pred)).astype(float) / n
    rank_error = y_rank - orig_rank  # 正值=排名偏低(需提升)
    
    feature_sets = {
        '核心': [feature_names.index(f) for f in [
            'topo_prop_loss_reduction', 'fault_probability',
            'isolated_load_fraction', 'betweenness_centrality',
        ] if f in feature_names],
        '精简': [feature_names.index(f) for f in [
            'topo_prop_loss_reduction', 'fault_probability',
            'isolated_load_fraction', 'expected_fault_hours',
            'is_normally_open', 'line_type_ac',
        ] if f in feature_names],
    }
    
    loo = LeaveOneOut()
    results = {}
    
    for fs_name, feat_idx in feature_sets.items():
        for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
            rank_correction_pred = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                reg = Ridge(alpha=alpha)
                reg.fit(X_train, rank_error[train_idx])
                rank_correction_pred[test_idx] = reg.predict(X_test)
            
            for damping in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                corrected_rank = orig_rank + damping * rank_correction_pred
                # 转回值: 用原始pred的值, 但按new rank排序
                new_order = np.argsort(corrected_rank)  # 越小rank越靠前
                sorted_orig = np.sort(orig_pred)[::-1]  # 原始值从大到小
                pred = np.zeros(n)
                for rank_pos, idx in enumerate(new_order):
                    pred[idx] = sorted_orig[rank_pos]
                
                pred = postprocess(pred, line_names, X_df)
                ev = quick_eval(y, pred)
                key = f"排名校正({fs_name},α={alpha},d={damping})"
                results[key] = (pred, ev)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][1]['spearman'], reverse=True)
    for name, (_, ev) in sorted_results[:5]:
        quick_eval(y, _, label=name)
    
    return results


# ═══════════════════════════════════════════════
#  策略7: 超强正则化 + 原始推理约束
# ═══════════════════════════════════════════════

def heavily_regularized(X, y, feature_names, line_names, X_df, orig_pred):
    """用极高正则化的模型，让ML只做微小修正"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    print("\n  ── 策略7: 超强正则化直接预测 ──")
    n = len(y)
    loo = LeaveOneOut()
    
    feature_sets = {
        '精简5': [feature_names.index(f) for f in [
            'orig_pred_combined',
            'topo_prop_loss_reduction',
            'expected_fault_hours', 'fault_probability',
            'isolated_load_fraction',
        ] if f in feature_names],
        '核心3': [feature_names.index(f) for f in [
            'orig_pred_combined',
            'topo_prop_loss_reduction',
            'fault_probability',
        ] if f in feature_names],
        '推理2': [feature_names.index(f) for f in [
            'orig_pred_combined',
            'topo_prop_loss_reduction',
        ] if f in feature_names],
    }
    
    results = {}
    for fs_name, feat_idx in feature_sets.items():
        for alpha in [500.0, 1000.0, 2000.0, 5000.0, 10000.0]:
            predictions = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                reg = Ridge(alpha=alpha)
                reg.fit(X_train, y[train_idx])
                predictions[test_idx] = reg.predict(X_test)
            
            predictions = postprocess(predictions, line_names, X_df)
            ev = quick_eval(y, predictions)
            key = f"超正则({fs_name},α={alpha})"
            results[key] = (predictions, ev)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][1]['spearman'], reverse=True)
    for name, (_, ev) in sorted_results[:5]:
        quick_eval(y, _, label=name)
    
    return results


# ═══════════════════════════════════════════════
#  策略8: KNN + 原始推理融合
# ═══════════════════════════════════════════════

def knn_approach(X, y, feature_names, line_names, X_df, orig_pred):
    """KNN: 找相似线路取平均值 (robust for small n)"""
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    print("\n  ── 策略8: KNN融合 ──")
    n = len(y)
    loo = LeaveOneOut()
    
    feature_sets = {
        '少量': [feature_names.index(f) for f in [
            'orig_pred_combined', 'topo_prop_loss_reduction',
            'expected_fault_hours', 'fault_probability',
            'isolated_load_fraction', 'betweenness_centrality',
            'is_normally_open', 'line_type_ac',
        ] if f in feature_names],
        '精简': [feature_names.index(f) for f in [
            'orig_pred_combined', 'topo_prop_loss_reduction',
            'fault_probability', 'isolated_load_fraction',
        ] if f in feature_names],
    }
    
    results = {}
    for fs_name, feat_idx in feature_sets.items():
        for k in [3, 5, 7, 9, 11]:
            predictions = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                
                dists = np.sum((X_train - X_test) ** 2, axis=1)
                k_actual = min(k, len(train_idx))
                nn_idx = np.argsort(dists)[:k_actual]
                
                # 距离加权平均
                nn_dists = dists[nn_idx]
                weights = 1.0 / (nn_dists + 1e-10)
                weights /= weights.sum()
                predictions[test_idx] = np.sum(weights * y[train_idx[nn_idx]])
            
            predictions = postprocess(predictions, line_names, X_df)
            
            # 融合 with orig_pred
            for w_ml in [0.3, 0.5, 0.7, 1.0]:
                pred = w_ml * predictions + (1 - w_ml) * orig_pred
                pred = postprocess(pred, line_names, X_df)
                ev = quick_eval(y, pred)
                key = f"KNN({fs_name},k={k},wML={w_ml})"
                results[key] = (pred, ev)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][1]['spearman'], reverse=True)
    for name, (_, ev) in sorted_results[:5]:
        quick_eval(y, _, label=name)
    
    return results


# ═══════════════════════════════════════════════
#  策略9: 对数变换 + Ridge
# ═══════════════════════════════════════════════

def log_transform_ridge(X, y, feature_names, line_names, X_df, orig_pred):
    """对y做log变换后回归"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    print("\n  ── 策略9: 对数变换+Ridge ──")
    n = len(y)
    
    # 对正值做log变换
    y_log = np.log1p(y * 1000)  # scale up then log
    loo = LeaveOneOut()
    
    feature_sets = {
        '精简': [feature_names.index(f) for f in [
            'orig_pred_combined', 'topo_prop_loss_reduction',
            'expected_fault_hours', 'fault_probability',
            'isolated_load_fraction', 'is_normally_open',
        ] if f in feature_names],
        '核心': [feature_names.index(f) for f in [
            'orig_pred_combined', 'topo_prop_loss_reduction',
            'fault_probability',
        ] if f in feature_names],
    }
    
    results = {}
    for fs_name, feat_idx in feature_sets.items():
        for alpha in [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
            predictions_log = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                reg = Ridge(alpha=alpha)
                reg.fit(X_train, y_log[train_idx])
                predictions_log[test_idx] = reg.predict(X_test)
            
            predictions = (np.expm1(predictions_log)) / 1000.0
            predictions = postprocess(predictions, line_names, X_df)
            ev = quick_eval(y, predictions)
            key = f"Log+Ridge({fs_name},α={alpha})"
            results[key] = (predictions, ev)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][1]['spearman'], reverse=True)
    for name, (_, ev) in sorted_results[:5]:
        quick_eval(y, _, label=name)
    
    return results


# ═══════════════════════════════════════════════
#  策略10: 阈值优化后的原始推理
# ═══════════════════════════════════════════════

def threshold_optimized_original(y, orig_pred, line_names, X_df, X, feature_names):
    """只对原始推理做阈值截断，看能否提升"""
    
    print("\n  ── 策略10: 原始推理+阈值优化 ──")
    
    results = {}
    for threshold in [0, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]:
        pred = orig_pred.copy()
        pred[pred < threshold] = 0
        pred = postprocess(pred, line_names, X_df)
        ev = quick_eval(y, pred)
        key = f"原始+阈值({threshold})"
        results[key] = (pred, ev)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][1]['spearman'], reverse=True)
    for name, (_, ev) in sorted_results[:5]:
        quick_eval(y, _, label=name)
    
    return results


# ═══════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  特征选择实验 Round 2: 智能建模策略")
    print("=" * 70)
    t_start = time.time()

    # 1. 加载数据
    print("\n[1] 加载数据...")
    merged, line_cols, baseline, data_dir, disp_path, topo_features = load_data()
    
    print("[2] 计算原始推理预测...")
    original_preds = compute_original_predictions(merged, line_cols, baseline, topo_features)
    
    gt_df = load_ground_truth()
    if gt_df is None:
        print("  ✗ 未找到 ground_truth.json")
        return 1
    
    # 2. 特征工程
    print("[3] 提取全量特征...")
    features_df = extract_features_for_all_lines(
        merged, line_cols, baseline, topo_features,
        gt_df=gt_df, original_preds=original_preds)
    
    common_lines = features_df.index.intersection(gt_df.index)
    X_df = features_df.loc[common_lines].copy()
    y = gt_df.loc[common_lines, 'actual_combined_improvement'].values
    orig_pred = np.array([original_preds.get(ln, {}).get('pred_combined', 0) for ln in common_lines])
    
    feature_cols = [c for c in X_df.columns if X_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    X = X_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    feature_names = feature_cols
    line_names = common_lines.tolist()
    
    print(f"  {len(line_names)} 条线路, {len(feature_names)} 个特征")
    
    # 3. Baseline
    print("\n[4] Baseline...")
    eval_orig = quick_eval(y, orig_pred, label="原始推理 (baseline)")
    
    # 4. 运行所有策略
    all_results = {}
    all_results['原始推理'] = (orig_pred, eval_orig)
    
    r1 = residual_learning(X, y, feature_names, line_names, X_df, orig_pred)
    all_results.update(r1)
    
    r2 = isotonic_calibration(y, orig_pred, line_names, X_df)
    all_results.update(r2)
    
    r3 = damped_residual(X, y, feature_names, line_names, X_df, orig_pred)
    all_results.update(r3)
    
    r4 = classification_correction(X, y, feature_names, line_names, X_df, orig_pred)
    all_results.update(r4)
    
    r5 = weighted_signal_fusion(X, y, feature_names, line_names, X_df, orig_pred)
    all_results.update(r5)
    
    r6 = rank_correction(X, y, feature_names, line_names, X_df, orig_pred)
    all_results.update(r6)
    
    r7 = heavily_regularized(X, y, feature_names, line_names, X_df, orig_pred)
    all_results.update(r7)
    
    r8 = knn_approach(X, y, feature_names, line_names, X_df, orig_pred)
    all_results.update(r8)
    
    r9 = log_transform_ridge(X, y, feature_names, line_names, X_df, orig_pred)
    all_results.update(r9)
    
    r10 = threshold_optimized_original(y, orig_pred, line_names, X_df, X, feature_names)
    all_results.update(r10)
    
    # 5. 最终排行
    print("\n" + "=" * 90)
    print("  ★ 最终排行榜 (Top 30)")
    print("=" * 90)
    
    sorted_all = sorted(all_results.items(), key=lambda x: x[1][1]['spearman'], reverse=True)
    
    print(f"\n  {'#':>3} {'策略':<55} {'Sp':>7} {'Kendall':>8} {'NDCG@5':>7} {'Top5':>5} {'Top10':>5}")
    print(f"  {'─'*100}")
    for rank, (name, (_, ev)) in enumerate(sorted_all[:30], 1):
        marker = " ◄BASELINE" if name == '原始推理' else ""
        beat = " ★" if ev['spearman'] > eval_orig['spearman'] and name != '原始推理' else ""
        print(f"  {rank:>3} {name:<55} {ev['spearman']:>7.4f} {ev['kendall']:>8.4f} "
              f"{ev['ndcg@5']:>7.4f} {ev['top5']:>5.0%} {ev['top10']:>5.0%}{marker}{beat}")
    
    # 找到是否有超越baseline的
    best_ml = max([(k, v) for k, v in all_results.items() if k != '原始推理'],
                  key=lambda x: x[1][1]['spearman'])
    print(f"\n  最佳ML: {best_ml[0]}")
    print(f"  Spearman: {best_ml[1][1]['spearman']:.4f} vs 原始 {eval_orig['spearman']:.4f} "
          f"(Δ={best_ml[1][1]['spearman'] - eval_orig['spearman']:+.4f})")
    
    # 打印超过baseline的所有策略
    winners = [(k, v) for k, v in sorted_all if v[1]['spearman'] > eval_orig['spearman'] and k != '原始推理']
    if winners:
        print(f"\n  🎉 超过baseline的策略共 {len(winners)} 个:")
        for name, (_, ev) in winners:
            print(f"    {name}: Sp={ev['spearman']:.4f} (+{ev['spearman']-eval_orig['spearman']:.4f})")
    else:
        print(f"\n  ⚠ 没有策略超过baseline (原始推理Sp={eval_orig['spearman']:.4f})")
        print("  建议: ML层应以原始推理为主, 仅做轻量校准")
    
    elapsed = time.time() - t_start
    print(f"\n  实验耗时: {elapsed:.1f} 秒")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
