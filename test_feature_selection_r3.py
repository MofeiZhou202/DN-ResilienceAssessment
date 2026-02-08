"""特征选择实验 Round 3: 精细化优化 + 稳定性验证
====================================================
Round 2 发现: 
  1. 分类矫正(少量,C=0.1,t=0.3) Sp=0.9173 — 最佳
  2. 阻尼残差(核心,α=50,d=0.1)  Sp=0.8916
  3. 原始+阈值(0.005)           Sp=0.8826

本轮:
  1. 精细化搜索分类矫正的最优参数
  2. 组合策略: 分类矫正 + 阻尼残差
  3. 稳定性验证: Leave-K-Out & Bootstrap
  4. 找出最终推荐方案
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
        print(f"    {label:<50} Sp={sp:.4f} Kt={kt:.4f} N@5={ndcg5:.4f} T5={t5:.0%} T10={t10:.0%}")
    return ev


def postprocess(pred, line_names, X_df):
    pred = np.maximum(pred, 0)
    for i, ln in enumerate(line_names):
        if X_df.loc[ln, 'is_normally_open'] == 1:
            pred[i] = 0
        if not ln.startswith('AC_Line_'):
            pred[i] = 0
    return pred


def main():
    print("=" * 70)
    print("  特征选择实验 Round 3: 精细化 + 稳定性验证")
    print("=" * 70)
    t_start = time.time()

    # === 加载数据 ===
    print("\n[1] 加载数据...")
    merged, line_cols, baseline, data_dir, disp_path, topo_features = load_data()
    original_preds = compute_original_predictions(merged, line_cols, baseline, topo_features)
    gt_df = load_ground_truth()
    if gt_df is None:
        return 1
    
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
    n = len(line_names)
    
    eval_orig = quick_eval(y, orig_pred, label="原始推理 BASELINE")
    
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import LeaveOneOut
    
    y_binary = (y > 0).astype(int)
    loo = LeaveOneOut()
    
    # ═══════════════════════════════════════════════
    #  Part 1: 分类矫正精细搜索
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Part 1: 分类矫正精细搜索")
    print("=" * 70)
    
    feature_combos = {
        'F1: is_no+lt_ac+fp+orig': ['is_normally_open', 'line_type_ac', 'fault_probability', 'orig_pred_combined'],
        'F2: F1+topo+iso': ['is_normally_open', 'line_type_ac', 'fault_probability', 'orig_pred_combined', 'topo_prop_loss_reduction', 'isolated_load_fraction'],
        'F3: F2+efh': ['is_normally_open', 'line_type_ac', 'fault_probability', 'orig_pred_combined', 'topo_prop_loss_reduction', 'isolated_load_fraction', 'expected_fault_hours'],
        'F4: F3+bc+bridge': ['is_normally_open', 'line_type_ac', 'fault_probability', 'orig_pred_combined', 'topo_prop_loss_reduction', 'isolated_load_fraction', 'expected_fault_hours', 'betweenness_centrality', 'is_bridge'],
        'F5: 只is_no+lt_ac+orig': ['is_normally_open', 'line_type_ac', 'orig_pred_combined'],
        'F6: 只fp+orig+topo': ['fault_probability', 'orig_pred_combined', 'topo_prop_loss_reduction'],
        'F7: fp+orig': ['fault_probability', 'orig_pred_combined'],
        'F8: fp+orig+efh+iso': ['fault_probability', 'orig_pred_combined', 'expected_fault_hours', 'isolated_load_fraction'],
        'F9: 拓扑核心无推理': ['is_normally_open', 'line_type_ac', 'fault_probability', 'topo_prop_loss_reduction', 'isolated_load_fraction'],
        'F10: 只orig_pred': ['orig_pred_combined'],
    }
    
    all_cls_results = []
    
    for f_label, feat_list in feature_combos.items():
        feat_idx = [feature_names.index(f) for f in feat_list if f in feature_names]
        
        for C_val in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
            # LOO predictions
            probs = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X[train_idx][:, feat_idx])
                X_test = scaler.transform(X[test_idx][:, feat_idx])
                clf = LogisticRegression(C=C_val, max_iter=500, random_state=42)
                clf.fit(X_train, y_binary[train_idx])
                probs[test_idx] = clf.predict_proba(X_test)[:, 1]
            
            for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                pred = orig_pred.copy()
                for i in range(n):
                    if probs[i] < threshold:
                        pred[i] = 0
                pred = postprocess(pred, line_names, X_df)
                ev = quick_eval(y, pred)
                all_cls_results.append({
                    'features': f_label,
                    'C': C_val,
                    'threshold': threshold,
                    'spearman': ev['spearman'],
                    'kendall': ev['kendall'],
                    'ndcg@5': ev['ndcg@5'],
                    'top5': ev['top5'],
                    'top10': ev['top10'],
                    'n_feat': len(feat_idx),
                    'probs': probs.copy(),
                    'predictions': pred.copy(),
                })
    
    # 排序
    all_cls_results.sort(key=lambda x: x['spearman'], reverse=True)
    
    print(f"\n  Top 15 分类矫正组合:")
    print(f"  {'#':>3} {'特征组合':<30} {'C':>5} {'阈值':>5} {'维度':>3} {'Sp':>7} {'Kt':>7} {'N@5':>6} {'T5':>4} {'T10':>4}")
    print(f"  {'─'*95}")
    for i, r in enumerate(all_cls_results[:15], 1):
        print(f"  {i:>3} {r['features']:<30} {r['C']:>5.2f} {r['threshold']:>5.2f} {r['n_feat']:>3} "
              f"{r['spearman']:>7.4f} {r['kendall']:>7.4f} {r['ndcg@5']:>6.4f} {r['top5']:>4.0%} {r['top10']:>4.0%}")
    
    best_cls = all_cls_results[0]
    
    # ═══════════════════════════════════════════════
    #  Part 2: 诊断分析 — 看分类器做了什么
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Part 2: 诊断分析 — 最佳分类器做了什么?")
    print("=" * 70)
    
    probs_best = best_cls['probs']
    pred_best = best_cls['predictions']
    threshold_best = best_cls['threshold']
    
    print(f"\n  最佳: {best_cls['features']}, C={best_cls['C']}, t={best_cls['threshold']}")
    print(f"  结果: Sp={best_cls['spearman']:.4f}, Kt={best_cls['kendall']:.4f}")
    
    print(f"\n  {'线路':<14} {'实际':>8} {'原始预测':>8} {'ML预测':>8} {'分类P':>7} {'决策':>6} {'判断':>6}")
    print(f"  {'─'*65}")
    
    for idx in np.argsort(-y):
        ln = line_names[idx]
        actual = y[idx]
        orig = orig_pred[idx]
        ml = pred_best[idx]
        prob = probs_best[idx]
        decision = "保留" if prob >= threshold_best else "置零"
        
        # 检查判断是否正确
        actual_status = "正" if actual > 0 else "零"
        pred_status = "正" if ml > 0 else "零"
        correct = "✓" if actual_status == pred_status else "✗"
        
        print(f"  {ln:<14} {actual*100:>7.3f}% {orig*100:>7.3f}% {ml*100:>7.3f}% {prob:>7.3f} {decision:>6} {correct:>6}")
    
    # ═══════════════════════════════════════════════
    #  Part 3: 组合策略 — 分类矫正 + 阻尼残差
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Part 3: 组合策略 — 分类矫正 + 阻尼残差")
    print("=" * 70)
    
    residual = y - orig_pred
    
    # 残差特征集
    residual_feature_sets = {
        '核心3': [feature_names.index(f) for f in [
            'topo_prop_loss_reduction', 'fault_probability', 'isolated_load_fraction',
        ] if f in feature_names],
        '精简5': [feature_names.index(f) for f in [
            'topo_prop_loss_reduction', 'expected_fault_hours', 'fault_probability',
            'isolated_load_fraction', 'is_normally_open',
        ] if f in feature_names],
        '少量8': [feature_names.index(f) for f in [
            'topo_prop_loss_reduction', 'plain_prop_loss_reduction',
            'expected_fault_hours', 'fault_probability',
            'betweenness_centrality', 'isolated_load_fraction',
            'is_normally_open', 'line_type_ac',
        ] if f in feature_names],
    }
    
    combo_results = []
    
    # 使用top5分类矫正结果
    for cls_res in all_cls_results[:5]:
        cls_pred = cls_res['predictions']
        cls_probs = cls_res['probs']
        cls_threshold = cls_res['threshold']
        cls_label = f"Cls({cls_res['features'][:10]},C={cls_res['C']},t={cls_res['threshold']})"
        
        for res_name, res_feat_idx in residual_feature_sets.items():
            for alpha in [10.0, 50.0, 100.0, 200.0]:
                # 先学残差
                raw_residual = np.zeros(n)
                for train_idx, test_idx in loo.split(X):
                    scaler = RobustScaler()
                    X_train = scaler.fit_transform(X[train_idx][:, res_feat_idx])
                    X_test = scaler.transform(X[test_idx][:, res_feat_idx])
                    reg = Ridge(alpha=alpha)
                    reg.fit(X_train, residual[train_idx])
                    raw_residual[test_idx] = reg.predict(X_test)
                
                for damping in [0.05, 0.1, 0.15, 0.2, 0.3]:
                    # 阻尼残差 + 分类矫正
                    pred = orig_pred + damping * raw_residual
                    for i in range(n):
                        if cls_probs[i] < cls_threshold:
                            pred[i] = 0
                    pred = postprocess(pred, line_names, X_df)
                    ev = quick_eval(y, pred)
                    label = f"{cls_label}+残差({res_name},α={alpha},d={damping})"
                    combo_results.append({
                        'label': label,
                        'spearman': ev['spearman'],
                        'kendall': ev['kendall'],
                        'ndcg@5': ev['ndcg@5'],
                        'top5': ev['top5'],
                        'top10': ev['top10'],
                        'predictions': pred.copy(),
                    })
    
    combo_results.sort(key=lambda x: x['spearman'], reverse=True)
    
    print(f"\n  Top 10 组合策略:")
    print(f"  {'#':>3} {'策略':<65} {'Sp':>7} {'Kt':>7} {'N@5':>6} {'T5':>4} {'T10':>4}")
    print(f"  {'─'*105}")
    for i, r in enumerate(combo_results[:10], 1):
        print(f"  {i:>3} {r['label'][:65]:<65} {r['spearman']:>7.4f} {r['kendall']:>7.4f} "
              f"{r['ndcg@5']:>6.4f} {r['top5']:>4.0%} {r['top10']:>4.0%}")
    
    # ═══════════════════════════════════════════════
    #  Part 4: 稳定性验证 — Bootstrap
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Part 4: 稳定性验证 (Bootstrap, 200次)")
    print("=" * 70)
    
    # 选几个候选策略验证稳定性
    candidates = {
        '原始推理': orig_pred,
        f'最佳分类矫正': best_cls['predictions'],
    }
    if combo_results:
        candidates['最佳组合'] = combo_results[0]['predictions']
    
    # 也加入阻尼残差
    # 重新计算阻尼残差(核心,α=50,d=0.1)
    res_feat_idx = [feature_names.index(f) for f in [
        'topo_prop_loss_reduction', 'fault_probability', 'isolated_load_fraction',
    ] if f in feature_names]
    damped_residual_pred = np.zeros(n)
    for train_idx, test_idx in loo.split(X):
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X[train_idx][:, res_feat_idx])
        X_test = scaler.transform(X[test_idx][:, res_feat_idx])
        reg = Ridge(alpha=50.0)
        reg.fit(X_train, residual[train_idx])
        damped_residual_pred[test_idx] = reg.predict(X_test)
    damped_pred = orig_pred + 0.1 * damped_residual_pred
    damped_pred = postprocess(damped_pred, line_names, X_df)
    candidates['阻尼残差(核心,α=50,d=0.1)'] = damped_pred
    
    # 阈值优化
    threshold_pred = orig_pred.copy()
    threshold_pred[threshold_pred < 0.005] = 0
    threshold_pred = postprocess(threshold_pred, line_names, X_df)
    candidates['原始+阈值(0.005)'] = threshold_pred
    
    np.random.seed(42)
    n_bootstrap = 200
    
    print(f"\n  {'策略':<40} {'Mean Sp':>8} {'Std Sp':>8} {'Min Sp':>8} {'Max Sp':>8} {'Mean Kt':>8}")
    print(f"  {'─'*90}")
    
    bootstrap_full = {}
    for name, pred in candidates.items():
        sps = []
        kts = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            # 确保有方差
            if len(np.unique(y[idx])) < 3 or len(np.unique(pred[idx])) < 3:
                continue
            sp = spearmanr(y[idx], pred[idx])[0]
            kt = kendalltau(y[idx], pred[idx])[0]
            if not np.isnan(sp):
                sps.append(sp)
            if not np.isnan(kt):
                kts.append(kt)
        
        sps = np.array(sps)
        kts = np.array(kts)
        bootstrap_full[name] = sps
        print(f"  {name:<40} {np.mean(sps):>8.4f} {np.std(sps):>8.4f} "
              f"{np.min(sps):>8.4f} {np.max(sps):>8.4f} {np.mean(kts):>8.4f}")
    
    # ═══════════════════════════════════════════════
    #  Part 5: Leave-3-Out 交叉验证
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Part 5: Leave-3-Out 稳定性检验 (分类矫正)")
    print("=" * 70)
    
    # 对最佳分类矫正方案做L3O验证
    best_feat_list = [f for f in feature_combos[best_cls['features']] if f in feature_names]
    best_feat_idx = [feature_names.index(f) for f in best_feat_list]
    best_C = best_cls['C']
    best_threshold = best_cls['threshold']
    
    from itertools import combinations
    
    np.random.seed(42)
    # 随机抽100组3-out
    all_indices = list(range(n))
    all_combos = list(combinations(all_indices, 3))
    np.random.shuffle(all_combos)
    selected_combos = all_combos[:min(200, len(all_combos))]
    
    l3o_sps_cls = []
    l3o_sps_orig = []
    
    for test_set in selected_combos:
        test_idx = list(test_set)
        train_idx = [i for i in range(n) if i not in test_idx]
        
        if len(np.unique(y[test_idx])) < 2:
            continue
        
        # 训练分类器
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X[np.array(train_idx)][:, best_feat_idx])
        X_test = scaler.transform(X[np.array(test_idx)][:, best_feat_idx])
        clf = LogisticRegression(C=best_C, max_iter=500, random_state=42)
        clf.fit(X_train, y_binary[train_idx])
        probs = clf.predict_proba(X_test)[:, 1]
        
        pred = orig_pred[test_idx].copy()
        for i in range(len(test_idx)):
            if probs[i] < best_threshold:
                pred[i] = 0
        
        # Spearman on test set
        if len(np.unique(pred)) >= 2 and len(np.unique(y[test_idx])) >= 2:
            sp_cls = spearmanr(y[test_idx], pred)[0]
            sp_orig = spearmanr(y[test_idx], orig_pred[test_idx])[0]
            if not np.isnan(sp_cls) and not np.isnan(sp_orig):
                l3o_sps_cls.append(sp_cls)
                l3o_sps_orig.append(sp_orig)
    
    l3o_sps_cls = np.array(l3o_sps_cls)
    l3o_sps_orig = np.array(l3o_sps_orig)
    
    print(f"\n  L3O验证 ({len(l3o_sps_cls)} 组):")
    print(f"    分类矫正: Mean Sp={np.mean(l3o_sps_cls):.4f} ± {np.std(l3o_sps_cls):.4f}")
    print(f"    原始推理: Mean Sp={np.mean(l3o_sps_orig):.4f} ± {np.std(l3o_sps_orig):.4f}")
    print(f"    分类矫正胜出比例: {np.mean(l3o_sps_cls > l3o_sps_orig):.1%}")
    
    # ═══════════════════════════════════════════════
    #  Part 6: 最终推荐
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ★ 最终推荐方案")
    print("=" * 70)
    
    print(f"""
  基于 3 轮系统实验的结论:

  1. 原始 40 维特征确实过多 (Sp=0.77), 最优特征维度为 3-8 维
  
  2. ML 最大价值不是预测值, 而是分类零/非零:
     - 最佳策略: 分类矫正
     - 特征: {best_cls['features']}
     - 参数: C={best_cls['C']}, threshold={best_cls['threshold']}
     - LOO-CV Spearman: {best_cls['spearman']:.4f} (vs baseline {eval_orig['spearman']:.4f})
  
  3. 阻尼残差学习也有效:
     - 思路: 学 y-orig_pred 残差, 但只用 10% 修正量
     - 特征: topo_prop_loss_reduction, fault_probability, isolated_load_fraction
     - LOO-CV Spearman: 0.8916

  4. 推荐的ML推理层架构:
     Step 1: 计算原始推理预测 (predict_single_line)
     Step 2: 用分类器(3-7个特征)判断零/非零 → 过滤假阳性
     Step 3: (可选) 用阻尼残差(3个特征)微调预测值
     Step 4: 后处理 (DC/VSC置零, 常开置零)
    """)
    
    elapsed = time.time() - t_start
    print(f"  实验耗时: {elapsed:.1f} 秒")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
