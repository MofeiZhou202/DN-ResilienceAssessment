"""配电网线路维修优先级排序 + Julia真实验证
========================================
对35条线路(26AC + 2DC + 7VSC)进行推理预测，排名并验证:
  1. 对所有35条线路运行推理层预测，得到加固改善率
  2. 按综合改善率排序，生成维修优先级
  3. 对排名前N的线路逐个运行Julia完整流程验证
  4. 输出对比报告文档 (Markdown)

线路编号与MC故障矩阵对应关系:
  AC_Line_1~26  → row_in_sample 1~26
  DC_Line_1~2   → row_in_sample 27~28
  VSC_Line_1~7  → row_in_sample 29~35

使用方法:
  python run_ranking_validation.py                  # 完整流程(预测+验证Top10)
  python run_ranking_validation.py --predict-only   # 仅预测排序，不跑Julia
  python run_ranking_validation.py --top 5          # 只验证Top5
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from validate_inference import (
    load_data, predict_single_line, validate_single, NumpyEncoder
)


def run_all_predictions(merged, line_cols, baseline, topo_features) -> List[Dict]:
    """对所有35条线路(26AC + 2DC + 7VSC)运行推理层预测"""
    # 构建完整线路列表: AC_Line_1~26, DC_Line_1~2, VSC_Line_1~7
    all_lines = []
    for i in range(1, 27):
        all_lines.append(f"AC_Line_{i}")
    for i in range(1, 3):
        all_lines.append(f"DC_Line_{i}")
    for i in range(1, 8):
        all_lines.append(f"VSC_Line_{i}")
    
    total = len(all_lines)
    results = []
    for idx, line_name in enumerate(all_lines, 1):
        print(f"  预测中: {line_name} ({idx}/{total}) ...", end="", flush=True)
        t0 = time.time()
        pred = predict_single_line(merged, line_cols, baseline, line_name, topo_features)
        elapsed = time.time() - t0
        pred["line_name"] = line_name
        pred["line_num"] = idx
        pred["predict_time"] = elapsed
        results.append(pred)

        loss_imp = pred.get("loss_improvement", 0) * 100
        over2h_imp = pred.get("over2h_improvement", 0) * 100
        comb = pred.get("combined_improvement", 0) * 100
        n_fault = pred.get("n_fault", 0)
        crit = "★" if pred.get("topo_is_critical") else " "
        print(f" {crit} loss={loss_imp:5.2f}% over2h={over2h_imp:5.2f}% combined={comb:5.2f}% faults={n_fault} ({elapsed:.1f}s)")

    return results


def rank_lines(predictions: List[Dict]) -> List[Dict]:
    """按 combined_improvement 降序排列"""
    ranked = sorted(predictions, key=lambda x: x.get("combined_improvement", 0), reverse=True)
    for rank, r in enumerate(ranked, 1):
        r["rank"] = rank
    return ranked


def run_julia_for_top_n(ranked: List[Dict], n: int, merged, line_cols, baseline,
                        data_dir, topo_features) -> List[Dict]:
    """对排名前N的线路逐个跑Julia验证"""
    verified_results = []
    top_n = ranked[:n]
    total = len(top_n)

    for idx, pred in enumerate(top_n, 1):
        line_name = pred["line_name"]
        rank = pred["rank"]
        print(f"\n{'#'*60}")
        print(f"  Julia验证 [{idx}/{total}] 排名#{rank}: {line_name}")
        print(f"  (预测: loss={pred.get('loss_improvement', 0):.2%}, "
              f"over2h={pred.get('over2h_improvement', 0):.2%}, "
              f"combined={pred.get('combined_improvement', 0):.2%})")
        print(f"{'#'*60}")

        result = validate_single(
            line_name, merged, line_cols, baseline, data_dir,
            predict_only=False, topo_features=topo_features
        )
        result["rank"] = rank
        result["line_name"] = line_name
        verified_results.append(result)

    return verified_results


def generate_report(ranked: List[Dict], verified: List[Dict], baseline: Dict,
                    output_path: Path):
    """生成Markdown排序+验证报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    base_loss = baseline.get("expected_load_shed_total", 0)
    base_viols = len(baseline.get("violations", []))

    # 将验证结果索引化
    verified_map = {}
    for v in verified:
        verified_map[v.get("line_name", v.get("target_line", ""))] = v

    lines = []
    lines.append(f"# 配电网线路维修优先级排序报告")
    lines.append(f"")
    lines.append(f"**生成时间**: {now}")
    lines.append(f"")
    lines.append(f"## 1. 基线信息")
    lines.append(f"")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 基线期望失负荷 | {base_loss:.2f} kW·h |")
    lines.append(f"| 基线超2h违规节点 | {base_viols} 个 |")
    lines.append(f"| 蒙特卡洛场景数 | 100 |")
    lines.append(f"| 时间步 | 48 (每步30分钟，共24小时) |")
    lines.append(f"| 联络开关 | Cable24-26 (考虑转供能力) |")
    lines.append(f"")

    # 全部26条线路排序
    lines.append(f"## 2. 全部线路维修优先级排序（推理预测）")
    lines.append(f"")
    lines.append(f"排序依据: **综合改善率** = 0.6×失负荷改善 + 0.4×超时改善")
    lines.append(f"")
    lines.append(f"| 排名 | 线路 | 综合改善 | 失负荷改善 | 超时改善 | 故障数 | 影响场景 | BC | 孤立负荷 | 关键 | 方法 |")
    lines.append(f"|------|------|----------|-----------|---------|--------|---------|------|---------|------|------|")

    for r in ranked:
        name = r.get("line_name", "?")
        rank = r.get("rank", "?")
        comb = r.get("combined_improvement", 0) * 100
        loss = r.get("loss_improvement", 0) * 100
        over2h = r.get("over2h_improvement", 0) * 100
        n_fault = r.get("n_fault", 0)
        n_scen = r.get("n_affected_scenarios", "?")
        bc = r.get("topo_betweenness", 0)
        iso = r.get("topo_iso_load_mva", 0)
        crit = "★" if r.get("topo_is_critical") else ""
        method = r.get("method_used", "?")
        # 简化method名
        if "CRITICAL" in str(method):
            method_short = "topo_prop"
        elif "NON_CRITICAL" in str(method):
            method_short = "avg(topo,conn)"
        else:
            method_short = str(method)[:15]

        lines.append(f"| {rank} | {name} | {comb:.2f}% | {loss:.2f}% | {over2h:.2f}% "
                     f"| {n_fault} | {n_scen} | {bc:.3f} | {iso:.0f}MVA | {crit} | {method_short} |")

    lines.append(f"")

    # Top10 验证详情
    if verified:
        lines.append(f"## 3. Top{len(verified)} 线路Julia真实验证对比")
        lines.append(f"")
        lines.append(f"| 排名 | 线路 | 预测失负荷 | 真实失负荷 | 误差 | 预测超时 | 真实超时 | 误差 | 真实期望失负荷(kW·h) | 耗时 |")
        lines.append(f"|------|------|-----------|-----------|------|---------|---------|------|---------------------|------|")

        total_loss_err = 0
        total_over2h_err = 0
        n_ok = 0

        for v in verified:
            name = v.get("line_name", v.get("target_line", "?"))
            rank = v.get("rank", "?")
            p_loss = v.get("loss_improvement", 0) * 100
            p_over2h = v.get("over2h_improvement", 0) * 100

            if v.get("status") == "validated":
                a_loss = v["actual_loss_improvement"] * 100
                a_over2h = v["actual_over2h_improvement"] * 100
                e_loss = v["loss_error"] * 100
                e_over2h = v["over2h_error"] * 100
                loss_mark = "✓" if e_loss < 2 else "△" if e_loss < 5 else "✗"
                over2h_mark = "✓" if e_over2h < 2 else "△" if e_over2h < 5 else "✗"
                cf_loss = v.get("loss_counterfactual", "?")
                elapsed = v.get("elapsed_seconds", 0)

                lines.append(f"| {rank} | {name} | {p_loss:.2f}% | {a_loss:.2f}% | {e_loss:.2f}%{loss_mark} "
                             f"| {p_over2h:.2f}% | {a_over2h:.2f}% | {e_over2h:.2f}%{over2h_mark} "
                             f"| {cf_loss:.1f} | {elapsed:.0f}s |")
                total_loss_err += e_loss
                total_over2h_err += e_over2h
                n_ok += 1
            else:
                note = v.get("note", v.get("error", v.get("julia_error", "失败")))
                lines.append(f"| {rank} | {name} | {p_loss:.2f}% | - | - | {p_over2h:.2f}% | - | - | {note} | - |")

        lines.append(f"")
        if n_ok > 0:
            avg_l = total_loss_err / n_ok
            avg_o = total_over2h_err / n_ok
            lines.append(f"**平均预测误差**: 失负荷 {avg_l:.2f}%  |  超时 {avg_o:.2f}%")
            lines.append(f"")

        # 排序一致性分析
        lines.append(f"### 排序一致性分析")
        lines.append(f"")
        
        # 按真实改善率重新排序
        verified_with_actual = [v for v in verified if v.get("status") == "validated"]
        if len(verified_with_actual) >= 2:
            pred_order = [(v.get("line_name", ""), v.get("loss_improvement", 0)) for v in verified_with_actual]
            actual_order = sorted(verified_with_actual,
                                  key=lambda x: x.get("actual_loss_improvement", 0), reverse=True)

            lines.append(f"| 推理排名 | 线路 | 推理综合改善 | 真实失负荷改善 | 真实排名 | 排名偏差 |")
            lines.append(f"|---------|------|------------|-------------|---------|---------|")
            
            actual_rank_map = {}
            for ai, av in enumerate(actual_order, 1):
                actual_rank_map[av.get("line_name", av.get("target_line", ""))] = ai

            rank_diffs = []
            for v in verified_with_actual:
                name = v.get("line_name", v.get("target_line", ""))
                pred_rank_idx = v.get("rank", "?")
                pred_comb = v.get("combined_improvement", 0) * 100
                a_loss = v.get("actual_loss_improvement", 0) * 100
                a_rank = actual_rank_map.get(name, "?")
                if isinstance(pred_rank_idx, int) and isinstance(a_rank, int):
                    diff = abs(pred_rank_idx - a_rank)
                    rank_diffs.append(diff)
                    lines.append(f"| #{pred_rank_idx} | {name} | {pred_comb:.2f}% | {a_loss:.2f}% | #{a_rank} | {diff} |")
                else:
                    lines.append(f"| #{pred_rank_idx} | {name} | {pred_comb:.2f}% | {a_loss:.2f}% | #{a_rank} | ? |")

            lines.append(f"")
            if rank_diffs:
                kendall_ok = sum(1 for d in rank_diffs if d <= 1)
                lines.append(f"- 排名偏差≤1的线路: {kendall_ok}/{len(rank_diffs)}")
                lines.append(f"- 平均排名偏差: {np.mean(rank_diffs):.1f}")
        lines.append(f"")

    # 结论
    lines.append(f"## 4. 维修建议")
    lines.append(f"")
    lines.append(f"根据推理层综合改善率排序，建议按以下优先级进行维修加固：")
    lines.append(f"")
    for r in ranked[:10]:
        name = r.get("line_name", "?")
        rank = r.get("rank", "?")
        comb = r.get("combined_improvement", 0) * 100
        loss = r.get("loss_improvement", 0) * 100
        crit = " (关键瓶颈)" if r.get("topo_is_critical") else ""
        verified_info = ""
        if name in verified_map:
            v = verified_map[name]
            if v.get("status") == "validated":
                a_loss = v["actual_loss_improvement"] * 100
                verified_info = f" → 真实验证: {a_loss:.2f}%"
        lines.append(f"{rank}. **{name}** — 预测综合改善 {comb:.2f}% (失负荷 {loss:.2f}%){crit}{verified_info}")
    lines.append(f"")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    return report_text


def main():
    parser = argparse.ArgumentParser(description="线路维修优先级排序 + Julia真实验证")
    parser.add_argument("--predict-only", "-p", action="store_true",
                        help="仅运行预测排序，不跑Julia验证")
    parser.add_argument("--top", "-t", type=int, default=10,
                        help="Julia验证排名前N的线路 (默认10)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出报告路径")
    args = parser.parse_args()

    print("=" * 70)
    print("  配电网线路维修优先级排序与验证")
    print("=" * 70)
    t_start = time.time()

    # 加载数据
    print("\n[1/4] 加载数据...")
    merged, line_cols, baseline, data_dir, disp_path, topo_features = load_data()
    print(f"  数据: {merged.shape[0]} 行, {len(line_cols)} 条线路")
    print(f"  基线失负荷: {baseline.get('expected_load_shed_total', 0):.2f} kW·h")

    # 全部预测
    print(f"\n[2/4] 运行35条线路推理预测(26AC + 2DC + 7VSC)...")
    predictions = run_all_predictions(merged, line_cols, baseline, topo_features)

    # 排序
    print(f"\n[3/4] 按综合改善率排序...")
    ranked = rank_lines(predictions)

    print(f"\n  {'排名':>4} {'线路':<12} {'综合':>8} {'失负荷':>8} {'超时':>8} {'关键':>4}")
    print(f"  {'─'*56}")
    for r in ranked:
        name = r.get("line_name", "?")
        rank = r.get("rank", "?")
        comb = r.get("combined_improvement", 0) * 100
        loss = r.get("loss_improvement", 0) * 100
        over2h = r.get("over2h_improvement", 0) * 100
        crit = "★" if r.get("topo_is_critical") else ""
        marker = " ◄" if rank <= args.top else ""
        print(f"  {rank:>4} {name:<12} {comb:>7.2f}% {loss:>7.2f}% {over2h:>7.2f}% {crit:>4}{marker}")

    # Julia验证
    verified = []
    if not args.predict_only:
        n_verify = min(args.top, len(ranked))
        print(f"\n[4/4] Julia真实验证 Top{n_verify} 线路 (每条约5-15分钟)...")
        print(f"  预计总耗时: {n_verify * 8:.0f}~{n_verify * 15:.0f} 分钟")
        verified = run_julia_for_top_n(ranked, n_verify, merged, line_cols,
                                       baseline, data_dir, topo_features)
    else:
        print(f"\n[4/4] 跳过Julia验证 (--predict-only)")

    # 生成报告
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "output" / "line_ranking_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(ranked, verified, baseline, output_path)

    # 保存JSON
    json_path = output_path.with_suffix(".json")
    all_data = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "expected_load_shed_total": baseline.get("expected_load_shed_total", 0),
            "n_violations": len(baseline.get("violations", [])),
        },
        "ranked_lines": ranked,
        "verified_lines": verified,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  完成! 总耗时: {elapsed_total/60:.1f} 分钟")
    print(f"  报告: {output_path}")
    print(f"  数据: {json_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
