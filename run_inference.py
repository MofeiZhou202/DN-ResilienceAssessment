"""配电网线路加固优先级推理 (一键运行)
========================================
读取数据 → 构建AC-DC-VSC混合网络拓扑 → 推理预测35条线路的加固改善率
→ 按综合改善率排序 → 输出维修优先级报告

所需输入文件 (均在 data/ 目录下):
  1. ac_dc_real_case.xlsx       — 配电网拓扑 (母线/线路/VSC/DC/负荷)
  2. topology_reconfiguration_results.xlsx — 拓扑重构结果 (100场景×48时间步)
  3. mess_dispatch_hourly.xlsx   — MESS调度结果 (弹性指标: 失负荷/供电率/超时)
  4. mess_dispatch_results_key_metrics.json — Julia基线统计 (期望失负荷/违规节点)

输出文件 (在 output/ 目录下):
  1. inference_result.md   — Markdown排序报告 (可直接查看)
  2. inference_result.json — JSON格式完整数据 (可程序化读取)

使用方法:
  python run_inference.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from validate_inference import (
    load_data, predict_single_line, NumpyEncoder
)


# ────────────────────────────────────────────────────────
#  预测
# ────────────────────────────────────────────────────────

def predict_all_lines(merged, line_cols, baseline, topo_features) -> List[Dict]:
    """对35条线路 (26AC + 2DC + 7VSC) 逐条运行推理预测"""
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
        print(f"  [{idx:>2}/{total}] {line_name:<14}", end="", flush=True)
        t0 = time.time()
        pred = predict_single_line(merged, line_cols, baseline, line_name, topo_features)
        elapsed = time.time() - t0
        pred["line_name"] = line_name
        pred["predict_time"] = elapsed

        loss_pct = pred.get("loss_improvement", 0) * 100
        over2h_pct = pred.get("over2h_improvement", 0) * 100
        comb_pct = pred.get("combined_improvement", 0) * 100
        crit = "★" if pred.get("topo_is_critical") else " "
        print(f" {crit} 综合={comb_pct:5.2f}%  失负荷={loss_pct:5.2f}%  超时={over2h_pct:5.2f}%  ({elapsed:.1f}s)")
        results.append(pred)
    return results


def rank_lines(predictions: List[Dict]) -> List[Dict]:
    """按 combined_improvement 降序排列"""
    ranked = sorted(predictions, key=lambda x: x.get("combined_improvement", 0), reverse=True)
    for rank, r in enumerate(ranked, 1):
        r["rank"] = rank
    return ranked


# ────────────────────────────────────────────────────────
#  报告生成
# ────────────────────────────────────────────────────────

def generate_report(ranked: List[Dict], baseline: Dict, output_md: Path, output_json: Path):
    """生成 Markdown + JSON 报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    base_loss = baseline.get("expected_load_shed_total", 0)
    base_viols = len(baseline.get("violations", []))

    md = []
    md.append("# 配电网线路加固优先级排序报告")
    md.append(f"\n**生成时间**: {now}\n")
    md.append("## 基线信息\n")
    md.append("| 指标 | 值 |")
    md.append("|------|-----|")
    md.append(f"| 基线期望失负荷 | {base_loss:.2f} kW·h |")
    md.append(f"| 基线超2h违规节点 | {base_viols} 个 |")
    md.append(f"| 蒙特卡洛场景数 | 100 |")
    md.append(f"| 时间步 | 48 (每步30分钟，共24小时) |")
    md.append(f"| 网络规模 | 30节点(25AC+5DC), 35线路(26AC+2DC+7VSC) |")
    md.append(f"| 联络开关 | Cable24-26, VSC1, VSC3 |")
    md.append(f"| MESS移动储能 | 3台 (1500+1000+500 kW) |")
    md.append("")

    md.append("## 线路加固优先级排序\n")
    md.append("排序依据: **综合改善率** = 0.6×失负荷改善 + 0.4×超时改善\n")
    md.append("| 排名 | 线路 | 综合改善 | 失负荷改善 | 超时改善 | 故障小时数 | 影响场景 | 关键瓶颈 |")
    md.append("|:----:|------|:--------:|:---------:|:-------:|:---------:|:-------:|:-------:|")

    for r in ranked:
        name = r.get("line_name", "?")
        rank = r.get("rank", "?")
        comb = r.get("combined_improvement", 0) * 100
        loss = r.get("loss_improvement", 0) * 100
        over2h = r.get("over2h_improvement", 0) * 100
        n_fault = r.get("n_fault", 0)
        n_scen = r.get("n_affected_scenarios", "?")
        crit = "★" if r.get("topo_is_critical") else ""
        md.append(f"| {rank} | {name} | {comb:.2f}% | {loss:.2f}% | {over2h:.2f}% "
                  f"| {n_fault} | {n_scen} | {crit} |")

    md.append("")
    md.append("## 维修建议\n")
    for r in ranked[:10]:
        name = r.get("line_name", "?")
        rank = r.get("rank", "?")
        comb = r.get("combined_improvement", 0) * 100
        loss = r.get("loss_improvement", 0) * 100
        crit = " (**关键瓶颈**)" if r.get("topo_is_critical") else ""
        md.append(f"{rank}. **{name}** — 预测综合改善 {comb:.2f}% (失负荷 {loss:.2f}%){crit}")

    md.append("")
    md.append("---\n")
    md.append("*关键瓶颈(★): BC>0.08 或 孤立负荷比例>20%*\n")
    md.append("*综合改善率 = 0.6×失负荷改善 + 0.4×超时改善*\n")
    md.append("*对比真实效果请参考 `output/ground_truth.md`*")

    report_text = "\n".join(md)
    output_md.write_text(report_text, encoding="utf-8")

    # JSON
    all_data = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "expected_load_shed_total": base_loss,
            "n_violations": base_viols,
        },
        "ranked_lines": ranked,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    return report_text


# ────────────────────────────────────────────────────────
#  主入口
# ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  配电网线路加固优先级推理")
    print("=" * 60)
    t_start = time.time()

    # Step 1: 加载数据
    print("\n[1/3] 加载数据...")
    try:
        merged, line_cols, baseline, data_dir, disp_path, topo_features = load_data()
    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        print(f"\n  请确保以下文件存在于 data/ 目录:")
        print(f"    1. ac_dc_real_case.xlsx")
        print(f"    2. topology_reconfiguration_results.xlsx")
        print(f"    3. mess_dispatch_hourly.xlsx (需含 HourlyDetails 工作表)")
        print(f"    4. mess_dispatch_results_key_metrics.json")
        return 1
    print(f"  数据: {merged.shape[0]} 行, {len(line_cols)} 条线路")
    print(f"  基线失负荷: {baseline.get('expected_load_shed_total', 0):.2f} kW·h")

    # Step 2: 推理预测
    print(f"\n[2/3] 推理预测 35 条线路 (26AC + 2DC + 7VSC)...")
    predictions = predict_all_lines(merged, line_cols, baseline, topo_features)

    # Step 3: 排序输出
    print(f"\n[3/3] 生成报告...")
    ranked = rank_lines(predictions)

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    output_md = output_dir / "inference_result.md"
    output_json = output_dir / "inference_result.json"
    generate_report(ranked, baseline, output_md, output_json)

    # 打印排名摘要
    print(f"\n  {'排名':>4} {'线路':<14} {'综合':>8} {'失负荷':>8} {'超时':>8} {'关键':>4}")
    print(f"  {'─'*58}")
    for r in ranked:
        name = r.get("line_name", "?")
        rank = r.get("rank", "?")
        comb = r.get("combined_improvement", 0) * 100
        loss = r.get("loss_improvement", 0) * 100
        over2h = r.get("over2h_improvement", 0) * 100
        crit = "★" if r.get("topo_is_critical") else ""
        print(f"  {rank:>4} {name:<14} {comb:>7.2f}% {loss:>7.2f}% {over2h:>7.2f}% {crit:>4}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  完成! 耗时 {elapsed:.0f} 秒")
    print(f"  报告: {output_md}")
    print(f"  数据: {output_json}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
