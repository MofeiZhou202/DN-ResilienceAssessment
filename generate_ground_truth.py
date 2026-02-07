"""生成真实验证文档 (Julia Ground Truth)
==========================================
对每条线路运行Julia完整弹性评估流程，得到真实的加固效果，
保存到 output/ground_truth.md 和 output/ground_truth.json。

以后只需对照该文档校验推理结果，无需重新运行Julia。

使用方法:
  python generate_ground_truth.py              # 所有35条线路 (约 35×8 = 280分钟)
  python generate_ground_truth.py --top 10      # 仅Top-10 (约 80分钟)
  python generate_ground_truth.py --lines AC_Line_12 AC_Line_1 VSC_Line_1  # 指定线路
  python generate_ground_truth.py --resume      # 断点续跑 (跳过已验证的线路)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from validate_inference import (
    NumpyEncoder, load_data, predict_single_line,
    run_julia_verification,
)


OUTPUT_DIR   = PROJECT_ROOT / "output"
GT_JSON_PATH = OUTPUT_DIR / "ground_truth.json"
GT_MD_PATH   = OUTPUT_DIR / "ground_truth.md"


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

ALL_LINES = (
    [f"AC_Line_{i}" for i in range(1, 27)]
    + [f"DC_Line_{i}" for i in range(1, 3)]
    + [f"VSC_Line_{i}" for i in range(1, 8)]
)


def load_existing_gt() -> Dict:
    """读取已有的ground_truth.json (用于断点续跑)"""
    if GT_JSON_PATH.exists():
        with open(GT_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"lines": {}}


def save_gt(gt: Dict):
    """保存ground_truth.json (每次验证完一条就写入)"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(GT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


# ─────────────────────────────────────────────────────────
# Julia验证 + 收集结果
# ─────────────────────────────────────────────────────────

def verify_one_line(line_name: str, data_dir: Path,
                    merged, line_cols, baseline, topo_features) -> Dict:
    """对单条线路同时获取推理预测 + Julia真实验证"""
    # 推理预测
    pred = predict_single_line(merged, line_cols, baseline, line_name, topo_features)

    # Julia真实验证
    print(f"\n  [Julia] 验证 {line_name} ...")
    t0 = time.time()
    julia_res = run_julia_verification([line_name], data_dir)
    elapsed = time.time() - t0

    result = {
        "line_name": line_name,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        # 推理预测值
        "predicted_loss_improvement": float(pred.get("loss_improvement", 0)),
        "predicted_over2h_improvement": float(pred.get("over2h_improvement", 0)),
        "predicted_combined_improvement": float(pred.get("combined_improvement", 0)),
        "n_fault": int(pred.get("n_fault", 0)),
        "n_affected_scenarios": int(pred.get("n_affected_scenarios", 0)),
        "topo_is_critical": bool(pred.get("topo_is_critical", False)),
        "topo_betweenness": float(pred.get("topo_betweenness", 0)),
    }

    if julia_res.get("status") == "validated":
        result["julia_status"] = "success"
        result["actual_loss_improvement"] = float(julia_res["loss_improvement_actual"])
        result["actual_over2h_improvement"] = float(julia_res["over2h_improvement_actual"])
        result["actual_combined_improvement"] = round(
            0.6 * julia_res["loss_improvement_actual"]
            + 0.4 * julia_res["over2h_improvement_actual"], 6
        )
        result["loss_original_kwh"] = float(julia_res["loss_original"])
        result["loss_counterfactual_kwh"] = float(julia_res["loss_counterfactual"])
        result["supply_ratio_original"] = float(julia_res.get("supply_ratio_original", 0))
        result["supply_ratio_counterfactual"] = float(julia_res.get("supply_ratio_counterfactual", 0))
        result["supply_ratio_improvement"] = float(julia_res.get("supply_ratio_improvement", 0))
        result["loss_error"] = round(abs(
            result["predicted_loss_improvement"] - result["actual_loss_improvement"]
        ), 6)
        result["over2h_error"] = round(abs(
            result["predicted_over2h_improvement"] - result["actual_over2h_improvement"]
        ), 6)
    else:
        result["julia_status"] = "failed"
        result["julia_error"] = julia_res.get("error", "unknown")

    return result


# ─────────────────────────────────────────────────────────
# Markdown报告生成
# ─────────────────────────────────────────────────────────

def generate_markdown(gt: Dict, baseline: Dict):
    """根据已收集的ground_truth数据生成Markdown"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    base_loss = baseline.get("expected_load_shed_total", 0)
    base_viols = len(baseline.get("violations", []))

    entries = list(gt.get("lines", {}).values())
    # 按真实综合改善率排序
    ok_entries = [e for e in entries if e.get("julia_status") == "success"]
    failed_entries = [e for e in entries if e.get("julia_status") != "success"]
    ok_entries.sort(key=lambda x: x.get("actual_combined_improvement", 0), reverse=True)

    md = []
    md.append("# 配电网线路加固效果 — Julia真实验证文档")
    md.append("")
    md.append(f"**生成时间**: {now}")
    md.append(f"**验证线路数**: {len(ok_entries)} 成功, {len(failed_entries)} 失败")
    md.append("")

    md.append("## 基线信息")
    md.append("")
    md.append("| 指标 | 值 |")
    md.append("|------|-----|")
    md.append(f"| 基线期望失负荷 | {base_loss:.2f} kW·h |")
    md.append(f"| 基线超2h违规节点 | {base_viols} 个 |")
    md.append(f"| 蒙特卡洛场景数 | 100 |")
    md.append(f"| 时间步 | 48 (每步30分钟，共24小时) |")
    md.append(f"| 网络规模 | 30节点(25AC+5DC), 35线路(26AC+2DC+7VSC) |")
    md.append("")

    md.append("## 真实验证结果排序 (按Julia综合改善率)")
    md.append("")
    md.append("综合改善率 = 0.6×失负荷改善 + 0.4×超时改善")
    md.append("")
    md.append("| 真实排名 | 线路 | Julia综合改善 | Julia失负荷改善 | Julia超时改善 | "
              "加固后失负荷(kW·h) | 供电率提升 | 关键 |")
    md.append("|:--------:|------|:------------:|:--------------:|:------------:|"
              ":------------------:|:---------:|:----:|")

    for rank, e in enumerate(ok_entries, 1):
        name = e["line_name"]
        ac = e["actual_combined_improvement"] * 100
        al = e["actual_loss_improvement"] * 100
        ao = e["actual_over2h_improvement"] * 100
        cf_loss = e["loss_counterfactual_kwh"]
        sr = e.get("supply_ratio_improvement", 0) * 100
        crit = "★" if e.get("topo_is_critical") else ""
        md.append(f"| {rank} | {name} | {ac:.2f}% | {al:.2f}% | {ao:.2f}% "
                  f"| {cf_loss:.1f} | +{sr:.2f}% | {crit} |")

    md.append("")

    # 推理vs真实对比
    md.append("## 推理预测 vs Julia真实 对比")
    md.append("")
    md.append("| 线路 | 推理综合 | Julia综合 | 推理失负荷 | Julia失负荷 | 失负荷误差 | "
              "推理超时 | Julia超时 | 超时误差 |")
    md.append("|------|:-------:|:--------:|:---------:|:----------:|:---------:|"
              ":------:|:--------:|:-------:|")

    loss_errs = []
    over2h_errs = []
    for e in ok_entries:
        name = e["line_name"]
        pc = e["predicted_combined_improvement"] * 100
        ac = e["actual_combined_improvement"] * 100
        pl = e["predicted_loss_improvement"] * 100
        al = e["actual_loss_improvement"] * 100
        le = e["loss_error"] * 100
        po = e["predicted_over2h_improvement"] * 100
        ao = e["actual_over2h_improvement"] * 100
        oe = e["over2h_error"] * 100
        loss_errs.append(e["loss_error"])
        over2h_errs.append(e["over2h_error"])

        le_tag = "✓" if e["loss_error"] < 0.02 else ("△" if e["loss_error"] < 0.05 else "✗")
        oe_tag = "✓" if e["over2h_error"] < 0.02 else ("△" if e["over2h_error"] < 0.05 else "✗")
        md.append(f"| {name} | {pc:.2f}% | {ac:.2f}% | {pl:.2f}% | {al:.2f}% | "
                  f"{le:.2f}%{le_tag} | {po:.2f}% | {ao:.2f}% | {oe:.2f}%{oe_tag} |")

    if loss_errs:
        md.append("")
        md.append(f"**平均失负荷误差**: {np.mean(loss_errs)*100:.2f}%  |  "
                  f"**平均超时误差**: {np.mean(over2h_errs)*100:.2f}%")
        md.append(f"**最大失负荷误差**: {np.max(loss_errs)*100:.2f}%  |  "
                  f"**最大超时误差**: {np.max(over2h_errs)*100:.2f}%")

    md.append("")

    # 每条线路详情
    md.append("## 各线路验证详情")
    md.append("")
    for e in ok_entries:
        name = e["line_name"]
        md.append(f"### {name}")
        md.append("")
        md.append(f"- **验证时间**: {e['timestamp']}")
        md.append(f"- **Julia耗时**: {e['elapsed_seconds']:.0f}秒")
        md.append(f"- **故障样本数**: {e['n_fault']}")
        md.append(f"- **影响场景数**: {e['n_affected_scenarios']}")
        md.append(f"- **拓扑关键度**: BC={e['topo_betweenness']:.4f}, "
                  f"{'关键瓶颈★' if e.get('topo_is_critical') else '普通'}")
        md.append(f"- **期望失负荷**: {e['loss_original_kwh']:.2f} → {e['loss_counterfactual_kwh']:.2f} kW·h")
        sr_orig = e.get('supply_ratio_original', 0)
        sr_cf = e.get('supply_ratio_counterfactual', 0)
        md.append(f"- **供电率**: {sr_orig:.2%} → {sr_cf:.2%}")
        md.append(f"- **真实失负荷改善**: {e['actual_loss_improvement']:.2%} "
                  f"(推理: {e['predicted_loss_improvement']:.2%}, 误差: {e['loss_error']:.2%})")
        md.append(f"- **真实超时改善**: {e['actual_over2h_improvement']:.2%} "
                  f"(推理: {e['predicted_over2h_improvement']:.2%}, 误差: {e['over2h_error']:.2%})")
        md.append("")

    if failed_entries:
        md.append("## 验证失败的线路")
        md.append("")
        for e in failed_entries:
            md.append(f"- **{e['line_name']}**: {e.get('julia_error', '未知错误')}")
        md.append("")

    md.append("---")
    md.append("*✓ 误差<2%  △ 误差2-5%  ✗ 误差>5%*")
    md.append("*综合改善率 = 0.6×失负荷改善 + 0.4×超时改善*")
    md.append("*本文档由 generate_ground_truth.py 自动生成，可直接用于校验推理结果*")

    text = "\n".join(md)
    GT_MD_PATH.write_text(text, encoding="utf-8")
    return text


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成线路加固真实验证文档")
    parser.add_argument("--top", type=int, default=0,
                        help="仅验证推理排名前N条 (需先运行 run_inference.py)")
    parser.add_argument("--lines", nargs="+", default=None,
                        help="指定要验证的线路, e.g. AC_Line_12 VSC_Line_1")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑 (跳过已验证成功的线路)")
    parser.add_argument("--all", action="store_true",
                        help="验证全部35条线路")
    args = parser.parse_args()

    print("=" * 60)
    print("  生成线路加固真实验证文档 (Julia Ground Truth)")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据...")
    merged, line_cols, baseline, data_dir, disp_path, topo_features = load_data()
    print(f"    基线失负荷: {baseline.get('expected_load_shed_total', 0):.2f} kW·h")

    # 确定要验证的线路列表
    if args.lines:
        lines_to_verify = args.lines
    elif args.top > 0:
        # 先跑推理得到排名
        inf_json = OUTPUT_DIR / "inference_result.json"
        if not inf_json.exists():
            print(f"\n  ✗ 未找到 {inf_json}")
            print(f"    请先运行: python run_inference.py")
            return 1
        with open(inf_json, "r", encoding="utf-8") as f:
            inf_data = json.load(f)
        ranked = inf_data.get("ranked_lines", [])
        lines_to_verify = [r["line_name"] for r in ranked[:args.top]]
    elif args.all:
        lines_to_verify = ALL_LINES.copy()
    else:
        # 默认: Top-10
        inf_json = OUTPUT_DIR / "inference_result.json"
        if inf_json.exists():
            with open(inf_json, "r", encoding="utf-8") as f:
                inf_data = json.load(f)
            ranked = inf_data.get("ranked_lines", [])
            lines_to_verify = [r["line_name"] for r in ranked[:10]]
            print(f"    默认验证推理排名 Top-10 (使用 --all 验证全部)")
        else:
            lines_to_verify = ALL_LINES.copy()
            print(f"    未找到推理结果，将验证全部 {len(lines_to_verify)} 条线路")

    # 断点续跑
    gt = load_existing_gt()
    if args.resume:
        already_done = {k for k, v in gt.get("lines", {}).items()
                        if v.get("julia_status") == "success"}
        before = len(lines_to_verify)
        lines_to_verify = [l for l in lines_to_verify if l not in already_done]
        skipped = before - len(lines_to_verify)
        if skipped:
            print(f"    跳过 {skipped} 条已验证成功的线路")

    if not lines_to_verify:
        print("\n  所有线路已验证完毕!")
        generate_markdown(gt, baseline)
        print(f"  报告: {GT_MD_PATH}")
        return 0

    total = len(lines_to_verify)
    est_min = total * 8
    print(f"\n[2] 即将验证 {total} 条线路 (预计 {est_min} 分钟 ≈ {est_min/60:.1f} 小时)")
    for i, name in enumerate(lines_to_verify, 1):
        print(f"    {i:>2}. {name}")

    # 逐条验证
    print(f"\n[3] 开始Julia验证...\n")
    t_start = time.time()
    if "lines" not in gt:
        gt["lines"] = {}
    gt["baseline"] = {
        "expected_load_shed_total": baseline.get("expected_load_shed_total", 0),
        "n_violations": len(baseline.get("violations", [])),
    }

    for idx, line_name in enumerate(lines_to_verify, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{total}] {line_name}")
        print(f"{'='*60}")
        t0 = time.time()
        result = verify_one_line(line_name, data_dir,
                                 merged, line_cols, baseline, topo_features)
        elapsed_line = time.time() - t0
        status = result.get("julia_status", "?")
        if status == "success":
            loss_imp = result["actual_loss_improvement"] * 100
            print(f"  ✓ {line_name}: Julia综合改善 {result['actual_combined_improvement']*100:.2f}% "
                  f"(失负荷 {loss_imp:.2f}%) | {elapsed_line:.0f}s")
        else:
            print(f"  ✗ {line_name}: {result.get('julia_error', '?')} | {elapsed_line:.0f}s")

        gt["lines"][line_name] = result
        save_gt(gt)  # 每条都保存，防丢失

        # 进度预估
        elapsed_total = time.time() - t_start
        remaining = total - idx
        avg_per_line = elapsed_total / idx
        eta = remaining * avg_per_line
        print(f"  进度: {idx}/{total}, 已用 {elapsed_total/60:.0f}分钟, "
              f"预计剩余 {eta/60:.0f}分钟")

    # 生成Markdown
    print(f"\n[4] 生成报告...")
    generate_markdown(gt, baseline)

    total_time = time.time() - t_start
    n_success = sum(1 for v in gt["lines"].values() if v.get("julia_status") == "success")
    n_fail = sum(1 for v in gt["lines"].values() if v.get("julia_status") != "success")

    print(f"\n{'='*60}")
    print(f"  完成! 成功 {n_success} 条, 失败 {n_fail} 条")
    print(f"  耗时: {total_time/60:.1f} 分钟")
    print(f"  真实验证文档: {GT_MD_PATH}")
    print(f"  原始数据: {GT_JSON_PATH}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
