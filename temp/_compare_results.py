import json

with open("output/ground_truth.json", encoding="utf-8") as f:
    gt = json.load(f)
with open("output/inference_result.json", encoding="utf-8") as f:
    inf = json.load(f)

# Build lookup  
gt_lines = gt["lines"]  # dict: line_id -> data
inf_map = {r["target_line"]: r for r in inf["ranked_lines"]}

print(f"{'Line':<16} {'GT actual%':>10} {'Pred%':>10} {'Error':>10} {'Pred rank':>9}")
print("-" * 65)

# Sort by ground truth actual improvement (filter for success only)
gt_items = [(lid, r) for lid, r in gt_lines.items() 
            if r.get("julia_status") == "success" and "actual_loss_improvement" in r]
gt_sorted = sorted(gt_items, key=lambda x: x[1]["actual_loss_improvement"], reverse=True)

for lid, r in gt_sorted:
    actual = r["actual_loss_improvement"] * 100
    ir = inf_map.get(lid, {})
    pred = ir.get("loss_improvement", 0) * 100
    err = pred - actual
    pred_rank = ir.get("rank", "?")
    flag = ""
    if actual > 1 and pred < actual * 0.3:
        flag = "  << UNDER"
    elif actual < 0.01 and pred > 0.5:
        flag = "  << FALSE+"
    print(f"{lid:<16} {actual:>10.2f} {pred:>10.2f} {err:>+10.2f} {pred_rank:>9}{flag}")

print()
# Summary stats
errors = []
for lid, r in gt_items:
    actual = r["actual_loss_improvement"] * 100
    ir = inf_map.get(lid, {})
    pred = ir.get("loss_improvement", 0) * 100
    errors.append(abs(pred - actual))

import statistics
print(f"MAE: {statistics.mean(errors):.3f}%")
print(f"Max error: {max(errors):.3f}%")

# Top-5 ranking comparison
print("\n=== Top-5 by Ground Truth ===")
for i, (lid, r) in enumerate(gt_sorted[:5]):
    print(f"  GT #{i+1}: {lid} ({r['actual_loss_improvement']*100:.2f}%)")

print("\n=== Top-5 by Inference ===")
for r in inf["ranked_lines"][:5]:
    print(f"  Pred #{r['rank']}: {r['target_line']} ({r['loss_improvement']*100:.2f}%)")

# Rank correlation for non-zero lines
print("\n=== Spearman rank correlation (lines with actual > 0) ===")
gt_nonzero = [(lid, r["actual_loss_improvement"]) for lid, r in gt_items if r["actual_loss_improvement"] > 0.001]
pred_vals = [(lid, inf_map.get(lid, {}).get("loss_improvement", 0)) for lid, _ in gt_nonzero]
from scipy.stats import spearmanr
gt_ranks = [v for _, v in gt_nonzero]
pr_ranks = [v for _, v in pred_vals]
if len(gt_ranks) >= 3:
    corr, pval = spearmanr(gt_ranks, pr_ranks)
    print(f"  Spearman rho = {corr:.3f}, p = {pval:.4f}")
