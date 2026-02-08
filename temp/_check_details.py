import json

with open("output/inference_result.json", encoding="utf-8") as f:
    inf = json.load(f)
with open("output/ground_truth.json", encoding="utf-8") as f:
    gt = json.load(f)

# Lines of interest
key_lines = ["AC_Line_6", "AC_Line_21", "AC_Line_22", "AC_Line_7", "AC_Line_5",
             "AC_Line_10", "AC_Line_11", "AC_Line_20", "AC_Line_23", "AC_Line_17",
             "AC_Line_8", "AC_Line_16"]

inf_map = {r["target_line"]: r for r in inf["ranked_lines"]}

print(f"{'Line':<16} {'actual%':>8} {'pred%':>8} {'between':>8} {'fz_load':>8} {'fz_imp%':>8} {'n_fault':>8} {'n_scen':>7} {'method'}")
print("-" * 120)
for lid in key_lines:
    ir = inf_map.get(lid, {})
    gr = gt["lines"].get(lid, {})
    actual = gr.get("actual_loss_improvement", 0) * 100
    pred = ir.get("loss_improvement", 0) * 100
    between = ir.get("topo_betweenness", 0)
    fz_load = ir.get("fault_zone_load_mva", 0)
    fz_imp = ir.get("fault_zone_improvement", 0) * 100
    n_fault = ir.get("n_fault", 0)
    n_scen = ir.get("n_affected_scenarios", 0) 
    method = ir.get("method_used", "")[:50]
    print(f"{lid:<16} {actual:>8.2f} {pred:>8.2f} {between:>8.3f} {fz_load:>8.0f} {fz_imp:>8.2f} {n_fault:>8} {n_scen:>7} {method}")
