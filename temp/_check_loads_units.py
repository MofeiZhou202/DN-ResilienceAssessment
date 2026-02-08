"""Quick check: load units and DC/VSC ground truth"""
import pandas as pd, json
from pathlib import Path

base = Path(__file__).resolve().parent.parent
xlsx = base / "data/ac_dc_real_case.xlsx"

# 1. AC loads (header=1 like validate_inference.py)
loads_ac = pd.read_excel(xlsx, sheet_name='lumpedload', header=1)
print("=== AC loads (lumpedload) ===")
print(f"Columns: {list(loads_ac.columns)}")
if 'MVA' in loads_ac.columns:
    print(loads_ac[['Bus', 'MVA']].to_string())
    total_ac = loads_ac['MVA'].sum()
    print(f"\nTotal AC load (MVA column sum): {total_ac}")
    print(f"AC load range: {loads_ac['MVA'].min()} to {loads_ac['MVA'].max()}")
else:
    print(loads_ac.head(5).to_string())

# 2. DC loads
print("\n=== DC loads (dclumpload) ===")
loads_dc = pd.read_excel(xlsx, sheet_name='dclumpload')
print(f"Columns: {list(loads_dc.columns)}")
print(loads_dc.to_string())

# 3. Ground truth for DC/VSC
print("\n=== DC/VSC Ground Truth ===")
gt = json.load(open(base / "output/ground_truth.json", encoding='utf-8'))
for line in gt:
    name = line['line_name']
    if 'DC' in name or 'VSC' in name:
        actual = line['actual_combined_improvement']
        loss = line.get('actual_loss_improvement', 0)
        over2h = line.get('actual_over2h_improvement', 0)
        print(f"  {name}: actual_combined={actual:.6f} (loss={loss:.6f}, over2h={over2h:.6f})")

# 4. Summary
print("\n=== Unit Analysis ===")
if 'MVA' in loads_ac.columns:
    vals = loads_ac['MVA'].dropna()
    print(f"AC 'MVA' column values: {sorted(vals.unique())}")
    if vals.max() > 10:
        print("  -> Values like 100, 300 are clearly kW or kVA, NOT MVA!")
        print("  -> The column name 'MVA' is misleading; actual unit is kW")
    else:
        print("  -> Values are in MVA range")
