import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from pathlib import Path

data_dir = Path("data")
case_path = data_dir / "ac_dc_real_case.xlsx"

# AC loads
loads_ac = pd.read_excel(case_path, sheet_name='lumpedload', header=1)
print("=== AC Loads (lumpedload) ===")
print(loads_ac[['Bus', 'MVA', 'InService']].to_string())
print(f"\nTotal AC loads: {len(loads_ac)}")
zero_mva = (loads_ac['MVA'] == 0).sum()
pos_mva = (loads_ac['MVA'] > 0).sum()
print(f"Loads with MVA=0: {zero_mva}")
print(f"Loads with MVA>0: {pos_mva}")
print(f"\nMVA values: {sorted(loads_ac['MVA'].unique())}")

# DC loads
loads_dc = pd.read_excel(case_path, sheet_name='dclumpload')
print(f"\n=== DC Loads (dclumpload) ===")
if 'KW' in loads_dc.columns:
    print(loads_dc[['Bus', 'KW']].to_string())
    print(f"\nDC loads with KW=0: {(loads_dc['KW'] == 0).sum()}")
    print(f"KW values: {sorted(loads_dc['KW'].unique())}")
else:
    print(loads_dc.columns.tolist())
    print(loads_dc.to_string())

# Compute what Julia would use vs what Python inference uses
print("\n=== Julia vs Python inference load comparison ===")
total_load_python = 0
total_load_julia = 0
for _, r in loads_ac.iterrows():
    bus = r['Bus']
    mva = r.get('MVA', 0)
    mva = float(mva) if pd.notna(mva) else 0.0
    
    # Python inference: uses MVA directly
    total_load_python += mva
    
    # Julia: MVA column is actually kVA, converted to MW (pd_mw = mva_kva / 1000)
    # Then if pd_mw <= 0, uses default 0.1 MW (100 kW)
    pd_mw = mva / 1000.0
    if pd_mw <= 0:
        pd_mw = 0.1  # 100 kW default
    pd_kw = pd_mw * 1000.0
    total_load_julia += pd_kw

print(f"Python inference total load (MVA, used as-is): {total_load_python}")
print(f"Julia total load (kW, with 100kW default): {total_load_julia}")

# Compare per bus
print("\n=== Per-bus load comparison ===")
print(f"{'Bus':<40} {'Python(MVA)':<12} {'Julia(kW)':<12}")
for _, r in loads_ac.iterrows():
    bus = str(r['Bus'])
    mva = float(r['MVA']) if pd.notna(r['MVA']) else 0.0
    pd_mw = mva / 1000.0
    if pd_mw <= 0:
        pd_mw = 0.1
    pd_kw = pd_mw * 1000.0
    print(f"{bus:<40} {mva:<12.1f} {pd_kw:<12.1f}")
