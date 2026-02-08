import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from pathlib import Path

data_dir = Path("data")
case_path = data_dir / "ac_dc_real_case.xlsx"

# Read HVCB (circuit breakers)
hvcb_df = pd.read_excel(case_path, sheet_name='hvcb')
print("=== HVCB Pairs ===")
hvcb_pairs = set()
for _, row in hvcb_df.iterrows():
    pair = (row['FromElement'], row['ToElement'])
    hvcb_pairs.add(pair)
    print(f"  {pair[0]} <-> {pair[1]}")

# Read cables
cables = pd.read_excel(case_path, sheet_name='cable')
print("\n=== AC Line Switch Flags ===")
for _, r in cables.iterrows():
    line_num = int(r['ID'].replace('Cable', ''))
    has_switch = (r['FromBus'], r['ToBus']) in hvcb_pairs
    in_service = r['InService']
    print(f"  AC_Line_{line_num:2d}: r={1 if has_switch else 0}  InService={in_service}  {r['FromBus']} -> {r['ToBus']}")

# DC lines
dc_lines = pd.read_excel(case_path, sheet_name='dcimpedance')
print("\n=== DC Line Switch Flags ===")
for idx, r in dc_lines.iterrows():
    dc_num = int(idx) + 1
    has_switch = (r['FromBus'], r['ToBus']) in hvcb_pairs
    print(f"  DC_Line_{dc_num}: r={1 if has_switch else 0}  {r['FromBus']} -> {r['ToBus']}")

# VSC converters
vsc = pd.read_excel(case_path, sheet_name='inverter')
print("\n=== VSC Switch Flags (always r=1 in code) ===")
for idx, r in vsc.iterrows():
    vsc_num = int(idx) + 1
    in_service = r.get('InService', True)
    print(f"  VSC_Line_{vsc_num}: r=1  InService={in_service}  AC={r['BusID']} DC={r.get('CZNetwork', r.get('CzNetwork', ''))}")

# Key lines with large actual improvements
print("\n=== Key Lines (from ground truth) ===")
print("Line         BC      is_crit  r  actual_impr  predicted")
key_lines = {
    'AC_Line_6':  (0.115, True, None, 0.0892, 0.0018),
    'AC_Line_7':  (0.117, True, None, 0.0207, 0.0008),
    'AC_Line_16': (0.370, True, None, 0.0500, 0.0460),
    'AC_Line_21': (0.063, False, None, 0.0814, 0.0005),
    'AC_Line_22': (0.062, False, None, 0.0273, 0.0004),
    'AC_Line_5':  (0.117, True, None, 0.0046, 0.0016),
}
for line, (bc, crit, r, actual, pred) in key_lines.items():
    line_num = int(line.split('_')[-1])
    row = cables[cables['ID'] == f'Cable{line_num}']
    if not row.empty:
        has_switch = (row.iloc[0]['FromBus'], row.iloc[0]['ToBus']) in hvcb_pairs
        r_val = 1 if has_switch else 0
    else:
        r_val = '?'
    print(f"  {line:<12} {bc:.3f}   {str(crit):<7} r={r_val}  {actual*100:.2f}%    {pred*100:.2f}%")
