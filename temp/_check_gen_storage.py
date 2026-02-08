import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from pathlib import Path

data_dir = Path("data")
case_path = data_dir / "ac_dc_real_case.xlsx"

xl = pd.ExcelFile(case_path)
print("Available sheets:", xl.sheet_names)

# Generators
try:
    gen = pd.read_excel(case_path, sheet_name='generator')
    print(f"\n=== Generators ({len(gen)}) ===")
    print(gen.to_string())
except:
    print("\n=== No 'generator' sheet ===")

# PV / solar
try:
    pv = pd.read_excel(case_path, sheet_name='pvarray')
    print(f"\n=== PV Arrays ({len(pv)}) ===")
    for col in pv.columns:
        print(f"  {col}: {pv[col].tolist()}")
except:
    print("\n=== No 'pvarray' sheet ===")

# Storage / Battery
try:
    storage = pd.read_excel(case_path, sheet_name='storageetap')
    print(f"\n=== Static Storage ({len(storage)}) ===")
    for col in storage.columns:
        print(f"  {col}: {storage[col].tolist()}")
except:
    print("\n=== No 'storageetap' sheet ===")

# Utility/source
try:
    util = pd.read_excel(case_path, sheet_name='utility')
    print(f"\n=== Utility/Source ({len(util)}) ===")
    for col in util.columns:
        print(f"  {col}: {util[col].tolist()}")
except:
    print("\n=== No 'utility' sheet ===")

# AC PV system
try:
    acpv = pd.read_excel(case_path, sheet_name='acpv_system')
    print(f"\n=== AC PV System ({len(acpv)}) ===")
    for col in acpv.columns:
        print(f"  {col}: {acpv[col].tolist()}")
except:
    print("\n=== No 'acpv_system' sheet ===")
