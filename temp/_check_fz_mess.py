import json, sys
sys.path.insert(0, '.')
from validate_inference import load_data, build_network_topology

merged, line_cols, baseline, data_dir, disp_path, topo = load_data()
net = topo['_network']
fz_info = net['fault_zone_info']
mess_reachable = set(net['mess_reachable_buses'])
all_line_info = net['all_line_info']

print("=== MESS Reachable Buses ===")
for b in sorted(mess_reachable):
    print(f"  {b}")

print("\n=== Key Lines: Fault Zone Buses + MESS Reachability ===")
key_lines = ["AC_Line_5", "AC_Line_6", "AC_Line_7", "AC_Line_9", "AC_Line_10", 
             "AC_Line_11", "AC_Line_20", "AC_Line_21", "AC_Line_22", "AC_Line_23"]

# Also get endpoints
endpoints = {}
for gidx, info in all_line_info.items():
    endpoints[info['line_id']] = (info['from'], info['to'])

for lid in key_lines:
    fz = fz_info.get(lid, {})
    fz_buses = fz.get('fault_zone_buses', set())
    fz_load = fz.get('fault_zone_load', 0)
    eff_fz = fz.get('effective_fz_load', 0)
    ep = endpoints.get(lid, ('?', '?'))
    switchable = fz.get('switchable', True)
    
    mess_in_fz = fz_buses & mess_reachable
    print(f"\n{lid}: ({ep[0]} → {ep[1]})")
    print(f"  switchable={switchable}, fz_load={fz_load}, eff_fz={eff_fz}")
    print(f"  fault zone buses: {sorted(fz_buses) if fz_buses else '(empty)'}")
    print(f"  MESS reachable in FZ: {sorted(mess_in_fz) if mess_in_fz else '(none)'}")

# Check tie switch connections 
print("\n=== Tie Switch (α=0) Endpoints ===")
for gidx, info in all_line_info.items():
    if info['alpha'] == 0:
        print(f"  {info['line_id']}: ({info['from']} → {info['to']})")

# For each fault zone bus, check if a tie switch connects to it
print("\n=== Fault Zone Buses: Tie Switch Rescue ===")
tie_buses = set()
for gidx, info in all_line_info.items():
    if info['alpha'] == 0:
        tie_buses.add(info['from'])
        tie_buses.add(info['to'])

for lid in key_lines:
    fz = fz_info.get(lid, {})
    fz_buses = fz.get('fault_zone_buses', set())
    if not fz_buses:
        continue
    rescue_buses = fz_buses & tie_buses
    print(f"  {lid}: fz_buses={sorted(fz_buses)}, tie_switch_connected={sorted(rescue_buses) if rescue_buses else 'NONE'}")
