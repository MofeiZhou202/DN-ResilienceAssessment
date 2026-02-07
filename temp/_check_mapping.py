import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

mc = pd.read_excel('data/mc_simulation_results_k100_clusters.xlsx', sheet_name='cluster_representatives')
tc = [f'Col_{i:02d}' for i in range(1, 49)]

# Find scenarios where AC_Line_1 (row=1) has damage (status=1)
damaged_scens = []
for cid in range(100):
    vals = mc[(mc['cluster_id']==cid) & (mc['row_in_sample']==1)][tc].values.flatten()
    if vals.sum() > 0:
        damaged_scens.append(cid)
print(f'AC_Line_1 damaged in MC clusters: {damaged_scens[:5]}...(total {len(damaged_scens)})')

cid = damaged_scens[0]
mc_vals = list(mc[(mc['cluster_id']==cid) & (mc['row_in_sample']==1)][tc].values.flatten())
print(f'MC cluster={cid}, AC_Line_1: {mc_vals}')

topo = pd.read_excel('data/topology_reconfiguration_results.xlsx', sheet_name='RollingDecisionsOriginal')
topo_vals = list(topo[topo['Scenario']==cid+1]['AC_Line_1'].values)
print(f'Topo Scenario={cid+1}, AC_Line_1: {topo_vals}')

# Also check AC_Line_26 (normally-open)
print()
cid2 = 0
mc26 = list(mc[(mc['cluster_id']==cid2) & (mc['row_in_sample']==26)][tc].values.flatten())
topo26 = list(topo[topo['Scenario']==cid2+1]['AC_Line_26'].values)
print(f'MC cluster={cid2}, AC_Line_26: {mc26}')
print(f'Topo Scenario={cid2+1}, AC_Line_26: {topo26}')

# Compare for a damaged AC_Line_26 scenario
for cid in range(100):
    vals = mc[(mc['cluster_id']==cid) & (mc['row_in_sample']==26)][tc].values.flatten()
    if vals.sum() > 0:
        tv = list(topo[topo['Scenario']==cid+1]['AC_Line_26'].values)
        mc_ones = int(vals.sum())
        topo_zeros = int(sum(1 for v in tv if v == 0))
        print(f'\nAC_Line_26 damaged in cluster={cid}: MC_ones={mc_ones}, Topo_zeros={topo_zeros}')
        print(f'  MC: {list(vals)}')
        print(f'  Topo: {tv}')
        break
