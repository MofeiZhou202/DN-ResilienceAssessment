"""
推理层预测精度验证脚本
========================
对35条线路(26AC + 2DC + 7VSC)的加固效果进行推理预测，并可选地通过
Julia完整弹性评估流程验证预测精度。

算法三层架构:
  1. 拓扑加权比例归因 (下界) — 按 (故障时长×拓扑权重) 比例分配失负荷
  2. XGBoost分位数回归反事实 — 数据驱动的非线性补充
  3. 连通性物理模型 (上界) — 按实际网络孤立负荷计算最大潜在收益
  
集成策略:
  - 关键瓶颈线路(BC>0.08 或 孤立负荷>20%) → 使用拓扑归因(优化器缓解手段多)
  - 普通分支线路 → avg(拓扑归因, 连通性)(缓解空间有限)

使用方法：
  python validate_inference.py --lines AC_Line_19                 # 单条线路
  python validate_inference.py --lines AC_Line_19 DC_Line_1       # 多条逐个
  python validate_inference.py --lines AC_Line_19 AC_Line_6 --multi  # 同时加固
  python validate_inference.py --batch                            # 批量测试
  python validate_inference.py --lines AC_Line_19 --predict-only  # 仅预测
"""

from __future__ import annotations

import json
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ============================================================
# 网络拓扑特征计算 (方案一: 拓扑结构特征)
# ============================================================

def  build_network_topology(data_dir: Path = None) -> Dict[str, Dict]:
    """
    从 ac_dc_real_case.xlsx 构建完整AC-DC混合配电网 networkx 图, 计算拓扑结构特征。
    
    网络模型包含:
      - 26条AC线路 (cable表, 含3条常开联络开关Cable24-26)
      - 2条DC线路 (dcimpedance表)
      - 7条VSC换流器 (inverter表, 桥接AC↔DC网络)
      - 5个DC母线 (dcbus表)
      - AC负荷 (lumpedload表) + DC负荷 (dclumpload表)
      - MESS移动储能 (3台, 可在交通网络节点间移动并向孤立节点注入功率)
    
    与Julia dispatch_main.jl一致:
      - 故障矩阵35行/场景: AC(1-26) + DC(27-28) + VSC(29-35)
      - 常开线路: {24,25,26,29,31} (AC联络开关+部分VSC)
      - 节点编号: AC母线1-25, DC母线26-30
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data"
    
    case_path = data_dir / "ac_dc_real_case.xlsx"
    if not case_path.exists():
        print(f"  [警告] 未找到配网拓扑文件 {case_path}，使用默认拓扑特征")
        return {}
    
    # --- 读取完整网络数据 (与Julia ETAPImporter一致) ---
    cables = pd.read_excel(case_path, sheet_name='cable')
    loads_ac = pd.read_excel(case_path, sheet_name='lumpedload', header=1)
    
    # DC网络数据
    try:
        dc_buses = pd.read_excel(case_path, sheet_name='dcbus')
        dc_lines = pd.read_excel(case_path, sheet_name='dcimpedance')
        vsc_converters = pd.read_excel(case_path, sheet_name='inverter')
        loads_dc = pd.read_excel(case_path, sheet_name='dclumpload')
        print(f"  DC网络: {len(dc_buses)}个DC母线, {len(dc_lines)}条DC线路, {len(vsc_converters)}个VSC换流器, {len(loads_dc)}个DC负荷")
    except Exception as e:
        print(f"  [警告] 读取DC网络数据失败: {e}，仅使用AC网络")
        dc_buses = pd.DataFrame()
        dc_lines = pd.DataFrame()
        vsc_converters = pd.DataFrame()
        loads_dc = pd.DataFrame()
    
    G = nx.Graph()
    cable_map = {}   # line_id -> (from_bus, to_bus)
    all_edges = {}   # line_num -> (from_bus, to_bus), AC线路1-26
    dc_edges = {}    # dc_line_num -> (from_bus, to_bus), DC线路 (对应故障矩阵行27-28)
    vsc_edges = {}   # vsc_num -> (ac_bus, dc_bus), VSC换流器 (对应故障矩阵行29-35)
    tie_switches = {} # line_num -> (from_bus, to_bus), 联络开关(常开)
    
    # 1. 添加AC电缆 (Cable1-26, 含联络开关Cable24-26)
    for _, r in cables.iterrows():
        line_num = int(r['ID'].replace('Cable', ''))
        line_id = f'AC_Line_{line_num}'
        all_edges[line_num] = (r['FromBus'], r['ToBus'])
        cable_map[line_id] = (r['FromBus'], r['ToBus'])
        # 所有线路都加入图(常开线路后续标记, 但图结构应完整)
        G.add_edge(r['FromBus'], r['ToBus'],
                   weight=float(r['LengthValue']),
                   line_id=line_id)
        if not r['InService']:
            tie_switches[line_num] = (r['FromBus'], r['ToBus'])
    
    # 2. 添加DC母线节点
    for _, r in dc_buses.iterrows():
        bus_name = r['ID']
        G.add_node(bus_name, node_type='dc')
    
    # 3. 添加DC线路 (dcimpedance, 故障矩阵行27-28)
    for idx, r in dc_lines.iterrows():
        dc_num = int(idx) + 1  # 1-based
        from_bus = r['FromBus']
        to_bus = r['ToBus']
        dc_edges[dc_num] = (from_bus, to_bus)
        G.add_edge(from_bus, to_bus, line_id=f'DC_Line_{dc_num}')
    
    # 4. 添加VSC换流器 (inverter, 故障矩阵行29-35, 桥接AC↔DC)
    for idx, r in vsc_converters.iterrows():
        vsc_num = int(idx) + 1  # 1-based
        ac_bus = r['BusID']     # AC侧母线
        dc_bus = r['CZNetwork'] if 'CZNetwork' in r.index else r.get('CzNetwork', '')  # DC侧母线
        vsc_edges[vsc_num] = (ac_bus, dc_bus)
        G.add_edge(ac_bus, dc_bus, line_id=f'VSC_Line_{vsc_num}')
        if not r.get('InService', True):
            # VSC 29,31 (vsc_num 1,3) 是常开
            tie_switches[26 + len(dc_edges) + vsc_num] = (ac_bus, dc_bus)
    
    # 5. 给节点添加AC负荷属性
    load_on_bus = {}
    for _, r in loads_ac.iterrows():
        bus = r.get('Bus')
        mva = r.get('MVA', 0)
        if pd.notna(bus) and pd.notna(mva):
            load_on_bus[bus] = float(mva)
            if bus in G.nodes:
                G.nodes[bus]['load_mva'] = float(mva)
    
    # 6. 给节点添加DC负荷属性
    # 与Julia一致：AC负荷的MVA列实际存储kVA值(如100,300)，Julia将其作为kW使用
    # DC负荷的KW列直接就是kW值，所以直接使用即可（不除以1000）
    for _, r in loads_dc.iterrows():
        bus = r.get('Bus')
        kw = r.get('KW', 0)
        if pd.notna(bus) and pd.notna(kw):
            # DC负荷直接使用kW值，与AC负荷的MVA列(实为kVA≈kW)保持同一量级
            load_on_bus[bus] = load_on_bus.get(bus, 0) + float(kw)
            if bus in G.nodes:
                G.nodes[bus]['load_mva'] = load_on_bus[bus]
    
    sources = ['Bus_草河F27', 'Bus_石楼F12']
    
    # --- MESS移动储能配置 (与Julia dispatch_main.jl DEFAULT_MESS一致) ---
    # MESS可在交通网络6个节点间移动，向接入的电网节点注入功率
    # Transport节点 → 电网母线: 1→Bus5, 2→Bus10, 3→Bus15, 4→Bus20, 5→Bus25, 6→Bus30
    mess_config = {
        'MESS-1': {'grid_bus': 'Bus_大刀沙水厂_交流', 'discharge_kw': 1500, 'capacity_kwh': 4500},
        'MESS-2': {'grid_bus': 'Bus_大刀沙村合作经济社_交流', 'discharge_kw': 1000, 'capacity_kwh': 4000},
        'MESS-3': {'grid_bus': 'Bus_联络虚拟节点', 'discharge_kw': 500, 'capacity_kwh': 3500},  # DC node 30
    }
    # MESS可到达的所有交通节点对应的电网母线
    mess_reachable_buses = [
        'Bus_大刀沙水厂_交流',          # Transport 1 → Grid 5
        'Bus_大刀沙村合作经济社_交流',    # Transport 2 → Grid 10
        'Bus_#4组团',                   # Transport 3 → Grid 15
        'Bus_观龙村19队#1公变',          # Transport 4 → Grid 20
        'Bus_石楼镇观龙岛路灯箱变_中压',  # Transport 5 → Grid 25
        'Bus_联络虚拟节点',              # Transport 6 → Grid 30 (DC)
    ]
    
    n_total_lines = len(all_edges) + len(dc_edges) + len(vsc_edges)
    print(f"  AC-DC混合图: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边(去重后)")
    print(f"  线路总数: {n_total_lines} (AC={len(all_edges)} + DC={len(dc_edges)} + VSC={len(vsc_edges)})")
    print(f"  联络开关(常开): {sorted(tie_switches.keys())}")
    print(f"  MESS移动储能: {len(mess_config)}台, 可达{len(mess_reachable_buses)}个节点")
    
    # --- 1. Edge Betweenness Centrality (仅AC部分用于排序) ---
    ebc = nx.edge_betweenness_centrality(G)
    
    # --- 2. Isolated Load (考虑联络开关+DC/VSC转供) ---
    total_load = sum(load_on_bus.values())
    
    topo_features = {}
    for line_id in sorted(cable_map.keys(), key=lambda x: int(x.split('_')[-1])):
        u, v = cable_map[line_id]
        line_num = int(line_id.split('_')[-1])
        bc = ebc.get((u, v), ebc.get((v, u), 0))
        
        G2 = G.copy()
        if G2.has_edge(u, v):
            G2.remove_edge(u, v)
        
        # 无转供时的原始孤立负荷
        isolated_load_raw = 0.0
        for node in G.nodes:
            mva = load_on_bus.get(node, 0)
            if mva <= 0:
                continue
            connected = any(src in G2 and nx.has_path(G2, node, src) for src in sources)
            if not connected:
                isolated_load_raw += mva
        
        # 加入联络开关进行转供
        G2_reconfig = G2.copy()
        for ts_num, (ts_u, ts_v) in tie_switches.items():
            if ts_num == line_num:
                continue
            G2_reconfig.add_edge(ts_u, ts_v)
        
        # 转供后的残余孤立负荷
        isolated_load = 0.0
        for node in G.nodes:
            mva = load_on_bus.get(node, 0)
            if mva <= 0:
                continue
            if node not in G2_reconfig:
                isolated_load += mva
                continue
            connected = any(src in G2_reconfig and nx.has_path(G2_reconfig, node, src)
                           for src in sources)
            if not connected:
                isolated_load += mva
        
        iso_load_frac = isolated_load / total_load if total_load > 0 else 0.0
        
        topo_features[line_id] = {
            'betweenness_centrality': bc,
            'isolated_load_mva': isolated_load,
            'isolated_load_mva_raw': isolated_load_raw,
            'isolated_load_fraction': iso_load_frac,
            'is_critical': bc > 0.08 or iso_load_frac > 0.2,
        }
    
    # DC线路拓扑特征
    for dc_num, (u, v) in dc_edges.items():
        line_id = f'DC_Line_{dc_num}'
        bc = ebc.get((u, v), ebc.get((v, u), 0))
        G2 = G.copy()
        if G2.has_edge(u, v):
            G2.remove_edge(u, v)
        isolated_load = 0.0
        for node in G.nodes:
            mva = load_on_bus.get(node, 0)
            if mva <= 0:
                continue
            if node not in G2:
                isolated_load += mva
                continue
            connected = any(src in G2 and nx.has_path(G2, node, src) for src in sources)
            if not connected:
                isolated_load += mva
        iso_load_frac = isolated_load / total_load if total_load > 0 else 0.0
        topo_features[line_id] = {
            'betweenness_centrality': bc,
            'isolated_load_mva': isolated_load,
            'isolated_load_mva_raw': isolated_load,
            'isolated_load_fraction': iso_load_frac,
            'is_critical': bc > 0.08 or iso_load_frac > 0.2,
        }
    
    # VSC换流器拓扑特征
    for vsc_num, (u, v) in vsc_edges.items():
        line_id = f'VSC_Line_{vsc_num}'
        bc = ebc.get((u, v), ebc.get((v, u), 0))
        G2 = G.copy()
        if G2.has_edge(u, v):
            G2.remove_edge(u, v)
        isolated_load = 0.0
        for node in G.nodes:
            mva = load_on_bus.get(node, 0)
            if mva <= 0:
                continue
            if node not in G2:
                isolated_load += mva
                continue
            connected = any(src in G2 and nx.has_path(G2, node, src) for src in sources)
            if not connected:
                isolated_load += mva
        iso_load_frac = isolated_load / total_load if total_load > 0 else 0.0
        topo_features[line_id] = {
            'betweenness_centrality': bc,
            'isolated_load_mva': isolated_load,
            'isolated_load_mva_raw': isolated_load,
            'isolated_load_fraction': iso_load_frac,
            'is_critical': bc > 0.08 or iso_load_frac > 0.2,
        }
    
    # --- 读取HVCB(断路器)数据, 构建开关标志 r ---
    # 与Julia juliapowercase2jpc_tp.jl 一致:
    #   r=1 表示有断路器(可在故障隔离中打开)
    #   r=0 表示无断路器(故障会传播通过该线路)
    switch_flag = {}  # line_global_idx -> r值 (1=有开关, 0=无开关)
    hvcb_pairs = set()
    try:
        hvcb_df = pd.read_excel(case_path, sheet_name='hvcb')
        for _, row in hvcb_df.iterrows():
            hvcb_pairs.add((row['FromElement'], row['ToElement']))
        print(f"  HVCB断路器: {len(hvcb_pairs)}对")
    except Exception as e:
        print(f"  [警告] 读取HVCB数据失败: {e}，所有线路假设无开关")
    
    # AC线路开关标志
    for _, row in cables.iterrows():
        line_num = int(row['ID'].replace('Cable', ''))
        has_switch = (row['FromBus'], row['ToBus']) in hvcb_pairs
        switch_flag[line_num] = 1 if has_switch else 0
    
    # DC线路开关标志 (与Julia一致: 匹配HVCB, 通常无匹配→r=0)
    for dc_num, (u, v) in dc_edges.items():
        global_idx = 26 + dc_num  # AC26条 + DC编号
        has_switch = (u, v) in hvcb_pairs
        switch_flag[global_idx] = 1 if has_switch else 0
    
    # VSC换流器: Julia中全部视为有开关 (r=1)
    for vsc_num in vsc_edges:
        global_idx = 28 + vsc_num  # AC26 + DC2 + VSC编号
        switch_flag[global_idx] = 1
    
    switchable_lines = [k for k, v in switch_flag.items() if v == 1]
    unswitchable_lines = [k for k, v in switch_flag.items() if v == 0]
    print(f"  开关标志: {len(switchable_lines)}条有断路器, {len(unswitchable_lines)}条无断路器")
    
    # --- 构建故障传播图 (Fault Propagation Graph) ---
    # 模拟Julia three_stage_tp_model.jl 的故障隔离约束:
    #   当线路l故障时, 其两端节点进入故障区域z=1
    #   故障通过r=0(无开关)的线路传播: 若β=1且r=0, 则z传播
    #   只有r=1(有断路器)的线路可以断开(β=0)来隔离故障
    #
    # 简化实现: 构建"不可切换图"(unswitchable graph),
    # 其中只包含r=0的在服线路. 在此图中, 如果一条线路故障,
    # 故障区域 = 通过r=0线路连通的所有节点.
    
    # 首先构建 line_global_idx -> (from_bus, to_bus, alpha) 的完整映射
    all_line_info = {}  # global_idx -> {from, to, alpha, r, line_id}
    for line_num, (u, v) in all_edges.items():
        alpha = 0 if line_num in tie_switches else 1
        all_line_info[line_num] = {
            'from': u, 'to': v, 'alpha': alpha,
            'r': switch_flag.get(line_num, 0),
            'line_id': f'AC_Line_{line_num}',
        }
    for dc_num, (u, v) in dc_edges.items():
        gidx = 26 + dc_num
        all_line_info[gidx] = {
            'from': u, 'to': v, 'alpha': 1,
            'r': switch_flag.get(gidx, 0),
            'line_id': f'DC_Line_{dc_num}',
        }
    for vsc_num, (u, v) in vsc_edges.items():
        gidx = 28 + vsc_num
        alpha = 0 if gidx in tie_switches else 1
        all_line_info[gidx] = {
            'from': u, 'to': v, 'alpha': alpha,
            'r': switch_flag.get(gidx, 1),
            'line_id': f'VSC_Line_{vsc_num}',
        }
    
    # 构建运行拓扑图: 仅包含常闭线路 (alpha=1)
    # 不含联络开关/常开VSC (alpha=0), 因为这些需要MILP决策才能合上
    # 联络开关虽然可用于转供, 但受辐射约束和容量约束限制, 不一定能完全补偿
    # 在运行拓扑中的桥线路 = 无替代路径的关键线路
    G_operating = nx.Graph()
    for gidx, info in all_line_info.items():
        if info['alpha'] == 1:
            G_operating.add_edge(info['from'], info['to'], global_idx=gidx)
    for bus in load_on_bus:
        if bus not in G_operating:
            G_operating.add_node(bus)
    for s in sources:
        if s not in G_operating:
            G_operating.add_node(s)
    
    # 对每条r=0线路计算: 在运行拓扑中移除后, 哪些负荷节点与馈线断开
    # 桥线路(割边)的故障区域大, 非桥线路(环路上)的故障区域小
    fault_zone_info = {}
    all_buses_with_load = set(load_on_bus.keys())
    
    for gidx, info in all_line_info.items():
        line_id = info['line_id']
        u, v = info['from'], info['to']
        
        if info['alpha'] == 0:
            fault_zone_info[line_id] = {
                'fault_zone_buses': set(),
                'fault_zone_load': 0.0,
                'effective_fz_load': 0.0,
                'switchable': True,
            }
            continue
        
        if info['r'] == 1:
            fault_zone_info[line_id] = {
                'fault_zone_buses': set(),
                'fault_zone_load': 0.0,
                'effective_fz_load': 0.0,
                'switchable': True,
            }
            continue
        
        # r=0, alpha=1: 在运行拓扑中检查是否为桥
        G_temp = G_operating.copy()
        if G_temp.has_edge(u, v):
            G_temp.remove_edge(u, v)
        
        feeder_reachable = set()
        for s in sources:
            if s in G_temp:
                feeder_reachable.update(nx.node_connected_component(G_temp, s))
        
        fault_zone = all_buses_with_load - feeder_reachable
        fault_zone_load = sum(load_on_bus.get(bus, 0) for bus in fault_zone)
        
        # 计算有效故障区域负荷: 对有备份路径的节点打折
        # 三级备份检测:
        # 1. α=1边 (已激活的VSC/DC连接) → 90%补偿
        # 2. α=0边连接到非故障区域 (联络开关可被MILP合上) → 负荷相关折扣
        # 3. 无任何备份 → 完全暴露
        effective_fz_load = 0.0
        for bus in fault_zone:
            bus_load = load_on_bus.get(bus, 0)
            if bus_load == 0:
                continue
            
            # Level 1: α=1备份 (已激活, 如VSC2)
            # 与Julia MILP一致: α=1备份已在运行拓扑中激活
            # 如果备份是通过VSC/DC桥接(可传输大功率), 补偿率更高
            # 如果备份只是另一条AC线路, 容量可能受限
            has_alpha1_backup = bus in G_temp and G_temp.degree(bus) > 0
            if has_alpha1_backup:
                # 检查备份类型: VSC/DC连接能提供更可靠的补偿
                has_vsc_dc_backup = False
                if bus in G_temp:
                    for neighbor in G_temp.neighbors(bus):
                        edge_data = G_temp.edges[bus, neighbor]
                        eid = edge_data.get('line_id', '')
                        if eid.startswith('VSC_') or eid.startswith('DC_'):
                            has_vsc_dc_backup = True
                            break
                if has_vsc_dc_backup:
                    # VSC/DC桥接: MILP可通过VSC传输功率, 高补偿率
                    effective_fz_load += bus_load * 0.05  # 95%被补偿
                else:
                    # 另一条AC线路备份: 受容量约束, 补偿率略低
                    effective_fz_load += bus_load * 0.20  # 80%被补偿
                continue
            
            # Level 2: α=0联络开关连接到非故障区域 (需MILP激活)
            has_alpha0_rescue = False
            for gidx_chk, info_chk in all_line_info.items():
                if gidx_chk == gidx:  # 跳过目标线路本身
                    continue
                if info_chk['alpha'] == 0:
                    other_bus = None
                    if info_chk['from'] == bus:
                        other_bus = info_chk['to']
                    elif info_chk['to'] == bus:
                        other_bus = info_chk['from']
                    if other_bus and other_bus not in fault_zone:
                        has_alpha0_rescue = True
                        break
            
            if has_alpha0_rescue:
                # α=0联络开关: MILP可能合上进行转供, 但受辐射约束和容量约束限制
                # 与Julia三阶段MILP一致: α=0开关能否激活取决于全局优化
                # 保守估计: 联络开关转供成功率有限
                # 小负荷(≤150): 联络开关容量通常足够 → 60%补偿
                # 中负荷(150-300): 容量可能不足 → 30%补偿
                # 大负荷(>300): 容量严重不足 → 10%补偿
                if bus_load <= 150:
                    effective_fz_load += bus_load * 0.40
                elif bus_load <= 300:
                    effective_fz_load += bus_load * 0.70
                else:
                    effective_fz_load += bus_load * 0.90
            else:
                # 无任何备份: 完全暴露
                effective_fz_load += bus_load
        
        fault_zone_info[line_id] = {
            'fault_zone_buses': fault_zone,
            'fault_zone_load': fault_zone_load,
            'effective_fz_load': effective_fz_load,
            'switchable': False,
        }
    
    # 补充计算: 不可切换簇(r=0连通分量)的总负荷
    # 运行拓扑的per-line故障区域只反映直接断开的节点(通常1-2个)
    # 但MILP的三阶段重构优化产生级联效应, 影响范围远超直接故障区域
    # cluster_load捕获这种级联效应的上界
    G_unswitchable = nx.Graph()
    for gidx, info in all_line_info.items():
        if info['alpha'] == 1 and info['r'] == 0:
            G_unswitchable.add_edge(info['from'], info['to'], global_idx=gidx)
    
    for gidx, info in all_line_info.items():
        line_id = info['line_id']
        if info['r'] != 0 or info['alpha'] != 1:
            continue
        u, v = info['from'], info['to']
        # 找到该线路所在的r=0连通分量(不可切换簇)
        cluster_buses = set()
        if u in G_unswitchable:
            cluster_buses.update(nx.node_connected_component(G_unswitchable, u))
        if v in G_unswitchable:
            cluster_buses.update(nx.node_connected_component(G_unswitchable, v))
        cluster_load = sum(load_on_bus.get(b, 0) for b in cluster_buses)
        fault_zone_info[line_id]['cluster_buses'] = cluster_buses
        fault_zone_info[line_id]['cluster_load'] = cluster_load
        fault_zone_info[line_id]['cluster_n_buses'] = len(cluster_buses)
    
    # 输出故障区域统计
    fz_lines = [(lid, fz.get('cluster_load', 0),
                  fz.get('effective_fz_load', fz['fault_zone_load']),
                  fz['fault_zone_load'])
                for lid, fz in fault_zone_info.items()
                if fz.get('effective_fz_load', 0) > 0 or fz.get('cluster_load', 0) > 0]
    fz_lines.sort(key=lambda x: -x[2])  # 按有效FZ排序
    if fz_lines:
        print(f"  故障传播区域 (Top-{min(10, len(fz_lines))}):")
        print(f"    {'线路':<14} {'簇负荷':>6} {'有效FZ':>6} {'直接FZ':>6}")
        for lid, clust_l, eff_l, direct_l in fz_lines[:10]:
            print(f"    {lid:<14} {clust_l:>6.0f} {eff_l:>6.0f} {direct_l:>6.0f} MVA")
    else:
        print(f"  故障传播区域: 所有r=0线路故障后均有替代路径 (无隔离风险)")
    
    # 将故障区域信息添加到各线路的topo_features中
    for line_id in list(topo_features.keys()):
        if line_id.startswith('_'):
            continue
        fz = fault_zone_info.get(line_id, {})
        topo_features[line_id]['fault_zone_load'] = fz.get('fault_zone_load', 0.0)
        topo_features[line_id]['fault_zone_buses'] = fz.get('fault_zone_buses', set())
        topo_features[line_id]['fault_zone_switchable'] = fz.get('switchable', True)
        # 更新is_critical: 加入故障传播区域判断
        # 无断路器且故障区域负荷大 → 关键瓶颈 (三阶段MILP会产生大的组合效应)
        bc = topo_features[line_id].get('betweenness_centrality', 0)
        iso_frac = topo_features[line_id].get('isolated_load_fraction', 0)
        fz_load = fz.get('fault_zone_load', 0.0)
        is_unswitchable_critical = (not fz.get('switchable', True)) and fz_load > 0
        topo_features[line_id]['is_critical'] = (
            bc > 0.08 or iso_frac > 0.2 or is_unswitchable_critical
        )
    
    # 存储完整网络结构数据
    topo_features['_network'] = {
        'all_edges': all_edges,           # AC线路 {1-26: (from, to)}
        'dc_edges': dc_edges,             # DC线路 {1-2: (from, to)}
        'vsc_edges': vsc_edges,           # VSC换流器 {1-7: (ac_bus, dc_bus)}
        'load_on_bus': load_on_bus,       # 所有负荷 (AC+DC)
        'sources': sources,
        'total_load': total_load,
        'tie_switches': tie_switches,
        'mess_config': mess_config,
        'mess_reachable_buses': mess_reachable_buses,
        'switch_flag': switch_flag,       # 开关标志 {gidx: r}
        'all_line_info': all_line_info,   # 线路完整信息
        'fault_zone_info': fault_zone_info,  # 故障传播区域信息
    }
    
    return topo_features


def compute_connectivity_benefit(merged: pd.DataFrame, line_cols: List[str],
                                  target_col: str, topo_features: Dict) -> float:
    """
    基于物理模型计算连通性收益 (含AC+DC+VSC完整网络):
    对每个场景×时间步, 计算加固目标线路后减少的孤立负荷 (MVA·h)
    """
    net = topo_features.get('_network', {})
    all_edges = net.get('all_edges', {})
    dc_edges = net.get('dc_edges', {})
    vsc_edges = net.get('vsc_edges', {})
    load_on_bus = net.get('load_on_bus', {})
    sources = net.get('sources', ['Bus_草河F27', 'Bus_石楼F12'])
    
    if not all_edges or not load_on_bus:
        return 0.0
    
    target_base = target_col.replace('_Status', '')
    
    def _compute_iso(statuses):
        G_tmp = nx.Graph()
        # AC线路
        for n, (u, v) in all_edges.items():
            col = f'AC_Line_{n}'
            if statuses.get(col, 1) == 0:
                continue
            G_tmp.add_edge(u, v)
        # DC线路
        for n, (u, v) in dc_edges.items():
            col = f'DC_Line_{n}'
            if statuses.get(col, 1) == 0:
                continue
            G_tmp.add_edge(u, v)
        # VSC换流器 (桥接AC↔DC)
        for n, (u, v) in vsc_edges.items():
            col = f'VSC_Line_{n}'
            if statuses.get(col, 1) == 0:
                continue
            G_tmp.add_edge(u, v)
        for bus in load_on_bus:
            if bus not in G_tmp:
                G_tmp.add_node(bus)
        for src in sources:
            if src not in G_tmp:
                G_tmp.add_node(src)
        iso = 0.0
        for bus, mva in load_on_bus.items():
            if any(src in G_tmp and nx.has_path(G_tmp, bus, src) for src in sources):
                continue
            iso += mva
        return iso
    
    # 只对原始status列名工作(不含_Status后缀)
    raw_line_cols = [c.replace('_Status', '') if c.endswith('_Status') else c for c in line_cols]
    
    scenarios = merged.groupby('Scenario_ID')
    total_benefit = 0.0
    
    for sid, g in scenarios:
        prob = g['Probability'].iloc[0] if 'Probability' in g.columns else 1.0 / merged['Scenario_ID'].nunique()
        for _, row in g.iterrows():
            # 构造line status dict (用不带_Status后缀的列名)
            statuses = {}
            for lc in line_cols:
                base = lc.replace('_Status', '') if lc.endswith('_Status') else lc
                statuses[base] = int(row[lc])
            
            if statuses.get(target_base, 1) == 1:
                continue  # 目标线路未故障,无收益
            
            iso_actual = _compute_iso(statuses)
            statuses_cf = statuses.copy()
            statuses_cf[target_base] = 1
            iso_cf = _compute_iso(statuses_cf)
            
            total_benefit += prob * (iso_actual - iso_cf)
    
    return total_benefit


# ============================================================
# 超时节点物理模型 (Graph-Connectivity Over-2h Model)
# ============================================================

def _compute_graph_violation_probs(merged: pd.DataFrame, line_cols: List[str],
                                   all_edges: Dict, load_on_bus: Dict,
                                   sources: List[str],
                                   dc_edges: Dict = None,
                                   vsc_edges: Dict = None,
                                   mess_reachable: set = None,
                                   reinforce_cols: List[str] = None) -> Dict[str, float]:
    """
    基于图连通性的超2h违规概率计算 (含AC+DC+VSC完整网络 + MESS启发式)。
    
    对每个场景×时间步，构建含AC/DC/VSC的networkx图并跟踪每个负荷节点的连续断电时长。
    当连续断电超过2小时（≥3个连续时间步）时，标记该节点为"违规"。
    MESS可达节点：如果断电连续达到2步时MESS到达，重置连续计数（启发式）。
    
    Args:
        merged: 合并后的拓扑+调度DataFrame
        line_cols: 线路状态列名列表 (AC+DC+VSC)
        all_edges: Dict {线路编号: (起始节点, 终止节点)} — AC线路
        load_on_bus: Dict {母线名: 负荷MVA}
        sources: 电源母线名列表
        dc_edges: Dict {线路编号: (起始节点, 终止节点)} — DC线路
        vsc_edges: Dict {线路编号: (起始节点, 终止节点)} — VSC换流器
        mess_reachable: set of bus names reachable by MESS transport
        reinforce_cols: 需要加固的列名（模拟状态=1）
    
    Returns:
        Dict {母线名: 违规概率}
    """
    if dc_edges is None:
        dc_edges = {}
    if vsc_edges is None:
        vsc_edges = {}
    if mess_reachable is None:
        mess_reachable = set()
    
    load_buses = sorted(load_on_bus.keys())
    
    # 加固列名集合 (直接用列名匹配, 避免编号冲突)
    reinforce_set = set(reinforce_cols) if reinforce_cols else set()
    
    # 构建列名→边映射
    col_to_edge = {}
    for c in line_cols:
        base = c.replace('_Status', '')
        parts = base.split('_')
        try:
            num = int(parts[-1])
        except ValueError:
            continue
        if base.startswith('AC_Line'):
            if num in all_edges:
                col_to_edge[c] = all_edges[num]
        elif base.startswith('DC_Line'):
            if num in dc_edges:
                col_to_edge[c] = dc_edges[num]
        elif base.startswith('VSC_Line'):
            if num in vsc_edges:
                col_to_edge[c] = vsc_edges[num]
    
    # 每节点违规概率 = Σ(场景概率 × 是否违规)
    node_viol_prob = {bus: 0.0 for bus in load_buses}
    
    for scen_id, scen_group in merged.groupby('Scenario_ID'):
        prob = scen_group['Probability'].iloc[0] if 'Probability' in scen_group.columns \
               else 1.0 / merged['Scenario_ID'].nunique()
        
        scen_sorted = scen_group.sort_values('TimeStep')
        consecutive = {bus: 0 for bus in load_buses}
        violated = {bus: False for bus in load_buses}
        
        for _, row in scen_sorted.iterrows():
            # 构建当前时间步的网络图 (AC + DC + VSC)
            G = nx.Graph()
            for col, (u, v) in col_to_edge.items():
                status = int(row.get(col, 1))
                if col in reinforce_set:
                    status = 1  # 加固线路永不故障
                if status == 1:
                    G.add_edge(u, v)
            
            # 确保所有节点存在
            for bus in load_buses:
                if bus not in G:
                    G.add_node(bus)
            for src in sources:
                if src not in G:
                    G.add_node(src)
            
            # 检查连通性，跟踪连续断电
            for bus in load_buses:
                connected = any(src in G and nx.has_path(G, bus, src) for src in sources)
                if connected:
                    consecutive[bus] = 0
                else:
                    consecutive[bus] += 1
                    # MESS启发式: MESS可达节点在连续断电2步时MESS到达, 重置计数
                    if bus in mess_reachable and consecutive[bus] == 2:
                        consecutive[bus] = 0  # MESS提供应急供电
                    elif consecutive[bus] > 2 and not violated[bus]:
                        violated[bus] = True
        
        # 累计场景概率
        for bus in load_buses:
            if violated[bus]:
                node_viol_prob[bus] += prob
    
    return node_viol_prob


def compute_over2h_physical(merged: pd.DataFrame, line_cols: List[str],
                            baseline: Dict, topo_features: Dict,
                            reinforce_cols: List[str] = None) -> Dict:
    """
    基于图连通性 + 逐节点校准的超2h违规预测模型。
    
    物理模型方法:
    1. 对每个场景×时间步构建networkx图，跟踪每个负荷节点的连续断电
    2. 计算图模型的逐节点违规概率（基线 + 反事实）
    3. 利用Julia基线校准: alpha(N) = Julia概率(N) / 图模型概率(N)
    4. 预测反事实: Julia_cf(N) = Julia_base(N) × (Graph_cf(N) / Graph_base(N))
    
    指标: total_violation_probability = Σ violation_prob(node)
    改善率 = (基线总概率 - 反事实总概率) / 基线总概率
    
    Args:
        merged: 合并后的拓扑+调度DataFrame
        line_cols: 线路状态列名
        baseline: Dict，包含Julia基线 'violations' 列表
        topo_features: Dict，包含 '_network' 键
        reinforce_cols: 需要加固的列名（None表示仅计算基线）
    
    Returns:
        Dict: 包含逐节点概率和改善率指标
    """
    net = topo_features.get('_network', {})
    all_edges = net.get('all_edges', {})
    dc_edges = net.get('dc_edges', {})
    vsc_edges = net.get('vsc_edges', {})
    load_on_bus = net.get('load_on_bus', {})
    sources = net.get('sources', ['Bus_草河F27', 'Bus_石楼F12'])
    mess_reachable = net.get('mess_reachable_buses', set())
    
    if not all_edges or not load_on_bus:
        return {
            'over2h_improvement': 0.0,
            'total_violation_prob': 0.0,
            'n_violations': 0,
            'per_node_probs': {},
            'note': '无网络拓扑数据'
        }
    
    # 读取Julia基线逐节点违规概率
    julia_base_probs = {}
    violations = baseline.get('violations', [])
    for v in violations:
        node_name = v.get('node_name', '')
        prob = v.get('violation_probability', 0.0)
        if node_name and prob > 0:
            julia_base_probs[node_name] = prob
    
    julia_base_total = sum(julia_base_probs.values())
    julia_base_count = len(julia_base_probs)
    
    if julia_base_total <= 0:
        return {
            'over2h_improvement': 0.0,
            'total_violation_prob': 0.0,
            'n_violations': 0,
            'per_node_probs': {},
            'note': '基线无违规节点'
        }
    
    # Step 1: 计算图模型基线违规概率 (含DC+VSC+MESS)
    graph_base = _compute_graph_violation_probs(
        merged, line_cols, all_edges, load_on_bus, sources,
        dc_edges=dc_edges, vsc_edges=vsc_edges, mess_reachable=mess_reachable)
    
    if reinforce_cols is None:
        # 仅基线: 返回Julia基线统计
        return {
            'over2h_improvement': 0.0,
            'total_violation_prob': julia_base_total,
            'n_violations': julia_base_count,
            'per_node_probs': julia_base_probs,
            'baseline_total_prob': julia_base_total,
            'baseline_n_violations': julia_base_count,
        }
    
    # Step 2: 计算图模型反事实违规概率 (含DC+VSC+MESS)
    graph_cf = _compute_graph_violation_probs(
        merged, line_cols, all_edges, load_on_bus, sources,
        dc_edges=dc_edges, vsc_edges=vsc_edges, mess_reachable=mess_reachable,
        reinforce_cols=reinforce_cols)
    
    # Step 3: 逐节点校准预测
    # predicted_cf_prob(N) = julia_base_prob(N) × (graph_cf(N) / graph_base(N))
    predicted_cf = {}
    dc_nodes = {}
    
    load_buses = sorted(load_on_bus.keys())
    for bus in load_buses:
        j_base = julia_base_probs.get(bus, 0.0)
        g_base = graph_base.get(bus, 0.0)
        g_cf = graph_cf.get(bus, 0.0)
        
        if g_base > 0 and j_base > 0:
            ratio = g_cf / g_base
            predicted_cf[bus] = j_base * ratio
        elif j_base > 0 and g_base == 0:
            # Julia有违规但图模型无 — 无法预测（DC母线等），保持不变
            dc_nodes[bus] = j_base
            predicted_cf[bus] = j_base
        else:
            predicted_cf[bus] = 0.0
    
    # 处理不在load_buses中的Julia违规节点（如DC母线）
    for node_name, prob in julia_base_probs.items():
        if node_name not in predicted_cf:
            dc_nodes[node_name] = prob
            predicted_cf[node_name] = prob
    
    cf_total = sum(predicted_cf.values())
    cf_count = sum(1 for p in predicted_cf.values() if p > 0)
    
    improvement_total = (julia_base_total - cf_total) / julia_base_total if julia_base_total > 0 else 0.0
    improvement_count = (julia_base_count - cf_count) / julia_base_count if julia_base_count > 0 else 0.0
    
    return {
        'over2h_improvement': max(0.0, improvement_total),
        'total_violation_prob': cf_total,
        'n_violations': cf_count,
        'per_node_probs': {k: v for k, v in predicted_cf.items() if v > 0},
        'baseline_total_prob': julia_base_total,
        'baseline_n_violations': julia_base_count,
        'improvement_total_prob': float(improvement_total),
        'improvement_count': float(improvement_count),
    }


# ============================================================
# 数据加载与推理预测
# ============================================================

def load_data():
    """加载并合并拓扑-调度数据，返回推理所需的全部对象"""
    data_dir = PROJECT_ROOT / "data"
    
    # 查找文件
    topo_path = data_dir / "topology_reconfiguration_results.xlsx"
    disp_candidates = [
        "mess_dispatch_hourly.xlsx",
        "mess_dispatch_report.xlsx",
        "mess_dispatch_results.xlsx",
    ]
    disp_path = None
    for name in disp_candidates:
        p = data_dir / name
        if p.exists():
            try:
                xl = pd.ExcelFile(p)
                if "HourlyDetails" in xl.sheet_names:
                    disp_path = p
                    break
            except:
                pass
    
    if not topo_path.exists():
        raise FileNotFoundError(f"未找到拓扑文件: {topo_path}")
    if disp_path is None:
        raise FileNotFoundError(f"未找到含HourlyDetails的调度文件")
    
    # 加载拓扑
    topo_df = pd.read_excel(topo_path, sheet_name="RollingDecisionsOriginal")
    if 'Scenario' in topo_df.columns:
        topo_df = topo_df.rename(columns={'Scenario': 'Scenario_ID'})
    
    # 加载调度
    disp_df = pd.read_excel(disp_path, sheet_name="HourlyDetails")
    
    # 合并
    if topo_df['TimeStep'].min() == 0:
        topo_df['TimeStep'] = topo_df['TimeStep'] + 1
    
    merged = pd.merge(topo_df, disp_df, on=['Scenario_ID', 'TimeStep'], how='inner')
    
    # 识别线路列 (与run_inference_pipeline_v2.py保持一致的逻辑)
    # 先尝试检测带_Status后缀的列（已有后缀的情况）
    line_cols = [c for c in merged.columns if c.endswith('_Status')]
    if not line_cols:
        # 没有_Status后缀, 给原始线路列加后缀
        raw_lines = [c for c in topo_df.columns if c.startswith(('AC_Line_', 'DC_Line_', 'VSC_'))
                     and c not in ['Scenario_ID', 'TimeStep']]
        for col in raw_lines:
            new_name = col + '_Status'
            if col in merged.columns:
                merged = merged.rename(columns={col: new_name})
        line_cols = [c for c in merged.columns if c.endswith('_Status')]
    
    # 过滤掉非二值列（确保只保留线路状态列）
    line_cols = [c for c in line_cols if set(merged[c].unique()).issubset({0, 1, 0.0, 1.0})]
    
    # 目标变量
    loss_col = 'Load_Shed' if 'Load_Shed' in merged.columns else 'Total_Load_Loss'
    if loss_col not in merged.columns:
        for c in merged.columns:
            if 'loss' in c.lower() or 'shed' in c.lower():
                loss_col = c
                break
    merged = merged.rename(columns={loss_col: 'Total_Load_Loss'})
    
    # 读取Julia基线
    baseline = {}
    for km_name in ["mess_dispatch_results_key_metrics.json", "mess_dispatch_report_key_metrics.json"]:
        km_path = data_dir / km_name
        if km_path.exists():
            with open(km_path, 'r', encoding='utf-8') as f:
                km_data = json.load(f)
            baseline = {
                'expected_load_shed_total': km_data.get('expected_load_shed_total', 0),
                'expected_supply_ratio': km_data.get('expected_supply_ratio', 0),
                'violations': km_data.get('violations', []),
            }
            break
    
    # 计算拓扑特征
    print("  构建网络拓扑,计算结构特征...")
    topo_features = build_network_topology(data_dir)
    if topo_features:
        n_lines = sum(1 for k in topo_features if k != '_network')
        n_crit = sum(1 for k, v in topo_features.items() if k != '_network' and v.get('is_critical'))
        print(f"  拓扑特征: {n_lines}条线路, {n_crit}条关键瓶颈线路")
    
    # 加载MC故障数据 (用于准确故障计数，替代拓扑状态数据)
    # MC数据编码: 0=正常(线路完好), 1=故障(台风损坏)
    # 拓扑数据编码: 1=闭合(通电), 0=断开(故障或常开)
    # 对于常开线路(联络开关): 拓扑status=0不代表故障，需用MC数据区分
    mc_path = data_dir / "mc_simulation_results_k100_clusters.xlsx"
    if mc_path.exists():
        try:
            mc_df = pd.read_excel(mc_path, sheet_name="cluster_representatives")
            time_cols_mc = [f'Col_{i:02d}' for i in range(1, 49)]
            
            # row_in_sample → 线路列名映射
            row_to_col = {}
            for i in range(1, 27):
                row_to_col[i] = f'AC_Line_{i}_Status'
            for i in range(1, 3):
                row_to_col[26 + i] = f'DC_Line_{i}_Status'
            for i in range(1, 8):
                row_to_col[28 + i] = f'VSC_Line_{i}_Status'
            
            # 构建 mc_fault_hours: {scenario_id: {line_col: n_damage_timesteps}}
            mc_fault_hours = {}
            for cluster_id, group in mc_df.groupby('cluster_id'):
                scen_id = int(cluster_id) + 1  # cluster_id 0-based → Scenario 1-based
                fh = {}
                for _, row in group.iterrows():
                    row_num = int(row['row_in_sample'])
                    col_name = row_to_col.get(row_num)
                    if col_name:
                        n_damaged = int(sum(1 for tc in time_cols_mc if row[tc] == 1))
                        fh[col_name] = n_damaged
                mc_fault_hours[scen_id] = fh
            
            # 存入 topo_features 以便 predict_single_line 使用
            topo_features['_mc_fault_hours'] = mc_fault_hours
            
            # 统计信息
            total_mc_damage = sum(sum(v.values()) for v in mc_fault_hours.values())
            print(f"  MC故障数据: {len(mc_fault_hours)}场景, 共{total_mc_damage}个故障时间步")
        except Exception as e:
            print(f"  [警告] MC故障数据加载失败: {e}, 将使用拓扑状态数据")
    else:
        print(f"  [警告] 未找到MC故障数据文件 ({mc_path.name}), 将使用拓扑状态数据")
    
    return merged, line_cols, baseline, data_dir, disp_path, topo_features


def predict_single_line(merged: pd.DataFrame, line_cols: List[str], baseline: Dict,
                        target_line: str, topo_features: Dict = None) -> Dict:
    """
    使用拓扑结构特征 + 分位数回归预测加固单条线路的系统级改善率
    
    三层方法:
      1. 拓扑加权比例归因 (Topology-Weighted Proportional Attribution)
         — 按 (故障小时 × 拓扑权重) 比例分配失负荷, 高介数/高孤立负荷线路获得更大份额
      2. XGBoost 分位数回归反事实 (Quantile Regression Counterfactual)
         — 用拓扑特征增强的XGBoost，在90%分位数处预测，捕获极端场景效应
      3. 集成策略: 取两方法最大值（对关键线路更保守的估计）
    """
    import xgboost as xgb
    from scipy.spatial.distance import cdist
    
    if topo_features is None:
        topo_features = {}
    
    all_line_cols = sorted(line_cols)
    X = merged[all_line_cols].values.astype(float)
    y_loss = merged['Total_Load_Loss'].values.astype(float)
    has_over2h = 'Nodes_Over_2h' in merged.columns
    y_over2h = merged['Nodes_Over_2h'].values.astype(float) if has_over2h else np.zeros(len(merged))
    
    # 场景级统计
    scenario_groups = merged.groupby('Scenario_ID')
    scenario_probs = {}
    scenario_total_loss = {}
    scenario_total_over2h = {}
    scenario_fault_hours = {}
    
    for scen_id, group in scenario_groups:
        prob = group['Probability'].iloc[0] if 'Probability' in group.columns else 1.0 / merged['Scenario_ID'].nunique()
        scenario_probs[scen_id] = prob
        scenario_total_loss[scen_id] = group['Total_Load_Loss'].sum()
        scenario_total_over2h[scen_id] = group['Nodes_Over_2h'].sum() if has_over2h else 0
        fh = {}
        for lc in all_line_cols:
            fh[lc] = int((group[lc] == 0).sum())
        scenario_fault_hours[scen_id] = fh
    
    data_expected_loss = sum(scenario_probs[s] * scenario_total_loss[s] for s in scenario_probs)
    data_expected_over2h = sum(scenario_probs[s] * scenario_total_over2h[s] for s in scenario_probs)
    
    total_expected_loss = baseline.get('expected_load_shed_total', data_expected_loss)
    if total_expected_loss <= 0:
        total_expected_loss = data_expected_loss
    total_expected_over2h = data_expected_over2h
    n_scenarios = len(scenario_probs)
    
    # 找目标线路
    if target_line + '_Status' in all_line_cols:
        target_col = target_line + '_Status'
    elif target_line in all_line_cols:
        target_col = target_line
    else:
        return {'error': f"未找到线路 {target_line}，可用线路: {all_line_cols[:5]}..."}
    
    line_idx = all_line_cols.index(target_col)
    other_indices = [i for i in range(len(all_line_cols)) if i != line_idx]
    
    fault_mask = X[:, line_idx] == 0
    normal_mask = X[:, line_idx] == 1
    n_fault = fault_mask.sum()
    n_normal = normal_mask.sum()
    
    # MC物理故障数 (更准确: 只计台风物理损坏, 不含常开默认状态)
    mc_fault_hours_data = topo_features.get('_mc_fault_hours', {})
    if mc_fault_hours_data:
        n_mc_fault = sum(mc_fault_hours_data.get(s, {}).get(target_col, 0)
                         for s in mc_fault_hours_data)
    else:
        n_mc_fault = int(n_fault)
    
    if n_mc_fault == 0 and n_fault == 0:
        return {'target_line': target_line, 'n_fault': 0,
                'loss_improvement': 0.0, 'over2h_improvement': 0.0,
                'combined_improvement': 0.0, 'note': '该线路无故障'}
    if n_normal == 0:
        return {'target_line': target_line, 'n_fault': int(n_fault),
                'loss_improvement': 0.0, 'over2h_improvement': 0.0,
                'combined_improvement': 0.0, 'note': '无对照组'}
    
    fault_probs = merged.loc[fault_mask, 'Probability'].values if 'Probability' in merged.columns else np.ones(n_fault) / n_scenarios
    y_fault_loss = y_loss[fault_mask]
    y_fault_over2h = y_over2h[fault_mask]
    
    # ===== 获取目标线路拓扑特征 =====
    target_line_base = target_col.replace('_Status', '')
    target_topo = topo_features.get(target_line_base, {})
    target_bc = target_topo.get('betweenness_centrality', 0.0)
    target_iso_load = target_topo.get('isolated_load_mva', 0.0)
    target_iso_frac = target_topo.get('isolated_load_fraction', 0.0)
    target_is_critical = target_topo.get('is_critical', False)
    
    # ===== 方法1: 拓扑加权比例归因法 =====
    # 权重 = BC + IsoLoadFraction，高介数/高孤立负荷线路获得更大归因份额
    def _get_line_topo_weight(col_name):
        """获取线路的拓扑权重"""
        base = col_name.replace('_Status', '')
        ft = topo_features.get(base, {})
        bc = ft.get('betweenness_centrality', 0.0)
        iso_frac = ft.get('isolated_load_fraction', 0.0)
        # 权重 = 介数中心性 + 孤立负荷比例，最低0.01防止除零
        return max(0.01, bc + iso_frac)
    
    topo_prop_loss_reduction = 0.0
    topo_prop_over2h_reduction = 0.0
    
    # 使用MC故障数据进行归因 (如果可用)
    # MC数据直接记录台风物理损坏，比拓扑重构状态更准确:
    #   - 常闭线路: 拓扑status=0 ≈ MC damage=1 (近似一致)
    #   - 常开线路: 拓扑status=0 大部分是正常开路(非故障), MC damage=1 才是真故障
    mc_fault_hours = topo_features.get('_mc_fault_hours', {})
    use_mc = len(mc_fault_hours) > 0
    
    for scen_id in scenario_probs:
        if use_mc:
            mc_fh = mc_fault_hours.get(scen_id, {})
            target_fh = mc_fh.get(target_col, 0)
        else:
            target_fh = scenario_fault_hours[scen_id].get(target_col, 0)
        if target_fh == 0:
            continue
        
        # 拓扑加权: weighted_fault_hours = fault_hours × topo_weight
        target_weighted_fh = target_fh * _get_line_topo_weight(target_col)
        if use_mc:
            total_weighted_fh = sum(
                mc_fh.get(lc, 0) * _get_line_topo_weight(lc)
                for lc in all_line_cols
            )
        else:
            total_weighted_fh = sum(
                scenario_fault_hours[scen_id].get(lc, 0) * _get_line_topo_weight(lc)
                for lc in all_line_cols
            )
        if total_weighted_fh <= 0:
            continue
        
        frac = target_weighted_fh / total_weighted_fh
        prob = scenario_probs[scen_id]
        topo_prop_loss_reduction += prob * scenario_total_loss[scen_id] * frac
        topo_prop_over2h_reduction += prob * scenario_total_over2h[scen_id] * frac
    
    topo_prop_loss_reduction = max(0, topo_prop_loss_reduction)
    topo_prop_over2h_reduction = max(0, topo_prop_over2h_reduction)
    
    # 同时保留原始等权比例归因用于对比 (也使用MC数据)
    plain_prop_loss_reduction = 0.0
    for scen_id in scenario_probs:
        if use_mc:
            mc_fh = mc_fault_hours.get(scen_id, {})
            target_fh = mc_fh.get(target_col, 0)
        else:
            target_fh = scenario_fault_hours[scen_id].get(target_col, 0)
        if target_fh == 0:
            continue
        if use_mc:
            total_fh = sum(mc_fh.get(lc, 0) for lc in all_line_cols)
        else:
            total_fh = sum(scenario_fault_hours[scen_id].get(lc, 0) for lc in all_line_cols)
        if total_fh == 0:
            continue
        frac = target_fh / total_fh
        prob = scenario_probs[scen_id]
        plain_prop_loss_reduction += prob * scenario_total_loss[scen_id] * frac
    plain_prop_loss_reduction = max(0, plain_prop_loss_reduction)
    
    # ===== 方法2: XGBoost 分位数回归反事实 (方案二) =====
    # 特征增强: 原始35列二值状态 + 拓扑聚合特征
    # 拓扑聚合特征在样本级变化（因故障模式不同）
    bc_array = np.array([topo_features.get(c.replace('_Status', ''), {}).get('betweenness_centrality', 0.0) 
                         for c in all_line_cols])
    iso_array = np.array([topo_features.get(c.replace('_Status', ''), {}).get('isolated_load_mva', 0.0) 
                          for c in all_line_cols])
    
    # 故障指示: 1 - status (故障=1, 正常=0)
    fault_indicator = 1.0 - X  # shape: (n_samples, n_lines)
    
    # 拓扑聚合特征 (每个样本不同, 因为故障模式不同)
    sum_bc_failed = (fault_indicator * bc_array).sum(axis=1, keepdims=True)
    sum_iso_failed = (fault_indicator * iso_array).sum(axis=1, keepdims=True)
    max_bc_failed = (fault_indicator * bc_array).max(axis=1, keepdims=True)
    max_iso_failed = (fault_indicator * iso_array).max(axis=1, keepdims=True)
    n_faults = fault_indicator.sum(axis=1, keepdims=True)
    
    # 构造增强特征矩阵
    X_aug = np.hstack([
        X,                    # 原始35列二值状态
        sum_bc_failed,        # 故障线路介数中心性之和
        sum_iso_failed,       # 故障线路孤立负荷之和  
        max_bc_failed,        # 最大故障线路介数
        max_iso_failed,       # 最大故障线路孤立负荷
        n_faults,             # 故障线路数量
    ])
    
    # 训练 XGBoost 分位数回归 (90%分位数)
    xgb_q90_loss = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='reg:quantileerror',
        quantile_alpha=0.90,
        random_state=42, verbosity=0,
    )
    xgb_q90_loss.fit(X_aug, y_loss)
    
    # 同时训练均值模型做对比
    xgb_mean_loss = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, verbosity=0,
    )
    xgb_mean_loss.fit(X_aug, y_loss)
    
    # 构造反事实特征: 将目标线路设为1(不故障)
    X_cf = X.copy()
    X_cf[:, line_idx] = 1  # 加固后该线路永不故障
    fault_indicator_cf = 1.0 - X_cf
    sum_bc_cf = (fault_indicator_cf * bc_array).sum(axis=1, keepdims=True)
    sum_iso_cf = (fault_indicator_cf * iso_array).sum(axis=1, keepdims=True)
    max_bc_cf = (fault_indicator_cf * bc_array).max(axis=1, keepdims=True)
    max_iso_cf = (fault_indicator_cf * iso_array).max(axis=1, keepdims=True)
    n_faults_cf = fault_indicator_cf.sum(axis=1, keepdims=True)
    X_aug_cf = np.hstack([X_cf, sum_bc_cf, sum_iso_cf, max_bc_cf, max_iso_cf, n_faults_cf])
    
    # 预测实际值和反事实值 (只对故障样本计算差异)
    pred_actual_q90 = xgb_q90_loss.predict(X_aug[fault_mask])
    pred_cf_q90 = xgb_q90_loss.predict(X_aug_cf[fault_mask])
    pred_actual_mean = xgb_mean_loss.predict(X_aug[fault_mask])
    pred_cf_mean = xgb_mean_loss.predict(X_aug_cf[fault_mask])
    
    # Q90反事实效应
    q90_effects = pred_actual_q90 - pred_cf_q90
    q90_loss_reduction = max(0, np.sum(fault_probs * q90_effects))
    
    # 均值反事实效应
    mean_effects = pred_actual_mean - pred_cf_mean
    mean_loss_reduction = max(0, np.sum(fault_probs * mean_effects))
    
    # ===== 方法3: 连通性物理模型 (Connectivity-Based Physical Model) =====
    # 对每个场景×时间步,计算加固后减少的孤立负荷
    # 这是物理上限,实际效果介于比例归因(下界)和连通性收益(上界)之间
    conn_benefit = compute_connectivity_benefit(merged, all_line_cols, target_col, topo_features)
    conn_benefit_pct = conn_benefit / (total_expected_loss + 1e-8) * 100  # 百分比
    
    # ===== 方法4: 场景级Ridge回归 (Scenario-Level Ridge Regression) =====
    # 核心思想: 利用100个蒙特卡洛场景的实际优化结果, 通过线性回归
    # 学习每条线路故障小时数对系统总失负荷的边际贡献.
    # 优势:
    #   - 控制了其他线路同时故障的混淆效应 (proportional attribution无法做到)
    #   - 使用优化器的实际结果 (而非纯拓扑推算)
    #   - Ridge正则化防止过拟合, 对常开线路系数自然趋近于0
    ridge_loss_improvement = 0.0
    ridge_o2h_improvement = 0.0
    ridge_loss_reduction = 0.0
    ridge_o2h_reduction = 0.0
    target_beta_loss = 0.0
    target_beta_o2h = 0.0
    
    if use_mc and len(mc_fault_hours) >= 10:
        scen_ids_sorted = sorted(scenario_probs.keys())
        n_scen_reg = len(scen_ids_sorted)
        n_cols_reg = len(all_line_cols)
        
        X_reg = np.zeros((n_scen_reg, n_cols_reg))  # MC故障小时数 (场景×线路)
        y_loss_reg = np.zeros(n_scen_reg)  # 场景总失负荷
        y_o2h_reg = np.zeros(n_scen_reg)   # 场景总超时节点数
        w_reg = np.zeros(n_scen_reg)       # 场景概率
        
        for i, scen_id in enumerate(scen_ids_sorted):
            mc_fh = mc_fault_hours.get(scen_id, {})
            for j, lc in enumerate(all_line_cols):
                X_reg[i, j] = mc_fh.get(lc, 0)
            y_loss_reg[i] = scenario_total_loss[scen_id]
            y_o2h_reg[i] = scenario_total_over2h[scen_id]
            w_reg[i] = scenario_probs[scen_id]
        
        # 加截距列 (场景无故障时也可能有基础负荷)
        X_reg_int = np.hstack([X_reg, np.ones((n_scen_reg, 1))])
        n_features = X_reg_int.shape[1]
        
        # 加权Ridge回归: min Σ w_i (y_i - X_i β)^2 + α ‖β‖^2
        # 解: β = (X'WX + αI)^{-1} X'Wy  (截距列不正则化)
        alpha_ridge = 10.0
        W_diag = np.diag(w_reg)
        XtWX = X_reg_int.T @ W_diag @ X_reg_int
        reg_mat = alpha_ridge * np.eye(n_features)
        reg_mat[-1, -1] = 0.0  # 不正则化截距
        A = XtWX + reg_mat
        
        try:
            beta_loss = np.linalg.solve(A, X_reg_int.T @ W_diag @ y_loss_reg)
            beta_o2h = np.linalg.solve(A, X_reg_int.T @ W_diag @ y_o2h_reg)
            
            target_beta_loss = beta_loss[line_idx]
            target_beta_o2h = beta_o2h[line_idx]
            
            # 期望改善 = β_target × E[MC故障小时数] / 总期望失负荷
            target_expected_fh = sum(w_reg[i] * X_reg[i, line_idx] for i in range(n_scen_reg))
            
            if target_beta_loss > 0 and target_expected_fh > 0:
                ridge_loss_reduction = target_beta_loss * target_expected_fh
                ridge_loss_improvement = max(0.0, min(1.0,
                    ridge_loss_reduction / (total_expected_loss + 1e-8)))
            
            if target_beta_o2h > 0 and target_expected_fh > 0:
                ridge_o2h_reduction = target_beta_o2h * target_expected_fh
                ridge_o2h_improvement = max(0.0, min(1.0,
                    ridge_o2h_reduction / (total_expected_over2h + 1e-8)))
        except np.linalg.LinAlgError:
            pass  # 矩阵奇异, 回退到其他方法
    
    # ===== 方法5: 多线路故障边际分析模型 (Multi-line Marginal Fault Zone Model) =====
    # 
    # 核心改进: 不再仅分析单条线路故障的影响,
    # 而是在每个蒙特卡洛场景中:
    #   1. 移除所有该簇中故障的r=0线路(含目标) → 计算断连负荷A
    #   2. 仅移除非目标线路(目标加固存活)        → 计算断连负荷B
    #   3. 边际收益 = A - B = 加固目标线路减少的断连负荷
    #
    # 这自然解决了:
    #   - 干线 vs 支线区别: 干线加固后保持干线通路, 大量下游节点受益 → 高边际
    #   - 环路上的冗余线路: 移除不影响连通性 → 边际=0, 自动抑制假阳性
    #   - 多线路同时故障: 准确模拟级联断连效应
    
    fault_zone_loss_reduction = 0.0
    fault_zone_info = topo_features.get('_network', {}).get('fault_zone_info', {})
    all_line_info = topo_features.get('_network', {}).get('all_line_info', {})
    net_total_load = topo_features.get('_network', {}).get('total_load', 0.0)
    target_fz = fault_zone_info.get(target_line_base, {})
    target_fz_load = target_fz.get('fault_zone_load', 0.0)  # 运行拓扑上的直接故障区
    target_fz_buses = target_fz.get('fault_zone_buses', set())
    target_fz_switchable = target_fz.get('switchable', True)
    # 不可切换簇负荷 (r=0连通分量): 级联效应的上界
    target_cluster_load = target_fz.get('cluster_load', target_fz_load)
    target_cluster_buses = target_fz.get('cluster_buses', target_fz_buses)
    target_cluster_n = target_fz.get('cluster_n_buses', len(target_fz_buses))
    
    if target_cluster_load > 0 and not target_fz_switchable and use_mc:
        # 从_network中获取拓扑数据, 重建运行拓扑图用于边际分析
        net_info = topo_features.get('_network', {})
        sources_list = net_info.get('sources', [])
        load_on_bus_dict = net_info.get('load_on_bus', {})
        all_buses_with_load_set = set(load_on_bus_dict.keys())
        
        # 重建运行拓扑 (α=1线路)
        G_op = nx.Graph()
        line_endpoints = {}
        for gidx, info in all_line_info.items():
            line_endpoints[info['line_id']] = (info['from'], info['to'])
            if info['alpha'] == 1:
                G_op.add_edge(info['from'], info['to'])
        for bus in load_on_bus_dict:
            if bus not in G_op:
                G_op.add_node(bus)
        for s in sources_list:
            if s not in G_op:
                G_op.add_node(s)
        
        # 找出同簇的其他r=0线路
        same_cluster_cols = []
        same_cluster_line_ids = []
        for gidx, info in all_line_info.items():
            if info['line_id'] == target_line_base:
                continue
            if info['alpha'] == 0:
                continue
            if info['r'] == 0:
                other_fz = fault_zone_info.get(info['line_id'], {})
                other_cluster = other_fz.get('cluster_buses', set())
                if other_cluster & target_cluster_buses:
                    col_name = info['line_id'] + '_Status'
                    if col_name in all_line_cols:
                        same_cluster_cols.append(col_name)
                        same_cluster_line_ids.append(info['line_id'])
        
        # 目标线路端点
        target_uv = line_endpoints.get(target_line_base)
        
        # MESS可达节点集合 (用于边际负荷折扣)
        mess_reachable_set = set(net_info.get('mess_reachable_buses', []))

        # 预计算α=0线路信息 (tie switches + normally-open VSCs)
        alpha0_lines = []
        for gidx_a0, info_a0 in all_line_info.items():
            if info_a0['alpha'] == 0:
                alpha0_lines.append(info_a0)
        
        # 逐场景计算边际收益 (含MESS/联络开关折扣)
        for scen_id in scenario_probs:
            mc_fh = mc_fault_hours.get(scen_id, {})
            target_mc_fh = mc_fh.get(target_col, 0)
            if target_mc_fh == 0:
                continue
            
            # 找出该场景中同簇也故障的线路
            faulted_other_ids = []
            for cc, lid in zip(same_cluster_cols, same_cluster_line_ids):
                if mc_fh.get(cc, 0) > 0:
                    faulted_other_ids.append(lid)
            
            # 情况1: 所有故障线路(含目标)都断开 → 可达集合
            G_all_out = G_op.copy()
            if target_uv and G_all_out.has_edge(*target_uv):
                G_all_out.remove_edge(*target_uv)
            for lid in faulted_other_ids:
                uv = line_endpoints.get(lid)
                if uv and G_all_out.has_edge(*uv):
                    G_all_out.remove_edge(*uv)
            
            reachable_all = set()
            for s in sources_list:
                if s in G_all_out:
                    reachable_all.update(nx.node_connected_component(G_all_out, s))
            disconnected_all = all_buses_with_load_set - reachable_all
            
            # 情况2: 目标加固存活, 仅其他故障线路断开 → 可达集合
            G_target_in = G_op.copy()
            for lid in faulted_other_ids:
                uv = line_endpoints.get(lid)
                if uv and G_target_in.has_edge(*uv):
                    G_target_in.remove_edge(*uv)
            
            reachable_target = set()
            for s in sources_list:
                if s in G_target_in:
                    reachable_target.update(nx.node_connected_component(G_target_in, s))
            disconnected_target = all_buses_with_load_set - reachable_target
            
            # 边际断连节点 = 目标故障时断连但加固后可达的节点
            marginal_buses = disconnected_all - disconnected_target
            if not marginal_buses:
                continue
            
            # 对每个边际节点应用MESS备份折扣 + α=0桥接检测:
            #
            #   A. MESS可达 → 95%补偿 (物理储能, 独立于电网拓扑)
            #   B. α=0连接(VSC/联络) → 不折扣:
            #      MILP对α=0的主要利用方式是创建"桥接"路径, 而非单方向"救援"
            #      保活此节点 → MILP可通过α=0桥接创建新供电路径 → 级联收益
            #   C. 无备份 → 完全暴露
            effective_marginal_load = 0.0
            has_bridge_potential = False  # 是否有α=0桥接潜力
            
            for bus in marginal_buses:
                bus_load = load_on_bus_dict.get(bus, 0)
                if bus_load == 0:
                    continue
                
                # Case A: MESS可达
                if bus in mess_reachable_set:
                    effective_marginal_load += bus_load * 0.05
                    continue
                
                # Case B: 检查α=0桥接潜力
                for info_a0 in alpha0_lines:
                    if info_a0['from'] == bus or info_a0['to'] == bus:
                        a0_col = info_a0['line_id'] + '_Status'
                        if mc_fh.get(a0_col, 0) == 0:  # α=0线路未故障
                            has_bridge_potential = True
                            break
                
                # Case C: 完全暴露
                effective_marginal_load += bus_load
            
            if effective_marginal_load <= 0:
                continue
            
            marginal_frac = effective_marginal_load / (net_total_load + 1e-8)
            
            prob = scenario_probs[scen_id]
            scen_loss = scenario_total_loss[scen_id]
            time_frac = target_mc_fh / 48.0
            
            # 桥接场景加权: 有α=0桥接潜力的场景获得额外权重
            bridge_weight = 3.0 if has_bridge_potential else 1.0
            
            fault_zone_loss_reduction += prob * scen_loss * marginal_frac * time_frac * bridge_weight
        
        # MILP重构放大系数:
        # 拓扑连通性+桥接分析捕获直接和一阶间接效应
        # MILP还有二阶以上效应: MESS路由优化、多阶段时间协调、全局开关重构
        # 桥接放大(3.0x)已在per-scenario循环中处理
        reconfig_amplification = 5.0 + (target_cluster_n / 25.0) * 5.0
        fault_zone_loss_reduction *= reconfig_amplification
    
    fault_zone_improvement = max(0.0, min(1.0,
        fault_zone_loss_reduction / (total_expected_loss + 1e-8)))
    
    # ===== 集成策略 (v3): 故障传播模型 + 常开检测 + XGBoost门控 =====
    # 
    # v3 新增: 故障传播区域模型 (方法5)
    #   核心突破: 模拟Julia三阶段MILP中的故障隔离约束
    #   - 无断路器线路故障 → 故障传播到整个不可切换簇 → 簇内全部切负荷
    #   - 这个机制是之前推理层完全缺失的, 导致对AC21/22/6/7预测偏低
    #   - 原因: XGBoost/比例归因只看数据层面的边际效应, 
    #     但三阶段MILP的组合优化效应是非线性的、不连续的
    #
    # 集成策略:
    #   1. 常开线路 → 0 (不变)
    #   2. 有故障传播效应的线路 → max(topo_prop, fault_zone_model) 
    #      (fault_zone模型能捕获组合优化效应)
    #   3. 有断路器的线路 → 原始策略 (XGBoost门控 + topo_prop归因)
    #   4. 其他无故障传播效应的线路 → 原始策略
    
    # 各方法参考值 (保留用于输出对比)
    xgb_q90_improvement = max(0.0, min(1.0, q90_loss_reduction / (total_expected_loss + 1e-8)))
    xgb_mean_improvement = max(0.0, min(1.0, mean_loss_reduction / (total_expected_loss + 1e-8)))
    topo_prop_improvement = max(0.0, min(1.0, topo_prop_loss_reduction / (total_expected_loss + 1e-8)))
    plain_prop_improvement = max(0.0, min(1.0, plain_prop_loss_reduction / (total_expected_loss + 1e-8)))
    conn_improvement = max(0.0, min(1.0, conn_benefit / (total_expected_loss + 1e-8)))
    
    # Step 1: 检测常开线路 (联络开关 + 常开VSC)
    net_info = topo_features.get('_network', {})
    tie_switches = net_info.get('tie_switches', {})
    
    is_normally_open = False
    try:
        _ln = int(target_line_base.split('_')[-1])
        if target_line_base.startswith('AC_Line'):
            is_normally_open = _ln in tie_switches
        elif target_line_base.startswith('DC_Line'):
            is_normally_open = (26 + _ln) in tie_switches
        elif target_line_base.startswith('VSC_Line'):
            is_normally_open = (28 + _ln) in tie_switches
    except (ValueError, IndexError):
        pass
    
    # Step 2: 选择最佳估计方法
    #
    # 关键改进: 对无断路器(r=0)且有显著故障传播效应的线路,
    # 使用故障传播区域模型估算, 因为它能捕捉三阶段MILP的组合优化效应.
    # 对有断路器(r=1)的线路, 保持原始XGBoost门控策略.
    
    XGB_GATE_THRESHOLD = 0.015  # 1.5% — XGBoost边际效应显著性阈值
    
    if is_normally_open:
        # 常开线路: MC损坏仅意味着"不能被合上用于转供",
        # 而非"正在供电的线路断开". 加固效果为0或负.
        expected_loss_reduction = 0.0
        method_used = 'zero [NORMALLY_OPEN]'
        system_loss_improvement = 0.0
        system_over2h_improvement = 0.0
    elif not target_fz_switchable and fault_zone_improvement > 0:
        # 无断路器线路, 多线路边际分析检测到正效益: 使用故障传播模型
        # 多线路分析已准确计算了加固该线路的边际断连负荷减少量
        # 优先使用fault_zone模型(基于物理拓扑, 不受XGBoost混淆影响)
        system_loss_improvement = fault_zone_improvement
        expected_loss_reduction = system_loss_improvement * total_expected_loss
        method_used = (f'marginal_fz '
                      f'[MFZ={fault_zone_improvement:.4f}, '
                      f'XGB={xgb_mean_improvement:.4f}]')
        # Over-2h: 故障区域越大, 超时风险越高
        system_over2h_improvement = min(ridge_o2h_improvement, 0.03) if ridge_o2h_improvement > 0 else 0.0
    elif not target_fz_switchable and target_cluster_load > 0:
        # 无断路器线路, 在不可切换簇中, 但多线路边际分析结果为0
        # 说明: 加固该线路不改变网络连通性 (冗余线路或支线, 上游仍断)
        # XGBoost/topo_prop的信号是混淆效应 (同簇其他线路共故障导致的虚假相关)
        # → 强制为0, 抑制假阳性
        expected_loss_reduction = 0.0
        method_used = (f'zero [R0_NO_MARGINAL, cluster_load={target_cluster_load:.0f}, '
                      f'XGB_CONFOUNDED={xgb_mean_improvement:.4f}]')
        system_loss_improvement = 0.0
        system_over2h_improvement = 0.0
    elif xgb_mean_improvement >= XGB_GATE_THRESHOLD:
        # 有断路器线路, XGBoost检测到显著的每时间步边际效应
        # → 该线路故障确实增加了可测量的失负荷
        # → topo_prop的归因量有数据支持, 直接使用
        expected_loss_reduction = topo_prop_loss_reduction
        method_used = f'topo_prop [XGB_VALIDATED={xgb_mean_improvement:.4f}]'
        system_loss_improvement = topo_prop_improvement
        # Over-2h: 若XGBoost确认有信号, 使用回归或图模型的保守估计
        system_over2h_improvement = min(ridge_o2h_improvement, 0.03) if ridge_o2h_improvement > 0 else 0.0
    else:
        # XGBoost未检测到显著的边际效应
        # → 优化器已通过拓扑重构/MESS调度补偿了该线路故障
        # → topo_prop的归因存在混淆, 施加90%缩减
        shrinkage = 0.10
        expected_loss_reduction = topo_prop_loss_reduction * shrinkage
        method_used = f'topo_prop_shrunk×{shrinkage} [XGB_UNSUPPORTED={xgb_mean_improvement:.4f}]'
        system_loss_improvement = max(0.0, min(1.0,
            expected_loss_reduction / (total_expected_loss + 1e-8)))
        system_over2h_improvement = 0.0
    
    # Over-2h: 使用图连通性模型 + 校准 (仅作参考, 上界为3%)
    over2h_result = compute_over2h_physical(merged, all_line_cols, baseline, topo_features,
                                           reinforce_cols=[target_col])
    graph_over2h = over2h_result.get('over2h_improvement', 0.0)
    # 如果图模型显示无改善, 覆盖为0
    if graph_over2h <= 0 and system_over2h_improvement > 0:
        system_over2h_improvement = 0.0
    # 上界: 单条线路加固对超时节点的影响有限
    system_over2h_improvement = min(system_over2h_improvement, 0.03)
    
    combined_improvement = 0.6 * system_loss_improvement + 0.4 * system_over2h_improvement
    
    # 使用MC故障数据计算受影响场景数 (如果可用)
    if use_mc:
        n_affected = sum(1 for s in scenario_probs if mc_fault_hours.get(s, {}).get(target_col, 0) > 0)
    else:
        n_affected = sum(1 for s in scenario_probs if scenario_fault_hours[s].get(target_col, 0) > 0)
    
    return {
        'target_line': target_line,
        'target_col': target_col,
        'n_fault': int(n_mc_fault),  # MC物理故障数 (更准确)
        'n_fault_topo': int(n_fault),  # 拓扑状态故障数 (含常开默认状态, 仅供参考)
        'n_normal': int(n_normal),
        'n_affected_scenarios': n_affected,
        'loss_improvement': float(system_loss_improvement),
        'over2h_improvement': float(system_over2h_improvement),
        'combined_improvement': float(combined_improvement),
        'expected_loss_reduction_kwh': float(expected_loss_reduction),
        'total_expected_loss': float(total_expected_loss),
        # 各方法明细
        'plain_prop_improvement': float(plain_prop_improvement),
        'topo_prop_improvement': float(topo_prop_improvement),
        'xgb_mean_improvement': float(xgb_mean_improvement),
        'xgb_q90_improvement': float(xgb_q90_improvement),
        'conn_improvement': float(conn_improvement),
        'ridge_improvement': float(ridge_loss_improvement),
        'ridge_beta': float(target_beta_loss),
        'fault_zone_improvement': float(fault_zone_improvement),
        'fault_zone_load_mva': float(target_fz_load),
        'fault_zone_switchable': bool(target_fz_switchable),
        'is_normally_open': bool(is_normally_open),
        'method_used': method_used,
        # 拓扑特征
        'topo_betweenness': float(target_bc),
        'topo_iso_load_mva': float(target_iso_load),
        'topo_iso_load_frac': float(target_iso_frac),
        'topo_is_critical': bool(target_is_critical),
        # 超时节点详情
        'over2h_detail': over2h_result,
    }


def predict_multi_lines(merged: pd.DataFrame, line_cols: List[str], baseline: Dict,
                        target_lines: List[str], topo_features: Dict = None) -> Dict:
    """
    预测同时加固多条线路的系统级改善率 (拓扑特征 + 分位数回归版本)
    """
    import xgboost as xgb
    
    if topo_features is None:
        topo_features = {}
    
    all_line_cols = sorted(line_cols)
    X = merged[all_line_cols].values.astype(float)
    y_loss = merged['Total_Load_Loss'].values.astype(float)
    has_over2h = 'Nodes_Over_2h' in merged.columns
    y_over2h = merged['Nodes_Over_2h'].values.astype(float) if has_over2h else np.zeros(len(merged))
    
    # 解析目标列名
    target_cols = []
    for t in target_lines:
        if t + '_Status' in all_line_cols:
            target_cols.append(t + '_Status')
        elif t in all_line_cols:
            target_cols.append(t)
        else:
            print(f"  [警告] 未找到线路 {t}，跳过")
    
    if not target_cols:
        return {'error': '没有有效的目标线路'}
    
    target_indices = [all_line_cols.index(tc) for tc in target_cols]
    
    # 场景级统计
    scenario_groups = merged.groupby('Scenario_ID')
    scenario_probs = {}
    scenario_total_loss = {}
    scenario_total_over2h = {}
    scenario_fault_hours = {}
    
    for scen_id, group in scenario_groups:
        prob = group['Probability'].iloc[0] if 'Probability' in group.columns else 1.0 / merged['Scenario_ID'].nunique()
        scenario_probs[scen_id] = prob
        scenario_total_loss[scen_id] = group['Total_Load_Loss'].sum()
        scenario_total_over2h[scen_id] = group['Nodes_Over_2h'].sum() if has_over2h else 0
        fh = {}
        for lc in all_line_cols:
            fh[lc] = int((group[lc] == 0).sum())
        scenario_fault_hours[scen_id] = fh
    
    data_expected_loss = sum(scenario_probs[s] * scenario_total_loss[s] for s in scenario_probs)
    data_expected_over2h = sum(scenario_probs[s] * scenario_total_over2h[s] for s in scenario_probs)
    total_expected_loss = baseline.get('expected_load_shed_total', data_expected_loss)
    if total_expected_loss <= 0:
        total_expected_loss = data_expected_loss
    total_expected_over2h = data_expected_over2h
    n_scenarios = len(scenario_probs)
    
    # 故障/对照掩码
    any_fault_mask = np.zeros(len(X), dtype=bool)
    for idx in target_indices:
        any_fault_mask |= (X[:, idx] == 0)
    all_normal_mask = np.ones(len(X), dtype=bool)
    for idx in target_indices:
        all_normal_mask &= (X[:, idx] == 1)
    
    n_fault = any_fault_mask.sum()
    n_normal = all_normal_mask.sum()
    
    if n_fault == 0 or n_normal == 0:
        return {
            'target_lines': target_lines,
            'loss_improvement': 0.0, 'over2h_improvement': 0.0,
            'combined_improvement': 0.0,
            'note': '无足够的故障/对照样本',
        }
    
    fault_probs = merged.loc[any_fault_mask, 'Probability'].values if 'Probability' in merged.columns else np.ones(n_fault) / n_scenarios
    
    # ===== 方法1: 拓扑加权比例归因法 (总量+按线路分解) =====
    def _get_line_topo_weight(col_name):
        base = col_name.replace('_Status', '')
        ft = topo_features.get(base, {})
        bc = ft.get('betweenness_centrality', 0.0)
        iso_frac = ft.get('isolated_load_fraction', 0.0)
        return max(0.01, bc + iso_frac)
    
    topo_prop_loss_reduction = 0.0
    topo_prop_over2h_reduction = 0.0
    # 按线路分解 topo_prop 贡献
    per_line_topo_loss = {tc: 0.0 for tc in target_cols}
    
    for scen_id in scenario_probs:
        target_fh = sum(scenario_fault_hours[scen_id].get(tc, 0) for tc in target_cols)
        if target_fh == 0:
            continue
        
        total_weighted_fh = sum(
            scenario_fault_hours[scen_id].get(lc, 0) * _get_line_topo_weight(lc)
            for lc in all_line_cols
        )
        if total_weighted_fh <= 0:
            continue
        
        prob = scenario_probs[scen_id]
        scen_loss = scenario_total_loss[scen_id]
        scen_over2h = scenario_total_over2h[scen_id]
        
        total_frac = 0.0
        for tc in target_cols:
            fh_i = scenario_fault_hours[scen_id].get(tc, 0)
            w_i = _get_line_topo_weight(tc)
            frac_i = (fh_i * w_i) / total_weighted_fh
            per_line_topo_loss[tc] += prob * scen_loss * frac_i
            total_frac += frac_i
        
        topo_prop_loss_reduction += prob * scen_loss * total_frac
        topo_prop_over2h_reduction += prob * scen_over2h * total_frac
    
    topo_prop_loss_reduction = max(0, topo_prop_loss_reduction)
    topo_prop_over2h_reduction = max(0, topo_prop_over2h_reduction)
    per_line_topo_loss = {tc: max(0, v) for tc, v in per_line_topo_loss.items()}
    
    # ===== 方法2: XGBoost 分位数回归反事实 =====
    bc_array = np.array([topo_features.get(c.replace('_Status', ''), {}).get('betweenness_centrality', 0.0) 
                         for c in all_line_cols])
    iso_array = np.array([topo_features.get(c.replace('_Status', ''), {}).get('isolated_load_mva', 0.0) 
                          for c in all_line_cols])
    
    fault_indicator = 1.0 - X
    sum_bc_failed = (fault_indicator * bc_array).sum(axis=1, keepdims=True)
    sum_iso_failed = (fault_indicator * iso_array).sum(axis=1, keepdims=True)
    max_bc_failed = (fault_indicator * bc_array).max(axis=1, keepdims=True)
    max_iso_failed = (fault_indicator * iso_array).max(axis=1, keepdims=True)
    n_faults = fault_indicator.sum(axis=1, keepdims=True)
    X_aug = np.hstack([X, sum_bc_failed, sum_iso_failed, max_bc_failed, max_iso_failed, n_faults])
    
    xgb_q90_loss = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='reg:quantileerror', quantile_alpha=0.90,
        random_state=42, verbosity=0,
    )
    xgb_q90_loss.fit(X_aug, y_loss)
    
    xgb_mean_loss = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, verbosity=0,
    )
    xgb_mean_loss.fit(X_aug, y_loss)
    
    # 反事实: 所有目标线路设为1
    X_cf = X.copy()
    for idx in target_indices:
        X_cf[:, idx] = 1
    fault_indicator_cf = 1.0 - X_cf
    sum_bc_cf = (fault_indicator_cf * bc_array).sum(axis=1, keepdims=True)
    sum_iso_cf = (fault_indicator_cf * iso_array).sum(axis=1, keepdims=True)
    max_bc_cf = (fault_indicator_cf * bc_array).max(axis=1, keepdims=True)
    max_iso_cf = (fault_indicator_cf * iso_array).max(axis=1, keepdims=True)
    n_faults_cf = fault_indicator_cf.sum(axis=1, keepdims=True)
    X_aug_cf = np.hstack([X_cf, sum_bc_cf, sum_iso_cf, max_bc_cf, max_iso_cf, n_faults_cf])
    
    q90_effects = xgb_q90_loss.predict(X_aug[any_fault_mask]) - xgb_q90_loss.predict(X_aug_cf[any_fault_mask])
    mean_effects = xgb_mean_loss.predict(X_aug[any_fault_mask]) - xgb_mean_loss.predict(X_aug_cf[any_fault_mask])
    q90_loss_reduction = max(0, np.sum(fault_probs * q90_effects))
    mean_loss_reduction = max(0, np.sum(fault_probs * mean_effects))
    
    # ===== 方法3: 多线路连通性物理模型 (按线路分解) =====
    # 对每条目标线路,计算单独加固该线路的连通性收益
    conn_total_benefit = 0.0
    per_line_conn_benefit = {tc: 0.0 for tc in target_cols}
    net = topo_features.get('_network', {})
    all_edges_net = net.get('all_edges', {})
    load_on_bus = net.get('load_on_bus', {})
    sources_net = net.get('sources', ['Bus_草河F27', 'Bus_石楼F12'])
    
    if all_edges_net and load_on_bus:
        raw_target_bases = [tc.replace('_Status', '') if tc.endswith('_Status') else tc for tc in target_cols]
        tc_to_base = {tc: tc.replace('_Status', '') if tc.endswith('_Status') else tc for tc in target_cols}
        scenarios_g = merged.groupby('Scenario_ID')
        
        for sid, g in scenarios_g:
            prob = g['Probability'].iloc[0] if 'Probability' in g.columns else 1.0 / merged['Scenario_ID'].nunique()
            for _, row in g.iterrows():
                statuses = {}
                for lc in all_line_cols:
                    base = lc.replace('_Status', '') if lc.endswith('_Status') else lc
                    statuses[base] = int(row[lc])
                
                # 任一目标线路故障才有收益
                faulted_targets = [tc for tc in target_cols if statuses.get(tc_to_base[tc], 1) == 0]
                if not faulted_targets:
                    continue
                
                def _build_iso(sts):
                    G_t = nx.Graph()
                    # AC线路
                    for n, (u, v) in all_edges_net.items():
                        col = f'AC_Line_{n}'
                        if sts.get(col, 1) == 0:
                            continue
                        G_t.add_edge(u, v)
                    # DC线路
                    dc_edges_net = net.get('dc_edges', {})
                    for n, (u, v) in dc_edges_net.items():
                        col = f'DC_Line_{n}'
                        if sts.get(col, 1) == 0:
                            continue
                        G_t.add_edge(u, v)
                    # VSC换流器
                    vsc_edges_net = net.get('vsc_edges', {})
                    for n, (u, v) in vsc_edges_net.items():
                        col = f'VSC_Line_{n}'
                        if sts.get(col, 1) == 0:
                            continue
                        G_t.add_edge(u, v)
                    for bus in load_on_bus:
                        if bus not in G_t: G_t.add_node(bus)
                    for src in sources_net:
                        if src not in G_t: G_t.add_node(src)
                    iso = 0.0
                    for bus, mva in load_on_bus.items():
                        if any(src in G_t and nx.has_path(G_t, bus, src) for src in sources_net):
                            continue
                        iso += mva
                    return iso
                
                iso_actual = _build_iso(statuses)
                
                # 按线路分解: 每条故障的目标线路单独修复的连通性收益
                for tc in faulted_targets:
                    tb = tc_to_base[tc]
                    sts_single_fix = statuses.copy()
                    sts_single_fix[tb] = 1
                    iso_single_fix = _build_iso(sts_single_fix)
                    per_line_conn_benefit[tc] += prob * (iso_actual - iso_single_fix)
        
        conn_total_benefit = sum(per_line_conn_benefit.values())
    
    conn_improvement = max(0.0, min(1.0, conn_total_benefit / (total_expected_loss + 1e-8)))
    
    # ===== 集成: 按线路分别应用 is_critical 判据后求和 =====
    # 关键瓶颈线路(高BC骨干): 优化器有大量缓解手段 → 连通性模型严重高估 → 用topo_prop
    # 普通分支线路(低BC末梢): 优化器缓解空间有限 → 实际在topo_prop和conn之间 → 取平均
    
    expected_loss_reduction = 0.0
    for tc in target_cols:
        line_name = tc.replace('_Status', '') if tc.endswith('_Status') else tc
        is_crit = topo_features.get(line_name, {}).get('is_critical', False)
        tp_i = per_line_topo_loss.get(tc, 0.0)
        cn_i = per_line_conn_benefit.get(tc, 0.0)
        if is_crit:
            # 关键瓶颈线路 → 使用 topo_prop (连通性过估)
            expected_loss_reduction += tp_i
        else:
            # 普通分支线路 → avg(topo_prop, connectivity) (实际在两者之间)
            expected_loss_reduction += (tp_i + cn_i) / 2.0
    
    system_loss_improvement = max(0.0, min(1.0, expected_loss_reduction / (total_expected_loss + 1e-8)))
    
    # Over-2h: 基于图连通性的物理模型 + 逐节点校准
    over2h_result = compute_over2h_physical(merged, all_line_cols, baseline, topo_features,
                                           reinforce_cols=target_cols)
    system_over2h_improvement = over2h_result['over2h_improvement']
    combined_improvement = 0.6 * system_loss_improvement + 0.4 * system_over2h_improvement
    
    n_affected = sum(1 for s in scenario_probs if any(scenario_fault_hours[s].get(tc, 0) > 0 for tc in target_cols))
    
    xgb_q90_improvement = max(0.0, min(1.0, q90_loss_reduction / (total_expected_loss + 1e-8)))
    xgb_mean_improvement = max(0.0, min(1.0, mean_loss_reduction / (total_expected_loss + 1e-8)))
    topo_prop_improvement = max(0.0, min(1.0, topo_prop_loss_reduction / (total_expected_loss + 1e-8)))
    
    return {
        'target_lines': target_lines,
        'target_cols': target_cols,
        'n_fault': int(n_fault),
        'n_normal': int(n_normal),
        'n_affected_scenarios': n_affected,
        'loss_improvement': float(system_loss_improvement),
        'over2h_improvement': float(system_over2h_improvement),
        'combined_improvement': float(combined_improvement),
        'expected_loss_reduction_kwh': float(expected_loss_reduction),
        'total_expected_loss': float(total_expected_loss),
        'topo_prop_improvement': float(topo_prop_improvement),
        'xgb_mean_improvement': float(xgb_mean_improvement),
        'xgb_q90_improvement': float(xgb_q90_improvement),
        'conn_improvement': float(conn_improvement),
        # 超时节点详情
        'over2h_detail': over2h_result,
    }


# ============================================================
# Julia验证
# ============================================================

def run_julia_verification(target_lines: List[str], data_dir: Path) -> Dict:
    """
    调用Julia完整弹性评估流程验证加固效果

    流程: 创建反事实MC数据 → 阶段划分 → 拓扑重构 → MESS调度 → 提取指标
    """
    import run_inference_pipeline_v2 as pipeline
    
    mc_path = data_dir / "mc_simulation_results_k100_clusters.xlsx"
    case_path = data_dir / "ac_dc_real_case.xlsx"
    output_dir = PROJECT_ROOT / "output"
    
    disp_candidates = [
        "mess_dispatch_hourly.xlsx",
        "mess_dispatch_report.xlsx",
        "mess_dispatch_results.xlsx",
    ]
    disp_path = None
    for name in disp_candidates:
        p = data_dir / name
        if p.exists():
            disp_path = p
            break
    
    if not mc_path.exists():
        return {'status': 'failed', 'error': f'未找到MC数据文件: {mc_path}'}
    if not case_path.exists():
        return {'status': 'failed', 'error': f'未找到配电网数据文件: {case_path}'}
    
    # 添加_Status后缀（Julia验证函数需要）
    target_lines_with_status = []
    for t in target_lines:
        if not t.endswith('_Status'):
            target_lines_with_status.append(t + '_Status')
        else:
            target_lines_with_status.append(t)
    
    result = pipeline._verify_counterfactual_with_julia(
        original_mc_path=str(mc_path),
        original_dispatch_path=str(disp_path) if disp_path else '',
        case_path=str(case_path),
        target_lines=target_lines_with_status,
        predicted_loss_improvement=0,
        predicted_over2h_improvement=0,
        output_dir=output_dir,
    )
    
    return result


# ============================================================
# 主控逻辑
# ============================================================

def validate_single(target_line: str, merged, line_cols, baseline, data_dir, predict_only=False, topo_features=None):
    """验证单条线路加固"""
    print(f"\n{'='*60}")
    print(f"  验证加固线路: {target_line}")
    print(f"{'='*60}")
    
    # Step 1: 推理预测
    print(f"\n  [1/2] 推理层预测 (拓扑特征+分位数回归)...")
    pred = predict_single_line(merged, line_cols, baseline, target_line, topo_features)
    
    if 'error' in pred:
        print(f"  错误: {pred['error']}")
        return pred
    
    print(f"    故障样本: {pred['n_fault']}/{len(merged)}")
    print(f"    受影响场景: {pred.get('n_affected_scenarios', '?')}/100")
    switch_tag = '★关键瓶颈' if pred.get('topo_is_critical') else '普通分支'
    print(f"    {switch_tag} (BC={pred.get('topo_betweenness', 0):.3f}, IsoLoad={pred.get('topo_iso_load_mva', 0):.0f}MVA)")
    print(f"    方法明细:")
    print(f"      等权比例归因:      {pred.get('plain_prop_improvement', 0):.2%}")
    print(f"      拓扑加权归因:      {pred.get('topo_prop_improvement', 0):.2%}")
    print(f"      连通性物理模型:    {pred.get('conn_improvement', 0):.2%}")
    print(f"      XGBoost均值:       {pred.get('xgb_mean_improvement', 0):.2%}")
    print(f"      XGBoost Q90:       {pred.get('xgb_q90_improvement', 0):.2%}")
    print(f"    → 最终预测: {pred['loss_improvement']:.2%} ({pred.get('method_used', '?')})")
    
    if predict_only:
        print(f"\n  [2/2] 跳过Julia验证 (--predict-only)")
        return {**pred, 'status': 'predict_only'}
    
    # Step 2: Julia验证
    print(f"\n  [2/2] Julia完整流程验证 (预计5-15分钟)...")
    t0 = time.time()
    julia_result = run_julia_verification([target_line], data_dir)
    elapsed = time.time() - t0
    
    if julia_result.get('status') != 'validated':
        print(f"  Julia验证失败: {julia_result.get('error', '未知错误')}")
        return {**pred, 'julia_status': 'failed', 'julia_error': julia_result.get('error')}
    
    actual_loss_imp = julia_result['loss_improvement_actual']
    actual_over2h_imp = julia_result['over2h_improvement_actual']
    
    loss_error = abs(pred['loss_improvement'] - actual_loss_imp)
    over2h_error = abs(pred['over2h_improvement'] - actual_over2h_imp)
    
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  验证结果: {target_line:<37s} │")
    print(f"  ├──────────┬──────────┬──────────┬────────────────┤")
    print(f"  │   指标   │  预测值  │  真实值  │    预测误差    │")
    print(f"  ├──────────┼──────────┼──────────┼────────────────┤")
    print(f"  │ 失负荷   │  {pred['loss_improvement']:6.2%}  │  {actual_loss_imp:6.2%}  │  {loss_error:6.2%}  {'✓' if loss_error < 0.02 else '△' if loss_error < 0.05 else '✗'}      │")
    print(f"  │ 超时节点 │  {pred['over2h_improvement']:6.2%}  │  {actual_over2h_imp:6.2%}  │  {over2h_error:6.2%}  {'✓' if over2h_error < 0.02 else '△' if over2h_error < 0.05 else '✗'}      │")
    print(f"  │ 供电率   │    -     │  +{julia_result.get('supply_ratio_improvement', 0):5.2%} │       -        │")
    print(f"  ├──────────┼──────────┼──────────┼────────────────┤")
    print(f"  │ 期望失负荷: {julia_result['loss_original']:.1f} → {julia_result['loss_counterfactual']:.1f} kW·h")
    print(f"  │ 耗时: {elapsed:.0f}秒")
    print(f"  └─────────────────────────────────────────────────┘")
    
    return {
        **pred,
        'status': 'validated',
        'actual_loss_improvement': float(actual_loss_imp),
        'actual_over2h_improvement': float(actual_over2h_imp),
        'loss_error': float(loss_error),
        'over2h_error': float(over2h_error),
        'loss_original': julia_result['loss_original'],
        'loss_counterfactual': julia_result['loss_counterfactual'],
        'supply_ratio_original': julia_result.get('supply_ratio_original', 0),
        'supply_ratio_counterfactual': julia_result.get('supply_ratio_counterfactual', 0),
        'supply_ratio_improvement': julia_result.get('supply_ratio_improvement', 0),
        'elapsed_seconds': elapsed,
    }


def validate_multi(target_lines: List[str], merged, line_cols, baseline, data_dir, predict_only=False, topo_features=None):
    """验证同时加固多条线路"""
    lines_str = ' + '.join(target_lines)
    print(f"\n{'='*60}")
    print(f"  验证同时加固: {lines_str}")
    print(f"{'='*60}")
    
    # Step 1: 推理预测
    print(f"\n  [1/2] 推理层预测（多线路联合, 拓扑特征+分位数回归）...")
    pred = predict_multi_lines(merged, line_cols, baseline, target_lines, topo_features)
    
    if 'error' in pred:
        print(f"  错误: {pred['error']}")
        return pred
    
    print(f"    故障样本(任一故障): {pred['n_fault']}/{len(merged)}")
    print(f"    对照样本(全部正常): {pred['n_normal']}/{len(merged)}")
    print(f"    受影响场景: {pred.get('n_affected_scenarios', '?')}/100")
    print(f"    方法明细:")
    print(f"      拓扑加权归因:   {pred.get('topo_prop_improvement', 0):.2%}")
    print(f"      XGBoost均值:    {pred.get('xgb_mean_improvement', 0):.2%}")
    print(f"      XGBoost Q90:    {pred.get('xgb_q90_improvement', 0):.2%}")
    print(f"      连通性物理模型: {pred.get('conn_improvement', 0):.2%}")
    print(f"    → 最终预测失负荷改善: {pred['loss_improvement']:.2%}")
    print(f"    预测超时改善: {pred['over2h_improvement']:.2%}")
    
    if predict_only:
        print(f"\n  [2/2] 跳过Julia验证 (--predict-only)")
        return {**pred, 'status': 'predict_only'}
    
    # Step 2: Julia验证
    print(f"\n  [2/2] Julia完整流程验证 (预计5-15分钟)...")
    t0 = time.time()
    julia_result = run_julia_verification(target_lines, data_dir)
    elapsed = time.time() - t0
    
    if julia_result.get('status') != 'validated':
        print(f"  Julia验证失败: {julia_result.get('error', '未知错误')}")
        return {**pred, 'julia_status': 'failed', 'julia_error': julia_result.get('error')}
    
    actual_loss_imp = julia_result['loss_improvement_actual']
    actual_over2h_imp = julia_result['over2h_improvement_actual']
    
    loss_error = abs(pred['loss_improvement'] - actual_loss_imp)
    over2h_error = abs(pred['over2h_improvement'] - actual_over2h_imp)
    
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  验证结果: {lines_str:<37s} │")
    print(f"  ├──────────┬──────────┬──────────┬────────────────┤")
    print(f"  │   指标   │  预测值  │  真实值  │    预测误差    │")
    print(f"  ├──────────┼──────────┼──────────┼────────────────┤")
    print(f"  │ 失负荷   │  {pred['loss_improvement']:6.2%}  │  {actual_loss_imp:6.2%}  │  {loss_error:6.2%}  {'✓' if loss_error < 0.02 else '△' if loss_error < 0.05 else '✗'}      │")
    print(f"  │ 超时节点 │  {pred['over2h_improvement']:6.2%}  │  {actual_over2h_imp:6.2%}  │  {over2h_error:6.2%}  {'✓' if over2h_error < 0.02 else '△' if over2h_error < 0.05 else '✗'}      │")
    print(f"  │ 供电率   │    -     │  +{julia_result.get('supply_ratio_improvement', 0):5.2%} │       -        │")
    print(f"  ├──────────┼──────────┼──────────┼────────────────┤")
    print(f"  │ 期望失负荷: {julia_result['loss_original']:.1f} → {julia_result['loss_counterfactual']:.1f} kW·h")
    print(f"  │ 耗时: {elapsed:.0f}秒")
    print(f"  └─────────────────────────────────────────────────┘")
    
    return {
        **pred,
        'status': 'validated',
        'actual_loss_improvement': float(actual_loss_imp),
        'actual_over2h_improvement': float(actual_over2h_imp),
        'loss_error': float(loss_error),
        'over2h_error': float(over2h_error),
        'loss_original': julia_result['loss_original'],
        'loss_counterfactual': julia_result['loss_counterfactual'],
        'supply_ratio_improvement': julia_result.get('supply_ratio_improvement', 0),
        'elapsed_seconds': elapsed,
    }


def print_summary_table(results: List[Dict]):
    """打印汇总对比表格"""
    print(f"\n\n{'='*90}")
    print(f"  推理层预测精度 - 汇总对比表")
    print(f"{'='*90}")
    
    # 表头
    print(f"\n  ┌{'─'*30}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┐")
    print(f"  │ {'加固方案':<28s} │ {'预测':^8s} │ {'真实':^8s} │ {'误差':^8s} │ {'预测':^8s} │ {'真实':^8s} │ {'误差':^8s} │")
    print(f"  │ {'':28s} │ {'失负荷':^8s} │ {'失负荷':^8s} │ {'失负荷':^8s} │ {'超时':^8s} │ {'超时':^8s} │ {'超时':^8s} │")
    print(f"  ├{'─'*30}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┤")
    
    total_loss_error = 0
    total_over2h_error = 0
    n_validated = 0
    
    for r in results:
        # 方案名
        if 'target_lines' in r:
            name = '+'.join([t.replace('AC_Line_', 'ACL').replace('DC_Line_', 'DCL').replace('VSC_Line_', 'VSC') for t in r['target_lines']])
        else:
            name = r.get('target_line', '?').replace('AC_Line_', 'ACL').replace('DC_Line_', 'DCL').replace('VSC_Line_', 'VSC')
        
        if len(name) > 28:
            name = name[:25] + '...'
        
        pred_loss = r.get('loss_improvement', 0) * 100
        pred_over2h = r.get('over2h_improvement', 0) * 100
        
        if r.get('status') == 'validated':
            actual_loss = r['actual_loss_improvement'] * 100
            actual_over2h = r['actual_over2h_improvement'] * 100
            loss_err = r['loss_error'] * 100
            over2h_err = r['over2h_error'] * 100
            
            loss_mark = '✓' if loss_err < 2 else '△' if loss_err < 5 else '✗'
            over2h_mark = '✓' if over2h_err < 2 else '△' if over2h_err < 5 else '✗'
            
            print(f"  │ {name:<28s} │ {pred_loss:6.2f}%  │ {actual_loss:6.2f}%  │ {loss_err:5.2f}%{loss_mark} │ {pred_over2h:6.2f}%  │ {actual_over2h:6.2f}%  │ {over2h_err:5.2f}%{over2h_mark} │")
            
            total_loss_error += loss_err
            total_over2h_error += over2h_err
            n_validated += 1
        elif r.get('status') == 'predict_only':
            print(f"  │ {name:<28s} │ {pred_loss:6.2f}%  │  待验证 │   -    │ {pred_over2h:6.2f}%  │  待验证 │   -    │")
        else:
            note = r.get('note', r.get('error', '失败'))
            print(f"  │ {name:<28s} │ {pred_loss:6.2f}%  │ {'失败':^8s} │ {'N/A':^8s} │ {pred_over2h:6.2f}%  │ {'失败':^8s} │ {'N/A':^8s} │")
    
    print(f"  └{'─'*30}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┘")
    
    if n_validated > 0:
        avg_loss_err = total_loss_error / n_validated
        avg_over2h_err = total_over2h_error / n_validated
        print(f"\n  平均预测误差:  失负荷 {avg_loss_err:.2f}%  |  超时 {avg_over2h_err:.2f}%")
        print(f"  验证方案数: {n_validated}")
        print(f"  精度评级: ", end='')
        max_err = max(avg_loss_err, avg_over2h_err)
        if max_err < 2:
            print("★★★ 优秀 (平均误差<2%)")
        elif max_err < 5:
            print("★★ 良好 (平均误差<5%)")
        elif max_err < 10:
            print("★ 一般 (平均误差<10%)")
        else:
            print("需要改进 (平均误差≥10%)")
    
    print(f"\n  说明: ✓ 误差<2%  △ 误差<5%  ✗ 误差≥5%")


def main():
    parser = argparse.ArgumentParser(
        description="推理层预测精度验证工具 - 对比推理预测与Julia真实计算结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 测试加固单条线路
  python validate_inference.py --lines AC_Line_19

  # 测试同时加固多条线路
  python validate_inference.py --lines AC_Line_19 AC_Line_6 --multi

  # 批量测试（默认Top3逐个 + 两两组合 + 全部）
  python validate_inference.py --batch

  # 自定义批量测试线路
  python validate_inference.py --batch --lines AC_Line_19 AC_Line_6 AC_Line_16

  # 只做预测不跑Julia（快速查看预测值）
  python validate_inference.py --lines AC_Line_19 AC_Line_6 --predict-only

  # 同时测试单独加固和组合加固
  python validate_inference.py --lines AC_Line_19 AC_Line_6 --both
        """
    )
    parser.add_argument('--lines', '-l', nargs='+', default=None,
                        help='要加固的线路名称（不含_Status后缀），如: AC_Line_19 AC_Line_6')
    parser.add_argument('--multi', '-m', action='store_true',
                        help='将所有指定线路作为一组同时加固（而非逐个测试）')
    parser.add_argument('--both', '-b', action='store_true',
                        help='同时测试逐个加固和组合加固')
    parser.add_argument('--batch', action='store_true',
                        help='批量测试模式：逐个测试 + 两两组合 + 全部组合')
    parser.add_argument('--predict-only', '-p', action='store_true',
                        help='只做推理预测，不运行Julia验证')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='结果保存路径（JSON格式）')
    
    args = parser.parse_args()
    
    # 加载数据
    print("加载数据...")
    merged, line_cols, baseline, data_dir, disp_path, topo_features = load_data()
    print(f"  数据维度: {merged.shape}")
    print(f"  线路数: {len(line_cols)}")
    if baseline:
        print(f"  Julia基线期望失负荷: {baseline.get('expected_load_shed_total', 'N/A')} kW·h")
        print(f"  Julia基线超标节点: {len(baseline.get('violations', []))}")
    
    all_results = []
    
    if args.batch:
        # 批量测试模式
        test_lines = args.lines if args.lines else ['AC_Line_19', 'AC_Line_6', 'AC_Line_16']
        
        print(f"\n{'#'*60}")
        print(f"  批量验证模式")
        print(f"  测试线路: {', '.join(test_lines)}")
        print(f"  计划: {len(test_lines)}个单线路 + {len(test_lines)*(len(test_lines)-1)//2}个两两组合 + 1个全部组合")
        if not args.predict_only:
            total_runs = len(test_lines) + len(test_lines)*(len(test_lines)-1)//2 + 1
            print(f"  预计总耗时: {total_runs * 10:.0f}~{total_runs * 15:.0f}分钟")
        print(f"{'#'*60}")
        
        # 1. 逐个测试
        print(f"\n\n{'━'*60}")
        print(f"  第一部分: 单线路加固")
        print(f"{'━'*60}")
        for line in test_lines:
            result = validate_single(line, merged, line_cols, baseline, data_dir, args.predict_only, topo_features)
            all_results.append(result)
        
        # 2. 两两组合
        if len(test_lines) >= 2:
            print(f"\n\n{'━'*60}")
            print(f"  第二部分: 两两组合加固")
            print(f"{'━'*60}")
            from itertools import combinations
            for combo in combinations(test_lines, 2):
                result = validate_multi(list(combo), merged, line_cols, baseline, data_dir, args.predict_only, topo_features)
                all_results.append(result)
        
        # 3. 全部组合
        if len(test_lines) >= 3:
            print(f"\n\n{'━'*60}")
            print(f"  第三部分: 全部线路同时加固")
            print(f"{'━'*60}")
            result = validate_multi(test_lines, merged, line_cols, baseline, data_dir, args.predict_only, topo_features)
            all_results.append(result)
    
    elif args.lines:
        if args.multi:
            # 所有线路作为一组同时加固
            result = validate_multi(args.lines, merged, line_cols, baseline, data_dir, args.predict_only, topo_features)
            all_results.append(result)
        elif args.both:
            # 逐个 + 组合
            for line in args.lines:
                result = validate_single(line, merged, line_cols, baseline, data_dir, args.predict_only, topo_features)
                all_results.append(result)
            if len(args.lines) >= 2:
                result = validate_multi(args.lines, merged, line_cols, baseline, data_dir, args.predict_only, topo_features)
                all_results.append(result)
        else:
            # 逐个测试
            for line in args.lines:
                result = validate_single(line, merged, line_cols, baseline, data_dir, args.predict_only, topo_features)
                all_results.append(result)
    else:
        parser.print_help()
        print("\n\n请指定 --lines 或 --batch 参数")
        return
    
    # 打印汇总表
    print_summary_table(all_results)
    
    # 保存结果
    output_path = args.output or str(PROJECT_ROOT / "output" / "validation_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\n  结果已保存: {output_path}")


if __name__ == "__main__":
    main()
