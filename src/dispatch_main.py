"""交直流混合配电网 + 微电网 + MESS 协同调度主程序。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.transportation_network import transportation_network

try:
    import gurobipy as gp
except ImportError as exc:
    raise RuntimeError("需要安装 gurobipy 才能求解混合配电网优化模型。") from exc

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

DEFAULT_CASE_XLSX = DATA_DIR / "ac_dc_real_case.xlsx"
DEFAULT_TOPOLOGY_XLSX = DATA_DIR / "topology_reconfiguration_results.xlsx"
DEFAULT_MC_XLSX = DATA_DIR / "mc_simulation_results_k100_clusters.xlsx"
DEFAULT_HOURS = 48
TIME_STEP_HOURS = 1.0

MESS_TRAVEL_ENERGY_LOSS_PER_HOUR = 0.0
"""每小时在移动过程中被认为的能源消耗量（kW·h）。设为0以消除移动阻碍。"""

NO_POWER_MAX_CONSECUTIVE_HOURS = 2
NO_POWER_WINDOW = NO_POWER_MAX_CONSECUTIVE_HOURS + 1
NO_POWER_SOFT_PENALTY = 1e5
NO_POWER_REL_TOL = 0.02
NO_POWER_ABS_TOL = 1e-3
"""连续断电软约束的窗口与惩罚设置。"""

TRANSPORT_NODE_TO_GRID: Mapping[int, int] = {
    1: 5,
    2: 10,
    3: 15,
    4: 20,
    5: 25,
    6: 30,
}


@dataclass(frozen=True)
class TrafficArc:
    origin: int
    destination: int
    travel_time: int


@dataclass
class MessMobilitySchedule:
    u_vars: List[gp.MVar]
    travel_vars: Dict[int, gp.MVar]
    travel_actions: Dict[int, List[TrafficArc]]
    departure_lookup: Dict[Tuple[int, int], List[Tuple[int, int]]]
    arrival_lookup: Dict[Tuple[int, int], List[Tuple[int, int]]]
    transit_lookup: Dict[int, List[Tuple[int, int]]]
    nb_transport: int


@dataclass(frozen=True)
class MESSConfig:
    """移动储能系统基础参数配置。"""

    name: str
    node: int
    charge_max: float
    discharge_max: float
    energy_max: float
    soc_initial: float
    eta_charge: float
    eta_discharge: float


@dataclass
class HybridGridCase:
    """交直流混合配电网算例数据容器。"""

    nb: int
    nb_ac: int
    nb_dc: int
    nl_ac: int
    nl_dc: int
    nl_vsc: int
    ng: int
    nmg: int
    nmess: int
    Cft_ac: sp.csr_matrix
    Cft_dc: sp.csr_matrix
    Cft_vsc: sp.csr_matrix
    Cg: sp.csr_matrix
    Cmg: sp.csr_matrix
    Cd: sp.csr_matrix
    Pd: np.ndarray
    Qd: np.ndarray
    Pgmax: np.ndarray
    Qgmax: np.ndarray
    Pmgmax: np.ndarray
    Pmgmin: np.ndarray
    Qmgmax: np.ndarray
    Qmgmin: np.ndarray
    Pvscmax: np.ndarray
    eta_vsc: np.ndarray
    Smax_ac: np.ndarray
    Smax_dc: np.ndarray
    alpha_ac: np.ndarray
    alpha_dc: np.ndarray
    alpha_vsc: np.ndarray
    switch_flag: np.ndarray
    VMAX: np.ndarray
    VMIN: np.ndarray
    R: np.ndarray
    X: np.ndarray
    bigM: float
    c_load: float
    c_sg: float
    c_mg: float
    c_vsc: float
    load_per_node: np.ndarray
    generator_nodes: np.ndarray
    microgrid_nodes: np.ndarray
    mess_nodes: np.ndarray
    mess_connect: sp.csr_matrix
    mess_charge_max: np.ndarray
    mess_discharge_max: np.ndarray
    mess_energy_max: np.ndarray
    mess_soc_initial: np.ndarray
    mess_eta_charge: np.ndarray
    mess_eta_discharge: np.ndarray
    line_types: List[str]
    transport_node_to_grid: Mapping[int, int]
    transport_grid_to_node: Mapping[int, int]
    mess_transport_initial: np.ndarray
    mess_names: Tuple[str, ...]


DEFAULT_MESS = [
    # MESS放在不同的初始位置
    MESSConfig("MESS-1", 5, 1500.0, 1500.0, 4500.0, 2500.0, 92, 90),   # 在节点5
    MESSConfig("MESS-2", 10, 1000.0, 1000.0, 4000.0, 2000.0, 92, 90),  # 在节点10
    MESSConfig("MESS-3", 30, 500.0, 500.0, 3500.0, 1500.0, 92, 90),   # 在节点30
]

# DEFAULT_MESS = [
#     MESSConfig("MESS-5", 5, 15.0, 15.0, 45.0, 25.0, 0.92, 0.9),
#     MESSConfig("MESS-15", 15, 10.0, 10.0, 40.0, 20.0, 0.92, 0.9),
#     MESSConfig("MESS-25", 25, 5.0, 5.0, 35.0, 15.0, 0.92, 0.9),
# ]


def _normalize_label(value: object) -> str:
    return str(value).strip()


def _parse_binary(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return 1 if value != 0 else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        return 1 if lowered in {"1", "true", "yes", "y"} else 0
    return 0


def _build_incidence_matrix(edges: Sequence[Tuple[int, int]], num_nodes: int) -> sp.csr_matrix:
    if not edges:
        return sp.csr_matrix((0, num_nodes))
    rows: List[int] = []
    cols: List[int] = []
    data: List[int] = []
    for idx, (frm, to) in enumerate(edges):
        rows.extend([idx, idx])
        cols.extend([frm, to])
        data.extend([1, -1])
    return sp.coo_matrix((data, (rows, cols)), shape=(len(edges), num_nodes)).tocsr()


def _build_connection_matrix(nodes: Sequence[int], num_nodes: int) -> sp.csr_matrix:
    if not nodes:
        return sp.csr_matrix((0, num_nodes))
    rows = np.arange(len(nodes), dtype=int)
    cols = np.asarray(nodes, dtype=int)
    data = np.ones(len(nodes), dtype=float)
    return sp.coo_matrix((data, (rows, cols)), shape=(len(nodes), num_nodes)).tocsr()


def _max_consecutive_true(values: Sequence[float], threshold: float = 0.5) -> int:
    best = 0
    current = 0
    for value in values:
        if value > threshold:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def load_hybrid_case(case_path: Path, mess_configs: Sequence[MESSConfig]) -> HybridGridCase:
    if not case_path.exists():
        raise FileNotFoundError(f"未找到电网算例文件 {case_path}")

    bus_df = pd.read_excel(case_path, sheet_name="bus")
    dcbus_df = pd.read_excel(case_path, sheet_name="dcbus")
    line_df = pd.read_excel(case_path, sheet_name="cable")
    dc_df = pd.read_excel(case_path, sheet_name="dcimpedance")
    inv_df = pd.read_excel(case_path, sheet_name="inverter")
    gen_df = pd.read_excel(case_path, sheet_name="util")
    mg_df = pd.read_excel(case_path, sheet_name="pvarray")
    load_df = pd.read_excel(case_path, sheet_name="lumpedload", header=1)
    dcload_df = pd.read_excel(case_path, sheet_name="dclumpload")
    switch_df = pd.read_excel(case_path, sheet_name="hvcb")

    name_to_index: dict[str, int] = {}
    VMAX_ac: List[float] = []
    VMIN_ac: List[float] = []
    ac_nodes: List[int] = []
    for row in bus_df.itertuples(index=False):
        label = _normalize_label(row.ID)
        if label in name_to_index:
            continue
        idx = len(name_to_index)
        name_to_index[label] = idx
        ac_nodes.append(idx)
        VMAX_ac.append(float(getattr(row, "VMaxLimit", 105.0)) / 100.0)
        VMIN_ac.append(float(getattr(row, "VMinLimit", 90.0)) / 100.0)

    VMAX_dc: List[float] = []
    VMIN_dc: List[float] = []
    dc_nodes: List[int] = []
    for row in dcbus_df.itertuples(index=False):
        label = _normalize_label(row.ID)
        if label in name_to_index:
            continue
        idx = len(name_to_index)
        name_to_index[label] = idx
        dc_nodes.append(idx)
        VMAX_dc.append(1.05)
        VMIN_dc.append(0.90)

    nb_ac = len(ac_nodes)
    nb_dc = len(dc_nodes)
    nb = nb_ac + nb_dc

    switch_pairs = set()
    for row in switch_df.itertuples(index=False):
        a = _normalize_label(row.FromElement)
        b = _normalize_label(row.ToElement)
        switch_pairs.add((a, b))
        switch_pairs.add((b, a))

    ac_edges: List[Tuple[int, int]] = []
    R_ac: List[float] = []
    X_ac: List[float] = []
    alpha_ac: List[int] = []
    switch_ac: List[int] = []
    for row in line_df.itertuples(index=False):
        from_label = _normalize_label(row.FromBus)
        to_label = _normalize_label(row.ToBus)
        if from_label not in name_to_index or to_label not in name_to_index:
            continue
        frm = name_to_index[from_label]
        to = name_to_index[to_label]
        ac_edges.append((frm, to))
        R_ac.append(float(getattr(row, "RPosValue", 0.0)))
        X_ac.append(float(getattr(row, "XPosValue", 0.0)))
        alpha_ac.append(_parse_binary(getattr(row, "InService", 1)))
        switch_ac.append(1 if (from_label, to_label) in switch_pairs else 0)

    dc_edges: List[Tuple[int, int]] = []
    R_dc: List[float] = []
    alpha_dc: List[int] = []
    switch_dc: List[int] = []
    for row in dc_df.itertuples(index=False):
        from_label = _normalize_label(row.FromBus)
        to_label = _normalize_label(row.ToBus)
        if from_label not in name_to_index or to_label not in name_to_index:
            continue
        frm = name_to_index[from_label]
        to = name_to_index[to_label]
        dc_edges.append((frm, to))
        R_dc.append(float(getattr(row, "RValue", 0.0)))
        alpha_dc.append(_parse_binary(getattr(row, "InService", 1)))
        switch_dc.append(1 if (from_label, to_label) in switch_pairs else 0)

    vsc_edges: List[Tuple[int, int]] = []
    Pvscmax: List[float] = []
    eta_vsc: List[float] = []
    alpha_vsc: List[int] = []
    for row in inv_df.itertuples(index=False):
        ac_label = _normalize_label(getattr(row, "BusID", ""))
        dc_label = _normalize_label(getattr(row, "CZNetwork", ""))
        if ac_label not in name_to_index or dc_label not in name_to_index:
            continue
        frm = name_to_index[ac_label]
        to = name_to_index[dc_label]
        vsc_edges.append((frm, to))
        pmax = float(getattr(row, "DckW", 0.0))
        eta = float(getattr(row, "DcPercentEFF", 95.0)) / 100.0
        Pvscmax.append(max(pmax, 0.0))
        eta_vsc.append(min(max(eta, 0.01), 1.0))
        alpha_vsc.append(_parse_binary(getattr(row, "InService", 1)))

    nl_ac = len(ac_edges)
    nl_dc = len(dc_edges)
    nl_vsc = len(vsc_edges)

    Cft_ac = _build_incidence_matrix(ac_edges, nb)
    Cft_dc = _build_incidence_matrix(dc_edges, nb)
    Cft_vsc = _build_incidence_matrix(vsc_edges, nb)

    gen_nodes: List[int] = []
    Pgmax_list: List[float] = []
    Qgmax_list: List[float] = []
    for row in gen_df.itertuples(index=False):
        bus_label = _normalize_label(getattr(row, "Bus", ""))
        if bus_label not in name_to_index:
            continue
        gen_nodes.append(name_to_index[bus_label])
        Pgmax_list.append(float(getattr(row, "OpMW", 0.0)) * 1000.0)
        Qgmax_list.append(float(getattr(row, "OpMvar", 0.0)) * 1000.0)
    Cg = _build_connection_matrix(gen_nodes, nb)

    mg_nodes: List[int] = []
    Pmgmax_list: List[float] = []
    for row in mg_df.itertuples(index=False):
        bus_label = _normalize_label(getattr(row, "Bus", ""))
        if bus_label not in name_to_index:
            continue
        mg_nodes.append(name_to_index[bus_label])
        Pmgmax_list.append(float(getattr(row, "PVAPower", 0.0)) * 1000.0)  # MW -> kW
    Cmg = _build_connection_matrix(mg_nodes, nb)

    load_nodes: List[int] = []
    Pd_list: List[float] = []
    Qd_list: List[float] = []
    DEFAULT_PF = 0.9  # 默认功率因数
    for row in load_df.itertuples(index=False):
        bus_label = _normalize_label(getattr(row, "Bus", ""))
        if bus_label not in name_to_index:
            continue
        load_nodes.append(name_to_index[bus_label])
        # MVA 列实际是 kVA 单位，MTLoadPercent 也是 kVA
        # 优先使用 MTLoadPercent（如果有值），否则使用 MVA
        mva_val = float(getattr(row, "MVA", 0.0))  # 实际是 kVA
        mt_load = float(getattr(row, "MTLoadPercent", 0.0))  # 实际是 kVA
        pf_val = float(getattr(row, "PF", 0.0)) / 100.0 if hasattr(row, "PF") else 0.0
        
        # 如果 MVA 和 PF 都有值，使用 S * PF 计算有功
        if mva_val > 0 and pf_val > 0:
            demand_kw = mva_val * pf_val  # kVA * PF = kW
        elif mt_load > 0:
            # 使用 MTLoadPercent 作为 kVA，乘以默认功率因数
            demand_kw = mt_load * DEFAULT_PF
        else:
            demand_kw = 0.0
        
        Pd_list.append(demand_kw)
        Qd_list.append(demand_kw * np.tan(np.arccos(DEFAULT_PF)) if demand_kw > 0 else 0.0)
    for row in dcload_df.itertuples(index=False):
        bus_label = _normalize_label(getattr(row, "Bus", ""))
        if bus_label not in name_to_index:
            continue
        load_nodes.append(name_to_index[bus_label])
        demand_kw = float(getattr(row, "KW", 0.0))
        Pd_list.append(demand_kw)
        Qd_list.append(0.0)  # DC负荷无功为0
    Cd = _build_connection_matrix(load_nodes, nb)

    Pd = np.asarray(Pd_list, dtype=float)
    Qd = np.asarray(Qd_list, dtype=float)
    load_per_node = np.asarray(Cd.transpose().dot(Pd)).ravel() if Pd.size else np.zeros(nb)

    mess_nodes: List[int] = []
    mess_charge_max: List[float] = []
    mess_discharge_max: List[float] = []
    mess_energy_max: List[float] = []
    mess_soc_initial: List[float] = []
    mess_eta_charge: List[float] = []
    mess_eta_discharge: List[float] = []
    for config in mess_configs:
        node_idx = config.node - 1
        if node_idx < 0 or node_idx >= nb:
            raise ValueError(f"MESS {config.name} 的接入节点 {config.node} 超出节点范围")
        mess_nodes.append(node_idx)
        mess_charge_max.append(config.charge_max)
        mess_discharge_max.append(config.discharge_max)
        mess_energy_max.append(config.energy_max)
        mess_soc_initial.append(min(config.soc_initial, config.energy_max))
        # 效率值如果>1则认为是百分比形式(如92)，需转换为小数(如0.92)
        eta_c = config.eta_charge / 100.0 if config.eta_charge > 1 else config.eta_charge
        eta_d = config.eta_discharge / 100.0 if config.eta_discharge > 1 else config.eta_discharge
        mess_eta_charge.append(eta_c)
        mess_eta_discharge.append(eta_d)
    mess_connect = _build_connection_matrix(mess_nodes, nb)

    Smax_ac = np.full(nl_ac, 4000.0, dtype=float)
    Smax_dc = np.full(nl_dc, 4000.0, dtype=float)
    if nl_dc >= 1:
        Smax_dc[0] = 200.0
    if nl_dc >= 2:
        Smax_dc[1] = 20000.0
    alpha_ac_arr = np.asarray(alpha_ac, dtype=float) if nl_ac else np.zeros(0, dtype=float)
    alpha_dc_arr = np.asarray(alpha_dc, dtype=float) if nl_dc else np.zeros(0, dtype=float)
    alpha_vsc_arr = np.asarray(alpha_vsc, dtype=float) if nl_vsc else np.zeros(0, dtype=float)

    switch_vec = np.concatenate(
        [
            np.asarray(switch_ac, dtype=float) if nl_ac else np.zeros(0, dtype=float),
            np.asarray(switch_dc, dtype=float) if nl_dc else np.zeros(0, dtype=float),
            np.ones(nl_vsc, dtype=float),
        ]
    )
    line_types = ["AC"] * nl_ac + ["DC"] * nl_dc + ["VSC"] * nl_vsc

    transport_node_to_grid_map, transport_grid_to_node_map = _build_transport_mapping(nb)
    mess_transport_initial_nodes = _resolve_mess_transport_initial(
        np.asarray(mess_nodes, dtype=int),
        transport_grid_to_node_map,
    )

    return HybridGridCase(
        nb=nb,
        nb_ac=nb_ac,
        nb_dc=nb_dc,
        nl_ac=nl_ac,
        nl_dc=nl_dc,
        nl_vsc=nl_vsc,
        ng=len(gen_nodes),
        nmg=len(mg_nodes),
        nmess=len(mess_nodes),
        Cft_ac=Cft_ac,
        Cft_dc=Cft_dc,
        Cft_vsc=Cft_vsc,
        Cg=Cg,
        Cmg=Cmg,
        Cd=Cd,
        Pd=Pd,
        Qd=Qd,
        Pgmax=np.asarray(Pgmax_list, dtype=float),
        Qgmax=np.asarray(Qgmax_list, dtype=float),
        Pmgmax=np.asarray(Pmgmax_list, dtype=float),
        Pmgmin=np.zeros(len(mg_nodes), dtype=float),
        Qmgmax=np.asarray(Pmgmax_list, dtype=float) / 5.0 if mg_nodes else np.zeros(0, dtype=float),
        Qmgmin=np.zeros(len(mg_nodes), dtype=float),
        Pvscmax=np.asarray(Pvscmax, dtype=float),
        eta_vsc=np.asarray(eta_vsc, dtype=float),
        Smax_ac=Smax_ac,
        Smax_dc=Smax_dc,
        alpha_ac=alpha_ac_arr,
        alpha_dc=alpha_dc_arr,
        alpha_vsc=alpha_vsc_arr,
        switch_flag=switch_vec,
        VMAX=np.asarray(VMAX_ac + VMAX_dc, dtype=float),
        VMIN=np.asarray(VMIN_ac + VMIN_dc, dtype=float),
        R=np.concatenate([np.asarray(R_ac, dtype=float), np.asarray(R_dc, dtype=float)]) if (R_ac or R_dc) else np.zeros(0, dtype=float),
        X=np.concatenate([np.asarray(X_ac, dtype=float), np.zeros(nl_dc, dtype=float)]) if (X_ac or nl_dc) else np.zeros(0, dtype=float),
        bigM=1000.0,
        c_load=1000.0,
        c_sg=5.0,
        c_mg=2.0,
        c_vsc=1.0,
        load_per_node=load_per_node,
        generator_nodes=np.asarray(gen_nodes, dtype=int),
        microgrid_nodes=np.asarray(mg_nodes, dtype=int),
        mess_nodes=np.asarray(mess_nodes, dtype=int),
        mess_connect=mess_connect,
        mess_charge_max=np.asarray(mess_charge_max, dtype=float),
        mess_discharge_max=np.asarray(mess_discharge_max, dtype=float),
        mess_energy_max=np.asarray(mess_energy_max, dtype=float),
        mess_soc_initial=np.asarray(mess_soc_initial, dtype=float),
        mess_eta_charge=np.asarray(mess_eta_charge, dtype=float),
        mess_eta_discharge=np.asarray(mess_eta_discharge, dtype=float),
        line_types=line_types,
        transport_node_to_grid=transport_node_to_grid_map,
        transport_grid_to_node=transport_grid_to_node_map,
        mess_transport_initial=mess_transport_initial_nodes,
        mess_names=tuple(config.name for config in mess_configs),
    )


def _build_transport_mapping(nb: int) -> tuple[Mapping[int, int], Mapping[int, int]]:
    node_map: Dict[int, int] = {}
    grid_map: Dict[int, int] = {}
    for transport_label, grid_label in TRANSPORT_NODE_TO_GRID.items():
        grid_index = grid_label - 1
        if grid_index < 0 or grid_index >= nb:
            raise ValueError(
                f"TRANSPORT_NODE_TO_GRID references grid node {grid_label} which exceeds case node count {nb}."
            )
        transport_index = transport_label - 1
        node_map[transport_index] = grid_index
        grid_map[grid_index] = transport_index
    if not node_map:
        raise ValueError("Transport node mapping must define at least one node.")
    return node_map, grid_map


def _resolve_mess_transport_initial(
    mess_nodes: np.ndarray, grid_to_transport: Mapping[int, int]
) -> np.ndarray:
    if mess_nodes.size == 0:
        return np.zeros(0, dtype=int)
    initial_nodes: List[int] = []
    for grid_idx in mess_nodes.astype(int):
        transport_node = grid_to_transport.get(int(grid_idx))
        if transport_node is None:
            raise ValueError(
                f"无法为电网节点 {grid_idx} 找到映射的交通节点，更新 TRANSPORT_NODE_TO_GRID。"
            )
        initial_nodes.append(transport_node)
    return np.asarray(initial_nodes, dtype=int)


def _setup_mess_mobility(model: gp.Model, case: HybridGridCase, hours: int, scenario_label: str) -> MessMobilitySchedule:
    transport = transportation_network()
    nb_transport = int(transport["bus"].shape[0])
    arcs: List[TrafficArc] = []
    for row in transport["branch"]:
        origin = int(row[0]) - 1
        destination = int(row[1]) - 1
        travel_time = max(int(row[2]), 1)
        if 0 <= origin < nb_transport and 0 <= destination < nb_transport:
            arcs.append(TrafficArc(origin, destination, travel_time))
            arcs.append(TrafficArc(destination, origin, travel_time))

    travel_actions: Dict[int, List[TrafficArc]] = {}
    for start_time in range(hours):
        actions: List[TrafficArc] = []
        for arc in arcs:
            if start_time + arc.travel_time <= hours:
                actions.append(arc)
        if actions:
            travel_actions[start_time] = actions

    u_vars: List[gp.MVar] = []
    label = str(scenario_label)
    for t in range(hours + 1):
        u_vars.append(
            model.addMVar(
                (case.nmess, nb_transport),
                vtype=gp.GRB.BINARY,
                name=f"mess_loc[{label},{t}]",
            )
        )

    for m_idx, start_node in enumerate(case.mess_transport_initial):
        for node_idx in range(nb_transport):
            value = 1.0 if node_idx == start_node else 0.0
            model.addConstr(
                u_vars[0][m_idx, node_idx] == value,
                name=f"mess_init[{label},{m_idx},{node_idx}]",
            )

    travel_vars: Dict[int, gp.MVar] = {}
    for start_time, actions in travel_actions.items():
        travel_vars[start_time] = model.addMVar(
            (case.nmess, len(actions)),
            vtype=gp.GRB.BINARY,
            name=f"mess_travel[{label},{start_time}]",
        )

    departure_lookup: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    arrival_lookup: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    transit_lookup: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for start_time, actions in travel_actions.items():
        var = travel_vars[start_time]
        for action_idx, action in enumerate(actions):
            departure_lookup[(start_time, action.origin)].append((start_time, action_idx))
            arrival_time = start_time + action.travel_time
            if arrival_time <= hours:
                arrival_lookup[(arrival_time, action.destination)].append((start_time, action_idx))
            # 修复：transit_lookup 只包含真正在路上的时间点
            # start_time 时刻MESS还在出发节点，不应该算作in_transit
            # 真正在路上的时间是 [start_time+1, start_time+travel_time-1]
            for active_time in range(start_time + 1, min(start_time + action.travel_time, hours)):
                transit_lookup[active_time].append((start_time, action_idx))
            for m_idx in range(case.nmess):
                model.addConstr(
                    var[m_idx, action_idx] <= u_vars[start_time][m_idx, action.origin],
                    name=f"mess_travel_link[{label},{start_time},{m_idx},{action_idx}]",
                )

    for t in range(hours):
        for node_idx in range(nb_transport):
            departures = departure_lookup.get((t, node_idx), [])
            arrivals = arrival_lookup.get((t + 1, node_idx), [])
            for m_idx in range(case.nmess):
                # 流量守恒: u[t+1] = u[t] - sum(departures at t) + sum(arrivals at t+1)
                # 变形: u[t+1] - u[t] + departures - arrivals = 0
                expr = u_vars[t + 1][m_idx, node_idx] - u_vars[t][m_idx, node_idx]
                if departures:
                    for start_time, action_idx in departures:
                        expr += travel_vars[start_time][m_idx, action_idx]
                if arrivals:
                    for start_time, action_idx in arrivals:
                        expr -= travel_vars[start_time][m_idx, action_idx]
                model.addConstr(
                    expr == 0,
                    name=f"mess_flow[{label},{t},{m_idx},{node_idx}]",
                )

    return MessMobilitySchedule(
        u_vars=u_vars,
        travel_vars=travel_vars,
        travel_actions=travel_actions,
        departure_lookup=departure_lookup,
        arrival_lookup=arrival_lookup,
        transit_lookup=transit_lookup,
        nb_transport=nb_transport,
    )


def _format_grid_number(grid_idx: int | None) -> str:
    return str(grid_idx + 1) if grid_idx is not None else "?"


def _collect_mess_vectors(
    case: HybridGridCase,
    schedule: MessMobilitySchedule,
    hours: int,
    mess_chg_vars: List,  # List of MVar for each time step
    mess_dis_vars: List,  # List of MVar for each time step
) -> Dict[str, Dict[str, List]]:
    """收集每个MESS的位置向量和净功率向量。
    
    Returns:
        dict: {
            'MESS-1': {'location': [5,5,0,0,30,...], 'power': [50.0,-20.0,...]},
            ...
        }
    """
    if case.nmess == 0:
        return {}
    
    result: Dict[str, Dict[str, List]] = {}
    
    for m_idx in range(case.nmess):
        name = case.mess_names[m_idx] if m_idx < len(case.mess_names) else f"MESS-{m_idx+1}"
        location_vector: List[int] = []
        power_vector: List[float] = []
        
        for t in range(hours):
            # 检查是否在移动中
            traveling = False
            for start_time, action_index in schedule.transit_lookup.get(t, []):
                travel_var = schedule.travel_vars[start_time]
                if travel_var.X[m_idx, action_index] > 0.5:
                    traveling = True
                    break
            
            if traveling:
                # 移动中用0表示
                location_vector.append(0)
                power_vector.append(0.0)
            else:
                # 找到当前所在节点
                location_values = schedule.u_vars[t].X[m_idx]
                if np.any(location_values > 0.5):
                    transport_node = int(np.argmax(location_values))
                    grid_idx = case.transport_node_to_grid.get(transport_node)
                    location_vector.append(grid_idx + 1 if grid_idx is not None else -1)
                else:
                    location_vector.append(-1)
                
                # 计算净功率: 放电为正，充电为负
                if t < len(mess_chg_vars) and t < len(mess_dis_vars):
                    total_charge = float(np.sum(mess_chg_vars[t].X[m_idx, :]))
                    total_discharge = float(np.sum(mess_dis_vars[t].X[m_idx, :]))
                    net_power = total_discharge - total_charge  # 正=放电, 负=充电
                    power_vector.append(round(net_power, 2))
                else:
                    power_vector.append(0.0)
        
        result[name] = {'location': location_vector, 'power': power_vector}
    
    return result


def _collect_mobility_log(
    case: HybridGridCase, schedule: MessMobilitySchedule, hours: int
) -> List[str]:
    """旧版轨迹日志(保留兼容性)"""
    if case.nmess == 0:
        return []
    per_mess_paths: List[List[str]] = [[] for _ in range(case.nmess)]
    for t in range(hours):
        for m_idx in range(case.nmess):
            traveling_action = None
            for start_time, action_index in schedule.transit_lookup.get(t, []):
                travel_var = schedule.travel_vars[start_time]
                if travel_var.X[m_idx, action_index] > 0.5:
                    traveling_action = schedule.travel_actions[start_time][action_index]
                    break
            if traveling_action is not None:
                origin_idx = case.transport_node_to_grid.get(traveling_action.origin)
                destination_idx = case.transport_node_to_grid.get(traveling_action.destination)
                entry = (
                    f"{_format_grid_number(origin_idx)}->"
                    f"{_format_grid_number(destination_idx)}"
                )
            else:
                location_values = schedule.u_vars[t].X[m_idx]
                if np.any(location_values > 0.5):
                    transport_node = int(np.argmax(location_values))
                    grid_idx = case.transport_node_to_grid.get(transport_node)
                    entry = _format_grid_number(grid_idx)
                else:
                    entry = "Unknown"
            per_mess_paths[m_idx].append(entry)
    log_entries: List[str] = []
    for m_idx, path in enumerate(per_mess_paths):
        label = (
            case.mess_names[m_idx]
            if case.mess_names and m_idx < len(case.mess_names)
            else f"MESS-{m_idx+1}"
        )
        path_text = ", ".join(path)
        log_entries.append(f"{label} 轨迹: [{path_text}] (共{len(path)}个元素)")
    return log_entries


def schedule_departure_grid(case: HybridGridCase, transport_node: int) -> str:
    grid_idx = case.transport_node_to_grid.get(transport_node)
    if grid_idx is None:
        return "Node?"
    return f"Node{grid_idx + 1}"


def schedule_transport_grid_label(case: HybridGridCase, transport_node: int) -> str:
    return schedule_departure_grid(case, transport_node)


def load_topology_status(
    topology_path: Path,
    total_lines: int,
    hours: int,
    fallback_path: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """加载拓扑状态数据。
    
    如果 topology_path 文件不存在或无法读取，且提供了 fallback_path，
    则尝试从 fallback_path 加载数据。
    """
    # 尝试主路径
    actual_path = topology_path
    use_fallback = False
    
    if not topology_path.exists():
        if fallback_path and fallback_path.exists():
            print(f"[警告] 未找到 {topology_path.name}，使用备用文件 {fallback_path.name}")
            actual_path = fallback_path
            use_fallback = True
        else:
            raise FileNotFoundError(f"未找到网络拓扑状态文件 {topology_path}")
    else:
        # 检查文件是否可读
        try:
            import openpyxl
            openpyxl.load_workbook(topology_path, read_only=True).close()
        except Exception as e:
            if fallback_path and fallback_path.exists():
                print(f"[警告] {topology_path.name} 无法读取 ({e})，使用备用文件 {fallback_path.name}")
                actual_path = fallback_path
                use_fallback = True
            else:
                raise

    outage_df = pd.read_excel(
        actual_path,
        sheet_name="cluster_representatives",
        header=None,
        usecols="E:AZ",
        skiprows=1,
        engine="openpyxl",
    )
    matrix = outage_df.to_numpy(dtype=float)
    if matrix.shape[1] != hours:
        raise ValueError(f"拓扑文件时间维度为 {matrix.shape[1]}，需为 {hours}")
    rows = matrix.shape[0]
    if rows % total_lines != 0:
        raise ValueError("拓扑状态行数与线路数不匹配，无法重构场景矩阵")
    scenario_count = rows // total_lines
    matrix = matrix.astype(int)
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError("拓扑状态矩阵应为 0/1 值")
    reshaped = matrix.reshape(scenario_count, total_lines, hours)
    outages = np.transpose(reshaped, (0, 2, 1)).astype(float)
    status = 1.0 - outages

    summary = pd.read_excel(actual_path, sheet_name="cluster_summary", engine="openpyxl")
    if "sample_fraction" not in summary.columns:
        raise ValueError("cluster_summary 缺少 sample_fraction 列")
    weights = summary["sample_fraction"].to_numpy(dtype=float)
    if weights.size != scenario_count:
        raise ValueError("概率向量长度与场景数不一致")
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("概率向量总和必须为正")
    weights = weights / weight_sum
    labels: List[str] = []
    for row in summary.itertuples(index=False):
        cluster_id = getattr(row, "cluster_id", None)
        samples = getattr(row, "sample_count", None)
        label_parts: List[str] = []
        if cluster_id is not None:
            label_parts.append(f"Cluster {int(cluster_id)}")
        if samples is not None:
            label_parts.append(f"samples {int(samples)}")
        labels.append(" / ".join(label_parts) if label_parts else "Unnamed scenario")
    return status, weights, labels


def optimize_hybrid_dispatch(
    case: HybridGridCase,
    statuses: np.ndarray,
    weights: Sequence[float] | None = None,
) -> Tuple[gp.Model, dict]:
    status_arr = np.asarray(statuses, dtype=float)
    if status_arr.ndim == 2:
        status_arr = status_arr[np.newaxis, ...]
    elif status_arr.ndim != 3:
        raise ValueError("拓扑状态数据必须为2维或3维数组")

    scenario_count, hours, total_lines = status_arr.shape
    expected_lines = case.nl_ac + case.nl_dc + case.nl_vsc
    if total_lines != expected_lines:
        raise ValueError("拓扑状态维度与线路数不一致")

    if weights is None:
        weights_arr = np.ones(scenario_count, dtype=float) / scenario_count
    else:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape != (scenario_count,):
            raise ValueError("概率向量长度必须与场景数一致")
        total_weight = float(weights_arr.sum())
        if total_weight <= 0:
            raise ValueError("概率向量总和必须为正")
        weights_arr = weights_arr / total_weight

    ac_slice = slice(0, case.nl_ac)
    dc_slice = slice(case.nl_ac, case.nl_ac + case.nl_dc)
    vsc_slice = slice(case.nl_ac + case.nl_dc, expected_lines)

    model = gp.Model("hybrid_ac_dc_mess_batch")
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    model.Params.TimeLimit = 360
    model.Params.MIPGap = 0.01

    objective = gp.LinExpr()
    scenario_records: List[dict] = []

    for scenario_idx in range(scenario_count):
        scenario_status = np.clip(status_arr[scenario_idx], 0.0, 1.0)
        scenario_weight = float(weights_arr[scenario_idx])
        status_ac = (
            scenario_status[:, ac_slice] * case.alpha_ac if case.nl_ac else np.zeros((hours, 0))
        )
        status_dc = (
            scenario_status[:, dc_slice] * case.alpha_dc if case.nl_dc else np.zeros((hours, 0))
        )
        status_vsc = (
            scenario_status[:, vsc_slice] * case.alpha_vsc if case.nl_vsc else np.zeros((hours, 0))
        )

        scenario_expr = gp.LinExpr()
        scenario_pg_vars: List[gp.MVar] = []
        scenario_pm_vars: List[gp.MVar] = []
        scenario_shed_vars: List[gp.MVar] = []
        scenario_soc_vars: List[gp.MVar] = []
        scenario_mess_chg_vars: List[gp.MVar] = []
        scenario_mess_dis_vars: List[gp.MVar] = []
        scenario_prec_vars: List[gp.MVar] = []
        scenario_pinv_vars: List[gp.MVar] = []
        scenario_no_power_vars: List[gp.MVar] = []
        scenario_no_power_violation_vars: List[gp.Var] = []
        mobility_schedule: MessMobilitySchedule | None = None
        if case.nmess:
            mobility_schedule = _setup_mess_mobility(
                model,
                case,
                hours,
                scenario_label=f"s{scenario_idx}",
            )
            soc_init = model.addMVar(
                case.nmess,
                lb=0.0,
                ub=case.mess_energy_max,
                name=f"soc[{scenario_idx},0]",
            )
            model.addConstr(
                soc_init == case.mess_soc_initial,
                name=f"soc_init[{scenario_idx}]",
            )
            scenario_soc_vars.append(soc_init)

        for t in range(hours):
            ac_var = None
            dc_var = None
            prec_var = None
            pinv_var = None
            if case.nl_ac:
                ub_ac = case.Smax_ac * status_ac[t]
                ac_var = model.addMVar(
                    case.nl_ac,
                    lb=-ub_ac,
                    ub=ub_ac,
                    name=f"p_ac[{scenario_idx},{t}]",
                )
            if case.nl_dc:
                ub_dc = case.Smax_dc * status_dc[t]
                dc_var = model.addMVar(
                    case.nl_dc,
                    lb=-ub_dc,
                    ub=ub_dc,
                    name=f"p_dc[{scenario_idx},{t}]",
                )
            if case.ng:
                pg_var = model.addMVar(
                    case.ng,
                    lb=0.0,
                    ub=case.Pgmax,
                    name=f"pg[{scenario_idx},{t}]",
                )
                scenario_expr += case.c_sg * gp.quicksum(pg_var)
                scenario_pg_vars.append(pg_var)
            else:
                pg_var = None
            if case.nmg:
                pm_var = model.addMVar(
                    case.nmg,
                    lb=case.Pmgmin,
                    ub=case.Pmgmax,
                    name=f"pm[{scenario_idx},{t}]",
                )
                scenario_expr += case.c_mg * gp.quicksum(pm_var)
                scenario_pm_vars.append(pm_var)
            else:
                pm_var = None
            if case.nl_vsc:
                ub_prec = case.Pvscmax * status_vsc[t]
                prec_var = model.addMVar(
                    case.nl_vsc,
                    lb=0.0,
                    ub=ub_prec,
                    name=f"prec[{scenario_idx},{t}]",
                )
                pinv_var = model.addMVar(
                    case.nl_vsc,
                    lb=0.0,
                    ub=ub_prec,
                    name=f"pinv[{scenario_idx},{t}]",
                )
                scenario_prec_vars.append(prec_var)
                scenario_pinv_vars.append(pinv_var)
                scenario_expr += case.c_vsc * (gp.quicksum(prec_var) + gp.quicksum(pinv_var))
            mess_chg_node = None
            mess_dis_node = None
            charge_state = None
            discharge_state = None
            total_charge_exprs: List[gp.LinExpr] = []
            total_discharge_exprs: List[gp.LinExpr] = []
            in_transit_exprs: List[gp.LinExpr | float] = []
            if case.nmess:
                if mobility_schedule is None:
                    raise RuntimeError("MESS mobility schedule is required when nmess > 0.")
                nb_transport = mobility_schedule.nb_transport
                charge_ub = np.tile(case.mess_charge_max.reshape(-1, 1), (1, nb_transport))
                discharge_ub = np.tile(case.mess_discharge_max.reshape(-1, 1), (1, nb_transport))
                mess_chg_node = model.addMVar(
                    (case.nmess, nb_transport),
                    lb=0.0,
                    ub=charge_ub,
                    name=f"mess_chg[{scenario_idx},{t}]",
                )
                mess_dis_node = model.addMVar(
                    (case.nmess, nb_transport),
                    lb=0.0,
                    ub=discharge_ub,
                    name=f"mess_dis[{scenario_idx},{t}]",
                )
                charge_state = model.addMVar(
                    case.nmess,
                    vtype=gp.GRB.BINARY,
                    name=f"mess_charge_state[{scenario_idx},{t}]",
                )
                discharge_state = model.addMVar(
                    case.nmess,
                    vtype=gp.GRB.BINARY,
                    name=f"mess_discharge_state[{scenario_idx},{t}]",
                )
                scenario_mess_chg_vars.append(mess_chg_node)
                scenario_mess_dis_vars.append(mess_dis_node)

                for m_idx in range(case.nmess):
                    location_expr = gp.quicksum(mobility_schedule.u_vars[t][m_idx, :])
                    transit_terms = [
                        mobility_schedule.travel_vars[start_time][m_idx, action_idx]
                        for start_time, action_idx in mobility_schedule.transit_lookup.get(t, [])
                    ]
                    in_transit = gp.quicksum(transit_terms) if transit_terms else 0.0
                    in_transit_exprs.append(in_transit)
                    model.addConstr(
                        location_expr + in_transit == 1.0,
                        name=f"mess_mode[{scenario_idx},{t},{m_idx}]",
                    )
                    model.addConstr(
                        charge_state[m_idx] <= location_expr,
                        name=f"mess_charge_location[{scenario_idx},{t},{m_idx}]",
                    )
                    model.addConstr(
                        discharge_state[m_idx] <= location_expr,
                        name=f"mess_discharge_location[{scenario_idx},{t},{m_idx}]",
                    )
                    model.addConstr(
                        charge_state[m_idx] + discharge_state[m_idx] + in_transit <= 1.0,
                        name=f"mess_state_exclusive[{scenario_idx},{t},{m_idx}]",
                    )
                    total_charge = gp.quicksum(
                        mess_chg_node[m_idx, node_idx] for node_idx in range(nb_transport)
                    )
                    total_discharge = gp.quicksum(
                        mess_dis_node[m_idx, node_idx] for node_idx in range(nb_transport)
                    )
                    total_charge_exprs.append(total_charge)
                    total_discharge_exprs.append(total_discharge)
                    model.addConstr(
                        total_charge <= case.mess_charge_max[m_idx] * charge_state[m_idx],
                        name=f"mess_charge_capacity[{scenario_idx},{t},{m_idx}]",
                    )
                    model.addConstr(
                        total_discharge <= case.mess_discharge_max[m_idx] * discharge_state[m_idx],
                        name=f"mess_discharge_capacity[{scenario_idx},{t},{m_idx}]",
                    )
                    for node_idx in range(nb_transport):
                        model.addConstr(
                            mess_chg_node[m_idx, node_idx]
                            <= case.mess_charge_max[m_idx] * mobility_schedule.u_vars[t][m_idx, node_idx],
                            name=f"mess_charge_location_lock[{scenario_idx},{t},{m_idx},{node_idx}]",
                        )
                        model.addConstr(
                            mess_dis_node[m_idx, node_idx]
                            <= case.mess_discharge_max[m_idx] * mobility_schedule.u_vars[t][m_idx, node_idx],
                            name=f"mess_discharge_location_lock[{scenario_idx},{t},{m_idx},{node_idx}]",
                        )
            shed_var = model.addMVar(
                case.nb,
                lb=0.0,
                ub=case.load_per_node + 1e-6,
                name=f"shed[{scenario_idx},{t}]",
            )
            scenario_shed_vars.append(shed_var)
            scenario_expr += case.c_load * gp.quicksum(shed_var)

            no_power_var = model.addMVar(
                case.nb,
                vtype=gp.GRB.BINARY,
                name=f"no_power[{scenario_idx},{t}]",
            )
            scenario_no_power_vars.append(no_power_var)
            for node_idx in range(case.nb):
                load_value = float(case.load_per_node[node_idx])
                if load_value <= NO_POWER_ABS_TOL:
                    model.addConstr(
                        no_power_var[node_idx] == 0.0,
                        name=f"no_power_zero[{scenario_idx},{t},{node_idx}]",
                    )
                    continue
                tolerance = max(NO_POWER_ABS_TOL, NO_POWER_REL_TOL * load_value)
                powered_upper = max(load_value - tolerance, 0.0)
                outage_lower = max(load_value - NO_POWER_ABS_TOL, 0.0)
                model.addGenConstrIndicator(
                    no_power_var[node_idx],
                    0,
                    shed_var[node_idx],
                    gp.GRB.LESS_EQUAL,
                    powered_upper,
                    name=f"no_power_upper[{scenario_idx},{t},{node_idx}]",
                )
                model.addGenConstrIndicator(
                    no_power_var[node_idx],
                    1,
                    shed_var[node_idx],
                    gp.GRB.GREATER_EQUAL,
                    outage_lower,
                    name=f"no_power_lower[{scenario_idx},{t},{node_idx}]",
                )

            node_expr = [gp.LinExpr() for _ in range(case.nb)]

            if case.nl_ac and ac_var is not None:
                ac_matrix = case.Cft_ac.tocsr()
                for line_idx in range(case.nl_ac):
                    start = ac_matrix.indptr[line_idx]
                    end = ac_matrix.indptr[line_idx + 1]
                    indices = ac_matrix.indices[start:end]
                    values = ac_matrix.data[start:end]
                    for node_idx, coeff in zip(indices, values, strict=False):
                        node_expr[node_idx] += coeff * ac_var[line_idx]

            if case.nl_dc and dc_var is not None:
                dc_matrix = case.Cft_dc.tocsr()
                for line_idx in range(case.nl_dc):
                    start = dc_matrix.indptr[line_idx]
                    end = dc_matrix.indptr[line_idx + 1]
                    indices = dc_matrix.indices[start:end]
                    values = dc_matrix.data[start:end]
                    for node_idx, coeff in zip(indices, values, strict=False):
                        node_expr[node_idx] += coeff * dc_var[line_idx]

            if case.nl_vsc and prec_var is not None and pinv_var is not None:
                vsc_matrix = case.Cft_vsc.tocsr()
                for v_idx in range(case.nl_vsc):
                    start = vsc_matrix.indptr[v_idx]
                    end = vsc_matrix.indptr[v_idx + 1]
                    indices = vsc_matrix.indices[start:end]
                    values = vsc_matrix.data[start:end]
                    for node_idx, coeff in zip(indices, values, strict=False):
                        if node_idx < case.nb_ac:
                            node_expr[node_idx] += coeff * prec_var[v_idx]
                            node_expr[node_idx] -= coeff * case.eta_vsc[v_idx] * pinv_var[v_idx]
                        else:
                            node_expr[node_idx] += coeff * case.eta_vsc[v_idx] * prec_var[v_idx]
                            node_expr[node_idx] -= coeff * pinv_var[v_idx]

            if case.ng and pg_var is not None:
                for g_idx, node_idx in enumerate(case.generator_nodes):
                    node_expr[node_idx] += pg_var[g_idx]

            if case.nmg and pm_var is not None:
                for mg_idx, node_idx in enumerate(case.microgrid_nodes):
                    node_expr[node_idx] += pm_var[mg_idx]

            if case.nmess and mess_chg_node is not None and mess_dis_node is not None:
                for m_idx in range(case.nmess):
                    for transport_node_idx, grid_node_idx in case.transport_node_to_grid.items():
                        if grid_node_idx < 0 or grid_node_idx >= case.nb:
                            continue
                        node_expr[grid_node_idx] += mess_dis_node[m_idx, transport_node_idx]
                        node_expr[grid_node_idx] -= mess_chg_node[m_idx, transport_node_idx]

            for node_idx in range(case.nb):
                node_expr[node_idx] += shed_var[node_idx]
                node_expr[node_idx] -= case.load_per_node[node_idx]
                model.addConstr(
                    node_expr[node_idx] == 0.0,
                    name=f"kcl[{scenario_idx},{t},{node_idx}]",
                )

            if case.nmess and mess_chg_node is not None and mess_dis_node is not None:
                prev_soc = scenario_soc_vars[-1]
                next_soc = model.addMVar(
                    case.nmess,
                    lb=0.0,
                    ub=case.mess_energy_max,
                    name=f"soc[{scenario_idx},{t+1}]",
                )
                for m_idx in range(case.nmess):
                    total_charge = total_charge_exprs[m_idx]
                    total_discharge = total_discharge_exprs[m_idx]
                    in_transit = in_transit_exprs[m_idx]
                    model.addConstr(
                        next_soc[m_idx]
                        == prev_soc[m_idx]
                        + case.mess_eta_charge[m_idx] * total_charge * TIME_STEP_HOURS
                        - (total_discharge / case.mess_eta_discharge[m_idx]) * TIME_STEP_HOURS
                        - MESS_TRAVEL_ENERGY_LOSS_PER_HOUR * in_transit * TIME_STEP_HOURS,
                        name=f"soc_balance[{scenario_idx},{t},{m_idx}]",
                    )
                scenario_soc_vars.append(next_soc)

        if case.nmess:
            final_soc = scenario_soc_vars[-1]
            for m_idx in range(case.nmess):
                min_terminal_soc = case.mess_soc_initial[m_idx] * 0.5
                max_terminal_soc = min(
                    case.mess_soc_initial[m_idx] * 1.5,
                    case.mess_energy_max[m_idx],
                )
                model.addConstr(
                    final_soc[m_idx] >= min_terminal_soc,
                    name=f"soc_terminal_min[{scenario_idx},{m_idx}]",
                )
                model.addConstr(
                    final_soc[m_idx] <= max_terminal_soc,
                    name=f"soc_terminal_max[{scenario_idx},{m_idx}]",
                )

        no_power_penalty_terms: List[gp.Var] = []
        if scenario_no_power_vars and hours >= NO_POWER_WINDOW:
            for node_idx in range(case.nb):
                for t in range(hours - NO_POWER_WINDOW + 1):
                    viol = model.addVar(
                        lb=0.0,
                        name=f"no_power_violation[{scenario_idx},{node_idx},{t}]",
                    )
                    window_expr = gp.quicksum(
                        scenario_no_power_vars[t + offset][node_idx]
                        for offset in range(NO_POWER_WINDOW)
                    )
                    model.addConstr(
                        window_expr
                        <= NO_POWER_MAX_CONSECUTIVE_HOURS + viol,
                        name=f"no_power_window[{scenario_idx},{node_idx},{t}]",
                    )
                    no_power_penalty_terms.append(viol)
        if no_power_penalty_terms:
            scenario_expr += NO_POWER_SOFT_PENALTY * gp.quicksum(no_power_penalty_terms)
            scenario_no_power_violation_vars.extend(no_power_penalty_terms)

        scenario_records.append(
            {
                "weight": scenario_weight,
                "hours": hours,
                "mobility_schedule": mobility_schedule,
                "mess_chg_vars": scenario_mess_chg_vars,
                "mess_dis_vars": scenario_mess_dis_vars,
                "shed_vars": scenario_shed_vars,
                "pg_vars": scenario_pg_vars,
                "pm_vars": scenario_pm_vars,
                "prec_vars": scenario_prec_vars,
                "pinv_vars": scenario_pinv_vars,
                "soc_vars": scenario_soc_vars,
                "no_power_vars": scenario_no_power_vars,
                "no_power_violation_vars": scenario_no_power_violation_vars,
            }
        )
        objective += scenario_weight * scenario_expr

    model.setObjective(objective, gp.GRB.MINIMIZE)
    model.optimize()
    status = model.Status
    if status == gp.GRB.OPTIMAL:
        pass
    elif status == gp.GRB.TIME_LIMIT and model.SolCount > 0:
        gap_value = model.MIPGap if model.MIPGap is not None else float("nan")
        print(
            f"Warning: Gurobi hit TIME_LIMIT (status {status}) but produced a feasible solution;"
            f" Gap={gap_value:.6f}",
        )
    else:
        raise RuntimeError(f"调度模型求解失败，Gurobi 状态码 {status}")

    scenario_results: List[dict] = []
    for record in scenario_records:
        shed_total = sum(float(np.sum(var.X)) for var in record["shed_vars"])
        load_demand_total = float(case.load_per_node.sum() * record["hours"])
        generation_total = sum(float(np.sum(var.X)) for var in record["pg_vars"])
        microgrid_total = sum(float(np.sum(var.X)) for var in record["pm_vars"])
        prec_total = sum(float(np.sum(var.X)) for var in record["prec_vars"])
        pinv_total = sum(float(np.sum(var.X)) for var in record["pinv_vars"])
        soc_terminal = (
            record["soc_vars"][-1].X.copy() if record["soc_vars"] else np.zeros(0)
        )
        mobility_log: List[str] = []
        mess_vectors: Dict[str, Dict[str, List]] = {}
        if record["mobility_schedule"]:
            mobility_log = _collect_mobility_log(
                case,
                record["mobility_schedule"],
                record["hours"],
            )
            mess_vectors = _collect_mess_vectors(
                case,
                record["mobility_schedule"],
                record["hours"],
                record["mess_chg_vars"],
                record["mess_dis_vars"],
            )
        no_power_violations: Dict[int, int] = {}
        if record.get("no_power_vars"):
            no_power_matrix = np.vstack([var.X for var in record["no_power_vars"]])
            for node_idx in range(no_power_matrix.shape[1]):
                max_run = _max_consecutive_true(no_power_matrix[:, node_idx])
                if max_run > NO_POWER_MAX_CONSECUTIVE_HOURS:
                    no_power_violations[node_idx + 1] = int(max_run)
        pg_cost = case.c_sg * generation_total
        pm_cost = case.c_mg * microgrid_total
        vsc_cost = case.c_vsc * (prec_total + pinv_total)
        load_cost = case.c_load * shed_total
        scenario_objective = pg_cost + pm_cost + vsc_cost + load_cost
        served = load_demand_total - shed_total
        served_ratio = served / load_demand_total if load_demand_total else 1.0
        scenario_results.append(
            {
                "probability": record["weight"],
                "load_demand_total": load_demand_total,
                "load_shed_total": shed_total,
                "generation_total": generation_total,
                "microgrid_total": microgrid_total,
                "mess_soc_terminal": soc_terminal,
                "mobility_log": mobility_log,
                "mess_vectors": mess_vectors,
                "served_ratio": served_ratio,
                "objective": scenario_objective,
                "no_power_violations": no_power_violations,
            }
        )

    expected_load_shed = sum(detail["probability"] * detail["load_shed_total"] for detail in scenario_results)
    expected_supply_ratio = sum(detail["probability"] * detail["served_ratio"] for detail in scenario_results)

    result = {
        "objective": model.ObjVal,
        "expected_load_shed_total": expected_load_shed,
        "expected_supply_ratio": expected_supply_ratio,
        "scenario_results": scenario_results,
    }
    return model, result


def main() -> None:
    case = load_hybrid_case(DEFAULT_CASE_XLSX, DEFAULT_MESS)
    
    # 打印负荷统计信息
    print("="*60)
    print("配电网数据加载统计")
    print("="*60)
    print(f"节点数: AC={case.nb_ac}, DC={case.nb_dc}, 总计={case.nb}")
    print(f"线路数: AC={case.nl_ac}, DC={case.nl_dc}, VSC={case.nl_vsc}")
    print(f"负荷数: {len(case.Pd)}")
    print(f"各负荷功率 (kW): {case.Pd}")
    print(f"单时刻总负荷功率: {case.Pd.sum():.2f} kW")
    print(f"总负荷电量 ({DEFAULT_HOURS}小时): {case.Pd.sum() * DEFAULT_HOURS:.2f} kW·h")
    print(f"微电网/光伏数: {case.nmg}, 最大出力: {case.Pmgmax} kW")
    print(f"MESS数量: {case.nmess}")
    print("="*60)
    
    total_lines = case.nl_ac + case.nl_dc + case.nl_vsc
    statuses, weights, labels = load_topology_status(
        DEFAULT_TOPOLOGY_XLSX, total_lines, DEFAULT_HOURS,
        fallback_path=DEFAULT_MC_XLSX
    )
    if not statuses.size:
        raise ValueError("未加载到任何拓扑场景")

    model, result = optimize_hybrid_dispatch(case, statuses, weights)
    scenario_details = result["scenario_results"]
    total_scenarios = len(scenario_details)
    for idx, detail in enumerate(scenario_details):
        label = labels[idx] if idx < len(labels) else f"Scenario {idx}"
        prob = detail["probability"]
        print(f"\n{'='*60}")
        print(f"场景 {idx + 1}/{total_scenarios} ({label})")
        print(f"{'='*60}")
        print(f"场景概率: {prob:.4f}")
        print(f"调度模型目标值: {detail['objective']:.4f}")
        print(f"总负荷需求: {detail['load_demand_total']:.4f} kW·h")
        print(f"削减负荷: {detail['load_shed_total']:.4f} kW·h (供电率 {detail['served_ratio']:.2%})")
        
        if case.nmess:
            soc_text = ", ".join(f"{val:.2f}" for val in detail["mess_soc_terminal"])
            print(f"MESS 末端 SOC: [{soc_text}] kW·h")
            
            print(f"\n--- MESS 向量化结果 (48时段) ---")
            mess_vectors = detail.get("mess_vectors", {})
            for mess_name, vectors in mess_vectors.items():
                location = vectors.get('location', [])
                power = vectors.get('power', [])
                print(f"\n{mess_name} Location: {location}")
                print(f"{mess_name} Power (kW): {power}")
                
                unique_locs = set(loc for loc in location if loc > 0)
                if len(unique_locs) > 1:
                    print(f"  → ✅ {mess_name} 发生了移动! 访问了节点: {sorted(unique_locs)}")
                else:
                    print(f"  → ⚠️ {mess_name} 未移动，始终在节点 {unique_locs}")

        violations = detail.get("no_power_violations", {})
        if violations:
            print("\n⚠️ 连续断电超过 2 小时的节点：")
            for node_id, duration in sorted(violations.items()):
                print(f"  - 节点 {node_id}: 最长 {duration} 小时无电")
        else:
            print("\n所有节点均满足“任意节点连续断电不超过2小时”的软约束。")

    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    print(f"期望调度目标: {result['objective']:.4f}")
    print(f"期望削减负荷: {result['expected_load_shed_total']:.4f} kW·h")
    print(f"期望供电率: {result['expected_supply_ratio']:.2%}")


if __name__ == "__main__":
    main()
