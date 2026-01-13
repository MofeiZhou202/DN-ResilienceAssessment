import numpy as np
from math import exp, radians, log

import pandas as pd
from scipy.io import loadmat, savemat
from scipy.optimize import fsolve

from src.config import (
    SST_data_path,
    hurricane_data_path,
)

LATITUDE_BOUNDS = (-89.0, 89.0)
LONGITUDE_BOUNDS = (-180.0, 180.0)


def normalize_lat(lat):
    """Clamp latitude into the SST grid coverage."""
    clipped = np.clip(lat, LATITUDE_BOUNDS[0], LATITUDE_BOUNDS[1])
    return float(clipped) if np.isscalar(lat) else clipped


def normalize_lng(lng):
    """Wrap longitude to (-180, 180] while avoiding the -180 index overflow."""
    normalized = ((np.asarray(lng) + 180.0) % 360.0) - 180.0
    normalized = np.where(normalized <= -180.0, np.nextafter(180.0, 0.0), normalized)
    return float(normalized) if np.isscalar(lng) else normalized


def get_interpolated_sst(lat, lng, monthly_sst):
    """使用双线性插值获取SST数据"""
    lat = normalize_lat(lat)
    lng = normalize_lng(lng)

    lat_idx = 90.0 - lat
    lng_idx = 180.0 - lng

    lat_idx_floor = np.floor(lat_idx).astype(int)
    lat_idx_ceil = np.ceil(lat_idx).astype(int)
    lng_idx_floor = np.floor(lng_idx).astype(int)
    lng_idx_ceil = np.ceil(lng_idx).astype(int)

    lat_idx_floor = np.clip(lat_idx_floor, 0, monthly_sst.shape[0] - 1)
    lat_idx_ceil = np.clip(lat_idx_ceil, 0, monthly_sst.shape[0] - 1)
    lng_idx_floor = np.clip(lng_idx_floor, 0, monthly_sst.shape[1] - 1)
    lng_idx_ceil = np.clip(lng_idx_ceil, 0, monthly_sst.shape[1] - 1)

    lat_weight = lat_idx - lat_idx_floor
    lng_weight = lng_idx - lng_idx_floor

    f00 = monthly_sst[lat_idx_floor, lng_idx_floor]
    f10 = monthly_sst[lat_idx_ceil, lng_idx_floor]
    f01 = monthly_sst[lat_idx_floor, lng_idx_ceil]
    f11 = monthly_sst[lat_idx_ceil, lng_idx_ceil]

    weights = np.array([
        (1 - lat_weight) * (1 - lng_weight),
        lat_weight * (1 - lng_weight),
        (1 - lat_weight) * lng_weight,
        lat_weight * lng_weight,
    ])
    values = np.array([f00, f10, f01, f11], dtype=float)
    mask = np.isnan(values)
    if mask.any():
        valid_weight = weights[~mask].sum()
        if valid_weight == 0:
            lat_idx_round = int(np.clip(round(lat_idx), 0, monthly_sst.shape[0] - 1))
            lng_idx_round = int(np.clip(round(lng_idx), 0, monthly_sst.shape[1] - 1))
            fallback_value = monthly_sst[lat_idx_round, lng_idx_round]
            if not np.isnan(fallback_value):
                return float(fallback_value)

            for radius in range(1, 5):
                lat_start = max(lat_idx_round - radius, 0)
                lat_end = min(lat_idx_round + radius + 1, monthly_sst.shape[0])
                lng_start = max(lng_idx_round - radius, 0)
                lng_end = min(lng_idx_round + radius + 1, monthly_sst.shape[1])

                subgrid = monthly_sst[lat_start:lat_end, lng_start:lng_end]
                if np.isnan(subgrid).all():
                    continue

                valid_values = subgrid[~np.isnan(subgrid)]
                return float(valid_values.mean())

            return float("nan")
        return float((values[~mask] * weights[~mask]).sum() / valid_weight)

    return float(
        f00 * (1 - lat_weight) * (1 - lng_weight)
        + f10 * lat_weight * (1 - lng_weight)
        + f01 * (1 - lat_weight) * lng_weight
        + f11 * lat_weight * lng_weight
    )


def get_interpolated_t0(lat, T0_table):
    """使用线性插值获取T0数据"""
    lat = normalize_lat(lat)
    idx = (90.0 - lat) / 10.0
    idx_floor = int(np.floor(idx))
    idx_ceil = int(np.ceil(idx))

    idx_floor = np.clip(idx_floor, 0, T0_table.shape[1] - 1)
    idx_ceil = np.clip(idx_ceil, 0, T0_table.shape[1] - 1)

    if idx_floor == idx_ceil:
        return T0_table[0, idx_floor]

    weight = idx - idx_floor
    return (
        T0_table[0, idx_floor] * (1 - weight)
        + T0_table[0, idx_ceil] * weight
    )

def is_valid_hurricane(hurricane):
    # 检查是否有 NaN 或 0
    return not (np.isnan(hurricane).any() or (hurricane == 0).any())

def run_hurricane_simulation(initialconditions, month, duration=24, max_attempts=10):
    attempts = 0
    while attempts < max_attempts:
        hurricane = hurricane_simulation(initialconditions, month, duration)
        if is_valid_hurricane(hurricane):
            return hurricane
        attempts += 1
    raise RuntimeError("Failed to generate a valid hurricane after multiple attempts.")

def hurricane_simulation(initialcondition, month, duration=24):
    # 加载必要的文件
    SST = loadmat(SST_data_path)['SST']  # 形状 (1812, 180, 360)

    T0_data = pd.read_excel(hurricane_data_path + 'T0.xlsx', sheet_name='T0_9_11',header=None).values.flatten()
    T0_9_11 = T0_data.reshape(1, -1)  # 将其形状调整为 (1, 19)
    T0_data = pd.read_excel(hurricane_data_path+'T0.xlsx', sheet_name='T0_6_8',header=None).values.flatten()
    T0_6_8 = T0_data.reshape(1, -1)  # 将其形状调整为 (1, 19)

    param_sheets = ['r_speed', 'r_heading', 'r_intensity', 'speed', 'heading', 'intensity']
    parameters = {}

    for sheet in param_sheets:
        parameters[sheet] = pd.read_excel(hurricane_data_path+'Parameter_hur_v5.xlsx',sheet_name=sheet, header=None).values.flatten()

    # 提取指定月份的 30 年 SST 平均值
    start_year = 1870
    monthly_sst_indices = [(year - start_year) * 12 + (month - 1) for year in range(1991, 2021)]
    monthly_sst = np.mean(SST[monthly_sst_indices, :, :], axis=0)

    # 默认初始条件
    if initialcondition is None:
        raise ValueError("Initial condition is required.")

    occurday, occurtime, lath, lngh, deltaP, IR, rmw, theta, transSpeed = initialcondition
    lath = normalize_lat(lath)
    lngh = normalize_lng(lngh)
    omega = 7.2921e-5
    airRho = 1.15
    landfalltime = 0
    errorRmw = 0

    # 初始化变量 - 使用动态长度
    Lath = np.zeros(duration)
    Lngh = np.zeros(duration)
    Transspeed = np.zeros(duration)
    Heading = np.zeros(duration)
    Relativeintensity = np.zeros(duration)
    DeltaP = np.zeros(duration)
    Rmw = np.zeros(duration)
    HollandB = np.zeros(duration)
    Vmax = np.zeros(duration)
    Fc = np.zeros(duration)

    # 初始化前两个时段
    j = len(parameters['r_speed'])
    k = len(parameters['r_heading'])
    l = len(parameters['r_intensity'])

    transSpeed2 = transSpeed + np.random.choice(parameters['r_speed'])
    theta2 = theta + np.random.choice(parameters['r_heading'])
    IR2 = IR + np.random.choice(parameters['r_intensity'])

    DIS = transSpeed * 0.009
    lath2 = normalize_lat(lath + DIS * np.cos(np.radians(theta)))
    # 经度步长取反，确保负向航向向东移动
    lngh2 = normalize_lng(lngh - DIS * np.sin(np.radians(theta)))

    # 设置初始值
    Lath[:2] = [lath, lath2]
    Lngh[:2] = [lngh, lngh2]
    Transspeed[:2] = [transSpeed, transSpeed2]
    Heading[:2] = [theta, theta2]
    Relativeintensity[:2] = [IR, IR2]

    # 计算初始 fc 和 DeltaP2
    fc = 2 * omega * np.sin(np.radians(lath))
    fc2 = 2 * omega * np.sin(np.radians(lath2))
    Fc[:2] = [fc, fc2]

    T0_table = np.concatenate([T0_6_8, T0_9_11])
    Ts2 = get_interpolated_sst(lath2, lngh2, monthly_sst)
    T02 = get_interpolated_t0(lath2, T0_table)

    Ts2 += 272.15
    T02 += 272.15
    epsilon = (Ts2 - T02) / Ts2

    e_s = 6.112 * exp(17.67 * (Ts2 - 273) / (Ts2 - 29.5))
    RH = 0.75
    p_da = 1013 - (RH * e_s)
    L_v = 2.5e6 - 2320 * (Ts2 - 273)
    R_v = 461

    A = epsilon * L_v * e_s / ((1 - epsilon) * R_v * Ts2 * p_da)
    B = RH * (1 + e_s * np.log(RH) / (p_da * A))

    def equation(x):
        return -A * (1 / x - B) - np.log(x)

    s = fsolve(equation, 1.0)[0]
    deltaP2 = -((1 - RH) * e_s + IR2 * (s - 1) * (1013 - RH * e_s))
    DeltaP[:2] = [deltaP, deltaP2]

    lnRmw = 3.858 - 7.7e-5 * deltaP**2 + errorRmw
    rmw = np.exp(lnRmw)

    lnRmw2 = 3.858 - 7.7e-5 * deltaP2**2 + errorRmw
    rmw2 = np.exp(lnRmw2)
    Rmw[:2] = [rmw, rmw2]

    hollandB = 1.881093 - 0.010917 * lath - 0.005567 * rmw
    vmax = ((hollandB * deltaP * 100 / airRho)**0.5) * exp(-0.5)

    hollandB2 = 1.881093 - 0.010917 * lath2 - 0.005567 * rmw2
    vmax2 = ((hollandB2 * deltaP2 * 100 / airRho)**0.5) * exp(-0.5)

    HollandB[:2] = [hollandB, hollandB2]
    Vmax[:2] = [vmax, vmax2]

    # 处理登陆和未登陆情况
    for i in range(2, duration):
        r_speed = np.random.choice(parameters['r_speed'])
        r_heading = np.random.choice(parameters['r_heading'])
        r_intensity = np.random.choice(parameters['r_intensity'])

        if Lath[i - 1] >= 22:
            if landfalltime == 0:
                landfalltime = i - 1

            sigma = 0.0025
            error_a = 10
            while abs(error_a) >= 3 * sigma:
                error_a = np.random.normal(0, sigma)

            decay = 0.006 + 0.00046 * DeltaP[i - 1] ** 2 / Rmw[i - 1] + error_a
            DeltaP[i] = DeltaP[i - 1] * np.exp(-decay * (i - landfalltime))
        else:
            Relativeintensity[i] = (
                parameters['intensity'][0] +
                parameters['intensity'][1] * Relativeintensity[i - 1] +
                parameters['intensity'][2] * Relativeintensity[i - 2] +
                r_intensity
            )

            # 使用插值获取SST和T0数据
            Ts = get_interpolated_sst(Lath[i - 1], Lngh[i - 1], monthly_sst)
            T0 = get_interpolated_t0(Lath[i - 1], T0_table)

            Ts += 272.15
            T0 += 272.15
            epsilon = (Ts - T0) / Ts

            e_s = 6.112 * exp(17.67 * (Ts - 273) / (Ts - 29.5))
            p_da = 1013 - (RH * e_s)
            A = epsilon * L_v * e_s / ((1 - epsilon) * R_v * Ts * p_da)
            B = RH * (1 + e_s * np.log(RH) / (p_da * A))
            s = fsolve(lambda x: -A * (1 / x - B) - log(x), 1.0)[0]
            DeltaP[i] = -((1 - RH) * e_s + Relativeintensity[i] * (s - 1) * (1013 - RH * e_s))

        # 更新其他参数
        Transspeed[i] = (
            parameters['speed'][0] +
            parameters['speed'][1] * Transspeed[i - 1] +
            parameters['speed'][2] * Transspeed[i - 2] +
            r_speed
        )

        Heading[i] = (
            parameters['heading'][0] +
            parameters['heading'][1] * Lath[i - 1] +
            parameters['heading'][2] * Lngh[i - 1] +
            parameters['heading'][3] * Transspeed[i - 1] +
            parameters['heading'][4] * Heading[i - 1] +
            parameters['heading'][5] * Heading[i - 2] +
            r_heading
        )

        # 检查并修正移动速度
        if Transspeed[i] <= 0:
            # 如果速度无效，使用略小于前一时刻的速度
            Transspeed[i] = max(1.0, Transspeed[i-1] * 0.8)  # 保证最小速度为1 m/s

        # 检查并修正移动方向，保持在合理范围内
        if abs(Heading[i]) > 90:
            # 如果方向角度过大，调整为向合理方向偏转
            target_heading = np.sign(Heading[i]) * 45  # 将极端角度调整为45度
            Heading[i] = Heading[i-1] + 0.3 * (target_heading - Heading[i-1])  # 平滑过渡

        DIS = Transspeed[i] * 0.009
        lath_new = normalize_lat(Lath[i - 1] + DIS * np.cos(np.radians(Heading[i])))
        # 在经度方向取反，确保航向符号与经度变化方向保持一致
        lngh_new = normalize_lng(Lngh[i - 1] - DIS * np.sin(np.radians(Heading[i])))

        Lath[i] = lath_new
        Lngh[i] = lngh_new

        Fc[i] = 2 * omega * np.sin(np.radians(Lath[i]))
        lnRmw = 3.858 - 7.7e-5 * DeltaP[i] ** 2
        Rmw[i] = np.exp(lnRmw)

        HollandB[i] = 1.881093 - 0.010917 * Lath[i] - 0.005567 * Rmw[i]
        Vmax[i] = ((HollandB[i] * DeltaP[i] * 100 / airRho) ** 0.5) * exp(-0.5)

    hurricane = np.column_stack([Lath, Lngh, Fc, DeltaP, HollandB, Rmw, Heading, Transspeed, Vmax])

    return hurricane

