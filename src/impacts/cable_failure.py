"""Cable failure probability utilities.

This module centralizes all logic needed to compute cable failure
probabilities under typhoon scenarios so that CLI entry-points can
remain thin orchestration layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import LEGACY_ORIGIN_LAT_LON, ORIGIN_LAT_LON
from src.impacts.site_rain import RainCalculator
from src.utils.coordinates import latlon_to_xy

CABLE_K1 = 100.0
CABLE_K2 = 10.0
CABLE_REQUIRED_COLUMNS = [
    "locationx",
    "locationy",
    "CN",
    "n",
    "S0",
    "Z",
    "phi",
    "theta",
    "Pbase",
]


def build_tower_and_cable_segments(
    tower_excel: Path | str,
) -> Tuple[List[dict], List[int], Dict[str, pd.DataFrame]]:
    """Load tower/segment and cable sheets from the Excel workbook."""

    path = Path(tower_excel).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Tower/segment workbook not found: {path}")

    sheets = pd.read_excel(path, sheet_name=None)
    tower_segments: List[dict] = []
    cable_ids: List[int] = []
    cable_segments: Dict[str, pd.DataFrame] = {}

    for key, tower_df in sheets.items():
        key_str = str(key)
        if key_str.startswith("Tower_"):
            try:
                identifier = int(key_str.split("_")[1])
            except (IndexError, ValueError):
                continue
            segment_key = f"Segment_{identifier}"
            segment_df = sheets.get(segment_key)
            if segment_df is None:
                continue
            tower_segments.append(
                {
                    "lineid": identifier,
                    "pairlineid": identifier + 1,
                    "tower": {"impactpara": tower_df.to_numpy()},
                    "segment": {"impactpara": segment_df.to_numpy()},
                }
            )
        elif key_str.startswith("CableSeg_"):
            try:
                cable_id = int(key_str.split("_")[1])
            except (IndexError, ValueError):
                continue
            cable_ids.append(cable_id)
            cable_segments[key_str] = tower_df.copy()

    if not tower_segments and not cable_ids:
        raise ValueError(f"No Tower_/Segment_ or CableSeg_ sheets found in {path}")

    return tower_segments, sorted(set(cable_ids)), cable_segments


def compute_cable_failure_for_hurricane(
    cable_segments: Dict[str, pd.DataFrame],
    hurricane: np.ndarray,
    sample_hour: int,
) -> Dict[int, np.ndarray]:
    """Compute system failure probabilities for every cable."""

    results: Dict[int, np.ndarray] = {}
    for sheet_name, df in cable_segments.items():
        try:
            cable_id = int(sheet_name.split("_")[1])
        except (IndexError, ValueError):
            print(f"[cable] Warning: unable to parse sheet '{sheet_name}', skipping")
            continue

        missing_cols = [col for col in CABLE_REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"[cable] Warning: {sheet_name} missing columns {missing_cols}, skipping")
            continue

        rainfall = _calculate_cable_segment_rainfall(df, hurricane, sample_hour)
        segment_failure_probs = _calculate_cable_failure_probability(df, rainfall, sample_hour)
        system_failure_prob = _calculate_cable_system_failure_probability(segment_failure_probs)
        results[cable_id] = system_failure_prob

        max_prob = float(np.max(system_failure_prob)) if system_failure_prob.size else 0.0
        mean_prob = float(np.mean(system_failure_prob)) if system_failure_prob.size else 0.0
        print(
            f"[cable] {sheet_name}: nodes={df.shape[0]}, max={max_prob:.6f}, mean={mean_prob:.6f}"
        )

    return results


def simulate_cable_rainfall(
    cable_segments: Dict[str, pd.DataFrame],
    hurricane: np.ndarray,
) -> Dict[str, pd.DataFrame]:
    """Return rainfall timeseries for every cable sheet."""

    if not cable_segments or hurricane.size == 0:
        return {}

    sample_hour = hurricane.shape[0]
    columns = [f"Hour_{i + 1}" for i in range(sample_hour)]
    rain_calc = RainCalculator()
    delta_p = hurricane[:, 3]
    dpdt = np.empty(sample_hour, dtype=float)
    dpdt[0] = 1.0
    dpdt[1:] = delta_p[:-1] - delta_p[1:]
    hurricane_xy = np.array(
        [_convert_hurricane_coordinates(lat, lon) for lat, lon in hurricane[:, :2]],
        dtype=float,
    )

    results: Dict[str, pd.DataFrame] = {}
    for sheet_name, df in cable_segments.items():
        if df.shape[1] < 2:
            raise ValueError(f"CableSeg sheet '{sheet_name}' needs at least two coordinate columns")

        coords = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        row_count = coords.shape[0]
        rainfall = np.full((row_count, sample_hour), np.nan, dtype=float)
        has_valid = ~np.isnan(coords).any(axis=1)

        if has_valid.any():
            for hour in range(sample_hour):
                hurr_xy = hurricane_xy[hour]
                vectors = coords[has_valid] - hurr_xy
                distances = np.linalg.norm(vectors, axis=1)
                base_angle = (np.degrees(np.arctan2(vectors[:, 0], vectors[:, 1])) + 360) % 360
                sector_angle = (base_angle + float(hurricane[hour, 8])) % 360
                rain_values = rain_calc.calculate_rain(
                    hurricane[hour, 5],
                    distances,
                    sector_angle,
                    hurricane[hour, 3],
                    dpdt[hour],
                    hurricane[hour, 8],
                )
                rainfall[has_valid, hour] = rain_values

        results[sheet_name] = pd.DataFrame(rainfall, columns=columns)

    return results


def write_cable_rainfall_workbook(results: Dict[str, pd.DataFrame], output_path: Path | str) -> None:
    """Persist rainfall simulations into an Excel workbook."""

    if not results:
        return

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for idx, (sheet_name, frame) in enumerate(results.items(), start=1):
            safe_name = _sanitize_sheet_name(sheet_name, idx)
            frame.to_excel(writer, sheet_name=safe_name, index=False)


def _calculate_cable_segment_rainfall(
    segment_df: pd.DataFrame,
    hurricane: np.ndarray,
    sample_hour: int,
) -> np.ndarray:
    rain_calc = RainCalculator()
    delta_p = hurricane[:, 3]
    dpdt = np.empty(sample_hour, dtype=float)
    dpdt[0] = 1.0
    dpdt[1:] = delta_p[:-1] - delta_p[1:]
    hurricane_xy = np.array(
        [_convert_hurricane_coordinates(lat, lon) for lat, lon in hurricane[:, :2]],
        dtype=float,
    )

    coords = segment_df[["locationx", "locationy"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    row_count = coords.shape[0]
    rainfall = np.full((row_count, sample_hour), 0.0, dtype=float)
    has_valid = ~np.isnan(coords).any(axis=1)

    if not has_valid.any():
        return rainfall

    for hour in range(sample_hour):
        hurr_xy = hurricane_xy[hour]
        vectors = coords[has_valid] - hurr_xy
        distances = np.linalg.norm(vectors, axis=1)
        base_angle = (np.degrees(np.arctan2(vectors[:, 0], vectors[:, 1])) + 360) % 360
        sector_angle = (base_angle + float(hurricane[hour, 8])) % 360
        rain_values = rain_calc.calculate_rain(
            hurricane[hour, 5],
            distances,
            sector_angle,
            hurricane[hour, 3],
            dpdt[hour],
            hurricane[hour, 8],
        )
        rainfall[has_valid, hour] = rain_values

    return rainfall


def _calculate_cable_failure_probability(
    segment_df: pd.DataFrame,
    rainfall: np.ndarray,
    sample_hour: int,
) -> np.ndarray:
    row_count = segment_df.shape[0]
    failure_prob = np.zeros((row_count, sample_hour), dtype=float)

    for row_idx in range(row_count):
        row = segment_df.iloc[row_idx]
        try:
            CN = float(row["CN"])
            n = float(row["n"])
            S0 = float(row["S0"])
            Z = float(row["Z"])
            phi = float(row["phi"])
            theta_init = float(row["theta"])
            Pbase = float(row["Pbase"]) / 10.0
        except (KeyError, ValueError) as exc:
            print(f"[cable] Warning: row {row_idx} invalid parameters ({exc}), skipping")
            continue

        if CN <= 0 or phi <= 0 or Z <= 0 or S0 <= 0:
            print(f"[cable] Warning: row {row_idx} invalid values (CN={CN}, phi={phi}, Z={Z}, S0={S0})")
            continue

        P_cum = 0.0
        Q_cum_prev = 0.0
        theta_current = theta_init
        S = 25400.0 / CN - 254.0
        Ia = 0.2 * S

        for t in range(sample_hour):
            delta_P = rainfall[row_idx, t]
            P_cum += delta_P

            if P_cum <= Ia:
                Q_cum = 0.0
            else:
                Q_cum = (P_cum - Ia) ** 2 / ((P_cum - Ia) + S)

            delta_Q = Q_cum - Q_cum_prev
            Q_cum_prev = Q_cum

            delta_F = max(0.0, delta_P - delta_Q)
            q_rate = delta_Q / 1000.0 / 3600.0
            if q_rate > 0 and S0 > 0:
                h_t = (q_rate * n / np.sqrt(S0)) ** 0.6
            else:
                h_t = 0.0

            theta_current += delta_F / (Z * 1000.0)
            theta_current = min(theta_current, phi)

            P_f = Pbase * np.exp(CABLE_K1 * h_t + CABLE_K2 * (theta_current / phi))
            failure_prob[row_idx, t] = min(P_f, 1.0)

    return failure_prob


def _calculate_cable_system_failure_probability(
    segment_failure_probs: np.ndarray,
) -> np.ndarray:
    if segment_failure_probs.size == 0:
        return np.zeros(segment_failure_probs.shape[1] if segment_failure_probs.ndim > 1 else 0)

    survival_probs = 1.0 - segment_failure_probs
    system_survival = np.prod(survival_probs, axis=0)
    return 1.0 - system_survival


def _convert_hurricane_coordinates(latitude: float, longitude: float) -> np.ndarray:
    origin = ORIGIN_LAT_LON if longitude >= 0 else LEGACY_ORIGIN_LAT_LON
    return latlon_to_xy(latitude, longitude, origin)


def _sanitize_sheet_name(name: str, fallback_index: int) -> str:
    cleaned = (name or f"Sheet{fallback_index:03d}").strip()
    if not cleaned:
        cleaned = f"Sheet{fallback_index:03d}"
    return cleaned[:31]
