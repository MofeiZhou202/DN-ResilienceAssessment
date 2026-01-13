"""Utility to consolidate Typhoon simulation outputs into ML-friendly datasets.

This script ingests the Excel workbooks produced by the existing pipeline
and emits either a compressed ``.npz`` bundle or a set of ``.parquet`` tables
with harmonised schemas.  The goal is to provide a single entry-point for
model-training data preparation so downstream experiments can read a consistent
format without re-implementing the extraction logic each time.

Usage examples
--------------

Generate a compressed numpy archive (default)::

    python prepare_dataset.py \
        --hurricane-dir output --hurricane-dir temp \
        --impact impact_assessment_simplified.xlsx \
        --wind wind_farms_output.xlsx \
        --mc mc_simulation_results.xlsx \
        --output data/typhoon_dataset.npz

Produce parquet tables instead (requires ``pyarrow`` or ``fastparquet``)::

    python prepare_dataset.py --format parquet --output data/parquet_dataset

The parquet mode will create multiple files in the target path, one for each
logical table (typhoon lifecycle, line impacts, wind farm output, Monte Carlo
states, etc.).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from src.config import GENERATED_DATA_DIR
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hurricane lifecycle ingestion
# ---------------------------------------------------------------------------


def _collect_hurricane_lifecycle(
    directories: Iterable[Path],
    horizon: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, List]]:
    """Load per-hour lifecycle features from every Excel file in *directories*.

    Returns a dense ``(N, horizon, features)`` tensor padded with ``NaN`` where
    individual simulations are shorter than the requested horizon.  Metadata
    is returned alongside so downstream consumers can restore the original
    durations or trace back to the source workbook/sheet.
    """

    entries: List[Dict[str, object]] = []
            "--output",
            default=str(GENERATED_DATA_DIR / "typhoon_dataset.npz"),
            help="输出文件（NPZ）或目录（parquet）。",
        )
            continue
        for workbook in sorted(directory.glob("*.xlsx")):
            try:
                excel = pd.ExcelFile(workbook, engine="openpyxl")
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[WARN] skipping workbook {workbook}: {exc}")
                continue

            for sheet_name in excel.sheet_names:
                try:
                    frame = excel.parse(sheet_name, engine="openpyxl")
                except Exception as exc:  # pragma: no cover
                    print(f"[WARN] skipping {workbook.name}::{sheet_name}: {exc}")
                    continue

                numeric_frame = frame.select_dtypes(include=[float, int])
                if numeric_frame.empty:
                    continue

                entries.append(
                    {
                        "id": f"{workbook.stem}:{sheet_name}",
                        "file": workbook.name,
                        "sheet": sheet_name,
                        "frame": numeric_frame,
                    }
                )

                for column in numeric_frame.columns.astype(str):
                    if column not in feature_order:
                        feature_order.append(column)

    if not entries:
        return np.zeros((0, 0, 0), dtype=float), {
            "ids": [],
            "files": [],
            "sheets": [],
            "lengths": [],
            "feature_names": feature_order,
        }

    max_len = horizon or max(entry["frame"].shape[0] for entry in entries)
    feature_dim = len(feature_order)
    tensor = np.full((len(entries), max_len, feature_dim), np.nan, dtype=float)
    lengths: List[int] = []

    for idx, entry in enumerate(entries):
        frame = entry["frame"].reindex(columns=feature_order)
        data = frame.to_numpy(dtype=float)
        length = min(data.shape[0], max_len)
        tensor[idx, :length, : data.shape[1]] = data[:length]
        lengths.append(int(data.shape[0]))

    metadata = {
        "ids": [entry["id"] for entry in entries],
        "files": [entry["file"] for entry in entries],
        "sheets": [entry["sheet"] for entry in entries],
        "lengths": lengths,
        "feature_names": feature_order,
    }
    return tensor, metadata


# ---------------------------------------------------------------------------
# Impact assessment workbooks
# ---------------------------------------------------------------------------


def _load_matrix_workbook(path: Path) -> Dict[str, Dict[str, object]]:
    """Load a workbook where column 0 holds identifiers and the rest is a grid."""

    if not path.exists():
        return {}

    excel = pd.ExcelFile(path, engine="openpyxl")
    sheet_data: Dict[str, Dict[str, object]] = {}

    for sheet in excel.sheet_names:
        frame = excel.parse(sheet, engine="openpyxl")
        ids = frame.iloc[:, 0].astype(str).tolist()
        matrix = frame.iloc[:, 1:].to_numpy(dtype=float)
        time_labels = frame.columns[1:].astype(str).tolist()
        sheet_data[sheet] = {
            "ids": ids,
            "matrix": matrix,
            "time_labels": time_labels,
        }

    return sheet_data


# ---------------------------------------------------------------------------
# Monte Carlo workbook parsing
# ---------------------------------------------------------------------------


def _extract_mc_section(
    raw: pd.DataFrame,
    start_idx: int,
    batch_size: int,
    component_count: int,
    horizon: int,
) -> Tuple[np.ndarray, List[str]]:
    """Slice a (sample, component, time) block from the raw MC sheet."""

    block = raw.iloc[start_idx + 1 : start_idx + 1 + batch_size * component_count, :]
    component_labels = block.iloc[:component_count, 1].astype(str).tolist()
    values = block.iloc[:, 2 : 2 + horizon].to_numpy(dtype=float)
    cube = values.reshape(batch_size, component_count, horizon)
    return cube, component_labels


def _load_mc_results(path: Path) -> Dict[str, Dict[str, object]]:
    """Parse the pseudo Monte-Carlo workbook into structured arrays."""

    if not path.exists():
        return {}

    results: Dict[str, Dict[str, object]] = {}
    excel = pd.ExcelFile(path, engine="openpyxl")

    for sheet in excel.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, engine="openpyxl", header=None)

        summary_rows = raw.loc[1:5, :2].dropna(how="all")
        summary = {str(row[0]): int(row[1]) for _, row in summary_rows.iterrows()}

        batch_size = summary.get("batch_size", 0)
        line_count = summary.get("line_count", 0)
        generator_count = summary.get("generator_count", 0)
        wind_count = summary.get("wind_farm_count", 0)
        horizon = summary.get("time_horizon", 0)

        faulty_header = raw.index[
            (raw.iloc[:, 0] == "Sample") & (raw.iloc[:, 1] == "FaultyLineCount")
        ]
        faulty_counts = (
            raw.iloc[faulty_header[0] + 1 : faulty_header[0] + 1 + batch_size, 1]
            .to_numpy(dtype=int)
            if len(faulty_header) and batch_size
            else np.zeros(0, dtype=int)
        )

        component_headers = [
            idx
            for idx, row in raw.iterrows()
            if row.iloc[0] == "Sample" and row.iloc[1] == "Component"
        ]

        line_cube, line_labels = (
            _extract_mc_section(raw, component_headers[0], batch_size, line_count, horizon)
            if len(component_headers) > 0 and line_count
            else (np.zeros((0, 0, 0)), [])
        )
        gen_cube, gen_labels = (
            _extract_mc_section(raw, component_headers[1], batch_size, generator_count, horizon)
            if len(component_headers) > 1 and generator_count
            else (np.zeros((0, 0, 0)), [])
        )
        wind_cube, wind_labels = (
            _extract_mc_section(raw, component_headers[2], batch_size, wind_count, horizon)
            if len(component_headers) > 2 and wind_count
            else (np.zeros((0, 0, 0)), [])
        )

        results[sheet] = {
            "summary": summary,
            "faulty_counts": faulty_counts,
            "line_states": line_cube,
            "line_labels": line_labels,
            "generator_states": gen_cube,
            "generator_labels": gen_labels,
            "wind_states": wind_cube,
            "wind_labels": wind_labels,
        }

    return results


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


def _ensure_parquet_engine() -> None:
    for module_name in ("pyarrow", "fastparquet"):
        if importlib.util.find_spec(module_name) is not None:
            return

    raise RuntimeError(
        "需要安装 'pyarrow' 或 'fastparquet' 以导出 parquet 文件。"
    )


def _build_long_dataframe(
    cube: np.ndarray,
    sample_labels: List[str],
    component_labels: List[str],
    sheet: str,
    value_name: str,
) -> pd.DataFrame:
    if cube.size == 0:
        return pd.DataFrame(columns=["sheet", "sample", "component", "step", value_name])

    batch, component_count, horizon = cube.shape
    samples = np.repeat(sample_labels, component_count * horizon)
    components = np.tile(
        np.repeat(component_labels, horizon),
        batch,
    )
    steps = np.tile(np.arange(horizon, dtype=int), batch * component_count)
    values = cube.reshape(-1)

    return pd.DataFrame(
        {
            "sheet": sheet,
            "sample": samples,
            "component": components,
            "step": steps,
            value_name: values,
        }
    )


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def build_dataset(args: argparse.Namespace) -> None:
    hurricane_dirs = [Path(path) for path in args.hurricane_dir]
    lifecycle_tensor, lifecycle_meta = _collect_hurricane_lifecycle(
        hurricane_dirs, horizon=args.horizon
    )

    impact_data = _load_matrix_workbook(Path(args.impact))
    wind_data = _load_matrix_workbook(Path(args.wind))
    mc_data = _load_mc_results(Path(args.mc))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "npz":
        payload = {
            "lifecycle_tensor": lifecycle_tensor,
            "lifecycle_meta": np.array([lifecycle_meta], dtype=object),
            "impact_data": np.array([impact_data], dtype=object),
            "wind_data": np.array([wind_data], dtype=object),
            "mc_data": np.array([mc_data], dtype=object),
        }
        np.savez_compressed(output_path, **payload)
        print(f"[OK] generated NPZ dataset: {output_path}")
        return

    # Parquet export -----------------------------------------------------
    _ensure_parquet_engine()
    if output_path.suffix:
        raise ValueError("parquet 模式需要提供目录路径，而不是单个文件名。")
    output_path.mkdir(parents=True, exist_ok=True)

    # Typhoon lifecycle table
    lifecycle_records: List[pd.DataFrame] = []
    feature_names = lifecycle_meta.get("feature_names", [])
    for idx, sample_id in enumerate(lifecycle_meta.get("ids", [])):
        valid_length = lifecycle_meta["lengths"][idx]
        frame = pd.DataFrame(
            lifecycle_tensor[idx, :valid_length, : len(feature_names)],
            columns=feature_names,
        )
        frame.insert(0, "step", np.arange(valid_length, dtype=int))
        frame.insert(0, "sheet", lifecycle_meta["sheets"][idx])
        frame.insert(0, "source_file", lifecycle_meta["files"][idx])
        frame.insert(0, "sample_id", sample_id)
        lifecycle_records.append(frame)

    if lifecycle_records:
        typhoon_df = pd.concat(lifecycle_records, ignore_index=True)
        typhoon_df.to_parquet(output_path / "typhoon_lifecycle.parquet", index=False)

    # Impact lines and wind
    if impact_data:
        line_records: List[pd.DataFrame] = []
        for sheet, payload in impact_data.items():
            matrix = payload["matrix"]
            ids = payload["ids"]
            horizon = matrix.shape[1]
            steps = np.arange(horizon, dtype=int)
            df = pd.DataFrame(
                matrix.reshape(len(ids), horizon),
                index=ids,
                columns=steps,
            )
            long_df = (
                df.reset_index()
                .melt(id_vars="index", var_name="step", value_name="line_fail_prob")
                .rename(columns={"index": "line_id"})
            )
            long_df.insert(0, "sheet", sheet)
            line_records.append(long_df)

        pd.concat(line_records, ignore_index=True).to_parquet(
            output_path / "impact_line_fail_prob.parquet", index=False
        )

    if wind_data:
        wind_records: List[pd.DataFrame] = []
        for sheet, payload in wind_data.items():
            matrix = payload["matrix"]
            ids = payload["ids"]
            horizon = matrix.shape[1]
            steps = np.arange(horizon, dtype=int)
            df = pd.DataFrame(
                matrix.reshape(len(ids), horizon),
                index=ids,
                columns=steps,
            )
            long_df = (
                df.reset_index()
                .melt(id_vars="index", var_name="step", value_name="wind_output")
                .rename(columns={"index": "wind_farm"})
            )
            long_df.insert(0, "sheet", sheet)
            wind_records.append(long_df)

        pd.concat(wind_records, ignore_index=True).to_parquet(
            output_path / "wind_farm_output.parquet", index=False
        )

    # Monte Carlo tables
    for sheet, payload in mc_data.items():
        summary = payload.get("summary", {})
        batch_size = summary.get("batch_size", payload.get("faulty_counts", np.array([])).size)
        sample_labels = [f"Sample_{i+1}" for i in range(batch_size)]

        faulty_df = pd.DataFrame(
            {
                "sheet": sheet,
                "sample": sample_labels,
                "faulty_line_count": payload.get("faulty_counts", np.array([])),
            }
        )
        faulty_df.to_parquet(output_path / f"mc_faulty_counts_sheet_{sheet}.parquet", index=False)

        lines_df = _build_long_dataframe(
            payload.get("line_states", np.zeros((0, 0, 0))),
            sample_labels,
            payload.get("line_labels", []),
            sheet,
            "line_state",
        )
        if not lines_df.empty:
            lines_df.to_parquet(
                output_path / f"mc_line_states_sheet_{sheet}.parquet",
                index=False,
            )

        gens_df = _build_long_dataframe(
            payload.get("generator_states", np.zeros((0, 0, 0))),
            sample_labels,
            payload.get("generator_labels", []),
            sheet,
            "generator_state",
        )
        if not gens_df.empty:
            gens_df.to_parquet(
                output_path / f"mc_generator_states_sheet_{sheet}.parquet",
                index=False,
            )

        wind_df = _build_long_dataframe(
            payload.get("wind_states", np.zeros((0, 0, 0))),
            sample_labels,
            payload.get("wind_labels", []),
            sheet,
            "wind_state",
        )
        if not wind_df.empty:
            wind_df.to_parquet(
                output_path / f"mc_wind_states_sheet_{sheet}.parquet",
                index=False,
            )

    print(f"[OK] wrote parquet tables to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate Typhoon simulation outputs")
    parser.add_argument(
        "--hurricane-dir",
        action="append",
        default=["output"],
        help="包含台风生命周期 Excel 文件的目录，可多次提供。",
    )
    parser.add_argument(
        "--impact",
        default="impact_assessment_simplified.xlsx",
        help="输电线路影响结果的工作簿路径。",
    )
    parser.add_argument(
        "--wind",
        default="wind_farms_output.xlsx",
        help="风电场影响结果的工作簿路径。",
    )
    parser.add_argument(
        "--mc",
        default="mc_simulation_results.xlsx",
        help="蒙特卡洛抽样结果的工作簿路径。",
    )
    parser.add_argument(
        "--output",
        default="data/typhoon_dataset.npz",
        help="输出文件（NPZ）或目录（parquet）。",
    )
    parser.add_argument(
        "--format",
        choices=["npz", "parquet"],
        default="npz",
        help="输出格式。",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="可选，统一台风生命周期的时间步长（默认使用数据中的最大长度）。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
