"""Unified entry point for typhoon simulation workflows.

The original project scattered runnable scripts across the repository. This
launcher exposes the major workflows the user requested:

1. Typhoon lifecycle generation via physical simulation.
2. Transmission-line failure probability estimation.
3. Wind farm output estimation during typhoon events.
4. Pseudo Monte Carlo sampling that combines lifecycle outputs and random failures.
5. Clustering of Monte Carlo results.

Supplementary helpers include a dedicated neural-network-based typhoon generator
and the auto-eval pipeline that chains the individual stages together with shared
artifacts.

Each workflow can be invoked through a dedicated sub-command or via the
interactive menu shown when the script runs without arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import numpy as np

from src.config import (
    CLUSTER_RESULTS_DIR,
    GENERATED_DATA_DIR,
    LEGACY_ORIGIN_LAT_LON,
    MC_RESULTS_DIR,
    ORIGIN_LAT_LON,
    RAW_DATA_DIR,
    ROOT_DIR,
)
from src.impacts.cable_failure import (
    build_tower_and_cable_segments,
    compute_cable_failure_for_hurricane,
    simulate_cable_rainfall,
    write_cable_rainfall_workbook,
)
from src.impacts.transmission import HurricaneImpactsOnTransmissionLines
from src.impacts.wind_farms import HurricaneImpactOnWindFarms
from src.monte_carlo.pseudo_mc_sampling import (
    PseudoMCSampling,
    _resolve_sheet_names as resolve_mc_sheets,
    _write_sheet as write_sampling_sheet,
)
from src.services.random_failure import failprob
from src.typhoon.seasonal import SeasonalHurricaneSimulator
from src.typhoon.initialization import MAX_INIT_LATITUDE
from src.models.train_init_models import (
    INPUT_COLUMNS as NN_INPUT_COLUMNS,
    TARGET_COLUMNS as NN_TARGET_COLUMNS,
    Normalization as NNNormalization,
    Regressor as NNRegressor,
)
from src.utils.coordinates import latlon_to_xy

DEFAULT_MONTHS = [6, 7, 8, 9, 10, 11]
DEFAULT_NN_SAMPLE = {
    "InitLatitude": 21.3979353,
    "InitLongitude": 116.6839997,
    "InitDeltaP": 46.12501565,
    "InitIR": 0.556285239,
    "InitRmw": 44.7543777,
    "InitTheta": 1.165574874,
    "InitTransSpeed": 5.400026797,
}
DEFAULT_CLUSTER_SAMPLE_ROWS = 32
DEFAULT_CLUSTER_COUNT = 100
DEFAULT_CLUSTER_COMPONENT_PREFIX = "Line_"
DEFAULT_CLUSTER_COVERAGE_TARGETS = [10, 20, 50, 100, 200]
DEFAULT_CLUSTER_COVERAGE_THRESHOLDS = [0.0, 0.02, 0.05, 0.1]
DEFAULT_NN_CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "init_models"
DEFAULT_TY_DURATION = 48
DEFAULT_TY_YEAR = 2025
DEFAULT_NN_JITTER = 0.05
SAMPLE_LINEFAIL_WORKBOOK = ROOT_DIR / "dataset" / "generated" / "impact" / "test_linefail.xlsx"
SAMPLE_WIND_WORKBOOK = ROOT_DIR / "dataset" / "generated" / "impact" / "test_wind.xlsx"
DEFAULT_CABLE_RAIN_OUTPUT = GENERATED_DATA_DIR / "impact" / "cable_rainfall.xlsx"


def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[-1]


DEFAULT_NN_DATA_ROOT = _first_existing_path(
    ROOT_DIR / "dataset" / "classified_typhoons",
    ROOT_DIR / "new_dataset" / "classified_typhoons",
    ROOT_DIR.parent / "new_dataset" / "classified_typhoons",
)
DEFAULT_CLUSTER_WORKBOOK = _first_existing_path(
    MC_RESULTS_DIR / "mc_simulation_results.xlsx",
    ROOT_DIR / "new_dataset" / "MC_results" / "mc_simulation_results.xlsx",
    ROOT_DIR.parent / "new_dataset" / "MC_results" / "mc_simulation_results.xlsx",
)
DEFAULT_CLUSTER_OUTPUT_DIR = CLUSTER_RESULTS_DIR
AUTO_EVAL_ROOT = ROOT_DIR / "dataset" / "auto_eval_runs"


def _prompt_yes_no(message: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    prompt = f"{message} ({hint}): "
    while True:
        try:
            choice = input(prompt).strip().lower()
        except EOFError:
            return default
        if not choice:
            return default
        if choice in {"y", "yes", "1"}:
            return True
        if choice in {"n", "no", "0"}:
            return False
        print("\u8bf7\u8f93\u5165 y/n")


def _prompt_text(message: str, default: str | Path | None = None) -> str | None:
    hint_value = "-" if default in (None, "") else str(default)
    try:
        response = input(f"{message} [{hint_value}]: ").strip()
    except EOFError:
        return str(default) if default is not None else None
    if response:
        return response
    return str(default) if default is not None else None


def _prompt_int(message: str, default: int | None = None) -> int | None:
    while True:
        raw = _prompt_text(message, str(default) if default is not None else None)
        if raw in (None, "", str(default)) and default is not None:
            return default
        if raw in (None, ""):
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            print("\u8bf7\u8f93\u5165\u6574\u6570")


def _prompt_float(message: str, default: float | None = None) -> float | None:
    while True:
        raw = _prompt_text(message, str(default) if default is not None else None)
        if raw in (None, "", str(default)) and default is not None:
            return default
        if raw in (None, ""):
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            print("\u8bf7\u8f93\u5165\u6570\u503c")


def _prompt_int_list(message: str, default: Sequence[int]) -> list[int]:
    default_str = " ".join(str(x) for x in default)
    while True:
        raw = _prompt_text(message, default_str)
        if not raw:
            return list(default)
        tokens = raw.replace(",", " ").split()
        try:
            return [int(tok) for tok in tokens]
        except ValueError:
            print("\u8bf7\u4f7f\u7528\u7a7a\u683c\u6216\u9017\u53f7\u5206\u9694\u7684\u6574\u6570")


def _prompt_float_list(message: str, default: Sequence[float]) -> list[float]:
    default_str = " ".join(str(x) for x in default)
    while True:
        raw = _prompt_text(message, default_str)
        if not raw:
            return list(default)
        tokens = raw.replace(",", " ").split()
        try:
            return [float(tok) for tok in tokens]
        except ValueError:
            print("\u8bf7\u4f7f\u7528\u7a7a\u683c\u6216\u9017\u53f7\u5206\u9694\u7684\u6570\u503c")


def _prompt_choice(message: str, choices: Sequence[str], default: str) -> str:
    choice_hint = ", ".join(choices)
    while True:
        raw = _prompt_text(f"{message} ({choice_hint})", default)
        if not raw:
            return default
        normalized = raw.lower()
        if normalized in choices:
            return normalized
        print("\u8bf7\u9009\u62e9\u5408\u6cd5\u9009\u9879")


def _prompt_string_list(message: str, default: Sequence[str] | None = None) -> list[str]:
    default_str = ",".join(default) if default else ""
    raw = _prompt_text(message, default_str if default_str else None)
    if not raw:
        return list(default) if default else []
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]


def _prompt_required_text(message: str, default: str | None = None) -> str:
    while True:
        value = _prompt_text(message, default)
        if value:
            return value
        print("\u6b64\u9879\u4e3a\u5fc5\u586b\uff0c\u8bf7\u8f93\u5165\u6709\u6548\u5185\u5bb9")


def _prompt_required_float(message: str, default: float | None = None) -> float:
    while True:
        value = _prompt_float(message, default)
        if value is not None:
            return value
        print("\u6b64\u9879\u4e3a\u5fc5\u586b\uff0c\u8bf7\u8f93\u5165\u6570\u503c")


def _nn_infer_hidden_sizes(state_dict: dict[str, object], output_dim: int) -> list[int]:
    dims: list[int] = []
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if not key.endswith("weight") or not hasattr(tensor, "shape"):
            continue
        shape = getattr(tensor, "shape")
        if len(shape) != 2:
            continue
        dims.append(int(shape[0]))
    if not dims or dims[-1] != output_dim:
        raise ValueError("\u65e0\u6cd5\u6839\u636e checkpoint \u63d0\u53d6\u7f51\u7edc\u5c42\u6570")
    return dims[:-1]


def _nn_load_input_sample(json_path: str | None) -> dict[str, float]:
    sample = dict(DEFAULT_NN_SAMPLE)
    if not json_path:
        return sample
    path = _resolve_path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"\u81ea\u5b9a\u4e49 JSON \u6587\u4ef6\u4e0d\u5b58\u5728: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON \u5fc5\u987b\u662f\u952e\u503c\u5bf9\u5bf9\u8c61")
    for key, value in payload.items():
        if key in sample and value is not None:
            sample[key] = float(value)
    missing = [col for col in NN_INPUT_COLUMNS if col != "Step" and col not in sample]
    if missing:
        raise ValueError(f"\u8f93\u5165\u6837\u672c\u7f3a\u5c11\u5b57\u6bb5: {missing}")
    if sample["InitLatitude"] > MAX_INIT_LATITUDE:
        print(
            f"[nn-typhoon] InitLatitude {sample['InitLatitude']:.3f} > {MAX_INIT_LATITUDE}, \u5df2\u81ea\u52a8\u622a\u65ad"
        )
        sample["InitLatitude"] = MAX_INIT_LATITUDE
    return sample


def _nn_sample_for_run(base_sample: dict[str, float], rng: np.random.Generator, jitter: float) -> dict[str, float]:
    if jitter <= 0.0:
        return dict(base_sample)
    perturbed: dict[str, float] = {}
    for key, value in base_sample.items():
        scale = 1.0 + rng.normal(0.0, jitter)
        new_value = max(0.0, float(value) * scale)
        if key == "InitLatitude":
            new_value = min(new_value, MAX_INIT_LATITUDE)
        perturbed[key] = new_value
    return perturbed


def _resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _resolve_workbook(
    path_like: str | Path,
    *,
    description: str,
    fallback: str | Path | None = None,
    tag: str = "mc",
) -> tuple[Path, bool]:
    primary = _resolve_path(path_like)
    if primary.exists() and primary.stat().st_size > 0:
        return primary, False
    if fallback is not None:
        fallback_path = _resolve_path(fallback)
        if fallback_path.exists() and fallback_path.stat().st_size > 0:
            if primary != fallback_path:
                print(
                    f"[{tag}] {description} workbook '{primary}' missing or empty, switched to sample '{fallback_path}'"
                )
            return fallback_path, True
    state = "missing" if not primary.exists() else "empty"
    raise FileNotFoundError(f"[{tag}] {description} workbook {state}: {primary}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sanitize_sheet_name(name: str, fallback_index: int) -> str:
    cleaned = (name or f"Sheet{fallback_index:03d}").strip()
    if not cleaned:
        cleaned = f"Sheet{fallback_index:03d}"
    return cleaned[:31]


def _prepare_run_directory(base_dir: Path, run_name: str | None) -> tuple[Path, str]:
    base_dir = _resolve_path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_label = run_name.strip() if run_name else f"run_{timestamp}"
    candidate = base_label
    counter = 1
    run_path = base_dir / candidate
    while run_path.exists():
        candidate = f"{base_label}_{counter:02d}"
        run_path = base_dir / candidate
        counter += 1
    run_path.mkdir(parents=True, exist_ok=False)
    return run_path, candidate


def _write_json(path: Path, payload: Dict[str, object]) -> Path:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _normalize_months(months: Sequence[int]) -> list[int]:
    normalized: list[int] = []
    for month in months:
        if month < 1 or month > 12:
            raise ValueError("月份必须在1-12之间")
        if month not in normalized:
            normalized.append(month)
    return normalized or list(DEFAULT_MONTHS)



def _load_hurricane_sheets(
    hurricane_file: Path,
    sheets: Sequence[str] | None,
) -> List[tuple[str, pd.DataFrame]]:
    hurricane_file = _resolve_path(hurricane_file)
    if not hurricane_file.exists():
        raise FileNotFoundError(f"Hurricane workbook not found: {hurricane_file}")

    sheet_items: List[tuple[str, pd.DataFrame]] = []
    with pd.ExcelFile(hurricane_file) as workbook:
        available = workbook.sheet_names
        if sheets:
            missing = [name for name in sheets if name not in available]
            if missing:
                raise ValueError(
                    f"Requested hurricane sheet(s) missing in {hurricane_file.name}: {missing}"
                )
            target = list(dict.fromkeys(sheets))
        else:
            target = available
        for sheet_name in target:
            sheet_items.append((sheet_name, workbook.parse(sheet_name=sheet_name)))
    if not sheet_items:
        raise ValueError(f"Workbook {hurricane_file} does not contain any data sheets.")
    return sheet_items

def _parse_int_tokens(values: Sequence[str] | None) -> list[int]:
    numbers: list[int] = []
    if not values:
        return numbers
    for entry in values:
        text = str(entry).replace(";", ",")
        for token in text.split(","):
            stripped = token.strip()
            if not stripped:
                continue
            try:
                numbers.append(int(stripped))
            except ValueError as exc:
                raise ValueError(f"无效的负荷编号: {token}") from exc
    return numbers

def _ensure_random_failure_workbook(path: Path) -> Path:
    path = _resolve_path(path)
    if path.exists():
        return path

    _ensure_parent(path)
    failure_prob = failprob()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for key, values in failure_prob.items():
            df = pd.DataFrame({key: values})
            df.to_excel(writer, sheet_name=key[:31], index=False)
    print(f"[random-failure] Generated baseline workbook at {path}")
    return path


def run_typhoon_lifecycle(args: argparse.Namespace) -> None:
    if args.month < 1 or args.month > 12:
        raise ValueError("\u6708\u4efd\u5fc5\u987b\u57281-12\u4e4b\u95f4")
    months = [int(args.month)]
    storm_count = max(1, int(args.storms))
    simulator = SeasonalHurricaneSimulator(
        start_year=DEFAULT_TY_YEAR,
        end_year=DEFAULT_TY_YEAR,
        storms_per_year=storm_count,
        months=months,
        sim_duration=DEFAULT_TY_DURATION,
        rng_seed=args.seed,
    )
    output_path = _resolve_path(args.output)
    _ensure_parent(output_path)
    result_path = simulator.to_excel(output_path)
    print(f"[typhoon] Generated seasonal hurricanes -> {result_path}")


def run_transmission_impacts(args: argparse.Namespace) -> None:
    tower_seg, cable_ids, cable_segments = build_tower_and_cable_segments(args.tower_excel)
    hurricane_sheets = _load_hurricane_sheets(args.hurricane_file, args.sheets)
    output_path = _resolve_path(args.output)
    _ensure_parent(output_path)
    reference_hurricane: np.ndarray | None = None
    reference_sheet: str | None = None

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for idx, (sheet_name, df) in enumerate(hurricane_sheets, start=1):
            hurricane = df.to_numpy(dtype=float)
            if hurricane.size == 0:
                print(f"[impacts] Skipping empty sheet '{sheet_name}'")
                continue
            if reference_hurricane is None:
                reference_hurricane = hurricane
                reference_sheet = sheet_name
            sample_hour = hurricane.shape[0]
            records: list[np.ndarray] = []
            line_ids: list[int] = []

            if tower_seg:
                simulator = HurricaneImpactsOnTransmissionLines(tower_seg, hurricane, sample_hour)
                result = simulator.calculate_impact()
                for entry in result:
                    records.append(entry["linefailprob"].flatten())
                    line_ids.append(entry["lineid"])

            if cable_segments:
                cable_failure_results = compute_cable_failure_for_hurricane(
                    cable_segments, hurricane, sample_hour
                )
                for cable_id in sorted(cable_failure_results.keys()):
                    records.append(cable_failure_results[cable_id])
                    line_ids.append(cable_id)

            if not records:
                print(f"[impacts] No results for sheet '{sheet_name}'")
                continue
            df_result = pd.DataFrame(records, index=line_ids)
            df_result.index.name = "LineId"
            df_result.columns = [f"Hour_{i + 1}" for i in range(df_result.shape[1])]
            safe_name = _sanitize_sheet_name(sheet_name, idx)
            df_result.to_excel(writer, sheet_name=safe_name)
            cable_note = f" + cables={len(cable_segments)}" if cable_segments else ""
            print(f"[impacts] {safe_name}: lines={len(line_ids)} hours={sample_hour}{cable_note}")
    cable_rain_output = getattr(args, "cable_rain_output", None)
    if cable_segments and cable_rain_output:
        if reference_hurricane is None:
            print("[impacts] Cable rainfall skipped: hurricane workbook contained no data")
        else:
            rainfall_results = simulate_cable_rainfall(cable_segments, reference_hurricane)
            if rainfall_results:
                cable_rain_path = _resolve_path(cable_rain_output)
                write_cable_rainfall_workbook(rainfall_results, cable_rain_path)
                suffix = ""
                if reference_sheet and len(hurricane_sheets) > 1:
                    suffix = f" (based on {reference_sheet})"
                print(f"[impacts] Cable rainfall saved{suffix} -> {cable_rain_path}")
    print(f"[impacts] Saved line failure workbook -> {output_path}")


def run_wind_impacts(args: argparse.Namespace) -> None:
    wind_excel = _resolve_path(args.wind_excel)
    if not wind_excel.exists():
        raise FileNotFoundError(f"Wind farm workbook not found: {wind_excel}")
    wind_farm_location_df = pd.read_excel(
        wind_excel,
        sheet_name=args.location_sheet,
        header=None,
    )
    hurricane_sheets = _load_hurricane_sheets(args.hurricane_file, args.sheets)
    output_path = _resolve_path(args.output)
    _ensure_parent(output_path)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for idx, (sheet_name, df) in enumerate(hurricane_sheets, start=1):
            hurricane = df.to_numpy(dtype=float)
            if hurricane.size == 0:
                print(f"[wind] Skipping empty sheet '{sheet_name}'")
                continue
            simulator = HurricaneImpactOnWindFarms(wind_farm_location_df, hurricane, hurricane.shape[0])
            wind_output = simulator.simulate()
            output_df = pd.DataFrame(
                wind_output,
                index=[f"Farm_{i + 1}" for i in range(wind_output.shape[0])],
                columns=[f"TimeStep_{j + 1}" for j in range(wind_output.shape[1])],
            )
            safe_name = _sanitize_sheet_name(sheet_name, idx)
            output_df.to_excel(writer, sheet_name=safe_name)
            print(f"[wind] \u6a21\u62df {safe_name}")
    print(f"[wind] Saved wind farm output workbook -> {output_path}")


def run_monte_carlo(args: argparse.Namespace) -> None:
    linefail_path, linefail_used_fallback = _resolve_workbook(
        args.linefail,
        description="line failure",
        fallback=SAMPLE_LINEFAIL_WORKBOOK,
    )
    wind_source = args.wind
    if linefail_used_fallback:
        resolved_default_wind = _resolve_path(args.wind)
        if resolved_default_wind != SAMPLE_WIND_WORKBOOK:
            print(
                f"[mc] Wind workbook '{resolved_default_wind}' paired with missing data, switched to sample '{SAMPLE_WIND_WORKBOOK}'"
            )
            wind_source = SAMPLE_WIND_WORKBOOK
    wind_path, _ = _resolve_workbook(
        wind_source,
        description="wind output",
        fallback=SAMPLE_WIND_WORKBOOK,
    )
    random_path = _ensure_random_failure_workbook(args.random)

    sheet_names = resolve_mc_sheets(str(linefail_path), str(wind_path), args.sheet)

    options = {
        "batch_size": args.batch,
        "consider_random_failures": 0 if args.no_random else 1,
        "consider_repair": 1 if args.repair else 0,
        "T_re": args.tre,
        "show_progress": True,
        "progress_updates": 10,
    }

    output_path = _resolve_path(args.output)
    _ensure_parent(output_path)
    summaries = []
    json_payloads = []

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        total = len(sheet_names)
        for idx, sheet in enumerate(sheet_names, start=1):
            sheet_options = dict(options)
            sheet_options["progress_prefix"] = f"[{idx}/{total}][{sheet}]"
            sampler = PseudoMCSampling.from_excel(
                linefail_path=str(linefail_path),
                random_failure_path=str(random_path),
                wind_output_path=str(wind_path),
                sheet_name=sheet,
                options=sheet_options,
            )
            print(f"[mc] Simulating sheet '{sheet}' ({idx}/{total}) ...")
            sampler.simulate()
            system_state, _component_failure, faulty_counts = sampler.get_results()
            write_sampling_sheet(writer, sheet, sampler, system_state, faulty_counts)
            summary_entry = {
                "sheet": sheet,
                "line_count": sampler.nl,
                "generator_count": sampler.ng,
                "wind_farm_count": sampler.nw,
                "time_horizon": sampler.T,
                "faulty_mean": float(pd.Series(faulty_counts).mean()) if faulty_counts else 0.0,
                "faulty_min": int(min(faulty_counts)) if faulty_counts else 0,
                "faulty_max": int(max(faulty_counts)) if faulty_counts else 0,
            }
            summaries.append(summary_entry)
            if args.dump_json:
                json_payloads.append(
                    {
                        "sheet": sheet,
                        "batch_size": args.batch,
                        "time_horizon": sampler.T,
                        "line_count": sampler.nl,
                        "generator_count": sampler.ng,
                        "wind_farm_count": sampler.nw,
                        "faulty_line_counts": faulty_counts,
                        "metadata": system_state["metadata"],
                    }
                )
            print(
                f"[mc] {sheet}: faulty_mean={summary_entry['faulty_mean']:.2f}, "
                f"range=[{summary_entry['faulty_min']}, {summary_entry['faulty_max']}]"
            )

    if args.dump_json and json_payloads:
        dump_path = _resolve_path(args.dump_json)
        _ensure_parent(dump_path)
        dump_path.write_text(json.dumps(json_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[mc] JSON summary saved -> {dump_path}")

    print(f"[mc] Monte Carlo workbook saved -> {output_path}")
    for entry in summaries:
        print(
            f"  - {entry['sheet']}: lines={entry['line_count']} horizon={entry['time_horizon']}h"
        )


def run_nn_typhoon(args: argparse.Namespace) -> None:
    import torch
    import sys

    torch.serialization.add_safe_globals([NNNormalization])
    setattr(sys.modules.setdefault("__main__", sys.modules[__name__]), "Normalization", NNNormalization)

    checkpoint_dir = _resolve_path(args.checkpoint_dir)
    data_root = _resolve_path(args.data_root)
    output_path = _resolve_path(args.output)
    _ensure_parent(output_path)

    sample = _nn_load_input_sample(args.input_json)
    rng = np.random.default_rng(args.seed)
    jitter = max(0.0, args.jitter if args.jitter is not None else DEFAULT_NN_JITTER)
    missing_fields = [col for col in NN_INPUT_COLUMNS if col != "Step" and col not in sample]
    if missing_fields:
        raise ValueError(f"\u8f93\u5165\u53c2\u6570\u7f3a\u5c11\u5b57\u6bb5: {missing_fields}")

    levels = args.levels or []
    if not levels:
        raise ValueError("\u81f3\u5c11\u6307\u5b9a\u4e00\u4e2a\u53f0\u98ce\u7ea7\u522b")

    count = max(1, int(args.count or 1))
    frames: dict[str, pd.DataFrame] = {}
    for level in levels:
        level = level.strip()
        if not level:
            continue
        checkpoint_path = checkpoint_dir / f"{level}_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{level} \u6a21\u578b checkpoint \u4e0d\u5b58\u5728: {checkpoint_path}")
        workbook = _resolve_path(data_root / f"{level}.xlsx")
        if not workbook.exists():
            raise FileNotFoundError(f"\u672a\u627e\u5230 {level} \u53f0\u98ceExcel: {workbook}")
        step_list = list(range(DEFAULT_TY_DURATION))

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict") or payload
        feature_norm: NNNormalization = payload["feature_norm"]
        target_norm: NNNormalization = payload["target_norm"]

        hidden_sizes = _nn_infer_hidden_sizes(state_dict, len(NN_TARGET_COLUMNS))
        model = NNRegressor(len(NN_INPUT_COLUMNS), hidden_sizes, len(NN_TARGET_COLUMNS))
        model.load_state_dict(state_dict)
        model.eval()

        for run_idx in range(1, count + 1):
            run_sample = _nn_sample_for_run(sample, rng, jitter)
            rows: list[dict[str, float]] = []
            with torch.no_grad():
                for step in step_list:
                    feature_vector: list[float] = []
                    for col in NN_INPUT_COLUMNS:
                        if col == "Step":
                            feature_vector.append(float(step))
                        else:
                            feature_vector.append(float(run_sample[col]))
                    inputs_tensor = torch.tensor(feature_vector, dtype=torch.float32)
                    normalized_inputs = feature_norm.normalize(inputs_tensor)
                    normalized_output = model(normalized_inputs.unsqueeze(0)).squeeze(0)
                    predicted = target_norm.denormalize(normalized_output)
                    row = {col: float(value) for col, value in zip(NN_TARGET_COLUMNS, predicted.tolist())}
                    rows.append(row)

            frame = pd.DataFrame(rows)
            sheet_key = level if count == 1 else f"{level}_run{run_idx}"
            frames[sheet_key] = frame
            print(f"[nn-typhoon] {sheet_key}: steps={len(step_list)}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for level, frame in frames.items():
            sheet_name = level[:31] if level else "Sheet"
            frame.to_excel(writer, sheet_name=sheet_name or "Sheet", index=False)
    print(f"[nn-typhoon] \u9884\u6d4b\u7ed3\u679c -> {output_path}")


def run_cluster_workflow(args: argparse.Namespace) -> List[Path]:
    try:
        from src.tools import cluster_mc_results as cluster_mod
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "\u6267\u884c\u805a\u7c07\u9700\u8981\u9884\u5148 pip install sklearn-extra scipy matplotlib"
        ) from exc

    workbook_input = _resolve_path(args.workbook)
    if not workbook_input.exists():
        raise FileNotFoundError(f"\u672a\u627e\u5230 MC \u7ed3\u679c Excel: {workbook_input}")

    if workbook_input.is_dir():
        workbooks = cluster_mod.iter_workbooks(workbook_input)
        base_dir = workbook_input
    else:
        if workbook_input.suffix.lower() != ".xlsx":
            raise ValueError(f"\u8bf7\u6307\u5411 Excel \u6587\u4ef6 (.xlsx): {workbook_input}")
        workbooks = [workbook_input]
        base_dir = workbook_input.parent

    if args.workbooks:
        wanted = {Path(name).stem for name in args.workbooks}
        filtered = []
        for workbook in workbooks:
            if workbook.stem in wanted or workbook.name in wanted:
                filtered.append(workbook)
        workbooks = filtered
    if args.limit:
        workbooks = workbooks[: args.limit]

    if not workbooks:
        raise FileNotFoundError("\u6309\u6761\u4ef6\u672a\u627e\u5230 Excel \u6587\u4ef6")

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Path] = []
    for workbook in workbooks:
        try:
            rel_display = workbook.relative_to(base_dir)
        except ValueError:
            rel_display = workbook.name
        print(f"[cluster] \u5f00\u59cb\u5904\u7406 {rel_display}")
        output_path = cluster_mod.process_workbook(
            workbook,
            args.sample_rows,
            args.clusters,
            args.component_prefix,
            args.coverage_targets,
            args.coverage_thresholds,
            output_dir=output_dir,
        )
        results.append(output_path)
    return results

def run_auto_evaluation(args: argparse.Namespace) -> None:
    base_dir = _resolve_path(args.output_root or AUTO_EVAL_ROOT)
    run_dir, run_name = _prepare_run_directory(base_dir, args.run_name)
    print(f"[auto-eval] Output directory: {run_dir}")

    typhoon_dir = run_dir / "typhoon"
    impact_dir = run_dir / "impact"
    mc_dir = run_dir / "monte_carlo"
    cluster_dir = run_dir / "cluster"

    hurricane_path = typhoon_dir / ("nn_typhoons.xlsx" if args.typhoon_source == "nn" else "seasonal_hurricanes.xlsx")
    linefail_path = impact_dir / "linefailprob.xlsx"
    cable_rain_path = Path(args.cable_rain_output) if args.cable_rain_output else impact_dir / "cable_rainfall.xlsx"
    wind_output_path = impact_dir / "wind_farms_output.xlsx"
    mc_output_path = mc_dir / "mc_simulation_results.xlsx"
    mc_dump_json = mc_dir / "mc_summary.json"
    summary_path = run_dir / "auto_eval_summary.json"

    summary: Dict[str, object] = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "typhoon_source": args.typhoon_source,
        "paths": {},
        "inputs": {
            "typhoon_source": args.typhoon_source,
            "months": args.months,
            "storms": args.storms,
            "levels": args.levels,
            "nn_count": args.nn_count,
            "nn_seed": args.nn_seed,
            "nn_jitter": args.nn_jitter,
            "mc_sheet": args.mc_sheet,
            "mc_batch": args.mc_batch,
            "mc_repair": args.mc_repair,
            "mc_no_random": args.mc_no_random,
            "mc_tre": args.mc_tre,
            "cluster_sample_rows": args.cluster_sample_rows,
            "cluster_count": args.cluster_count,
        },
        "steps": [],
    }

    for target_path in (
        hurricane_path,
        linefail_path,
        cable_rain_path,
        wind_output_path,
        mc_output_path,
        mc_dump_json,
        summary_path,
    ):
        _ensure_parent(Path(target_path))

    # 1. Typhoon generation
    if args.typhoon_source == "seasonal":
        months = _normalize_months(args.months or DEFAULT_MONTHS)
        simulator = SeasonalHurricaneSimulator(
            start_year=DEFAULT_TY_YEAR,
            end_year=DEFAULT_TY_YEAR,
            storms_per_year=max(1, int(args.storms)),
            months=months,
            sim_duration=DEFAULT_TY_DURATION,
            rng_seed=args.typhoon_seed,
        )
        result_path = simulator.to_excel(hurricane_path)
        typhoon_meta = {
            "months": months,
            "storms_per_year": max(1, int(args.storms)),
            "duration_hours": DEFAULT_TY_DURATION,
            "seed": args.typhoon_seed,
        }
        print(f"[auto-eval] Seasonal hurricanes saved -> {result_path}")
    else:
        levels = args.levels or ["05_STY"]
        if not levels:
            raise ValueError("nn-typhoon source requires at least one --levels entry")
        nn_args = argparse.Namespace(
            levels=levels,
            data_root=args.nn_data_root,
            checkpoint_dir=args.nn_checkpoint_dir,
            count=max(1, int(args.nn_count)),
            seed=args.nn_seed,
            jitter=args.nn_jitter,
            input_json=args.nn_input_json,
            output=str(hurricane_path),
        )
        run_nn_typhoon(nn_args)
        typhoon_meta = {
            "levels": levels,
            "count": max(1, int(args.nn_count)),
            "seed": args.nn_seed,
            "jitter": args.nn_jitter,
        }
    summary["paths"]["hurricane"] = str(hurricane_path)
    summary["steps"].append({"name": "typhoon", "output": str(hurricane_path), "meta": typhoon_meta})

    # 2. Transmission impacts
    impact_args = argparse.Namespace(
        hurricane_file=str(hurricane_path),
        tower_excel=args.tower_excel,
        output=str(linefail_path),
        cable_rain_output=str(cable_rain_path),
        sheets=None,
    )
    run_transmission_impacts(impact_args)
    summary["paths"]["linefail"] = str(linefail_path)
    summary["paths"]["cable_rain"] = str(cable_rain_path)
    summary["steps"].append({"name": "impact", "output": str(linefail_path)})
    summary["steps"].append({"name": "cable-rain", "output": str(cable_rain_path)})

    # 3. Wind farm outputs
    wind_args = argparse.Namespace(
        hurricane_file=str(hurricane_path),
        wind_excel=args.wind_excel,
        location_sheet=args.wind_location_sheet,
        output=str(wind_output_path),
        sheets=None,
    )
    run_wind_impacts(wind_args)
    summary["paths"]["wind"] = str(wind_output_path)
    summary["steps"].append({"name": "wind", "output": str(wind_output_path)})

    # 4. Monte Carlo simulation
    mc_args = argparse.Namespace(
        linefail=str(linefail_path),
        random=args.random_failure,
        wind=str(wind_output_path),
        sheet=args.mc_sheet,
        batch=max(1, int(args.mc_batch)),
        repair=args.mc_repair,
        tre=int(args.mc_tre),
        no_random=args.mc_no_random,
        output=str(mc_output_path),
        dump_json=str(mc_dump_json),
    )
    run_monte_carlo(mc_args)
    summary["paths"]["monte_carlo"] = str(mc_output_path)
    summary["steps"].append({"name": "monte-carlo", "output": str(mc_output_path), "summary_json": str(mc_dump_json)})

    # 5. Clustering
    cluster_args = argparse.Namespace(
        workbook=str(mc_output_path),
        sample_rows=int(args.cluster_sample_rows),
        clusters=int(args.cluster_count),
        component_prefix=args.cluster_component_prefix,
        coverage_targets=args.cluster_coverage_targets,
        coverage_thresholds=args.cluster_coverage_thresholds,
        output_dir=str(cluster_dir),
        workbooks=None,
        limit=None,
    )
    cluster_outputs = run_cluster_workflow(cluster_args)
    if not cluster_outputs:
        raise RuntimeError("Clustering did not produce any workbook outputs")
    scenario_file = cluster_outputs[0]
    summary["paths"]["cluster"] = [str(path) for path in cluster_outputs]
    summary["steps"].append({"name": "cluster", "output": str(scenario_file)})

    _write_json(summary_path, summary)
    print(f"[auto-eval] Summary saved -> {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Typhoon simulation control center")
    subparsers = parser.add_subparsers(dest="command")

    typhoon = subparsers.add_parser("typhoon", help="Generate typhoon lifecycle Excel files")
    typhoon.add_argument(
        "--month",
        type=int,
        default=9,
        help="Single month to simulate (default 9)",
    )
    typhoon.add_argument("--storms", type=int, default=5, help="Total storms to simulate")
    typhoon.add_argument("--seed", type=int, help="Optional RNG seed for reproducibility")
    typhoon.add_argument(
        "--output",
        default=str(GENERATED_DATA_DIR / "typhoons" / "seasonal_hurricanes.xlsx"),
    )
    typhoon.set_defaults(func=run_typhoon_lifecycle)

    impacts = subparsers.add_parser(
        "impacts",
        help="Compute transmission line failure probabilities",
    )
    impacts.add_argument(
        "--hurricane-file",
        default=str(RAW_DATA_DIR / "hurricane.xlsx"),
    )
    impacts.add_argument(
        "--tower-excel",
        default=str(RAW_DATA_DIR / "TowerSeg.xlsx"),
    )
    impacts.add_argument(
        "--output",
        default=str(GENERATED_DATA_DIR / "impact" / "linefailprob.xlsx"),
    )
    impacts.add_argument(
        "--cable-rain-output",
        default=str(DEFAULT_CABLE_RAIN_OUTPUT),
        help="Excel workbook for CableSeg rainfall data",
    )
    impacts.add_argument(
        "--sheets",
        nargs="+",
        help="Optional subset of hurricane sheet names to process",
    )
    impacts.set_defaults(func=run_transmission_impacts)

    wind = subparsers.add_parser(
        "wind",
        help="Estimate wind farm outputs under typhoon scenarios",
    )
    wind.add_argument(
        "--hurricane-file",
        default=str(RAW_DATA_DIR / "hurricane.xlsx"),
    )
    wind.add_argument(
        "--wind-excel",
        default=str(RAW_DATA_DIR / "wind_farms.xlsx"),
    )
    wind.add_argument("--location-sheet", default="wind_farm_location")
    wind.add_argument(
        "--output",
        default=str(GENERATED_DATA_DIR / "impact" / "wind_farms_output.xlsx"),
    )
    wind.add_argument(
        "--sheets",
        nargs="+",
        help="Optional subset of hurricane sheet names to process",
    )
    wind.set_defaults(func=run_wind_impacts)

    mc = subparsers.add_parser("monte-carlo", help="Run pseudo Monte Carlo sampling")
    mc.add_argument(
        "--linefail",
        default=str(GENERATED_DATA_DIR / "impact" / "linefailprob.xlsx"),
    )
    mc.add_argument(
        "--random",
        default=str(RAW_DATA_DIR / "random_failure_prob.xlsx"),
    )
    mc.add_argument(
        "--wind",
        default=str(GENERATED_DATA_DIR / "impact" / "wind_farms_output.xlsx"),
    )
    mc.add_argument("--sheet", default="all", help="Sheet selection (all or comma list)")
    mc.add_argument("--batch", type=int, default=1000)
    mc.add_argument("--repair", action="store_true", help="Enable repair logic")
    mc.add_argument("--tre", type=int, default=24, help="Repair window length (hours)")
    mc.add_argument("--no-random", action="store_true", help="Disable base random failures")
    mc.add_argument(
        "--output",
        default=str(MC_RESULTS_DIR / "mc_simulation_results.xlsx"),
    )
    mc.add_argument(
        "--cable-rain-output",
        help="Optional CableSeg rainfall workbook path",
    )
    mc.add_argument("--dump-json", help="Optional JSON summary path")
    mc.set_defaults(func=run_monte_carlo)

    nn_typhoon = subparsers.add_parser(
        "nn-typhoon",
        help="Use trained neural nets to generate specific typhoon levels",
    )
    nn_typhoon.add_argument(
        "--levels",
        nargs="+",
        default=["05_STY"],
        help="Typhoon level workbook names, e.g. 01_TD 02_TS",
    )
    nn_typhoon.add_argument(
        "--data-root",
        default=str(DEFAULT_NN_DATA_ROOT),
        help="Directory containing classified typhoon Excel files",
    )
    nn_typhoon.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_NN_CHECKPOINT_DIR),
        help="Directory containing *_model.pt checkpoints",
    )
    nn_typhoon.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of independent sequences to generate per level",
    )
    nn_typhoon.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible NN sampling",
    )
    nn_typhoon.add_argument(
        "--jitter",
        type=float,
        default=DEFAULT_NN_JITTER,
        help="Relative std dev applied to Init* parameters (e.g. 0.05 = ±5%)",
    )
    nn_typhoon.add_argument(
        "--input-json",
        help="JSON file with Init* fields to override default sample",
    )
    nn_typhoon.add_argument(
        "--output",
        default=str(GENERATED_DATA_DIR / "typhoons" / "nn_generated.xlsx"),
        help="Output Excel path",
    )
    nn_typhoon.set_defaults(func=run_nn_typhoon)

    cluster = subparsers.add_parser(
        "cluster",
        help="Cluster Monte Carlo simulation results per typhoon level",
    )
    cluster.add_argument(
        "--workbook",
        default=str(DEFAULT_CLUSTER_WORKBOOK),
        help="Path to the MC simulation Excel file (or a directory of files)",
    )
    cluster.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_CLUSTER_SAMPLE_ROWS,
        help="Rows per sample (default 32)",
    )
    cluster.add_argument(
        "--clusters",
        type=int,
        default=DEFAULT_CLUSTER_COUNT,
        help="Target number of clusters per workbook",
    )
    cluster.add_argument(
        "--component-prefix",
        default=DEFAULT_CLUSTER_COMPONENT_PREFIX,
        help="Prefix of component rows, e.g. Line_",
    )
    cluster.add_argument(
        "--coverage-targets",
        type=int,
        nargs="*",
        default=list(DEFAULT_CLUSTER_COVERAGE_TARGETS),
        help="Top-N medoid counts for coverage evaluation",
    )
    cluster.add_argument(
        "--coverage-thresholds",
        type=float,
        nargs="*",
        default=list(DEFAULT_CLUSTER_COVERAGE_THRESHOLDS),
        help="Hamming distance thresholds for coverage curves",
    )
    cluster.add_argument(
        "--output-dir",
        default=str(DEFAULT_CLUSTER_OUTPUT_DIR),
        help="Directory where clustered Excel files will be written",
    )
    cluster.add_argument(
        "--workbooks",
        nargs="*",
        help="Optional subset of workbook names to process",
    )
    cluster.add_argument(
        "--limit",
        type=int,
        help="Process at most this many files",
    )
    cluster.set_defaults(func=run_cluster_workflow)

    auto = subparsers.add_parser(
        "auto-eval",
        help="Run the full typhoon → impact → MC → clustering pipeline",
    )
    auto.add_argument("--run-name", help="Custom name for this evaluation run")
    auto.add_argument(
        "--output-root",
        default=str(AUTO_EVAL_ROOT),
        help="Directory under dataset/ where run folders will live",
    )
    auto.add_argument(
        "--typhoon-source",
        choices=["seasonal", "nn"],
        default="seasonal",
        help="Choose physical seasonal simulator or NN generator",
    )
    auto.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=[9],
        help="Months to sample when typhoon-source=seasonal",
    )
    auto.add_argument(
        "--storms",
        type=int,
        default=100,
        help="Total storms to synthesize for seasonal runs",
    )
    auto.add_argument("--typhoon-seed", type=int, help="RNG seed for seasonal simulator")
    auto.add_argument(
        "--levels",
        nargs="+",
        default=["05_STY"],
        help="Typhoon intensity levels when typhoon-source=nn",
    )
    auto.add_argument(
        "--nn-count",
        type=int,
        default=100,
        help="Independent NN samples per level",
    )
    auto.add_argument("--nn-seed", type=int, help="RNG seed for NN sampling")
    auto.add_argument(
        "--nn-jitter",
        type=float,
        default=DEFAULT_NN_JITTER,
        help="Relative perturbation applied to Init* fields",
    )
    auto.add_argument("--nn-input-json", help="Override NN Init* sample via JSON file")
    auto.add_argument(
        "--nn-data-root",
        default=str(DEFAULT_NN_DATA_ROOT),
        help="Directory containing classified typhoon Excel files",
    )
    auto.add_argument(
        "--nn-checkpoint-dir",
        default=str(DEFAULT_NN_CHECKPOINT_DIR),
        help="Directory with NN checkpoints",
    )
    auto.add_argument(
        "--tower-excel",
        default=str(RAW_DATA_DIR / "TowerSeg.xlsx"),
        help="Tower/segment workbook for transmission impacts",
    )
    auto.add_argument(
        "--cable-rain-output",
        help="Optional CableSeg rainfall workbook path",
    )
    auto.add_argument(
        "--wind-excel",
        default=str(RAW_DATA_DIR / "wind_farms.xlsx"),
        help="Wind farm workbook for wind outputs",
    )
    auto.add_argument(
        "--wind-location-sheet",
        default="wind_farm_location",
        help="Sheet name listing wind farm coordinates",
    )
    auto.add_argument(
        "--random-failure",
        default=str(RAW_DATA_DIR / "random_failure_prob.xlsx"),
        help="Random failure workbook consumed by Monte Carlo",
    )
    auto.add_argument(
        "--mc-sheet",
        default="all",
        help="Sheet selection for Monte Carlo stage",
    )
    auto.add_argument("--mc-batch", type=int, default=1000, help="Samples per MC batch")
    auto.add_argument("--mc-repair", action="store_true", help="Enable repair logic during MC")
    auto.add_argument("--mc-no-random", action="store_true", help="Disable base random failures")
    auto.add_argument(
        "--mc-tre",
        type=int,
        default=24,
        help="Repair window length (hours) passed to Monte Carlo",
    )
    auto.add_argument(
        "--cluster-sample-rows",
        type=int,
        default=DEFAULT_CLUSTER_SAMPLE_ROWS,
        help="Rows per sample when clustering MC results",
    )
    auto.add_argument(
        "--cluster-count",
        type=int,
        default=DEFAULT_CLUSTER_COUNT,
        help="Target number of clusters",
    )
    auto.add_argument(
        "--cluster-component-prefix",
        default=DEFAULT_CLUSTER_COMPONENT_PREFIX,
        help="Component prefix (e.g. Line_) for clustering",
    )
    auto.add_argument(
        "--cluster-coverage-targets",
        type=int,
        nargs="*",
        default=list(DEFAULT_CLUSTER_COVERAGE_TARGETS),
        help="Top-N medoid coverage cutoffs",
    )
    auto.add_argument(
        "--cluster-coverage-thresholds",
        type=float,
        nargs="*",
        default=list(DEFAULT_CLUSTER_COVERAGE_THRESHOLDS),
        help="Normalized Hamming distance thresholds",
    )
    auto.set_defaults(func=run_auto_evaluation)

    return parser


def _display_defaults(command: str, defaults: argparse.Namespace) -> dict:
    values = {
        key: value
        for key, value in vars(defaults).items()
        if key not in {"func", "command"}
    }
    print("\n\u5f53\u524d\u547d\u4ee4\u7684\u9ed8\u8ba4\u53c2\u6570:")
    for key, value in values.items():
        print(f"  - {key}: {value}")
    return values


def _build_interactive_args(parser: argparse.ArgumentParser, command: str) -> List[str]:
    defaults = parser.parse_args([command])
    _display_defaults(command, defaults)
    if _prompt_yes_no("\u662f\u5426\u76f4\u63a5\u4f7f\u7528\u9ed8\u8ba4\u503c?", True):
        return [command]
    builder = INTERACTIVE_BUILDERS.get(command)
    if not builder:
        print("\u6682\u65e0\u4ea4\u4e92\u914d\u7f6e, \u5c06\u4f7f\u7528\u9ed8\u8ba4\u503c")
        return [command]
    extra_args = builder(defaults)
    return [command] + extra_args


def _collect_typhoon_args(defaults: argparse.Namespace) -> List[str]:
    tokens: List[str] = []
    storms_default = getattr(defaults, "storms", 5)
    storms = _prompt_int("\u5171\u9700\u751f\u6210\u51e0\u4e2a\u53f0\u98ce", storms_default)
    if storms is not None:
        tokens += ["--storms", str(max(1, storms))]
    month_default = getattr(defaults, "month", 9)
    month = _prompt_int("\u6a21\u62df\u7684\u6708\u4efd(1-12)", month_default)
    if month is not None:
        tokens += ["--month", str(month)]
    seed = _prompt_int("\u968f\u673a\u79cd\u5b50(\u53ef\u7559\u7a7a)", getattr(defaults, "seed", None))
    if seed is not None:
        tokens += ["--seed", str(seed)]
    output = _prompt_text("\u8f93\u51faExcel\u8def\u5f84", defaults.output)
    if output:
        tokens += ["--output", output]
    return tokens


def _collect_impacts_args(defaults: argparse.Namespace) -> List[str]:
    tokens: List[str] = []
    hurricane = _prompt_text("\u53f0\u98ceExcel\u8def\u5f84", defaults.hurricane_file)
    if hurricane:
        tokens += ["--hurricane-file", hurricane]
    tower = _prompt_text("TowerSeg Excel \u8def\u5f84", defaults.tower_excel)
    if tower:
        tokens += ["--tower-excel", tower]
    output = _prompt_text("\u8f93\u51faExcel\u8def\u5f84", defaults.output)
    if output:
        tokens += ["--output", output]
    cable_rain = _prompt_text("CableSeg 降雨量输出路径", defaults.cable_rain_output)
    if cable_rain:
        tokens += ["--cable-rain-output", cable_rain]
    sheets = _prompt_string_list("\u6307\u5b9a\u5de5\u4f5c\u8868(\u9017\u53f7\u5206\u9694)" , defaults.sheets)
    if sheets:
        tokens.append("--sheets")
        tokens.extend(sheets)
    return tokens


def _collect_wind_args(defaults: argparse.Namespace) -> List[str]:
    tokens: List[str] = []
    hurricane = _prompt_text("\u53f0\u98ceExcel\u8def\u5f84", defaults.hurricane_file)
    if hurricane:
        tokens += ["--hurricane-file", hurricane]
    wind_excel = _prompt_text("\u98ce\u573aExcel\u8def\u5f84", defaults.wind_excel)
    if wind_excel:
        tokens += ["--wind-excel", wind_excel]
    location_sheet = _prompt_text("\u4f4d\u7f6e\u8868\u540d", defaults.location_sheet)
    if location_sheet:
        tokens += ["--location-sheet", location_sheet]
    output = _prompt_text("\u8f93\u51faExcel\u8def\u5f84", defaults.output)
    if output:
        tokens += ["--output", output]
    sheets = _prompt_string_list("\u6307\u5b9a\u53f0\u98ce\u5de5\u4f5c\u8868", defaults.sheets)
    if sheets:
        tokens.append("--sheets")
        tokens.extend(sheets)
    return tokens


def _collect_mc_args(defaults: argparse.Namespace) -> List[str]:
    tokens: List[str] = []
    linefail = _prompt_text("\u7ebf\u8def\u6545\u969cExcel", defaults.linefail)
    if linefail:
        tokens += ["--linefail", linefail]
    random_excel = _prompt_text("\u968f\u673a\u6545\u969cExcel", defaults.random)
    if random_excel:
        tokens += ["--random", random_excel]
    wind = _prompt_text("\u98ce\u7535\u573aExcel", defaults.wind)
    if wind:
        tokens += ["--wind", wind]
    cable_rain = _prompt_text("CableSeg 降雨量输出路径", defaults.cable_rain_output)
    if cable_rain:
        tokens += ["--cable-rain-output", cable_rain]
    sheet = _prompt_text("\u5de5\u4f5c\u8868(all/\u9017\u53f7 list)", defaults.sheet)
    if sheet:
        tokens += ["--sheet", sheet]
    batch = _prompt_int("\u6279\u6b21\u5927\u5c0f(\u8499\u7279\u5361\u6d1b\u6837\u672c\u6570)", defaults.batch)
    if batch is not None:
        tokens += ["--batch", str(batch)]
    output = _prompt_text("\u7edf\u8ba1Excel\u8def\u5f84", defaults.output)
    if output:
        tokens += ["--output", output]
    dump_json = _prompt_text("JSON \u5185\u5bb9\u8def\u5f84", defaults.dump_json)
    if dump_json:
        tokens += ["--dump-json", dump_json]
    return tokens


def _collect_nn_typhoon_args(defaults: argparse.Namespace) -> List[str]:
    tokens: List[str] = []
    levels = _prompt_string_list("\u751f\u6210\u53f0\u98ce\u7ea7\u522b(\u7a7a\u683c/\u9017\u53f7)", defaults.levels)
    if levels:
        tokens.append("--levels")
        tokens.extend(levels)
    data_root = _prompt_text("\u5206\u7ea7\u53f0\u98ceExcel\u76ee\u5f55", defaults.data_root)
    if data_root:
        tokens += ["--data-root", data_root]
    checkpoint_dir = _prompt_text("\u795e\u7ecf\u7f51\u7edccheckpoint\u76ee\u5f55", defaults.checkpoint_dir)
    if checkpoint_dir:
        tokens += ["--checkpoint-dir", checkpoint_dir]
    count = _prompt_int("\u6bcf\u7ea7\u751f\u6210\u51e0\u6761", getattr(defaults, "count", 1))
    if count is not None:
        tokens += ["--count", str(max(1, count))]
    jitter = _prompt_float(
        "Init* \u6b21\u6570\u6444\u52a8\u7a0b\u5ea6 (0-0.5)",
        getattr(defaults, "jitter", DEFAULT_NN_JITTER),
    )
    if jitter is not None:
        tokens += ["--jitter", str(max(0.0, jitter))]
    seed = _prompt_int("\u968f\u673a\u79cd\u5b50(\u53ef\u7559\u7a7a)", getattr(defaults, "seed", None))
    if seed is not None:
        tokens += ["--seed", str(seed)]
    input_json = _prompt_text("\u8f93\u5165 JSON \u6837\u672c\u8def\u5f84", defaults.input_json)
    if input_json:
        tokens += ["--input-json", input_json]
    output = _prompt_text("\u8f93\u51fa Excel \u8def\u5f84", defaults.output)
    if output:
        tokens += ["--output", output]
    return tokens


def _collect_cluster_args(defaults: argparse.Namespace) -> List[str]:
    tokens: List[str] = []
    workbook = _prompt_text("MC \u7ed3\u679cExcel\u8def\u5f84", defaults.workbook)
    if workbook:
        tokens += ["--workbook", workbook]
    sample_rows = _prompt_int("\u6bcf\u4e2a\u6837\u672c\u5305\u542b\u884c\u6570", defaults.sample_rows)
    if sample_rows is not None:
        tokens += ["--sample-rows", str(sample_rows)]
    clusters = _prompt_int("\u76ee\u6807\u805a\u7c07\u6570", defaults.clusters)
    if clusters is not None:
        tokens += ["--clusters", str(clusters)]
    component_prefix = _prompt_text("\u7ec4\u4ef6\u540d\u79f0\u524d\u7f00", defaults.component_prefix)
    if component_prefix:
        tokens += ["--component-prefix", component_prefix]
    coverage_targets = _prompt_int_list(
        "Top-N \u5bb9\u91cf\u8bc4\u4f30\u5b57\u6bb5",
        defaults.coverage_targets or DEFAULT_CLUSTER_COVERAGE_TARGETS,
    )
    if coverage_targets:
        tokens.append("--coverage-targets")
        tokens.extend(str(val) for val in coverage_targets)
    coverage_thresholds = _prompt_float_list(
        "\u8ddd\u79bb\u9608\u503c(0-1)",
        defaults.coverage_thresholds or DEFAULT_CLUSTER_COVERAGE_THRESHOLDS,
    )
    if coverage_thresholds:
        tokens.append("--coverage-thresholds")
        tokens.extend(str(val) for val in coverage_thresholds)
    output_dir = _prompt_text("\u805a\u7c07\u7ed3\u679c\u8f93\u51fa\u76ee\u5f55", defaults.output_dir)
    if output_dir:
        tokens += ["--output-dir", output_dir]
    workbooks = _prompt_string_list("\u6307\u5b9a Excel \u6587\u4ef6(\u53ef\u7559\u7a7a)", defaults.workbooks)
    if workbooks:
        tokens.append("--workbooks")
        tokens.extend(workbooks)
    limit = _prompt_int("\u6700\u591a\u5904\u7406\u6587\u4ef6\u6570", defaults.limit)
    if limit is not None:
        tokens += ["--limit", str(limit)]
    return tokens


def _collect_auto_eval_args(defaults: argparse.Namespace) -> List[str]:
    tokens: List[str] = []
    run_name = _prompt_text("\u8bc4\u4f30\u8fd0\u884c\u540d", defaults.run_name)
    if run_name:
        tokens += ["--run-name", run_name]
    output_root = _prompt_text("\u8f93\u51fa\u6839\u76ee\u5f55", defaults.output_root)
    if output_root:
        tokens += ["--output-root", output_root]

    source = _prompt_choice("\u53f0\u98ce\u6a21\u5757(seasonal/nn)", ["seasonal", "nn"], defaults.typhoon_source)
    tokens += ["--typhoon-source", source] 

    if source == "seasonal":
        months = _prompt_int_list("\u5b63\u8282\u53c2\u4e0e\u6708\u4efd", defaults.months or [9])
        if months:
            tokens.append("--months")
            tokens.extend(str(m) for m in months)
        storms = _prompt_int("\u53f0\u98ce\u6a21\u62df\u6570", defaults.storms)
        if storms is not None:
            tokens += ["--storms", str(storms)]
        seed = _prompt_int("\u968f\u673a\u79cd\u5b50", defaults.typhoon_seed)
        if seed is not None:
            tokens += ["--typhoon-seed", str(seed)]
    else: 
        levels = _prompt_string_list("NN \u53f0\u98ce\u7ea7\u522b", defaults.levels)
        if levels:
            tokens.append("--levels")
            tokens.extend(levels)
        nn_count  = _prompt_int("NN \u751f\u6210\u6b21\u6570", defaults.nn_count)
        if nn_count is not None:
            tokens += ["--nn-count", str(nn_count)]
        nn_seed = _prompt_int("NN \u968f\u673a\u79cd\u5b50", defaults.nn_seed)
        if nn_seed is not None:
            tokens += ["--nn-seed", str(nn_seed)]
        nn_jitter = _prompt_float("NN \u6b21\u6570\u6444\u52a8", defaults.nn_jitter)
        if nn_jitter is not None:
            tokens += ["--nn-jitter", str(nn_jitter)]
        nn_input = _prompt_text("NN JSON \u6837\u672c", defaults.nn_input_json)
        if nn_input:
            tokens += ["--nn-input-json", nn_input]

    mc_batch = _prompt_int("MC \u6279\u6b21\u5927\u5c0f", defaults.mc_batch)
    if mc_batch is not None:
        tokens += ["--mc-batch", str(mc_batch)]
    mc_sheet = _prompt_text("MC \u5de5\u4f5c\u8868", defaults.mc_sheet)
    if mc_sheet:
        tokens += ["--mc-sheet", mc_sheet]
    if _prompt_yes_no("\u542f\u7528MC\u4fee\u590d\u903b\u8f91?", defaults.mc_repair):
        tokens.append("--mc-repair")
    if _prompt_yes_no("\u5173\u95ed\u968f\u673a\u6545\u969c?", defaults.mc_no_random):
        tokens.append("--mc-no-random")
    mc_tre = _prompt_int("MC \u4fee\u590d\u7a97\u53e3(\u5c0f\u65f6)", defaults.mc_tre)
    if mc_tre is not None:
        tokens += ["--mc-tre", str(mc_tre)]

    cluster_count = _prompt_int("\u805a\u7c07\u6570", defaults.cluster_count)
    if cluster_count is not None:
        tokens += ["--cluster-count", str(cluster_count)]
    cluster_rows = _prompt_int("\u6bcf\u6837\u672c\u884c\u6570", defaults.cluster_sample_rows)
    if cluster_rows is not None:
        tokens += ["--cluster-sample-rows", str(cluster_rows)]

    return tokens


INTERACTIVE_BUILDERS = {
    "typhoon": _collect_typhoon_args,
    "impacts": _collect_impacts_args,
    "wind": _collect_wind_args,
    "monte-carlo": _collect_mc_args,
    "nn-typhoon": _collect_nn_typhoon_args,
    "cluster": _collect_cluster_args,
    "auto-eval": _collect_auto_eval_args,
}


def _interactive_menu(parser: argparse.ArgumentParser) -> None:
    options = [
        ("Typhoon lifecycle simulation", "typhoon"),
        ("NN typhoon generator", "nn-typhoon"),
        ("Transmission line impacts", "impacts"),
        ("Wind farm output", "wind"),
        ("Pseudo Monte Carlo", "monte-carlo"),
        ("MC result clustering", "cluster"),
        ("Auto evaluation pipeline", "auto-eval"),
    ]
    print("\nAvailable workflows:")
    for idx, (label, _) in enumerate(options, start=1):
        print(f"  {idx}. {label}")
    print("  q. Exit")

    try:
        choice = input(f"Select a workflow [1-{len(options)}/q]: ").strip().lower()
    except EOFError:
        return
    if choice in {"q", "quit", "exit", ""}:
        return
    try:
        index = int(choice) - 1
    except ValueError:
        print("Invalid selection, exiting.")
        return
    if index < 0 or index >= len(options):
        print("Invalid selection, exiting.")
        return
    _, command = options[index]
    args_list = _build_interactive_args(parser, command)
    args = parser.parse_args(args_list)
    if hasattr(args, "func"):
        args.func(args)


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(list(argv))
    if getattr(args, "command", None):
        args.func(args)
        return
    if sys.stdin.isatty():
        _interactive_menu(parser)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
