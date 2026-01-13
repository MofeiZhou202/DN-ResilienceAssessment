"""Compare predicted hurricane impacts against actual results across all worksheets."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate similarity between predicted and actual hurricane impact workbooks.",
    )
    parser.add_argument(
        "--actual",
        type=Path,
        default=Path("testset") / "batch_hurricane_impact_simple.xlsx",
        help="Path to the workbook containing actual impact data.",
    )
    parser.add_argument(
        "--predicted",
        type=Path,
        default=Path("testset") / "batch_hurricane_predictions.xlsx",
        help="Path to the workbook containing predicted impact data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("testset") / "evaluation",
        help="Directory where metrics and plots will be written.",
    )
    parser.add_argument(
        "--summary-name",
        default="evaluation_summary.csv",
        help="File name for the metrics summary CSV.",
    )
    return parser.parse_args()


def read_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")
    return pd.read_excel(path, sheet_name=None)


def sanitize_name(name: str) -> str:
    allowed = []
    for ch in str(name):
        if ch.isalnum() or ch in {"_", "-"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "sheet"


def maybe_set_index(frame: pd.DataFrame) -> pd.DataFrame:
    first_column = frame.columns[0]
    column_series = frame[first_column]
    if not pd.api.types.is_numeric_dtype(column_series):
        if column_series.is_unique:
            frame = frame.set_index(first_column)
    return frame


def prepare_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    working = maybe_set_index(frame.copy())
    numeric = working.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(how="all").dropna(axis=1, how="all")
    return numeric


def align_frames(actual: pd.DataFrame, predicted: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aligned_actual, aligned_pred = actual.align(predicted, join="inner", axis=None)
    aligned_actual = aligned_actual.dropna(how="all").dropna(axis=1, how="all")
    aligned_pred = aligned_pred.dropna(how="all").dropna(axis=1, how="all")
    aligned_actual, aligned_pred = aligned_actual.align(aligned_pred, join="inner", axis=None)
    return aligned_actual, aligned_pred


def flatten_values(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=float, copy=True)
    return values.reshape(-1)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    diff = predicted - actual
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    max_abs = float(np.max(np.abs(diff)))
    bias = float(np.mean(diff))
    if actual.var() == 0:
        r2 = float("nan")
        corr = float("nan")
    else:
        ss_res = float(np.sum(np.square(diff)))
        ss_tot = float(np.sum(np.square(actual - np.mean(actual))))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
        corr = float(np.corrcoef(actual, predicted)[0, 1])
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MaxAbsError": max_abs,
        "Bias": bias,
        "R2": r2,
        "PearsonR": corr,
    }


def draw_aggregate_plots(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
    metrics_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Scatter plot with identity line summarizing all pairs
    fig, ax = plt.subplots(figsize=(6, 6))
    min_val = float(np.min(np.concatenate([actual_values, predicted_values])))
    max_val = float(np.max(np.concatenate([actual_values, predicted_values])))
    padding = 0.02 * (max_val - min_val) if max_val > min_val else 1.0
    lower = min_val - padding
    upper = max_val + padding
    ax.scatter(actual_values, predicted_values, s=8, alpha=0.4, edgecolor="none")
    ax.plot([lower, upper], [lower, upper], "r--", linewidth=1.0)
    ax.set_title("Overall Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    fig.tight_layout()
    fig.savefig(output_dir / "overall_scatter.png", dpi=150)
    plt.close(fig)

    # Error histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    errors = predicted_values - actual_values
    ax.hist(errors, bins=40, color="#1f77b4", alpha=0.75)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_title("Overall Error Distribution")
    ax.set_xlabel("Prediction - Actual")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "overall_error_hist.png", dpi=150)
    plt.close(fig)

    if not metrics_frame.empty:
        sorted_frame = metrics_frame.sort_values("RMSE")
        fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_frame) * 0.3)))
        ax.barh(sorted_frame["sheet"], sorted_frame["RMSE"], color="#2ca02c")
        ax.set_xlabel("RMSE")
        ax.set_title("RMSE by Worksheet")
        fig.tight_layout()
        fig.savefig(output_dir / "rmse_by_sheet.png", dpi=150)
        plt.close(fig)


def evaluate_workbooks(
    actual_sheets: Dict[str, pd.DataFrame],
    predicted_sheets: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    common_sheets = sorted(set(actual_sheets) & set(predicted_sheets))
    if not common_sheets:
        raise RuntimeError("No matching worksheets found between actual and predicted workbooks.")

    metrics_records: List[Dict[str, object]] = []
    all_actual: List[np.ndarray] = []
    all_predicted: List[np.ndarray] = []

    for sheet_name in common_sheets:
        actual_numeric = prepare_numeric(actual_sheets[sheet_name])
        predicted_numeric = prepare_numeric(predicted_sheets[sheet_name])
        aligned_actual, aligned_predicted = align_frames(actual_numeric, predicted_numeric)
        if aligned_actual.empty or aligned_predicted.empty:
            print(f"[WARN] Sheet {sheet_name} yielded empty alignment; skipping.")
            continue

        actual_values = flatten_values(aligned_actual)
        predicted_values = flatten_values(aligned_predicted)
        mask = np.isfinite(actual_values) & np.isfinite(predicted_values)
        actual_values = actual_values[mask]
        predicted_values = predicted_values[mask]
        if actual_values.size == 0:
            print(f"[WARN] Sheet {sheet_name} contains no comparable values; skipping.")
            continue

        sheet_metrics = compute_metrics(actual_values, predicted_values)
        metrics_record = {"sheet": sheet_name, **sheet_metrics}
        metrics_records.append(metrics_record)
        all_actual.append(actual_values)
        all_predicted.append(predicted_values)

    summary_df = pd.DataFrame(metrics_records)
    overall_actual = np.concatenate(all_actual) if all_actual else np.array([], dtype=float)
    overall_predicted = np.concatenate(all_predicted) if all_predicted else np.array([], dtype=float)
    return summary_df, overall_actual, overall_predicted


def main() -> None:
    args = parse_args()
    actual_sheets = read_workbook(args.actual)
    predicted_sheets = read_workbook(args.predicted)
    summary, overall_actual, overall_predicted = evaluate_workbooks(actual_sheets, predicted_sheets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if overall_actual.size and overall_predicted.size:
        draw_aggregate_plots(overall_actual, overall_predicted, summary, args.output_dir)
    summary_path = args.output_dir / args.summary_name
    summary.sort_values("RMSE", inplace=True)
    summary.to_csv(summary_path, index=False)
    if not summary.empty:
        print("Evaluation completed. Key metrics:")
        print(summary)
    print(f"Summary written to {summary_path.resolve()}.")
    print(f"Plots saved under {args.output_dir.resolve()}.")


if __name__ == "__main__":
    main()
