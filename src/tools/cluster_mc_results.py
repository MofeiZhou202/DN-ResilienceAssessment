"""Cluster binary Monte Carlo simulation samples by typhoon intensity level.

This script traverses the Excel workbooks stored under ``new_dataset/MC_results``.
Each workbook contains multiple worksheets, and every worksheet is structured as
640 rows by 48 columns of 0/1 data. Every consecutive block of 32 rows
represents one simulation sample (i.e., a single power line outage scenario).

For every workbook (corresponding to one typhoon intensity level), this script
collects all samples across all worksheets, performs k-means clustering, and
saves the assignments plus per-cluster statistics to a new Excel file located
next to the source workbook.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple
from types import ModuleType
import subprocess

try:  # Python 3.13+: distutils removed from stdlib
    import distutils  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - compatibility shim
    import importlib
    import sys

    _distutils = importlib.import_module("setuptools._distutils")
    sys.modules.setdefault("distutils", _distutils)
    sys.modules.setdefault(
        "distutils.version",
        importlib.import_module("setuptools._distutils.version"),
    )

import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids

try:
    from matplotlib import pyplot as plt  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - optional dependency
    _scipy_stats = None

# Default values that reflect the data layout described by the user.
DEFAULT_SAMPLE_ROWS = 32
DEFAULT_CLUSTER_COUNT = 100
DEFAULT_COMPONENT_PREFIX = "Line_"
TARGET_COLUMNS = 48
RANDOM_STATE = 42
KMEDOIDS_MAX_RESTARTS = 10
CONFIDENCE_LEVEL = 0.95
Z_FALLBACK = 1.96
_MATPLOTLIB_READY: Optional[bool] = None
DEFAULT_COVERAGE_TARGETS = (10, 20, 50, 100, 200)
DEFAULT_COVERAGE_THRESHOLDS = (0.0, 0.02, 0.05, 0.1)
DISTANCE_BATCH_SIZE = 128


def _install_matplotlib() -> bool:
    """Attempt to install matplotlib via pip."""

    try:
        print("  matplotlib not found, attempting automatic installation...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "matplotlib",
                "--quiet",
                "--disable-pip-version-check",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("  matplotlib installation completed.")
        return True
    except Exception:
        print("  Automatic matplotlib installation failed, please install manually.")
        return False


def _ensure_matplotlib() -> Optional[ModuleType]:
    """Ensure matplotlib is importable and configured for headless use."""

    global plt, _MATPLOTLIB_READY

    if plt is not None:
        return plt
    if _MATPLOTLIB_READY is False:
        return None

    try:
        import importlib

        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg", force=True)
        plt_module = importlib.import_module("matplotlib.pyplot")
        plt = plt_module  # type: ignore[assignment]
        _MATPLOTLIB_READY = True
        return plt_module
    except Exception:
        if _install_matplotlib():
            try:
                import importlib

                matplotlib = importlib.import_module("matplotlib")
                matplotlib.use("Agg", force=True)
                plt_module = importlib.import_module("matplotlib.pyplot")
                plt = plt_module  # type: ignore[assignment]
                _MATPLOTLIB_READY = True
                return plt_module
            except Exception:
                _MATPLOTLIB_READY = False
                return None
        _MATPLOTLIB_READY = False
        return None


def _norm_quantile(confidence: float) -> float:
    """Return the two-sided normal quantile for the given confidence level."""

    try:
        return NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    except ValueError:
        return Z_FALLBACK


def _t_quantile(df: int, confidence: float) -> float:
    if df <= 0:
        return 0.0
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df))
    return _norm_quantile(confidence)


def _wilson_interval(successes: int, trials: int, confidence: float) -> Tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = _norm_quantile(confidence)
    phat = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = phat + z2 / (2.0 * trials)
    adjustment = z * sqrt((phat * (1.0 - phat) + z2 / (4.0 * trials)) / trials)
    lower = (center - adjustment) / denominator
    upper = (center + adjustment) / denominator
    return max(0.0, lower), min(1.0, upper)


def compute_cluster_confidence_intervals(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    sample_rows: int,
    model: KMedoids,
) -> pd.DataFrame:
    """Compute confidence intervals for cluster coverage and outage rates."""

    total_samples = feature_matrix.shape[0]
    if total_samples == 0:
        raise ValueError("Cannot compute confidence intervals on empty data.")

    cluster_ids = np.sort(np.unique(labels.astype(int)))
    flat_features = feature_matrix.reshape(feature_matrix.shape[0], -1).astype(float)
    observations_per_sample = float(sample_rows * TARGET_COLUMNS)
    z_value = _norm_quantile(CONFIDENCE_LEVEL)

    medoid_indices = getattr(model, "medoid_indices_", None)

    records: List[Dict[str, float]] = []
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        cluster_size = int(mask.sum())
        if cluster_size == 0:
            continue

        sample_fraction = cluster_size / total_samples
        frac_lower, frac_upper = _wilson_interval(cluster_size, total_samples, CONFIDENCE_LEVEL)

        cluster_samples = flat_features[mask]
        sample_rates = cluster_samples.mean(axis=1)
        mean_rate = float(sample_rates.mean())
        if cluster_size > 1:
            std_rate = float(sample_rates.std(ddof=1))
            t_value = _t_quantile(cluster_size - 1, CONFIDENCE_LEVEL)
            margin = t_value * std_rate / sqrt(cluster_size)
        else:
            margin = 0.0
        rate_lower = max(0.0, mean_rate - margin)
        rate_upper = min(1.0, mean_rate + margin)

        mean_outage_count = mean_rate * observations_per_sample
        count_margin = margin * observations_per_sample

        medoid_vector: Optional[np.ndarray] = None
        if medoid_indices is not None and 0 <= cluster_id < len(medoid_indices):
            medoid_index = int(medoid_indices[cluster_id])
            if 0 <= medoid_index < feature_matrix.shape[0]:
                medoid_vector = feature_matrix[medoid_index].astype(float)

        if medoid_vector is None:
            medoid_vector = cluster_samples.mean(axis=0).round()

        distances = np.mean(cluster_samples != medoid_vector, axis=1)
        mean_distance = float(distances.mean())
        if cluster_size > 1:
            std_distance = float(distances.std(ddof=1))
            distance_margin = _t_quantile(cluster_size - 1, CONFIDENCE_LEVEL) * std_distance / sqrt(
                cluster_size
            )
        else:
            distance_margin = 0.0
        distance_lower = max(0.0, mean_distance - distance_margin)
        distance_upper = min(1.0, mean_distance + distance_margin)

        records.append(
            {
                "cluster_id": int(cluster_id),
                "sample_count": cluster_size,
                "sample_fraction": sample_fraction,
                "sample_fraction_ci_lower": frac_lower,
                "sample_fraction_ci_upper": frac_upper,
                "mean_outage_fraction": mean_rate,
                "outage_fraction_ci_lower": rate_lower,
                "outage_fraction_ci_upper": rate_upper,
                "mean_outage_count": mean_outage_count,
                "outage_count_ci_lower": max(0.0, mean_outage_count - count_margin),
                "outage_count_ci_upper": mean_outage_count + count_margin,
                "mean_hamming_to_medoid": mean_distance,
                "medoid_distance_ci_lower": distance_lower,
                "medoid_distance_ci_upper": distance_upper,
                "z_value": z_value,
            }
        )

    confidence_df = pd.DataFrame(records).sort_values("cluster_id").reset_index(drop=True)
    numeric_columns = [
        "sample_fraction",
        "sample_fraction_ci_lower",
        "sample_fraction_ci_upper",
        "mean_outage_fraction",
        "outage_fraction_ci_lower",
        "outage_fraction_ci_upper",
        "mean_outage_count",
        "outage_count_ci_lower",
        "outage_count_ci_upper",
        "mean_hamming_to_medoid",
        "medoid_distance_ci_lower",
        "medoid_distance_ci_upper",
        "z_value",
    ]
    confidence_df[numeric_columns] = confidence_df[numeric_columns].astype(float)
    return confidence_df


def plot_cluster_confidence_intervals(
    confidence_df: pd.DataFrame,
    output_path: Path,
) -> Optional[Path]:
    """Create bar charts with confidence intervals per cluster."""

    if confidence_df.empty:
        return None

    plt_module = _ensure_matplotlib()
    if plt_module is None:
        return None

    plot_df = confidence_df.sort_values("cluster_id").reset_index(drop=True)
    positions = np.arange(len(plot_df))
    cluster_labels = plot_df["cluster_id"].astype(str).to_numpy()

    fraction_mean = plot_df["sample_fraction"].to_numpy()
    fraction_lower = fraction_mean - plot_df["sample_fraction_ci_lower"].to_numpy()
    fraction_upper = plot_df["sample_fraction_ci_upper"].to_numpy() - fraction_mean
    fraction_yerr = np.vstack([fraction_lower, fraction_upper])

    outage_mean = plot_df["mean_outage_fraction"].to_numpy()
    outage_lower = outage_mean - plot_df["outage_fraction_ci_lower"].to_numpy()
    outage_upper = plot_df["outage_fraction_ci_upper"].to_numpy() - outage_mean
    outage_yerr = np.vstack([outage_lower, outage_upper])

    fig, axes = plt_module.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].bar(positions, fraction_mean, color="#1f77b4", alpha=0.8)
    axes[0].errorbar(
        positions,
        fraction_mean,
        yerr=fraction_yerr,
        fmt="none",
        ecolor="#2ca02c",
        capsize=3,
        linewidth=1,
    )
    axes[0].set_ylabel("Sample Fraction")
    axes[0].set_title("Cluster Sample Fractions with 95% CI")

    axes[1].bar(positions, outage_mean, color="#ff7f0e", alpha=0.8)
    axes[1].errorbar(
        positions,
        outage_mean,
        yerr=outage_yerr,
        fmt="none",
        ecolor="#d62728",
        capsize=3,
        linewidth=1,
    )
    axes[1].set_ylabel("Mean Outage Fraction")
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_title("Mean Outage Fraction per Sample with 95% CI")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(cluster_labels, rotation=45, ha="right")

    fig.tight_layout()

    plot_path = output_path.with_name(f"{output_path.stem}_confidence.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt_module.close(fig)
    return plot_path


def plot_cluster_distance_intervals(
    confidence_df: pd.DataFrame,
    output_path: Path,
) -> Optional[Path]:
    """Plot mean Hamming distance to medoid with confidence intervals."""

    required_columns = {
        "cluster_id",
        "mean_hamming_to_medoid",
        "medoid_distance_ci_lower",
        "medoid_distance_ci_upper",
    }
    if confidence_df.empty or not required_columns.issubset(confidence_df.columns):
        return None

    plt_module = _ensure_matplotlib()
    if plt_module is None:
        return None

    plot_df = confidence_df.sort_values("cluster_id").reset_index(drop=True)
    positions = np.arange(len(plot_df))
    cluster_labels = plot_df["cluster_id"].astype(str).to_numpy()

    mean_distance = plot_df["mean_hamming_to_medoid"].to_numpy()
    distance_lower = mean_distance - plot_df["medoid_distance_ci_lower"].to_numpy()
    distance_upper = plot_df["medoid_distance_ci_upper"].to_numpy() - mean_distance
    distance_lower = np.clip(distance_lower, 0.0, None)
    distance_upper = np.clip(distance_upper, 0.0, None)
    distance_yerr = np.vstack([distance_lower, distance_upper])

    fig, ax = plt_module.subplots(figsize=(12, 4))
    ax.bar(positions, mean_distance, color="#9467bd", alpha=0.8)
    ax.errorbar(
        positions,
        mean_distance,
        yerr=distance_yerr,
        fmt="none",
        ecolor="#17becf",
        capsize=3,
        linewidth=1,
    )
    ax.set_ylabel("Mean Hamming Distance")
    ax.set_xlabel("Cluster ID")
    ax.set_title("Average Hamming Distance to Cluster Medoid with 95% CI")
    ax.set_xticks(positions)
    ax.set_xticklabels(cluster_labels, rotation=45, ha="right")
    upper_totals = mean_distance + distance_upper
    y_max = float(np.max(upper_totals)) if upper_totals.size else 0.0
    ax.set_ylim(0.0, min(1.0, max(0.05, y_max * 1.05)))

    fig.tight_layout()

    plot_path = output_path.with_name(f"{output_path.stem}_distance_ci.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt_module.close(fig)
    return plot_path


def plot_cluster_coverage_summary(
    confidence_df: pd.DataFrame,
    output_path: Path,
) -> Optional[Path]:
    """Plot cumulative coverage of clusters with confidence intervals."""

    required_columns = {
        "cluster_id",
        "sample_fraction",
        "sample_fraction_ci_lower",
        "sample_fraction_ci_upper",
    }
    if confidence_df.empty or not required_columns.issubset(confidence_df.columns):
        return None

    plt_module = _ensure_matplotlib()
    if plt_module is None:
        return None

    plot_df = confidence_df.sort_values("sample_fraction", ascending=False).reset_index(drop=True)
    ranks = np.arange(1, len(plot_df) + 1)

    mean_fraction = np.clip(plot_df["sample_fraction"].to_numpy(), 0.0, 1.0)
    cumulative_mean = np.minimum(1.0, np.cumsum(mean_fraction))

    cumulative_counts = np.cumsum(plot_df["sample_count"].to_numpy(dtype=int))
    total_samples = int(plot_df["sample_count"].sum())
    lower_list: List[float] = []
    upper_list: List[float] = []
    for count in cumulative_counts:
        lower, upper = _wilson_interval(int(count), total_samples, CONFIDENCE_LEVEL)
        lower_list.append(lower)
        upper_list.append(upper)
    cumulative_lower = np.minimum(1.0, np.array(lower_list))
    cumulative_upper = np.minimum(1.0, np.array(upper_list))

    fig, ax = plt_module.subplots(figsize=(12, 5))
    ax.plot(ranks, cumulative_mean, color="#1f77b4", label="Cumulative coverage (observed)")
    ax.fill_between(
        ranks,
        cumulative_lower,
        cumulative_upper,
        color="#1f77b4",
        alpha=0.25,
        label="95% confidence band",
    )
    ax.axhline(0.9, color="#ff7f0e", linestyle="--", linewidth=1, label="90% coverage threshold")
    ax.axhline(0.95, color="#2ca02c", linestyle=":", linewidth=1, label="95% coverage threshold")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Cumulative sample coverage")
    ax.set_title("Top-N Cluster Coverage with 95% Confidence Band")
    ax.set_xlim(1, len(plot_df))
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()

    plot_path = output_path.with_name(f"{output_path.stem}_coverage_ci.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt_module.close(fig)
    return plot_path


def evaluate_medoid_coverage(
    feature_matrix: np.ndarray,
    cluster_summary: pd.DataFrame,
    medoid_indices: Optional[np.ndarray],
    coverage_targets: Optional[Sequence[int]],
    coverage_thresholds: Optional[Sequence[float]],
) -> Optional[pd.DataFrame]:
    """Evaluate how well top-N medoids cover samples under distance thresholds."""

    if medoid_indices is None:
        return None
    if not coverage_targets or not coverage_thresholds:
        return None

    sorted_targets = sorted({int(target) for target in coverage_targets if int(target) > 0})
    if not sorted_targets:
        return None

    sorted_thresholds = sorted({float(th) for th in coverage_thresholds if float(th) >= 0.0})
    if not sorted_thresholds:
        return None

    available_clusters = len(medoid_indices)
    valid_targets = [target for target in sorted_targets if target <= available_clusters]
    if not valid_targets:
        return None

    cluster_order = (
        cluster_summary.sort_values("sample_count", ascending=False)["cluster_id"].to_numpy(dtype=int)
    )
    feature_len = feature_matrix.shape[1]
    medoid_matrix = feature_matrix[medoid_indices]

    n_samples = feature_matrix.shape[0]
    n_medoids = medoid_matrix.shape[0]
    dist_matrix = np.empty((n_samples, n_medoids), dtype=float)

    for start in range(0, n_samples, DISTANCE_BATCH_SIZE):
        end = min(start + DISTANCE_BATCH_SIZE, n_samples)
        chunk = feature_matrix[start:end]
        diff = chunk[:, np.newaxis, :] != medoid_matrix[np.newaxis, :, :]
        dist_matrix[start:end] = diff.sum(axis=2) / feature_len

    records: List[Dict[str, float]] = []

    for top_n in valid_targets:
        top_indices = cluster_order[:top_n]
        min_distances = dist_matrix[:, top_indices].min(axis=1)
        p95_distance = float(np.percentile(min_distances, 95))
        max_distance = float(min_distances.max())
        mean_distance = float(min_distances.mean())
        for threshold in sorted_thresholds:
            coverage_fraction = float(np.mean(min_distances <= threshold))
            records.append(
                {
                    "top_n": int(top_n),
                    "threshold": float(threshold),
                    "coverage_fraction": coverage_fraction,
                    "mean_distance": mean_distance,
                    "percentile_95_distance": p95_distance,
                    "max_distance": max_distance,
                }
            )

    coverage_df = pd.DataFrame(records)
    coverage_df.sort_values(["top_n", "threshold"], inplace=True)
    coverage_df.reset_index(drop=True, inplace=True)
    return coverage_df


def plot_medoid_coverage_curves(
    coverage_df: Optional[pd.DataFrame],
    output_path: Path,
) -> Optional[Path]:
    """Plot coverage achieved by top-N medoids under distance thresholds."""

    if coverage_df is None or coverage_df.empty:
        return None

    plt_module = _ensure_matplotlib()
    if plt_module is None:
        return None

    pivot = coverage_df.pivot_table(
        index="threshold", columns="top_n", values="coverage_fraction"
    ).sort_index()

    fig, ax = plt_module.subplots(figsize=(10, 6))
    for top_n in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[top_n],
            marker="o",
            label=f"Top {int(top_n)} clusters",
        )

    ax.set_xlabel("Normalized Hamming distance threshold")
    ax.set_ylabel("Fraction of samples within threshold")
    ax.set_title("Coverage of top-N medoids under distance thresholds")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    plot_path = output_path.with_name(f"{output_path.stem}_medoid_coverage.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt_module.close(fig)
    return plot_path


@dataclass
class SampleMeta:
    """Lightweight container for sample metadata."""

    excel_file: str
    sheet_name: str
    sample_index: int

    def make_uid(self) -> str:
        """Return a human-readable identifier for the sample."""

        return f"{self.sheet_name}#{self.sample_index:03d}"


def _parse_sample_number(sample_label: str) -> int:
    """Try to parse the trailing number inside labels such as ``Sample_1``."""

    if not sample_label:
        return 0

    for token in reversed(sample_label.split("_")):
        if token.isdigit():
            return int(token)
    return 0


def load_samples_from_excel(
    excel_path: Path,
    sample_rows: int,
    component_prefix: str,
) -> Tuple[np.ndarray, List[SampleMeta], List[Tuple[str, int]]]:
    """Load and reshape binary samples from an Excel workbook.

    Parameters
    ----------
    excel_path:
        Path to the workbook that houses the simulation samples.
    sample_rows:
        Number of consecutive rows that represent one sample.

    Returns
    -------
    feature_matrix:
        ``(num_samples, sample_rows * num_cols)`` array ready for clustering.
    metadata:
        Parallel list describing the origin of each sample.
    """

    features: List[np.ndarray] = []
    metadata: List[SampleMeta] = []
    sheet_stats: List[Tuple[str, int]] = []

    workbook = pd.ExcelFile(excel_path, engine="openpyxl")

    for sheet_name in workbook.sheet_names:
        df = workbook.parse(sheet_name, header=None)
        if df.empty:
            continue

        first_col = df.iloc[:, 0].astype(str).str.strip()
        sentinel_mask = first_col.eq("Sample")
        if sentinel_mask.any():
            sentinel_indices = np.flatnonzero(sentinel_mask.to_numpy())
            anchor = sentinel_indices[1] if sentinel_indices.size > 1 else sentinel_indices[0]
            df = df.iloc[int(anchor) + 1 :].copy()
            if df.empty:
                continue

        component_series = df.iloc[:, 1].astype(str)
        mask_components = component_series.str.startswith(
            component_prefix, na=False
        )
        line_rows = df.loc[mask_components].copy()
        if line_rows.empty:
            continue

        line_rows.iloc[:, 0] = line_rows.iloc[:, 0].ffill()

        value_block = (
            line_rows.iloc[:, 2:]
            .dropna(axis=1, how="all")
            .iloc[:, :TARGET_COLUMNS]
        )

        if value_block.shape[1] != TARGET_COLUMNS:
            raise ValueError(
                f"Worksheet '{sheet_name}' in '{excel_path.name}' has {value_block.shape[1]} "
                f"columns, expected {TARGET_COLUMNS}."
            )

        numeric_block = value_block.apply(pd.to_numeric, errors="raise")
        matrix = numeric_block.to_numpy(dtype=float)
        if not np.isin(matrix, (0.0, 1.0)).all():
            raise ValueError(
                f"Worksheet '{sheet_name}' in '{excel_path.name}' contains non-binary values."
            )

        rows, _ = matrix.shape

        if rows % sample_rows != 0:
            raise ValueError(
                f"Worksheet '{sheet_name}' in '{excel_path.name}' yielded {rows} rows, "
                f"which is not divisible by sample height {sample_rows}."
            )

        sample_labels = line_rows.iloc[:, 0].astype(str).to_numpy()
        sample_count = rows // sample_rows
        sheet_stats.append((sheet_name, sample_count))

        for sample_idx in range(sample_count):
            start = sample_idx * sample_rows
            end = (sample_idx + 1) * sample_rows
            block = matrix[start:end, :]
            features.append(block.reshape(-1))

            sample_label = sample_labels[start]
            numeric_index = _parse_sample_number(sample_label)
            metadata.append(
                SampleMeta(
                    excel_file=excel_path.name,
                    sheet_name=sheet_name,
                    sample_index=numeric_index or (sample_idx + 1),
                )
            )

    if not features:
        raise ValueError(f"No valid samples found in '{excel_path}'.")

    feature_matrix = np.vstack(features).astype(np.uint8)
    return feature_matrix, metadata, sheet_stats


def cluster_feature_matrix(
    feature_matrix: np.ndarray,
    target_clusters: int,
) -> Tuple[np.ndarray, KMedoids]:
    """Apply k-medoids clustering with Hamming distance."""

    if feature_matrix.shape[0] == 0:
        raise ValueError("Cannot cluster an empty feature matrix.")

    cluster_count = min(target_clusters, feature_matrix.shape[0])
    labels: np.ndarray | None = None
    model: KMedoids | None = None

    for attempt in range(KMEDOIDS_MAX_RESTARTS):
        model = KMedoids(
            n_clusters=cluster_count,
            metric="hamming",
            init="k-medoids++",
            random_state=RANDOM_STATE + attempt,
        )
        labels = model.fit_predict(feature_matrix)
        if np.unique(labels).size == cluster_count:
            break
    else:
        assert model is not None and labels is not None
        print(
            "[cluster] Warning: some medoids ended empty after"
            f" {KMEDOIDS_MAX_RESTARTS} attempts; proceeding with"
            f" {np.unique(labels).size} active clusters."
        )

    assert labels is not None and model is not None
    return labels, model


def export_cluster_results(
    excel_path: Path,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    metadata: List[SampleMeta],
    sample_rows: int,
    model: KMedoids,
    output_suffix: str,
    coverage_targets: Optional[Sequence[int]],
    coverage_thresholds: Optional[Sequence[float]],
    output_dir: Optional[Path] = None,
) -> Path:
    """Persist clustering assignments and summaries to a companion workbook."""

    if feature_matrix.shape[0] != len(metadata):
        raise ValueError(
            "Feature matrix row count does not match metadata entries; cannot export results."
        )

    assignments = pd.DataFrame(
        {
            "excel_file": [meta.excel_file for meta in metadata],
            "sheet_name": [meta.sheet_name for meta in metadata],
            "sample_index": [meta.sample_index for meta in metadata],
            "sample_uid": [meta.make_uid() for meta in metadata],
            "cluster_id": labels.astype(int),
        }
    )

    cluster_summary = (
        assignments.groupby("cluster_id")
        .agg(
            sample_count=("sample_uid", "size"),
            unique_sheets=("sheet_name", "nunique"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )
    cluster_summary["sample_fraction"] = (
        cluster_summary["sample_count"] / len(assignments)
    ).round(6)

    confidence_df = compute_cluster_confidence_intervals(
        feature_matrix,
        labels,
        sample_rows,
        model,
    )

    medoid_indices = getattr(model, "medoid_indices_", None)
    coverage_df = evaluate_medoid_coverage(
        feature_matrix,
        cluster_summary,
        medoid_indices,
        coverage_targets,
        coverage_thresholds,
    )

    column_headers = [f"Col_{idx+1:02d}" for idx in range(TARGET_COLUMNS)]
    cluster_ids = np.sort(assignments["cluster_id"].unique().astype(int))

    prob_center_dict: Dict[int, np.ndarray] = {}
    binary_center_dict: Dict[int, np.ndarray] = {}
    medoid_index_dict: Dict[int, int] = {}

    for cluster_id in cluster_ids:
        cluster_mask = labels == cluster_id
        cluster_indices = np.flatnonzero(cluster_mask)
        if cluster_indices.size == 0:
            continue

        cluster_samples = feature_matrix[cluster_indices].astype(float)
        prob_center = cluster_samples.mean(axis=0)
        prob_center_dict[cluster_id] = prob_center
        binary_center_dict[cluster_id] = (prob_center >= 0.5).astype(int)
        if medoid_indices is not None and 0 <= cluster_id < len(medoid_indices):
            medoid_index_dict[cluster_id] = int(medoid_indices[cluster_id])

    representative_frames: List[pd.DataFrame] = []
    representative_records: List[dict] = []

    for cluster_id in cluster_ids:
        cluster_mask = labels == cluster_id
        cluster_indices = np.flatnonzero(cluster_mask)
        if cluster_indices.size == 0:
            continue

        best_global_index = medoid_index_dict.get(cluster_id)
        if best_global_index is None or best_global_index >= feature_matrix.shape[0]:
            cluster_samples = feature_matrix[cluster_indices]
            prob_center = prob_center_dict.get(cluster_id)
            if prob_center is None:
                continue
            distances = np.mean(cluster_samples != prob_center, axis=1)
            best_local = int(np.argmin(distances))
            best_global_index = int(cluster_indices[best_local])

        best_sample = feature_matrix[best_global_index].reshape(
            sample_rows, TARGET_COLUMNS
        ).astype(int)
        best_meta = metadata[best_global_index]

        df_block = pd.DataFrame(best_sample, columns=column_headers, dtype=int)
        df_block.insert(0, "row_in_sample", np.arange(1, sample_rows + 1))
        df_block.insert(0, "sample_uid", best_meta.make_uid())
        df_block.insert(0, "sheet_name", best_meta.sheet_name)
        df_block.insert(0, "cluster_id", cluster_id)
        representative_frames.append(df_block)

        representative_records.append(
            {
                "cluster_id": cluster_id,
                "sheet_name": best_meta.sheet_name,
                "sample_index": best_meta.sample_index,
                "sample_uid": best_meta.make_uid(),
            }
        )

    if representative_frames:
        cluster_representatives = pd.concat(representative_frames, ignore_index=True)
    else:
        cluster_representatives = pd.DataFrame(
            columns=["cluster_id", "sheet_name", "sample_uid", "row_in_sample", *column_headers]
        )

    centroid_prob_frames: List[pd.DataFrame] = []
    centroid_binary_frames: List[pd.DataFrame] = []

    for cluster_id in cluster_ids:
        prob_center = prob_center_dict.get(cluster_id)
        if prob_center is None:
            continue

        prob_block = prob_center.reshape(sample_rows, TARGET_COLUMNS)
        df_prob = pd.DataFrame(prob_block, columns=column_headers)
        df_prob.insert(0, "row_in_sample", np.arange(1, sample_rows + 1))
        df_prob.insert(0, "cluster_id", cluster_id)
        centroid_prob_frames.append(df_prob)

        binary_block = binary_center_dict[cluster_id].reshape(
            sample_rows, TARGET_COLUMNS
        )
        df_binary = pd.DataFrame(binary_block, columns=column_headers, dtype=int)
        df_binary.insert(0, "row_in_sample", np.arange(1, sample_rows + 1))
        df_binary.insert(0, "cluster_id", cluster_id)
        centroid_binary_frames.append(df_binary)

    centroid_prob = pd.concat(centroid_prob_frames, ignore_index=True)
    centroid_prob_binary = pd.concat(centroid_binary_frames, ignore_index=True)

    representative_overview = pd.DataFrame(representative_records)

    output_name = f"{excel_path.stem}_{output_suffix}.xlsx"
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name
    else:
        output_path = excel_path.with_name(output_name)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        assignments.to_excel(writer, sheet_name="cluster_assignments", index=False)
        cluster_summary.to_excel(writer, sheet_name="cluster_summary", index=False)
        cluster_representatives.to_excel(
            writer, sheet_name="cluster_representatives", index=False
        )
        centroid_prob.to_excel(writer, sheet_name="cluster_centers_prob", index=False)
        centroid_prob_binary.to_excel(
            writer, sheet_name="cluster_centers_binary", index=False
        )
        representative_overview.to_excel(
            writer, sheet_name="cluster_representative_map", index=False
        )
        confidence_df.to_excel(
            writer, sheet_name="cluster_confidence_intervals", index=False
        )
        if coverage_df is not None:
            coverage_df.to_excel(writer, sheet_name="medoid_coverage", index=False)

    plot_path = plot_cluster_confidence_intervals(confidence_df, output_path)
    if plot_path is not None:
        print(f"  Confidence interval figure saved: {plot_path.name}")
    else:
        print("  matplotlib unavailable, skipped confidence interval figure.")

    distance_plot_path = plot_cluster_distance_intervals(confidence_df, output_path)
    if distance_plot_path is not None:
        print(f"  Cluster compactness figure saved: {distance_plot_path.name}")
    else:
        print("  matplotlib unavailable, skipped compactness figure.")

    coverage_plot_path = plot_cluster_coverage_summary(confidence_df, output_path)
    if coverage_plot_path is not None:
        print(f"  Coverage trend figure saved: {coverage_plot_path.name}")
    else:
        print("  matplotlib unavailable, skipped coverage trend figure.")

    medoid_coverage_plot_path = plot_medoid_coverage_curves(coverage_df, output_path)
    if medoid_coverage_plot_path is not None:
        print(f"  Medoid coverage figure saved: {medoid_coverage_plot_path.name}")
    else:
        print("  matplotlib unavailable or coverage data missing, skipped medoid coverage figure.")

    return output_path


def process_workbook(
    excel_path: Path,
    sample_rows: int,
    target_clusters: int,
    component_prefix: str,
    coverage_targets: Optional[Sequence[int]],
    coverage_thresholds: Optional[Sequence[float]],
    output_dir: Optional[Path] = None,
) -> Path:
    """Load, cluster, and export results for a single workbook."""

    print(f"Processing workbook: {excel_path.name}")
    feature_matrix, metadata, sheet_stats = load_samples_from_excel(
        excel_path, sample_rows, component_prefix
    )
    for sheet_name, sample_count in sheet_stats:
        print(f"  Worksheet {sheet_name}: extracted {sample_count} samples")
    print(f"  Total samples {feature_matrix.shape[0]}, start clustering...")
    labels, model = cluster_feature_matrix(feature_matrix, target_clusters)
    active_clusters = int(np.unique(labels).size)
    output_suffix = f"k{active_clusters}_clusters"
    output_path = export_cluster_results(
        excel_path,
        feature_matrix,
        labels,
        metadata,
        sample_rows,
        model,
        output_suffix,
        coverage_targets,
        coverage_thresholds,
        output_dir,
    )
    print(
        "  Clustering finished, results saved: {name} (sheets: cluster_assignments, "
        "cluster_summary, cluster_representatives, cluster_centers_prob, "
        "cluster_centers_binary, cluster_representative_map, cluster_confidence_intervals, "
        "medoid_coverage)".format(
            name=output_path.name
        )
    )
    return output_path


def iter_workbooks(root_dir: Path) -> List[Path]:
    """Yield workbook paths under ``root_dir`` while skipping temporary files."""

    workbooks: List[Path] = []
    for excel_path in sorted(root_dir.rglob("*.xlsx")):
        if excel_path.name.startswith("~$"):
            continue
        if "_k" in excel_path.stem and excel_path.stem.endswith("_clusters"):
            # Skip cluster result files
            continue
        workbooks.append(excel_path)
    return workbooks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster MC simulation samples for each typhoon intensity level "
            "and export the assignments."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "new_dataset" / "MC_results",
        help="Directory that contains the intensity subfolders with Excel files.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help="Number of consecutive rows that form a single sample.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=DEFAULT_CLUSTER_COUNT,
        help="Target number of clusters per workbook.",
    )
    parser.add_argument(
        "--component-prefix",
        type=str,
        default=DEFAULT_COMPONENT_PREFIX,
        help="Prefix used to filter eligible component rows, e.g. Line_.",
    )
    parser.add_argument(
        "--coverage-targets",
        type=int,
        nargs="*",
        default=list(DEFAULT_COVERAGE_TARGETS),
        help=(
            "Top-N cluster counts (e.g. 10 20 50 100) for evaluating medoid coverage. "
            "Counts greater than the actual number of clusters will be ignored."
        ),
    )
    parser.add_argument(
        "--coverage-thresholds",
        type=float,
        nargs="*",
        default=list(DEFAULT_COVERAGE_THRESHOLDS),
        help=(
            "Normalized Hamming distance thresholds (0-1) for medoid coverage curves."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory where clustered workbooks and figures are stored.",
    )

    args = parser.parse_args()

    workbooks = iter_workbooks(args.root)
    if not workbooks:
        raise FileNotFoundError(
            f"No Excel workbooks were found under '{args.root}'."
        )

    for workbook in workbooks:
        output_path = process_workbook(
            workbook,
            args.sample_rows,
            args.clusters,
            args.component_prefix,
            args.coverage_targets,
            args.coverage_thresholds,
            args.output_dir,
        )
    print(f"Done: {workbook.relative_to(args.root)} -> {output_path.name}\n")


if __name__ == "__main__":
    main()
