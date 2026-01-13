"""Classify typhoon workbooks by landfall wind speed and export per-level files."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class WindCategory:
    name: str
    lower: float
    upper: float

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class WorkbookEntry:
    label: str
    path: Path
    month: Optional[int]


DEFAULT_TYPHOON_ROOT = Path(__file__).resolve().parent / "new_dataset" / "typhoons"
DEFAULT_OUTPUT_DIR = Path("new_dataset") / "classified_typhoons"

DEFAULT_CATEGORIES: Tuple[WindCategory, ...] = (
    WindCategory("TD", 0.0, 17.1),
    WindCategory("TS", 17.1, 24.4),
    WindCategory("STS", 24.4, 32.6),
    WindCategory("TY", 32.6, 41.4),
    WindCategory("STY", 41.4, 50.9),
    WindCategory("SuperTY", 50.9, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify typhoon simulations by the first landfall wind speed (latitude >= threshold)."
        ),
    )
    parser.add_argument(
        "--source-excel",
        type=Path,
        help="Optional path to a single workbook. If omitted, the script scans typhoon-root/month_* directories.",
    )
    parser.add_argument(
        "--typhoon-root",
        type=Path,
        default=DEFAULT_TYPHOON_ROOT,
        help="Root directory that contains monthly typhoon workbooks (default: new_dataset/typhoons).",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        help="Restrict processing to the listed months (numbers 1-12).",
    )
    parser.add_argument(
        "--month-range",
        type=str,
        help="Inclusive month range such as 5-11. Overrides --months when provided.",
    )
    parser.add_argument(
        "--hurricane-file-name",
        default="batch_hurricane.xlsx",
        help="Expected workbook name inside each month directory (default: batch_hurricane.xlsx).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives the per-level Excel files (default: new_dataset/classified_typhoons).",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional CSV that records month, sheet, landfall wind speed, and assigned category.",
    )
    parser.add_argument(
        "--vmax-column",
        default="Vmax",
        help="Column name used for maximum wind speed (default: Vmax).",
    )
    parser.add_argument(
        "--lat-column",
        default="Lath",
        help="Column name used for latitude values (default: Lath).",
    )
    parser.add_argument(
        "--lat-threshold",
        type=float,
        default=22.0,
        help="Latitude threshold that represents landfall (default: 22.0 degrees).",
    )
    return parser.parse_args()


def categorize_vmax(value: float, categories: Iterable[WindCategory]) -> WindCategory:
    for category in categories:
        if category.contains(value):
            return category
    raise ValueError(f"No wind speed category matched value {value}. Please adjust thresholds.")


def build_month_filter(months: Optional[Sequence[int]], month_range: Optional[str]) -> Optional[List[int]]:
    if month_range:
        try:
            start_str, end_str = month_range.split("-", 1)
            start_month = int(start_str)
            end_month = int(end_str)
        except ValueError as exc:
            raise ValueError("month-range must look like 5-11") from exc
        if start_month > end_month:
            raise ValueError("month-range start cannot be greater than end")
        values = list(range(start_month, end_month + 1))
    elif months:
        values = list(months)
    else:
        return None

    unique = sorted(set(values))
    for month in unique:
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month: {month}")
    return unique


def extract_month(label: str) -> Optional[int]:
    digits = "".join(ch for ch in label if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def discover_workbooks(args: argparse.Namespace) -> List[WorkbookEntry]:
    if args.source_excel:
        path = args.source_excel.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Source workbook not found: {path}")
        return [WorkbookEntry(label=path.stem, path=path, month=None)]

    month_filter = build_month_filter(args.months, args.month_range)
    root = args.typhoon_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Typhoon root not found: {root}")

    entries: List[WorkbookEntry] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        month_value = extract_month(child.name)
        if month_filter and (month_value is None or month_value not in month_filter):
            continue
        excel_path = (child / args.hurricane_file_name).resolve()
        if not excel_path.exists():
            print(f"[WARN] Skipping {child}: missing {args.hurricane_file_name}")
            continue
        entries.append(WorkbookEntry(label=child.name, path=excel_path, month=month_value))

    if not entries:
        raise RuntimeError("No workbooks discovered. Check typhoon-root and month filters.")
    return entries


def compute_landfall_vmax(
    frame: pd.DataFrame,
    lat_column: str,
    lat_threshold: float,
    vmax_column: str,
) -> Tuple[float, bool]:
    if lat_column not in frame.columns:
        raise KeyError(f"Missing latitude column {lat_column} in worksheet.")
    if vmax_column not in frame.columns:
        raise KeyError(f"Missing wind speed column {vmax_column} in worksheet.")

    lat_series = pd.to_numeric(frame[lat_column], errors="coerce")
    vmax_series = pd.to_numeric(frame[vmax_column], errors="coerce")

    first_valid_idx = lat_series.first_valid_index()
    if first_valid_idx is None:
        raise ValueError("Latitude series is empty after dropping NaN.")
    initial_lat = lat_series.loc[first_valid_idx]
    if initial_lat >= lat_threshold:
        raise ValueError("Initial latitude already exceeds threshold; skipping.")

    landfall_indices = lat_series.index[lat_series >= lat_threshold]
    landfall_indices = [idx for idx in landfall_indices if idx >= first_valid_idx]
    if len(landfall_indices) == 0:
        raise ValueError("Typhoon never reached latitude threshold; skipping.")

    first_idx = landfall_indices[0]
    landfall_value = float(vmax_series.loc[first_idx])
    if pd.isna(landfall_value):
        raise ValueError("Landfall Vmax is NaN; skipping.")
    return landfall_value, False


def write_category_workbooks(
    grouped: Dict[WindCategory, List[Tuple[str, pd.DataFrame]]],
    target_dir: Path,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, (category, items) in enumerate(grouped.items(), start=1):
        if not items:
            continue
        sanitized = category.name.replace("/", "_")
        excel_path = target_dir / f"{index:02d}_{sanitized}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet_name, frame in items:
                writer_sheet = str(sheet_name)[:31]
                frame.to_excel(writer, sheet_name=writer_sheet, index=False)


def classify_workbook(
    entry: WorkbookEntry,
    categories: Iterable[WindCategory],
    grouped: Dict[WindCategory, List[Tuple[str, pd.DataFrame]]],
    lat_column: str,
    lat_threshold: float,
    vmax_column: str,
) -> List[Dict[str, object]]:
    workbook_summary: List[Dict[str, object]] = []

    sheets = pd.read_excel(entry.path, sheet_name=None)
    for sheet_name, frame in sheets.items():
        try:
            value, _ = compute_landfall_vmax(frame, lat_column, lat_threshold, vmax_column)
        except ValueError as exc:
            print(f"[INFO] Skip {entry.label}/{sheet_name}: {exc}")
            continue
        category = categorize_vmax(value, categories)
        grouped[category].append((f"{entry.label}_{sheet_name}", frame))
        meta_snapshot: Dict[str, object] = {}
        for column in frame.columns:
            if column.startswith("Init"):
                column_series = frame[column]
                numeric_series = pd.to_numeric(column_series, errors="coerce")
                cleaned = numeric_series.dropna()
                if not cleaned.empty:
                    meta_snapshot[column] = cleaned.iloc[-1]
                else:
                    non_numeric = column_series.dropna()
                    if not non_numeric.empty:
                        meta_snapshot[column] = non_numeric.iloc[-1]
        workbook_summary.append(
            {
                "workbook_label": entry.label,
                "workbook_path": str(entry.path),
                "month": entry.month,
                "sheet_name": sheet_name,
                "landfall_vmax": value,
                "category": category.name,
                **meta_snapshot,
            }
        )

    return workbook_summary


def write_summary(summary: List[Dict[str, object]], summary_path: Path) -> None:
    frame = pd.DataFrame(summary)
    frame.sort_values(
        by=["month", "landfall_vmax", "sheet_name"],
        ascending=[True, False, True],
        inplace=True,
        na_position="last",
    )
    frame.to_csv(summary_path, index=False)


def main() -> None:
    args = parse_args()

    try:
        entries = discover_workbooks(args)
    except Exception as exc:
        raise SystemExit(f"Failed to locate workbooks: {exc}") from exc

    overall_summary: List[Dict[str, object]] = []
    all_grouped: Dict[WindCategory, List[Tuple[str, pd.DataFrame]]] = {cat: [] for cat in DEFAULT_CATEGORIES}
    for entry in entries:
        print(f"[INFO] Processing {entry.label} ({entry.path})")
        try:
            summary = classify_workbook(
                entry,
                categories=DEFAULT_CATEGORIES,
                grouped=all_grouped,
                lat_column=args.lat_column,
                lat_threshold=args.lat_threshold,
                vmax_column=args.vmax_column,
            )
            overall_summary.extend(summary)
        except Exception as exc:
            print(f"[ERROR] Failed to classify {entry.path}: {exc}")

    write_category_workbooks(all_grouped, args.output_dir)
    print(f"Classification finished. Results saved under {args.output_dir.resolve()}.")
    if args.summary_path and overall_summary:
        write_summary(overall_summary, args.summary_path)
        print(f"Summary written to {args.summary_path.resolve()}.")


if __name__ == "__main__":
    main()
