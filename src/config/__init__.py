"""Centralized project paths and shared constants."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "dataset"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = (DATASET_DIR / "raw").resolve()
GENERATED_DATA_DIR = (DATASET_DIR / "generated").resolve()
OUTPUT_DIR = (ROOT_DIR / "output").resolve()
TEMP_DIR = (ROOT_DIR / "temp").resolve()

for directory in (RAW_DATA_DIR, GENERATED_DATA_DIR, OUTPUT_DIR, TEMP_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _dir_str(path: Path) -> str:
    """Return a string path with a trailing separator for legacy joins."""
    value = str(path)
    if not value.endswith(os.sep):
        value += os.sep
    return value


SST_DATA_FILE = RAW_DATA_DIR / "SST.mat"
HURRICANE_DATA_DIR = RAW_DATA_DIR
INFRASTRUCTURE_DATA_DIR = RAW_DATA_DIR
GENERATED_API_DIR = GENERATED_DATA_DIR / "api_results"
MC_RESULTS_DIR = GENERATED_DATA_DIR / "MC_results"
CLUSTER_RESULTS_DIR = GENERATED_DATA_DIR / "cluster"

GENERATED_API_DIR.mkdir(parents=True, exist_ok=True)
MC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CLUSTER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Backwards compatibility aliases -------------------------------------------------
excel_file_path = str(RAW_DATA_DIR / "wind_farms_output.xlsx")
SST_data_path = str(SST_DATA_FILE)
hurricane_data_path = _dir_str(HURRICANE_DATA_DIR)
infrastructure_data_path = _dir_str(INFRASTRUCTURE_DATA_DIR)


# Geographic constants -----------------------------------------------------------
ORIGIN_LAT_LON = (19.255954070367995, 110.9011439735295)
LEGACY_ORIGIN_LAT_LON = (27.8, -90.3)


__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "DATASET_DIR",
    "RAW_DATA_DIR",
    "GENERATED_DATA_DIR",
    "OUTPUT_DIR",
    "TEMP_DIR",
    "SST_DATA_FILE",
    "HURRICANE_DATA_DIR",
    "INFRASTRUCTURE_DATA_DIR",
    "GENERATED_API_DIR",
    "MC_RESULTS_DIR",
    "CLUSTER_RESULTS_DIR",
    "excel_file_path",
    "SST_data_path",
    "hurricane_data_path",
    "infrastructure_data_path",
    "ORIGIN_LAT_LON",
    "LEGACY_ORIGIN_LAT_LON",
]
