"""High-level helpers for batch hurricane lifecycle simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.typhoon.initialization import hurricane_initialization
from src.typhoon.simulation import run_hurricane_simulation

LIFECYCLE_COLUMNS = [
    "Lath",
    "Lngh",
    "Fc",
    "DeltaP",
    "HollandB",
    "Rmw",
    "Heading",
    "Transspeed",
    "Vmax",
]


@dataclass
class HurricaneRecord:
    sheet_name: str
    dataframe: pd.DataFrame
    initial_condition: List[float]
    month: int


class SeasonalHurricaneSimulator:
    """Generate physical hurricane lifecycles across years and seasons."""

    def __init__(
        self,
        *,
        start_year: int,
        end_year: int,
        storms_per_year: int = 5,
        months: Sequence[int] | None = None,
        sim_duration: int = 48,
        rng_seed: int | None = None,
    ) -> None:
        if end_year < start_year:
            raise ValueError("end_year must be greater than or equal to start_year")
        if storms_per_year <= 0:
            raise ValueError("storms_per_year must be positive")
        if sim_duration <= 0:
            raise ValueError("sim_duration must be positive")
        self.start_year = start_year
        self.end_year = end_year
        self.storms_per_year = storms_per_year
        self.months = list(months) if months else list(range(6, 12))
        self.sim_duration = sim_duration
        self.rng = np.random.default_rng(rng_seed)

    def _pick_month(self) -> int:
        return int(self.rng.choice(self.months))

    def simulate(self) -> List[HurricaneRecord]:
        """Simulate hurricanes and return per-sheet dataframes."""

        records: List[HurricaneRecord] = []
        sheet_counter = 1
        for year in range(self.start_year, self.end_year + 1):
            for _ in range(self.storms_per_year):
                initial_condition = hurricane_initialization()
                month = self._pick_month()
                lifecycle = run_hurricane_simulation(initial_condition, month, self.sim_duration)
                df = pd.DataFrame(lifecycle, columns=LIFECYCLE_COLUMNS)
                sheet_name = f"{year}_{sheet_counter:03d}"
                sheet_counter += 1
                records.append(
                    HurricaneRecord(
                        sheet_name=sheet_name,
                        dataframe=df,
                        initial_condition=list(map(float, initial_condition)),
                        month=month,
                    )
                )
        return records

    def to_excel(self, output_path: Path | str) -> Path:
        """Simulate and write the results into a multi-sheet Excel file."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self.simulate()
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for record in records:
                df = record.dataframe.copy()
                for idx, column in enumerate(
                    [
                        "InitLatitude",
                        "InitLongitude",
                        "InitDeltaP",
                        "InitIR",
                        "InitRmw",
                        "InitTheta",
                        "InitTransSpeed",
                    ]
                ):
                    if column not in df.columns:
                        df[column] = None
                    df.at[0, column] = record.initial_condition[2 + idx]
                df.to_excel(writer, sheet_name=record.sheet_name[:31], index=False)
        return path


__all__ = ["SeasonalHurricaneSimulator", "HurricaneRecord", "LIFECYCLE_COLUMNS"]