from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from train_init_models import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    Normalization,
    Regressor,
    load_workbook_frames,
)

torch.serialization.add_safe_globals([Normalization])

DATA_ROOT = Path("new_dataset/classified_typhoons")
CHECKPOINT_ROOT = Path("checkpoints/init_models")
OUTPUT_PATH = Path("output/init_single_prediction.xlsx")

INPUT_SAMPLE: Dict[str, float] = {
    "InitLatitude": 21.3979353,
    "InitLongitude": 116.6839997,
    "InitDeltaP": 46.12501565,
    "InitIR": 0.556285239,
    "InitRmw": 44.7543777,
    "InitTheta": 1.165574874,
    "InitTransSpeed": 5.400026797,
}

def _resolve_steps(workbook_name: str) -> List[float]:
    workbook_path = DATA_ROOT / f"{workbook_name}.xlsx"
    if not workbook_path.exists():
        raise FileNotFoundError(f"未找到 {workbook_path}")
    frames = load_workbook_frames(workbook_path)
    max_step = 0.0
    for frame in frames.values():
        if "Step" not in frame.columns:
            continue
        max_step = max(max_step, float(frame["Step"].max()))
    total_steps = int(max_step) + 1
    return [float(step) for step in range(total_steps)]


def infer_hidden_sizes(state_dict: Dict[str, torch.Tensor], output_dim: int) -> List[int]:
    linear_layers = [key for key in state_dict.keys() if key.endswith("weight")]
    linear_layers.sort()
    dims: List[int] = []
    for key in linear_layers:
        weight = state_dict[key]
        dims.append(int(weight.shape[0]))
    if not dims or dims[-1] != output_dim:
        raise ValueError("无法从权重推断出网络结构")
    return dims[:-1]


def make_predictions() -> Dict[str, pd.DataFrame]:
    forecasts: Dict[str, pd.DataFrame] = {}
    for checkpoint_path in sorted(CHECKPOINT_ROOT.glob("*_model.pt")):
        workbook_name = checkpoint_path.stem.replace("_model", "")
        steps = _resolve_steps(workbook_name)

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict: Dict[str, torch.Tensor] = payload["state_dict"]
        feature_norm: Normalization = payload["feature_norm"]
        target_norm: Normalization = payload["target_norm"]

        hidden_sizes = infer_hidden_sizes(state_dict, len(TARGET_COLUMNS))
        model = Regressor(len(INPUT_COLUMNS), hidden_sizes, len(TARGET_COLUMNS))
        model.load_state_dict(state_dict)
        model.eval()

        rows: List[Dict[str, float]] = []
        for step in steps:
            feature_values: List[float] = []
            for col in INPUT_COLUMNS:
                if col == "Step":
                    feature_values.append(step)
                else:
                    feature_values.append(float(INPUT_SAMPLE[col]))
            inputs_tensor = torch.tensor(feature_values, dtype=torch.float32)
            normalized_inputs = feature_norm.normalize(inputs_tensor)
            with torch.no_grad():
                normalized_output = model(normalized_inputs.unsqueeze(0)).squeeze(0)
            predicted = target_norm.denormalize(normalized_output)
            row = {"Step": step}
            for col, value in zip(TARGET_COLUMNS, predicted.tolist()):
                row[col] = float(value)
            rows.append(row)
        forecasts[workbook_name] = pd.DataFrame(rows)
    return forecasts


def main() -> None:
    forecasts = make_predictions()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        for name, frame in forecasts.items():
            sheet_name = name[:31]
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"{name} 预测 {len(frame)} 条")
            print(frame.head().to_string(index=False))
    print(f"预测结果写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
