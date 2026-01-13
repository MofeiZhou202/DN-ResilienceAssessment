"""Reusable helpers for Net-A lifecycle inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from src.models.train_init_models import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    Normalization,
    Regressor,
)

torch.serialization.add_safe_globals([Normalization])

INIT_MODEL_DIR = Path("checkpoints/init_models")
LEVEL_MODEL_MAP = {
    1: "01_TD",
    2: "02_TS",
    3: "03_STS",
    4: "04_TY",
    5: "05_STY",
    6: "06_SuperTY",
}
TARGET_FIELD_MAP = {
    "Lath": "latitude",
    "Lngh": "longitude",
    "DeltaP": "deltaP",
    "Vmax": "vmax",
    "Rmw": "rmw",
    "Heading": "heading",
    "Transspeed": "transSpeed",
    "Fc": "fc",
    "HollandB": "hollandB",
}
INIT_MODEL_CACHE: Dict[int, Dict[str, Any]] = {}


def _infer_hidden_sizes(state_dict: Dict[str, torch.Tensor]) -> List[int]:
    linear_keys: List[tuple[int, str]] = []
    for key in state_dict.keys():
        if key.startswith("net") and key.endswith("weight"):
            try:
                index = int(key.split(".")[1])
            except (IndexError, ValueError):
                index = len(linear_keys)
            linear_keys.append((index, key))
    linear_keys.sort(key=lambda item: item[0])
    dims = [int(state_dict[key].shape[0]) for _, key in linear_keys]
    if not dims:
        raise ValueError("无法从模型权重推断隐藏层结构")
    if dims[-1] != len(TARGET_COLUMNS):
        raise ValueError("模型输出维度与预期不一致，无法加载")
    return dims[:-1]


def load_init_model(level: int) -> Dict[str, Any]:
    if level not in LEVEL_MODEL_MAP:
        raise ValueError(f"未知的台风等级: {level}")
    cached = INIT_MODEL_CACHE.get(level)
    if cached is not None:
        return cached

    model_name = LEVEL_MODEL_MAP[level]
    checkpoint_path = INIT_MODEL_DIR / f"{model_name}_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到等级 {level} 对应的模型: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict: Dict[str, torch.Tensor] = payload["state_dict"]
    feature_norm: Normalization = payload["feature_norm"]
    target_norm: Normalization = payload["target_norm"]

    hidden_sizes = _infer_hidden_sizes(state_dict)
    model = Regressor(len(INPUT_COLUMNS), hidden_sizes, len(TARGET_COLUMNS))
    model.load_state_dict(state_dict)
    model.eval()

    cache_entry = {
        "model": model,
        "feature_norm": feature_norm,
        "target_norm": target_norm,
    }
    INIT_MODEL_CACHE[level] = cache_entry
    return cache_entry


def initial_condition_to_features(initial_condition: Iterable[float]) -> Dict[str, float]:
    initial_list = list(initial_condition)
    if len(initial_list) < 9:
        raise ValueError("initial_condition must contain at least 9 values")
    return {
        "InitLatitude": float(initial_list[2]),
        "InitLongitude": float(initial_list[3]),
        "InitDeltaP": float(initial_list[4]),
        "InitIR": float(initial_list[5]),
        "InitRmw": float(initial_list[6]),
        "InitTheta": float(initial_list[7]),
        "InitTransSpeed": float(initial_list[8]),
    }


def predict_lifecycle(level: int, base_inputs: Dict[str, float], steps: Iterable[float]) -> List[Dict[str, float]]:
    bundle = load_init_model(level)
    model: Regressor = bundle["model"]
    feature_norm: Normalization = bundle["feature_norm"]
    target_norm: Normalization = bundle["target_norm"]

    lifecycle: List[Dict[str, float]] = []
    with torch.no_grad():
        for step in steps:
            features: List[float] = []
            for column in INPUT_COLUMNS:
                if column == "Step":
                    features.append(float(step))
                else:
                    features.append(float(base_inputs[column]))
            inputs_tensor = torch.tensor(features, dtype=torch.float32)
            normalized_inputs = feature_norm.normalize(inputs_tensor)
            normalized_output = model(normalized_inputs.unsqueeze(0)).squeeze(0)
            denormalized = target_norm.denormalize(normalized_output)
            step_result = {
                target_key: float(value)
                for target_key, value in zip(TARGET_COLUMNS, denormalized.tolist())
            }
            lifecycle.append(step_result)
    return lifecycle


def map_targets_to_response(target_values: Dict[str, float]) -> Dict[str, float]:
    return {
        output_key: float(target_values[target_key])
        for target_key, output_key in TARGET_FIELD_MAP.items()
    }


__all__ = [
    "LEVEL_MODEL_MAP",
    "TARGET_FIELD_MAP",
    "initial_condition_to_features",
    "load_init_model",
    "map_targets_to_response",
    "predict_lifecycle",
]