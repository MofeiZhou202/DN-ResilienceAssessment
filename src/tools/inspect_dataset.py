"""数据集检查工具。

示例:
    python inspect_dataset.py --npz dataset/generated/test_dataset.npz --list 10
    python inspect_dataset.py --npz dataset/generated/test_dataset.npz --sample 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from src.config import GENERATED_DATA_DIR


def load_npz(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "lifecycle_tensor": data["lifecycle_tensor"],
        "lifecycle_meta": data["lifecycle_meta"].item(),
        "impact_data": data["impact_data"].item(),
        "wind_data": data["wind_data"].item(),
        "mc_data": data["mc_data"].item(),
    }


def print_summary(bundle: Dict[str, Any]) -> None:
    lifecycle = bundle["lifecycle_tensor"]
    meta = bundle["lifecycle_meta"]
    impact = bundle["impact_data"]
    wind = bundle["wind_data"]
    mc = bundle["mc_data"]

    print("=== 数据集概要 ===")
    print(f"样本数量: {lifecycle.shape[0]}")
    print(f"统一时间步长: {lifecycle.shape[1]}")
    print(f"特征维度: {lifecycle.shape[2]}")
    print(f"可用影响 sheets: {list(impact.keys())}")
    print(f"可用风电 sheets: {list(wind.keys())}")
    print(f"蒙特卡洛 sheets: {list(mc.keys())}")
    print(f"前 5 条样本 ID: {meta['ids'][:5]}")


def list_samples(bundle: Dict[str, Any], count: int) -> None:
    meta = bundle["lifecycle_meta"]
    impact = bundle["impact_data"]
    wind = bundle["wind_data"]
    ids = meta["ids"]
    files = meta["files"]
    lengths = meta["lengths"]

    print("=== 样本清单 ===")
    for idx in range(min(count, len(ids))):
        sheet = str(ids[idx]).split(":")[-1]
        print(f"#{idx:03d} | id={ids[idx]} | file={files[idx]} | length={lengths[idx]}")
        print(f"       impact_sheet={sheet in impact} wind_sheet={sheet in wind}")


def show_sample(bundle: Dict[str, Any], index: int) -> None:
    lifecycle = bundle["lifecycle_tensor"]
    meta = bundle["lifecycle_meta"]
    impact = bundle["impact_data"]
    wind = bundle["wind_data"]

    if index < 0 or index >= lifecycle.shape[0]:
        raise IndexError(f"索引 {index} 超出范围 0..{lifecycle.shape[0]-1}")

    sample_len = int(meta["lengths"][index])
    sample_id = meta["ids"][index]
    sheet = str(sample_id).split(":")[-1]

    seq = np.nan_to_num(lifecycle[index][:sample_len], nan=0.0)
    impact_matrix = impact[sheet]["matrix"][:, :sample_len]
    wind_matrix = wind[sheet]["matrix"][:, :sample_len]

    payload = {
        "id": sample_id,
        "file": meta["files"][index],
        "length": sample_len,
        "feature_names": meta.get("feature_names", []),
        "lifecycle_first_row": seq[0].tolist() if sample_len else [],
        "impact_shape": impact_matrix.shape,
        "impact_first_row": impact_matrix[0].tolist() if impact_matrix.size else [],
        "wind_shape": wind_matrix.shape,
        "wind_first_row": wind_matrix[0].tolist() if wind_matrix.size else [],
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 prepare_dataset 生成的 npz")
    parser.add_argument(
        "--npz",
        default=str(GENERATED_DATA_DIR / "test_dataset.npz"),
        help="npz 文件路径",
    )
    parser.add_argument("--list", type=int, default=0, help="列出前多少条样本")
    parser.add_argument("--sample", type=int, default=-1, help="查看指定索引的样本细节")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.npz)
    if not path.exists():
        raise FileNotFoundError(f"npz 文件不存在: {path}")

    bundle = load_npz(path)
    print_summary(bundle)

    if args.list > 0:
        list_samples(bundle, args.list)

    if args.sample >= 0:
        show_sample(bundle, args.sample)


if __name__ == "__main__":
    main()
