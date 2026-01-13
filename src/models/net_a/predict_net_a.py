"""使用训练好的 Net-A 模型对新的台风生命周期 Excel 进行推断，输出预测概率 Excel。

示例:


python train_net_a.py `
    --hurricane-excel dataset/batch_hurricane.xlsx `
    --impact-excel dataset/batch_hurricane_impact_simple.xlsx `
    --test-last 200 `
    --seed 42 `
    --epochs 120 `
    --batch-size 32 `
    --focal-gamma 1.5 `
    --neg-weight 3.0 `
    --grad-clip 1.0

脚本会逐个工作表读取台风特征，使用模型计算输电线路的故障概率，生成与简化影响结果相同结构的 Excel 便于对比。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from train_net_a import TyphoonImpactNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Net-A 台风影响预测")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/net_a_best.pt",
        help="模型权重文件路径 (默认使用最近训练生成的 checkpoints/net_a_best.pt)",
    )
    parser.add_argument(
        "--hurricane-excel",
        default="testset/batch_hurricane.xlsx",
        help="需要推断的台风生命周期 Excel (默认读取 testset/batch_hurricane.xlsx)",
    )
    parser.add_argument(
        "--reference-impact",
        default="dataset/batch_hurricane_impact_simple.xlsx",
        help="用于提供线路名称/时间列的参考影响 Excel（可选）",
    )
    parser.add_argument("--sheet-limit", type=int, help="仅对前 N 个工作表推断")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="testset/batch_hurricane_predictions.xlsx",
        help="输出预测 Excel 文件路径 (默认保存到 testset/batch_hurricane_predictions.xlsx)",
    )
    return parser.parse_args()


def _sort_sheet_names(sheet_names: List[str]) -> List[str]:
    def sort_key(name: str) -> Tuple[int, str]:
        if str(name).isdigit():
            return (0, int(name))
        # 尝试提取末尾数字
        digits = ''.join(ch for ch in str(name) if ch.isdigit())
        return (1, int(digits) if digits else 0)

    return sorted(sheet_names, key=sort_key)


def load_hurricane_sequences(
    hurricane_path: Path,
    sheet_limit: int | None,
    expected_features: List[str] | None = None,
) -> Tuple[np.ndarray, List[int], List[str]]:
    book = pd.read_excel(hurricane_path, sheet_name=None, engine="openpyxl")
    sheet_names = _sort_sheet_names(list(book.keys()))
    if sheet_limit is not None:
        sheet_names = sheet_names[:sheet_limit]

    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    ids: List[str] = []

    feature_columns: List[str] | None = list(expected_features) if expected_features else None

    for name in sheet_names:
        df = book[name]
        if feature_columns is None:
            feature_columns = list(df.columns)
        if expected_features is not None:
            missing = [col for col in feature_columns if col not in df.columns]
            if missing:
                print(f"[WARN] 工作表 {name} 缺少以下特征列，使用 0 填充: {missing}")
            df = df.reindex(columns=feature_columns)
        else:
            # 确保列顺序一致
            df = df[feature_columns]

        arr = df.to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        sequences.append(arr)
        lengths.append(arr.shape[0])
        ids.append(str(name))

    max_len = max(lengths)
    feature_dim = sequences[0].shape[1]
    if expected_features is not None and feature_dim != len(expected_features):
        raise ValueError(
            f"特征列数量不一致: 模型期望 {len(expected_features)} 列，当前数据有 {feature_dim} 列。"
        )
    tensor = np.zeros((len(sequences), max_len, feature_dim), dtype=np.float32)
    for idx, arr in enumerate(sequences):
        tensor[idx, : arr.shape[0], :] = arr

    return tensor, lengths, ids


def extract_model_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    # LSTM 权重格式: weight_hh_l{k} shape = (4*hidden_dim, hidden_dim)
    hidden_dim = state_dict["encoder.weight_hh_l0"].shape[1]

    layer_indices = []
    prefix = "encoder.weight_hh_l"
    for key in state_dict:
        if key.startswith(prefix):
            remainder = key[len(prefix):]
            idx_str = remainder.split(".")[0]
            idx_str = idx_str.split("_")[0]
            try:
                layer_indices.append(int(idx_str))
            except ValueError:
                continue
    num_layers = max(layer_indices) + 1 if layer_indices else 1

    line_dim = state_dict["line_head.weight"].shape[0]
    wind_dim = state_dict["wind_head.weight"].shape[0] if "wind_head.weight" in state_dict else 0
    return hidden_dim, int(num_layers), line_dim, wind_dim


def load_reference_metadata(
    reference_path: Path | None,
    sheet_names: List[str],
    line_dim: int,
    max_len: int,
) -> Dict[str, Tuple[List[str], List[str]]]:
    metadata: Dict[str, Tuple[List[str], List[str]]] = {}
    reference_book = None
    if reference_path and reference_path.exists():
        reference_book = pd.read_excel(reference_path, sheet_name=None, engine="openpyxl", index_col=0)

    for name in sheet_names:
        line_ids = [f"Line_{idx+1}" for idx in range(line_dim)]
        hours = [f"Hour {idx+1}" for idx in range(max_len)]
        if reference_book and name in reference_book:
            ref_df = reference_book[name]
            ref_line_ids = [str(idx) for idx in ref_df.index]
            ref_hours = [str(col) for col in ref_df.columns]
            if len(ref_line_ids) == line_dim:
                line_ids = ref_line_ids
            if len(ref_hours) >= max_len:
                hours = ref_hours[:max_len]
            else:
                hours = ref_hours + [f"Hour {i+1}" for i in range(len(ref_hours), max_len)]
        metadata[str(name)] = (line_ids, hours)
    return metadata


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    hurricane_path = Path(args.hurricane_excel)
    if not hurricane_path.exists():
        raise FileNotFoundError(f"找不到台风 Excel: {hurricane_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    hidden_dim, num_layers, line_dim, wind_dim = extract_model_dims(state_dict)

    expected_feature_dim = state_dict["encoder.weight_ih_l0"].shape[1]
    feature_names = checkpoint.get("feature_names")
    if feature_names is not None:
        feature_names = [str(col) for col in feature_names]
        if len(feature_names) != expected_feature_dim:
            print(
                "[WARN] checkpoint 中记录的特征列数量与模型结构不一致，按照模型结构截取/补齐。"
            )
            if len(feature_names) > expected_feature_dim:
                feature_names = feature_names[:expected_feature_dim]
            else:
                start = len(feature_names)
                feature_names = feature_names + [f"Feature_{idx+1}" for idx in range(start, expected_feature_dim)]
    else:
        ckpt_args = checkpoint.get("args") or {}
        hurricane_source = ckpt_args.get("hurricane_excel")
        if hurricane_source:
            source_path = Path(hurricane_source)
            if not source_path.exists() and not source_path.is_absolute():
                source_path = (checkpoint_path.parent / hurricane_source).resolve()
            if source_path.exists():
                try:
                    training_book = pd.read_excel(source_path, sheet_name=None, engine="openpyxl")
                    order = _sort_sheet_names(list(training_book.keys()))
                    if order:
                        feature_names = list(training_book[order[0]].columns)
                        print(
                            f"[INFO] 从训练数据 {source_path} 推断到 {len(feature_names)} 个特征列。"
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] 无法从 {source_path} 读取特征列: {exc}")

    lifecycle_tensor, lengths, sheet_ids = load_hurricane_sequences(
        hurricane_path,
        args.sheet_limit,
        expected_features=feature_names,
    )
    feature_dim = lifecycle_tensor.shape[2]
    max_len = lifecycle_tensor.shape[1]

    if feature_dim != expected_feature_dim:
        raise RuntimeError(
            "台风 Excel 的特征列数量与模型输入维度不一致。"
            "请确保使用与训练阶段相同的特征列，或重新训练模型以在 checkpoint 中包含 feature_names。"
        )

    model = TyphoonImpactNet(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        line_dim=line_dim,
        wind_dim=wind_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    metadata = load_reference_metadata(
        Path(args.reference_impact) if args.reference_impact else None,
        sheet_ids,
        line_dim,
        max_len,
    )

    predictions: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for idx, sheet_name in enumerate(sheet_ids):
            seq_tensor = torch.from_numpy(lifecycle_tensor[idx : idx + 1]).to(device)
            length_tensor = torch.tensor([lengths[idx]], device=device)
            line_pred, _ = model(seq_tensor, length_tensor)
            prob = torch.sigmoid(line_pred)[0, : lengths[idx], :].cpu().numpy()
            predictions[sheet_name] = prob

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in sheet_ids:
            prob = predictions[sheet_name]
            line_ids, hours = metadata[sheet_name]
            # 只取有效长度
            hours = hours[: prob.shape[0]]
            line_ids = line_ids[:line_dim]
            df = pd.DataFrame(prob.T, index=line_ids, columns=hours)
            df.to_excel(writer, sheet_name=str(sheet_name)[:31])

    print(f"预测结果已保存至 {output_path}")


if __name__ == "__main__":
    main()
