from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

INPUT_COLUMNS: List[str] = [
    "InitLatitude",
    "InitLongitude",
    "InitDeltaP",
    "InitIR",
    "InitRmw",
    "InitTheta",
    "InitTransSpeed",
    "Step",
]

TARGET_COLUMNS: List[str] = [
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
class Normalization:
    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean


class SheetRegressionDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        feature_norm: Normalization,
        target_norm: Normalization,
    ) -> None:
        self.features = feature_norm.normalize(features)
        self.targets = target_norm.normalize(targets)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class Regressor(nn.Module):
    def __init__(self, input_dim: int, hidden: Iterable[int], output_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def list_workbooks(root: Path) -> List[Path]:
    return sorted([p for p in root.glob("*.xlsx") if not p.name.startswith("~$")])


def load_workbook_frames(book_path: Path) -> Dict[str, pd.DataFrame]:
    book = pd.read_excel(book_path, sheet_name=None, engine="openpyxl")
    frames: Dict[str, pd.DataFrame] = {}
    for sheet_name, frame in book.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        frame = frame.copy()
        frame["Step"] = np.arange(len(frame), dtype=float)
        required = INPUT_COLUMNS + TARGET_COLUMNS
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"{book_path.name} / {sheet_name}: 缺少列 {missing}")
        prepared = frame[required].copy()
        init_cols = [col for col in INPUT_COLUMNS if col != "Step"]
        if init_cols:
            prepared[init_cols] = prepared[init_cols].ffill().bfill()
        prepared = prepared.dropna(subset=required)
        if prepared.empty:
            continue
        prepared["__sheet__"] = sheet_name
        frames[str(sheet_name)] = prepared
    if not frames:
        raise ValueError(f"{book_path.name}: 没有可用数据工作表")
    return frames


def split_frames(
    frames: Dict[str, pd.DataFrame],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not frames:
        return (
            pd.DataFrame(columns=INPUT_COLUMNS + TARGET_COLUMNS),
            pd.DataFrame(columns=INPUT_COLUMNS + TARGET_COLUMNS),
            pd.DataFrame(columns=INPUT_COLUMNS + TARGET_COLUMNS),
        )

    full_df = pd.concat(frames.values(), ignore_index=True)
    if full_df.empty:
        return (
            pd.DataFrame(columns=INPUT_COLUMNS + TARGET_COLUMNS),
            pd.DataFrame(columns=INPUT_COLUMNS + TARGET_COLUMNS),
            pd.DataFrame(columns=INPUT_COLUMNS + TARGET_COLUMNS),
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(full_df))
    rng.shuffle(indices)

    train_end = max(1, int(round(len(indices) * train_ratio)))
    val_end = int(round(len(indices) * (train_ratio + val_ratio)))
    val_end = min(len(indices), max(train_end + 1 if len(indices) > 1 else len(indices), val_end))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def _subset(idx: np.ndarray) -> pd.DataFrame:
        if idx.size == 0:
            return pd.DataFrame(columns=INPUT_COLUMNS + TARGET_COLUMNS)
        return full_df.iloc[idx].reset_index(drop=True)

    return _subset(train_idx), _subset(val_idx), _subset(test_idx)


def to_tensors(frame: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    if frame.empty:
        return torch.zeros((0, len(INPUT_COLUMNS))), torch.zeros((0, len(TARGET_COLUMNS)))
    features = torch.tensor(frame[INPUT_COLUMNS].to_numpy(dtype=np.float32))
    targets = torch.tensor(frame[TARGET_COLUMNS].to_numpy(dtype=np.float32))
    return features, targets


def build_norm(tensor: torch.Tensor) -> Normalization:
    if tensor.shape[0] == 0:
        raise ValueError("训练集为空，无法计算归一化参数")
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0).clamp_min(1e-6)
    return Normalization(mean=mean, std=std)


def train_one_workbook(
    book_path: Path,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    hidden: List[int],
    lr: float,
    seed: int,
) -> Dict[str, float]:
    frames = load_workbook_frames(book_path)
    train_df, val_df, test_df = split_frames(frames, 0.7, 0.15, seed)

    train_x, train_y = to_tensors(train_df)
    val_x, val_y = to_tensors(val_df)
    test_x, test_y = to_tensors(test_df)

    feature_norm = build_norm(train_x)
    target_norm = build_norm(train_y)

    train_dataset = SheetRegressionDataset(train_x, train_y, feature_norm, target_norm)
    val_dataset = SheetRegressionDataset(val_x, val_y, feature_norm, target_norm) if len(val_df) else None
    test_dataset = SheetRegressionDataset(test_x, test_y, feature_norm, target_norm) if len(test_df) else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    model = Regressor(len(INPUT_COLUMNS), hidden, len(TARGET_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.shape[0]
        train_loss /= len(train_dataset)

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb)
                    batch_loss = criterion(preds, yb)
                    val_loss += batch_loss.item() * xb.shape[0]
            val_loss /= len(val_dataset)  # type: ignore[arg-type]
        else:
            val_loss = train_loss

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "epoch": epoch,
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
            }

    if best_state is None:
        raise RuntimeError("未能保存最佳模型状态")

    model.load_state_dict(best_state["model"])  # type: ignore[arg-type]

    def _rmse(x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> float:
        if x_tensor.shape[0] == 0:
            return float("nan")
        dataset = SheetRegressionDataset(x_tensor, y_tensor, feature_norm, target_norm)
        loader = DataLoader(dataset, batch_size=batch_size)
        total_loss = 0.0
        total_items = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                preds = model(xb)
                preds_real = target_norm.denormalize(preds)
                y_real = target_norm.denormalize(yb)
                mse = torch.mean((preds_real - y_real) ** 2, dim=0)
                total_loss += mse.sum().item() * xb.shape[0]
                total_items += xb.shape[0]
        if total_items == 0:
            return float("nan")
        avg_mse = total_loss / (total_items * len(TARGET_COLUMNS))
        return math.sqrt(avg_mse)

    train_rmse = _rmse(train_x, train_y)
    val_rmse = _rmse(val_x, val_y)
    test_rmse = _rmse(test_x, test_y)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{book_path.stem}_model.pt"
    torch.save({"state_dict": model.state_dict(), "feature_norm": feature_norm, "target_norm": target_norm}, model_path)

    report = {
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "test_samples": int(test_x.shape[0]),
        "best_epoch": int(best_state["epoch"]),
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "test_rmse": float(test_rmse),
    }

    report_path = output_dir / f"{book_path.stem}_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为每个台风类别Excel训练独立的回归网络")
    parser.add_argument("--data-root", type=Path, default=Path("new_dataset/classified_typhoons"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/init_models"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        default=[128, 64],
        help="隐藏层宽度序列，例如 --hidden-sizes 128 64",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workbooks = list_workbooks(args.data_root)
    if not workbooks:
        raise SystemExit(f"未在 {args.data_root} 找到Excel文件")

    summary: Dict[str, Dict[str, float]] = {}
    for book in workbooks:
        print(f"开始训练: {book.name}")
        report = train_one_workbook(
            book,
            args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            hidden=args.hidden_sizes,
            lr=args.learning_rate,
            seed=args.seed,
        )
        summary[book.name] = report
        print(f"完成 {book.name}: 最佳epoch {report['best_epoch']}, 验证RMSE {report['val_rmse']:.4f}")

    summary_path = args.output_dir / "summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"训练完成，摘要写入 {summary_path}")


if __name__ == "__main__":
    main()
