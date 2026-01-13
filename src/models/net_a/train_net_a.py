"""Net-A 训练脚本：台风生命周期 -> 输电线路故障概率 / 风电场出力.

使用方式（在虚拟环境下，确保安装 torch>=2.0）:

    python train_net_a.py \
        --npz-path data/test_dataset.npz \
        --train-split 0.7 --val-split 0.15 \
        --batch-size 32 --epochs 80

或直接使用批量生成的Excel数据：

    python train_net_a.py \
        --hurricane-excel dataset/batch_hurricane.xlsx \
        --impact-excel dataset/batch_hurricane_impact_simple.xlsx \
        --epochs 80 --batch-size 32

脚本会自动：
1. 读取 prepare_dataset.py 生成的 npz 数据包。
2. 构建 PyTorch Dataset/DataLoader, 支持变长序列。
3. 训练一个 LSTM 编码器，输出对应的线路/风电场预测（支持加权 BCE/Focal Loss、梯度裁剪、学习率调度和早停）。
4. 在验证集上评估并保存最好模型权重。

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
    python train_net_a.py --hurricane-excel dataset/batch_hurricane.xlsx --impact-excel dataset/batch_hurricane_impact_simple.xlsx --test-last 200 --epochs 120 --batch-size 32 --neg-weight 3.0 --focal-gamma 0.0python train_net_a.py --hurricane-excel dataset/batch_hurricane.xlsx --impact-excel dataset/batch_hurricane_impact_simple.xlsx --test-last 200 --epochs 120 --batch-size 32 --neg-weight 3.0 --focal-gamma 0.0

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import GENERATED_DATA_DIR


# ---------------------------------------------------------------------------
# 数据集定义
# ---------------------------------------------------------------------------


def _ensure_tensor(data: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float32)


@dataclass
class Sample:
    sequence: torch.Tensor  # (T, F)
    length: int
    line_target: torch.Tensor  # (T, L)
    wind_target: torch.Tensor  # (T, W) 可能是零维矩阵


class TyphoonImpactDataset(Dataset):
    """把 npz 数据集封装成 PyTorch Dataset."""

    def __init__(
        self,
        lifecycle: np.ndarray,
        meta: Dict[str, np.ndarray],
        impact: Dict[str, Dict[str, np.ndarray]],
        wind: Dict[str, Dict[str, np.ndarray]] | None,
        indices: Sequence[int],
    ) -> None:
        self.lifecycle = lifecycle
        self.meta = meta
        self.impact = impact
        self.wind = wind or {}
        self.indices = list(indices)
        self.ids = meta["ids"]
        self.lengths = np.asarray(meta["lengths"], dtype=np.int64)
        self.impact_keys = {str(k) for k in self.impact.keys()}
        self.wind_keys = {str(k) for k in self.wind.keys()}

    def _resolve_key(self, name: str, key_set: set[str]) -> str:
        name = str(name)
        if name in key_set:
            return name

        lower = name.lower()
        if lower.startswith("sheet"):
            suffix = name[5:]
            if suffix in key_set:
                return suffix
            suffix_stripped = suffix.lstrip("_ ")
            if suffix_stripped in key_set:
                return suffix_stripped
            digits = ''.join(ch for ch in suffix if ch.isdigit())
            if digits and digits in key_set:
                return digits

        digits_only = ''.join(ch for ch in name if ch.isdigit())
        if digits_only and digits_only in key_set:
            return digits_only

        raise KeyError(f"无法匹配到对应的 sheet 名称: {name}; 可用键 {sorted(key_set)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Sample:
        sample_idx = self.indices[item]
        seq_full = self.lifecycle[sample_idx]
        raw_length = int(self.lengths[sample_idx])

        raw_sheet = str(self.ids[sample_idx]).split(":")[-1]
        impact_key = self._resolve_key(raw_sheet, self.impact_keys)
        impact_payload = self.impact[impact_key]
        wind_payload = self.wind.get(self._resolve_key(raw_sheet, self.wind_keys)) if self.wind_keys else None

        line_matrix = np.asarray(impact_payload["matrix"], dtype=np.float32)
        if wind_payload is not None:
            wind_matrix = np.asarray(wind_payload["matrix"], dtype=np.float32)
        else:
            wind_matrix = np.zeros((0, line_matrix.shape[1]), dtype=np.float32)

        effective_len = min(
            max(raw_length, 1),
            seq_full.shape[0],
            line_matrix.shape[1],
            wind_matrix.shape[1] if wind_matrix.size else seq_full.shape[0],
        )

        seq = np.nan_to_num(seq_full[:effective_len], nan=0.0)
        line_target = np.nan_to_num(line_matrix[:, :effective_len], nan=0.0).T
        wind_target = np.nan_to_num(wind_matrix[:, :effective_len], nan=0.0).T if wind_matrix.size else np.zeros((effective_len, 0), dtype=np.float32)

        return Sample(
            sequence=_ensure_tensor(seq),
            length=effective_len,
            line_target=_ensure_tensor(line_target),
            wind_target=_ensure_tensor(wind_target),
        )


def sanitize_impact_data(
    impact: Dict[str, Dict[str, np.ndarray]],
    zero_threshold: float | None,
    min_positive: float | None,
    max_positive: float | None,
    binarize: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    """对影响矩阵做清洗，缓解极端噪声与类别失衡。"""

    if not impact:
        return impact

    sanitized: Dict[str, Dict[str, np.ndarray]] = {}
    for key, payload in impact.items():
        matrix = np.asarray(payload["matrix"], dtype=np.float32)
        matrix = matrix.copy()

        if zero_threshold is not None and zero_threshold > 0.0:
            matrix[np.abs(matrix) < zero_threshold] = 0.0

        if binarize:
            mask = matrix > 0.0
            matrix[mask] = 1.0
        else:
            mask = matrix > 0.0
            if min_positive is not None and min_positive > 0.0:
                matrix[mask] = np.maximum(matrix[mask], min_positive)
            if max_positive is not None and max_positive > 0.0:
                matrix[mask] = np.minimum(matrix[mask], max_positive)

        sanitized[key] = {**payload, "matrix": matrix}

    return sanitized


def summarize_impact_stats(impact: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
    """统计正样本比例等信息，便于调参观察。"""

    if not impact:
        return {"positive_ratio": 0.0, "mean": 0.0}

    total = 0
    positives = 0
    running_sum = 0.0
    for payload in impact.values():
        matrix = np.asarray(payload["matrix"], dtype=np.float32)
        total += matrix.size
        positives += np.count_nonzero(matrix > 0.0)
        running_sum += float(matrix.sum())

    ratio = positives / max(total, 1)
    mean = running_sum / max(total, 1)
    return {
        "positive_ratio": float(ratio),
        "mean": float(mean),
    }


def initialize_output_bias(linear: nn.Linear, positive_ratio: float | None) -> None:
    """用全局正样本率初始化输出层偏置，便于模型学习极小概率。"""

    if positive_ratio is None:
        return

    eps = 1e-6
    prob = float(positive_ratio)
    if not (eps < prob < 1 - eps):
        prob = min(max(prob, eps), 1 - eps)

    prior_logit = math.log(prob / (1.0 - prob))
    with torch.no_grad():
        linear.bias.fill_(prior_logit)


def collate_samples(batch: Sequence[Sample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([s.length for s in batch], dtype=torch.long)
    max_len = int(lengths.max())

    def pad_stack(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        feature_dim = tensors[0].shape[-1]
        stacked = torch.zeros((len(tensors), max_len, feature_dim), dtype=torch.float32)
        for idx, tensor in enumerate(tensors):
            stacked[idx, : tensor.shape[0]] = tensor
        return stacked

    sequences = pad_stack([s.sequence for s in batch])
    line_targets = pad_stack([s.line_target for s in batch])
    wind_targets = pad_stack([s.wind_target for s in batch])
    return sequences, lengths, line_targets, wind_targets


def load_excel_dataset(
    hurricane_excel: Path,
    impact_excel: Path,
    sheet_limit: int | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    hurricane_book = pd.read_excel(hurricane_excel, sheet_name=None, engine="openpyxl")
    impact_book = pd.read_excel(impact_excel, sheet_name=None, engine="openpyxl", index_col=0)

    common_sheets = sorted(set(hurricane_book.keys()) & set(impact_book.keys()), key=lambda x: int(str(x)) if str(x).isdigit() else str(x))
    if not common_sheets:
        raise ValueError("台风与影响Excel没有共同的工作表名称")
    if sheet_limit is not None:
        common_sheets = common_sheets[:sheet_limit]

    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    sheet_ids: List[str] = []
    impact_data: Dict[str, Dict[str, np.ndarray]] = {}
    wind_data: Dict[str, Dict[str, np.ndarray]] = {}

    example_features: List[str] | None = None

    for sheet_name in common_sheets:
        typhoon_df = hurricane_book[sheet_name]
        impact_df = impact_book[sheet_name]

        typhoon_array = typhoon_df.to_numpy(dtype=np.float32)
        impact_array = impact_df.to_numpy(dtype=np.float32)

        if example_features is None:
            example_features = list(map(str, typhoon_df.columns))

        seq_len, feature_dim = typhoon_array.shape
        if impact_array.shape[1] < seq_len:
            # impact小于生命周期长度时截断
            seq_len = impact_array.shape[1]
            typhoon_array = typhoon_array[:seq_len]
        elif impact_array.shape[1] > seq_len:
            # impact列更多时补零方便训练
            pad_width = impact_array.shape[1] - seq_len
            typhoon_array = np.pad(typhoon_array, ((0, pad_width), (0, 0)), mode="constant")
            seq_len = impact_array.shape[1]

        sequences.append(typhoon_array)
        lengths.append(seq_len)
        sheet_id = str(sheet_name)
        sheet_ids.append(sheet_id)

        impact_data[sheet_id] = {
            "matrix": impact_array,
            "line_ids": list(map(str, impact_df.index)),
            "hours": list(map(str, impact_df.columns)),
        }
        wind_data[sheet_id] = {
            "matrix": np.zeros((0, impact_array.shape[1]), dtype=np.float32)
        }

    max_len = max(lengths)
    feature_dim = sequences[0].shape[1]
    lifecycle_tensor = np.zeros((len(sequences), max_len, feature_dim), dtype=np.float32)
    for idx, seq in enumerate(sequences):
        lifecycle_tensor[idx, : seq.shape[0], :] = seq

    lifecycle_meta = {
        "ids": np.array(sheet_ids),
        "lengths": np.array(lengths),
        "feature_names": example_features or [],
    }

    return lifecycle_tensor, lifecycle_meta, impact_data, wind_data


def compute_class_weights(dataset: TyphoonImpactDataset, line_dim: int, clamp: float) -> torch.Tensor:
    """估计正类权重，缓解极端不平衡。"""
    if len(dataset) == 0:
        raise ValueError("训练集为空，无法计算类别权重")

    pos_counts = torch.zeros(line_dim, dtype=torch.float64)
    total_counts = torch.zeros(line_dim, dtype=torch.float64)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        length = sample.length
        line_sum = sample.line_target[:length].sum(dim=0).double()
        pos_counts += line_sum
        total_counts += float(length)

    pos_counts = pos_counts.clamp_min(1.0)
    neg_counts = (total_counts - pos_counts).clamp_min(1.0)
    weights = (neg_counts / pos_counts).clamp_max(clamp).float()
    return weights


def compute_line_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
    focal_gamma: float,
    neg_weight: float,
) -> torch.Tensor:
    pos_weight_param = None
    if pos_weight is not None:
        pos_weight_param = pos_weight.view(1, 1, -1)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight_param,
    )

    if neg_weight != 1.0:
        neg_factor = 1.0 + (neg_weight - 1.0) * (1.0 - targets)
        loss = loss * neg_factor

    if focal_gamma > 0.0:
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = loss * (1 - p_t).pow(focal_gamma)

    mask = mask.expand_as(loss).float()
    masked_loss_sum = (loss * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return masked_loss_sum / denom


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = max(1, patience)
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------


class TyphoonImpactNet(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, line_dim: int, wind_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.line_head = nn.Linear(hidden_dim * 2, line_dim)
        self.wind_head = nn.Linear(hidden_dim * 2, wind_dim) if wind_dim > 0 else None

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        line_pred = self.line_head(encoded)
        if self.wind_head is not None:
            wind_pred = self.wind_head(encoded)
        else:
            wind_pred = encoded.new_zeros(encoded.size(0), encoded.size(1), 0)
        return line_pred, wind_pred


# ---------------------------------------------------------------------------
# 训练与评估逻辑
# ---------------------------------------------------------------------------


def compute_masks(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    idx = torch.arange(max_len, device=lengths.device)[None, :]
    mask = idx < lengths[:, None]
    return mask


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    loss = criterion(pred, target)
    mask = mask.expand_as(loss).float()
    masked_loss_sum = (loss * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return masked_loss_sum / denom


def train_one_epoch(
    model: TyphoonImpactNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    focal_gamma: float,
    neg_weight: float,
    grad_clip: float,
    wind_criterion: nn.Module,
) -> Dict[str, float]:
    model.train()
    total_line = 0.0
    total_wind = 0.0
    total_count = 0

    for sequences, lengths, line_targets, wind_targets in loader:
        optimizer.zero_grad()
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        line_targets = line_targets.to(device)
        wind_targets = wind_targets.to(device)

        line_pred, wind_pred = model(sequences, lengths)
        mask = compute_masks(lengths, line_pred.size(1)).unsqueeze(-1)

        line_loss = compute_line_loss(line_pred, line_targets, mask, pos_weight, focal_gamma, neg_weight)
        if wind_pred.size(-1) == 0:
            wind_loss = torch.tensor(0.0, device=device)
            loss = line_loss
        else:
            wind_loss = masked_loss(wind_pred, wind_targets, mask, wind_criterion)
            loss = line_loss + wind_loss
        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_line += line_loss.item() * sequences.size(0)
        total_wind += wind_loss.item() * sequences.size(0)
        total_count += sequences.size(0)

    return {
        "line_loss": total_line / max(total_count, 1),
        "wind_loss": total_wind / max(total_count, 1),
    }


def evaluate(
    model: TyphoonImpactNet,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    focal_gamma: float,
    neg_weight: float,
    wind_criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_line = 0.0
    total_wind = 0.0
    total_count = 0

    with torch.no_grad():
        for sequences, lengths, line_targets, wind_targets in loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            line_targets = line_targets.to(device)
            wind_targets = wind_targets.to(device)

            line_pred, wind_pred = model(sequences, lengths)
            mask = compute_masks(lengths, line_pred.size(1)).unsqueeze(-1)

            line_loss = compute_line_loss(line_pred, line_targets, mask, pos_weight, focal_gamma, neg_weight)
            if wind_pred.size(-1) == 0:
                wind_loss = torch.tensor(0.0, device=device)
            else:
                wind_loss = masked_loss(wind_pred, wind_targets, mask, wind_criterion)

            total_line += line_loss.item() * sequences.size(0)
            total_wind += wind_loss.item() * sequences.size(0)
            total_count += sequences.size(0)

    return {
        "line_loss": total_line / max(total_count, 1),
        "wind_loss": total_wind / max(total_count, 1),
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Net-A (typhoon lifecycle -> impacts)")
    parser.add_argument(
        "--npz-path",
        default=str(GENERATED_DATA_DIR / "test_dataset.npz"),
        help="prepare_dataset 输出的 npz 文件",
    )
    parser.add_argument("--hurricane-excel", help="批量台风生命周期 Excel 文件路径")
    parser.add_argument("--impact-excel", help="对应的输电线路影响简化结果 Excel 路径")
    parser.add_argument("--sheet-limit", type=int, help="仅使用前N个工作表进行训练")
    parser.add_argument("--test-last", type=int, default=0, help="固定使用最后N个工作表作为测试集")
    parser.add_argument("--seed", type=int, help="随机种子，便于复现")
    parser.add_argument("--train-split", type=float, default=0.7, help="训练集占比")
    parser.add_argument("--val-split", type=float, default=0.15, help="验证集占比")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="checkpoints/net_a_best.pt", help="保存最优模型的路径")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=10, help="早停容忍的epoch数量")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="早停判定的最小改进幅度")
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Focal Loss 的 γ 参数，0表示禁用")
    parser.add_argument("--neg-weight", type=float, default=1.0, help="负样本权重系数 (>1 可抑制假阳性)")
    parser.add_argument("--pos-weight-cap", type=float, default=50.0, help="正类权重上限，防止过大")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="梯度裁剪上限 (0 表示不裁剪)")
    parser.add_argument("--lr-patience", type=int, default=5, help="学习率调度的耐心轮数")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="学习率降低因子")
    parser.add_argument("--label-zero-threshold", type=float, default=1e-4, help="低于该阈值的标签视为0")
    parser.add_argument("--label-min-positive", type=float, default=1e-3, help="保留连续标签时的最小正值")
    parser.add_argument("--label-max-positive", type=float, default=1.0, help="保留连续标签时的最大正值")
    parser.add_argument("--label-binarize", dest="label_binarize", action="store_true", help="把正样本标签统一为1")
    parser.add_argument("--no-label-binarize", dest="label_binarize", action="store_false", help="保留连续标签值")
    parser.add_argument("--no-logit-prior", dest="use_logit_prior", action="store_false", help="不使用正样本先验初始化输出层偏置")
    parser.set_defaults(label_binarize=False)
    parser.set_defaults(use_logit_prior=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_excel = args.hurricane_excel and args.impact_excel
    if use_excel:
        lifecycle_tensor, lifecycle_meta, impact_data, wind_data = load_excel_dataset(
            Path(args.hurricane_excel), Path(args.impact_excel), args.sheet_limit
        )
    else:
        npz_path = Path(args.npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"找不到 npz 文件: {npz_path}，或者提供 --hurricane-excel 与 --impact-excel"
            )
        data = np.load(npz_path, allow_pickle=True)
        lifecycle_tensor = data["lifecycle_tensor"]
        lifecycle_meta = data["lifecycle_meta"].item()
        impact_data = data["impact_data"].item()
        wind_data = data["wind_data"].item()

    if (not args.label_binarize) and (
        args.label_max_positive is not None
    ) and (
        args.label_min_positive is not None
    ) and (args.label_min_positive > args.label_max_positive):
        raise ValueError("label_min_positive 不能大于 label_max_positive")

    impact_data = sanitize_impact_data(
        impact_data,
        zero_threshold=args.label_zero_threshold,
        min_positive=None if args.label_binarize else args.label_min_positive,
        max_positive=None if args.label_binarize else args.label_max_positive,
        binarize=args.label_binarize,
    )

    label_stats = summarize_impact_stats(impact_data)
    print(
        "[INFO] 标签统计 -> positive_ratio={:.6f}, mean={:.6f}".format(
            label_stats["positive_ratio"], label_stats["mean"]
        )
    )

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    num_samples, max_len, feature_dim = lifecycle_tensor.shape
    example_sheet = next(iter(impact_data.values()))
    line_dim = example_sheet["matrix"].shape[0]
    wind_dim = next(iter(wind_data.values()))["matrix"].shape[0] if wind_data else 0

    ordered_indices = np.arange(num_samples)
    if args.test_last:
        if args.test_last >= num_samples:
            raise ValueError("test_last 必须小于样本总数")
        test_idx = ordered_indices[-args.test_last:]
        base_indices = ordered_indices[:-args.test_last]
    else:
        base_indices = ordered_indices
        test_idx = None

    permuted = np.random.permutation(len(base_indices)) if len(base_indices) else np.array([], dtype=int)
    base_indices = base_indices[permuted]

    train_end = int(len(base_indices) * args.train_split)
    val_end = train_end + int(len(base_indices) * args.val_split)

    train_idx = base_indices[:train_end]
    val_idx = base_indices[train_end:val_end]

    if args.test_last:
        remaining = base_indices[val_end:]
        if remaining.size:
            train_idx = np.concatenate([train_idx, remaining])
        final_test_idx = test_idx
    else:
        final_test_idx = base_indices[val_end:]

    test_idx = final_test_idx

    dataset_kwargs = {
        "lifecycle": lifecycle_tensor,
        "meta": lifecycle_meta,
        "impact": impact_data,
        "wind": wind_data,
    }

    train_dataset = TyphoonImpactDataset(indices=train_idx, **dataset_kwargs)
    val_dataset = TyphoonImpactDataset(indices=val_idx, **dataset_kwargs)
    test_dataset = TyphoonImpactDataset(indices=test_idx, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_samples)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)

    device = torch.device(args.device)
    model = TyphoonImpactNet(feature_dim, args.hidden_dim, line_dim, wind_dim, num_layers=args.num_layers)
    if args.use_logit_prior:
        initialize_output_bias(model.line_head, label_stats.get("positive_ratio"))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    wind_criterion = nn.MSELoss(reduction="none")

    pos_weight_tensor: torch.Tensor | None = None
    if len(train_dataset) > 0:
        pos_weight_tensor = compute_class_weights(train_dataset, line_dim, args.pos_weight_cap)
        stats = pos_weight_tensor.cpu().numpy()
        print(
            f"[INFO] 正类权重统计 -> min: {stats.min():.2f}, max: {stats.max():.2f}, mean: {stats.mean():.2f}"
        )
        pos_weight_tensor = pos_weight_tensor.to(device)
    else:
        print("[WARN] 训练集为空，跳过正类权重估计")

    scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.lr_patience, min_lr=1e-6)
    early_stopper = EarlyStopping(args.patience, args.min_delta)

    best_val = float("inf")
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    has_val = len(val_dataset) > 0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pos_weight_tensor,
            args.focal_gamma,
            args.neg_weight,
            args.grad_clip,
            wind_criterion,
        )
        if has_val:
            val_stats = evaluate(
                model,
                val_loader,
                device,
                pos_weight_tensor,
                args.focal_gamma,
                args.neg_weight,
                wind_criterion,
            )
        else:
            val_stats = train_stats.copy()
        history.append({"epoch": epoch, **train_stats, **val_stats})

        print(
            f"Epoch {epoch:03d} | train_line={train_stats['line_loss']:.4f} train_wind={train_stats['wind_loss']:.4f} "
            f"| val_line={val_stats['line_loss']:.4f} val_wind={val_stats['wind_loss']:.4f}"
        )

        current_val = val_stats["line_loss"] + val_stats["wind_loss"]
        if current_val < best_val:
            best_val = current_val
            feature_names = None
            if isinstance(lifecycle_meta, dict):
                feature_names = lifecycle_meta.get("feature_names")

            checkpoint_payload = {
                "model": model.state_dict(),
                "args": vars(args),
                "pos_weight": pos_weight_tensor.detach().cpu().numpy().tolist() if pos_weight_tensor is not None else None,
            }
            if feature_names is not None:
                checkpoint_payload["feature_names"] = list(map(str, feature_names))

            torch.save(checkpoint_payload, checkpoint_path)
            print(f"  -> new best model saved to {checkpoint_path}")

        if has_val:
            scheduler.step(current_val)
            if early_stopper.step(current_val):
                print("[INFO] Early stopping triggered.")
                break

    if len(test_dataset) > 0:
        print("[INFO] evaluating best checkpoint on test set...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        test_stats = evaluate(
            model,
            test_loader,
            device,
            pos_weight_tensor,
            args.focal_gamma,
            args.neg_weight,
            wind_criterion,
        )
        print(
            f"Test | line_loss={test_stats['line_loss']:.4f} wind_loss={test_stats['wind_loss']:.4f}"
        )
    else:
        print("[WARN] 测试集为空，跳过测试评估。")

    history_path = checkpoint_path.with_suffix(".history.json")
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)
    print(f"[INFO] training metrics saved to {history_path}")


if __name__ == "__main__":
    main()
