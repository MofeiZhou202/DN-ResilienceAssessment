"""评估 Net-A 模型在指定数据集上的表现。

用法示例：

    python evaluate_net_a.py \
        --checkpoint checkpoints/net_a_best_last200.pt \
        --hurricane-excel dataset/batch_hurricane.xlsx \
        --impact-excel dataset/batch_hurricane_impact_simple.xlsx \
        --test-last 200 --seed 42

支持与 train_net_a.py 相同的数据加载/划分策略，评估指标包括：
- BCEWithLogitsLoss（与训练一致）
- Brier Score（概率预测的均方误差）
- MAE（平均绝对误差）
并输出部分样本的预测/真实概率对比，便于快速核查模型行为。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_net_a import (
    TyphoonImpactDataset,
    TyphoonImpactNet,
    collate_samples,
    load_excel_dataset,
    compute_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Net-A 模型")
    parser.add_argument("--checkpoint", required=True, help="模型权重文件路径")
    parser.add_argument("--npz-path", help="prepare_dataset 生成的 npz 文件")
    parser.add_argument("--hurricane-excel", help="批量台风生命周期 Excel 文件")
    parser.add_argument("--impact-excel", help="对应的影响概率 Excel 文件")
    parser.add_argument("--sheet-limit", type=int, help="只使用前 N 个工作表")
    parser.add_argument("--test-last", type=int, default=0, help="固定使用最后 N 个工作表作为测试集")
    parser.add_argument("--seed", type=int, help="随机种子，保持与训练一致")
    parser.add_argument("--train-split", type=float, default=0.7, help="训练集占比（对非 test-last 部分生效）")
    parser.add_argument("--val-split", type=float, default=0.15, help="验证集占比")
    parser.add_argument("--batch-size", type=int, default=32, help="评估时的 batch 大小")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-report", help="将评估指标保存到 JSON 文件")
    parser.add_argument("--preview-count", type=int, default=3, help="打印多少条样本的预测预览")
    return parser.parse_args()


def load_dataset_and_split(args: argparse.Namespace) -> Tuple[
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    use_excel = args.hurricane_excel and args.impact_excel
    if use_excel:
        lifecycle_tensor, lifecycle_meta, impact_data, wind_data = load_excel_dataset(
            Path(args.hurricane_excel), Path(args.impact_excel), args.sheet_limit
        )
    else:
        if not args.npz_path:
            raise ValueError("未指定数据来源，请提供 --hurricane-excel 与 --impact-excel 或 --npz-path")
        npz_path = Path(args.npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"找不到 npz 文件: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        lifecycle_tensor = data["lifecycle_tensor"]
        lifecycle_meta = data["lifecycle_meta"].item()
        impact_data = data["impact_data"].item()
        wind_data = data["wind_data"].item()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    num_samples = lifecycle_tensor.shape[0]
    ordered_indices = np.arange(num_samples)

    if args.test_last:
        if args.test_last >= num_samples:
            raise ValueError("test_last 必须小于样本总数")
        test_idx = ordered_indices[-args.test_last:]
        base_indices = ordered_indices[:-args.test_last]
    else:
        base_indices = ordered_indices
        test_idx = np.array([], dtype=int)

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

    return (
        lifecycle_tensor,
        lifecycle_meta,
        impact_data,
        wind_data,
        train_idx,
        val_idx,
        final_test_idx,
    )


def masked_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    pred_prob = torch.sigmoid(pred)
    mask_float = mask.float()
    count = mask_float.sum().clamp_min(1.0)
    diff = torch.abs(pred_prob - target)
    mse = torch.square(pred_prob - target)
    return {
        "mae": float((diff * mask_float).sum() / count),
        "brier": float((mse * mask_float).sum() / count),
    }


def evaluate_model(
    model: TyphoonImpactNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    line_criterion = nn.BCEWithLogitsLoss(reduction="none")
    model.eval()
    total_line = 0.0
    total_count = 0
    metrics_sum = {"mae": 0.0, "brier": 0.0}

    with torch.no_grad():
        for sequences, lengths, line_targets, wind_targets in loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            line_targets = line_targets.to(device)

            line_pred, wind_pred = model(sequences, lengths)
            mask = compute_masks(lengths, line_pred.size(1)).unsqueeze(-1)

            line_loss = line_criterion(line_pred, line_targets)
            mask_float = mask.float()
            loss_sum = (line_loss * mask_float).sum()
            denom = mask_float.sum().clamp_min(1.0)
            total_line += float(loss_sum / denom) * sequences.size(0)
            total_count += sequences.size(0)

            batch_metrics = masked_metrics(line_pred, line_targets, mask)
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key] * sequences.size(0)

    if total_count == 0:
        raise ValueError("测试集为空，无法评估")

    return {
        "line_bce": total_line / total_count,
        "mae": metrics_sum["mae"] / total_count,
        "brier": metrics_sum["brier"] / total_count,
    }


def preview_samples(
    model: TyphoonImpactNet,
    dataset: TyphoonImpactDataset,
    device: torch.device,
    count: int,
) -> Dict[str, Dict[str, list]]:
    results: Dict[str, Dict[str, list]] = {}
    if count <= 0:
        return results

    with torch.no_grad():
        preview_total = min(count, len(dataset))
        for local_idx in range(preview_total):
            sample = dataset[local_idx]
            seq = sample.sequence.unsqueeze(0).to(device)
            length = torch.tensor([sample.length], device=device)
            line_target = sample.line_target.unsqueeze(0).to(device)

            line_pred, _ = model(seq, length)
            mask = compute_masks(length, line_pred.size(1)).unsqueeze(-1)
            prob = torch.sigmoid(line_pred).cpu().numpy()[0]
            target = line_target.cpu().numpy()[0]
            valid_len = int(length.item())

            sheet_name = str(dataset.ids[dataset.indices[local_idx]])
            results[sheet_name] = {
                "hours": list(range(1, valid_len + 1)),
                "pred_first_line": prob[:valid_len, 0].tolist() if prob.shape[1] else [],
                "target_first_line": target[:valid_len, 0].tolist() if target.shape[1] else [],
                "pred_stats": {
                    "min": float(prob[:valid_len].min()) if prob.size else 0.0,
                    "max": float(prob[:valid_len].max()) if prob.size else 0.0,
                    "mean": float(prob[:valid_len].mean()) if prob.size else 0.0,
                },
            }
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    (
        lifecycle_tensor,
        lifecycle_meta,
        impact_data,
        wind_data,
        train_idx,
        val_idx,
        test_idx,
    ) = load_dataset_and_split(args)

    if test_idx.size == 0:
        raise ValueError("测试集为空，请检查 --test-last / --train-split / --val-split 设置")

    feature_dim = lifecycle_tensor.shape[2]
    line_dim = next(iter(impact_data.values()))["matrix"].shape[0]
    wind_dim = next(iter(wind_data.values()))["matrix"].shape[0] if wind_data else 0

    dataset_kwargs = {
        "lifecycle": lifecycle_tensor,
        "meta": lifecycle_meta,
        "impact": impact_data,
        "wind": wind_data,
    }

    test_dataset = TyphoonImpactDataset(indices=test_idx, **dataset_kwargs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    hidden_dim = saved_args.get("hidden_dim", 128)
    num_layers = saved_args.get("num_layers", 2)

    model = TyphoonImpactNet(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        line_dim=line_dim,
        wind_dim=wind_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    metrics = evaluate_model(model, test_loader, device)

    preview = preview_samples(model, test_dataset, device, args.preview_count)

    print("=== Net-A 评估结果 ===")
    print(f"测试样本数: {len(test_dataset)}")
    print(f"平均 BCE Loss: {metrics['line_bce']:.6f}")
    print(f"Brier Score : {metrics['brier']:.6f}")
    print(f"MAE         : {metrics['mae']:.6f}")
    if preview:
        print("\n--- 部分样本预测预览（第一条线路） ---")
        for sheet, info in preview.items():
            pred_slice = info["pred_first_line"][:5]
            tgt_slice = info["target_first_line"][:5]
            print(f"工作表 {sheet}: pred={pred_slice}, target={tgt_slice}, stats={info['pred_stats']}")

    if args.save_report:
        report_path = Path(args.save_report)
        report = {
            "metrics": metrics,
            "preview": preview,
            "checkpoint": str(checkpoint_path),
            "test_size": len(test_dataset),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] 评估报告已保存至 {report_path}")


if __name__ == "__main__":
    main()
