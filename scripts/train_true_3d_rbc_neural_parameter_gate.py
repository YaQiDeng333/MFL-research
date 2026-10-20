# 公共依赖摘录：保留原算法定义，移除未被发布入口调用的历史实验流程。

from __future__ import annotations

import argparse

import copy

import csv

import math

import random

from pathlib import Path

from typing import Any

import numpy as np

import torch

from torch import nn

from torch.utils.data import DataLoader, TensorDataset

from load_true_3d_rbc_pilot_dataset import (
    DATASET_ID,
    PARAM_NAMES,
    ROOT,
    aggregate_prediction_rows,
    check_no_overwrite,
    denormalize_y,
    evaluate_param_predictions,
    load_dataset,
    normalize_x,
    normalize_y,
    run_name_for_dataset,
    split_indices,
    train_normalization,
    write_csv,
)

PARAM_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=torch.float32)

METRIC_FIELDS = [
    "model",
    "seed",
    "selected_seed",
    "split",
    "param",
    "sample_count",
    "normalized_mae",
    "physical_mae",
    "physical_mae_mm",
    "relative_mae",
    "r2",
]

EPOCH_FIELDS = [
    "seed",
    "epoch",
    "train_loss",
    "train_selection_metric",
    "val_selection_metric",
    "train_normalized_param_mae",
    "val_normalized_param_mae",
    "train_dimension_mae_norm",
    "val_dimension_mae_norm",
    "train_curvature_mae_norm",
    "val_curvature_mae_norm",
]

GROUP_FIELDS = [
    "seed",
    "selected_seed",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "normalized_param_mae_mean",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "curvature_mae_mean",
    "projected_mask_iou",
    "projected_mask_dice",
    "profile_depth_rmse_m",
]

PROFILE_FIELDS = [
    "seed",
    "selected_seed",
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "size_bin",
    "clip_applied",
    "clip_fraction",
    "normalized_param_mae_mean",
    "dimension_param_mae_norm",
    "curvature_param_mae_norm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "curvature_mae_mean",
    "projected_mask_iou",
    "projected_mask_dice",
    "projected_mask_area_error",
    "projected_mask_center_error_px",
    "profile_depth_rmse_m",
    "volume_proxy_rel_error",
]

class RBCConvRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 96),
            nn.GELU(),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Linear(32, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)

def selection_components(y_true_norm: np.ndarray, y_pred_norm: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_true_norm - y_pred_norm)
    return {
        "normalized_param_mae": float(abs_err.mean()),
        "dimension_mae_norm": float(abs_err[:, :3].mean()),
        "curvature_mae_norm": float(abs_err[:, 3:].mean()),
    }

def selection_metric(components: dict[str, float]) -> float:
    return components["normalized_param_mae"] + 0.25 * components["dimension_mae_norm"] + 0.10 * components["curvature_mae_norm"]

def predict_norm(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
    return pred.astype(np.float32)

def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = PARAM_WEIGHTS.to(pred.device)
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weights).mean()

def train_one_seed(seed: int, x_norm: np.ndarray, y_norm: np.ndarray, splits: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    set_seed(seed)
    model = RBCConvRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_idx = splits["train"]
    train_ds = TensorDataset(torch.as_tensor(x_norm[train_idx], dtype=torch.float32), torch.as_tensor(y_norm[train_idx], dtype=torch.float32))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_score = math.inf
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = weighted_smooth_l1(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        pred_train = predict_norm(model, x_norm[train_idx])
        pred_val = predict_norm(model, x_norm[splits["val"]])
        train_comp = selection_components(y_norm[train_idx], pred_train)
        val_comp = selection_components(y_norm[splits["val"]], pred_val)
        train_score = selection_metric(train_comp)
        val_score = selection_metric(val_comp)
        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_selection_metric": train_score,
                "val_selection_metric": val_score,
                "train_normalized_param_mae": train_comp["normalized_param_mae"],
                "val_normalized_param_mae": val_comp["normalized_param_mae"],
                "train_dimension_mae_norm": train_comp["dimension_mae_norm"],
                "val_dimension_mae_norm": val_comp["dimension_mae_norm"],
                "train_curvature_mae_norm": train_comp["curvature_mae_norm"],
                "val_curvature_mae_norm": val_comp["curvature_mae_norm"],
            }
        )
    if best_state is None:
        raise RuntimeError("no validation state selected")
    min_train_row = min(epoch_rows, key=lambda row: float(row["train_normalized_param_mae"]))
    model.load_state_dict(best_state)
    pred_norm_all = predict_norm(model, x_norm)
    return {
        "seed": seed,
        "model": model,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "min_train_epoch": int(min_train_row["epoch"]),
        "min_train_normalized_param_mae": float(min_train_row["train_normalized_param_mae"]),
        "val_normalized_at_min_train": float(min_train_row["val_normalized_param_mae"]),
        "pred_norm": pred_norm_all,
        "epoch_rows": epoch_rows,
    }

def compute_param_metric_rows(seed: int, selected_seed: bool, y_true_raw: np.ndarray, y_pred_raw: np.ndarray, y_true_norm: np.ndarray, y_pred_norm: np.ndarray, splits: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, idx in splits.items():
        true_raw = y_true_raw[idx]
        pred_raw = y_pred_raw[idx]
        true_norm = y_true_norm[idx]
        pred_norm = y_pred_norm[idx]
        for pidx, param_name in enumerate(PARAM_NAMES):
            abs_raw = np.abs(pred_raw[:, pidx] - true_raw[:, pidx])
            abs_norm = np.abs(pred_norm[:, pidx] - true_norm[:, pidx])
            denom = np.maximum(np.abs(true_raw[:, pidx]), 1.0e-12)
            ss_res = float(np.sum((true_raw[:, pidx] - pred_raw[:, pidx]) ** 2))
            ss_tot = float(np.sum((true_raw[:, pidx] - np.mean(true_raw[:, pidx])) ** 2))
            r2 = float("nan") if ss_tot <= 1.0e-20 else 1.0 - ss_res / ss_tot
            rows.append(
                {
                    "model": "conv1d_rbc_param_gate",
                    "seed": seed,
                    "selected_seed": selected_seed,
                    "split": split_name,
                    "param": param_name,
                    "sample_count": len(idx),
                    "normalized_mae": float(abs_norm.mean()),
                    "physical_mae": float(abs_raw.mean()),
                    "physical_mae_mm": float(abs_raw.mean() * 1000.0) if pidx < 3 else "",
                    "relative_mae": float(np.mean(abs_raw / denom)) if pidx < 3 else "",
                    "r2": r2,
                }
            )
        abs_norm_all = np.abs(pred_norm - true_norm)
        abs_raw_all = np.abs(pred_raw - true_raw)
        rows.append(
            {
                "model": "conv1d_rbc_param_gate",
                "seed": seed,
                "selected_seed": selected_seed,
                "split": split_name,
                "param": "ALL",
                "sample_count": len(idx),
                "normalized_mae": float(abs_norm_all.mean()),
                "physical_mae": float(abs_raw_all.mean()),
                "physical_mae_mm": "",
                "relative_mae": "",
                "r2": "",
            }
        )
        rows.append(
            {
                "model": "conv1d_rbc_param_gate",
                "seed": seed,
                "selected_seed": selected_seed,
                "split": split_name,
                "param": "DIMENSION_MEAN",
                "sample_count": len(idx),
                "normalized_mae": float(abs_norm_all[:, :3].mean()),
                "physical_mae": float(abs_raw_all[:, :3].mean()),
                "physical_mae_mm": float(abs_raw_all[:, :3].mean() * 1000.0),
                "relative_mae": "",
                "r2": "",
            }
        )
        rows.append(
            {
                "model": "conv1d_rbc_param_gate",
                "seed": seed,
                "selected_seed": selected_seed,
                "split": split_name,
                "param": "CURVATURE_MEAN",
                "sample_count": len(idx),
                "normalized_mae": float(abs_norm_all[:, 3:].mean()),
                "physical_mae": float(abs_raw_all[:, 3:].mean()),
                "physical_mae_mm": "",
                "relative_mae": "",
                "r2": "",
            }
        )
    return rows

def read_feature_baseline_scores(path: Path) -> dict[str, float]:
    scores = {"mean_test": math.nan, "feature_test": math.nan}
    if not path.exists():
        return scores
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "test":
                continue
            value = float(row.get("normalized_param_mae_mean_mean") or "nan")
            if row.get("model") == "mean_train_target":
                scores["mean_test"] = value
            if str(row.get("selected_by_val")).lower() == "true":
                scores["feature_test"] = value
    return scores
