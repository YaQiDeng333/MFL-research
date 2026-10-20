#!/usr/bin/env python
"""Formal rerun of the Stage 20.77 true-3D RBC neural candidate.

This script intentionally reuses the 20.77 Conv1D architecture, loss, and
validation-selection protocol. It loads data only through the explicit
COMSOL_DATA_REGISTRY + manifest gate and writes benchmark-candidate metrics;
it does not run COMSOL, modify NPZ files, or create a baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import train_true_3d_rbc_neural_parameter_gate as gate
from load_true_3d_rbc_pilot_dataset import (
    V3_240_DATASET_ID,
    ROOT,
    aggregate_prediction_rows,
    check_no_overwrite,
    clip_params_to_train_bounds,
    denormalize_y,
    depth_grid_from_params,
    depth_map_from_params,
    evaluate_param_predictions,
    gate_manifest,
    load_dataset,
    normalize_x,
    normalize_y,
    resolve_dataset,
    split_indices,
    train_normalization,
    write_csv,
)


PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_formal_benchmark_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_formal_benchmark_20_77_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_seed_summary.csv"
METRICS = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_metrics.csv"
EPOCH_LOG = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_epoch_log.csv"
GROUP_SUMMARY = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_group_summary.csv"
PROFILE_METRICS = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_profile_metrics.csv"

REFERENCE_REQUIRED = [
    ROOT / "results/metrics/true_3d_rbc_v3_240_feature_baseline_metrics.csv",
]

SEED_FIELDS = [
    "seed",
    "selected_seed",
    "best_epoch",
    "best_val_selection_metric",
    "min_train_epoch",
    "min_train_normalized_param_mae",
    "train_normalized_param_mae",
    "val_normalized_param_mae",
    "test_normalized_param_mae",
    "train_dimension_mae_norm",
    "val_dimension_mae_norm",
    "test_dimension_mae_norm",
    "train_curvature_mae_norm",
    "val_curvature_mae_norm",
    "test_curvature_mae_norm",
    "test_L_mae_mm",
    "test_W_mae_mm",
    "test_D_mae_mm",
    "test_wLD_abs_error",
    "test_wWD_abs_error",
    "test_wLW_abs_error",
    "test_wMAE_auxiliary",
    "test_projected_mask_iou",
    "test_projected_mask_dice",
    "test_profile_depth_rmse_m",
    "test_er_like_profile_error",
    "test_max_depth_error_m",
    "test_volume_proxy_rel_error",
    "can_fit_train",
    "beats_mean_baseline_test",
    "beats_feature_baseline_test",
]

PROFILE_FIELDS = gate.PROFILE_FIELDS + ["er_like_profile_error", "max_depth_error_m"]

GROUP_FIELDS = gate.GROUP_FIELDS + ["er_like_profile_error", "max_depth_error_m", "volume_proxy_rel_error"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return float(np.mean(vals)) if vals else math.nan


def add_profile_error_rows(dataset: Any, pred_params_raw: np.ndarray, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pred_params, _ = clip_params_to_train_bounds(pred_params_raw, dataset)
    enriched: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        pred_depth = depth_grid_from_params(pred_params[idx])
        true_depth = dataset.profile_depth_grid_m[idx]
        denom = float(np.sum(true_depth**2))
        er_like = 0.0 if denom <= 1.0e-20 else float(np.sqrt(np.sum((pred_depth - true_depth) ** 2) / denom))
        updated = dict(row)
        updated["er_like_profile_error"] = er_like
        updated["max_depth_error_m"] = abs(float(pred_params[idx, 2] - dataset.rbc_params[idx, 2]))
        # Keep volume semantics aligned with the historical profile metrics.
        true_volume = float(dataset.profile_depth_map_xy_m[idx].sum())
        pred_volume = float(depth_map_from_params(pred_params[idx], dataset.profile_pose[idx]).sum())
        updated["volume_proxy_rel_error"] = 0.0 if abs(true_volume) < 1.0e-12 else abs(pred_volume - true_volume) / abs(true_volume)
        enriched.append(updated)
    return enriched


def split_aggregate(rows: list[dict[str, Any]], split_name: str) -> dict[str, float]:
    agg = aggregate_prediction_rows(rows, "formal_20_77_conv1d_rbc_param_gate", split_name)
    subset = [row for row in rows if row["split"] == split_name]
    agg["er_like_profile_error_mean"] = mean(subset, "er_like_profile_error")
    agg["max_depth_error_m_mean"] = mean(subset, "max_depth_error_m")
    return agg


def group_rows(seed: int, selected: bool, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        for group_field in ("curvature_template", "depth_bin", "aspect_bin", "size_bin"):
            for group_value in sorted({str(row[group_field]) for row in split_rows}):
                subset = [row for row in split_rows if str(row[group_field]) == group_value]
                if not subset:
                    continue
                base = {
                    "seed": seed,
                    "selected_seed": selected,
                    "split": split_name,
                    "group_field": group_field,
                    "group_value": group_value,
                    "sample_count": len(subset),
                    "normalized_param_mae_mean": mean(subset, "normalized_param_mae_mean"),
                    "L_mae_mm": mean(subset, "L_mae_mm"),
                    "W_mae_mm": mean(subset, "W_mae_mm"),
                    "D_mae_mm": mean(subset, "D_mae_mm"),
                    "curvature_mae_mean": mean(subset, "curvature_mae_mean"),
                    "projected_mask_iou": mean(subset, "projected_mask_iou"),
                    "projected_mask_dice": mean(subset, "projected_mask_dice"),
                    "profile_depth_rmse_m": mean(subset, "profile_depth_rmse_m"),
                    "er_like_profile_error": mean(subset, "er_like_profile_error"),
                    "max_depth_error_m": mean(subset, "max_depth_error_m"),
                    "volume_proxy_rel_error": mean(subset, "volume_proxy_rel_error"),
                }
                out.append(base)
    return out


def write_preflight(args: argparse.Namespace) -> None:
    entry, manifest, npz_path = resolve_dataset(args.dataset_id)
    checks = gate_manifest(entry, manifest, npz_path, args.dataset_id)
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset)
    missing = [str(path.relative_to(ROOT)) for path in REFERENCE_REQUIRED if not path.exists()]
    expected_split = {"train": 162, "val": 39, "test": 39}
    failed = [row for row in checks if not row["pass"]]
    schema_pass = (
        dataset.delta_b.shape == (240, 3, 3, 201)
        and dataset.x_channels.shape == (240, 9, 201)
        and {name: len(idx) for name, idx in splits.items()} == expected_split
        and bool(np.isfinite(dataset.delta_b).all())
        and bool(np.isfinite(dataset.rbc_params).all())
    )
    lines = [
        "20.85 formal true 3D RBC benchmark preflight summary",
        "",
        f"dataset_id: {args.dataset_id}",
        f"registry_manifest_gate_pass: {not failed}",
        f"schema_pass: {schema_pass}",
        f"npz_path_resolved_from_manifest: {npz_path}",
        f"input_shape: delta_b={list(dataset.delta_b.shape)}, conv1d={list(dataset.x_channels.shape)}",
        f"split_counts: {{'train': {len(splits['train'])}, 'val': {len(splits['val'])}, 'test': {len(splits['test'])}}}",
        f"reference_material_missing: {', '.join(missing) if missing else 'none'}",
        "latest_newest_npz_scan: false",
        "COMSOL_run: false",
        "new_data_generation: false",
        "NPZ_modification: false",
        "model_protocol: 20.77 Conv1D architecture, weighted SmoothL1 loss, validation-only checkpoint/seed selection",
        f"actual_seed_plan: {list(args.seeds)}",
        "forbidden_submit: data/, NPZ, checkpoint, preview PNG, notes, __pycache__, baseline docs, CURRENT_BASELINE.md, scripts/visualize_current_baseline.py",
        "stage_gate: pass" if not failed and schema_pass and not missing else "stage_gate: blocker",
    ]
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed or not schema_pass or missing:
        raise RuntimeError("formal benchmark preflight blocker; see preflight summary")


def run(args: argparse.Namespace) -> int:
    outputs = [PREFLIGHT_SUMMARY, SUMMARY, SEED_SUMMARY, METRICS, EPOCH_LOG, GROUP_SUMMARY, PROFILE_METRICS]
    check_no_overwrite(outputs, args.overwrite)
    write_preflight(args)
    dataset = load_dataset(args.dataset_id)
    stats = train_normalization(dataset)
    splits = split_indices(dataset)
    x_norm = normalize_x(dataset, stats)
    y_norm = normalize_y(dataset, stats)
    feature_scores = gate.read_feature_baseline_scores(ROOT / "results/metrics/true_3d_rbc_v3_240_feature_baseline_metrics.csv")

    seed_outputs = [gate.train_one_seed(seed, x_norm, y_norm, splits, args) for seed in args.seeds]
    selected = min(seed_outputs, key=lambda item: item["best_val_score"])
    selected_seed = int(selected["seed"])

    all_seed_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_epoch_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    all_profile_rows: list[dict[str, Any]] = []
    selected_profile_rows: list[dict[str, Any]] = []

    for seed_out in seed_outputs:
        seed = int(seed_out["seed"])
        is_selected = seed == selected_seed
        pred_raw = denormalize_y(seed_out["pred_norm"], stats)
        profile_rows = add_profile_error_rows(dataset, pred_raw, evaluate_param_predictions(dataset, pred_raw, stats))
        for row in profile_rows:
            row["seed"] = seed
            row["selected_seed"] = is_selected
        metric_rows = gate.compute_param_metric_rows(seed, is_selected, dataset.rbc_params, pred_raw, y_norm, seed_out["pred_norm"], splits)
        all_metric_rows.extend(metric_rows)
        all_profile_rows.extend(profile_rows)
        all_group_rows.extend(group_rows(seed, is_selected, profile_rows))
        all_epoch_rows.extend(seed_out["epoch_rows"])
        train_agg = split_aggregate(profile_rows, "train")
        val_agg = split_aggregate(profile_rows, "val")
        test_agg = split_aggregate(profile_rows, "test")
        if is_selected:
            selected_profile_rows = profile_rows
        all_seed_rows.append(
            {
                "seed": seed,
                "selected_seed": is_selected,
                "best_epoch": seed_out["best_epoch"],
                "best_val_selection_metric": seed_out["best_val_score"],
                "min_train_epoch": seed_out["min_train_epoch"],
                "min_train_normalized_param_mae": seed_out["min_train_normalized_param_mae"],
                "train_normalized_param_mae": train_agg["normalized_param_mae_mean_mean"],
                "val_normalized_param_mae": val_agg["normalized_param_mae_mean_mean"],
                "test_normalized_param_mae": test_agg["normalized_param_mae_mean_mean"],
                "train_dimension_mae_norm": train_agg["dimension_param_mae_norm_mean"],
                "val_dimension_mae_norm": val_agg["dimension_param_mae_norm_mean"],
                "test_dimension_mae_norm": test_agg["dimension_param_mae_norm_mean"],
                "train_curvature_mae_norm": train_agg["curvature_param_mae_norm_mean"],
                "val_curvature_mae_norm": val_agg["curvature_param_mae_norm_mean"],
                "test_curvature_mae_norm": test_agg["curvature_param_mae_norm_mean"],
                "test_L_mae_mm": test_agg["L_mae_mm_mean"],
                "test_W_mae_mm": test_agg["W_mae_mm_mean"],
                "test_D_mae_mm": test_agg["D_mae_mm_mean"],
                "test_wLD_abs_error": test_agg["wLD_abs_error_mean"],
                "test_wWD_abs_error": test_agg["wWD_abs_error_mean"],
                "test_wLW_abs_error": test_agg["wLW_abs_error_mean"],
                "test_wMAE_auxiliary": test_agg["curvature_mae_mean_mean"],
                "test_projected_mask_iou": test_agg["projected_mask_iou_mean"],
                "test_projected_mask_dice": test_agg["projected_mask_dice_mean"],
                "test_profile_depth_rmse_m": test_agg["profile_depth_rmse_m_mean"],
                "test_er_like_profile_error": test_agg["er_like_profile_error_mean"],
                "test_max_depth_error_m": test_agg["max_depth_error_m_mean"],
                "test_volume_proxy_rel_error": test_agg["volume_proxy_rel_error_mean"],
                "can_fit_train": seed_out["min_train_normalized_param_mae"] < 0.20,
                "beats_mean_baseline_test": bool(test_agg["normalized_param_mae_mean_mean"] < feature_scores["mean_test"]) if not math.isnan(feature_scores["mean_test"]) else "",
                "beats_feature_baseline_test": bool(test_agg["normalized_param_mae_mean_mean"] < feature_scores["feature_test"]) if not math.isnan(feature_scores["feature_test"]) else "",
            }
        )

    write_csv(SEED_SUMMARY, all_seed_rows, SEED_FIELDS)
    write_csv(METRICS, all_metric_rows, gate.METRIC_FIELDS)
    write_csv(EPOCH_LOG, all_epoch_rows, gate.EPOCH_FIELDS)
    write_csv(GROUP_SUMMARY, all_group_rows, GROUP_FIELDS)
    write_csv(PROFILE_METRICS, all_profile_rows, PROFILE_FIELDS)

    selected_test = split_aggregate(selected_profile_rows, "test")
    selected_train = split_aggregate(selected_profile_rows, "train")
    selected_val = split_aggregate(selected_profile_rows, "val")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.85 formal true 3D RBC benchmark rerun summary",
                "",
                f"dataset_id: {args.dataset_id}",
                "baseline_update: false",
                "COMSOL_run: false",
                "data_or_NPZ_modified: false",
                "model_family: 20.77 small Conv1D encoder + MLP six-parameter head",
                "loss: 20.77 weighted SmoothL1, dimension weights=1.0, curvature weights=0.5",
                "selection: validation-only best checkpoint per seed, validation-only seed selection",
                f"seeds_run: {list(args.seeds)}",
                f"selected_seed: {selected_seed}",
                f"train_normalized_mae: {selected_train['normalized_param_mae_mean_mean']:.6f}",
                f"val_normalized_mae: {selected_val['normalized_param_mae_mean_mean']:.6f}",
                f"test_normalized_mae: {selected_test['normalized_param_mae_mean_mean']:.6f}",
                f"test_LWD_mae_mm: L={selected_test['L_mae_mm_mean']:.6f}, W={selected_test['W_mae_mm_mean']:.6f}, D={selected_test['D_mae_mm_mean']:.6f}",
                f"test_wMAE_auxiliary: {selected_test['curvature_mae_mean_mean']:.6f}",
                f"test_wLD_wWD_wLW_auxiliary: {selected_test['wLD_abs_error_mean']:.6f}, {selected_test['wWD_abs_error_mean']:.6f}, {selected_test['wLW_abs_error_mean']:.6f}",
                f"test_profile_depth_rmse_m: {selected_test['profile_depth_rmse_m_mean']:.9f}",
                f"test_er_like_profile_error: {selected_test['er_like_profile_error_mean']:.6f}",
                f"test_projected_mask_iou_dice: {selected_test['projected_mask_iou_mean']:.6f}, {selected_test['projected_mask_dice_mean']:.6f}",
                f"test_max_depth_error_m: {selected_test['max_depth_error_m_mean']:.9f}",
                f"test_volume_proxy_rel_error: {selected_test['volume_proxy_rel_error_mean']:.6f}",
                "Boundary: this is a formal benchmark candidate rerun, not a baseline replacement.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
