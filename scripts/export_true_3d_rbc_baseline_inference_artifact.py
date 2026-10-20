#!/usr/bin/env python
"""Recover the fixed true-3D RBC baseline inference artifact.

This is an artifact-recovery utility for Stage 20.88a. It intentionally
replays the 20.77/20.85 seed=42 Conv1D six-parameter model protocol so later
robustness audits can re-run inference on perturbed delta_b observations.

It does not run COMSOL, generate new data, modify NPZ files, tune
hyperparameters, or update CURRENT_BASELINE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import train_true_3d_rbc_neural_parameter_gate as gate
from load_true_3d_rbc_pilot_dataset import (
    PARAM_NAMES,
    ROOT,
    V3_240_DATASET_ID,
    aggregate_prediction_rows,
    check_no_overwrite,
    denormalize_y,
    evaluate_param_predictions,
    gate_manifest,
    load_dataset,
    normalize_x,
    normalize_y,
    resolve_dataset,
    sha256_file,
    split_indices,
    train_normalization,
    write_csv,
)
from run_true_3d_rbc_formal_benchmark_20_77_candidate import (
    add_profile_error_rows,
    split_aggregate,
)


PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_baseline_artifact_recovery_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_baseline_artifact_recovery_summary.txt"
VERIFY_SUMMARY = ROOT / "results/summaries/true_3d_rbc_baseline_artifact_verification_summary.txt"
RECOVERY_METRICS = ROOT / "results/metrics/true_3d_rbc_baseline_artifact_recovery_metrics.csv"
VERIFY_CSV = ROOT / "results/metrics/true_3d_rbc_baseline_artifact_verification.csv"
ARTIFACT_MANIFEST = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"

ARTIFACT_DIR = ROOT / "checkpoints/true_3d_rbc_baseline_artifacts"
CHECKPOINT_PATH = ARTIFACT_DIR / "true_3d_rbc_v3_240_seed42_20_77_baseline.pt"
PREDICTION_PATH = ARTIFACT_DIR / "true_3d_rbc_v3_240_seed42_20_77_predictions.npz"

REFERENCE_SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_seed_summary.csv"
REFERENCE_METRICS = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_metrics.csv"
REFERENCE_PROFILE_METRICS = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_profile_metrics.csv"
REFERENCE_SUMMARY = ROOT / "results/summaries/true_3d_rbc_formal_benchmark_20_77_summary.txt"

RECOVERY_FIELDS = [
    "split",
    "sample_count",
    "normalized_param_mae",
    "dimension_mae_norm",
    "curvature_mae_norm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "wMAE_auxiliary",
    "projected_mask_iou",
    "projected_mask_dice",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "max_depth_error_m",
    "volume_proxy_rel_error",
]

VERIFY_FIELDS = ["metric", "reference_value", "recovered_value", "delta", "tolerance", "pass"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def selected_reference_row() -> dict[str, str]:
    rows = read_csv(REFERENCE_SEED_SUMMARY)
    selected = [row for row in rows if str(row.get("selected_seed", "")).lower() == "true"]
    if len(selected) != 1:
        raise RuntimeError(f"expected exactly one selected 20.85 reference seed row, found {len(selected)}")
    return selected[0]


def as_float(row: dict[str, Any], key: str) -> float:
    value = row[key]
    if value in {"", None}:
        return math.nan
    return float(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def recovery_rows(profile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        agg = split_aggregate(profile_rows, split_name)
        rows.append(
            {
                "split": split_name,
                "sample_count": agg["sample_count"],
                "normalized_param_mae": agg["normalized_param_mae_mean_mean"],
                "dimension_mae_norm": agg["dimension_param_mae_norm_mean"],
                "curvature_mae_norm": agg["curvature_param_mae_norm_mean"],
                "L_mae_mm": agg["L_mae_mm_mean"],
                "W_mae_mm": agg["W_mae_mm_mean"],
                "D_mae_mm": agg["D_mae_mm_mean"],
                "wLD_abs_error": agg["wLD_abs_error_mean"],
                "wWD_abs_error": agg["wWD_abs_error_mean"],
                "wLW_abs_error": agg["wLW_abs_error_mean"],
                "wMAE_auxiliary": agg["curvature_mae_mean_mean"],
                "projected_mask_iou": agg["projected_mask_iou_mean"],
                "projected_mask_dice": agg["projected_mask_dice_mean"],
                "profile_depth_rmse_m": agg["profile_depth_rmse_m_mean"],
                "er_like_profile_error": agg["er_like_profile_error_mean"],
                "max_depth_error_m": agg["max_depth_error_m_mean"],
                "volume_proxy_rel_error": agg["volume_proxy_rel_error_mean"],
            }
        )
    return rows


def write_preflight(args: argparse.Namespace) -> None:
    entry, manifest, npz_path = resolve_dataset(args.dataset_id)
    checks = gate_manifest(entry, manifest, npz_path, args.dataset_id)
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset)
    expected_split = {"train": 162, "val": 39, "test": 39}
    reference_paths = [REFERENCE_SEED_SUMMARY, REFERENCE_METRICS, REFERENCE_PROFILE_METRICS, REFERENCE_SUMMARY]
    missing_refs = [str(path.relative_to(ROOT)) for path in reference_paths if not path.exists()]
    failed_gate = [row for row in checks if not row["pass"]]
    schema_pass = (
        dataset.delta_b.shape == (240, 3, 3, 201)
        and dataset.x_channels.shape == (240, 9, 201)
        and {name: len(idx) for name, idx in splits.items()} == expected_split
        and dataset.axis_names == ["Bx", "By", "Bz"]
        and bool(np.isfinite(dataset.delta_b).all())
        and bool(np.isfinite(dataset.rbc_params).all())
    )
    output_artifacts_exist = {
        "checkpoint_path_exists": CHECKPOINT_PATH.exists(),
        "prediction_artifact_exists": PREDICTION_PATH.exists(),
        "artifact_manifest_exists": ARTIFACT_MANIFEST.exists(),
    }
    lines = [
        "20.88a true 3D RBC baseline artifact recovery preflight summary",
        "",
        f"dataset_id: {args.dataset_id}",
        f"registry_manifest_gate_pass: {not failed_gate}",
        f"schema_pass: {schema_pass}",
        f"npz_path_resolved_from_manifest: {npz_path}",
        f"input_shape: delta_b={list(dataset.delta_b.shape)}, conv1d={list(dataset.x_channels.shape)}",
        f"split_counts: {{'train': {len(splits['train'])}, 'val': {len(splits['val'])}, 'test': {len(splits['test'])}}}",
        f"reference_material_missing: {', '.join(missing_refs) if missing_refs else 'none'}",
        f"existing_output_artifacts: {output_artifacts_exist}",
        "artifact_recovery_seed: 42",
        "model_protocol: 20.77/20.85 small Conv1D encoder + MLP six-parameter head",
        "loss: weighted SmoothL1 with dimension weights=1.0 and curvature weights=0.5",
        "selection: validation-only best checkpoint for fixed seed=42",
        "latest_newest_npz_scan: false",
        "COMSOL_run: false",
        "new_data_generation: false",
        "NPZ_modification: false",
        "tuning_or_architecture_change: false",
        "forbidden_submit: data/, NPZ, checkpoint, preview PNG, notes, __pycache__, baseline docs, CURRENT_BASELINE.md, scripts/visualize_current_baseline.py",
        "stage_gate: pass" if not failed_gate and schema_pass and not missing_refs else "stage_gate: blocker",
    ]
    write_text(PREFLIGHT_SUMMARY, lines)
    if failed_gate or not schema_pass or missing_refs:
        raise RuntimeError("artifact recovery preflight blocker; see preflight summary")


def build_metrics(dataset: Any, stats: dict[str, np.ndarray], pred_norm: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    pred_raw = denormalize_y(pred_norm, stats)
    profile_rows = add_profile_error_rows(dataset, pred_raw, evaluate_param_predictions(dataset, pred_raw, stats))
    for row in profile_rows:
        row["seed"] = 42
        row["selected_seed"] = True
    return pred_raw, profile_rows, recovery_rows(profile_rows)


def save_checkpoint(args: argparse.Namespace, model: torch.nn.Module, stats: dict[str, np.ndarray], seed_out: dict[str, Any], dataset: Any) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "artifact_stage": "20.88a_baseline_inference_artifact_recovery",
        "dataset_id": args.dataset_id,
        "seed": args.seed,
        "model_class": "RBCConvRegressor",
        "model_family": "20.77_small_conv1d_encoder_mlp_six_parameter_head",
        "model_state_dict": model.state_dict(),
        "param_names": PARAM_NAMES,
        "input_shape_delta_b": tuple(dataset.delta_b.shape),
        "input_shape_conv1d": tuple(dataset.x_channels.shape),
        "normalization": {
            "x_mean": stats["x_mean"],
            "x_std": stats["x_std"],
            "y_mean": stats["y_mean"],
            "y_std": stats["y_std"],
        },
        "protocol": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": "weighted_smooth_l1",
            "param_weights": gate.PARAM_WEIGHTS.detach().cpu().numpy(),
            "selection_metric": "val_normalized_param_mae + 0.25*val_dimension_mae_norm + 0.10*val_curvature_mae_norm",
            "test_selection": False,
            "architecture_tuning": False,
        },
        "best_epoch": seed_out["best_epoch"],
        "best_val_selection_metric": seed_out["best_val_score"],
        "min_train_epoch": seed_out["min_train_epoch"],
        "min_train_normalized_param_mae": seed_out["min_train_normalized_param_mae"],
    }
    torch.save(checkpoint, CHECKPOINT_PATH)


def save_predictions(args: argparse.Namespace, dataset: Any, pred_norm: np.ndarray, pred_raw: np.ndarray, seed_out: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PREDICTION_PATH,
        dataset_id=np.array([args.dataset_id]),
        seed=np.array([args.seed], dtype=np.int64),
        best_epoch=np.array([seed_out["best_epoch"]], dtype=np.int64),
        best_val_selection_metric=np.array([seed_out["best_val_score"]], dtype=np.float64),
        pred_norm=np.asarray(pred_norm, dtype=np.float32),
        pred_params_m=np.asarray(pred_raw, dtype=np.float32),
        sample_ids=np.asarray(dataset.sample_ids).astype(str),
        split=np.asarray(dataset.split).astype(str),
        curvature_template=np.asarray(dataset.curvature_template).astype(str),
        depth_bin=np.asarray(dataset.depth_bin).astype(str),
        aspect_bin=np.asarray(dataset.aspect_bin).astype(str),
        size_bin=np.asarray(dataset.size_bin).astype(str),
    )


def write_artifact_manifest(args: argparse.Namespace, dataset: Any, stats: dict[str, np.ndarray], seed_out: dict[str, Any], metrics_rows: list[dict[str, Any]]) -> None:
    test_row = next(row for row in metrics_rows if row["split"] == "test")
    manifest = {
        "artifact_id": "true_3d_rbc_v3_240_seed42_20_77_baseline_inference_artifact",
        "stage": "20.88a_baseline_artifact_recovery",
        "dataset_id": args.dataset_id,
        "route": "true_3d_piao_style",
        "exact_piao_rbc": False,
        "rbc_style_approximation": True,
        "purpose": "Enable 20.88 observation perturbation robustness evaluation without retraining.",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "prediction_artifact_path": str(PREDICTION_PATH),
        "prediction_artifact_sha256": sha256_file(PREDICTION_PATH),
        "checkpoint_committed": False,
        "prediction_artifact_committed": False,
        "artifact_paths_are_ignored": True,
        "model_class": "RBCConvRegressor",
        "model_family": "20.77 small Conv1D encoder + MLP six-parameter head",
        "seed": args.seed,
        "best_epoch": seed_out["best_epoch"],
        "best_val_selection_metric": seed_out["best_val_score"],
        "input": {
            "delta_b_shape": list(dataset.delta_b.shape),
            "conv1d_shape": list(dataset.x_channels.shape),
            "axis_names": dataset.axis_names,
            "model_input_fields": ["delta_b"],
            "forbidden_model_input_fields": ["rbc_params", "projected_mask_2d", "profile_depth_grid_m", "split", "curvature_template", "depth_bin", "aspect_bin", "sample_id"],
        },
        "normalization": {
            "x_mean_shape": list(stats["x_mean"].shape),
            "x_std_shape": list(stats["x_std"].shape),
            "y_mean_shape": list(stats["y_mean"].shape),
            "y_std_shape": list(stats["y_std"].shape),
            "fit_scope": "train_split_only",
        },
        "split_counts": {name: int(np.sum(dataset.split == name)) for name in ("train", "val", "test")},
        "protocol": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": "weighted_smooth_l1",
            "selection": "validation-only best checkpoint for fixed seed=42",
            "test_final_only": True,
            "hyperparameter_tuning": False,
            "architecture_change": False,
        },
        "test_metrics": test_row,
        "forbidden_commit_artifacts": ["checkpoint .pt", "prediction .npz", "data/", "NPZ", "checkpoint", "preview PNG", "notes", "__pycache__"],
    }
    ARTIFACT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_MANIFEST.write_text(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verification_rows(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ref = selected_reference_row()
    test_row = next(row for row in metrics_rows if row["split"] == "test")
    comparisons = [
        ("test_normalized_param_mae", "test_normalized_param_mae", "normalized_param_mae", 1.0e-6),
        ("test_L_mae_mm", "test_L_mae_mm", "L_mae_mm", 1.0e-6),
        ("test_W_mae_mm", "test_W_mae_mm", "W_mae_mm", 1.0e-6),
        ("test_D_mae_mm", "test_D_mae_mm", "D_mae_mm", 1.0e-6),
        ("test_wMAE_auxiliary", "test_wMAE_auxiliary", "wMAE_auxiliary", 1.0e-6),
        ("test_projected_mask_iou", "test_projected_mask_iou", "projected_mask_iou", 1.0e-6),
        ("test_projected_mask_dice", "test_projected_mask_dice", "projected_mask_dice", 1.0e-6),
        ("test_profile_depth_rmse_m", "test_profile_depth_rmse_m", "profile_depth_rmse_m", 1.0e-12),
        ("test_er_like_profile_error", "test_er_like_profile_error", "er_like_profile_error", 1.0e-9),
    ]
    rows = []
    for name, ref_key, got_key, tol in comparisons:
        reference = float(ref[ref_key])
        recovered = float(test_row[got_key])
        delta = abs(recovered - reference)
        rows.append(
            {
                "metric": name,
                "reference_value": reference,
                "recovered_value": recovered,
                "delta": delta,
                "tolerance": tol,
                "pass": delta <= tol,
            }
        )
    return rows


def verify_artifact(args: argparse.Namespace, dataset: Any, stats: dict[str, np.ndarray], y_norm: np.ndarray, metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not CHECKPOINT_PATH.exists() or not PREDICTION_PATH.exists():
        raise RuntimeError("checkpoint or prediction artifact missing after export")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = gate.RBCConvRegressor()
    model.load_state_dict(checkpoint["model_state_dict"])
    x_norm = ((dataset.x_channels - checkpoint["normalization"]["x_mean"]) / checkpoint["normalization"]["x_std"]).astype(np.float32)
    pred_norm = gate.predict_norm(model, x_norm)
    pred_raw, _, reload_metrics_rows = build_metrics(dataset, stats, pred_norm)
    with np.load(PREDICTION_PATH, allow_pickle=True) as pred_npz:
        pred_artifact_raw = np.asarray(pred_npz["pred_params_m"], dtype=np.float32)
    if not np.allclose(pred_raw, pred_artifact_raw, atol=1.0e-7):
        raise RuntimeError("reloaded checkpoint predictions do not match prediction artifact")
    if not np.isfinite(y_norm).all():
        raise RuntimeError("normalized labels are not finite")
    verify_rows = verification_rows(reload_metrics_rows)
    verify_rows.extend(
        [
            {"metric": "checkpoint_exists", "reference_value": True, "recovered_value": CHECKPOINT_PATH.exists(), "delta": 0, "tolerance": 0, "pass": CHECKPOINT_PATH.exists()},
            {"metric": "prediction_artifact_exists", "reference_value": True, "recovered_value": PREDICTION_PATH.exists(), "delta": 0, "tolerance": 0, "pass": PREDICTION_PATH.exists()},
            {"metric": "checkpoint_reload_prediction_match", "reference_value": True, "recovered_value": True, "delta": 0, "tolerance": 0, "pass": True},
            {"metric": "manifest_sufficient_for_20_88", "reference_value": True, "recovered_value": ARTIFACT_MANIFEST.exists(), "delta": 0, "tolerance": 0, "pass": ARTIFACT_MANIFEST.exists()},
        ]
    )
    write_csv(VERIFY_CSV, verify_rows, VERIFY_FIELDS)
    failed = [row for row in verify_rows if not bool(row["pass"])]
    test_row = next(row for row in reload_metrics_rows if row["split"] == "test")
    write_text(
        VERIFY_SUMMARY,
        [
            "20.88a true 3D RBC baseline artifact verification summary",
            "",
            f"artifact_manifest: {ARTIFACT_MANIFEST}",
            f"checkpoint_reloaded: {CHECKPOINT_PATH}",
            f"prediction_artifact_reloaded: {PREDICTION_PATH}",
            f"checkpoint_reload_prediction_match: {not failed or all(row['metric'] != 'checkpoint_reload_prediction_match' for row in failed)}",
            f"test_normalized_mae: {test_row['normalized_param_mae']:.12f}",
            f"test_profile_depth_rmse_m: {test_row['profile_depth_rmse_m']:.15f}",
            f"test_er_like_profile_error: {test_row['er_like_profile_error']:.12f}",
            f"test_LWD_mae_mm: L={test_row['L_mae_mm']:.12f}, W={test_row['W_mae_mm']:.12f}, D={test_row['D_mae_mm']:.12f}",
            f"test_projected_mask_iou_dice: {test_row['projected_mask_iou']:.12f}, {test_row['projected_mask_dice']:.12f}",
            f"clean_20_85_reproduction_pass: {not failed}",
            f"failed_checks: {', '.join(str(row['metric']) for row in failed) if failed else 'none'}",
        ],
    )
    if failed:
        raise RuntimeError("artifact verification failed; see verification summary/csv")
    return reload_metrics_rows


def run(args: argparse.Namespace) -> int:
    if args.dataset_id != V3_240_DATASET_ID:
        raise ValueError(
            "20.88a artifact recovery is fixed to "
            f"{V3_240_DATASET_ID}; got {args.dataset_id}"
        )
    if args.seed != 42:
        raise ValueError(f"20.88a artifact recovery is fixed to seed=42; got {args.seed}")
    outputs = [
        PREFLIGHT_SUMMARY,
        SUMMARY,
        VERIFY_SUMMARY,
        RECOVERY_METRICS,
        VERIFY_CSV,
        ARTIFACT_MANIFEST,
        CHECKPOINT_PATH,
        PREDICTION_PATH,
    ]
    check_no_overwrite(outputs, args.overwrite)
    write_preflight(args)
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset)
    stats = train_normalization(dataset)
    x_norm = normalize_x(dataset, stats)
    y_norm = normalize_y(dataset, stats)

    seed_out = gate.train_one_seed(args.seed, x_norm, y_norm, splits, args)
    pred_raw, profile_rows, metrics_rows = build_metrics(dataset, stats, seed_out["pred_norm"])
    write_csv(RECOVERY_METRICS, metrics_rows, RECOVERY_FIELDS)
    save_checkpoint(args, seed_out["model"], stats, seed_out, dataset)
    save_predictions(args, dataset, seed_out["pred_norm"], pred_raw, seed_out)
    write_artifact_manifest(args, dataset, stats, seed_out, metrics_rows)
    verified_rows = verify_artifact(args, dataset, stats, y_norm, metrics_rows)

    train_row = next(row for row in verified_rows if row["split"] == "train")
    val_row = next(row for row in verified_rows if row["split"] == "val")
    test_row = next(row for row in verified_rows if row["split"] == "test")
    write_text(
        SUMMARY,
        [
            "20.88a true 3D RBC baseline artifact recovery summary",
            "",
            f"dataset_id: {args.dataset_id}",
            "artifact_recovery_only: true",
            "COMSOL_run: false",
            "new_data_generation: false",
            "NPZ_modification: false",
            "CURRENT_BASELINE_update: false",
            "hyperparameter_tuning: false",
            "architecture_change: false",
            "test_based_selection: false",
            "model_family: 20.77/20.85 small Conv1D encoder + MLP six-parameter head",
            f"seed: {args.seed}",
            f"best_epoch: {seed_out['best_epoch']}",
            f"best_val_selection_metric: {seed_out['best_val_score']:.12f}",
            f"checkpoint_path_ignored_not_committed: {CHECKPOINT_PATH}",
            f"prediction_artifact_path_ignored_not_committed: {PREDICTION_PATH}",
            f"artifact_manifest_committable: {ARTIFACT_MANIFEST}",
            f"checkpoint_sha256: {sha256_file(CHECKPOINT_PATH)}",
            f"prediction_artifact_sha256: {sha256_file(PREDICTION_PATH)}",
            f"train_val_test_normalized_mae: {train_row['normalized_param_mae']:.6f} / {val_row['normalized_param_mae']:.6f} / {test_row['normalized_param_mae']:.6f}",
            f"test_profile_depth_rmse_m: {test_row['profile_depth_rmse_m']:.9f}",
            f"test_er_like_profile_error: {test_row['er_like_profile_error']:.6f}",
            f"test_LWD_mae_mm: L={test_row['L_mae_mm']:.6f}, W={test_row['W_mae_mm']:.6f}, D={test_row['D_mae_mm']:.6f}",
            f"test_wMAE_auxiliary: {test_row['wMAE_auxiliary']:.6f}",
            f"test_projected_mask_iou_dice: {test_row['projected_mask_iou']:.6f}, {test_row['projected_mask_dice']:.6f}",
            "clean_20_85_reproduction: pass",
            "next_step: return to 20.88 observation perturbation robustness audit using the artifact manifest.",
        ],
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
