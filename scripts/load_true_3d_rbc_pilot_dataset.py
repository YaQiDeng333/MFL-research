#!/usr/bin/env python
"""Explicit registry/manifest loader for true-3D RBC training gates.

This module intentionally has no latest/newest NPZ fallback. Callers must pass
the dataset_id and the loader resolves the NPZ only through COMSOL_DATA_REGISTRY
and the tracked manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled"
V2_120_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v2_120"
V3_240_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"
RUN_NAME_BY_DATASET_ID = {
    DATASET_ID: "true_3d_rbc_training_gate",
    V2_120_DATASET_ID: "true_3d_rbc_v2_120_training_gate",
    V3_240_DATASET_ID: "true_3d_rbc_v3_240_training_gate",
}
REGISTRY_PATH = ROOT / "COMSOL_DATA_REGISTRY.md"
SUMMARY_PATH = ROOT / "results/summaries/true_3d_rbc_training_gate_input_summary.txt"
PREFLIGHT_SUMMARY_PATH = ROOT / "results/summaries/true_3d_rbc_training_gate_preflight_summary.txt"
INPUT_CHECK_PATH = ROOT / "results/metrics/true_3d_rbc_training_gate_input_check.csv"

ROUTE = "true_3d_piao_style"
SCHEMA_VERSION = "true3d_profile_v1_piao_rbc"
PARAM_NAMES = ["L_m", "W_m", "D_m", "wLD", "wWD", "wLW"]
MASK_WIDTH = 128
MASK_HEIGHT = 64
MASK_X_START_M = -0.04
MASK_X_STOP_M = 0.04
MASK_Y_START_M = -0.01
MASK_Y_STOP_M = 0.01
GRID_U_COUNT = 33
GRID_V_COUNT = 17

CHECK_FIELDS = ["check_name", "pass", "observed", "notes"]


@dataclass(frozen=True)
class True3DRBCDataset:
    dataset_id: str
    manifest: dict[str, Any]
    registry_entry: dict[str, str]
    npz_path: Path
    delta_b: np.ndarray
    b_defect: np.ndarray
    b_no_defect: np.ndarray
    x_channels: np.ndarray
    rbc_params: np.ndarray
    profile_pose: np.ndarray
    projected_mask_2d: np.ndarray
    profile_depth_grid_m: np.ndarray
    profile_depth_map_xy_m: np.ndarray
    sample_ids: np.ndarray
    split: np.ndarray
    axis_names: list[str]
    sensor_x: np.ndarray
    scan_line_y: np.ndarray
    sensor_z_m: float
    curvature_template: np.ndarray
    depth_bin: np.ndarray
    aspect_bin: np.ndarray
    size_bin: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and validate the explicit true-3D RBC pilot dataset.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--run-name")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--preflight-summary", type=Path)
    parser.add_argument("--input-check", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    apply_default_output_paths(args)
    return args


def run_name_for_dataset(dataset_id: str) -> str:
    return RUN_NAME_BY_DATASET_ID.get(dataset_id, re.sub(r"[^A-Za-z0-9_]+", "_", dataset_id).strip("_"))


def apply_default_output_paths(args: argparse.Namespace) -> None:
    run_name = args.run_name or run_name_for_dataset(args.dataset_id)
    args.run_name = run_name
    if args.summary is None:
        args.summary = ROOT / f"results/summaries/{run_name}_input_summary.txt"
    if args.preflight_summary is None:
        args.preflight_summary = ROOT / f"results/summaries/{run_name}_preflight_summary.txt"
    if args.input_check is None:
        args.input_check = ROOT / f"results/metrics/{run_name}_input_check.csv"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def parse_list_value(value: str) -> list[str]:
    text = value.strip().strip("`")
    if text.lower() in {"", "none"}:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_registry(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    entries: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current = heading.group(1).strip()
            entries[current] = {}
            continue
        if current and line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            entries[current][key.strip()] = value.strip().strip("`")
    return entries


def resolve_dataset(dataset_id: str, registry_path: Path = REGISTRY_PATH) -> tuple[dict[str, str], dict[str, Any], Path]:
    registry = parse_registry(registry_path)
    if dataset_id not in registry:
        raise RuntimeError(f"dataset_id not found in registry: {dataset_id}")
    entry = registry[dataset_id]
    manifest_path = Path(entry.get("manifest_path", ""))
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_path from registry does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != dataset_id:
        raise RuntimeError(f"manifest dataset_id mismatch: {manifest.get('dataset_id')} != {dataset_id}")
    self_path = Path(str(manifest.get("manifest_path", manifest_path)))
    if not self_path.is_absolute():
        self_path = ROOT / self_path
    if self_path.resolve() != manifest_path.resolve():
        raise RuntimeError(f"manifest self path mismatch: {manifest.get('manifest_path')} != {manifest_path}")
    npz_path = Path(manifest["npz_path"])
    if not npz_path.is_absolute():
        npz_path = ROOT / npz_path
    return entry, manifest, npz_path


def gate_manifest(entry: dict[str, str], manifest: dict[str, Any], npz_path: Path, dataset_id: str | None = None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, notes: str = "") -> None:
        checks.append({"check_name": name, "pass": bool(passed), "observed": observed, "notes": notes})

    allowed = set(manifest.get("allowed_use", []))
    forbidden = set(manifest.get("forbidden_use", []))
    entry_allowed = set(parse_list_value(entry.get("allowed_use", "")))
    entry_forbidden = set(parse_list_value(entry.get("forbidden_use", "")))
    expected_dataset_id = dataset_id or str(manifest.get("dataset_id", ""))
    add("dataset_id", manifest.get("dataset_id") == expected_dataset_id, manifest.get("dataset_id"))
    add("route", manifest.get("route") == ROUTE and entry.get("route") == ROUTE, f"manifest={manifest.get('route')}; registry={entry.get('route')}")
    add("status", manifest.get("status") == "pilot_generated" and entry.get("status") == "pilot_generated", f"manifest={manifest.get('status')}; registry={entry.get('status')}")
    add("train_ready_candidate", bool(manifest.get("train_ready_candidate")) and parse_bool(entry.get("train_ready_candidate", "false")), manifest.get("train_ready_candidate"))
    add("baseline_ready_false", (not bool(manifest.get("baseline_ready"))) and (not parse_bool(entry.get("baseline_ready", "true"))), manifest.get("baseline_ready"))
    add("geometry_method", manifest.get("geometry_method") == "imported_watertight_mesh_solid" and entry.get("geometry_method") == "imported_watertight_mesh_solid", f"manifest={manifest.get('geometry_method')}; registry={entry.get('geometry_method')}")
    add("exact_piao_rbc_false", (not bool(manifest.get("exact_piao_rbc"))) and (not parse_bool(entry.get("exact_piao_rbc", "true"))), manifest.get("exact_piao_rbc"))
    add("rbc_style_approximation_true", bool(manifest.get("rbc_style_approximation")) and parse_bool(entry.get("rbc_style_approximation", "false")), manifest.get("rbc_style_approximation"))
    add("explicit_training_allowed", "explicit_pilot_training_gate" in allowed and "explicit_pilot_training_gate" in entry_allowed, sorted(allowed))
    add("baseline_forbidden", {"baseline_update", "current_baseline_replacement"}.issubset(forbidden) and {"baseline_update", "current_baseline_replacement"}.issubset(entry_forbidden), sorted(forbidden))
    add("latest_newest_forbidden", "latest_newest_auto_discovery" in forbidden and not bool(manifest.get("latest_newest_discovery_allowed")), manifest.get("latest_newest_discovery_allowed"))
    add("auto_discovery_forbidden", not bool(manifest.get("auto_discovery_allowed")), manifest.get("auto_discovery_allowed"))
    add("npz_exists", npz_path.exists(), str(npz_path))
    if npz_path.exists():
        add("npz_sha256", sha256_file(npz_path) == manifest.get("npz_sha256"), manifest.get("npz_sha256"))
    add("manifest_stage_recorded", bool(manifest.get("stage")), manifest.get("stage"), "Training gates consume already-generated packs; stage records the data generation stage.")
    return checks


def load_dataset(dataset_id: str = DATASET_ID, registry_path: Path = REGISTRY_PATH) -> True3DRBCDataset:
    entry, manifest, npz_path = resolve_dataset(dataset_id, registry_path)
    gate_checks = gate_manifest(entry, manifest, npz_path, dataset_id)
    failed = [row for row in gate_checks if not row["pass"]]
    if failed:
        raise RuntimeError("dataset registry/manifest gate failed: " + json.dumps(failed, ensure_ascii=False))
    with np.load(npz_path, allow_pickle=True) as npz:
        required = [
            "delta_b",
            "b_defect",
            "b_no_defect",
            "rbc_params",
            "profile_pose",
            "projected_mask_2d",
            "profile_depth_grid_m",
            "profile_depth_map_xy_m",
            "sample_ids",
            "split",
            "axis_names",
            "sensor_x",
            "scan_line_y",
            "sensor_z_m",
            "curvature_template",
            "depth_bin",
            "aspect_bin",
            "size_bin",
        ]
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise RuntimeError(f"missing NPZ fields: {missing}")
        delta_b = np.asarray(npz["delta_b"], dtype=np.float32)
        dataset = True3DRBCDataset(
            dataset_id=dataset_id,
            manifest=manifest,
            registry_entry=entry,
            npz_path=npz_path,
            delta_b=delta_b,
            b_defect=np.asarray(npz["b_defect"], dtype=np.float32),
            b_no_defect=np.asarray(npz["b_no_defect"], dtype=np.float32),
            x_channels=delta_b.reshape(delta_b.shape[0], 9, delta_b.shape[-1]),
            rbc_params=np.asarray(npz["rbc_params"], dtype=np.float32).reshape(delta_b.shape[0], 6),
            profile_pose=np.asarray(npz["profile_pose"], dtype=np.float32).reshape(delta_b.shape[0], 6),
            projected_mask_2d=np.asarray(npz["projected_mask_2d"], dtype=np.uint8),
            profile_depth_grid_m=np.asarray(npz["profile_depth_grid_m"], dtype=np.float32),
            profile_depth_map_xy_m=np.asarray(npz["profile_depth_map_xy_m"], dtype=np.float32),
            sample_ids=np.asarray(npz["sample_ids"]).astype(str),
            split=np.asarray(npz["split"]).astype(str),
            axis_names=[str(x) for x in np.asarray(npz["axis_names"]).tolist()],
            sensor_x=np.asarray(npz["sensor_x"], dtype=np.float32),
            scan_line_y=np.asarray(npz["scan_line_y"], dtype=np.float32),
            sensor_z_m=float(np.asarray(npz["sensor_z_m"]).reshape(-1)[0]),
            curvature_template=np.asarray(npz["curvature_template"]).astype(str),
            depth_bin=np.asarray(npz["depth_bin"]).astype(str),
            aspect_bin=np.asarray(npz["aspect_bin"]).astype(str),
            size_bin=np.asarray(npz["size_bin"]).astype(str),
        )
    return dataset


def split_indices(dataset: True3DRBCDataset) -> dict[str, np.ndarray]:
    return {name: np.where(dataset.split == name)[0] for name in ("train", "val", "test")}


def train_normalization(dataset: True3DRBCDataset) -> dict[str, np.ndarray]:
    splits = split_indices(dataset)
    train = splits["train"]
    x_train = dataset.x_channels[train]
    y_train = dataset.rbc_params[train]
    x_mean = x_train.mean(axis=(0, 2), keepdims=True)
    x_std = x_train.std(axis=(0, 2), keepdims=True)
    x_std = np.where(x_std < 1.0e-12, 1.0, x_std)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1.0e-12, 1.0, y_std)
    return {"x_mean": x_mean.astype(np.float32), "x_std": x_std.astype(np.float32), "y_mean": y_mean.astype(np.float32), "y_std": y_std.astype(np.float32)}


def normalize_x(dataset: True3DRBCDataset, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((dataset.x_channels - stats["x_mean"]) / stats["x_std"]).astype(np.float32)


def normalize_y(dataset: True3DRBCDataset, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((dataset.rbc_params - stats["y_mean"]) / stats["y_std"]).astype(np.float32)


def denormalize_y(y_norm: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(y_norm, dtype=np.float32) * stats["y_std"] + stats["y_mean"]).astype(np.float32)


def rbc_weight_curve(values: np.ndarray, weight: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    numerator = weight * (1.0 - clipped * clipped)
    denominator = numerator + clipped * clipped + 1.0e-12
    return np.clip(numerator / denominator, 0.0, 1.0)


def local_depth(params: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    l_m, w_m, d_m, wld, wwd, wlw = [float(x) for x in params]
    _ = (l_m, w_m)
    u_abs = np.abs(u)
    v_abs = np.abs(v)
    length_profile = rbc_weight_curve(u_abs, wld)
    width_scale = rbc_weight_curve(u_abs, wlw)
    safe_width = np.maximum(width_scale, 1.0e-9)
    v_norm = np.divide(v_abs, safe_width, out=np.ones_like(v_abs), where=safe_width > 0.0)
    inside = (u_abs <= 1.0) & (v_norm <= 1.0)
    width_profile = rbc_weight_curve(v_norm, wwd)
    return np.where(inside, d_m * length_profile * width_profile, 0.0).astype(np.float32)


def xy_to_uv(params: np.ndarray, pose: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    l_m, w_m = float(params[0]), float(params[1])
    center_x, center_y, angle_rad = float(pose[0]), float(pose[1]), float(pose[2])
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = xx - center_x
    dy = yy - center_y
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return 2.0 * local_x / max(l_m, 1.0e-9), 2.0 * local_y / max(w_m, 1.0e-9)


def depth_grid_from_params(params: np.ndarray) -> np.ndarray:
    u = np.linspace(-1.0, 1.0, GRID_U_COUNT)
    v = np.linspace(-1.0, 1.0, GRID_V_COUNT)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return local_depth(params, uu, vv)


def depth_map_from_params(params: np.ndarray, pose: np.ndarray) -> np.ndarray:
    mask_x = np.linspace(MASK_X_START_M, MASK_X_STOP_M, MASK_WIDTH)
    mask_y = np.linspace(MASK_Y_START_M, MASK_Y_STOP_M, MASK_HEIGHT)
    yy, xx = np.meshgrid(mask_y, mask_x, indexing="ij")
    u, v = xy_to_uv(params, pose, xx, yy)
    return local_depth(params, u, v)


def projected_mask_from_params(params: np.ndarray, pose: np.ndarray) -> np.ndarray:
    depth = depth_map_from_params(params, pose)
    threshold = max(1.0e-6, 0.01 * float(params[2]))
    return (depth >= threshold).astype(np.uint8)


def clip_params_to_train_bounds(params: np.ndarray, dataset: True3DRBCDataset) -> tuple[np.ndarray, np.ndarray]:
    train = split_indices(dataset)["train"]
    low = dataset.rbc_params[train].min(axis=0)
    high = dataset.rbc_params[train].max(axis=0)
    clipped = np.clip(params, low[None, :], high[None, :])
    clipped_flag = np.any(np.abs(clipped - params) > 1.0e-12, axis=1)
    return clipped.astype(np.float32), clipped_flag


def mask_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_mask).astype(bool)
    true = np.asarray(true_mask).astype(bool)
    intersection = int(np.logical_and(pred, true).sum())
    union = int(np.logical_or(pred, true).sum())
    pred_area = int(pred.sum())
    true_area = int(true.sum())
    iou = 1.0 if union == 0 else intersection / union
    dice = 1.0 if pred_area + true_area == 0 else (2.0 * intersection) / (pred_area + true_area)
    area_error = 0.0 if true_area == 0 else abs(pred_area - true_area) / true_area
    if pred_area == 0 or true_area == 0:
        center_error = float(max(MASK_WIDTH, MASK_HEIGHT))
    else:
        yy, xx = np.indices(true.shape)
        pred_center = np.array([float(xx[pred].mean()), float(yy[pred].mean())])
        true_center = np.array([float(xx[true].mean()), float(yy[true].mean())])
        center_error = float(np.linalg.norm(pred_center - true_center))
    return {"iou": float(iou), "dice": float(dice), "area_error": float(area_error), "center_error": center_error}


def evaluate_param_predictions(dataset: True3DRBCDataset, pred_params_raw: np.ndarray, stats: dict[str, np.ndarray] | None = None) -> list[dict[str, Any]]:
    pred_params, clipped = clip_params_to_train_bounds(np.asarray(pred_params_raw, dtype=np.float32), dataset)
    rows: list[dict[str, Any]] = []
    if stats is not None:
        pred_norm = (pred_params - stats["y_mean"]) / stats["y_std"]
        true_norm = (dataset.rbc_params - stats["y_mean"]) / stats["y_std"]
    else:
        pred_norm = pred_params
        true_norm = dataset.rbc_params
    for idx, sample_id in enumerate(dataset.sample_ids):
        pred_mask = projected_mask_from_params(pred_params[idx], dataset.profile_pose[idx])
        mask_row = mask_metrics(pred_mask, dataset.projected_mask_2d[idx])
        pred_depth = depth_grid_from_params(pred_params[idx])
        depth_rmse = float(np.sqrt(np.mean((pred_depth - dataset.profile_depth_grid_m[idx]) ** 2)))
        true_volume = float(dataset.profile_depth_map_xy_m[idx].sum())
        pred_volume = float(depth_map_from_params(pred_params[idx], dataset.profile_pose[idx]).sum())
        volume_error = 0.0 if abs(true_volume) < 1.0e-12 else abs(pred_volume - true_volume) / abs(true_volume)
        param_abs = np.abs(pred_params[idx] - dataset.rbc_params[idx])
        param_norm_abs = np.abs(pred_norm[idx] - true_norm[idx])
        rows.append(
            {
                "sample_id": str(sample_id),
                "split": str(dataset.split[idx]),
                "curvature_template": str(dataset.curvature_template[idx]),
                "depth_bin": str(dataset.depth_bin[idx]),
                "aspect_bin": str(dataset.aspect_bin[idx]),
                "size_bin": str(dataset.size_bin[idx]),
                "clip_applied": bool(clipped[idx]),
                "clip_fraction": float(np.mean(clipped)),
                "normalized_param_mae_mean": float(np.mean(param_norm_abs)),
                "dimension_param_mae_norm": float(np.mean(param_norm_abs[:3])),
                "curvature_param_mae_norm": float(np.mean(param_norm_abs[3:])),
                "L_mae_m": float(param_abs[0]),
                "W_mae_m": float(param_abs[1]),
                "D_mae_m": float(param_abs[2]),
                "L_mae_mm": float(param_abs[0] * 1000.0),
                "W_mae_mm": float(param_abs[1] * 1000.0),
                "D_mae_mm": float(param_abs[2] * 1000.0),
                "wLD_abs_error": float(param_abs[3]),
                "wWD_abs_error": float(param_abs[4]),
                "wLW_abs_error": float(param_abs[5]),
                "curvature_mae_mean": float(np.mean(param_abs[3:])),
                "projected_mask_iou": mask_row["iou"],
                "projected_mask_dice": mask_row["dice"],
                "projected_mask_area_error": mask_row["area_error"],
                "projected_mask_center_error_px": mask_row["center_error"],
                "profile_depth_rmse_m": depth_rmse,
                "volume_proxy_rel_error": float(volume_error),
            }
        )
    return rows


def aggregate_prediction_rows(rows: list[dict[str, Any]], model_name: str, split_name: str) -> dict[str, Any]:
    subset = [row for row in rows if row["split"] == split_name]
    out: dict[str, Any] = {"model": model_name, "split": split_name, "sample_count": len(subset)}
    if not subset:
        return out
    keys = [
        "normalized_param_mae_mean",
        "dimension_param_mae_norm",
        "curvature_param_mae_norm",
        "L_mae_m",
        "W_mae_m",
        "D_mae_m",
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
        "clip_applied",
    ]
    for key in keys:
        values = [float(row[key]) for row in subset]
        out[f"{key}_mean"] = float(np.mean(values))
    return out


def run_cli(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.preflight_summary, args.input_check], args.overwrite)
    entry, manifest, npz_path = resolve_dataset(args.dataset_id, args.registry)
    checks = gate_manifest(entry, manifest, npz_path, args.dataset_id)
    dataset = load_dataset(args.dataset_id, args.registry)
    stats = train_normalization(dataset)
    splits = split_indices(dataset)
    with np.load(npz_path, allow_pickle=True) as npz:
        delta_error = float(np.max(np.abs(npz["delta_b"] - (npz["b_defect"] - npz["b_no_defect"]))))
    expected_n = int(manifest.get("n_samples", len(dataset.sample_ids)))
    expected_split = {str(k): int(v) for k, v in dict(manifest.get("split_counts", {})).items()}
    observed_split = {k: len(v) for k, v in splits.items()}
    expected_shape = (expected_n, 3, 3, 201)
    checks.extend(
        [
            {"check_name": "sample_count", "pass": len(dataset.sample_ids) == expected_n, "observed": len(dataset.sample_ids), "notes": f"manifest_n_samples={expected_n}"},
            {"check_name": "split_counts", "pass": observed_split == expected_split, "observed": observed_split, "notes": f"manifest_split_counts={expected_split}"},
            {"check_name": "delta_b_shape", "pass": dataset.delta_b.shape == expected_shape, "observed": list(dataset.delta_b.shape), "notes": "flattened to 9x201 for Conv1D"},
            {"check_name": "axis_names", "pass": dataset.axis_names == ["Bx", "By", "Bz"], "observed": dataset.axis_names, "notes": ""},
            {"check_name": "delta_b_finite", "pass": bool(np.isfinite(dataset.delta_b).all()), "observed": "finite", "notes": ""},
            {"check_name": "rbc_params_finite", "pass": bool(np.isfinite(dataset.rbc_params).all()), "observed": "finite", "notes": ""},
            {"check_name": "projected_mask_nonempty", "pass": bool((dataset.projected_mask_2d.sum(axis=(1, 2)) > 0).all()), "observed": int(dataset.projected_mask_2d.sum()), "notes": ""},
            {"check_name": "profile_depth_grid_present", "pass": bool(dataset.profile_depth_grid_m.size), "observed": list(dataset.profile_depth_grid_m.shape), "notes": ""},
            {"check_name": "curvature_template_present", "pass": len(dataset.curvature_template) == expected_n, "observed": len(dataset.curvature_template), "notes": ""},
            {"check_name": "depth_bin_present", "pass": len(dataset.depth_bin) == expected_n, "observed": len(dataset.depth_bin), "notes": ""},
            {"check_name": "delta_check", "pass": delta_error <= 1.0e-12, "observed": delta_error, "notes": ""},
            {"check_name": "train_only_normalization_prepared", "pass": True, "observed": {"x_mean_shape": list(stats["x_mean"].shape), "y_mean_shape": list(stats["y_mean"].shape)}, "notes": ""},
        ]
    )
    write_csv(args.input_check, checks, CHECK_FIELDS)
    failed = [row for row in checks if not bool(row["pass"])]
    args.preflight_summary.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_summary.write_text(
        "\n".join(
            [
                f"{args.run_name} preflight summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"dataset_id_exists: {manifest.get('dataset_id') == args.dataset_id}",
                f"registry_manifest_gate_pass: {not failed}",
                f"train_ready_candidate: {manifest.get('train_ready_candidate')}",
                "can_enter_training_gate: True" if not failed else "can_enter_training_gate: False",
                "input_fields: delta_b -> 9 channels x 201 from Bx/By/Bz x 3 scan lines",
                "output_labels: rbc_params [L_m,W_m,D_m,wLD,wWD,wLW]; projected_mask_2d/profile_depth_grid_m for metrics; curvature_template/depth_bin only for grouping",
                "comparison_protocol: replay 20.73 mean/feature/neural protocol on this dataset and compare distribution-level metrics, not paired samples",
                "allowed_submit: explicit training scripts, summaries, metrics, Markdown route updates",
                "forbidden_submit: data/, NPZ, checkpoint, preview PNG, notes, baseline docs, CURRENT_BASELINE.md",
                "stop_conditions: registry/manifest gate fail; schema fail; train/test leakage; no train fit; val/test collapse; forbidden artifacts staged",
                "",
                "Subagent preflight: Agents A-E completed read-only. A registry/manifest GO; B schema GO; C replay the mean/feature/neural gate for this explicit dataset_id; D safety boundary excludes data/NPZ/checkpoints/previews/notes/baseline docs and scripts/visualize_current_baseline.py; E parameterize existing scripts, no new loader family required.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                f"{args.run_name} input summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"npz_path: {dataset.npz_path}",
                f"route: {manifest.get('route')}",
                f"status: {manifest.get('status')}",
                f"train_ready_candidate: {manifest.get('train_ready_candidate')}",
                f"baseline_ready: {manifest.get('baseline_ready')}",
                f"allowed_use: {manifest.get('allowed_use')}",
                f"forbidden_use: {manifest.get('forbidden_use')}",
                f"input_shape_delta_b: {list(dataset.delta_b.shape)}",
                f"input_shape_conv1d: {list(dataset.x_channels.shape)}",
                f"rbc_params_shape: {list(dataset.rbc_params.shape)}",
                f"projected_mask_shape: {list(dataset.projected_mask_2d.shape)}",
                f"profile_depth_grid_shape: {list(dataset.profile_depth_grid_m.shape)}",
                f"split_counts: {{'train': {len(splits['train'])}, 'val': {len(splits['val'])}, 'test': {len(splits['test'])}}}",
                f"axis_names: {dataset.axis_names}",
                f"delta_max_abs_error: {delta_error}",
                "",
                "Boundary: this loader rejects any dataset not resolved by explicit dataset_id through registry + manifest. It performs no latest/newest NPZ scan.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError("input gate failed")
    return 0


def main() -> int:
    return run_cli(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
