# 公共依赖摘录：保留原算法定义，移除未被发布入口调用的历史实验流程。

from __future__ import annotations

import argparse

import csv

import json

import math

import sys

from dataclasses import dataclass

from pathlib import Path

from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from load_true_3d_rbc_pilot_dataset import (  # noqa: E402
    MASK_X_START_M,
    MASK_X_STOP_M,
    MASK_Y_START_M,
    MASK_Y_STOP_M,
    V3_240_DATASET_ID,
    load_dataset,
    split_indices,
)

from train_phase11a_ld_targeted_retraining import (  # noqa: E402
    EXPECTED_DELTA_B_SHAPE,
    as_delta_b,
    extract_all_141_features,
    parse_shape,
)

PHASE11A_NPZ = ROOT / "data/processed/phase11a_ld_targeted_expansion_delta_b.npz"

PHASE11A_EXECUTION_MANIFEST = ROOT / "results/manifests/phase11a_ld_targeted_execution_manifest.csv"

WIDTH_TARGET_NAMES = ["W_m", "W_eff_mask_50", "W_eff_mask_20", "W_eff_area", "W_eff_moment"]

W_EFF_TARGET_NAMES = ["W_eff_mask_50", "W_eff_mask_20", "W_eff_area", "W_eff_moment"]

EXPECTED_V3_SPLITS = {"train": 162, "val": 39, "test": 39}

EXTRA_TREES_PARAMS = {
    "n_estimators": 360,
    "random_state": 42,
    "min_samples_leaf": 1,
    "max_features": 0.75,
    "bootstrap": False,
    "n_jobs": -1,
}

@dataclass
class Phase11aWidthPack:
    sample_ids: np.ndarray
    delta_b: np.ndarray
    rbc_params: np.ndarray
    depth_maps: np.ndarray
    families: np.ndarray
    sensor_x: np.ndarray
    success_count: int
    failed_sample_ids: list[str]

@dataclass
class WidthCandidate:
    name: str
    target_name: str
    model: Any
    feature_names: list[str]
    feature_schema: str
    old_features: np.ndarray
    expansion_features: np.ndarray
    train_source: str
    uses_phase11a: bool
    family_conditioned: bool
    notes: str

def json_dumps(value: Any, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing files without --overwrite: {existing}")

def _as_params(params: np.ndarray) -> np.ndarray:
    arr = np.asarray(params, dtype=np.float64)
    return arr.reshape(arr.shape[0], -1)[:, :6]

def _mask_y_extent(mask: np.ndarray, dy: float) -> float:
    rows = np.any(mask, axis=1)
    return float(np.sum(rows) * dy)

def _mask_x_extent(mask: np.ndarray, dx: float) -> float:
    cols = np.any(mask, axis=0)
    return float(np.sum(cols) * dx)

def _moment_width(depth: np.ndarray, y_centers: np.ndarray) -> float:
    weights = np.maximum(np.asarray(depth, dtype=np.float64), 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    y = y_centers[:, None]
    mean_y = float(np.sum(weights * y) / total)
    var_y = float(np.sum(weights * (y - mean_y) ** 2) / total)
    return float(math.sqrt(max(0.0, 12.0 * var_y)))

def compute_width_targets_from_depth_maps(
    depth_maps: np.ndarray,
    rbc_params: np.ndarray,
    *,
    x_span_m: tuple[float, float] = (MASK_X_START_M, MASK_X_STOP_M),
    y_span_m: tuple[float, float] = (MASK_Y_START_M, MASK_Y_STOP_M),
) -> dict[str, np.ndarray]:
    """Build nominal and observable y-width targets from depth-map geometry."""
    depth = np.asarray(depth_maps, dtype=np.float64)
    if depth.ndim != 3:
        raise ValueError(f"expected depth maps with shape (N,Y,X), got {depth.shape}")
    params = _as_params(rbc_params)
    if depth.shape[0] != params.shape[0]:
        raise ValueError("depth map count and rbc_params count differ")
    n, ny, nx = depth.shape
    dx = (float(x_span_m[1]) - float(x_span_m[0])) / float(nx)
    dy = (float(y_span_m[1]) - float(y_span_m[0])) / float(ny)
    y_centers = float(y_span_m[0]) + (np.arange(ny, dtype=np.float64) + 0.5) * dy

    w_m = params[:, 1].astype(np.float64)
    d_m = np.maximum(params[:, 2].astype(np.float64), 1.0e-12)
    w50 = np.zeros(n, dtype=np.float64)
    w20 = np.zeros(n, dtype=np.float64)
    w_area = np.zeros(n, dtype=np.float64)
    w_moment = np.zeros(n, dtype=np.float64)
    for idx in range(n):
        mask50 = depth[idx] >= 0.5 * d_m[idx]
        mask20 = depth[idx] >= 0.2 * d_m[idx]
        w50[idx] = _mask_y_extent(mask50, dy)
        w20[idx] = _mask_y_extent(mask20, dy)
        area20 = float(np.sum(mask20) * dx * dy)
        l_eff = _mask_x_extent(mask20, dx)
        w_area[idx] = area20 / max(l_eff, 1.0e-12)
        w_moment[idx] = _moment_width(depth[idx], y_centers)
    return {
        "W_m": w_m,
        "W_eff_mask_50": w50,
        "W_eff_mask_20": w20,
        "W_eff_area": w_area,
        "W_eff_moment": w_moment,
    }

def assign_quantile_bins_from_train(
    values: np.ndarray,
    train_indices: np.ndarray,
    labels: tuple[str, str, str] = ("low", "mid", "high"),
) -> tuple[np.ndarray, list[float]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    train = np.asarray(train_indices, dtype=int)
    thresholds = [float(x) for x in np.quantile(arr[train], [1.0 / 3.0, 2.0 / 3.0])]
    codes = np.searchsorted(np.asarray(thresholds, dtype=np.float64), arr, side="left")
    return np.asarray([labels[int(code)] for code in codes], dtype=object), thresholds

def interval_metrics(y_true_m: np.ndarray, lower_m: np.ndarray, upper_m: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true_m, dtype=np.float64)
    lower = np.asarray(lower_m, dtype=np.float64)
    upper = np.asarray(upper_m, dtype=np.float64)
    covered = (truth >= lower) & (truth <= upper)
    width_mm = (upper - lower) * 1000.0
    return {
        "coverage": float(np.mean(covered)) if truth.size else math.nan,
        "median_width_mm": float(np.median(width_mm)) if width_mm.size else math.nan,
        "mean_width_mm": float(np.mean(width_mm)) if width_mm.size else math.nan,
    }

def extract_signal_width_features(delta_b: np.ndarray, sensor_x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    signal = as_delta_b(np.asarray(delta_b, dtype=np.float64))
    x = np.asarray(sensor_x, dtype=np.float64).reshape(-1)
    if signal.shape[-1] != x.size:
        raise ValueError("sensor_x length does not match delta_b x dimension")
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    axis_names = ["Bx", "By", "Bz"]
    line_names = ["ym", "yc", "yp"]
    features: list[np.ndarray] = []
    names: list[str] = []
    for axis_index, axis_name in enumerate(axis_names):
        for line_index, line_name in enumerate(line_names):
            curve = signal[:, axis_index, line_index, :]
            abs_curve = np.abs(curve)
            max_abs = np.max(abs_curve, axis=1)
            threshold = 0.5 * max_abs[:, None]
            above = abs_curve >= threshold
            half_width = np.sum(above, axis=1) * abs(dx)
            weights = abs_curve + 1.0e-18
            weight_sum = np.sum(weights, axis=1)
            x_mean = np.sum(weights * x[None, :], axis=1) / weight_sum
            x_var = np.sum(weights * (x[None, :] - x_mean[:, None]) ** 2, axis=1) / weight_sum
            energy_width = np.sqrt(np.maximum(0.0, 12.0 * x_var))
            grad = np.gradient(curve, x, axis=1)
            grad_abs = np.abs(grad)
            grad_sum = np.sum(grad_abs + 1.0e-18, axis=1)
            grad_mean = np.sum((grad_abs + 1.0e-18) * x[None, :], axis=1) / grad_sum
            grad_var = np.sum((grad_abs + 1.0e-18) * (x[None, :] - grad_mean[:, None]) ** 2, axis=1) / grad_sum
            grad_width = np.sqrt(np.maximum(0.0, 12.0 * grad_var))
            peak_x = x[np.argmax(abs_curve, axis=1)]
            for name, value in [
                ("abs_halfmax_x_width_m", half_width),
                ("abs_energy_x_width_m", energy_width),
                ("grad_energy_x_width_m", grad_width),
                ("abs_peak_x_m", peak_x),
            ]:
                features.append(np.asarray(value, dtype=np.float64))
                names.append(f"sig_width_{axis_name}_{line_name}_{name}")
    for axis_index, axis_name in enumerate(axis_names):
        side_mean = 0.5 * (
            np.mean(np.abs(signal[:, axis_index, 0, :]), axis=1)
            + np.mean(np.abs(signal[:, axis_index, 2, :]), axis=1)
        )
        center_mean = np.mean(np.abs(signal[:, axis_index, 1, :]), axis=1)
        features.append(center_mean / np.maximum(side_mean, 1.0e-18))
        names.append(f"sig_width_{axis_name}_center_side_mean_abs_ratio")
    matrix = np.column_stack(features).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise RuntimeError("non-finite deterministic signal width features")
    return matrix, names

def load_phase11a_width_pack(npz_path: Path, execution_manifest: Path) -> Phase11aWidthPack:
    if not npz_path.exists():
        raise FileNotFoundError(f"Phase11a NPZ missing: {npz_path}")
    if not execution_manifest.exists():
        raise FileNotFoundError(f"Phase11a execution manifest missing: {execution_manifest}")
    rows = read_csv(execution_manifest)
    pass_ids = [
        str(row.get("sample_id", ""))
        for row in rows
        if str(row.get("status", "")).strip().lower() in {"pass", "passed", "success"}
    ]
    failed_ids = sorted(
        str(row.get("sample_id", ""))
        for row in rows
        if str(row.get("status", "")).strip().lower() not in {"pass", "passed", "success"}
    )
    bad_shape_ids = [
        str(row.get("sample_id", ""))
        for row in rows
        if str(row.get("status", "")).strip().lower() in {"pass", "passed", "success"}
        and parse_shape(row.get("signal_shape") or row.get("expected_delta_b_shape_json") or "")
        not in (None, EXPECTED_DELTA_B_SHAPE)
    ]
    if bad_shape_ids:
        raise RuntimeError(f"successful Phase11a rows with bad signal shape: {bad_shape_ids[:8]}")

    with np.load(npz_path, allow_pickle=True) as npz:
        sample_ids_all = np.asarray(npz["sample_ids"]).astype(str)
        keep = np.asarray([sample_id in set(pass_ids) for sample_id in sample_ids_all], dtype=bool)
        missing = sorted(set(pass_ids).difference(set(sample_ids_all)))
        if missing:
            raise RuntimeError(f"Phase11a pass ids missing from NPZ: {missing[:8]}")
        delta_b = as_delta_b(np.asarray(npz["delta_b"], dtype=np.float64))[keep]
        rbc_params = _as_params(np.asarray(npz["rbc_params"], dtype=np.float64))[keep]
        depth_maps = np.asarray(npz["profile_depth_map_xy_m"], dtype=np.float64)[keep]
        families = np.asarray(npz["curvature_template"]).astype(str)[keep]
        sensor_x = np.asarray(npz["sensor_x"], dtype=np.float64)
        sample_ids = sample_ids_all[keep]
    if delta_b.shape[1:] != tuple(EXPECTED_DELTA_B_SHAPE):
        raise RuntimeError(f"bad Phase11a delta_b shape: {delta_b.shape}")
    if not np.isfinite(delta_b).all():
        raise RuntimeError("non-finite Phase11a delta_b values")
    return Phase11aWidthPack(
        sample_ids=sample_ids,
        delta_b=delta_b,
        rbc_params=rbc_params,
        depth_maps=depth_maps,
        families=families,
        sensor_x=sensor_x,
        success_count=int(delta_b.shape[0]),
        failed_sample_ids=failed_ids,
    )

def extra_trees_regressor(random_state: int = 42, max_features: float = 0.75) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    params = dict(EXTRA_TREES_PARAMS)
    params.update({"random_state": random_state, "max_features": max_features})
    return ExtraTreesRegressor(**params)

def fit_global_candidate(
    name: str,
    target_name: str,
    old_features: np.ndarray,
    expansion_features: np.ndarray,
    old_targets: dict[str, np.ndarray],
    expansion_targets: dict[str, np.ndarray],
    train_idx: np.ndarray,
    use_phase11a: bool,
    feature_names: list[str],
    feature_schema: str,
    random_state: int,
    notes: str,
) -> WidthCandidate:
    x_train = old_features[train_idx]
    y_train = old_targets[target_name][train_idx]
    train_source = "original_v3_train"
    if use_phase11a:
        x_train = np.vstack([x_train, expansion_features])
        y_train = np.concatenate([y_train, expansion_targets[target_name]])
        train_source = "original_v3_train_plus_phase11a_train_side"
    model = extra_trees_regressor(random_state=random_state).fit(x_train, y_train)
    return WidthCandidate(
        name=name,
        target_name=target_name,
        model=model,
        feature_names=feature_names,
        feature_schema=feature_schema,
        old_features=old_features,
        expansion_features=expansion_features,
        train_source=train_source,
        uses_phase11a=use_phase11a,
        family_conditioned=False,
        notes=notes,
    )

def predict_candidate(candidate: WidthCandidate, indices: np.ndarray, families: np.ndarray | None = None) -> np.ndarray:
    x = candidate.old_features[indices]
    if candidate.family_conditioned:
        if families is None:
            raise ValueError("families are required for family-conditioned prediction")
        model = candidate.model
        pred = np.asarray(model["fallback"].predict(x), dtype=np.float64)
        fam_values = np.asarray(families, dtype=str)[indices]
        for family, family_model in model["family_models"].items():
            mask = fam_values == family
            if np.any(mask):
                pred[mask] = np.asarray(family_model.predict(x[mask]), dtype=np.float64)
        return pred
    return np.asarray(candidate.model.predict(x), dtype=np.float64)

def mae_mm(y_true_m: np.ndarray, y_pred_m: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred_m) - np.asarray(y_true_m)) * 1000.0))

def group_mae_mm(y_true_m: np.ndarray, y_pred_m: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    groups = np.asarray(groups).astype(str)
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        out[str(group)] = mae_mm(np.asarray(y_true_m)[mask], np.asarray(y_pred_m)[mask])
    return out

def build_high_nuisance_mask(wWD_values: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    arr = np.asarray(wWD_values, dtype=np.float64)
    train = np.asarray(train_idx, dtype=int)
    center = float(np.median(arr[train]))
    abs_dev = np.abs(arr - center)
    threshold = float(np.quantile(abs_dev[train], 0.75))
    return abs_dev >= threshold, {"train_median_wWD": center, "train_abs_deviation_q75": threshold}
