#!/usr/bin/env python
"""Train Phase 11a L/D-targeted L/W/D-only candidates and audit old test non-regression."""

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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_true_3d_rbc_pilot_dataset import V3_240_DATASET_ID, load_dataset, split_indices  # noqa: E402


EXPANSION_NPZ = ROOT / "data/processed/phase11a_ld_targeted_expansion_delta_b.npz"
EXECUTION_MANIFEST = ROOT / "results/manifests/phase11a_ld_targeted_execution_manifest.csv"
METRICS_JSON = ROOT / "results/metrics/phase11a_ld_targeted_retraining_metrics.json"
SUMMARY_MD = ROOT / "results/summaries/phase11a_ld_targeted_retraining_summary.md"
MODEL_PATH = ROOT / "results/models/phase11a_ld_targeted_candidate.joblib"

AXIS_NAMES = ["Bx", "By", "Bz"]
LINE_NAMES = ["y-", "yc", "y+"]
TARGET_NAMES = ["L_m", "W_m", "D_m"]
EXPECTED_DELTA_B_SHAPE = [3, 3, 201]
OFFICIAL_BASELINE = {
    "candidate": "extra_trees+all_141+L_m/W_m/D_m",
    "L_mae_mm": 1.987,
    "W_mae_mm": 2.065,
    "D_mae_mm": 0.854,
    "mean_lwd_mae_mm": 1.635,
}
GATE_PASS_THRESHOLDS = {
    "L_mae_mm": 1.75,
    "W_mae_mm": 2.15,
    "D_mae_mm": 0.75,
    "mean_lwd_mae_mm": 1.55,
}
EXTRA_TREES_PARAMS = {
    "n_estimators": 600,
    "random_state": 42,
    "min_samples_leaf": 1,
    "max_features": 0.75,
    "bootstrap": False,
    "n_jobs": -1,
}


@dataclass
class ExpansionPack:
    sample_ids: np.ndarray
    delta_b: np.ndarray
    rbc_params: np.ndarray
    sensor_x: np.ndarray
    success_count: int
    failed_sample_ids: list[str]


@dataclass
class Candidate:
    name: str
    model: Any
    feature_names: list[str]
    x_old: np.ndarray
    x_expansion: np.ndarray
    notes: str


def json_dumps(value: Any, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def as_delta_b(delta_b: np.ndarray) -> np.ndarray:
    arr = np.asarray(delta_b, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[1:] == (3, 3, 201):
        return arr
    if arr.ndim == 3 and arr.shape[1:] == (9, 201):
        return arr.reshape(arr.shape[0], 3, 3, 201)
    raise ValueError(f"expected delta_b shape (N,3,3,201) or (N,9,201), got {arr.shape}")


def _safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    return np.asarray(num, dtype=np.float64) / np.maximum(np.asarray(den, dtype=np.float64), eps)


def _width_at_fraction(y_abs: np.ndarray, x: np.ndarray, fraction: float) -> np.ndarray:
    thresholds = np.max(y_abs, axis=1) * fraction
    widths = np.zeros(y_abs.shape[0], dtype=np.float64)
    for row_index, threshold in enumerate(thresholds):
        mask = y_abs[row_index] >= threshold
        if np.any(mask):
            xs = x[mask]
            widths[row_index] = float(xs[-1] - xs[0])
    return widths


def extract_all_141_features(delta_b: np.ndarray, sensor_x: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Extract the official 141 signal-derived feature contract."""
    arr = as_delta_b(delta_b)
    n = arr.shape[0]
    x = np.asarray(sensor_x if sensor_x is not None else np.linspace(-1.0, 1.0, arr.shape[-1]), dtype=np.float64)
    if x.shape != (arr.shape[-1],):
        raise ValueError(f"sensor_x shape mismatch: {x.shape} vs {arr.shape[-1]}")
    center_mask = x >= np.median(x)
    left_mask = x < 0.0
    right_mask = x > 0.0

    features: list[np.ndarray] = []
    names: list[str] = []
    p2p_by_axis_line = np.zeros((n, 3, 3), dtype=np.float64)

    for axis_index, axis_name in enumerate(AXIS_NAMES):
        for line_index, line_name in enumerate(LINE_NAMES):
            y = arr[:, axis_index, line_index, :]
            y_abs = np.abs(y)
            y_max = np.max(y, axis=1)
            y_min = np.min(y, axis=1)
            p2p = y_max - y_min
            p2p_by_axis_line[:, axis_index, line_index] = p2p
            arg_peak = np.argmax(y, axis=1)
            arg_valley = np.argmin(y, axis=1)
            x_peak = x[arg_peak]
            x_valley = x[arg_valley]
            grad = np.gradient(y, x, axis=1)
            abs_grad = np.abs(grad)
            grad_arg = np.argmax(abs_grad, axis=1)
            left_energy = np.mean(y[:, left_mask] ** 2, axis=1) if np.any(left_mask) else np.zeros(n)
            right_energy = np.mean(y[:, right_mask] ** 2, axis=1) if np.any(right_mask) else np.zeros(n)
            front_slope = np.mean(abs_grad[:, ~center_mask], axis=1)
            back_slope = np.mean(abs_grad[:, center_mask], axis=1)
            block = [
                ("max", y_max),
                ("min", y_min),
                ("p2p", p2p),
                ("mean", np.mean(y, axis=1)),
                ("mean_abs", np.mean(y_abs, axis=1)),
                ("x_peak", x_peak),
                ("x_valley", x_valley),
                ("pv_dist", np.abs(x_peak - x_valley)),
                ("eff_width", _width_at_fraction(y_abs, x, 0.50)),
                ("area", np.trapz(y, x=x, axis=1)),
                ("abs_area", np.trapz(y_abs, x=x, axis=1)),
                ("lr_energy_diff", _safe_ratio(left_energy - right_energy, left_energy + right_energy)),
                ("grad_max", np.max(abs_grad, axis=1)),
                ("x_grad_max", x[grad_arg]),
                ("fb_slope_ratio", _safe_ratio(front_slope, back_slope)),
            ]
            for feature_name, values in block:
                features.append(np.asarray(values, dtype=np.float64).reshape(n, 1))
                names.append(f"{axis_name}@{line_name}_{feature_name}")

    center = p2p_by_axis_line[:, :, 1]
    cross = [
        ("Bx_By_center_p2p_ratio", _safe_ratio(center[:, 0], center[:, 1])),
        ("Bx_Bz_center_p2p_ratio", _safe_ratio(center[:, 0], center[:, 2])),
        ("By_Bz_center_p2p_ratio", _safe_ratio(center[:, 1], center[:, 2])),
    ]
    for axis_index, axis_name in enumerate(AXIS_NAMES):
        side_mean = 0.5 * (p2p_by_axis_line[:, axis_index, 0] + p2p_by_axis_line[:, axis_index, 2])
        cross.append((f"{axis_name}_center_vs_side_p2p_contrast", _safe_ratio(center[:, axis_index] - side_mean, side_mean)))
    for name, values in cross:
        features.append(np.asarray(values, dtype=np.float64).reshape(n, 1))
        names.append(name)

    matrix = np.concatenate(features, axis=1)
    if matrix.shape[1] != 141 or len(names) != 141:
        raise RuntimeError(f"all_141 feature contract broken: matrix={matrix.shape}, names={len(names)}")
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), names


def extract_ld_specific_features(delta_b: np.ndarray, sensor_x: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Deterministic signal-only width/amplitude features for L/D candidates."""
    arr = as_delta_b(delta_b)
    n = arr.shape[0]
    x = np.asarray(sensor_x if sensor_x is not None else np.linspace(-1.0, 1.0, arr.shape[-1]), dtype=np.float64)
    features: list[np.ndarray] = []
    names: list[str] = []
    p2p_by_axis_line = np.ptp(arr, axis=3)
    abs_area_by_axis_line = np.trapz(np.abs(arr), x=x, axis=3)

    for axis_index, axis_name in enumerate(AXIS_NAMES):
        for line_index, line_name in enumerate(LINE_NAMES):
            y = arr[:, axis_index, line_index, :]
            y_abs = np.abs(y)
            grad = np.gradient(y, x, axis=1)
            abs_grad = np.abs(grad)
            blocks = [
                ("width25", _width_at_fraction(y_abs, x, 0.25)),
                ("width50", _width_at_fraction(y_abs, x, 0.50)),
                ("width75", _width_at_fraction(y_abs, x, 0.75)),
                ("energy", np.mean(y * y, axis=1)),
                ("grad_energy", np.mean(grad * grad, axis=1)),
                ("abs_grad_area", np.trapz(abs_grad, x=x, axis=1)),
                ("abs_area_norm", _safe_ratio(abs_area_by_axis_line[:, axis_index, line_index], p2p_by_axis_line[:, axis_index, line_index])),
            ]
            for feature_name, values in blocks:
                features.append(np.asarray(values, dtype=np.float64).reshape(n, 1))
                names.append(f"{axis_name}@{line_name}_{feature_name}")

    for axis_index, axis_name in enumerate(AXIS_NAMES):
        side_mean = 0.5 * (p2p_by_axis_line[:, axis_index, 0] + p2p_by_axis_line[:, axis_index, 2])
        features.append((_safe_ratio(p2p_by_axis_line[:, axis_index, 1], side_mean)).reshape(n, 1))
        names.append(f"{axis_name}_center_side_p2p_ratio_ld")
        features.append((abs_area_by_axis_line[:, axis_index, 1] - side_mean).reshape(n, 1))
        names.append(f"{axis_name}_center_abs_area_minus_side_p2p_ld")

    matrix = np.concatenate(features, axis=1)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), names


def build_training_index_policy(old_split: np.ndarray, expansion_count: int) -> dict[str, list[int]]:
    split = np.asarray(old_split).astype(str)
    original_train = np.where(split == "train")[0].astype(int).tolist()
    original_val = np.where(split == "val")[0].astype(int).tolist()
    original_test = np.where(split == "test")[0].astype(int).tolist()
    expansion_train = list(range(len(split), len(split) + int(expansion_count)))
    return {
        "original_train": original_train,
        "original_val": original_val,
        "original_test": original_test,
        "expansion_train": expansion_train,
        "augmented_train": original_train + expansion_train,
    }


def parse_shape(value: str) -> list[int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    except Exception:
        return None
    return None


def load_phase11a_expansion(npz_path: Path = EXPANSION_NPZ, execution_manifest: Path = EXECUTION_MANIFEST) -> ExpansionPack:
    if not npz_path.exists():
        raise FileNotFoundError(f"Phase 11a delta_b NPZ missing: {npz_path}")
    if not execution_manifest.exists():
        raise FileNotFoundError(f"Phase 11a execution manifest missing: {execution_manifest}")

    manifest_rows = read_csv(execution_manifest)
    failed_sample_ids = sorted(
        str(row.get("sample_id", ""))
        for row in manifest_rows
        if str(row.get("status", "")).strip().lower() not in {"pass", "passed", "success"}
    )
    passed_ids = [
        str(row.get("sample_id", ""))
        for row in manifest_rows
        if str(row.get("status", "")).strip().lower() in {"pass", "passed", "success"}
    ]
    bad_shape_ids = [
        str(row.get("sample_id", ""))
        for row in manifest_rows
        if str(row.get("status", "")).strip().lower() in {"pass", "passed", "success"}
        and parse_shape(row.get("signal_shape") or row.get("expected_delta_b_shape_json") or "") not in (None, EXPECTED_DELTA_B_SHAPE)
    ]
    if bad_shape_ids:
        raise RuntimeError(f"successful rows with bad signal shape: {bad_shape_ids[:8]}")

    with np.load(npz_path, allow_pickle=True) as npz:
        sample_ids = np.asarray(npz["sample_ids"]).astype(str)
        delta_b = as_delta_b(np.asarray(npz["delta_b"], dtype=np.float64))
        rbc_params = np.asarray(npz["rbc_params"], dtype=np.float64).reshape(delta_b.shape[0], 6)
        sensor_x = np.asarray(npz["sensor_x"], dtype=np.float64)

    if delta_b.shape[1:] != tuple(EXPECTED_DELTA_B_SHAPE):
        raise RuntimeError(f"bad expansion delta_b shape: {delta_b.shape}")
    if not np.isfinite(delta_b).all():
        raise RuntimeError("non-finite Bx/By/Bz values found in expansion NPZ")

    pass_set = set(passed_ids)
    keep = np.asarray([sample_id in pass_set for sample_id in sample_ids], dtype=bool)
    missing_pass_ids = sorted(pass_set.difference(set(sample_ids)))
    if missing_pass_ids:
        raise RuntimeError(f"execution manifest pass ids missing from NPZ: {missing_pass_ids[:8]}")
    sample_ids = sample_ids[keep]
    delta_b = delta_b[keep]
    rbc_params = rbc_params[keep]
    return ExpansionPack(
        sample_ids=sample_ids,
        delta_b=delta_b,
        rbc_params=rbc_params,
        sensor_x=sensor_x,
        success_count=int(delta_b.shape[0]),
        failed_sample_ids=failed_sample_ids,
    )


def extra_trees_model(random_state: int = 42, max_features: float = 0.75, min_samples_leaf: int = 1) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    params = dict(EXTRA_TREES_PARAMS)
    params.update({"random_state": random_state, "max_features": max_features, "min_samples_leaf": min_samples_leaf})
    return ExtraTreesRegressor(**params)


def fit_extra_trees_candidate(
    name: str,
    x_old: np.ndarray,
    x_expansion: np.ndarray,
    y_old: np.ndarray,
    y_expansion: np.ndarray,
    train_idx: np.ndarray,
    use_expansion: bool,
    feature_names: list[str],
    notes: str,
) -> Candidate:
    x_train = x_old[train_idx]
    y_train = y_old[train_idx]
    if use_expansion:
        x_train = np.vstack([x_train, x_expansion])
        y_train = np.vstack([y_train, y_expansion])
    model = extra_trees_model()
    model.fit(x_train, y_train)
    return Candidate(name=name, model=model, feature_names=feature_names, x_old=x_old, x_expansion=x_expansion, notes=notes)


def select_feature_indices(names: list[str], keywords: tuple[str, ...]) -> list[int]:
    indices = [index for index, name in enumerate(names) if any(keyword in name for keyword in keywords)]
    if not indices:
        raise RuntimeError(f"no feature indices matched keywords: {keywords}")
    return indices


def fit_per_target_candidate(
    x_old: np.ndarray,
    x_expansion: np.ndarray,
    y_old: np.ndarray,
    y_expansion: np.ndarray,
    train_idx: np.ndarray,
    feature_names: list[str],
) -> Candidate:
    x_train = np.vstack([x_old[train_idx], x_expansion])
    y_train = np.vstack([y_old[train_idx], y_expansion])
    l_indices = select_feature_indices(feature_names, ("eff_width", "width", "pv_dist", "x_peak", "x_valley", "center_side", "center_vs_side"))
    d_indices = select_feature_indices(feature_names, ("max", "min", "p2p", "mean_abs", "area", "energy", "grad", "abs_area"))
    w_indices = list(range(x_old.shape[1]))
    models = {
        "L_m": extra_trees_model(random_state=52, max_features=0.90, min_samples_leaf=1).fit(x_train[:, l_indices], y_train[:, 0]),
        "W_m": extra_trees_model(random_state=42, max_features=0.75, min_samples_leaf=1).fit(x_train[:, w_indices], y_train[:, 1]),
        "D_m": extra_trees_model(random_state=62, max_features=0.90, min_samples_leaf=1).fit(x_train[:, d_indices], y_train[:, 2]),
    }
    model = {"models": models, "indices": {"L_m": l_indices, "W_m": w_indices, "D_m": d_indices}}
    return Candidate(
        name="per_target_L_width_D_amplitude_W_shared",
        model=model,
        feature_names=feature_names,
        x_old=x_old,
        x_expansion=x_expansion,
        notes="L uses x-width features; D uses amplitude/energy/gradient features; W remains shared all-feature tree target",
    )


def predict_candidate(candidate: Candidate, indices: np.ndarray) -> np.ndarray:
    x = candidate.x_old[indices]
    if isinstance(candidate.model, dict):
        models = candidate.model["models"]
        idx = candidate.model["indices"]
        return np.column_stack(
            [
                models["L_m"].predict(x[:, idx["L_m"]]),
                models["W_m"].predict(x[:, idx["W_m"]]),
                models["D_m"].predict(x[:, idx["D_m"]]),
            ]
        )
    return np.asarray(candidate.model.predict(x), dtype=np.float64)


def mae_metrics(y_true_m: np.ndarray, y_pred_m: np.ndarray) -> dict[str, float]:
    abs_mm = np.abs(np.asarray(y_pred_m, dtype=np.float64) - np.asarray(y_true_m, dtype=np.float64)) * 1000.0
    return {
        "L_mae_mm": float(np.mean(abs_mm[:, 0])),
        "W_mae_mm": float(np.mean(abs_mm[:, 1])),
        "D_mae_mm": float(np.mean(abs_mm[:, 2])),
        "mean_lwd_mae_mm": float(np.mean(abs_mm)),
        "LD_mean_mae_mm": float(np.mean(abs_mm[:, [0, 2]])),
    }


def per_sample_abs_mm(y_true_m: np.ndarray, y_pred_m: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y_pred_m, dtype=np.float64) - np.asarray(y_true_m, dtype=np.float64)) * 1000.0


def choose_candidate_by_validation(
    candidate_metrics: dict[str, dict[str, float]],
    reference_name: str,
    selectable_names: list[str],
) -> str:
    reference_w = candidate_metrics[reference_name]["W_mae_mm"]
    selectable = {name: candidate_metrics[name] for name in selectable_names}
    eligible = {
        name: metrics
        for name, metrics in selectable.items()
        if metrics["W_mae_mm"] <= reference_w + 0.10
    }
    if not eligible:
        eligible = selectable
    return min(eligible, key=lambda name: (eligible[name]["LD_mean_mae_mm"], eligible[name]["mean_lwd_mae_mm"]))


def gate11a(final_test: dict[str, float], leakage: bool) -> tuple[str, str]:
    if leakage:
        return "FAIL", "test leakage or forbidden label leakage detected"
    l_gain = OFFICIAL_BASELINE["L_mae_mm"] - final_test["L_mae_mm"]
    d_gain = OFFICIAL_BASELINE["D_mae_mm"] - final_test["D_mae_mm"]
    w_degradation = final_test["W_mae_mm"] - OFFICIAL_BASELINE["W_mae_mm"]
    pass_gate = (
        final_test["L_mae_mm"] <= GATE_PASS_THRESHOLDS["L_mae_mm"]
        and final_test["D_mae_mm"] <= GATE_PASS_THRESHOLDS["D_mae_mm"]
        and final_test["W_mae_mm"] <= GATE_PASS_THRESHOLDS["W_mae_mm"]
        and final_test["mean_lwd_mae_mm"] <= GATE_PASS_THRESHOLDS["mean_lwd_mae_mm"]
    )
    if pass_gate:
        return "PASS", "L/D and mean non-regression thresholds met with W guard satisfied"
    partial_gate = (l_gain >= 0.10 or d_gain >= 0.10) and w_degradation <= 0.10
    if partial_gate:
        return "PARTIAL", "at least one of L/D improved by >=0.10 mm and W degradation stayed <=0.10 mm, but PASS thresholds were not all met"
    return "FAIL", "L/D did not improve materially or W degraded beyond the guard"


def high_error_reduction(reference_abs_mm: np.ndarray, final_abs_mm: np.ndarray) -> dict[str, Any]:
    ld_reference = np.mean(reference_abs_mm[:, [0, 2]], axis=1)
    threshold = float(np.percentile(ld_reference, 75.0))
    high = ld_reference >= threshold
    final_ld = np.mean(final_abs_mm[:, [0, 2]], axis=1)
    return {
        "threshold_reference_LD_abs_error_mm": threshold,
        "high_error_count": int(np.sum(high)),
        "reduced_count": int(np.sum(final_ld[high] < ld_reference[high])),
        "reference_high_error_mean_LD_mm": float(np.mean(ld_reference[high])) if np.any(high) else math.nan,
        "final_high_error_mean_LD_mm": float(np.mean(final_ld[high])) if np.any(high) else math.nan,
    }


def save_model(candidate: Candidate, model_path: Path, metadata: dict[str, Any]) -> bool:
    try:
        import joblib
    except Exception:
        return False
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"candidate": candidate, "metadata": metadata}, model_path)
    return True


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    check_no_overwrite([args.metrics_json, args.summary_md] + ([args.model_path] if args.save_model else []), args.overwrite)

    old = load_dataset(V3_240_DATASET_ID)
    old_splits = split_indices(old)
    train_idx = old_splits["train"]
    val_idx = old_splits["val"]
    test_idx = old_splits["test"]
    if (len(train_idx), len(val_idx), len(test_idx)) != (162, 39, 39):
        raise RuntimeError(f"unexpected v3_240 split counts: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    if tuple(old.delta_b.shape) != (240, 3, 3, 201):
        raise RuntimeError(f"unexpected old delta_b shape: {old.delta_b.shape}")
    if float(old.sensor_z_m) != 0.008:
        raise RuntimeError(f"unexpected old sensor_z_m: {old.sensor_z_m}")

    expansion = load_phase11a_expansion(args.expansion_npz, args.execution_manifest)
    gate2_pass = expansion.success_count >= 88
    if not gate2_pass:
        raise RuntimeError(f"Gate 2 failed: success_count={expansion.success_count}/96, threshold=88")

    old_all, all_names = extract_all_141_features(old.delta_b, old.sensor_x)
    new_all, new_all_names = extract_all_141_features(expansion.delta_b, expansion.sensor_x)
    if new_all_names != all_names:
        raise RuntimeError("all_141 feature names mismatch between old and Phase11a expansion")
    old_ld, ld_names = extract_ld_specific_features(old.delta_b, old.sensor_x)
    new_ld, new_ld_names = extract_ld_specific_features(expansion.delta_b, expansion.sensor_x)
    if new_ld_names != ld_names:
        raise RuntimeError("L/D-specific feature names mismatch between old and Phase11a expansion")

    old_y = np.asarray(old.rbc_params[:, :3], dtype=np.float64)
    new_y = np.asarray(expansion.rbc_params[:, :3], dtype=np.float64)
    x_old_all_ld = np.hstack([old_all, old_ld])
    x_new_all_ld = np.hstack([new_all, new_ld])
    all_ld_names = all_names + ld_names

    candidates = [
        fit_extra_trees_candidate(
            "official_original_train_only",
            old_all,
            new_all,
            old_y,
            new_y,
            train_idx,
            use_expansion=False,
            feature_names=all_names,
            notes="official spec ExtraTrees on original train only; reference reproduction candidate",
        ),
        fit_extra_trees_candidate(
            "official_train_plus_phase11a",
            old_all,
            new_all,
            old_y,
            new_y,
            train_idx,
            use_expansion=True,
            feature_names=all_names,
            notes="official ExtraTrees/all_141 trained on original train plus successful Phase11a train-only expansion",
        ),
        fit_extra_trees_candidate(
            "extra_trees_all_141_plus_ld_features",
            x_old_all_ld,
            x_new_all_ld,
            old_y,
            new_y,
            train_idx,
            use_expansion=True,
            feature_names=all_ld_names,
            notes="ExtraTrees with deterministic signal-only L/D-specific features added to all_141",
        ),
        fit_per_target_candidate(
            x_old_all_ld,
            x_new_all_ld,
            old_y,
            new_y,
            train_idx,
            all_ld_names,
        ),
    ]

    val_metrics: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        pred_val = predict_candidate(candidate, val_idx)
        val_metrics[candidate.name] = mae_metrics(old_y[val_idx], pred_val)
    selectable_names = [candidate.name for candidate in candidates if candidate.name != "official_original_train_only"]
    selected_name = choose_candidate_by_validation(val_metrics, "official_original_train_only", selectable_names)
    selected = next(candidate for candidate in candidates if candidate.name == selected_name)

    final_pred_test = predict_candidate(selected, test_idx)
    final_test = mae_metrics(old_y[test_idx], final_pred_test)
    reference_candidate = next(candidate for candidate in candidates if candidate.name == "official_original_train_only")
    reference_pred_test = predict_candidate(reference_candidate, test_idx)
    reference_test = mae_metrics(old_y[test_idx], reference_pred_test)

    final_abs = per_sample_abs_mm(old_y[test_idx], final_pred_test)
    reference_abs = per_sample_abs_mm(old_y[test_idx], reference_pred_test)
    high_error = high_error_reduction(reference_abs, final_abs)

    leakage = False
    policy = build_training_index_policy(old.split, expansion.success_count)
    if set(policy["original_test"]).intersection(policy["augmented_train"]):
        leakage = True
    forbidden_feature_label_terms = {"wLD", "wWD", "wLW", "skew", "flatness", "slope", "mode", "boxiness"}
    if forbidden_feature_label_terms.intersection(all_ld_names):
        leakage = True
    verdict, rationale = gate11a(final_test, leakage)

    model_saved = save_model(
        selected,
        args.model_path,
        {
            "selected_candidate": selected.name,
            "feature_count": len(selected.feature_names),
            "target_names": TARGET_NAMES,
            "official_baseline": OFFICIAL_BASELINE,
            "gate11a_verdict": verdict,
        },
    ) if args.save_model else False

    metrics = {
        "phase": "phase11a_ld_targeted_expansion_and_retraining",
        "boundary": {
            "real_data_used": False,
            "measurement_redesign": False,
            "sensor_z_changed": False,
            "scan_lines_changed": False,
            "x_grid_changed": False,
            "shape_labels_trained": False,
            "excluded_labels_trained": False,
            "phase8c_8d_small_sample_training": False,
            "current_baseline_modified": False,
            "w_targeted_expansion": False,
            "w_decorrelated_stage2": False,
            "old_test_used_for_training_or_selection": False,
        },
        "dataset": {
            "context": "v3_240",
            "old_dataset_id": V3_240_DATASET_ID,
            "old_delta_b_shape": list(old.delta_b.shape),
            "old_split_counts": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "expansion_success_count": expansion.success_count,
            "expansion_failed_sample_ids": expansion.failed_sample_ids,
            "expansion_delta_b_shape": list(expansion.delta_b.shape),
            "expected_per_sample_delta_b_shape": EXPECTED_DELTA_B_SHAPE,
            "gate2_pass": gate2_pass,
        },
        "features": {
            "official_feature_set": "all_141",
            "all_141_feature_count": len(all_names),
            "ld_specific_feature_count": len(ld_names),
            "deterministic_signal_only": True,
            "forbidden_label_feature_terms": sorted(forbidden_feature_label_terms),
        },
        "training_policy": {
            "target_names": TARGET_NAMES,
            "original_train_count": len(policy["original_train"]),
            "original_val_count": len(policy["original_val"]),
            "original_test_count": len(policy["original_test"]),
            "expansion_train_count": len(policy["expansion_train"]),
            "augmented_train_count": len(policy["augmented_train"]),
            "original_test_frozen": True,
            "selection_split": "original_val",
            "final_test_split": "original_v3_240_test",
        },
        "official_baseline_reference": OFFICIAL_BASELINE,
        "candidate_validation_metrics": val_metrics,
        "selected_candidate": selected.name,
        "reference_candidate_selectable_for_final": False,
        "selection_pool": selectable_names,
        "selected_candidate_notes": selected.notes,
        "reference_reproduction_test_metrics": reference_test,
        "final_old_test_metrics": final_test,
        "delta_vs_official_baseline_mm": {
            "L_mae_change_mm": float(final_test["L_mae_mm"] - OFFICIAL_BASELINE["L_mae_mm"]),
            "W_mae_change_mm": float(final_test["W_mae_mm"] - OFFICIAL_BASELINE["W_mae_mm"]),
            "D_mae_change_mm": float(final_test["D_mae_mm"] - OFFICIAL_BASELINE["D_mae_mm"]),
            "mean_lwd_mae_change_mm": float(final_test["mean_lwd_mae_mm"] - OFFICIAL_BASELINE["mean_lwd_mae_mm"]),
        },
        "high_error_ld_reduction": high_error,
        "gate11a": {
            "verdict": verdict,
            "rationale": rationale,
            "thresholds": GATE_PASS_THRESHOLDS,
            "baseline_transition_audit_worth_entering": verdict == "PASS",
        },
        "model_artifact": str(args.model_path) if model_saved else "",
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json_dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_summary(args.summary_md, metrics)
    return metrics


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    final = metrics["final_old_test_metrics"]
    delta = metrics["delta_vs_official_baseline_mm"]
    gate = metrics["gate11a"]
    lines = [
        "# Phase 11a L/D-Targeted Retraining Summary",
        "",
        f"-新增 COMSOL 成功数: `{metrics['dataset']['expansion_success_count']}/96`",
        f"- best candidate: `{metrics['selected_candidate']}`",
        f"- old test L MAE: `{final['L_mae_mm']:.6f} mm` (change `{delta['L_mae_change_mm']:.6f} mm` vs 1.987)",
        f"- old test W MAE: `{final['W_mae_mm']:.6f} mm` (change `{delta['W_mae_change_mm']:.6f} mm` vs 2.065)",
        f"- old test D MAE: `{final['D_mae_mm']:.6f} mm` (change `{delta['D_mae_change_mm']:.6f} mm` vs 0.854)",
        f"- old test L/W/D mean MAE: `{final['mean_lwd_mae_mm']:.6f} mm` (change `{delta['mean_lwd_mae_change_mm']:.6f} mm` vs 1.635)",
        f"- Gate 11a verdict: `{gate['verdict']}`",
        f"- baseline-transition audit worth entering: `{gate['baseline_transition_audit_worth_entering']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-npz", type=Path, default=EXPANSION_NPZ)
    parser.add_argument("--execution-manifest", type=Path, default=EXECUTION_MANIFEST)
    parser.add_argument("--metrics-json", type=Path, default=METRICS_JSON)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--no-save-model", dest="save_model", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = train_and_evaluate(args)
    print(
        json_dumps(
            {
                "selected_candidate": metrics["selected_candidate"],
                "gate11a": metrics["gate11a"]["verdict"],
                "expansion_success_count": metrics["dataset"]["expansion_success_count"],
                "metrics_json": str(args.metrics_json),
                "summary_md": str(args.summary_md),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
