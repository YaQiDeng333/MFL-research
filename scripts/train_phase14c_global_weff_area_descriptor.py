#!/usr/bin/env python
"""Train Phase 14c: global W_eff_area reporting descriptor calibration.

Freezes the only inference-eligible width candidate from Phase 14b
(`global_W_eff_area_head`, all_141 + deterministic signal width features)
into a reporting descriptor:

  W_eff_area point estimate + val-calibrated conformal interval
  + a high-nuisance risk flag built from the W_m-vs-W_eff_area
  prediction discrepancy.

No true family, no predicted family, no mixture routing, no W_m
residual/stacking, no COMSOL, no CURRENT_BASELINE.md update.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_true_3d_rbc_pilot_dataset import (  # noqa: E402
    V3_240_DATASET_ID,
    load_dataset,
    split_indices,
)
from train_phase11a_ld_targeted_retraining import extract_all_141_features  # noqa: E402
from train_phase14a_observable_width_conditioned_models import (  # noqa: E402
    EXPECTED_V3_SPLITS,
    PHASE11A_EXECUTION_MANIFEST,
    PHASE11A_NPZ,
    W_EFF_TARGET_NAMES,
    build_high_nuisance_mask,
    check_no_overwrite,
    compute_width_targets_from_depth_maps,
    extra_trees_regressor,
    extract_signal_width_features,
    fit_global_candidate,
    group_mae_mm,
    interval_metrics,
    json_dumps,
    load_phase11a_width_pack,
    mae_mm,
    predict_candidate,
    write_csv,
)


PHASE14B_METRICS = ROOT / "results/metrics/phase14b_family_conditioned_weff_inference_realism_metrics.json"
METRICS_JSON = ROOT / "results/metrics/phase14c_global_weff_area_descriptor_metrics.json"
SUMMARY_MD = ROOT / "results/summaries/phase14c_global_weff_area_descriptor_summary.md"
MANIFEST_CSV = ROOT / "results/manifests/phase14c_global_weff_area_descriptor_manifest.csv"
MODEL_PATH = ROOT / "results/models/phase14c_global_weff_area_descriptor.joblib"

TARGET_NAME = "W_eff_area"
GLOBAL_CANDIDATE_NAME = "plain_W_eff_signal_width_features__W_eff_area"
W_M_CONTROL_CANDIDATE_NAME = "plain_W_m_head"
COVERAGE_TARGETS = (0.80, 0.90)
ENSEMBLE_SEEDS = list(range(1420, 1430))

GATE_MAE_MM = 1.50
GATE_COVERAGE_80 = 0.80
GATE_COVERAGE_90 = 0.90
RISK_AUC_THRESHOLD = 0.70
RISK_SPEARMAN_THRESHOLD = 0.30
RISK_ENRICHMENT_THRESHOLD = 1.50
RISK_FLAG_VAL_QUANTILE = 0.75
REPRODUCTION_STRICT_TOL_MM = 1.0e-6
REPRODUCTION_ACCEPT_TOL_MM = 5.0e-3


def split_conformal_radius(abs_residuals: np.ndarray, coverage: float) -> float:
    """Finite-sample split-conformal radius: k = ceil((n+1)*coverage) order statistic."""
    arr = np.sort(np.asarray(abs_residuals, dtype=np.float64).reshape(-1))
    n = arr.size
    if n == 0:
        return math.inf
    k = int(math.ceil((n + 1) * float(coverage)))
    if k > n:
        return math.inf
    return float(arr[k - 1])


def scaled_conformal_quantile(abs_residuals: np.ndarray, scales: np.ndarray, coverage: float) -> float:
    """Finite-sample quantile of residual/scale ratios for scaled conformal intervals."""
    residuals = np.asarray(abs_residuals, dtype=np.float64).reshape(-1)
    scale = np.maximum(np.asarray(scales, dtype=np.float64).reshape(-1), 1.0e-12)
    if residuals.shape != scale.shape:
        raise ValueError("residuals and scales must have the same shape")
    return split_conformal_radius(residuals / scale, coverage)


def rank_auc(scores: np.ndarray, positive_mask: np.ndarray) -> float:
    """Rank-sum AUC of `scores` for separating positive_mask from the rest."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive_mask, dtype=bool).reshape(-1)
    n_pos = int(np.sum(positive))
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # average ranks over ties
    unique_values, inverse, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    if unique_values.size != scores.size:
        cumulative = np.cumsum(counts)
        start = cumulative - counts + 1
        avg_rank_per_value = (start + cumulative) / 2.0
        ranks[order] = avg_rank_per_value[inverse]
    rank_sum_pos = float(np.sum(ranks[positive]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 3:
        return math.nan

    def _ranks(v: np.ndarray) -> np.ndarray:
        order = v.argsort(kind="mergesort")
        ranks = np.empty(v.size, dtype=np.float64)
        ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
        unique_values, inverse, counts = np.unique(v[order], return_inverse=True, return_counts=True)
        if unique_values.size != v.size:
            cumulative = np.cumsum(counts)
            start = cumulative - counts + 1
            avg = (start + cumulative) / 2.0
            ranks[order] = avg[inverse]
        return ranks

    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float(np.sum(ra**2)) * float(np.sum(rb**2)))
    if denom <= 0.0:
        return math.nan
    return float(np.sum(ra * rb) / denom)


def assess_discrepancy_risk_flag(
    discrepancy_val_mm: np.ndarray,
    discrepancy_test_mm: np.ndarray,
    high_nuisance_test: np.ndarray,
    abs_error_test_mm: np.ndarray,
    flag_quantile: float = RISK_FLAG_VAL_QUANTILE,
) -> dict[str, Any]:
    """Assess |W_m_pred - W_eff_area_pred| as a risk flag.

    Threshold comes from the val quantile only; test labels are never used to
    tune the flag. Effectiveness is pre-declared as ANY of:
      high-nuisance AUC >= 0.70, |spearman(discrepancy, |error|)| >= 0.30,
      flagged/unflagged W_eff error enrichment >= 1.5.
    """
    disc_val = np.asarray(discrepancy_val_mm, dtype=np.float64).reshape(-1)
    disc_test = np.asarray(discrepancy_test_mm, dtype=np.float64).reshape(-1)
    high = np.asarray(high_nuisance_test, dtype=bool).reshape(-1)
    err = np.asarray(abs_error_test_mm, dtype=np.float64).reshape(-1)
    threshold = float(np.quantile(disc_val, float(flag_quantile)))
    flagged = disc_test >= threshold

    large_error = err >= float(np.quantile(err, 0.75))
    auc_high = rank_auc(disc_test, high)
    auc_large_error = rank_auc(disc_test, large_error)
    sp = spearman_corr(disc_test, err)
    flagged_mean = float(np.mean(err[flagged])) if np.any(flagged) else math.nan
    unflagged_mean = float(np.mean(err[~flagged])) if np.any(~flagged) else math.nan
    enrichment = (
        flagged_mean / unflagged_mean
        if np.any(flagged) and np.any(~flagged) and unflagged_mean > 0.0
        else math.nan
    )

    def _safe_ge(value: float, bound: float) -> bool:
        return bool(np.isfinite(value) and value >= bound)

    criteria = {
        "high_nuisance_auc_ge_0p70": _safe_ge(auc_high, RISK_AUC_THRESHOLD),
        "abs_spearman_ge_0p30": bool(np.isfinite(sp) and abs(sp) >= RISK_SPEARMAN_THRESHOLD),
        "error_enrichment_ge_1p50": _safe_ge(enrichment, RISK_ENRICHMENT_THRESHOLD),
    }
    n_flag_high = int(np.sum(flagged & high))
    return {
        "flag_source": "abs(W_m_pred - W_eff_area_pred), threshold = val q%.2f" % flag_quantile,
        "threshold_mm": threshold,
        "test_flagged_count": int(np.sum(flagged)),
        "test_flagged_fraction": float(np.mean(flagged)),
        "high_nuisance_auc": float(auc_high),
        "large_width_error_auc": float(auc_large_error),
        "spearman_discrepancy_vs_abs_error": float(sp),
        "flagged_mean_abs_error_mm": flagged_mean,
        "unflagged_mean_abs_error_mm": unflagged_mean,
        "error_enrichment_ratio": float(enrichment) if np.isfinite(enrichment) else math.nan,
        "high_nuisance_recall_at_flag": float(n_flag_high / max(1, int(np.sum(high)))),
        "high_nuisance_precision_at_flag": float(n_flag_high / max(1, int(np.sum(flagged)))),
        "criteria": criteria,
        "effective": bool(any(criteria.values())),
        "test_flag_per_sample": [bool(v) for v in flagged.tolist()],
    }


def gate14c(
    test_mae_mm: float,
    coverage_80: float,
    coverage_90: float,
    risk_flag_effective: bool,
    audit_valid: bool,
) -> dict[str, Any]:
    """Gate 14c.

    PASS    : MAE <= 1.50 mm AND coverage80 >= 0.80 AND coverage90 >= 0.90
              AND the discrepancy risk flag is effective.
    PARTIAL : MAE and calibration pass but the risk flag is weak
              (descriptor usable as optional reporting field only).
    FAIL    : MAE above 1.50 mm, or interval calibration below target,
              or the Phase14b reproduction check failed.
    """
    checks = {
        "mae_le_1p50": bool(test_mae_mm <= GATE_MAE_MM),
        "coverage_80_ge_0p80": bool(coverage_80 >= GATE_COVERAGE_80),
        "coverage_90_ge_0p90": bool(coverage_90 >= GATE_COVERAGE_90),
        "risk_flag_effective": bool(risk_flag_effective),
        "reproduction_valid": bool(audit_valid),
    }
    if not checks["reproduction_valid"]:
        verdict, rationale = "FAIL", "Phase14b/14a reproduction mismatch invalidates the descriptor calibration"
    elif not checks["mae_le_1p50"]:
        verdict, rationale = "FAIL", "global W_eff_area head does not reproduce <=1.50 mm on the frozen test"
    elif not (checks["coverage_80_ge_0p80"] and checks["coverage_90_ge_0p90"]):
        verdict, rationale = "FAIL", "conformal interval calibration misses the 80/90 coverage targets"
    elif checks["risk_flag_effective"]:
        verdict, rationale = "PASS", "point MAE, 80/90 conformal coverage and the discrepancy risk flag all meet targets"
    else:
        verdict, rationale = (
            "PARTIAL",
            "point MAE and conformal calibration hold but the discrepancy risk flag is weak; optional descriptor only",
        )
    return {
        "verdict": verdict,
        "rationale": rationale,
        "checks": checks,
        "thresholds": {
            "mae_mm": GATE_MAE_MM,
            "coverage_80": GATE_COVERAGE_80,
            "coverage_90": GATE_COVERAGE_90,
            "risk_flag_criteria": {
                "high_nuisance_auc": RISK_AUC_THRESHOLD,
                "abs_spearman": RISK_SPEARMAN_THRESHOLD,
                "error_enrichment": RISK_ENRICHMENT_THRESHOLD,
            },
        },
        "descriptor_joins_reporting_layer": verdict == "PASS",
        "descriptor_optional_only": verdict == "PARTIAL",
        "baseline_update_allowed": False,
        "w_m_point_baseline_replaced": False,
    }


def build_manifest_rows(gate: dict[str, Any], summary_values: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "phase": "phase14c_global_weff_area_descriptor",
            "descriptor_name": "global_W_eff_area_descriptor",
            "point_model": GLOBAL_CANDIDATE_NAME,
            "target_name": TARGET_NAME,
            "uses_true_family": "False",
            "uses_predicted_family": "False",
            "uses_mixture_routing": "False",
            "w_m_residual_or_stacking_trained": "False",
            "test_mae_mm": "%.6f" % summary_values["test_mae_mm"],
            "coverage_80": "%.6f" % summary_values["coverage_80"],
            "coverage_90": "%.6f" % summary_values["coverage_90"],
            "width_80_mm": "%.6f" % summary_values["width_80_mm"],
            "width_90_mm": "%.6f" % summary_values["width_90_mm"],
            "high_nuisance_mae_mm": "%.6f" % summary_values["high_nuisance_mae_mm"],
            "risk_flag_effective": str(bool(summary_values["risk_flag_effective"])),
            "gate14c_verdict": gate["verdict"],
            "descriptor_joins_reporting_layer": str(bool(gate["descriptor_joins_reporting_layer"])),
            "calibration_split": "original_v3_240_val",
            "final_eval_split": "original_v3_240_test",
            "comsol_run": "False",
            "real_data_used": "False",
            "bz_only": "False",
            "current_baseline_update": "False",
            "old_v3_test_used_for_training_or_selection": "False",
            "phase11a_train_side_only": "True",
            "phase13c_samples_added_to_training": "False",
        }
    ]


def train_and_calibrate(args: argparse.Namespace) -> dict[str, Any]:
    output_paths = [args.metrics_json, args.summary_md, args.manifest_csv]
    if args.save_model:
        output_paths.append(args.model_path)
    check_no_overwrite(output_paths, args.overwrite)

    if not args.phase14b_metrics.exists():
        raise FileNotFoundError(f"Phase14b metrics missing: {args.phase14b_metrics}")
    stored_14b = json.loads(args.phase14b_metrics.read_text(encoding="utf-8"))
    ref_global_test = float(stored_14b["routing_evaluations"]["global_W_eff_area_head"]["test_mae_mm"])
    ref_global_high = float(stored_14b["routing_evaluations"]["global_W_eff_area_head"]["high_nuisance_mae_mm"])
    ref_w_m_test = float(stored_14b["w_m_control"]["test_mae_mm"])

    old = load_dataset(V3_240_DATASET_ID)
    splits = split_indices(old)
    split_counts = {name: int(len(idx)) for name, idx in splits.items()}
    if split_counts != EXPECTED_V3_SPLITS:
        raise RuntimeError(f"unexpected v3_240 split counts: {split_counts}")
    phase11a = load_phase11a_width_pack(args.phase11a_npz, args.phase11a_execution_manifest)
    if phase11a.success_count < 88:
        raise RuntimeError(f"Phase11a success count below threshold: {phase11a.success_count}/96")

    old_targets = compute_width_targets_from_depth_maps(old.profile_depth_map_xy_m, old.rbc_params)
    expansion_targets = compute_width_targets_from_depth_maps(phase11a.depth_maps, phase11a.rbc_params)
    old_all, all_names = extract_all_141_features(old.delta_b, old.sensor_x)
    new_all, new_all_names = extract_all_141_features(phase11a.delta_b, phase11a.sensor_x)
    if all_names != new_all_names:
        raise RuntimeError("all_141 feature schema mismatch between v3_240 and Phase11a")
    old_width, width_names = extract_signal_width_features(old.delta_b, old.sensor_x)
    new_width, new_width_names = extract_signal_width_features(phase11a.delta_b, phase11a.sensor_x)
    if width_names != new_width_names:
        raise RuntimeError("signal width feature schema mismatch between v3_240 and Phase11a")
    old_all_width = np.hstack([old_all, old_width])
    new_all_width = np.hstack([new_all, new_width])
    all_width_names = all_names + width_names

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    old_wWD = np.asarray(old.rbc_params[:, 4], dtype=np.float64)
    high_nuisance_mask, high_nuisance_policy = build_high_nuisance_mask(old_wWD, train_idx)
    high_val = high_nuisance_mask[val_idx]
    high_test = high_nuisance_mask[test_idx]
    families_test = np.asarray(old.curvature_template).astype(str)[test_idx]

    target_offset = W_EFF_TARGET_NAMES.index(TARGET_NAME)
    y_true_val = old_targets[TARGET_NAME][val_idx]
    y_true_test = old_targets[TARGET_NAME][test_idx]

    # --- point model: exact Phase14a/14b global head ---
    global_candidate = fit_global_candidate(
        GLOBAL_CANDIDATE_NAME,
        TARGET_NAME,
        old_all_width,
        new_all_width,
        old_targets,
        expansion_targets,
        train_idx,
        use_phase11a=True,
        feature_names=all_width_names,
        feature_schema="all_141_plus_fixed_signal_width_features",
        random_state=200 + target_offset,
        notes="global W_eff head with fixed deterministic signal-only width features",
    )
    pred_val = predict_candidate(global_candidate, val_idx)
    pred_test = predict_candidate(global_candidate, test_idx)
    test_mae = mae_mm(y_true_test, pred_test)
    val_mae = mae_mm(y_true_val, pred_val)
    high_test_mae = mae_mm(y_true_test[high_test], pred_test[high_test])

    # --- W_m control head (discrepancy source only; not a baseline restoration) ---
    w_m_control = fit_global_candidate(
        W_M_CONTROL_CANDIDATE_NAME,
        "W_m",
        old_all,
        new_all,
        old_targets,
        expansion_targets,
        train_idx,
        use_phase11a=False,
        feature_names=all_names,
        feature_schema="all_141",
        random_state=42,
        notes="control head: official all_141 signal features to nominal W_m on original train only",
    )
    w_m_pred_val = predict_candidate(w_m_control, val_idx)
    w_m_pred_test = predict_candidate(w_m_control, test_idx)
    w_m_test_mae = mae_mm(old_targets["W_m"][test_idx], w_m_pred_test)

    # --- reproduction checks against Phase14b stored values ---
    repro_rows = {
        "global_test_mae_mm": (test_mae, ref_global_test),
        "global_high_nuisance_mae_mm": (high_test_mae, ref_global_high),
        "w_m_control_test_mae_mm": (w_m_test_mae, ref_w_m_test),
    }
    reproduction: dict[str, Any] = {}
    audit_valid = True
    for key, (recomputed, stored_value) in repro_rows.items():
        diff = float(abs(float(recomputed) - float(stored_value)))
        status = (
            "exact" if diff <= REPRODUCTION_STRICT_TOL_MM
            else "acceptable" if diff <= REPRODUCTION_ACCEPT_TOL_MM
            else "mismatch"
        )
        audit_valid = audit_valid and status != "mismatch"
        reproduction[key] = {
            "recomputed_mm": float(recomputed),
            "phase14b_stored_mm": float(stored_value),
            "abs_diff_mm": diff,
            "status": status,
        }

    # --- seeded ensemble for per-sample uncertainty scale (uncertainty only) ---
    x_train_full = np.vstack([old_all_width[train_idx], new_all_width])
    y_train_full = np.concatenate([old_targets[TARGET_NAME][train_idx], expansion_targets[TARGET_NAME]])
    ensemble_preds_val = []
    ensemble_preds_test = []
    for seed in ENSEMBLE_SEEDS:
        member = extra_trees_regressor(random_state=seed).fit(x_train_full, y_train_full)
        ensemble_preds_val.append(np.asarray(member.predict(old_all_width[val_idx]), dtype=np.float64))
        ensemble_preds_test.append(np.asarray(member.predict(old_all_width[test_idx]), dtype=np.float64))
    ens_val = np.vstack(ensemble_preds_val)
    ens_test = np.vstack(ensemble_preds_test)
    scale_val = np.maximum(np.std(ens_val, axis=0), 1.0e-12)
    scale_test = np.maximum(np.std(ens_test, axis=0), 1.0e-12)
    ensemble_mean_test_mae = mae_mm(y_true_test, np.mean(ens_test, axis=0))

    # --- conformal calibration on val only ---
    abs_resid_val = np.abs(pred_val - y_true_val)
    conformal: dict[str, Any] = {"calibration_split": "original_v3_240_val", "n_calibration": int(len(val_idx))}
    plain_radii: dict[str, float] = {}
    for coverage in COVERAGE_TARGETS:
        key = str(int(coverage * 100))
        radius = split_conformal_radius(abs_resid_val, coverage)
        plain_radii[key] = radius
        lower = pred_test - radius
        upper = pred_test + radius
        conformal.setdefault("plain_split_conformal", {})[key] = {
            "radius_mm": radius * 1000.0,
            **interval_metrics(y_true_test, lower, upper),
        }
    scaled_q: dict[str, float] = {}
    for coverage in COVERAGE_TARGETS:
        key = str(int(coverage * 100))
        q = scaled_conformal_quantile(abs_resid_val, scale_val, coverage)
        scaled_q[key] = q
        lower = pred_test - q * scale_test
        upper = pred_test + q * scale_test
        conformal.setdefault("ensemble_std_scaled_conformal", {})[key] = {
            "quantile_ratio": q,
            **interval_metrics(y_true_test, lower, upper),
        }
    primary = conformal["plain_split_conformal"]
    coverage_80 = float(primary["80"]["coverage"])
    coverage_90 = float(primary["90"]["coverage"])
    conformal["primary_method"] = "plain_split_conformal"
    conformal["secondary_method_reported_only"] = "ensemble_std_scaled_conformal"

    # --- risk flag: |W_m_pred - W_eff_area_pred| (val-thresholded) ---
    disc_val_mm = np.abs(w_m_pred_val - pred_val) * 1000.0
    disc_test_mm = np.abs(w_m_pred_test - pred_test) * 1000.0
    abs_err_test_mm = np.abs(pred_test - y_true_test) * 1000.0
    risk = assess_discrepancy_risk_flag(disc_val_mm, disc_test_mm, high_test, abs_err_test_mm)
    ens_risk = {
        "signal": "ensemble_std_mm (reported only, not the gated flag)",
        "high_nuisance_auc": float(rank_auc(scale_test * 1000.0, high_test)),
        "spearman_vs_abs_error": float(spearman_corr(scale_test * 1000.0, abs_err_test_mm)),
    }

    gate = gate14c(
        test_mae_mm=test_mae,
        coverage_80=coverage_80,
        coverage_90=coverage_90,
        risk_flag_effective=bool(risk["effective"]),
        audit_valid=audit_valid,
    )

    summary_values = {
        "test_mae_mm": test_mae,
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "width_80_mm": float(primary["80"]["mean_width_mm"]),
        "width_90_mm": float(primary["90"]["mean_width_mm"]),
        "high_nuisance_mae_mm": high_test_mae,
        "risk_flag_effective": bool(risk["effective"]),
    }

    metrics = {
        "phase": "phase14c_global_weff_area_descriptor",
        "purpose": "freeze the Phase14b inference-eligible global W_eff_area head into a calibrated reporting descriptor",
        "boundary": {
            "comsol_run": False,
            "real_data_used": False,
            "bz_only": False,
            "current_baseline_update": False,
            "uses_true_family": False,
            "uses_predicted_family": False,
            "uses_mixture_routing": False,
            "w_m_residual_or_stacking_trained": False,
            "w_m_point_baseline_replaced": False,
            "old_v3_test_used_for_training_or_selection": False,
            "phase11a_train_side_only": True,
            "phase13c_samples_added_to_training_or_calibration": False,
        },
        "dataset": {
            "v3_dataset_id": V3_240_DATASET_ID,
            "v3_split_counts": split_counts,
            "phase11a_success_count": phase11a.success_count,
            "target_name": TARGET_NAME,
            "feature_schema": "all_141_plus_fixed_signal_width_features",
            "feature_count": len(all_width_names),
        },
        "phase14b_reproduction": {
            "checks": reproduction,
            "strict_tolerance_mm": REPRODUCTION_STRICT_TOL_MM,
            "acceptable_tolerance_mm": REPRODUCTION_ACCEPT_TOL_MM,
            "audit_valid": audit_valid,
        },
        "point_estimate": {
            "candidate": GLOBAL_CANDIDATE_NAME,
            "validation_mae_mm": val_mae,
            "test_mae_mm": test_mae,
            "high_nuisance_test_mae_mm": high_test_mae,
            "high_nuisance_test_count": int(np.sum(high_test)),
            "per_true_family_test_mae_mm": group_mae_mm(y_true_test, pred_test, families_test),
            "ensemble_mean_test_mae_mm_reported_only": ensemble_mean_test_mae,
            "w_m_control_test_mae_mm": w_m_test_mae,
            "w_eff_area_advantage_vs_w_m_control_mm": float(w_m_test_mae - test_mae),
        },
        "conformal": conformal,
        "risk_flag": {
            "primary_discrepancy_flag": risk,
            "secondary_ensemble_std_signal": ens_risk,
        },
        "nuisance_stratification": {
            "high_nuisance_policy": high_nuisance_policy,
            "high_nuisance_val_count": int(np.sum(high_val)),
            "high_nuisance_test_count": int(np.sum(high_test)),
        },
        "gate14c": gate,
        "reporting_layer_recommendation": {
            "join_posterior_reporting_layer": gate["descriptor_joins_reporting_layer"],
            "optional_descriptor_only": gate["descriptor_optional_only"],
            "descriptor_fields": [
                "W_eff_area_pred_m",
                "W_eff_area_interval_80_m",
                "W_eff_area_interval_90_m",
                "w_m_weff_discrepancy_risk_flag",
            ],
            "note": "W_m remains the official nominal-width output; W_eff_area is a report-only observable-width descriptor",
        },
        "model_artifact": "",
    }

    manifest_rows = build_manifest_rows(gate, summary_values)
    write_csv(args.manifest_csv, manifest_rows)

    if args.save_model:
        try:
            import joblib

            args.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "descriptor": "global_W_eff_area_descriptor",
                    "point_model": global_candidate.model,
                    "w_m_control_model": w_m_control.model,
                    "feature_names": all_width_names,
                    "w_m_feature_names": all_names,
                    "plain_conformal_radius_m": plain_radii,
                    "scaled_conformal_quantile": scaled_q,
                    "risk_flag_threshold_mm": risk["threshold_mm"],
                    "metadata": {
                        "phase": metrics["phase"],
                        "target_name": TARGET_NAME,
                        "gate14c": gate,
                    },
                },
                args.model_path,
            )
            metrics["model_artifact"] = str(args.model_path)
        except Exception as exc:  # pragma: no cover - optional artifact only
            metrics["model_artifact_error"] = str(exc)

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json_dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_summary(args.summary_md, metrics)
    return metrics


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    gate = metrics["gate14c"]
    point = metrics["point_estimate"]
    conf = metrics["conformal"]["plain_split_conformal"]
    scaled = metrics["conformal"]["ensemble_std_scaled_conformal"]
    risk = metrics["risk_flag"]["primary_discrepancy_flag"]
    repro = metrics["phase14b_reproduction"]
    lines = [
        "# Phase 14c Global W_eff_area Descriptor Summary",
        "",
        "- Gate 14c verdict: `%s`" % gate["verdict"],
        "- rationale: %s" % gate["rationale"],
        "- descriptor joins posterior/reporting layer: `%s`" % gate["descriptor_joins_reporting_layer"],
        "- optional descriptor only: `%s`" % gate["descriptor_optional_only"],
        "- Phase14b reproduction: audit_valid=`%s` (global test recomputed `%.6f` vs stored `%.6f`)"
        % (
            repro["audit_valid"],
            repro["checks"]["global_test_mae_mm"]["recomputed_mm"],
            repro["checks"]["global_test_mae_mm"]["phase14b_stored_mm"],
        ),
        "",
        "## Point Estimate (target = W_eff_area, frozen v3 test n=39)",
        "",
        "- test MAE: `%.6f mm` (target <= 1.50)" % point["test_mae_mm"],
        "- validation MAE: `%.6f mm`" % point["validation_mae_mm"],
        "- high-nuisance test MAE: `%.6f mm` (n=%d)"
        % (point["high_nuisance_test_mae_mm"], point["high_nuisance_test_count"]),
        "- W_m control test MAE: `%.6f mm`; W_eff_area advantage: `%.6f mm`"
        % (point["w_m_control_test_mae_mm"], point["w_eff_area_advantage_vs_w_m_control_mm"]),
        "",
        "## Conformal Intervals (val-calibrated, primary = plain split conformal)",
        "",
        "| method | level | coverage | mean width mm |",
        "|---|---|---:|---:|",
        "| plain split | 80 | %.6f | %.6f |" % (conf["80"]["coverage"], conf["80"]["mean_width_mm"]),
        "| plain split | 90 | %.6f | %.6f |" % (conf["90"]["coverage"], conf["90"]["mean_width_mm"]),
        "| ensemble-std scaled | 80 | %.6f | %.6f |" % (scaled["80"]["coverage"], scaled["80"]["mean_width_mm"]),
        "| ensemble-std scaled | 90 | %.6f | %.6f |" % (scaled["90"]["coverage"], scaled["90"]["mean_width_mm"]),
        "",
        "## Risk Flag (|W_m_pred - W_eff_area_pred|, val q%.2f threshold)" % RISK_FLAG_VAL_QUANTILE,
        "",
        "- threshold: `%.6f mm`; flagged on test: `%d/39`" % (risk["threshold_mm"], risk["test_flagged_count"]),
        "- high-nuisance AUC: `%.6f`; large-width-error AUC: `%.6f`"
        % (risk["high_nuisance_auc"], risk["large_width_error_auc"]),
        "- spearman(discrepancy, |error|): `%.6f`" % risk["spearman_discrepancy_vs_abs_error"],
        "- flagged vs unflagged mean |error|: `%.6f` / `%.6f` mm (enrichment `%.6f`)"
        % (risk["flagged_mean_abs_error_mm"], risk["unflagged_mean_abs_error_mm"], risk["error_enrichment_ratio"]),
        "- high-nuisance recall/precision at flag: `%.6f` / `%.6f`"
        % (risk["high_nuisance_recall_at_flag"], risk["high_nuisance_precision_at_flag"]),
        "- effective: `%s` (criteria: %s)" % (risk["effective"], risk["criteria"]),
        "",
        "## Boundary Confirmation",
        "",
        "- no COMSOL run",
        "- no real-data input",
        "- no Bz-only route",
        "- no CURRENT_BASELINE.md update",
        "- no true or predicted family routing; no mixture; no W_m residual/stacking",
        "- W_m remains the official nominal-width output; W_eff_area is report-only",
        "- conformal calibration and risk-flag threshold use val only; test frozen and final-only",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase11a-npz", type=Path, default=PHASE11A_NPZ)
    parser.add_argument("--phase11a-execution-manifest", type=Path, default=PHASE11A_EXECUTION_MANIFEST)
    parser.add_argument("--phase14b-metrics", type=Path, default=PHASE14B_METRICS)
    parser.add_argument("--metrics-json", type=Path, default=METRICS_JSON)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    parser.add_argument("--manifest-csv", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = train_and_calibrate(args)
    print(
        json_dumps(
            {
                "gate14c": metrics["gate14c"]["verdict"],
                "test_mae_mm": metrics["point_estimate"]["test_mae_mm"],
                "coverage_80": metrics["conformal"]["plain_split_conformal"]["80"]["coverage"],
                "coverage_90": metrics["conformal"]["plain_split_conformal"]["90"]["coverage"],
                "risk_flag_effective": metrics["risk_flag"]["primary_discrepancy_flag"]["effective"],
                "metrics_json": str(args.metrics_json),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
