#!/usr/bin/env python
"""Train Phase 14d: W_eff_area ensemble-risk flag validation.

Validates whether the Phase 14c global W_eff_area descriptor can carry an
EFFECTIVE risk flag when the per-sample uncertainty comes from a seeded
ExtraTrees ensemble (point = ensemble mean, risk score = ensemble std).

Forbidden: true/predicted family routing, mixture heads, W_m
residual/stacking, test threshold tuning, COMSOL, CURRENT_BASELINE.md update.
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
from train_phase14c_global_weff_area_descriptor import (  # noqa: E402
    rank_auc,
    scaled_conformal_quantile,
    spearman_corr,
    split_conformal_radius,
)


PHASE14C_METRICS = ROOT / "results/metrics/phase14c_global_weff_area_descriptor_metrics.json"
METRICS_JSON = ROOT / "results/metrics/phase14d_weff_area_ensemble_risk_flag_metrics.json"
SUMMARY_MD = ROOT / "results/summaries/phase14d_weff_area_ensemble_risk_flag_summary.md"
MANIFEST_CSV = ROOT / "results/manifests/phase14d_weff_area_ensemble_risk_flag_manifest.csv"
MODEL_PATH = ROOT / "results/models/phase14d_weff_area_ensemble_descriptor.joblib"

TARGET_NAME = "W_eff_area"
SINGLE_MODEL_CANDIDATE_NAME = "plain_W_eff_signal_width_features__W_eff_area"
ENSEMBLE_SEEDS = list(range(101, 113))
COVERAGE_TARGETS = (0.80, 0.90)
RISK_FLAG_VAL_QUANTILE = 0.75

GATE_MAE_MM = 1.50
GATE_COVERAGE_80 = 0.80
GATE_COVERAGE_90 = 0.90
GATE_AUC_PASS = 0.70
GATE_AUC_PARTIAL = 0.60
GATE_FLAG_DELTA_MM = 0.30
REPRODUCTION_STRICT_TOL_MM = 1.0e-6
REPRODUCTION_ACCEPT_TOL_MM = 5.0e-3


def ensemble_mean_std(member_predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over ensemble members; expects shape (n_members, n_samples)."""
    arr = np.asarray(member_predictions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"expected (n_members>=2, n_samples), got {arr.shape}")
    return arr.mean(axis=0), np.maximum(arr.std(axis=0), 1.0e-12)


def assess_ensemble_std_risk_flag(
    std_val_mm: np.ndarray,
    std_test_mm: np.ndarray,
    high_nuisance_test: np.ndarray,
    abs_error_test_mm: np.ndarray,
    flag_quantile: float = RISK_FLAG_VAL_QUANTILE,
) -> dict[str, Any]:
    """Assess ensemble std as the descriptor risk flag.

    The flag threshold is the val quantile of the ensemble std; test labels
    and test errors are never used to tune the threshold.
    """
    s_val = np.asarray(std_val_mm, dtype=np.float64).reshape(-1)
    s_test = np.asarray(std_test_mm, dtype=np.float64).reshape(-1)
    high = np.asarray(high_nuisance_test, dtype=bool).reshape(-1)
    err = np.asarray(abs_error_test_mm, dtype=np.float64).reshape(-1)
    threshold = float(np.quantile(s_val, float(flag_quantile)))
    flagged = s_test >= threshold
    large_error = err >= float(np.quantile(err, 0.75))

    flagged_mae = float(np.mean(err[flagged])) if np.any(flagged) else math.nan
    unflagged_mae = float(np.mean(err[~flagged])) if np.any(~flagged) else math.nan
    delta = (
        flagged_mae - unflagged_mae
        if np.isfinite(flagged_mae) and np.isfinite(unflagged_mae)
        else math.nan
    )
    enrichment = (
        flagged_mae / unflagged_mae
        if np.isfinite(flagged_mae) and np.isfinite(unflagged_mae) and unflagged_mae > 0.0
        else math.nan
    )
    return {
        "flag_source": "ensemble_std, threshold = val q%.2f" % flag_quantile,
        "threshold_mm": threshold,
        "test_flagged_count": int(np.sum(flagged)),
        "test_flagged_fraction": float(np.mean(flagged)),
        "spearman_std_vs_abs_error": float(spearman_corr(s_test, err)),
        "high_nuisance_auc": float(rank_auc(s_test, high)),
        "large_width_error_auc": float(rank_auc(s_test, large_error)),
        "flagged_mae_mm": flagged_mae,
        "unflagged_mae_mm": unflagged_mae,
        "flagged_minus_unflagged_mm": float(delta) if np.isfinite(delta) else math.nan,
        "error_enrichment_ratio": float(enrichment) if np.isfinite(enrichment) else math.nan,
        "high_nuisance_recall_at_flag": float(np.sum(flagged & high) / max(1, int(np.sum(high)))),
        "high_nuisance_precision_at_flag": float(np.sum(flagged & high) / max(1, int(np.sum(flagged)))),
        "test_flag_per_sample": [bool(v) for v in flagged.tolist()],
    }


def gate14d(
    test_mae_mm: float,
    coverage_80: float,
    coverage_90: float,
    high_nuisance_auc: float,
    large_error_auc: float,
    flagged_minus_unflagged_mm: float,
    audit_valid: bool,
) -> dict[str, Any]:
    """Gate 14d.

    PASS    : MAE <= 1.50 AND coverage80 >= 0.80 AND coverage90 >= 0.90
              AND (high-nuisance AUC >= 0.70 OR large-error AUC >= 0.70)
              AND flagged MAE >= unflagged MAE + 0.30 mm.
    PARTIAL : MAE and coverage hold, flag has positive signal
              (best AUC >= 0.60 and flagged error above unflagged) but misses
              a PASS condition (AUC in 0.60-0.70 band, or delta < 0.30 mm);
              descriptor reportable with the flag marked weak.
    FAIL    : best AUC < 0.60, or flagged samples are not worse than
              unflagged, or MAE/coverage/reproduction fail.
    """
    best_auc = max(
        high_nuisance_auc if np.isfinite(high_nuisance_auc) else -math.inf,
        large_error_auc if np.isfinite(large_error_auc) else -math.inf,
    )
    delta_ok = bool(np.isfinite(flagged_minus_unflagged_mm) and flagged_minus_unflagged_mm > 0.0)
    checks = {
        "mae_le_1p50": bool(test_mae_mm <= GATE_MAE_MM),
        "coverage_80_ge_0p80": bool(coverage_80 >= GATE_COVERAGE_80),
        "coverage_90_ge_0p90": bool(coverage_90 >= GATE_COVERAGE_90),
        "best_risk_auc": float(best_auc),
        "best_risk_auc_ge_0p70": bool(best_auc >= GATE_AUC_PASS),
        "best_risk_auc_ge_0p60": bool(best_auc >= GATE_AUC_PARTIAL),
        "flagged_above_unflagged": delta_ok,
        "flag_delta_ge_0p30_mm": bool(
            np.isfinite(flagged_minus_unflagged_mm) and flagged_minus_unflagged_mm >= GATE_FLAG_DELTA_MM
        ),
        "reproduction_valid": bool(audit_valid),
    }
    core_ok = checks["mae_le_1p50"] and checks["coverage_80_ge_0p80"] and checks["coverage_90_ge_0p90"]
    if not checks["reproduction_valid"]:
        verdict, rationale = "FAIL", "Phase14c reproduction mismatch invalidates the ensemble descriptor validation"
    elif not core_ok:
        verdict, rationale = "FAIL", "point MAE or conformal coverage misses the descriptor targets"
    elif not checks["best_risk_auc_ge_0p60"] or not checks["flagged_above_unflagged"]:
        verdict, rationale = (
            "FAIL",
            "ensemble-std risk flag is ineffective: best AUC below 0.60 or flagged samples not worse than unflagged",
        )
    elif checks["best_risk_auc_ge_0p70"] and checks["flag_delta_ge_0p30_mm"]:
        verdict, rationale = (
            "PASS",
            "descriptor MAE/coverage hold and the ensemble-std flag separates risky samples (AUC >= 0.70, flag delta >= 0.30 mm)",
        )
    else:
        verdict, rationale = (
            "PARTIAL",
            "descriptor MAE/coverage hold and the flag has positive signal, but it misses a PASS condition; flag must be reported as weak",
        )
    return {
        "verdict": verdict,
        "rationale": rationale,
        "checks": checks,
        "thresholds": {
            "mae_mm": GATE_MAE_MM,
            "coverage_80": GATE_COVERAGE_80,
            "coverage_90": GATE_COVERAGE_90,
            "auc_pass": GATE_AUC_PASS,
            "auc_partial": GATE_AUC_PARTIAL,
            "flag_delta_mm": GATE_FLAG_DELTA_MM,
        },
        "risk_flag_status": "effective" if verdict == "PASS" else "weak" if verdict == "PARTIAL" else "ineffective",
        "descriptor_reportable": verdict in {"PASS", "PARTIAL"},
        "baseline_update_allowed": False,
        "w_m_point_baseline_replaced": False,
    }


def build_manifest_rows(gate: dict[str, Any], values: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "phase": "phase14d_weff_area_ensemble_risk_flag",
            "descriptor_name": "global_W_eff_area_ensemble_descriptor",
            "target_name": TARGET_NAME,
            "ensemble_seeds_json": json_dumps(ENSEMBLE_SEEDS),
            "n_members": str(len(ENSEMBLE_SEEDS)),
            "point_estimate": "ensemble_mean",
            "risk_score": "ensemble_std",
            "uses_true_family": "False",
            "uses_predicted_family": "False",
            "uses_mixture_routing": "False",
            "w_m_residual_or_stacking_trained": "False",
            "test_threshold_tuning": "False",
            "ensemble_mean_test_mae_mm": "%.6f" % values["test_mae_mm"],
            "phase14c_single_test_mae_mm": "%.6f" % values["single_test_mae_mm"],
            "coverage_80": "%.6f" % values["coverage_80"],
            "coverage_90": "%.6f" % values["coverage_90"],
            "width_80_mm": "%.6f" % values["width_80_mm"],
            "width_90_mm": "%.6f" % values["width_90_mm"],
            "high_nuisance_auc": "%.6f" % values["high_nuisance_auc"],
            "large_width_error_auc": "%.6f" % values["large_width_error_auc"],
            "flagged_minus_unflagged_mm": "%.6f" % values["flagged_minus_unflagged_mm"],
            "risk_flag_status": gate["risk_flag_status"],
            "gate14d_verdict": gate["verdict"],
            "calibration_split": "original_v3_240_val",
            "final_eval_split": "original_v3_240_test",
            "comsol_run": "False",
            "real_data_used": "False",
            "current_baseline_update": "False",
            "old_v3_test_used_for_training_or_selection": "False",
            "phase11a_train_side_only": "True",
            "phase13c_samples_added_to_training": "False",
        }
    ]


def train_and_validate(args: argparse.Namespace) -> dict[str, Any]:
    output_paths = [args.metrics_json, args.summary_md, args.manifest_csv]
    if args.save_model:
        output_paths.append(args.model_path)
    check_no_overwrite(output_paths, args.overwrite)

    if not args.phase14c_metrics.exists():
        raise FileNotFoundError(f"Phase14c metrics missing: {args.phase14c_metrics}")
    stored_14c = json.loads(args.phase14c_metrics.read_text(encoding="utf-8"))
    ref_single_test = float(stored_14c["point_estimate"]["test_mae_mm"])
    ref_single_high = float(stored_14c["point_estimate"]["high_nuisance_test_mae_mm"])

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
    high_test = high_nuisance_mask[test_idx]
    families_test = np.asarray(old.curvature_template).astype(str)[test_idx]
    y_true_val = old_targets[TARGET_NAME][val_idx]
    y_true_test = old_targets[TARGET_NAME][test_idx]
    target_offset = W_EFF_TARGET_NAMES.index(TARGET_NAME)

    # --- reproduction anchor: exact Phase14c single global head ---
    single_candidate = fit_global_candidate(
        SINGLE_MODEL_CANDIDATE_NAME,
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
        notes="Phase14c single global head, refit as reproduction anchor only",
    )
    single_pred_test = predict_candidate(single_candidate, test_idx)
    single_test_mae = mae_mm(y_true_test, single_pred_test)
    single_high_mae = mae_mm(y_true_test[high_test], single_pred_test[high_test])

    reproduction: dict[str, Any] = {}
    audit_valid = True
    for key, (recomputed, stored_value) in {
        "single_global_test_mae_mm": (single_test_mae, ref_single_test),
        "single_global_high_nuisance_mae_mm": (single_high_mae, ref_single_high),
    }.items():
        diff = float(abs(float(recomputed) - float(stored_value)))
        status = (
            "exact" if diff <= REPRODUCTION_STRICT_TOL_MM
            else "acceptable" if diff <= REPRODUCTION_ACCEPT_TOL_MM
            else "mismatch"
        )
        audit_valid = audit_valid and status != "mismatch"
        reproduction[key] = {
            "recomputed_mm": float(recomputed),
            "phase14c_stored_mm": float(stored_value),
            "abs_diff_mm": diff,
            "status": status,
        }

    # --- seeded ensemble: point = mean, risk score = std ---
    x_train_full = np.vstack([old_all_width[train_idx], new_all_width])
    y_train_full = np.concatenate([old_targets[TARGET_NAME][train_idx], expansion_targets[TARGET_NAME]])
    members = []
    member_val = []
    member_test = []
    for seed in ENSEMBLE_SEEDS:
        member = extra_trees_regressor(random_state=seed).fit(x_train_full, y_train_full)
        members.append(member)
        member_val.append(np.asarray(member.predict(old_all_width[val_idx]), dtype=np.float64))
        member_test.append(np.asarray(member.predict(old_all_width[test_idx]), dtype=np.float64))
    pred_val, std_val = ensemble_mean_std(np.vstack(member_val))
    pred_test, std_test = ensemble_mean_std(np.vstack(member_test))
    test_mae = mae_mm(y_true_test, pred_test)
    val_mae = mae_mm(y_true_val, pred_val)
    high_test_mae = mae_mm(y_true_test[high_test], pred_test[high_test])

    # --- conformal on val only (primary: plain split; secondary: std-scaled) ---
    abs_resid_val = np.abs(pred_val - y_true_val)
    conformal: dict[str, Any] = {"calibration_split": "original_v3_240_val", "n_calibration": int(len(val_idx))}
    plain_radii: dict[str, float] = {}
    scaled_q: dict[str, float] = {}
    for coverage in COVERAGE_TARGETS:
        key = str(int(coverage * 100))
        radius = split_conformal_radius(abs_resid_val, coverage)
        plain_radii[key] = radius
        conformal.setdefault("plain_split_conformal", {})[key] = {
            "radius_mm": radius * 1000.0,
            **interval_metrics(y_true_test, pred_test - radius, pred_test + radius),
        }
        q = scaled_conformal_quantile(abs_resid_val, std_val, coverage)
        scaled_q[key] = q
        conformal.setdefault("ensemble_std_scaled_conformal", {})[key] = {
            "quantile_ratio": q,
            **interval_metrics(y_true_test, pred_test - q * std_test, pred_test + q * std_test),
        }
    conformal["primary_method"] = "plain_split_conformal"
    primary = conformal["plain_split_conformal"]
    coverage_80 = float(primary["80"]["coverage"])
    coverage_90 = float(primary["90"]["coverage"])

    # --- ensemble-std risk flag (val-thresholded) ---
    abs_err_test_mm = np.abs(pred_test - y_true_test) * 1000.0
    risk = assess_ensemble_std_risk_flag(std_val * 1000.0, std_test * 1000.0, high_test, abs_err_test_mm)

    gate = gate14d(
        test_mae_mm=test_mae,
        coverage_80=coverage_80,
        coverage_90=coverage_90,
        high_nuisance_auc=float(risk["high_nuisance_auc"]),
        large_error_auc=float(risk["large_width_error_auc"]),
        flagged_minus_unflagged_mm=float(risk["flagged_minus_unflagged_mm"]),
        audit_valid=audit_valid,
    )

    values = {
        "test_mae_mm": test_mae,
        "single_test_mae_mm": single_test_mae,
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "width_80_mm": float(primary["80"]["mean_width_mm"]),
        "width_90_mm": float(primary["90"]["mean_width_mm"]),
        "high_nuisance_auc": float(risk["high_nuisance_auc"]),
        "large_width_error_auc": float(risk["large_width_error_auc"]),
        "flagged_minus_unflagged_mm": float(risk["flagged_minus_unflagged_mm"]),
    }

    metrics = {
        "phase": "phase14d_weff_area_ensemble_risk_flag",
        "purpose": "validate an ensemble-std risk flag for the Phase14c global W_eff_area descriptor",
        "boundary": {
            "comsol_run": False,
            "real_data_used": False,
            "current_baseline_update": False,
            "uses_true_family": False,
            "uses_predicted_family": False,
            "uses_mixture_routing": False,
            "w_m_residual_or_stacking_trained": False,
            "w_m_point_baseline_replaced": False,
            "test_threshold_tuning": False,
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
        "ensemble": {
            "seeds": ENSEMBLE_SEEDS,
            "n_members": len(ENSEMBLE_SEEDS),
            "point_estimate": "ensemble_mean",
            "risk_score": "ensemble_std",
            "member_spec": "Phase14a EXTRA_TREES_PARAMS with per-member random_state",
        },
        "phase14c_reproduction": {
            "checks": reproduction,
            "strict_tolerance_mm": REPRODUCTION_STRICT_TOL_MM,
            "acceptable_tolerance_mm": REPRODUCTION_ACCEPT_TOL_MM,
            "audit_valid": audit_valid,
        },
        "point_estimate": {
            "ensemble_mean_validation_mae_mm": val_mae,
            "ensemble_mean_test_mae_mm": test_mae,
            "phase14c_single_test_mae_mm": single_test_mae,
            "ensemble_minus_single_mm": float(test_mae - single_test_mae),
            "high_nuisance_test_mae_mm": high_test_mae,
            "high_nuisance_test_count": int(np.sum(high_test)),
            "per_true_family_test_mae_mm": group_mae_mm(y_true_test, pred_test, families_test),
        },
        "conformal": conformal,
        "risk_flag": risk,
        "nuisance_stratification": {
            "high_nuisance_policy": high_nuisance_policy,
            "high_nuisance_test_count": int(np.sum(high_test)),
        },
        "gate14d": gate,
        "reporting_layer_recommendation": {
            "descriptor_reportable": gate["descriptor_reportable"],
            "risk_flag_status": gate["risk_flag_status"],
            "descriptor_fields": [
                "W_eff_area_pred_m (ensemble mean)",
                "W_eff_area_interval_80_m",
                "W_eff_area_interval_90_m",
                "ensemble_std_risk_flag",
            ],
            "note": "W_m remains the official nominal-width output; W_eff_area stays report-only",
        },
        "model_artifact": "",
    }

    write_csv(args.manifest_csv, build_manifest_rows(gate, values))

    if args.save_model:
        try:
            import joblib

            args.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "descriptor": "global_W_eff_area_ensemble_descriptor",
                    "ensemble_members": members,
                    "ensemble_seeds": ENSEMBLE_SEEDS,
                    "feature_names": all_width_names,
                    "plain_conformal_radius_m": plain_radii,
                    "scaled_conformal_quantile": scaled_q,
                    "risk_flag_threshold_mm": risk["threshold_mm"],
                    "metadata": {
                        "phase": metrics["phase"],
                        "target_name": TARGET_NAME,
                        "gate14d": gate,
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
    gate = metrics["gate14d"]
    point = metrics["point_estimate"]
    conf = metrics["conformal"]["plain_split_conformal"]
    scaled = metrics["conformal"]["ensemble_std_scaled_conformal"]
    risk = metrics["risk_flag"]
    repro = metrics["phase14c_reproduction"]
    lines = [
        "# Phase 14d W_eff_area Ensemble Risk Flag Summary",
        "",
        "- Gate 14d verdict: `%s`" % gate["verdict"],
        "- rationale: %s" % gate["rationale"],
        "- risk flag status: `%s`" % gate["risk_flag_status"],
        "- descriptor reportable: `%s`" % gate["descriptor_reportable"],
        "- Phase14c reproduction: audit_valid=`%s` (single head recomputed `%.6f` vs stored `%.6f`)"
        % (
            repro["audit_valid"],
            repro["checks"]["single_global_test_mae_mm"]["recomputed_mm"],
            repro["checks"]["single_global_test_mae_mm"]["phase14c_stored_mm"],
        ),
        "",
        "## Point Estimate (ensemble mean, %d members, seeds 101-112)" % metrics["ensemble"]["n_members"],
        "",
        "- ensemble mean test MAE: `%.6f mm` vs Phase14c single `%.6f mm` (delta `%+.6f`)"
        % (point["ensemble_mean_test_mae_mm"], point["phase14c_single_test_mae_mm"], point["ensemble_minus_single_mm"]),
        "- validation MAE: `%.6f mm`" % point["ensemble_mean_validation_mae_mm"],
        "- high-nuisance test MAE: `%.6f mm` (n=%d)"
        % (point["high_nuisance_test_mae_mm"], point["high_nuisance_test_count"]),
        "",
        "## Conformal Intervals (val-calibrated)",
        "",
        "| method | level | coverage | mean width mm |",
        "|---|---|---:|---:|",
        "| plain split | 80 | %.6f | %.6f |" % (conf["80"]["coverage"], conf["80"]["mean_width_mm"]),
        "| plain split | 90 | %.6f | %.6f |" % (conf["90"]["coverage"], conf["90"]["mean_width_mm"]),
        "| ensemble-std scaled | 80 | %.6f | %.6f |" % (scaled["80"]["coverage"], scaled["80"]["mean_width_mm"]),
        "| ensemble-std scaled | 90 | %.6f | %.6f |" % (scaled["90"]["coverage"], scaled["90"]["mean_width_mm"]),
        "",
        "## Ensemble-Std Risk Flag (threshold = val q%.2f = `%.6f mm`)" % (RISK_FLAG_VAL_QUANTILE, risk["threshold_mm"]),
        "",
        "- spearman(ensemble_std, |error|): `%.6f`" % risk["spearman_std_vs_abs_error"],
        "- high-nuisance AUC: `%.6f`; large-width-error AUC: `%.6f`"
        % (risk["high_nuisance_auc"], risk["large_width_error_auc"]),
        "- flagged: `%d/39`; flagged MAE `%.6f` vs unflagged `%.6f` mm (delta `%+.6f`, enrichment `%.6f`)"
        % (
            risk["test_flagged_count"],
            risk["flagged_mae_mm"],
            risk["unflagged_mae_mm"],
            risk["flagged_minus_unflagged_mm"],
            risk["error_enrichment_ratio"],
        ),
        "- high-nuisance recall/precision at flag: `%.6f` / `%.6f`"
        % (risk["high_nuisance_recall_at_flag"], risk["high_nuisance_precision_at_flag"]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase11a-npz", type=Path, default=PHASE11A_NPZ)
    parser.add_argument("--phase11a-execution-manifest", type=Path, default=PHASE11A_EXECUTION_MANIFEST)
    parser.add_argument("--phase14c-metrics", type=Path, default=PHASE14C_METRICS)
    parser.add_argument("--metrics-json", type=Path, default=METRICS_JSON)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    parser.add_argument("--manifest-csv", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = train_and_validate(args)
    print(
        json_dumps(
            {
                "gate14d": metrics["gate14d"]["verdict"],
                "ensemble_mean_test_mae_mm": metrics["point_estimate"]["ensemble_mean_test_mae_mm"],
                "coverage_80": metrics["conformal"]["plain_split_conformal"]["80"]["coverage"],
                "coverage_90": metrics["conformal"]["plain_split_conformal"]["90"]["coverage"],
                "high_nuisance_auc": metrics["risk_flag"]["high_nuisance_auc"],
                "large_width_error_auc": metrics["risk_flag"]["large_width_error_auc"],
                "flagged_minus_unflagged_mm": metrics["risk_flag"]["flagged_minus_unflagged_mm"],
                "metrics_json": str(args.metrics_json),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
