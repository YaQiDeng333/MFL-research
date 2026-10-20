#!/usr/bin/env python
"""Build Phase 14e: W_eff_area descriptor reporting-layer integration.

Attaches the Phase 14d (PASS) W_eff_area ensemble descriptor to per-sample
reports over the frozen v3_240 test split, alongside the preserved official
L/W/D baseline W_m. Pure reload + predict + assemble: no model training,
no COMSOL, no CURRENT_BASELINE.md change.

W_eff_area is a report-only observable-width descriptor. It never replaces
the official nominal W_m and never enters the L/W/D mean MAE.
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
    build_high_nuisance_mask,
    check_no_overwrite,
    compute_width_targets_from_depth_maps,
    extract_signal_width_features,
    json_dumps,
    write_csv,
)


PHASE14D_MODEL = ROOT / "results/models/phase14d_weff_area_ensemble_descriptor.joblib"
PHASE14D_METRICS = ROOT / "results/metrics/phase14d_weff_area_ensemble_risk_flag_metrics.json"
PHASE10A_OFFICIAL_MODEL = ROOT / "results/models/phase10a_w_residual_corrected_candidate.joblib"
METRICS_JSON = ROOT / "results/metrics/phase14e_weff_area_reporting_layer_metrics.json"
SUMMARY_MD = ROOT / "results/summaries/phase14e_weff_area_reporting_layer_summary.md"
MANIFEST_CSV = ROOT / "results/manifests/phase14e_weff_area_reporting_layer_manifest.csv"
REPORTS_JSONL = ROOT / "results/reports/phase14e_weff_area_test_reports.jsonl"

TARGET_NAME = "W_eff_area"
RISK_REASON = "ensemble_std_large_error_risk"
WIDTH_DESCRIPTOR_STATUS = "report_only"

REQUIRED_WIDTH_DESCRIPTOR_FIELDS = [
    "W_eff_area_pred_mm",
    "W_eff_area_ci80_lower_mm",
    "W_eff_area_ci80_upper_mm",
    "W_eff_area_ci90_lower_mm",
    "W_eff_area_ci90_upper_mm",
    "W_eff_area_uncertainty_std_mm",
    "W_eff_area_risk_flag",
    "W_eff_area_risk_reason",
    "width_descriptor_status",
]

OFFICIAL_LWD_REFERENCE_MAE_MM = {"L_m": 2.0005228062972233, "W_m": 2.095925210234827, "D_m": 0.8601579644854588}
OFFICIAL_W_M_RECONSTRUCTION_TOL_MM = 1.0e-9

REPRO_WEFF_MAE_MM = 1.444350870555334
REPRO_COVERAGE_80 = 0.8974358974358975
REPRO_COVERAGE_90 = 0.9487179487179487
REPRO_FLAG_COUNT = 10
REPRO_MAE_TOL_MM = 1.0e-4
REPRO_COVERAGE_TOL = 1.0e-6


def reconstruct_official_141_features(delta_b: np.ndarray, sensor_x: np.ndarray) -> np.ndarray:
    """Reconstruct the official Phase-3a 141-feature schema for the frozen L/W/D baseline.

    Ported from the first-principles audit reconstruction. 132/141 features match
    the official x_scaler train-mean fingerprint exactly; the 9 lr_energy_diff
    features are approximate, so the reloaded official baseline reproduces official
    L/W/D to <= 0.031 mm. Used only to feed the reloaded official baseline for the
    preserved W_m field; the W_eff_area descriptor uses the Phase-14 feature world.
    """
    signal = np.asarray(delta_b, dtype=np.float64)
    if signal.ndim != 4 or signal.shape[1:3] != (3, 3):
        raise ValueError("expected delta_b (N,3,3,X), got %r" % (signal.shape,))
    x = np.asarray(sensor_x, dtype=np.float64).reshape(-1)
    n_x = signal.shape[3]
    if x.size != n_x:
        raise ValueError("sensor_x length does not match delta_b x dimension")
    dx = float(x[1] - x[0])
    half = n_x // 2
    n = signal.shape[0]

    def line_feats(s: np.ndarray) -> list[float]:
        smax = float(s.max())
        smin = float(s.min())
        p2p = smax - smin
        mean = float(s.mean())
        t = s - mean
        mean_abs = float(np.abs(t).mean())
        ipk = int(np.argmax(s))
        ivl = int(np.argmin(s))
        at = np.abs(t)
        abs_area = float(np.trapezoid(at, x))
        eff_w = float((at > 0.5 * at.max()).sum() * dx)
        a_left = float(at[:half].sum())
        a_right = float(at[half + 1:].sum())
        lr = (a_left - a_right) / (a_left + a_right) if (a_left + a_right) > 0 else 0.0
        g = np.gradient(s, x)
        ig = int(np.argmax(np.abs(g)))
        gpos = g[g > 0].max() if (g > 0).any() else 0.0
        gneg = -g[g < 0].min() if (g < 0).any() else 0.0
        return [
            smax, smin, p2p, mean, mean_abs,
            float(x[ipk]), float(x[ivl]), float(abs(x[ipk] - x[ivl])),
            eff_w, float(np.trapezoid(s, x)), abs_area, lr,
            float(np.abs(g).max()), float(x[ig]), float(gpos / (gneg + 1e-30)),
        ]

    features = np.zeros((n, 141), dtype=np.float64)
    for i in range(n):
        col = 0
        for axis in range(3):
            for line in range(3):
                features[i, col:col + 15] = line_feats(signal[i, axis, line, :])
                col += 15
        p2pl = [signal[i, axis, 1, :].max() - signal[i, axis, 1, :].min() for axis in range(3)]
        features[i, 135] = p2pl[0] / p2pl[1]
        features[i, 136] = p2pl[0] / p2pl[2]
        features[i, 137] = p2pl[1] / p2pl[2]
        for axis in range(3):
            per_line = [signal[i, axis, line, :].max() - signal[i, axis, line, :].min() for line in range(3)]
            features[i, 138 + axis] = per_line[1] / (0.5 * (per_line[0] + per_line[2]))
    if not np.isfinite(features).all():
        raise RuntimeError("non-finite reconstructed official features")
    return features


def load_official_lwd_baseline(path: Path) -> dict[str, Any]:
    """Reload the serialized official L/W/D ExtraTrees baseline (no training)."""
    import joblib

    if not path.exists():
        raise FileNotFoundError("official baseline artifact missing: %s" % path)
    payload = joblib.load(path)
    official = payload["candidate_payload"]["official_fit"]
    return {
        "estimator": official["estimator"],
        "x_scaler": official["x_scaler"],
        "y_scaler": official["y_scaler"],
        "target_order": ["L_m", "W_m", "D_m"],
    }


def official_lwd_predict_m(baseline: dict[str, Any], delta_b: np.ndarray, sensor_x: np.ndarray) -> np.ndarray:
    """Predict official L/W/D in meters via reloaded baseline + reconstructed features."""
    features = reconstruct_official_141_features(delta_b, sensor_x)
    scaled = baseline["x_scaler"].transform(features)
    return np.asarray(baseline["y_scaler"].inverse_transform(baseline["estimator"].predict(scaled)), dtype=np.float64)


def load_weff_descriptor(path: Path) -> dict[str, Any]:
    """Reload the Phase 14d W_eff_area ensemble descriptor (12 fitted members)."""
    import joblib

    if not path.exists():
        raise FileNotFoundError("Phase14d descriptor artifact missing: %s" % path)
    payload = joblib.load(path)
    if payload.get("metadata", {}).get("gate14d", {}).get("verdict") != "PASS":
        raise RuntimeError("Phase14d descriptor did not pass gate14d; refusing to integrate")
    return payload


def weff_area_descriptor_predict(descriptor: dict[str, Any], features_180: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble mean (point) and std (uncertainty) in meters over the 12 reloaded members."""
    members = descriptor["ensemble_members"]
    if len(members) < 2:
        raise RuntimeError("descriptor ensemble has fewer than 2 members")
    preds = np.vstack([np.asarray(member.predict(features_180), dtype=np.float64) for member in members])
    return preds.mean(axis=0), np.maximum(preds.std(axis=0), 1.0e-12)


def build_sample_report(
    sample_id: str,
    official_lwd_m: np.ndarray,
    weff_mean_m: float,
    weff_std_m: float,
    radius_80_m: float,
    radius_90_m: float,
    risk_threshold_mm: float,
    weff_true_m: float | None,
) -> dict[str, Any]:
    """Assemble one per-sample report: official W_m preserved + W_eff_area descriptor."""
    mm = 1000.0
    pred_mm = float(weff_mean_m * mm)
    std_mm = float(weff_std_m * mm)
    risk_flag = bool(std_mm >= float(risk_threshold_mm))
    width_descriptor = {
        "target": TARGET_NAME,
        "width_descriptor_status": WIDTH_DESCRIPTOR_STATUS,
        "W_eff_area_pred_mm": pred_mm,
        "W_eff_area_ci80_lower_mm": float((weff_mean_m - radius_80_m) * mm),
        "W_eff_area_ci80_upper_mm": float((weff_mean_m + radius_80_m) * mm),
        "W_eff_area_ci90_lower_mm": float((weff_mean_m - radius_90_m) * mm),
        "W_eff_area_ci90_upper_mm": float((weff_mean_m + radius_90_m) * mm),
        "W_eff_area_uncertainty_std_mm": std_mm,
        "W_eff_area_risk_flag": risk_flag,
        "W_eff_area_risk_reason": RISK_REASON,
        "descriptor_source": "phase14d_weff_area_ensemble_descriptor",
        "descriptor_gate": "PASS",
        "point_estimate_method": "ensemble_mean",
        "uncertainty_method": "ensemble_std",
        "interval_method": "val_calibrated_plain_split_conformal",
        "risk_flag_threshold_mm": float(risk_threshold_mm),
    }
    report = {
        "sample_id": sample_id,
        "status": "COMPATIBLE_REPORTED",
        "report_schema_version": "phase14e_weff_area_reporting_v1",
        "official_lwd_baseline": {
            "source": "phase6a_9d_official_lwd_baseline_reloaded_phase10a_official_fit",
            "targets": ["L_m", "W_m", "D_m"],
            "L_m_pred_mm": float(official_lwd_m[0] * mm),
            "W_m_pred_mm": float(official_lwd_m[1] * mm),
            "D_m_pred_mm": float(official_lwd_m[2] * mm),
            "w_m_is_official_nominal_width": True,
            "w_m_replaced_by_w_eff_area": False,
            "reconstruction_note": "official features reconstructed to Phase-3a schema (132/141 exact); reproduces official L/W/D to <=0.031 mm",
        },
        "width_descriptor": width_descriptor,
        "descriptor_participates_in_lwd_mean_mae": False,
        "boundary": {
            "training_performed": False,
            "comsol_run": False,
            "current_baseline_modified": False,
            "uses_true_family": False,
            "uses_predicted_family": False,
            "uses_mixture_routing": False,
            "w_m_residual_or_stacking": False,
            "w_eff_area_replaces_w_m": False,
        },
    }
    if weff_true_m is not None:
        report["eval_context"] = {
            "W_eff_area_true_mm": float(weff_true_m * mm),
            "W_eff_area_abs_error_mm": float(abs(weff_mean_m - weff_true_m) * mm),
            "note": "frozen v3_240 test has ground-truth width targets; eval context is report-only",
        }
    return report


def validate_report_schema(report: dict[str, Any]) -> dict[str, Any]:
    """Validate one report: required descriptor fields, constant fields, W_m preservation."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "observed": observed})

    wd = report.get("width_descriptor", {})
    for field in REQUIRED_WIDTH_DESCRIPTOR_FIELDS:
        add("field_present:%s" % field, field in wd, field in wd)
    add("width_descriptor_status_report_only", wd.get("width_descriptor_status") == WIDTH_DESCRIPTOR_STATUS, wd.get("width_descriptor_status"))
    add("risk_reason_constant", wd.get("W_eff_area_risk_reason") == RISK_REASON, wd.get("W_eff_area_risk_reason"))
    add("risk_flag_is_bool", isinstance(wd.get("W_eff_area_risk_flag"), bool), type(wd.get("W_eff_area_risk_flag")).__name__)

    ci80_ok = (
        "W_eff_area_ci80_lower_mm" in wd
        and "W_eff_area_ci80_upper_mm" in wd
        and wd["W_eff_area_ci80_lower_mm"] <= wd["W_eff_area_ci80_upper_mm"]
    )
    ci90_ok = (
        "W_eff_area_ci90_lower_mm" in wd
        and "W_eff_area_ci90_upper_mm" in wd
        and wd["W_eff_area_ci90_lower_mm"] <= wd["W_eff_area_ci90_upper_mm"]
    )
    nested_ok = (
        ci90_ok and ci80_ok
        and wd["W_eff_area_ci90_lower_mm"] <= wd["W_eff_area_ci80_lower_mm"]
        and wd["W_eff_area_ci80_upper_mm"] <= wd["W_eff_area_ci90_upper_mm"]
    )
    add("ci80_lower_le_upper", ci80_ok)
    add("ci90_lower_le_upper", ci90_ok)
    add("ci80_nested_in_ci90", nested_ok)

    official = report.get("official_lwd_baseline", {})
    add("official_w_m_present", "W_m_pred_mm" in official, "W_m_pred_mm" in official)
    add("w_m_is_official_nominal", official.get("w_m_is_official_nominal_width") is True)
    add("w_m_not_replaced", official.get("w_m_replaced_by_w_eff_area") is False)
    add("descriptor_excluded_from_lwd_mae", report.get("descriptor_participates_in_lwd_mean_mae") is False)
    add("status_compatible_reported", report.get("status") == "COMPATIBLE_REPORTED")

    passed = all(check["passed"] for check in checks)
    return {"passed": passed, "checks": checks, "n_checks": len(checks), "n_failed": sum(1 for c in checks if not c["passed"])}


def gate14e(
    n_reports: int,
    n_test: int,
    all_schemas_pass: bool,
    weff_mae_mm: float,
    coverage_80: float,
    coverage_90: float,
    flag_count: int,
    flagged_mae_mm: float,
    unflagged_mae_mm: float,
    official_w_m_reproduced: bool,
) -> dict[str, Any]:
    """Gate 14e: reporting-layer integration acceptance.

    PASS requires: schema validation passes for every report; 39/39 frozen-test
    reports generated; W_eff_area MAE reproduces 14d (~1.444351 mm); 80/90 interval
    coverage reproduces ~0.897/0.949; risk flag count reproduces 10/39; flagged MAE
    strictly above unflagged MAE. Official W_m reproduction is a required provenance
    guard (reload validity), not a numeric reproduction target.
    """
    checks = {
        "reports_generated_39_of_39": bool(n_reports == n_test == 39),
        "schema_validation_pass": bool(all_schemas_pass),
        "weff_mae_reproduced": bool(abs(weff_mae_mm - REPRO_WEFF_MAE_MM) <= REPRO_MAE_TOL_MM),
        "coverage_80_reproduced": bool(abs(coverage_80 - REPRO_COVERAGE_80) <= REPRO_COVERAGE_TOL),
        "coverage_90_reproduced": bool(abs(coverage_90 - REPRO_COVERAGE_90) <= REPRO_COVERAGE_TOL),
        "coverage_80_meets_target": bool(coverage_80 >= 0.80),
        "coverage_90_meets_target": bool(coverage_90 >= 0.90),
        "flag_count_reproduced": bool(flag_count == REPRO_FLAG_COUNT),
        "flagged_worse_than_unflagged": bool(
            np.isfinite(flagged_mae_mm) and np.isfinite(unflagged_mae_mm) and flagged_mae_mm > unflagged_mae_mm
        ),
        "official_w_m_reload_valid": bool(official_w_m_reproduced),
    }
    passed = all(checks.values())
    if passed:
        verdict = "PASS"
        rationale = "39/39 reports built and schema-valid; W_eff_area MAE/coverage/flag reproduce Phase14d; W_m preserved from official baseline"
    else:
        failed = [name for name, ok in checks.items() if not ok]
        verdict = "FAIL"
        rationale = "reporting-layer integration failed checks: %s" % ", ".join(failed)
    return {
        "verdict": verdict,
        "rationale": rationale,
        "checks": checks,
        "reproduction_targets": {
            "weff_area_mae_mm": REPRO_WEFF_MAE_MM,
            "coverage_80": REPRO_COVERAGE_80,
            "coverage_90": REPRO_COVERAGE_90,
            "flag_count": REPRO_FLAG_COUNT,
        },
        "descriptor_joined_reporting_layer": verdict == "PASS",
        "current_baseline_update": False,
        "w_m_point_baseline_replaced": False,
    }


def build_manifest_rows(gate: dict[str, Any], values: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "phase": "phase14e_weff_area_reporting_layer",
            "descriptor_name": "global_W_eff_area_ensemble_descriptor",
            "descriptor_source_gate": "phase14d_PASS",
            "target_name": TARGET_NAME,
            "width_descriptor_status": WIDTH_DESCRIPTOR_STATUS,
            "n_reports": str(values["n_reports"]),
            "frozen_test_count": str(values["n_test"]),
            "schema_validation_pass": str(bool(values["all_schemas_pass"])),
            "weff_area_test_mae_mm": "%.6f" % values["weff_mae_mm"],
            "coverage_80": "%.6f" % values["coverage_80"],
            "coverage_90": "%.6f" % values["coverage_90"],
            "risk_flag_count": str(values["flag_count"]),
            "flagged_mae_mm": "%.6f" % values["flagged_mae_mm"],
            "unflagged_mae_mm": "%.6f" % values["unflagged_mae_mm"],
            "official_w_m_test_mae_mm": "%.6f" % values["official_w_m_mae_mm"],
            "w_m_source": "official_phase6a_9d_baseline_reload",
            "w_m_replaced_by_w_eff_area": "False",
            "descriptor_in_lwd_mean_mae": "False",
            "gate14e_verdict": gate["verdict"],
            "reports_jsonl": str(REPORTS_JSONL.relative_to(ROOT)).replace("\\", "/"),
            "comsol_run": "False",
            "real_data_used": "False",
            "bz_only": "False",
            "current_baseline_update": "False",
            "training_performed": "False",
            "uses_true_family": "False",
            "uses_predicted_family": "False",
            "uses_mixture_routing": "False",
            "old_v3_test_used_for_training_or_selection": "False",
        }
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_paths = [args.metrics_json, args.summary_md, args.manifest_csv, args.reports_jsonl]
    check_no_overwrite(output_paths, args.overwrite)

    old = load_dataset(V3_240_DATASET_ID)
    splits = split_indices(old)
    test_idx = splits["test"]
    train_idx = splits["train"]
    n_test = int(len(test_idx))

    old_all, all_names = extract_all_141_features(old.delta_b, old.sensor_x)
    old_width, width_names = extract_signal_width_features(old.delta_b, old.sensor_x)
    features_180 = np.hstack([old_all, old_width])
    weff_targets = compute_width_targets_from_depth_maps(old.profile_depth_map_xy_m, old.rbc_params)
    weff_true_test = weff_targets[TARGET_NAME][test_idx]

    old_wWD = np.asarray(old.rbc_params[:, 4], dtype=np.float64)
    high_mask, _ = build_high_nuisance_mask(old_wWD, train_idx)
    high_test = high_mask[test_idx]

    descriptor = load_weff_descriptor(args.phase14d_model)
    radius_80_m = float(descriptor["plain_conformal_radius_m"]["80"])
    radius_90_m = float(descriptor["plain_conformal_radius_m"]["90"])
    risk_threshold_mm = float(descriptor["risk_flag_threshold_mm"])
    weff_mean_all, weff_std_all = weff_area_descriptor_predict(descriptor, features_180)
    weff_mean_test = weff_mean_all[test_idx]
    weff_std_test = weff_std_all[test_idx]

    baseline = load_official_lwd_baseline(args.official_model)
    official_lwd_test_m = official_lwd_predict_m(baseline, old.delta_b[test_idx], old.sensor_x)
    official_w_m_mae_mm = float(np.mean(np.abs(official_lwd_test_m[:, 1] - old.rbc_params[test_idx, 1]) * 1000.0))
    official_l_mae_mm = float(np.mean(np.abs(official_lwd_test_m[:, 0] - old.rbc_params[test_idx, 0]) * 1000.0))
    official_d_mae_mm = float(np.mean(np.abs(official_lwd_test_m[:, 2] - old.rbc_params[test_idx, 2]) * 1000.0))
    official_w_m_reproduced = abs(official_w_m_mae_mm - OFFICIAL_LWD_REFERENCE_MAE_MM["W_m"]) <= OFFICIAL_W_M_RECONSTRUCTION_TOL_MM

    sample_ids = np.asarray(old.sample_ids).astype(str)[test_idx]
    reports: list[dict[str, Any]] = []
    schema_results: list[dict[str, Any]] = []
    abs_err_mm = np.abs(weff_mean_test - weff_true_test) * 1000.0
    for i in range(n_test):
        report = build_sample_report(
            sample_id=str(sample_ids[i]),
            official_lwd_m=official_lwd_test_m[i],
            weff_mean_m=float(weff_mean_test[i]),
            weff_std_m=float(weff_std_test[i]),
            radius_80_m=radius_80_m,
            radius_90_m=radius_90_m,
            risk_threshold_mm=risk_threshold_mm,
            weff_true_m=float(weff_true_test[i]),
        )
        schema = validate_report_schema(report)
        report["schema_validation"] = schema
        reports.append(report)
        schema_results.append(schema)
    all_schemas_pass = all(s["passed"] for s in schema_results)

    weff_mae_mm = float(np.mean(abs_err_mm))
    coverage_80 = float(np.mean((weff_true_test >= weff_mean_test - radius_80_m) & (weff_true_test <= weff_mean_test + radius_80_m)))
    coverage_90 = float(np.mean((weff_true_test >= weff_mean_test - radius_90_m) & (weff_true_test <= weff_mean_test + radius_90_m)))
    flagged = weff_std_test * 1000.0 >= risk_threshold_mm
    flag_count = int(np.sum(flagged))
    flagged_mae_mm = float(np.mean(abs_err_mm[flagged])) if np.any(flagged) else math.nan
    unflagged_mae_mm = float(np.mean(abs_err_mm[~flagged])) if np.any(~flagged) else math.nan

    gate = gate14e(
        n_reports=len(reports),
        n_test=n_test,
        all_schemas_pass=all_schemas_pass,
        weff_mae_mm=weff_mae_mm,
        coverage_80=coverage_80,
        coverage_90=coverage_90,
        flag_count=flag_count,
        flagged_mae_mm=flagged_mae_mm,
        unflagged_mae_mm=unflagged_mae_mm,
        official_w_m_reproduced=official_w_m_reproduced,
    )

    values = {
        "n_reports": len(reports),
        "n_test": n_test,
        "all_schemas_pass": all_schemas_pass,
        "weff_mae_mm": weff_mae_mm,
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "flag_count": flag_count,
        "flagged_mae_mm": flagged_mae_mm,
        "unflagged_mae_mm": unflagged_mae_mm,
        "official_w_m_mae_mm": official_w_m_mae_mm,
    }
    metrics = assemble_metrics(
        gate, values, descriptor, official_l_mae_mm, official_w_m_mae_mm, official_d_mae_mm,
        official_w_m_reproduced, risk_threshold_mm, radius_80_m, radius_90_m,
        int(np.sum(high_test)), int(np.sum(flagged & high_test)),
    )

    args.reports_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.reports_jsonl.open("w", encoding="utf-8") as f:
        for report in reports:
            f.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
    write_csv(args.manifest_csv, build_manifest_rows(gate, values))
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json_dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_summary(args.summary_md, metrics)
    return metrics


def assemble_metrics(
    gate: dict[str, Any],
    values: dict[str, Any],
    descriptor: dict[str, Any],
    official_l_mae_mm: float,
    official_w_m_mae_mm: float,
    official_d_mae_mm: float,
    official_w_m_reproduced: bool,
    risk_threshold_mm: float,
    radius_80_m: float,
    radius_90_m: float,
    high_nuisance_test_count: int,
    flagged_and_high_count: int,
) -> dict[str, Any]:
    return {
        "phase": "phase14e_weff_area_reporting_layer",
        "purpose": "integrate the Phase14d W_eff_area ensemble descriptor into per-sample reports while preserving official W_m",
        "boundary": {
            "comsol_run": False,
            "real_data_used": False,
            "bz_only": False,
            "current_baseline_update": False,
            "training_performed": False,
            "uses_true_family": False,
            "uses_predicted_family": False,
            "uses_mixture_routing": False,
            "w_m_residual_or_stacking": False,
            "w_eff_area_replaces_w_m": False,
            "descriptor_in_lwd_mean_mae": False,
            "old_v3_test_used_for_training_or_selection": False,
        },
        "dataset": {
            "v3_dataset_id": V3_240_DATASET_ID,
            "frozen_test_count": values["n_test"],
            "target_name": TARGET_NAME,
            "feature_schema": "all_141_plus_fixed_signal_width_features (180 dims) for descriptor; reconstructed official 141 for W_m",
        },
        "reload_sources": {
            "weff_descriptor": str(PHASE14D_MODEL.relative_to(ROOT)).replace("\\", "/"),
            "weff_descriptor_gate": descriptor.get("metadata", {}).get("gate14d", {}).get("verdict"),
            "n_ensemble_members": len(descriptor["ensemble_members"]),
            "official_lwd_baseline": str(PHASE10A_OFFICIAL_MODEL.relative_to(ROOT)).replace("\\", "/"),
            "plain_conformal_radius_m": {"80": radius_80_m, "90": radius_90_m},
            "risk_flag_threshold_mm": risk_threshold_mm,
        },
        "official_lwd_preservation": {
            "source": "reloaded phase10a official_fit (official Phase6a/9d L/W/D baseline)",
            "official_test_mae_mm": {
                "L_m": official_l_mae_mm,
                "W_m": official_w_m_mae_mm,
                "D_m": official_d_mae_mm,
            },
            "official_reference_mae_mm": OFFICIAL_LWD_REFERENCE_MAE_MM,
            "w_m_reconstruction_tol_mm": OFFICIAL_W_M_RECONSTRUCTION_TOL_MM,
            "official_w_m_reload_valid": bool(official_w_m_reproduced),
            "w_m_replaced_by_w_eff_area": False,
            "w_eff_area_in_lwd_mean_mae": False,
            "note": "official features reconstructed to Phase-3a schema (132/141 exact); source extractor unavailable, reload reproduces official W_m to <=0.031 mm",
        },
        "width_descriptor_reproduction": {
            "weff_area_test_mae_mm": values["weff_mae_mm"],
            "phase14d_reference_mae_mm": REPRO_WEFF_MAE_MM,
            "coverage_80": values["coverage_80"],
            "coverage_90": values["coverage_90"],
            "phase14d_reference_coverage": {"80": REPRO_COVERAGE_80, "90": REPRO_COVERAGE_90},
            "risk_flag_count": values["flag_count"],
            "phase14d_reference_flag_count": REPRO_FLAG_COUNT,
            "flagged_mae_mm": values["flagged_mae_mm"],
            "unflagged_mae_mm": values["unflagged_mae_mm"],
            "flagged_minus_unflagged_mm": float(values["flagged_mae_mm"] - values["unflagged_mae_mm"]),
            "high_nuisance_test_count": high_nuisance_test_count,
            "flagged_and_high_nuisance_count": flagged_and_high_count,
        },
        "report_schema": {
            "required_width_descriptor_fields": REQUIRED_WIDTH_DESCRIPTOR_FIELDS,
            "all_reports_schema_valid": values["all_schemas_pass"],
            "n_reports": values["n_reports"],
            "reports_jsonl": str(REPORTS_JSONL.relative_to(ROOT)).replace("\\", "/"),
        },
        "gate14e": gate,
    }


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    gate = metrics["gate14e"]
    rep = metrics["width_descriptor_reproduction"]
    off = metrics["official_lwd_preservation"]
    sch = metrics["report_schema"]
    lines = [
        "# Phase 14e W_eff_area Reporting-Layer Integration Summary",
        "",
        "- Gate 14e verdict: `%s`" % gate["verdict"],
        "- rationale: %s" % gate["rationale"],
        "- descriptor joined reporting layer: `%s`" % gate["descriptor_joined_reporting_layer"],
        "- reports generated: `%d/%d` (frozen v3 test); all schema-valid: `%s`"
        % (sch["n_reports"], metrics["dataset"]["frozen_test_count"], sch["all_reports_schema_valid"]),
        "",
        "## W_eff_area Descriptor Reproduction (report-only ground-truth eval)",
        "",
        "- W_eff_area test MAE: `%.6f mm` (Phase14d `%.6f mm`)" % (rep["weff_area_test_mae_mm"], rep["phase14d_reference_mae_mm"]),
        "- 80%% coverage: `%.6f` (ref `%.6f`); 90%%: `%.6f` (ref `%.6f`)"
        % (rep["coverage_80"], rep["phase14d_reference_coverage"]["80"], rep["coverage_90"], rep["phase14d_reference_coverage"]["90"]),
        "- risk flag count: `%d/%d` (ref `%d`)" % (rep["risk_flag_count"], metrics["dataset"]["frozen_test_count"], rep["phase14d_reference_flag_count"]),
        "- flagged MAE `%.6f` vs unflagged `%.6f` mm (delta `%+.6f`)"
        % (rep["flagged_mae_mm"], rep["unflagged_mae_mm"], rep["flagged_minus_unflagged_mm"]),
        "",
        "## Official W_m Preservation",
        "",
        "- source: %s" % off["source"],
        "- 独立 W/D 来源模型的重载检查（不是完整组合尺寸基准）：L `%.6f`, W `%.6f`, D `%.6f` mm；完整尺寸结果见 phase17a_composite_reporting_layer_metrics.json"
        % (off["official_test_mae_mm"]["L_m"], off["official_test_mae_mm"]["W_m"], off["official_test_mae_mm"]["D_m"]),
        "- official W_m reload valid (within %.2f mm): `%s`" % (off["w_m_reconstruction_tol_mm"], off["official_w_m_reload_valid"]),
        "- W_m replaced by W_eff_area: `%s`; W_eff_area in L/W/D mean MAE: `%s`"
        % (off["w_m_replaced_by_w_eff_area"], off["w_eff_area_in_lwd_mean_mae"]),
        "",
        "## New Report Fields (per sample)",
        "",
    ]
    for field in sch["required_width_descriptor_fields"]:
        lines.append("- `%s`" % field)
    lines += [
        "",
        "## Boundary Confirmation",
        "",
        "- no COMSOL run",
        "- no real-data input",
        "- no Bz-only route",
        "- no CURRENT_BASELINE.md update",
        "- no model training (pure reload of Phase14d descriptor + official baseline)",
        "- no true/predicted family routing, no mixture heads, no W_m residual/stacking",
        "- W_m stays official nominal width; W_eff_area is report-only and excluded from L/W/D mean MAE",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase14d-model", type=Path, default=PHASE14D_MODEL)
    parser.add_argument("--official-model", type=Path, default=PHASE10A_OFFICIAL_MODEL)
    parser.add_argument("--metrics-json", type=Path, default=METRICS_JSON)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    parser.add_argument("--manifest-csv", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--reports-jsonl", type=Path, default=REPORTS_JSONL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = run(args)
    print(
        json_dumps(
            {
                "gate14e": metrics["gate14e"]["verdict"],
                "n_reports": metrics["report_schema"]["n_reports"],
                "weff_area_test_mae_mm": metrics["width_descriptor_reproduction"]["weff_area_test_mae_mm"],
                "coverage_80": metrics["width_descriptor_reproduction"]["coverage_80"],
                "coverage_90": metrics["width_descriptor_reproduction"]["coverage_90"],
                "risk_flag_count": metrics["width_descriptor_reproduction"]["risk_flag_count"],
                "official_w_m_mae_mm": metrics["official_lwd_preservation"]["official_test_mae_mm"]["W_m"],
                "metrics_json": str(args.metrics_json),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
