#!/usr/bin/env python
"""Build Phase 17a: composite-baseline reporting layer v2.

Aligns the per-sample reporting layer with the Phase 16e promoted baseline:
L_m from the 16b 12-member ensemble (point + std + val-calibrated conformal),
W_m/D_m from the unchanged official fit, W_eff_area report-only descriptor
fields carried over from Phase 14e. All three MAEs are computed from aligned
predictions on the same frozen test samples. No training, no COMSOL.
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

import train_phase11a_ld_targeted_retraining as p11  # noqa: E402
import __main__ as _M  # noqa: E402
for _n in dir(p11):
    if not _n.startswith("_"):
        setattr(_M, _n, getattr(p11, _n))

from load_true_3d_rbc_pilot_dataset import V3_240_DATASET_ID, load_dataset, split_indices, sha256_file  # noqa: E402
from train_phase14a_observable_width_conditioned_models import (  # noqa: E402
    build_high_nuisance_mask,
    check_no_overwrite,
    compute_width_targets_from_depth_maps,
    extract_signal_width_features,
    json_dumps,
    write_csv,
)
from train_phase11a_ld_targeted_retraining import extract_all_141_features, extract_ld_specific_features  # noqa: E402
from build_phase14e_weff_area_reporting_layer import (  # noqa: E402
    PHASE10A_OFFICIAL_MODEL,
    PHASE14D_MODEL,
    load_official_lwd_baseline,
    load_weff_descriptor,
    official_lwd_predict_m,
    weff_area_descriptor_predict,
)
from eval_phase16a_l_upgrade_composite_candidate import mae_mm  # noqa: E402

L_ENSEMBLE_JOBLIB = ROOT / "results/models/phase16b_l_head_ensemble_descriptor.joblib"
METRICS_JSON = ROOT / "results/metrics/phase17a_composite_reporting_layer_metrics.json"
MANIFEST_CSV = ROOT / "results/manifests/phase17a_composite_reporting_layer_manifest.csv"
REPORTS_JSONL = ROOT / "results/reports/phase17a_composite_test_reports.jsonl"

PREDICTIONS_CSV = ROOT / "results/metrics/composite_unified_test_predictions.csv"

REPRO = {
    "L_mae_mm": 1.6262799071882357, "L_cov80": 0.9743589743589743, "L_cov90": 1.0,
    "weff_mae_mm": 1.4443508705553338, "weff_cov80": 0.8974358974358975,
    "weff_cov90": 0.9487179487179487, "weff_flag_count": 10,
    "W_mae_mm": 2.095925210234827, "D_mae_mm": 0.8601579644854588,
    "composite_mean_lwd_mm": 1.527454360636174,
}
TOL = 1.0e-9

REQUIRED_FIELDS = [
    "L_m_pred_mm", "L_m_ci80_lower_mm", "L_m_ci80_upper_mm",
    "L_m_ci90_lower_mm", "L_m_ci90_upper_mm", "L_m_uncertainty_std_mm",
    "W_m_pred_mm", "D_m_pred_mm",
]


def build_report(sid: str, l_mean_m: float, l_std_m: float, l_r80: float, l_r90: float,
                 w_m: float, d_m: float, weff_block: dict[str, Any]) -> dict[str, Any]:
    mm = 1000.0
    return {
        "sample_id": sid,
        "status": "COMPATIBLE_REPORTED",
        "report_schema_version": "phase17a_composite_reporting_v2",
        "official_composite_baseline": {
            "source": "unified frozen-test evaluation of fixed L/W/D composite",
            "L_m_pred_mm": float(l_mean_m * mm),
            "L_m_ci80_lower_mm": float((l_mean_m - l_r80) * mm),
            "L_m_ci80_upper_mm": float((l_mean_m + l_r80) * mm),
            "L_m_ci90_lower_mm": float((l_mean_m - l_r90) * mm),
            "L_m_ci90_upper_mm": float((l_mean_m + l_r90) * mm),
            "L_m_uncertainty_std_mm": float(l_std_m * mm),
            "L_source": "phase16b_l_head_ensemble (12-member ET-600, 11a 210-dim features)",
            "W_m_pred_mm": float(w_m * mm),
            "D_m_pred_mm": float(d_m * mm),
            "W_D_source": "official phase10a official_fit (unchanged; reconstructed official 141 features)",
        },
        "width_descriptor": weff_block,
        "descriptor_participates_in_lwd_mean_mae": False,
        "boundary": {
            "training_performed": False, "comsol_run": False,
            "current_baseline_modified": False,
            "w_m_replaced_by_w_eff_area": False,
            "uses_true_family": False, "uses_predicted_family": False,
        },
    }


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    checks = []
    block = report.get("official_composite_baseline", {})
    for f in REQUIRED_FIELDS:
        checks.append({"name": "field:%s" % f, "passed": f in block})
    wd = report.get("width_descriptor", {})
    checks.append({"name": "weff_fields", "passed": all(k in wd for k in [
        "W_eff_area_pred_mm", "W_eff_area_risk_flag", "width_descriptor_status"])})
    checks.append({"name": "weff_report_only", "passed": wd.get("width_descriptor_status") == "report_only"})
    ok80 = block.get("L_m_ci80_lower_mm", 0) <= block.get("L_m_pred_mm", 0) <= block.get("L_m_ci80_upper_mm", 0)
    nested = (block.get("L_m_ci90_lower_mm", 1) <= block.get("L_m_ci80_lower_mm", 0)
              and block.get("L_m_ci80_upper_mm", 0) <= block.get("L_m_ci90_upper_mm", -1))
    checks.append({"name": "L_ci80_contains_point", "passed": bool(ok80)})
    checks.append({"name": "L_ci80_nested_in_ci90", "passed": bool(nested)})
    checks.append({"name": "not_in_lwd_mae", "passed": report.get("descriptor_participates_in_lwd_mean_mae") is False})
    return {"passed": all(c["passed"] for c in checks), "n_checks": len(checks),
            "n_failed": sum(1 for c in checks if not c["passed"]), "checks": checks}


def gate17a(n_reports: int, schemas_ok: bool, repro: dict[str, bool]) -> dict[str, Any]:
    checks = {"reports_39_of_39": n_reports == 39, "schema_all_pass": bool(schemas_ok)}
    checks.update(repro)
    verdict = "PASS" if all(checks.values()) else "FAIL"
    return {"verdict": verdict, "checks": checks,
            "rationale": "组合报告与同一测试集的逐样本预测一致，指标在数值容差内复现"
            if verdict == "PASS" else "failed: %s" % [k for k, v in checks.items() if not v],
            "current_baseline_update": False}


def composite_metrics(prediction_m: np.ndarray, truth_m: np.ndarray) -> dict[str, float]:
    """从同一组逐样本预测计算三个维度和平均 MAE，不读取历史汇总常量。"""
    prediction = np.asarray(prediction_m, dtype=np.float64)
    truth = np.asarray(truth_m, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 3 or not len(prediction):
        raise ValueError("expected matching nonempty (N, 3) predictions and truth")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("predictions and truth must be finite")
    errors = np.mean(np.abs(prediction - truth), axis=0) * 1000.0
    return {"L_mae_mm": float(errors[0]), "W_mae_mm": float(errors[1]),
            "D_mae_mm": float(errors[2]), "composite_mean_lwd_mm": float(errors.mean())}


def main() -> int:
    import joblib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    check_no_overwrite([METRICS_JSON, MANIFEST_CSV, REPORTS_JSONL, PREDICTIONS_CSV], args.overwrite)

    old = load_dataset(V3_240_DATASET_ID)
    splits = split_indices(old)
    test_idx = splits["test"]
    y = np.asarray(old.rbc_params, dtype=np.float64)
    sample_ids = np.asarray(old.sample_ids).astype(str)[test_idx]

    # L: recompute the fixed 210-dimensional signal features from the same observations.
    lpack = joblib.load(L_ENSEMBLE_JOBLIB)
    f141, _ = extract_all_141_features(old.delta_b, old.sensor_x)
    ld_features, _ = extract_ld_specific_features(old.delta_b, old.sensor_x)
    x_old = np.hstack([f141, ld_features])
    lp = np.vstack([np.asarray(m.predict(x_old[test_idx]), dtype=np.float64) for m in lpack["ensemble_members"]])
    l_mean, l_std = lp.mean(axis=0), np.maximum(lp.std(axis=0), 1e-12)
    l_r80 = float(lpack["plain_conformal_radius_m"]["80"]); l_r90 = float(lpack["plain_conformal_radius_m"]["90"])
    l_mae = mae_mm(l_mean, y[test_idx, 0])
    l_cov80 = float(np.mean(np.abs(l_mean - y[test_idx, 0]) <= l_r80))
    l_cov90 = float(np.mean(np.abs(l_mean - y[test_idx, 0]) <= l_r90))

    # W/D: official reload
    off = official_lwd_predict_m(load_official_lwd_baseline(PHASE10A_OFFICIAL_MODEL),
                                 old.delta_b[test_idx], old.sensor_x)

    # W_eff_area descriptor: 14d joblib on 180-dim features
    f141, _ = extract_all_141_features(old.delta_b, old.sensor_x)
    fw, _ = extract_signal_width_features(old.delta_b, old.sensor_x)
    x180 = np.hstack([f141, fw])
    wdesc = load_weff_descriptor(PHASE14D_MODEL)
    we_mean, we_std = weff_area_descriptor_predict(wdesc, x180)
    we_r80 = float(wdesc["plain_conformal_radius_m"]["80"]); we_r90 = float(wdesc["plain_conformal_radius_m"]["90"])
    thr = float(wdesc["risk_flag_threshold_mm"])
    weff_targets = compute_width_targets_from_depth_maps(old.profile_depth_map_xy_m, old.rbc_params)
    wt = weff_targets["W_eff_area"][test_idx]
    wm_t, ws_t = we_mean[test_idx], we_std[test_idx]
    weff_mae = mae_mm(wm_t, wt)
    weff_cov80 = float(np.mean(np.abs(wm_t - wt) <= we_r80))
    weff_cov90 = float(np.mean(np.abs(wm_t - wt) <= we_r90))
    flags = ws_t * 1000.0 >= thr

    reports, schemas = [], []
    for i, sid in enumerate(sample_ids):
        weff_block = {
            "target": "W_eff_area", "width_descriptor_status": "report_only",
            "W_eff_area_pred_mm": float(wm_t[i] * 1000.0),
            "W_eff_area_ci80_lower_mm": float((wm_t[i] - we_r80) * 1000.0),
            "W_eff_area_ci80_upper_mm": float((wm_t[i] + we_r80) * 1000.0),
            "W_eff_area_ci90_lower_mm": float((wm_t[i] - we_r90) * 1000.0),
            "W_eff_area_ci90_upper_mm": float((wm_t[i] + we_r90) * 1000.0),
            "W_eff_area_uncertainty_std_mm": float(ws_t[i] * 1000.0),
            "W_eff_area_risk_flag": bool(flags[i]),
            "W_eff_area_risk_reason": "ensemble_std_large_error_risk",
        }
        r = build_report(str(sid), float(l_mean[i]), float(l_std[i]), l_r80, l_r90,
                         float(off[i, 1]), float(off[i, 2]), weff_block)
        s = validate_report(r)
        r["schema_validation"] = s
        reports.append(r); schemas.append(s["passed"])

    prediction_m = np.column_stack([l_mean, off[:, 1:3]])
    dimension_metrics = composite_metrics(prediction_m, y[test_idx, :3])
    w_mae, d_mae = dimension_metrics["W_mae_mm"], dimension_metrics["D_mae_mm"]
    composite_mean = dimension_metrics["composite_mean_lwd_mm"]
    prediction_rows = []
    for i, sid in enumerate(sample_ids):
        row = {"sample_id": str(sid), "split": "test"}
        for j, axis in enumerate(("L", "W", "D")):
            row[f"{axis}_true_mm"] = float(y[test_idx[i], j] * 1000.0)
            row[f"{axis}_pred_mm"] = float(prediction_m[i, j] * 1000.0)
            row[f"{axis}_absolute_error_mm"] = float(abs(prediction_m[i, j] - y[test_idx[i], j]) * 1000.0)
        prediction_rows.append(row)
    write_csv(PREDICTIONS_CSV, prediction_rows)
    repro_checks = {
        "L_mae_reproduced": abs(l_mae - REPRO["L_mae_mm"]) <= TOL,
        "W_mae_reproduced": abs(w_mae - REPRO["W_mae_mm"]) <= TOL,
        "D_mae_reproduced": abs(d_mae - REPRO["D_mae_mm"]) <= TOL,
        "L_cov80_reproduced": abs(l_cov80 - REPRO["L_cov80"]) <= TOL,
        "L_cov90_reproduced": abs(l_cov90 - REPRO["L_cov90"]) <= TOL,
        "weff_mae_reproduced": abs(weff_mae - REPRO["weff_mae_mm"]) <= TOL,
        "weff_cov_reproduced": abs(weff_cov80 - REPRO["weff_cov80"]) <= TOL and abs(weff_cov90 - REPRO["weff_cov90"]) <= TOL,
        "weff_flag_count_reproduced": int(flags.sum()) == REPRO["weff_flag_count"],
        "composite_mean_reproduced": abs(composite_mean - REPRO["composite_mean_lwd_mm"]) <= TOL,
    }
    gate = gate17a(len(reports), all(schemas), repro_checks)

    REPORTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with REPORTS_JSONL.open("w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = {
        "phase": "phase17a_composite_reporting_layer", "gate17a": gate,
        "evaluation": {"dataset_id": V3_240_DATASET_ID, "split": "test", "sample_count": len(test_idx),
                       "prediction_evidence": str(PREDICTIONS_CSV.relative_to(ROOT)).replace("\\", "/"),
                       "aggregation": "mean of L/W/D MAE calculated from the same sample predictions",
                       "dataset_npz_sha256": old.manifest["npz_sha256"],
                       "model_sha256": {path.name: sha256_file(path) for path in (L_ENSEMBLE_JOBLIB, PHASE10A_OFFICIAL_MODEL, PHASE14D_MODEL)}},
        "numbers": {"W_mae_mm": w_mae, "D_mae_mm": d_mae, "L_mae_mm": l_mae, "L_cov80": l_cov80, "L_cov90": l_cov90,
                    "weff_mae_mm": weff_mae, "weff_flag_count": int(flags.sum()),
                    "composite_mean_lwd_mm": composite_mean},
        "supersedes": "results/reports/phase14e_weff_area_test_reports.jsonl (L field now composite)",
        "reports_jsonl": str(REPORTS_JSONL.relative_to(ROOT)).replace("\\", "/"),
        "boundary": {"comsol_run": False, "training_performed": False, "current_baseline_update": False},
    }
    METRICS_JSON.write_text(json_dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_csv(MANIFEST_CSV, [{
        "phase": metrics["phase"], "verdict": gate["verdict"],
        "n_reports": str(len(reports)), "L_mae_mm": "%.6f" % l_mae,
        "W_mae_mm": "%.6f" % w_mae, "D_mae_mm": "%.6f" % d_mae,
        "composite_mean_lwd_mm": "%.6f" % composite_mean,
        "comsol_run": "False", "training_performed": "False", "current_baseline_update": "False",
    }])
    print(json_dumps({"gate17a": gate["verdict"], "n_reports": len(reports),
                      "L_mae_mm": l_mae, "composite_mean_lwd_mm": composite_mean,
                      "weff_flag_count": int(flags.sum())}))
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
