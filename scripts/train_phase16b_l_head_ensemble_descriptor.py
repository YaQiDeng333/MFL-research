#!/usr/bin/env python
"""Phase 16b: L-head ensemble descriptor package.

Turns the Phase 16a TRANSITION_AUDIT_READY L upgrade into a deployable unit:
12-seed ExtraTrees ensemble (11a feature world, train = v3 162 + 11a 90),
ensemble-mean point estimate, val-only plain split conformal 80/90 intervals,
ensemble-std risk signal, serialized joblib.

No COMSOL. No CURRENT_BASELINE.md change. Frozen v3 test, final eval only.
"""

from __future__ import annotations

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

from load_true_3d_rbc_pilot_dataset import V3_240_DATASET_ID, load_dataset, split_indices  # noqa: E402
from train_phase14a_observable_width_conditioned_models import (  # noqa: E402
    interval_metrics,
    json_dumps,
    load_phase11a_width_pack,
    write_csv,
)
from train_phase14c_global_weff_area_descriptor import split_conformal_radius, rank_auc, spearman_corr  # noqa: E402
from eval_phase16a_l_upgrade_composite_candidate import (  # noqa: E402
    PHASE11A_CANDIDATE,
    PHASE11A_MANIFEST,
    PHASE11A_NPZ,
    SEEDS,
    fit_l_head,
    mae_mm,
)

METRICS_JSON = ROOT / "results/metrics/phase16b_l_head_ensemble_descriptor_metrics.json"
MANIFEST_CSV = ROOT / "results/manifests/phase16b_l_head_ensemble_descriptor_manifest.csv"
MODEL_PATH = ROOT / "results/models/phase16b_l_head_ensemble_descriptor.joblib"

GATE_L_MAE_MM = 1.70
GATE_COV80 = 0.80
GATE_COV90 = 0.90


def main() -> int:
    import joblib

    old = load_dataset(V3_240_DATASET_ID)
    splits = split_indices(old)
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    pack = load_phase11a_width_pack(PHASE11A_NPZ, PHASE11A_MANIFEST)
    cand = joblib.load(PHASE11A_CANDIDATE)["candidate"]
    x_old = np.asarray(cand.x_old, dtype=np.float64)
    x_exp = np.asarray(cand.x_expansion, dtype=np.float64)
    y_L = np.asarray(old.rbc_params[:, 0], dtype=np.float64)
    y_L_exp = np.asarray(pack.rbc_params[:, 0], dtype=np.float64)

    x_train = np.vstack([x_old[train_idx], x_exp])
    y_train = np.concatenate([y_L[train_idx], y_L_exp])

    members = []
    pv, pt = [], []
    for seed in SEEDS:
        m = fit_l_head(x_train, y_train, seed)
        members.append(m)
        pv.append(np.asarray(m.predict(x_old[val_idx]), dtype=np.float64))
        pt.append(np.asarray(m.predict(x_old[test_idx]), dtype=np.float64))
    pv = np.vstack(pv); pt = np.vstack(pt)
    mean_val, std_val = pv.mean(axis=0), np.maximum(pv.std(axis=0), 1e-12)
    mean_test, std_test = pt.mean(axis=0), np.maximum(pt.std(axis=0), 1e-12)

    ens_val_mae = mae_mm(mean_val, y_L[val_idx])
    ens_test_mae = mae_mm(mean_test, y_L[test_idx])
    abs_resid_val = np.abs(mean_val - y_L[val_idx])
    conformal = {}
    radii = {}
    for cov in (0.80, 0.90):
        key = str(int(cov * 100))
        r = split_conformal_radius(abs_resid_val, cov)
        radii[key] = float(r)
        conformal[key] = {"radius_mm": r * 1000.0,
                          **interval_metrics(y_L[test_idx], mean_test - r, mean_test + r)}

    err_mm = np.abs(mean_test - y_L[test_idx]) * 1000.0
    large = err_mm >= np.quantile(err_mm, 0.75)
    risk = {
        "spearman_std_vs_abs_error": float(spearman_corr(std_test * 1000.0, err_mm)),
        "large_error_auc": float(rank_auc(std_test * 1000.0, large)),
    }

    cov80, cov90 = conformal["80"]["coverage"], conformal["90"]["coverage"]
    gate_pass = bool(ens_test_mae <= GATE_L_MAE_MM and cov80 >= GATE_COV80 and cov90 >= GATE_COV90)
    verdict = "PASS" if gate_pass else "FAIL"

    metrics = {
        "phase": "phase16b_l_head_ensemble_descriptor",
        "boundary": {"comsol_run": False, "current_baseline_update": False,
                     "test_used_for_selection_or_calibration": False,
                     "phase11a_train_side_only": True, "probe_samples_in_training": False},
        "ensemble": {"seeds": SEEDS, "n_members": len(SEEDS),
                     "spec": "ExtraTrees-600 mf0.75, 11a 210-dim feature world, train 162+90"},
        "point_estimate": {
            "ensemble_mean_val_mae_mm": ens_val_mae,
            "ensemble_mean_test_mae_mm": ens_test_mae,
            "phase16a_12seed_mean_of_maes_mm": 1.6276,
            "official_stored_L_mae_mm": 1.987,
        },
        "conformal_val_calibrated": conformal,
        "risk_signal": risk,
        "gate16b": {
            "verdict": verdict,
            "checks": {"L_mae_le_1p70": ens_test_mae <= GATE_L_MAE_MM,
                       "cov80_ge_0p80": cov80 >= GATE_COV80,
                       "cov90_ge_0p90": cov90 >= GATE_COV90},
            "thresholds": {"L_mae_mm": GATE_L_MAE_MM, "cov80": GATE_COV80, "cov90": GATE_COV90},
        },
        "model_artifact": str(MODEL_PATH),
    }
    METRICS_JSON.write_text(json_dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_csv(MANIFEST_CSV, [{
        "phase": "phase16b_l_head_ensemble_descriptor", "verdict": verdict,
        "ens_mean_test_mae_mm": "%.6f" % ens_test_mae,
        "cov80": "%.6f" % cov80, "cov90": "%.6f" % cov90,
        "width80_mm": "%.6f" % conformal["80"]["mean_width_mm"],
        "width90_mm": "%.6f" % conformal["90"]["mean_width_mm"],
        "comsol_run": "False", "current_baseline_update": "False",
    }])
    joblib.dump({
        "descriptor": "l_head_ensemble_descriptor",
        "ensemble_members": members, "seeds": SEEDS,
        "feature_world": "phase11a_210dim(all_141_plus_ld_features)",
        "plain_conformal_radius_m": radii,
        "metadata": {"phase": metrics["phase"], "gate16b": metrics["gate16b"]},
    }, MODEL_PATH)
    print(json_dumps({"gate16b": verdict, "ens_mean_test_mae_mm": ens_test_mae,
                      "cov80": cov80, "cov90": cov90,
                      "width80_mm": conformal["80"]["mean_width_mm"],
                      "width90_mm": conformal["90"]["mean_width_mm"],
                      "risk": risk}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
