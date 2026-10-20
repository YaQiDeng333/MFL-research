from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _module():
    return importlib.import_module("train_phase14d_weff_area_ensemble_risk_flag")




def test_ensemble_mean_std_shapes_and_floor() -> None:
    mod = _module()
    members = np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    mean, std = mod.ensemble_mean_std(members)
    assert np.allclose(mean, [2.0, 2.0, 2.0])
    assert np.allclose(std, [1.0, 1.0e-12, 1.0])
    try:
        mod.ensemble_mean_std(np.asarray([1.0, 2.0]))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_assess_ensemble_std_risk_flag_informative_case() -> None:
    mod = _module()
    err = np.linspace(0.1, 3.0, 39)
    std_test = err.copy()  # perfectly informative uncertainty
    std_val = np.linspace(0.1, 3.0, 39)
    high = err >= np.quantile(err, 0.75)

    risk = mod.assess_ensemble_std_risk_flag(std_val, std_test, high, err)

    assert risk["high_nuisance_auc"] == 1.0
    assert risk["large_width_error_auc"] == 1.0
    assert np.isclose(risk["spearman_std_vs_abs_error"], 1.0)
    assert risk["flagged_minus_unflagged_mm"] > 0.0
    assert 0 < risk["test_flagged_count"] < 39


def test_assess_ensemble_std_risk_flag_threshold_comes_from_val_only() -> None:
    mod = _module()
    err = np.linspace(0.1, 3.0, 20)
    std_test = np.linspace(0.1, 3.0, 20)
    high = err >= np.quantile(err, 0.75)
    low_val = np.full(20, 0.05)   # low val stds -> low threshold -> everything flagged
    risk_low = mod.assess_ensemble_std_risk_flag(low_val, std_test, high, err)
    high_val = np.full(20, 99.0)  # high val stds -> nothing flagged
    risk_high = mod.assess_ensemble_std_risk_flag(high_val, std_test, high, err)
    assert risk_low["test_flagged_count"] == 20
    assert risk_high["test_flagged_count"] == 0
    assert math.isnan(risk_high["flagged_mae_mm"])
