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
    return importlib.import_module("train_phase14c_global_weff_area_descriptor")


def test_split_conformal_radius_uses_finite_sample_order_statistic() -> None:
    mod = _module()
    residuals = np.arange(1.0, 11.0)  # 1..10, n=10
    # k = ceil(11 * 0.8) = 9 -> 9th smallest = 9.0
    assert mod.split_conformal_radius(residuals, 0.80) == 9.0
    # k = ceil(11 * 0.95) = 11 > n -> infinite radius
    assert math.isinf(mod.split_conformal_radius(residuals, 0.95))
    assert math.isinf(mod.split_conformal_radius(np.asarray([]), 0.8))


def test_scaled_conformal_quantile_is_ratio_order_statistic() -> None:
    mod = _module()
    residuals = np.asarray([1.0, 4.0, 9.0, 16.0])
    scales = np.asarray([1.0, 2.0, 3.0, 4.0])
    # ratios = [1, 2, 3, 4], n=4, k = ceil(5*0.5) = 3 -> 3.0
    assert mod.scaled_conformal_quantile(residuals, scales, 0.5) == 3.0


def test_rank_auc_perfect_separation_and_ties() -> None:
    mod = _module()
    scores = np.asarray([1.0, 2.0, 3.0, 4.0])
    positive = np.asarray([False, False, True, True])
    assert mod.rank_auc(scores, positive) == 1.0
    tied = np.asarray([5.0, 5.0, 5.0, 5.0])
    assert mod.rank_auc(tied, positive) == 0.5
    assert math.isnan(mod.rank_auc(scores, np.asarray([True, True, True, True])))


def test_spearman_corr_monotone_and_anticorrelated() -> None:
    mod = _module()
    a = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(mod.spearman_corr(a, a * 3.0 + 1.0), 1.0)
    assert np.isclose(mod.spearman_corr(a, -a), -1.0)
