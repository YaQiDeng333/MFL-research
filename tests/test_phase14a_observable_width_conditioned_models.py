from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_width_targets_from_depth_map_are_deterministic_y_widths() -> None:
    train = importlib.import_module("train_phase14a_observable_width_conditioned_models")
    depth = np.zeros((2, 10, 20), dtype=np.float64)
    depth[0, 2:8, 4:14] = 0.004
    depth[1, 3:7, 5:15] = 0.002
    params = np.asarray(
        [
            [0.010, 0.006, 0.004, 1.0, 1.0, 1.0],
            [0.010, 0.004, 0.002, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    targets = train.compute_width_targets_from_depth_maps(
        depth,
        params,
        x_span_m=(-0.010, 0.010),
        y_span_m=(-0.005, 0.005),
    )

    assert list(targets.keys()) == train.WIDTH_TARGET_NAMES
    assert np.allclose(targets["W_m"], [0.006, 0.004])
    assert np.allclose(targets["W_eff_mask_50"], [0.006, 0.004])
    assert np.allclose(targets["W_eff_mask_20"], [0.006, 0.004])
    assert np.allclose(targets["W_eff_area"], [0.006, 0.004])
    assert np.all(targets["W_eff_moment"] > 0.0)
    assert targets["W_eff_moment"][0] > targets["W_eff_moment"][1]


def test_train_only_wWD_bins_do_not_use_val_or_test_thresholds() -> None:
    train = importlib.import_module("train_phase14a_observable_width_conditioned_models")
    values = np.asarray([0.80, 0.90, 1.10, 1.30, 8.00, 9.00], dtype=np.float64)
    train_idx = np.asarray([0, 1, 2, 3])

    labels, thresholds = train.assign_quantile_bins_from_train(values, train_idx)

    assert labels.tolist() == ["low", "low", "mid", "high", "high", "high"]
    assert max(thresholds) < 2.0


def test_posterior_interval_metrics_report_coverage_and_width_mm() -> None:
    train = importlib.import_module("train_phase14a_observable_width_conditioned_models")
    truth = np.asarray([0.010, 0.012, 0.014], dtype=np.float64)
    lo = np.asarray([0.009, 0.011, 0.013], dtype=np.float64)
    hi = np.asarray([0.011, 0.013, 0.0135], dtype=np.float64)

    metrics = train.interval_metrics(truth, lo, hi)

    assert metrics["coverage"] == 2.0 / 3.0
    assert np.isclose(metrics["median_width_mm"], 2.0)
    assert np.isclose(metrics["mean_width_mm"], (2.0 + 2.0 + 0.5) / 3.0)
