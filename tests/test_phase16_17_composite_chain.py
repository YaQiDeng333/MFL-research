from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))








def _report(mod):
    weff_block = {"target": "W_eff_area", "width_descriptor_status": "report_only",
                  "W_eff_area_pred_mm": 10.0, "W_eff_area_risk_flag": False,
                  "W_eff_area_risk_reason": "ensemble_std_large_error_risk"}
    return mod.build_report("sid_001", 0.020, 0.0002, 0.0046, 0.0058, 0.011, 0.003, weff_block)


def test_17a_report_builder_fields_and_interval_nesting() -> None:
    mod = importlib.import_module("build_phase17a_composite_reporting_layer")
    r = _report(mod)
    block = r["official_composite_baseline"]
    for f in mod.REQUIRED_FIELDS:
        assert f in block, f
    assert np.isclose(block["L_m_pred_mm"], 20.0)
    assert np.isclose(block["W_m_pred_mm"], 11.0)
    assert block["L_m_ci80_lower_mm"] < block["L_m_pred_mm"] < block["L_m_ci80_upper_mm"]
    assert block["L_m_ci90_lower_mm"] <= block["L_m_ci80_lower_mm"]
    assert r["descriptor_participates_in_lwd_mean_mae"] is False
    result = mod.validate_report(r)
    assert result["passed"] is True and result["n_failed"] == 0


def test_17a_validate_report_flags_broken_reports() -> None:
    mod = importlib.import_module("build_phase17a_composite_reporting_layer")
    r = _report(mod)
    del r["official_composite_baseline"]["L_m_ci80_lower_mm"]
    r["width_descriptor"]["width_descriptor_status"] = "primary"
    result = mod.validate_report(r)
    assert result["passed"] is False and result["n_failed"] >= 2


def test_composite_metrics_use_all_three_prediction_columns():
    mod = importlib.import_module("build_phase17a_composite_reporting_layer")
    truth = np.array([[0.020, 0.010, 0.003], [0.030, 0.012, 0.004]])
    assert mod.composite_metrics(truth, truth)["composite_mean_lwd_mm"] == 0.0
    prediction = truth + np.array([0.001, -0.002, 0.003])
    result = mod.composite_metrics(prediction, truth)
    assert np.allclose([result["L_mae_mm"], result["W_mae_mm"], result["D_mae_mm"]], [1, 2, 3])
    assert np.isclose(result["composite_mean_lwd_mm"], 2.0)


def test_public_metrics_match_per_sample_prediction_evidence():
    import csv
    import json
    mod = importlib.import_module("build_phase17a_composite_reporting_layer")
    with (ROOT / "results/metrics/composite_unified_test_predictions.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 39 and len({r["sample_id"] for r in rows}) == 39
    assert {r["split"] for r in rows} == {"test"}
    predictions = np.array([[float(r[f"{axis}_pred_mm"]) for axis in ("L", "W", "D")] for r in rows]) / 1000
    truth = np.array([[float(r[f"{axis}_true_mm"]) for axis in ("L", "W", "D")] for r in rows]) / 1000
    computed = mod.composite_metrics(predictions, truth)
    recorded = json.loads((ROOT / "results/metrics/phase17a_composite_reporting_layer_metrics.json").read_text(encoding="utf-8"))["numbers"]
    for key, value in computed.items():
        assert abs(recorded[key] - value) < 1e-10
