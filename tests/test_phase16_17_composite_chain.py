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
