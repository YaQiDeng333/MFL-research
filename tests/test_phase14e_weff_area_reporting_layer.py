from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _module():
    return importlib.import_module("build_phase14e_weff_area_reporting_layer")


def _good_report(mod):
    return mod.build_sample_report(
        sample_id="rbc_test_001",
        official_lwd_m=np.asarray([0.020, 0.011, 0.003]),
        weff_mean_m=0.012,
        weff_std_m=0.0002,
        radius_80_m=0.0029,
        radius_90_m=0.0037,
        risk_threshold_mm=0.1481,
        weff_true_m=0.0125,
    )




def test_build_sample_report_has_all_required_fields():
    mod = _module()
    report = _good_report(mod)
    wd = report["width_descriptor"]
    for field in mod.REQUIRED_WIDTH_DESCRIPTOR_FIELDS:
        assert field in wd, field
    assert wd["width_descriptor_status"] == "report_only"
    assert wd["W_eff_area_risk_reason"] == "ensemble_std_large_error_risk"
    assert isinstance(wd["W_eff_area_risk_flag"], bool)


def test_build_sample_report_preserves_official_w_m_and_excludes_from_lwd():
    mod = _module()
    report = _good_report(mod)
    off = report["official_lwd_baseline"]
    assert off["w_m_is_official_nominal_width"] is True
    assert off["w_m_replaced_by_w_eff_area"] is False
    assert abs(off["W_m_pred_mm"] - 11.0) < 1e-9  # 0.011 m -> 11 mm, not the W_eff_area value (12 mm)
    assert report["width_descriptor"]["W_eff_area_pred_mm"] != off["W_m_pred_mm"]
    assert report["descriptor_participates_in_lwd_mean_mae"] is False


def test_risk_flag_threshold_semantics():
    mod = _module()
    flagged = mod.build_sample_report("s", np.asarray([0.02, 0.011, 0.003]), 0.012,
                                      0.0003, 0.0029, 0.0037, 0.1481, 0.012)
    unflagged = mod.build_sample_report("s", np.asarray([0.02, 0.011, 0.003]), 0.012,
                                        0.0001, 0.0029, 0.0037, 0.1481, 0.012)
    assert flagged["width_descriptor"]["W_eff_area_risk_flag"] is True   # 0.3mm >= 0.1481
    assert unflagged["width_descriptor"]["W_eff_area_risk_flag"] is False  # 0.1mm < 0.1481


def test_validate_report_schema_passes_good_and_flags_bad():
    mod = _module()
    good = mod.validate_report_schema(_good_report(mod))
    assert good["passed"] is True
    assert good["n_failed"] == 0

    bad = _good_report(mod)
    del bad["width_descriptor"]["W_eff_area_ci80_lower_mm"]
    bad["official_lwd_baseline"]["w_m_replaced_by_w_eff_area"] = True
    res = mod.validate_report_schema(bad)
    assert res["passed"] is False
    assert res["n_failed"] >= 2


def test_validate_report_schema_catches_inverted_intervals():
    mod = _module()
    bad = _good_report(mod)
    bad["width_descriptor"]["W_eff_area_ci80_lower_mm"] = 99.0  # lower > upper
    res = mod.validate_report_schema(bad)
    names = {c["name"]: c["passed"] for c in res["checks"]}
    assert names["ci80_lower_le_upper"] is False
    assert res["passed"] is False
