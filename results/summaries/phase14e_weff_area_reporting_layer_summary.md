# Phase 14e W_eff_area Reporting-Layer Integration Summary

- Gate 14e verdict: `PASS`
- rationale: 39/39 reports built and schema-valid; W_eff_area MAE/coverage/flag reproduce Phase14d; W_m preserved from official baseline
- descriptor joined reporting layer: `True`
- reports generated: `39/39` (frozen v3 test); all schema-valid: `True`

## W_eff_area Descriptor Reproduction (report-only ground-truth eval)

- W_eff_area test MAE: `1.444351 mm` (Phase14d `1.444351 mm`)
- 80% coverage: `0.897436` (ref `0.897436`); 90%: `0.948718` (ref `0.948718`)
- risk flag count: `10/39` (ref `10`)
- flagged MAE `2.169448` vs unflagged `1.194317` mm (delta `+0.975131`)

## New Report Fields (per sample)

- `W_eff_area_pred_mm`
- `W_eff_area_ci80_lower_mm`
- `W_eff_area_ci80_upper_mm`
- `W_eff_area_ci90_lower_mm`
- `W_eff_area_ci90_upper_mm`
- `W_eff_area_uncertainty_std_mm`
- `W_eff_area_risk_flag`
- `W_eff_area_risk_reason`
- `width_descriptor_status`
