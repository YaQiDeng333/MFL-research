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

## 尺寸指标入口

本摘要仅记录 W_eff_area 描述符。完整 L/W/D 尺寸指标统一见 [组合评估记录](../metrics/phase17a_composite_reporting_layer_metrics.json)，不在描述符摘要中另设一套尺寸基准。W_eff_area 不替代 W_m，不参与 L/W/D 平均 MAE。

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

## Boundary Confirmation

- no COMSOL run
- no real-data input
- no Bz-only route
- no CURRENT_BASELINE.md update
- no model training (pure reload of Phase14d descriptor + official baseline)
- no true/predicted family routing, no mixture heads, no W_m residual/stacking
- W_m stays official nominal width; W_eff_area is report-only and excluded from L/W/D mean MAE
