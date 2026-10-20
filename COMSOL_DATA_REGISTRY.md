# 数据登记

此登记描述已完成基准的数据合同；NPZ 不随仓库提供。路径相对仓库根目录。

## comsol_true_3d_rbc_imported_watertight_pilot_v3_240

- dataset_role: assembled
- status: pilot_generated
- route: true_3d_piao_style
- stage: 20.76
- schema_version: true3d_profile_v1_piao_rbc
- geometry_method: imported_watertight_mesh_solid
- exact_piao_rbc: false
- rbc_style_approximation: true
- path: `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.npz`
- manifest_path: `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json`
- n_samples: 240
- split_counts: {'val': 39, 'test': 39, 'train': 162}
- curvature_counts: {'sharp': 48, 'round': 49, 'boxy': 47, 'LD_dominant': 46, 'WD_dominant': 50}
- train_ready_candidate: true
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, explicit_pilot_training_gate
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: comsol_true_3d_rbc_imported_watertight_pilot_v2_120, comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76
- generator_script: `scripts/generate_mfl_true_3d_rbc_dataset_240_topup_pack.py`
- validation_script: `scripts/validate_true_3d_rbc_dataset_240_pack.py`
- npz_sha256: 6cfd6c4dd32c474dbe4843ac520f541123a54a8f0082a94d3e28d0c8cd196817
- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.
