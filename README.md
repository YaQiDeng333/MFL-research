# MFL-research：漏磁缺陷几何反演

本仓库整理三轴漏磁信号反演缺陷几何的已完成研究，包含尺寸预测、三维轮廓重建、有效宽度描述符及其结果记录。它是从原研究工程中筛选出的公开代码与成果快照。

输入为缺陷相对于无缺陷状态的漏磁变化 `delta_b = b_defect - b_no_defect`，包含 Bx、By、Bz 三个分量。每个样本有 3 条扫描线、每条 201 个采样点，数组形状为 `(N, 3, 3, 201)`。数据来自 COMSOL 的三维闭合网格实体仿真；RBC-style 几何是工程近似，尚未在真实试样上验证。

尺寸模型采用 ExtraTrees，轮廓模型采用 Conv1D 编码器和六参数回归头。

## 已完成成果

数据集 `v3_240` 有 240 个样本，训练 / 验证 / 测试划分为 162 / 39 / 39。L 头另使用 90 个训练扩充样本。模型选择和区间、风险阈值校准使用验证集，测试集用于最终评估。

| 成果 | 已记录结果 | 证据 |
|---|---|---|
| L/W/D 尺寸组合模型 | MAE：1.626280 / 2.095925 / 0.860158 mm，平均 1.527454 mm | [统一评估记录](results/metrics/phase17a_composite_reporting_layer_metrics.json)、[逐样本预测与误差](results/metrics/composite_unified_test_predictions.csv) |
| 三维轮廓模型（20.77 / 20.85） | 深度 RMSE 0.387737 mm，投影 mask IoU 0.750650、Dice 0.847727 | [轮廓结果摘要](results/summaries/true_3d_rbc_formal_benchmark_20_77_summary.txt)、[种子结果](results/metrics/true_3d_rbc_formal_benchmark_20_77_seed_summary.csv) |
| 有效面积宽度 W_eff_area | 测试 MAE 1.444351 mm；80% / 90% 区间覆盖率 0.897436 / 0.948718 | [描述符报告摘要](results/summaries/phase14e_weff_area_reporting_layer_summary.md) |

尺寸组合模型的 L 来自 12 个 ExtraTrees-600 模型的均值，W/D 来自既有 ExtraTrees-400 模型。2026-10-20 固定模型，在同一批 39 个冻结测试样本上重新提取特征、推理并统一计算三个维度的 MAE；平均误差为三项实算 MAE 的算术平均。L 头使用 210 维信号特征，W/D 使用固定的 141 维重构特征。表中数值是本仓库唯一公开尺寸基准。

轮廓模型的六个参数为 `L_m, W_m, D_m, wLD, wWD, wLW`；后三项是几何形状辅助参数。

尺寸结果经过上述统一重载评估，未重新训练、调参或改变测试划分；轮廓及有效宽度结果保留各自独立的成果作用域。

## 代码入口

文件保留阶段名称，便于与结果文件对应。以下是全部 12 个研究代码文件。

| 文件 | 用途 |
|---|---|
| [load_true_3d_rbc_pilot_dataset.py](scripts/load_true_3d_rbc_pilot_dataset.py) | 数据登记与哈希检查、训练集归一化、六参数转深度表面、预测指标 |
| [train_phase11a_ld_targeted_retraining.py](scripts/train_phase11a_ld_targeted_retraining.py) | 信号特征、训练扩充样本读取与尺寸候选训练 |
| [train_phase14a_observable_width_conditioned_models.py](scripts/train_phase14a_observable_width_conditioned_models.py) | 摘出的公共宽度特征、标签、模型和区间函数；已移除旧实验主流程 |
| [eval_phase16a_l_upgrade_composite_candidate.py](scripts/eval_phase16a_l_upgrade_composite_candidate.py) | 摘出的 L 头拟合函数和固定参数；已移除旧阶段审计入口 |
| [train_phase16b_l_head_ensemble_descriptor.py](scripts/train_phase16b_l_head_ensemble_descriptor.py) | L 头集成训练与验证集区间校准 |
| [train_phase14c_global_weff_area_descriptor.py](scripts/train_phase14c_global_weff_area_descriptor.py) | 全局有效宽度描述符、区间校准与风险统计 |
| [train_phase14d_weff_area_ensemble_risk_flag.py](scripts/train_phase14d_weff_area_ensemble_risk_flag.py) | 有效宽度集成模型和风险旗标 |
| [build_phase14e_weff_area_reporting_layer.py](scripts/build_phase14e_weff_area_reporting_layer.py) | 官方尺寸特征重构、模型预测、有效宽度报告 |
| [build_phase17a_composite_reporting_layer.py](scripts/build_phase17a_composite_reporting_layer.py) | L/W/D 组合预测和报告校验 |
| [train_true_3d_rbc_neural_parameter_gate.py](scripts/train_true_3d_rbc_neural_parameter_gate.py) | 摘出的 Conv1D 模型、训练、预测与指标函数 |
| [run_true_3d_rbc_formal_benchmark_20_77_candidate.py](scripts/run_true_3d_rbc_formal_benchmark_20_77_candidate.py) | 轮廓基准训练与评估入口 |
| [export_true_3d_rbc_baseline_inference_artifact.py](scripts/export_true_3d_rbc_baseline_inference_artifact.py) | 按固定 seed=42 协议重训轮廓模型，并保存后回读校验 checkpoint |

`results/metrics/` 保存统一尺寸评估和精选指标；`results/manifests/` 保存数据合同与训练扩充样本状态清单。

## 安装与独立测试

发布验证环境为 Python 3.10.20，直接依赖版本见 [requirements.txt](requirements.txt)。在仓库根目录运行：

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

测试使用合成小数组和临时文件，覆盖信号特征、训练/验证/测试隔离、宽度计算、区间与风险阈值、报告结构、轮廓模型及相对路径解析。

## 运行研究代码的条件

复现表中结果需准备以下文件：

| 用途 | 需要另外准备的文件 |
|---|---|
| 轮廓训练与 checkpoint 恢复 | [数据登记](COMSOL_DATA_REGISTRY.md) 指定的 v3_240 NPZ，须匹配 manifest 中的 SHA-256 |
| L 头与有效宽度训练 | v3_240 NPZ，以及 `data/processed/phase11a_ld_targeted_expansion_delta_b.npz`；L 头还读取 Phase 11a 的候选模型对象 |
| 尺寸组合报告 | `results/models/` 下的 `phase16b_l_head_ensemble_descriptor.joblib`、`phase10a_w_residual_corrected_candidate.joblib`、`phase14d_weff_area_ensemble_descriptor.joblib`，以及对应数据 |

当前公开代码提供既有 W/D 模型的特征重构与加载预测路径；尺寸组合报告使用对应的官方 W/D 模型。旧 `.joblib` 对象可能要求与原训练一致的软件版本。

查看带参数入口的使用方式不会启动训练：

```bash
python scripts/run_true_3d_rbc_formal_benchmark_20_77_candidate.py --help
python scripts/train_phase11a_ld_targeted_retraining.py --help
python scripts/train_phase14d_weff_area_ensemble_risk_flag.py --help
python scripts/export_true_3d_rbc_baseline_inference_artifact.py --help
```

补齐文件后，带参数入口可指定输出位置；轮廓基准使用固定结果路径且默认拒绝覆盖已有结果。L 头训练和组合报告入口也使用固定路径，实际运行前应在独立副本中安排输出，避免覆盖随仓库提供的历史证据。
