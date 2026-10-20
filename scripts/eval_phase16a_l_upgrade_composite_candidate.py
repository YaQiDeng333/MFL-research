# 公共依赖摘录：保留原算法定义，移除未被发布入口调用的历史实验流程。

from __future__ import annotations

import json

import sys

from pathlib import Path

from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import train_phase11a_ld_targeted_retraining as p11  # noqa: E402

import __main__ as _M  # noqa: E402

from load_true_3d_rbc_pilot_dataset import (  # noqa: E402
    V3_240_DATASET_ID,
    load_dataset,
    split_indices,
)

from train_phase14a_observable_width_conditioned_models import (  # noqa: E402
    EXPECTED_V3_SPLITS,
    json_dumps,
    load_phase11a_width_pack,
    write_csv,
)

from build_phase14e_weff_area_reporting_layer import (  # noqa: E402
    PHASE10A_OFFICIAL_MODEL,
    load_official_lwd_baseline,
    official_lwd_predict_m,
)

PHASE11A_CANDIDATE = ROOT / "results/models/phase11a_ld_targeted_candidate.joblib"

PHASE11A_NPZ = ROOT / "data/processed/phase11a_ld_targeted_expansion_delta_b.npz"

PHASE11A_MANIFEST = ROOT / "results/manifests/phase11a_ld_targeted_execution_manifest.csv"

SEEDS = [42, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]

def mae_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b)) * 1000.0))

def fit_l_head(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=600, random_state=seed, min_samples_leaf=1,
        max_features=0.75, bootstrap=False, n_jobs=-1,
    ).fit(x_train, y_train)
