from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))








def test_all_141_and_ld_feature_extractors_are_signal_only_and_finite() -> None:
    train = importlib.import_module("train_phase11a_ld_targeted_retraining")
    rng = np.random.default_rng(11)
    delta_b = rng.normal(size=(7, 3, 3, 201)).astype(np.float64)
    sensor_x = np.linspace(-0.04, 0.04, 201)

    all_141, all_names = train.extract_all_141_features(delta_b, sensor_x=sensor_x)
    ld_extra, ld_names = train.extract_ld_specific_features(delta_b, sensor_x=sensor_x)

    assert all_141.shape == (7, 141)
    assert len(all_names) == 141
    assert ld_extra.shape[0] == 7
    assert len(ld_names) == ld_extra.shape[1]
    assert np.isfinite(all_141).all()
    assert np.isfinite(ld_extra).all()
    forbidden = {"wLD", "wWD", "wLW", "skew", "flatness", "slope", "mode", "boxiness"}
    assert forbidden.isdisjoint(set(all_names))
    assert forbidden.isdisjoint(set(ld_names))


def test_training_indices_keep_old_test_frozen_and_new_samples_train_only() -> None:
    train = importlib.import_module("train_phase11a_ld_targeted_retraining")
    old_split = np.asarray(["train"] * 162 + ["val"] * 39 + ["test"] * 39)
    expansion_count = 91

    indices = train.build_training_index_policy(old_split, expansion_count)

    assert len(indices["original_train"]) == 162
    assert len(indices["original_val"]) == 39
    assert len(indices["original_test"]) == 39
    assert len(indices["augmented_train"]) == 162 + expansion_count
    assert max(indices["original_test"]) == 239
    assert min(indices["expansion_train"]) == 240
    assert set(indices["original_test"]).isdisjoint(set(indices["augmented_train"]))
