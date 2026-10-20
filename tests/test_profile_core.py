from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import load_true_3d_rbc_pilot_dataset as data
import train_true_3d_rbc_neural_parameter_gate as neural


def test_normalization_ignores_validation_and_test_values():
    dataset = SimpleNamespace(
        split=np.array(['train', 'train', 'val', 'test']),
        x_channels=np.stack([np.full((9, 201), v) for v in [1., 3., 1000., 2000.]]),
        rbc_params=np.stack([np.full(6, v) for v in [1., 3., 1000., 2000.]]),
    )
    stats = data.train_normalization(dataset)
    assert np.allclose(stats['x_mean'], 2.)
    assert np.allclose(stats['x_std'], 1.)
    assert np.allclose(stats['y_mean'], 2.)
    assert np.allclose(data.denormalize_y(data.normalize_y(dataset, stats), stats), dataset.rbc_params)


def test_depth_surface_preserves_depth_and_scales_linearly():
    params = np.array([.020, .012, .003, 1., 1., 1.])
    depth = data.depth_grid_from_params(params)
    assert depth.shape == (33, 17)
    assert np.isfinite(depth).all() and (depth >= 0).all()
    assert np.isclose(depth.max(), params[2])
    doubled = params.copy()
    doubled[2] *= 2
    assert np.allclose(data.depth_grid_from_params(doubled), 2 * depth)
    assert np.allclose(depth, depth[::-1, :])
    assert np.allclose(depth, depth[:, ::-1])


def test_neural_training_does_not_select_on_test_labels():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(6, 9, 201)).astype(np.float32)
    y = rng.normal(size=(6, 6)).astype(np.float32)
    splits = {'train': np.array([0, 1, 2]), 'val': np.array([3, 4]), 'test': np.array([5])}
    args = SimpleNamespace(epochs=2, batch_size=3, lr=.003, weight_decay=.0001)
    first = neural.train_one_seed(42, x, y, splits, args)
    changed = y.copy()
    changed[5] = 1e6
    second = neural.train_one_seed(42, x, changed, splits, args)
    assert first['pred_norm'].shape == (6, 6)
    assert np.isfinite(first['pred_norm']).all()
    assert first['best_epoch'] == second['best_epoch']
    assert first['best_val_score'] == second['best_val_score']
    for key, value in first['model'].state_dict().items():
        assert torch.equal(value, second['model'].state_dict()[key])


def test_dataset_paths_resolve_from_checkout_not_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    entry, manifest, npz = data.resolve_dataset(data.V3_240_DATASET_ID)
    assert manifest['dataset_id'] == data.V3_240_DATASET_ID
    assert npz == data.ROOT / Path(manifest['npz_path'])
    assert (data.ROOT / entry['manifest_path']).is_file()
