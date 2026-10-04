import json
import sys
import types

import numpy as np

# Keep this unit test focused on Dataset_mix initialization even in minimal envs.
sys.modules.setdefault("mediapy", types.ModuleType("mediapy"))
if "decord" not in sys.modules:
    decord_stub = types.ModuleType("decord")
    decord_stub.VideoReader = object
    decord_stub.cpu = lambda *_args, **_kwargs: None
    sys.modules["decord"] = decord_stub

from config import wm_orca_args
from dataset.dataset_orca import Dataset_mix
from utils.config_loader import load_experiment_config


def _write_meta(root, dataset_name, train_samples, val_samples=None):
    dataset_dir = root / dataset_name
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train_sample.json").write_text(json.dumps(train_samples))
    (dataset_dir / "val_sample.json").write_text(json.dumps(val_samples or train_samples))
    (dataset_dir / "stat.json").write_text(
        json.dumps({"state_01": [0.0, 0.0], "state_99": [1.0, 1.0]})
    )


def test_config_loader_preserves_episode_exclusion_mapping(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "dataset:\n"
        "  dataset_names: dataset_a\n"
        "  exclude_episode_ids_by_dataset:\n"
        "    dataset_a: [1, '0002']\n"
    )

    args = load_experiment_config(str(config_path), wm_orca_args())

    assert args.exclude_episode_ids_by_dataset == {"dataset_a": [1, "0002"]}


def test_dataset_mix_excludes_configured_episode_ids_per_dataset(tmp_path):
    meta_root = tmp_path / "meta"
    _write_meta(
        meta_root,
        "dataset_a",
        [
            {"episode_id": 1, "frame_ids": [0]},
            {"episode_id": 2, "frame_ids": [1]},
            {"episode_id": 2, "frame_ids": [2]},
        ],
    )
    _write_meta(
        meta_root,
        "dataset_b",
        [
            {"episode_id": "keep", "frame_ids": [0]},
            {"episode_id": "drop", "frame_ids": [1]},
        ],
    )

    args = wm_orca_args()
    args.dataset_root_path = str(tmp_path / "data")
    args.dataset_names = "dataset_a+dataset_b"
    args.dataset_meta_info_path = str(meta_root)
    args.prob = [0.5, 0.5]
    args.max_num_samples = 100
    args.max_num_samples_for_validation = 100
    args.exclude_episode_ids_by_dataset = {
        "dataset_a": [2],
        "dataset_b": ["drop"],
    }

    dataset = Dataset_mix(args, mode="train")

    assert dataset.samples_len == [1, 1]
    assert [sample["episode_id"] for sample in dataset.samples_all[0]] == [1]
    assert [sample["episode_id"] for sample in dataset.samples_all[1]] == ["keep"]


def test_dataset_mix_uses_dataset_stat_path_override(tmp_path):
    meta_root = tmp_path / "meta"
    _write_meta(
        meta_root,
        "dataset_a",
        [{"episode_id": 1, "frame_ids": [0]}],
    )
    override_stat_path = tmp_path / "shared_stat.json"
    override_stat_path.write_text(
        json.dumps({"state_01": [-2.0, -3.0], "state_99": [2.0, 3.0]})
    )

    args = wm_orca_args()
    args.dataset_root_path = str(tmp_path / "data")
    args.dataset_names = "dataset_a"
    args.dataset_meta_info_path = str(meta_root)
    args.dataset_stat_path = str(override_stat_path)
    args.max_num_samples = 100
    args.max_num_samples_for_validation = 100

    dataset = Dataset_mix(args, mode="train")

    state_p01, state_p99 = dataset.norm_all[0]
    assert state_p01.tolist() == [[-2.0, -3.0]]
    assert state_p99.tolist() == [[2.0, 3.0]]


def test_dataset_mix_loads_model_specific_stat_path_overrides(tmp_path):
    meta_root = tmp_path / "meta"
    _write_meta(
        meta_root,
        "dataset_a",
        [{"episode_id": 1, "frame_ids": [0]}],
    )
    wm1_stat_path = tmp_path / "wm1_stat.json"
    wm1_stat_path.write_text(
        json.dumps({"state_01": [-1.0, -1.0], "state_99": [1.0, 1.0]})
    )
    wm2_stat_path = tmp_path / "wm2_stat.json"
    wm2_stat_path.write_text(
        json.dumps({"state_01": [-2.0, -3.0], "state_99": [2.0, 3.0]})
    )

    args = wm_orca_args()
    args.dataset_root_path = str(tmp_path / "data")
    args.dataset_names = "dataset_a"
    args.dataset_meta_info_path = str(meta_root)
    args.wm1_dataset_stat_path = str(wm1_stat_path)
    args.wm2_dataset_stat_path = str(wm2_stat_path)
    args.max_num_samples = 100
    args.max_num_samples_for_validation = 100

    dataset = Dataset_mix(args, mode="train")

    wm1_p01, wm1_p99 = dataset.model_norm_all["wm1"][0]
    wm2_p01, wm2_p99 = dataset.model_norm_all["wm2"][0]
    assert wm1_p01.tolist() == [[-1.0, -1.0]]
    assert wm1_p99.tolist() == [[1.0, 1.0]]
    assert wm2_p01.tolist() == [[-2.0, -3.0]]
    assert wm2_p99.tolist() == [[2.0, 3.0]]


def test_dataset_mix_can_apply_model_specific_action_swaps_without_mutating_raw_sequence():
    dataset = object.__new__(Dataset_mix)
    raw = np.array([[10.0, 1.0, 2.0, 3.0]], dtype=np.float32)

    unswapped = dataset._maybe_swap_abd_with_mcp(raw, False)
    swapped = dataset._maybe_swap_abd_with_mcp(raw, True)

    assert unswapped is raw
    assert raw.tolist() == [[10.0, 1.0, 2.0, 3.0]]
    assert swapped.tolist() == [[10.0, 2.0, 1.0, 3.0]]
    assert raw.tolist() == [[10.0, 1.0, 2.0, 3.0]]
