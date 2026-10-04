"""Controllability evaluation: random target-state rollouts.

Two independent axes, both built here:
  - sampling_scope: "whole_pose" (all 23 dims randomized jointly, one full random
    target pose) or "per_dim" (one dim at a time, others held at the base pose --
    mirrors inference_wm1_to_wm2_sine_actions.py's per-dimension structure).
  - approach_style: "direct" (the target is presented as a step function from
    frame 0 -- the model must jump straight there) or "linear" (the action
    conditioning ramps from the base pose to the target, then holds).

Two modes, so a target catalog is generated once and shared across separate
baseline / WM1->WM2 invocations (identical targets -> a fair comparison):
  --mode targets   builds and saves a target catalog (no model loaded).
  --mode rollout   loads model(s), expands each target into direct/linear
                   rollouts, and writes rollout_manifest_{baseline,wm1wm2}.json.
  --mode both      runs targets then rollout in one process.

See faive_lab/scripts/wm_evaluation/render_controllability_targets.py for the
ground-truth side, which reads targets_manifest.json's `target_action_abs`
directly (hand dims already in canonical post-swap_abd_with_mcp order).
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

import argparse
import copy
import gc
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from accelerate import Accelerator
import mediapy

from config import wm_orca_args
from utils.config_loader import load_experiment_config
from dataset.dataset_orca import Dataset_mix

from scripts.inference_wm1_to_wm2_sine_actions import (
    denormalize_bound,
    normalize_bound,
    get_action_bounds_from_stat,
    get_current_observed_action_abs,
    get_model_stat_path,
    get_stat_path,
    make_video_grid_uint8,
    parse_float_list,
    parse_int_list,
    repeat_batch,
    resolve_dataset_name,
    resolve_sample_indices,
    run_autoregressive_baseline_wm_with_actions,
    run_autoregressive_wm1_wm2_with_actions,
    to_device_batch,
    _load_ctrl_world_from_checkpoint,
)
from scripts.inference_wm1_to_wm2_motion_suite import (
    ACTION_LABELS,
    HAND_JOINT_LABELS,
    _interpolate_hold,
    _repeat_base,
)


# ---------------------------------------------------------------------------
# Target sampling (pure, no model/GPU needed)
# ---------------------------------------------------------------------------

# Physical range-of-motion limits (radians) for the 17 ORCA hand joints, sourced from
# faive_lab's URDF (faive_lab/source/faive_lab/faive_lab/assets/data/orca_v1/retargeter/
# orca_v1.urdf <limit lower=".." upper=".."/> tags) and mapped to this project's
# canonical hand-joint order (HAND_JOINT_LABELS above) via the authoritative real->sim
# joint identity mapping in faive_lab's mdp/observations.py
# (_ORCA_TO_SIM_JOINT_NAME_MAP) -- NOT guessed from joint names, which are actively
# misleading here (real "thumb_abd" maps to the URDF joint literally named
# "root2thumb_base", not the more name-obvious "thumb_base2pp", which is "thumb_mcp").
# Order matches HAND_DIMS (global indices 6-22, i.e. ACTION_LABELS[6:23]): wrist,
# thumb_mcp, thumb_abd, thumb_pip, thumb_dip, index_abd, index_mcp, index_pip,
# middle_abd, middle_mcp, middle_pip, ring_abd, ring_mcp, ring_pip, pinky_abd,
# pinky_mcp, pinky_pip.
#
# Why this matters: confirmed empirically 2026-09-09 -- faive_lab's physics settle step
# frequently fails to converge (saturated_dims) on whole-pose targets sampled purely
# from stat.json's 1st/99th percentile of *recorded* real-world values, and the
# recurring offenders are almost entirely the four "_abd" (abduction) joints plus two
# thumb joints -- exactly the dims where the recorded/percentile range exceeds this
# mechanical range-of-motion table below (a known data-quality issue -- see
# [[faive-lab-isaac-sim-gotchas]] on motor-calibration-outlier stat.json bounds).
# Intersecting with the true mechanical limits directly fixes sampling at the source,
# rather than only detecting the resulting unreachable targets after the fact.
HAND_JOINT_LIMITS_RAD: Tuple[Tuple[float, float], ...] = (
    (-1.0471975511965976, 1.0471975511965976),  # wrist (tower2root)
    (-0.7853981633974483, 0.7853981633974483),  # thumb_mcp (thumb_base2pp)
    (-0.9250245035569946, 0.8377580409572782),  # thumb_abd (root2thumb_base)
    (-0.3490658503988659, 2.007128639793479),   # thumb_pip (thumb_pp2mp)
    (-0.3490658503988659, 1.7453292519943295),  # thumb_dip (thumb_mp2dp)
    (-0.5235987755982988, 0.5235987755982988),  # index_abd
    (-0.3490658503988659, 1.9198621771937625),  # index_mcp (root2index_pp)
    (-0.3490658503988659, 2.2689280275926285),  # index_pip (index_pp2mp)
    (-0.5235987755982988, 0.5235987755982988),  # middle_abd
    (-0.3490658503988659, 1.9198621771937625),  # middle_mcp (root2middle_pp)
    (-0.3490658503988659, 2.2689280275926285),  # middle_pip (middle_pp2mp)
    (-0.5235987755982988, 0.5235987755982988),  # ring_abd
    (-0.3490658503988659, 1.9198621771937625),  # ring_mcp (root2ring_pp)
    (-0.3490658503988659, 2.2689280275926285),  # ring_pip (ring_pp2mp)
    (-0.5235987755982988, 0.5235987755982988),  # pinky_abd
    (-0.3490658503988659, 1.9198621771937625),  # pinky_mcp (root2pinky_pp)
    (-0.3490658503988659, 2.2689280275926285),  # pinky_pip (pinky_pp2mp)
)
HAND_DIMS_START = 6  # matches HAND_DIMS = tuple(range(6, 23)) in motion_suite.py


def constrain_hand_bounds_to_urdf_limits(
    action_low: np.ndarray, action_high: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Intersects the hand-joint slice of [action_low, action_high] with
    HAND_JOINT_LIMITS_RAD (tighter of the two wins per-dimension). Returns the
    (possibly) tightened bounds plus the list of ACTION_LABELS that were actually
    tightened, for logging. No-ops (returns inputs unchanged, empty list) if
    action_low doesn't have exactly HAND_DIMS_START + 17 dims."""
    expected_dim = HAND_DIMS_START + len(HAND_JOINT_LIMITS_RAD)
    if action_low.shape[0] != expected_dim:
        return action_low, action_high, []
    action_low = action_low.copy()
    action_high = action_high.copy()
    tightened = []
    for i, (lo, hi) in enumerate(HAND_JOINT_LIMITS_RAD):
        dim = HAND_DIMS_START + i
        new_low = max(float(action_low[dim]), lo)
        new_high = min(float(action_high[dim]), hi)
        if new_low != action_low[dim] or new_high != action_high[dim]:
            tightened.append(ACTION_LABELS[dim])
        action_low[dim] = new_low
        action_high[dim] = new_high
    return action_low, action_high, tightened


@dataclass(frozen=True)
class TargetSpec:
    trial_group_id: str
    split: str
    sample_idx: int
    sampling_scope: str  # "whole_pose" | "per_dim"
    component: Optional[int]
    component_label: Optional[str]
    target_idx: int
    base_action_abs: List[float]
    base_action_source: str
    target_action_abs: List[float]
    dims_changed: List[int]


def sample_whole_pose_target_norm(rng: np.random.Generator, action_dim: int) -> np.ndarray:
    """Uniform i.i.d. draw in [-1, 1]^action_dim -- one full random target pose."""
    return rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)


def sample_per_dim_target_norm(rng: np.random.Generator, min_norm_distance: float, base_norm_value: float) -> float:
    """Uniform draw in [-1, 1], resampled while too close to base_norm_value.

    Avoids a near-zero-delta, uninformative trial. Falls back to whichever
    boundary is farthest from base_norm_value if no draw clears min_norm_distance
    within a bounded number of attempts (can happen if base_norm_value is itself
    near 0 and min_norm_distance is large).
    """
    for _ in range(64):
        candidate = float(rng.uniform(-1.0, 1.0))
        if abs(candidate - base_norm_value) >= min_norm_distance:
            return candidate
    return 1.0 if abs(1.0 - base_norm_value) >= abs(-1.0 - base_norm_value) else -1.0


def build_target_catalog_for_sample(
    split: str,
    sample_idx: int,
    base_action_abs: np.ndarray,
    base_action_source: str,
    action_low: np.ndarray,
    action_high: np.ndarray,
    component_indices: List[int],
    scopes: List[str],
    num_whole_pose_targets: int,
    num_per_dim_targets: int,
    rng: np.random.Generator,
    min_norm_distance: float,
) -> List[TargetSpec]:
    action_dim = int(base_action_abs.shape[-1])
    base_norm = normalize_bound(base_action_abs[None, :], action_low[None, :], action_high[None, :])[0]
    specs: List[TargetSpec] = []

    if "whole_pose" in scopes:
        for target_idx in range(num_whole_pose_targets):
            target_norm = sample_whole_pose_target_norm(rng, action_dim)
            target_abs = denormalize_bound(target_norm[None, :], action_low[None, :], action_high[None, :])[0]
            trial_group_id = f"{split}_s{sample_idx:05d}_wholepose_full_t{target_idx:02d}"
            specs.append(
                TargetSpec(
                    trial_group_id=trial_group_id,
                    split=split,
                    sample_idx=int(sample_idx),
                    sampling_scope="whole_pose",
                    component=None,
                    component_label=None,
                    target_idx=target_idx,
                    base_action_abs=base_action_abs.astype(float).tolist(),
                    base_action_source=base_action_source,
                    target_action_abs=target_abs.astype(float).tolist(),
                    dims_changed=list(range(action_dim)),
                )
            )

    if "per_dim" in scopes:
        for dim_idx in component_indices:
            for target_idx in range(num_per_dim_targets):
                target_norm_dim = sample_per_dim_target_norm(rng, min_norm_distance, float(base_norm[dim_idx]))
                target_abs = base_action_abs.copy()
                dim_abs = denormalize_bound(
                    np.array([[target_norm_dim]], dtype=np.float32),
                    action_low[None, dim_idx : dim_idx + 1],
                    action_high[None, dim_idx : dim_idx + 1],
                )[0, 0]
                target_abs[dim_idx] = dim_abs
                component_label = ACTION_LABELS[dim_idx] if dim_idx < len(ACTION_LABELS) else str(dim_idx)
                trial_group_id = f"{split}_s{sample_idx:05d}_perdim_dim{dim_idx:02d}_t{target_idx:02d}"
                specs.append(
                    TargetSpec(
                        trial_group_id=trial_group_id,
                        split=split,
                        sample_idx=int(sample_idx),
                        sampling_scope="per_dim",
                        component=int(dim_idx),
                        component_label=component_label,
                        target_idx=target_idx,
                        base_action_abs=base_action_abs.astype(float).tolist(),
                        base_action_source=base_action_source,
                        target_action_abs=target_abs.astype(float).tolist(),
                        dims_changed=[int(dim_idx)],
                    )
                )

    return specs


# Datasets written by dataset_example/extract_latent_orca.py store one annotation state per
# latent frame, so a sample's base pose is the state at its current frame ("frame_aligned").
# "legacy_default" reads it with wm_orca_args' default fps fields (latent_original_fps=10,
# down_sample=5); that is how the published targets manifests were generated, and manifests
# without a "frame_mapping" field use it.
FRAME_MAPPING_ALIGNED = "frame_aligned"
FRAME_MAPPING_LEGACY = "legacy_default"


def _frame_mapping_args(dataset_args: wm_orca_args, frame_mapping: str) -> wm_orca_args:
    """Copy of dataset_args whose fps fields map a sample to the state that defines its base pose."""
    mapped = copy.copy(dataset_args)
    if frame_mapping == FRAME_MAPPING_ALIGNED:
        mapped.latent_original_fps = mapped.fps
        mapped.down_sample = 1
    elif frame_mapping == FRAME_MAPPING_LEGACY:
        defaults = wm_orca_args()
        mapped.fps = defaults.fps
        mapped.latent_original_fps = defaults.latent_original_fps
        mapped.down_sample = defaults.down_sample
    else:
        raise ValueError(f"Unknown frame mapping {frame_mapping!r}")
    return mapped


def _check_sample_matches_targets(
    test_dataset: Dataset_mix,
    dataset_args: wm_orca_args,
    targets: Dict,
    sample_idx: int,
    sample_targets: List[Dict],
    dataset_id: int,
) -> None:
    """Fail if a rollout loaded a different scene than the one its targets were built from."""
    target = sample_targets[0]
    if target.get("base_action_source") != "current_observation":
        return
    check_args = _frame_mapping_args(dataset_args, targets.get("frame_mapping", FRAME_MAPPING_LEGACY))
    check_args.swap_abd_with_mcp = bool(targets.get("swap_abd_with_mcp_applied", False))
    observed = get_current_observed_action_abs(
        test_dataset=test_dataset,
        dataset_args=check_args,
        sample_idx=sample_idx,
        dataset_id=dataset_id,
        expected_action_dim=int(targets["action_dim"]),
    )
    expected = np.asarray(target["base_action_abs"], dtype=np.float32)
    if observed is None or not np.allclose(observed, expected, atol=1e-4):
        raise RuntimeError(
            f"sample_idx={sample_idx} is not the scene its targets were built from (base pose "
            "mismatch). Use the same dataset arguments and --seed as the targets phase."
        )


def main_targets(args_cli: argparse.Namespace) -> str:
    dataset_args = wm_orca_args()
    if args_cli.dataset_config_path is not None:
        dataset_args = load_experiment_config(args_cli.dataset_config_path, dataset_args)
    if args_cli.dataset_root_path is not None:
        dataset_args.dataset_root_path = args_cli.dataset_root_path
    if args_cli.dataset_names is not None:
        dataset_args.dataset_names = args_cli.dataset_names
    if args_cli.dataset_meta_info_path is not None:
        dataset_args.dataset_meta_info_path = args_cli.dataset_meta_info_path
    if args_cli.dataset_meta_info_name is not None:
        dataset_args.dataset_meta_info_name = args_cli.dataset_meta_info_name
    if args_cli.dataset_stat_path is not None:
        dataset_args.dataset_stat_path = args_cli.dataset_stat_path
    # Training-time episode exclusions must not change which validation samples are evaluated.
    dataset_args.exclude_episode_ids_by_dataset = {}
    frame_mapping = FRAME_MAPPING_LEGACY if args_cli.legacy_default_frame_mapping else FRAME_MAPPING_ALIGNED

    np.random.seed(args_cli.seed)
    rng = np.random.default_rng(args_cli.seed)

    test_dataset = Dataset_mix(dataset_args, mode=args_cli.mode)
    sample_indices = resolve_sample_indices(args_cli, test_dataset)
    print(f"[targets] split={args_cli.split_name} sample_indices={sample_indices}")

    stat_path = get_stat_path(dataset_args, args_cli.dataset_id)
    with open(stat_path, "r", encoding="utf-8") as f:
        stat = json.load(f)
    action_dim = len(stat["state_01"])

    percentile_low = float(args_cli.target_bound_percentile_low)
    percentile_high = float(args_cli.target_bound_percentile_high)
    if percentile_low == 1.0 and percentile_high == 99.0:
        action_low, action_high = get_action_bounds_from_stat(stat_path, dataset_args, action_dim)
    else:
        # Recompute bounds at custom percentiles directly from the raw dataset
        # state values used to build stat.json, mirroring
        # dataset_meta_info/create_meta_info_orca.py's own np.percentile call.
        # stat.json only stores the 1/99 percentiles, so a non-default
        # percentile pair requires the caller to point at a stat file that was
        # itself regenerated at those percentiles; here we just clip the
        # stored 1/99 bounds proportionally as a lightweight tightening knob.
        state_01 = np.array(stat["state_01"], dtype=np.float32)
        state_99 = np.array(stat["state_99"], dtype=np.float32)
        shrink_low = (percentile_low - 1.0) / 98.0
        shrink_high = (99.0 - percentile_high) / 98.0
        span = state_99 - state_01
        action_low = state_01 + span * max(0.0, shrink_low)
        action_high = state_99 - span * max(0.0, shrink_high)

    if not args_cli.no_constrain_hand_joints_to_urdf_limits:
        action_low, action_high, tightened_dims = constrain_hand_bounds_to_urdf_limits(action_low, action_high)
        if tightened_dims:
            print(
                f"[targets] Constrained hand-joint sampling bounds to URDF range-of-motion "
                f"limits (tighter than the dataset percentile bounds) for: {tightened_dims}"
            )

    scopes = [s.strip() for s in args_cli.sampling_scopes.split(",") if s.strip()]
    for scope in scopes:
        if scope not in ("whole_pose", "per_dim"):
            raise ValueError(f"Unknown sampling scope '{scope}'; expected 'whole_pose' and/or 'per_dim'.")

    if args_cli.component_indices:
        component_indices = parse_int_list(args_cli.component_indices)
    else:
        component_indices = list(range(action_dim))

    os.makedirs(args_cli.output_path, exist_ok=True)

    bounds_report = {
        "stat_path": stat_path,
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "dims": [
            {
                "dim": i,
                "label": ACTION_LABELS[i] if i < len(ACTION_LABELS) else str(i),
                "low": float(action_low[i]),
                "high": float(action_high[i]),
                "width": float(action_high[i] - action_low[i]),
            }
            for i in range(action_dim)
        ],
    }
    bounds_report_path = os.path.join(args_cli.output_path, "bounds_report.json")
    with open(bounds_report_path, "w", encoding="utf-8") as f:
        json.dump(bounds_report, f, indent=2)
    print(f"[targets] Saved bounds report to {bounds_report_path} -- inspect before a full sweep.")

    all_targets: List[TargetSpec] = []
    for sample_idx in sample_indices:
        base_action_abs = get_current_observed_action_abs(
            test_dataset=test_dataset,
            dataset_args=_frame_mapping_args(dataset_args, frame_mapping),
            sample_idx=sample_idx,
            dataset_id=args_cli.dataset_id,
            expected_action_dim=action_dim,
        )
        if base_action_abs is None:
            sample = test_dataset.__getitem__(index=sample_idx, dataset_id=args_cli.dataset_id)
            base_action_norm = sample["action"][dataset_args.num_history].detach().cpu().numpy()
            base_action_abs = denormalize_bound(
                base_action_norm[None, :], action_low[None, :], action_high[None, :]
            )[0].astype(np.float32)
            base_action_source = "dataset_action"
        else:
            base_action_source = "current_observation"

        specs = build_target_catalog_for_sample(
            split=args_cli.split_name,
            sample_idx=sample_idx,
            base_action_abs=base_action_abs,
            base_action_source=base_action_source,
            action_low=action_low,
            action_high=action_high,
            component_indices=component_indices,
            scopes=scopes,
            num_whole_pose_targets=args_cli.num_whole_pose_targets,
            num_per_dim_targets=args_cli.num_per_dim_targets,
            rng=rng,
            min_norm_distance=args_cli.min_norm_target_distance,
        )
        all_targets.extend(specs)
        print(f"[targets] sample_idx={sample_idx}: {len(specs)} targets ({base_action_source})")

    manifest = {
        "seed": int(args_cli.seed),
        "split_name": args_cli.split_name,
        "action_dim": int(action_dim),
        "action_labels": ACTION_LABELS,
        "swap_abd_with_mcp_applied": bool(getattr(dataset_args, "swap_abd_with_mcp", False)),
        "dataset_name": resolve_dataset_name(dataset_args, args_cli.dataset_id),
        "dataset_id": int(args_cli.dataset_id),
        "stat_path": stat_path,
        "sample_indices": [int(i) for i in sample_indices],
        "sampling_scopes": scopes,
        "component_indices": [int(i) for i in component_indices],
        "num_whole_pose_targets": int(args_cli.num_whole_pose_targets),
        "num_per_dim_targets": int(args_cli.num_per_dim_targets),
        "min_norm_target_distance": float(args_cli.min_norm_target_distance),
        "targets": [asdict(spec) for spec in all_targets],
    }
    if frame_mapping != FRAME_MAPPING_LEGACY:
        manifest["frame_mapping"] = frame_mapping
    targets_manifest_path = os.path.join(args_cli.output_path, "targets_manifest.json")
    with open(targets_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[targets] Saved {len(all_targets)} targets to {targets_manifest_path}")
    return targets_manifest_path


# ---------------------------------------------------------------------------
# Approach-style action-sequence builders (pure, no model/GPU needed)
# ---------------------------------------------------------------------------

def build_direct_step_actions_abs(target_action_abs: np.ndarray, total_future_frames: int) -> np.ndarray:
    """Step function: the target is held for the whole horizon from frame 0."""
    return _repeat_base(target_action_abs, total_future_frames)


def build_linear_approach_actions_abs(
    base_action_abs: np.ndarray,
    target_action_abs: np.ndarray,
    total_future_frames: int,
    transition_fraction: float = 0.6,
) -> np.ndarray:
    """Ramp base->target over transition_fraction of the horizon, then hold."""
    return _interpolate_hold(base_action_abs, target_action_abs, total_future_frames, transition_fraction)


def expand_target_to_actions_abs(
    base_action_abs: np.ndarray,
    target_action_abs: np.ndarray,
    total_future_frames: int,
    approach_style: str,
    transition_fraction: float,
) -> np.ndarray:
    if approach_style == "direct":
        return build_direct_step_actions_abs(target_action_abs, total_future_frames)
    if approach_style == "linear":
        return build_linear_approach_actions_abs(
            base_action_abs, target_action_abs, total_future_frames, transition_fraction
        )
    raise ValueError(f"Unknown approach_style '{approach_style}'; expected 'direct' or 'linear'.")


# ---------------------------------------------------------------------------
# Rollout expansion (models loaded; existing autoregressive runners reused,
# not modified)
# ---------------------------------------------------------------------------

def _load_targets_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_chunks(seq: List, size: Optional[int]) -> List[List]:
    """Splits seq into sub-lists of at most `size` (the whole seq in one chunk if
    size is None) -- lets a single sample's batched_cases (num_targets *
    num_approach_styles, e.g. 24-52 in practice) be processed in smaller
    sub-batches to fit GPU memory, without changing anything for callers that
    don't pass --max_batch_size (default None reproduces the prior
    batch-everything-at-once behavior exactly)."""
    if size is None or size >= len(seq):
        return [seq]
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def main_rollout_baseline(baseline_wm_args: wm_orca_args, args_cli: argparse.Namespace) -> None:
    targets = _load_targets_manifest(args_cli.targets_manifest_path)
    approach_styles = [s.strip() for s in args_cli.approach_styles.split(",") if s.strip()]
    variant_label = args_cli.variant_label or "baseline"

    step_horizon = int(baseline_wm_args.num_frames)
    total_future_frames = int(args_cli.ar_num_steps) * step_horizon

    dataset_args = copy.deepcopy(baseline_wm_args)
    dataset_args.num_frames = total_future_frames
    dataset_args.dataset_names = targets["dataset_name"]
    # Evaluate on the same validation samples as the targets phase, whatever episodes the
    # model's training config excluded.
    dataset_args.exclude_episode_ids_by_dataset = {}

    use_wandb = bool(args_cli.wandb_project_name or args_cli.wandb_run_name)
    accelerator = Accelerator(
        mixed_precision=baseline_wm_args.mixed_precision,
        log_with="wandb" if use_wandb else None,
    )
    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            args_cli.wandb_project_name or "ctrl-world",
            init_kwargs={"wandb": {"name": args_cli.wandb_run_name or "baseline_controllability_eval"}},
        )

    model = _load_ctrl_world_from_checkpoint(
        baseline_wm_args, args_cli.baseline_wm_model_ckpt_path, model_label="baseline_wm"
    )
    model = accelerator.prepare(model)

    np.random.seed(args_cli.seed)  # same validation subset as the targets phase
    torch.manual_seed(args_cli.seed)  # repeatable diffusion noise
    test_dataset = Dataset_mix(dataset_args, mode=args_cli.mode)
    action_dim = int(targets["action_dim"])
    baseline_stat_path = get_model_stat_path(baseline_wm_args, "baseline_wm", args_cli.dataset_id)
    baseline_action_low, baseline_action_high = get_action_bounds_from_stat(
        baseline_stat_path, baseline_wm_args, action_dim
    )

    os.makedirs(args_cli.output_path, exist_ok=True)
    videos_dir = os.path.join(args_cli.output_path, "videos")
    tensors_dir = os.path.join(args_cli.output_path, "tensors")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    manifest = {
        "targets_manifest_path": args_cli.targets_manifest_path,
        "model_variant": variant_label,
        "baseline_wm_stat_path": baseline_stat_path,
        "ar_num_steps": int(args_cli.ar_num_steps),
        "step_horizon": step_horizon,
        "total_future_frames": total_future_frames,
        "approach_styles": approach_styles,
        "transition_fraction": float(args_cli.transition_fraction),
        "results": [],
    }

    targets_by_sample: Dict[int, List[Dict]] = {}
    for target in targets["targets"]:
        targets_by_sample.setdefault(target["sample_idx"], []).append(target)

    # Bug fix (2026-09-09): see the identical note in main_rollout_wm1wm2 -- a model
    # configured with swap_abd_with_mcp=True needs the shared, model-agnostic
    # targets_manifest.json's future action swapped to match, since the manifest is
    # always built in the raw/unswapped convention.
    target_swap_applied = bool(targets.get("swap_abd_with_mcp_applied", False))
    baseline_hand_swap_correction = bool(getattr(baseline_wm_args, "swap_abd_with_mcp", False)) != target_swap_applied
    if baseline_hand_swap_correction:
        print(
            f"[rollout/baseline] Applying hand abd/mcp swap correction to future actions "
            f"(targets_manifest swap_abd_with_mcp_applied={target_swap_applied})"
        )

    rollout_idx = 0
    for sample_pos, (sample_idx, sample_targets) in enumerate(targets_by_sample.items()):
        _check_sample_matches_targets(
            test_dataset, dataset_args, targets, sample_idx, sample_targets, args_cli.dataset_id
        )
        sample = test_dataset.__getitem__(index=sample_idx, dataset_id=args_cli.dataset_id)
        base_batch = to_device_batch(sample, accelerator.device)

        case_specs = []
        for target in sample_targets:
            base_action_abs = np.array(target["base_action_abs"], dtype=np.float32)
            target_action_abs = np.array(target["target_action_abs"], dtype=np.float32)
            for approach_style in approach_styles:
                actions_abs_future = expand_target_to_actions_abs(
                    base_action_abs, target_action_abs, total_future_frames,
                    approach_style, args_cli.transition_fraction,
                )
                case_specs.append((target, approach_style, actions_abs_future))

        case_chunks = _iter_chunks(case_specs, args_cli.max_batch_size)
        print(
            f"[rollout/baseline] sample_idx={sample_idx} "
            f"({sample_pos + 1}/{len(targets_by_sample)}), total_cases={len(case_specs)}, "
            f"chunks={len(case_chunks)}"
        )

        for chunk_pos, chunk in enumerate(case_chunks):
            chunk_actions_abs_cases = [c[2] for c in chunk]
            batched_base_batch = repeat_batch(base_batch, len(chunk_actions_abs_cases))
            actions_abs_batch = np.stack(chunk_actions_abs_cases, axis=0)

            print(
                f"[rollout/baseline]   chunk {chunk_pos + 1}/{len(case_chunks)}, "
                f"batched_cases={len(chunk_actions_abs_cases)}"
            )

            with torch.no_grad():
                with accelerator.autocast():
                    out = run_autoregressive_baseline_wm_with_actions(
                        model=model,
                        model_args=baseline_wm_args,
                        accelerator=accelerator,
                        base_batch=batched_base_batch,
                        actions_abs_future=actions_abs_batch,
                        action_low=baseline_action_low,
                        action_high=baseline_action_high,
                        ar_num_steps=args_cli.ar_num_steps,
                        snap_to_palette=args_cli.snap_to_palette,
                        hand_swap_correction=baseline_hand_swap_correction,
                    )

            for case_pos, (target, approach_style, actions_abs_future) in enumerate(chunk):
                view_start = case_pos * baseline_wm_args.num_views
                view_end = (case_pos + 1) * baseline_wm_args.num_views
                pred_video = out["pred_video"][view_start:view_end]
                pred_latents = out["pred_latents"][view_start:view_end]
                video_grid = make_video_grid_uint8(pred_video, num_views=baseline_wm_args.num_views)

                trial_id = f"{target['trial_group_id']}_{approach_style}_{variant_label}"
                video_path = os.path.join(videos_dir, f"{trial_id}.mp4")
                latent_path = os.path.join(tensors_dir, f"{trial_id}_pred_latents.pt")
                action_path = os.path.join(tensors_dir, f"{trial_id}_actions_abs.npy")

                mediapy.write_video(video_path, video_grid, fps=baseline_wm_args.fps)
                torch.save(pred_latents.detach().cpu(), latent_path)
                np.save(action_path, actions_abs_future)

                if use_wandb and accelerator.is_main_process:
                    accelerator.log(
                        {f"video/{trial_id}": wandb.Video(video_path, fps=baseline_wm_args.wandb_video_display_fps, format="mp4")},
                        step=rollout_idx,
                    )

                manifest["results"].append(
                    {
                        "trial_id": trial_id,
                        "trial_group_id": target["trial_group_id"],
                        "split": target["split"],
                        "sampling_scope": target["sampling_scope"],
                        "component": target["component"],
                        "component_label": target["component_label"],
                        "approach_style": approach_style,
                        "model_variant": variant_label,
                        "video_path": video_path,
                        "latent_path": latent_path,
                        "action_path": action_path,
                    }
                )
                rollout_idx += 1
                print(f"[rollout/baseline] Saved {video_path}")

    manifest_path = os.path.join(args_cli.output_path, f"rollout_manifest_{variant_label}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    accelerator.end_training()
    print(f"[rollout/baseline] Done. Manifest: {manifest_path}")


def main_rollout_wm1wm2(
    wm1_args: wm_orca_args, wm2_args: wm_orca_args, args_cli: argparse.Namespace
) -> None:
    targets = _load_targets_manifest(args_cli.targets_manifest_path)
    approach_styles = [s.strip() for s in args_cli.approach_styles.split(",") if s.strip()]
    # Two separate defaults preserve the pre-existing "wm1_wm2" (metadata field) vs
    # "wm1wm2" (trial_id/filename) mismatch exactly when --variant_label is unset;
    # when it is set, both collapse to the same custom label.
    variant_label = args_cli.variant_label or "wm1_wm2"
    variant_id_tag = args_cli.variant_label or "wm1wm2"

    if wm1_args.num_frames != wm2_args.num_frames:
        raise ValueError(f"Requires wm1_num_frames == wm2_num_frames. Got {wm1_args.num_frames} vs {wm2_args.num_frames}.")
    step_horizon = int(wm1_args.num_frames)
    total_future_frames = int(args_cli.ar_num_steps) * step_horizon

    dataset_args = copy.deepcopy(wm2_args)
    dataset_args.num_frames = total_future_frames
    dataset_args.dataset_names = targets["dataset_name"]
    # Evaluate on the same validation samples as the targets phase, whatever episodes the
    # model's training config excluded.
    dataset_args.exclude_episode_ids_by_dataset = {}
    # Bug fix (2026-09-09): dataset_args is a copy of wm2_args only, so without this,
    # WM1's own --wm1_dataset_stat_path / swap_abd_with_mcp never reach Dataset_mix --
    # it silently never creates an "action_wm1" field, and use_model_action(..., "wm1")
    # (used for WM1's action HISTORY at rollout time) falls back to the shared "action"
    # field, which is normalized under WM2's own stat/swap instead of WM1's own. This
    # was a real, confirmed cause of WM1 receiving a wrongly-normalized action history.
    dataset_args.wm1_dataset_stat_path = getattr(wm1_args, "wm1_dataset_stat_path", None)
    dataset_args.wm1_swap_abd_with_mcp = bool(getattr(wm1_args, "swap_abd_with_mcp", False))

    accelerator = Accelerator(mixed_precision=wm2_args.mixed_precision, log_with="wandb" if args_cli.wandb_project_name else None)
    if args_cli.sequential_wm1_wm2_loading and accelerator.num_processes > 1:
        raise ValueError("--sequential_wm1_wm2_loading currently supports single-process inference only.")
    if args_cli.wandb_project_name and accelerator.is_main_process:
        accelerator.init_trackers(
            args_cli.wandb_project_name, init_kwargs={"wandb": {"name": args_cli.wandb_run_name or "wm1wm2_controllability_eval"}}
        )

    wm1 = _load_ctrl_world_from_checkpoint(wm1_args, args_cli.wm1_model_ckpt_path, model_label="WM1")
    wm2 = None
    if not args_cli.sequential_wm1_wm2_loading:
        wm2 = _load_ctrl_world_from_checkpoint(wm2_args, args_cli.wm2_model_ckpt_path, model_label="WM2")
        wm1, wm2 = accelerator.prepare(wm1, wm2)
    sequential_model_cache = {"wm1": wm1, "wm2": wm2} if args_cli.sequential_wm1_wm2_loading else None

    np.random.seed(args_cli.seed)  # same validation subset as the targets phase
    torch.manual_seed(args_cli.seed)  # repeatable diffusion noise
    test_dataset = Dataset_mix(dataset_args, mode=args_cli.mode)
    action_dim = int(targets["action_dim"])
    wm1_stat_path = get_model_stat_path(wm1_args, "wm1", args_cli.dataset_id)
    wm2_stat_path = get_model_stat_path(wm2_args, "wm2", args_cli.dataset_id)
    wm1_action_low, wm1_action_high = get_action_bounds_from_stat(wm1_stat_path, wm1_args, action_dim)
    wm2_action_low, wm2_action_high = get_action_bounds_from_stat(wm2_stat_path, wm2_args, action_dim)

    os.makedirs(args_cli.output_path, exist_ok=True)
    videos_dir = os.path.join(args_cli.output_path, "videos")
    tensors_dir = os.path.join(args_cli.output_path, "tensors")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    manifest = {
        "targets_manifest_path": args_cli.targets_manifest_path,
        "model_variant": variant_label,
        "wm1_stat_path": wm1_stat_path,
        "wm2_stat_path": wm2_stat_path,
        "ar_num_steps": int(args_cli.ar_num_steps),
        "step_horizon": step_horizon,
        "total_future_frames": total_future_frames,
        "approach_styles": approach_styles,
        "transition_fraction": float(args_cli.transition_fraction),
        "sequential_wm1_wm2_loading": bool(args_cli.sequential_wm1_wm2_loading),
        "results": [],
    }

    targets_by_sample: Dict[int, List[Dict]] = {}
    for target in targets["targets"]:
        targets_by_sample.setdefault(target["sample_idx"], []).append(target)

    # Bug fix (2026-09-09): targets_manifest.json is built once, model-agnostically, and is
    # always in whatever convention main_targets' bare (unconfigured) dataset_args resolved
    # to -- in practice always swap_abd_with_mcp=False (see "swap_abd_with_mcp_applied" in
    # the manifest). A model whose own config sets swap_abd_with_mcp=True (e.g. both
    # sim-midtrained WM1 variants, and WM2) was trained with hand-joint columns 1/2 swapped
    # relative to that raw convention, so its future/target action needs the same swap
    # applied here -- exactly what Dataset_mix already does for real recorded history
    # frames, which this call previously left unmatched for the future portion.
    target_swap_applied = bool(targets.get("swap_abd_with_mcp_applied", False))
    wm1_hand_swap_correction = bool(getattr(wm1_args, "swap_abd_with_mcp", False)) != target_swap_applied
    wm2_hand_swap_correction = bool(getattr(wm2_args, "swap_abd_with_mcp", False)) != target_swap_applied
    if wm1_hand_swap_correction or wm2_hand_swap_correction:
        print(
            f"[rollout/wm1wm2] Applying hand abd/mcp swap correction to future actions: "
            f"wm1={wm1_hand_swap_correction}, wm2={wm2_hand_swap_correction} "
            f"(targets_manifest swap_abd_with_mcp_applied={target_swap_applied})"
        )

    rollout_idx = 0
    for sample_pos, (sample_idx, sample_targets) in enumerate(targets_by_sample.items()):
        _check_sample_matches_targets(
            test_dataset, dataset_args, targets, sample_idx, sample_targets, args_cli.dataset_id
        )
        sample = test_dataset.__getitem__(index=sample_idx, dataset_id=args_cli.dataset_id)
        base_batch = to_device_batch(sample, accelerator.device)

        case_specs = []
        for target in sample_targets:
            base_action_abs = np.array(target["base_action_abs"], dtype=np.float32)
            target_action_abs = np.array(target["target_action_abs"], dtype=np.float32)
            for approach_style in approach_styles:
                actions_abs_future = expand_target_to_actions_abs(
                    base_action_abs, target_action_abs, total_future_frames,
                    approach_style, args_cli.transition_fraction,
                )
                case_specs.append((target, approach_style, actions_abs_future))

        case_chunks = _iter_chunks(case_specs, args_cli.max_batch_size)
        print(
            f"[rollout/wm1wm2] sample_idx={sample_idx} "
            f"({sample_pos + 1}/{len(targets_by_sample)}), total_cases={len(case_specs)}, "
            f"chunks={len(case_chunks)}"
        )

        wm1_view_span = wm1_args.num_views
        view_span = wm2_args.num_views
        for chunk_pos, chunk in enumerate(case_chunks):
            chunk_actions_abs_cases = [c[2] for c in chunk]
            batched_base_batch = repeat_batch(base_batch, len(chunk_actions_abs_cases))
            actions_abs_batch = np.stack(chunk_actions_abs_cases, axis=0)

            print(
                f"[rollout/wm1wm2]   chunk {chunk_pos + 1}/{len(case_chunks)}, "
                f"batched_cases={len(chunk_actions_abs_cases)}"
            )

            with torch.no_grad():
                with accelerator.autocast():
                    out = run_autoregressive_wm1_wm2_with_actions(
                        wm1=wm1,
                        wm2=wm2,
                        wm1_args=wm1_args,
                        wm2_args=wm2_args,
                        accelerator=accelerator,
                        base_batch=batched_base_batch,
                        actions_abs_future=actions_abs_batch,
                        wm1_action_low=wm1_action_low,
                        wm1_action_high=wm1_action_high,
                        wm2_action_low=wm2_action_low,
                        wm2_action_high=wm2_action_high,
                        ar_num_steps=args_cli.ar_num_steps,
                        snap_to_palette=args_cli.snap_to_palette,
                        wm2_disable_action_conditioning=args_cli.wm2_disable_action_conditioning,
                        sequential_wm1_wm2_loading=args_cli.sequential_wm1_wm2_loading,
                        wm1_model_ckpt_path=args_cli.wm1_model_ckpt_path,
                        wm2_model_ckpt_path=args_cli.wm2_model_ckpt_path,
                        model_cache=sequential_model_cache,
                        wm1_hand_swap_correction=wm1_hand_swap_correction,
                        wm2_hand_swap_correction=wm2_hand_swap_correction,
                    )

            for case_pos, (target, approach_style, actions_abs_future) in enumerate(chunk):
                wm1_view_start = case_pos * wm1_view_span
                wm1_view_end = (case_pos + 1) * wm1_view_span
                view_start = case_pos * view_span
                view_end = (case_pos + 1) * view_span

                wm1_pred_seg_video = out["wm1_pred_seg_video"][wm1_view_start:wm1_view_end]
                wm1_pred_seg_latents = out["wm1_pred_seg_latents"][wm1_view_start:wm1_view_end]
                wm2_pred_video = out["wm2_pred_video"][view_start:view_end]
                wm2_pred_latents = out["wm2_pred_latents"][view_start:view_end]
                wm1_seg_video_grid = make_video_grid_uint8(wm1_pred_seg_video, num_views=wm1_view_span)
                video_grid = make_video_grid_uint8(wm2_pred_video, num_views=view_span)

                trial_id = f"{target['trial_group_id']}_{approach_style}_{variant_id_tag}"
                video_path = os.path.join(videos_dir, f"{trial_id}.mp4")
                wm1_seg_video_path = os.path.join(videos_dir, f"{trial_id}_wm1_seg.mp4")
                latent_path = os.path.join(tensors_dir, f"{trial_id}_pred_latents.pt")
                wm1_seg_latent_path = os.path.join(tensors_dir, f"{trial_id}_wm1_seg_latents.pt")
                action_path = os.path.join(tensors_dir, f"{trial_id}_actions_abs.npy")

                mediapy.write_video(video_path, video_grid, fps=wm2_args.fps)
                mediapy.write_video(wm1_seg_video_path, wm1_seg_video_grid, fps=wm1_args.fps)
                torch.save(wm2_pred_latents.detach().cpu(), latent_path)
                torch.save(wm1_pred_seg_latents.detach().cpu(), wm1_seg_latent_path)
                np.save(action_path, actions_abs_future)

                if args_cli.wandb_project_name and accelerator.is_main_process:
                    accelerator.log(
                        {
                            f"video/{trial_id}": wandb.Video(video_path, fps=wm2_args.wandb_video_display_fps, format="mp4"),
                            f"video_wm1_seg/{trial_id}": wandb.Video(wm1_seg_video_path, fps=wm1_args.wandb_video_display_fps, format="mp4"),
                        },
                        step=rollout_idx,
                    )

                manifest["results"].append(
                    {
                        "trial_id": trial_id,
                        "trial_group_id": target["trial_group_id"],
                        "split": target["split"],
                        "sampling_scope": target["sampling_scope"],
                        "component": target["component"],
                        "component_label": target["component_label"],
                        "approach_style": approach_style,
                        "model_variant": variant_label,
                        "video_path": video_path,
                        "wm1_seg_video_path": wm1_seg_video_path,
                        "latent_path": latent_path,
                        "wm1_seg_latent_path": wm1_seg_latent_path,
                        "action_path": action_path,
                    }
                )
                rollout_idx += 1
                print(f"[rollout/wm1wm2] Saved {video_path}")

    manifest_path = os.path.join(args_cli.output_path, f"rollout_manifest_{variant_id_tag}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    accelerator.end_training()
    print(f"[rollout/wm1wm2] Done. Manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, choices=["targets", "rollout", "both"], default="both")
    parser.add_argument("--mode", type=str, default="val", help="Dataset split passed to Dataset_mix (e.g. 'val', 'train').")
    parser.add_argument("--output_path", type=str, default=f"inference_output/controllability_eval_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--split_name", type=str, default="id", help="Label only ('id'/'ood'); the dataset that is loaded is set by the --dataset_* arguments.")

    # dataset / sample selection (mirrors inference_wm1_to_wm2_sine_actions.py)
    parser.add_argument("--dataset_config_path", type=str, default=None)
    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, default=None)
    parser.add_argument("--dataset_meta_info_path", type=str, default=None)
    parser.add_argument("--dataset_meta_info_name", type=str, default=None)
    parser.add_argument("--dataset_stat_path", "--data_stat_path", dest="dataset_stat_path", type=str, default=None)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--sample_indices", type=str, default="")
    parser.add_argument("--num_random_samples", type=int, default=None)
    parser.add_argument("--episode_ids", type=str, default="")
    parser.add_argument("--num_episode_samples", type=int, default=None)
    parser.add_argument("--dataset_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--legacy_default_frame_mapping", action="store_true",
        help="Read base poses with wm_orca_args' default fps fields (how the published targets "
             "manifests were generated) instead of frame-aligned annotations.",
    )

    # target sampling (mode=targets)
    parser.add_argument("--sampling_scopes", type=str, default="whole_pose,per_dim")
    parser.add_argument("--num_whole_pose_targets", type=int, default=3)
    parser.add_argument("--num_per_dim_targets", type=int, default=1)
    parser.add_argument("--component_indices", type=str, default="")
    parser.add_argument("--min_norm_target_distance", type=float, default=0.2)
    parser.add_argument("--target_bound_percentile_low", type=float, default=1.0)
    parser.add_argument("--target_bound_percentile_high", type=float, default=99.0)
    parser.add_argument(
        "--no_constrain_hand_joints_to_urdf_limits",
        action="store_true",
        default=False,
        help=(
            "By default, hand-joint (finger) sampling bounds are intersected with the "
            "ORCA hand's actual physical range-of-motion limits (HAND_JOINT_LIMITS_RAD, "
            "sourced from faive_lab's URDF) -- the dataset's own 1st/99th percentile "
            "bounds for the abduction joints in particular are often wider than what's "
            "mechanically reachable (motor-calibration-outlier artifacts), which is why "
            "whole-pose targets frequently failed to converge in the sim GT render. Pass "
            "this flag to disable the constraint and sample purely from dataset "
            "percentiles (the old behavior)."
        ),
    )
    parser.add_argument("--targets_manifest_path", type=str, default=None, help="Reuse an existing target catalog instead of building one (required if --mode rollout).")

    # rollout expansion (mode=rollout)
    parser.add_argument("--approach_styles", type=str, default="direct,linear")
    parser.add_argument("--transition_fraction", type=float, default=0.6)
    parser.add_argument("--ar_num_steps", type=int, default=30)
    parser.add_argument(
        "--max_batch_size", type=int, default=None,
        help=(
            "Cap on how many (target x approach_style) cases are batched into a single "
            "model forward call at once (default: unlimited -- batch an entire sample's "
            "cases together, the prior behavior). Lower this if a rollout OOMs; cases "
            "beyond the cap are processed in additional sequential sub-batches instead of "
            "failing."
        ),
    )
    parser.add_argument("--snap_to_palette", action="store_true", default=False)
    parser.add_argument("--wm2_disable_action_conditioning", action="store_true", default=False)
    parser.add_argument("--sequential_wm1_wm2_loading", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--variant_label",
        type=str,
        default=None,
        help=(
            "Override the model_variant tag used in trial_id / model_variant / "
            "rollout_manifest filename (default: 'baseline' or 'wm1_wm2'/'wm1wm2' "
            "depending on rollout type). Needed to distinguish multiple checkpoint "
            "combos of the same shape (e.g. several WM1+WM2 pairings) written into "
            "the same --output_path -- without this they'd collide on identical "
            "trial_ids and overwrite each other's manifest."
        ),
    )

    # model selection -- baseline XOR wm1+wm2, same convention as sine_actions.py
    parser.add_argument("--baseline_wm_model_ckpt_path", type=str, default=None)
    parser.add_argument("--baseline_wm_config_path", type=str, default=None)
    parser.add_argument("--baseline_wm_finetune_lora_ckpt_path", type=str, default=None)
    parser.add_argument("--baseline_wm_dataset_stat_path", type=str, default=None)
    parser.add_argument("--baseline_wm_num_inference_steps", type=int, default=None)
    parser.add_argument("--wm1_model_ckpt_path", type=str, default=None)
    parser.add_argument("--wm1_config_path", type=str, default=None)
    parser.add_argument("--wm1_finetune_lora_ckpt_path", type=str, default=None)
    parser.add_argument("--wm1_dataset_stat_path", type=str, default=None)
    parser.add_argument("--wm1_num_frames", type=int, default=5)
    parser.add_argument("--wm1_num_history", type=int, default=None)
    parser.add_argument("--wm1_num_inference_steps", type=int, default=None)
    parser.add_argument("--wm2_model_ckpt_path", type=str, default=None)
    parser.add_argument("--wm2_config_path", type=str, default=None)
    parser.add_argument("--wm2_finetune_lora_ckpt_path", type=str, default=None)
    parser.add_argument("--wm2_dataset_stat_path", type=str, default=None)
    parser.add_argument("--wm2_num_frames", type=int, default=5)
    parser.add_argument("--wm2_num_history", type=int, default=None)
    parser.add_argument("--wm2_num_inference_steps", type=int, default=None)

    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args_cli = parser.parse_args()

    if args_cli.debug:
        args_cli.ar_num_steps = min(2, args_cli.ar_num_steps)
        args_cli.num_whole_pose_targets = min(1, args_cli.num_whole_pose_targets)
        args_cli.num_per_dim_targets = min(1, args_cli.num_per_dim_targets)

    targets_manifest_path = args_cli.targets_manifest_path
    if args_cli.phase in ("targets", "both"):
        targets_manifest_path = main_targets(args_cli)

    if args_cli.phase in ("rollout", "both"):
        if targets_manifest_path is None:
            raise ValueError("--phase rollout requires --targets_manifest_path (or --phase both to build one first).")
        args_cli.targets_manifest_path = targets_manifest_path

        baseline_mode = args_cli.baseline_wm_model_ckpt_path is not None
        if baseline_mode:
            baseline_wm_args = wm_orca_args()
            if args_cli.baseline_wm_config_path is None:
                raise ValueError("--baseline_wm_config_path is required when using --baseline_wm_model_ckpt_path.")
            baseline_wm_args = load_experiment_config(args_cli.baseline_wm_config_path, baseline_wm_args)
            if args_cli.dataset_root_path is not None:
                baseline_wm_args.dataset_root_path = args_cli.dataset_root_path
            if args_cli.dataset_names is not None:
                baseline_wm_args.dataset_names = args_cli.dataset_names
            if args_cli.dataset_meta_info_path is not None:
                baseline_wm_args.dataset_meta_info_path = args_cli.dataset_meta_info_path
            if args_cli.dataset_meta_info_name is not None:
                baseline_wm_args.dataset_meta_info_name = args_cli.dataset_meta_info_name
            if args_cli.dataset_stat_path is not None:
                baseline_wm_args.dataset_stat_path = args_cli.dataset_stat_path
            if args_cli.baseline_wm_finetune_lora_ckpt_path is not None:
                baseline_wm_args.finetune_lora_ckpt_path = args_cli.baseline_wm_finetune_lora_ckpt_path
            if args_cli.baseline_wm_dataset_stat_path is not None:
                baseline_wm_args.baseline_wm_dataset_stat_path = args_cli.baseline_wm_dataset_stat_path
            if args_cli.baseline_wm_num_inference_steps is not None:
                baseline_wm_args.num_inference_steps = int(args_cli.baseline_wm_num_inference_steps)
            if args_cli.debug:
                baseline_wm_args.num_inference_steps = 1
            main_rollout_baseline(baseline_wm_args, args_cli)
        else:
            if args_cli.wm1_config_path is None or args_cli.wm1_model_ckpt_path is None:
                raise ValueError("--wm1_config_path and --wm1_model_ckpt_path are required unless --baseline_wm_model_ckpt_path is set.")
            if args_cli.wm2_config_path is None or args_cli.wm2_model_ckpt_path is None:
                raise ValueError("--wm2_config_path and --wm2_model_ckpt_path are required unless --baseline_wm_model_ckpt_path is set.")
            wm1_args = load_experiment_config(args_cli.wm1_config_path, wm_orca_args())
            wm2_args = load_experiment_config(args_cli.wm2_config_path, wm_orca_args())
            for _model_args in (wm1_args, wm2_args):
                if args_cli.dataset_root_path is not None:
                    _model_args.dataset_root_path = args_cli.dataset_root_path
                if args_cli.dataset_names is not None:
                    _model_args.dataset_names = args_cli.dataset_names
                if args_cli.dataset_meta_info_path is not None:
                    _model_args.dataset_meta_info_path = args_cli.dataset_meta_info_path
                if args_cli.dataset_meta_info_name is not None:
                    _model_args.dataset_meta_info_name = args_cli.dataset_meta_info_name
                if args_cli.dataset_stat_path is not None:
                    _model_args.dataset_stat_path = args_cli.dataset_stat_path
            if args_cli.wm1_finetune_lora_ckpt_path is not None:
                wm1_args.finetune_lora_ckpt_path = args_cli.wm1_finetune_lora_ckpt_path
            if args_cli.wm2_finetune_lora_ckpt_path is not None:
                wm2_args.finetune_lora_ckpt_path = args_cli.wm2_finetune_lora_ckpt_path
            if args_cli.wm1_dataset_stat_path is not None:
                wm1_args.wm1_dataset_stat_path = args_cli.wm1_dataset_stat_path
            if args_cli.wm2_dataset_stat_path is not None:
                wm2_args.wm2_dataset_stat_path = args_cli.wm2_dataset_stat_path
            if args_cli.wm1_num_history is not None:
                wm1_args.num_history = int(args_cli.wm1_num_history)
            if args_cli.wm2_num_history is not None:
                wm2_args.num_history = int(args_cli.wm2_num_history)
            if args_cli.wm1_num_inference_steps is not None:
                wm1_args.num_inference_steps = int(args_cli.wm1_num_inference_steps)
            if args_cli.wm2_num_inference_steps is not None:
                wm2_args.num_inference_steps = int(args_cli.wm2_num_inference_steps)
            if args_cli.debug:
                wm1_args.num_inference_steps = 1
                wm2_args.num_inference_steps = 1
            main_rollout_wm1wm2(wm1_args, wm2_args, args_cli)
