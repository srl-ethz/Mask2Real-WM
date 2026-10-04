import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

import argparse
import copy
import gc
import itertools
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from accelerate import Accelerator
import mediapy

from config import wm_orca_args
from utils.config_loader import load_experiment_config
from dataset.dataset_orca import Dataset_mix

from scripts.inference_wm1_to_wm2 import (
    load_ctrlworld_model_for_inference,
    infer_pred_latents_raw,
    decode_and_format_outputs_for_metrics,
    build_windowed_batch,
    build_wm2_cascade_batch,
    apply_snap_to_palette_on_seg_latents,
    use_model_action,
)


def _load_ctrl_world_from_checkpoint(
    model_args: wm_orca_args,
    ckpt_path: str,
    model_label: str = "model",
):
    return load_ctrlworld_model_for_inference(model_args, ckpt_path, model_label=model_label)


def _activate_ctrl_world_model(
    model,
    model_args: wm_orca_args,
    ckpt_path: str,
    accelerator: Accelerator,
    model_label: str = "model",
):
    if model is None:
        model = _load_ctrl_world_from_checkpoint(model_args, ckpt_path, model_label=model_label)
    accelerator.unwrap_model(model).to(accelerator.device)
    accelerator.unwrap_model(model).eval()
    return model


def _offload_ctrl_world_model(model, accelerator: Accelerator):
    if model is None:
        return None
    accelerator.unwrap_model(model).to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def normalize_bound(
    data: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    clip_min: float = -1,
    clip_max: float = 1,
    eps: float = 1e-8,
) -> np.ndarray:
    ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
    return np.clip(ndata, clip_min, clip_max)


def denormalize_bound(
    data: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    clip_min: float = -1,
    clip_max: float = 1,
) -> np.ndarray:
    clip_range = clip_max - clip_min
    return (data - clip_min) / clip_range * (data_max - data_min) + data_min


def to_device_batch(sample: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0).to(device, non_blocking=True)
        else:
            out[k] = [v]
    return out


def repeat_batch(batch: Dict, batch_size: int) -> Dict:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.shape[0] != 1:
                raise ValueError(f"Can only repeat singleton batches, got batch['{k}'].shape={tuple(v.shape)}.")
            out[k] = v.repeat_interleave(batch_size, dim=0)
        elif isinstance(v, list) and len(v) == 1:
            out[k] = v * batch_size
        else:
            out[k] = v
    return out


def tensor_batch_to_cpu(batch: Dict) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return out


def parse_float_list(raw: str) -> List[float]:
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return [float(v) for v in vals]


def parse_int_list(raw: str) -> List[int]:
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(v) for v in vals]


def resolve_dataset_name(args: wm_orca_args, dataset_id: int) -> str:
    names = args.dataset_names.split("+")
    if dataset_id < 0 or dataset_id >= len(names):
        raise ValueError(f"dataset_id={dataset_id} is out of range for dataset_names={args.dataset_names}.")
    return names[dataset_id]


def get_stat_path(args: wm_orca_args, dataset_id: int) -> str:
    if args.dataset_stat_path is not None:
        return args.dataset_stat_path
    if args.dataset_meta_info_name is not None:
        folder = args.dataset_meta_info_name
    else:
        folder = resolve_dataset_name(args, dataset_id)
    return os.path.join(args.dataset_meta_info_path, folder, "stat.json")


def get_model_stat_path(args: wm_orca_args, model_key: str, dataset_id: int) -> str:
    stat_path = getattr(args, f"{model_key}_dataset_stat_path", None)
    if stat_path is not None:
        return stat_path
    return get_stat_path(args, dataset_id)


def get_action_bounds_from_stat(stat_path: str, args: wm_orca_args, action_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    with open(stat_path, "r", encoding="utf-8") as f:
        stat = json.load(f)

    state_p01 = np.array(stat["state_01"], dtype=np.float32)
    state_p99 = np.array(stat["state_99"], dtype=np.float32)

    if args.use_only_hand_actions:
        low = state_p01[6:]
        high = state_p99[6:]
        if args.use_average_scalar_hand_action:
            low = np.array([float(np.mean(low))], dtype=np.float32)
            high = np.array([float(np.mean(high))], dtype=np.float32)
    elif args.use_only_ee_pose_actions:
        low = state_p01[:6]
        high = state_p99[:6]
    else:
        low = state_p01
        high = state_p99

    if low.shape[0] != action_dim:
        raise ValueError(
            "Action/stat dimension mismatch. "
            f"Derived bounds dim={low.shape[0]}, but model action_dim={action_dim}."
        )
    return low, high


def swap_hand_abd_mcp_columns(actions_abs: np.ndarray, ee_pose_dims: int = 6) -> np.ndarray:
    """Mirrors dataset_orca.py's Dataset_mix._maybe_swap_abd_with_mcp, but applied to an
    absolute-action array whose last axis is the full [ee_pose_dims + hand_joints] layout
    (e.g. the 23-dim controllability-eval target trajectory) rather than an already-sliced
    hand-only array. Swaps the two hand-joint columns immediately after the EE-pose block
    (local hand indices 1 and 2 -- thumb_mcp/thumb_abd in this project's joint layout).

    Needed because a shared targets_manifest.json (built once, model-agnostic) is always in
    the raw/unswapped convention, but a model trained with swap_abd_with_mcp=True expects
    this swap applied consistently -- same as Dataset_mix already does for real recorded
    history frames. Only call this for a model whose own swap_abd_with_mcp differs from
    whatever convention the manifest was built with.
    """
    if actions_abs.shape[-1] <= ee_pose_dims + 2:
        raise ValueError(
            f"swap_hand_abd_mcp_columns requires at least {ee_pose_dims + 3} action dims, "
            f"got shape {actions_abs.shape}."
        )
    swapped = actions_abs.copy()
    i, j = ee_pose_dims + 1, ee_pose_dims + 2
    swapped[..., [i, j]] = swapped[..., [j, i]]
    return swapped


@dataclass
class VariantSpec:
    variant_id: int
    cycles: float
    amplitude_scale: float


def build_variant_specs(
    cycles_list: List[float],
    amp_scales: List[float],
    min_variants: int,
) -> List[VariantSpec]:
    all_pairs = list(itertools.product(cycles_list, amp_scales))
    if len(all_pairs) < min_variants:
        raise ValueError(
            f"Need at least {min_variants} (frequency, amplitude) pairs, got {len(all_pairs)}."
        )
    # Prefer \"diagonal\" pairing first so both frequency and amplitude vary
    # across the first variants (e.g. (f0,a0), (f1,a1), (f2,a2)).
    selected = []
    i = 0
    while len(selected) < min_variants and i < (len(cycles_list) * len(amp_scales) * 2):
        candidate = (cycles_list[i % len(cycles_list)], amp_scales[i % len(amp_scales)])
        if candidate not in selected:
            selected.append(candidate)
        i += 1
    if len(selected) < min_variants:
        for candidate in all_pairs:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= min_variants:
                break
    return [
        VariantSpec(variant_id=i, cycles=float(c), amplitude_scale=float(a))
        for i, (c, a) in enumerate(selected)
    ]


def generate_sine_future_actions_abs(
    base_action_abs: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    total_future_frames: int,
    target_dim: int,
    cycles: float,
    amplitude_scale: float,
) -> np.ndarray:
    if total_future_frames <= 0:
        raise ValueError(f"total_future_frames must be positive, got {total_future_frames}.")

    out = np.repeat(base_action_abs[None, :], total_future_frames, axis=0)
    t = np.arange(total_future_frames, dtype=np.float32)
    phase = 2.0 * np.pi * cycles * t / max(1, (total_future_frames - 1))

    denom = high[target_dim] - low[target_dim] + 1e-8
    base_norm = 2.0 * (base_action_abs[target_dim] - low[target_dim]) / denom - 1.0
    base_norm = float(np.clip(base_norm, -1.0, 1.0))

    raw_norm = np.sin(phase + np.arcsin(base_norm))
    delta = raw_norm - base_norm
    max_up = float(np.max(delta))
    max_down = float(abs(np.min(delta)))
    amplitude_scale = float(np.clip(amplitude_scale, 0.0, 1.0))

    sine_norm = np.full_like(raw_norm, base_norm)
    if max_up > 1e-8:
        up_delta = delta / max_up * (1.0 - base_norm) * amplitude_scale
        sine_norm = np.where(delta > 0.0, base_norm + up_delta, sine_norm)
    if max_down > 1e-8:
        down_delta = delta / max_down * (base_norm + 1.0) * amplitude_scale
        sine_norm = np.where(delta < 0.0, base_norm + down_delta, sine_norm)

    sine_norm = np.clip(sine_norm, -1.0, 1.0)
    out[:, target_dim] = (sine_norm + 1.0) * 0.5 * (high[target_dim] - low[target_dim]) + low[target_dim]
    return out.astype(np.float32)


def resolve_sample_indices(args_cli: argparse.Namespace, test_dataset: Dataset_mix) -> List[int]:
    dataset_id = int(args_cli.dataset_id)
    if dataset_id < 0 or dataset_id >= len(test_dataset.samples_all):
        raise ValueError(
            f"dataset_id={dataset_id} is out of range for {len(test_dataset.samples_all)} dataset(s)."
        )

    num_available = int(test_dataset.samples_len[dataset_id])
    if args_cli.sample_indices:
        sample_indices = parse_int_list(args_cli.sample_indices)
    elif args_cli.episode_ids:
        episode_ids = {int(v) for v in parse_int_list(args_cli.episode_ids)}
        sample_indices = [
            idx
            for idx, sample in enumerate(test_dataset.samples_all[dataset_id])
            if int(sample.get("episode_id", -1)) in episode_ids
        ]
        if not sample_indices:
            raise ValueError(
                f"No samples remain in dataset_id={dataset_id} for episode_ids={sorted(episode_ids)} "
                f"after dataset filtering/subsetting."
            )
        if args_cli.num_episode_samples is not None:
            if args_cli.num_episode_samples <= 0:
                raise ValueError("--num_episode_samples must be positive when provided.")
            sample_indices = sample_indices[: int(args_cli.num_episode_samples)]
    elif args_cli.num_random_samples is not None:
        if args_cli.num_random_samples <= 0:
            raise ValueError("--num_random_samples must be positive when provided.")
        if args_cli.num_random_samples > num_available:
            raise ValueError(
                f"Requested {args_cli.num_random_samples} random samples, but dataset_id={dataset_id} "
                f"only has {num_available} sample(s)."
            )
        rng = np.random.default_rng(int(args_cli.seed))
        sample_indices = rng.choice(
            num_available,
            size=int(args_cli.num_random_samples),
            replace=False,
        ).astype(int).tolist()
    else:
        sample_indices = [int(args_cli.sample_idx)]

    invalid = [idx for idx in sample_indices if idx < 0 or idx >= num_available]
    if invalid:
        raise ValueError(
            f"Sample indices {invalid} are invalid for dataset_id={dataset_id}; "
            f"valid range is [0, {num_available - 1}]."
        )
    return [int(idx) for idx in sample_indices]


def get_current_observed_action_abs(
    test_dataset: Dataset_mix,
    dataset_args: wm_orca_args,
    sample_idx: int,
    dataset_id: int,
    expected_action_dim: int,
) -> Optional[np.ndarray]:
    samples = test_dataset.samples_all[dataset_id]
    sample_idx = int(sample_idx) % len(samples)
    sample = samples[sample_idx]
    dataset_dir = test_dataset.dataset_path_all[dataset_id][sample_idx]
    ann_file = test_dataset._resolve_annotation_file(dataset_dir, sample)

    with open(ann_file, "r", encoding="utf-8") as f:
        label = json.load(f)

    if "observation.state.cartesian_position" not in label:
        return None

    cartesian_seq = np.asarray(label["observation.state.cartesian_position"], dtype=np.float32)
    hand_seq_raw = label.get("observation.state.hand_joint_position")
    if hand_seq_raw is None:
        hand_seq_raw = label.get("action.hand_joint_position", label.get("action.hand_joints"))
    if hand_seq_raw is None:
        return None
    hand_seq = np.asarray(hand_seq_raw, dtype=np.float32)

    fps_ratio = max(1, int(dataset_args.latent_original_fps / dataset_args.fps))
    down_sample = max(1, int(dataset_args.down_sample))
    state_id = int(sample["frame_ids"][0] // fps_ratio) * down_sample
    max_state_id = min(len(cartesian_seq), len(hand_seq)) - 1
    if max_state_id < 0:
        return None
    state_id = int(np.clip(state_id, 0, max_state_id))

    hand_current = hand_seq[state_id].copy()
    if bool(getattr(dataset_args, "swap_abd_with_mcp", False)):
        if hand_current.shape[-1] <= 2:
            raise ValueError(
                "swap_abd_with_mcp requires at least 3 hand joints, "
                f"got shape {hand_current.shape}."
            )
        hand_current[[1, 2]] = hand_current[[2, 1]]

    observed = np.concatenate([cartesian_seq[state_id], hand_current], axis=-1).astype(np.float32)
    if observed.shape[-1] != expected_action_dim:
        raise ValueError(
            "Observed current state/action dimension mismatch. "
            f"Expected {expected_action_dim}, got {observed.shape[-1]} from {ann_file}."
        )
    return observed


def make_video_grid_uint8(video_btchw: torch.Tensor, num_views: int) -> np.ndarray:
    # Input [B*num_views, T, C, H, W], B is expected to be 1 for this script.
    video = ((video_btchw / 2.0 + 0.5).clamp(0, 1) * 255.0).to(torch.uint8)
    video = video.permute(0, 1, 3, 4, 2).detach().cpu().numpy()  # [V, T, H, W, C]
    if video.shape[0] != num_views:
        raise ValueError(f"Expected first dimension num_views={num_views}, got {video.shape[0]}.")
    return np.concatenate([video[v] for v in range(num_views)], axis=2)  # [T, H, V*W, C]



def run_autoregressive_baseline_wm_with_actions(
    model,
    model_args: wm_orca_args,
    accelerator: Accelerator,
    base_batch: Dict,
    actions_abs_future: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    ar_num_steps: int,
    snap_to_palette: bool,
    hand_swap_correction: bool = False,
):
    step_horizon = int(model_args.num_frames)
    total_future = int(ar_num_steps) * step_horizon
    batch_size = int(base_batch["action"].shape[0])
    action_dim = int(base_batch["action"].shape[-1])

    actions_abs_future = np.asarray(actions_abs_future, dtype=np.float32)
    if actions_abs_future.ndim == 2:
        actions_abs_future = actions_abs_future[None, ...]
    expected_shape = (batch_size, total_future, action_dim)
    if actions_abs_future.shape != expected_shape:
        raise ValueError(
            f"Expected actions_abs_future shape {expected_shape}, got {actions_abs_future.shape}."
        )

    if hand_swap_correction:
        actions_abs_future = swap_hand_abd_mcp_columns(actions_abs_future)

    actions_norm_future = normalize_bound(
        actions_abs_future,
        action_low[None, None, :],
        action_high[None, None, :],
    ).astype(np.float32)
    actions_norm_future = torch.tensor(
        actions_norm_future,
        dtype=base_batch["action"].dtype,
        device=accelerator.device,
    )

    pipeline = accelerator.unwrap_model(model).pipeline

    state_full = base_batch[model_args.predicted_datatype]
    history = state_full[:, : model_args.num_history]
    current = state_full[:, model_args.num_history]
    action_history = use_model_action(base_batch, "baseline_wm")["action"][:, : model_args.num_history]

    pred_video_chunks = []
    pred_latent_chunks = []

    for step_idx in range(ar_num_steps):
        future_offset = step_idx * step_horizon

        step_batch = build_windowed_batch(
            batch=base_batch,
            num_history=model_args.num_history,
            num_frames=model_args.num_frames,
            future_offset=future_offset,
        )

        action_chunk = actions_norm_future[:, future_offset : future_offset + step_horizon]
        step_action = torch.cat([action_history, action_chunk], dim=1)
        step_batch["action"] = step_action

        pred_raw = infer_pred_latents_raw(
            model=model,
            args=model_args,
            accelerator=accelerator,
            batch=step_batch,
            history_latent=history,
            current_latent=current,
        )

        gt_raw = step_batch[model_args.predicted_datatype].to(
            accelerator.device,
            non_blocking=True,
        )
        pred_video, _, pred_lat, _ = decode_and_format_outputs_for_metrics(
            pred_raw,
            gt_raw,
            model_args,
            pipeline,
        )
        pred_video_chunks.append(pred_video)
        pred_latent_chunks.append(pred_lat)

        feedback_raw = pred_raw
        if snap_to_palette:
            if model_args.predicted_datatype != "latent_segmentation_videos":
                raise ValueError(
                    "--snap_to_palette is only supported when predicted_datatype='latent_segmentation_videos'."
                )
            feedback_raw = apply_snap_to_palette_on_seg_latents(
                seg_latents_btchw=pred_raw,
                pipeline=pipeline,
                decode_chunk_size=model_args.decode_chunk_size,
                enable_snap_to_palette=True,
            )

        history = torch.cat([history, feedback_raw], dim=1)[:, -model_args.num_history :]
        current = feedback_raw[:, -1]
        action_history = torch.cat([action_history, action_chunk], dim=1)[:, -model_args.num_history :]

    return {
        "pred_video": torch.cat(pred_video_chunks, dim=1),
        "pred_latents": torch.cat(pred_latent_chunks, dim=1),
    }


def run_autoregressive_wm1_wm2_with_actions(
    wm1,
    wm2,
    wm1_args: wm_orca_args,
    wm2_args: wm_orca_args,
    accelerator: Accelerator,
    base_batch: Dict,
    actions_abs_future: np.ndarray,
    wm1_action_low: np.ndarray,
    wm1_action_high: np.ndarray,
    wm2_action_low: np.ndarray,
    wm2_action_high: np.ndarray,
    ar_num_steps: int,
    snap_to_palette: bool,
    wm2_disable_action_conditioning: bool = False,
    sequential_wm1_wm2_loading: bool = False,
    wm1_model_ckpt_path: Optional[str] = None,
    wm2_model_ckpt_path: Optional[str] = None,
    model_cache: Optional[Dict[str, object]] = None,
    wm1_hand_swap_correction: bool = False,
    wm2_hand_swap_correction: bool = False,
):
    if wm1_args.num_frames != wm2_args.num_frames:
        raise ValueError(
            f"Autoregressive rollout requires same horizon. "
            f"WM1={wm1_args.num_frames}, WM2={wm2_args.num_frames}."
        )

    step_horizon = int(wm1_args.num_frames)
    total_future = int(ar_num_steps) * step_horizon
    batch_size = int(base_batch["action"].shape[0])
    action_dim = int(base_batch["action"].shape[-1])

    actions_abs_future = np.asarray(actions_abs_future, dtype=np.float32)
    if actions_abs_future.ndim == 2:
        actions_abs_future = actions_abs_future[None, ...]
    expected_shape = (batch_size, total_future, action_dim)
    if actions_abs_future.shape != expected_shape:
        raise ValueError(
            f"Expected actions_abs_future shape {expected_shape}, got {actions_abs_future.shape}."
        )

    wm1_actions_abs_future = (
        swap_hand_abd_mcp_columns(actions_abs_future) if wm1_hand_swap_correction else actions_abs_future
    )
    wm2_actions_abs_future = (
        swap_hand_abd_mcp_columns(actions_abs_future) if wm2_hand_swap_correction else actions_abs_future
    )
    wm1_actions_norm_future = normalize_bound(
        wm1_actions_abs_future,
        wm1_action_low[None, None, :],
        wm1_action_high[None, None, :],
    ).astype(np.float32)
    wm2_actions_norm_future = normalize_bound(
        wm2_actions_abs_future,
        wm2_action_low[None, None, :],
        wm2_action_high[None, None, :],
    ).astype(np.float32)
    wm1_actions_norm_future = torch.tensor(
        wm1_actions_norm_future,
        dtype=base_batch["action"].dtype,
        device=accelerator.device,
    )
    wm2_actions_norm_future = torch.tensor(
        wm2_actions_norm_future,
        dtype=base_batch["action"].dtype,
        device=accelerator.device,
    )

    if sequential_wm1_wm2_loading:
        if wm1_model_ckpt_path is None or wm2_model_ckpt_path is None:
            raise ValueError("Sequential WM1/WM2 loading requires both checkpoint paths.")
        if accelerator.num_processes > 1:
            raise ValueError("--sequential_wm1_wm2_loading currently supports single-process inference only.")
        if model_cache is None:
            model_cache = {"wm1": wm1, "wm2": wm2}
        wm1_pipeline = None
        wm2_pipeline = None
    else:
        wm1_pipeline = accelerator.unwrap_model(wm1).pipeline
        wm2_pipeline = accelerator.unwrap_model(wm2).pipeline

    wm1_state_full = base_batch["latent_segmentation_videos"]
    wm2_state_full = base_batch["latent_videos"]

    wm1_history = wm1_state_full[:, : wm1_args.num_history]
    wm1_current = wm1_state_full[:, wm1_args.num_history]
    wm2_history = wm2_state_full[:, : wm2_args.num_history]
    wm2_current = wm2_state_full[:, wm2_args.num_history]
    wm1_action_history = use_model_action(base_batch, "wm1")["action"][:, : wm1_args.num_history]
    wm2_action_history = use_model_action(base_batch, "wm2")["action"][:, : wm2_args.num_history]

    wm2_pred_video_chunks = []
    wm2_pred_latent_chunks = []
    wm1_pred_seg_video_chunks = []
    wm1_pred_seg_latent_chunks = []

    if sequential_wm1_wm2_loading:
        wm2_cascade_batches = []

        wm1 = _activate_ctrl_world_model(
            model_cache.get("wm1"),
            wm1_args,
            wm1_model_ckpt_path,
            accelerator,
        )
        model_cache["wm1"] = wm1
        wm1_pipeline = accelerator.unwrap_model(wm1).pipeline

        # First pass: roll out the full WM1 segmentation trajectory for this sine action sequence.
        for step_idx in range(ar_num_steps):
            future_offset = step_idx * step_horizon
            wm1_step_batch = build_windowed_batch(
                batch=base_batch,
                num_history=wm1_args.num_history,
                num_frames=wm1_args.num_frames,
                future_offset=future_offset,
            )
            wm2_step_batch = build_windowed_batch(
                batch=base_batch,
                num_history=wm2_args.num_history,
                num_frames=wm2_args.num_frames,
                future_offset=future_offset,
            )

            wm1_action_chunk = wm1_actions_norm_future[:, future_offset : future_offset + step_horizon]
            wm2_action_chunk = wm2_actions_norm_future[:, future_offset : future_offset + step_horizon]
            wm1_step_action = torch.cat([wm1_action_history, wm1_action_chunk], dim=1)
            wm2_step_action = torch.cat([wm2_action_history, wm2_action_chunk], dim=1)
            wm1_step_batch["action"] = wm1_step_action
            wm2_step_batch["action"] = wm2_step_action

            wm1_pred_seg_raw = infer_pred_latents_raw(
                model=wm1,
                args=wm1_args,
                accelerator=accelerator,
                batch=wm1_step_batch,
                history_latent=wm1_history,
                current_latent=wm1_current,
            )
            wm1_gt_seg_raw = wm1_step_batch[wm1_args.predicted_datatype].to(
                accelerator.device,
                non_blocking=True,
            )
            w1_pred_seg_video, _, w1_pred_seg_lat, _ = decode_and_format_outputs_for_metrics(
                wm1_pred_seg_raw,
                wm1_gt_seg_raw,
                wm1_args,
                wm1_pipeline,
            )
            wm1_pred_seg_video_chunks.append(w1_pred_seg_video.detach().cpu())
            wm1_pred_seg_latent_chunks.append(w1_pred_seg_lat.detach().cpu())

            wm1_pred_for_wm2 = apply_snap_to_palette_on_seg_latents(
                seg_latents_btchw=wm1_pred_seg_raw,
                pipeline=wm1_pipeline,
                decode_chunk_size=wm1_args.decode_chunk_size,
                enable_snap_to_palette=snap_to_palette,
            )

            wm2_step_cascade_batch = build_wm2_cascade_batch(
                batch=wm2_step_batch,
                seg_history_latents=wm1_history,
                wm1_pred_seg_latents=wm1_pred_for_wm2,
                wm2_args=wm2_args,
                wm1_pipeline=wm1_pipeline,
            )
            wm2_step_cascade_batch["action"] = wm2_step_action
            wm2_cascade_batches.append(tensor_batch_to_cpu(wm2_step_cascade_batch))

            wm1_history = torch.cat([wm1_history, wm1_pred_seg_raw], dim=1)[:, -wm1_args.num_history :]
            wm1_current = wm1_pred_seg_raw[:, -1]
            wm1_action_history = torch.cat([wm1_action_history, wm1_action_chunk], dim=1)[:, -wm1_args.num_history :]
            wm2_action_history = torch.cat([wm2_action_history, wm2_action_chunk], dim=1)[:, -wm2_args.num_history :]

        model_cache["wm1"] = _offload_ctrl_world_model(wm1, accelerator)
        wm1 = model_cache["wm1"]
        wm1_pipeline = None

        wm2 = _activate_ctrl_world_model(
            model_cache.get("wm2"),
            wm2_args,
            wm2_model_ckpt_path,
            accelerator,
        )
        model_cache["wm2"] = wm2
        wm2_pipeline = accelerator.unwrap_model(wm2).pipeline

        # Second pass: roll out WM2 over the full trajectory using the cached WM1-conditioned batches.
        for wm2_step_cascade_batch in wm2_cascade_batches:
            wm2_pred_raw = infer_pred_latents_raw(
                model=wm2,
                args=wm2_args,
                accelerator=accelerator,
                batch=wm2_step_cascade_batch,
                history_latent=wm2_history,
                current_latent=wm2_current,
                disable_action_conditioning=wm2_disable_action_conditioning,
            )
            wm2_gt_raw = wm2_step_cascade_batch[wm2_args.predicted_datatype].to(
                accelerator.device,
                non_blocking=True,
            )
            w2_pred_video, _, w2_pred_lat, _ = decode_and_format_outputs_for_metrics(
                wm2_pred_raw,
                wm2_gt_raw,
                wm2_args,
                wm2_pipeline,
            )
            wm2_pred_video_chunks.append(w2_pred_video)
            wm2_pred_latent_chunks.append(w2_pred_lat)

            wm2_history = torch.cat([wm2_history, wm2_pred_raw], dim=1)[:, -wm2_args.num_history :]
            wm2_current = wm2_pred_raw[:, -1]

        model_cache["wm2"] = _offload_ctrl_world_model(wm2, accelerator)

        return {
            "wm1_pred_seg_video": torch.cat(wm1_pred_seg_video_chunks, dim=1),
            "wm1_pred_seg_latents": torch.cat(wm1_pred_seg_latent_chunks, dim=1),
            "wm2_pred_video": torch.cat(wm2_pred_video_chunks, dim=1),
            "wm2_pred_latents": torch.cat(wm2_pred_latent_chunks, dim=1),
        }

    for step_idx in range(ar_num_steps):
        future_offset = step_idx * step_horizon

        wm1_step_batch = build_windowed_batch(
            batch=base_batch,
            num_history=wm1_args.num_history,
            num_frames=wm1_args.num_frames,
            future_offset=future_offset,
        )
        wm2_step_batch = build_windowed_batch(
            batch=base_batch,
            num_history=wm2_args.num_history,
            num_frames=wm2_args.num_frames,
            future_offset=future_offset,
        )

        wm1_action_chunk = wm1_actions_norm_future[:, future_offset : future_offset + step_horizon]
        wm2_action_chunk = wm2_actions_norm_future[:, future_offset : future_offset + step_horizon]
        wm1_step_action = torch.cat([wm1_action_history, wm1_action_chunk], dim=1)
        wm2_step_action = torch.cat([wm2_action_history, wm2_action_chunk], dim=1)
        wm1_step_batch["action"] = wm1_step_action
        wm2_step_batch["action"] = wm2_step_action

        wm1_pred_seg_raw = infer_pred_latents_raw(
            model=wm1,
            args=wm1_args,
            accelerator=accelerator,
            batch=wm1_step_batch,
            history_latent=wm1_history,
            current_latent=wm1_current,
        )
        wm1_gt_seg_raw = wm1_step_batch[wm1_args.predicted_datatype].to(
            accelerator.device,
            non_blocking=True,
        )
        w1_pred_seg_video, _, w1_pred_seg_lat, _ = decode_and_format_outputs_for_metrics(
            wm1_pred_seg_raw,
            wm1_gt_seg_raw,
            wm1_args,
            wm1_pipeline,
        )
        wm1_pred_seg_video_chunks.append(w1_pred_seg_video.detach().cpu())
        wm1_pred_seg_latent_chunks.append(w1_pred_seg_lat.detach().cpu())

        wm1_pred_for_wm2 = apply_snap_to_palette_on_seg_latents(
            seg_latents_btchw=wm1_pred_seg_raw,
            pipeline=wm1_pipeline,
            decode_chunk_size=wm1_args.decode_chunk_size,
            enable_snap_to_palette=snap_to_palette,
        )

        wm2_step_cascade_batch = build_wm2_cascade_batch(
            batch=wm2_step_batch,
            seg_history_latents=wm1_history,
            wm1_pred_seg_latents=wm1_pred_for_wm2,
            wm2_args=wm2_args,
            wm1_pipeline=wm1_pipeline,
        )
        wm2_step_cascade_batch["action"] = wm2_step_action

        wm2_pred_raw = infer_pred_latents_raw(
            model=wm2,
            args=wm2_args,
            accelerator=accelerator,
            batch=wm2_step_cascade_batch,
            history_latent=wm2_history,
            current_latent=wm2_current,
            disable_action_conditioning=wm2_disable_action_conditioning,
        )

        wm2_gt_raw = wm2_step_batch[wm2_args.predicted_datatype].to(
            accelerator.device,
            non_blocking=True,
        )
        w2_pred_video, _, w2_pred_lat, _ = decode_and_format_outputs_for_metrics(
            wm2_pred_raw,
            wm2_gt_raw,
            wm2_args,
            wm2_pipeline,
        )
        wm2_pred_video_chunks.append(w2_pred_video)
        wm2_pred_latent_chunks.append(w2_pred_lat)

        wm1_history = torch.cat([wm1_history, wm1_pred_seg_raw], dim=1)[:, -wm1_args.num_history :]
        wm1_current = wm1_pred_seg_raw[:, -1]
        wm2_history = torch.cat([wm2_history, wm2_pred_raw], dim=1)[:, -wm2_args.num_history :]
        wm2_current = wm2_pred_raw[:, -1]
        wm1_action_history = torch.cat([wm1_action_history, wm1_action_chunk], dim=1)[:, -wm1_args.num_history :]
        wm2_action_history = torch.cat([wm2_action_history, wm2_action_chunk], dim=1)[:, -wm2_args.num_history :]

    return {
        "wm1_pred_seg_video": torch.cat(wm1_pred_seg_video_chunks, dim=1),
        "wm1_pred_seg_latents": torch.cat(wm1_pred_seg_latent_chunks, dim=1),
        "wm2_pred_video": torch.cat(wm2_pred_video_chunks, dim=1),
        "wm2_pred_latents": torch.cat(wm2_pred_latent_chunks, dim=1),
    }


def main_baseline_wm(baseline_wm_args: wm_orca_args, args_cli: argparse.Namespace):
    if args_cli.baseline_wm_num_frames is not None:
        baseline_wm_args.num_frames = int(args_cli.baseline_wm_num_frames)
    if args_cli.baseline_wm_num_history is not None:
        baseline_wm_args.num_history = int(args_cli.baseline_wm_num_history)
    if args_cli.baseline_wm_num_inference_steps is not None:
        baseline_wm_args.num_inference_steps = int(args_cli.baseline_wm_num_inference_steps)

    step_horizon = int(baseline_wm_args.num_frames)
    total_future_frames = int(args_cli.ar_num_steps) * step_horizon

    dataset_args = copy.deepcopy(baseline_wm_args)
    # Evaluate on the same validation samples for every model, whatever episodes its
    # training config excluded.
    dataset_args.exclude_episode_ids_by_dataset = {}
    dataset_args.num_frames = total_future_frames

    use_wandb = bool(args_cli.wandb_project_name or args_cli.wandb_run_name)
    accelerator = Accelerator(
        mixed_precision=baseline_wm_args.mixed_precision,
        log_with="wandb" if use_wandb else None,
    )

    if use_wandb and accelerator.is_main_process:
        wandb_project_name = args_cli.wandb_project_name or getattr(
            baseline_wm_args, "wandb_project_name", "ctrl-world"
        )
        wandb_run_name = args_cli.wandb_run_name or getattr(
            baseline_wm_args, "wandb_run_name", "baseline_wm_sine_actions"
        )
        accelerator.init_trackers(
            wandb_project_name,
            init_kwargs={"wandb": {"name": wandb_run_name}},
        )

    model = _load_ctrl_world_from_checkpoint(
        baseline_wm_args,
        args_cli.baseline_wm_model_ckpt_path,
        model_label="baseline_wm",
    )
    model = accelerator.prepare(model)

    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    test_dataset = Dataset_mix(dataset_args, mode=args_cli.mode)
    sample_indices = resolve_sample_indices(args_cli, test_dataset)
    print(f"Selected sample ids for baseline sine evaluation: {sample_indices}")

    first_sample = test_dataset.__getitem__(index=sample_indices[0], dataset_id=args_cli.dataset_id)
    first_base_batch = to_device_batch(first_sample, accelerator.device)

    action_dim = int(first_base_batch["action"].shape[-1])
    stat_path = get_stat_path(dataset_args, args_cli.dataset_id)
    action_low, action_high = get_action_bounds_from_stat(stat_path, dataset_args, action_dim)
    baseline_stat_path = get_model_stat_path(baseline_wm_args, "baseline_wm", args_cli.dataset_id)
    baseline_action_low, baseline_action_high = get_action_bounds_from_stat(
        baseline_stat_path,
        baseline_wm_args,
        action_dim,
    )

    cycles_list = parse_float_list(args_cli.sine_cycles)
    amp_scales = parse_float_list(args_cli.sine_amplitude_scales)
    variant_specs = build_variant_specs(cycles_list, amp_scales, args_cli.variants_per_component)

    if args_cli.component_indices:
        component_indices = parse_int_list(args_cli.component_indices)
        invalid = [idx for idx in component_indices if idx < 0 or idx >= action_dim]
        if invalid:
            raise ValueError(
                f"component_indices contains invalid indices {invalid}; action_dim={action_dim}."
            )
    else:
        max_dims = action_dim if args_cli.max_components <= 0 else min(action_dim, args_cli.max_components)
        component_indices = list(range(max_dims))

    excluded_component_indices = set()
    if args_cli.exclude_component_indices:
        excluded_component_indices = set(parse_int_list(args_cli.exclude_component_indices))
        invalid = [idx for idx in excluded_component_indices if idx < 0 or idx >= action_dim]
        if invalid:
            raise ValueError(
                f"exclude_component_indices contains invalid indices {invalid}; action_dim={action_dim}."
            )
        component_indices = [idx for idx in component_indices if idx not in excluded_component_indices]

    if not component_indices:
        raise ValueError("No action components remain after applying component filters.")

    os.makedirs(args_cli.output_path, exist_ok=True)
    videos_dir = os.path.join(args_cli.output_path, "videos")
    tensors_dir = os.path.join(args_cli.output_path, "tensors")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    manifest = {
        "sample_indices": [int(idx) for idx in sample_indices],
        "episode_ids": parse_int_list(args_cli.episode_ids) if args_cli.episode_ids else None,
        "num_episode_samples": args_cli.num_episode_samples,
        "num_random_samples": args_cli.num_random_samples,
        "dataset_id": int(args_cli.dataset_id),
        "dataset_name": resolve_dataset_name(dataset_args, args_cli.dataset_id),
        "stat_path": stat_path,
        "baseline_wm_stat_path": baseline_stat_path,
        "action_dim": action_dim,
        "component_indices": [int(idx) for idx in component_indices],
        "excluded_component_indices": [int(idx) for idx in sorted(excluded_component_indices)],
        "num_history": int(baseline_wm_args.num_history),
        "step_horizon": step_horizon,
        "ar_num_steps": int(args_cli.ar_num_steps),
        "total_future_frames": total_future_frames,
        "variants_per_component": int(args_cli.variants_per_component),
        "predicted_datatype": baseline_wm_args.predicted_datatype,
        "variants": [
            {
                "variant_id": int(v.variant_id),
                "cycles": float(v.cycles),
                "amplitude_scale": float(v.amplitude_scale),
            }
            for v in variant_specs
        ],
        "results": [],
    }

    total_rollouts = len(sample_indices) * len(component_indices) * len(variant_specs)
    print(
        f"Running baseline sine sweep: samples={sample_indices}, components={component_indices}, "
        f"variants/component={len(variant_specs)}, total_rollouts={total_rollouts}"
    )

    rollout_idx = 0
    for sample_pos, sample_idx in enumerate(sample_indices):
        if sample_pos == 0:
            base_batch = first_base_batch
        else:
            sample = test_dataset.__getitem__(index=sample_idx, dataset_id=args_cli.dataset_id)
            base_batch = to_device_batch(sample, accelerator.device)

        base_action_abs = get_current_observed_action_abs(
            test_dataset=test_dataset,
            dataset_args=dataset_args,
            sample_idx=sample_idx,
            dataset_id=args_cli.dataset_id,
            expected_action_dim=action_dim,
        )
        if base_action_abs is None:
            base_action_norm = base_batch["action"][0, baseline_wm_args.num_history].detach().cpu().numpy()
            base_action_abs = denormalize_bound(
                base_action_norm[None, :],
                action_low[None, :],
                action_high[None, :],
            )[0].astype(np.float32)
            base_action_source = "dataset_action"
        else:
            base_action_source = "current_observation"

        case_specs = []
        actions_abs_cases = []
        for dim_idx in component_indices:
            for spec in variant_specs:
                actions_abs_future = generate_sine_future_actions_abs(
                    base_action_abs=base_action_abs,
                    low=action_low,
                    high=action_high,
                    total_future_frames=total_future_frames,
                    target_dim=dim_idx,
                    cycles=spec.cycles,
                    amplitude_scale=spec.amplitude_scale,
                )
                case_specs.append((dim_idx, spec, actions_abs_future))
                actions_abs_cases.append(actions_abs_future)

        batched_base_batch = repeat_batch(base_batch, len(actions_abs_cases))
        actions_abs_batch = np.stack(actions_abs_cases, axis=0)

        print(
            f"Running baseline sample_id={sample_idx} ({sample_pos + 1}/{len(sample_indices)}), "
            f"base_action_source={base_action_source}, "
            f"batched_component_rollouts={len(actions_abs_cases)}"
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
                )

        for case_pos, (dim_idx, spec, actions_abs_future) in enumerate(case_specs):
            view_start = case_pos * baseline_wm_args.num_views
            view_end = (case_pos + 1) * baseline_wm_args.num_views
            pred_video = out["pred_video"][view_start:view_end]
            pred_latents = out["pred_latents"][view_start:view_end]
            video_grid = make_video_grid_uint8(pred_video, num_views=baseline_wm_args.num_views)

            case_name = (
                f"sample_{sample_idx:05d}_dim_{dim_idx:02d}_var_{spec.variant_id:02d}"
                f"_cycles_{spec.cycles:.3f}_amp_{spec.amplitude_scale:.3f}"
            )
            video_path = os.path.join(videos_dir, f"{case_name}.mp4")
            latent_path = os.path.join(tensors_dir, f"{case_name}_pred_latents.pt")
            action_path = os.path.join(tensors_dir, f"{case_name}_actions_abs.npy")

            mediapy.write_video(video_path, video_grid, fps=baseline_wm_args.fps)
            torch.save(pred_latents.detach().cpu(), latent_path)
            np.save(action_path, actions_abs_future)

            if use_wandb and accelerator.is_main_process:
                accelerator.log(
                    {
                        f"video/{case_name}": wandb.Video(
                            video_path,
                            fps=baseline_wm_args.wandb_video_display_fps,
                            format="mp4",
                        ),
                    },
                    step=rollout_idx,
                )

            amp_max = float(
                min(
                    base_action_abs[dim_idx] - action_low[dim_idx],
                    action_high[dim_idx] - base_action_abs[dim_idx],
                )
            )
            actions_norm_future = normalize_bound(
                actions_abs_future,
                action_low[None, :],
                action_high[None, :],
            )
            manifest["results"].append(
                {
                    "sample_idx": int(sample_idx),
                    "component": int(dim_idx),
                    "variant_id": int(spec.variant_id),
                    "cycles": float(spec.cycles),
                    "amplitude_scale": float(spec.amplitude_scale),
                    "base_action_source": base_action_source,
                    "base_action_abs": float(base_action_abs[dim_idx]),
                    "safe_amplitude_max": amp_max,
                    "action_abs_min": float(actions_abs_future[:, dim_idx].min()),
                    "action_abs_max": float(actions_abs_future[:, dim_idx].max()),
                    "action_norm_min": float(actions_norm_future[:, dim_idx].min()),
                    "action_norm_max": float(actions_norm_future[:, dim_idx].max()),
                    "video_path": video_path,
                    "latent_path": latent_path,
                    "action_path": action_path,
                }
            )
            rollout_idx += 1
            print(f"Saved {video_path}")

    manifest_path = os.path.join(args_cli.output_path, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if use_wandb:
        accelerator.end_training()
    print(f"Done. Manifest: {manifest_path}")



def main(wm1_args: wm_orca_args, wm2_args: wm_orca_args, args_cli: argparse.Namespace):
    if args_cli.wm1_num_frames is not None:
        wm1_args.num_frames = int(args_cli.wm1_num_frames)
    if args_cli.wm2_num_frames is not None:
        wm2_args.num_frames = int(args_cli.wm2_num_frames)

    if wm1_args.num_frames != wm2_args.num_frames:
        raise ValueError(
            f"Autoregressive sine test requires wm1_num_frames == wm2_num_frames. "
            f"Got {wm1_args.num_frames} vs {wm2_args.num_frames}."
        )

    step_horizon = int(wm1_args.num_frames)
    total_future_frames = int(args_cli.ar_num_steps) * step_horizon

    dataset_args = copy.deepcopy(wm2_args)
    # Evaluate on the same validation samples for every model, whatever episodes its
    # training config excluded.
    dataset_args.exclude_episode_ids_by_dataset = {}
    dataset_args.num_frames = total_future_frames

    accelerator = Accelerator(mixed_precision=wm2_args.mixed_precision, log_with="wandb")
    if args_cli.sequential_wm1_wm2_loading and accelerator.num_processes > 1:
        raise ValueError("--sequential_wm1_wm2_loading currently supports single-process inference only.")

    if accelerator.is_main_process:
        wandb_project_name = getattr(args_cli, "wandb_project_name", None) or getattr(wm2_args, "wandb_project_name", "ctrl-world")
        wandb_run_name = getattr(args_cli, "wandb_run_name", None) or getattr(wm2_args, "wandb_run_name", "wm1_to_wm2_sine_actions")
        accelerator.init_trackers(
            wandb_project_name,
            init_kwargs={"wandb": {"name": wandb_run_name}},
        )

    wm1 = _load_ctrl_world_from_checkpoint(
        wm1_args,
        args_cli.wm1_model_ckpt_path,
        model_label="WM1",
    )
    wm2 = None
    if not args_cli.sequential_wm1_wm2_loading:
        wm2 = _load_ctrl_world_from_checkpoint(
            wm2_args,
            args_cli.wm2_model_ckpt_path,
            model_label="WM2",
        )
        wm1, wm2 = accelerator.prepare(wm1, wm2)

    sequential_model_cache = None
    if args_cli.sequential_wm1_wm2_loading:
        sequential_model_cache = {"wm1": wm1, "wm2": wm2}

    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    test_dataset = Dataset_mix(dataset_args, mode=args_cli.mode)
    sample_indices = resolve_sample_indices(args_cli, test_dataset)
    print(f"Selected sample ids for sine evaluation: {sample_indices}")

    first_sample = test_dataset.__getitem__(index=sample_indices[0], dataset_id=args_cli.dataset_id)
    first_base_batch = to_device_batch(first_sample, accelerator.device)

    action_dim = int(first_base_batch["action"].shape[-1])
    stat_path = get_stat_path(dataset_args, args_cli.dataset_id)
    action_low, action_high = get_action_bounds_from_stat(stat_path, dataset_args, action_dim)
    wm1_stat_path = get_model_stat_path(wm1_args, "wm1", args_cli.dataset_id)
    wm2_stat_path = get_model_stat_path(wm2_args, "wm2", args_cli.dataset_id)
    wm1_action_low, wm1_action_high = get_action_bounds_from_stat(wm1_stat_path, wm1_args, action_dim)
    wm2_action_low, wm2_action_high = get_action_bounds_from_stat(wm2_stat_path, wm2_args, action_dim)

    cycles_list = parse_float_list(args_cli.sine_cycles)
    amp_scales = parse_float_list(args_cli.sine_amplitude_scales)
    variant_specs = build_variant_specs(cycles_list, amp_scales, args_cli.variants_per_component)

    if args_cli.component_indices:
        component_indices = parse_int_list(args_cli.component_indices)
        invalid = [idx for idx in component_indices if idx < 0 or idx >= action_dim]
        if invalid:
            raise ValueError(
                f"component_indices contains invalid indices {invalid}; action_dim={action_dim}."
            )
    else:
        max_dims = action_dim if args_cli.max_components <= 0 else min(action_dim, args_cli.max_components)
        component_indices = list(range(max_dims))

    excluded_component_indices = set()
    if args_cli.exclude_component_indices:
        excluded_component_indices = set(parse_int_list(args_cli.exclude_component_indices))
        invalid = [idx for idx in excluded_component_indices if idx < 0 or idx >= action_dim]
        if invalid:
            raise ValueError(
                f"exclude_component_indices contains invalid indices {invalid}; action_dim={action_dim}."
            )
        component_indices = [idx for idx in component_indices if idx not in excluded_component_indices]

    if not component_indices:
        raise ValueError("No action components remain after applying component filters.")

    os.makedirs(args_cli.output_path, exist_ok=True)
    videos_dir = os.path.join(args_cli.output_path, "videos")
    tensors_dir = os.path.join(args_cli.output_path, "tensors")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    manifest = {
        "sample_indices": [int(idx) for idx in sample_indices],
        "episode_ids": parse_int_list(args_cli.episode_ids) if args_cli.episode_ids else None,
        "num_episode_samples": args_cli.num_episode_samples,
        "num_random_samples": args_cli.num_random_samples,
        "dataset_id": int(args_cli.dataset_id),
        "dataset_name": resolve_dataset_name(dataset_args, args_cli.dataset_id),
        "stat_path": stat_path,
        "wm1_stat_path": wm1_stat_path,
        "wm2_stat_path": wm2_stat_path,
        "action_dim": action_dim,
        "component_indices": [int(idx) for idx in component_indices],
        "excluded_component_indices": [int(idx) for idx in sorted(excluded_component_indices)],
        "num_history": int(wm1_args.num_history),
        "step_horizon": step_horizon,
        "ar_num_steps": int(args_cli.ar_num_steps),
        "total_future_frames": total_future_frames,
        "variants_per_component": int(args_cli.variants_per_component),
        "sequential_wm1_wm2_loading": bool(args_cli.sequential_wm1_wm2_loading),
        "variants": [
            {
                "variant_id": int(v.variant_id),
                "cycles": float(v.cycles),
                "amplitude_scale": float(v.amplitude_scale),
            }
            for v in variant_specs
        ],
        "results": [],
    }

    total_rollouts = len(sample_indices) * len(component_indices) * len(variant_specs)
    print(
        f"Running sine sweep: samples={sample_indices}, components={component_indices}, "
        f"variants/component={len(variant_specs)}, total_rollouts={total_rollouts}"
    )

    rollout_idx = 0
    for sample_pos, sample_idx in enumerate(sample_indices):
        if sample_pos == 0:
            base_batch = first_base_batch
        else:
            sample = test_dataset.__getitem__(index=sample_idx, dataset_id=args_cli.dataset_id)
            base_batch = to_device_batch(sample, accelerator.device)

        base_action_abs = get_current_observed_action_abs(
            test_dataset=test_dataset,
            dataset_args=dataset_args,
            sample_idx=sample_idx,
            dataset_id=args_cli.dataset_id,
            expected_action_dim=action_dim,
        )
        if base_action_abs is None:
            base_action_norm = base_batch["action"][0, wm1_args.num_history].detach().cpu().numpy()
            base_action_abs = denormalize_bound(
                base_action_norm[None, :],
                action_low[None, :],
                action_high[None, :],
            )[0].astype(np.float32)
            base_action_source = "dataset_action"
        else:
            base_action_source = "current_observation"

        case_specs = []
        actions_abs_cases = []
        for dim_idx in component_indices:
            for spec in variant_specs:
                actions_abs_future = generate_sine_future_actions_abs(
                    base_action_abs=base_action_abs,
                    low=action_low,
                    high=action_high,
                    total_future_frames=total_future_frames,
                    target_dim=dim_idx,
                    cycles=spec.cycles,
                    amplitude_scale=spec.amplitude_scale,
                )
                case_specs.append((dim_idx, spec, actions_abs_future))
                actions_abs_cases.append(actions_abs_future)

        batched_base_batch = repeat_batch(base_batch, len(actions_abs_cases))
        actions_abs_batch = np.stack(actions_abs_cases, axis=0)

        print(
            f"Running sample_id={sample_idx} ({sample_pos + 1}/{len(sample_indices)}), "
            f"base_action_source={base_action_source}, "
            f"batched_component_rollouts={len(actions_abs_cases)}"
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
                )

        for case_pos, (dim_idx, spec, actions_abs_future) in enumerate(case_specs):
            view_start = case_pos * wm2_args.num_views
            view_end = (case_pos + 1) * wm2_args.num_views
            wm1_view_start = case_pos * wm1_args.num_views
            wm1_view_end = (case_pos + 1) * wm1_args.num_views

            wm1_pred_seg_video = out["wm1_pred_seg_video"][wm1_view_start:wm1_view_end]
            wm1_pred_seg_latents = out["wm1_pred_seg_latents"][wm1_view_start:wm1_view_end]
            wm2_pred_video = out["wm2_pred_video"][view_start:view_end]
            wm2_pred_latents = out["wm2_pred_latents"][view_start:view_end]
            wm1_seg_video_grid = make_video_grid_uint8(wm1_pred_seg_video, num_views=wm1_args.num_views)
            video_grid = make_video_grid_uint8(wm2_pred_video, num_views=wm2_args.num_views)

            case_name = (
                f"sample_{sample_idx:05d}_dim_{dim_idx:02d}_var_{spec.variant_id:02d}"
                f"_cycles_{spec.cycles:.3f}_amp_{spec.amplitude_scale:.3f}"
            )
            video_path = os.path.join(videos_dir, f"{case_name}.mp4")
            wm1_seg_video_path = os.path.join(videos_dir, f"{case_name}_wm1_seg.mp4")
            latent_path = os.path.join(tensors_dir, f"{case_name}_pred_latents.pt")
            wm1_seg_latent_path = os.path.join(tensors_dir, f"{case_name}_wm1_seg_latents.pt")
            action_path = os.path.join(tensors_dir, f"{case_name}_actions_abs.npy")

            mediapy.write_video(video_path, video_grid, fps=wm2_args.fps)
            mediapy.write_video(wm1_seg_video_path, wm1_seg_video_grid, fps=wm1_args.fps)
            torch.save(wm2_pred_latents.detach().cpu(), latent_path)
            torch.save(wm1_pred_seg_latents.detach().cpu(), wm1_seg_latent_path)
            np.save(action_path, actions_abs_future)

            if accelerator.is_main_process:
                accelerator.log(
                    {
                        f"video/{case_name}": wandb.Video(
                            video_path,
                            fps=wm2_args.wandb_video_display_fps,
                            format="mp4",
                        ),
                        f"video_wm1_seg/{case_name}": wandb.Video(
                            wm1_seg_video_path,
                            fps=wm1_args.wandb_video_display_fps,
                            format="mp4",
                        ),
                    },
                    step=rollout_idx,
                )

            amp_max = float(
                min(
                    base_action_abs[dim_idx] - action_low[dim_idx],
                    action_high[dim_idx] - base_action_abs[dim_idx],
                )
            )
            actions_norm_future = normalize_bound(
                actions_abs_future,
                action_low[None, :],
                action_high[None, :],
            )
            manifest["results"].append(
                {
                    "sample_idx": int(sample_idx),
                    "component": int(dim_idx),
                    "variant_id": int(spec.variant_id),
                    "cycles": float(spec.cycles),
                    "amplitude_scale": float(spec.amplitude_scale),
                    "base_action_source": base_action_source,
                    "base_action_abs": float(base_action_abs[dim_idx]),
                    "safe_amplitude_max": amp_max,
                    "action_abs_min": float(actions_abs_future[:, dim_idx].min()),
                    "action_abs_max": float(actions_abs_future[:, dim_idx].max()),
                    "action_norm_min": float(actions_norm_future[:, dim_idx].min()),
                    "action_norm_max": float(actions_norm_future[:, dim_idx].max()),
                    "video_path": video_path,
                    "wm1_seg_video_path": wm1_seg_video_path,
                    "latent_path": latent_path,
                    "wm1_seg_latent_path": wm1_seg_latent_path,
                    "action_path": action_path,
                }
            )
            rollout_idx += 1
            print(f"Saved {video_path}")
            print(f"Saved {wm1_seg_video_path}")

    manifest_path = os.path.join(args_cli.output_path, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    accelerator.end_training()
    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm1_model_ckpt_path", type=str, default=None)
    parser.add_argument("--wm2_model_ckpt_path", type=str, default=None)
    parser.add_argument("--baseline_wm_model_ckpt_path", type=str, default=None)
    parser.add_argument("--wm1_finetune_lora_ckpt_path", type=str, default=None)
    parser.add_argument("--wm2_finetune_lora_ckpt_path", type=str, default=None)
    parser.add_argument("--baseline_wm_finetune_lora_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--output_path",
        type=str,
        default=f"inference_output/wm1_to_wm2_sine_actions_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--wm1_config_path", type=str, default=None)
    parser.add_argument("--wm2_config_path", type=str, default=None)
    parser.add_argument("--baseline_wm_config_path", type=str, default=None)

    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, default=None)
    parser.add_argument("--dataset_meta_info_path", type=str, default=None)
    parser.add_argument("--dataset_meta_info_name", type=str, default=None)
    parser.add_argument(
        "--dataset_stat_path",
        "--data_stat_path",
        dest="dataset_stat_path",
        type=str,
        default=None,
        help="Optional stat.json path used for dataset/action normalization instead of per-dataset meta stats.",
    )
    parser.add_argument(
        "--wm1_dataset_stat_path",
        type=str,
        default=None,
        help="Optional stat.json path used to normalize WM1 action conditioning.",
    )
    parser.add_argument(
        "--wm2_dataset_stat_path",
        type=str,
        default=None,
        help="Optional stat.json path used to normalize WM2 action conditioning.",
    )
    parser.add_argument(
        "--baseline_wm_dataset_stat_path",
        type=str,
        default=None,
        help="Optional stat.json path used to normalize baseline WM action conditioning.",
    )
    parser.add_argument("--mode", type=str, default="val")

    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="",
        help="Optional comma-separated sample indices. Overrides --sample_idx and --num_random_samples.",
    )
    parser.add_argument(
        "--num_random_samples",
        type=int,
        default=None,
        help="Randomly sample this many unique indices from the selected dataset split.",
    )
    parser.add_argument(
        "--episode_ids",
        type=str,
        default="",
        help="Optional comma-separated episode IDs; selects matching samples after dataset filtering/subsetting.",
    )
    parser.add_argument(
        "--num_episode_samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples selected by --episode_ids.",
    )
    parser.add_argument("--dataset_id", type=int, default=0)

    parser.add_argument("--wm1_num_frames", type=int, default=5)
    parser.add_argument("--wm2_num_frames", type=int, default=5)
    parser.add_argument("--baseline_wm_num_frames", type=int, default=5)
    parser.add_argument("--wm1_num_history", type=int, default=None)
    parser.add_argument("--wm2_num_history", type=int, default=None)
    parser.add_argument("--baseline_wm_num_history", type=int, default=None)
    parser.add_argument("--wm1_num_inference_steps", type=int, default=None)
    parser.add_argument("--wm2_num_inference_steps", type=int, default=None)
    parser.add_argument("--baseline_wm_num_inference_steps", type=int, default=None)
    parser.add_argument("--ar_num_steps", type=int, default=30)

    parser.add_argument("--sine_cycles", type=str, default="2.0")
    parser.add_argument("--sine_amplitude_scales", type=str, default="0.6")
    parser.add_argument("--variants_per_component", type=int, default=1)
    parser.add_argument("--max_components", type=int, default=0, help="<=0 means all action components")
    parser.add_argument(
        "--component_indices",
        type=str,
        default="",
        help="Optional comma-separated action component indices to sweep; overrides --max_components.",
    )
    parser.add_argument(
        "--exclude_component_indices",
        type=str,
        default="",
        help="Optional comma-separated action component indices to remove from the sweep.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--snap_to_palette", action="store_true", default=False)
    parser.add_argument("--wm2_disable_action_conditioning", action="store_true", default=False)
    parser.add_argument(
        "--sequential_wm1_wm2_loading",
        action="store_true",
        default=False,
        help=(
            "Keep only one cascade model on GPU at a time: run the full WM1 sine rollout, "
            "offload WM1 to CPU, then load WM2 for the full conditioned video rollout."
        ),
    )
    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args_cli = parser.parse_args()

    baseline_mode = args_cli.baseline_wm_model_ckpt_path is not None
    if not baseline_mode:
        missing = [
            flag for flag in ("wm1_model_ckpt_path", "wm2_model_ckpt_path", "wm1_config_path", "wm2_config_path")
            if getattr(args_cli, flag) is None
        ]
        if missing:
            parser.error(
                "cascade mode needs " + ", ".join(f"--{flag}" for flag in missing)
                + " (or pass --baseline_wm_model_ckpt_path for a monolithic model)"
            )

    wm1_args = wm_orca_args()
    if args_cli.wm1_config_path is not None and not baseline_mode:
        wm1_args = load_experiment_config(args_cli.wm1_config_path, wm1_args)

    wm2_args = wm_orca_args()
    if args_cli.wm2_config_path is not None and not baseline_mode:
        wm2_args = load_experiment_config(args_cli.wm2_config_path, wm2_args)

    baseline_wm_args = wm_orca_args()
    if args_cli.baseline_wm_config_path is not None:
        baseline_wm_args = load_experiment_config(args_cli.baseline_wm_config_path, baseline_wm_args)
    elif baseline_mode:
        raise ValueError("--baseline_wm_config_path is required when using --baseline_wm_model_ckpt_path.")

    if args_cli.dataset_root_path is not None:
        wm1_args.dataset_root_path = args_cli.dataset_root_path
        wm2_args.dataset_root_path = args_cli.dataset_root_path
        baseline_wm_args.dataset_root_path = args_cli.dataset_root_path
    if args_cli.dataset_names is not None:
        wm1_args.dataset_names = args_cli.dataset_names
        wm2_args.dataset_names = args_cli.dataset_names
        baseline_wm_args.dataset_names = args_cli.dataset_names
    if args_cli.dataset_meta_info_path is not None:
        wm1_args.dataset_meta_info_path = args_cli.dataset_meta_info_path
        wm2_args.dataset_meta_info_path = args_cli.dataset_meta_info_path
        baseline_wm_args.dataset_meta_info_path = args_cli.dataset_meta_info_path
    if args_cli.dataset_meta_info_name is not None:
        wm1_args.dataset_meta_info_name = args_cli.dataset_meta_info_name
        wm2_args.dataset_meta_info_name = args_cli.dataset_meta_info_name
        baseline_wm_args.dataset_meta_info_name = args_cli.dataset_meta_info_name
    if args_cli.dataset_stat_path is not None:
        wm1_args.dataset_stat_path = args_cli.dataset_stat_path
        wm2_args.dataset_stat_path = args_cli.dataset_stat_path
        baseline_wm_args.dataset_stat_path = args_cli.dataset_stat_path

    model_specific_stat_paths = {
        "wm1_dataset_stat_path": args_cli.wm1_dataset_stat_path
        or getattr(wm1_args, "wm1_dataset_stat_path", None),
        "wm2_dataset_stat_path": args_cli.wm2_dataset_stat_path
        or getattr(wm2_args, "wm2_dataset_stat_path", None),
        "baseline_wm_dataset_stat_path": args_cli.baseline_wm_dataset_stat_path
        or getattr(baseline_wm_args, "baseline_wm_dataset_stat_path", None),
    }
    for stat_attr, stat_path in model_specific_stat_paths.items():
        if stat_path is None:
            continue
        setattr(wm1_args, stat_attr, stat_path)
        setattr(wm2_args, stat_attr, stat_path)
        setattr(baseline_wm_args, stat_attr, stat_path)
    if args_cli.wm1_finetune_lora_ckpt_path is not None:
        wm1_args.finetune_lora_ckpt_path = args_cli.wm1_finetune_lora_ckpt_path
    if args_cli.wm2_finetune_lora_ckpt_path is not None:
        wm2_args.finetune_lora_ckpt_path = args_cli.wm2_finetune_lora_ckpt_path
    if args_cli.baseline_wm_finetune_lora_ckpt_path is not None:
        baseline_wm_args.finetune_lora_ckpt_path = args_cli.baseline_wm_finetune_lora_ckpt_path

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
        baseline_wm_args.num_inference_steps = 1
        args_cli.ar_num_steps = min(2, args_cli.ar_num_steps)
        args_cli.max_components = min(2, args_cli.max_components) if args_cli.max_components > 0 else 2

    if args_cli.wm1_num_frames <= 0 or args_cli.wm2_num_frames <= 0:
        raise ValueError("wm1_num_frames and wm2_num_frames must be positive.")
    if args_cli.baseline_wm_num_frames <= 0:
        raise ValueError("baseline_wm_num_frames must be positive.")
    if args_cli.ar_num_steps <= 0:
        raise ValueError("ar_num_steps must be positive.")
    if args_cli.variants_per_component < 1:
        raise ValueError("variants_per_component must be >= 1.")

    if baseline_mode:
        main_baseline_wm(baseline_wm_args, args_cli)
    else:
        main(wm1_args, wm2_args, args_cli)
