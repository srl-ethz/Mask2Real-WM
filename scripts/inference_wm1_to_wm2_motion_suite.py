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
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import mediapy
import numpy as np
import torch
import wandb
from accelerate import Accelerator

from config import wm_orca_args
from dataset.dataset_orca import Dataset_mix
from utils.config_loader import load_experiment_config

from scripts.inference_wm1_to_wm2_sine_actions import (
    denormalize_bound,
    get_action_bounds_from_stat,
    get_current_observed_action_abs,
    get_model_stat_path,
    get_stat_path,
    make_video_grid_uint8,
    normalize_bound,
    repeat_batch,
    resolve_dataset_name,
    resolve_sample_indices,
    run_autoregressive_baseline_wm_with_actions,
    run_autoregressive_wm1_wm2_with_actions,
    to_device_batch,
    _load_ctrl_world_from_checkpoint,
    _offload_ctrl_world_model,
)


HAND_JOINT_LABELS = {
    6: "wrist",
    7: "thumb_mcp",
    8: "thumb_abd",
    9: "thumb_pip",
    10: "thumb_dip",
    11: "index_abd",
    12: "index_mcp",
    13: "index_pip",
    14: "middle_abd",
    15: "middle_mcp",
    16: "middle_pip",
    17: "ring_abd",
    18: "ring_mcp",
    19: "ring_pip",
    20: "pinky_abd",
    21: "pinky_mcp",
    22: "pinky_pip",
}

EE_LABELS = ["x", "y", "z", "roll", "pitch", "yaw"]
ACTION_LABELS = EE_LABELS + [HAND_JOINT_LABELS[i] for i in range(6, 23)]
HAND_DIMS = tuple(range(6, 23))
FINGER_JOINT_DIMS = tuple(range(7, 23))
THUMB_SWAP_FULL_DIMS = (7, 8)


@dataclass(frozen=True)
class MotionCase:
    name: str
    category: str
    actions_abs: np.ndarray
    metadata: Dict


def parse_csv(raw: str) -> List[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _phase(total_frames: int, cycles: float = 1.0) -> np.ndarray:
    return np.linspace(0.0, 2.0 * math.pi * float(cycles), total_frames, endpoint=False, dtype=np.float32)


def _clip_to_bounds(actions_abs: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    return np.clip(actions_abs, action_low[None, :], action_high[None, :]).astype(np.float32)


def _repeat_base(base_action_abs: np.ndarray, total_frames: int) -> np.ndarray:
    return np.repeat(base_action_abs[None, :], total_frames, axis=0).astype(np.float32)


def _norm_to_abs(norm_value: float, dim: int, action_low: np.ndarray, action_high: np.ndarray) -> float:
    norm_arr = np.zeros((1, action_low.shape[0]), dtype=np.float32)
    norm_arr[0, dim] = float(np.clip(norm_value, -1.0, 1.0))
    abs_arr = denormalize_bound(norm_arr, action_low[None, :], action_high[None, :])
    return float(abs_arr[0, dim])


def _set_norm_targets(
    action_abs: np.ndarray,
    targets_by_dim: Dict[int, float],
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    out = action_abs.copy()
    for dim, norm_value in targets_by_dim.items():
        if dim < 0 or dim >= out.shape[-1]:
            raise ValueError(f"Target dim {dim} is out of range for action_dim={out.shape[-1]}.")
        out[dim] = _norm_to_abs(norm_value, dim, action_low, action_high)
    return out.astype(np.float32)


def _interpolate_hold(
    base_action_abs: np.ndarray,
    target_action_abs: np.ndarray,
    total_frames: int,
    transition_fraction: float,
) -> np.ndarray:
    transition_steps = int(round(total_frames * float(transition_fraction)))
    transition_steps = max(1, min(total_frames, transition_steps))
    alphas = np.linspace(0.0, 1.0, transition_steps + 1, dtype=np.float32)[1:]
    transition = np.stack(
        [base_action_abs + alpha * (target_action_abs - base_action_abs) for alpha in alphas],
        axis=0,
    )
    if transition_steps == total_frames:
        return transition.astype(np.float32)
    hold = np.repeat(target_action_abs[None, :], total_frames - transition_steps, axis=0)
    return np.concatenate([transition, hold], axis=0).astype(np.float32)


def _bounded_sine_dim(
    base_action_abs: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    total_frames: int,
    dim: int,
    cycles: float,
    amplitude_scale: float,
) -> np.ndarray:
    out = _repeat_base(base_action_abs, total_frames)
    denom = action_high[dim] - action_low[dim] + 1e-8
    base_norm = 2.0 * (base_action_abs[dim] - action_low[dim]) / denom - 1.0
    base_norm = float(np.clip(base_norm, -1.0, 1.0))
    raw = np.sin(_phase(total_frames, cycles) + np.arcsin(base_norm))
    delta = raw - base_norm
    amp = float(np.clip(amplitude_scale, 0.0, 1.0))
    norm = np.full_like(raw, base_norm)
    max_up = float(np.max(delta))
    max_down = float(abs(np.min(delta)))
    if max_up > 1e-8:
        norm = np.where(delta > 0.0, base_norm + delta / max_up * (1.0 - base_norm) * amp, norm)
    if max_down > 1e-8:
        norm = np.where(delta < 0.0, base_norm + delta / max_down * (base_norm + 1.0) * amp, norm)
    out[:, dim] = (np.clip(norm, -1.0, 1.0) + 1.0) * 0.5 * (action_high[dim] - action_low[dim]) + action_low[dim]
    return out.astype(np.float32)


def maybe_swap_thumb_abd_mcp(actions_abs: np.ndarray, swap_abd_with_mcp: bool) -> np.ndarray:
    if not swap_abd_with_mcp:
        return actions_abs.astype(np.float32, copy=True)
    out = actions_abs.astype(np.float32, copy=True)
    out[..., list(THUMB_SWAP_FULL_DIMS)] = out[..., list(reversed(THUMB_SWAP_FULL_DIMS))]
    return out


def build_symbol_targets(
    extended_norm: float,
    curled_norm: float,
    spread_norm: float,
) -> Dict[str, Dict[int, float]]:
    thumb_extended = extended_norm
    thumb_curled = curled_norm
    return {
        "peace": {
            7: thumb_curled, 8: thumb_curled, 9: thumb_curled, 10: thumb_curled,
            11: -spread_norm, 12: extended_norm, 13: extended_norm,
            14: spread_norm, 15: extended_norm, 16: extended_norm,
            17: 0.0, 18: curled_norm, 19: curled_norm,
            20: 0.0, 21: curled_norm, 22: curled_norm,
        },
        "phone": {
            7: thumb_extended, 8: thumb_extended, 9: thumb_extended, 10: thumb_extended,
            12: curled_norm, 13: curled_norm,
            15: curled_norm, 16: curled_norm,
            18: curled_norm, 19: curled_norm,
            21: extended_norm, 22: extended_norm,
        },
        "fist": {dim: curled_norm for dim in FINGER_JOINT_DIMS},
        "gun": {
            7: thumb_extended, 8: thumb_extended, 9: thumb_extended, 10: thumb_extended,
            11: -spread_norm, 12: extended_norm, 13: extended_norm,
            15: curled_norm, 16: curled_norm,
            18: curled_norm, 19: curled_norm,
            21: curled_norm, 22: curled_norm,
        },
        "ok": {
            7: thumb_curled, 8: thumb_curled, 9: thumb_curled, 10: thumb_curled,
            11: -0.5 * spread_norm, 12: curled_norm, 13: curled_norm,
            14: 0.4 * spread_norm, 15: extended_norm, 16: extended_norm,
            17: 0.2 * spread_norm, 18: extended_norm, 19: extended_norm,
            20: 0.0, 21: extended_norm, 22: extended_norm,
        },
    }


def build_motion_suite(
    base_action_abs: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    total_frames: int,
    args_cli: argparse.Namespace,
) -> List[MotionCase]:
    if base_action_abs.ndim != 1:
        raise ValueError(f"base_action_abs must be 1D, got shape {base_action_abs.shape}.")
    if base_action_abs.shape[0] < 23:
        raise ValueError(
            "The requested motion suite expects the full 23D action layout "
            f"(6 EE pose + 17 hand joints), got action_dim={base_action_abs.shape[0]}."
        )

    cases: List[MotionCase] = []

    circle = _repeat_base(base_action_abs, total_frames)
    theta = _phase(total_frames, args_cli.ee_cycles)
    circle[:, 0] = base_action_abs[0] + float(args_cli.xy_radius) * np.cos(theta)
    circle[:, 1] = base_action_abs[1] + float(args_cli.xy_radius) * np.sin(theta)
    circle = _clip_to_bounds(circle, action_low, action_high)
    cases.append(MotionCase(
        name="ee_circle_xy",
        category="end_effector",
        actions_abs=circle,
        metadata={"xy_radius": float(args_cli.xy_radius), "cycles": float(args_cli.ee_cycles)},
    ))

    z_motion = _repeat_base(base_action_abs, total_frames)
    z_motion[:, 2] = base_action_abs[2] + float(args_cli.z_amplitude) * np.sin(theta)
    z_motion = _clip_to_bounds(z_motion, action_low, action_high)
    cases.append(MotionCase(
        name="ee_up_down_z",
        category="end_effector",
        actions_abs=z_motion,
        metadata={"z_amplitude": float(args_cli.z_amplitude), "cycles": float(args_cli.ee_cycles)},
    ))

    rotations = _repeat_base(base_action_abs, total_frames)
    for dim, phase_offset in zip((3, 4, 5), (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)):
        rotations[:, dim] = base_action_abs[dim] + float(args_cli.rotation_amplitude) * np.sin(theta + phase_offset)
    rotations = _clip_to_bounds(rotations, action_low, action_high)
    cases.append(MotionCase(
        name="ee_combined_rotations_rpy",
        category="end_effector",
        actions_abs=rotations,
        metadata={
            "rotation_amplitude_radians": float(args_cli.rotation_amplitude),
            "rotation_amplitude_degrees": float(math.degrees(args_cli.rotation_amplitude)),
            "cycles": float(args_cli.ee_cycles),
        },
    ))

    if not args_cli.skip_individual_joints:
        for dim in HAND_DIMS:
            actions = _bounded_sine_dim(
                base_action_abs=base_action_abs,
                action_low=action_low,
                action_high=action_high,
                total_frames=total_frames,
                dim=dim,
                cycles=args_cli.joint_cycles,
                amplitude_scale=args_cli.joint_amplitude_scale,
            )
            cases.append(MotionCase(
                name=f"joint_{dim:02d}_{HAND_JOINT_LABELS.get(dim, f'dim_{dim}')}",
                category="individual_joint",
                actions_abs=actions,
                metadata={
                    "dim": int(dim),
                    "label": HAND_JOINT_LABELS.get(dim, f"dim_{dim}"),
                    "cycles": float(args_cli.joint_cycles),
                    "amplitude_scale": float(args_cli.joint_amplitude_scale),
                },
            ))

    combined = _repeat_base(base_action_abs, total_frames)
    for offset, dim in enumerate(HAND_DIMS):
        dim_actions = _bounded_sine_dim(
            base_action_abs=base_action_abs,
            action_low=action_low,
            action_high=action_high,
            total_frames=total_frames,
            dim=dim,
            cycles=args_cli.joint_cycles,
            amplitude_scale=args_cli.combined_joint_amplitude_scale,
        )
        phase_sign = 1.0 if offset % 2 == 0 else -1.0
        combined[:, dim] = base_action_abs[dim] + phase_sign * (dim_actions[:, dim] - base_action_abs[dim])
    combined = _clip_to_bounds(combined, action_low, action_high)
    cases.append(MotionCase(
        name="joints_combined_wave",
        category="combined_joint",
        actions_abs=combined,
        metadata={
            "dims": [int(dim) for dim in HAND_DIMS],
            "cycles": float(args_cli.joint_cycles),
            "amplitude_scale": float(args_cli.combined_joint_amplitude_scale),
            "phase_pattern": "alternating_sign",
        },
    ))

    symbol_targets = build_symbol_targets(
        extended_norm=args_cli.symbol_extended_norm,
        curled_norm=args_cli.symbol_curled_norm,
        spread_norm=args_cli.symbol_spread_norm,
    )
    for symbol_name in ("peace", "phone", "fist", "gun", "ok"):
        target = _set_norm_targets(base_action_abs, symbol_targets[symbol_name], action_low, action_high)
        actions = _interpolate_hold(
            base_action_abs=base_action_abs,
            target_action_abs=target,
            total_frames=total_frames,
            transition_fraction=args_cli.symbol_transition_fraction,
        )
        actions = _clip_to_bounds(actions, action_low, action_high)
        cases.append(MotionCase(
            name=f"symbol_{symbol_name}",
            category="symbol",
            actions_abs=actions,
            metadata={
                "symbol": symbol_name,
                "transition_fraction": float(args_cli.symbol_transition_fraction),
                "norm_targets": {
                    HAND_JOINT_LABELS.get(dim, f"dim_{dim}"): float(norm_value)
                    for dim, norm_value in symbol_targets[symbol_name].items()
                },
            },
        ))

    requested = set(parse_csv(args_cli.case_names))
    if requested:
        known = {case.name for case in cases}
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f"Unknown --case_names entries {missing}. Known cases: {sorted(known)}")
        cases = [case for case in cases if case.name in requested]

    if args_cli.max_cases > 0:
        cases = cases[: int(args_cli.max_cases)]
    if not cases:
        raise ValueError("No motion cases selected.")

    expected_shape = (total_frames, base_action_abs.shape[0])
    for case in cases:
        if case.actions_abs.shape != expected_shape:
            raise ValueError(f"{case.name} has shape {case.actions_abs.shape}; expected {expected_shape}.")
    return cases


def prepare_args(args_cli: argparse.Namespace) -> Tuple[wm_orca_args, wm_orca_args, wm_orca_args]:
    wm1_args = load_experiment_config(args_cli.wm1_config_path, wm_orca_args())
    wm2_args = load_experiment_config(args_cli.wm2_config_path, wm_orca_args())
    baseline_args = load_experiment_config(args_cli.baseline_wm_config_path, wm_orca_args())

    for cfg in (wm1_args, wm2_args, baseline_args):
        cfg.dataset_root_path = args_cli.dataset_root_path
        cfg.dataset_names = args_cli.dataset_names
        cfg.dataset_meta_info_path = args_cli.dataset_meta_info_path
        cfg.dataset_meta_info_name = args_cli.dataset_meta_info_name
        cfg.dataset_stat_path = args_cli.dataset_stat_path
        cfg.wm1_dataset_stat_path = args_cli.wm1_dataset_stat_path
        cfg.wm2_dataset_stat_path = args_cli.wm2_dataset_stat_path
        cfg.baseline_wm_dataset_stat_path = args_cli.baseline_wm_dataset_stat_path

    wm1_args.finetune_lora_ckpt_path = args_cli.wm1_finetune_lora_ckpt_path
    if args_cli.wm2_finetune_lora_ckpt_path is not None:
        wm2_args.finetune_lora_ckpt_path = args_cli.wm2_finetune_lora_ckpt_path
    if args_cli.baseline_wm_finetune_lora_ckpt_path is not None:
        baseline_args.finetune_lora_ckpt_path = args_cli.baseline_wm_finetune_lora_ckpt_path

    wm1_args.num_frames = int(args_cli.wm1_num_frames)
    wm2_args.num_frames = int(args_cli.wm2_num_frames)
    baseline_args.num_frames = int(args_cli.baseline_wm_num_frames)
    wm1_args.num_history = int(args_cli.wm1_num_history)
    wm2_args.num_history = int(args_cli.wm2_num_history)
    baseline_args.num_history = int(args_cli.baseline_wm_num_history)
    wm1_args.num_inference_steps = int(args_cli.wm1_num_inference_steps)
    wm2_args.num_inference_steps = int(args_cli.wm2_num_inference_steps)
    baseline_args.num_inference_steps = int(args_cli.baseline_wm_num_inference_steps)

    return wm1_args, wm2_args, baseline_args


def prepare_dataset_args(
    wm1_args: wm_orca_args,
    wm2_args: wm_orca_args,
    baseline_args: wm_orca_args,
    total_future_frames: int,
) -> wm_orca_args:
    dataset_args = copy.deepcopy(wm2_args)
    dataset_args.num_frames = int(total_future_frames)
    dataset_args.num_history = max(
        int(wm1_args.num_history),
        int(wm2_args.num_history),
        int(baseline_args.num_history),
    )
    dataset_args.max_num_samples_for_validation = 10**18
    dataset_args.min_stride = 1
    dataset_args.max_stride = 1

    # Keep the default action in raw physical order; model-specific action_* keys
    # below apply the training-time joint swap and normalization for each model.
    dataset_args.swap_abd_with_mcp = False
    dataset_args.wm1_swap_abd_with_mcp = bool(getattr(wm1_args, "swap_abd_with_mcp", False))
    dataset_args.wm2_swap_abd_with_mcp = bool(getattr(wm2_args, "swap_abd_with_mcp", False))
    dataset_args.baseline_wm_swap_abd_with_mcp = bool(getattr(baseline_args, "swap_abd_with_mcp", False))
    return dataset_args


def save_baseline_outputs(
    out: Dict[str, torch.Tensor],
    batch_cases: Sequence[MotionCase],
    case_offset: int,
    model_args: wm_orca_args,
    action_low: np.ndarray,
    action_high: np.ndarray,
    videos_dir: str,
    tensors_dir: str,
    fps: int,
) -> List[Dict]:
    records = []
    for local_idx, case in enumerate(batch_cases):
        view_start = local_idx * int(model_args.num_views)
        view_end = (local_idx + 1) * int(model_args.num_views)
        pred_video = out["pred_video"][view_start:view_end]
        pred_latents = out["pred_latents"][view_start:view_end]
        video_grid = make_video_grid_uint8(pred_video, num_views=model_args.num_views)

        case_name = f"{case_offset + local_idx:03d}_{case.name}"
        video_path = os.path.join(videos_dir, f"{case_name}.mp4")
        latent_path = os.path.join(tensors_dir, f"{case_name}_pred_latents.pt")
        action_abs_path = os.path.join(tensors_dir, f"{case_name}_actions_abs.npy")
        action_norm_path = os.path.join(tensors_dir, f"{case_name}_actions_norm_dataset.npy")

        mediapy.write_video(video_path, video_grid, fps=fps)
        torch.save(pred_latents.detach().cpu(), latent_path)
        np.save(action_abs_path, case.actions_abs)
        action_norm = normalize_bound(case.actions_abs, action_low[None, :], action_high[None, :])
        np.save(action_norm_path, action_norm)

        records.append({
            "case": case.name,
            "category": case.category,
            "metadata": case.metadata,
            "video_path": video_path,
            "latent_path": latent_path,
            "actions_abs_path": action_abs_path,
            "actions_norm_dataset_path": action_norm_path,
            "action_norm_min": action_norm.min(axis=0).tolist(),
            "action_norm_max": action_norm.max(axis=0).tolist(),
        })
        print(f"Saved baseline rollout: {video_path}")
    return records


def save_cascade_outputs(
    out: Dict[str, torch.Tensor],
    batch_cases: Sequence[MotionCase],
    case_offset: int,
    wm1_args: wm_orca_args,
    wm2_args: wm_orca_args,
    action_low: np.ndarray,
    action_high: np.ndarray,
    videos_dir: str,
    tensors_dir: str,
) -> List[Dict]:
    records = []
    for local_idx, case in enumerate(batch_cases):
        wm2_view_start = local_idx * int(wm2_args.num_views)
        wm2_view_end = (local_idx + 1) * int(wm2_args.num_views)
        wm1_view_start = local_idx * int(wm1_args.num_views)
        wm1_view_end = (local_idx + 1) * int(wm1_args.num_views)

        wm2_video_grid = make_video_grid_uint8(
            out["wm2_pred_video"][wm2_view_start:wm2_view_end],
            num_views=wm2_args.num_views,
        )
        wm1_seg_video_grid = make_video_grid_uint8(
            out["wm1_pred_seg_video"][wm1_view_start:wm1_view_end],
            num_views=wm1_args.num_views,
        )

        case_name = f"{case_offset + local_idx:03d}_{case.name}"
        wm2_video_path = os.path.join(videos_dir, f"{case_name}_wm2.mp4")
        wm1_seg_video_path = os.path.join(videos_dir, f"{case_name}_wm1_seg.mp4")
        wm2_latent_path = os.path.join(tensors_dir, f"{case_name}_wm2_pred_latents.pt")
        wm1_seg_latent_path = os.path.join(tensors_dir, f"{case_name}_wm1_seg_latents.pt")
        action_abs_path = os.path.join(tensors_dir, f"{case_name}_actions_abs.npy")
        action_norm_path = os.path.join(tensors_dir, f"{case_name}_actions_norm_dataset.npy")

        mediapy.write_video(wm2_video_path, wm2_video_grid, fps=wm2_args.fps)
        mediapy.write_video(wm1_seg_video_path, wm1_seg_video_grid, fps=wm1_args.fps)
        torch.save(out["wm2_pred_latents"][wm2_view_start:wm2_view_end].detach().cpu(), wm2_latent_path)
        torch.save(out["wm1_pred_seg_latents"][wm1_view_start:wm1_view_end].detach().cpu(), wm1_seg_latent_path)
        np.save(action_abs_path, case.actions_abs)
        action_norm = normalize_bound(case.actions_abs, action_low[None, :], action_high[None, :])
        np.save(action_norm_path, action_norm)

        records.append({
            "case": case.name,
            "category": case.category,
            "metadata": case.metadata,
            "wm2_video_path": wm2_video_path,
            "wm1_seg_video_path": wm1_seg_video_path,
            "wm2_latent_path": wm2_latent_path,
            "wm1_seg_latent_path": wm1_seg_latent_path,
            "actions_abs_path": action_abs_path,
            "actions_norm_dataset_path": action_norm_path,
            "action_norm_min": action_norm.min(axis=0).tolist(),
            "action_norm_max": action_norm.max(axis=0).tolist(),
        })
        print(f"Saved cascade rollout: {wm2_video_path}")
        print(f"Saved cascade WM1 segmentation rollout: {wm1_seg_video_path}")
    return records


def log_records_to_wandb(
    accelerator: Accelerator,
    records: Sequence[Dict],
    model_key: str,
    display_fps: int,
    step_offset: int,
) -> None:
    if not accelerator.is_main_process:
        return
    for idx, record in enumerate(records):
        payload = {}
        if model_key == "baseline_wm":
            payload[f"video/{model_key}/{record['case']}"] = wandb.Video(
                record["video_path"],
                fps=display_fps,
                format="mp4",
            )
        else:
            payload[f"video/{model_key}/{record['case']}_wm2"] = wandb.Video(
                record["wm2_video_path"],
                fps=display_fps,
                format="mp4",
            )
            payload[f"video/{model_key}/{record['case']}_wm1_seg"] = wandb.Video(
                record["wm1_seg_video_path"],
                fps=display_fps,
                format="mp4",
            )
        accelerator.log(payload, step=step_offset + idx)


def run_baseline_suite(
    accelerator: Accelerator,
    baseline_args: wm_orca_args,
    args_cli: argparse.Namespace,
    base_batch: Dict,
    cases: Sequence[MotionCase],
    baseline_action_low: np.ndarray,
    baseline_action_high: np.ndarray,
    dataset_action_low: np.ndarray,
    dataset_action_high: np.ndarray,
    output_path: str,
    use_wandb: bool,
) -> List[Dict]:
    videos_dir = os.path.join(output_path, "videos", "baseline_wm")
    tensors_dir = os.path.join(output_path, "tensors", "baseline_wm")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    model = _load_ctrl_world_from_checkpoint(
        baseline_args,
        args_cli.baseline_wm_model_ckpt_path,
        model_label="baseline_wm",
    )
    model = accelerator.prepare(model)

    records: List[Dict] = []
    for start in range(0, len(cases), args_cli.case_batch_size):
        batch_cases = cases[start:start + args_cli.case_batch_size]
        batch = repeat_batch(base_batch, len(batch_cases))
        actions_for_model = np.stack(
            [
                maybe_swap_thumb_abd_mcp(case.actions_abs, bool(getattr(baseline_args, "swap_abd_with_mcp", False)))
                for case in batch_cases
            ],
            axis=0,
        )
        print(f"Running baseline motion cases {start}..{start + len(batch_cases) - 1}")
        with torch.no_grad():
            with accelerator.autocast():
                out = run_autoregressive_baseline_wm_with_actions(
                    model=model,
                    model_args=baseline_args,
                    accelerator=accelerator,
                    base_batch=batch,
                    actions_abs_future=actions_for_model,
                    action_low=baseline_action_low,
                    action_high=baseline_action_high,
                    ar_num_steps=args_cli.ar_num_steps,
                    snap_to_palette=args_cli.snap_to_palette,
                )
        batch_records = save_baseline_outputs(
            out=out,
            batch_cases=batch_cases,
            case_offset=start,
            model_args=baseline_args,
            action_low=dataset_action_low,
            action_high=dataset_action_high,
            videos_dir=videos_dir,
            tensors_dir=tensors_dir,
            fps=baseline_args.fps,
        )
        records.extend(batch_records)
        if use_wandb:
            log_records_to_wandb(
                accelerator,
                batch_records,
                model_key="baseline_wm",
                display_fps=baseline_args.wandb_video_display_fps,
                step_offset=start,
            )

    _offload_ctrl_world_model(model, accelerator)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def run_cascade_suite(
    accelerator: Accelerator,
    wm1_args: wm_orca_args,
    wm2_args: wm_orca_args,
    args_cli: argparse.Namespace,
    base_batch: Dict,
    cases: Sequence[MotionCase],
    wm1_action_low: np.ndarray,
    wm1_action_high: np.ndarray,
    wm2_action_low: np.ndarray,
    wm2_action_high: np.ndarray,
    dataset_action_low: np.ndarray,
    dataset_action_high: np.ndarray,
    output_path: str,
    use_wandb: bool,
) -> List[Dict]:
    videos_dir = os.path.join(output_path, "videos", "wm1_finetuned_wm2")
    tensors_dir = os.path.join(output_path, "tensors", "wm1_finetuned_wm2")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    if bool(getattr(wm1_args, "swap_abd_with_mcp", False)) != bool(getattr(wm2_args, "swap_abd_with_mcp", False)):
        raise ValueError(
            "This script expects WM1 and WM2 to use the same swap_abd_with_mcp setting "
            "so a single absolute action sequence can condition both cascade stages."
        )

    wm1 = _load_ctrl_world_from_checkpoint(wm1_args, args_cli.wm1_model_ckpt_path, model_label="WM1")
    wm2 = None
    if not args_cli.sequential_wm1_wm2_loading:
        wm2 = _load_ctrl_world_from_checkpoint(wm2_args, args_cli.wm2_model_ckpt_path, model_label="WM2")
        wm1, wm2 = accelerator.prepare(wm1, wm2)
    model_cache = {"wm1": wm1, "wm2": wm2} if args_cli.sequential_wm1_wm2_loading else None

    records: List[Dict] = []
    for start in range(0, len(cases), args_cli.case_batch_size):
        batch_cases = cases[start:start + args_cli.case_batch_size]
        batch = repeat_batch(base_batch, len(batch_cases))
        actions_for_model = np.stack(
            [
                maybe_swap_thumb_abd_mcp(case.actions_abs, bool(getattr(wm1_args, "swap_abd_with_mcp", False)))
                for case in batch_cases
            ],
            axis=0,
        )
        print(f"Running WM1-finetuned -> WM2 motion cases {start}..{start + len(batch_cases) - 1}")
        with torch.no_grad():
            with accelerator.autocast():
                out = run_autoregressive_wm1_wm2_with_actions(
                    wm1=wm1,
                    wm2=wm2,
                    wm1_args=wm1_args,
                    wm2_args=wm2_args,
                    accelerator=accelerator,
                    base_batch=batch,
                    actions_abs_future=actions_for_model,
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
                    model_cache=model_cache,
                )
        batch_records = save_cascade_outputs(
            out=out,
            batch_cases=batch_cases,
            case_offset=start,
            wm1_args=wm1_args,
            wm2_args=wm2_args,
            action_low=dataset_action_low,
            action_high=dataset_action_high,
            videos_dir=videos_dir,
            tensors_dir=tensors_dir,
        )
        records.extend(batch_records)
        if use_wandb:
            log_records_to_wandb(
                accelerator,
                batch_records,
                model_key="wm1_finetuned_wm2",
                display_fps=wm2_args.wandb_video_display_fps,
                step_offset=start,
            )

    if model_cache is not None:
        _offload_ctrl_world_model(model_cache.get("wm1"), accelerator)
        _offload_ctrl_world_model(model_cache.get("wm2"), accelerator)
    else:
        _offload_ctrl_world_model(wm1, accelerator)
        _offload_ctrl_world_model(wm2, accelerator)
    del wm1
    del wm2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def main(args_cli: argparse.Namespace) -> None:
    if args_cli.ar_num_steps <= 0:
        raise ValueError("--ar_num_steps must be positive.")
    if args_cli.case_batch_size <= 0:
        raise ValueError("--case_batch_size must be positive.")

    wm1_args, wm2_args, baseline_args = prepare_args(args_cli)
    if wm1_args.num_frames != wm2_args.num_frames:
        raise ValueError(f"WM1 and WM2 horizons must match, got {wm1_args.num_frames} and {wm2_args.num_frames}.")
    if baseline_args.num_frames != wm1_args.num_frames:
        raise ValueError(
            "The baseline and cascade horizons must match for paired motion-suite rollouts, "
            f"got baseline={baseline_args.num_frames}, cascade={wm1_args.num_frames}."
        )

    step_horizon = int(wm1_args.num_frames)
    total_future_frames = int(args_cli.ar_num_steps) * step_horizon
    dataset_args = prepare_dataset_args(wm1_args, wm2_args, baseline_args, total_future_frames)

    use_wandb = bool(args_cli.wandb_project_name or args_cli.wandb_run_name)
    accelerator = Accelerator(
        mixed_precision=wm2_args.mixed_precision,
        log_with="wandb" if use_wandb else None,
        project_dir=args_cli.output_path,
    )
    if args_cli.sequential_wm1_wm2_loading and accelerator.num_processes > 1:
        raise ValueError("--sequential_wm1_wm2_loading supports single-process inference only.")
    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            args_cli.wandb_project_name or getattr(wm2_args, "wandb_project_name", "ctrl-world"),
            init_kwargs={"wandb": {"name": args_cli.wandb_run_name or "motion_suite_baseline_lora_vs_wm1_finetuned_wm2"}},
        )

    set_seed(args_cli.seed)
    test_dataset = Dataset_mix(dataset_args, mode=args_cli.mode)
    sample_indices = resolve_sample_indices(args_cli, test_dataset)
    sample_idx = int(sample_indices[0])
    sample_record = test_dataset.samples_all[args_cli.dataset_id][sample_idx]
    print(f"Selected sample_idx={sample_idx}, sample_record={sample_record}")

    sample = test_dataset.__getitem__(index=sample_idx, dataset_id=args_cli.dataset_id)
    base_batch = to_device_batch(sample, accelerator.device)
    action_dim = int(base_batch["action"].shape[-1])

    dataset_stat_path = get_stat_path(dataset_args, args_cli.dataset_id)
    dataset_action_low, dataset_action_high = get_action_bounds_from_stat(dataset_stat_path, dataset_args, action_dim)
    wm1_stat_path = get_model_stat_path(wm1_args, "wm1", args_cli.dataset_id)
    wm2_stat_path = get_model_stat_path(wm2_args, "wm2", args_cli.dataset_id)
    baseline_stat_path = get_model_stat_path(baseline_args, "baseline_wm", args_cli.dataset_id)
    wm1_action_low, wm1_action_high = get_action_bounds_from_stat(wm1_stat_path, wm1_args, action_dim)
    wm2_action_low, wm2_action_high = get_action_bounds_from_stat(wm2_stat_path, wm2_args, action_dim)
    baseline_action_low, baseline_action_high = get_action_bounds_from_stat(baseline_stat_path, baseline_args, action_dim)

    base_action_abs = get_current_observed_action_abs(
        test_dataset=test_dataset,
        dataset_args=dataset_args,
        sample_idx=sample_idx,
        dataset_id=args_cli.dataset_id,
        expected_action_dim=action_dim,
    )
    if base_action_abs is None:
        base_action_norm = base_batch["action"][0, dataset_args.num_history].detach().cpu().numpy()
        base_action_abs = denormalize_bound(
            base_action_norm[None, :],
            dataset_action_low[None, :],
            dataset_action_high[None, :],
        )[0].astype(np.float32)
        base_action_source = "dataset_action"
    else:
        base_action_source = "current_observation"

    cases = build_motion_suite(
        base_action_abs=base_action_abs,
        action_low=dataset_action_low,
        action_high=dataset_action_high,
        total_frames=total_future_frames,
        args_cli=args_cli,
    )
    print(f"Built {len(cases)} motion cases with {total_future_frames} future action frames.")

    os.makedirs(args_cli.output_path, exist_ok=True)
    manifest = {
        "script": os.path.basename(__file__),
        "dataset_name": resolve_dataset_name(dataset_args, args_cli.dataset_id),
        "dataset_root_path": dataset_args.dataset_root_path,
        "dataset_meta_info_path": dataset_args.dataset_meta_info_path,
        "dataset_meta_info_name": dataset_args.dataset_meta_info_name,
        "mode": args_cli.mode,
        "dataset_id": int(args_cli.dataset_id),
        "sample_idx": sample_idx,
        "sample_record": sample_record,
        "base_action_source": base_action_source,
        "base_action_abs_raw_order": base_action_abs.tolist(),
        "action_dim": action_dim,
        "action_labels": ACTION_LABELS,
        "seed": int(args_cli.seed),
        "step_horizon": step_horizon,
        "ar_num_steps": int(args_cli.ar_num_steps),
        "total_future_frames": total_future_frames,
        "long_rollout_note": (
            "Dataset frame ids are clamped by Dataset_mix when the seed clip is shorter than "
            "num_history + total_future_frames; generated outputs are synthetic rollouts, not metric claims."
        ),
        "paths": {
            "wm1_config_path": args_cli.wm1_config_path,
            "wm2_config_path": args_cli.wm2_config_path,
            "baseline_wm_config_path": args_cli.baseline_wm_config_path,
            "wm1_model_ckpt_path": args_cli.wm1_model_ckpt_path,
            "wm1_finetune_lora_ckpt_path": args_cli.wm1_finetune_lora_ckpt_path,
            "wm2_model_ckpt_path": args_cli.wm2_model_ckpt_path,
            "baseline_wm_model_ckpt_path": args_cli.baseline_wm_model_ckpt_path,
            "dataset_stat_path": dataset_stat_path,
            "wm1_stat_path": wm1_stat_path,
            "wm2_stat_path": wm2_stat_path,
            "baseline_wm_stat_path": baseline_stat_path,
        },
        "model_action_layout": {
            "raw_order_for_saved_actions": True,
            "wm1_swap_abd_with_mcp": bool(getattr(wm1_args, "swap_abd_with_mcp", False)),
            "wm2_swap_abd_with_mcp": bool(getattr(wm2_args, "swap_abd_with_mcp", False)),
            "baseline_wm_swap_abd_with_mcp": bool(getattr(baseline_args, "swap_abd_with_mcp", False)),
            "swapped_full_dims_when_enabled": list(THUMB_SWAP_FULL_DIMS),
        },
        "motion_generation": {
            "xy_radius": float(args_cli.xy_radius),
            "z_amplitude": float(args_cli.z_amplitude),
            "rotation_amplitude": float(args_cli.rotation_amplitude),
            "ee_cycles": float(args_cli.ee_cycles),
            "joint_cycles": float(args_cli.joint_cycles),
            "joint_amplitude_scale": float(args_cli.joint_amplitude_scale),
            "combined_joint_amplitude_scale": float(args_cli.combined_joint_amplitude_scale),
            "symbol_extended_norm": float(args_cli.symbol_extended_norm),
            "symbol_curled_norm": float(args_cli.symbol_curled_norm),
            "symbol_spread_norm": float(args_cli.symbol_spread_norm),
            "symbol_transition_fraction": float(args_cli.symbol_transition_fraction),
        },
        "cases": [
            {"name": case.name, "category": case.category, "metadata": case.metadata}
            for case in cases
        ],
        "results": {},
    }

    if args_cli.models in ("both", "baseline"):
        manifest["results"]["baseline_wm"] = run_baseline_suite(
            accelerator=accelerator,
            baseline_args=baseline_args,
            args_cli=args_cli,
            base_batch=base_batch,
            cases=cases,
            baseline_action_low=baseline_action_low,
            baseline_action_high=baseline_action_high,
            dataset_action_low=dataset_action_low,
            dataset_action_high=dataset_action_high,
            output_path=args_cli.output_path,
            use_wandb=use_wandb,
        )

    if args_cli.models in ("both", "cascade"):
        manifest["results"]["wm1_finetuned_wm2"] = run_cascade_suite(
            accelerator=accelerator,
            wm1_args=wm1_args,
            wm2_args=wm2_args,
            args_cli=args_cli,
            base_batch=base_batch,
            cases=cases,
            wm1_action_low=wm1_action_low,
            wm1_action_high=wm1_action_high,
            wm2_action_low=wm2_action_low,
            wm2_action_high=wm2_action_high,
            dataset_action_low=dataset_action_low,
            dataset_action_high=dataset_action_high,
            output_path=args_cli.output_path,
            use_wandb=use_wandb,
        )

    manifest_path = os.path.join(args_cli.output_path, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if use_wandb:
        accelerator.end_training()
    print(f"Done. Saved manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_wm_model_ckpt_path",
        type=str,
        default=os.path.join(project_root, "checkpoints/mono_sr_58000/model.safetensors"),
    )
    parser.add_argument(
        "--wm1_model_ckpt_path",
        type=str,
        default=os.path.join(project_root, "checkpoints/wm1_cascade_sr/model.safetensors"),
    )
    parser.add_argument("--wm1_finetune_lora_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--wm2_model_ckpt_path",
        type=str,
        default=os.path.join(project_root, "checkpoints/wm2/model.safetensors"),
    )
    parser.add_argument("--baseline_wm_finetune_lora_ckpt_path", type=str, default=None)
    parser.add_argument("--wm2_finetune_lora_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--baseline_wm_config_path",
        type=str,
        default=os.path.join(project_root, "experiments/mask2real/mono_sr.yaml"),
    )
    parser.add_argument(
        "--wm1_config_path",
        type=str,
        default=os.path.join(project_root, "experiments/mask2real/wm1_cascade_sr.yaml"),
    )
    parser.add_argument(
        "--wm2_config_path",
        type=str,
        default=os.path.join(project_root, "experiments/mask2real/wm2.yaml"),
    )
    parser.add_argument(
        "--dataset_root_path",
        type=str,
        default=os.path.join(project_root, "datasets/2026-03-14T13-34-49"),
    )
    parser.add_argument("--dataset_names", type=str, default="large_real_dataset_5fps_135_240")
    parser.add_argument(
        "--dataset_meta_info_path",
        type=str,
        default=os.path.join(project_root, "dataset_meta_info/2026-03-14T13-34-49"),
    )
    parser.add_argument("--dataset_meta_info_name", type=str, default=None)
    parser.add_argument(
        "--dataset_stat_path",
        type=str,
        default=os.path.join(
            project_root,
            "dataset_meta_info/2026-03-14T13-34-49/large_real_dataset_5fps_135_240/stat.json",
        ),
    )
    parser.add_argument(
        "--baseline_wm_dataset_stat_path",
        type=str,
        default=os.path.join(
            project_root,
            "dataset_meta_info/2026-03-14T13-34-49/large_real_dataset_5fps_135_240/stat.json",
        ),
    )
    parser.add_argument(
        "--wm1_dataset_stat_path",
        type=str,
        default=os.path.join(
            project_root,
            "dataset_meta_info/2026-05-19T03-28-53/SIM_PRETRAINING_DATA_ALL_lerobot_5fps/stat.json",
        ),
    )
    parser.add_argument(
        "--wm2_dataset_stat_path",
        type=str,
        default=os.path.join(
            project_root,
            "dataset_meta_info/2026-05-19T03-28-53/SIM_PRETRAINING_DATA_ALL_lerobot_5fps/stat.json",
        ),
    )

    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--dataset_id", type=int, default=0)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--sample_indices", type=str, default="")
    parser.add_argument("--episode_ids", type=str, default="")
    parser.add_argument("--num_episode_samples", type=int, default=None)
    parser.add_argument("--num_random_samples", type=int, default=None)

    parser.add_argument("--baseline_wm_num_frames", type=int, default=5)
    parser.add_argument("--wm1_num_frames", type=int, default=5)
    parser.add_argument("--wm2_num_frames", type=int, default=5)
    parser.add_argument("--baseline_wm_num_history", type=int, default=5)
    parser.add_argument("--wm1_num_history", type=int, default=5)
    parser.add_argument("--wm2_num_history", type=int, default=5)
    parser.add_argument("--baseline_wm_num_inference_steps", type=int, default=50)
    parser.add_argument("--wm1_num_inference_steps", type=int, default=50)
    parser.add_argument("--wm2_num_inference_steps", type=int, default=50)
    parser.add_argument("--ar_num_steps", type=int, default=120)

    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(
            project_root,
            f"inference_output/motion_suite_baseline_lora_vs_wm1_finetuned_wm2_{time.strftime('%Y%m%d_%H%M%S')}",
        ),
    )
    parser.add_argument("--models", choices=["both", "baseline", "cascade"], default="both")
    parser.add_argument("--case_batch_size", type=int, default=1)
    parser.add_argument("--case_names", type=str, default="")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--skip_individual_joints", action="store_true", default=False)

    parser.add_argument("--xy_radius", type=float, default=0.04)
    parser.add_argument("--z_amplitude", type=float, default=0.05)
    parser.add_argument("--rotation_amplitude", type=float, default=0.60)
    parser.add_argument("--ee_cycles", type=float, default=2.0)
    parser.add_argument("--joint_cycles", type=float, default=2.0)
    parser.add_argument("--joint_amplitude_scale", type=float, default=0.85)
    parser.add_argument("--combined_joint_amplitude_scale", type=float, default=0.55)
    parser.add_argument("--symbol_extended_norm", type=float, default=-1.0)
    parser.add_argument("--symbol_curled_norm", type=float, default=1.0)
    parser.add_argument("--symbol_spread_norm", type=float, default=0.75)
    parser.add_argument("--symbol_transition_fraction", type=float, default=0.25)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snap_to_palette", action="store_true", default=False)
    parser.add_argument("--wm2_disable_action_conditioning", action="store_true", default=False)
    parser.add_argument("--sequential_wm1_wm2_loading", action="store_true", default=True)
    parser.add_argument("--no_sequential_wm1_wm2_loading", dest="sequential_wm1_wm2_loading", action="store_false")
    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    if args.debug:
        args.baseline_wm_num_inference_steps = 1
        args.wm1_num_inference_steps = 1
        args.wm2_num_inference_steps = 1
        args.ar_num_steps = min(args.ar_num_steps, 2)
        args.max_cases = 2 if args.max_cases <= 0 else min(args.max_cases, 2)
        args.case_batch_size = 1

    main(args)
