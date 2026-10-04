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
import json
import datetime
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import wandb

import einops
import lpips
import torch
import torch.nn.functional as F
from accelerate import Accelerator

torch.backends.cudnn.enabled = True

# config related imports
from config import wm_orca_args
from utils.config_loader import load_experiment_config
from utils.utils import send_discord_message, snap_to_palette, SEMANTIC_SEGMENTATION_MAPPING

# dataset related imports
from dataset.dataset_orca import Dataset_mix

# model related imports
from models.ctrl_world import CtrlWorld, latent_seg_heightstack_to_channelstack
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from scripts.train_wm import (
    _is_lora_adapter_key,
    _load_torch_checkpoint,
    args_to_dict,
    load_finetune_lora_checkpoint,
)


def _extract_model_state_dict(checkpoint_payload):
    """Return model weights from either bare state_dict or wrapped training checkpoint."""
    if not isinstance(checkpoint_payload, dict):
        raise ValueError(
            f"Unsupported checkpoint format: expected dict, got {type(checkpoint_payload)}."
        )

    def _looks_like_tensor_state_dict(candidate):
        return bool(candidate) and all(torch.is_tensor(v) for v in candidate.values())

    # Training checkpoints may be nested, e.g.
    # {"state_dict": {"model": {...}, "optimizer": {...}}}.
    unwrap_keys = ("model", "state_dict", "model_state_dict", "module")
    for _ in range(8):
        if _looks_like_tensor_state_dict(checkpoint_payload):
            break
        nested_payload = None
        for key in unwrap_keys:
            nested = checkpoint_payload.get(key)
            if isinstance(nested, dict):
                nested_payload = nested
                break
        if nested_payload is None:
            break
        checkpoint_payload = nested_payload

    if not isinstance(checkpoint_payload, dict):
        raise ValueError("Checkpoint does not contain a valid model state_dict.")
    if not _looks_like_tensor_state_dict(checkpoint_payload):
        raise ValueError(
            "Checkpoint did not resolve to a tensor state_dict. "
            f"Top-level keys: {list(checkpoint_payload.keys())[:8]}"
        )

    if any(k.startswith("module.") for k in checkpoint_payload.keys()):
        checkpoint_payload = {k.replace("module.", "", 1): v for k, v in checkpoint_payload.items()}
    return checkpoint_payload


def set_inference_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Set inference seed to {seed}")


def load_ctrlworld_model_for_inference(
    args: wm_orca_args,
    ckpt_path: str,
    model_label: str,
) -> CtrlWorld:
    """Build CtrlWorld and load base + optional fine-tuned LoRA weights."""
    if ckpt_path is None:
        raise ValueError(f"{model_label} checkpoint path is required.")

    finetune_before_lora = bool(getattr(args, "finetune_from_checkpoint_before_lora", False))
    use_lora = bool(getattr(args, "use_unet_lora", False))
    enable_lora_after_checkpoint = finetune_before_lora and use_lora

    if enable_lora_after_checkpoint:
        args.use_unet_lora = False
    try:
        model = CtrlWorld(args)
    finally:
        if enable_lora_after_checkpoint:
            args.use_unet_lora = use_lora

    print(f"Loading {model_label} checkpoint from {ckpt_path}!")
    checkpoint = _load_torch_checkpoint(ckpt_path, model_label)
    state_dict = _extract_model_state_dict(checkpoint)

    if enable_lora_after_checkpoint and any(_is_lora_adapter_key(k) for k in state_dict):
        # Self-contained LoRA-finetuned checkpoint (e.g. the released Cascade-SR / Mono-SR
        # weights): it already holds the base weights plus the adapters, so wrap the UNet
        # with LoRA first and then load everything strictly.
        model.enable_unet_lora(
            rank=getattr(args, "unet_lora_rank", 8),
            alpha=getattr(args, "unet_lora_alpha", 8.0),
        )
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
        if enable_lora_after_checkpoint:
            model.enable_unet_lora(
                rank=getattr(args, "unet_lora_rank", 8),
                alpha=getattr(args, "unet_lora_alpha", 8.0),
            )

    load_finetune_lora_checkpoint(
        model,
        getattr(args, "finetune_lora_ckpt_path", None),
        restore_optimizer=False,
    )
    model.eval()
    return model


def _latent_control_passthrough_adapter(segmentation_videos, target_frames: int, target_size):
    """Use precomputed latent control tensor directly (no RGB adapter CNN)."""
    bsz, num_frames = segmentation_videos.shape[:2]
    if num_frames < target_frames:
        pad_frames = target_frames - num_frames
        pad = segmentation_videos[:, -1:].repeat(1, pad_frames, 1, 1, 1)
        segmentation_videos = torch.cat([segmentation_videos, pad], dim=1)
    elif num_frames > target_frames:
        segmentation_videos = segmentation_videos[:, :target_frames]

    x = segmentation_videos.flatten(0, 1)
    if x.shape[-2:] != target_size:
        x = F.adaptive_avg_pool2d(x, output_size=target_size)
    x = x.view(bsz, target_frames, x.shape[1], target_size[0], target_size[1])
    return x


def _resolve_controlnet_inference_inputs(
    acc_model: CtrlWorld,
    args: wm_orca_args,
    batch: Dict,
    gt_latents_dtype: torch.dtype,
    device: torch.device,
):
    if not args.use_controlnet_conditioning:
        return None, acc_model.segmentation_to_control

    use_latent_control = bool(getattr(args, "use_latent_segmentation_for_controlnet", False))
    use_vae_roundtrip = bool(getattr(args, "use_vae_roundtrip_for_controlnet", False))
    use_vis_seg_actions = bool(getattr(args, "use_vis_seg_actions_for_controlnet", False))
    enabled_control_sources = int(use_latent_control) + int(use_vae_roundtrip) + int(use_vis_seg_actions)
    if enabled_control_sources > 1:
        raise ValueError(
            "Ambiguous ControlNet source flags. Enable at most one of "
            "use_latent_segmentation_for_controlnet, "
            "use_vae_roundtrip_for_controlnet, "
            "use_vis_seg_actions_for_controlnet."
        )

    controlnet_segmentation_videos = batch.get("controlnet_segmentation_videos", None)
    if controlnet_segmentation_videos is not None:
        controlnet_segmentation_videos = controlnet_segmentation_videos.to(
            device=device, dtype=gt_latents_dtype, non_blocking=True
        )

    if use_latent_control:
        if "latent_segmentation_videos" not in batch:
            raise KeyError("use_latent_segmentation_for_controlnet requires batch['latent_segmentation_videos'].")
        control_latents = batch["latent_segmentation_videos"].to(device=device, dtype=gt_latents_dtype, non_blocking=True)
        control_latents = latent_seg_heightstack_to_channelstack(control_latents, num_views=args.num_views)
        if control_latents.shape[2] != acc_model.unet.config.in_channels:
            raise ValueError(
                "Latent ControlNet conditioning channels mismatch during inference: "
                f"got {control_latents.shape[2]}, expected {acc_model.unet.config.in_channels}."
            )
        return control_latents, _latent_control_passthrough_adapter

    if use_vae_roundtrip:
        if controlnet_segmentation_videos is not None:
            return controlnet_segmentation_videos, acc_model.segmentation_to_control
        if "latent_segmentation_videos" not in batch:
            raise KeyError("use_vae_roundtrip_for_controlnet requires batch['latent_segmentation_videos'].")
        latent_segs = batch["latent_segmentation_videos"].to(device=device, dtype=gt_latents_dtype, non_blocking=True)
        segmentation_videos = acc_model._vae_decode_latent_segs(latent_segs)
        return segmentation_videos, acc_model.segmentation_to_control

    if use_vis_seg_actions:
        if "vis_seg_actions" not in batch:
            raise KeyError("use_vis_seg_actions_for_controlnet requires batch['vis_seg_actions'].")
        segmentation_videos = batch["vis_seg_actions"].to(device=device, dtype=gt_latents_dtype, non_blocking=True)
        return segmentation_videos, acc_model.segmentation_to_control

    if controlnet_segmentation_videos is not None:
        return controlnet_segmentation_videos, acc_model.segmentation_to_control
    if "segmentation_videos" not in batch:
        raise KeyError("ControlNet conditioning requires batch['segmentation_videos'].")
    segmentation_videos = batch["segmentation_videos"].to(device=device, dtype=gt_latents_dtype, non_blocking=True)
    return segmentation_videos, acc_model.segmentation_to_control


def _build_action_latent(
    model: CtrlWorld,
    args: wm_orca_args,
    batch: Dict,
    text,
    device: torch.device,
):
    if "dino_visual" in args.action_encoder:
        visual_actions = batch["segmentation_videos"].to(device, non_blocking=True)
        assert visual_actions.shape[1:] == (
            int(args.num_frames + args.num_history),
            args.num_views * 3,
            256,
            256,
        )
        action_latent = model.action_encoder(visual_actions)
    elif args.action_encoder in ("action_encoder", "action_encoder_dual"):
        actions = batch["action"].to(device, non_blocking=True)
        assert actions.shape[1:] == (int(args.num_frames + args.num_history), args.action_dim)
        action_latent = model.action_encoder(
            actions,
            text,
            model.tokenizer,
            model.text_encoder,
            args.frame_level_cond,
        )
    else:
        action_latent = torch.zeros((*batch["action"].shape[:2], 1024), device=device)
    return action_latent


def infer_pred_latents_raw(
    model: CtrlWorld,
    args: wm_orca_args,
    accelerator: Accelerator,
    batch: Dict,
    history_latent: Optional[torch.Tensor] = None,
    current_latent: Optional[torch.Tensor] = None,
    disable_action_conditioning: bool = False,
) -> torch.Tensor:
    """
    Returns predicted latents in shape [B, num_frames, 4, H, W]
    (before the per-view rearrange used for metric computation).
    """
    device = accelerator.device
    acc_model = accelerator.unwrap_model(model)
    pipeline = acc_model.pipeline

    text = batch["text"]
    gt_latents = batch[args.predicted_datatype].to(device, non_blocking=True)

    concatenate_latent = None
    if args.concatenate_latent:
        concatenate_latent = batch[args.concatenate_latent].to(device, non_blocking=True)
    segmentation_videos_for_controlnet, segmentation_to_control = _resolve_controlnet_inference_inputs(
        acc_model=acc_model,
        args=args,
        batch=batch,
        gt_latents_dtype=gt_latents.dtype,
        device=device,
    )

    if history_latent is None or current_latent is None:
        his_latent_gt = gt_latents[:, : args.num_history]
        future_latent_gt = gt_latents[:, args.num_history :]
        current_latent_gt = future_latent_gt[:, 0]
    else:
        his_latent_gt = history_latent
        current_latent_gt = current_latent

    # Validate latent layout without hardcoding square latent sizes.
    expected_channels = 4
    expected_latent_width = args.width // args.vae_compression_rate
    # Match dataset construction order: compress each view first, then stack views.
    # This avoids off-by-one mismatches for heights not divisible by compression rate
    # (e.g., 180 // 8 = 22, then for 2 views => 44).
    expected_stacked_latent_height = (
        args.num_views * (args.height // args.vae_compression_rate)
    )
    assert current_latent_gt.shape[1:] == (
        expected_channels,
        expected_stacked_latent_height,
        expected_latent_width,
    ), (
        "Unexpected latent shape. "
        f"Got {tuple(current_latent_gt.shape[1:])}, "
        f"expected ({expected_channels}, {expected_stacked_latent_height}, {expected_latent_width})."
    )
    assert his_latent_gt.shape[1] == args.num_history, (
        f"History length mismatch for model conditioning. Got {his_latent_gt.shape[1]}, "
        f"expected {args.num_history}."
    )

    action_latent = _build_action_latent(acc_model, args, batch, text, device)

    if disable_action_conditioning:
        action_latent = torch.zeros_like(action_latent)

    _, pred_latents_raw = CtrlWorldDiffusionPipeline.__call__(
        pipeline,
        image=current_latent_gt,
        text=action_latent,
        width=args.width,
        height=int(args.num_views * args.height),
        num_frames=args.num_frames,
        history=his_latent_gt,
        num_inference_steps=args.num_inference_steps,
        decode_chunk_size=args.decode_chunk_size,
        max_guidance_scale=args.guidance_scale,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        mask=None,
        output_type="latent",
        return_dict=False,
        frame_level_cond=args.frame_level_cond,
        his_cond_zero=args.his_cond_zero,
        concatenate_latent=concatenate_latent,
        num_channels_concatenate=args.num_channels_concatenate,
        use_controlnet_conditioning=args.use_controlnet_conditioning,
        controlnet=acc_model.controlnet,
        segmentation_videos=segmentation_videos_for_controlnet,
        segmentation_to_control=segmentation_to_control,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )
    return pred_latents_raw


def decode_and_format_outputs_for_metrics(
    pred_latents_raw: torch.Tensor,
    gt_latents_raw: torch.Tensor,
    args: wm_orca_args,
    pipeline,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      pred_video, gt_video, pred_latents_metric, gt_latents_metric
    where latent tensors are shaped [B*num_views, T, 4, 32, 32]-style,
    matching your existing metric code path.
    """
    pred_latents = einops.rearrange(
        pred_latents_raw,
        "b f c (m h) (n w) -> (b m n) f c h w",
        m=args.num_views,
        n=1,
    )
    gt_latents = einops.rearrange(
        gt_latents_raw,
        "b f c (m h) (n w) -> (b m n) f c h w",
        m=args.num_views,
        n=1,
    )

    # decode gt
    decoded_video = []
    gt_bsz, gt_time_steps = gt_latents.shape[:2]
    gt_latents_flat = gt_latents.flatten(0, 1)
    for i in range(0, gt_latents_flat.shape[0], args.decode_chunk_size):
        chunk = gt_latents_flat[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
        decoded_video.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
    gt_video = torch.cat(decoded_video, dim=0).reshape(gt_bsz, gt_time_steps, -1, gt_latents.shape[-2] * 8, gt_latents.shape[-1] * 8)
    gt_latents = gt_latents_flat.reshape(gt_bsz, gt_time_steps, *gt_latents.shape[2:])

    # decode pred
    decoded_video = []
    pred_bsz, pred_time_steps = pred_latents.shape[:2]
    pred_latents_flat = pred_latents.flatten(0, 1)
    for i in range(0, pred_latents_flat.shape[0], args.decode_chunk_size):
        chunk = pred_latents_flat[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
        decoded_video.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
    pred_video = torch.cat(decoded_video, dim=0).reshape(pred_bsz, pred_time_steps, -1, pred_latents.shape[-2] * 8, pred_latents.shape[-1] * 8)
    pred_latents = pred_latents_flat.reshape(pred_bsz, pred_time_steps, *pred_latents.shape[2:])

    return pred_video, gt_video, pred_latents, gt_latents


def decode_seg_latents_to_spatial_seg_video(
    seg_latents_btchw: torch.Tensor,
    pipeline,
    decode_chunk_size: int,
) -> torch.Tensor:
    """
    Input : [B, T, 4, H_lat, W_lat]
    Output: [B, T, 3, H_px, W_px] where H_px typically = num_views * 256, W_px = 256
    """
    bsz, t, c, h, w = seg_latents_btchw.shape
    flat = seg_latents_btchw.reshape(bsz * t, c, h, w) / pipeline.vae.config.scaling_factor

    decoded = []
    for i in range(0, flat.shape[0], decode_chunk_size):
        chunk = flat[i : i + decode_chunk_size]
        decoded.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)

    out = torch.cat(decoded, dim=0).reshape(bsz, t, -1, decoded[0].shape[-2], decoded[0].shape[-1])
    return out


def encode_spatial_video_to_latents(
    video_btchw: torch.Tensor,
    pipeline,
    encode_chunk_size: int,
) -> torch.Tensor:
    """
    Input : [B, T, 3, H_px, W_px] in [-1, 1]
    Output: [B, T, 4, H_lat, W_lat]
    """
    bsz, t, c, h, w = video_btchw.shape
    flat = video_btchw.reshape(bsz * t, c, h, w).to(dtype=pipeline.vae.dtype)

    latents = []
    for i in range(0, flat.shape[0], encode_chunk_size):
        chunk = flat[i : i + encode_chunk_size]
        latent_chunk = pipeline.vae.encode(chunk).latent_dist.sample().mul_(pipeline.vae.config.scaling_factor)
        latents.append(latent_chunk)

    return torch.cat(latents, dim=0).reshape(bsz, t, 4, latents[0].shape[-2], latents[0].shape[-1])


def apply_snap_to_palette_on_seg_latents(
    seg_latents_btchw: torch.Tensor,
    pipeline,
    decode_chunk_size: int,
    enable_snap_to_palette: bool,
) -> torch.Tensor:
    """
    Converts segmentation latents -> decoded segmentation video, snaps to semantic palette,
    and encodes back to latents for autoregressive feedback.
    """
    if not enable_snap_to_palette:
        return seg_latents_btchw

    seg_spatial = decode_seg_latents_to_spatial_seg_video(
        seg_latents_btchw=seg_latents_btchw,
        pipeline=pipeline,
        decode_chunk_size=decode_chunk_size,
    )  # [B, T, 3, H_px, W_px], in [-1, 1]

    seg_palette_rgb = torch.tensor(
        list(SEMANTIC_SEGMENTATION_MAPPING.values()),
        dtype=torch.float32,
        device=seg_spatial.device,
    )
    seg_spatial_flat = seg_spatial.flatten(0, 1)
    seg_spatial_rgb = ((seg_spatial_flat / 2.0 + 0.5).clamp(0, 1) * 255.0).to(torch.float32)
    seg_spatial_rgb_snapped = snap_to_palette(seg_spatial_rgb, palette=seg_palette_rgb)
    seg_spatial_snapped = (
        (seg_spatial_rgb_snapped.to(torch.float32) / 127.5 - 1.0)
        .to(seg_spatial.dtype)
        .reshape_as(seg_spatial)
    )

    return encode_spatial_video_to_latents(
        video_btchw=seg_spatial_snapped,
        pipeline=pipeline,
        encode_chunk_size=decode_chunk_size,
    )


class CircularBuffer:
    """Fixed-capacity FIFO buffer along the temporal dimension of a latent tensor.

    When ``stride > 1`` the buffer retains ``capacity * stride`` raw frames
    internally, but ``data`` returns every ``stride``-th frame (always
    ``capacity`` frames total).  This lets you expose a wider temporal window
    to the model without changing ``num_history``.

    Example — capacity=5, stride=2:
      internal size = 10  (last 10 pushed frames are kept)
      data          = frames at internal positions 1, 3, 5, 7, 9
                      → 5 frames spaced 2 steps apart
    """

    def __init__(self, initial: torch.Tensor, capacity: int, stride: int = 1):
        """
        initial  : [B, T_init, *] — seeds the buffer.
        capacity : number of frames exposed via ``data``.
        stride   : spacing between exposed frames; internal size = capacity * stride.
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}.")
        self.capacity = capacity
        self.stride = stride
        self._icap = capacity * stride  # internal capacity

        # Left-pad with first frame if initial has fewer frames than internal capacity.
        if initial.shape[1] < self._icap:
            pad_len = self._icap - initial.shape[1]
            pad = initial[:, :1].expand(-1, pad_len, *initial.shape[2:]).clone()
            initial = torch.cat([pad, initial], dim=1)
        self._buf = initial[:, -self._icap :]

    @property
    def data(self) -> torch.Tensor:
        """Strided history: [B, capacity, *] with spacing ``stride``."""
        return self._buf[:, self.stride - 1 :: self.stride]

    @property
    def current(self) -> torch.Tensor:
        """Most recent frame regardless of stride: [B, *]."""
        return self._buf[:, -1]

    def push(self, frames: torch.Tensor) -> None:
        """Append *frames* [B, T_new, *] and drop oldest to stay within internal capacity."""
        self._buf = torch.cat([self._buf, frames], dim=1)[:, -self._icap :]


def match_temporal_length(latents_btchw: torch.Tensor, target_t: int) -> torch.Tensor:
    """
    Match temporal length by truncating or repeating the last frame.
    """
    current_t = latents_btchw.shape[1]
    if current_t == target_t:
        return latents_btchw
    if current_t > target_t:
        return latents_btchw[:, :target_t]
    pad_count = target_t - current_t
    pad = latents_btchw[:, -1:].repeat(1, pad_count, 1, 1, 1)
    return torch.cat([latents_btchw, pad], dim=1)


def use_model_action(batch: Dict, model_key: str) -> Dict:
    action_key = f"action_{model_key}"
    if action_key not in batch:
        return batch
    out = dict(batch)
    out["action"] = batch[action_key]
    return out


def build_windowed_batch(
    batch: Dict,
    num_history: int,
    num_frames: int,
    future_offset: int = 0,
) -> Dict:
    """
    Build a temporal slice window with length num_history + num_frames.
    `future_offset` moves the prediction start forward in time.
    """
    window_start = int(future_offset)
    window_len = int(num_history + num_frames)
    window_end = window_start + window_len

    temporal_keys = (
        "action",
        "action_wm1",
        "action_wm2",
        "action_baseline_wm",
        "latent_videos",
        "latent_segmentation_videos",
        "segmentation_videos",
        "vis_seg_actions",
        "controlnet_segmentation_videos",
        "interpolated_downsampled_segmentation",
        "hand_mask",
    )
    out = dict(batch)
    for key in temporal_keys:
        value = batch.get(key, None)
        if value is None or not torch.is_tensor(value):
            continue
        if value.shape[1] < window_end:
            raise ValueError(
                f"Insufficient temporal length in batch['{key}']: "
                f"got T={value.shape[1]}, need at least {window_end}."
            )
        out[key] = value[:, window_start:window_end]
    return out


def build_wm2_cascade_batch(
    batch: Dict,
    seg_history_latents: torch.Tensor,   # [B, num_history, 4, H, W]
    wm1_pred_seg_latents: torch.Tensor,  # [B, wm1_num_frames, 4, H, W]
    wm2_args: wm_orca_args,
    wm1_pipeline,
) -> Dict:
    """
    Builds a new batch for WM2 where segmentation conditioning comes from
    GT history + WM1 future prediction.
    """
    batch2 = dict(batch)

    # Build full segmentation latent sequence: autoregressive history + WM1 future prediction.
    wm1_pred_seg_latents = match_temporal_length(wm1_pred_seg_latents, wm2_args.num_frames)
    seg_full = torch.cat([seg_history_latents, wm1_pred_seg_latents], dim=1)  # [B, num_history+num_frames, 4, H, W]

    # For latent concatenation path in WM2.
    if wm2_args.concatenate_latent == "latent_segmentation_videos":
        batch2["latent_segmentation_videos"] = seg_full

    # If WM2 needs decoded segmentation frames, decode once and derive required format.
    needs_decoded_seg = (
        "dino_visual" in wm2_args.action_encoder
        or wm2_args.concatenate_latent == "interpolated_downsampled_segmentation"
        or wm2_args.use_controlnet_conditioning
    )
    if needs_decoded_seg:
        seg_spatial = decode_seg_latents_to_spatial_seg_video(
            seg_full, wm1_pipeline, wm2_args.decode_chunk_size
        )  # [B, T, 3, H_px, W_px]

        # Build visual action encoder input: [B, T, num_views*3, 256, 256]
        if "dino_visual" in wm2_args.action_encoder:
            if wm2_args.num_views == 1:
                seg_action = seg_spatial
            else:
                # spatial stacked views -> channel stacked views
                seg_action = einops.rearrange(
                    seg_spatial,
                    "b t c (m h) w -> b t (m c) h w",
                    m=wm2_args.num_views,
                )
            # Make sure action encoder sees exactly 256x256 per view.
            if seg_action.shape[-2:] != (256, 256):
                bt, cc = seg_action.shape[0] * seg_action.shape[1], seg_action.shape[2]
                seg_action = F.interpolate(
                    seg_action.reshape(bt, cc, seg_action.shape[-2], seg_action.shape[-1]),
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(seg_action.shape[0], seg_action.shape[1], cc, 256, 256)
            batch2["segmentation_videos"] = seg_action

        # ControlNet expects segmentation in pixel space with views stacked spatially:
        # [B, T, 3, num_views*256, 256].
        if wm2_args.use_controlnet_conditioning:
            batch2["controlnet_segmentation_videos"] = seg_spatial
            if getattr(wm2_args, "use_vis_seg_actions_for_controlnet", False):
                batch2["vis_seg_actions"] = seg_spatial

        # Build interpolated-downsampled latent conditioning path.
        if wm2_args.concatenate_latent == "interpolated_downsampled_segmentation":
            compressed = wm2_args.width // wm2_args.vae_compression_rate
            target_h = wm2_args.num_views * compressed
            target_w = compressed
            bt = seg_spatial.shape[0] * seg_spatial.shape[1]
            seg_flat = seg_spatial.reshape(bt, 3, seg_spatial.shape[-2], seg_spatial.shape[-1])

            mode = wm2_args.downsample_method_for_segmentation
            if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
                seg_down = F.interpolate(
                    seg_flat,
                    size=(target_h, target_w),
                    mode=mode,
                    align_corners=False,
                )
            else:
                seg_down = F.interpolate(
                    seg_flat,
                    size=(target_h, target_w),
                    mode=mode,
                )
            batch2["interpolated_downsampled_segmentation"] = seg_down.reshape(
                seg_spatial.shape[0], seg_spatial.shape[1], 3, target_h, target_w
            )

    return batch2


METRIC_KEYS = ("psnr", "ssim", "lpips", "mse_img", "mse_lat", "mae_img", "mae_lat")


def _group_flattened_views(tensor: torch.Tensor, num_views: int, name: str) -> torch.Tensor:
    if tensor.shape[0] % num_views != 0:
        raise ValueError(
            f"{name} has first dimension {tensor.shape[0]}, which is not divisible by "
            f"num_views={num_views}. Expected tensors shaped [B*num_views, T, ...]."
        )
    batch_size = tensor.shape[0] // num_views
    return tensor.reshape(batch_size, num_views, *tensor.shape[1:])


def _compute_psnr_series(pred_frames: torch.Tensor, gt_frames: torch.Tensor) -> np.ndarray:
    mse = torch.mean((pred_frames - gt_frames) ** 2, dim=(3, 4, 5))
    psnr_values = 10.0 * torch.log10(torch.full_like(mse, 4.0) / mse.clamp_min(1e-12))
    psnr_values = torch.where(mse == 0, torch.full_like(psnr_values, float("inf")), psnr_values)
    return psnr_values.cpu().numpy()


def _compute_ssim_series(pred_frames: torch.Tensor, gt_frames: torch.Tensor) -> np.ndarray:
    from skimage.metrics import structural_similarity as ssim

    pred_np = pred_frames.cpu().numpy()
    gt_np = gt_frames.cpu().numpy()
    batch_size, num_views, num_frames = pred_np.shape[:3]
    height, width = pred_np.shape[-2], pred_np.shape[-1]
    ssim_values = np.full((batch_size, num_views, num_frames), np.nan, dtype=np.float32)
    if min(height, width) < 3:
        return ssim_values

    win_size = min(7, height - 1 if height % 2 == 0 else height, width - 1 if width % 2 == 0 else width)
    win_size = max(3, win_size)
    if win_size % 2 == 0:
        win_size -= 1

    for sample_idx in range(batch_size):
        for view_idx in range(num_views):
            for frame_idx in range(num_frames):
                ssim_values[sample_idx, view_idx, frame_idx] = ssim(
                    gt_np[sample_idx, view_idx, frame_idx],
                    pred_np[sample_idx, view_idx, frame_idx],
                    channel_axis=0,
                    data_range=2,
                    win_size=win_size,
                )
    return ssim_values


def _compute_lpips_series(pred_frames: torch.Tensor, gt_frames: torch.Tensor) -> np.ndarray:
    batch_size, num_views, num_frames, _, height, width = pred_frames.shape
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        # Frames are in [-1, 1]; pass directly without intermediate round-trip to [0,1].
        pred_torch = pred_frames.reshape(
            batch_size * num_views * num_frames, 3, height, width
        ).float().to(device)
        gt_torch = gt_frames.reshape(
            batch_size * num_views * num_frames, 3, height, width
        ).float().to(device)
        scores: list = []
        # Process one frame pair at a time to avoid OOM on large batches.
        for j in range(pred_torch.shape[0]):
            with torch.no_grad():
                scores.append(float(lpips_fn(pred_torch[j : j + 1], gt_torch[j : j + 1]).item()))
        return np.array(scores, dtype=np.float32).reshape(batch_size, num_views, num_frames)
    except Exception as exc:
        import traceback
        print(f"LPIPS computation failed: {exc}\n{traceback.format_exc()}")
        return np.full((batch_size, num_views, num_frames), float("nan"), dtype=np.float32)


def compute_video_metrics_per_view(
    pred_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    pred_latents: torch.Tensor,
    gt_latents: torch.Tensor,
    num_views: int,
    fps: int = 10,
    time_window_seconds: float = 1.0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return metric tensors with explicit [sample, view, time] semantics."""
    frames_in_window = int(fps * time_window_seconds)
    frames_to_evaluate = min(
        frames_in_window,
        pred_frames.shape[1],
        gt_frames.shape[1],
        pred_latents.shape[1],
        gt_latents.shape[1],
    )
    if frames_to_evaluate <= 0:
        raise ValueError("No frames available for metric computation.")

    pred_frames = _group_flattened_views(pred_frames[:, :frames_to_evaluate], num_views, "pred_frames")
    gt_frames = _group_flattened_views(gt_frames[:, :frames_to_evaluate], num_views, "gt_frames")
    pred_latents = _group_flattened_views(pred_latents[:, :frames_to_evaluate], num_views, "pred_latents")
    gt_latents = _group_flattened_views(gt_latents[:, :frames_to_evaluate], num_views, "gt_latents")

    print(
        f"Evaluating metrics per view over {frames_to_evaluate} frames "
        f"({frames_to_evaluate/fps:.2f}s at {fps} fps)"
    )

    pred_frames = pred_frames.detach().cpu().to(torch.float32)
    gt_frames = gt_frames.detach().cpu().to(torch.float32)
    pred_latents = pred_latents.detach().cpu().to(torch.float32)
    gt_latents = gt_latents.detach().cpu().to(torch.float32)

    per_frame = {
        "psnr": _compute_psnr_series(pred_frames, gt_frames),
        "ssim": _compute_ssim_series(pred_frames, gt_frames),
        "lpips": _compute_lpips_series(pred_frames, gt_frames),
        "mse_img": torch.mean((pred_frames - gt_frames) ** 2, dim=(3, 4, 5)).cpu().numpy(),
        "mse_lat": torch.mean((pred_latents - gt_latents) ** 2, dim=(3, 4, 5)).cpu().numpy(),
        "mae_img": torch.mean(torch.abs(pred_frames - gt_frames), dim=(3, 4, 5)).cpu().numpy(),
        "mae_lat": torch.mean(torch.abs(pred_latents - gt_latents), dim=(3, 4, 5)).cpu().numpy(),
    }
    per_sample_view = {key: np.nanmean(values, axis=2) for key, values in per_frame.items()}
    per_sample = {key: np.nanmean(values, axis=1) for key, values in per_sample_view.items()}
    per_view = {key: np.nanmean(values, axis=0) for key, values in per_sample_view.items()}

    return {
        "per_frame": per_frame,
        "per_sample_view": per_sample_view,
        "per_sample": per_sample,
        "per_view": per_view,
    }


def _make_metric_store() -> Dict:
    return {
        **{metric_key: [] for metric_key in METRIC_KEYS},
        "per_sample_view": {metric_key: [] for metric_key in METRIC_KEYS},
        "time_rows": [],
        "ar_step_rows": [],       # [sample_id, view_idx, ar_step_idx, *METRIC_KEYS] averaged over step frames
        "num_views": None,
        "videos": [],        # wandb.Video objects for per-sample RGB
        "latent_videos": [], # wandb.Video objects for per-sample latent
    }


def _collect_batch_metrics(
    pred_video: torch.Tensor,
    gt_video: torch.Tensor,
    pred_lat: torch.Tensor,
    gt_lat: torch.Tensor,
    model_args,
    metrics_store: Dict,
    batch_idx: int,
    accelerator,
    prefix: str,
    step_horizon: Optional[int] = None,
) -> None:
    """Compute per-sample metrics for one batch, save videos, accumulate, and upload to wandb."""
    if accelerator.num_processes > 1:
        pred_video = accelerator.gather_for_metrics(pred_video)
        gt_video = accelerator.gather_for_metrics(gt_video)
        pred_lat = accelerator.gather_for_metrics(pred_lat)
        gt_lat = accelerator.gather_for_metrics(gt_lat)

    if not accelerator.is_main_process:
        return

    gt_future_video = gt_video[:, model_args.num_history:]
    gt_future_lat = gt_lat[:, model_args.num_history:]
    # In AR mode cover all predicted frames; in oneshot mode keep the 1-second window.
    _time_window = (
        float(pred_video.shape[1]) / model_args.fps
        if step_horizon is not None
        else 1.0
    )
    batch_metrics = compute_video_metrics_per_view(
        pred_video,
        gt_future_video,
        pred_lat,
        gt_future_lat,
        num_views=int(model_args.num_views),
        fps=model_args.fps,
        time_window_seconds=_time_window,
    )

    # Build combined RGB video array: GT (future) on top, pred below, samples side by side.
    # pred_video/gt_video shape: [B*views, T, 3, H, W]
    gt_np = (
        (gt_video.detach().cpu().float() / 2.0 + 0.5).clamp(0, 1) * 255
    ).permute(0, 1, 3, 4, 2).to(torch.uint8).numpy()
    pred_np = (
        (pred_video.detach().cpu().float() / 2.0 + 0.5).clamp(0, 1) * 255
    ).permute(0, 1, 3, 4, 2).to(torch.uint8).numpy()
    # [B*views, T, 2H, W, 3] -> [T, 2H, W*B*views, 3]
    combined = np.concatenate([gt_np[:, model_args.num_history:], pred_np], axis=-3)
    combined_horiz = np.concatenate(list(combined), axis=-2).astype(np.uint8)

    # Build combined latent array.
    # pred_lat/gt_lat shape: [B*views, T, 4, H_lat, W_lat]
    pred_lat_np = ((pred_lat.detach().cpu().float() / 2.0 + 0.5).clamp(0, 1) * 255).numpy()
    gt_lat_np = ((gt_lat.detach().cpu().float() / 2.0 + 0.5).clamp(0, 1) * 255).numpy()
    # [B*views, T, 4, 2*H_lat, W_lat] -> [T, 4, 2*H_lat, W_lat*B*views]
    combined_lat = np.concatenate([gt_lat_np[:, model_args.num_history:], pred_lat_np], axis=-2)
    combined_lat_horiz = np.concatenate(list(combined_lat), axis=-1).astype(np.uint8)

    num_views = int(model_args.num_views)
    if metrics_store["num_views"] is None:
        metrics_store["num_views"] = num_views
    elif metrics_store["num_views"] != num_views:
        raise ValueError(
            f"Metric store already contains num_views={metrics_store['num_views']}, "
            f"but current batch has num_views={num_views}."
        )

    width_per_sample = num_views * model_args.width
    width_per_sample_lat = num_views * model_args.width // model_args.vae_compression_rate
    n = int(batch_metrics["per_sample"]["psnr"].shape[0])

    for i in range(n):
        for metric_key in METRIC_KEYS:
            metrics_store[metric_key].append(float(batch_metrics["per_sample"][metric_key][i]))
            metrics_store["per_sample_view"][metric_key].append(
                [float(v) for v in batch_metrics["per_sample_view"][metric_key][i]]
            )

        for view_idx in range(num_views):
            for frame_idx in range(batch_metrics["per_frame"]["psnr"].shape[2]):
                metrics_store["time_rows"].append([
                    len(metrics_store["psnr"]) - 1,
                    view_idx,
                    frame_idx,
                    (frame_idx + 1) / float(model_args.fps),
                    *[float(batch_metrics["per_frame"][metric_key][i, view_idx, frame_idx]) for metric_key in METRIC_KEYS],
                ])

        # Per-AR-step rows: average metrics over the step_horizon frames in each step.
        if step_horizon is not None and "ar_step_rows" in metrics_store:
            _n_eval = batch_metrics["per_frame"]["psnr"].shape[2]
            _ar_steps = _n_eval // step_horizon
            _global_sid = len(metrics_store["psnr"]) - 1
            for _vi in range(num_views):
                for _si in range(_ar_steps):
                    _f0, _f1 = _si * step_horizon, (_si + 1) * step_horizon
                    metrics_store["ar_step_rows"].append([
                        _global_sid,
                        _vi,
                        _si,
                        *[float(np.nanmean(
                            batch_metrics["per_frame"][metric_key][i, _vi, _f0:_f1]
                        )) for metric_key in METRIC_KEYS],
                    ])

        # combined_horiz is [T, H, W, 3] (THWC); wandb.Video requires [T, C, H, W] (TCHW).
        video_clip = combined_horiz[:, :, i * width_per_sample:(i + 1) * width_per_sample]
        metrics_store["videos"].append(
            wandb.Video(video_clip.transpose(0, 3, 1, 2), fps=model_args.wandb_video_display_fps, format="mp4")
        )

        # combined_lat_horiz is [T, 4, H, W] (TCHW); take first 3 channels directly.
        lat_clip = combined_lat_horiz[:, :3, :, i * width_per_sample_lat:(i + 1) * width_per_sample_lat]
        metrics_store["latent_videos"].append(
            wandb.Video(lat_clip, fps=model_args.wandb_video_display_fps, format="mp4")
        )

    # Upload per-sample metrics + videos for this batch immediately.
    global_offset = len(metrics_store["psnr"]) - n
    view_metric_columns = [
        f"{metric_key}_v{view_idx:02d}"
        for metric_key in METRIC_KEYS
        for view_idx in range(num_views)
    ]
    batch_table = wandb.Table(columns=[
        "sample_id", *METRIC_KEYS, *view_metric_columns, "video", "latent_video"
    ])
    for i in range(n):
        g = global_offset + i
        view_metric_values: List[float] = []
        for metric_key in METRIC_KEYS:
            view_metric_values.extend(metrics_store["per_sample_view"][metric_key][g])
        batch_table.add_data(
            g,
            *[metrics_store[metric_key][g] for metric_key in METRIC_KEYS],
            *view_metric_values,
            metrics_store["videos"][g],
            metrics_store["latent_videos"][g],
        )
    accelerator.log({f"{prefix}/batch_{batch_idx}_samples": batch_table}, step=batch_idx)


def _build_metric_plots(metrics_store: Dict, prefix: str) -> Dict:
    """One line-chart per metric, x=sample_id, y=metric value."""
    plots = {}
    for metric_key, display_name in [
        ("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS"),
        ("mse_img", "MSE (image)"), ("mae_img", "MAE (image)"),
    ]:
        values = metrics_store[metric_key]
        table = wandb.Table(
            data=[[i, v] for i, v in enumerate(values)],
            columns=["Sample ID", display_name],
        )
        plots[f"{prefix}/{metric_key}_per_sample"] = wandb.plot.line(
            table, "Sample ID", display_name, title=f"{prefix} — {display_name} per sample"
        )
    return plots


def _view_metric_columns(num_views: int) -> List[str]:
    return [
        f"{metric_key}_v{view_idx:02d}"
        for metric_key in METRIC_KEYS
        for view_idx in range(num_views)
    ]


def _per_view_metric_values(metrics_store: Dict, sample_idx: int, num_views: int) -> List[float]:
    values: List[float] = []
    for metric_key in METRIC_KEYS:
        metric_views = metrics_store["per_sample_view"][metric_key][sample_idx]
        if len(metric_views) != num_views:
            raise ValueError(
                f"Sample {sample_idx} metric {metric_key} has {len(metric_views)} views; "
                f"expected {num_views}."
            )
        values.extend(metric_views)
    return values


def _build_per_view_avg_table(metrics_store: Dict) -> "wandb.Table":
    num_views = int(metrics_store["num_views"] or 0)
    table = wandb.Table(columns=["view_idx", *METRIC_KEYS])
    if num_views == 0:
        return table
    for view_idx in range(num_views):
        table.add_data(
            view_idx,
            *[
                float(np.nanmean([row[view_idx] for row in metrics_store["per_sample_view"][metric_key]]))
                for metric_key in METRIC_KEYS
            ],
        )
    return table


def _build_per_sample_view_table(metrics_store: Dict) -> "wandb.Table":
    num_views = int(metrics_store["num_views"] or 0)
    table = wandb.Table(columns=["sample_id", "view_idx", *METRIC_KEYS])
    for sample_idx in range(len(metrics_store["psnr"])):
        for view_idx in range(num_views):
            table.add_data(
                sample_idx,
                view_idx,
                *[metrics_store["per_sample_view"][metric_key][sample_idx][view_idx] for metric_key in METRIC_KEYS],
            )
    return table


def _build_metric_time_table(metrics_store: Dict) -> "wandb.Table":
    return wandb.Table(
        columns=["sample_id", "view_idx", "frame_idx", "time_seconds", *METRIC_KEYS],
        data=metrics_store["time_rows"],
    )


def _build_ar_step_table(metrics_store: Dict) -> "wandb.Table":
    """Per-AR-step metrics averaged over the step_horizon frames, all samples/views."""
    return wandb.Table(
        columns=["sample_id", "view_idx", "ar_step_idx", *METRIC_KEYS],
        data=metrics_store.get("ar_step_rows", []),
    )


def _build_ar_step_plots(metrics_store: Dict, prefix: str) -> Dict:
    """Line charts of mean metric vs AR step index (averaged over samples and views)."""
    from collections import defaultdict
    rows = metrics_store.get("ar_step_rows", [])
    if not rows:
        return {}

    step_vals: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {k: [] for k in METRIC_KEYS})
    for row in rows:
        ar_step = int(row[2])
        for mi, metric_key in enumerate(METRIC_KEYS):
            val = float(row[3 + mi])
            if not np.isnan(val):
                step_vals[ar_step][metric_key].append(val)

    plots = {}
    for metric_key, display_name in [
        ("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS"),
        ("mse_img", "MSE (image)"), ("mae_img", "MAE (image)"),
    ]:
        table_data = [
            [step, float(np.mean(step_vals[step][metric_key]))]
            for step in sorted(step_vals)
            if step_vals[step].get(metric_key)
        ]
        if not table_data:
            continue
        tbl = wandb.Table(data=table_data, columns=["AR Step", display_name])
        plots[f"{prefix}/{metric_key}_over_ar_steps"] = wandb.plot.line(
            tbl, "AR Step", display_name,
            title=f"{prefix} — {display_name} over AR steps",
        )
    return plots


def _build_video_table(metrics_store: Dict, model_args) -> "wandb.Table":
    """Flat table of all samples sorted by PSNR ascending (worst -> best)."""
    n = len(metrics_store["psnr"])
    num_views = int(metrics_store["num_views"] or model_args.num_views)
    order = sorted(range(n), key=lambda i: metrics_store["psnr"][i])
    table = wandb.Table(columns=[
        "id", *METRIC_KEYS, *_view_metric_columns(num_views), "video", "latent_video"
    ])
    for i in order:
        table.add_data(
            i,
            *[metrics_store[metric_key][i] for metric_key in METRIC_KEYS],
            *_per_view_metric_values(metrics_store, i, num_views),
            metrics_store["videos"][i],
            metrics_store["latent_videos"][i],
        )
    return table


def main(wm1_args: wm_orca_args, wm2_args: wm_orca_args, baseline_wm_args: wm_orca_args, args_cli: argparse.Namespace):
    wm1_enabled = args_cli.wm1_model_ckpt_path is not None
    wm2_enabled = args_cli.wm2_model_ckpt_path is not None
    baseline_wm_enabled = args_cli.baseline_wm_model_ckpt_path is not None

    if not any([wm1_enabled, wm2_enabled, baseline_wm_enabled]):
        raise ValueError("At least one of wm1, wm2, or baseline_wm checkpoint paths must be provided.")

    # Basic compatibility checks.
    if wm1_enabled:
        assert wm1_args.predicted_datatype == "latent_segmentation_videos", (
            "WM1 should predict segmentation latents for this cascade/autoregressive experiment."
        )
    if wm2_enabled:
        assert wm2_args.predicted_datatype == "latent_videos", (
            "WM2 should predict video latents for this cascade/autoregressive experiment."
        )
    if baseline_wm_enabled:
        assert baseline_wm_args.predicted_datatype == "latent_videos", (
            "baseline_wm should predict video latents."
        )
    if wm1_enabled and wm2_enabled:
        assert wm1_args.num_views == wm2_args.num_views, "WM1 and WM2 num_views must match."
        # Training-only settings can differ between separately trained WM1/WM2 checkpoints.
        # Inference uses the accelerator settings selected below from the first enabled output model.

    if args_cli.wm1_num_frames is not None:
        wm1_args.num_frames = int(args_cli.wm1_num_frames)
    if args_cli.wm2_num_frames is not None:
        wm2_args.num_frames = int(args_cli.wm2_num_frames)
    if args_cli.wm1_num_history is not None:
        wm1_args.num_history = int(args_cli.wm1_num_history)
    if args_cli.wm2_num_history is not None:
        wm2_args.num_history = int(args_cli.wm2_num_history)
    if args_cli.baseline_wm_num_frames is not None:
        baseline_wm_args.num_frames = int(args_cli.baseline_wm_num_frames)
    if args_cli.baseline_wm_num_history is not None:
        baseline_wm_args.num_history = int(args_cli.baseline_wm_num_history)
    if args_cli.wm1_num_inference_steps is not None:
        wm1_args.num_inference_steps = int(args_cli.wm1_num_inference_steps)
    if args_cli.wm2_num_inference_steps is not None:
        wm2_args.num_inference_steps = int(args_cli.wm2_num_inference_steps)
    if args_cli.baseline_wm_num_inference_steps is not None:
        baseline_wm_args.num_inference_steps = int(args_cli.baseline_wm_num_inference_steps)

    if args_cli.inference_mode == "autoregressive":
        if wm1_enabled and wm2_enabled:
            assert wm1_args.num_frames == wm2_args.num_frames, (
                "Autoregressive mode requires equal WM1 and WM2 horizons. "
                f"Got wm1_num_frames={wm1_args.num_frames}, wm2_num_frames={wm2_args.num_frames}."
            )
        ar_ref_args = wm1_args if wm1_enabled else (wm2_args if wm2_enabled else baseline_wm_args)
        step_horizon = int(ar_ref_args.num_frames)
        if baseline_wm_enabled and (wm1_enabled or wm2_enabled):
            assert baseline_wm_args.num_frames == step_horizon, (
                "Autoregressive mode requires equal num_frames across all enabled models. "
                f"Got baseline_wm_num_frames={baseline_wm_args.num_frames}, step_horizon={step_horizon}."
            )
        total_future_frames = int(args_cli.ar_num_steps) * step_horizon
    else:
        step_horizon = None
        candidate_frames = []
        if wm1_enabled:
            candidate_frames.append(wm1_args.num_frames)
        if wm2_enabled:
            candidate_frames.append(wm2_args.num_frames)
        if baseline_wm_enabled:
            candidate_frames.append(baseline_wm_args.num_frames)
        total_future_frames = int(max(candidate_frames))

    # Pick dataset settings from the first enabled model (priority: wm2 > wm1 > baseline_wm).
    if wm2_enabled:
        dataset_args = copy.deepcopy(wm2_args)
    elif wm1_enabled:
        dataset_args = copy.deepcopy(wm1_args)
    else:
        dataset_args = copy.deepcopy(baseline_wm_args)
    # Evaluate on the same validation samples for every model, whatever episodes its
    # training config excluded.
    dataset_args.exclude_episode_ids_by_dataset = {}
    dataset_args.num_frames = total_future_frames
    # Ensure the dataset loads enough frames to seed the largest history buffer.
    all_histories = []
    if wm1_enabled:
        all_histories.append(wm1_args.num_history)
    if wm2_enabled:
        all_histories.append(wm2_args.num_history)
    if baseline_wm_enabled:
        all_histories.append(baseline_wm_args.num_history)
    dataset_args.num_history = max(all_histories)
    # Force stride=1 at inference so every frame is unique and the ground truth
    # is never a repeated/clipped frame.
    dataset_args.min_stride = 1
    dataset_args.max_stride = 1

    # The dataloader is shared across enabled models, but action joint-order
    # preprocessing can be model-specific. Keep each model's training-time
    # swap_abd_with_mcp setting for its own normalized action tensor.
    if wm1_enabled:
        dataset_args.wm1_swap_abd_with_mcp = bool(getattr(wm1_args, "swap_abd_with_mcp", False))
    if wm2_enabled:
        dataset_args.wm2_swap_abd_with_mcp = bool(getattr(wm2_args, "swap_abd_with_mcp", False))
    if baseline_wm_enabled:
        dataset_args.baseline_wm_swap_abd_with_mcp = bool(
            getattr(baseline_wm_args, "swap_abd_with_mcp", False)
        )

    # Pick accelerator settings from the first enabled model.
    accel_args = wm2_args if wm2_enabled else (wm1_args if wm1_enabled else baseline_wm_args)
    accelerator = Accelerator(
        gradient_accumulation_steps=accel_args.gradient_accumulation_steps,
        mixed_precision=accel_args.mixed_precision,
        log_with="wandb",
        project_dir=args_cli.output_path,
    )

    if accelerator.is_main_process:
        log_args = wm2_args if wm2_enabled else (wm1_args if wm1_enabled else baseline_wm_args)
        wandb_project_name = getattr(log_args, "wandb_project_name", "ctrl-world")
        wandb_run_name = getattr(log_args, "wandb_run_name", "inference")
        merged_cfg = {
            "wm1_ckpt": args_cli.wm1_model_ckpt_path,
            "wm2_ckpt": args_cli.wm2_model_ckpt_path,
            "baseline_wm_ckpt": args_cli.baseline_wm_model_ckpt_path,
            **{f"wm1_{k}": v for k, v in args_to_dict(wm1_args).items()},
            **{f"wm2_{k}": v for k, v in args_to_dict(wm2_args).items()},
            **{f"baseline_wm_{k}": v for k, v in args_to_dict(baseline_wm_args).items()},
        }
        accelerator.init_trackers(
            wandb_project_name,
            config=merged_cfg,
            init_kwargs={"wandb": {"name": wandb_run_name}},
        )

    # load models
    wm1 = None
    if wm1_enabled:
        wm1 = load_ctrlworld_model_for_inference(
            wm1_args,
            args_cli.wm1_model_ckpt_path,
            "WM1",
        )
        wm1.unet = torch.compile(wm1.unet, mode="reduce-overhead")

    wm2 = None
    if wm2_enabled:
        wm2 = load_ctrlworld_model_for_inference(
            wm2_args,
            args_cli.wm2_model_ckpt_path,
            "WM2",
        )
        wm2.unet = torch.compile(wm2.unet, mode="reduce-overhead")
        if args_cli.wm2_disable_controlnet_conditioning:
            wm2_args.use_controlnet_conditioning = False

    baseline_wm = None
    if baseline_wm_enabled:
        baseline_wm = load_ctrlworld_model_for_inference(
            baseline_wm_args,
            args_cli.baseline_wm_model_ckpt_path,
            "baseline_wm",
        )
        baseline_wm.unet = torch.compile(baseline_wm.unet, mode="reduce-overhead")

    # notification
    active_models = ", ".join(filter(None, [
        "WM1" if wm1_enabled else None,
        "WM2" if wm2_enabled else None,
        "baseline_wm" if baseline_wm_enabled else None,
    ]))
    start_msg = (
        f"🚀 **Starting Inference** ({active_models})\n"
        f"📁 WM1 Config: `{args_cli.wm1_config_path}`\n"
        f"📁 WM2 Config: `{args_cli.wm2_config_path}`\n"
        f"📁 Baseline WM Config: `{args_cli.baseline_wm_config_path}`\n"
        f"🧪 Mode: `{args_cli.inference_mode}`\n"
        f"📏 Dataset future window: `{total_future_frames}`\n"
        f"⏰ Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    send_discord_message(start_msg)

    # dataset/loader
    # In autoregressive mode the AR-length filter runs on the full validation set so that
    # the requested number of samples is drawn from *all* qualifying episodes, not from a
    # pre-subsampled pool that might lose valid episodes before the filter runs.
    _requested_max_samples = int(dataset_args.max_num_samples_for_validation)
    if args_cli.inference_mode == "autoregressive":
        dataset_args.max_num_samples_for_validation = sys.maxsize
    test_dataset = Dataset_mix(dataset_args, mode=args_cli.mode)

    # Filter out samples that don't have enough frames for the full autoregressive
    # rollout. With stride=1 forced above, the last future frame lands at
    # frame_now + (total_future_frames - 1), so we need video_length > frame_now + total_future_frames - 1.
    if args_cli.inference_mode == "autoregressive":
        for ds_idx, (samples, dataset_paths) in enumerate(
            zip(test_dataset.samples_all, test_dataset.dataset_path_all)
        ):
            dataset_dir = dataset_paths[0]
            ann_cache: dict = {}

            def _get_video_length(episode_id: int) -> int:
                if episode_id not in ann_cache:
                    ann_path = os.path.join(dataset_dir, "annotation", f"{episode_id}.json")
                    with open(ann_path) as _f:
                        ann_cache[episode_id] = json.load(_f)["video_length"]
                return ann_cache[episode_id]

            kept = [
                s for s in samples
                if _get_video_length(s["episode_id"]) - s["frame_ids"][0] > total_future_frames - 1
            ]
            n_removed = len(samples) - len(kept)
            if n_removed:
                print(
                    f"[filter] Removed {n_removed}/{len(samples)} samples from dataset {ds_idx} "
                    f"with fewer than {total_future_frames} available frames."
                )
            # Re-apply the originally requested cap on the filtered pool.
            if len(kept) > _requested_max_samples:
                kept = kept[:_requested_max_samples]
                print(
                    f"[filter] Capped dataset {ds_idx} to {_requested_max_samples} samples "
                    f"after AR-length filter."
                )
            test_dataset.samples_all[ds_idx] = kept
            test_dataset.dataset_path_all[ds_idx] = dataset_paths[:len(kept)]
            test_dataset.samples_len[ds_idx] = len(kept)
        test_dataset.max_id = max(test_dataset.samples_len)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=accel_args.validation_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # prepare each enabled model + loader separately
    if wm1_enabled:
        wm1 = accelerator.prepare(wm1)
    if wm2_enabled:
        wm2 = accelerator.prepare(wm2)
    if baseline_wm_enabled:
        baseline_wm = accelerator.prepare(baseline_wm)
    test_loader = accelerator.prepare(test_loader)

    # Per-model per-sample metric accumulators (populated batch by batch).
    wm1_seg_metrics = _make_metric_store()
    wm2_baseline_metrics = _make_metric_store()
    wm2_cascade_metrics = _make_metric_store()
    wm2_ar_metrics = _make_metric_store()
    wm2_standalone_ar_metrics = _make_metric_store()
    baseline_wm_metrics = _make_metric_store()
    baseline_wm_ar_metrics = _make_metric_store()

    for batch_idx, batch in enumerate(test_loader):
        with torch.no_grad():
            with accelerator.autocast():
                if args_cli.inference_mode == "oneshot":
                    # --- WM1 (predicts seg latents) ---
                    if wm1_enabled:
                        wm1_batch = build_windowed_batch(
                            batch=batch,
                            num_history=wm1_args.num_history,
                            num_frames=wm1_args.num_frames,
                            future_offset=0,
                        )
                        wm1_pred_seg_raw = infer_pred_latents_raw(
                            model=wm1,
                            args=wm1_args,
                            accelerator=accelerator,
                            batch=use_model_action(wm1_batch, "wm1"),
                        )
                        gt_seg_raw = wm1_batch[wm1_args.predicted_datatype].to(
                            accelerator.device, non_blocking=True
                        )
                        b_pred_seg, b_gt_seg, b_pred_seg_lat, b_gt_seg_lat = decode_and_format_outputs_for_metrics(
                            wm1_pred_seg_raw,
                            gt_seg_raw,
                            wm1_args,
                            accelerator.unwrap_model(wm1).pipeline,
                        )
                        _collect_batch_metrics(
                            b_pred_seg, b_gt_seg, b_pred_seg_lat, b_gt_seg_lat,
                            wm1_args, wm1_seg_metrics, batch_idx, accelerator,

                            prefix="wm1_seg",
                        )

                    # --- WM2 baseline (GT-seg → video) ---
                    if wm2_enabled:
                        wm2_batch = build_windowed_batch(
                            batch=batch,
                            num_history=wm2_args.num_history,
                            num_frames=wm2_args.num_frames,
                            future_offset=0,
                        )
                        wm2_pred_baseline_raw = infer_pred_latents_raw(
                            model=wm2,
                            args=wm2_args,
                            accelerator=accelerator,
                            batch=use_model_action(wm2_batch, "wm2"),
                            disable_action_conditioning=args_cli.wm2_disable_action_conditioning,
                        )
                        gt_wm2_raw = wm2_batch[wm2_args.predicted_datatype].to(
                            accelerator.device, non_blocking=True
                        )
                        b_pred_video, b_gt_video, b_pred_lat, b_gt_lat = decode_and_format_outputs_for_metrics(
                            wm2_pred_baseline_raw,
                            gt_wm2_raw,
                            wm2_args,
                            accelerator.unwrap_model(wm2).pipeline,
                        )
                        _collect_batch_metrics(
                            b_pred_video, b_gt_video, b_pred_lat, b_gt_lat,
                            wm2_args, wm2_baseline_metrics, batch_idx, accelerator,
                            prefix="wm2_baseline",
                        )

                    # --- WM2 cascade (WM1-predicted seg → video, only when both enabled) ---
                    if wm1_enabled and wm2_enabled:
                        wm1_hist_for_wm2 = wm2_batch["latent_segmentation_videos"][:, : wm2_args.num_history]
                        wm1_pred_for_wm2 = apply_snap_to_palette_on_seg_latents(
                            seg_latents_btchw=wm1_pred_seg_raw,
                            pipeline=accelerator.unwrap_model(wm1).pipeline,
                            decode_chunk_size=wm2_args.decode_chunk_size,
                            enable_snap_to_palette=args_cli.snap_to_palette,
                        )
                        batch_cascade = build_wm2_cascade_batch(
                            batch=wm2_batch,
                            seg_history_latents=wm1_hist_for_wm2,
                            wm1_pred_seg_latents=wm1_pred_for_wm2,
                            wm2_args=wm2_args,
                            wm1_pipeline=accelerator.unwrap_model(wm1).pipeline,
                        )
                        wm2_pred_cascade_raw = infer_pred_latents_raw(
                            model=wm2,
                            args=wm2_args,
                            accelerator=accelerator,
                            batch=use_model_action(batch_cascade, "wm2"),
                            disable_action_conditioning=args_cli.wm2_disable_action_conditioning,
                        )
                        c_pred_video, c_gt_video, c_pred_lat, c_gt_lat = decode_and_format_outputs_for_metrics(
                            wm2_pred_cascade_raw,
                            gt_wm2_raw,
                            wm2_args,
                            accelerator.unwrap_model(wm2).pipeline,
                        )
                        _collect_batch_metrics(
                            c_pred_video, c_gt_video, c_pred_lat, c_gt_lat,
                            wm2_args, wm2_cascade_metrics, batch_idx, accelerator,
                            prefix="wm2_cascade",
                        )

                    # --- Baseline WM (standalone video prediction) ---
                    if baseline_wm_enabled:
                        baseline_wm_batch = build_windowed_batch(
                            batch=batch,
                            num_history=baseline_wm_args.num_history,
                            num_frames=baseline_wm_args.num_frames,
                            future_offset=0,
                        )
                        baseline_wm_pred_raw = infer_pred_latents_raw(
                            model=baseline_wm,
                            args=baseline_wm_args,
                            accelerator=accelerator,
                            batch=use_model_action(baseline_wm_batch, "baseline_wm"),
                            disable_action_conditioning=args_cli.wm2_disable_action_conditioning,
                        )
                        gt_baseline_wm_raw = baseline_wm_batch[baseline_wm_args.predicted_datatype].to(
                            accelerator.device, non_blocking=True
                        )
                        bwm_pred_video, bwm_gt_video, bwm_pred_lat, bwm_gt_lat = decode_and_format_outputs_for_metrics(
                            baseline_wm_pred_raw,
                            gt_baseline_wm_raw,
                            baseline_wm_args,
                            accelerator.unwrap_model(baseline_wm).pipeline,
                        )
                        _collect_batch_metrics(
                            bwm_pred_video, bwm_gt_video, bwm_pred_lat, bwm_gt_lat,
                            baseline_wm_args, baseline_wm_metrics, batch_idx, accelerator,
                            prefix="baseline_wm",
                        )

                else:
                    assert step_horizon is not None
                    wm1_pipeline = accelerator.unwrap_model(wm1).pipeline if wm1_enabled else None
                    wm2_pipeline = accelerator.unwrap_model(wm2).pipeline if wm2_enabled else None
                    baseline_wm_pipeline = accelerator.unwrap_model(baseline_wm).pipeline if baseline_wm_enabled else None

                    if baseline_wm_enabled:
                        baseline_wm_state_full = batch["latent_videos"].to(
                            accelerator.device, non_blocking=True
                        )
                        baseline_wm_hist_buf = CircularBuffer(
                            baseline_wm_state_full[:, : baseline_wm_args.num_history],
                            capacity=baseline_wm_args.num_history,
                            stride=args_cli.baseline_wm_history_stride,
                        )
                        baseline_wm_current = baseline_wm_state_full[:, baseline_wm_args.num_history]

                    if wm1_enabled:
                        wm1_state_full = batch["latent_segmentation_videos"].to(
                            accelerator.device, non_blocking=True
                        )
                        wm1_hist_buf = CircularBuffer(
                            wm1_state_full[:, : wm1_args.num_history],
                            capacity=wm1_args.num_history,
                            stride=args_cli.wm1_history_stride,
                        )
                        wm1_current = wm1_state_full[:, wm1_args.num_history]

                    if wm2_enabled:
                        wm2_state_full = batch["latent_videos"].to(
                            accelerator.device, non_blocking=True
                        )
                        wm2_hist_buf = CircularBuffer(
                            wm2_state_full[:, : wm2_args.num_history],
                            capacity=wm2_args.num_history,
                            stride=args_cli.wm2_history_stride,
                        )
                        wm2_current = wm2_state_full[:, wm2_args.num_history]
                        if wm1_enabled:
                            # Separate buffer for WM1 seg frames used as cascade conditioning
                            # for WM2 — capacity and stride match WM2's history settings.
                            wm2_seg_hist_buf = CircularBuffer(
                                wm1_state_full[:, : wm2_args.num_history],
                                capacity=wm2_args.num_history,
                                stride=args_cli.wm2_history_stride,
                            )

                    batch_wm1_pred_seg = []
                    batch_wm1_pred_seg_lat = []
                    batch_wm2_pred_video = []
                    batch_wm2_pred_lat = []
                    batch_wm1_gt_seg_future = []
                    batch_wm1_gt_seg_lat_future = []
                    batch_wm2_gt_video_future = []
                    batch_wm2_gt_lat_future = []
                    batch_wm2_standalone_pred_video = []
                    batch_wm2_standalone_pred_lat = []
                    batch_wm2_standalone_gt_video_future = []
                    batch_wm2_standalone_gt_lat_future = []
                    batch_baseline_wm_pred_video = []
                    batch_baseline_wm_pred_lat = []
                    batch_baseline_wm_gt_video_future = []
                    batch_baseline_wm_gt_lat_future = []
                    wm1_gt_seg_history = None
                    wm1_gt_seg_lat_history = None
                    wm2_gt_video_history = None
                    wm2_gt_lat_history = None
                    wm2_standalone_gt_video_history = None
                    wm2_standalone_gt_lat_history = None
                    baseline_wm_gt_video_history = None
                    baseline_wm_gt_lat_history = None

                    for step_idx in range(int(args_cli.ar_num_steps)):
                        future_offset = step_idx * step_horizon
                        wm1_step_batch = None
                        if wm1_enabled:
                            wm1_step_batch = build_windowed_batch(
                                batch=batch,
                                num_history=wm1_args.num_history,
                                num_frames=wm1_args.num_frames,
                                future_offset=future_offset,
                            )
                        wm2_step_batch = None
                        if wm2_enabled:
                            wm2_step_batch = build_windowed_batch(
                                batch=batch,
                                num_history=wm2_args.num_history,
                                num_frames=wm2_args.num_frames,
                                future_offset=future_offset,
                            )

                        if wm1_enabled:
                            wm1_pred_seg_raw = infer_pred_latents_raw(
                                model=wm1,
                                args=wm1_args,
                                accelerator=accelerator,
                                batch=use_model_action(wm1_step_batch, "wm1"),
                                history_latent=wm1_hist_buf.data,
                                current_latent=wm1_current,
                            )
                            wm1_gt_seg_raw = wm1_step_batch[wm1_args.predicted_datatype].to(
                                accelerator.device, non_blocking=True
                            )
                            w1_pred_seg, w1_gt_seg, w1_pred_lat, w1_gt_lat = decode_and_format_outputs_for_metrics(
                                wm1_pred_seg_raw,
                                wm1_gt_seg_raw,
                                wm1_args,
                                wm1_pipeline,
                            )
                            wm1_pred_for_wm2 = apply_snap_to_palette_on_seg_latents(
                                seg_latents_btchw=wm1_pred_seg_raw,
                                pipeline=wm1_pipeline,
                                decode_chunk_size=wm1_args.decode_chunk_size,
                                enable_snap_to_palette=args_cli.snap_to_palette,
                            )

                        if wm2_enabled and wm1_enabled:
                            wm2_step_cascade_batch = build_wm2_cascade_batch(
                                batch=wm2_step_batch,
                                seg_history_latents=wm2_seg_hist_buf.data,
                                wm1_pred_seg_latents=wm1_pred_for_wm2,
                                wm2_args=wm2_args,
                                wm1_pipeline=wm1_pipeline,
                            )
                            wm2_pred_raw = infer_pred_latents_raw(
                                model=wm2,
                                args=wm2_args,
                                accelerator=accelerator,
                                batch=use_model_action(wm2_step_cascade_batch, "wm2"),
                                history_latent=wm2_hist_buf.data,
                                current_latent=wm2_current,
                                disable_action_conditioning=args_cli.wm2_disable_action_conditioning,
                            )
                            wm2_gt_raw = wm2_step_batch[wm2_args.predicted_datatype].to(
                                accelerator.device, non_blocking=True
                            )
                            w2_pred_video, w2_gt_video, w2_pred_lat, w2_gt_lat = decode_and_format_outputs_for_metrics(
                                wm2_pred_raw,
                                wm2_gt_raw,
                                wm2_args,
                                wm2_pipeline,
                            )

                        elif wm2_enabled and not wm1_enabled:
                            # WM2 standalone: use GT segmentation (or no conditioning) each step.
                            if wm2_args.use_controlnet_conditioning:
                                gt_seg_history = wm2_step_batch["latent_segmentation_videos"][:, : wm2_args.num_history]
                                gt_seg_future = wm2_step_batch["latent_segmentation_videos"][:, wm2_args.num_history :]
                                wm2_standalone_step_batch = build_wm2_cascade_batch(
                                    batch=wm2_step_batch,
                                    seg_history_latents=gt_seg_history,
                                    wm1_pred_seg_latents=gt_seg_future,
                                    wm2_args=wm2_args,
                                    wm1_pipeline=wm2_pipeline,
                                )
                            else:
                                wm2_standalone_step_batch = wm2_step_batch
                            wm2_standalone_pred_raw = infer_pred_latents_raw(
                                model=wm2,
                                args=wm2_args,
                                accelerator=accelerator,
                                batch=use_model_action(wm2_standalone_step_batch, "wm2"),
                                history_latent=wm2_hist_buf.data,
                                current_latent=wm2_current,
                                disable_action_conditioning=args_cli.wm2_disable_action_conditioning,
                            )
                            wm2_standalone_gt_raw = wm2_step_batch[wm2_args.predicted_datatype].to(
                                accelerator.device, non_blocking=True
                            )
                            ws_pred_video, ws_gt_video, ws_pred_lat, ws_gt_lat = decode_and_format_outputs_for_metrics(
                                wm2_standalone_pred_raw,
                                wm2_standalone_gt_raw,
                                wm2_args,
                                wm2_pipeline,
                            )

                        if baseline_wm_enabled:
                            baseline_wm_step_batch = build_windowed_batch(
                                batch=batch,
                                num_history=baseline_wm_args.num_history,
                                num_frames=baseline_wm_args.num_frames,
                                future_offset=future_offset,
                            )
                            baseline_wm_pred_raw = infer_pred_latents_raw(
                                model=baseline_wm,
                                args=baseline_wm_args,
                                accelerator=accelerator,
                                batch=use_model_action(baseline_wm_step_batch, "baseline_wm"),
                                history_latent=baseline_wm_hist_buf.data,
                                current_latent=baseline_wm_current,
                                disable_action_conditioning=args_cli.wm2_disable_action_conditioning,
                            )
                            baseline_wm_gt_raw = baseline_wm_step_batch[baseline_wm_args.predicted_datatype].to(
                                accelerator.device, non_blocking=True
                            )
                            bwm_pred_video, bwm_gt_video, bwm_pred_lat, bwm_gt_lat = decode_and_format_outputs_for_metrics(
                                baseline_wm_pred_raw,
                                baseline_wm_gt_raw,
                                baseline_wm_args,
                                baseline_wm_pipeline,
                            )

                        # Feed back predictions into the circular buffers.
                        if wm1_enabled:
                            wm1_hist_buf.push(wm1_pred_seg_raw)
                            wm1_current = wm1_hist_buf.current
                        if wm2_enabled and wm1_enabled:
                            wm2_hist_buf.push(wm2_pred_raw)
                            wm2_seg_hist_buf.push(wm1_pred_for_wm2)
                            wm2_current = wm2_hist_buf.current
                        elif wm2_enabled and not wm1_enabled:
                            wm2_hist_buf.push(wm2_standalone_pred_raw)
                            wm2_current = wm2_hist_buf.current
                        if baseline_wm_enabled:
                            baseline_wm_hist_buf.push(baseline_wm_pred_raw)
                            baseline_wm_current = baseline_wm_hist_buf.current

                        if wm1_enabled:
                            batch_wm1_pred_seg.append(w1_pred_seg)
                            batch_wm1_pred_seg_lat.append(w1_pred_lat)
                            batch_wm1_gt_seg_future.append(w1_gt_seg[:, wm1_args.num_history :])
                            batch_wm1_gt_seg_lat_future.append(w1_gt_lat[:, wm1_args.num_history :])
                        if wm2_enabled and wm1_enabled:
                            batch_wm2_pred_video.append(w2_pred_video)
                            batch_wm2_pred_lat.append(w2_pred_lat)
                            batch_wm2_gt_video_future.append(w2_gt_video[:, wm2_args.num_history :])
                            batch_wm2_gt_lat_future.append(w2_gt_lat[:, wm2_args.num_history :])
                        elif wm2_enabled and not wm1_enabled:
                            batch_wm2_standalone_pred_video.append(ws_pred_video)
                            batch_wm2_standalone_pred_lat.append(ws_pred_lat)
                            batch_wm2_standalone_gt_video_future.append(ws_gt_video[:, wm2_args.num_history :])
                            batch_wm2_standalone_gt_lat_future.append(ws_gt_lat[:, wm2_args.num_history :])
                        if baseline_wm_enabled:
                            batch_baseline_wm_pred_video.append(bwm_pred_video)
                            batch_baseline_wm_pred_lat.append(bwm_pred_lat)
                            batch_baseline_wm_gt_video_future.append(bwm_gt_video[:, baseline_wm_args.num_history :])
                            batch_baseline_wm_gt_lat_future.append(bwm_gt_lat[:, baseline_wm_args.num_history :])

                        if step_idx == 0:
                            if wm1_enabled:
                                wm1_gt_seg_history = w1_gt_seg[:, : wm1_args.num_history]
                                wm1_gt_seg_lat_history = w1_gt_lat[:, : wm1_args.num_history]
                            if wm2_enabled and wm1_enabled:
                                wm2_gt_video_history = w2_gt_video[:, : wm2_args.num_history]
                                wm2_gt_lat_history = w2_gt_lat[:, : wm2_args.num_history]
                            elif wm2_enabled and not wm1_enabled:
                                wm2_standalone_gt_video_history = ws_gt_video[:, : wm2_args.num_history]
                                wm2_standalone_gt_lat_history = ws_gt_lat[:, : wm2_args.num_history]
                            if baseline_wm_enabled:
                                baseline_wm_gt_video_history = bwm_gt_video[:, : baseline_wm_args.num_history]
                                baseline_wm_gt_lat_history = bwm_gt_lat[:, : baseline_wm_args.num_history]

                    if wm1_enabled:
                        ar_pred_seg = torch.cat(batch_wm1_pred_seg, dim=1)
                        ar_gt_seg = torch.cat([wm1_gt_seg_history] + batch_wm1_gt_seg_future, dim=1)
                        ar_pred_seg_lat = torch.cat(batch_wm1_pred_seg_lat, dim=1)
                        ar_gt_seg_lat = torch.cat([wm1_gt_seg_lat_history] + batch_wm1_gt_seg_lat_future, dim=1)
                        _collect_batch_metrics(
                            ar_pred_seg, ar_gt_seg, ar_pred_seg_lat, ar_gt_seg_lat,
                            wm1_args, wm1_seg_metrics, batch_idx, accelerator,

                            prefix="wm1_seg",
                            step_horizon=step_horizon,
                        )
                    if wm2_enabled and wm1_enabled:
                        ar_pred_video = torch.cat(batch_wm2_pred_video, dim=1)
                        ar_gt_video = torch.cat([wm2_gt_video_history] + batch_wm2_gt_video_future, dim=1)
                        ar_pred_lat = torch.cat(batch_wm2_pred_lat, dim=1)
                        ar_gt_lat = torch.cat([wm2_gt_lat_history] + batch_wm2_gt_lat_future, dim=1)
                        _collect_batch_metrics(
                            ar_pred_video, ar_gt_video, ar_pred_lat, ar_gt_lat,
                            wm2_args, wm2_ar_metrics, batch_idx, accelerator,
                            prefix="wm2_ar",
                            step_horizon=step_horizon,
                        )
                    elif wm2_enabled and not wm1_enabled:
                        ar_pred_video = torch.cat(batch_wm2_standalone_pred_video, dim=1)
                        ar_gt_video = torch.cat([wm2_standalone_gt_video_history] + batch_wm2_standalone_gt_video_future, dim=1)
                        ar_pred_lat = torch.cat(batch_wm2_standalone_pred_lat, dim=1)
                        ar_gt_lat = torch.cat([wm2_standalone_gt_lat_history] + batch_wm2_standalone_gt_lat_future, dim=1)
                        _collect_batch_metrics(
                            ar_pred_video, ar_gt_video, ar_pred_lat, ar_gt_lat,
                            wm2_args, wm2_standalone_ar_metrics, batch_idx, accelerator,
                            prefix="wm2_standalone_ar",
                            step_horizon=step_horizon,
                        )
                    if baseline_wm_enabled:
                        ar_pred_video_bwm = torch.cat(batch_baseline_wm_pred_video, dim=1)
                        ar_gt_video_bwm = torch.cat([baseline_wm_gt_video_history] + batch_baseline_wm_gt_video_future, dim=1)
                        ar_pred_lat_bwm = torch.cat(batch_baseline_wm_pred_lat, dim=1)
                        ar_gt_lat_bwm = torch.cat([baseline_wm_gt_lat_history] + batch_baseline_wm_gt_lat_future, dim=1)
                        _collect_batch_metrics(
                            ar_pred_video_bwm, ar_gt_video_bwm, ar_pred_lat_bwm, ar_gt_lat_bwm,
                            baseline_wm_args, baseline_wm_ar_metrics, batch_idx, accelerator,
                            prefix="baseline_wm_ar",
                            step_horizon=step_horizon,
                        )

    # Compute and log average metrics over the whole dataset.
    if accelerator.is_main_process:
        def _log_avg_metrics(metrics_store: Dict, prefix: str, log_dict: Dict) -> None:
            if not metrics_store["psnr"]:
                return
            n = len(metrics_store["psnr"])
            for metric_key in METRIC_KEYS:
                log_dict[f"{prefix}/avg_{metric_key}"] = float(np.nanmean(metrics_store[metric_key]))

            num_views = int(metrics_store["num_views"] or 0)
            for view_idx in range(num_views):
                for metric_key in METRIC_KEYS:
                    log_dict[f"{prefix}/view_{view_idx:02d}/avg_{metric_key}"] = float(
                        np.nanmean([row[view_idx] for row in metrics_store["per_sample_view"][metric_key]])
                    )

            print(
                f"[{prefix}] avg over {n} samples and {num_views} views — "
                f"PSNR={log_dict[f'{prefix}/avg_psnr']:.4f}, "
                f"SSIM={log_dict[f'{prefix}/avg_ssim']:.4f}, "
                f"LPIPS={log_dict[f'{prefix}/avg_lpips']:.4f}"
            )

        avg_log: Dict = {}

        def _add_model_logs(store, prefix, margs):
            _log_avg_metrics(store, prefix, avg_log)
            avg_log.update(_build_metric_plots(store, prefix))
            avg_log[f"{prefix}/per_view_avg_metrics"] = _build_per_view_avg_table(store)
            avg_log[f"{prefix}/per_sample_view_metrics"] = _build_per_sample_view_table(store)
            avg_log[f"{prefix}/metrics_over_time_per_sample"] = _build_metric_time_table(store)
            avg_log[f"{prefix}/all_videos"] = _build_video_table(store, margs)
            if store.get("ar_step_rows"):
                avg_log[f"{prefix}/metrics_over_ar_steps"] = _build_ar_step_table(store)
                avg_log.update(_build_ar_step_plots(store, prefix))

        if wm1_enabled:
            _add_model_logs(wm1_seg_metrics, "wm1_seg", wm1_args)
        if wm2_enabled:
            if args_cli.inference_mode == "oneshot":
                _add_model_logs(wm2_baseline_metrics, "wm2_baseline", wm2_args)
                if wm1_enabled:
                    _add_model_logs(wm2_cascade_metrics, "wm2_cascade", wm2_args)
            elif wm1_enabled:
                _add_model_logs(wm2_ar_metrics, "wm2_ar", wm2_args)
            else:
                _add_model_logs(wm2_standalone_ar_metrics, "wm2_standalone_ar", wm2_args)
        if baseline_wm_enabled:
            if args_cli.inference_mode == "oneshot":
                _add_model_logs(baseline_wm_metrics, "baseline_wm", baseline_wm_args)
            else:
                _add_model_logs(baseline_wm_ar_metrics, "baseline_wm_ar", baseline_wm_args)

        if avg_log:
            accelerator.log(avg_log, step=batch_idx + 1)

        summary_lines = [f"  {k}: {v:.4f}" for k, v in avg_log.items() if isinstance(v, float)]
        send_discord_message(
            f"✅ **Inference complete** ({args_cli.inference_mode})\n"
            + "\n".join(summary_lines)
        )


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
        default=f"inference_output/wm1_to_wm2_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--wm1_config_path", type=str, default=None)
    parser.add_argument("--wm2_config_path", type=str, default=None)
    parser.add_argument("--baseline_wm_config_path", type=str, default=None)
    parser.add_argument("--wandb_project_name", type=str, default="mask2real-wm-inference")
    parser.add_argument("--wandb_run_name", type=str, default="inference")
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
        help="Optional stat.json path used for action normalization instead of per-dataset meta stats.",
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
    parser.add_argument("--num_of_samples_for_inference", type=int, default=None)
    parser.add_argument("--inference_mode", type=str, choices=["oneshot", "autoregressive"], default="autoregressive")
    parser.add_argument("--wm1_num_frames", type=int, default=5)
    parser.add_argument("--wm2_num_frames", type=int, default=5)
    parser.add_argument("--baseline_wm_num_frames", type=int, default=5)
    parser.add_argument("--wm1_num_history", type=int, default=None,
                        help="Override WM1 num_history from config.")
    parser.add_argument("--wm2_num_history", type=int, default=None,
                        help="Override WM2 num_history from config.")
    parser.add_argument("--baseline_wm_num_history", type=int, default=None,
                        help="Override baseline_wm num_history from config.")
    parser.add_argument("--wm1_history_stride", type=int, default=1,
                        help="Stride for WM1 circular history buffer. "
                             "stride=2 keeps every 2nd frame, doubling the covered window.")
    parser.add_argument("--wm2_history_stride", type=int, default=1,
                        help="Stride for WM2 circular history buffers (video + seg cascade). "
                             "stride=2 keeps every 2nd frame, doubling the covered window.")
    parser.add_argument("--baseline_wm_history_stride", type=int, default=1,
                        help="Stride for baseline_wm circular history buffer. "
                             "stride=2 keeps every 2nd frame, doubling the covered window.")
    parser.add_argument("--ar_num_steps", type=int, default=30)
    parser.add_argument("--wm1_num_inference_steps", type=int, default=None,
                        help="Override WM1 denoising steps (num_inference_steps) from config.")
    parser.add_argument("--wm2_num_inference_steps", type=int, default=None,
                        help="Override WM2 denoising steps (num_inference_steps) from config.")
    parser.add_argument("--baseline_wm_num_inference_steps", type=int, default=None,
                        help="Override baseline_wm denoising steps (num_inference_steps) from config.")
    parser.add_argument("--validation_batch_size", type=int, default=None,
                        help="Override validation batch size for all models.")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional inference seed for Python, NumPy, and Torch RNGs.",
    )
    parser.add_argument("--snap_to_palette", action="store_true", default=False)
    parser.add_argument("--wm2_disable_action_conditioning", action="store_true")
    parser.add_argument("--wm2_disable_controlnet_conditioning", action="store_true",
                        help="Override WM2 use_controlnet_conditioning to False at inference time.")

    args_cli = parser.parse_args()
    set_inference_seed(args_cli.seed)

    wm1_args = wm_orca_args()
    if args_cli.wm1_config_path is not None:
        wm1_args = load_experiment_config(args_cli.wm1_config_path, wm1_args)

    wm2_args = wm_orca_args()
    if args_cli.wm2_config_path is not None:
        wm2_args = load_experiment_config(args_cli.wm2_config_path, wm2_args)

    baseline_wm_args = wm_orca_args()
    if args_cli.baseline_wm_config_path is not None:
        baseline_wm_args = load_experiment_config(args_cli.baseline_wm_config_path, baseline_wm_args)

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

    if args_cli.wandb_project_name is not None:
        wm1_args.wandb_project_name = args_cli.wandb_project_name
        wm2_args.wandb_project_name = args_cli.wandb_project_name
        baseline_wm_args.wandb_project_name = args_cli.wandb_project_name
    if args_cli.wandb_run_name is not None:
        wm1_args.wandb_run_name = args_cli.wandb_run_name
        wm2_args.wandb_run_name = args_cli.wandb_run_name
        baseline_wm_args.wandb_run_name = args_cli.wandb_run_name

    if args_cli.num_of_samples_for_inference is not None:
        wm1_args.max_num_samples_for_validation = args_cli.num_of_samples_for_inference
        wm2_args.max_num_samples_for_validation = args_cli.num_of_samples_for_inference
        baseline_wm_args.max_num_samples_for_validation = args_cli.num_of_samples_for_inference
    else:
        wm1_args.max_num_samples_for_validation = sys.maxsize
        wm2_args.max_num_samples_for_validation = sys.maxsize
        baseline_wm_args.max_num_samples_for_validation = sys.maxsize

    if args_cli.validation_batch_size is not None:
        wm1_args.validation_batch_size = args_cli.validation_batch_size
        wm2_args.validation_batch_size = args_cli.validation_batch_size
        baseline_wm_args.validation_batch_size = args_cli.validation_batch_size

    if args_cli.debug:
        if args_cli.num_of_samples_for_inference is None:
            wm1_args.max_num_samples_for_validation = 6
            wm2_args.max_num_samples_for_validation = 6
            baseline_wm_args.max_num_samples_for_validation = 6

        wm1_args.validation_batch_size = 2
        wm2_args.validation_batch_size = 2
        baseline_wm_args.validation_batch_size = 2

        if args_cli.wm1_num_inference_steps is None:
            wm1_args.num_inference_steps = 50
        if args_cli.wm2_num_inference_steps is None:
            wm2_args.num_inference_steps = 50
        if args_cli.baseline_wm_num_inference_steps is None:
            baseline_wm_args.num_inference_steps = 50

    if args_cli.wm1_num_frames is not None and args_cli.wm1_num_frames <= 0:
        raise ValueError(f"wm1_num_frames must be positive, got {args_cli.wm1_num_frames}.")
    if args_cli.wm2_num_frames is not None and args_cli.wm2_num_frames <= 0:
        raise ValueError(f"wm2_num_frames must be positive, got {args_cli.wm2_num_frames}.")
    if args_cli.baseline_wm_num_frames is not None and args_cli.baseline_wm_num_frames <= 0:
        raise ValueError(f"baseline_wm_num_frames must be positive, got {args_cli.baseline_wm_num_frames}.")
    if args_cli.wm1_num_inference_steps is not None and args_cli.wm1_num_inference_steps <= 0:
        raise ValueError(
            f"wm1_num_inference_steps must be positive, got {args_cli.wm1_num_inference_steps}."
        )
    if args_cli.wm2_num_inference_steps is not None and args_cli.wm2_num_inference_steps <= 0:
        raise ValueError(
            f"wm2_num_inference_steps must be positive, got {args_cli.wm2_num_inference_steps}."
        )
    if (
        args_cli.baseline_wm_num_inference_steps is not None
        and args_cli.baseline_wm_num_inference_steps <= 0
    ):
        raise ValueError(
            "baseline_wm_num_inference_steps must be positive, "
            f"got {args_cli.baseline_wm_num_inference_steps}."
        )
    if args_cli.ar_num_steps <= 0:
        raise ValueError(f"ar_num_steps must be positive, got {args_cli.ar_num_steps}.")

    main(wm1_args, wm2_args, baseline_wm_args, args_cli)

