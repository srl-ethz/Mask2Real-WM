#!/usr/bin/env python3
"""
visualize_wm1_seg_iou_overlap.py — see what the IoU score in
compute_wm1_controllability_seg_iou.py is actually measuring.

For one or more trial_ids, renders a row of 3 panels on the requested camera
view's final rollout frame:

  1. Predicted hand mask   — green-thresholded pixels from WM1's own
                              predicted `wm1_seg_video_path` (see
                              green_hand_mask() in compute_wm1_controllability_seg_iou.py)
  2. Ground-truth hand mask — `hand_instance_ids` applied to the faive_lab
                              Isaac-Sim render's `<camera>_seg_raw_ids.npy`
  3. Overlap                — both masks combined pixel-by-pixel:
                                green  = TP (both agree it's hand)
                                red    = FP (predicted hand, GT says no)
                                blue   = FN (GT says hand, predicted no)
                                dark   = both agree it's not hand
                              IoU = TP / (TP + FP + FN), i.e. green pixels
                              divided by every colored (non-dark) pixel.

Pass multiple --trial_id values to stack several examples as rows in one
image — e.g. three unrelated trials spanning the IoU range (good/medium/bad),
or the same trial_group_id's 3 WM1-variant trial_ids side by side to compare
models on one shared target.

Usage
-----
  python scripts/visualize_wm1_seg_iou_overlap.py \\
      --gt_manifest inference_output/full_run_gt_renders/gt_manifest.json \\
      --rollout_manifest inference_output/controllability_eval_4variant_id_.../ \\
      --rollout_manifest inference_output/controllability_eval_4variant_ood_.../ \\
      --trial_id id_s00017_wholepose_full_t00_direct_wm1_midtrain_lora45000 \\
      --trial_id id_s00083_wholepose_full_t01_direct_wm1_midtrain_lora45000 \\
      --camera side_camera_one \\
      --output overlap_examples.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compute_wm1_controllability_seg_iou import (
    CAMERA_VIEW_ORDER,
    DEFAULT_COLOR_THRESHOLD,
    REPO_ROOT,
    _load_json,
    discover_rollout_manifests,
    green_hand_mask,
    iou,
    load_gt_hand_mask,
    read_last_frames,
    resize_mask_to,
    resolve_path,
    split_views,
)

PANEL_LABELS = ("1. Predicted hand mask (WM1)", "2. Ground-truth hand mask", "3. Overlap (green=TP red=FP blue=FN)")
BG_COLOR = (25, 25, 25)
TP_COLOR = (60, 220, 60)
FP_COLOR = (220, 60, 60)
FN_COLOR = (60, 110, 230)
PRED_COLOR = (60, 220, 60)
GT_COLOR = (230, 230, 230)
TITLE_H = 20
HEADER_H = 20
GAP = 4


def _load_all_trials(rollout_manifest_paths: List[str], repo_root: Path) -> Dict[str, Dict[str, Any]]:
    trials: Dict[str, Dict[str, Any]] = {}
    for path in discover_rollout_manifests(rollout_manifest_paths):
        manifest = _load_json(resolve_path(path, repo_root))
        for trial in manifest["results"]:
            trials[trial["trial_id"]] = trial
    return trials


def _upscale(arr: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)


def _solid(mask: np.ndarray, color: tuple) -> np.ndarray:
    out = np.full(mask.shape + (3,), BG_COLOR, dtype=np.uint8)
    out[mask] = color
    return out


def _text_strip(width: int, height: int, text: str, bg=BG_COLOR) -> np.ndarray:
    strip = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    return np.array(strip)


def render_trial_row(
    trial: Dict[str, Any],
    gt_entry: Dict[str, Any],
    camera: str,
    repo_root: Path,
    color_threshold: float,
    upscale: int,
) -> np.ndarray:
    seg_video_path = resolve_path(trial["wm1_seg_video_path"], repo_root)
    frame = read_last_frames(seg_video_path, 1)[0]
    num_views = len(CAMERA_VIEW_ORDER)
    views = split_views(frame, num_views)
    cam_idx = CAMERA_VIEW_ORDER.index(camera)
    view = views[cam_idx]

    pred_mask = green_hand_mask(view, color_threshold)
    gt_mask = load_gt_hand_mask(gt_entry, camera, repo_root)
    gt_mask_resized = resize_mask_to(gt_mask, view.shape[:2]) if gt_mask is not None else np.zeros(view.shape[:2], dtype=bool)
    iou_val = iou(gt_mask_resized, pred_mask)

    pred_panel = _solid(pred_mask, PRED_COLOR)
    gt_panel = _solid(gt_mask_resized, GT_COLOR)
    overlap = np.full(pred_mask.shape + (3,), BG_COLOR, dtype=np.uint8)
    tp = pred_mask & gt_mask_resized
    fp = pred_mask & ~gt_mask_resized
    fn = ~pred_mask & gt_mask_resized
    overlap[fp] = FP_COLOR
    overlap[fn] = FN_COLOR
    overlap[tp] = TP_COLOR

    panels = [_upscale(p, upscale) for p in (pred_panel, gt_panel, overlap)]
    gap = np.full((panels[0].shape[0], GAP, 3), BG_COLOR, dtype=np.uint8)
    row = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)

    label = (
        f"{trial['trial_id']}  |  variant={trial['model_variant']}  split={trial.get('split')}  "
        f"scope={trial.get('sampling_scope')}({trial.get('component_label') or 'all'})  "
        f"approach={trial.get('approach_style')}  camera={camera}  |  IoU_hand = {iou_val:.3f}"
    )
    title = _text_strip(row.shape[1], TITLE_H, label)
    return np.concatenate([title, row], axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt_manifest", required=True)
    p.add_argument("--rollout_manifest", action="append", required=True, dest="rollout_manifests")
    p.add_argument("--trial_id", action="append", required=True, dest="trial_ids")
    p.add_argument("--camera", default="side_camera_one", choices=CAMERA_VIEW_ORDER)
    p.add_argument("--color_threshold", type=float, default=DEFAULT_COLOR_THRESHOLD)
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--repo_root", default=str(REPO_ROOT))
    p.add_argument("--output", required=True)
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    gt_manifest = _load_json(resolve_path(args.gt_manifest, repo_root))
    gt_by_trial_group = {r["trial_group_id"]: r for r in gt_manifest["results"]}
    trials_by_id = _load_all_trials(args.rollout_manifests, repo_root)

    rows = []
    for trial_id in args.trial_ids:
        trial = trials_by_id.get(trial_id)
        if trial is None:
            raise KeyError(f"trial_id={trial_id!r} not found in any given --rollout_manifest")
        gt_entry = gt_by_trial_group.get(trial["trial_group_id"])
        if gt_entry is None:
            raise KeyError(f"no GT entry for trial_group_id={trial['trial_group_id']!r}")
        rows.append(render_trial_row(trial, gt_entry, args.camera, repo_root, args.color_threshold, args.upscale))

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    header = _text_strip(max_w, HEADER_H, "   |   ".join(PANEL_LABELS))
    padded_rows.append(header)
    for i, r in enumerate(rows):
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), BG_COLOR, dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded_rows.append(r)
        if i != len(rows) - 1:
            padded_rows.append(np.full((GAP * 2, max_w, 3), (0, 0, 0), dtype=np.uint8))

    figure = np.concatenate(padded_rows, axis=0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(figure).save(args.output)
    print(f"[saved] {args.output}  ({figure.shape[1]}x{figure.shape[0]})")


if __name__ == "__main__":
    main()
