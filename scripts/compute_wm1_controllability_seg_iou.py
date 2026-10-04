#!/usr/bin/env python3
"""
compute_wm1_controllability_seg_iou.py — WM1 controllability-eval hand IoU

Ranks WM1 checkpoint variants (e.g. real-data-only vs. sim-midtrained vs.
sim-midtrained+finetuned) by how well WM1's own predicted hand segmentation
matches the *true* target hand pose, for the controllability-eval trials
produced by scripts/inference_wm1_to_wm2_controllability_eval.py (see the
`controllability_eval` branch).

For each trial this joins:
  - a rollout_manifest_wm1_<variant>.json entry's `wm1_seg_video_path` — WM1's
    own predicted segmentation video for that rollout (hand rendered green,
    decoded from the model's latent seg-channel prediction), and
  - the matching gt_manifest.json entry for the same `trial_group_id` — the
    ground-truth hand mask, rendered in Isaac Sim by faive_lab's
    render_controllability_targets.py from the actual sampled target pose
    (`hand_instance_ids` indexing into `<camera>_seg_raw_ids.npy`).

...and computes Intersection-over-Union between WM1's predicted green-hand
mask and the true target hand mask, on the final frame of the rollout (the
frame WM1 was asked to land the hand on the target by). Only the hand class
is scored — object/background/arm pixels are ignored.

This gives a cheap, objective, per-model controllability proxy that doesn't
require the LLM judge or a human rater: if WM1's predicted hand ends up where
the true target hand is, IoU is high.

Usage
-----
  python scripts/compute_wm1_controllability_seg_iou.py \\
      --gt_manifest inference_output/full_run_gt_renders/gt_manifest.json \\
      --rollout_manifest inference_output/controllability_eval_4variant_id_.../ \\
      --rollout_manifest inference_output/controllability_eval_4variant_ood_.../ \\
      --output_csv inference_output/wm1_controllability_seg_iou_per_trial.csv

A --rollout_manifest argument may be a directory (in which case every
rollout_manifest_wm1_*.json inside it is used — baseline's
rollout_manifest_baseline_*.json is deliberately not matched, since the
baseline model has no wm1_seg_video_path) or a direct path to one manifest
file. Pass it multiple times to combine several eval runs (e.g. ID + OOD).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CAMERA_VIEW_ORDER = ["side_camera_one", "wrist_camera"]

# Distance (Euclidean, in 0-255 RGB space) from pure green (0, 255, 0) below
# which a decoded wm1_seg pixel counts as "hand". Chosen from direct
# measurement across a random sample of real wm1_seg.mp4 outputs (all 3 WM1
# variants, both id/ood splits): hand pixels (picked via a coarse green-
# dominance test) never exceeded distance 204, but every video had a clean
# gap with the *nearest* non-hand pixel always >= ~154. 130 sits safely below
# that observed floor (leaves ~24 units of margin against compression-noise
# outliers) while still catching all but the most extreme anti-aliased edge
# pixels of the hand silhouette.
DEFAULT_COLOR_THRESHOLD = 130.0
PURE_GREEN = np.array([0.0, 255.0, 0.0])


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: str, repo_root: Path) -> str:
    """Manifests store paths relative to the repo root the pipeline was run
    from, not relative to the manifest file itself — resolve against
    --repo_root (default: this script's repo) when the raw path doesn't
    exist as given (e.g. cwd differs from the original run)."""
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    candidate = str(repo_root / path)
    return candidate if os.path.isfile(candidate) else path


def discover_rollout_manifests(raw_paths: List[str]) -> List[str]:
    """Expand directories into their rollout_manifest_wm1_*.json files;
    pass individual .json files through unchanged. Explicitly excludes
    rollout_manifest_baseline_*.json — the baseline model isn't a WM1
    variant and has no wm1_seg_video_path."""
    manifests: List[str] = []
    for raw in raw_paths:
        if os.path.isdir(raw):
            found = sorted(glob.glob(os.path.join(raw, "rollout_manifest_wm1_*.json")))
            if not found:
                print(f"[WARN] no rollout_manifest_wm1_*.json under directory {raw}")
            manifests.extend(found)
        else:
            manifests.append(raw)
    return manifests


def green_hand_mask(frame_rgb: np.ndarray, color_threshold: float) -> np.ndarray:
    """Boolean mask of pixels within color_threshold of pure green — WM1's
    predicted hand class in its decoded segmentation video."""
    dist = np.linalg.norm(frame_rgb.astype(np.float32) - PURE_GREEN, axis=-1)
    return dist <= color_threshold


def split_views(frame: np.ndarray, num_views: int) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    if num_views != len(CAMERA_VIEW_ORDER):
        raise ValueError(f"Expected num_views={len(CAMERA_VIEW_ORDER)} (CAMERA_VIEW_ORDER), got {num_views}.")
    if w % num_views != 0:
        raise ValueError(f"Frame width {w} not divisible by num_views={num_views}.")
    view_w = w // num_views
    return [frame[:, v * view_w:(v + 1) * view_w] for v in range(num_views)]


def read_last_frames(video_path: str, num_last_frames: int) -> List[np.ndarray]:
    import mediapy
    frames = mediapy.read_video(video_path)
    n = len(frames)
    if n == 0:
        return []
    k = min(num_last_frames, n)
    return [np.asarray(frames[i])[..., :3] for i in range(n - k, n)]


def iou(gt: np.ndarray, pred: np.ndarray) -> float:
    intersection = int((gt & pred).sum())
    union = int((gt | pred).sum())
    if union == 0:
        return 1.0  # both empty -> perfect agreement (e.g. hand out of camera view)
    return intersection / union


def load_gt_hand_mask(gt_entry: Dict[str, Any], camera: str, repo_root: Path) -> Optional[np.ndarray]:
    hand_ids_by_camera = gt_entry.get("hand_instance_ids")
    if not hand_ids_by_camera:
        return None
    render_paths = gt_entry["render_paths"]
    seg_key = f"{camera}_seg_raw_ids"
    if seg_key not in render_paths:
        return None
    seg_ids = np.load(resolve_path(render_paths[seg_key], repo_root))
    hand_ids = hand_ids_by_camera.get(camera, [])
    if not hand_ids:
        return np.zeros_like(seg_ids, dtype=bool)
    return np.isin(seg_ids, np.array(hand_ids, dtype=seg_ids.dtype))


def resize_mask_to(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == target_hw:
        return mask
    h, w = target_hw
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def compute_trial_rows(
    trial: Dict[str, Any],
    gt_entry: Dict[str, Any],
    repo_root: Path,
    color_threshold: float,
    num_views: int,
    num_last_frames: int,
) -> List[Dict[str, Any]]:
    seg_video_path = resolve_path(trial["wm1_seg_video_path"], repo_root)
    frames = read_last_frames(seg_video_path, num_last_frames)
    if not frames:
        return []

    rows: List[Dict[str, Any]] = []
    per_camera_ious: Dict[str, List[float]] = {cam: [] for cam in CAMERA_VIEW_ORDER}

    for frame in frames:
        views = split_views(frame, num_views)
        for camera, view in zip(CAMERA_VIEW_ORDER, views):
            gt_mask = load_gt_hand_mask(gt_entry, camera, repo_root)
            if gt_mask is None:
                continue
            gt_mask_resized = resize_mask_to(gt_mask, view.shape[:2])
            pred_mask = green_hand_mask(view, color_threshold)
            per_camera_ious[camera].append(iou(gt_mask_resized, pred_mask))

    for camera, vals in per_camera_ious.items():
        if not vals:
            continue
        rows.append({
            "trial_id": trial["trial_id"],
            "trial_group_id": trial["trial_group_id"],
            "model_variant": trial["model_variant"],
            "split": trial.get("split"),
            "sampling_scope": trial.get("sampling_scope"),
            "component_label": trial.get("component_label"),
            "approach_style": trial.get("approach_style"),
            "camera": camera,
            "iou_hand": float(np.mean(vals)),
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute WM1-predicted vs. ground-truth hand-mask IoU for the controllability eval, per WM1 variant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gt_manifest", required=True, help="Path to gt_manifest.json (faive_lab GT renders).")
    p.add_argument(
        "--rollout_manifest", action="append", required=True, dest="rollout_manifests",
        help="Path to a rollout_manifest_wm1_<variant>.json, or a directory containing one or more "
             "such files (auto-globbed). Pass multiple times to combine eval runs (e.g. id + ood).",
    )
    p.add_argument("--repo_root", default=str(REPO_ROOT), help="Root to resolve relative manifest paths against.")
    p.add_argument("--color_threshold", type=float, default=DEFAULT_COLOR_THRESHOLD,
                   help="Euclidean RGB distance from pure green below which a wm1_seg pixel counts as hand.")
    p.add_argument("--num_views", type=int, default=2, help="Number of camera views concatenated per frame.")
    p.add_argument("--num_last_frames", type=int, default=1,
                   help="Average IoU over the last N frames of each rollout instead of just the final frame.")
    p.add_argument("--output_csv", default=None, help="If set, save per-trial-per-camera IoU rows to this CSV.")
    p.add_argument("--output_summary_csv", default=None, help="If set, save the per-model ranking table to this CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)

    gt_manifest = _load_json(resolve_path(args.gt_manifest, repo_root))
    gt_by_trial_group: Dict[str, Dict[str, Any]] = {r["trial_group_id"]: r for r in gt_manifest["results"]}
    print(f"[gt] {len(gt_by_trial_group)} ground-truth trial_group entries loaded from {args.gt_manifest}")

    rollout_manifest_paths = discover_rollout_manifests(args.rollout_manifests)
    if not rollout_manifest_paths:
        print("[ERROR] no rollout manifests found.")
        raise SystemExit(1)
    print(f"[rollout] {len(rollout_manifest_paths)} manifest file(s): {rollout_manifest_paths}")

    all_rows: List[Dict[str, Any]] = []
    n_missing_gt = 0
    n_trials = 0
    for manifest_path in rollout_manifest_paths:
        manifest = _load_json(resolve_path(manifest_path, repo_root))
        results = manifest["results"]
        print(f"  [load] {manifest_path}: {len(results)} trials")
        for i, trial in enumerate(results):
            n_trials += 1
            print(f"    [{i + 1}/{len(results)}] {trial['trial_id']}", end="\r", flush=True)
            gt_entry = gt_by_trial_group.get(trial["trial_group_id"])
            if gt_entry is None:
                n_missing_gt += 1
                continue
            try:
                rows = compute_trial_rows(
                    trial, gt_entry, repo_root,
                    color_threshold=args.color_threshold,
                    num_views=args.num_views,
                    num_last_frames=args.num_last_frames,
                )
            except Exception as exc:
                print(f"\n  [!] {trial['trial_id']}: {exc}")
                rows = []
            all_rows.extend(rows)
        print()

    print(f"\n[data] {n_trials} trials scanned, {n_missing_gt} skipped (no matching GT), {len(all_rows)} camera-view IoU rows")

    if not all_rows:
        print("[ERROR] no IoU rows computed. Exiting.")
        raise SystemExit(1)

    df = pd.DataFrame(all_rows)

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"[csv] per-trial rows saved -> {args.output_csv}")

    # Per-trial mean across camera views, then per-model aggregation.
    per_trial = df.groupby(["model_variant", "trial_id"], as_index=False)["iou_hand"].mean()
    summary = (
        per_trial.groupby("model_variant")["iou_hand"]
        .agg(mean_iou="mean", std_iou="std", n_trials="count")
        .sort_values("mean_iou", ascending=False)
        .reset_index()
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))

    print("\n" + "=" * 72)
    print("WM1 CONTROLLABILITY-EVAL HAND-IoU RANKING (higher = better controllability)")
    print("=" * 72)
    print(summary.to_string(index=False))
    print("=" * 72)

    if "split" in df.columns and df["split"].notna().any():
        per_trial_split = df.groupby(["model_variant", "split", "trial_id"], as_index=False)["iou_hand"].mean()
        split_summary = (
            per_trial_split.groupby(["model_variant", "split"])["iou_hand"]
            .agg(mean_iou="mean", n_trials="count")
            .round(4)
            .sort_values(["split", "mean_iou"], ascending=[True, False])
        )
        print("\n── Breakdown by split (id / ood) ──────────────────────────────────")
        print(split_summary.to_string())

    if args.output_summary_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_summary_csv)), exist_ok=True)
        summary.to_csv(args.output_summary_csv, index=False)
        print(f"\n[csv] ranking summary saved -> {args.output_summary_csv}")


if __name__ == "__main__":
    main()
