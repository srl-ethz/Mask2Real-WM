#!/usr/bin/env python3
"""
make_wm1_iou_video.py — animated version of make_wm1_iou_figure.py.

Same layout as the static figure (one row per example hand state; columns GT /
Cascade-R / Cascade-S / Cascade-SR; overlap green=TP, red=FP, blue=FN; IoU under each
model panel), but instead of only the rollout's final frame, every frame of
the rollout is shown in turn: the predicted hand masks move, and each IoU
number is recomputed live against the (static) ground-truth target mask.
The last frame's IoU matches the static figure exactly.

The crop window covers the GT and every model's predicted hand over ALL
frames (not just the last), so the hand's whole path stays in view — panels
are therefore zoomed out somewhat vs. the static figure.

Canvas defaults to a 16:9 1920x1080 PowerPoint slide with slide-sized text;
the panels are widened to fill it (see --width_px/--height_px/--font_scale).
Styled white-on-black in Arial by default (--bg/--text_color/--font); the
mask panels share the background color, so masks sit directly on black.

Usage
-----
  python scripts/paper/make_wm1_iou_video.py \\
      --gt_manifest inference_output/full_run_gt_renders/gt_manifest.json \\
      --rollout_manifest inference_output/controllability_eval_4variant_id_.../ \\
      --rollout_manifest inference_output/controllability_eval_4variant_ood_.../ \\
      --output figures/wm1_controllability_iou.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import mediapy
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for compute_wm1_controllability_seg_iou

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
from make_wm1_iou_figure import (
    DEFAULT_EXAMPLES,
    FN_COLOR,
    FP_COLOR,
    GRID_LEFT,
    GRID_RIGHT,
    GT_COLOR,
    MODEL_ORDER,
    TP_COLOR,
    add_legend,
    compute_crop_windows,
    create_figure,
    draw_qual_row,
)


def load_example_sequence(
    trial_group_id: str,
    approach_style: str,
    gt_by_trial_group: Dict[str, Dict],
    trials_by_id: Dict[str, Dict],
    camera: str,
    repo_root: Path,
    color_threshold: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """(gt_mask [H,W], {model: pred_masks [T,H,W]}, union over GT + all frames + all models)."""
    gt_native = load_gt_hand_mask(gt_by_trial_group[trial_group_id], camera, repo_root)
    cam_idx = CAMERA_VIEW_ORDER.index(camera)

    preds: Dict[str, np.ndarray] = {}
    for model in MODEL_ORDER:
        trial = trials_by_id[f"{trial_group_id}_{approach_style}_{model}"]
        frames = read_last_frames(resolve_path(trial["wm1_seg_video_path"], repo_root), 10**9)
        views = [split_views(f, len(CAMERA_VIEW_ORDER))[cam_idx] for f in frames]
        preds[model] = np.stack([green_hand_mask(v, color_threshold) for v in views])

    n_frames = min(p.shape[0] for p in preds.values())
    preds = {m: p[:n_frames] for m, p in preds.items()}
    ref_shape = preds[MODEL_ORDER[0]].shape[1:]
    gt = resize_mask_to(gt_native, ref_shape)

    union = gt.copy()
    for p in preds.values():
        union |= p.any(axis=0)
    return gt, preds, union


def overlap_frames(gt: np.ndarray, pred: np.ndarray, bg_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """[T,H,W,3] TP/FP/FN overlap images and [T] IoU values for one model."""
    T = pred.shape[0]
    out = np.tile(bg_rgb, (T,) + gt.shape + (1,)).astype(np.uint8)
    g = gt[None]
    out[pred & ~g] = FP_COLOR
    out[~pred & g] = FN_COLOR
    out[pred & g] = TP_COLOR
    ious = np.array([iou(gt, pred[t]) for t in range(T)])
    return out, ious


def hex_to_rgb(color: str) -> np.ndarray:
    c = color.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)])


def crop_and_upscale(arr: np.ndarray, window: Tuple[int, int, int, int], scale: int) -> np.ndarray:
    """Works on [H,W,3] and [T,H,W,3] alike (H/W are always axes -3/-2)."""
    r0, r1, c0, c1 = window
    cropped = arr[..., r0:r1, c0:c1, :]
    return np.repeat(np.repeat(cropped, scale, axis=-3), scale, axis=-2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt_manifest", required=True)
    p.add_argument("--rollout_manifest", action="append", required=True, dest="rollout_manifests")
    p.add_argument(
        "--example", action="append", dest="examples", default=None,
        help="trial_group_id:approach_style:row_label — pass 3 times to override the default examples.",
    )
    p.add_argument("--camera", default="side_camera_one", choices=CAMERA_VIEW_ORDER)
    p.add_argument("--color_threshold", type=float, default=DEFAULT_COLOR_THRESHOLD)
    p.add_argument("--repo_root", default=str(REPO_ROOT))
    p.add_argument("--fps", type=float, default=5.0, help="Playback fps (the rollouts themselves are 5 fps).")
    p.add_argument("--hold_end", type=int, default=10, help="Extra frames to hold the final frame.")
    p.add_argument("--width_px", type=int, default=1920, help="Video width (default 1920x1080 = 16:9 slide).")
    p.add_argument("--height_px", type=int, default=1080)
    p.add_argument("--dpi", type=int, default=144, help="Figure dpi; with 1920x1080 this is a 13.33x7.5in slide.")
    p.add_argument("--font_scale", type=float, default=2.3,
                   help="Text size relative to the 3.45in single-column figure (9pt labels at 1.0).")
    p.add_argument("--bg", default="#000000", help="Background (and mask-panel) color, hex.")
    p.add_argument("--text_color", default="#ffffff", help="Text color, hex.")
    p.add_argument("--font", default="Arial", help="Font family (must be installed).")
    p.add_argument("--output", required=True, help="Output .mp4 path.")
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    if args.examples:
        examples = [tuple(e.split(":", 2)) for e in args.examples]
    else:
        examples = DEFAULT_EXAMPLES

    gt_manifest = _load_json(resolve_path(args.gt_manifest, repo_root))
    gt_by_trial_group = {r["trial_group_id"]: r for r in gt_manifest["results"]}
    trials_by_id: Dict[str, Dict] = {}
    for path in discover_rollout_manifests(args.rollout_manifests):
        for t in _load_json(resolve_path(path, repo_root))["results"]:
            trials_by_id[t["trial_id"]] = t

    seqs = [
        load_example_sequence(tg, ap, gt_by_trial_group, trials_by_id, args.camera, repo_root, args.color_threshold)
        for tg, ap, _ in examples
    ]
    n_frames = min(pred.shape[0] for _, preds, _ in seqs for pred in preds.values())

    # Slide canvas layout (inches). Margins are derived from the font sizes so the
    # column titles, per-row IoU labels and the legend just fit, which fixes the
    # panel shape; the crop windows are then grown to that same aspect so the
    # panels fill the slide without imshow shrinking them.
    dpi = args.dpi
    fs = args.font_scale
    bg_rgb = hex_to_rgb(args.bg)
    gt_rgb = np.array([235, 235, 235]) if bg_rgb.mean() < 128 else GT_COLOR  # keep the GT mask contrasting
    fig_w, fig_h = args.width_px / dpi, args.height_px / dpi
    n_rows = len(examples)
    label_in = 11 * fs / 72 + 0.03                  # IoU label under a row
    legend_in = 1.3 * 8 * fs / 72                   # legend row
    top_in = 12.5 * fs / 72 + 0.12                  # column titles
    bottom_in = label_in + 0.04 + legend_in + 0.03  # last row's IoU label + legend, tight
    gap_in = label_in + 0.10                        # between rows: room for the IoU label
    panel_w = fig_w * (GRID_RIGHT - GRID_LEFT) / (4 + 3 * 0.06)
    panel_h = (fig_h - top_in - bottom_in - (n_rows - 1) * gap_in) / n_rows

    windows, box_h = compute_crop_windows(
        [(union, union.shape) for _, _, union in seqs],
        aspect=panel_w / panel_h, pad_frac=0.1, center="bbox",
    )
    scale = max(1, int(np.ceil(panel_h * dpi / box_h)))

    # rows[i][col] -> [T,h,w,3] panel frames ; ious[i][model] -> [T]
    rows: List[Dict[str, np.ndarray]] = []
    ious: List[Dict[str, np.ndarray]] = []
    for (gt, preds, _), window in zip(seqs, windows):
        gt_panel = np.tile(bg_rgb, gt.shape + (1,)).astype(np.uint8)
        gt_panel[gt] = gt_rgb
        row = {"gt": crop_and_upscale(gt_panel, window, scale)}
        row_iou = {}
        for m in MODEL_ORDER:
            frames_m, iou_m = overlap_frames(gt, preds[m][:n_frames], bg_rgb)
            row[m] = crop_and_upscale(frames_m, window, scale)
            row_iou[m] = iou_m
        rows.append(row)
        ious.append(row_iou)

    fig, gs = create_figure(
        n_rows, hspace=gap_in / panel_h, fig_height=fig_h, width=fig_w,
        top=1 - top_in / fig_h, bottom=bottom_in / fig_h, font_scale=fs,
        font_family=args.font, facecolor=args.bg, text_color=args.text_color,
    )
    fig.set_dpi(dpi)

    def panels_at(i: int, t: int) -> Dict[str, np.ndarray]:
        d = {k: v if k == "gt" else v[t] for k, v in rows[i].items()}
        d.update({f"{m}_iou": ious[i][m][t] for m in MODEL_ORDER})
        d["gt"] = rows[i]["gt"]
        return d

    handles = [
        draw_qual_row(fig, gs[i].subgridspec(1, 4, wspace=0.06), panels_at(i, 0), show_col_titles=(i == 0), font_scale=fs)
        for i in range(n_rows)
    ]
    add_legend(fig, font_scale=fs)

    video: List[np.ndarray] = []
    for t in list(range(n_frames)) + [n_frames - 1] * args.hold_end:
        for i in range(n_rows):
            for m in MODEL_ORDER:
                ax, im = handles[i][m]
                im.set_data(rows[i][m][t])
                ax.xaxis.label.set_text(f"IoU={ious[i][m][t]:.2f}")
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        h, w = rgb.shape[:2]
        video.append(rgb[: h - h % 2, : w - w % 2])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(out), video, fps=args.fps)
    print(f"[saved] {out}  ({video[0].shape[1]}x{video[0].shape[0]}, {len(video)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
