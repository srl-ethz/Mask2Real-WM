#!/usr/bin/env python3
"""
make_wm1_iou_figure.py — ICRA single-column (3.45in) qualitative figure for the WM1
controllability hand-IoU comparison (see compute_wm1_controllability_seg_iou.py).

Purely qualitative (the paper's results table already carries the numbers):
one column per model (+ ground truth), one row per example hand state, each
panel showing the predicted-vs-GT overlap (green=TP, red=FP, blue=FN) on that
rollout's final frame, cropped/zoomed to the hand region and annotated with
its IoU.

The 3 default example rows are: an OOD
whole-pose target where the real-data-only model fails to localize the hand
at all, an ID whole-pose target where all three models are comparable, and a
single-DOF (per-dimension) target where only the finetuned model succeeds.
Override with --example to show different trials.

Usage
-----
  python scripts/paper/make_wm1_iou_figure.py \\
      --gt_manifest inference_output/full_run_gt_renders/gt_manifest.json \\
      --rollout_manifest inference_output/controllability_eval_4variant_id_.../ \\
      --rollout_manifest inference_output/controllability_eval_4variant_ood_.../ \\
      --output figures/wm1_controllability_iou

  # Override examples: trial_group_id:approach_style:row_label
  python scripts/paper/make_wm1_iou_figure.py ... \\
      --example ood_s00038_wholepose_full_t02:direct:"OOD, whole pose" \\
      --example id_s00063_wholepose_full_t02:direct:"ID, whole pose" \\
      --example ood_s00024_perdim_dim06_t00:direct:"OOD, wrist only"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    load_gt_hand_mask,
    read_last_frames,
    resize_mask_to,
    resolve_path,
    split_views,
)

MODEL_ORDER = ["wm1_real_only", "wm1_midtrain_only", "wm1_midtrain_lora45000"]
MODEL_LABELS = {
    "wm1_real_only": "Cascade-R",
    "wm1_midtrain_only": "Cascade-S",
    "wm1_midtrain_lora45000": "Cascade-SR",
}

TP_COLOR = np.array([60, 200, 60])
FP_COLOR = np.array([210, 60, 60])
FN_COLOR = np.array([60, 100, 220])
BG_COLOR = np.array([235, 235, 235])
GT_COLOR = np.array([40, 40, 40])

GRID_LEFT, GRID_RIGHT = 0.02, 0.98

DEFAULT_EXAMPLES = [
    ("id_s00083_wholepose_full_t03", "direct", "ID, whole pose"),
    ("ood_s00013_wholepose_full_t02", "direct", "OOD, whole pose"),
    ("ood_s00021_perdim_dim14_t00", "direct", "OOD, middle finger only"),
]


def render_raw_example(
    trial_group_id: str,
    approach_style: str,
    gt_manifest_path: str,
    rollout_manifest_paths: List[str],
    camera: str,
    repo_root: Path,
    color_threshold: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Tuple[int, int]]:
    """Uncropped panels (at native camera-view resolution) + the union mask
    used to decide where/how big to crop — cropping is deferred so it can be
    made identical in size across every row (see equalize_and_crop)."""
    gt_manifest = _load_json(resolve_path(gt_manifest_path, repo_root))
    gt_by_trial_group = {r["trial_group_id"]: r for r in gt_manifest["results"]}
    gt_entry = gt_by_trial_group[trial_group_id]

    trials_by_id = {}
    for path in discover_rollout_manifests(rollout_manifest_paths):
        manifest = _load_json(resolve_path(path, repo_root))
        for t in manifest["results"]:
            trials_by_id[t["trial_id"]] = t

    gt_mask_native = load_gt_hand_mask(gt_entry, camera, repo_root)
    panels: Dict[str, np.ndarray] = {}
    all_masks: List[np.ndarray] = []
    cam_idx = CAMERA_VIEW_ORDER.index(camera)

    ref_view_shape = None
    for model in MODEL_ORDER:
        trial_id = f"{trial_group_id}_{approach_style}_{model}"
        trial = trials_by_id[trial_id]
        frame = read_last_frames(resolve_path(trial["wm1_seg_video_path"], repo_root), 1)[0]
        view = split_views(frame, len(CAMERA_VIEW_ORDER))[cam_idx]
        ref_view_shape = view.shape[:2]
        pred_mask = green_hand_mask(view, color_threshold)
        gt_mask = resize_mask_to(gt_mask_native, view.shape[:2])
        overlap = np.tile(BG_COLOR, view.shape[:2] + (1,)).astype(np.uint8)
        overlap[pred_mask & ~gt_mask] = FP_COLOR
        overlap[~pred_mask & gt_mask] = FN_COLOR
        overlap[pred_mask & gt_mask] = TP_COLOR
        union = (pred_mask | gt_mask).sum()
        iou_val = float((pred_mask & gt_mask).sum() / union) if union > 0 else 1.0
        panels[model] = overlap
        panels[f"{model}_iou"] = iou_val
        all_masks.extend([pred_mask, gt_mask])

    gt_panel = np.tile(BG_COLOR, ref_view_shape + (1,)).astype(np.uint8)
    gt_mask_ref = resize_mask_to(gt_mask_native, ref_view_shape)
    gt_panel[gt_mask_ref] = GT_COLOR
    panels["gt"] = gt_panel

    union_mask = np.zeros(ref_view_shape, dtype=bool)
    for m in all_masks:
        union_mask |= m
    return panels, union_mask, ref_view_shape


def _padded_box_size(union_mask: np.ndarray, pad_frac: float = 0.25) -> Tuple[int, int]:
    rows, cols = np.where(union_mask)
    if rows.size == 0:
        return union_mask.shape
    pad_r = max(6, int(pad_frac * (rows.max() - rows.min() + 1)))
    pad_c = max(6, int(pad_frac * (cols.max() - cols.min() + 1)))
    return (rows.max() - rows.min() + 1 + 2 * pad_r, cols.max() - cols.min() + 1 + 2 * pad_c)


def compute_crop_windows(
    rows_meta: List[Tuple[np.ndarray, Tuple[int, int]]],
    aspect: float | None = None,
    pad_frac: float = 0.25,
    center: str = "centroid",
) -> Tuple[List[Tuple[int, int, int, int]], int]:
    """One (r0, r1, c0, c1) window per row, all the SAME (height, width) —
    sized to fit the most demanding row's padded bounding box, centered on
    each row's own mask centroid. rows_meta = [(union_mask, ref_view_shape)].
    If aspect (width/height) is given, the window is grown along whichever
    axis is short so it has that aspect (clamped to the frame). pad_frac is the
    padding around the mask extent; center is "centroid" (mask pixel mean) or
    "bbox" (center of the mask's bounding box).
    Also returns the shared window height (for choosing an upscale factor)."""
    box_h = max(_padded_box_size(u, pad_frac)[0] for u, _ in rows_meta)
    box_w = max(_padded_box_size(u, pad_frac)[1] for u, _ in rows_meta)
    if aspect is not None:
        box_h, box_w = max(box_h, int(np.ceil(box_w / aspect))), max(box_w, int(np.ceil(box_h * aspect)))

    windows: List[Tuple[int, int, int, int]] = []
    for union_mask, (H, W) in rows_meta:
        box_h_c, box_w_c = min(box_h, H), min(box_w, W)
        rows, cols = np.where(union_mask)
        if not rows.size:
            cy, cx = H // 2, W // 2
        elif center == "bbox":
            cy, cx = (rows.min() + rows.max()) // 2, (cols.min() + cols.max()) // 2
        else:
            cy, cx = int(round(rows.mean())), int(round(cols.mean()))
        r0 = min(max(cy - box_h_c // 2, 0), H - box_h_c)
        c0 = min(max(cx - box_w_c // 2, 0), W - box_w_c)
        windows.append((r0, r0 + box_h_c, c0, c0 + box_w_c))
    return windows, box_h


def equalize_and_crop(
    rows_data: List[Tuple[Dict[str, np.ndarray], np.ndarray, Tuple[int, int]]],
    target_h: int = 220,
) -> List[Dict[str, np.ndarray]]:
    """Crop every row to the same-size window (see compute_crop_windows), then
    upscale all rows by the same factor. Every panel in the final figure ends
    up with an identical pixel shape, so all rows render at the same size and
    stay aligned."""
    windows, box_h = compute_crop_windows([(u, shape) for _, u, shape in rows_data])

    all_panels: List[Dict[str, np.ndarray]] = []
    for (panels, _, _), (r0, r1, c0, c1) in zip(rows_data, windows):
        cropped_panels: Dict[str, np.ndarray] = {}
        for key in ["gt"] + MODEL_ORDER:
            cropped_panels[key] = panels[key][r0:r1, c0:c1]
            cropped_panels[f"{key}_iou"] = panels.get(f"{key}_iou")
        all_panels.append(cropped_panels)

    scale = target_h / box_h
    for cropped_panels in all_panels:
        for key in ["gt"] + MODEL_ORDER:
            arr = cropped_panels[key]
            cropped_panels[key] = np.repeat(np.repeat(arr, round(scale), axis=0), round(scale), axis=1)
    return all_panels


def draw_qual_row(
    fig: plt.Figure, gs_row, panels: Dict[str, np.ndarray], show_col_titles: bool, font_scale: float = 1.0
) -> Dict[str, Tuple]:
    """Returns {column: (axes, image)} so callers (e.g. the video script) can
    update the panels in place. font_scale=1 is sized for a 3.45in column."""
    cols = ["gt"] + MODEL_ORDER
    titles = ["GT"] + [MODEL_LABELS[m] for m in MODEL_ORDER]
    handles: Dict[str, Tuple] = {}
    for i, (col, title) in enumerate(zip(cols, titles)):
        ax = fig.add_subplot(gs_row[i])
        im = ax.imshow(panels[col])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.5 * font_scale); spine.set_color("#888888")
        if show_col_titles:
            ax.set_title(title, fontsize=9.5 * font_scale, pad=3 * font_scale)
        if col != "gt":
            ax.set_xlabel(f"IoU={panels[col + '_iou']:.2f}", fontsize=9 * font_scale, labelpad=2 * font_scale)
        handles[col] = (ax, im)
    return handles


def create_figure(
    n_rows: int,
    row_height: float = 0.92,
    hspace: float = 0.08,
    fig_height: float | None = None,
    width: float = 3.45,
    top: float | None = None,
    bottom: float = 0.16,
    font_scale: float = 1.0,
    font_family: str = "sans-serif",
    facecolor: str = "white",
    text_color: str = "black",
):
    """Figure + outer row gridspec (default: single-column 3.45in); the legend is
    added separately by add_legend(). font_scale=1 is sized for a 3.45in column —
    scale it up with width for slides. facecolor/text_color/font_family restyle
    the whole figure (e.g. white-on-black Arial for slides)."""
    plt.rcParams.update({
        "font.family": font_family, "font.size": 9 * font_scale,
        "text.color": text_color, "axes.labelcolor": text_color, "axes.titlecolor": text_color,
        "legend.labelcolor": text_color, "axes.facecolor": facecolor, "figure.facecolor": facecolor,
    })
    fig = plt.figure(
        figsize=(width, fig_height if fig_height is not None else row_height * n_rows + 0.48),
        facecolor=facecolor,
    )
    grid_kwargs = {} if top is None else {"top": top}
    gs = fig.add_gridspec(n_rows, 1, hspace=hspace, bottom=bottom, left=GRID_LEFT, right=GRID_RIGHT, **grid_kwargs)
    return fig, gs


def add_legend(fig: plt.Figure, font_scale: float = 1.0) -> None:
    """Anchored+expanded to exactly [GRID_LEFT, GRID_RIGHT] so the legend's
    width matches the mask grid above it, not the full figure canvas."""
    legend_handles = [
        Patch(facecolor=TP_COLOR / 255, edgecolor="none", label="Correct (TP)"),
        Patch(facecolor=FP_COLOR / 255, edgecolor="none", label="False positive"),
        Patch(facecolor=FN_COLOR / 255, edgecolor="none", label="Missed (FN)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(GRID_LEFT, 0.0, GRID_RIGHT - GRID_LEFT, 0.001), mode="expand",
        ncol=3, frameon=False, fontsize=8 * font_scale, handlelength=1.0, handleheight=1.0,
        columnspacing=1.0, borderaxespad=0.1,
    )


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
    p.add_argument("--output", required=True, help="Output path stem (no extension) — saves .pdf and .png")
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    if args.examples:
        examples: List[Tuple[str, str, str]] = []
        for e in args.examples:
            trial_group_id, approach_style, label = e.split(":", 2)
            examples.append((trial_group_id, approach_style, label))
    else:
        examples = DEFAULT_EXAMPLES

    rows_data = [
        render_raw_example(
            trial_group_id, approach_style, args.gt_manifest, args.rollout_manifests,
            args.camera, repo_root, args.color_threshold,
        )
        for trial_group_id, approach_style, _ in examples
    ]
    all_panels = equalize_and_crop(rows_data)

    fig, gs = create_figure(len(examples))
    for row_i, panels in enumerate(all_panels):
        gs_row = gs[row_i].subgridspec(1, 4, wspace=0.06)
        draw_qual_row(fig, gs_row, panels, show_col_titles=(row_i == 0))
    add_legend(fig)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    print(f"[saved] {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
