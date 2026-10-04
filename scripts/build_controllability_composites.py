"""Composite/overlay builder for the controllability-eval pipeline.

Joins a rollout_manifest_{baseline,wm1wm2}.json (WM-generated rollouts) with a
gt_manifest.json (faive_lab ground-truth renders, from
faive_lab/scripts/wm_evaluation/render_controllability_targets.py) on
trial_group_id, and for every trial builds one composite image: per camera
view, a 3-panel row [Generated RGB | GT RGB | GT segmentation mask]. This is
what both the LLM judge (Phase 4) and the human rating game (Phase 5) look at
to score "how close did it get."

Column 3 is a standalone segmentation mask (hand=green, arm=blue, everything
else black) -- NOT an overlay blended onto the generated photo, which is what
this used to be (a semi-transparent fill + outline on top of column 1). That
overlay design was replaced after an A/B test against a genuinely-perfect-match
sanity check (feeding the GT's own render back in as "generated", so columns 1
and 2 are byte-identical -- see build_gt_self_check_composites): the LLM judge
scored that non-existent mismatch a mean 0.74 (range 0.60-0.85) with the old
overlay design, vs. 0.85 (0.82-0.88, much tighter) with a standalone mask
panel, and an outline-only variant (no fill) landed in between at 0.79. The
mask panel's better calibration held up on genuinely-bad real rollouts too
(scores were within 0.03 of the old overlay design's on 6 real failed
baseline-model trials), so it isn't just uniformly inflating scores -- it's a
strictly better-calibrated design. Working hypothesis: alpha-blending a color
wash over the photo obscures exactly the fine finger-edge detail a vision
judge needs to confirm alignment, so it reads illusory offsets into a frame
that's actually identical; a clean silhouette panel removes that confound.
See mask_panel_image's docstring. None of the three designs reached a literal
1.0 even on the byte-identical case, though -- some conservative bias seems to
be a property of the judge model itself, not fixable by composite design
alone. compute_controllability_stats.py's --gt_self_scores support addresses
this by scoring rollouts *relative to* the judge's own per-scene ceiling
rather than against an unreachable literal 1.0 (see build_gt_self_check_composites).

The mask is drawn from the *hand-only* ids in gt_manifest.json's
`hand_instance_ids` field (per-camera lists produced by
render_controllability_targets.py via a prim-path substring match, default
"orcahand"), NOT from "any non-background pixel" (id != 0). That naive test
was tried first and is wrong: Isaac Sim's instance segmentation gives
background/environment geometry (walls, floor, arena panels) their own real,
non-zero instance ids too -- one observed case had id=0 "BACKGROUND" and a
*second*, much larger region at id=1 labeled "UNLABELLED" covering nearly the
whole frame (the arena), which a naive id!=0 contour traced instead of the
hand. Always source the mask from `hand_instance_ids`, never from `seg_ids != 0`.

The Franka arm also gets marked, in a different color (blue), drawn from
gt_manifest.json's `arm_instance_ids` field. This requires two things on the
faive_lab side: (1) ORCA_FRANKA_CFG carrying a root-level ("class", "arm")
semantic tag, and (2) the GT renderer (render_controllability_targets.py)
re-showing the arm's visual meshes, which the shared WM-training-data env cfg
hides by default via its `hide_franka_arm_visuals` prestartup event (see
orca_synthetic_data_gen_mimic_env_cfg.py) -- without both, the arm has no
separate instance id / isn't rendered at all, and `arm_instance_ids` is just
empty/missing (not an error -- older gt_manifest.json files without this field
still work fine, they just render without the arm region). hand_mask and
arm_mask never overlap in practice (the hand's joint bodies each carry their
own more specific semantic tag that takes precedence over the arm's
root-level one). This addresses a real prior failure mode (see
llm_judge_controllability.py's prompt language, still kept as a second line of
defense): the LLM judge mistaking the Franka arm for the hand when the arm
wasn't marked at all.

Video view order: video frames are [T, H, num_views*W, C] with views
concatenated along width in dataset order (see
faive_lab/scripts/data_collection/real_convert_to_lerobot.py's
`camera_keys = [("oakd_side_view", "0"), ("oakd_wrist_view", "1")]`) --
view 0 = side_camera_one, view 1 = wrist_camera. This must match
CAMERA_VIEW_ORDER below or panels will be paired with the wrong GT camera.

Example:
    python scripts/build_controllability_composites.py \\
        --rollout_manifest inference_output/.../rollout_manifest_baseline.json \\
        --rollout_manifest inference_output/.../rollout_manifest_wm1wm2.json \\
        --gt_manifest /path/to/gt_renders/gt_manifest.json \\
        --output_dir inference_output/.../composites

    # GT-self-check composites, for compute_controllability_stats.py --gt_self_scores:
    python scripts/build_controllability_composites.py \\
        --gt_self_check \\
        --gt_manifest /path/to/gt_renders/gt_manifest.json \\
        --output_dir inference_output/.../gt_self_check_composites
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERA_VIEW_ORDER = ["side_camera_one", "wrist_camera"]
PANEL_LABELS = (
    "1. Generated end state",
    "2. Target state (simulation)",
    "3. GT segmentation mask (hand=green, arm=blue)",
)
HAND_COLOR_RGB = (0, 255, 0)
# Matches faive_lab's SEMANTIC_SEGMENTATION_MAPPING["class:arm"] = (0, 0, 255, 255) RGBA.
ARM_COLOR_RGB = (0, 0, 255)
TITLE_STRIP_HEIGHT = 28
PANEL_GAP = 4
BACKGROUND_COLOR = (30, 30, 30)


def extract_final_frame_per_view(video_path: str, num_views: int) -> Dict[str, np.ndarray]:
    """Last frame of a [T, H, num_views*W, C] video, split into per-camera crops."""
    video = mediapy.read_video(video_path)
    last_frame = np.asarray(video[-1])[..., :3]
    total_width = last_frame.shape[1]
    if total_width % num_views != 0:
        raise ValueError(f"Video width {total_width} not divisible by num_views={num_views}: {video_path}")
    view_width = total_width // num_views
    if num_views != len(CAMERA_VIEW_ORDER):
        raise ValueError(f"Expected num_views={len(CAMERA_VIEW_ORDER)} (CAMERA_VIEW_ORDER), got {num_views}.")
    return {
        CAMERA_VIEW_ORDER[v]: last_frame[:, v * view_width : (v + 1) * view_width]
        for v in range(num_views)
    }


def load_gt_frame(
    render_paths: Dict[str, str], hand_ids: List[int], camera: str, arm_ids: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    rgb = np.array(Image.open(render_paths[f"{camera}_rgb"]).convert("RGB"))
    seg_ids = np.load(render_paths[f"{camera}_seg_raw_ids"])
    hand_mask = np.isin(seg_ids, np.array(hand_ids, dtype=seg_ids.dtype)) if hand_ids else np.zeros_like(seg_ids, dtype=bool)
    arm_mask = np.isin(seg_ids, np.array(arm_ids, dtype=seg_ids.dtype)) if arm_ids else np.zeros_like(seg_ids, dtype=bool)
    return {"rgb": rgb, "hand_mask": hand_mask, "arm_mask": arm_mask}


def _resize_nearest(arr: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Nearest-neighbor resize that preserves arr's dtype/value range.

    Uses cv2 (not PIL) specifically so this is safe for the segmentation
    raw-id arrays (int32, values are arbitrary instance ids that can exceed
    255) as well as uint8 RGB -- a uint8 cast before resizing would silently
    truncate/collide large instance ids.
    """
    target_h, target_w = target_hw
    if arr.shape[:2] == (target_h, target_w):
        return arr
    return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def mask_panel_image(base_shape: tuple, hand_mask: np.ndarray, arm_mask: Optional[np.ndarray]) -> np.ndarray:
    """Standalone GT segmentation mask, resized to base_shape: hand=green, arm=blue,
    everything else black -- no RGB detail, not overlaid on any photo.

    See the module docstring for the A/B test this design won (mean judge score
    0.85 on a genuinely-perfect-match sanity check, vs. 0.74 for the previous
    alpha-blended overlay design, without losing any discriminative power on real
    failures). hand_mask/arm_mask never overlap in practice (arm_instance_ids
    excludes the hand's more specific ids -- see faive_lab's
    INSTANCE_SEGMENTATION_MAPPING), so draw order between them doesn't matter.
    """
    h, w = base_shape[:2]
    hand_resized = _resize_nearest(hand_mask.astype(np.uint8), (h, w)).astype(bool)
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    if arm_mask is not None:
        arm_resized = _resize_nearest(arm_mask.astype(np.uint8), (h, w)).astype(bool)
        panel[arm_resized] = ARM_COLOR_RGB
    panel[hand_resized] = HAND_COLOR_RGB
    return panel


def _title_strip(width: int, text: str) -> np.ndarray:
    strip = Image.new("RGB", (width, TITLE_STRIP_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((6, 6), text, fill=(255, 255, 255), font=font)
    return np.array(strip)


def _panel_header_strip(total_width: int, panel_gap: int, labels: tuple) -> np.ndarray:
    """One label per panel, positioned directly above its column so it's
    unambiguous which of the three images each label names -- a single
    combined caption (e.g. in the title strip) leaves that mapping implicit.
    """
    n = len(labels)
    panel_width = (total_width - panel_gap * (n - 1)) // n
    strip = Image.new("RGB", (total_width, TITLE_STRIP_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, label in enumerate(labels):
        x = i * (panel_width + panel_gap)
        draw.text((x + 6, 6), label, fill=(255, 255, 255), font=font)
    return np.array(strip)


def build_composite_image(
    generated_by_view: Dict[str, np.ndarray],
    gt_by_view: Dict[str, Dict[str, np.ndarray]],
    trial_meta: Dict,
) -> np.ndarray:
    """2 camera views (rows) x 3 panels (Generated | GT | GT segmentation mask)."""
    rows = []
    for camera in CAMERA_VIEW_ORDER:
        gen = generated_by_view[camera]
        gt_rgb = _resize_nearest(gt_by_view[camera]["rgb"], gen.shape[:2])
        mask_panel = mask_panel_image(gen.shape, gt_by_view[camera]["hand_mask"], gt_by_view[camera].get("arm_mask"))
        gap = np.full((gen.shape[0], PANEL_GAP, 3), BACKGROUND_COLOR, dtype=np.uint8)
        row = np.concatenate([gen, gap, gt_rgb, gap, mask_panel], axis=1)
        rows.append(row)

    row_gap = np.full((PANEL_GAP, rows[0].shape[1], 3), BACKGROUND_COLOR, dtype=np.uint8)
    grid = np.concatenate([rows[0], row_gap, rows[1]], axis=0)

    scope = trial_meta.get("sampling_scope", "?")
    component = trial_meta.get("component_label") or "all dims"
    converged = trial_meta.get("gt_converged")
    conv_str = "" if converged is None else (" | GT converged" if converged else " | GT NOT converged")
    # Deliberately excludes model_variant and trial_id (whose suffix names the variant,
    # e.g. "..._wm1wm2") from the *rendered pixels* -- both the LLM judge (Phase 4) and
    # the human rating game (Phase 5) are meant to be blind to which model produced a
    # trial, and a vision model can read text baked into an image just as well as a
    # human can. trial_group_id is safe: it identifies the target, not the model.
    title_text = (
        f"{trial_meta.get('trial_group_id', '?')} | scope={scope} ({component}) | "
        f"approach={trial_meta.get('approach_style', '?')}"
        f"{conv_str} | rows: {' / '.join(CAMERA_VIEW_ORDER)}"
    )
    title = _title_strip(grid.shape[1], title_text)
    panel_header = _panel_header_strip(grid.shape[1], PANEL_GAP, PANEL_LABELS)
    return np.concatenate([title, panel_header, grid], axis=0)


def _load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_all_composites(
    rollout_manifest_paths: List[str],
    gt_manifest_path: str,
    output_dir: str,
    num_views: int = 2,
) -> str:
    gt_manifest = _load_manifest(gt_manifest_path)
    gt_by_trial_group: Dict[str, Dict] = {r["trial_group_id"]: r for r in gt_manifest["results"]}

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    composite_manifest = {
        "rollout_manifest_paths": rollout_manifest_paths,
        "gt_manifest_path": gt_manifest_path,
        "results": [],
    }

    n_missing_gt = 0
    n_built = 0
    for rollout_manifest_path in rollout_manifest_paths:
        rollout_manifest = _load_manifest(rollout_manifest_path)
        for trial in rollout_manifest["results"]:
            trial_group_id = trial["trial_group_id"]
            gt_entry = gt_by_trial_group.get(trial_group_id)
            if gt_entry is None:
                n_missing_gt += 1
                print(f"[WARN]: No GT render for trial_group_id={trial_group_id} (trial_id={trial['trial_id']}); skipping.")
                continue

            hand_ids_by_camera = gt_entry.get("hand_instance_ids")
            if not hand_ids_by_camera:
                raise KeyError(
                    f"gt_manifest.json entry for trial_group_id={trial_group_id} has no 'hand_instance_ids' -- "
                    "this GT render predates the hand-label fix (see render_controllability_targets.py); "
                    "re-render with --overwrite_existing."
                )
            # Unlike hand_instance_ids, missing/empty is not an error here -- older
            # gt_manifest.json files predate the arm tag and simply render without
            # the arm region (see mask_panel_image).
            arm_ids_by_camera = gt_entry.get("arm_instance_ids") or {}

            generated_by_view = extract_final_frame_per_view(trial["video_path"], num_views)
            gt_by_view = {
                cam: load_gt_frame(
                    gt_entry["render_paths"], hand_ids_by_camera.get(cam, []), cam, arm_ids_by_camera.get(cam, [])
                )
                for cam in CAMERA_VIEW_ORDER
            }

            trial_meta = dict(trial)
            trial_meta["gt_converged"] = gt_entry["diagnostics"]["converged"]
            composite = build_composite_image(generated_by_view, gt_by_view, trial_meta)

            composite_path = output_dir_path / f"{trial['trial_id']}.png"
            Image.fromarray(composite).save(composite_path)

            composite_manifest["results"].append(
                {
                    "trial_id": trial["trial_id"],
                    "trial_group_id": trial_group_id,
                    "split": trial["split"],
                    "sampling_scope": trial["sampling_scope"],
                    "component": trial["component"],
                    "component_label": trial["component_label"],
                    "approach_style": trial["approach_style"],
                    "model_variant": trial["model_variant"],
                    "composite_path": str(composite_path),
                    "source_video_path": trial["video_path"],
                    "source_gt_render_paths": gt_entry["render_paths"],
                    "gt_converged": gt_entry["diagnostics"]["converged"],
                    "gt_saturated_dims": gt_entry["diagnostics"]["saturated_dims"],
                }
            )
            n_built += 1

    composite_manifest_path = output_dir_path / "composite_manifest.json"
    with open(composite_manifest_path, "w", encoding="utf-8") as f:
        json.dump(composite_manifest, f, indent=2)

    print(f"[INFO]: Built {n_built} composites ({n_missing_gt} skipped for missing GT). {composite_manifest_path}")
    return str(composite_manifest_path)


def build_gt_self_check_composites(gt_manifest_path: str, output_dir: str) -> str:
    """One composite per trial_group_id in gt_manifest.json, with the GT's own render
    fed back in as the "generated" panel too -- so columns 1 and 2 are byte-identical,
    a calibration check for the LLM judge's own achievable ceiling per scene (see the
    module docstring's A/B test). Feed the resulting composite_manifest.json through
    llm_judge_controllability.py to get a gt_self_scores.jsonl for
    compute_controllability_stats.py's --gt_self_scores.

    trial_id is set to trial_group_id (not e.g. an "..._GT_SELF" suffix) so
    compute_controllability_stats.py can join scored rows straight onto trial
    metadata's trial_group_id column without extra bookkeeping -- there's exactly one
    of these per trial_group_id, reused across every model_variant/approach_style
    that shares it, same as gt_manifest.json itself.

    sampling_scope/component_label are taken from the real target (gt_manifest.json
    carries them), and approach_style is set to the ordinary value "direct" (not
    something like "gt_self_check") deliberately: this composite's title text and
    metadata are shown to the judge like any other trial, so the measured ceiling
    reflects the judge's normal behavior, not a judge that's been tipped off it's
    looking at a calibration check.
    """
    gt_manifest = _load_manifest(gt_manifest_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    composite_manifest = {"gt_manifest_path": gt_manifest_path, "results": []}
    for gt_entry in gt_manifest["results"]:
        trial_group_id = gt_entry["trial_group_id"]
        hand_ids_by_camera = gt_entry.get("hand_instance_ids") or {}
        arm_ids_by_camera = gt_entry.get("arm_instance_ids") or {}
        gt_by_view = {
            cam: load_gt_frame(
                gt_entry["render_paths"], hand_ids_by_camera.get(cam, []), cam, arm_ids_by_camera.get(cam, [])
            )
            for cam in CAMERA_VIEW_ORDER
        }
        generated_by_view = {cam: gt_by_view[cam]["rgb"] for cam in CAMERA_VIEW_ORDER}

        trial_meta = {
            "trial_group_id": trial_group_id,
            "sampling_scope": gt_entry.get("sampling_scope", "whole_pose"),
            "component_label": gt_entry.get("component_label"),
            "approach_style": "direct",
            "gt_converged": gt_entry["diagnostics"]["converged"],
        }
        composite = build_composite_image(generated_by_view, gt_by_view, trial_meta)

        composite_path = output_dir_path / f"{trial_group_id}.png"
        Image.fromarray(composite).save(composite_path)

        composite_manifest["results"].append(
            {
                "trial_id": trial_group_id,
                "trial_group_id": trial_group_id,
                "sampling_scope": trial_meta["sampling_scope"],
                "component_label": trial_meta["component_label"],
                "approach_style": trial_meta["approach_style"],
                "composite_path": str(composite_path),
                "source_gt_render_paths": gt_entry["render_paths"],
                "gt_converged": gt_entry["diagnostics"]["converged"],
            }
        )

    composite_manifest_path = output_dir_path / "composite_manifest.json"
    with open(composite_manifest_path, "w", encoding="utf-8") as f:
        json.dump(composite_manifest, f, indent=2)

    print(f"[INFO]: Built {len(composite_manifest['results'])} GT-self-check composites. {composite_manifest_path}")
    return str(composite_manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rollout_manifest", action="append", default=[], dest="rollout_manifests",
        help="Path to a rollout_manifest_{baseline,wm1wm2}.json. Pass multiple times to combine variants. "
        "Required unless --gt_self_check is set.",
    )
    parser.add_argument("--gt_manifest", type=str, required=True, help="Path to gt_manifest.json.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save composite PNGs and composite_manifest.json.")
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument(
        "--gt_self_check", action="store_true", default=False,
        help="Build one composite per trial_group_id in --gt_manifest instead of joining against "
        "--rollout_manifest, using the GT's own render as the 'generated' panel too -- see "
        "build_gt_self_check_composites. Ignores --rollout_manifest/--num_views.",
    )
    args = parser.parse_args()

    if args.gt_self_check:
        build_gt_self_check_composites(args.gt_manifest, args.output_dir)
    else:
        if not args.rollout_manifests:
            parser.error("--rollout_manifest is required unless --gt_self_check is set.")
        build_all_composites(args.rollout_manifests, args.gt_manifest, args.output_dir, args.num_views)
