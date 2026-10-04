"""Pairwise A/B composite builder for the controllability-eval pipeline (v2).

Unlike build_controllability_composites.py (which scores one generated rollout
against GT independently), this builds head-to-head comparison images: for
every (trial_group_id, approach_style) pair, it forms a 5-candidate pool --
the 4 model variants' generated final frames, plus a synthetic "gt_self"
candidate that reuses the GT's own render (the same trick
build_gt_self_check_composites uses, folded directly into the comparison set
here instead of a separate calibration pass) -- and builds all C(5,2)=10
pairwise composites among them.

Each composite is 2 camera rows x 4 panels:
    [GT reference: RGB | GT reference: segmentation mask | Model A | Model B]

"Model A"/"Model B" are anonymous, per-comparison labels -- which real
candidate identity lands in which slot is randomized per comparison_id (see
assign_left_right), so raters/the LLM judge never see real model-variant
names, and "gt_self" isn't reliably found in the same slot. The reference
panels are fixed/always GT; only the two candidate panels rotate.

This is a validity test as much as an eval: since one candidate in 4 of the
10 pairs per target is a literal, pixel-identical copy of the reference RGB
(gt_self), a sound comparison methodology should pick it as the better match
essentially every time. See llm_judge_controllability_pairwise.py and
gradio_controllability_pairwise.py for the two judges, and
compute_controllability_pairwise_stats.py for how that signal gets reported.

Example:
    python scripts/build_controllability_pairwise_composites.py \\
        --rollout_manifest inference_output/.../rollout_manifest_baseline_midtrain_lora.json \\
        --rollout_manifest inference_output/.../rollout_manifest_wm1_midtrain_only.json \\
        --rollout_manifest inference_output/.../rollout_manifest_wm1_midtrain_lora45000.json \\
        --rollout_manifest inference_output/.../rollout_manifest_wm1_real_only.json \\
        --gt_manifest inference_output/full_run_gt_renders/gt_manifest.json \\
        --output_dir inference_output/full_run_results/pairwise_composites
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from scripts.build_controllability_composites import (  # noqa: E402
    CAMERA_VIEW_ORDER,
    PANEL_GAP,
    BACKGROUND_COLOR,
    extract_final_frame_per_view,
    load_gt_frame,
    _resize_nearest,
    mask_panel_image,
    _title_strip,
    _panel_header_strip,
)

GT_SELF_IDENTITY = "gt_self"
# Kept short enough to fit a quarter-width panel header without overlapping its neighbor (measured
# against a real built composite: ~240px/panel, ~228px usable after margins at the default PIL
# font -- v1's longer "GT segmentation mask (hand=green, arm=blue)..." label only fit because it
# had a third of the width instead of a quarter).
PAIRWISE_PANEL_LABELS = (
    "GT reference: RGB",
    "GT reference: mask (green=hand, blue=arm)",
    "Model A",
    "Model B",
)


def _load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_candidate_pairs(candidates: List[str]) -> List[Tuple[str, str]]:
    """All C(n,2) unordered pairs, fixed canonical order (itertools.combinations),
    with no family/variant filter: every pair here is a wanted comparison.
    """
    return list(itertools.combinations(candidates, 2))


def assign_left_right(identity_a: str, identity_b: str, seed: int, comparison_id: str) -> Tuple[str, str]:
    """Randomized (label "A", label "B") assignment, deterministic per (seed, comparison_id).
    The same comparison_id always renders the same way (so the LLM run and every human build_subset call
    against a given seed see an identical image), while different comparison_ids get independent
    coin flips, so a given identity (especially "gt_self") isn't reliably found in the same slot.
    """
    rng = random.Random(f"{seed}:{comparison_id}")
    if rng.random() < 0.5:
        return identity_a, identity_b
    return identity_b, identity_a


def build_pairwise_composite_image(
    gt_by_view: Dict[str, Dict[str, np.ndarray]],
    candidate_a_by_view: Dict[str, np.ndarray],
    candidate_b_by_view: Dict[str, np.ndarray],
    base_shape_by_view: Dict[str, Tuple[int, int]],
    trial_meta: Dict,
) -> np.ndarray:
    """2 camera views (rows) x 4 panels (GT RGB | GT segmentation mask | Model A | Model B).

    Every panel is resized (nearest-neighbor) to base_shape_by_view[camera] -- derived once per
    trial group from a real rollout video's resolution (see build_all_pairwise_composites), since
    either candidate slot (A or B) may independently be "gt_self" (native GT-render resolution,
    not the video's) depending on the per-comparison shuffle.
    """
    rows = []
    for camera in CAMERA_VIEW_ORDER:
        h, w = base_shape_by_view[camera]
        gt_rgb = _resize_nearest(gt_by_view[camera]["rgb"], (h, w))
        mask_panel = mask_panel_image((h, w, 3), gt_by_view[camera]["hand_mask"], gt_by_view[camera].get("arm_mask"))
        cand_a = _resize_nearest(candidate_a_by_view[camera], (h, w))
        cand_b = _resize_nearest(candidate_b_by_view[camera], (h, w))
        gap = np.full((h, PANEL_GAP, 3), BACKGROUND_COLOR, dtype=np.uint8)
        row = np.concatenate([gt_rgb, gap, mask_panel, gap, cand_a, gap, cand_b], axis=1)
        rows.append(row)

    row_gap = np.full((PANEL_GAP, rows[0].shape[1], 3), BACKGROUND_COLOR, dtype=np.uint8)
    grid = np.concatenate([rows[0], row_gap, rows[1]], axis=0)

    scope = trial_meta.get("sampling_scope", "?")
    component = trial_meta.get("component_label") or "all dims"
    converged = trial_meta.get("gt_converged")
    conv_str = "" if converged is None else (" | GT converged" if converged else " | GT NOT converged")
    # Deliberately excludes any candidate identity from the *rendered pixels* -- same blinding
    # rule as build_controllability_composites.py's title strip (a vision model can read text
    # baked into an image just as well as a human can). trial_group_id is safe: it identifies
    # the target, not which model produced either candidate.
    title_text = (
        f"{trial_meta.get('trial_group_id', '?')} | scope={scope} ({component}) | "
        f"approach={trial_meta.get('approach_style', '?')}"
        f"{conv_str} | rows: {' / '.join(CAMERA_VIEW_ORDER)}"
    )
    title = _title_strip(grid.shape[1], title_text)
    panel_header = _panel_header_strip(grid.shape[1], PANEL_GAP, PAIRWISE_PANEL_LABELS)
    return np.concatenate([title, panel_header, grid], axis=0)


def build_all_pairwise_composites(
    rollout_manifest_paths: List[str],
    gt_manifest_path: str,
    output_dir: str,
    seed: int = 0,
    num_views: int = 2,
) -> str:
    gt_manifest = _load_manifest(gt_manifest_path)
    gt_by_trial_group: Dict[str, Dict] = {r["trial_group_id"]: r for r in gt_manifest["results"]}

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Group rollout trials by (trial_group_id, approach_style) -> {model_variant: trial}.
    grouped: Dict[Tuple[str, str], Dict[str, Dict]] = {}
    all_variants: set = set()
    for rollout_manifest_path in rollout_manifest_paths:
        rollout_manifest = _load_manifest(rollout_manifest_path)
        for trial in rollout_manifest["results"]:
            key = (trial["trial_group_id"], trial["approach_style"])
            grouped.setdefault(key, {})[trial["model_variant"]] = trial
            all_variants.add(trial["model_variant"])

    variants_sorted = sorted(all_variants)
    candidates = variants_sorted + [GT_SELF_IDENTITY]
    candidate_pairs = build_candidate_pairs(candidates)
    print(f"[INFO]: {len(variants_sorted)} model variant(s) found: {variants_sorted}. "
          f"{len(candidate_pairs)} pairwise comparisons per (trial_group_id, approach_style).")

    pairwise_manifest = {
        "rollout_manifest_paths": rollout_manifest_paths,
        "gt_manifest_path": gt_manifest_path,
        "seed": seed,
        "results": [],
    }

    n_missing_gt = 0
    n_missing_variant = 0
    n_built = 0

    for (trial_group_id, approach_style), variant_trials in sorted(grouped.items()):
        gt_entry = gt_by_trial_group.get(trial_group_id)
        if gt_entry is None:
            n_missing_gt += 1
            print(f"[WARN]: No GT render for trial_group_id={trial_group_id}; "
                  f"skipping all pairs for approach_style={approach_style}.")
            continue

        missing_variants = [v for v in variants_sorted if v not in variant_trials]
        if missing_variants:
            n_missing_variant += 1
            print(f"[WARN]: trial_group_id={trial_group_id} approach_style={approach_style} "
                  f"missing rollout(s) for {missing_variants}; skipping.")
            continue

        hand_ids_by_camera = gt_entry.get("hand_instance_ids")
        if not hand_ids_by_camera:
            raise KeyError(
                f"gt_manifest.json entry for trial_group_id={trial_group_id} has no 'hand_instance_ids' -- "
                "this GT render predates the hand-label fix (see render_controllability_targets.py); "
                "re-render with --overwrite_existing."
            )
        # Unlike hand_instance_ids, missing/empty arm_instance_ids is not an error -- see
        # build_controllability_composites.py's mask_panel_image.
        arm_ids_by_camera = gt_entry.get("arm_instance_ids") or {}

        gt_by_view = {
            cam: load_gt_frame(gt_entry["render_paths"], hand_ids_by_camera.get(cam, []), cam, arm_ids_by_camera.get(cam, []))
            for cam in CAMERA_VIEW_ORDER
        }

        # Extract each real variant's generated final frame once per group -- reused across
        # every pairwise comparison it appears in within this group (4 pairs each).
        frames_by_identity: Dict[str, Dict[str, np.ndarray]] = {}
        for variant in variants_sorted:
            frames_by_identity[variant] = extract_final_frame_per_view(variant_trials[variant]["video_path"], num_views)
        frames_by_identity[GT_SELF_IDENTITY] = {cam: gt_by_view[cam]["rgb"] for cam in CAMERA_VIEW_ORDER}

        # Base panel resolution = a real rollout video's resolution (never gt_self's, which is
        # the GT render's native resolution and may differ) -- see build_pairwise_composite_image.
        base_shape_by_view = {
            cam: frames_by_identity[variants_sorted[0]][cam].shape[:2] for cam in CAMERA_VIEW_ORDER
        }

        # split/sampling_scope/component/component_label are properties of the target, shared by
        # every variant's trial for this group -- source them from any one (here, the first
        # variant in sorted order, for determinism), matching how build_all_composites sources
        # these fields from its one rollout trial per iteration.
        anchor_trial = variant_trials[variants_sorted[0]]
        trial_meta_base = {
            "trial_group_id": trial_group_id,
            "split": anchor_trial["split"],
            "sampling_scope": anchor_trial["sampling_scope"],
            "component": anchor_trial["component"],
            "component_label": anchor_trial["component_label"],
            "approach_style": approach_style,
            "gt_converged": gt_entry["diagnostics"]["converged"],
            "gt_saturated_dims": gt_entry["diagnostics"]["saturated_dims"],
        }

        for pair_idx, (identity_a, identity_b) in enumerate(candidate_pairs):
            comparison_id = f"{trial_group_id}_{approach_style}_pair{pair_idx:02d}"
            left_identity, right_identity = assign_left_right(identity_a, identity_b, seed, comparison_id)

            composite = build_pairwise_composite_image(
                gt_by_view,
                frames_by_identity[left_identity],
                frames_by_identity[right_identity],
                base_shape_by_view,
                trial_meta_base,
            )

            composite_path = output_dir_path / f"{comparison_id}.png"
            Image.fromarray(composite).save(composite_path)

            pairwise_manifest["results"].append(
                {
                    "comparison_id": comparison_id,
                    "trial_group_id": trial_group_id,
                    "split": trial_meta_base["split"],
                    "sampling_scope": trial_meta_base["sampling_scope"],
                    "component": trial_meta_base["component"],
                    "component_label": trial_meta_base["component_label"],
                    "approach_style": approach_style,
                    "left_identity": left_identity,
                    "right_identity": right_identity,
                    "composite_path": str(composite_path),
                    "source_gt_render_paths": gt_entry["render_paths"],
                    "gt_converged": trial_meta_base["gt_converged"],
                    "gt_saturated_dims": trial_meta_base["gt_saturated_dims"],
                }
            )
            n_built += 1

    pairwise_manifest_path = output_dir_path / "pairwise_manifest.json"
    with open(pairwise_manifest_path, "w", encoding="utf-8") as f:
        json.dump(pairwise_manifest, f, indent=2)

    print(f"[INFO]: Built {n_built} pairwise composites "
          f"({n_missing_gt} group(s) skipped for missing GT, {n_missing_variant} skipped for missing variant rollout). "
          f"{pairwise_manifest_path}")
    return str(pairwise_manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rollout_manifest", action="append", default=[], dest="rollout_manifests", required=True,
        help="Path to a rollout_manifest_<variant>.json. Pass once per variant/split-dir to combine "
        "(a full run needs 4 variants x {id,ood} = 8 files).",
    )
    parser.add_argument("--gt_manifest", type=str, required=True, help="Path to gt_manifest.json.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save comparison PNGs and pairwise_manifest.json.")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the per-comparison left/right (Model A/B) shuffle. Must stay fixed across "
        "the LLM run and every human build_subset run against this --output_dir.",
    )
    parser.add_argument("--num_views", type=int, default=2)
    args = parser.parse_args()

    build_all_pairwise_composites(args.rollout_manifests, args.gt_manifest, args.output_dir, args.seed, args.num_views)


if __name__ == "__main__":
    main()
