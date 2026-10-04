"""Incremental pairwise composite builder for adding ONE new candidate identity
against a FIXED, EXPLICIT set of opponents -- without touching the existing
pairwise_manifest.json or reassigning any existing comparison_id.

Why this script exists instead of re-running build_controllability_pairwise_composites.py
over the full candidate set: that script derives pair ordering from
sorted(all_variants) + ["gt_self"] fed through itertools.combinations. Adding a
new variant and recomputing over the full set reshuffles the pairNN suffix of
EVERY existing pair (confirmed by hand-tracing combinations()'s output order
for n vs n+1 -- inserting one element anywhere, even at the end, shifts nearly
every subsequent pair's index), silently desyncing the already-collected LLM
votes in llm_comparisons.jsonl. This script instead enumerates only ONE new
pairing at a time (new_identity vs one explicit opponent), assigns it a
caller-supplied pairNN index that continues past the existing pool (pair10+),
and appends into a SEPARATE addendum manifest file. The original
pairwise_manifest.json is never opened for writing.

Run once per new opponent (up to 3 for this study's targeted ablation). Each
run appends to the same --output_manifest, skipping any
comparison_id already present (safe to re-run).

Usage:
    python scripts/build_controllability_pairwise_composites_addendum.py \\
        --new_identity baseline_real_only \\
        --new_identity_rollout_manifest inference_output/.../rollout_manifest_baseline_real_only.json \\
        --new_identity_rollout_manifest inference_output/.../rollout_manifest_baseline_real_only.json \\
        --opponent_identity wm1_midtrain_lora45000 \\
        --opponent_rollout_manifest inference_output/.../rollout_manifest_wm1_midtrain_lora45000.json \\
        --opponent_rollout_manifest inference_output/.../rollout_manifest_wm1_midtrain_lora45000.json \\
        --gt_manifest inference_output/full_run_gt_renders/gt_manifest.json \\
        --output_dir inference_output/full_run_results/pairwise_composites \\
        --output_manifest inference_output/full_run_results/pairwise_composites/pairwise_manifest_baseline_real_only_addendum.json \\
        --pair_index 10 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from scripts.build_controllability_composites import (  # noqa: E402
    CAMERA_VIEW_ORDER,
    extract_final_frame_per_view,
    load_gt_frame,
)
from scripts.build_controllability_pairwise_composites import (  # noqa: E402
    assign_left_right,
    build_pairwise_composite_image,
)


def _load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_trials_by_variant(rollout_manifest_paths: List[str], variant: str) -> Dict[Tuple[str, str], Dict]:
    """(trial_group_id, approach_style) -> trial row, filtered to one model_variant, across
    however many rollout manifest files (e.g. one per split) are passed for that variant."""
    trials: Dict[Tuple[str, str], Dict] = {}
    for path in rollout_manifest_paths:
        manifest = _load_manifest(path)
        for trial in manifest["results"]:
            if trial["model_variant"] != variant:
                continue
            trials[(trial["trial_group_id"], trial["approach_style"])] = trial
    return trials


def _allowed_keys_from_subset(subset_path: Optional[str]) -> Optional[Set[Tuple[str, str]]]:
    """Restrict to the (trial_group_id, approach_style) pairs in an existing subset.json --
    used for the budget-constrained tier, which reuses the human study's own
    50-trial subset instead of the full 560. None (no --restrict_to_subset) means no restriction."""
    if subset_path is None:
        return None
    subset = _load_manifest(subset_path)
    return {(t["trial_group_id"], t["approach_style"]) for t in subset["trials"]}


def build_addendum(
    new_identity: str,
    new_identity_rollout_manifests: List[str],
    opponent_identity: str,
    opponent_rollout_manifests: List[str],
    gt_manifest_path: str,
    output_dir: str,
    output_manifest_path: str,
    pair_index: int,
    seed: int = 0,
    restrict_to_subset: Optional[str] = None,
    num_views: int = 2,
) -> str:
    gt_manifest = _load_manifest(gt_manifest_path)
    gt_by_trial_group: Dict[str, Dict] = {r["trial_group_id"]: r for r in gt_manifest["results"]}

    new_trials = _load_trials_by_variant(new_identity_rollout_manifests, new_identity)
    opponent_trials = _load_trials_by_variant(opponent_rollout_manifests, opponent_identity)
    keys = sorted(set(new_trials) & set(opponent_trials))

    allowed = _allowed_keys_from_subset(restrict_to_subset)
    if allowed is not None:
        keys = [k for k in keys if k in allowed]

    print(f"[INFO]: {new_identity} vs {opponent_identity}: {len(keys)} (trial_group_id, approach_style) "
          f"pairs to build (pair index {pair_index:02d}, seed={seed}"
          f"{', restricted to subset' if restrict_to_subset else ', full trial set'}).")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    existing_manifest: Dict = {"rollout_manifest_paths": [], "gt_manifest_path": gt_manifest_path, "seed": seed, "results": []}
    if os.path.exists(output_manifest_path):
        existing_manifest = _load_manifest(output_manifest_path)
    existing_ids = {r["comparison_id"] for r in existing_manifest["results"]}
    all_rollout_paths_seen = set(existing_manifest.get("rollout_manifest_paths", []))
    all_rollout_paths_seen.update(new_identity_rollout_manifests)
    all_rollout_paths_seen.update(opponent_rollout_manifests)

    n_built = n_skipped_existing = n_missing_gt = 0

    for trial_group_id, approach_style in keys:
        comparison_id = f"{trial_group_id}_{approach_style}_pair{pair_index:02d}"
        if comparison_id in existing_ids:
            n_skipped_existing += 1
            continue

        gt_entry = gt_by_trial_group.get(trial_group_id)
        if gt_entry is None:
            n_missing_gt += 1
            print(f"[WARN]: No GT render for trial_group_id={trial_group_id}; skipping.")
            continue

        hand_ids_by_camera = gt_entry.get("hand_instance_ids")
        if not hand_ids_by_camera:
            raise KeyError(
                f"gt_manifest.json entry for trial_group_id={trial_group_id} has no 'hand_instance_ids' -- "
                "this GT render predates the hand-label fix; re-render with --overwrite_existing."
            )
        arm_ids_by_camera = gt_entry.get("arm_instance_ids") or {}
        gt_by_view = {
            cam: load_gt_frame(gt_entry["render_paths"], hand_ids_by_camera.get(cam, []), cam, arm_ids_by_camera.get(cam, []))
            for cam in CAMERA_VIEW_ORDER
        }

        new_frames = extract_final_frame_per_view(new_trials[(trial_group_id, approach_style)]["video_path"], num_views)
        opponent_frames = extract_final_frame_per_view(opponent_trials[(trial_group_id, approach_style)]["video_path"], num_views)
        base_shape_by_view = {cam: opponent_frames[cam].shape[:2] for cam in CAMERA_VIEW_ORDER}

        anchor_trial = new_trials[(trial_group_id, approach_style)]
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

        left_identity, right_identity = assign_left_right(new_identity, opponent_identity, seed, comparison_id)
        frames_by_identity = {new_identity: new_frames, opponent_identity: opponent_frames}

        composite = build_pairwise_composite_image(
            gt_by_view,
            frames_by_identity[left_identity],
            frames_by_identity[right_identity],
            base_shape_by_view,
            trial_meta_base,
        )
        composite_path = output_dir_path / f"{comparison_id}.png"
        Image.fromarray(composite).save(composite_path)

        existing_manifest["results"].append(
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
        existing_ids.add(comparison_id)
        n_built += 1

    existing_manifest["rollout_manifest_paths"] = sorted(all_rollout_paths_seen)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(existing_manifest, f, indent=2)

    print(f"[INFO]: Built {n_built} new composites ({n_skipped_existing} already present, "
          f"{n_missing_gt} skipped for missing GT). Total rows in {output_manifest_path}: "
          f"{len(existing_manifest['results'])}.")
    return output_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new_identity", required=True)
    parser.add_argument("--new_identity_rollout_manifest", action="append", default=[], dest="new_identity_rollout_manifests", required=True)
    parser.add_argument("--opponent_identity", required=True)
    parser.add_argument("--opponent_rollout_manifest", action="append", default=[], dest="opponent_rollout_manifests", required=True)
    parser.add_argument("--gt_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_manifest", required=True, help="Addendum manifest path -- appended to across multiple runs (one per opponent).")
    parser.add_argument("--pair_index", type=int, required=True, help="Fixed pairNN suffix for this opponent -- must not collide with pair00-pair09 (existing pool) or any other addendum opponent's index.")
    parser.add_argument("--seed", type=int, default=0, help="Must match the existing pool's seed (0) for consistent Model A/B assignment behavior.")
    parser.add_argument("--restrict_to_subset", default=None, help="Optional path to human_comparisons/subset.json -- if given, only builds for (trial_group_id, approach_style) pairs in that subset (the budget-constrained tier). Omit for the full 560-trial tier.")
    parser.add_argument("--num_views", type=int, default=2)
    args = parser.parse_args()

    build_addendum(
        args.new_identity, args.new_identity_rollout_manifests,
        args.opponent_identity, args.opponent_rollout_manifests,
        args.gt_manifest, args.output_dir, args.output_manifest,
        args.pair_index, args.seed, args.restrict_to_subset, args.num_views,
    )


if __name__ == "__main__":
    main()
