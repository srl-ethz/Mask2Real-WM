"""Build the combined, blinded subset.json for the 5-model sine-actions human rating
study (scripts/gradio_sine_actions_rating.py).

Merges the 10 (model, dataset) manifests written by scripts/paper/run_sine_sweeps.sh into one flat trial list, assigns
each of the 5 models an opaque random slot (model_A..model_E), and writes:
  - subset.json      -- all 1150 trials, no model identity (rater-facing)
  - blinding_key.json -- slot -> real model name (analysis-only, never read by the app)

Usage:
    python scripts/build_sine_actions_rating_subset.py --sweeps_dir outputs/sine_sweeps \
        --output_dir outputs/sine_actions_human_study --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random

ACTION_LABELS = [
    "x", "y", "z", "roll", "pitch", "yaw",
    "wrist",
    "thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip",
    "index_abd", "index_mcp", "index_pip",
    "middle_abd", "middle_mcp", "middle_pip",
    "ring_abd", "ring_mcp", "ring_pip",
    "pinky_abd", "pinky_mcp", "pinky_pip",
]

# (model_key, display name for the blinding key, model name used by scripts/paper/run_sine_sweeps.sh)
MODELS = [
    ("baseline_midtrain_lora", "baseline_midtrain_lora (checkpoint-58000)", "mono_sr_58000"),
    ("baseline_real_only", "baseline_real_only (checkpoint-22000)", "mono_r"),
    ("wm1_realonly_wm2", "WM1(real-only)->WM2", "cascade_r"),
    ("wm1_midtrain_wm2", "WM1(midtrain)->WM2", "cascade_s"),
    ("wm1_midtrain_lora_wm2", "WM1(midtrain+LoRA)->WM2", "cascade_sr"),
]
# (dataset key in the trial ids, split name used by scripts/paper/run_sine_sweeps.sh)
DATASETS = [("id", "id"), ("ood_no_cube", "ood")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps_dir", default="outputs/sine_sweeps",
                   help="Folder with the <split>_<model>/manifest.json outputs of scripts/paper/run_sine_sweeps.sh")
    p.add_argument("--output_dir", default="outputs/sine_actions_human_study")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(args.seed)
    slots = ["model_A", "model_B", "model_C", "model_D", "model_E"]
    rng.shuffle(slots)

    blinding_key = {}
    trials = []
    for (model_key, display_name, sweep_model), slot in zip(MODELS, slots):
        blinding_key[slot] = {"model_key": model_key, "display_name": display_name}
        for dataset_key, split in DATASETS:
            manifest_path = os.path.join(args.sweeps_dir, f"{split}_{sweep_model}", "manifest.json")
            if not os.path.isfile(manifest_path):
                raise SystemExit(f"No manifest found: {manifest_path}")
            with open(manifest_path) as f:
                manifest = json.load(f)
            results = manifest["results"]
            for entry in results:
                sample_idx = entry["sample_idx"]
                component = entry["component"]
                video_path = entry["video_path"]
                if not os.path.isfile(video_path):
                    raise SystemExit(f"Video missing on disk: {video_path} (from {manifest_path})")
                trial_id = f"{model_key}_{dataset_key}_sample{sample_idx:05d}_dim{component:02d}"
                trials.append({
                    "trial_id": trial_id,
                    "model_slot": slot,
                    "dataset": dataset_key,
                    "sample_idx": sample_idx,
                    "component": component,
                    "component_label": ACTION_LABELS[component],
                    "video_path": os.path.abspath(video_path),
                })
            print(f"{model_key} / {dataset_key}: {len(results)} trials from {manifest_path}")

    subset_path = os.path.join(args.output_dir, "subset.json")
    with open(subset_path, "w") as f:
        json.dump({"seed": args.seed, "trials": trials}, f, indent=2)

    blinding_key_path = os.path.join(args.output_dir, "blinding_key.json")
    with open(blinding_key_path, "w") as f:
        json.dump(blinding_key, f, indent=2)

    print(f"\n[INFO] Wrote {len(trials)} trials -> {subset_path}")
    print(f"[INFO] Blinding key (not shown to raters) -> {blinding_key_path}")
    assert len(trials) == 5 * 2 * 5 * 23, f"expected 1150 trials, got {len(trials)}"
    assert len(blinding_key) == 5


if __name__ == "__main__":
    main()
