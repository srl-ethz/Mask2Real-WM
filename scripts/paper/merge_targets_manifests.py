"""Merge the in-distribution and OOD controllability targets manifests into the single
manifest that the Isaac Sim ground-truth renderer consumes.

The combined manifest keeps the ID manifest's header fields, concatenates the two target
lists (ID first) and records both dataset names. Trial-group IDs already encode the split,
so they stay unique.

Example:
    python scripts/paper/merge_targets_manifests.py \
        --id_manifest outputs/controllability/id/targets_manifest.json \
        --ood_manifest outputs/controllability/ood/targets_manifest.json \
        --output outputs/controllability/combined_targets_manifest.json
"""

import argparse
import json


def merge_targets_manifests(id_manifest: dict, ood_manifest: dict) -> dict:
    merged = dict(id_manifest)
    merged["targets"] = id_manifest["targets"] + ood_manifest["targets"]
    merged["id_dataset_name"] = id_manifest["dataset_name"]
    merged["ood_dataset_name"] = ood_manifest["dataset_name"]
    trial_group_ids = [t["trial_group_id"] for t in merged["targets"]]
    if len(set(trial_group_ids)) != len(trial_group_ids):
        raise ValueError("trial_group_id collision between the ID and OOD manifests")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--id_manifest", required=True)
    parser.add_argument("--ood_manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.id_manifest) as f:
        id_manifest = json.load(f)
    with open(args.ood_manifest) as f:
        ood_manifest = json.load(f)
    merged = merge_targets_manifests(id_manifest, ood_manifest)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {len(merged['targets'])} targets to {args.output}")


if __name__ == "__main__":
    main()
