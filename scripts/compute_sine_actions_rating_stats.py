"""Compute per-model/dataset/joint summary stats and inter-rater agreement for the
5-model sine-actions human rating study.

Purpose-built for this study's 3-point {0, 0.5, 1.0} data -- the existing
compute_controllability_stats.py targets a different (continuous-score,
LLM-vs-human) pipeline and manifest schema and isn't a fit here.

Usage:
    python scripts/compute_sine_actions_rating_stats.py \\
        --study_dir outputs/sine_actions_human_study \\
        --output_path outputs/sine_actions_human_study/stats_report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CATEGORIES = [0.0, 0.5, 1.0]

# Mirrors EE_LABELS / HAND_JOINT_LABELS / HAND_DIMS in inference_wm1_to_wm2_motion_suite.py:50-73 --
# dims 0-5 are the 6-DOF end-effector pose (x,y,z,roll,pitch,yaw), dims 6-22 are the 17 hand
# joints (wrist + 4 fingers + thumb).
EE_DIMS = set(range(0, 6))
HAND_DIMS = set(range(6, 23))


def component_group(component: int) -> str:
    if component in EE_DIMS:
        return "end_effector"
    if component in HAND_DIMS:
        return "hand_joint"
    raise ValueError(f"component {component} is not in the expected 0-22 action-dim range")


def load_ratings(study_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(study_dir, "human_ratings", "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No ratings found under {study_dir}/human_ratings/*.jsonl")
    return pd.DataFrame(rows)


def load_blinding_key(study_dir: str) -> Dict[str, str]:
    with open(os.path.join(study_dir, "blinding_key.json")) as f:
        key = json.load(f)
    return {slot: v["display_name"] for slot, v in key.items()}


def fleiss_kappa(item_category_counts: np.ndarray) -> float:
    """Standard Fleiss' kappa. item_category_counts: (N items, K categories) count matrix,
    every row must sum to the same n (raters per item) -- caller is responsible for only
    passing items with a consistent number of ratings."""
    N, K = item_category_counts.shape
    n = item_category_counts.sum(axis=1)
    assert np.all(n == n[0]) and n[0] > 1, "Fleiss' kappa requires the same n>1 raters on every included item"
    n = int(n[0])

    p_j = item_category_counts.sum(axis=0) / (N * n)
    P_i = (np.square(item_category_counts).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e_bar = np.square(p_j).sum()
    if P_e_bar >= 1.0:
        return 1.0
    return float((P_bar - P_e_bar) / (1.0 - P_e_bar))


def pairwise_agreement(df: pd.DataFrame) -> float:
    """Mean, over all rater pairs, of exact-match fraction on their commonly-rated trials."""
    raters = sorted(df["rater"].unique())
    if len(raters) < 2:
        return float("nan")
    by_rater = {r: df[df["rater"] == r].set_index("trial_id")["answer"] for r in raters}
    agreements = []
    for r1, r2 in combinations(raters, 2):
        s1, s2 = by_rater[r1], by_rater[r2]
        common = s1.index.intersection(s2.index)
        if len(common) == 0:
            continue
        agreements.append(float((s1.loc[common].values == s2.loc[common].values).mean()))
    return float(np.mean(agreements)) if agreements else float("nan")


def compute_group_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        counts = {c: int((g["answer"] == c).sum()) for c in CATEGORIES}
        majority = max(counts, key=counts.get)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_ratings": len(g),
            "mean_score": round(g["answer"].mean(), 3),
            "majority_vote": majority,
            "n_0": counts[0.0], "n_0.5": counts[0.5], "n_1.0": counts[1.0],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study_dir", default="outputs/sine_actions_human_study")
    p.add_argument("--output_path", default=None, help="Defaults to <study_dir>/stats_report.md")
    args = p.parse_args()
    output_path = args.output_path or os.path.join(args.study_dir, "stats_report.md")

    df = load_ratings(args.study_dir)
    display_name = load_blinding_key(args.study_dir)
    df["model"] = df["model_slot"].map(display_name)
    df["component_group"] = df["component"].map(component_group)

    n_raters = df["rater"].nunique()
    n_trials_total = df["trial_id"].nunique()
    counts_per_trial = df.groupby("trial_id").size()
    full_coverage_trials = counts_per_trial[counts_per_trial == n_raters].index
    df_full = df[df["trial_id"].isin(full_coverage_trials)]

    lines = ["# Sine-actions human rating study -- stats report", ""]
    lines.append(f"Raters: {n_raters} ({', '.join(sorted(df['rater'].unique()))})")
    lines.append(f"Distinct trials with >=1 rating: {n_trials_total} / 1150")
    lines.append(f"Trials rated by all {n_raters} raters (used for agreement stats): {len(full_coverage_trials)}")
    lines.append("")

    if n_raters >= 2 and len(full_coverage_trials) > 0:
        pw = pairwise_agreement(df_full)
        lines.append(f"**Pairwise exact-match agreement** (mean over rater pairs, on commonly-rated trials): {pw:.3f}")
        counts_matrix = (
            df_full.groupby("trial_id")["answer"]
            .apply(lambda s: [int((s == c).sum()) for c in CATEGORIES])
        )
        matrix = np.array(counts_matrix.tolist())
        try:
            kappa = fleiss_kappa(matrix)
            lines.append(f"**Fleiss' kappa** (3 categories, {n_raters} raters, n={len(full_coverage_trials)} fully-rated trials): {kappa:.3f}")
        except AssertionError as e:
            lines.append(f"Fleiss' kappa: not computed ({e})")
    else:
        lines.append("Fewer than 2 raters (or no fully-overlapping trials) so far -- agreement stats will appear once more ratings are in.")
    lines.append("")

    lines.append("## Per (model, dataset) summary")
    lines.append("")
    md_summary = compute_group_stats(df, ["model", "dataset"])
    lines.append(md_summary.sort_values(["dataset", "mean_score"], ascending=[True, False]).to_markdown(index=False))
    lines.append("")

    lines.append("## End-effector vs. hand-joint controllability")
    lines.append("")
    lines.append(
        "End-effector = the 6-DOF wrist pose (x, y, z, roll, pitch, yaw). Hand joint = the "
        "17 finger/wrist joints. Averages are kept separate rather than pooled into one "
        "hand-wide number, since the two action groups have very different dimensionality "
        "(6 vs. 17) and pooling would let whichever group has more dims dominate the average."
    )
    lines.append("")
    group_stats = compute_group_stats(df, ["model", "dataset", "component_group"])
    lines.append("### End-effector actions (6 dims: x, y, z, roll, pitch, yaw)")
    lines.append("")
    ee_stats = group_stats[group_stats["component_group"] == "end_effector"].drop(columns=["component_group"])
    lines.append(ee_stats.sort_values(["dataset", "mean_score"], ascending=[True, False]).to_markdown(index=False))
    lines.append("")
    lines.append("### Hand joints (17 dims: wrist + thumb/index/middle/ring/pinky)")
    lines.append("")
    hand_stats = group_stats[group_stats["component_group"] == "hand_joint"].drop(columns=["component_group"])
    lines.append(hand_stats.sort_values(["dataset", "mean_score"], ascending=[True, False]).to_markdown(index=False))
    lines.append("")

    lines.append("## Per (model, dataset, joint) detail")
    lines.append("")
    joint_detail = compute_group_stats(df, ["model", "dataset", "component_label"])
    lines.append(joint_detail.sort_values(["dataset", "model", "component_label"]).to_markdown(index=False))
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] wrote {output_path}")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
