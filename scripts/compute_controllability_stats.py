"""Statistics/report script for the controllability-eval pipeline.

Joins rollout manifests (for trial metadata: model_variant, split,
sampling_scope, approach_style, component_label) with llm_scores.jsonl
(Phase 4, full coverage) and human_ratings/*.jsonl (Phase 5, a smaller shared
subset) on trial_id, and reports:

- Mean/median +/- bootstrap 95% CI per (model_variant x split x
  sampling_scope x approach_style), separately for LLM scores (full) and
  human scores (subset) -- the direct upgrade over the old sine-sweep
  protocol's un-intervaled mean.
- The same, again, as *relative* scores (llm_score / gt_self_score) if
  --gt_self_scores is given -- see "GT-relative scoring" below.
- Spearman's rho between the LLM judge's score and each trial's mean human
  score, on the human-rated subset -- the primary "is the automated judge
  trustworthy" statistic.
- Inter-rater reliability, reported two ways deliberately (see the
  controllability-eval plan's Phase 6 section for the full rationale): ICC
  (intraclass correlation, via pingouin -- the statistically correct
  instrument for continuous multi-rater agreement) as the primary number,
  and weighted Cohen's kappa on scores binned back to the old {0, 0.5, 1}
  scale, so results stay comparable to what the old coarse protocol reported.
  Kappa alone would discard information on a genuinely continuous scale;
  ICC alone would break comparability with the old protocol's framing.

GT-relative scoring (--gt_self_scores, optional): a genuinely perfect
generation -- byte-identical to its GT render -- does not reliably score 1.0
from the LLM judge (confirmed empirically: mean 0.85, range 0.82-0.88 across
6 scenes with the current mask-panel composite design; see
build_controllability_composites.py's module docstring for the full A/B
test). That ceiling also isn't constant across scenes. Comparing raw
llm_score against a literal 1.0 therefore both underestimates how good a
rollout really is and conflates two different things: model quality and
per-scene judge calibration. --gt_self_scores points at a jsonl (produced by
running llm_judge_controllability.py over
build_controllability_composites.py --gt_self_check composites -- one score
per trial_group_id, the judge's own achievable ceiling for that specific
scene) which this script joins onto every trial via trial_group_id and uses
to compute relative_llm_score = llm_score / gt_self_score, reported
alongside (never instead of) the raw llm_score. relative_llm_score is left
uncapped (not clamped to 1.0) deliberately: a rollout scoring above its own
scene's GT self-score is rare noise, not a bug to hide, and clamping would
throw that signal away.

Output: <output_dir>/summary.json (machine-readable), report.md (narrative),
plots/*.png.

Example:
    python scripts/compute_controllability_stats.py \\
        --rollout_manifest inference_output/.../rollout_manifest_baseline.json \\
        --rollout_manifest inference_output/.../rollout_manifest_wm1wm2.json \\
        --llm_scores inference_output/.../composites/llm_scores.jsonl \\
        --gt_self_scores inference_output/.../gt_self_check_composites/llm_scores.jsonl \\
        --human_ratings_dir inference_output/.../human_ratings/human_ratings \\
        --output_dir inference_output/.../stats
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

try:
    import pingouin as pg

    HAVE_PINGOUIN = True
except ImportError:
    HAVE_PINGOUIN = False

GROUP_COLS = ["model_variant", "split", "sampling_scope", "approach_style"]
DEFAULT_KAPPA_THRESHOLDS = (0.25, 0.75)


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trial_metadata(rollout_manifest_paths: List[str]) -> pd.DataFrame:
    """One row per trial_id, with the grouping/identifying columns every score joins against."""
    rows = []
    for path in rollout_manifest_paths:
        manifest = _load_json(path)
        for trial in manifest["results"]:
            rows.append(
                {
                    "trial_id": trial["trial_id"],
                    "trial_group_id": trial["trial_group_id"],
                    "model_variant": trial["model_variant"],
                    "split": trial["split"],
                    "sampling_scope": trial["sampling_scope"],
                    "component_label": trial.get("component_label"),
                    "approach_style": trial["approach_style"],
                }
            )
    if not rows:
        raise ValueError("No trials found in the given --rollout_manifest file(s).")
    df = pd.DataFrame(rows)
    dupes = df["trial_id"][df["trial_id"].duplicated()]
    if not dupes.empty:
        raise ValueError(f"Duplicate trial_id(s) across rollout manifests: {sorted(set(dupes))[:5]}...")
    return df


def load_llm_scores(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)[["trial_id", "score", "failure_mode"]].rename(
        columns={"score": "llm_score", "failure_mode": "llm_failure_mode"}
    )


def load_gt_self_scores(path: str) -> pd.DataFrame:
    """gt_self_scores.jsonl has the same row shape as llm_scores.jsonl (it's produced
    by the same llm_judge_controllability.py), but its "trial_id" is actually a
    trial_group_id -- see build_controllability_composites.py's
    build_gt_self_check_composites, which deliberately sets trial_id=trial_group_id
    so this join needs no extra bookkeeping."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)[["trial_id", "score"]].rename(
        columns={"trial_id": "trial_group_id", "score": "gt_self_score"}
    )
    dupes = df["trial_group_id"][df["trial_group_id"].duplicated()]
    if not dupes.empty:
        raise ValueError(f"Duplicate trial_group_id(s) in --gt_self_scores: {sorted(set(dupes))[:5]}...")
    return df


def load_human_ratings(human_ratings_dir: str) -> pd.DataFrame:
    rows = []
    ratings_dir = Path(human_ratings_dir)
    if ratings_dir.exists():
        for jsonl_path in sorted(ratings_dir.glob("*.jsonl")):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["trial_id", "rater", "score", "failure_mode"])
    return pd.DataFrame(rows)[["trial_id", "rater", "score", "failure_mode"]]


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


def group_stats(df: pd.DataFrame, score_col: str, group_cols: List[str] = GROUP_COLS) -> pd.DataFrame:
    records = []
    for group_key, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        mean, lo, hi = bootstrap_ci(group_df[score_col].to_numpy())
        median = float(group_df[score_col].median())
        record = dict(zip(group_cols, group_key))
        record.update({"n": int(len(group_df)), "mean": mean, "median": median, "ci_lo": lo, "ci_hi": hi})
        records.append(record)
    return pd.DataFrame(records)


def per_dim_stats(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Per-action-dimension mean score for per_dim trials -- the direct analogue of
    the old sine-sweep protocol's headline "mean score across all 23 action
    dimensions" number."""
    per_dim = df[df["sampling_scope"] == "per_dim"]
    return group_stats(per_dim, score_col, group_cols=["model_variant", "component_label"])


def add_relative_scores(df: pd.DataFrame, gt_self_scores: pd.DataFrame, score_col: str, relative_col: str) -> pd.DataFrame:
    """Left-join gt_self_scores onto df via trial_group_id and add
    relative_col = score_col / gt_self_score. Left join, not inner: a trial whose
    trial_group_id has no matching gt_self_score keeps relative_col as NaN rather
    than being dropped from df entirely, so raw-score stats computed from the same
    df afterward still cover every trial even when --gt_self_scores only covers a
    subset (e.g. a gt_self_check run that hasn't finished, or predates some
    targets). group_stats/bootstrap_ci already drop NaN internally."""
    merged = df.merge(gt_self_scores, on="trial_group_id", how="left")
    merged[relative_col] = merged[score_col] / merged["gt_self_score"].replace(0, np.nan)
    return merged


def compute_spearman(merged: pd.DataFrame) -> Dict:
    """merged must have llm_score and human_score_mean columns, one row per trial_id."""
    from scipy.stats import spearmanr

    valid = merged.dropna(subset=["llm_score", "human_score_mean"])
    if len(valid) < 3:
        return {"rho": None, "pvalue": None, "n": len(valid), "note": "fewer than 3 paired trials -- not computed"}
    rho, pvalue = spearmanr(valid["llm_score"], valid["human_score_mean"])
    result = {"rho": float(rho), "pvalue": float(pvalue), "n": int(len(valid))}

    stratified = {}
    for group_key, group_df in valid.groupby(["split", "sampling_scope"], dropna=False):
        if len(group_df) >= 5:
            g_rho, g_p = spearmanr(group_df["llm_score"], group_df["human_score_mean"])
            stratified[f"{group_key[0]}/{group_key[1]}"] = {"rho": float(g_rho), "pvalue": float(g_p), "n": int(len(group_df))}
    result["stratified"] = stratified
    return result


def bin_score(score: float, thresholds: Tuple[float, float]) -> int:
    """Maps to the old {0, 0.5, 1} scale, but as an ordered integer *category* code
    (0/1/2), not a float -- sklearn's cohen_kappa_score raises "continuous is not
    supported" if given float bin values like 0.0/0.5/1.0 (it can't distinguish a
    3-value float scale from truly continuous data by dtype alone). weights="linear"
    still penalizes by rank distance correctly with these integer codes."""
    low, high = thresholds
    if score < low:
        return 0
    if score > high:
        return 2
    return 1


def compute_icc(human_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """ICC across raters on trials with >=2 ratings. None if pingouin is unavailable
    or there isn't enough overlapping data (needs >=2 raters x >=2 shared trials)."""
    if not HAVE_PINGOUIN or human_df.empty:
        return None
    rating_counts = human_df.groupby("trial_id")["rater"].nunique()
    multi_rated = rating_counts[rating_counts >= 2].index
    subset = human_df[human_df["trial_id"].isin(multi_rated)]
    if subset["rater"].nunique() < 2 or subset["trial_id"].nunique() < 2:
        return None
    # pingouin needs exactly one rating per (target, rater) pair -- average duplicates if any.
    subset = subset.groupby(["trial_id", "rater"], as_index=False)["score"].mean()
    return pg.intraclass_corr(data=subset, targets="trial_id", raters="rater", ratings="score")


def compute_pairwise_weighted_kappa(human_df: pd.DataFrame, llm_df: pd.DataFrame, thresholds: Tuple[float, float]) -> Dict:
    """Weighted (linear) Cohen's kappa, pairwise between every rater and between each
    rater and the LLM judge, on scores binned to {0, 0.5, 1}. See module docstring for
    why this is reported alongside ICC rather than instead of it."""
    results: Dict[str, float] = {}
    raters = sorted(human_df["rater"].unique()) if not human_df.empty else []

    wide = human_df.pivot_table(index="trial_id", columns="rater", values="score", aggfunc="mean")
    if not llm_df.empty:
        wide = wide.join(llm_df.set_index("trial_id")["llm_score"], how="outer")

    columns = list(wide.columns)
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            pair = wide[[col_a, col_b]].dropna()
            if len(pair) < 3:
                continue
            binned_a = pair[col_a].apply(lambda s: bin_score(s, thresholds))
            binned_b = pair[col_b].apply(lambda s: bin_score(s, thresholds))
            if binned_a.nunique() < 2 and binned_b.nunique() < 2:
                continue  # cohen_kappa_score is undefined/degenerate with a single class on both sides
            kappa = cohen_kappa_score(binned_a, binned_b, weights="linear")
            label_a = "llm_judge" if col_a == "llm_score" else f"rater:{col_a}"
            label_b = "llm_judge" if col_b == "llm_score" else f"rater:{col_b}"
            results[f"{label_a} vs {label_b} (n={len(pair)})"] = float(kappa)
    return results


def _plot_per_dim(per_dim_df: pd.DataFrame, score_label: str, title: str, out_path: Path, ylim: Optional[Tuple[float, float]]) -> None:
    fig, ax = plt.subplots(figsize=(max(8, per_dim_df["component_label"].nunique() * 0.5), 5))
    for variant, group in per_dim_df.groupby("model_variant"):
        group = group.sort_values("component_label")
        yerr = np.array([group["mean"] - group["ci_lo"], group["ci_hi"] - group["mean"]])
        ax.errorbar(group["component_label"], group["mean"], yerr=yerr, marker="o", capsize=3, label=variant)
    ax.set_ylabel(score_label)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    plt.xticks(rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_plots(
    per_dim_df: pd.DataFrame, merged: pd.DataFrame, output_dir: Path, per_dim_df_relative: Optional[pd.DataFrame] = None
) -> set:
    """Returns the set of plot filenames actually written, so write_report_md can
    list exactly those rather than assuming a fixed set -- per_dim plots in
    particular are skipped whenever a run has no per_dim-scope trials (e.g. a
    whole_pose-only sweep), which write_report_md needs to know rather than
    unconditionally claiming the file exists."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = set()

    if not per_dim_df.empty:
        _plot_per_dim(
            per_dim_df, "Mean LLM score", "Per-dimension mean score (95% bootstrap CI)",
            plots_dir / "per_dim_scores.png", ylim=(-0.05, 1.05),
        )
        written.add("per_dim_scores.png")

    if per_dim_df_relative is not None and not per_dim_df_relative.empty:
        # No fixed ylim here, unlike the raw plot: relative scores (llm_score /
        # gt_self_score, see add_relative_scores) are deliberately left uncapped, so
        # a point can legitimately land above 1.0 -- clamping the axis to [0,1.05]
        # would clip that real signal off the chart.
        _plot_per_dim(
            per_dim_df_relative, "Mean relative LLM score (llm_score / gt_self_score)",
            "Per-dimension mean score relative to GT self-score ceiling (95% bootstrap CI)",
            plots_dir / "per_dim_scores_relative.png", ylim=None,
        )
        written.add("per_dim_scores_relative.png")

    valid = merged.dropna(subset=["llm_score", "human_score_mean"])
    if len(valid) >= 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(valid["human_score_mean"], valid["llm_score"], alpha=0.6)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="y=x")
        ax.set_xlabel("Mean human score")
        ax.set_ylabel("LLM judge score")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("LLM judge vs. human rating")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "llm_vs_human_scatter.png", dpi=150)
        plt.close(fig)
        written.add("llm_vs_human_scatter.png")

    return written


def log_stats_to_wandb(
    wandb_project_name: str,
    wandb_run_name: Optional[str],
    llm_group_stats: pd.DataFrame,
    human_group_stats: pd.DataFrame,
    spearman: Dict,
    icc_df: Optional[pd.DataFrame],
    kappa: Dict,
    plots_dir: Path,
    plots_written: set,
) -> None:
    """Logs the same numbers already written to summary.json/report.md/plots/ to a
    dedicated wandb run, for the final-report side of the 4-variant comparison
    (see the controllability-eval plan's "Uploading to Weights & Biases" section).
    Purely additive -- summary.json/report.md/plots/*.png are still written to disk
    exactly as before regardless of whether this is called."""
    import wandb

    run = wandb.init(project=wandb_project_name, name=wandb_run_name, job_type="controllability_eval_stats")
    try:
        if not llm_group_stats.empty:
            run.log({"llm_group_stats": wandb.Table(dataframe=llm_group_stats.reset_index(drop=True))})
        if not human_group_stats.empty:
            run.log({"human_group_stats": wandb.Table(dataframe=human_group_stats.reset_index(drop=True))})
        if spearman.get("rho") is not None:
            run.log({"spearman/rho": spearman["rho"], "spearman/pvalue": spearman["pvalue"], "spearman/n": spearman["n"]})
        if icc_df is not None and not icc_df.empty:
            run.log({"icc": wandb.Table(dataframe=icc_df.reset_index(drop=True))})
        for label, value in kappa.items():
            run.log({f"kappa/{label}": value})
        for plot_name in sorted(plots_written):
            run.log({f"plots/{plot_name}": wandb.Image(str(plots_dir / plot_name))})
    finally:
        run.finish()


def write_report_md(
    output_dir: Path,
    llm_group_stats: pd.DataFrame,
    human_group_stats: pd.DataFrame,
    spearman: Dict,
    icc_df: Optional[pd.DataFrame],
    kappa: Dict,
    kappa_thresholds: Tuple[float, float],
    n_llm: int,
    n_human_ratings: int,
    n_human_trials: int,
    n_human_raters: int,
    llm_group_stats_relative: Optional[pd.DataFrame] = None,
    human_group_stats_relative: Optional[pd.DataFrame] = None,
    plots_written: Optional[set] = None,
    overall_by_variant: Optional[pd.DataFrame] = None,
    overall_by_variant_relative: Optional[pd.DataFrame] = None,
) -> None:
    plots_written = plots_written or set()
    lines = ["# Controllability-eval statistics report", ""]
    lines.append(f"LLM judge: {n_llm} scored trials. Human study: {n_human_raters} rater(s), "
                 f"{n_human_ratings} ratings across {n_human_trials} distinct trials.")
    lines.append("")

    if overall_by_variant is not None and not overall_by_variant.empty:
        lines.append("## Overall ranking by model variant (headline table)")
        lines.append("")
        lines.append("Collapses every split/sampling_scope/approach_style into one number per "
                      "model_variant -- the number to cite as \"the\" score for a variant. The "
                      "per-condition breakdowns below are the source of truth for anything more "
                      "specific than that.")
        lines.append("")
        ranked = overall_by_variant.sort_values("mean", ascending=False).reset_index(drop=True)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        lines.append(ranked.round(3).to_markdown(index=False))
        lines.append("")
        if overall_by_variant_relative is not None and not overall_by_variant_relative.empty:
            lines.append("Relative to GT self-score (score / gt_self_score for that trial's own scene; "
                          "see the full explanation under the per-condition relative table below):")
            lines.append("")
            ranked_rel = overall_by_variant_relative.sort_values("mean", ascending=False).reset_index(drop=True)
            ranked_rel.insert(0, "rank", range(1, len(ranked_rel) + 1))
            lines.append(ranked_rel.round(3).to_markdown(index=False))
            lines.append("")

    lines.append("## Mean score by condition (LLM judge, full coverage)")
    lines.append("")
    lines.append(llm_group_stats.round(3).to_markdown(index=False) if not llm_group_stats.empty else "_no data_")
    lines.append("")

    if llm_group_stats_relative is not None and not llm_group_stats_relative.empty:
        lines.append("## Mean score by condition, relative to GT self-score (LLM judge)")
        lines.append("")
        lines.append("A genuinely perfect generation doesn't reliably score a literal 1.0 from the LLM judge "
                      "(confirmed empirically -- see build_controllability_composites.py's module docstring), "
                      "and that ceiling varies by scene. These numbers are score / gt_self_score for that "
                      "trial's own scene -- how close the rollout got to the judge's own achievable maximum, "
                      "not to an unreachable 1.0. Deliberately uncapped: a value above 1.0 is real (noisy) "
                      "signal, not an error.")
        lines.append("")
        lines.append(llm_group_stats_relative.round(3).to_markdown(index=False))
        lines.append("")

    if not human_group_stats.empty:
        lines.append("## Mean score by condition (human ratings, subset)")
        lines.append("")
        lines.append(human_group_stats.round(3).to_markdown(index=False))
        lines.append("")

    if human_group_stats_relative is not None and not human_group_stats_relative.empty:
        lines.append("## Mean score by condition, relative to GT self-score (human ratings, subset)")
        lines.append("")
        lines.append(human_group_stats_relative.round(3).to_markdown(index=False))
        lines.append("")

    lines.append("## Validity: LLM judge vs. human ratings (Spearman's rho)")
    lines.append("")
    if spearman.get("rho") is not None:
        lines.append(f"Overall: rho={spearman['rho']:.3f} (p={spearman['pvalue']:.3g}, n={spearman['n']}). "
                      f"Closer to 1 means the automated judge tracks human judgment well; "
                      f"small n means treat this as indicative, not definitive.")
        for key, val in spearman.get("stratified", {}).items():
            lines.append(f"- {key}: rho={val['rho']:.3f} (p={val['pvalue']:.3g}, n={val['n']})")
    else:
        lines.append(f"Not computed: {spearman.get('note', 'insufficient paired data')}.")
    lines.append("")

    lines.append("## Inter-rater reliability")
    lines.append("")
    lines.append("Reported two ways deliberately: ICC is the statistically correct instrument for "
                  "continuous multi-rater agreement (primary number below); weighted Cohen's kappa is "
                  f"computed after binning scores back to the old {{0, 0.5, 1}} scale (thresholds "
                  f"<{kappa_thresholds[0]}, {kappa_thresholds[0]}-{kappa_thresholds[1]}, "
                  f">{kappa_thresholds[1]}) specifically so this stays comparable to what the prior "
                  "coarse 3-point protocol would have reported. Kappa alone would discard information "
                  "on a genuinely continuous scale; ICC alone would break that comparability.")
    lines.append("")
    if icc_df is not None:
        lines.append("**ICC** (all six variants; ICC(A,1) is the recommended primary reading -- "
                      "absolute agreement, single rater):")
        lines.append("")
        lines.append(icc_df.round(3).to_markdown(index=False))
    else:
        lines.append("**ICC**: not computed -- needs >=2 raters with overlapping rated trials"
                      + ("" if HAVE_PINGOUIN else " (pingouin not installed)") + ".")
    lines.append("")
    if kappa:
        lines.append("**Weighted (linear) Cohen's kappa**, pairwise:")
        lines.append("")
        for label, value in kappa.items():
            lines.append(f"- {label}: kappa={value:.3f}")
    else:
        lines.append("**Weighted Cohen's kappa**: not computed -- needs at least one pair of raters/judge "
                      "with >=3 overlapping rated trials.")
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    # Listed only if make_plots actually wrote them (plots_written) -- a run with no
    # per_dim-scope trials (e.g. whole_pose-only) skips per_dim_scores*.png entirely,
    # and this used to unconditionally claim they existed regardless.
    if "per_dim_scores.png" in plots_written:
        lines.append("- `plots/per_dim_scores.png` -- mean LLM score per action dimension, the direct "
                      "analogue of the old sine-sweep protocol's headline number.")
    if "per_dim_scores_relative.png" in plots_written:
        lines.append("- `plots/per_dim_scores_relative.png` -- same, but score/gt_self_score (see the "
                      "relative-score table above); note this plot is not on a fixed 0-1 axis.")
    if "llm_vs_human_scatter.png" in plots_written:
        lines.append("- `plots/llm_vs_human_scatter.png` -- visualizes the Spearman correlation above.")
    if not plots_written:
        lines.append("_no plots generated (insufficient data)._")

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollout_manifest", action="append", required=True, dest="rollout_manifests", help="Repeatable: one or more rollout_manifest_{baseline,wm1wm2}.json.")
    parser.add_argument("--llm_scores", type=str, required=True, help="Path to llm_scores.jsonl.")
    parser.add_argument(
        "--gt_self_scores", type=str, default=None,
        help="Optional: path to a gt_self_scores.jsonl (llm_judge_controllability.py run over "
        "build_controllability_composites.py --gt_self_check composites). When given, adds "
        "score-relative-to-GT-self-score tables/plots -- see module docstring.",
    )
    parser.add_argument("--human_ratings_dir", type=str, default=None, help="Directory of per-rater human_ratings/<name>.jsonl files (optional).")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--kappa_low", type=float, default=DEFAULT_KAPPA_THRESHOLDS[0])
    parser.add_argument("--kappa_high", type=float, default=DEFAULT_KAPPA_THRESHOLDS[1])
    parser.add_argument(
        "--wandb_project_name", type=str, default=None,
        help="Optional: when set (along with --wandb_run_name), also logs the group-stats "
        "tables, Spearman/ICC/kappa numbers, and plots/*.png to a dedicated wandb run.",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kappa_thresholds = (args.kappa_low, args.kappa_high)

    metadata = load_trial_metadata(args.rollout_manifests)
    llm_scores = load_llm_scores(args.llm_scores)
    human_ratings = load_human_ratings(args.human_ratings_dir) if args.human_ratings_dir else pd.DataFrame(columns=["trial_id", "rater", "score", "failure_mode"])
    gt_self_scores = load_gt_self_scores(args.gt_self_scores) if args.gt_self_scores else None

    llm_df = metadata.merge(llm_scores, on="trial_id", how="inner")
    human_with_meta = metadata.merge(human_ratings, on="trial_id", how="inner")

    if gt_self_scores is not None:
        llm_df = add_relative_scores(llm_df, gt_self_scores, "llm_score", "relative_llm_score")
        if not human_with_meta.empty:
            human_with_meta = add_relative_scores(human_with_meta, gt_self_scores, "score", "relative_human_score")

    llm_group_stats = group_stats(llm_df, "llm_score")
    llm_group_stats_relative = group_stats(llm_df, "relative_llm_score") if gt_self_scores is not None else pd.DataFrame()
    overall_by_variant = group_stats(llm_df, "llm_score", group_cols=["model_variant"])
    overall_by_variant_relative = (
        group_stats(llm_df, "relative_llm_score", group_cols=["model_variant"])
        if gt_self_scores is not None
        else pd.DataFrame()
    )
    human_group_stats = group_stats(human_with_meta.rename(columns={"score": "human_score"}), "human_score") if not human_with_meta.empty else pd.DataFrame()
    human_group_stats_relative = (
        group_stats(human_with_meta.rename(columns={"score": "human_score"}), "relative_human_score")
        if not human_with_meta.empty and gt_self_scores is not None
        else pd.DataFrame()
    )

    per_dim_df = per_dim_stats(llm_df, "llm_score")
    per_dim_df_relative = per_dim_stats(llm_df, "relative_llm_score") if gt_self_scores is not None else pd.DataFrame()

    human_mean_per_trial = (
        human_ratings.groupby("trial_id")["score"].mean().reset_index().rename(columns={"score": "human_score_mean"})
        if not human_ratings.empty
        else pd.DataFrame(columns=["trial_id", "human_score_mean"])
    )
    merged = llm_df.merge(human_mean_per_trial, on="trial_id", how="inner")
    spearman = compute_spearman(merged)

    icc_df = compute_icc(human_ratings)
    kappa = compute_pairwise_weighted_kappa(human_ratings, llm_scores, kappa_thresholds)

    plots_written = make_plots(per_dim_df, merged, output_dir, per_dim_df_relative)

    summary = {
        "n_llm_scored_trials": int(len(llm_df)),
        "n_human_ratings": int(len(human_ratings)),
        "n_human_rated_trials": int(human_ratings["trial_id"].nunique()) if not human_ratings.empty else 0,
        "n_human_raters": int(human_ratings["rater"].nunique()) if not human_ratings.empty else 0,
        "gt_self_scores_path": args.gt_self_scores,
        "n_gt_self_scores": int(len(gt_self_scores)) if gt_self_scores is not None else 0,
        "overall_by_variant": overall_by_variant.sort_values("mean", ascending=False).to_dict(orient="records"),
        "overall_by_variant_relative_to_gt_self": (
            overall_by_variant_relative.sort_values("mean", ascending=False).to_dict(orient="records")
            if not overall_by_variant_relative.empty else []
        ),
        "llm_group_stats": llm_group_stats.to_dict(orient="records"),
        "llm_group_stats_relative_to_gt_self": llm_group_stats_relative.to_dict(orient="records") if not llm_group_stats_relative.empty else [],
        "human_group_stats": human_group_stats.to_dict(orient="records") if not human_group_stats.empty else [],
        "human_group_stats_relative_to_gt_self": human_group_stats_relative.to_dict(orient="records") if not human_group_stats_relative.empty else [],
        "per_dim_llm_stats": per_dim_df.to_dict(orient="records"),
        "per_dim_llm_stats_relative_to_gt_self": per_dim_df_relative.to_dict(orient="records") if not per_dim_df_relative.empty else [],
        "spearman_llm_vs_human": spearman,
        "icc": icc_df.to_dict(orient="records") if icc_df is not None else None,
        "weighted_kappa_pairwise": kappa,
        "kappa_bin_thresholds": list(kappa_thresholds),
    }
    def _json_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    write_report_md(
        output_dir, llm_group_stats, human_group_stats, spearman, icc_df, kappa, kappa_thresholds,
        n_llm=len(llm_df),
        n_human_ratings=len(human_ratings),
        n_human_trials=int(human_ratings["trial_id"].nunique()) if not human_ratings.empty else 0,
        n_human_raters=int(human_ratings["rater"].nunique()) if not human_ratings.empty else 0,
        llm_group_stats_relative=llm_group_stats_relative,
        human_group_stats_relative=human_group_stats_relative,
        plots_written=plots_written,
        overall_by_variant=overall_by_variant,
        overall_by_variant_relative=overall_by_variant_relative,
    )

    print(f"[INFO]: Wrote {output_dir / 'summary.json'}, {output_dir / 'report.md'}, and plots/ to {output_dir}")

    if args.wandb_project_name:
        log_stats_to_wandb(
            args.wandb_project_name, args.wandb_run_name,
            llm_group_stats, human_group_stats, spearman, icc_df, kappa,
            output_dir / "plots", plots_written,
        )
        print(f"[INFO]: Logged stats to wandb project '{args.wandb_project_name}'")


if __name__ == "__main__":
    main()
