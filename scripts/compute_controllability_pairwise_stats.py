"""Statistics/report script for the pairwise A/B controllability-eval pipeline (v2).

Joins pairwise_manifest.json (for left_identity/right_identity de-blinding)
with llm_comparisons.jsonl (Phase 2, full coverage) and human_comparisons/*.jsonl
(Phase 3, a smaller shared subset) on comparison_id, and reports:

- A full per-identity Elo/win-rate ranking (LLM, and pooled across all human
  raters). Elo itself is compute_elo() below -- its {left_run_id, right_run_id,
  vote} input shape
  is exactly what a de-blinded pairwise-vote row already looks like once
  joined against the manifest -- but see compute_stable_elo's docstring: a
  single pass of compute_elo() is a sequential rating update and is sensitive
  to the (arbitrary) order its input list happens to be in, confirmed
  empirically to swing a candidate's Elo by >100 points and change its rank
  on this pipeline's own real data. compute_stable_elo() averages many
  random-order passes over the same fixed win/loss/tie outcomes to get a
  number that reflects the outcomes rather than processing order, and reports
  elo_std alongside so residual uncertainty stays visible.
- The headline "test of the comparison" result: the synthetic "gt_self"
  identity's row in that same Elo table. Since gt_self is a literal,
  pixel-identical copy of the GT reference in every comparison it appears in
  (see build_controllability_pairwise_composites.py), a sound comparison
  methodology should give it a win_rate far above the 50% no-signal baseline
  -- ideally close to 100%. If it doesn't, that's a real methodology finding
  (blinding leak, confusing prompt/UI, judge miscalibration), not a finding
  about the models.
- LLM-vs-human agreement: simple % agreement + unweighted Cohen's kappa
  between the LLM's vote and each human rater's vote on shared comparison_ids
  (plus a pooled figure across all raters). No de-blinding is needed for this
  specific computation -- both consumers view the identical pre-built image
  (same Model A/B assignment) for a given comparison_id, so comparing raw
  votes directly is already meaningful.

Output: <output_dir>/summary.json (machine-readable), report.md (narrative),
plots/*.png.

Example:
    python scripts/compute_controllability_pairwise_stats.py \\
        --pairwise_manifest inference_output/.../pairwise_composites/pairwise_manifest.json \\
        --llm_comparisons inference_output/.../llm_comparisons.jsonl \\
        --human_comparisons_dir inference_output/.../human_comparisons/human_comparisons \\
        --output_dir inference_output/.../pairwise_stats
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)


def compute_elo(
    pairs: List[Dict],
    run_labels: Dict[str, str],
    k: float = 32.0,
    initial: float = 1500.0,
) -> List[Dict]:
    """Sequential Elo update over {left_run_id, right_run_id, vote} rows.

    vote is "left", "right" or anything else (a tie). Returns one row per
    identity in run_labels, sorted by Elo (highest first).
    """
    ratings = {rid: initial for rid in run_labels}
    wins    = {rid: 0 for rid in run_labels}
    losses  = {rid: 0 for rid in run_labels}
    ties    = {rid: 0 for rid in run_labels}

    for pair in pairs:
        vote = pair.get("vote")
        if vote is None:
            continue

        a, b = pair["left_run_id"], pair["right_run_id"]
        ra, rb = ratings[a], ratings[b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea

        if vote == "left":
            sa, sb = 1.0, 0.0
            wins[a] += 1; losses[b] += 1
        elif vote == "right":
            sa, sb = 0.0, 1.0
            wins[b] += 1; losses[a] += 1
        else:
            sa = sb = 0.5
            ties[a] += 1; ties[b] += 1

        ratings[a] += k * (sa - ea)
        ratings[b] += k * (sb - eb)

    results = []
    for rid, label in run_labels.items():
        n = wins[rid] + losses[rid] + ties[rid]
        results.append({
            "run_id":        rid,
            "run_label":     label,
            "elo":           round(ratings[rid], 1),
            "wins":          wins[rid],
            "losses":        losses[rid],
            "ties":          ties[rid],
            "n_comparisons": n,
            "win_rate":      wins[rid] / n if n > 0 else None,
        })

    results.sort(key=lambda x: x["elo"], reverse=True)
    return results

GT_SELF_IDENTITY = "gt_self"


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pairwise_metadata(pairwise_manifest_paths: List[str]) -> pd.DataFrame:
    """One row per comparison_id, with the grouping/de-blinding columns everything joins against."""
    rows = []
    for path in pairwise_manifest_paths:
        manifest = _load_json(path)
        rows.extend(manifest["results"])
    if not rows:
        raise ValueError("No comparisons found in the given --pairwise_manifest file(s).")
    df = pd.DataFrame(rows)[
        ["comparison_id", "trial_group_id", "split", "sampling_scope", "component_label",
         "approach_style", "left_identity", "right_identity"]
    ]
    dupes = df["comparison_id"][df["comparison_id"].duplicated()]
    if not dupes.empty:
        raise ValueError(f"Duplicate comparison_id(s) across pairwise manifests: {sorted(set(dupes))[:5]}...")
    return df


def load_llm_comparisons(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["comparison_id", "vote"])
    return pd.DataFrame(rows)[["comparison_id", "vote"]]


def load_human_comparisons(human_comparisons_dir: str) -> pd.DataFrame:
    rows = []
    comparisons_dir = Path(human_comparisons_dir)
    if comparisons_dir.exists():
        for jsonl_path in sorted(comparisons_dir.glob("*.jsonl")):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["comparison_id", "rater", "vote"])
    return pd.DataFrame(rows)[["comparison_id", "rater", "vote"]]


def all_identities(metadata: pd.DataFrame) -> List[str]:
    return sorted(set(metadata["left_identity"]) | set(metadata["right_identity"]))


def to_elo_input(comparisons_df: pd.DataFrame, metadata: pd.DataFrame) -> List[Dict]:
    """De-blind + reshape: join on comparison_id, map vote "A"/"B"/"tie" -> "left"/"right"/"tie"
    against left_identity/right_identity -- the exact {left_run_id, right_run_id, vote} triples
    compute_elo() consumes."""
    if comparisons_df.empty:
        return []
    merged = comparisons_df.merge(
        metadata[["comparison_id", "left_identity", "right_identity"]], on="comparison_id", how="inner"
    )
    vote_map = {"A": "left", "B": "right", "tie": "tie"}
    return [
        {
            "left_run_id": row["left_identity"],
            "right_run_id": row["right_identity"],
            "vote": vote_map.get(row["vote"], "tie"),
        }
        for _, row in merged.iterrows()
    ]


def compute_stable_elo(pairs: List[Dict], run_labels: Dict, n_shuffles: int = 200, seed: int = 0) -> List[Dict]:
    """compute_elo() is a sequential rating update, so it is
    order-dependent: the exact same fixed set of win/loss/tie outcomes, processed in a different
    order, produces a different Elo. Confirmed empirically on this run's real data -- reshuffling
    the same 5599 comparisons swung one candidate's Elo by >100 points and changed its rank among
    the 4 real model variants across different orderings, even though its win/loss/tie counts
    (which ARE order-independent) were identical every time and already showed it as the
    strongest. The order these comparisons happen to arrive in from `client.messages.batches.results()`
    is explicitly documented as arbitrary, not meaningful -- so a single-order Elo here isn't a
    measurement, it's closer to a coin flip. Averaging Elo over many random processing orders of
    the same outcomes converges to a stable estimate that matches the (order-independent) win rate
    picture; win/loss/tie/win_rate are taken from one representative run since they never change
    across shuffles, and elo_std is kept so the residual uncertainty stays visible rather than
    hidden behind a falsely precise single number.
    """
    accum: Dict[str, List[float]] = {rid: [] for rid in run_labels}
    representative_row: Dict[str, Dict] = {}
    rng = random.Random(seed)
    for _ in range(n_shuffles):
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        for row in compute_elo(shuffled, run_labels):
            accum[row["run_id"]].append(row["elo"])
            representative_row[row["run_id"]] = row

    results = []
    for rid, label in run_labels.items():
        if not accum[rid]:
            continue
        row = dict(representative_row[rid])
        # compute_elo() always returns a row for every identity in run_labels, even ones with zero
        # actual comparisons (e.g. gt_self against human raters, who never see it at all) -- such a
        # row is stuck at the untouched initial rating (1500) in every shuffle, which would
        # otherwise show up here looking like a real (if middling) result rather than "never
        # evaluated." Exclude identities that never actually appeared in a comparison.
        if row["wins"] + row["losses"] + row["ties"] == 0:
            continue
        row["elo"] = round(float(np.mean(accum[rid])), 1)
        row["elo_std"] = round(float(np.std(accum[rid])), 1)
        results.append(row)
    results.sort(key=lambda x: x["elo"], reverse=True)
    return results


def compute_agreement(llm_df: pd.DataFrame, human_df: pd.DataFrame, min_overlap: int = 3) -> Dict:
    """Simple % agreement + unweighted Cohen's kappa (A/B/tie is categorical, not ordinal, so no
    linear weighting here unlike v1's binned-score kappa) between the LLM's vote and each human
    rater's vote on shared comparison_ids, plus one pooled figure across all raters. No
    de-blinding needed for this computation: both consumers view the identical pre-built image
    (same Model A/B assignment) for a given comparison_id, so comparing raw votes directly is
    already meaningful."""
    results: Dict[str, Dict] = {}
    if llm_df.empty or human_df.empty:
        return results

    llm_by_id = llm_df.set_index("comparison_id")["vote"]
    for rater, rater_df in human_df.groupby("rater"):
        rater_by_id = rater_df.set_index("comparison_id")["vote"]
        shared = llm_by_id.index.intersection(rater_by_id.index)
        if len(shared) < min_overlap:
            continue
        llm_votes = llm_by_id.loc[shared]
        human_votes = rater_by_id.loc[shared]
        results[rater] = {
            "n": int(len(shared)),
            "agreement_rate": float((llm_votes.values == human_votes.values).mean()),
            "kappa": float(cohen_kappa_score(llm_votes.values, human_votes.values)),
        }

    pooled = human_df.merge(llm_df, on="comparison_id", suffixes=("_human", "_llm"))
    if len(pooled) >= min_overlap:
        results["__pooled__"] = {
            "n": int(len(pooled)),
            "agreement_rate": float((pooled["vote_human"] == pooled["vote_llm"]).mean()),
            "kappa": float(cohen_kappa_score(pooled["vote_human"], pooled["vote_llm"])),
        }

    return results


def _elo_row_for(elo_ranking: List[Dict], identity: str) -> Dict:
    return next((r for r in elo_ranking if r["run_id"] == identity), {})


def make_plot(llm_elo: List[Dict], human_elo: List[Dict], output_dir: Path) -> Optional[str]:
    if not llm_elo and not human_elo:
        return None
    identities = sorted({r["run_id"] for r in llm_elo} | {r["run_id"] for r in human_elo})
    llm_by_id = {r["run_id"]: r for r in llm_elo}
    human_by_id = {r["run_id"]: r for r in human_elo}

    x = np.arange(len(identities))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(identities) * 1.2), 4.5))
    llm_vals = [llm_by_id.get(i, {}).get("win_rate") for i in identities]
    human_vals = [human_by_id.get(i, {}).get("win_rate") for i in identities]
    ax.bar(x - width / 2, [v if v is not None else 0 for v in llm_vals], width, label="LLM judge")
    ax.bar(x + width / 2, [v if v is not None else 0 for v in human_vals], width, label="Human (pooled)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(identities, rotation=30, ha="right")
    ax.set_ylabel("Win rate")
    ax.set_title("Pairwise win rate by candidate identity")
    ax.legend()
    fig.tight_layout()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "win_rate_by_identity.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return str(plot_path)


def write_pairwise_report_md(
    output_dir: Path,
    llm_elo: List[Dict],
    human_elo: List[Dict],
    agreement: Dict,
    n_llm: int,
    n_human: int,
    plot_path: Optional[str],
) -> None:
    lines = ["# Pairwise A/B controllability-eval report", ""]

    lines.append("## 1. Headline: does the comparison methodology recognize a perfect match?")
    lines.append("")
    lines.append(
        "`gt_self` is a literal, pixel-identical copy of the GT reference slipped in as a "
        "candidate in some comparisons (see build_controllability_pairwise_composites.py). A "
        "sound methodology should pick it as the better match far above the 50% no-signal "
        "baseline -- ideally close to 100%. If it doesn't, treat that as a methodology problem "
        "(blinding leak, confusing prompt/UI) to investigate before trusting the model rankings "
        "below, not a finding about the models themselves. **LLM-only by design**: human raters "
        "never see gt_self at all (gradio_controllability_pairwise.py excludes it) -- an expert "
        "human would trivially spot a literal copy of the reference shown right next to it, so "
        "this check only tests something for an automated judge."
    )
    lines.append("")
    llm_gt_row = _elo_row_for(llm_elo, GT_SELF_IDENTITY)
    human_gt_row = _elo_row_for(human_elo, GT_SELF_IDENTITY)
    lines.append("| Judge | gt_self win rate | gt_self Elo (± std over 200 shuffles) | n comparisons |")
    lines.append("|---|---|---|---|")
    if llm_gt_row:
        wr = f"{llm_gt_row['win_rate']:.1%}" if llm_gt_row.get("win_rate") is not None else "—"
        lines.append(f"| LLM | {wr} | {llm_gt_row.get('elo', '—')} ± {llm_gt_row.get('elo_std', '—')} | {llm_gt_row.get('n_comparisons', 0)} |")
    if human_gt_row:
        wr = f"{human_gt_row['win_rate']:.1%}" if human_gt_row.get("win_rate") is not None else "—"
        lines.append(f"| Human (pooled) | {wr} | {human_gt_row.get('elo', '—')} ± {human_gt_row.get('elo_std', '—')} | {human_gt_row.get('n_comparisons', 0)} |")
    lines.append("")

    lines.append("## 2. Full ranking by candidate identity")
    lines.append("")
    lines.append(
        "Elo here is averaged over 200 random re-orderings of the same comparisons, not a single "
        "pass. Standard Elo is a sequential rating update, so a single pass is sensitive to the "
        "arbitrary order results happen to arrive in (confirmed on this run's data: a single "
        "ordering swung one candidate's Elo by >100 points relative to this average, though "
        "win/loss/tie counts never change with order). Read `elo_std` before treating a small gap "
        "between two candidates as a real difference -- a gap smaller than the larger of the two "
        "candidates' `elo_std` values isn't distinguishable from noise, and win rate (also shown, "
        "and order-independent by construction) is the more direct number for that comparison."
    )
    lines.append("")
    for label, elo_ranking, n in (("LLM judge", llm_elo, n_llm), ("Human (pooled)", human_elo, n_human)):
        lines.append(f"### {label} ({n} comparisons)")
        lines.append("| Rank | Identity | Elo | ± std | W / L / T | Win rate |")
        lines.append("|---|---|---|---|---|---|")
        for i, r in enumerate(elo_ranking, 1):
            wr = f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "—"
            lines.append(f"| {i} | {r['run_id']} | {r['elo']} | {r.get('elo_std', '—')} | {r['wins']}/{r['losses']}/{r['ties']} | {wr} |")
        lines.append("")

    lines.append("## 3. LLM-vs-human agreement")
    lines.append("")
    if not agreement:
        lines.append("Not computed -- no overlapping (LLM, human) comparisons met --min_overlap.")
    else:
        lines.append("| Rater | n shared | Agreement rate | Kappa (unweighted) |")
        lines.append("|---|---|---|---|")
        for rater, stats in agreement.items():
            label = "**Pooled (all raters)**" if rater == "__pooled__" else rater
            lines.append(f"| {label} | {stats['n']} | {stats['agreement_rate']:.1%} | {stats['kappa']:.3f} |")
    lines.append("")

    if plot_path:
        rel = os.path.relpath(plot_path, output_dir)
        lines.append("## 4. Plots")
        lines.append("")
        lines.append(f"![Win rate by identity]({rel})")
        lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairwise_manifest", action="append", default=[], dest="pairwise_manifest", required=True, help="Path to pairwise_manifest.json (repeatable).")
    parser.add_argument("--llm_comparisons", type=str, required=True, help="Path to llm_comparisons.jsonl.")
    parser.add_argument("--human_comparisons_dir", type=str, default=None, help="Path to human_comparisons/ (the dir of per-rater .jsonl files).")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_overlap", type=int, default=3, help="Minimum shared (LLM, human) comparisons required to report agreement for a rater.")
    parser.add_argument("--elo_shuffles", type=int, default=200, help="Number of random re-orderings to average Elo over -- see compute_stable_elo's docstring for why a single pass is unreliable.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_pairwise_metadata(args.pairwise_manifest)
    llm_df = load_llm_comparisons(args.llm_comparisons)
    human_df = load_human_comparisons(args.human_comparisons_dir) if args.human_comparisons_dir else pd.DataFrame(columns=["comparison_id", "rater", "vote"])

    identities = all_identities(metadata)
    run_labels = {identity: identity for identity in identities}

    llm_elo = compute_stable_elo(to_elo_input(llm_df, metadata), run_labels, args.elo_shuffles) if not llm_df.empty else []
    human_elo = compute_stable_elo(to_elo_input(human_df, metadata), run_labels, args.elo_shuffles) if not human_df.empty else []

    agreement = compute_agreement(llm_df, human_df, args.min_overlap)

    plot_path = make_plot(llm_elo, human_elo, output_dir)

    summary = {
        "n_llm_comparisons": int(len(llm_df)),
        "n_human_comparisons": int(len(human_df)),
        "identities": identities,
        "llm_elo_ranking": llm_elo,
        "human_elo_ranking_pooled": human_elo,
        "gt_self_headline": {
            "llm": _elo_row_for(llm_elo, GT_SELF_IDENTITY),
            "human_pooled": _elo_row_for(human_elo, GT_SELF_IDENTITY),
        },
        "agreement_llm_vs_human": agreement,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_pairwise_report_md(output_dir, llm_elo, human_elo, agreement, len(llm_df), len(human_df), plot_path)

    print(f"[INFO]: Wrote {output_dir / 'summary.json'} and {output_dir / 'report.md'}.")
    llm_gt = _elo_row_for(llm_elo, GT_SELF_IDENTITY)
    if llm_gt:
        wr = llm_gt.get("win_rate")
        print(f"[INFO]: Headline -- gt_self LLM win rate: {wr:.1%} over {llm_gt.get('n_comparisons', 0)} comparisons "
              f"(chance = 50%)." if wr is not None else "[INFO]: gt_self had no LLM comparisons.")


if __name__ == "__main__":
    main()
