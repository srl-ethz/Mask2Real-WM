"""Controllability numbers of the paper, from the released judge and rater files.

Writes to --out_dir:
  ab_win_rates.{csv,md}           human A/B win rate per model (ties count as non-wins) with 95%
                                  Wilson interval and stable Elo (200 shuffles), main study plus
                                  Mono-R follow-up
  head_to_head.{csv,md}           exact binomial tests between the four pool models, LLM judge and
                                  main-study raters who completed the survey; Bonferroni over 12
  ablation_head_to_head.{csv,md}  Mono-R against each other model, LLM judge and follow-up raters
                                  who completed the survey; Bonferroni over the 20 tests
  agreement.{csv,md}              agreement rate and unweighted Cohen's kappa, LLM vs. each main-study
                                  rater and between main-study raters
  sine_scores.{csv,md}            sine-sweep rating (0 / 0.5 / 1) per model: mean and standard error
  wm1_iou.{csv,md}                WM1 hand-mask IoU per model (both cameras)
  table_controllability.tex       the paper's controllability table

Example:
    python scripts/paper/compute_paper_controllability_tables.py --out_dir results/controllability
"""

import argparse
import csv
import glob
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compute_controllability_pairwise_stats import compute_stable_elo  # noqa: E402

FR = "inference_output/full_run_results"
POOL = ["wm1_real_only", "wm1_midtrain_only", "wm1_midtrain_lora45000", "baseline_midtrain_lora"]
MONO_R = "baseline_real_only"
PAPER_NAME = {
    "wm1_real_only": "Cascade-R", "wm1_midtrain_only": "Cascade-S", "wm1_midtrain_lora45000": "Cascade-SR",
    "baseline_midtrain_lora": "Mono-SR", "baseline_real_only": "Mono-R", "gt_self": "gt_self",
}
SINE_KEY_TO_IDENTITY = {
    "wm1_realonly_wm2": "wm1_real_only", "wm1_midtrain_wm2": "wm1_midtrain_only",
    "wm1_midtrain_lora_wm2": "wm1_midtrain_lora45000", "baseline_midtrain_lora": "baseline_midtrain_lora",
    "baseline_real_only": "baseline_real_only",
}
TABLE_ORDER = ["baseline_real_only", "baseline_midtrain_lora", "wm1_real_only", "wm1_midtrain_only", "wm1_midtrain_lora45000"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pairs(manifest_paths):
    pairs = {}
    for path in manifest_paths:
        with open(path) as f:
            for row in json.load(f)["results"]:
                pairs[row["comparison_id"]] = (row["left_identity"], row["right_identity"])
    return pairs


def load_votes(votes_dir):
    """rater -> list of vote rows; the rater is the file's name."""
    return {os.path.basename(p)[:-len(".jsonl")]: read_jsonl(p) for p in sorted(glob.glob(f"{votes_dir}/*.jsonl"))}


def completed(votes_by_rater, subset_path):
    with open(subset_path) as f:
        ids = {t["comparison_id"] for t in json.load(f)["trials"]}
    return {r for r, rows in votes_by_rater.items() if {x["comparison_id"] for x in rows} >= ids}


def elo_pairs(rows, pairs):
    vote_map = {"A": "left", "B": "right", "tie": "tie"}
    return [{"left_run_id": pairs[r["comparison_id"]][0], "right_run_id": pairs[r["comparison_id"]][1],
             "vote": vote_map[r["vote"]]} for r in rows]


def head_to_head(rows, pairs, a, b):
    wins_a = wins_b = ties = 0
    for r in rows:
        left, right = pairs[r["comparison_id"]]
        if {left, right} != {a, b}:
            continue
        if r["vote"] == "tie":
            ties += 1
        elif (left if r["vote"] == "A" else right) == a:
            wins_a += 1
        else:
            wins_b += 1
    return wins_a, wins_b, ties


def test_row(judge, a, b, counts, n_tests):
    wins_a, wins_b, ties = counts
    decisive = wins_a + wins_b
    p = binomtest(wins_a, decisive, 0.5).pvalue if decisive else float("nan")
    return {"model_a": PAPER_NAME[a], "model_b": PAPER_NAME[b], "judge": judge, "a_wins": wins_a, "b_wins": wins_b,
            "ties": ties, "a_win_rate_decisive": wins_a / decisive if decisive else float("nan"),
            "p_value": p, "significant_bonferroni": bool(p < 0.05 / n_tests)}


def write_table(rows, out_dir, name, title, note=""):
    with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = [f"# {title}", ""] + ([note, ""] if note else [])
    lines += ["| " + " | ".join(rows[0].keys()) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(fmt(v) for v in row.values()) + " |" for row in rows]
    with open(os.path.join(out_dir, f"{name}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise_manifests", nargs="+", default=[
        f"{FR}/pairwise_composites/pairwise_manifest.json",
        f"{FR}/pairwise_composites/pairwise_manifest_baseline_real_only_addendum.json"])
    parser.add_argument("--llm_comparisons", default=f"{FR}/llm_comparisons.jsonl")
    parser.add_argument("--main_votes_dir", default=f"{FR}/human_comparisons/human_comparisons")
    parser.add_argument("--main_subset", default=f"{FR}/human_comparisons/subset.json")
    parser.add_argument("--followup_votes_dir", default=f"{FR}/human_comparisons_baseline_real_only_followup/human_comparisons")
    parser.add_argument("--followup_subset", default=f"{FR}/human_comparisons/subset_baseline_real_only_followup.json")
    parser.add_argument("--sine_ratings_dir", default="inference_output/sine_actions_human_study/human_ratings")
    parser.add_argument("--sine_blinding_key", default="inference_output/sine_actions_human_study/blinding_key.json")
    parser.add_argument("--iou_csv", default="inference_output/wm1_controllability_seg_iou/wm1_iou_per_trial.csv")
    parser.add_argument("--out_dir", default="results/controllability")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pairs = load_pairs(args.pairwise_manifests)
    llm = read_jsonl(args.llm_comparisons)
    main_votes = load_votes(args.main_votes_dir)
    followup_votes = load_votes(args.followup_votes_dir)
    main_complete = completed(main_votes, args.main_subset)
    followup_complete = completed(followup_votes, args.followup_subset)

    # A/B win rate and Elo: every main-study vote plus the follow-up raters who completed it.
    human_rows = [r for rows in main_votes.values() for r in rows]
    human_rows += [r for rater in sorted(followup_complete) for r in followup_votes[rater]]  # fixed order: Elo is order-dependent
    identities = sorted({i for r in human_rows for i in pairs[r["comparison_id"]]})
    elo = {row["run_id"]: row for row in compute_stable_elo(elo_pairs(human_rows, pairs), {i: i for i in identities})}
    ab_rows = []
    for identity in TABLE_ORDER:
        e = elo[identity]
        n = e["n_comparisons"]
        ci = binomtest(e["wins"], n).proportion_ci(confidence_level=0.95, method="wilson")
        ab_rows.append({"model": PAPER_NAME[identity], "identity": identity, "wins": e["wins"], "losses": e["losses"],
                        "ties": e["ties"], "n": n, "win_rate": e["wins"] / n, "wilson_low": ci.low,
                        "wilson_high": ci.high, "elo": e["elo"], "elo_std": e["elo_std"]})
    write_table(ab_rows, args.out_dir, "ab_win_rates", "Human A/B win rate and Elo",
                f"{len(human_rows)} human votes: all main-study votes plus the {len(followup_complete)} follow-up "
                "raters who completed the follow-up survey. Ties count as non-wins; 95% Wilson interval.")

    # Head-to-head among the pool models: LLM, and main-study raters who completed the survey.
    llm_by_id = {r["comparison_id"]: r for r in llm}
    main_complete_rows = [r for rater in sorted(main_complete) for r in main_votes[rater]]
    h2h = []
    for a, b in itertools.combinations(POOL, 2):
        h2h.append(test_row("LLM", a, b, head_to_head(llm, pairs, a, b), 12))
        h2h.append(test_row("Human", a, b, head_to_head(main_complete_rows, pairs, a, b), 12))
    write_table(h2h, args.out_dir, "head_to_head", "Head-to-head significance among the pool models",
                f"Exact two-sided binomial test on decisive votes; Bonferroni over 12 tests (alpha = {0.05 / 12:.5f}). "
                f"Human: {len(main_complete)} main-study raters who completed the survey, {len(main_complete_rows)} votes.")

    # Mono-R ablation: LLM, and follow-up raters who completed the survey.
    followup_rows = [r for rater in sorted(followup_complete) for r in followup_votes[rater]]
    ablation = []
    for opponent in POOL[::-1]:
        ablation.append(test_row("LLM", MONO_R, opponent, head_to_head(llm, pairs, MONO_R, opponent), 20))
        ablation.append(test_row("Human", MONO_R, opponent, head_to_head(followup_rows, pairs, MONO_R, opponent), 20))
    write_table(ablation, args.out_dir, "ablation_head_to_head", "Mono-R against the other models",
                f"Exact two-sided binomial test on decisive votes; Bonferroni over the 20 tests of this and the "
                f"head-to-head table (alpha = {0.05 / 20:.4f}). Human: {len(followup_complete)} follow-up raters who "
                f"completed the survey, {len(followup_rows)} votes.")

    # Agreement: LLM vs. each main-study rater, and between main-study raters.
    agreement = []
    pooled_llm, pooled_human = [], []
    for rater, rows in main_votes.items():
        shared = [(llm_by_id[r["comparison_id"]]["vote"], r["vote"]) for r in rows if r["comparison_id"] in llm_by_id]
        x, y = zip(*shared)
        pooled_llm += x
        pooled_human += y
        if len(shared) < 3:  # kappa is undefined on a single vote; the votes still count in the pooled row
            continue
        agreement.append({"pair": f"LLM vs {rater}", "n_shared": len(shared),
                          "agreement": float(np.mean(np.array(x) == np.array(y))), "kappa": cohen_kappa_score(x, y)})
    agreement.append({"pair": "LLM vs all raters (pooled)", "n_shared": len(pooled_llm),
                      "agreement": float(np.mean(np.array(pooled_llm) == np.array(pooled_human))),
                      "kappa": cohen_kappa_score(pooled_llm, pooled_human)})
    by_rater = {rater: {r["comparison_id"]: r["vote"] for r in rows} for rater, rows in main_votes.items()}
    for r1, r2 in itertools.combinations(sorted(by_rater), 2):
        shared_ids = sorted(set(by_rater[r1]) & set(by_rater[r2]))
        if len(shared_ids) < 3:
            continue
        x = [by_rater[r1][c] for c in shared_ids]
        y = [by_rater[r2][c] for c in shared_ids]
        agreement.append({"pair": f"{r1} vs {r2}", "n_shared": len(shared_ids),
                          "agreement": float(np.mean(np.array(x) == np.array(y))), "kappa": cohen_kappa_score(x, y)})
    write_table(agreement, args.out_dir, "agreement", "Agreement between judges (main study)",
                "Votes A / B / tie on the same comparisons; unweighted Cohen's kappa.")

    # Sine-sweep ratings.
    with open(args.sine_blinding_key) as f:
        slot_to_identity = {slot: SINE_KEY_TO_IDENTITY[v["model_key"]] for slot, v in json.load(f).items()}
    scores = defaultdict(list)
    for path in sorted(glob.glob(f"{args.sine_ratings_dir}/*.jsonl")):
        for r in read_jsonl(path):
            identity = slot_to_identity[r["model_slot"]]
            scores[(identity, "all")].append(r["answer"])
            scores[(identity, r["dataset"])].append(r["answer"])
    sine_rows = []
    for identity in TABLE_ORDER:
        for split in ("all", "id", "ood_no_cube"):
            values = np.array(scores.get((identity, split), []), dtype=float)
            if len(values):
                sine_rows.append({"model": PAPER_NAME[identity], "split": split, "n_ratings": len(values),
                                  "mean": values.mean(), "sem": values.std(ddof=1) / np.sqrt(len(values))})
    write_table(sine_rows, args.out_dir, "sine_scores", "Sine-sweep rating (0 / 0.5 / 1)",
                "Mean and standard error over all ratings, per model and split.")

    # WM1 hand-mask IoU (final frame, both cameras).
    iou = defaultdict(list)
    with open(args.iou_csv) as f:
        for r in csv.DictReader(f):
            iou[(r["model_variant"], "all")].append(float(r["iou_hand"]))
            iou[(r["model_variant"], r["split"])].append(float(r["iou_hand"]))
    iou_rows = [{"model": PAPER_NAME[m], "split": s, "n": len(iou[(m, s)]), "mean_iou": float(np.mean(iou[(m, s)]))}
                for m in ("wm1_real_only", "wm1_midtrain_only", "wm1_midtrain_lora45000") for s in ("all", "id", "ood")]
    write_table(iou_rows, args.out_dir, "wm1_iou", "WM1 hand-mask IoU", "Final rollout frame, both cameras.")

    # The paper's table.
    sine_all = {r["model"]: r for r in sine_rows if r["split"] == "all"}
    iou_all = {r["model"]: r["mean_iou"] for r in iou_rows if r["split"] == "all"}
    tex = [r"\begin{tabular}{lccc}", r"\toprule", r"Model & A/B win rate & Sine-sweep score & WM1 IoU \\", r"\midrule"]
    for r in ab_rows:
        s = sine_all[r["model"]]
        iou_cell = f"{iou_all[r['model']]:.3f}" if r["model"] in iou_all else "--"
        tex.append(f"{r['model']} & {100 * r['win_rate']:.1f}\\% [{100 * r['wilson_low']:.1f}, {100 * r['wilson_high']:.1f}] & "
                   f"{s['mean']:.2f} $\\pm$ {s['sem']:.2f} & {iou_cell} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(args.out_dir, "table_controllability.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"wrote controllability tables to {args.out_dir}")


if __name__ == "__main__":
    main()
