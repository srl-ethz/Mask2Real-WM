"""Human pairwise A/B comparison game for the controllability-eval pipeline (v2).

Scoped the same way as v1's gradio_controllability_rating.py (see that file's
module docstring): a small, time-boxed study, up to ~10 named raters, each in
one sitting under an hour. All raters see the SAME shared subset (not a
partition), to maximize overlap for inter-rater agreement.

Task shape differs from v1: instead of scoring one generated end-state
against GT on a continuous scale, each item here shows the GT reference
(RGB + segmentation mask) alongside two anonymous candidates -- "Model A" and
"Model B" -- and the rater picks whichever better matches, or calls it a tie.

Unlike the LLM judge (llm_judge_controllability_pairwise.py), the synthetic
"gt_self" identity -- the GT's own render slipped in as a "candidate", used
there as a built-in validity check on whether an automated judge can be
fooled -- is deliberately EXCLUDED from every comparison a human sees
(build_subset filters it out). An expert human rater would immediately
recognize a candidate that's a literal, pixel-identical copy of the reference
shown right next to it; showing it would test nothing and would just read as
a confusing or condescending item. gt_self therefore has no human ratings at
all, by design -- see compute_controllability_pairwise_stats.py's report,
which only reports its Elo/win-rate from the LLM side.

The subset unit is a whole (trial_group_id, approach_style) *target*, not an
individual comparison: each selected target contributes all 6 of its
non-gt_self pairwise comparisons (every one of the 4 real model variants vs.
every other), so every rater's session gets a complete local signal per
target they see, not a scattered sample of unrelated pairs.

Blinding: raters never see comparison_id, real model identity, or a
filename anywhere in the UI -- "Model A"/"Model B" are randomly reassigned
per comparison (baked into the composite image at build time, not decided by
this app). No source rollout video is shown here (unlike v1): the GT-self
candidate has no rollout video, and showing one for 3 of 4 slots but not the
4th would itself be a blinding leak. The done-screen recap is a plain vote
tally (A/B/tie counts) with no per-candidate breakdown, since any breakdown
meaningful enough to be useful would necessarily expose real model/gt_self
identity -- there is no stable "Model A" identity across comparisons to
tabulate that isn't also the real underlying identity. (v1's own done-screen
recap already excludes model_variant/trial_id for the same reason; this is
that same rule taken to its conclusion for a pairwise task shape.)

Usage:
    # 1. Build a shared subset once (20 targets x 6 non-gt_self pairs = 120
    #    comparisons by default), producing subset.json + blinding_key.json:
    python scripts/gradio_controllability_pairwise.py --mode build_subset \\
        --pairwise_manifest inference_output/.../pairwise_composites/pairwise_manifest.json \\
        --output_dir inference_output/.../human_comparisons --num_trial_groups 20

    # 2. Launch the comparison app against that subset:
    python scripts/gradio_controllability_pairwise.py --mode serve \\
        --subset_path inference_output/.../human_comparisons/subset.json \\
        --output_dir inference_output/.../human_comparisons
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

SECONDS_PER_ITEM_ESTIMATE = 20  # a single forced choice -- simpler than v1's slider+dropdown;
# only used before a rater has any timing data of their own.

GT_SELF_IDENTITY = "gt_self"

VOTE_CHOICES = ["Model A is closer", "Tie / can't tell", "Model B is closer"]
_CHOICE_TO_VOTE = {"Model A is closer": "A", "Tie / can't tell": "tie", "Model B is closer": "B"}

# Shown on both the login screen and every comparison screen -- panel names here are kept
# identical to build_controllability_pairwise_composites.py's PAIRWISE_PANEL_LABELS so the
# wording in the image and the wording on screen match exactly.
EXPLANATION_MD = (
    "For each comparison you'll see one image and vote on which of two anonymous candidates -- "
    '"Model A" and "Model B" -- better reached the target hand pose.\n\n'
    "**The comparison image has four panels, left to right:**\n"
    "1. **GT reference: RGB** -- the goal pose, rendered in the physics simulator.\n"
    "2. **GT reference: mask (green=hand, blue=arm)** -- a standalone silhouette of panel 1's "
    "target: green pixels are the target hand pose, blue pixels are the Franka arm (marked only "
    "so you don't mistake it for the hand -- it isn't part of what you're comparing).\n"
    "3. **Model A** and 4. **Model B** -- two candidate outcomes. Which one's hand shape/position "
    "more closely matches panels 1-2?\n\n"
    '"Model A"/"Model B" are randomly reassigned for every comparison -- there is no consistent '
    "identity behind either label from one image to the next, so please judge each comparison "
    "independently.\n\n"
    "You'll see the same panels 1-2 reference repeated for several comparisons in a row before it "
    "changes -- that's expected, not a stuck screen: several model pairs are compared against the "
    "same target before moving to the next one."
)


def _load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_subset(
    pairwise_manifest_paths: List[str],
    num_trial_groups: int,
    seed: int,
    scope_filter: Optional[str] = None,
    exclude_groups: Optional[set] = None,
    component_min: Optional[int] = None,
    component_max: Optional[int] = None,
    stratify_by_component: bool = False,
) -> List[Dict]:
    """Selects whole (trial_group_id, approach_style) targets, then includes all 6 of their
    non-gt_self pairwise comparisons (every one of the 4 real model variants vs. every other) --
    not a flat shuffle+cap over individual comparisons. This is what guarantees every rater's
    session sees a complete local signal for each target they're shown, rather than a scattered
    sample of unrelated pairs. gt_self comparisons are excluded entirely (see module docstring --
    an expert human would trivially spot a literal copy of the reference, so that check is
    reserved for the LLM judge). Deterministic shuffle of targets, then capped to num_trial_groups.

    scope_filter, when given, restricts candidates to that sampling_scope before selecting --
    e.g. "whole_pose" to specifically add multi-finger-articulation-heavy targets: whole_pose is
    the only existing scope that moves several finger joints at once (all 23 action dims move
    together), since per_dim only ever moves exactly one. No new targets/GT/rollouts are generated
    for this -- targets for every scope already exist across the full run and are already fully
    processed (composites built, LLM-judged), so this just pulls more of them into the
    human-facing subset. exclude_groups, when given, drops targets already in that set before
    selecting -- used by --extend_subset (see cmd_build_subset) so re-running this doesn't
    reselect targets already in an existing subset, preserving in-progress raters' resumability.
    component_min/component_max, when given, further restrict per_dim candidates to that
    (inclusive) action-dim index range -- e.g. 7-22 for finger joints only (indices 0-5 are the
    arm pose, 6 is the hand's own wrist-rotation joint, 7-22 are the 16 actual finger joints
    across thumb + the four fingers; see inference_wm1_to_wm2_controllability_eval.py's
    HAND_JOINT_LIMITS_RAD comment for the exact per-index mapping). No-op for whole_pose
    candidates, which have component=None.
    stratify_by_component, when True, distributes num_trial_groups as evenly as possible across
    the distinct `component` values among candidates (e.g. one row per finger joint) instead of a
    flat uniform draw over all candidates. This matters in practice: a flat draw of 30 from 320
    available finger per_dim targets left 4 of the 16 joints with zero coverage and others with up
    to 5, purely by chance -- stratifying guarantees every joint present in the candidate pool
    gets at least floor(num_trial_groups / n_joints) picks, with the remainder distributed to a
    randomly-chosen subset of joints (not always the same ones) so nobody's systematically
    favored. No-op (falls back to the flat draw) if every candidate has component=None (e.g. a
    whole_pose-only pool, which has no single component to stratify by).
    """
    comparisons: List[Dict] = []
    for path in pairwise_manifest_paths:
        comparisons.extend(_load_manifest(path)["results"])

    comparisons = [
        c for c in comparisons
        if GT_SELF_IDENTITY not in (c["left_identity"], c["right_identity"])
    ]
    if scope_filter:
        comparisons = [c for c in comparisons if c["sampling_scope"] == scope_filter]
    if component_min is not None or component_max is not None:
        lo = component_min if component_min is not None else float("-inf")
        hi = component_max if component_max is not None else float("inf")
        comparisons = [c for c in comparisons if c["component"] is not None and lo <= c["component"] <= hi]

    group_keys = sorted({(c["trial_group_id"], c["approach_style"]) for c in comparisons})
    if exclude_groups:
        group_keys = [k for k in group_keys if k not in exclude_groups]
    rng = random.Random(seed)

    component_by_group: Dict[Tuple[str, str], Optional[int]] = {}
    for c in comparisons:
        key = (c["trial_group_id"], c["approach_style"])
        if key not in component_by_group:
            component_by_group[key] = c["component"]
    strata_present = {component_by_group[k] for k in group_keys}

    if stratify_by_component and strata_present != {None}:
        groups_by_stratum: Dict[Optional[int], List[Tuple[str, str]]] = {}
        for k in group_keys:
            groups_by_stratum.setdefault(component_by_group[k], []).append(k)
        for pool in groups_by_stratum.values():
            rng.shuffle(pool)

        stratum_order = sorted(groups_by_stratum.keys(), key=lambda s: (s is None, s))
        rng.shuffle(stratum_order)  # which strata get the "remainder" extra pick is randomized, not fixed

        selected_list: List[Tuple[str, str]] = []
        next_idx = {s: 0 for s in stratum_order}
        while len(selected_list) < num_trial_groups:
            progressed = False
            for s in stratum_order:
                if len(selected_list) >= num_trial_groups:
                    break
                i = next_idx[s]
                if i < len(groups_by_stratum[s]):
                    selected_list.append(groups_by_stratum[s][i])
                    next_idx[s] += 1
                    progressed = True
            if not progressed:
                break  # every stratum exhausted -- fewer candidates available than num_trial_groups
        selected_keys = set(selected_list)
    else:
        shuffled_keys = group_keys[:]
        rng.shuffle(shuffled_keys)
        selected_keys = set(shuffled_keys[:num_trial_groups])

    subset = [c for c in comparisons if (c["trial_group_id"], c["approach_style"]) in selected_keys]
    subset.sort(key=lambda c: (c["trial_group_id"], c["approach_style"], c["comparison_id"]))
    return subset


def cmd_build_subset(args: argparse.Namespace) -> None:
    existing_trials: List[Dict] = []
    exclude_groups = None
    if args.extend_subset:
        existing = _load_manifest(args.extend_subset)
        existing_trials = existing["trials"]
        exclude_groups = {(t["trial_group_id"], t["approach_style"]) for t in existing_trials}

    new_trials = build_subset(
        args.pairwise_manifest, args.num_trial_groups, args.seed, args.scope_filter, exclude_groups,
        args.component_min, args.component_max, args.stratify_by_component,
    )
    if not new_trials and not existing_trials:
        raise SystemExit("No comparisons found in the given --pairwise_manifest file(s).")

    subset = existing_trials + new_trials
    subset.sort(key=lambda c: (c["trial_group_id"], c["approach_style"], c["comparison_id"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_path = output_dir / "subset.json"
    with open(subset_path, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "trials": subset}, f, indent=2)

    blinding_key = {
        c["comparison_id"]: {
            "left_identity": c["left_identity"],
            "right_identity": c["right_identity"],
            "trial_group_id": c["trial_group_id"],
        }
        for c in subset
    }
    blinding_key_path = output_dir / "blinding_key.json"
    with open(blinding_key_path, "w", encoding="utf-8") as f:
        json.dump(blinding_key, f, indent=2)

    n_groups = len({(c["trial_group_id"], c["approach_style"]) for c in subset})
    est_minutes = len(subset) * SECONDS_PER_ITEM_ESTIMATE / 60
    action = "Extended to a" if args.extend_subset else "Built a"
    print(f"[INFO]: {action} {n_groups}-target ({len(subset)}-comparison) subset -> {subset_path}")
    print(f"[INFO]: Blinding key (not shown to raters) -> {blinding_key_path}")
    print(f"[INFO]: Estimated ~{est_minutes:.0f} min total at ~{SECONDS_PER_ITEM_ESTIMATE}s/comparison -- "
          f"raters can split this across multiple sittings by resuming with the same name; "
          f"reduce --num_trial_groups if a single sitting matters more than full coverage.")


def _rater_order(subset: List[Dict], username: str, seed: int) -> List[int]:
    """Shuffles which target (trial_group_id, approach_style) a rater sees next, but keeps a
    target's 6 pairwise comparisons together and in a fixed order once reached -- so a rater's
    mental model of one target persists across its comparisons instead of being reloaded every
    item. Deterministic per (seed, username): re-opening with the same name resumes in the same
    order, same contract as v1's _rater_order.
    """
    group_keys = sorted({(c["trial_group_id"], c["approach_style"]) for c in subset})
    rng = random.Random(f"{seed}:{username}")
    rng.shuffle(group_keys)

    indices_by_group: Dict[Tuple[str, str], List[int]] = {}
    for i, c in enumerate(subset):
        indices_by_group.setdefault((c["trial_group_id"], c["approach_style"]), []).append(i)

    order: List[int] = []
    for key in group_keys:
        order.extend(indices_by_group[key])
    return order


def _comparisons_path(output_dir: Path, username: str) -> Path:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in username.strip())
    return output_dir / "human_comparisons" / f"{safe_name}.jsonl"


def _load_voted_comparison_ids(comparisons_path: Path) -> set:
    voted = set()
    if comparisons_path.exists():
        with open(comparisons_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    voted.add(json.loads(line)["comparison_id"])
    return voted


def _load_own_vote_summary(comparisons_path: Path) -> str:
    """Plain vote-count tally only (A/B/tie counts), no per-candidate breakdown -- see module
    docstring for why a per-candidate leaderboard isn't offered here (it would necessarily expose
    real model/gt_self identity to be meaningful)."""
    counts = {"A": 0, "B": 0, "tie": 0}
    if comparisons_path.exists():
        with open(comparisons_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vote = json.loads(line).get("vote")
                if vote in counts:
                    counts[vote] += 1
    total = sum(counts.values())
    if total == 0:
        return ""
    return (
        f"You made {total} comparisons: picked the first-shown candidate {counts['A']} time(s), "
        f"the second-shown candidate {counts['B']} time(s), and called it a tie {counts['tie']} time(s)."
    )


def build_app(subset: List[Dict], output_dir: Path, seed: int) -> gr.Blocks:
    with gr.Blocks(title="Controllability pairwise comparison") as demo:
        username_state = gr.State("")
        order_state = gr.State([])  # list of indices into `subset`, already filtered to unvoted
        position_state = gr.State(0)
        session_start_state = gr.State(0.0)
        item_shown_at_state = gr.State(0.0)

        est_minutes = len(subset) * SECONDS_PER_ITEM_ESTIMATE / 60
        with gr.Column(visible=True) as login_col:
            gr.Markdown(
                "# Controllability pairwise comparison\n"
                + EXPLANATION_MD
                + f"\n\nThere's no time limit -- this set is about {est_minutes:.0f} min at a "
                "realistic pace. Feel free to split it across multiple sittings; re-enter the "
                "same name to resume exactly where you left off."
            )
            username_box = gr.Textbox(label="Your name", placeholder="e.g. alice")
            start_btn = gr.Button("Start", variant="primary")

        with gr.Column(visible=False) as rating_col:
            progress_md = gr.Markdown()
            gr.Markdown(EXPLANATION_MD)
            composite_image = gr.Image(label="Comparison", interactive=False)
            vote_radio = gr.Radio(VOTE_CHOICES, value=None, label="Which better matches the GT reference?")
            submit_btn = gr.Button("Submit and next", variant="primary")

        with gr.Column(visible=False) as done_col:
            gr.Markdown(
                "# Done -- thank you!\n"
                "Your comparisons have been saved. You can close this tab."
            )
            done_summary_md = gr.Markdown()

        trial_by_id: Dict[str, Dict] = {c["comparison_id"]: c for c in subset}

        def _current_item(order: List[int], position: int) -> Optional[Dict]:
            if position >= len(order):
                return None
            return subset[order[position]]

        def _render_item(order: List[int], position: int, session_start: float, comparisons_path: Path):
            item = _current_item(order, position)
            if item is None:
                summary_text = _load_own_vote_summary(comparisons_path)
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(value=None, visible=False),
                    "",
                    gr.update(value=None),
                    time.time(),
                    summary_text,
                )
            elapsed = time.time() - session_start
            done = position
            remaining = len(order) - position
            pace = (elapsed / done) if done > 0 else SECONDS_PER_ITEM_ESTIMATE
            eta_min = max(0.0, remaining * pace / 60.0)
            progress_text = f"**Comparison {position + 1} of {len(order)}** -- about {eta_min:.0f} min left at your current pace."
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=item["composite_path"], visible=True),
                progress_text,
                gr.update(value=None),
                time.time(),
                "",
            )

        def on_start(username: str):
            username = (username or "").strip()
            if not username:
                gr.Warning("Please enter a name first.")
                return (
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    "", [], 0, 0.0, None, "", gr.update(value=None), 0.0, "",
                )
            comparisons_path = _comparisons_path(output_dir, username)
            already_voted = _load_voted_comparison_ids(comparisons_path)
            full_order = _rater_order(subset, username, seed)
            order = [i for i in full_order if subset[i]["comparison_id"] not in already_voted]

            session_start = time.time()
            rating_visible, done_visible, img, progress_text, vote_upd, shown_at, summary_text = _render_item(
                order, 0, session_start, comparisons_path
            )
            return (
                gr.update(visible=False), rating_visible, done_visible,
                username, order, 0, session_start, img, progress_text, vote_upd, shown_at, summary_text,
            )

        def on_submit(username: str, order: List[int], position: int, session_start: float, item_shown_at: float, vote_choice: Optional[str]):
            item = _current_item(order, position)
            if item is None:
                return gr.update(), gr.update(), position, gr.update(), gr.update(), gr.update(), item_shown_at, gr.update()

            vote = _CHOICE_TO_VOTE.get(vote_choice)
            if vote is None:
                gr.Warning("Please pick one before submitting.")
                return gr.update(), gr.update(), position, gr.update(), gr.update(), gr.update(), item_shown_at, gr.update()

            comparisons_path = _comparisons_path(output_dir, username)
            comparisons_path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "comparison_id": item["comparison_id"],
                "vote": vote,
                "rater": username,
                "rated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds_spent": round(time.time() - item_shown_at, 1),
            }
            with open(comparisons_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            next_position = position + 1
            rating_visible, done_visible, img, progress_text, vote_upd, shown_at, summary_text = _render_item(
                order, next_position, session_start, comparisons_path
            )
            return rating_visible, done_visible, next_position, img, progress_text, vote_upd, shown_at, summary_text

        start_btn.click(
            on_start,
            inputs=[username_box],
            outputs=[
                login_col, rating_col, done_col,
                username_state, order_state, position_state, session_start_state,
                composite_image, progress_md, vote_radio, item_shown_at_state,
                done_summary_md,
            ],
            api_name="start",
        )

        submit_btn.click(
            on_submit,
            inputs=[username_state, order_state, position_state, session_start_state, item_shown_at_state, vote_radio],
            outputs=[rating_col, done_col, position_state, composite_image, progress_md, vote_radio, item_shown_at_state, done_summary_md],
            api_name="submit",
        )

    return demo


def cmd_serve(args: argparse.Namespace) -> None:
    subset_data = _load_manifest(args.subset_path)
    subset = subset_data["trials"]
    seed = subset_data.get("seed", 0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demo = build_app(subset, output_dir, seed)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", type=str, choices=["build_subset", "serve"], required=True)
    parser.add_argument("--pairwise_manifest", action="append", default=[], dest="pairwise_manifest", help="Path to pairwise_manifest.json (--mode build_subset; repeatable).")
    parser.add_argument(
        "--num_trial_groups", type=int, default=20,
        help="--mode build_subset: number of (trial_group_id, approach_style) targets to include "
        "-- each contributes its 6 non-gt_self pairwise comparisons (gt_self is excluded from "
        "every human-facing comparison; see module docstring). Sized for <1hr at ~20s/comparison "
        "(20 targets x 6 pairs = 120 comparisons =~ 40 min).",
    )
    parser.add_argument("--seed", type=int, default=0, help="--mode build_subset: subset selection seed; also used to derive each rater's per-user target order.")
    parser.add_argument(
        "--scope_filter", type=str, default=None, choices=["whole_pose", "per_dim"],
        help="--mode build_subset: restrict candidates to this sampling_scope before selecting "
        "-- e.g. 'whole_pose' to add multi-finger-articulation-heavy targets (the only existing "
        "scope that moves several finger joints at once; per_dim moves exactly one). No new "
        "targets/GT/rollouts are generated -- this only selects among what's already been "
        "through the full pipeline. Omit for no filtering (any scope).",
    )
    parser.add_argument(
        "--extend_subset", type=str, default=None,
        help="--mode build_subset: path to an existing subset.json to extend rather than replace "
        "-- its targets are kept as-is (preserving any in-progress raters' resumability) and "
        "--num_trial_groups new ones are added on top, excluding anything already present.",
    )
    parser.add_argument(
        "--component_min", type=int, default=None,
        help="--mode build_subset: with --scope_filter per_dim, only include targets whose "
        "action-dim index is >= this (inclusive). E.g. 7 for finger joints only (0-5 are the arm "
        "pose, 6 is the hand's own wrist-rotation joint, 7-22 are the 16 finger joints).",
    )
    parser.add_argument(
        "--component_max", type=int, default=None,
        help="--mode build_subset: with --scope_filter per_dim, only include targets whose "
        "action-dim index is <= this (inclusive). E.g. 22 for finger joints only (paired with "
        "--component_min 7).",
    )
    parser.add_argument(
        "--stratify_by_component", action="store_true", default=False,
        help="--mode build_subset: distribute --num_trial_groups as evenly as possible across "
        "every distinct action-dim (e.g. one row per finger joint) instead of a flat uniform draw "
        "-- guarantees coverage of every joint present in the candidate pool rather than leaving "
        "some to chance. No-op if candidates have no single component (e.g. whole_pose).",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where subset.json/blinding_key.json live (build_subset) and human_comparisons/ is written (serve).")
    parser.add_argument("--subset_path", type=str, help="--mode serve: path to subset.json from a prior build_subset run.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False, help="Create a public Gradio share link (off by default).")
    args = parser.parse_args()

    if args.mode == "build_subset":
        if not args.pairwise_manifest:
            raise SystemExit("--pairwise_manifest is required for --mode build_subset (repeatable).")
        cmd_build_subset(args)
    else:
        if not args.subset_path:
            raise SystemExit("--subset_path is required for --mode serve.")
        cmd_serve(args)


if __name__ == "__main__":
    main()
