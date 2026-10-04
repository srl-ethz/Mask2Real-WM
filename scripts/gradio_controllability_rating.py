"""Human rating game for the controllability-eval pipeline.

Scoped as its own small, time-boxed study (see the controllability-eval plan):
up to ~10 named raters, each in one sitting under an hour, rating a shared
fixed subset of composites (from build_controllability_composites.py) on a
continuous 0.00-1.00 slider. All raters see the SAME subset (not a partition)
to maximize overlap for inter-rater reliability and for the Spearman check
against the LLM judge (Phase 4) -- see compute_controllability_stats.py.

Blinding: raters never see trial_id, model_variant, or a filename anywhere in
the UI -- items are shown in a per-rater-seeded shuffled order under a plain
"Item N of M" label. The trial_id -> model_variant mapping is written once to
blinding_key.json and is not read by this app after that (only by the stats
script, later). The composite PNGs themselves must also not leak model_variant
in their pixels -- build_controllability_composites.py's title strip was
fixed to exclude it (was baked in as "model=wm1_wm2" until 2026-09-05; a
vision-based judge or a rater who reads the image text could otherwise see it
directly, even if the surrounding UI hides it).

Usage:
    # 1. Build a shared subset once (stratified sample across the composites
    #    you want raters to see), producing subset.json + blinding_key.json:
    python scripts/gradio_controllability_rating.py --mode build_subset \\
        --composite_manifest inference_output/.../composites/composite_manifest.json \\
        --output_dir inference_output/.../human_ratings --subset_size 100

    # 2. Launch the rating app against that subset:
    python scripts/gradio_controllability_rating.py --mode serve \\
        --subset_path inference_output/.../human_ratings/subset.json \\
        --output_dir inference_output/.../human_ratings
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

FAILURE_MODES = [
    "none",
    "no_response",
    "coupled_articulation",
    "undershoot",
    "overshoot",
    "wrong_direction",
    "visual_artifact_or_collapse",
    "other",
]

SECONDS_PER_ITEM_ESTIMATE = 25  # only used before a rater has any timing data of their own

# Shown on both the login screen and every rating screen (not just once up front) --
# panel names here are kept identical to build_controllability_composites.py's
# PANEL_LABELS so the wording in the image and the wording on screen match exactly.
EXPLANATION_MD = (
    "For each item you'll see a comparison image (and the generated video) and rate, "
    "on a 0.00-1.00 slider, how closely the final generated frame reached the target "
    "hand pose shown in green in panel 3.\n\n"
    "**The comparison image has three panels, left to right:**\n"
    "1. **Generated end state** -- the final frame the model actually produced.\n"
    "2. **Target state (simulation)** -- the goal pose, rendered in the physics simulator.\n"
    "3. **GT segmentation mask (hand=green, arm=blue)** -- a standalone silhouette of "
    "panel 2's target: green pixels are the target hand pose, blue pixels are the Franka "
    "arm (marked only so you don't mistake it for the hand -- it isn't part of what "
    "you're rating). Compare this silhouette's shape/position against panel 1's hand."
)


def _load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_subset(composite_manifest_paths: List[str], subset_size: int, seed: int) -> List[Dict]:
    """Stratified-ish random subset: shuffle deterministically, then cap to subset_size.

    A full stratification (even coverage of split x scope x approach x variant) is
    not implemented here -- with a single random seed and a subset_size comfortably
    larger than the number of strata, a plain shuffle+cap already spreads reasonably
    evenly in expectation. Revisit with explicit per-stratum quotas if a specific
    condition turns out under-represented in practice.
    """
    trials: List[Dict] = []
    for path in composite_manifest_paths:
        trials.extend(_load_manifest(path)["results"])
    rng = random.Random(seed)
    shuffled = trials[:]
    rng.shuffle(shuffled)
    return shuffled[:subset_size]


def cmd_build_subset(args: argparse.Namespace) -> None:
    subset = build_subset(args.composite_manifest, args.subset_size, args.seed)
    if not subset:
        raise SystemExit("No trials found in the given --composite_manifest file(s).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_path = output_dir / "subset.json"
    with open(subset_path, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "trials": subset}, f, indent=2)

    blinding_key = {
        t["trial_id"]: {"model_variant": t["model_variant"], "trial_group_id": t["trial_group_id"]}
        for t in subset
    }
    blinding_key_path = output_dir / "blinding_key.json"
    with open(blinding_key_path, "w", encoding="utf-8") as f:
        json.dump(blinding_key, f, indent=2)

    est_minutes = len(subset) * SECONDS_PER_ITEM_ESTIMATE / 60
    print(f"[INFO]: Built a {len(subset)}-trial subset -> {subset_path}")
    print(f"[INFO]: Blinding key (not shown to raters) -> {blinding_key_path}")
    print(f"[INFO]: Estimated ~{est_minutes:.0f} min per rater at ~{SECONDS_PER_ITEM_ESTIMATE}s/item -- "
          f"reduce --subset_size if that's over an hour.")


def _rater_order(num_items: int, username: str, seed: int) -> List[int]:
    rng = random.Random(f"{seed}:{username}")
    order = list(range(num_items))
    rng.shuffle(order)
    return order


def _ratings_path(output_dir: Path, username: str) -> Path:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in username.strip())
    return output_dir / "human_ratings" / f"{safe_name}.jsonl"


def _load_rated_trial_ids(ratings_path: Path) -> set:
    rated = set()
    if ratings_path.exists():
        with open(ratings_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rated.add(json.loads(line)["trial_id"])
    return rated


def _load_own_ratings_summary(ratings_path: Path, trial_by_id: Dict[str, Dict]) -> List[list]:
    """This rater's own submitted scores, ranked highest to lowest.

    Deliberately excludes model_variant/trial_id (same blinding rule as the
    composite images themselves) -- sampling_scope/component_label are safe,
    they identify the target dimension, not which model produced the rollout.
    """
    rows = []
    if ratings_path.exists():
        with open(ratings_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                trial = trial_by_id.get(r["trial_id"], {})
                rows.append(
                    {
                        "score": r["score"],
                        "scope": trial.get("sampling_scope", "?"),
                        "component": trial.get("component_label") or "all dims",
                        "failure_mode": r.get("failure_mode", "?"),
                    }
                )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return [
        [i + 1, round(row["score"], 2), row["scope"], row["component"], row["failure_mode"]]
        for i, row in enumerate(rows)
    ]


def build_app(subset: List[Dict], output_dir: Path, seed: int) -> gr.Blocks:
    with gr.Blocks(title="Controllability rating") as demo:
        username_state = gr.State("")
        order_state = gr.State([])  # list of indices into `subset`, already filtered to unrated
        position_state = gr.State(0)
        session_start_state = gr.State(0.0)
        item_shown_at_state = gr.State(0.0)

        with gr.Column(visible=True) as login_col:
            gr.Markdown(
                "# Controllability rating\n"
                + EXPLANATION_MD
                + "\n\nThere's no time limit, but this is sized to take under an hour."
            )
            username_box = gr.Textbox(label="Your name", placeholder="e.g. alice")
            start_btn = gr.Button("Start", variant="primary")

        with gr.Column(visible=False) as rating_col:
            progress_md = gr.Markdown()
            gr.Markdown(EXPLANATION_MD)
            composite_image = gr.Image(label="Comparison", interactive=False)
            rollout_video = gr.Video(label="Generated rollout", interactive=False)
            score_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="How close did it get to the target? (0 = not at all, 1 = fully reached)")
            failure_mode_dropdown = gr.Dropdown(FAILURE_MODES, value="none", label="Failure mode (optional, best guess)")
            submit_btn = gr.Button("Submit and next", variant="primary")

        with gr.Column(visible=False) as done_col:
            gr.Markdown(
                "# Done -- thank you!\n"
                "Your ratings have been saved. You can close this tab.\n\n"
                "Here's a recap of your own scores, ranked highest to lowest:"
            )
            ranking_table = gr.Dataframe(
                headers=["Rank", "Your score", "Scope", "Component", "Failure mode"],
                interactive=False,
                visible=False,
            )

        trial_by_id: Dict[str, Dict] = {t["trial_id"]: t for t in subset}

        def _current_item(order: List[int], position: int) -> Optional[Dict]:
            if position >= len(order):
                return None
            return subset[order[position]]

        def _render_item(order: List[int], position: int, session_start: float, ratings_path: Path):
            item = _current_item(order, position)
            if item is None:
                summary_rows = _load_own_ratings_summary(ratings_path, trial_by_id)
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    "",
                    gr.update(),
                    gr.update(),
                    time.time(),
                    gr.update(value=summary_rows, visible=True),
                )
            elapsed = time.time() - session_start
            done = position
            remaining = len(order) - position
            pace = (elapsed / done) if done > 0 else SECONDS_PER_ITEM_ESTIMATE
            eta_min = max(0.0, remaining * pace / 60.0)
            progress_text = f"**Item {position + 1} of {len(order)}** -- about {eta_min:.0f} min left at your current pace."
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=item["composite_path"], visible=True),
                gr.update(value=item.get("source_video_path"), visible=True),
                progress_text,
                gr.update(value=0.5),
                gr.update(value="none"),
                time.time(),
                gr.update(visible=False),
            )

        def on_start(username: str):
            username = (username or "").strip()
            if not username:
                gr.Warning("Please enter a name first.")
                return (
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    "", [], 0, 0.0, None, None, "", 0.5, "none", 0.0, gr.update(visible=False),
                )
            ratings_path = _ratings_path(output_dir, username)
            already_rated = _load_rated_trial_ids(ratings_path)
            full_order = _rater_order(len(subset), username, seed)
            order = [i for i in full_order if subset[i]["trial_id"] not in already_rated]

            session_start = time.time()
            rating_visible, done_visible, img, vid, progress_text, score_upd, fm_upd, shown_at, ranking_upd = _render_item(
                order, 0, session_start, ratings_path
            )
            return (
                gr.update(visible=False), rating_visible, done_visible,
                username, order, 0, session_start, img, vid, progress_text, score_upd, fm_upd, shown_at, ranking_upd,
            )

        def on_submit(username: str, order: List[int], position: int, session_start: float, item_shown_at: float, score: float, failure_mode: str):
            item = _current_item(order, position)
            ratings_path = _ratings_path(output_dir, username)
            if item is not None:
                ratings_path.parent.mkdir(parents=True, exist_ok=True)
                row = {
                    "trial_id": item["trial_id"],
                    "score": float(score),
                    "failure_mode": failure_mode,
                    "rater": username,
                    "rated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "seconds_spent": round(time.time() - item_shown_at, 1),
                }
                with open(ratings_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

            next_position = position + 1
            rating_visible, done_visible, img, vid, progress_text, score_upd, fm_upd, shown_at, ranking_upd = _render_item(
                order, next_position, session_start, ratings_path
            )
            return rating_visible, done_visible, next_position, img, vid, progress_text, score_upd, fm_upd, shown_at, ranking_upd

        start_btn.click(
            on_start,
            inputs=[username_box],
            outputs=[
                login_col, rating_col, done_col,
                username_state, order_state, position_state, session_start_state,
                composite_image, rollout_video, progress_md, score_slider, failure_mode_dropdown, item_shown_at_state,
                ranking_table,
            ],
            api_name="start",
        )

        submit_btn.click(
            on_submit,
            inputs=[username_state, order_state, position_state, session_start_state, item_shown_at_state, score_slider, failure_mode_dropdown],
            outputs=[rating_col, done_col, position_state, composite_image, rollout_video, progress_md, score_slider, failure_mode_dropdown, item_shown_at_state, ranking_table],
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
    parser.add_argument("--composite_manifest", action="append", default=[], dest="composite_manifest", help="Path to composite_manifest.json (--mode build_subset; repeatable).")
    parser.add_argument("--subset_size", type=int, default=100, help="--mode build_subset: shared subset size, sized for <1hr at ~25s/item.")
    parser.add_argument("--seed", type=int, default=0, help="--mode build_subset: subset selection seed; also used to derive each rater's per-user shuffle.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where subset.json/blinding_key.json live (build_subset) and human_ratings/ is written (serve).")
    parser.add_argument("--subset_path", type=str, help="--mode serve: path to subset.json from a prior build_subset run.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False, help="Create a public Gradio share link (off by default).")
    args = parser.parse_args()

    if args.mode == "build_subset":
        if not args.composite_manifest:
            raise SystemExit("--composite_manifest is required for --mode build_subset (repeatable).")
        cmd_build_subset(args)
    else:
        if not args.subset_path:
            raise SystemExit("--subset_path is required for --mode serve.")
        cmd_serve(args)


if __name__ == "__main__":
    main()
