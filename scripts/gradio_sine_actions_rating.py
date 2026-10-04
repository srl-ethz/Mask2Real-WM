"""Human rating game for the 5-model sine-actions controllability study.

Scoped as a full-coverage study: every rater sees the SAME
1150-trial subset (5 models x 2 datasets x 5 samples x 23 action dims), built
ahead of time by build_sine_actions_rating_subset.py. Not time-boxed to under
an hour -- ratings are saved incrementally, so a rater can stop and resume
across as many sessions as they need; expect roughly 4-5 hours of total rating
time per rater at a realistic pace.

For each trial the rater sees the generated rollout video plus a reference
diagram + plain-language description of which joint is being tested, and picks
one of three answers:
    0.0 -- the joint does not move
    0.5 -- the joint moves, but coupled with other parts also moving
    1.0 -- only this joint moves (a clean, isolated response)

Blinding: raters never see which model produced a trial, anywhere in the UI.
Trials are shown in a per-rater-seeded shuffled order under a plain "Item N of
M" label. The model_slot -> real model name mapping lives only in
blinding_key.json (written by build_sine_actions_rating_subset.py) and is
never read by this app -- only by the analysis script, later.

Usage:
    python scripts/gradio_sine_actions_rating.py \\
        --subset_path outputs/sine_actions_human_study/subset.json \\
        --output_dir outputs/sine_actions_human_study \\
        --diagram_dir assets/hand_joint_diagrams
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr

SCORE_CHOICES = [
    "0 -- no movement",
    "0.5 -- moves, but coupled with other parts also moving",
    "1.0 -- only this joint moves",
]
SCORE_VALUE = {
    "0 -- no movement": 0.0,
    "0.5 -- moves, but coupled with other parts also moving": 0.5,
    "1.0 -- only this joint moves": 1.0,
}

SECONDS_PER_ITEM_ESTIMATE = 15  # only used before a rater has any timing data of their own

# gr.Video has no built-in playback-rate setting, so set it via JS after each video swap.
# Runs client-side after the Python callback's outputs are applied to the DOM.
SET_PLAYBACK_RATE_JS = """
() => {
    setTimeout(() => {
        const video = document.querySelector('#rollout-video video');
        if (video) {
            video.playbackRate = 2.0;
            video.play().catch(() => {});
        }
    }, 100);
}
"""

EXPLANATION_MD = (
    "For each item, watch the short generated video, then look at the diagram below it "
    "for which joint this trial is testing. Answer: does **that specific joint** move?\n\n"
    "- **0 -- no movement**: the joint shown in the diagram does not visibly move.\n"
    "- **0.5 -- coupled movement**: it moves, but so do other parts of the hand/wrist at the same time.\n"
    "- **1.0 -- isolated movement**: only the joint shown in the diagram moves, nothing else."
)


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rater_order(num_items: int, username: str, seed: int) -> List[int]:
    rng = random.Random(f"{seed}:{username}")
    order = list(range(num_items))
    rng.shuffle(order)
    return order


def _ratings_path(output_dir: Path, username: str) -> Path:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in username.strip())
    ratings_dir = output_dir / "human_ratings"
    # Case-insensitive match against existing files first, so a rater who typed their name with
    # different capitalization on a later login (e.g. "chenyu Yang" vs "Chenyu_Yang") resumes into
    # their existing file instead of silently starting a second, fragmented identity.
    if ratings_dir.exists():
        for existing in ratings_dir.glob("*.jsonl"):
            if existing.stem.lower() == safe_name.lower():
                return existing
    return ratings_dir / f"{safe_name}.jsonl"


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
    """This rater's own submitted answers. Deliberately excludes model identity."""
    rows = []
    if ratings_path.exists():
        with open(ratings_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                trial = trial_by_id.get(r["trial_id"], {})
                rows.append({
                    "answer": r["answer"],
                    "dataset": trial.get("dataset", "?"),
                    "component": trial.get("component_label", "?"),
                })
    rows.sort(key=lambda row: row["answer"], reverse=True)
    return [[i + 1, row["answer"], row["dataset"], row["component"]] for i, row in enumerate(rows)]


def _diagram_path(diagram_dir: Path, component: int, component_label: str) -> str:
    path = diagram_dir / f"dim_{component:02d}_{component_label}.png"
    return str(path) if path.exists() else None


def build_app(subset: List[Dict], output_dir: Path, diagram_dir: Path, seed: int) -> gr.Blocks:
    with gr.Blocks(title="Sine-actions controllability rating") as demo:
        username_state = gr.State("")
        order_state = gr.State([])
        position_state = gr.State(0)
        session_start_state = gr.State(0.0)
        item_shown_at_state = gr.State(0.0)

        with gr.Column(visible=True) as login_col:
            gr.Markdown(
                "# Sine-actions controllability rating\n"
                + EXPLANATION_MD
                + "\n\nThis is a full-coverage study (1150 items) -- there's no time limit, "
                "and your progress is saved after every answer, so feel free to split it "
                "across multiple sessions. Just log back in with the same name to resume."
            )
            username_box = gr.Textbox(label="Your name", placeholder="e.g. alice")
            start_btn = gr.Button("Start", variant="primary")

        with gr.Column(visible=False) as rating_col:
            progress_md = gr.Markdown()
            gr.Markdown(EXPLANATION_MD)
            rollout_video = gr.Video(
                label="Generated rollout", interactive=False,
                autoplay=True, loop=True, elem_id="rollout-video",
            )
            joint_diagram = gr.Image(label="Which joint is this trial testing?", interactive=False)
            answer_radio = gr.Radio(SCORE_CHOICES, label="Does that joint move?")
            submit_btn = gr.Button("Submit and next", variant="primary")

        with gr.Column(visible=False) as done_col:
            gr.Markdown(
                "# Done -- thank you!\n"
                "Your ratings have been saved. You can close this tab.\n\n"
                "Here's a recap of your own answers:"
            )
            ranking_table = gr.Dataframe(
                headers=["#", "Your answer", "Dataset", "Joint"],
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
                    gr.update(value=None),
                    time.time(),
                    gr.update(value=summary_rows, visible=True),
                )
            elapsed = time.time() - session_start
            done = position
            remaining = len(order) - position
            pace = (elapsed / done) if done > 0 else SECONDS_PER_ITEM_ESTIMATE
            eta_min = max(0.0, remaining * pace / 60.0)
            progress_text = f"**Item {position + 1} of {len(order)}** -- about {eta_min:.0f} min left at your current pace."
            diagram = _diagram_path(diagram_dir, item["component"], item["component_label"])
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=item["video_path"], visible=True),
                gr.update(value=diagram, visible=True),
                progress_text,
                gr.update(value=None),
                time.time(),
                gr.update(visible=False),
            )

        def on_start(username: str):
            username = (username or "").strip()
            if not username:
                gr.Warning("Please enter a name first.")
                return (
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    "", [], 0, 0.0, None, None, "", None, 0.0, gr.update(visible=False),
                )
            ratings_path = _ratings_path(output_dir, username)
            # Canonicalize to the resolved file's own name (matches an existing rater
            # case-insensitively) so the shuffle seed and the stored "rater" field stay
            # consistent across logins, even if this typing differs in capitalization.
            username = ratings_path.stem
            already_rated = _load_rated_trial_ids(ratings_path)
            full_order = _rater_order(len(subset), username, seed)
            order = [i for i in full_order if subset[i]["trial_id"] not in already_rated]

            session_start = time.time()
            rating_visible, done_visible, vid, diagram_upd, progress_text, answer_upd, shown_at, ranking_upd = _render_item(
                order, 0, session_start, ratings_path
            )
            return (
                gr.update(visible=False), rating_visible, done_visible,
                username, order, 0, session_start, vid, diagram_upd, progress_text, answer_upd, shown_at, ranking_upd,
            )

        def on_submit(username: str, order: List[int], position: int, session_start: float, item_shown_at: float, answer_label: str):
            item = _current_item(order, position)
            ratings_path = _ratings_path(output_dir, username)
            if item is not None:
                if not answer_label:
                    gr.Warning("Please pick an answer first.")
                    rating_visible, done_visible, vid, diagram_upd, progress_text, answer_upd, shown_at, ranking_upd = _render_item(
                        order, position, session_start, ratings_path
                    )
                    return rating_visible, done_visible, position, vid, diagram_upd, progress_text, answer_upd, shown_at, ranking_upd
                ratings_path.parent.mkdir(parents=True, exist_ok=True)
                row = {
                    "trial_id": item["trial_id"],
                    "component": item["component"],
                    "component_label": item["component_label"],
                    "dataset": item["dataset"],
                    "model_slot": item["model_slot"],
                    "answer": SCORE_VALUE[answer_label],
                    "rater": username,
                    "rated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "seconds_spent": round(time.time() - item_shown_at, 1),
                }
                with open(ratings_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

            next_position = position + 1
            rating_visible, done_visible, vid, diagram_upd, progress_text, answer_upd, shown_at, ranking_upd = _render_item(
                order, next_position, session_start, ratings_path
            )
            return rating_visible, done_visible, next_position, vid, diagram_upd, progress_text, answer_upd, shown_at, ranking_upd

        start_btn.click(
            on_start,
            inputs=[username_box],
            outputs=[
                login_col, rating_col, done_col,
                username_state, order_state, position_state, session_start_state,
                rollout_video, joint_diagram, progress_md, answer_radio, item_shown_at_state,
                ranking_table,
            ],
            api_name="start",
        ).then(fn=None, js=SET_PLAYBACK_RATE_JS)

        submit_btn.click(
            on_submit,
            inputs=[username_state, order_state, position_state, session_start_state, item_shown_at_state, answer_radio],
            outputs=[rating_col, done_col, position_state, rollout_video, joint_diagram, progress_md, answer_radio, item_shown_at_state, ranking_table],
            api_name="submit",
        ).then(fn=None, js=SET_PLAYBACK_RATE_JS)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset_path", type=str, required=True, help="Path to subset.json from build_sine_actions_rating_subset.py.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where human_ratings/ is written (should match --output_dir used to build the subset).")
    parser.add_argument("--diagram_dir", type=str, default="assets/hand_joint_diagrams")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False, help="Create a public Gradio share link (off by default).")
    args = parser.parse_args()

    subset_data = _load_json(args.subset_path)
    subset = subset_data["trials"]
    seed = subset_data.get("seed", 0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demo = build_app(subset, output_dir, Path(args.diagram_dir), seed)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
