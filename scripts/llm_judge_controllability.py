"""LLM judge for the controllability-eval pipeline.

Scores each composite image (from build_controllability_composites.py) 0.0-1.0
for how closely the generated final frame reached its faive_lab ground-truth
target, using Claude vision with structured output. The judge is deliberately
blind to model_variant (never included in the prompt) for the same validity
reason the human rating game (Phase 5) blinds raters.

Composite layout the judge is told about (must match build_controllability_composites.py):
2 rows (side_camera_one, wrist_camera) x 3 columns (Generated | Ground truth |
GT-contour-outlined-on-Generated).

Two modes:
  --mode sync   one call per trial, immediate results -- for testing/small runs.
  --mode batch  Message Batches API (default) -- ~50% cost, for a full sweep;
                submits then exits (results are ended asynchronously, up to a
                few hours later) -- rerun with --mode collect to poll/retrieve.

Example:
    python scripts/llm_judge_controllability.py --mode sync \\
        --composite_manifest /path/to/composites/composite_manifest.json \\
        --output_path /path/to/composites/llm_scores.jsonl \\
        --max_trials 5

    python scripts/llm_judge_controllability.py --mode batch \\
        --composite_manifest /path/to/composites/composite_manifest.json \\
        --output_path /path/to/composites/llm_scores.jsonl
    # ... later ...
    python scripts/llm_judge_controllability.py --mode collect \\
        --batch_id_file /path/to/composites/llm_judge_batch_id.txt \\
        --output_path /path/to/composites/llm_scores.jsonl
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MODEL_ID = "claude-opus-5"

FAILURE_MODES = (
    "none",
    "no_response",
    "coupled_articulation",
    "undershoot",
    "overshoot",
    "wrong_direction",
    "visual_artifact_or_collapse",
    "other",
)

PROMPT_VERSION = "v2"  # v1 described an overlay-blended column 3; v2 describes the mask-panel design
# (see build_controllability_composites.py's module docstring for the A/B test behind the switch) --
# scores from the two prompt versions are not directly comparable, hence the version tag.

SYSTEM_PROMPT = """You are evaluating a robotic hand world-model's controllability: given a commanded \
target hand/arm pose, did the model's generated rollout actually reach it?

You will see one composite image with two rows and three columns:
- Row 1: side_camera_one view. Row 2: wrist_camera view.
- Column 1 (left): the world model's generated final frame.
- Column 2 (middle): the true ground-truth target state, rendered directly in a physics \
simulator (faive_lab) -- this is the answer key, not another model output.
- Column 3 (right): the ground-truth segmentation mask BY ITSELF (not overlaid on any photo) -- \
solid green pixels = the target HAND region, solid blue pixels = the Franka ARM region (marked \
only so you don't confuse it with the hand, not because the arm itself is being scored), \
everything else solid black. There is no photo detail in this column; it is a pure silhouette. \
Compare column 3's green silhouette shape/position directly against column 1's generated photo \
to judge whether the generated hand's shape and position match the target.

IMPORTANT -- hand vs. arm: the robot is a multi-fingered hand (5 articulated digits: thumb, \
index, middle, ring, pinky) mounted on a gray/black Franka robot arm. The commanded target \
refers ONLY to the hand -- never the arm, even where the arm is marked blue in column 3. The \
arm's own links and wrist mount are frequently the most visually prominent gray/mechanical \
shape in the generated frame, especially in the wrist_camera view -- do not mistake it for the \
hand just because it is large or centrally placed. If you cannot clearly identify 5 articulated \
fingers in the generated frame matching column 3's green silhouette, that counts as the hand not \
having reached the target, even if a robot arm segment is visible nearby -- do not describe the \
arm's position as if it were the hand's.

Score how closely the generated final frame (column 1) reached the ground-truth target (column \
2 / column 3), on a continuous 0.0-1.0 scale. Anchors:
- 0.0: no discernible response toward the target -- the hand did not move as commanded, or \
the generated result is unrecognizable as a hand.
- 0.5: motion is present and roughly plausible, but with a clear mismatch or spurious \
coupling -- unintended joints moved, or the reached pose is noticeably off from the target.
- 1.0: the generated hand accurately reaches the target pose, with no spurious coupling.

Use the full 0-1 range -- interpolate based on how close the alignment actually looks, don't \
just pick 0/0.5/1. Weight both camera views: if they disagree (e.g. good alignment in one \
view, poor in the other), reflect that with a score in between rather than judging from only \
one view. If the ground-truth column itself shows an unusual or extreme pose (e.g. the hand \
mostly out of frame, or contorted against a joint limit), that is the correct target to score \
against -- do not penalize the generated frame for matching an extreme target, and do not \
assume the target was reasonable.

Respond with your score, a failure_mode tag summarizing what went wrong (or "none" if it \
reached the target well), and a one-to-two sentence rationale citing what you actually saw in \
the image -- specifically naming hand/finger features you identified, not just "the robot" or \
"the arm.\""""


class ControllabilityScore(BaseModel):
    """No `ge=`/`le=` bounds on `score`: the Batches API's output_config.format
    schema validator rejects `minimum`/`maximum` on a 'number' type property
    ("not supported") even though client.messages.parse()'s schema path
    accepts them -- confirmed by hitting this in testing. The [0, 1] range is
    stated in the field description (for the model) and enforced by the
    validator below (for both call paths, since both construct this model)."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(description="0.0=no response, 0.5=motion but coupled/off-target, 1.0=accurately reached target. Must be in [0.0, 1.0].")
    failure_mode: Literal[
        "none", "no_response", "coupled_articulation", "undershoot",
        "overshoot", "wrong_direction", "visual_artifact_or_collapse", "other",
    ]
    rationale: str = Field(description="One or two sentences citing what was actually seen in the image.")

    @field_validator("score")
    @classmethod
    def _clamp_score(cls, value: float) -> float:
        return max(0.0, min(1.0, value))


def _user_text_for_trial(trial: Dict) -> str:
    scope = trial.get("sampling_scope", "?")
    approach = trial.get("approach_style", "?")
    text = f"sampling_scope={scope}, approach_style={approach}"
    if scope == "per_dim" and trial.get("component_label"):
        text += f", target dimension: {trial['component_label']}"
    return text + "."


def _image_content_block(composite_path: str) -> Dict:
    with open(composite_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
    }


def _build_messages(trial: Dict) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": [
                _image_content_block(trial["composite_path"]),
                {"type": "text", "text": _user_text_for_trial(trial)},
            ],
        }
    ]


def _load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_existing_scores(output_path: str) -> Dict[str, Dict]:
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    existing[row["trial_id"]] = row
    return existing


def _append_score(output_path: str, row: Dict) -> None:
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_sync(client: anthropic.Anthropic, trials: List[Dict], output_path: str, existing_ids: set) -> None:
    for i, trial in enumerate(trials):
        if trial["trial_id"] in existing_ids:
            continue
        # 4096, not 1024: MODEL_ID emits a thinking block before the structured
        # output (confirmed empirically -- a real response for this prompt used 709
        # thinking tokens out of a 1024 max_tokens budget), and thinking tokens count
        # against max_tokens. At 1024 this intermittently truncated the JSON output
        # mid-string once thinking ran long. 4096 leaves comfortable headroom for
        # both, but thinking length isn't bounded, so this can still happen on rare
        # trials -- caught below rather than losing every remaining trial's progress
        # to one bad response (these calls aren't cheap to redo).
        try:
            response = client.messages.parse(
                model=MODEL_ID,
                max_tokens=4096,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=_build_messages(trial),
                output_format=ControllabilityScore,
            )
        except ValidationError as e:
            print(f"[WARN]: [{i + 1}/{len(trials)}] {trial['trial_id']}: response failed schema validation "
                  f"(likely truncated mid-JSON) -- skipping, rerun to retry. {e}")
            continue
        result: ControllabilityScore | None = response.parsed_output
        if result is None:
            print(f"[WARN]: [{i + 1}/{len(trials)}] {trial['trial_id']}: response.parsed_output was None "
                  f"(stop_reason={response.stop_reason}, output_tokens={response.usage.output_tokens}) -- skipping, rerun to retry.")
            continue
        row = {
            "trial_id": trial["trial_id"],
            "score": result.score,
            "failure_mode": result.failure_mode,
            "rationale": result.rationale,
            "judge_model": MODEL_ID,
            "prompt_version": PROMPT_VERSION,
            "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _append_score(output_path, row)
        print(f"[{i + 1}/{len(trials)}] {trial['trial_id']}: score={result.score:.2f} failure_mode={result.failure_mode}")


def run_batch_submit(client: anthropic.Anthropic, trials: List[Dict], batch_id_file: str, existing_ids: set) -> None:
    schema = ControllabilityScore.model_json_schema()
    output_format = {"type": "json_schema", "schema": schema}

    requests = []
    for trial in trials:
        if trial["trial_id"] in existing_ids:
            continue
        requests.append(
            Request(
                custom_id=trial["trial_id"],
                params=MessageCreateParamsNonStreaming(
                    model=MODEL_ID,
                    max_tokens=4096,  # see run_sync's comment -- thinking tokens count against this budget
                    system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                    messages=_build_messages(trial),
                    output_config={"format": output_format},
                ),
            )
        )

    if not requests:
        print("[INFO]: Nothing to submit -- all trials already scored.")
        return

    batch = client.messages.batches.create(requests=requests)
    print(f"[INFO]: Submitted batch {batch.id} with {len(requests)} requests (status={batch.processing_status}).")
    with open(batch_id_file, "w", encoding="utf-8") as f:
        f.write(batch.id)
    print(f"[INFO]: Batch id saved to {batch_id_file}. Rerun with --mode collect once it's done "
          f"(most batches finish within an hour; up to 24h).")


def run_batch_collect(client: anthropic.Anthropic, batch_id_file: str, output_path: str) -> None:
    with open(batch_id_file, "r", encoding="utf-8") as f:
        batch_id = f.read().strip()

    batch = client.messages.batches.retrieve(batch_id)
    print(f"[INFO]: Batch {batch_id} status={batch.processing_status}")
    if batch.processing_status != "ended":
        print("[INFO]: Not done yet -- rerun --mode collect later.")
        return

    n_ok, n_err = 0, 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), None)
            if text is None:
                print(f"[WARN]: {result.custom_id}: succeeded but no text block found; skipping.")
                n_err += 1
                continue
            parsed = ControllabilityScore.model_validate_json(text)
            row = {
                "trial_id": result.custom_id,
                "score": parsed.score,
                "failure_mode": parsed.failure_mode,
                "rationale": parsed.rationale,
                "judge_model": MODEL_ID,
                "prompt_version": PROMPT_VERSION,
                "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _append_score(output_path, row)
            n_ok += 1
        else:
            print(f"[WARN]: {result.custom_id}: {result.result.type}")
            n_err += 1

    print(f"[INFO]: Collected {n_ok} scores ({n_err} errors/skips). {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", type=str, choices=["sync", "batch", "collect"], default="batch")
    parser.add_argument("--composite_manifest", type=str, help="Path to composite_manifest.json (required for sync/batch).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to llm_scores.jsonl (appended incrementally).")
    parser.add_argument("--batch_id_file", type=str, default=None, help="Where the submitted batch id is saved/read (default: alongside --output_path).")
    parser.add_argument("--max_trials", type=int, default=None, help="Optional cap, for testing.")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set (checked process env and .env). "
            "Set it before running this script -- see readme.md's .env section."
        )

    batch_id_file = args.batch_id_file or (os.path.splitext(args.output_path)[0] + "_batch_id.txt")
    default_headers = {}
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace_id:
        default_headers["anthropic-workspace-id"] = workspace_id
    client = anthropic.Anthropic(default_headers=default_headers or None)

    if args.mode == "collect":
        run_batch_collect(client, batch_id_file, args.output_path)
        return

    if not args.composite_manifest:
        raise SystemExit("--composite_manifest is required for --mode sync/batch.")

    composite_manifest = _load_manifest(args.composite_manifest)
    trials = composite_manifest["results"]
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    existing = _load_existing_scores(args.output_path)
    print(f"[INFO]: {len(trials)} trials in manifest, {len(existing)} already scored, mode={args.mode}.")

    if args.mode == "sync":
        run_sync(client, trials, args.output_path, set(existing.keys()))
    else:
        run_batch_submit(client, trials, batch_id_file, set(existing.keys()))


if __name__ == "__main__":
    main()
