"""LLM judge for the pairwise A/B controllability-eval pipeline (v2).

For each comparison image (from build_controllability_pairwise_composites.py),
asks Claude vision which of two anonymous candidates ("Model A" / "Model B")
better reached its faive_lab ground-truth target -- a forced-choice vote, not
a 0-1 score. The judge is blind to which real model (or the synthetic
"gt_self" identity -- the GT's own render, slipped in as a candidate in some
comparisons as a validity check) produced either candidate: that mapping
lives only in pairwise_manifest.json, never in the prompt or the image.

Composite layout the judge is told about (must match
build_controllability_pairwise_composites.py): 2 rows (side_camera_one,
wrist_camera) x 4 columns (GT RGB | GT segmentation mask | Model A | Model B).

Two modes, same as llm_judge_controllability.py:
  --mode sync   one call per comparison, immediate results -- for testing/small runs.
  --mode batch  Message Batches API (default) -- ~50% cost, for a full sweep;
                submits then exits -- rerun with --mode collect to poll/retrieve.

Example:
    python scripts/llm_judge_controllability_pairwise.py --mode sync \\
        --pairwise_manifest /path/to/pairwise_composites/pairwise_manifest.json \\
        --output_path /path/to/pairwise_composites/llm_comparisons.jsonl \\
        --max_trials 10

    python scripts/llm_judge_controllability_pairwise.py --mode batch \\
        --pairwise_manifest /path/to/pairwise_composites/pairwise_manifest.json \\
        --output_path /path/to/pairwise_composites/llm_comparisons.jsonl
    # ... later ...
    python scripts/llm_judge_controllability_pairwise.py --mode collect \\
        --batch_id_file /path/to/pairwise_composites/llm_comparisons_batch_id.txt \\
        --output_path /path/to/pairwise_composites/llm_comparisons.jsonl
"""

import argparse
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
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.llm_judge_controllability import (  # noqa: E402
    MODEL_ID,
    _image_content_block,
    _append_score,
    _user_text_for_trial,
    _build_messages,
)

PAIRWISE_PROMPT_VERSION = "v1"  # independent counter from v1-scoring's PROMPT_VERSION (currently
# "v2") -- this is a different schema/task, not a continuation of that versioning.

PAIRWISE_SYSTEM_PROMPT = """You are evaluating a robotic hand world-model's controllability: given a \
commanded target hand/arm pose, which of two candidate outcomes better reached it?

You will see one composite image with two rows and four columns:
- Row 1: side_camera_one view. Row 2: wrist_camera view.
- Column 1: the true ground-truth target state, rendered directly in a physics simulator \
(faive_lab) -- this is the answer key, not another candidate.
- Column 2: the ground-truth segmentation mask BY ITSELF (not overlaid on any photo) -- solid \
green pixels = the target HAND region, solid blue pixels = the Franka ARM region (marked only so \
you don't confuse it with the hand, not because the arm itself is being judged), everything else \
solid black. There is no photo detail in this column; it is a pure silhouette.
- Column 3 ("Model A") and column 4 ("Model B"): two independent candidate outcomes. Compare each \
against columns 1-2 to judge how closely its hand's shape and position match the target.

IMPORTANT -- hand vs. arm: the robot is a multi-fingered hand (5 articulated digits: thumb, \
index, middle, ring, pinky) mounted on a gray/black Franka robot arm. The commanded target refers \
ONLY to the hand -- never the arm, even where the arm is marked blue in column 2. The arm's own \
links and wrist mount are frequently the most visually prominent gray/mechanical shape in a \
candidate frame, especially in the wrist_camera view -- do not mistake it for the hand just \
because it is large or centrally placed. If you cannot clearly identify 5 articulated fingers in \
a candidate frame matching column 2's green silhouette, that candidate has not reached the \
target, even if a robot arm segment is visible nearby.

IMPORTANT -- "Model A" and "Model B" are anonymous labels only, reassigned independently and at \
random for every image you are shown. Do not assume anything from a candidate's left/right \
position, and do not let any pattern you think you've noticed in a previous comparison influence \
this one -- judge only what is visible in this specific image.

Decide which candidate's hand more accurately reached the target pose shown in columns 1-2, with \
no spurious coupling of unintended joints. Respond with:
- vote: "A" if Model A is clearly closer to the target, "B" if Model B is clearly closer, or \
"tie" if they are equally close (both clearly right, both clearly wrong in the same way, or \
genuinely indistinguishable).
- rationale: one or two sentences citing what you actually saw in the image -- specifically \
naming hand/finger features you identified in each candidate, not just "the robot" or "the arm.\""""

# User-tested alternative: deliberately much shorter, close to verbatim what the user tried by
# hand (a few samples) and found agreed with their own judgment. Kept minimal on purpose -- the
# whole point of this variant is to test whether the long v1 framing above is helping or just
# adding noise, not to re-add the same scaffolding under a new name.
PAIRWISE_SYSTEM_PROMPT_V2 = """You will see one image with two rows (two camera views) and four columns:
- Column 1: ground-truth final hand state, rendered in simulation.
- Column 2: ground-truth segmentation mask (green = hand, blue = arm), for reference only.
- Column 3 ("Model A") and column 4 ("Model B"): two candidate final hand states, anonymous \
labels reassigned at random for every image -- don't assume anything from left/right position.

Which of model A or B's generated final hand state is closest to the one shown on the left in \
simulation?

Respond with:
- vote: "A", "B", or "tie" if indistinguishably close.
- rationale: one or two sentences citing what you saw."""

# v3: v2 + a targeted fix for a diagnosed failure mode -- manual inspection of disagreement
# cases (see conversation history) found the judge frequently anchoring on gross hand/wrist
# *position* in the frame rather than the specific joint under test, even though the joint name
# is already available in the per-trial user text. This makes that instruction explicit instead
# of implicit.
PAIRWISE_SYSTEM_PROMPT_V3 = """You will see one image with two rows (two camera views) and four columns:
- Column 1: ground-truth final hand state, rendered in simulation.
- Column 2: ground-truth segmentation mask (green = hand, blue = arm), for reference only.
- Column 3 ("Model A") and column 4 ("Model B"): two candidate final hand states, anonymous \
labels reassigned at random for every image -- don't assume anything from left/right position.

The user message below may name a specific target joint (e.g. "target dimension: middle_pip") \
for single-joint trials. When one is named, focus specifically on that joint's angle in each \
candidate -- do NOT judge candidates by overall hand or wrist position/placement in the frame; \
that is not the thing being tested and can look different between two otherwise-correct \
candidates. When no specific joint is named (whole-pose trials), judge the whole hand \
configuration as usual.

Which of model A or B's generated final hand state is closest to the one shown on the left in \
simulation?

Respond with:
- vote: "A", "B", or "tie" if indistinguishably close.
- rationale: one or two sentences citing what you saw, specifically at the named joint if one \
was given."""

# v4: v3 + restored, explicit tie permission. Measured tie rates: humans 22.9%, Opus5/v1 prompt
# 12.5%, Fable/v2-v3 (tie mentioned once, only in the output-format line) 3-8%. The v1 prompt's
# three-case tie description ("both clearly right, both clearly wrong in the same way, or
# genuinely indistinguishable") is restored here, moved into the main question so it reads as an
# invited answer rather than a schema footnote -- the terse v2/v3 phrasing looks like it was
# under-eliciting tie, forcing artificial A/B picks on calls that should honestly be close.
PAIRWISE_SYSTEM_PROMPT_V4 = """You will see one image with two rows (two camera views) and four columns:
- Column 1: ground-truth final hand state, rendered in simulation.
- Column 2: ground-truth segmentation mask (green = hand, blue = arm), for reference only.
- Column 3 ("Model A") and column 4 ("Model B"): two candidate final hand states, anonymous \
labels reassigned at random for every image -- don't assume anything from left/right position.

The user message below may name a specific target joint (e.g. "target dimension: middle_pip") \
for single-joint trials. When one is named, focus specifically on that joint's angle in each \
candidate -- do NOT judge candidates by overall hand or wrist position/placement in the frame; \
that is not the thing being tested and can look different between two otherwise-correct \
candidates. When no specific joint is named (whole-pose trials), judge the whole hand \
configuration as usual.

Which of model A or B's generated final hand state is closest to the one shown on the left in \
simulation -- or are they a tie? Answer "tie" whenever that is honestly the closest description: \
both clearly right, both clearly wrong in the same way, or genuinely too close to call. Do not \
force a pick between A and B just to avoid saying tie.

Respond with:
- vote: "A", "B", or "tie".
- rationale: one or two sentences citing what you saw, specifically at the named joint if one \
was given."""


class ControllabilityComparison(BaseModel):
    """Same field-type shape as v1's ControllabilityScore.failure_mode (a bare Literal[...],
    no Field(ge=/le=) numeric bounds) -- already confirmed safe in both client.messages.parse()
    and the Batches API's stricter output_config.format schema validator, unlike score's original
    numeric-bounds Field which that validator rejected. No cross-field permutation constraint
    here either, so no custom validator is needed."""

    model_config = ConfigDict(extra="forbid")

    vote: Literal["A", "B", "tie"] = Field(
        description="Which candidate better matches the GT reference (RGB + segmentation mask): "
        "'A', 'B', or 'tie' if indistinguishably close."
    )
    rationale: str = Field(description="One or two sentences citing what was actually seen in the image.")


def _load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_existing_votes(output_path: str) -> Dict[str, Dict]:
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    existing[row["comparison_id"]] = row
    return existing


def run_sync(
    client: anthropic.Anthropic, trials: List[Dict], output_path: str, existing_ids: set,
    model_id: str = MODEL_ID, temperature: Optional[float] = None,
    system_prompt: str = PAIRWISE_SYSTEM_PROMPT, prompt_version: str = PAIRWISE_PROMPT_VERSION,
    thinking: bool = False,
) -> None:
    # client.messages.parse()'s typed signature doesn't expose temperature directly in this SDK
    # version (confirmed via inspect.signature -- it's present on the raw Messages API and on
    # MessageCreateParamsNonStreaming used by run_batch_submit below, just not on this convenience
    # wrapper), so it has to go through extra_body instead of a named kwarg. Same for adaptive
    # thinking -- confirmed via direct API probing that {"effort": ...} and a nested
    # thinking.adaptive.effort field are both rejected ("Extra inputs are not permitted"); the
    # only lever exposed is the binary thinking={"type": "adaptive"} toggle, no discrete level.
    extra_body = {}
    if temperature is not None:
        extra_body["temperature"] = temperature
    if thinking:
        extra_body["thinking"] = {"type": "adaptive"}
    extra_kwargs = {} if not extra_body else {"extra_body": extra_body}
    for i, trial in enumerate(trials):
        if trial["comparison_id"] in existing_ids:
            continue
        try:
            response = client.messages.parse(
                model=model_id,
                max_tokens=4096,  # see llm_judge_controllability.py's run_sync comment -- thinking
                # tokens count against this budget and can truncate structured output at 1024.
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                messages=_build_messages(trial),
                output_format=ControllabilityComparison,
                **extra_kwargs,
            )
        except ValidationError as e:
            print(f"[WARN]: [{i + 1}/{len(trials)}] {trial['comparison_id']}: response failed schema validation "
                  f"(likely truncated mid-JSON) -- skipping, rerun to retry. {e}")
            continue
        result: Optional[ControllabilityComparison] = response.parsed_output
        if result is None:
            print(f"[WARN]: [{i + 1}/{len(trials)}] {trial['comparison_id']}: response.parsed_output was None "
                  f"(stop_reason={response.stop_reason}, output_tokens={response.usage.output_tokens}) -- skipping, rerun to retry.")
            continue
        thinking_tokens = getattr(response.usage.output_tokens_details, "thinking_tokens", None) if getattr(response.usage, "output_tokens_details", None) else None
        row = {
            "comparison_id": trial["comparison_id"],
            "vote": result.vote,
            "rationale": result.rationale,
            "judge_model": model_id,
            "prompt_version": prompt_version,
            "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "thinking_tokens": thinking_tokens,
        }
        _append_score(output_path, row)
        print(f"[{i + 1}/{len(trials)}] {trial['comparison_id']}: vote={result.vote} (thinking_tokens={thinking_tokens})")


def _chunk_batch_id_paths(batch_id_file: str, n_chunks: int) -> List[str]:
    stem, ext = os.path.splitext(batch_id_file)
    return [f"{stem}_chunk{i}{ext}" for i in range(n_chunks)]


def run_batch_submit(
    client: anthropic.Anthropic,
    trials: List[Dict],
    batch_id_file: str,
    existing_ids: set,
    max_requests_per_batch: int = 700,
    model_id: str = MODEL_ID, temperature: Optional[float] = None,
    system_prompt: str = PAIRWISE_SYSTEM_PROMPT,
) -> None:
    """Splits into multiple Batches API submissions when needed. The Message Batches API caps
    total request size at 256MB, and each request here embeds one base64-encoded composite image
    (~230KB PNG -> ~310KB with the request's JSON/prompt/schema overhead) -- a full pairwise run
    (5600 comparisons) is ~1.7GB unchunked, well over the cap (confirmed empirically: an unchunked
    submit of all 5600 raised anthropic.RequestTooLargeError). max_requests_per_batch=700 keeps
    each chunk to roughly 700*310KB =~ 217MB, comfortably under the 256MB limit. Below that
    threshold (e.g. a --max_trials smoke test), this submits exactly one batch, same as before --
    chunking is transparent to a small run.
    """
    schema = ControllabilityComparison.model_json_schema()
    output_format = {"type": "json_schema", "schema": schema}
    extra_kwargs = {} if temperature is None else {"temperature": temperature}

    all_requests = []
    for trial in trials:
        if trial["comparison_id"] in existing_ids:
            continue
        all_requests.append(
            Request(
                custom_id=trial["comparison_id"],
                params=MessageCreateParamsNonStreaming(
                    model=model_id,
                    max_tokens=4096,
                    system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                    messages=_build_messages(trial),
                    output_config={"format": output_format},
                    **extra_kwargs,
                ),
            )
        )

    if not all_requests:
        print("[INFO]: Nothing to submit -- all comparisons already judged.")
        return

    chunks = [all_requests[i : i + max_requests_per_batch] for i in range(0, len(all_requests), max_requests_per_batch)]

    if len(chunks) == 1:
        batch = client.messages.batches.create(requests=chunks[0])
        print(f"[INFO]: Submitted batch {batch.id} with {len(chunks[0])} requests (status={batch.processing_status}).")
        with open(batch_id_file, "w", encoding="utf-8") as f:
            f.write(batch.id)
        print(f"[INFO]: Batch id saved to {batch_id_file}. Rerun with --mode collect once it's done "
              f"(most batches finish within an hour; up to 24h).")
        return

    chunk_paths = _chunk_batch_id_paths(batch_id_file, len(chunks))
    print(f"[INFO]: {len(all_requests)} requests exceeds --max_requests_per_batch={max_requests_per_batch} "
          f"-- splitting into {len(chunks)} separate batch submissions.")
    for i, (chunk, path) in enumerate(zip(chunks, chunk_paths)):
        batch = client.messages.batches.create(requests=chunk)
        print(f"[INFO]: Submitted chunk {i + 1}/{len(chunks)}: batch {batch.id} with {len(chunk)} requests "
              f"(status={batch.processing_status}).")
        with open(path, "w", encoding="utf-8") as f:
            f.write(batch.id)
    print(f"[INFO]: {len(chunks)} batch ids saved to: {', '.join(chunk_paths)}")
    print(f"[INFO]: Rerun with --mode collect --batch_id_file {batch_id_file} once done -- it "
          f"auto-discovers and collects all {len(chunks)} chunk files under that same base name.")


def _resolve_batch_id_files(batch_id_file: str) -> List[str]:
    """If batch_id_file itself exists, that's the only batch to collect. Otherwise, look for
    chunked sibling files ({stem}_chunk{N}{ext}) written by run_batch_submit when a run was too
    large for a single Batches API submission -- collect all of them."""
    if os.path.exists(batch_id_file):
        return [batch_id_file]
    stem, ext = os.path.splitext(batch_id_file)
    directory = os.path.dirname(batch_id_file) or "."
    pattern = f"{os.path.basename(stem)}_chunk*{ext}"
    return sorted(str(p) for p in Path(directory).glob(pattern))


def run_batch_collect(
    client: anthropic.Anthropic, batch_id_file: str, output_path: str,
    model_id: str = MODEL_ID, prompt_version: str = PAIRWISE_PROMPT_VERSION,
) -> None:
    batch_id_paths = _resolve_batch_id_files(batch_id_file)
    if not batch_id_paths:
        print(f"[WARN]: No batch id file found at {batch_id_file} (or *_chunk*.txt siblings) -- nothing to collect.")
        return

    # Resumable across chunks too: a comparison_id already in output_path (from an earlier
    # partial collect, e.g. only some chunks had finished last time) is skipped rather than
    # appended twice.
    existing_ids = set(_load_existing_votes(output_path).keys())
    any_pending = False
    total_ok = total_err = total_skip = 0

    for path in batch_id_paths:
        with open(path, "r", encoding="utf-8") as f:
            batch_id = f.read().strip()

        batch = client.messages.batches.retrieve(batch_id)
        print(f"[INFO]: Batch {batch_id} ({os.path.basename(path)}) status={batch.processing_status}")
        if batch.processing_status != "ended":
            any_pending = True
            continue

        n_ok = n_err = n_skip = 0
        for result in client.messages.batches.results(batch_id):
            if result.custom_id in existing_ids:
                n_skip += 1
                continue
            if result.result.type == "succeeded":
                msg = result.result.message
                text = next((b.text for b in msg.content if b.type == "text"), None)
                if text is None:
                    print(f"[WARN]: {result.custom_id}: succeeded but no text block found; skipping.")
                    n_err += 1
                    continue
                parsed = ControllabilityComparison.model_validate_json(text)
                row = {
                    "comparison_id": result.custom_id,
                    "vote": parsed.vote,
                    "rationale": parsed.rationale,
                    "judge_model": model_id,
                    "prompt_version": prompt_version,
                    "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                }
                _append_score(output_path, row)
                existing_ids.add(result.custom_id)
                n_ok += 1
            else:
                print(f"[WARN]: {result.custom_id}: {result.result.type}")
                n_err += 1
        print(f"[INFO]:   -> collected {n_ok} ({n_err} errors, {n_skip} already collected).")
        total_ok += n_ok
        total_err += n_err
        total_skip += n_skip

    print(f"[INFO]: Collected {total_ok} votes total ({total_err} errors, {total_skip} already-collected skipped) "
          f"across {len(batch_id_paths)} batch(es). {output_path}")
    if any_pending:
        print("[INFO]: Some batch(es) not done yet -- rerun --mode collect later to pick up the rest.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", type=str, choices=["sync", "batch", "collect"], default="batch")
    parser.add_argument("--pairwise_manifest", type=str, help="Path to pairwise_manifest.json (required for sync/batch).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to llm_comparisons.jsonl (appended incrementally).")
    parser.add_argument("--batch_id_file", type=str, default=None, help="Where the submitted batch id is saved/read (default: alongside --output_path). If a run needs multiple batches (see --max_requests_per_batch), chunk files are written/read alongside this same base name.")
    parser.add_argument("--max_trials", type=int, default=None, help="Optional cap, for testing.")
    parser.add_argument("--max_requests_per_batch", type=int, default=700, help="Split --mode batch submissions larger than this into multiple Batches API calls -- each request embeds one composite image, and the API caps total request size at 256MB.")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help=f"Override the judge model (default: {MODEL_ID}).")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature. Omit to use the API default (1).")
    parser.add_argument("--prompt_version", type=str, choices=["v1", "v2", "v3", "v4"], default="v1", help="v1 = existing detailed system prompt (default). v2 = shorter user-tested alternative. v3 = v2 + explicit instruction to focus on the named joint instead of gross hand position. v4 = v3 + restored explicit tie permission (measured tie-rate gap vs humans/v1).")
    parser.add_argument("--comparison_ids_file", type=str, default=None, help="Optional path to a text file (one comparison_id per line) to restrict trials to -- for a small targeted test run instead of the whole manifest.")
    parser.add_argument("--thinking", action="store_true", help="Enable adaptive extended thinking (thinking={'type':'adaptive'}). No discrete effort level exists in the API -- confirmed via direct probing, only this binary toggle.")
    args = parser.parse_args()

    system_prompt = {
        "v1": PAIRWISE_SYSTEM_PROMPT, "v2": PAIRWISE_SYSTEM_PROMPT_V2,
        "v3": PAIRWISE_SYSTEM_PROMPT_V3, "v4": PAIRWISE_SYSTEM_PROMPT_V4,
    }[args.prompt_version]

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
        run_batch_collect(client, batch_id_file, args.output_path, args.model_id, args.prompt_version)
        return

    if not args.pairwise_manifest:
        raise SystemExit("--pairwise_manifest is required for --mode sync/batch.")

    pairwise_manifest = _load_manifest(args.pairwise_manifest)
    trials = pairwise_manifest["results"]
    if args.comparison_ids_file:
        with open(args.comparison_ids_file, "r", encoding="utf-8") as f:
            wanted_ids = {line.strip() for line in f if line.strip()}
        trials = [t for t in trials if t["comparison_id"] in wanted_ids]
        missing = wanted_ids - {t["comparison_id"] for t in trials}
        if missing:
            print(f"[WARN]: {len(missing)} id(s) from --comparison_ids_file not found in the manifest: {sorted(missing)[:5]}...")
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    existing = _load_existing_votes(args.output_path)
    print(f"[INFO]: {len(trials)} comparisons selected, {len(existing)} already judged, mode={args.mode}, "
          f"model={args.model_id}, temperature={args.temperature}, prompt_version={args.prompt_version}.")

    if args.mode == "sync":
        run_sync(client, trials, args.output_path, set(existing.keys()), args.model_id, args.temperature, system_prompt, args.prompt_version, args.thinking)
    else:
        run_batch_submit(client, trials, batch_id_file, set(existing.keys()), args.max_requests_per_batch, args.model_id, args.temperature, system_prompt)


if __name__ == "__main__":
    main()
