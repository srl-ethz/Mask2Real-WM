# Controllability evaluation

The controllability evaluation asks whether a world model moves the hand and arm where the
actions tell it to. It has six steps, each of which hands a set of files to the next:

1. **Targets**: random goal poses for the hand and arm, shared by every model.
2. **Rollouts and ground truth**: every model is rolled out towards each target, and a physics
   simulation renders what reaching the target really looks like.
3. **Composites**: images that put a model's final frame next to the ground truth.
4. **LLM judge**: a vision-language model scores each composite without knowing which model
   produced it.
5. **Human ratings**: a small study that checks the judge against people.
6. **Statistics**: the report that joins everything.

[Part 1](#part-1-absolute-scores) scores one rollout at a time on a 0 to 1 scale.
[Part 2](#part-2-pairwise-ab-comparisons) reuses the same rollouts for pairwise A/B
comparisons, which is the protocol behind the paper's win rates and Elo ratings.

All commands run from the repository root. The LLM judge needs `ANTHROPIC_API_KEY` in `.env`
(see [`.env.example`](../.env.example)).

```bash
OUTPUT_DIR=outputs/controllability/id
```

[`scripts/paper/run_controllability_rollouts.sh`](../scripts/paper/run_controllability_rollouts.sh)
runs steps 1 and 2b for all five models with the paper's settings, on the in-distribution set
and on the out-of-distribution `no_cube` category.

## Part 1: absolute scores

### 1. Targets

One catalog of goal poses, used by every model and both approach styles so that the comparison
is fair. Whole-pose targets change all 23 action dimensions at once, per-dim targets exactly
one.

```bash
python scripts/inference_wm1_to_wm2_controllability_eval.py \
  --phase targets --split_name id \
  --dataset_root_path datasets/2026-03-14T13-34-49 \
  --dataset_names large_real_dataset_5fps_135_240 \
  --dataset_meta_info_path dataset_meta_info/2026-03-14T13-34-49 \
  --dataset_stat_path dataset_meta_info/2026-03-14T13-34-49/large_real_dataset_5fps_135_240/stat.json \
  --mode val --sample_indices 83,63,94,17,89 --seed 42 \
  --sampling_scopes whole_pose,per_dim \
  --num_whole_pose_targets 5 --num_per_dim_targets 1 \
  --output_path "${OUTPUT_DIR}"
```

This writes `${OUTPUT_DIR}/targets_manifest.json`. Useful options:

- `--sample_indices`, or `--num_random_samples N` to draw the start scenes at random.
- `--target_bound_percentile_low` / `--target_bound_percentile_high` (default 1 / 99): the
  percentiles of the dataset statistics that bound each dimension's sampling range.
- `--min_norm_target_distance` (default 0.2): how far a per-dim target must be from the start
  pose.
- `--legacy_default_frame_mapping`: reads the start pose with the frame mapping used for the
  paper's targets; without it the start pose is taken from the frame the rollout starts at.

Steps 2a and 2b only need `targets_manifest.json` and are independent of each other.

### 2a. Ground truth from simulation

For each target, the simulation holds the target as a setpoint until the robot has settled and
renders RGB and instance segmentation from both cameras. The renderer is part of
[Mask2Real-SimDataGen](https://github.com/srl-ethz/Mask2Real-SimDataGen), the Isaac Lab
extension (`faive_lab`) for the ORCA hand:

```bash
# in the Mask2Real-SimDataGen repository
python scripts/wm_evaluation/render_controllability_targets.py \
  --targets_manifest /path/to/Mask2Real-WM/${OUTPUT_DIR}/targets_manifest.json \
  --output_dir /path/to/gt_renders \
  --headless --enable_cameras
```

It writes `gt_manifest.json`, keyed by `trial_group_id`: one render per target, shared by all
models and approach styles. Running it again with the same `--output_dir` skips targets that
are already rendered. To render the in-distribution and out-of-distribution targets in one
go, merge the two manifests first with
[`scripts/paper/merge_targets_manifests.py`](../scripts/paper/merge_targets_manifests.py).

### 2b. Rollouts

Run once per model, always with the same targets manifest:

```bash
# A monolithic model
python scripts/inference_wm1_to_wm2_controllability_eval.py \
  --phase rollout --targets_manifest_path "${OUTPUT_DIR}/targets_manifest.json" \
  --baseline_wm_model_ckpt_path checkpoints/mono_r/model.safetensors \
  --baseline_wm_config_path experiments/mask2real/mono_r.yaml \
  --baseline_wm_dataset_stat_path checkpoints/mono_r/stat.json \
  --baseline_wm_num_inference_steps 50 --max_batch_size 8 \
  --dataset_root_path datasets/2026-03-14T13-34-49 \
  --dataset_names large_real_dataset_5fps_135_240 \
  --dataset_meta_info_path dataset_meta_info/2026-03-14T13-34-49 \
  --dataset_stat_path dataset_meta_info/2026-03-14T13-34-49/large_real_dataset_5fps_135_240/stat.json \
  --mode val --seed 42 --approach_styles direct,linear --ar_num_steps 5 \
  --variant_label baseline_real_only --output_path "${OUTPUT_DIR}"

# A cascade
python scripts/inference_wm1_to_wm2_controllability_eval.py \
  --phase rollout --targets_manifest_path "${OUTPUT_DIR}/targets_manifest.json" \
  --wm1_model_ckpt_path checkpoints/wm1_cascade_sr/model.safetensors \
  --wm1_config_path experiments/mask2real/wm1_cascade_sr.yaml \
  --wm1_dataset_stat_path checkpoints/wm1_cascade_sr/stat.json \
  --wm2_model_ckpt_path checkpoints/wm2/model.safetensors \
  --wm2_config_path experiments/mask2real/wm2.yaml \
  --wm2_dataset_stat_path checkpoints/wm2/stat.json \
  --wm1_num_inference_steps 50 --wm2_num_inference_steps 50 \
  --sequential_wm1_wm2_loading --max_batch_size 8 \
  --dataset_root_path datasets/2026-03-14T13-34-49 \
  --dataset_names large_real_dataset_5fps_135_240 \
  --dataset_meta_info_path dataset_meta_info/2026-03-14T13-34-49 \
  --dataset_stat_path dataset_meta_info/2026-03-14T13-34-49/large_real_dataset_5fps_135_240/stat.json \
  --mode val --seed 42 --approach_styles direct,linear --ar_num_steps 5 \
  --variant_label wm1_midtrain_lora45000 --output_path "${OUTPUT_DIR}"
```

Each run writes `rollout_manifest_<variant_label>.json`, the videos and the predicted
latents. Notes:

- `--sequential_wm1_wm2_loading` keeps only one of WM1 and WM2 on the GPU at a time, which is
  what makes a cascade fit on a 24 GB GPU.
- Give every model its own `--variant_label`. It names the manifest and is part of every
  `trial_id`; two models with the same label overwrite each other.
- Pass each model the `stat.json` it was trained with (it is stored next to the checkpoint).
  With the wrong statistics the action conditioning can saturate without any error.
- The rollout refuses to run if the scene it loads does not have the start pose recorded in
  the targets manifest, so use the same dataset arguments and `--seed` as in step 1.
- `--debug` runs a tiny version of the whole step.

### 3. Composites

One image per trial: two camera views, three panels each (generated frame, ground truth,
ground-truth segmentation mask with the hand in green and the arm in blue). This is what the
judge and the human raters see.

```bash
python scripts/build_controllability_composites.py \
  --rollout_manifest "${OUTPUT_DIR}/rollout_manifest_baseline_real_only.json" \
  --rollout_manifest "${OUTPUT_DIR}/rollout_manifest_wm1_midtrain_lora45000.json" \
  --gt_manifest /path/to/gt_renders/gt_manifest.json \
  --output_dir "${OUTPUT_DIR}/composites"
```

`--rollout_manifest` can be repeated. The title strip of an image names the target and the
trial settings but never the model, which keeps the judge and the raters blind.

With `--gt_self_check` the script instead builds one composite per target in which the ground
truth is also used as the "generated" frame. Judging those gives the score the judge assigns
to a perfect match, which is below 1.0 and varies between scenes; step 6 can report scores
relative to it.

```bash
python scripts/build_controllability_composites.py --gt_self_check \
  --gt_manifest /path/to/gt_renders/gt_manifest.json \
  --output_dir "${OUTPUT_DIR}/gt_self_check_composites"
```

### 4. LLM judge

The judge (`claude-opus-5`) scores each composite between 0 and 1 for how close the generated
frame is to the target, and adds a failure-mode tag and a short rationale.

```bash
# A handful first
python scripts/llm_judge_controllability.py --mode sync \
  --composite_manifest "${OUTPUT_DIR}/composites/composite_manifest.json" \
  --output_path "${OUTPUT_DIR}/llm_scores.jsonl" --max_trials 4

# The full set as a batch, and later its results
python scripts/llm_judge_controllability.py --mode batch \
  --composite_manifest "${OUTPUT_DIR}/composites/composite_manifest.json" \
  --output_path "${OUTPUT_DIR}/llm_scores.jsonl"
python scripts/llm_judge_controllability.py --mode collect \
  --output_path "${OUTPUT_DIR}/llm_scores.jsonl"
```

Run the same three commands on `gt_self_check_composites/composite_manifest.json` with
`--output_path "${OUTPUT_DIR}/gt_self_scores.jsonl"` for the perfect-match scores. Read a few
rationales next to their images before trusting a full run.

### 5. Human ratings

A shared subset of trials that every rater scores with a slider:

```bash
python scripts/gradio_controllability_rating.py --mode build_subset \
  --composite_manifest "${OUTPUT_DIR}/composites/composite_manifest.json" \
  --output_dir "${OUTPUT_DIR}/human_ratings" --subset_size 100

python scripts/gradio_controllability_rating.py --mode serve \
  --subset_path "${OUTPUT_DIR}/human_ratings/subset.json" \
  --output_dir "${OUTPUT_DIR}/human_ratings"
```

Raters enter a name, which becomes their file `human_ratings/human_ratings/<name>.jsonl`;
entering the same name again resumes the session. `blinding_key.json` maps trials to models
and is only read by step 6.

### 6. Statistics

```bash
python scripts/compute_controllability_stats.py \
  --rollout_manifest "${OUTPUT_DIR}/rollout_manifest_baseline_real_only.json" \
  --rollout_manifest "${OUTPUT_DIR}/rollout_manifest_wm1_midtrain_lora45000.json" \
  --llm_scores "${OUTPUT_DIR}/llm_scores.jsonl" \
  --gt_self_scores "${OUTPUT_DIR}/gt_self_scores.jsonl" \
  --human_ratings_dir "${OUTPUT_DIR}/human_ratings/human_ratings" \
  --output_dir "${OUTPUT_DIR}/stats"
```

`--gt_self_scores` and `--human_ratings_dir` are optional. The script writes
`stats/report.md`, `stats/summary.json` and `stats/plots/`:

- mean score with a 95% confidence interval per model, split, sampling scope and approach
  style, also relative to the perfect-match score when `--gt_self_scores` is given;
- Spearman correlation between the judge and the mean human score;
- the intraclass correlation between raters and judge on the continuous scale;
- weighted Cohen's kappa after rounding the scores to {0, 0.5, 1}.

## Part 2: pairwise A/B comparisons

Instead of grading one rollout, this protocol shows two candidates next to the same ground
truth and asks which one is closer. It needs no new rollouts or renders. For the LLM judge the
ground truth itself is added as a hidden candidate (`gt_self`): a sound setup has to prefer
it almost every time.

### 2.1 Comparison images

For every target and approach style, the candidates are the models' final frames plus
`gt_self`, and every pair of them becomes one image with four panels per camera view (ground
truth RGB, ground-truth mask, Model A, Model B). A and B are assigned at random per
comparison.

```bash
python scripts/build_controllability_pairwise_composites.py \
  --rollout_manifest "${OUTPUT_DIR}/rollout_manifest_baseline_midtrain_lora.json" \
  --rollout_manifest "${OUTPUT_DIR}/rollout_manifest_wm1_midtrain_only.json" \
  --rollout_manifest "${OUTPUT_DIR}/rollout_manifest_wm1_midtrain_lora45000.json" \
  --rollout_manifest "${OUTPUT_DIR}/rollout_manifest_wm1_real_only.json" \
  --gt_manifest /path/to/gt_renders/gt_manifest.json \
  --output_dir "${OUTPUT_DIR}/pairwise_composites" --seed 0
```

Pass the manifests of every split you have. The output is `pairwise_manifest.json` and one
image per comparison, named `<trial_group_id>_<approach_style>_pair<NN>`. Keep `--seed` fixed
so the judge and all raters see the same arrangement.

Adding a model later would renumber the existing pairs.
[`scripts/build_controllability_pairwise_composites_addendum.py`](../scripts/build_controllability_pairwise_composites_addendum.py)
instead adds one new model against explicitly chosen opponents with new pair numbers and
writes a separate manifest; this is how Mono-R (`baseline_real_only`) was added to the study.

### 2.2 LLM judge

```bash
python scripts/llm_judge_controllability_pairwise.py --mode sync \
  --pairwise_manifest "${OUTPUT_DIR}/pairwise_composites/pairwise_manifest.json" \
  --output_path "${OUTPUT_DIR}/llm_comparisons.jsonl" --max_trials 10

python scripts/llm_judge_controllability_pairwise.py --mode batch \
  --pairwise_manifest "${OUTPUT_DIR}/pairwise_composites/pairwise_manifest.json" \
  --output_path "${OUTPUT_DIR}/llm_comparisons.jsonl"
python scripts/llm_judge_controllability_pairwise.py --mode collect \
  --output_path "${OUTPUT_DIR}/llm_comparisons.jsonl"
```

The judge answers `A`, `B` or `tie` with a short rationale.

### 2.3 Human A/B study

The subset is chosen per target: every selected target contributes all of its pairs between
real models, so each rater sees the complete picture for the targets they rate. `gt_self` is
never shown to people.

```bash
python scripts/gradio_controllability_pairwise.py --mode build_subset \
  --pairwise_manifest "${OUTPUT_DIR}/pairwise_composites/pairwise_manifest.json" \
  --output_dir "${OUTPUT_DIR}/human_comparisons" --num_trial_groups 20

python scripts/gradio_controllability_pairwise.py --mode serve \
  --subset_path "${OUTPUT_DIR}/human_comparisons/subset.json" \
  --output_dir "${OUTPUT_DIR}/human_comparisons"
```

`--extend_subset <subset.json>` adds targets to an existing subset without disturbing raters
who have already started, and `--scope_filter whole_pose` or `per_dim` restricts the
candidates to one kind of target. Votes are saved per rater in
`human_comparisons/human_comparisons/<name>.jsonl`.

### 2.4 Statistics

```bash
python scripts/compute_controllability_pairwise_stats.py \
  --pairwise_manifest "${OUTPUT_DIR}/pairwise_composites/pairwise_manifest.json" \
  --llm_comparisons "${OUTPUT_DIR}/llm_comparisons.jsonl" \
  --human_comparisons_dir "${OUTPUT_DIR}/human_comparisons/human_comparisons" \
  --output_dir "${OUTPUT_DIR}/pairwise_stats"
```

The report starts with the win rate of `gt_self`, which should be close to 100%; only then are
the win rates and Elo ratings of the models and the agreement between judge and raters worth
reading. [`scripts/paper/compute_paper_controllability_tables.py`](../scripts/paper/compute_paper_controllability_tables.py)
computes the numbers of the paper's controllability table from the judge and rater files.

## Troubleshooting

**A dimension is often listed in `saturated_dims` in `gt_manifest.json`, or `converged` is
often false.** The simulation could not reach the target, usually because the statistics of
that dimension contain outliers. Tighten the percentile bounds in step 1 or exclude the
dimension with `--component_indices`, then render again.

**The judge's rationale describes the wrong object, for example the arm instead of the hand.**
Check that `gt_manifest.json` has `arm_instance_ids`, so that the arm is drawn in blue in the
mask panel, and read the rationale next to the composite.

**A perfect match does not get a score of 1.0.** This is expected; use the `gt_self_check`
composites and `--gt_self_scores` to report scores relative to the judge's own ceiling.

**Agreement statistics close to 1.0 with only a couple of raters.** With few raters a small
sample can agree by chance. Check that the rater files are not copies of each other and treat
the numbers as indicative.

**The Anthropic API answers 400 "not scoped to a workspace".** Set `ANTHROPIC_WORKSPACE_ID` in
`.env`.

**`gt_self` does not win most of its comparisons.** Treat this as a problem of the setup, not
as a result. Look at comparisons that `gt_self` lost: the candidate panel should be identical
to the ground-truth panel, and the title strip must not reveal an identity.
