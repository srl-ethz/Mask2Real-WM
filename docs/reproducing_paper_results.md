# Reproducing the paper's evaluations

All commands run from the repository root and expect the checkpoints in `checkpoints/` and the
datasets in `datasets/` (see the [README](../README.md)). The launch scripts in
[`scripts/paper/`](../scripts/paper/) take their settings from environment variables that are
listed at the top of each script, for example `PYTHON`, `OUT_ROOT` and `DEBUG=1` for a quick
test run.

| Paper name | Name in the scripts | Identity in the result files |
|---|---|---|
| Cascade-R | `cascade_r` | `wm1_real_only` |
| Cascade-S | `cascade_s` | `wm1_midtrain_only` |
| Cascade-SR | `cascade_sr` | `wm1_midtrain_lora45000` |
| Mono-SR | `mono_sr_57500`, `mono_sr_58000` | `baseline_midtrain_lora` |
| Mono-R | `mono_r` | `baseline_real_only` |

## 1. Video fidelity

```bash
scripts/paper/run_fidelity_eval.sh                                   # all models, all datasets
MODELS="cascade_sr mono_sr_58000" scripts/paper/run_fidelity_eval.sh id ood_cube
```

Every model rolls out 10 autoregressive steps (50 frames, 10 s) from 150 validation samples of
the real-world set and from 50 validation samples of each out-of-distribution category, with
seed 0 and 50 denoising steps. Each run logs its per-sample metrics and comparison videos to
W&B; the remaining steps read them from there.

The runs are seeded. With the batch size of the paper run (24), Mono-SR on an RTX 3090
reproduces that run's per-sample metrics exactly. The noise of a batch is drawn at once, so a
different batch size gives different individual samples. The cascades need a GPU with more
than 24 GB for this evaluation, because WM1 and WM2 are on the GPU at the same time.

1. **Runs spec.** [`scripts/paper/fidelity_runs.json`](../scripts/paper/fidelity_runs.json)
   lists the W&B run of every (dataset, model) pair behind the paper's tables. For runs of
   your own, copy the file and replace the project and the run ids.

2. **PSNR, SSIM, LPIPS, MAE, MSE** per sample and per autoregressive step:

   ```bash
   python scripts/paper/export_fidelity_metrics.py --entity <wandb_entity> \
     --runs scripts/paper/fidelity_runs.json --out_dir results/fidelity
   ```

3. **FVD** (I3D, 16 frames per clip, per camera view; the tables report view 0):

   ```bash
   python scripts/recompute_metrics_and_visualize_on_wandb.py --entity <wandb_entity> \
     --project <wandb_project> --run_ids <run_id> [<run_id> ...] \
     --only_fvd --fvd_csv results/fidelity/fvd.csv
   ```

4. **FID** (Inception-v3, 8 frames per video and view, pooled over both views, plus per-view
   values with a bootstrap standard deviation):

   ```bash
   python scripts/paper/compute_fid.py --entity <wandb_entity> \
     --runs scripts/paper/fidelity_runs.json --work_dir outputs/fid \
     --out_csv results/fidelity/fid.csv
   ```

5. **Tables and error-accumulation plots:**

   ```bash
   python scripts/paper/build_fidelity_tables.py --results_dir results/fidelity
   ```

## 2. Controllability

```bash
scripts/paper/run_controllability_rollouts.sh            # targets and rollouts, all five models
```

The script samples the targets (5 start scenes of the real-world set and 5 of the
out-of-distribution `no_cube` category; per scene 5 whole-pose targets and one target per
action dimension) and rolls every model out towards each target directly and with a linear
ramp: 560 trials per model. Rollouts are seeded (`--seed 42`), so repeating a run on the same
hardware gives the same videos. The ground-truth renders, the composites, the LLM judge, the
human A/B study and the statistics are described step by step in
[controllability_eval.md](controllability_eval.md).

The numbers of the paper's controllability table (A/B win rates with Wilson intervals, Elo,
head-to-head tests, judge-rater agreement, sine-sweep scores and WM1 IoU) are computed from
the judge and rater files by

```bash
python scripts/paper/compute_paper_controllability_tables.py --out_dir results/controllability
```

Its defaults point to the file layout of the result files; every input can be set with an
argument (`--help`).

## 3. WM1 hand-mask IoU

The IoU between the hand mask that WM1 predicts in the last frame of a controllability rollout
and the hand mask of the simulated ground truth:

```bash
python scripts/compute_wm1_controllability_seg_iou.py \
  --gt_manifest /path/to/gt_renders/gt_manifest.json \
  --rollout_manifest outputs/controllability/id \
  --rollout_manifest outputs/controllability/ood \
  --output_csv results/wm1_iou_per_trial.csv \
  --output_summary_csv results/wm1_iou_summary.csv
```

`scripts/visualize_wm1_seg_iou_overlap.py` draws the overlap of the two masks for single
trials, and `scripts/paper/make_wm1_iou_figure.py` and `make_wm1_iou_video.py` build the
qualitative figure and its animated version.

## 4. Sine-sweep rating study

```bash
scripts/paper/run_sine_sweeps.sh                         # sweeps of all five models

python scripts/build_sine_actions_rating_subset.py \
  --sweeps_dir outputs/sine_sweeps --output_dir outputs/sine_actions_human_study
python scripts/gradio_sine_actions_rating.py \
  --subset_path outputs/sine_actions_human_study/subset.json \
  --output_dir outputs/sine_actions_human_study --diagram_dir assets/hand_joint_diagrams
python scripts/compute_sine_actions_rating_stats.py --study_dir outputs/sine_actions_human_study
```

Each of the 23 action dimensions follows one sine cycle while the others hold, for 5 start
scenes of the real-world set and 5 of the `no_cube` category: 1,150 clips over the five
models. Raters see one clip at a time without the model's name and answer whether the joint
did not move (0), moved together with other parts (0.5) or moved on its own (1).

## 5. WM2 with ground-truth masks

See [wm2_oracle_eval.md](wm2_oracle_eval.md).
