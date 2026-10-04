#!/usr/bin/env bash
# Controllability evaluation, world-model side: sample the target poses and roll out all five
# models towards them.
#
#   id : large_real_dataset_5fps_135_240 validation samples 83,63,94,17,89
#   ood: processed_OOD_combined_all4_5fps, "no_cube" category, episodes 0-4, samples 38,24,21,20,13
#
# Per sample: 5 whole-pose targets and 1 per-dim target for each of the 23 action dims (seed 42),
# each approached directly and with a linear ramp: 280 targets x 2 styles = 560 trials per model.
# The Isaac Sim ground-truth renders for the targets come from Mask2Real-SimDataGen
# (https://github.com/srl-ethz/Mask2Real-SimDataGen, scripts/wm_evaluation/
# render_controllability_targets.py) and are not needed for the rollouts.
#
# Usage:
#   scripts/paper/run_controllability_rollouts.sh [targets|rollouts|all] [variant ...]
#
# Variants (the labels are the identities used in the released result files):
#   wm1_real_only          Cascade-R   (WM1 cascade_r  -> WM2)
#   wm1_midtrain_only      Cascade-S   (WM1 cascade_s  -> WM2)
#   wm1_midtrain_lora45000 Cascade-SR  (WM1 cascade_sr -> WM2)
#   baseline_midtrain_lora Mono-SR     (checkpoint 57,500)
#   baseline_real_only     Mono-R
#
# Environment:
#   PYTHON            interpreter (default: python)
#   OUT_ROOT          output folder (default: outputs/controllability)
#   PAPER_TARGETS     1 (default) regenerates the published targets manifests
#                     (--legacy_default_frame_mapping); 0 uses frame-aligned base poses
#   WANDB_PROJECT     log to this W&B project (default: no W&B logging)
#   DEBUG             1 runs the scripts' --debug mode
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
OUT_ROOT=${OUT_ROOT:-outputs/controllability}
PAPER_TARGETS=${PAPER_TARGETS:-1}
STAGE=${1:-all}
shift || true
VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
  VARIANTS=(wm1_real_only wm1_midtrain_only wm1_midtrain_lora45000 baseline_midtrain_lora baseline_real_only)
fi

CKPT=checkpoints
CFG=experiments/mask2real
REAL_STAT=dataset_meta_info/2026-03-14T13-34-49/large_real_dataset_5fps_135_240/stat.json
SIM_STAT=dataset_meta_info/2026-05-19T03-28-53/SIM_PRETRAINING_DATA_ALL_lerobot_5fps/stat.json

EXTRA=()
[[ "${DEBUG:-0}" == "1" ]] && EXTRA+=(--debug)

dataset_args() {  # dataset_args <split>
  case "$1" in
    id)  echo --dataset_root_path datasets/2026-03-14T13-34-49 \
              --dataset_names large_real_dataset_5fps_135_240 \
              --dataset_meta_info_path dataset_meta_info/2026-03-14T13-34-49 ;;
    ood) echo --dataset_root_path datasets/2026-04-05T20-09-04 \
              --dataset_names processed_OOD_combined_all4_5fps \
              --dataset_meta_info_path dataset_meta_info/2026-04-05T20-09-04 \
              --dataset_meta_info_name processed_OOD_combined_all4_5fps_val_no_cube ;;
    *) echo "unknown split $1" >&2; exit 1 ;;
  esac
}

sample_args() {  # sample_args <split>
  case "$1" in
    id)  echo --sample_indices 83,63,94,17,89 ;;
    ood) echo --episode_ids 0,1,2,3,4 --sample_indices 38,24,21,20,13 ;;
  esac
}

model_args() {  # model_args <variant>
  local cascade_common=(--wm2_model_ckpt_path "$CKPT/wm2/model.safetensors" --wm2_config_path "$CFG/wm2.yaml"
                        --wm2_dataset_stat_path "$SIM_STAT" --wm1_num_inference_steps 50 --wm2_num_inference_steps 50
                        --sequential_wm1_wm2_loading --max_batch_size 8)
  case "$1" in
    wm1_real_only)
      echo --wm1_model_ckpt_path "$CKPT/wm1_cascade_r/model.safetensors" --wm1_config_path "$CFG/wm1_cascade_r.yaml" \
           "${cascade_common[@]}" ;;
    wm1_midtrain_only)
      echo --wm1_model_ckpt_path "$CKPT/wm1_cascade_s/model.safetensors" --wm1_config_path "$CFG/wm1_cascade_s.yaml" \
           --wm1_dataset_stat_path "$SIM_STAT" "${cascade_common[@]}" ;;
    wm1_midtrain_lora45000)
      echo --wm1_model_ckpt_path "$CKPT/wm1_cascade_sr/model.safetensors" --wm1_config_path "$CFG/wm1_cascade_sr.yaml" \
           --wm1_dataset_stat_path "$SIM_STAT" "${cascade_common[@]}" ;;
    baseline_midtrain_lora)
      # The Mono-SR rollouts batch all of a sample's cases into one call (no --max_batch_size).
      echo --baseline_wm_model_ckpt_path "$CKPT/mono_sr_57500/model.safetensors" --baseline_wm_config_path "$CFG/mono_sr.yaml" \
           --baseline_wm_dataset_stat_path "$REAL_STAT" --baseline_wm_num_inference_steps 50 ;;
    baseline_real_only)
      echo --baseline_wm_model_ckpt_path "$CKPT/mono_r/model.safetensors" --baseline_wm_config_path "$CFG/mono_r.yaml" \
           --baseline_wm_dataset_stat_path "$REAL_STAT" --baseline_wm_num_inference_steps 50 --max_batch_size 8 ;;
    *) echo "unknown variant $1" >&2; exit 1 ;;
  esac
}

run_targets() {
  local legacy=()
  [[ "$PAPER_TARGETS" == "1" ]] && legacy=(--legacy_default_frame_mapping)
  for split in id ood; do
    # shellcheck disable=SC2046
    "$PYTHON" scripts/inference_wm1_to_wm2_controllability_eval.py \
      --phase targets --split_name "$split" \
      $(dataset_args "$split") --dataset_stat_path "$REAL_STAT" --mode val \
      $(sample_args "$split") --dataset_id 0 --seed 42 \
      --sampling_scopes whole_pose,per_dim --num_whole_pose_targets 5 --num_per_dim_targets 1 \
      ${legacy[@]+"${legacy[@]}"} --output_path "$OUT_ROOT/$split" ${EXTRA[@]+"${EXTRA[@]}"}
  done
  "$PYTHON" scripts/paper/merge_targets_manifests.py \
    --id_manifest "$OUT_ROOT/id/targets_manifest.json" \
    --ood_manifest "$OUT_ROOT/ood/targets_manifest.json" \
    --output "$OUT_ROOT/combined_targets_manifest.json"
}

run_rollouts() {
  for split in id ood; do
    for variant in "${VARIANTS[@]}"; do
      local wandb=()
      if [[ -n "${WANDB_PROJECT:-}" ]]; then
        wandb=(--wandb_project_name "$WANDB_PROJECT" --wandb_run_name "controllability_${split}_${variant}")
      fi
      # shellcheck disable=SC2046
      "$PYTHON" scripts/inference_wm1_to_wm2_controllability_eval.py \
        --phase rollout --targets_manifest_path "$OUT_ROOT/$split/targets_manifest.json" \
        $(dataset_args "$split") --dataset_stat_path "$REAL_STAT" --mode val --dataset_id 0 --seed 42 \
        --approach_styles direct,linear --ar_num_steps 5 \
        $(model_args "$variant") --variant_label "$variant" \
        --output_path "$OUT_ROOT/$split" ${wandb[@]+"${wandb[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
    done
  done
}

case "$STAGE" in
  targets)  run_targets ;;
  rollouts) run_rollouts ;;
  all)      run_targets; run_rollouts ;;
  *) echo "usage: $0 [targets|rollouts|all] [variant ...]" >&2; exit 1 ;;
esac
