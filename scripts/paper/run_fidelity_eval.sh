#!/usr/bin/env bash
# Video-fidelity evaluation: every model rolls out 10 autoregressive steps (5 frames each, 10 s
# at 5 fps) from validation samples with seed 0 and 50 denoising steps. Each run logs the
# per-sample PSNR / SSIM / LPIPS / MAE / MSE tables and the rollout videos to W&B; FVD, FID and
# the tables are computed from those runs (see docs/reproducing_paper_results.md).
#
# Datasets:
#   id                 large_real_dataset_5fps_135_240, 150 validation samples
#   ood_no_cube, ood_cube, ood_rotated_arena, ood_duck
#                      categories of processed_OOD_combined_all4_5fps, 50 validation samples each
#
# Usage:
#   scripts/paper/run_fidelity_eval.sh [dataset ...]      (default: all five)
#
# Environment:
#   MODELS         space-separated subset of
#                  "cascade_r cascade_s cascade_sr mono_sr_58000 mono_r" (default: all)
#   PYTHON         interpreter (default: python)
#   OUT_ROOT       output folder (default: outputs/fidelity)
#   WANDB_PROJECT  W&B project (default: mask2real-wm-fidelity)
#   BATCH_SIZE     validation batch size (default: 30 for the cascades, 24 for the monolithic
#                  models). The noise of a batch is drawn at once, so the batch size changes the
#                  individual samples.
#   DEBUG          1 runs the script's --debug mode
#
# GPU memory: the monolithic models fit on a 24 GB GPU with the default batch size. For the
# cascades, WM1 and WM2 are on the GPU at the same time, which needs more than 24 GB.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
OUT_ROOT=${OUT_ROOT:-outputs/fidelity}
WANDB_PROJECT=${WANDB_PROJECT:-mask2real-wm-fidelity}
read -r -a MODELS <<< "${MODELS:-cascade_r cascade_s cascade_sr mono_sr_58000 mono_r}"
DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(id ood_no_cube ood_cube ood_rotated_arena ood_duck)
fi

CKPT=checkpoints
CFG=experiments/mask2real
REAL_STAT=dataset_meta_info/2026-03-14T13-34-49/large_real_dataset_5fps_135_240/stat.json
SIM_STAT=dataset_meta_info/2026-05-19T03-28-53/SIM_PRETRAINING_DATA_ALL_lerobot_5fps/stat.json

EXTRA=()
[[ "${DEBUG:-0}" == "1" ]] && EXTRA+=(--debug)

dataset_args() {  # dataset_args <dataset>
  case "$1" in
    id)
      echo --dataset_root_path datasets/2026-03-14T13-34-49 \
           --dataset_names large_real_dataset_5fps_135_240 \
           --dataset_meta_info_path dataset_meta_info/2026-03-14T13-34-49 \
           --num_of_samples_for_inference 150 ;;
    ood_no_cube|ood_cube|ood_rotated_arena|ood_duck)
      echo --dataset_root_path datasets/2026-04-05T20-09-04 \
           --dataset_names processed_OOD_combined_all4_5fps \
           --dataset_meta_info_path dataset_meta_info/2026-04-05T20-09-04 \
           --dataset_meta_info_name "processed_OOD_combined_all4_5fps_val_${1#ood_}" \
           --num_of_samples_for_inference 50 ;;
    *) echo "unknown dataset $1" >&2; exit 1 ;;
  esac
}

model_args() {  # model_args <model>
  local wm2=(--wm2_model_ckpt_path "$CKPT/wm2/model.safetensors" --wm2_config_path "$CFG/wm2.yaml"
             --wm2_dataset_stat_path "$SIM_STAT"
             --wm1_num_frames 5 --wm2_num_frames 5 --wm1_num_history 5 --wm2_num_history 5
             --wm1_history_stride 1 --wm2_history_stride 1
             --wm1_num_inference_steps 50 --wm2_num_inference_steps 50
             --validation_batch_size "${BATCH_SIZE:-30}")
  local mono=(--baseline_wm_num_frames 5 --baseline_wm_num_history 5 --baseline_wm_history_stride 1
              --baseline_wm_num_inference_steps 50 --baseline_wm_dataset_stat_path "$REAL_STAT"
              --validation_batch_size "${BATCH_SIZE:-24}")
  case "$1" in
    cascade_r)
      echo --wm1_model_ckpt_path "$CKPT/wm1_cascade_r/model.safetensors" --wm1_config_path "$CFG/wm1_cascade_r.yaml" \
           --wm1_dataset_stat_path "$REAL_STAT" "${wm2[@]}" ;;
    cascade_s)
      echo --wm1_model_ckpt_path "$CKPT/wm1_cascade_s/model.safetensors" --wm1_config_path "$CFG/wm1_cascade_s.yaml" \
           --wm1_dataset_stat_path "$SIM_STAT" "${wm2[@]}" ;;
    cascade_sr)
      echo --wm1_model_ckpt_path "$CKPT/wm1_cascade_sr/model.safetensors" --wm1_config_path "$CFG/wm1_cascade_sr.yaml" \
           --wm1_dataset_stat_path "$SIM_STAT" "${wm2[@]}" ;;
    mono_sr_58000)
      echo --baseline_wm_model_ckpt_path "$CKPT/mono_sr_58000/model.safetensors" --baseline_wm_config_path "$CFG/mono_sr.yaml" \
           "${mono[@]}" ;;
    mono_r)
      echo --baseline_wm_model_ckpt_path "$CKPT/mono_r/model.safetensors" --baseline_wm_config_path "$CFG/mono_r.yaml" \
           "${mono[@]}" ;;
    *) echo "unknown model $1" >&2; exit 1 ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    # shellcheck disable=SC2046
    "$PYTHON" scripts/inference_wm1_to_wm2.py \
      $(dataset_args "$dataset") --mode val --inference_mode autoregressive --ar_num_steps 10 --seed 0 \
      $(model_args "$model") \
      --output_path "$OUT_ROOT/${dataset}_${model}" \
      --wandb_project_name "$WANDB_PROJECT" --wandb_run_name "fidelity_${dataset}_${model}" \
      ${EXTRA[@]+"${EXTRA[@]}"}
  done
done
