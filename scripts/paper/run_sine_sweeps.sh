#!/usr/bin/env bash
# Sine-sweep rollouts for the human rating study: each of the 23 action dimensions follows one
# sine cycle (amplitude scale 1.0) while the others hold, rolled out autoregressively for 10
# steps (10 s at 5 fps). The same samples and per-model normalization statistics are used for
# all five models:
#
#   id : large_real_dataset_5fps_135_240 validation samples 83,63,94,17,89
#   ood: processed_OOD_combined_all4_5fps, "no_cube" category, episodes 0-4, samples 38,24,21,20,13
#
# Denoising steps come from the model configs (10 for WM1 and the monolithic models, 20 for WM2).
#
# Usage:
#   scripts/paper/run_sine_sweeps.sh [model ...]
#   models: cascade_r cascade_s cascade_sr mono_sr_58000 mono_r   (default: all)
#
# Environment: PYTHON (default python), OUT_ROOT (default outputs/sine_sweeps),
#              WANDB_MODE=offline to keep W&B logging local.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
OUT_ROOT=${OUT_ROOT:-outputs/sine_sweeps}
MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(cascade_r cascade_s cascade_sr mono_sr_58000 mono_r)
fi

CKPT=checkpoints
CFG=experiments/mask2real
REAL_STAT=dataset_meta_info/2026-03-14T13-34-49/large_real_dataset_5fps_135_240/stat.json
SIM_STAT=dataset_meta_info/2026-05-19T03-28-53/SIM_PRETRAINING_DATA_ALL_lerobot_5fps/stat.json

split_args() {  # split_args <split>
  case "$1" in
    id)  echo --dataset_root_path datasets/2026-03-14T13-34-49 \
              --dataset_names large_real_dataset_5fps_135_240 \
              --dataset_meta_info_path dataset_meta_info/2026-03-14T13-34-49 \
              --sample_indices 83,63,94,17,89 ;;
    ood) echo --dataset_root_path datasets/2026-04-05T20-09-04 \
              --dataset_names processed_OOD_combined_all4_5fps \
              --dataset_meta_info_path dataset_meta_info/2026-04-05T20-09-04 \
              --dataset_meta_info_name processed_OOD_combined_all4_5fps_val_no_cube \
              --episode_ids 0,1,2,3,4 --sample_indices 38,24,21,20,13 ;;
  esac
}

model_args() {  # model_args <model>
  local wm2=(--wm2_model_ckpt_path "$CKPT/wm2/model.safetensors" --wm2_config_path "$CFG/wm2.yaml"
             --wm2_dataset_stat_path "$SIM_STAT" --wm1_num_frames 5 --wm2_num_frames 5 --sequential_wm1_wm2_loading)
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
           --baseline_wm_dataset_stat_path "$REAL_STAT" --baseline_wm_num_frames 5 ;;
    mono_r)
      echo --baseline_wm_model_ckpt_path "$CKPT/mono_r/model.safetensors" --baseline_wm_config_path "$CFG/mono_r.yaml" \
           --baseline_wm_dataset_stat_path "$REAL_STAT" --baseline_wm_num_frames 5 ;;
    *) echo "unknown model $1" >&2; exit 1 ;;
  esac
}

for model in "${MODELS[@]}"; do
  for split in id ood; do
    # shellcheck disable=SC2046
    "$PYTHON" scripts/inference_wm1_to_wm2_sine_actions.py \
      $(split_args "$split") --dataset_stat_path "$REAL_STAT" --mode val --dataset_id 0 \
      $(model_args "$model") \
      --ar_num_steps 10 --sine_cycles 1.0 --sine_amplitude_scales 1.0 --variants_per_component 1 \
      --output_path "$OUT_ROOT/${split}_${model}" \
      --wandb_project_name mask2real-wm-sine-sweeps --wandb_run_name "sine_${split}_${model}"
  done
done
