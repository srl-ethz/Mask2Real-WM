#!/usr/bin/env bash
# Export the eight released models from their training checkpoints (model_ckpt/<run>/...) into
# checkpoints/<variant>/ with scripts/export_release_checkpoint.py. Documents which training
# checkpoint each released file comes from; it needs the original training checkpoints.
#
# Usage: scripts/export_release_checkpoints.sh [output_root] [variant ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python}
CKPT_ROOT=${CKPT_ROOT:-model_ckpt}
OUT=${1:-checkpoints}
shift || true

REAL_STAT=dataset_meta_info/2026-03-14T13-34-49/large_real_dataset_5fps_135_240/stat.json
SIM_STAT=dataset_meta_info/2026-05-19T03-28-53/SIM_PRETRAINING_DATA_ALL_lerobot_5fps/stat.json
SIM_RGB_STAT=dataset_meta_info/2026-09-03T19-36-55/sim_data_rgb_5fps/stat.json
CFG="$REPO_ROOT/experiments/mask2real"

# variant | config | checkpoint | LoRA finetune checkpoint | stat.json
SPECS=(
  "wm2|wm2.yaml|real_world_big_data_wm2_svd/checkpoint-70000.pt||$SIM_STAT"
  "wm1_cascade_r|wm1_cascade_r.yaml|real_world_big_data_wm1_svd_pretrained_weights/checkpoint-15000.pt||$REAL_STAT"
  "wm1_cascade_s|wm1_cascade_s.yaml|midtraining_wm1_svd/checkpoint-55000.pt||$SIM_STAT"
  "wm1_cascade_sr|wm1_cascade_sr.yaml|midtraining_wm1_svd/checkpoint-55000.pt|real_world_big_data_wm1_svd_finetune_pretrained_model/checkpoint-45000.pt|$SIM_STAT"
  "mono_r|mono_r.yaml|baseline_ctrl_world_svd_pretrained_weights/checkpoint-22000.pt||$REAL_STAT"
  "mono_s_base|mono_s_base.yaml|baseline_ctrl_world_with_svd_sim_midtraining/checkpoint-55000.pt||$SIM_RGB_STAT"
  "mono_sr_57500|mono_sr.yaml|baseline_ctrl_world_with_svd_sim_midtraining/checkpoint-55000.pt|baseline_ctrl_world_with_svd_finetuning_resumed_from_sim_midtrained/checkpoint-57500.pt|$REAL_STAT"
  "mono_sr_58000|mono_sr.yaml|baseline_ctrl_world_with_svd_sim_midtraining/checkpoint-55000.pt|baseline_ctrl_world_with_svd_finetuning_resumed_from_sim_midtrained/checkpoint-58000.pt|$REAL_STAT"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r variant config ckpt lora stat <<< "$spec"
  if [[ $# -gt 0 && ! " $* " =~ " $variant " ]]; then
    continue
  fi
  lora_args=()
  [[ -n "$lora" ]] && lora_args=(--finetune_lora_ckpt "$CKPT_ROOT/$lora")
  echo "=== $variant"
  "$PYTHON" "$REPO_ROOT/scripts/export_release_checkpoint.py" \
    --config "$CFG/$config" --ckpt "$CKPT_ROOT/$ckpt" ${lora_args[@]+"${lora_args[@]}"} \
    --stat "$stat" --output_dir "$OUT/$variant"
done
