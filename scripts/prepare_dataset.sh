#!/usr/bin/env bash
# Turn a converted ORCA recording folder (annotation/ + per-episode RGB and segmentation videos)
# into the latent dataset and meta-info that training and evaluation read:
#
#   1. dataset_example/extract_latent_orca.py   resample to 5 fps, resize to 135x240 and encode
#                                                the RGB and segmentation videos with the SVD VAE
#   2. dataset_meta_info/create_meta_info_orca.py  train/val sample lists and stat.json
#
# Usage:
#   scripts/prepare_dataset.sh <converted_dataset_dir> <output_root> [create_meta_info_orca.py args...]
#
# Outputs:
#   <output_root>/<timestamp>/<name>_5fps/                 latents, videos, annotations
#   dataset_meta_info/<timestamp>/<name>_5fps/{train,val}_sample.json, stat.json
#
# Environment overrides: ORIGINAL_FPS (source frame rate, default 25).
# Extra arguments are passed to create_meta_info_orca.py (see its --help), e.g.
# --sequence_length, --samples_per_traj or --val_episode_groups_json.
set -euo pipefail

usage="usage: $0 <converted_dataset_dir> <output_root> [create_meta_info_orca.py args...]"
SRC="${1:?$usage}"
OUT_ROOT="${2:?$usage}"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y-%m-%dT%H-%M-%S)"
NAME="$(basename "${SRC%/}")"

python dataset_example/extract_latent_orca.py \
  --orca_dataset_path "$SRC" \
  --orca_output_path "$OUT_ROOT" \
  --output_timestamp "$STAMP" \
  --frame_size "(135,240)" \
  --desired_fps 5 \
  --original_fps "${ORIGINAL_FPS:-25}" \
  --num_views 2

python dataset_meta_info/create_meta_info_orca.py \
  --orca_output_path "$OUT_ROOT/$STAMP/${NAME}_5fps" \
  --dataset_name "${NAME}_5fps" \
  "$@"
