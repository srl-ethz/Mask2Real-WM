"""Export a training checkpoint as a release weight file.

Rebuilds the model the way inference does (base checkpoint plus an optional LoRA finetune
checkpoint, through load_ctrlworld_model_for_inference) and saves its complete state_dict as a
single fp32 safetensors file, without optimizer state. The file is then loaded back through
the release loading path and compared tensor by tensor with the source model.

Example:
    python scripts/export_release_checkpoint.py \
        --config experiments/mask2real/wm1_cascade_sr.yaml \
        --ckpt model_ckpt/midtraining_wm1_svd/checkpoint-55000.pt \
        --finetune_lora_ckpt model_ckpt/real_world_big_data_wm1_svd_finetune_pretrained_model/checkpoint-45000.pt \
        --stat dataset_meta_info/2026-05-19T03-28-53/SIM_PRETRAINING_DATA_ALL_lerobot_5fps/stat.json \
        --output_dir checkpoints/wm1_cascade_sr
"""

import argparse
import copy
import gc
import hashlib
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

from config import wm_orca_args
from scripts.inference_wm1_to_wm2 import load_ctrlworld_model_for_inference
from utils.config_loader import load_experiment_config, save_config_to_yaml


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="Experiment config of the model")
    parser.add_argument("--ckpt", required=True, help="Training checkpoint (.pt), or the base checkpoint of a LoRA stage")
    parser.add_argument("--finetune_lora_ckpt", default=None, help="LoRA finetune checkpoint applied on top of --ckpt")
    parser.add_argument("--stat", required=True, help="stat.json the model normalizes actions with")
    parser.add_argument("--output_dir", required=True)
    args_cli = parser.parse_args()

    os.makedirs(args_cli.output_dir, exist_ok=True)
    weights_path = os.path.join(args_cli.output_dir, "model.safetensors")

    model_args = load_experiment_config(args_cli.config, wm_orca_args())
    model_args.finetune_lora_ckpt_path = args_cli.finetune_lora_ckpt
    model = load_ctrlworld_model_for_inference(copy.deepcopy(model_args), args_cli.ckpt, "source")
    state = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()

    save_file(state, weights_path, metadata={
        "format": "pt",
        "config": os.path.basename(args_cli.config),
        "source_checkpoint": os.path.basename(os.path.dirname(args_cli.ckpt)) + "/" + os.path.basename(args_cli.ckpt),
        "source_lora_checkpoint": (
            os.path.basename(os.path.dirname(args_cli.finetune_lora_ckpt)) + "/" + os.path.basename(args_cli.finetune_lora_ckpt)
            if args_cli.finetune_lora_ckpt else ""
        ),
    })
    print(f"Saved {len(state)} tensors to {weights_path}")

    # Load the exported file the way released inference does and compare with the source model.
    reload_args = copy.deepcopy(model_args)
    reload_args.finetune_lora_ckpt_path = None
    reloaded = load_ctrlworld_model_for_inference(reload_args, weights_path, "exported")
    reloaded_state = reloaded.state_dict()
    if set(reloaded_state) != set(state):
        raise RuntimeError("Exported checkpoint loads into a model with different keys than the source model")
    mismatched = [k for k, v in state.items() if not torch.equal(v, reloaded_state[k].detach().cpu())]
    if mismatched:
        raise RuntimeError(f"{len(mismatched)} tensors differ after reloading, e.g. {mismatched[:5]}")
    print(f"Verified: all {len(state)} tensors identical after reloading {weights_path}")
    del reloaded, reloaded_state
    gc.collect()

    shutil.copyfile(args_cli.stat, os.path.join(args_cli.output_dir, "stat.json"))
    save_config_to_yaml(
        load_experiment_config(args_cli.config, wm_orca_args()),
        os.path.join(args_cli.output_dir, "resolved_config.yaml"),
    )
    with open(weights_path + ".sha256", "w") as f:
        f.write(f"{sha256_of(weights_path)}  model.safetensors\n")
    print(f"Done: {args_cli.output_dir}")


if __name__ == "__main__":
    main()
