# from diffusers import StableVideoDiffusionPipeline
import sys, os
import zipfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CtrlWorld, latent_seg_heightstack_to_channelstack
from typing import Dict

import numpy as np
import torch
import einops
import lpips
from accelerate import Accelerator
import datetime
import time
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import wandb
import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from diffusers.optimization import SchedulerType, get_scheduler

from config import wm_orca_args

import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# If using GPU
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True


def _latent_control_passthrough_adapter(segmentation_videos, target_frames: int, target_size):
    """Use precomputed latent control tensor directly (no RGB adapter CNN)."""
    bsz, num_frames = segmentation_videos.shape[:2]
    if num_frames < target_frames:
        pad_frames = target_frames - num_frames
        pad = segmentation_videos[:, -1:].repeat(1, pad_frames, 1, 1, 1)
        segmentation_videos = torch.cat([segmentation_videos, pad], dim=1)
    elif num_frames > target_frames:
        segmentation_videos = segmentation_videos[:, :target_frames]

    x = segmentation_videos.flatten(0, 1)
    if x.shape[-2:] != target_size:
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=target_size)
    x = x.view(bsz, target_frames, x.shape[1], target_size[0], target_size[1])
    return x


def _resolve_controlnet_inference_inputs(
    acc_model: CtrlWorld,
    args: wm_orca_args,
    batch: Dict,
    gt_latents_dtype: torch.dtype,
    device: torch.device,
):
    if not args.use_controlnet_conditioning:
        return None, acc_model.segmentation_to_control

    use_latent_control = bool(getattr(args, "use_latent_segmentation_for_controlnet", False))
    use_vae_roundtrip = bool(getattr(args, "use_vae_roundtrip_for_controlnet", False))
    use_vis_seg_actions = bool(getattr(args, "use_vis_seg_actions_for_controlnet", False))
    enabled_control_sources = int(use_latent_control) + int(use_vae_roundtrip) + int(use_vis_seg_actions)
    if enabled_control_sources > 1:
        raise ValueError(
            "Ambiguous ControlNet source flags. Enable at most one of "
            "use_latent_segmentation_for_controlnet, "
            "use_vae_roundtrip_for_controlnet, "
            "use_vis_seg_actions_for_controlnet."
        )

    if use_latent_control:
        if "latent_segmentation_videos" not in batch:
            raise KeyError("use_latent_segmentation_for_controlnet requires batch['latent_segmentation_videos'].")
        control_latents = batch["latent_segmentation_videos"].to(device=device, dtype=gt_latents_dtype, non_blocking=True)
        control_latents = latent_seg_heightstack_to_channelstack(control_latents, num_views=args.num_views)
        if control_latents.shape[2] != acc_model.unet.config.in_channels:
            raise ValueError(
                "Latent ControlNet conditioning channels mismatch during inference: "
                f"got {control_latents.shape[2]}, expected {acc_model.unet.config.in_channels}."
            )
        return control_latents, _latent_control_passthrough_adapter

    if use_vae_roundtrip:
        if "latent_segmentation_videos" not in batch:
            raise KeyError("use_vae_roundtrip_for_controlnet requires batch['latent_segmentation_videos'].")
        latent_segs = batch["latent_segmentation_videos"].to(device=device, dtype=gt_latents_dtype, non_blocking=True)
        segmentation_videos = acc_model._vae_decode_latent_segs(latent_segs)
        return segmentation_videos, acc_model.segmentation_to_control

    if use_vis_seg_actions:
        if "vis_seg_actions" not in batch:
            raise KeyError("use_vis_seg_actions_for_controlnet requires batch['vis_seg_actions'].")
        segmentation_videos = batch["vis_seg_actions"].to(device=device, dtype=gt_latents_dtype, non_blocking=True)
        return segmentation_videos, acc_model.segmentation_to_control

    if "segmentation_videos" not in batch:
        raise KeyError("ControlNet conditioning requires batch['segmentation_videos'].")
    segmentation_videos = batch["segmentation_videos"].to(device=device, dtype=gt_latents_dtype, non_blocking=True)
    return segmentation_videos, acc_model.segmentation_to_control


def get_exponential_lr_range_test_scheduler(
    optimizer: torch.optim.Optimizer,
    start_lr: float,
    end_lr: float,
    num_steps: int,
):
    """Exponentially increase LR from start_lr to end_lr over num_steps."""
    if start_lr <= 0.0 or end_lr <= 0.0:
        raise ValueError("lr_range_test_start_lr and lr_range_test_end_lr must be > 0.")
    if end_lr <= start_lr:
        raise ValueError("lr_range_test_end_lr must be greater than lr_range_test_start_lr.")
    if num_steps < 2:
        raise ValueError("lr_range_test_num_steps must be >= 2.")

    gamma = (end_lr / start_lr) ** (1.0 / float(num_steps - 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = start_lr

    def lr_lambda(current_step: int):
        capped_step = min(current_step, num_steps - 1)
        return gamma ** capped_step

    return LambdaLR(optimizer, lr_lambda)


def build_lr_scheduler(args, optimizer):
    if args.lr_range_test:
        return get_exponential_lr_range_test_scheduler(
            optimizer=optimizer,
            start_lr=args.lr_range_test_start_lr,
            end_lr=args.lr_range_test_end_lr,
            num_steps=args.lr_range_test_num_steps,
        )

    num_warmup_steps = args.lr_warmup_steps
    if num_warmup_steps is None:
        num_warmup_steps = max(1, int(0.03 * args.max_train_steps))

    return get_scheduler(
        name=SchedulerType(args.lr_scheduler),
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

def save_training_checkpoint(save_path, accelerator, model, optimizer, logger, global_step=None, log_to_tracker=False):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    start_time = time.perf_counter()
    torch.save({
        "model": accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }, save_path)
    save_seconds = time.perf_counter() - start_time

    size_mb = os.path.getsize(save_path) / (1024 ** 2)
    throughput_mb_s = size_mb / save_seconds if save_seconds > 0 else float("inf")
    logger.info(
        f"Saved checkpoint to {save_path} "
        f"({size_mb:.2f} MiB in {save_seconds:.2f}s, {throughput_mb_s:.2f} MiB/s)"
    )

    if log_to_tracker and global_step is not None:
        accelerator.log(
            {
                "checkpoint_save_seconds": save_seconds,
                "checkpoint_size_mb": size_mb,
                "checkpoint_save_mb_per_sec": throughput_mb_s,
            },
            step=global_step,
        )

    return save_seconds, size_mb, throughput_mb_s


def _nearby_readable_checkpoint_candidates(checkpoint_path: str, max_candidates: int = 5):
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if not os.path.isdir(checkpoint_dir):
        return []

    candidates = []
    for filename in os.listdir(checkpoint_dir):
        if not (filename.endswith(".pt") or filename.endswith(".pth")):
            continue
        candidate = os.path.join(checkpoint_dir, filename)
        if candidate == checkpoint_path or not os.path.isfile(candidate):
            continue
        try:
            if not zipfile.is_zipfile(candidate):
                continue
            stat = os.stat(candidate)
        except OSError:
            continue
        candidates.append((stat.st_mtime, candidate, stat.st_size))

    candidates.sort(reverse=True)
    return candidates[:max_candidates]


def _load_torch_checkpoint(checkpoint_path: str, checkpoint_label: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_label} checkpoint does not exist: {checkpoint_path}")
    if os.path.isdir(checkpoint_path):
        raise IsADirectoryError(
            f"{checkpoint_label} checkpoint path is a directory, expected a .pt/.pth/.safetensors file: "
            f"{checkpoint_path}"
        )

    if checkpoint_path.endswith(".safetensors"):
        # Released weights are model-only safetensors files (a flat state_dict, no optimizer).
        from safetensors.torch import load_file as load_safetensors_file
        return {"model": load_safetensors_file(checkpoint_path, device="cpu")}

    try:
        return torch.load(checkpoint_path, map_location='cpu')
    except RuntimeError as exc:
        message = str(exc)
        is_zip_read_error = (
            "PytorchStreamReader failed reading zip archive" in message
            or "failed finding central directory" in message
        )
        if not is_zip_read_error:
            raise

        size_bytes = os.path.getsize(checkpoint_path)
        suggestions = _nearby_readable_checkpoint_candidates(checkpoint_path)
        suggestion_text = ""
        if suggestions:
            formatted = [
                f"{path} ({size / (1024 ** 3):.2f} GiB)"
                for _, path, size in suggestions
            ]
            suggestion_text = " Nearby readable checkpoints: " + "; ".join(formatted)

        raise RuntimeError(
            f"Could not load {checkpoint_label} checkpoint from {checkpoint_path}. "
            "PyTorch reported an unreadable zip archive, which usually means the "
            f"checkpoint file is incomplete or truncated. File size is "
            f"{size_bytes / (1024 ** 3):.2f} GiB. Use an earlier complete checkpoint "
            f"or regenerate this checkpoint.{suggestion_text}"
        ) from exc


def _extract_checkpoint_model_state(checkpoint):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def _extract_checkpoint_optimizer_state(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get('optimizer', None)
    return None


def _is_lora_adapter_key(key: str) -> bool:
    return '.lora_down.' in key or '.lora_up.' in key


def load_finetune_lora_checkpoint(model, lora_ckpt_path: str, restore_optimizer: bool = True):
    if lora_ckpt_path is None:
        return None
    if not getattr(model, "_unet_lora_enabled", False):
        raise RuntimeError(
            "finetune_lora_ckpt_path requires use_unet_lora=True so LoRA modules "
            "exist before loading the LoRA checkpoint."
        )

    print(f"Loading finetune LoRA checkpoint from {lora_ckpt_path}!")
    checkpoint = _load_torch_checkpoint(lora_ckpt_path, 'finetune LoRA')
    state_dict = _extract_checkpoint_model_state(checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"LoRA checkpoint must contain a state_dict mapping, got {type(state_dict).__name__}."
        )

    current_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    unexpected_keys = sorted(checkpoint_keys - current_keys)
    if unexpected_keys:
        raise RuntimeError(
            "LoRA checkpoint contains keys that are not present in the current model. "
            f"First unexpected keys: {unexpected_keys[:8]}"
        )

    missing_keys = sorted(current_keys - checkpoint_keys)
    adapter_only = bool(checkpoint_keys) and all(
        _is_lora_adapter_key(key) for key in checkpoint_keys
    )
    if missing_keys and not adapter_only:
        raise RuntimeError(
            "finetune_lora_ckpt_path must point to either a full LoRA-wrapped model "
            "checkpoint or an adapter-only checkpoint containing only lora_down/lora_up "
            f"weights. Missing {len(missing_keys)} current model keys; first missing keys: "
            f"{missing_keys[:8]}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=not adapter_only)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading LoRA checkpoint: {unexpected[:8]}")
    if adapter_only:
        print(
            "Loaded adapter-only LoRA checkpoint; base checkpoint weights remain from ckpt_path."
        )
    else:
        print("Loaded full LoRA-wrapped finetune checkpoint.")

    optimizer_state = _extract_checkpoint_optimizer_state(checkpoint)
    if optimizer_state is not None:
        if restore_optimizer:
            print("Optimizer state found in LoRA checkpoint; will restore after prepare().")
        else:
            print("Optimizer state found in LoRA checkpoint; ignored for inference.")
    else:
        if restore_optimizer:
            print("No optimizer state in LoRA checkpoint; optimizer starts cold.")
        else:
            print("No optimizer state in LoRA checkpoint.")
    return optimizer_state


def build_model_and_maybe_load_checkpoint(args):
    finetune_before_lora = bool(getattr(args, "finetune_from_checkpoint_before_lora", False))
    enable_lora_after_checkpoint = finetune_before_lora and bool(getattr(args, "use_unet_lora", False))
    if enable_lora_after_checkpoint:
        args.use_unet_lora = False
    model = CtrlWorld(args)
    if enable_lora_after_checkpoint:
        args.use_unet_lora = True

    optimizer_state_to_restore = None
    if args.ckpt_path is not None:
        print(f"Loading existing checkpoint from {args.ckpt_path}!")
        checkpoint = _load_torch_checkpoint(args.ckpt_path, 'base')
        # New format: {'model': ..., 'optimizer': ...}
        # Old format: bare state dict (backward compat)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(_extract_checkpoint_model_state(checkpoint), strict=True)
            if finetune_before_lora:
                print("Finetuning from checkpoint; optimizer state ignored and Adam starts cold.")
            else:
                optimizer_state_to_restore = checkpoint.get('optimizer', None)
                if optimizer_state_to_restore is not None:
                    print("Optimizer state found in checkpoint; will restore after prepare().")
                else:
                    print("No optimizer state in checkpoint (old format); optimizer starts cold.")
        else:
            model.load_state_dict(checkpoint, strict=True)
            print("Old-format checkpoint (model weights only); optimizer starts cold.")

    if enable_lora_after_checkpoint:
        model.enable_unet_lora(
            rank=getattr(args, "unet_lora_rank", 8),
            alpha=getattr(args, "unet_lora_alpha", 8.0),
        )

    lora_optimizer_state = load_finetune_lora_checkpoint(
        model, getattr(args, "finetune_lora_ckpt_path", None)
    )
    if lora_optimizer_state is not None:
        optimizer_state_to_restore = lora_optimizer_state

    return model, optimizer_state_to_restore


def main(args):
    logger = get_logger(__name__, log_level="INFO")
    
    # allows you to log when using multiple GPUs
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_dir=args.output_dir
    )

    model, _optimizer_state_to_restore = build_model_and_maybe_load_checkpoint(args)
    model.to(accelerator.device)
    model.train()

    # using AdamW as optimizer (same learning rate for all modules)
    base_lr = args.lr_range_test_start_lr if args.lr_range_test else args.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    # Showing a few intersting stats about the model that we're about to train
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        tag = args.tag
        run_name = f"train_{now}_{tag}"
        # Convert args to dict for wandb
        config_dict = args_to_dict(args)
        accelerator.init_trackers(args.wandb_project_name, config=config_dict, init_kwargs={"wandb":{"name":run_name}})
        os.makedirs(args.output_dir, exist_ok=True)
        # count parameters num in each part
        num_params = sum(p.numel() for p in model.unet.parameters())
        print(f"Number of parameters in the unet: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.vae.parameters())
        print(f"Number of parameters in the vae: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"Number of parameters in the image_encoder: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.text_encoder.parameters())
        print(f"Number of parameters in the text_encoder: {num_params/1000000:.2f}M")
        if model.action_encoder is not None:
            num_params = sum(p.numel() for p in model.action_encoder.parameters())
            print(f"Number of parameters in the action_encoder: {num_params/1000000:.2f}M")

    # Loading both the train and validation dataset
    from dataset.dataset_orca import Dataset_mix
    train_dataset = Dataset_mix(args,mode='train')
    print(f"Number of training samples: {len(train_dataset)}")
    val_dataset = Dataset_mix(args,mode='val')
    print(f"Number of validation samples: {len(val_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        num_workers=args.num_workers,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.validation_batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )


    # Prepare everything with our accelerator
    # does the GPU distribution
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    # Restore optimizer state after prepare() so Adam's m/v moment buffers are on the
    # correct device and the loss trajectory continues without a cold-start plateau.
    if _optimizer_state_to_restore is not None:
        optimizer.load_state_dict(_optimizer_state_to_restore)
        print("Optimizer state restored — Adam moments carried over from checkpoint.")

    # Build scheduler after `prepare` and keep it unwrapped so its step frequency
    # is controlled only by this script (once per optimizer update), not by
    # Accelerate's process-aware scheduler wrapper.
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # If resuming (or intentionally re-indexing) training, align the LR scheduler with the
    # intended optimizer-step index. Diffusers `get_scheduler()` returns a `LambdaLR`, and
    # calling `step(n)` sets the internal `last_epoch` and updates the optimizer LR.
    if args.starting_global_step and args.starting_global_step > 0:
        try:
            lr_scheduler.step(args.starting_global_step)
        except TypeError:
            # Fallback for schedulers that don't accept an explicit epoch argument
            for _ in range(args.starting_global_step):
                lr_scheduler.step()
   
    ############################ training ##############################
    # printing training information
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # `len(train_dataloader)` is per-process after `accelerator.prepare()`.
    # We want `max_train_steps` to mean *optimizer updates* (global_step), so compute epochs from
    # update-steps-per-epoch (which already accounts for multi-GPU via the sharded dataloader).
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = max(1, math.ceil(args.max_train_steps / num_update_steps_per_epoch))
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  checkpointing_steps = {args.checkpointing_steps}")
    logger.info(f"  validation_steps = {args.validation_steps}")
    if args.lr_range_test:
        logger.info(
            "  LR range test enabled: "
            f"{args.lr_range_test_start_lr} -> {args.lr_range_test_end_lr} "
            f"over {args.lr_range_test_num_steps} steps"
        )

    if accelerator.is_main_process and not args.lr_range_test:
        warmup = args.lr_warmup_steps or 0
        peak_lr = args.learning_rate
        total = args.max_train_steps
        num_cycles = args.lr_num_cycles if args.lr_num_cycles is not None else 0.5
        def _lr_at(step):
            if step < warmup:
                return peak_lr * step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return peak_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
        print("\n  ---- LR schedule preview ----")
        print(f"  {'Step':>8}  {'LR':>12}  {'% of peak':>10}")
        print(f"  {'-'*35}")
        milestones = sorted(set([
            0, warmup,
            args.starting_global_step if args.starting_global_step else 0,
            total // 4, total // 2, total * 3 // 4, total,
        ]))
        for s in milestones:
            lr = _lr_at(s)
            tag = ""
            if s == (args.starting_global_step or 0):
                tag = "  <-- resume"
            elif s == warmup:
                tag = "  <-- peak"
            elif s == total:
                tag = "  <-- end"
            print(f"  {s:>8}  {lr:>12.4e}  {lr/peak_lr*100:>9.1f}%{tag}")
        print(f"  {'-'*35}\n")


    first_validation_step = True
    # Initialize training counters
    global_step = args.starting_global_step  # Counts optimizer steps (after gradient accumulation)
    forward_step = 0  # Counts forward passes (can be multiple per optimizer step)
    train_loss = 0.0  # Accumulates loss for logging
    range_test_records = []
    # Create progress bar (only on main process to avoid duplicates)
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    try:
        # Main training loop over epochs
        for epoch in range(num_train_epochs):
            # Iterate through batches in the training dataloader
            for step, batch in enumerate(train_dataloader):
                # Context manager handles gradient accumulation logic
                with accelerator.accumulate(model):
                    # Enable automatic mixed precision (fp16/bf16) for faster training
                    with accelerator.autocast():
                        # Forward pass: compute loss
                        loss_gen, _ = model(batch)
                    # Gather losses from all GPUs and compute mean for logging
                    avg_loss = accelerator.gather(loss_gen.repeat(args.train_batch_size)).mean()
                    # Accumulate loss (divide by grad accum steps for correct averaging)
                    train_loss += avg_loss.item()/ args.gradient_accumulation_steps
                    # Backward pass: compute gradients (handles multi-GPU synchronization)
                    accelerator.backward(loss_gen)
                    # Get model parameters for gradient clipping
                    params_to_clip = model.parameters()
                    # Check if gradients should be synchronized (true after gradient_accumulation_steps)
                    if accelerator.sync_gradients:
                        # Clip gradients to prevent exploding gradients
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        # Update model weights (only on real optimizer steps)
                        optimizer.step()
                        lr_scheduler.step()
                        # Clear gradients for next iteration
                        optimizer.zero_grad()
                    # Increment forward pass counter
                    forward_step += 1

                # LOGGING 
                # Only execute when gradients have been synchronized (i.e., actual optimizer step)
                if accelerator.sync_gradients:
                    # Update progress bar
                    progress_bar.update(1)
                    # Increment global step counter
                    global_step += 1
                    if global_step >= args.max_train_steps:
                        break

                    # Log average loss every 100 steps
                    if global_step %100 == 0:
                        # Update progress bar display with current loss
                        current_lr = optimizer.param_groups[0]["lr"]
                        progress_bar.set_postfix({"loss": train_loss, "lr": current_lr})
                        # Log to wandb
                        accelerator.log({"train_loss": train_loss/100, "learning_rate": current_lr}, step=global_step)
                        # Reset accumulated loss
                        train_loss = 0.0

                    if args.lr_range_test:
                        current_lr = optimizer.param_groups[0]["lr"]
                        range_test_records.append(
                            {
                                "step": global_step,
                                "learning_rate": current_lr,
                                "loss": avg_loss.item(),
                            }
                        )
                        accelerator.log(
                            {
                                "lr_range_test_loss": avg_loss.item(),
                                "learning_rate": current_lr,
                            },
                            step=global_step,
                        )
                        if len(range_test_records) >= args.lr_range_test_num_steps:
                            if accelerator.is_main_process:
                                os.makedirs(args.output_dir, exist_ok=True)
                                out_path = os.path.join(args.output_dir, "lr_range_test.csv")
                                with open(out_path, "w", encoding="utf-8") as f:
                                    f.write("step,learning_rate,loss\n")
                                    for rec in range_test_records:
                                        f.write(
                                            f"{rec['step']},{rec['learning_rate']},{rec['loss']}\n"
                                        )
                                logger.info(f"Saved LR range test results to {out_path}")
                            accelerator.wait_for_everyone()
                            return
                    # Save model checkpoint at specified intervals (only on main process)
                    if (not args.lr_range_test) and global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                        save_training_checkpoint(
                            save_path=save_path,
                            accelerator=accelerator,
                            model=model,
                            optimizer=optimizer,
                            logger=logger,
                            global_step=global_step,
                            log_to_tracker=True,
                        )
                    # Generate validation videos at specified intervals (only on main process)
                    remainder = 1 if first_validation_step else 0
                    if (not args.lr_range_test) and global_step % args.validation_steps == remainder:
                        accelerator.wait_for_everyone()

                        model.eval()

                        # Aggregate metrics across validation dataloader
                        all_pred_videos = []
                        all_gt_videos = []
                        all_pred_latents = []
                        all_gt_latents = []

                        # Determine how many batches to validate
                        if first_validation_step:
                            max_samples = 4
                        else:
                            max_samples = args.max_num_samples_for_validation

                        max_batches = max(1, math.ceil(max_samples / args.validation_batch_size))

                        for batch_idx, batch in enumerate(val_dataloader):
                            if batch_idx >= max_batches:
                                break

                            # Generate predictions
                            with torch.no_grad():
                                with accelerator.autocast():
                                    pred_video, gt_video, pred_latents, gt_latents = validate_video_generation(
                                        model=model,
                                        batch=batch,
                                        args=args,
                                        accelerator=accelerator
                                    )

                            all_pred_videos.append(pred_video)
                            all_gt_videos.append(gt_video)
                            all_pred_latents.append(pred_latents)
                            all_gt_latents.append(gt_latents)

                        pred_videos = torch.cat(all_pred_videos, dim=0)
                        gt_videos = torch.cat(all_gt_videos, dim=0)
                        pred_latents = torch.cat(all_pred_latents, dim=0)
                        gt_latents = torch.cat(all_gt_latents, dim=0)

                        if accelerator.num_processes > 1:
                            # gather_for_metrics can trim tensors at end-of-dataloader based on remainder.
                            # Here we aggregate pre-collected validation tensors, so use plain gather.
                            pred_videos = accelerator.gather(pred_videos)
                            gt_videos = accelerator.gather(gt_videos)
                            pred_latents = accelerator.gather(pred_latents)
                            gt_latents = accelerator.gather(gt_latents)

                        # Compute metrics on all collected samples
                        if accelerator.is_main_process:
                            compute_and_visualize_metrics(
                                pred_videos,
                                gt_videos,
                                pred_latents,
                                gt_latents,
                                args,
                                accelerator.unwrap_model(model).pipeline,
                                global_step,
                                args.output_dir,
                                accelerator
                            )

                        accelerator.wait_for_everyone()                        

                        # Cleanup
                        del pred_videos, gt_videos, pred_latents, gt_latents
                        model.train()
                        
                        if first_validation_step:
                            first_validation_step = False
                            print("First validation step completed")
            if global_step >= args.max_train_steps:
                break
                        
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"checkpoint-interrupted-{global_step}.pt")
            save_training_checkpoint(
                save_path=save_path,
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                logger=logger,
            )
            logger.info(f"Training interrupted by user. Saved checkpoint to {save_path}")
        raise
    except Exception as e:
        logger.error(f"Error in training: {e}")
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"checkpoint-interrupted-{global_step}.pt")
            save_training_checkpoint(
                save_path=save_path,
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                logger=logger,
            )
            logger.info(f"Error in training. Saved checkpoint to {save_path}")
        raise


def main_val(args):
    accelerator = Accelerator()
    model = CtrlWorld(args)
    # load form val_model_path
    print("load from val_model_path",args.val_model_path)
    model.load_state_dict(torch.load(args.val_model_path))
    model.to(accelerator.device)
    model.eval()
    validate_video_generation(model, None, args, accelerator)
    
            

def validate_video_generation(model: CtrlWorld, batch: Dict, args: wm_orca_args, accelerator: Accelerator):
    device = accelerator.device
    acc_model = accelerator.unwrap_model(model)
    pipeline = acc_model.pipeline

    # get all necessary information from batch
    text = batch['text']
    gt_latents = batch[args.predicted_datatype].to(device, non_blocking=True)

    ## Concatenation Conditioning 
    concatenate_latent = None
    if args.concatenate_latent:
        concatenate_latent = batch[args.concatenate_latent].to(device, non_blocking=True)
    segmentation_videos, segmentation_to_control = _resolve_controlnet_inference_inputs(
        acc_model=acc_model,
        args=args,
        batch=batch,
        gt_latents_dtype=gt_latents.dtype,
        device=device,
    )

    ## split gt_latents into history and future
    his_latent_gt = gt_latents[:,:args.num_history]
    future_latent_gt = gt_latents[:,args.num_history:]
    current_latent_gt = future_latent_gt[:,0]

    # Validate latent layout without hardcoding square latent sizes.
    expected_channels = 4
    expected_latent_width = args.width // args.vae_compression_rate
    # Match dataset construction order: compress each view first, then stack views.
    # This avoids off-by-one mismatches for heights not divisible by compression rate
    # (e.g., 180 // 8 = 22, then for 2 views => 44).
    expected_stacked_latent_height = (
        args.num_views * (args.height // args.vae_compression_rate)
    )
    assert current_latent_gt.shape[1:] == (
        expected_channels,
        expected_stacked_latent_height,
        expected_latent_width,
    ), (
        "Unexpected latent shape. "
        f"Got {tuple(current_latent_gt.shape[1:])}, "
        f"expected ({expected_channels}, {expected_stacked_latent_height}, {expected_latent_width})."
    )
    
    bsz, time_steps = gt_latents.shape[:2]
    if "dino_visual" in args.action_encoder:
        visual_actions = batch['segmentation_videos'].to(device, non_blocking=True)
        assert visual_actions.shape[1:] == (int(args.num_frames+args.num_history), args.num_views*3, 256, 256)
        action_latent = acc_model.action_encoder(visual_actions) # (B, T, 1024)
    elif args.action_encoder in ("action_encoder", "action_encoder_dual"):
        actions = batch['action'].to(device, non_blocking=True)
        assert actions.shape[1:] == (int(args.num_frames+args.num_history), args.action_dim)
        action_latent = acc_model.action_encoder(actions, text, acc_model.tokenizer, acc_model.text_encoder, args.frame_level_cond) # (8, 1, 1024)
    else:
        action_latent = torch.zeros((bsz, time_steps, 1024), device=device)

    _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
        pipeline,
        image=current_latent_gt,
        text=action_latent,
        width=args.width,
        height=int(args.num_views*args.height),
        num_frames=args.num_frames,
        history=his_latent_gt,
        num_inference_steps=args.num_inference_steps,
        decode_chunk_size=args.decode_chunk_size,
        max_guidance_scale=args.guidance_scale,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        mask=None,
        output_type='latent',
        return_dict=False,
        frame_level_cond=args.frame_level_cond,
        his_cond_zero=args.his_cond_zero,
        concatenate_latent=concatenate_latent,
        num_channels_concatenate=args.num_channels_concatenate,
        use_controlnet_conditioning=args.use_controlnet_conditioning,
        controlnet=acc_model.controlnet,
        segmentation_videos=segmentation_videos,
        segmentation_to_control=segmentation_to_control,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )
    
    # Turns tensor from (batch, time, channels, height, width) to (batch*num_views, time, channels, height, width)
    pred_latents = einops.rearrange(pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=args.num_views,n=1) # (B, 8, 4, 32,32)
    gt_latents = einops.rearrange(gt_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=args.num_views, n=1) # (B, 8, 4, 32,32)
    
    # decode latent
    decoded_video = []
    gt_bsz, gt_time_steps = gt_latents.shape[:2]
    gt_latents = gt_latents.flatten(0,1)
    decode_kwargs = {}
    for i in range(0,gt_latents.shape[0],args.decode_chunk_size):
        chunk = gt_latents[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
        decode_kwargs["num_frames"] = chunk.shape[0]
        decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
    gt_video = torch.cat(decoded_video,dim=0)
    gt_video = gt_video.reshape(gt_bsz, gt_time_steps, *gt_video.shape[1:])
    gt_latents = gt_latents.reshape(gt_bsz, gt_time_steps, *gt_latents.shape[1:])
    
    decoded_video = []
    bsz, time_steps = pred_latents.shape[:2]
    pred_latents = pred_latents.flatten(0,1)
    decode_kwargs = {}
    for i in range(0,pred_latents.shape[0],args.decode_chunk_size):
        chunk = pred_latents[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
        decode_kwargs["num_frames"] = chunk.shape[0]
        decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
    pred_video = torch.cat(decoded_video,dim=0)
    pred_video = pred_video.reshape(bsz, time_steps,*pred_video.shape[1:])

    # Turns from (batch*num_views*time, channels, height, width) to (batch, time, channels, height, width)
    pred_latents = pred_latents.reshape(bsz, time_steps, *pred_latents.shape[1:])

    print("pred_video",pred_video.shape)
    print("gt_video",gt_video.shape)
    print("pred_latents",pred_latents.shape)
    print("gt_latents",gt_latents.shape)

    return pred_video, gt_video, pred_latents, gt_latents


def compute_and_visualize_metrics(pred_videos, gt_videos, pred_latents, gt_latents, args, pipeline, train_steps, videos_dir, accelerator, additional_name_suffix=""):
    print("compute_and_visualize_metrics")
    print("pred_videos",pred_videos.shape)
    print("gt_videos",gt_videos.shape)
    print("pred_latents",pred_latents.shape)
    print("gt_latents",gt_latents.shape)

    # Compute metrics over a 1-second time window for fair comparison across different fps
    metrics = compute_video_metrics(pred_videos, gt_videos[:,args.num_history:], pred_latents, gt_latents[:,args.num_history:], fps=args.fps, time_window_seconds=1.0)
    print("after compute_video_metrics")
    print("pred_videos",pred_videos.shape)
    print("gt_videos",gt_videos.shape)
    print("pred_latents",pred_latents.shape)
    print("gt_latents",gt_latents.shape)
    # Torch versions: same operations but kept as torch tensors, detached and moved to CPU
    gt_video_torch = ((gt_videos.detach().cpu().float() / 2.0 + 0.5).clamp(0, 1)*255)
    ## From (batch*num_views, time, channels, height, width) to (batch*num_views, time, height, width, channels)
    gt_video_torch = gt_video_torch.permute(0,1,3,4,2).to(torch.uint8)
    pred_video_torch = ((pred_videos.detach().cpu().float() / 2.0 + 0.5).clamp(0, 1)*255)
    ## From (batch*num_views, time, channels, height, width) to (batch*num_views, time, height, width, channels)
    pred_video_torch = pred_video_torch.permute(0,1,3,4,2).to(torch.uint8)

    pred_videos = pred_video_torch.numpy()
    gt_videos = gt_video_torch.numpy()

    ## From (batch*num_views, time, height, width, channels) to (batch*num_views, time, height*2, width, channels) (ground ruth above and pred below)
    videos = np.concatenate([gt_videos[:,args.num_history:], pred_videos],axis=-3) #(2,16,512,256,3)
    ## From (batch*num_views, time, height*2, width, channels) to (time, height*2, width*batch*num_views, channels)
    videos = np.concatenate([video for video in videos],axis=-2).astype(np.uint8) # (16,512,256*batch,3)
    print("videos",videos.shape)

    # convert latents to video format
    pred_latents = ((pred_latents.detach().cpu().float() / 2.0 + 0.5).clamp(0, 1)*255).numpy()
    gt_latents = ((gt_latents.detach().cpu().float() / 2.0 + 0.5).clamp(0, 1)*255).numpy()

    # concatenate gt and pred latents
    concatenated_latents = np.concatenate([gt_latents[:, args.num_history:], pred_latents], axis=-2)
    concatenated_latents = np.concatenate([cl for cl in concatenated_latents], axis=-1).astype(np.uint8)
    print("concatenated_latents",concatenated_latents.shape)

    # Upload validation clips to W&B without persisting per-checkpoint mp4 files.
    log_dict = {}

    if additional_name_suffix != "":
        additional_name_suffix = "_" + additional_name_suffix
    width_of_one_sample_video = args.num_views * args.width
    width_of_one_sample_latent_video = args.num_views * args.width // args.vae_compression_rate
    available_video_num = min(
        videos.shape[-2] // width_of_one_sample_video,
        concatenated_latents.shape[-1] // width_of_one_sample_latent_video,
        len(metrics['psnr']),
    )
    requested_video_num = max(4, int(args.video_num))
    actual_video_num = min(requested_video_num, available_video_num)
    if actual_video_num <= 0:
        raise ValueError("Validation produced no videos to log to W&B.")
    if available_video_num < requested_video_num:
        print(
            f"Only {available_video_num} validation videos are available; "
            f"logging all of them instead of requested {requested_video_num}."
        )

    rng = np.random.default_rng(seed=int(train_steps))
    selected_video_ids = rng.choice(
        available_video_num, size=actual_video_num, replace=False
    ).tolist()
    print(f"Logging validation video ids to W&B: {selected_video_ids}")

    for slot_id, sample_id in enumerate(selected_video_ids):
        # Log per-video scalar metrics for the random visualization subset.
        log_dict[f"val_selected_sample_id_{slot_id}{additional_name_suffix}"] = sample_id
        log_dict[f"val_psnr_{slot_id}{additional_name_suffix}"] = metrics['psnr'][sample_id].item()
        log_dict[f"val_ssim_{slot_id}{additional_name_suffix}"] = metrics['ssim'][sample_id].item()
        log_dict[f"val_lpips_{slot_id}{additional_name_suffix}"] = metrics['lpips'][sample_id].item()
        log_dict[f"val_mse_latent_space_over_time_{slot_id}{additional_name_suffix}"] = metrics['mse_latent_space_over_time'][sample_id]
        log_dict[f"val_mse_img_space_over_time_{slot_id}{additional_name_suffix}"] = metrics['mse_img_space_over_time'][sample_id]
        log_dict[f"val_mae_latent_space_over_time_{slot_id}{additional_name_suffix}"] = metrics['mae_latent_space_over_time'][sample_id]
        log_dict[f"val_mae_img_space_over_time_{slot_id}{additional_name_suffix}"] = metrics['mae_img_space_over_time'][sample_id]

    # Build PSNR bins for the random visualization subset and log a WandB table.
    psnr_values = [metrics['psnr'][sample_id].item() for sample_id in selected_video_ids]
    psnr_min = min(psnr_values)
    psnr_max = max(psnr_values)
    num_bins = min(10, actual_video_num)
    if psnr_min == psnr_max or num_bins == 1:
        bin_edges = [psnr_min, psnr_max + 1e-6]
        num_bins = 1
    else:
        bin_width = (psnr_max - psnr_min) / num_bins
        bin_edges = [psnr_min + i * bin_width for i in range(num_bins + 1)]

    table = wandb.Table(columns=["slot_id", "sample_id", "psnr_bin", "psnr", "ssim", "lpips", "video", "latent_video"])
    for slot_id, sample_id in enumerate(selected_video_ids):
        psnr_val = psnr_values[slot_id]
        if num_bins == 1:
            bin_idx = 0
        else:
            bin_idx = min(int((psnr_val - psnr_min) / (psnr_max - psnr_min) * num_bins), num_bins - 1)
        bin_low = bin_edges[bin_idx]
        bin_high = bin_edges[bin_idx + 1]
        psnr_bin_label = f"psnr_{bin_low:.2f}_to_{bin_high:.2f}"

        # Add row to the WandB table
        video_clip = videos[:, :, sample_id*width_of_one_sample_video:(sample_id+1)*width_of_one_sample_video]
        video_clip = np.moveaxis(video_clip, -1, 1)
        latent_clip = concatenated_latents[:, :, :, sample_id*width_of_one_sample_latent_video:(sample_id+1)*width_of_one_sample_latent_video]
        table.add_data(
            slot_id,
            sample_id,
            psnr_bin_label,
            psnr_val,
            metrics['ssim'][sample_id].item(),
            metrics['lpips'][sample_id].item(),
            wandb.Video(video_clip, fps=args.wandb_video_display_fps, format="mp4"),
            wandb.Video(latent_clip, fps=args.wandb_video_display_fps, format="mp4"),
        )
    log_dict[f"videos_by_psnr{additional_name_suffix}"] = table

    log_dict[f"val_psnr_avg{additional_name_suffix}"] = np.mean(metrics['psnr'],axis=0).item()
    log_dict[f"val_ssim_avg{additional_name_suffix}"] = np.mean(metrics['ssim'],axis=0).item()
    
    log_dict[f"val_mse_img_space_over_time_avg{additional_name_suffix}"] = np.mean(metrics['mse_img_space_over_time'],axis=0)
    log_dict[f"val_mse_latent_space_over_time_avg{additional_name_suffix}"] = np.mean(metrics['mse_latent_space_over_time'],axis=0)
    log_dict[f"val_mae_img_space_over_time_avg{additional_name_suffix}"] = np.mean(metrics['mae_img_space_over_time'],axis=0)
    log_dict[f"val_mae_latent_space_over_time_avg{additional_name_suffix}"] = np.mean(metrics['mae_latent_space_over_time'],axis=0)
    log_dict[f"val_lpips_avg{additional_name_suffix}"] = np.mean(metrics['lpips'],axis=0).item()

    # Create plots for metrics over time (averaging over spatial and channel dimensions)
    log_dict[f"val_mse_img_space_over_batch_plot{additional_name_suffix}"] = plot_metric_over_time(
        metrics['mse_img_space_over_batch'], 
        'MSE (-1, 1) Space', 
        fps=args.fps
    )
    log_dict[f"val_mse_latent_space_over_batch_plot{additional_name_suffix}"] = plot_metric_over_time(
        metrics['mse_latent_space_over_batch'], 
        'MSE Latent Space', 
        fps=args.fps
    )
    log_dict[f"val_mae_img_space_over_batch_plot{additional_name_suffix}"] = plot_metric_over_time(
        metrics['mae_img_space_over_batch'], 
        'MAE (-1, 1) Space', 
        fps=args.fps
    )
        
    accelerator.log(log_dict, step=train_steps)
    
    return log_dict[f"val_lpips_avg{additional_name_suffix}"], log_dict[f"val_mse_latent_space_over_time_avg{additional_name_suffix}"], log_dict[f"val_psnr_avg{additional_name_suffix}"], log_dict[f"val_ssim_avg{additional_name_suffix}"]
    

def compute_video_metrics(pred_frames, gt_frames, pred_latents, gt_latents, fps=10, time_window_seconds=1.0):
    """
    Compute multiple metrics to evaluate predicted video quality against ground truth
    over a specific time window.
    
    Args:
        pred_frames: (B, T, 3, H, W) predicted frames in range [-1, 1], uint8
        gt_frames: (B, T, 3, H, W) ground truth frames in range [-1, 1], uint8
        fps: frame rate of the video (default: 10)
        time_window_seconds: time window in seconds to evaluate (default: 1.0)
    
    Returns:
        dict with metrics: psnr, ssim, mse, mae, lpips
    """
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    # Calculate number of frames corresponding to the time window
    frames_in_window = int(fps * time_window_seconds)
    total_frames = pred_frames.shape[1]
    
    # Ensure we don't exceed available frames
    frames_to_evaluate = min(frames_in_window, total_frames)
    
    # Slice to only evaluate the specified time window (first N frames)
    pred_latents = pred_latents[:, :frames_to_evaluate]
    gt_latents = gt_latents[:, :frames_to_evaluate]
    pred_frames = pred_frames[:, :frames_to_evaluate]
    gt_frames = gt_frames[:, :frames_to_evaluate]
    
    print(f"Evaluating metrics over {frames_to_evaluate} frames ({frames_to_evaluate/fps:.2f}s at {fps} fps)")

    # Move all data to CPU for metric computation (WandB requires numpy)
    pred_latents = pred_latents.detach().cpu().to(torch.float32)
    gt_latents = gt_latents.detach().cpu().to(torch.float32)
    pred_frames = pred_frames.detach().cpu().to(torch.float32)
    gt_frames = gt_frames.detach().cpu().to(torch.float32)
    
    # 1. MSE (Mean Squared Error) - lower is better
    mse_frames_over_time = torch.mean((pred_frames - gt_frames) ** 2, dim=(1, 2, 3, 4)) # we want to compute the mean over time dimension
    mse_frames_over_batch = torch.mean((pred_frames - gt_frames) ** 2, dim=(0, 2, 3, 4)) # we want to compute the mean over batch dimension to view the error accumulation over time
    mse_latents_over_time = torch.mean((pred_latents - gt_latents) ** 2, dim=(1, 2, 3, 4)) # we want to compute the mean over time dimension
    mse_latents_over_batch = torch.mean((pred_latents - gt_latents) ** 2, dim=(0, 2, 3, 4)) # we want to compute the mean over batch dimension to view the error accumulation over time
    
    # 2. MAE (Mean Absolute Error) - lower is better
    mae_frames_over_time = torch.mean(torch.abs(pred_frames - gt_frames), dim=(1, 2, 3, 4)) # we want to compute the mean over time dimension
    mae_frames_over_batch = torch.mean(torch.abs(pred_frames - gt_frames), dim=(0, 2, 3, 4)) # we want to compute the mean over batch dimension to view the error accumulation over time
    mae_latents_over_time = torch.mean(torch.abs(pred_latents - gt_latents), dim=(1, 2, 3, 4)) # we want to compute the mean over time dimension
    mae_latents_over_batch = torch.mean(torch.abs(pred_latents - gt_latents), dim=(0, 2, 3, 4)) # we want to compute the mean over batch dimension to view the error accumulation over time

    # Convert to numpy for image metrics
    gt_frames_np = gt_frames.numpy()
    pred_frames_np = pred_frames.numpy()
    
    # Get image dimensions for win_size calculation
    H, W = pred_frames_np.shape[-2], pred_frames_np.shape[-1]
    # SSIM win_size must be odd and smaller than image dimensions
    # Default is 7, but we'll use a smaller value if image is too small
    win_size = min(7, H - 1 if H % 2 == 0 else H, W - 1 if W % 2 == 0 else W)
    if win_size < 3:
        win_size = 3  # Minimum window size
    if win_size % 2 == 0:
        win_size -= 1  # Ensure odd
    
    # 3. PSNR (Peak Signal-to-Noise Ratio) - higher is better
    # Compute per-batch-item PSNR (averaged over time)
    psnr_per_batch = []
    for b in range(pred_frames_np.shape[0]):
        batch_psnr_values = []
        for t in range(pred_frames_np.shape[1]):
            psnr_val = psnr(gt_frames_np[b, t], pred_frames_np[b, t], data_range=2)
            batch_psnr_values.append(psnr_val)
        psnr_per_batch.append(np.mean(batch_psnr_values))
    psnr_per_batch = np.array(psnr_per_batch)
    
    # 4. SSIM (Structural Similarity Index) - higher is better (range [0,1])
    # Compute per-batch-item SSIM (averaged over time)
    ssim_per_batch = []
    for b in range(pred_frames_np.shape[0]):
        batch_ssim_values = []
        for t in range(pred_frames_np.shape[1]):
            ssim_val = ssim(gt_frames_np[b, t], pred_frames_np[b, t], 
                          channel_axis=0, data_range=2, win_size=win_size)
            batch_ssim_values.append(ssim_val)
        ssim_per_batch.append(np.mean(batch_ssim_values))
    ssim_per_batch = np.array(ssim_per_batch)
    
    # 5. LPIPS (Learned Perceptual Image Patch Similarity) - lower is better
    # Compute per-batch-item LPIPS (averaged over time)
    try:
        # LPIPS needs GPU for computation, but we'll move results back to CPU
        lpips_fn = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')

        # Reshape to (B*T, 3, H, W) for LPIPS computation
        B, T, _, H, W = pred_frames.shape
        # pred_frames is (B, T, 3, H, W), need (B*T, 3, H, W)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pred_flat = (pred_frames / 2.0 + 0.5).clamp(0, 1).reshape(B * T, 3, H, W).float()
        gt_flat = (gt_frames / 2.0 + 0.5).clamp(0, 1).reshape(B * T, 3, H, W).float()

        # Run LPIPS in fixed-size chunks rather than one B*T-sized forward pass:
        # a single shot over the whole batch scales peak GPU memory with B*T and
        # can OOM once B*T reaches a few hundred frames on an already-loaded GPU
        # (observed failing at B*T=1500 on a 24GB GPU with the WM2 model resident).
        # Chunking keeps this step's footprint constant regardless of dataset size.
        lpips_chunk_size = 32
        lpips_chunks = []
        with torch.no_grad():
            for start in range(0, pred_flat.shape[0], lpips_chunk_size):
                end = start + lpips_chunk_size
                pred_chunk = pred_flat[start:end].to(device)
                gt_chunk = gt_flat[start:end].to(device)
                chunk_values = lpips_fn(pred_chunk, gt_chunk, normalize=True)  # (chunk, 1, 1, 1)
                lpips_chunks.append(chunk_values.squeeze(-1).squeeze(-1).squeeze(-1).detach().cpu())
        lpips_values = torch.cat(lpips_chunks, dim=0).reshape(B, T).numpy()  # (B, T) on CPU
        lpips_per_batch = lpips_values.mean(axis=1)  # Average over time, per batch, as numpy
    except Exception as e:
        # Report NaN rather than 0 so a failed computation can't pass for a perfect score.
        print(f"WARNING: LPIPS computation failed ({e}); reporting NaN for this batch.")
        lpips_per_batch = np.full(pred_frames_np.shape[0], np.nan)
    
    # Convert all tensors to numpy arrays for WandB compatibility
    print("mse_frames_over_time",mse_frames_over_time.shape)
    print("mse_frames_over_batch",mse_frames_over_batch.shape)
    print("mse_latents_over_time",mse_latents_over_time.shape)
    print("mse_latents_over_batch",mse_latents_over_batch.shape)
    print("mae_frames_over_time",mae_frames_over_time.shape)
    print("mae_frames_over_batch",mae_frames_over_batch.shape)
    print("mae_latents_over_time",mae_latents_over_time.shape)
    print("mae_latents_over_batch",mae_latents_over_batch.shape)
    print("psnr_per_batch",psnr_per_batch.shape)
    print("ssim_per_batch",ssim_per_batch.shape)
    print("lpips_per_batch",lpips_per_batch.shape)
    return {
        'lpips': lpips_per_batch,
        'mse_img_space_over_time': mse_frames_over_time.numpy(),
        'mse_img_space_over_batch': mse_frames_over_batch.numpy(),
        'mse_latent_space_over_time': mse_latents_over_time.numpy(),
        'mse_latent_space_over_batch': mse_latents_over_batch.numpy(),
        'mae_img_space_over_time': mae_frames_over_time.numpy(),
        'mae_img_space_over_batch': mae_frames_over_batch.numpy(),
        'mae_latent_space_over_time': mae_latents_over_time.numpy(),
        'mae_latent_space_over_batch': mae_latents_over_batch.numpy(),
        'psnr': psnr_per_batch,
        'ssim': ssim_per_batch,
    }


def plot_metric_over_time(metric_array, metric_name, fps=None):
    """
    Plot a metric over time by averaging over spatial and channel dimensions.
    
    Args:
        metric_array: numpy array with shape (time, height, width, channel) or (time, channel, height, width)
        metric_name: string name of the metric for the plot title
        fps: optional frame rate for x-axis labeling
    
    Returns:
        wandb.Image object containing the plot
    """
    # Create time axis
    time_steps = np.arange(metric_array.shape[0]) + 1 # start from 1 since the first frame in prediction is at t+1
    if fps is not None:
        time_seconds = time_steps / fps
        x_label = 'Time (seconds)'
        x_data = time_seconds
    else:
        x_label = 'Time (frames)'
        x_data = time_steps
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_data, metric_array, linewidth=2)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert to wandb Image
    wandb_image = wandb.Image(fig)
    plt.close(fig)  # Close to free memory
    
    return wandb_image


def args_to_dict(args):
    """Convert args object to dictionary for wandb config."""
    config_dict = {}
    for attr_name in dir(args):
        if attr_name.startswith('_') or callable(getattr(args, attr_name)):
            continue
        try:
            value = getattr(args, attr_name)
            # Convert torch dtypes to strings
            if isinstance(value, torch.dtype):
                dtype_str_map = {
                    torch.float32: 'float32',
                    torch.float16: 'float16',
                    torch.bfloat16: 'bfloat16',
                }
                value = dtype_str_map.get(value, str(value))
            config_dict[attr_name] = value
        except Exception:
            continue
    return config_dict

if __name__ == "__main__":
    # reset parameters with command line
    from argparse import ArgumentParser
    from utils.config_loader import load_experiment_config, save_config_to_yaml, list_available_experiments

    parser = ArgumentParser()

    # Main way to specify experiment config
    parser.add_argument('--config', type=str, default=None,
                       help='Direct path to experiment config file')
    parser.add_argument('--list_experiments', action='store_true',
                       help='List all available experiment configurations')    

    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--dataset_stat_path', type=str, default=None,
                       help='stat.json used to normalize actions (overrides the dataset\'s own stat.json)')
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_names', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--lr_scheduler', type=str, default=None)
    parser.add_argument('--lr_warmup_steps', type=int, default=None)
    parser.add_argument('--lr_num_cycles', type=float, default=None)
    parser.add_argument('--lr_power', type=float, default=None)
    parser.add_argument('--validation_steps', type=int, default=None)
    parser.add_argument('--checkpointing_steps', type=int, default=None)
    parser.add_argument('--controlnet_conditioning_scale', type=float, default=None)
    parser.add_argument('--cond_dropout_mask_only_prob', type=float, default=None)
    parser.add_argument('--cond_dropout_action_only_prob', type=float, default=None)
    parser.add_argument('--cond_dropout_both_prob', type=float, default=None)
    parser.add_argument('--lr_range_test', action='store_true', default=None)
    parser.add_argument('--lr_range_test_start_lr', type=float, default=None)
    parser.add_argument('--lr_range_test_end_lr', type=float, default=None)
    parser.add_argument('--lr_range_test_num_steps', type=int, default=None)
    
    args_cli = parser.parse_args()
    # List experiments if requested
    if args_cli.list_experiments:
        print("\n" + "="*60)
        print("Available Experiments:")
        print("="*60)
        experiments = list_available_experiments()
        for exp in experiments:
            print(f"\n📋 {exp['name']}")
            print(f"   File: {exp['file']}")
            print(f"   Description: {exp['description']}")
        print("\n" + "="*60)
        exit(0)

    args = wm_orca_args()

    # Load experiment config if specified
    if args_cli.config:
        print(f"Loading experiment config from: {args_cli.config}")
        args = load_experiment_config(args_cli.config, args)
    else:
        print("No experiment config specified, using default config")
        exit(0)
    
    # Apply command-line overrides (these take precedence)
    for key, value in vars(args_cli).items():
        if value is not None and hasattr(args, key):
            print(f"Overriding {key}: {getattr(args, key)} -> {value}")
            setattr(args, key, value)
    
    # Compute derived values
    if args_cli.fps is not None:
        args.down_sample = int(args.original_fps / args_cli.fps)
    
    if args_cli.tag is not None:
        args.output_dir = f"model_ckpt/{args_cli.tag}"
        args.wandb_run_name = args_cli.tag
    
    # Save the final config used for this run
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    main(args)
