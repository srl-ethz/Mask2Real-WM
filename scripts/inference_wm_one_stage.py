"""
Script for inference on the WM. Obtains all the outputs needed for evaluation.

We perform here both open loop and closed loop inference.

Open loop inference:
- We generate a video from a given action and given history and predict a specified horizon.

Closed loop inference:
- We generate a video from a given action and given history. Then feedback the generated video to the WM and generate the next frames.

Informations we store here for evaluation later:
- The decoded predicted video (uint8, (T, H, W, 3)) both as pt.
- The whole video (uint8, (T, H, W, 3)) as mp4 and uploaded to wandb. History and future frames concatenated. In the video it is marked when predicted and when ground truth.
- The action used to generate the video.
- The latents of the predicted video and its ground truth.
- Attention maps of the action conditioning as pt
- Noisy outputs of the unet: At t = 10, 20, 30, 40 as pt
"""

import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

import argparse
import datetime
from typing import Dict

import einops
import torch
import torch.nn.functional as F
import time
from accelerate import Accelerator

# config related imports
from config import wm_orca_args
from utils.config_loader import load_experiment_config
from utils.utils import send_discord_message

# dataset related imports
from dataset.dataset_orca import Dataset_mix

# model related imports
from models.ctrl_world import CtrlWorld, latent_seg_heightstack_to_channelstack
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from scripts.train_wm import _load_torch_checkpoint, compute_and_visualize_metrics, args_to_dict


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
        x = F.adaptive_avg_pool2d(x, output_size=target_size)
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


def perform_inference_one_stage_on_single_batch(
    model: CtrlWorld,
    args: wm_orca_args,
    accelerator: Accelerator,
    batch: Dict,
):
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
    

    # start generate
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

    return pred_video, gt_video, pred_latents, gt_latents


def main(wm_args: wm_orca_args, args_cli: argparse.Namespace): 
    accelerator = Accelerator(
        gradient_accumulation_steps=wm_args.gradient_accumulation_steps,
        mixed_precision=wm_args.mixed_precision,
        log_with='wandb',
        project_dir=args_cli.output_path
    )
    if accelerator.is_main_process:
        wandb_project_name = getattr(args_cli, "wandb_project_name", "ctrl-world")
        wandb_run_name = getattr(args_cli, "wandb_run_name", "inference")
        accelerator.init_trackers(
            wandb_project_name,
            config=args_to_dict(wm_args),
            init_kwargs={"wandb": {"name": wandb_run_name}},
        )

    # load model
    model = CtrlWorld(wm_args)
    checkpoint_payload = _load_torch_checkpoint(args_cli.model_ckpt_path, "model")
    state_dict = CtrlWorld._unwrap_state_dict(checkpoint_payload)
    model.load_state_dict(state_dict, strict=True)
    model.to(accelerator.device)
    model.eval()

    # Send notification when group starts
    start_msg = f"🚀 **Starting Inference**\n" \
                        f"📁 Config: `{args_cli.config_path}`\n" \
                        f"⏰ Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    send_discord_message(start_msg)

    # load test_dataset
    test_dataset = Dataset_mix(wm_args, mode=args_cli.mode)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=wm_args.validation_batch_size, shuffle=False)

    # wrap for accelerate 
    model, test_dataset_loader = accelerator.prepare(model, test_dataset_loader)

    # perform inference for each batch
    all_pred_videos = []
    all_gt_videos = []
    all_pred_latents = []
    all_gt_latents = []
    for batch in test_dataset_loader:
        with torch.no_grad():
            with accelerator.autocast():
                pred_video, gt_video, pred_latents, gt_latents = perform_inference_one_stage_on_single_batch(
                    model=model,
                    args=wm_args,
                    accelerator=accelerator,
                    batch=batch
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
        pred_videos = accelerator.gather_for_metrics(pred_videos)
        gt_videos = accelerator.gather_for_metrics(gt_videos)
        pred_latents = accelerator.gather_for_metrics(pred_latents)
        gt_latents = accelerator.gather_for_metrics(gt_latents)

    if accelerator.is_main_process:
        # compute metrics
        compute_and_visualize_metrics(
            pred_videos,
            gt_videos,
            pred_latents,
            gt_latents,
            wm_args,
            accelerator.unwrap_model(model).pipeline,
            0,
            args_cli.output_path,
            accelerator,
        )

        # save outputs
        os.makedirs(args_cli.output_path, exist_ok=True)
        torch.save(pred_videos, os.path.join(args_cli.output_path, "pred_videos.pt"))
        torch.save(gt_videos, os.path.join(args_cli.output_path, "gt_videos.pt"))
        torch.save(pred_latents, os.path.join(args_cli.output_path, "pred_latents.pt"))
        torch.save(gt_latents, os.path.join(args_cli.output_path, "gt_latents.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt_path", type=str, required=True,
                        help="Checkpoint to evaluate (.pt or released .safetensors)")
    parser.add_argument("--output_path", type=str, default=f"inference_output/test_analysis_wm_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Experiment config, e.g. experiments/mask2real/wm2_oracle_sam3/ood_all.yaml")
    parser.add_argument("--wandb_project_name", type=str, default="mask2real-wm-inference")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--debug", action="store_true", default=False)

    args_cli = parser.parse_args()

    wm_args = wm_orca_args()
    wm_args = load_experiment_config(args_cli.config_path, wm_args)

    if args_cli.debug:
        wm_args.wandb_run_name = "debug"
        wm_args.max_num_samples_for_validation = 6
        wm_args.max_num_samples = 6
        wm_args.validation_batch_size = 2
        wm_args.num_inference_steps = 1

    # starting inference
    main(wm_args, args_cli)
