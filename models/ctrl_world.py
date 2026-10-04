# from diffusers import StableVideoDiffusionPipeline
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

import numpy as np
import copy
import torch
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import json
from decord import VideoReader, cpu
import wandb
import mediapy
from typing import Dict, Any


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# x: (B, T, 4, V*H, W) -> (B, T, V*4, H, W)
def latent_seg_heightstack_to_channelstack(x, num_views: int):
    if num_views <= 1:
        return x
    B, T, C, Htot, W = x.shape
    assert C == 4, f"Expected latent C=4, got {C}"
    assert Htot % num_views == 0, f"Htot={Htot} not divisible by num_views={num_views}"
    H = Htot // num_views
    x = x.contiguous().view(B, T, C, num_views, H, W)   # split stacked height
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()        # (B,T,V,4,H,W)
    x = x.view(B, T, num_views * C, H, W)               # (B,T,V*4,H,W)
    return x


class Action_encoder2(nn.Module):
    def __init__(self, action_dim, action_num, hidden_sizes, text_cond=True):
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        self.text_cond = text_cond

        input_dim = int(action_dim)

        # Create hidden layers: flatten list comprehension properly
        hidden_layers = []
        for hidden_size in self.hidden_sizes:
            hidden_layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
        
        self.action_encode = nn.Sequential(
            nn.Linear(input_dim, self.hidden_sizes[0]),
            nn.SiLU(),
            *hidden_layers,
            nn.Linear(self.hidden_sizes[-1], 1024),
        )
        # kaiming initialization
        nn.init.kaiming_normal_(self.action_encode[0].weight, mode='fan_in', nonlinearity='relu')
        if len(hidden_layers) > 0:
            nn.init.kaiming_normal_(self.action_encode[2].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, action,  texts=None, text_tokinizer=None, text_encoder=None, frame_level_cond=True,):
        # action: (B, action_num, action_dim)
        B,T,D = action.shape
        if not frame_level_cond:
            action = einops.rearrange(action, 'b t d -> b 1 (t d)')
        action = self.action_encode(action)

        if texts is not None and self.text_cond:
            # with 50% probability, add text condition
            with torch.no_grad():
                inputs = text_tokinizer(texts, padding='max_length', return_tensors="pt", truncation=True).to(text_encoder.device)
                outputs = text_encoder(**inputs)
                hidden_text = outputs.text_embeds # (B, 512)
                hidden_text = einops.repeat(hidden_text, 'b c -> b 1 (n c)', n=2) # (B, 1, 1024)
            
            action = action + hidden_text # (B, T, hidden_size)
        return action # (B, 1, hidden_size) or (B, T, hidden_size) if frame_level_cond


class DualActionEncoder(nn.Module):
    """
    Split action conditioning into:
      - EE pose branch (first `relative_pose_dims` values, e.g. xyz+rot6d),
      - absolute joint-angle branch (remaining values, normalized from radians).
    Each branch uses an Action_encoder2-style MLP to produce 1024-d features.
    The two embeddings are summed before being used as UNet cross-attention cond.
    """
    def __init__(self, action_dim, action_num, hidden_sizes, relative_pose_dims=9, text_cond=True):
        super().__init__()
        if action_dim <= relative_pose_dims:
            raise ValueError(
                f"DualActionEncoder requires action_dim > relative_pose_dims; got action_dim={action_dim}, "
                f"relative_pose_dims={relative_pose_dims}"
            )

        self.relative_pose_dims = int(relative_pose_dims)
        self.absolute_joint_dims = int(action_dim) - self.relative_pose_dims
        self.text_cond = text_cond

        # Keep branch MLPs text-free so optional text is added exactly once after fusion.
        self.relative_encoder = Action_encoder2(
            action_dim=self.relative_pose_dims,
            action_num=action_num,
            hidden_sizes=hidden_sizes,
            text_cond=False,
        )
        self.absolute_encoder = Action_encoder2(
            action_dim=self.absolute_joint_dims,
            action_num=action_num,
            hidden_sizes=hidden_sizes,
            text_cond=False,
        )

    @staticmethod
    def _normalize_joint_angles_radians(joint_angles: torch.Tensor) -> torch.Tensor:
        # Wrap to [-pi, pi] and scale to [-1, 1] for stable optimization.
        wrapped = torch.atan2(torch.sin(joint_angles), torch.cos(joint_angles))
        return wrapped / torch.pi

    def forward(self, action, texts=None, text_tokinizer=None, text_encoder=None, frame_level_cond=True):
        rel_pose = action[..., :self.relative_pose_dims]
        abs_joints = action[..., self.relative_pose_dims:]
        # TODO: remove this — dataloader already normalizes actions to [-1, 1] via p01/p99 stats,
        # so _normalize_joint_angles_radians compresses the joint branch by ~1/π unnecessarily,
        # creating a branch imbalance with the relative pose branch and reducing finger sensitivity.
        abs_joints_norm = self._normalize_joint_angles_radians(abs_joints)

        rel_embed = self.relative_encoder(
            rel_pose,
            texts=None,
            text_tokinizer=None,
            text_encoder=None,
            frame_level_cond=frame_level_cond,
        )
        abs_embed = self.absolute_encoder(
            abs_joints_norm,
            texts=None,
            text_tokinizer=None,
            text_encoder=None,
            frame_level_cond=frame_level_cond,
        )
        action_embed = rel_embed + abs_embed

        if texts is not None and self.text_cond:
            with torch.no_grad():
                inputs = text_tokinizer(
                    texts, padding='max_length', return_tensors="pt", truncation=True
                ).to(text_encoder.device)
                outputs = text_encoder(**inputs)
                hidden_text = outputs.text_embeds  # (B, 512)
                hidden_text = einops.repeat(hidden_text, 'b c -> b 1 (n c)', n=2)  # (B, 1, 1024)

            action_embed = action_embed + hidden_text

        return action_embed

class DinoV2VisualActionEncoder1(nn.Module):
    # it takes a sequnce of RGB frames which represent the segmentation of the hand. (B, T, C, H, W)
    def __init__(self, num_views=1, num_in_channels_per_view=3, embed_dim=1024, hub_dir=None, dinov2_size=224):
        super().__init__()
        
        self.num_views = num_views
        self.num_in_channels_per_view = num_in_channels_per_view

        # Set custom directory for torch.hub cache if provided
        if hub_dir is not None:
            torch.hub.set_dir(hub_dir)
        
        # Pretrained spatial encoder (frozen)
        self.spatial_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for param in self.spatial_backbone.parameters():
            param.requires_grad = False
        
        # DINOv2 requires input size divisible by patch size (14)
        # Common sizes: 224 (224/14=16), 280 (280/14=20), 392 (392/14=28)
        assert dinov2_size % 14 == 0, f"dinov2_size must be divisible by 14 (patch size), got {dinov2_size}"
        self.dinov2_size = dinov2_size
        
        # Trainable task-specific adapter with learnable downsampling
        # Input: (B, num_in_channels, H, W) -> Output: (B, 3, dinov2_size, dinov2_size)
        # Strategy: Use learnable convolutions to downsample to target size
        # The adapter learns features at multiple scales, then adjusts to target size
        
        # Calculate kernel size for final size adjustment (for common case of 256->224)
        # Formula: output_size = (input_size - kernel_size + 2*padding) / stride + 1
        # For 256->224 with stride=1, padding=0: kernel_size = 256 - 224 + 1 = 33
        # We'll use this for the expected input size, but handle variable sizes in forward
        expected_input_size = 256  # Common case
        if expected_input_size > dinov2_size:
            final_kernel_size = expected_input_size - dinov2_size + 1
        else:
            final_kernel_size = 1
        
        self.seg_adapter = nn.Sequential(
            # First conv: channel transformation
            nn.Conv2d(num_views*num_in_channels_per_view, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Learnable downsampling: use strided conv to reduce spatial size
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # H/2, W/2 (e.g., 256 -> 128)
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Learnable upsampling: transposed conv to get back to larger size
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Final channel reduction to num_in_channels
            nn.Conv2d(64, num_views*num_in_channels_per_view, 1),
        )
        
        # Learnable size adjustment layer (applied conditionally in forward)
        # For 256->224: kernel_size=33, stride=1, padding=0 gives (256-33+1)=224
        if final_kernel_size > 1:
            self.size_adjust_conv = nn.Conv2d(num_views*num_in_channels_per_view, num_views*num_in_channels_per_view, kernel_size=final_kernel_size, stride=1, padding=0)
        else:
            self.size_adjust_conv = None
        
        # Trainable projection
        self.proj = nn.Linear(768*num_views, embed_dim)
    
    def forward(self, seg_sequence):
        B, T, C, H, W = seg_sequence.shape
        
        # Extract spatial features with frozen DINOv2
        spatial_feats = []
        for t in range(T):
            x = self.seg_adapter(seg_sequence[:, t])  # (B, 3, H, W) - after adapter, size is back to ~HxW
            
            # Apply learnable size adjustment if available (for exact size matching)
            if self.size_adjust_conv is not None and x.shape[2] == H and x.shape[3] == W:
                # Use learnable conv to adjust size (e.g., 256 -> 224)
                x = self.size_adjust_conv(x)  # (B, 3, dinov2_size, dinov2_size)
            elif x.shape[2] != self.dinov2_size or x.shape[3] != self.dinov2_size:
                # Fallback to interpolation if size doesn't match (for variable input sizes)
                x = torch.nn.functional.interpolate(
                    x, 
                    size=(self.dinov2_size, self.dinov2_size), 
                    mode='bilinear', 
                    align_corners=False
                )  # (B, 3, dinov2_size, dinov2_size)
            
            features = []
            for view in range(self.num_views):
                feat = self.spatial_backbone(x[:, view*self.num_in_channels_per_view:(view+1)*self.num_in_channels_per_view])
                print("feat shape: ", feat.shape)
                features.append(feat)
            features_stack = torch.concat(features, dim=1)
            spatial_feats.append(features_stack)
        
        spatial_feats = torch.stack(spatial_feats, dim=1)  # [B, T, 768]
        
        # Project to output
        output = self.proj(spatial_feats)
        
        return output

class DinoV2VisualActionEncoder(nn.Module):
    # it takes a sequnce of RGB frames which represent the segmentation of the hand. (B, T, C, H, W)
    def __init__(self, num_views=1, num_in_channels_per_view=3, embed_dim=1024, hub_dir=None, dinov2_size=224):
        super().__init__()
        
        self.num_views = num_views
        self.num_in_channels_per_view = num_in_channels_per_view

        # Set custom directory for torch.hub cache if provided
        if hub_dir is not None:
            torch.hub.set_dir(hub_dir)
        
        # Pretrained spatial encoder (frozen)
        self.spatial_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for param in self.spatial_backbone.parameters():
            param.requires_grad = False
        
        # DINOv2 requires input size divisible by patch size (14)
        # Common sizes: 224 (224/14=16), 280 (280/14=20), 392 (392/14=28)
        assert dinov2_size % 14 == 0, f"dinov2_size must be divisible by 14 (patch size), got {dinov2_size}"
        self.dinov2_size = dinov2_size
        
        # Trainable task-specific adapter with learnable downsampling
        # Input: (B, num_in_channels, H, W) -> Output: (B, 3, dinov2_size, dinov2_size)
        # Strategy: Use learnable convolutions to downsample to target size
        # The adapter learns features at multiple scales, then adjusts to target size
        
        # Calculate kernel size for final size adjustment (for common case of 256->224)
        # Formula: output_size = (input_size - kernel_size + 2*padding) / stride + 1
        # For 256->224 with stride=1, padding=0: kernel_size = 256 - 224 + 1 = 33
        # We'll use this for the expected input size, but handle variable sizes in forward
        expected_input_size = 256  # Common case
        if expected_input_size > dinov2_size:
            final_kernel_size = expected_input_size - dinov2_size + 1
        else:
            final_kernel_size = 1
        
        self.seg_adapter = nn.Sequential(
            # First conv: channel transformation
            nn.Conv2d(num_views*num_in_channels_per_view, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Learnable downsampling: use strided conv to reduce spatial size
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # H/2, W/2 (e.g., 256 -> 128)
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Learnable upsampling: transposed conv to get back to larger size
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Final channel reduction to num_in_channels
            nn.Conv2d(64, num_views*num_in_channels_per_view, 1),
        )
        
        # Learnable size adjustment layer (applied conditionally in forward)
        # For 256->224: kernel_size=33, stride=1, padding=0 gives (256-33+1)=224
        if final_kernel_size > 1:
            self.size_adjust_conv = nn.Conv2d(num_views*num_in_channels_per_view, num_views*num_in_channels_per_view, kernel_size=final_kernel_size, stride=1, padding=0)
        else:
            self.size_adjust_conv = None
        
        # Trainable temporal reasoning module
        self.temporal_module = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768*num_views, nhead=8, dim_feedforward=2048),
            num_layers=2
        )
        
        # Trainable projection
        self.proj = nn.Linear(768*num_views, embed_dim)
    
    def forward(self, seg_sequence):
        B, T, C, H, W = seg_sequence.shape
        
        # Extract spatial features with frozen DINOv2
        spatial_feats = []
        for t in range(T):
            x = self.seg_adapter(seg_sequence[:, t])  # (B, 3, H, W) - after adapter, size is back to ~HxW
            
            # Apply learnable size adjustment if available (for exact size matching)
            if self.size_adjust_conv is not None and x.shape[2] == H and x.shape[3] == W:
                # Use learnable conv to adjust size (e.g., 256 -> 224)
                x = self.size_adjust_conv(x)  # (B, 3, dinov2_size, dinov2_size)
            elif x.shape[2] != self.dinov2_size or x.shape[3] != self.dinov2_size:
                # Fallback to interpolation if size doesn't match (for variable input sizes)
                x = torch.nn.functional.interpolate(
                    x, 
                    size=(self.dinov2_size, self.dinov2_size), 
                    mode='bilinear', 
                    align_corners=False
                )  # (B, 3, dinov2_size, dinov2_size)
            
            features = []
            for view in range(self.num_views):
                feat = self.spatial_backbone(x[:, view*self.num_in_channels_per_view:(view+1)*self.num_in_channels_per_view])
                print("feat shape: ", feat.shape)
                features.append(feat)
            features_stack = torch.concat(features, dim=1)
            spatial_feats.append(features_stack)
        
        spatial_feats = torch.stack(spatial_feats, dim=1)  # [B, T, 768]
        
        # Learn temporal patterns
        spatial_feats = spatial_feats.transpose(0, 1)  # [T, B, 768]
        temporal_feats = self.temporal_module(spatial_feats)
        temporal_feats = temporal_feats.transpose(0, 1)  # [B, T, 768]
        
        # Project to output
        output = self.proj(temporal_feats)
        
        return output


class DinoV2VisualActionEncoder2(nn.Module):
    # it takes a sequnce of RGB frames which represent the segmentation of the hand. (B, T, C, H, W)
    def __init__(self, num_views=1, num_in_channels_per_view=3, embed_dim=1024, hub_dir=None, dinov2_size=224):
        super().__init__()
        
        self.num_views = num_views
        self.num_in_channels_per_view = num_in_channels_per_view

        # Set custom directory for torch.hub cache if provided
        if hub_dir is not None:
            torch.hub.set_dir(hub_dir)
        
        # Pretrained spatial encoder (frozen)
        self.spatial_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for param in self.spatial_backbone.parameters():
            param.requires_grad = False
        
        # DINOv2 requires input size divisible by patch size (14)
        # Common sizes: 224 (224/14=16), 280 (280/14=20), 392 (392/14=28)
        assert dinov2_size % 14 == 0, f"dinov2_size must be divisible by 14 (patch size), got {dinov2_size}"
        self.dinov2_size = dinov2_size
        self.num_patches = (dinov2_size // 14) ** 2
        self.backbone_embed_dim = getattr(self.spatial_backbone, "embed_dim", 768)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.backbone_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Trainable task-specific adapter with learnable downsampling
        # Input: (B, num_in_channels, H, W) -> Output: (B, 3, dinov2_size, dinov2_size)
        # Strategy: Use learnable convolutions to downsample to target size
        # The adapter learns features at multiple scales, then adjusts to target size
        
        # Calculate kernel size for final size adjustment (for common case of 256->224)
        # Formula: output_size = (input_size - kernel_size + 2*padding) / stride + 1
        # For 256->224 with stride=1, padding=0: kernel_size = 256 - 224 + 1 = 33
        # We'll use this for the expected input size, but handle variable sizes in forward
        expected_input_size = 256  # Common case
        if expected_input_size > dinov2_size:
            final_kernel_size = expected_input_size - dinov2_size + 1
        else:
            final_kernel_size = 1
        
        self.seg_adapter = nn.Sequential(
            # First conv: channel transformation
            nn.Conv2d(num_views*num_in_channels_per_view, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Learnable downsampling: use strided conv to reduce spatial size
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # H/2, W/2 (e.g., 256 -> 128)
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Learnable upsampling: transposed conv to get back to larger size
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            # Final channel reduction to num_in_channels
            nn.Conv2d(64, num_views*num_in_channels_per_view, 1),
        )
        
        # Learnable size adjustment layer (applied conditionally in forward)
        # For 256->224: kernel_size=33, stride=1, padding=0 gives (256-33+1)=224
        if final_kernel_size > 1:
            self.size_adjust_conv = nn.Conv2d(num_views*num_in_channels_per_view, num_views*num_in_channels_per_view, kernel_size=final_kernel_size, stride=1, padding=0)
        else:
            self.size_adjust_conv = None
        
        # Trainable projection
        self.proj = nn.Linear(self.backbone_embed_dim * num_views, embed_dim)

    def _extract_patch_tokens(self, image_tensor):
        # DINOv2 forward() often returns a global token; use forward_features() for patch tokens.
        if hasattr(self.spatial_backbone, "forward_features"):
            features = self.spatial_backbone.forward_features(image_tensor)
            if isinstance(features, dict):
                if "x_norm_patchtokens" in features:
                    patch_tokens = features["x_norm_patchtokens"]
                elif "x_prenorm" in features and features["x_prenorm"].dim() == 3:
                    patch_tokens = features["x_prenorm"][:, 1:]
                elif "x" in features and features["x"].dim() == 3:
                    patch_tokens = features["x"][:, 1:] if features["x"].shape[1] == self.num_patches + 1 else features["x"]
                else:
                    raise ValueError("DINOv2 forward_features() returned an unexpected dict format.")
            else:
                patch_tokens = features
        else:
            patch_tokens = self.spatial_backbone(image_tensor)

        if patch_tokens.dim() == 2:
            raise ValueError(
                "Expected patch tokens [B, N, D], but got global embedding [B, D]. "
                "Use DINOv2 forward_features() to retrieve patch tokens."
            )
        if patch_tokens.dim() != 3:
            raise ValueError(f"Expected a 3D tensor [B, N, D], got shape {tuple(patch_tokens.shape)}")

        if patch_tokens.shape[1] == self.num_patches + 1:
            patch_tokens = patch_tokens[:, 1:]
        if patch_tokens.shape[1] != self.num_patches:
            raise ValueError(
                f"Patch token count mismatch: expected {self.num_patches}, got {patch_tokens.shape[1]}"
            )
        if patch_tokens.shape[2] != self.backbone_embed_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.backbone_embed_dim}, got {patch_tokens.shape[2]}"
            )

        return patch_tokens
    
    def forward(self, seg_sequence):
        B, T, C, H, W = seg_sequence.shape
        
        # Extract spatial features with frozen DINOv2
        spatial_feats = []
        for t in range(T):
            x = self.seg_adapter(seg_sequence[:, t])  # (B, 3, H, W) - after adapter, size is back to ~HxW
            
            # Apply learnable size adjustment if available (for exact size matching)
            if self.size_adjust_conv is not None and x.shape[2] == H and x.shape[3] == W:
                # Use learnable conv to adjust size (e.g., 256 -> 224)
                x = self.size_adjust_conv(x)  # (B, 3, dinov2_size, dinov2_size)
            elif x.shape[2] != self.dinov2_size or x.shape[3] != self.dinov2_size:
                # Fallback to interpolation if size doesn't match (for variable input sizes)
                x = torch.nn.functional.interpolate(
                    x, 
                    size=(self.dinov2_size, self.dinov2_size), 
                    mode='bilinear', 
                    align_corners=False
                )  # (B, 3, dinov2_size, dinov2_size)
            
            features = []
            for view in range(self.num_views):
                feat = self._extract_patch_tokens(
                    x[:, view*self.num_in_channels_per_view:(view+1)*self.num_in_channels_per_view]
                )
                feat = feat + self.pos_embed.to(dtype=feat.dtype, device=feat.device)
                features.append(feat)
            features_stack = torch.concat(features, dim=2)  # [B, num_patches, 768*num_views]
            spatial_feats.append(features_stack)
        
        spatial_feats = torch.stack(spatial_feats, dim=1)  # [B, T, num_patches, 768*num_views]
        
        # Project to output
        output = self.proj(spatial_feats)
        
        return output


class DinoV2VisualActionEncoder3(DinoV2VisualActionEncoder2):
    # Hybrid conditioning: one global token + gated patch tokens per frame.
    def __init__(self, num_views=1, num_in_channels_per_view=3, embed_dim=1024, hub_dir=None, dinov2_size=224):
        super().__init__(
            num_views=num_views,
            num_in_channels_per_view=num_in_channels_per_view,
            embed_dim=embed_dim,
            hub_dir=hub_dir,
            dinov2_size=dinov2_size,
        )
        self.token_norm = nn.LayerNorm(self.backbone_embed_dim * num_views)
        # Start with weak patch contribution to improve early-stage alignment stability.
        self.patch_gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, seg_sequence):
        B, T, C, H, W = seg_sequence.shape

        # Extract spatial features with frozen DINOv2
        spatial_feats = []
        for t in range(T):
            x = self.seg_adapter(seg_sequence[:, t])  # (B, 3, H, W) - after adapter, size is back to ~HxW

            # Apply learnable size adjustment if available (for exact size matching)
            if self.size_adjust_conv is not None and x.shape[2] == H and x.shape[3] == W:
                # Use learnable conv to adjust size (e.g., 256 -> 224)
                x = self.size_adjust_conv(x)  # (B, 3, dinov2_size, dinov2_size)
            elif x.shape[2] != self.dinov2_size or x.shape[3] != self.dinov2_size:
                # Fallback to interpolation if size doesn't match (for variable input sizes)
                x = torch.nn.functional.interpolate(
                    x,
                    size=(self.dinov2_size, self.dinov2_size),
                    mode='bilinear',
                    align_corners=False
                )  # (B, 3, dinov2_size, dinov2_size)

            features = []
            for view in range(self.num_views):
                feat = self._extract_patch_tokens(
                    x[:, view*self.num_in_channels_per_view:(view+1)*self.num_in_channels_per_view]
                )
                feat = feat + self.pos_embed.to(dtype=feat.dtype, device=feat.device)
                features.append(feat)
            features_stack = torch.concat(features, dim=2)  # [B, num_patches, 768*num_views]
            features_stack = self.token_norm(features_stack)
            global_token = features_stack.mean(dim=1, keepdim=True)  # [B, 1, 768*num_views]
            patch_gate = torch.sigmoid(self.patch_gate_logit).to(features_stack.dtype)
            hybrid_tokens = torch.cat(
                [global_token, patch_gate * features_stack], dim=1
            )  # [B, 1+num_patches, 768*num_views]
            spatial_feats.append(hybrid_tokens)

        spatial_feats = torch.stack(spatial_feats, dim=1)  # [B, T, 1+num_patches, 768*num_views]

        # Project to output
        output = self.proj(spatial_feats)

        return output


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base_layer(x) + self.lora_up(self.lora_down(x)) * self.scaling


def _replace_submodule(root: nn.Module, module_name: str, replacement_module: nn.Module) -> None:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = replacement_module
    else:
        setattr(parent, last, replacement_module)


def _apply_lora_to_video_unet(unet: nn.Module, rank: int = 8, alpha: float = 8.0) -> int:
    replaced = 0
    target_tokens = ("to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out")
    target_prefixes = ("down_blocks", "mid_block", "up_blocks")
    for module_name, module in list(unet.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not module_name.startswith(target_prefixes):
            continue
        if not any(token in module_name for token in target_tokens):
            continue
        _replace_submodule(unet, module_name, LoRALinear(module, rank=rank, alpha=alpha))
        replaced += 1
    return replaced


class SegmentationToControlLatentAdapter(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, segmentation_videos: torch.Tensor, target_frames: int, target_size: tuple):
        bsz, num_frames = segmentation_videos.shape[:2]
        if num_frames < target_frames:
            pad_frames = target_frames - num_frames
            pad = segmentation_videos[:, -1:].repeat(1, pad_frames, 1, 1, 1)
            segmentation_videos = torch.cat([segmentation_videos, pad], dim=1)
        elif num_frames > target_frames:
            segmentation_videos = segmentation_videos[:, :target_frames]

        x = segmentation_videos.flatten(0, 1)
        x = self.encoder(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=target_size)
        x = x.view(bsz, target_frames, x.shape[1], target_size[0], target_size[1])
        return x


class CtrlWorldControlNet(nn.Module):
    def __init__(self, unet: UNetSpatioTemporalConditionModel):
        super().__init__()
        self.conv_in = copy.deepcopy(unet.conv_in)
        self.time_proj = copy.deepcopy(unet.time_proj)
        self.time_embedding = copy.deepcopy(unet.time_embedding)
        self.add_time_proj = copy.deepcopy(unet.add_time_proj)
        self.add_embedding = copy.deepcopy(unet.add_embedding)
        self.down_blocks = copy.deepcopy(unet.down_blocks)
        self.mid_block = copy.deepcopy(unet.mid_block)
        self.num_upsamplers = unet.num_upsamplers

        block_out_channels = list(unet.config.block_out_channels)
        layers_per_block = unet.config.layers_per_block
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(block_out_channels)
        else:
            layers_per_block = list(layers_per_block)

        self.controlnet_down_blocks = nn.ModuleList(
            [_zero_module(nn.Conv2d(block_out_channels[0], block_out_channels[0], kernel_size=1))]
        )
        for i, out_channels in enumerate(block_out_channels):
            is_final_block = i == len(block_out_channels) - 1
            num_residuals_for_block = layers_per_block[i] + (0 if is_final_block else 1)
            for _ in range(num_residuals_for_block):
                self.controlnet_down_blocks.append(
                    _zero_module(nn.Conv2d(out_channels, out_channels, kernel_size=1))
                )
        self.controlnet_mid_block = _zero_module(
            nn.Conv2d(block_out_channels[-1], block_out_channels[-1], kernel_size=1)
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        frame_level_cond: bool = False,
    ):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.int64, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1)).to(emb.dtype)
        emb = emb + self.add_embedding(time_embeds)

        sample = sample.flatten(0, 1)
        emb = emb.repeat_interleave(num_frames, dim=0, output_size=emb.shape[0] * num_frames)

        if not frame_level_cond:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
        else:
            encoder_hidden_states = encoder_hidden_states.reshape(
                batch_size * num_frames, -1, encoder_hidden_states.shape[-1]
            )

        sample = self.conv_in(sample)
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )
            down_block_res_samples += res_samples

        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        controlnet_down_block_res_samples = tuple(
            control_block(res_sample)
            for control_block, res_sample in zip(self.controlnet_down_blocks, down_block_res_samples)
        )
        controlnet_mid_block_res_sample = self.controlnet_mid_block(sample)
        return controlnet_down_block_res_samples, controlnet_mid_block_res_sample


class CtrlWorld(nn.Module):
    def __init__(self, args):
        super(CtrlWorld, self).__init__()

        self.args = args

        # load from pretrained stable video diffusion
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(args.svd_model_path)
        # repalce the unet to support frame_level pose condition
        print("replace the unet to support action condition and frame_level pose!")
        unet = UNetSpatioTemporalConditionModel()
        unet.load_state_dict(self.pipeline.unet.state_dict(), strict=False)

        if args.concatenate_latent:
            num_channels_concatenate = 0
            if args.concatenate_latent == "interpolated_downsampled_segmentation":
                num_channels_concatenate = 3
            elif args.concatenate_latent == "latent_segmentation_videos":
                num_channels_concatenate = 4

            old_conv_in = unet.conv_in
            new_conv_in = nn.Conv2d(
                in_channels=old_conv_in.in_channels + num_channels_concatenate,
                out_channels=old_conv_in.out_channels,
                kernel_size=old_conv_in.kernel_size,
                padding=old_conv_in.padding,
            )

            with torch.no_grad():
                new_conv_in.weight[:, :old_conv_in.in_channels, :, :].copy_(old_conv_in.weight)
                if old_conv_in.bias is not None:
                    new_conv_in.bias.copy_(old_conv_in.bias)

                new_conv_in.weight[:, old_conv_in.in_channels:, :, :].zero_()

            unet.conv_in = new_conv_in
            unet.config.in_channels = old_conv_in.in_channels + num_channels_concatenate
            print("concatenate latent with input is activated.")

        self.pipeline.unet = unet
        
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.image_encoder = self.pipeline.image_encoder
        self.scheduler = self.pipeline.scheduler
        self.use_controlnet_conditioning = getattr(args, "use_controlnet_conditioning", False)
        self.controlnet_conditioning_scale = getattr(args, "controlnet_conditioning_scale", 1.0)
        self.cond_dropout_mask_only_prob = float(getattr(args, "cond_dropout_mask_only_prob", 0.10))
        self.cond_dropout_action_only_prob = float(getattr(args, "cond_dropout_action_only_prob", 0.10))
        self.cond_dropout_both_prob = float(getattr(args, "cond_dropout_both_prob", 0.05))
        self.controlnet = None
        self.segmentation_to_control = None
        self.use_vae_roundtrip_for_controlnet = getattr(args, "use_vae_roundtrip_for_controlnet", False)
        self._unet_lora_enabled = False

        # freeze vae, image_encoder, enable unet gradient ckpt
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(not self.use_controlnet_conditioning)

        if self.use_controlnet_conditioning:
            self.controlnet = CtrlWorldControlNet(self.unet)
            self.controlnet.requires_grad_(True)
            self.segmentation_to_control = SegmentationToControlLatentAdapter(
                out_channels=self.unet.config.in_channels
            )
            self.segmentation_to_control.requires_grad_(True)
            print("ControlNet-style conditioning branch is enabled.")

        if getattr(args, "use_unet_lora", False):
            self.enable_unet_lora(
                rank=getattr(args, "unet_lora_rank", 8),
                alpha=getattr(args, "unet_lora_alpha", 8.0),
            )
        
        # Optionally freeze temporal attention layers
        if args.freeze_temporal_transformer_blocks:
            for name, param in self.unet.named_parameters():
                if 'temporal' in name.lower():
                    param.requires_grad = False
                    print(f"Freezing temporal layer: {name}")
        
        self.unet.enable_gradient_checkpointing()

        # SVD is a img2video model, load a clip text encoder
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(args.clip_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.clip_model_path,use_fast=False)
        self.text_encoder.requires_grad_(False)

        # initialize an action projector
        if args.action_encoder == "dino_visual":
            self.action_encoder = DinoV2VisualActionEncoder(num_views=args.num_views, num_in_channels_per_view=3, embed_dim=1024)
        elif args.action_encoder == "dino_visual2":
            self.action_encoder = DinoV2VisualActionEncoder2(num_views=args.num_views, num_in_channels_per_view=3, embed_dim=1024)
        elif args.action_encoder == "dino_visual3":
            self.action_encoder = DinoV2VisualActionEncoder3(num_views=args.num_views, num_in_channels_per_view=3, embed_dim=1024)
        elif args.action_encoder == "action_encoder":
            self.action_encoder = Action_encoder2(action_dim=args.action_dim, action_num=int(args.num_history+args.num_frames), hidden_sizes=args.action_encoder_hidden_dims, text_cond=args.text_cond)
        elif args.action_encoder == "action_encoder_dual":
            self.action_encoder = DualActionEncoder(
                action_dim=args.action_dim,
                action_num=int(args.num_history + args.num_frames),
                hidden_sizes=args.action_encoder_hidden_dims,
                relative_pose_dims=getattr(args, "relative_pose_dims", 6),
                text_cond=args.text_cond,
            )
        else:
            self.action_encoder = None

        self._maybe_load_ctrl_world_pretrained()

    def enable_unet_lora(self, rank: int = 8, alpha: float = 8.0) -> int:
        if self._unet_lora_enabled:
            raise RuntimeError("UNet LoRA is already enabled for this model.")

        self.unet.requires_grad_(False)
        replaced_count = _apply_lora_to_video_unet(
            self.unet,
            rank=rank,
            alpha=alpha,
        )
        self._unet_lora_enabled = True
        print(f"UNet LoRA enabled on {replaced_count} important linear blocks.")
        return replaced_count

    @staticmethod
    def _unwrap_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported checkpoint payload type: {type(payload)}")

        def _looks_like_tensor_state_dict(candidate: Dict[str, Any]) -> bool:
            if not candidate:
                return False
            return all(isinstance(v, torch.Tensor) for v in candidate.values())

        # Some checkpoints are nested multiple levels deep, e.g.
        # {"state_dict": {"model": {...}, "optimizer": {...}}}.
        # Repeatedly unwrap known container keys until we reach actual tensors.
        unwrap_keys = ("state_dict", "model_state_dict", "model", "module")
        for _ in range(8):
            if _looks_like_tensor_state_dict(payload):
                break

            nested_payload = None
            for key in unwrap_keys:
                nested = payload.get(key)
                if isinstance(nested, dict):
                    nested_payload = nested
                    break

            if nested_payload is None:
                break
            payload = nested_payload

        if not isinstance(payload, dict):
            raise ValueError("Checkpoint did not contain a valid state_dict.")
        if not _looks_like_tensor_state_dict(payload):
            raise ValueError(
                "Checkpoint did not resolve to a tensor state_dict. "
                f"Top-level keys: {list(payload.keys())[:8]}"
            )

        has_module_prefix = any(k.startswith("module.") for k in payload.keys())
        if has_module_prefix:
            payload = {k.replace("module.", "", 1): v for k, v in payload.items()}
        return payload

    def _load_state_dict_from_file(self, ckpt_file: str) -> Dict[str, torch.Tensor]:
        if ckpt_file.endswith(".safetensors"):
            from safetensors.torch import load_file as load_safetensors_file
            payload = load_safetensors_file(ckpt_file, device="cpu")
        else:
            payload = torch.load(ckpt_file, map_location="cpu")
        return self._unwrap_state_dict(payload)

    def _resolve_pretrained_ckpt_file(self, source: str) -> str:
        if os.path.isfile(source):
            return source

        search_dir = source
        if not os.path.isdir(search_dir):
            from huggingface_hub import snapshot_download
            print(f"Downloading Ctrl-World pretrained weights from HuggingFace repo: {source}")
            search_dir = snapshot_download(repo_id=source)

        candidates = []
        for root, _, files in os.walk(search_dir):
            for fname in files:
                if fname.endswith((".pt", ".pth", ".bin", ".safetensors")):
                    candidates.append(os.path.join(root, fname))

        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint file found under '{search_dir}'. "
                "Expected one of: .pt, .pth, .bin, .safetensors"
            )

        preferred = sorted(
            candidates,
            key=lambda p: (
                0 if "checkpoint" in os.path.basename(p) else 1,
                0 if p.endswith(".pt") else 1,
                len(p),
            ),
        )
        return preferred[0]

    def _select_pretrained_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Select which checkpoint keys to load.
        Default behavior loads only video-predictor modules to avoid pulling
        incompatible action/text encoder weights from older checkpoints.
        """
        load_mode = str(getattr(self.args, "ctrl_world_pretrained_load_mode", "video_predictor")).lower()
        if load_mode == "all":
            return state_dict

        target_prefixes = ("unet.", "controlnet.", "segmentation_to_control.", "action_encoder.")
        selected = {k: v for k, v in state_dict.items() if k.startswith(target_prefixes)}
        if not selected:
            print(
                "Warning: no selected pretrained keys found in checkpoint; "
                "falling back to loading all available keys."
            )
            return state_dict
        return selected

    def _filter_shape_compatible_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> tuple[
        Dict[str, torch.Tensor],
        list[tuple[str, tuple[int, ...], tuple[int, ...]]],
        list[str],
        list[str],
    ]:
        """
        Keep only keys that exist in the current model and match tensor shape.
        Returns:
          - filtered state_dict
          - list of shape mismatches as (key, ckpt_shape, model_shape)
          - list of keys missing in current model
          - list of adapted keys
        """
        current = self.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        missing_in_model: list[str] = []
        adapted_keys: list[str] = []

        # Backward-compatibility for dual action encoder:
        # map legacy single-encoder keys onto the relative-pose branch.
        use_dual_action_encoder = str(getattr(self.args, "action_encoder", "")) == "action_encoder_dual"
        if use_dual_action_encoder:
            remapped_state_dict: Dict[str, torch.Tensor] = {}
            for key, value in state_dict.items():
                if key.startswith("action_encoder.action_encode."):
                    remapped_key = key.replace(
                        "action_encoder.action_encode.",
                        "action_encoder.relative_encoder.action_encode.",
                        1,
                    )
                    remapped_state_dict[remapped_key] = value
                    adapted_keys.append(f"{key}->{remapped_key}")
                else:
                    remapped_state_dict[key] = value
            state_dict = remapped_state_dict

        for key, value in state_dict.items():
            if key not in current:
                missing_in_model.append(key)
                continue
            if tuple(value.shape) != tuple(current[key].shape):
                # Special handling for action encoder input expansion:
                # copy first N pretrained input columns and keep the rest
                # from the model's default initialization.
                if key in (
                    "action_encoder.action_encode.0.weight",
                    "action_encoder.relative_encoder.action_encode.0.weight",
                ):
                    curr = current[key]
                    if value.dim() == 2 and curr.dim() == 2 and value.shape[0] == curr.shape[0]:
                        default_copy_dims = int(getattr(self.args, "relative_pose_dims", 9))
                        copy_dims = int(getattr(self.args, "ctrl_world_pretrained_action_copy_dims", default_copy_dims))
                        copy_dims = max(0, min(copy_dims, default_copy_dims, value.shape[1], curr.shape[1]))
                        if copy_dims > 0:
                            adapted = curr.clone()
                            adapted[:, :copy_dims] = value[:, :copy_dims]
                            filtered[key] = adapted
                            adapted_keys.append(f"{key}[:,:{copy_dims}]")
                            continue
                mismatched.append((key, tuple(value.shape), tuple(current[key].shape)))
                continue
            filtered[key] = value

        return filtered, mismatched, missing_in_model, adapted_keys

    def _maybe_init_controlnet_from_loaded_unet(self):
        """
        Optional warm-start for ControlNet shared trunk.
        Copies UNet weights into ControlNet's mirrored modules after pretrained
        loading, so ControlNet starts from the same representation.
        """
        if not getattr(self.args, "ctrl_world_pretrained_init_controlnet_from_unet", False):
            return
        if self.controlnet is None:
            print("ControlNet warm-start skipped: controlnet branch is disabled.")
            return

        pairs = [
            ("conv_in", "conv_in"),
            ("time_proj", "time_proj"),
            ("time_embedding", "time_embedding"),
            ("add_time_proj", "add_time_proj"),
            ("add_embedding", "add_embedding"),
            ("down_blocks", "down_blocks"),
            ("mid_block", "mid_block"),
        ]

        copied = 0
        for src_name, dst_name in pairs:
            src = getattr(self.unet, src_name)
            dst = getattr(self.controlnet, dst_name)
            dst.load_state_dict(src.state_dict(), strict=True)
            copied += 1

        print(
            "Initialized ControlNet shared modules from loaded UNet "
            f"(copied {copied} module groups)."
        )

    def _maybe_load_ctrl_world_pretrained(self):
        if not getattr(self.args, "load_ctrl_world_pretrained", False):
            return

        source = getattr(self.args, "ctrl_world_pretrained_path", None)
        if not source:
            raise ValueError(
                "load_ctrl_world_pretrained=True requires ctrl_world_pretrained_path "
                "(local checkpoint path, local folder, or HF repo id)."
            )

        ckpt_file = self._resolve_pretrained_ckpt_file(source)
        print(f"Loading Ctrl-World pretrained weights from: {ckpt_file}")
        raw_state_dict = self._load_state_dict_from_file(ckpt_file)
        selected_state_dict = self._select_pretrained_state_dict(raw_state_dict)
        state_dict, mismatched, missing_in_model, adapted_keys = self._filter_shape_compatible_state_dict(selected_state_dict)

        if mismatched:
            preview = ", ".join(
                f"{k} (ckpt={ckpt_shape}, model={model_shape})"
                for k, ckpt_shape, model_shape in mismatched[:3]
            )
            print(
                f"Skipping {len(mismatched)} mismatched pretrained tensors. "
                f"Examples: {preview}"
            )
        if missing_in_model:
            print(f"Skipping {len(missing_in_model)} checkpoint tensors not present in current model.")
        if adapted_keys:
            print(
                f"Adapted {len(adapted_keys)} pretrained tensor(s) for shape compatibility: "
                + ", ".join(adapted_keys)
            )

        strict_load = bool(getattr(self.args, "ctrl_world_pretrained_strict", False))
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if strict_load:
            missing_relevant = [k for k in missing if k.startswith(("unet.", "controlnet.", "segmentation_to_control."))]
            require_all_keys = bool(
                getattr(self.args, "ctrl_world_pretrained_strict_require_all_keys", False)
            )
            if mismatched or unexpected or (require_all_keys and missing_relevant):
                raise RuntimeError(
                    "Strict Ctrl-World pretrained load failed for video-predictor modules: "
                    f"mismatched={len(mismatched)}, missing={len(missing_relevant)}, "
                    f"unexpected={len(unexpected)}, require_all_keys={require_all_keys}"
                )
            if missing_relevant and not require_all_keys:
                print(
                    f"Strict mode notice: {len(missing_relevant)} video-predictor keys were not "
                    "present in checkpoint and were left at initialization defaults."
                )

        # Optional: after UNet is loaded from Ctrl-World weights, mirror those
        # shared weights into ControlNet encoder/mid trunk.
        self._maybe_init_controlnet_from_loaded_unet()

        print(
            f"Ctrl-World pretrained load finished (strict={strict_load}). "
            f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
        )


    @torch.no_grad()
    def _vae_decode_latent_segs(self, latent_segs: torch.Tensor) -> torch.Tensor:
        """Decode pre-encoded segmentation latents back to pixel space for the ControlNet CNN.

        Processes one frame at a time to avoid temporal-attention cross-contamination
        in the AutoencoderKLTemporalDecoder when frames come from different views/timesteps.

        Args:
            latent_segs: (B, T, 4, H_lat, W_lat), scaled latents from disk
        Returns:
            (B, T, 3, H, W) pixel frames clipped to [-1, 1]
        """
        B, T, C, H_lat, W_lat = latent_segs.shape
        flat = latent_segs.flatten(0, 1)  # (B*T, 4, H_lat, W_lat)
        flat = flat / self.vae.config.scaling_factor

        decoded_frames = []
        for i in range(flat.shape[0]):
            frame = flat[i : i + 1]  # (1, 4, H_lat, W_lat)
            decoded_frames.append(self.vae.decode(frame, num_frames=1).sample)

        decoded = torch.cat(decoded_frames, dim=0)  # (B*T, 3, H, W)
        decoded = decoded.clamp(-1.0, 1.0)
        return decoded.view(B, T, decoded.shape[1], decoded.shape[2], decoded.shape[3])

    def forward(self, batch):
        latents = batch[self.args.predicted_datatype] # (B, 16, 4, 32, 32)
        latent_hand_masks = batch['hand_mask'] if self.args.use_hand_mask else None
        texts = batch['text']
        dtype = self.unet.dtype
        device = self.unet.device
        P_mean=0.7
        P_std=1.6
        noise_aug_strength = 0.0

        num_history  = self.args.num_history
        latents = latents.to(device) #[B, num_history + num_frames]

        # current img as condition image to stack at channel wise, add random noise to current image, noise strength 0.0~0.2
        current_img = latents[:,num_history:(num_history+1)] # (B, 1, 4, 32, 32)
        bsz,num_frames = latents.shape[:2]
        current_img = current_img[:,0] # (B, 4, 32, 32)
        sigma = torch.rand([bsz, 1, 1, 1], device=device) * 0.2
        c_in = 1 / (sigma**2 + 1) ** 0.5
        current_img = c_in*(current_img + torch.randn_like(current_img) * sigma)
        condition_latent = einops.repeat(current_img, 'b c h w -> b f c h w', f=num_frames) # (8, 16,12, 32,32)
        if self.args.his_cond_zero:
            condition_latent[:, :num_history] = 0.0 # (B, num_history+num_frames, 4, 32, 32)

        if self.args.concatenate_latent:
            concatenate_latent = batch[self.args.concatenate_latent] # (B, num_history+num_frames, 4, 32, 32)
            condition_latent = torch.cat([condition_latent, concatenate_latent], dim=2) # (B, num_history+num_frames, 8, 32, 32)

        # action condition
        action = batch['action'] # (B, f, 7)
        action = action.to(device)
        if "dino_visual" in self.args.action_encoder:
            visual_actions = batch['segmentation_videos'] # (B, T, 3, 256, 256)
            visual_actions = visual_actions.to(device)
            action_hidden = self.action_encoder(visual_actions) # (B, T, 1024)
        elif self.args.action_encoder in ("action_encoder", "action_encoder_dual"):
            action_hidden = self.action_encoder(action, texts, self.tokenizer, self.text_encoder, frame_level_cond=self.args.frame_level_cond) # (B, f, 1024)
        else:
            action_hidden = torch.zeros((bsz, num_frames, 1024), device=device)

        # Per-sample conditioning dropout hierarchy (when both action + ControlNet are active):
        # 5% drop both, 10% drop mask-only, 10% drop action-only.
        drop_action_cond = torch.zeros(bsz, dtype=torch.bool, device=device)
        drop_controlnet_cond = torch.zeros(bsz, dtype=torch.bool, device=device)
        use_joint_cond_dropout = self.training and self.use_controlnet_conditioning and (self.action_encoder is not None)
        if use_joint_cond_dropout:
            p_both = self.cond_dropout_both_prob
            p_mask_only = self.cond_dropout_mask_only_prob
            p_action_only = self.cond_dropout_action_only_prob
            r = torch.rand(bsz, device=device)
            drop_both = r < p_both
            drop_mask_only = (r >= p_both) & (r < (p_both + p_mask_only))
            drop_action_only = (r >= (p_both + p_mask_only)) & (r < (p_both + p_mask_only + p_action_only))
            drop_controlnet_cond = drop_both | drop_mask_only
            drop_action_cond = drop_both | drop_action_only

        if drop_action_cond.any():
            action_keep_mask = (~drop_action_cond).view(bsz, *([1] * (action_hidden.dim() - 1))).to(action_hidden.dtype)
            action_hidden = action_hidden * action_keep_mask

        # diffusion forward process on future latent
        rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        c_skip = 1 / (sigma**2 + 1)
        c_out =  -sigma / (sigma**2 + 1) ** 0.5
        c_in = 1 / (sigma**2 + 1) ** 0.5
        c_noise = (sigma.log() / 4).reshape([bsz])
        loss_weight = (sigma ** 2 + 1) / sigma ** 2
        noisy_latents = (latents + torch.randn_like(latents) * sigma)

        # add 0~0.3 noise to history, history as condition
        sigma_h = torch.randn([bsz, num_history, 1, 1, 1], device=device) * 0.3
        history = latents[:,:num_history] # (B, num_history, 4, 32, 32)
        noisy_history = 1/(sigma_h**2+1)**0.5 *(history + sigma_h * torch.randn_like(history)) # (B, num_history, 4, 32, 32)
        input_latents = torch.cat([noisy_history, c_in*noisy_latents[:,num_history:]], dim=1) # (B, num_history+num_frames, 4, 32, 32)

        # svd stack a img at channel wise
        input_latents = torch.cat([input_latents, condition_latent/self.vae.config.scaling_factor], dim=2)
        motion_bucket_id = self.args.motion_bucket_id
        fps = self.args.fps
        added_time_ids = self.pipeline._get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, action_hidden.dtype, bsz, 1, False)
        added_time_ids = added_time_ids.to(device)

        # forward unet
        loss = 0
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        if self.use_controlnet_conditioning and self.controlnet is not None:
            use_latent_control = bool(getattr(self.args, "use_latent_segmentation_for_controlnet", False))
            use_vae_roundtrip = bool(getattr(self, "use_vae_roundtrip_for_controlnet", False))
            use_vis_seg_actions = bool(getattr(self.args, "use_vis_seg_actions_for_controlnet", False))
            enabled_control_sources = int(use_latent_control) + int(use_vae_roundtrip) + int(use_vis_seg_actions)
            if enabled_control_sources > 1:
                raise ValueError(
                    "Ambiguous ControlNet source flags. Enable at most one of "
                    "use_latent_segmentation_for_controlnet, "
                    "use_vae_roundtrip_for_controlnet, "
                    "use_vis_seg_actions_for_controlnet."
                )

            if use_latent_control:
                controlnet_cond = batch["latent_segmentation_videos"].to(device=device, dtype=input_latents.dtype)
                controlnet_cond = latent_seg_heightstack_to_channelstack(controlnet_cond, num_views=self.args.num_views)
                if controlnet_cond.shape[2] != self.unet.config.in_channels:
                    raise ValueError(
                        "Latent ControlNet conditioning channels mismatch: "
                        f"got {controlnet_cond.shape[2]}, expected {self.unet.config.in_channels}."
                    )
            else:
                if use_vae_roundtrip:
                    if "latent_segmentation_videos" not in batch:
                        raise KeyError("use_vae_roundtrip_for_controlnet requires batch['latent_segmentation_videos'].")
                    latent_segs = batch["latent_segmentation_videos"].to(device=device, dtype=input_latents.dtype)
                    segmentation_videos = self._vae_decode_latent_segs(latent_segs)
                elif use_vis_seg_actions:
                    if "vis_seg_actions" not in batch:
                        raise KeyError("use_vis_seg_actions_for_controlnet requires batch['vis_seg_actions'].")
                    segmentation_videos = batch["vis_seg_actions"].to(device=device, dtype=input_latents.dtype)
                else:
                    if "segmentation_videos" not in batch:
                        raise KeyError("ControlNet conditioning requires batch['segmentation_videos'].")
                    segmentation_videos = batch["segmentation_videos"].to(device=device, dtype=input_latents.dtype)
                controlnet_cond = self.segmentation_to_control(
                    segmentation_videos=segmentation_videos,
                    target_frames=input_latents.shape[1],
                    target_size=input_latents.shape[-2:],
                )
            down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                controlnet_cond,
                c_noise,
                encoder_hidden_states=action_hidden,
                added_time_ids=added_time_ids,
                frame_level_cond=self.args.frame_level_cond,
            )
            if drop_controlnet_cond.any():
                # Residual tensors are flattened over frames (B*T,...), so repeat sample decisions over T.
                drop_controlnet_bt = drop_controlnet_cond.repeat_interleave(input_latents.shape[1])
                keep_controlnet_bt = (~drop_controlnet_bt).view(-1, 1, 1, 1).to(input_latents.dtype)
                down_block_additional_residuals = tuple(
                    residual * keep_controlnet_bt for residual in down_block_additional_residuals
                )
                mid_block_additional_residual = mid_block_additional_residual * keep_controlnet_bt

        model_pred = self.unet(
            input_latents,
            c_noise,
            encoder_hidden_states=action_hidden,
            added_time_ids=added_time_ids,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            frame_level_cond=self.args.frame_level_cond,
        ).sample
        predict_x0 = c_out * model_pred + c_skip * noisy_latents 

        # only calculate loss on future frames
        pred_difference = predict_x0[:,num_history:] - latents[:,num_history:]
        if self.args.use_hand_mask:
            print("pred_difference shape: ", pred_difference.shape)
            print("latent_hand_masks: ", latent_hand_masks[:, num_history:].shape)

            pred_difference = pred_difference * latent_hand_masks[:, num_history:] * self.args.hand_weight + pred_difference * (1 - latent_hand_masks[:, num_history:])
        loss += (pred_difference**2 * loss_weight).mean()

        return loss, torch.tensor(0.0, device=device,dtype=dtype)
