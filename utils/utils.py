import numpy as np
import os
from typing import Tuple, List
import torch
import requests

from typing import Dict
import itertools


SEMANTIC_SEGMENTATION_MAPPING: Dict[str, Tuple[int, int, int]] = {
    "object": (0, 0, 255),
    "hand": (0, 255, 0),
    # "robot": (0, 255, 0),
    # "ground": (255, 0, 0),
    # "table": (255, 255, 0),
    "background": (0, 0, 0),
}

def snap_to_palette(predicted_mask: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """
    predicted_mask: Tensor [B, 3, H, W] - Output from Model 1 (dirty/blurry)
    palette: Tensor [N, 3] - Your 'True' RGB colors for each class
    """
    B, C, H, W = predicted_mask.shape

    # Reshape mask and palette for distance calculation
    # [B, H*W, 2]
    flat_mask = predicted_mask.reshape(-1, 3)
    
    # Calculate Euclidean distance from every pixel to every palette color
    # dists shape: [TotalPixels, N_Classes]
    dists = torch.cdist(flat_mask, palette)
    
    # Find the index of the closest color
    best_color_idx = torch.argmin(dists, dim=1)
    
    # Map back to the palette colors
    for i in range(best_color_idx.shape[0]):
        flat_mask[i] = palette[best_color_idx[i]]
    
    # Reshape back to original image dimensions
    snapped_mask = flat_mask.reshape(B, C, H, W).to(torch.uint8)
    
    return snapped_mask

def find_closest_color_mask(seg_rgb: np.ndarray, entities_to_include_in_mask: List[str], color_threshold: float = 15.0) -> np.ndarray:
    """
    Find pixels in segmentation RGB that match any of the target entity colors within a threshold.
    
    Uses Euclidean distance in RGB space to handle video compression artifacts.
    Returns a combined mask where True indicates pixels matching ANY of the specified entities.
    
    Args:
        seg_rgb: Segmentation RGB array (T, H, W, 4) or (H, W, 4) uint8
        entities_to_include_in_mask: List of entity names from SEMANTIC_SEGMENTATION_MAPPING
        color_threshold: Maximum Euclidean distance in RGB space to consider a match (default: 15.0)
    
    Returns:
        Boolean mask (T, H, W) or (H, W) indicating pixels matching any of the target entities.
        True where at least one entity matches.
    
    Note:
        To extract RGB values while preserving structure, use:
        - seg_rgb * mask[..., None]  (zeros out non-matching pixels)
        - np.where(mask[..., None], seg_rgb, 0)  (same effect)
        NOT seg_rgb[mask] which returns a flattened 1D array (N, 3)
    """
    seg = seg_rgb.astype(np.float32)  # Convert to float for distance calculation
    targets = [np.array(SEMANTIC_SEGMENTATION_MAPPING[entity], dtype=np.float32) if entity in SEMANTIC_SEGMENTATION_MAPPING else None for entity in entities_to_include_in_mask]
    targets = [target for target in targets if target is not None]
    
    if seg.shape[-1] == 4:
        seg = seg[:, :, :, :3]

    # Calculate Euclidean distance in RGB space
    # seg: (T, H, W, 3) or (H, W, 3), target: (3,)
    # Broadcasting: subtract target from each pixel
    masks = []
    if "hand" in entities_to_include_in_mask:
        # get mask including everything except the hand
        for key, value in SEMANTIC_SEGMENTATION_MAPPING.items():
            if key != "hand":
                target = np.array(value, dtype=np.float32)
                diff = seg - target  # (T, H, W, 3) or (H, W, 3)
                distances = np.linalg.norm(diff, axis=-1)  # (T, H, W) or (H, W)
                mask = distances > color_threshold
                masks.append(mask)
        # contains only hand pixels
        hand_mask = np.logical_and.reduce(masks)
        masks = []
        masks.append(hand_mask)
    
    for target in targets:
        diff = seg - target  # (T, H, W, 3) or (H, W, 3)
        distances = np.linalg.norm(diff, axis=-1)  # (T, H, W) or (H, W)
 
        # Threshold: pixels with distance <= threshold are considered matches
        mask = distances <= color_threshold
        masks.append(mask)
    
    # Combine masks: True if ANY entity matches (logical OR)
    # More efficient than np.sum(masks, axis=0) > 0
    if len(masks) == 1:
        return masks[0].astype(bool)
    return np.logical_or.reduce(masks).astype(bool)


def ensure_uint8(frames: np.ndarray) -> np.ndarray:
    # Expect (T, H, W, 3) float in [-1,1] or [0,1] or uint8; convert to uint8 [0..255]
    if frames.dtype == np.uint8:
        return frames
    f = frames.astype(np.float32)
    if f.max() <= 1.0 and f.min() >= -1.0:
        # map [-1,1] or [0,1] to [0,255]
        f = np.clip((f + 1.0) / 2.0, 0.0, 1.0) if f.min() < 0 else np.clip(f, 0.0, 1.0)
        f = (f * 255.0).round()
    f = np.clip(f, 0.0, 255.0).astype(np.uint8)
    return f


def resize_mask_to(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # frames: (T, H, W, 3), mask: (T, 1, Hm, Wm) or (T, Hm, Wm)
    # returns boolean mask (T, H, W)
    import torch.nn.functional as F
    t, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    m = mask
    if m.ndim == 3:
        m = m[:, None, :, :]
    # Convert to float tensor for interpolation
    mt = torch.from_numpy(m.astype(np.float32))  # (T, 1, Hm, Wm)
    mt = F.interpolate(mt, size=(h, w), mode="nearest")
    mb = (mt.numpy() > 0.5).astype(bool)
    return mb[:, 0]


def build_roi_masks_from_seg(seg_rgb: np.ndarray, color_threshold: float = 15.0) -> Dict[str, np.ndarray]:
    # seg_rgb: (T, H, W, 3) uint8
    seg = seg_rgb.astype(np.uint8)  # Ensure uint8 dtype
    
    # Extract object color and use closest color matching
    object_color = SEMANTIC_SEGMENTATION_MAPPING["class:object"]
    class_mask_obj = find_closest_color_mask(seg, object_color, color_threshold=color_threshold)
    
    return {
        "object": class_mask_obj,
    }


def apply_mask(frames: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # frames: (T, H, W, 3) uint8, mask: (T, H, W) bool
    # returns (masked_frames, valid_mask) where masked frames are zeroed outside ROI; valid_mask for weighting
    m = mask
    f = frames.copy().astype(np.float32)
    f[~m] = 0.0
    return f.astype(np.uint8), m


# LOGGING FUNCTIONS
def send_discord_message(message: str):
    """Sends a message to a Discord channel via a webhook."""
    if requests is None:
        return
    
    # Using an environment variable for the webhook URL is a good practice
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        # Silently skip if webhook URL is not set (don't spam console)
        return

    data = {"content": message}
    try:
        response = requests.post(webhook_url, json=data, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to send Discord notification: {e}")

def generate_test_config_combinations(test_configs):
    """
    Expand a test_configs dict into all possible combinations
    for list-valued keys, keeping scalar keys fixed.
    """
    # Keys that are lists -> will be combined
    keys_to_expand = ["guidance_scale", "horizons", "num_inference_steps"]

    # Fixed keys (copied into every combination)
    fixed = {
        k: v
        for k, v in test_configs.items()
        if k not in keys_to_expand
    }

    # Values to combine
    values_product = itertools.product(
        test_configs["guidance_scale"],
        test_configs["horizons"],
        test_configs["num_inference_steps"],
    )

    # Build list of config dicts
    combinations = []
    for gs, horizon, steps in values_product:
        cfg = {
            **fixed,
            "guidance_scale": gs,
            "horizon": horizon,          # singular name if you prefer
            "num_inference_steps": steps,
        }
        combinations.append(cfg)

    return combinations