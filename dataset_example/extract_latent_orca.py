import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mediapy
import argparse
from pathlib import Path
from diffusers.models import AutoencoderKL
import torch
import numpy as np
import json
from diffusers.models import AutoencoderKL,AutoencoderKLTemporalDecoder
from torch.utils.data import Dataset

import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R
from accelerate import Accelerator
from dotenv import load_dotenv
import requests
import datetime

from utils.utils import find_closest_color_mask

load_dotenv()

def send_discord_message(message: str):
    """Sends a message to a Discord channel via a webhook."""
    # Using an environment variable for the webhook URL is a good practice
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("Warning: DISCORD_WEBHOOK_URL environment variable not set. Skipping notification.")
        return

    data = {"content": message}
    try:
        response = requests.post(webhook_url, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to send Discord notification: {e}")


def parse_tuple(s):
    """Parse a string representation of a tuple into a tuple of integers."""
    try:
        # Remove parentheses and split by comma
        s = s.strip('()')
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("Tuple must be in format: (width,height) or width,height")



def _json_sort_key(path):
    stem = Path(path).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def _annotation_root_from_path(path):
    path = Path(path)
    if path.is_dir() and any(path.glob("*.json")):
        return path
    return path if path.name == "annotation" else path / "annotation"


def _require_2d_sequence(traj_data, key, annotation_path, expected_dim=None):
    sequence = traj_data.get(key)
    if sequence is None:
        raise KeyError(f"{annotation_path}: missing required annotation key: {key}")
    array = np.asarray(sequence, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{annotation_path}: {key} must be 2D, got shape {array.shape}.")
    if len(array) == 0:
        raise ValueError(f"{annotation_path}: {key} is empty.")
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise ValueError(f"{annotation_path}: {key} must have dim {expected_dim}, got {array.shape[1]}.")
    return array


def absolute_action_cartesian_pose_sequence(traj_data, annotation_path="<annotation>", quaternion_order="wxyz"):
    """Return absolute action EE pose as xyz+rpy from source action annotations."""
    existing_pose = traj_data.get("action.cartesian_pose")
    if existing_pose is not None:
        return _require_2d_sequence(traj_data, "action.cartesian_pose", annotation_path, expected_dim=6).astype(float).tolist()

    position = _require_2d_sequence(traj_data, "action.cartesian_position", annotation_path, expected_dim=3)
    if traj_data.get("action.cartesian_orientation_euler") is not None:
        orientation = _require_2d_sequence(traj_data, "action.cartesian_orientation_euler", annotation_path, expected_dim=3)
    else:
        quat = _require_2d_sequence(traj_data, "action.cartesian_orientation_quat", annotation_path, expected_dim=4)
        quat_norm = np.linalg.norm(quat, axis=1)
        if np.any(quat_norm < 1e-8):
            raise ValueError(f"{annotation_path}: action.cartesian_orientation_quat contains zero quaternions.")
        if quaternion_order == "wxyz":
            quat_xyzw = quat[:, [1, 2, 3, 0]]
        elif quaternion_order == "xyzw":
            quat_xyzw = quat
        else:
            raise ValueError(f"Unsupported quaternion_order: {quaternion_order}")
        orientation = R.from_quat(quat_xyzw).as_euler("xyz")

    if len(position) != len(orientation):
        raise ValueError(f"{annotation_path}: action cartesian position/orientation lengths differ: {len(position)} vs {len(orientation)}.")
    return np.concatenate((position, orientation), axis=1).astype(float).tolist()


def _target_annotation_length(traj_data, annotation_path):
    sequence_keys = (
        "observation.state.cartesian_position",
        "observation.state.cartesian_pose_absolute",
        "observation.state.cartesian_pose",
        "states",
    )
    lengths = []
    if "video_length" in traj_data:
        lengths.append(int(traj_data["video_length"]))
    for key in sequence_keys:
        sequence = traj_data.get(key)
        if sequence is not None:
            lengths.append(len(sequence))
    if not lengths:
        raise ValueError(f"{annotation_path}: could not infer target annotation length.")
    if len(set(lengths)) != 1:
        raise ValueError(f"{annotation_path}: inconsistent target sequence lengths: {lengths}.")
    return lengths[0]


def patch_action_cartesian_pose_annotations(target_path, source_path, original_fps, desired_fps, dry_run_num_episodes=None, skip_errors=False, action_quaternion_order="wxyz"):
    target_annotation_dir = _annotation_root_from_path(target_path)
    source_annotation_dir = _annotation_root_from_path(source_path)
    if not target_annotation_dir.is_dir():
        raise FileNotFoundError(f"Target annotation directory does not exist: {target_annotation_dir}")
    if not source_annotation_dir.is_dir():
        raise FileNotFoundError(f"Source annotation directory does not exist: {source_annotation_dir}")
    if desired_fps <= 0:
        raise ValueError(f"desired_fps must be positive, got {desired_fps}.")
    if original_fps % desired_fps != 0:
        raise ValueError(f"original_fps ({original_fps}) must be divisible by desired_fps ({desired_fps}).")

    down_sample = original_fps // desired_fps
    annotation_paths = sorted(target_annotation_dir.rglob("*.json"), key=_json_sort_key)
    total_annotation_files = len(annotation_paths)
    if dry_run_num_episodes is not None:
        annotation_paths = annotation_paths[:dry_run_num_episodes]
        print(f"Dry run: limiting annotation patch to {len(annotation_paths)} of {total_annotation_files} episodes.", flush=True)

    patched = 0
    skipped = 0
    for idx, target_annotation_path in enumerate(annotation_paths, start=1):
        rel_path = target_annotation_path.relative_to(target_annotation_dir)
        source_annotation_path = source_annotation_dir / rel_path
        if not source_annotation_path.is_file():
            source_annotation_path = source_annotation_dir / target_annotation_path.name
        try:
            if not source_annotation_path.is_file():
                raise FileNotFoundError(f"Missing source annotation for {target_annotation_path}: {source_annotation_path}")
            with target_annotation_path.open("r") as f:
                target_info = json.load(f)
            with source_annotation_path.open("r") as f:
                source_info = json.load(f)
            action_pose = absolute_action_cartesian_pose_sequence(source_info, annotation_path=str(source_annotation_path), quaternion_order=action_quaternion_order)[::down_sample]
            target_length = _target_annotation_length(target_info, target_annotation_path)
            if len(action_pose) > target_length:
                action_pose = action_pose[:target_length]
            if len(action_pose) != target_length:
                raise ValueError(f"{target_annotation_path}: downsampled action.cartesian_pose length {len(action_pose)} does not match target length {target_length}.")
            target_info["action.cartesian_pose"] = action_pose
            with target_annotation_path.open("w") as f:
                json.dump(target_info, f, indent=2)
            patched += 1
            if idx % 1000 == 0:
                print(f"Patched {idx}/{len(annotation_paths)} annotations.", flush=True)
        except Exception as e:
            if not skip_errors:
                raise
            skipped += 1
            print(f"Warning: skipping {target_annotation_path}: {e}", flush=True)
    return {
        "target_annotation_dir": str(target_annotation_dir),
        "source_annotation_dir": str(source_annotation_dir),
        "total": len(annotation_paths),
        "patched": patched,
        "skipped": skipped,
        "down_sample": down_sample,
    }

def _source_video_roots(dataset_path):
    return [
        os.path.join(dataset_path, folder_name)
        for folder_name in ('videos', 'segmentation_videos')
        if os.path.isdir(os.path.join(dataset_path, folder_name))
    ]


def _video_file_candidates(view_id):
    return [
        f'{view_id}_rgb.mp4',
        f'{view_id}.mp4',
        f'{view_id}_segmentation.mp4',
    ]


def get_video_frame_size(dataset_path):
    """Read the native frame size (height, width) from the first video found in the dataset."""
    for videos_dir in _source_video_roots(dataset_path):
        for traj_id in sorted(os.listdir(videos_dir)):
            for file_name in _video_file_candidates(0):
                video_path = os.path.join(videos_dir, traj_id, file_name)
                if os.path.isfile(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        return height, width
                    cap.release()
    return None


def get_video_fps(dataset_path):
    """Read source FPS from the first video found in the dataset."""
    for videos_dir in _source_video_roots(dataset_path):
        for traj_id in sorted(os.listdir(videos_dir)):
            for file_name in _video_file_candidates(0):
                video_path = os.path.join(videos_dir, traj_id, file_name)
                if os.path.isfile(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                        if fps and fps > 0:
                            return fps
                    cap.release()
    return None


class EncodeLatentDataset(Dataset): 
    def __init__(self, args, old_path, new_path, vae, entities_to_include_in_mask, size=(192, 320), val_ratio=0.3, split_train_val=True):
        self.args = args
        self.entities_to_include_in_mask = entities_to_include_in_mask
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.vae = vae
        self.val_ratio = val_ratio
        self.split_train_val = split_train_val
        self.has_rgb_videos = os.path.isdir(os.path.join(old_path, "videos"))

        annotation_root = os.path.join(old_path, "annotation")
        annotation_files = []
        for root, _, files in os.walk(annotation_root):
            for file_name in files:
                if file_name.lower().endswith(".json"):
                    annotation_files.append(os.path.join(root, file_name))

        annotation_files.sort(key=_json_sort_key)
        dry_run_num_episodes = getattr(self.args, "dry_run_num_episodes", None)
        if dry_run_num_episodes is not None:
            total_annotation_files = len(annotation_files)
            annotation_files = annotation_files[:dry_run_num_episodes]
            print(
                f"Dry run: limiting extraction to {len(annotation_files)} of "
                f"{total_annotation_files} episodes.",
                flush=True,
            )

        # # randomize order of annotation files
        np.random.shuffle(annotation_files)

        self.data = []
        for annotation_file in annotation_files:
            with open(annotation_file, 'r') as f:
                traj_data = json.loads(f.read())
            # Preserve source split (e.g. train/val) so input video paths are resolved correctly.
            rel_parent = os.path.relpath(os.path.dirname(annotation_file), annotation_root)
            source_data_type = None if rel_parent in (".", "") else rel_parent
            self.data.append(
                {
                    "traj_data": traj_data,
                    "source_data_type": source_data_type,
                    "annotation_file": annotation_file,
                }
            )

    def _traj_output_dir(self, save_root, folder_name, traj_id, data_type):
        if data_type is None:
            return f"{save_root}/{folder_name}/{traj_id}"
        return f"{save_root}/{folder_name}/{data_type}/{traj_id}"

    def _traj_rel_path(self, folder_name, traj_id, view_id, extension, data_type):
        if data_type is None:
            return f"{folder_name}/{traj_id}/{view_id}.{extension}"
        return f"{folder_name}/{data_type}/{traj_id}/{view_id}.{extension}"

    def _traj_output_complete(self, save_root, traj_id, data_type):
        annotation_dir = f"{save_root}/annotation" if data_type is None else f"{save_root}/annotation/{data_type}"
        required_paths = [f"{annotation_dir}/{traj_id}.json"]
        num_views = max(1, int(self.args.num_views))

        if not getattr(self.args, "annotation_only", False):
            for view_id in range(num_views):
                if self.has_rgb_videos:
                    required_paths.extend([
                        f"{self._traj_output_dir(save_root, 'videos', traj_id, data_type)}/{view_id}.mp4",
                        f"{self._traj_output_dir(save_root, 'latent_videos', traj_id, data_type)}/{view_id}.pt",
                    ])
                if getattr(self.args, "encode_segmentation_with_svd", False):
                    required_paths.extend([
                        f"{self._traj_output_dir(save_root, 'segmentation_videos', traj_id, data_type)}/{view_id}.mp4",
                        f"{self._traj_output_dir(save_root, 'latent_segmentation_videos', traj_id, data_type)}/{view_id}.pt",
                    ])
                if getattr(self.args, "use_hand_mask", False):
                    required_paths.append(
                        f"{self._traj_output_dir(save_root, 'downsampled_binary_masks', traj_id, data_type)}/{view_id}.pt"
                    )

        return all(os.path.isfile(path) for path in required_paths)

    def _require_sequence(self, traj_data, key):
        sequence = traj_data.get(key)
        if sequence is None:
            raise KeyError(f"Missing required annotation key: {key}")
        if len(sequence) == 0:
            raise ValueError(f"Annotation key {key} is empty.")
        return sequence

    def _optional_sequence(self, traj_data, key):
        sequence = traj_data.get(key)
        return [] if sequence is None else sequence

    def _absolute_ee_pose_sequence(self, traj_data):
        cartesian_pose = traj_data.get('observation.state.cartesian_pose')
        if cartesian_pose is not None and len(cartesian_pose) > 0 and len(cartesian_pose[0]) == 6:
            return cartesian_pose

        cartesian_position = self._require_sequence(traj_data, 'observation.state.cartesian_position')
        if len(cartesian_position[0]) == 6:
            return cartesian_position
        if len(cartesian_position[0]) != 3:
            raise ValueError(
                'observation.state.cartesian_position must contain either 3D xyz or 6D xyz+rpy, '
                f'got dim={len(cartesian_position[0])}.'
            )

        cartesian_euler = self._require_sequence(traj_data, 'observation.state.cartesian_orientation_euler')
        if len(cartesian_euler) != len(cartesian_position):
            raise ValueError(
                'Cartesian xyz and Euler orientation lengths differ: '
                f'{len(cartesian_position)} vs {len(cartesian_euler)}.'
            )
        return [list(position) + list(euler) for position, euler in zip(cartesian_position, cartesian_euler)]

    def _align_hand_action_dim(self, action_hand_joint, obs_hand_joint, traj_id):
        if len(action_hand_joint) == 0 or len(obs_hand_joint) == 0:
            return action_hand_joint

        action_dim = len(action_hand_joint[0])
        obs_dim = len(obs_hand_joint[0])
        if action_dim == obs_dim:
            return action_hand_joint
        if action_dim > obs_dim:
            print(
                f"Warning: trajectory {traj_id} action.hand_joint_position dim {action_dim} "
                f"exceeds observation hand dim {obs_dim}; truncating actions to {obs_dim} dims.",
                flush=True,
            )
            return [list(action)[:obs_dim] for action in action_hand_joint]
        raise ValueError(
            f"Trajectory {traj_id} action.hand_joint_position dim {action_dim} is smaller than "
            f"observation.state.hand_joint_position dim {obs_dim}."
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        traj_data = sample["traj_data"]
        source_data_type = sample["source_data_type"]
        annotation_file = sample["annotation_file"]
        instruction = traj_data['texts'][0]
        traj_id = traj_data['episode_id']

        # Use hash for better distribution, then threshold based on val_ratio.
        # Optionally skip train/val split entirely.
        if self.split_train_val:
            data_type = 'val' if (hash(traj_id) % 100) < (self.val_ratio * 100) else 'train'
        else:
            data_type = None
        obs_car = self._absolute_ee_pose_sequence(traj_data)
        length = len(obs_car)
        obs_joint = list(self._optional_sequence(traj_data, 'observation.state.joint_position'))
        obs_hand_joint_source = traj_data.get('observation.state.hand_joint_position')
        if obs_hand_joint_source is None:
            obs_hand_joint_source = self._require_sequence(traj_data, 'action.hand_joint_position')
        obs_hand_joint = list(obs_hand_joint_source)
        action_car = list(
            traj_data.get(
                'action.cartesian_relative_pose',
                traj_data.get('action.cartesian', [[0.0] * 6 for _ in range(length)]),
            )
        )
        action_abs_car = None
        if traj_data.get('action.cartesian_pose') is not None or traj_data.get('action.cartesian_position') is not None:
            action_abs_car = absolute_action_cartesian_pose_sequence(traj_data, annotation_path=annotation_file)
        action_hand_joint = list(self._require_sequence(traj_data, 'action.hand_joint_position'))
        action_hand_joint = self._align_hand_action_dim(action_hand_joint, obs_hand_joint, traj_id)

        sequence_lengths = {
            'observation.state.cartesian_pose_absolute': len(obs_car),
            'observation.state.hand_joint_position': len(obs_hand_joint),
            'action.cartesian_relative_pose': len(action_car),
            'action.hand_joint_position': len(action_hand_joint),
        }
        if action_abs_car is not None:
            sequence_lengths['action.cartesian_pose'] = len(action_abs_car)
        if obs_joint:
            sequence_lengths['observation.state.joint_position'] = len(obs_joint)
        mismatched = {key: value for key, value in sequence_lengths.items() if value != length}
        if mismatched:
            raise ValueError(
                f"Trajectory {traj_id} has sequence lengths that do not match EE pose length {length}: "
                f"{mismatched}"
            )

        success = traj_data['success']
        num_views = max(1, int(self.args.num_views))
        source_video_roots = _source_video_roots(self.old_path)
        if not source_video_roots:
            raise FileNotFoundError(
                f"Expected either a videos/ or segmentation_videos/ directory under {self.old_path}."
            )

        candidate_roots = []
        for videos_root in source_video_roots:
            split_video_root = videos_root if source_data_type is None else f"{videos_root}/{source_data_type}"
            # Be robust to both layouts:
            # - split:   videos/train/<traj_id>/...
            # - unsplit: videos/<traj_id>/...
            # - segmentation-only: segmentation_videos/<traj_id>/...
            if self.split_train_val:
                candidate_roots.extend([split_video_root, videos_root])
            else:
                candidate_roots.extend([videos_root, split_video_root])
        # Keep order while removing duplicates.
        candidate_roots = list(dict.fromkeys(candidate_roots))

        def _first_existing(base_dir, file_candidates):
            for name in file_candidates:
                path = f"{base_dir}/{name}"
                if os.path.isfile(path):
                    return path
            return f"{base_dir}/{file_candidates[0]}"

        def _rgb_video_candidates(view_id):
            # Prefer RGB if present, but support segmentation-only sim datasets.
            return [f"{view_id}.mp4", f"{view_id}_rgb.mp4", f"{view_id}_segmentation.mp4"]

        def _segmentation_candidates(view_id):
            # Prefer video files for VAE encoding; .pt masks are a fallback for sim-only utilities.
            return [
                f"{view_id}_segmentation.mp4",
                f"{view_id}.mp4",
                f"{view_id}_segmentation.pt",
                f"{view_id}.pt",
            ]

        input_video_root = candidate_roots[0]
        for candidate_root in candidate_roots:
            traj_dir = f"{candidate_root}/{traj_id}"
            probe_video_path = _first_existing(traj_dir, _rgb_video_candidates(0))
            if os.path.isfile(probe_video_path):
                input_video_root = candidate_root
                break

        input_traj_dir = f"{input_video_root}/{traj_id}"
        video_paths = [
            _first_existing(input_traj_dir, _rgb_video_candidates(view_id))
            for view_id in range(num_views)
        ]

        needs_segmentation = bool(getattr(self.args, "encode_segmentation_with_svd", False)) or bool(
            getattr(self.args, "use_hand_mask", False)
        )
        seg_pt_paths = None
        if needs_segmentation:
            seg_pt_paths = [
                _first_existing(input_traj_dir, _segmentation_candidates(view_id))
                for view_id in range(num_views)
            ]

        missing_paths = [path for path in video_paths if not os.path.isfile(path)]
        if seg_pt_paths is not None:
            missing_paths.extend(path for path in seg_pt_paths if not os.path.isfile(path))
        if missing_paths:
            raise FileNotFoundError(
                f"Trajectory {traj_id} is missing required video/segmentation files: {missing_paths}"
            )

        traj_info = {'success': success,
                     'data_motion_type': traj_data.get('data_motion_type'),
                     'observation.state.cartesian_position': obs_car,
                     'observation.state.cartesian_pose_absolute': obs_car,
                     'observation.state.joint_position': obs_joint,
                     'observation.state.hand_joint_position': obs_hand_joint,
                     'action.cartesian_relative_pose': action_car,
                     'action.cartesian_pose': action_abs_car,
                     'action.hand_joint_position': action_hand_joint,
                    }
        

        # if output for this trajectory exists, skip this trajectory
        if self._traj_output_complete(self.new_path, traj_id, data_type):
            print(f"Trajectory {traj_id} already processed, skipping.", flush=True)
            return 0

        try:
            process_device = self.vae.device if self.vae is not None else "cpu"
            self.process_traj(video_paths, seg_pt_paths, traj_info, instruction, self.new_path, traj_id=traj_id, data_type=data_type, size=self.size, device=process_device)
        except Exception as e:
            if getattr(self.args, "skip_errors", False):
                print(f"Error processing trajectory {traj_id}, skipping... Error: {e}")
                return 0
            raise
    
        return 0

    def downsample_binary_mask(self, video_id, seg_video_path, data_type, save_root, traj_id, target_height, target_width, folder_name='downsampled_binary_masks'):
        """
        Downsample a binary mask to the target height and width using nearest neighbor interpolation.
        
        Args:
            mask: Input binary mask as a numpy array or tensor of shape (T, H, W, 1)
            target_height: Desired height after downsampling
            target_width: Desired width after downsampling
            weight: Weight to multiply the downsampled mask
            
        Returns:
            Downsampled mask tensor of shape (B, 1, target_height, target_width)
        """
        seg_video = mediapy.read_video(seg_video_path)

        mask = torch.tensor(seg_video).permute(0, 3, 1, 2)
        mask = torch.sum(mask, dim=-3)  # Keep only one channel for binary mask
        mask = mask.unsqueeze(-1)  # Add channel dimension back

        # reduce temporal length
        mask = mask[::self.args.down_sample]

        # Convert to tensor if it's a numpy array
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        else:
            mask = mask.float()
        
        # Ensure mask is in the correct range [0, 1]
        if torch.max(mask) > 1.0:
            mask = mask / torch.max(mask)
            mask = torch.where(mask > 0.5, 1.0, 0.0)
        
        # Convert from (T, H, W, 1) to (T, 1, H, W) for interpolate function
        # PyTorch's interpolate expects (N, C, H, W) format
        mask = mask.permute(0, 3, 1, 2)  # (T, H, W, 1) -> (T, 1, H, W)
        
        # Downsample using nearest neighbor interpolation
        downsampled_mask = torch.nn.functional.interpolate(
            mask, 
            size=(target_height, target_width), 
            mode='nearest'
        )

        output_dir = self._traj_output_dir(save_root, folder_name, traj_id, data_type)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(downsampled_mask, f"{output_dir}/{video_id}.pt")

        return len(mask)
        
    def extract_rgb_latents(self, video_id, video_path, save_root, traj_id, data_type, size, device, folder_name='latent_videos'):
        # load and resize video and save
        if "segmentation" in folder_name:
            seg_pt_path = video_path
            if self.args.real_data:
                seg_pt_path = seg_pt_path.replace('.pt', '.mp4')
                video = mediapy.read_video(seg_pt_path)
            else:
                seg_pt = torch.load(seg_pt_path)
                video = seg_pt.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[:, :, :, :3]
        else:
            video = mediapy.read_video(video_path) # (T, H, W, 3) => numpy array

        desired_fps = self.args.original_fps / self.args.down_sample
        video = video[::self.args.down_sample]
        if "segmentation" in folder_name:
            if not self.args.real_data: 
                mask = find_closest_color_mask(video, self.entities_to_include_in_mask, color_threshold=1.0)
                # want to preserve the shape of the video but only keep the pixels that are in the mask
                video = video * mask[..., None]  # mask[..., None] adds channel dimension for broadcasting

        frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0*2-1
        print("frames shape: ", frames.shape)
        x = torch.nn.functional.interpolate(frames, size=size, mode='bilinear', align_corners=False)
        resize_video = ((x / 2.0 + 0.5).clamp(0, 1)*255)
        resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        
        if "segmentation" in folder_name:
            seg_dir = self._traj_output_dir(save_root, "segmentation_videos", traj_id, data_type)
            os.makedirs(seg_dir, exist_ok=True)
            mediapy.write_video(f"{seg_dir}/{video_id}.mp4", resize_video, fps=desired_fps)
            print("segmentation video")
        else:
            video_dir = self._traj_output_dir(save_root, "videos", traj_id, data_type)
            os.makedirs(video_dir, exist_ok=True)
            mediapy.write_video(f"{video_dir}/{video_id}.mp4", resize_video, fps=desired_fps)

        # save svd latent
        x = x.to(device)
        with torch.no_grad():
            batch_size = self.args.vae_batch_size
            latents = []
            # Handle both wrapped and unwrapped VAE models
            vae_model = self.vae.module if hasattr(self.vae, 'module') else self.vae
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                latent = vae_model.encode(batch).latent_dist.sample().mul_(vae_model.config.scaling_factor).cpu()
                latents.append(latent)
            x = torch.cat(latents, dim=0)
        output_dir = self._traj_output_dir(save_root, folder_name, traj_id, data_type)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(x, f"{output_dir}/{video_id}.pt")
        return len(frames)

    def _downsample_sequence(self, sequence, video_length=None):
        if sequence is None:
            return []
        sequence = list(sequence)
        down_sample = max(1, int(self.args.down_sample))
        sequence = sequence[::down_sample]
        if video_length is not None:
            sequence = sequence[:video_length]
        return sequence
    
    def process_traj(self, video_paths, seg_video_paths, traj_info, instruction, save_root,traj_id=0,data_type='val', size=(192,320), device='cuda'):
        video_length = None
        if not getattr(self.args, "annotation_only", False):
            video_lengths = []
            seg_video_paths_iter = seg_video_paths if seg_video_paths is not None else [None] * len(video_paths)
            for video_id, (video_path, seg_video_path) in enumerate(zip(video_paths, seg_video_paths_iter)):
                if self.has_rgb_videos:
                    # extract rgb latents using VAE
                    video_length = self.extract_rgb_latents(video_id, video_path, save_root, traj_id, data_type, size, device, folder_name='latent_videos')
                    video_lengths.append(video_length)
                else:
                    video_length = None

                if self.args.encode_segmentation_with_svd:
                    seg_latent_length = self.extract_rgb_latents(video_id, seg_video_path, save_root, traj_id, data_type, size, device, folder_name='latent_segmentation_videos')
                    if video_length is None:
                        video_length = seg_latent_length
                        video_lengths.append(video_length)
                    else:
                        assert seg_latent_length == video_length

                # get downsample binary mask using interpolate
                if self.args.use_hand_mask:
                    seg_length = self.downsample_binary_mask(video_id, seg_video_path, data_type, save_root, traj_id, int(self.args.height / self.args.vae_compression_rate), int(self.args.width / self.args.vae_compression_rate))
                    assert seg_length == video_length

                if self.args.debug:
                    break # for debugging
            if len(set(video_lengths)) > 1:
                raise ValueError(f"View lengths differ for trajectory {traj_id}: {video_lengths}")
            video_length = min(video_lengths) if video_lengths else 0
        else:
            # Annotation-only mode: skip video/latent generation.
            video_length = len(traj_info['observation.state.cartesian_position'][::self.args.down_sample])
        
        # Record frame-aligned annotations at the same FPS as the emitted videos/latents.
        obs_cartesian = self._downsample_sequence(traj_info['observation.state.cartesian_position'], video_length)
        obs_joint = self._downsample_sequence(traj_info['observation.state.joint_position'], video_length)
        obs_hand_joint = self._downsample_sequence(traj_info['observation.state.hand_joint_position'], video_length)
        abs_ee_pose = self._downsample_sequence(traj_info['observation.state.cartesian_pose_absolute'], video_length)
        action_cartesian = self._downsample_sequence(traj_info['action.cartesian_relative_pose'], video_length)
        action_cartesian_pose = None
        if traj_info.get('action.cartesian_pose') is not None:
            action_cartesian_pose = self._downsample_sequence(traj_info['action.cartesian_pose'], video_length)
        action_hand_joint = self._downsample_sequence(traj_info['action.hand_joint_position'], video_length)

        aligned_lengths = [
            len(obs_cartesian),
            len(abs_ee_pose),
            len(obs_hand_joint),
            len(action_cartesian),
            len(action_hand_joint),
            int(video_length),
        ]
        if obs_joint:
            aligned_lengths.append(len(obs_joint))
        if action_cartesian_pose is not None:
            aligned_lengths.append(len(action_cartesian_pose))
        aligned_length = min(aligned_lengths)
        if aligned_length != video_length:
            print(
                f"Warning: trajectory {traj_id} annotation/video lengths differ after downsampling; "
                f"using aligned length {aligned_length} instead of {video_length}.",
                flush=True,
            )
            video_length = aligned_length
            obs_cartesian = obs_cartesian[:aligned_length]
            abs_ee_pose = abs_ee_pose[:aligned_length]
            obs_joint = obs_joint[:aligned_length]
            obs_hand_joint = obs_hand_joint[:aligned_length]
            action_cartesian = action_cartesian[:aligned_length]
            if action_cartesian_pose is not None:
                action_cartesian_pose = action_cartesian_pose[:aligned_length]
            action_hand_joint = action_hand_joint[:aligned_length]

        if aligned_length <= 0:
            raise ValueError(f"Trajectory {traj_id} is empty after downsampling.")

        cartesian_pose = np.array(obs_cartesian)
        cartesian_hand_joint = np.array(obs_hand_joint)
        cartesian_states = np.concatenate((cartesian_pose, cartesian_hand_joint), axis=-1).tolist()
        
        info = {
            "texts": [instruction],
            "episode_id": traj_id,
            "success": int(traj_info['success']),
            "video_length": video_length,
            "state_length": len(cartesian_states),
            "raw_length": len(traj_info['observation.state.cartesian_position']),
            "fps": self.args.fps,
            "original_fps": self.args.original_fps,
            "down_sample": self.args.down_sample,
            'states': cartesian_states,
            'observation.state.cartesian_position': obs_cartesian,
            'observation.state.cartesian_pose_absolute': abs_ee_pose,
            'observation.state.joint_position': obs_joint,
            'observation.state.hand_joint_position': obs_hand_joint,
            'action.cartesian_relative_pose': action_cartesian,
            'action.hand_joint_position': action_hand_joint,
        }
        if action_cartesian_pose is not None:
            info['action.cartesian_pose'] = action_cartesian_pose

        data_motion_type = traj_info.get('data_motion_type')
        if data_motion_type is not None:
            info['data_motion_type'] = data_motion_type

        if self.has_rgb_videos:
            if not getattr(self.args, "annotation_only", False):
                info["videos"] = [
                    {"video_path": self._traj_rel_path("videos", traj_id, view_id, "mp4", data_type)}
                    for view_id in range(len(video_paths))
                ]
                info["latent_videos"] = [
                    {"latent_video_path": self._traj_rel_path("latent_videos", traj_id, view_id, "pt", data_type)}
                    for view_id in range(len(video_paths))
                ]
            else:
                # Keep source paths in annotation-only mode, but skip any VAE extraction outputs.
                info["videos"] = [
                    {"video_path": video_path}
                    for video_path in video_paths
                ]
                info["latent_videos"] = []

        if seg_video_paths is not None:
            if not getattr(self.args, "annotation_only", False):
                info['latent_segmentation_videos'] = [
                    {"latent_video_path": self._traj_rel_path("latent_segmentation_videos", traj_id, view_id, "pt", data_type)}
                    for view_id in range(len(seg_video_paths))
                ]
                info['segmentation_videos'] = [
                    {"video_path": self._traj_rel_path("segmentation_videos", traj_id, view_id, "mp4", data_type)}
                    for view_id in range(len(seg_video_paths))
                ]
            else:
                info['segmentation_videos'] = [
                    {"video_path": seg_video_path.replace(".pt", ".mp4") if getattr(self.args, "real_data", False) else seg_video_path}
                    for seg_video_path in seg_video_paths
                ]

        annotation_dir = f"{save_root}/annotation" if data_type is None else f"{save_root}/annotation/{data_type}"
        os.makedirs(annotation_dir, exist_ok=True)
        with open(f"{annotation_dir}/{traj_id}.json", "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    from config import wm_orca_args
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--orca_dataset_path', type=str, required=True, help='Converted ORCA dataset folder (annotation/ + videos)')
    parser.add_argument('--orca_output_path', type=str, required=True, help='Root folder for the extracted dataset; a timestamped subfolder is created')
    parser.add_argument('--output_timestamp', type=str, default=None, help='Override timestamp folder under --orca_output_path for resumable restarts.')
    parser.add_argument('--svd_path', type=str, default='stabilityai/stable-video-diffusion-img2vid')
    parser.add_argument('--frame_size', type=parse_tuple, default=(135, 240))
    parser.add_argument('--desired_fps', type=int, default=5, help='Target FPS for emitted videos, latents, and frame-aligned annotations')
    parser.add_argument('--original_fps', type=int, default=25, help='Source dataset FPS; must be divisible by desired FPS')
    parser.add_argument('--num_views', type=int, default=2, help='Number of camera views to process per trajectory')
    parser.add_argument('--use_hand_mask', action='store_true')
    parser.add_argument('--encode_segmentation_with_svd', type=bool, default=True)
    parser.add_argument('--annotation_only', action='store_true', help='Only write annotation JSON files; skip video/latent extraction')
    parser.add_argument('--patch_action_cartesian_pose_only', action='store_true', help='Patch existing annotation JSON files in-place with action.cartesian_pose and exit')
    parser.add_argument('--action_source_annotation_path', type=str, default=None, help='Source dataset or annotation directory for --patch_action_cartesian_pose_only')
    parser.add_argument('--action_quaternion_order', type=str, default='wxyz', choices=('wxyz', 'xyzw'), help='Quaternion component order for source action.cartesian_orientation_quat')
    parser.add_argument('--val_ratio', type=float, default=0.3)
    parser.add_argument('--no_train_val_split', action='store_true', default=True, help='Disable train/validation split and save all samples without train/val subfolders')
    parser.add_argument('--skip_errors', action='store_true', help='Skip malformed trajectories instead of failing loudly')
    parser.add_argument('--vae_batch_size', type=int, default=2048, help='Number of frames to encode per VAE micro-batch.')
    parser.add_argument(
        '--dry_run_num_episodes',
        '--dry-run-num-episodes',
        type=int,
        default=None,
        help='Limit extraction to the first N episodes for downstream smoke testing.',
    )
    # debug
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    if args.dry_run_num_episodes is not None and args.dry_run_num_episodes <= 0:
        raise ValueError('--dry_run_num_episodes must be a positive integer when set.')
    if args.vae_batch_size <= 0:
        raise ValueError('--vae_batch_size must be positive.')

    if args.patch_action_cartesian_pose_only:
        if args.action_source_annotation_path is None:
            raise ValueError('--action_source_annotation_path is required with --patch_action_cartesian_pose_only.')
        patch_stats = patch_action_cartesian_pose_annotations(
            target_path=args.orca_dataset_path,
            source_path=args.action_source_annotation_path,
            original_fps=args.original_fps,
            desired_fps=args.desired_fps,
            dry_run_num_episodes=args.dry_run_num_episodes,
            skip_errors=args.skip_errors,
            action_quaternion_order=args.action_quaternion_order,
        )
        print(
            'Patched action.cartesian_pose annotations: '
            f"{patch_stats['patched']}/{patch_stats['total']} "
            f"(skipped={patch_stats['skipped']}, down_sample={patch_stats['down_sample']})."
        )
        raise SystemExit(0)

    wm_arguments = wm_orca_args()
    # Propagate CLI camera-view override to wm config used downstream.
    wm_arguments.num_views = args.num_views
    wm_arguments.vae_batch_size = args.vae_batch_size

    if args.use_hand_mask:
        wm_arguments.use_hand_mask = True

    if args.encode_segmentation_with_svd:
        wm_arguments.encode_segmentation_with_svd = True
    wm_arguments.annotation_only = args.annotation_only
    wm_arguments.skip_errors = args.skip_errors
    wm_arguments.dry_run_num_episodes = args.dry_run_num_episodes

    annotation_dir = os.path.join(args.orca_dataset_path, "annotation")
    if not (os.path.isdir(annotation_dir) and len(_source_video_roots(args.orca_dataset_path)) > 0):
        raise FileNotFoundError(
            "--orca_dataset_path must be a converted ORCA (LeRobot-style) dataset folder with an "
            f"'annotation/' directory and source videos: {args.orca_dataset_path}"
        )
    inferred_fps = get_video_fps(args.orca_dataset_path)
    inferred_original_fps = int(round(inferred_fps)) if inferred_fps is not None else None
    data_list = [
        {
            'data_path': args.orca_dataset_path,
            'desired_fps': args.desired_fps if args.desired_fps is not None else 5,
            'original_fps': args.original_fps if args.original_fps is not None else inferred_original_fps,
            'folder_name': os.path.basename(os.path.normpath(args.orca_dataset_path)),
            # possible entities: 'hand', 'object', 'robot', 'ground', 'table', 'background'
            'entities_to_include_in_mask': [],
            'real_data': True,
        }
    ]

    accelerator = Accelerator()
    
    # Load VAE only when latent extraction is enabled.
    vae = None
    if not args.annotation_only:
        if accelerator.is_main_process:
            print(f"Loading VAE from {args.svd_path}...")
            print(f"Number of processes: {accelerator.num_processes}")
            print(f"Current process index: {accelerator.process_index}")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_path, subfolder="vae")
        vae = accelerator.prepare(vae)
        vae.eval()
    
    # Process each dataset in data_dict
    now = args.output_timestamp if args.output_timestamp is not None else str(datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    for dataset_config in data_list:
        dataset_path = dataset_config['data_path']
        desired_fps = args.desired_fps if args.desired_fps is not None else dataset_config['desired_fps']
        folder_name = dataset_config['folder_name']

        if dataset_config.get('real_data', False):
            wm_arguments.real_data = True

        entities_to_include_in_mask = dataset_config['entities_to_include_in_mask']

        # set wm_arguments
        wm_arguments.fps = desired_fps
        wm_arguments.latent_original_fps = desired_fps
        wm_arguments.original_fps = args.original_fps if args.original_fps is not None else dataset_config.get('original_fps', wm_arguments.original_fps)
        if wm_arguments.original_fps is None:
            raise ValueError(
                "Could not infer source FPS from the input dataset; pass --original_fps explicitly."
            )
        if desired_fps <= 0:
            raise ValueError(f"desired_fps must be positive, got {desired_fps}.")
        if wm_arguments.original_fps % desired_fps != 0:
            raise ValueError(
                f"original_fps ({wm_arguments.original_fps}) must be divisible by "
                f"desired_fps ({desired_fps}) for deterministic frame skipping."
            )
        wm_arguments.down_sample = wm_arguments.original_fps // desired_fps
        
        # Create output path for this specific dataset
        output_path = os.path.join(os.path.join(args.orca_output_path, now), folder_name + f'_{desired_fps}fps')
        os.makedirs(output_path, exist_ok=True)
        
        start_time = datetime.datetime.now()
        native_frame_size = get_video_frame_size(dataset_path)
        if accelerator.is_main_process:
            start_msg = f"🚀 **Starting Dataset Processing**\n" \
                       f"📁 Dataset: `{folder_name}`\n" \
                       f"🎯 Desired FPS: {desired_fps} Hz\n" \
                       f"⏰ Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            send_discord_message(start_msg)
            
            print(f"\n{'='*80}")
            print(f"Processing dataset: {folder_name}")
            print(f"Source path: {dataset_path}")
            print(f"Output path: {output_path}")
            print(f"Original FPS: {wm_arguments.original_fps} Hz")
            print(f"Desired FPS: {desired_fps} Hz")
            print(f"RGB skip: {wm_arguments.down_sample}")
            print(f"Actual FPS after skipping: {wm_arguments.original_fps / wm_arguments.down_sample} Hz")
            print(f"Native frame size (H x W): {native_frame_size}")
            print(f"Target frame size (H x W): {args.frame_size}")
            print(f"{'='*80}\n")
        
        dataset = EncodeLatentDataset(
            args=wm_arguments,
            entities_to_include_in_mask=entities_to_include_in_mask,
            old_path=dataset_path,
            new_path=output_path,
            vae=vae,
            size=args.frame_size,
            val_ratio=args.val_ratio,
            split_train_val=not args.no_train_val_split,
        )
        
        tmp_data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=0,  # Keep 0 because VAE is used in __getitem__
                pin_memory=True,
            )
        # Accelerate will automatically distribute dataset across GPUs
        tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)

        for idx, _ in enumerate(tmp_data_loader):
            if idx == 1 and args.debug:
                break
            if idx % 100 == 0 and accelerator.is_main_process:
                print(f"[{folder_name}] Precomputed {idx} samples", flush=True)
        
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        if accelerator.is_main_process:
            end_msg = f"✅ **Completed Dataset Processing**\n" \
                     f"📁 Dataset: `{folder_name}`\n" \
                     f"🎯 Desired FPS: {desired_fps} Hz\n" \
                     f"⏰ Finish Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n" \
                     f"⏱️ Duration: {str(duration).split('.')[0]}"
            send_discord_message(end_msg)
            print(f"\nCompleted processing {folder_name}\n")
