import json
import os
import random
from typing import Dict, List, Optional, Tuple

import mediapy
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torch.utils.data import Dataset

class Dataset_mix(Dataset):
    def __init__(
            self,
            args,
            mode = 'val',
    ):
        """Constructor."""
        super().__init__()
        self.args = args
        self.mode = mode

        # samples:{'ann_file':xxx, 'frame_idx':xxx, 'dataset_name':xxx}
        self.dataset_path_all: List[List[str]] = []
        self.samples_all: List[List[Dict]] = []
        self.samples_len: List[int] = []
        self.norm_all: List[Tuple[np.ndarray, np.ndarray]] = []
        self.model_norm_all: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

        dataset_root_path = args.dataset_root_path
        dataset_names = args.dataset_names.split('+')
        dataset_meta_info_path = args.dataset_meta_info_path
        dataset_meta_info_name = args.dataset_meta_info_name
        dataset_stat_path = getattr(args, "dataset_stat_path", None)
        model_stat_paths = {
            "wm1": getattr(args, "wm1_dataset_stat_path", None),
            "wm2": getattr(args, "wm2_dataset_stat_path", None),
            "baseline_wm": getattr(args, "baseline_wm_dataset_stat_path", None),
        }
        model_stat_paths = {
            model_key: stat_path
            for model_key, stat_path in model_stat_paths.items()
            if stat_path is not None
        }
        self.model_norm_all = {model_key: [] for model_key in model_stat_paths}
        self.dataset_names = dataset_names
        self.prob = self._normalize_dataset_prob(args.prob, len(dataset_names))
        self.balanced_validation_sampling = bool(
            getattr(args, "balanced_validation_sampling", True)
        )

        for dataset_name in dataset_names:
            if dataset_meta_info_name is not None:
                data_json_path = os.path.join(dataset_meta_info_path, dataset_meta_info_name, f"{mode}_sample.json")
            else:
                data_json_path = os.path.join(dataset_meta_info_path, dataset_name, f"{mode}_sample.json")
            print(data_json_path)
     
            with open(data_json_path, "r") as f:
                samples = json.load(f)

            print(f"ALL dataset, {len(samples)} samples in total")
            samples = self._exclude_configured_episodes(samples, dataset_name)
            
            if mode == 'train' and (len(samples) > args.max_num_samples):
                samples = self._random_subset(samples, args.max_num_samples)
            if mode == 'val' and (len(samples) > args.max_num_samples_for_validation):
                samples = self._random_subset(samples, args.max_num_samples_for_validation)

            if len(samples) == 0:
                raise ValueError(
                    f"No samples remain for dataset {dataset_name!r} in mode {mode!r} "
                    "after applying episode exclusions and sample limits."
                )

            dataset_path = [
                os.path.join(dataset_root_path, dataset_name) for sample in samples
            ]
            self.dataset_path_all.append(dataset_path)
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))

            if dataset_stat_path is not None:
                data_json_path = dataset_stat_path
                print(
                    f"Loading normalization statistics for dataset {dataset_name!r} "
                    f"from override: {data_json_path}"
                )
            elif dataset_meta_info_name is not None:
                data_json_path = f'{dataset_meta_info_path}/{dataset_meta_info_name}/stat.json'
            else:
                data_json_path = f'{dataset_meta_info_path}/{dataset_name}/stat.json'
            state_p01, state_p99 = self._load_norm_stats(data_json_path)
            self.norm_all.append((state_p01, state_p99))

            for model_key, stat_path in model_stat_paths.items():
                print(
                    f"Loading {model_key} normalization statistics for dataset {dataset_name!r} "
                    f"from override: {stat_path}"
                )
                self.model_norm_all[model_key].append(
                    self._load_norm_stats(stat_path)
                )
        
        self.max_id = max(self.samples_len)
        print('samples_len:',self.samples_len, 'max_id:',self.max_id)
        if self.mode == "val" and self.balanced_validation_sampling and len(self.samples_all) > 1:
            print(
                "Balanced validation sampling enabled: cycling evenly over "
                f"{len(self.samples_all)} datasets and sampling random examples within each dataset."
            )

    def __len__(self):
        if self.mode == "val" and self.balanced_validation_sampling and len(self.samples_all) > 1:
            return self.max_id * len(self.samples_all)
        return self.max_id

    def _normalize_dataset_prob(self, prob: List[float], num_datasets: int) -> List[float]:
        if prob is None or len(prob) == 0:
            return [1.0 / num_datasets] * num_datasets

        prob = [float(p) for p in prob]
        if len(prob) == 1 and num_datasets > 1:
            print(
                "Only one dataset probability was provided for multiple datasets; "
                "using equal probabilities."
            )
            return [1.0 / num_datasets] * num_datasets

        if len(prob) != num_datasets:
            raise ValueError(
                f"Expected {num_datasets} dataset probabilities for "
                f"dataset_names={self.args.dataset_names!r}, got {len(prob)}: {prob}"
            )

        prob_sum = sum(prob)
        if prob_sum <= 0:
            raise ValueError(f"Dataset probabilities must sum to a positive value, got {prob}.")

        return [p / prob_sum for p in prob]

    def _random_subset(self, samples: List[Dict], max_samples: int) -> List[Dict]:
        ids = np.random.choice(len(samples), max_samples, replace=False)
        return np.array(samples)[ids].tolist()

    def _load_norm_stats(self, stat_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(stat_path, "r") as f:
            data_stat = json.load(f)

        if "state_01" not in data_stat or "state_99" not in data_stat:
            raise KeyError(
                f"Statistics file {stat_path} must contain 'state_01' and 'state_99'."
            )

        state_p01 = np.array(data_stat["state_01"])[None, :]
        state_p99 = np.array(data_stat["state_99"])[None, :]
        if state_p01.shape != state_p99.shape:
            raise ValueError(
                f"Statistics file {stat_path} has mismatched bounds: "
                f"state_01 shape {state_p01.shape}, state_99 shape {state_p99.shape}."
            )
        return state_p01, state_p99

    def _exclude_configured_episodes(
        self, samples: List[Dict], dataset_name: str
    ) -> List[Dict]:
        exclusions_by_dataset = getattr(
            self.args, "exclude_episode_ids_by_dataset", {}
        ) or {}

        if not isinstance(exclusions_by_dataset, dict):
            raise TypeError(
                "exclude_episode_ids_by_dataset must be a mapping from dataset name "
                f"to a list of episode IDs, got {type(exclusions_by_dataset).__name__}."
            )

        excluded_episode_ids = list(exclusions_by_dataset.get(dataset_name, []))
        excluded_episode_ids += list(exclusions_by_dataset.get("__all__", []))

        if not excluded_episode_ids:
            return samples

        excluded_episode_ids = {str(episode_id) for episode_id in excluded_episode_ids}
        filtered_samples = [
            sample
            for sample in samples
            if str(sample.get("episode_id")) not in excluded_episode_ids
        ]
        removed_count = len(samples) - len(filtered_samples)
        print(
            f"Excluded {removed_count} samples from dataset {dataset_name!r} "
            f"for episode IDs: {sorted(excluded_episode_ids)}"
        )
        return filtered_samples

    def _resolve_annotation_file(self, dataset_dir: str, sample: Dict) -> str:
        ann_name = self.args.annotation_name

        # Prefer explicit annotation path from meta info when available.
        sample_ann_file = sample.get("ann_file")
        if sample_ann_file is not None:
            if os.path.isabs(sample_ann_file):
                if os.path.exists(sample_ann_file):
                    return sample_ann_file
            else:
                rel_candidate = os.path.join(dataset_dir, sample_ann_file)
                if os.path.exists(rel_candidate):
                    return rel_candidate

        episode_id = sample["episode_id"]
        candidates = [
            os.path.join(dataset_dir, ann_name, self.mode, f"{episode_id}.json"),
            os.path.join(dataset_dir, ann_name, "validation", f"{episode_id}.json"),
            os.path.join(dataset_dir, ann_name, "val", f"{episode_id}.json"),
            os.path.join(dataset_dir, ann_name, "train", f"{episode_id}.json"),
            os.path.join(dataset_dir, ann_name, f"{episode_id}.json"),
        ]

        for ann_file in dict.fromkeys(candidates):
            if os.path.exists(ann_file):
                return ann_file

        raise FileNotFoundError(
            f"Could not find annotation file for episode_id={episode_id}. "
            f"Tried: {list(dict.fromkeys(candidates))}"
        )

    def _load_latent_video(
        self,
        video_path: str,
        frame_ids: List[int],
    ) -> torch.Tensor:
        with open(video_path,'rb') as file:
            video_tensor = torch.load(file)
            video_tensor.requires_grad = False
        max_frames = video_tensor.size()[0]
        frame_ids =  [
            int(frame_id) if frame_id < max_frames else max_frames-1 
            for frame_id in frame_ids
        ]
        return video_tensor[frame_ids]

    def _read_and_normalize_rgb_video(
        self, video_path: str, frame_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        if frame_ids is None:
            video = mediapy.read_video(video_path) # (T, H, W, 3)
        else:
            vr = VideoReader(video_path, ctx=cpu(0))
            max_frames = len(vr)
            safe_ids = [
                int(frame_id) if frame_id < max_frames else max_frames - 1
                for frame_id in frame_ids
            ]
            video = vr.get_batch(safe_ids).asnumpy()
        frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1 # (T, 3, H, W)
        return frames

    def _get_frames(
        self,
        label: Dict,
        datatype: str,
        frame_ids: List[int],
        cam_id: int,
        video_dir: str,
    ) -> torch.Tensor:
        assert cam_id is not None
        if "latent" in datatype: 
            video_path = label[datatype][cam_id]['latent_video_path']
            video_path = os.path.join(video_dir,video_path)
            return self._load_latent_video(video_path, frame_ids)

        video_path = label[datatype][cam_id]['video_path']
        return self._read_and_normalize_rgb_video(
            os.path.join(video_dir, video_path)
        )[frame_ids]
    
    def _get_hand_mask(
        self,
        label: Dict,
        frame_ids: List[int],
        cam_id: int,
        pre_encode: bool,
        video_dir: str,
    ) -> torch.Tensor:
        assert cam_id is not None
        assert pre_encode is True
        
        mask_video_path = label["latent_segmentation_videos"][cam_id]["latent_video_path"]
        mask_video_path = os.path.join(video_dir,mask_video_path)
        try:
            frames = self._load_latent_video(mask_video_path, frame_ids)
        except Exception as e:
            raise RuntimeError(
                f"Error loading hand mask video from {mask_video_path}: {e}"
            ) from e

        return frames * self.args.hand_weight

    def _input_latent_datatype(self, label: Dict) -> str:
        if label.get("latent_videos"):
            return "latent_videos"
        if label.get("latent_segmentation_videos"):
            return "latent_segmentation_videos"
        raise KeyError(
            "Annotation must contain either 'latent_videos' or 'latent_segmentation_videos'."
        )

    def _get_obs(self,
        label: Dict,
        datatype: str,
        frame_ids: List[int],
        cam_id: int,
        video_dir: str,
    ) -> Tuple[torch.Tensor, int]:
        temp_cam_id = (
            random.choice(range(self.args.num_views)) if cam_id is None else cam_id
        )
        frames = self._get_frames(
            label, datatype, frame_ids, cam_id=temp_cam_id, video_dir=video_dir
        )
        return frames, temp_cam_id

    def _load_segmentation_video_and_normalize(
        self,
        label: Dict,
        frame_ids: List[int],
        cam_id: int,
        video_dir: str,
    ) -> torch.Tensor:
        video_path = label["segmentation_videos"][cam_id]['video_path']
        video_path = os.path.join(video_dir,video_path)
        return self._read_and_normalize_rgb_video(video_path, frame_ids=frame_ids)

    def _load_vis_seg_actions_if_available(
        self,
        label: Dict,
        frame_ids: List[int],
        cam_id: int,
        video_dir: str,
    ) -> Optional[torch.Tensor]:
        vis_seg_actions = label.get("vis_seg_actions")
        if vis_seg_actions is None:
            return None

        if isinstance(vis_seg_actions, dict):
            cam_entry = vis_seg_actions.get(cam_id, vis_seg_actions.get(str(cam_id)))
        else:
            try:
                cam_entry = vis_seg_actions[cam_id]
            except (IndexError, KeyError, TypeError):
                return None

        if cam_entry is None:
            return None

        if "video_path" in cam_entry:
            video_path = os.path.join(video_dir, cam_entry["video_path"])
            frames = self._read_and_normalize_rgb_video(video_path)
            max_frames = frames.size()[0]
            frame_ids = [
                int(frame_id) if frame_id < max_frames else max_frames - 1
                for frame_id in frame_ids
            ]
            return frames[frame_ids]

        if "latent_video_path" in cam_entry:
            video_path = os.path.join(video_dir, cam_entry["latent_video_path"])
            return self._load_latent_video(video_path, frame_ids)

        return None

    def _annotation_array(
        self,
        label: Dict,
        keys: Tuple[str, ...],
    ) -> np.ndarray:
        for key in keys:
            if key in label:
                return np.array(label[key])
        raise KeyError(f"Annotation is missing required field. Tried keys: {keys}")

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps=1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata

    def _maybe_swap_abd_with_mcp(
        self,
        full_action_seq: np.ndarray,
        swap_abd_with_mcp: bool,
    ) -> np.ndarray:
        if not swap_abd_with_mcp:
            return full_action_seq
        if full_action_seq.shape[-1] <= 2:
            raise ValueError(
                "swap_abd_with_mcp requires at least 3 hand action joints, "
                f"got shape {full_action_seq.shape}."
            )
        swapped = full_action_seq.copy()
        swapped[:, [1, 2]] = swapped[:, [2, 1]]
        return swapped

    def _shift_action_sequence(self, full_action_seq: np.ndarray) -> np.ndarray:
        shift_steps = max(1, int(self.args.down_sample))
        num_actions = full_action_seq.shape[0]
        effective_shift = min(shift_steps, num_actions)
        if effective_shift == num_actions:
            return np.zeros_like(full_action_seq)
        pad_shape = (effective_shift, *full_action_seq.shape[1:])
        return np.concatenate(
            (
                np.zeros(pad_shape, dtype=full_action_seq.dtype),
                full_action_seq[:-effective_shift],
            ),
            axis=0,
        )

    def _normalize_action_from_bounds(
        self,
        cartesian_pose: np.ndarray,
        gripper_action: np.ndarray,
        state_p01: np.ndarray,
        state_p99: np.ndarray,
        ee_pose_dims: int,
    ) -> np.ndarray:
        if self.args.use_only_hand_actions:
            action = self.normalize_bound(
                gripper_action,
                state_p01[:, ee_pose_dims:],
                state_p99[:, ee_pose_dims:],
            )
            if self.args.use_average_scalar_hand_action:
                action = np.mean(action, axis=-1, keepdims=True)
            return action

        if self.args.use_only_ee_pose_actions:
            return self.normalize_bound(
                cartesian_pose,
                state_p01[:, :ee_pose_dims],
                state_p99[:, :ee_pose_dims],
            )

        action = np.concatenate((cartesian_pose, gripper_action), axis=-1)
        return self.normalize_bound(action, state_p01, state_p99)

    def _build_frame_ids(
        self, frame_ids: List[int], state_len: int
    ) -> Tuple[List[int], np.ndarray]:
        down_sample = max(1, int(self.args.down_sample))
        # `joint_len` may be -1 for malformed/empty trajectories; never allow
        # a negative clip upper bound since that can yield -1 indices.
        max_frame_idx = max(0, int(np.floor(state_len / down_sample)))

        skip = random.randint(self.args.min_stride, self.args.max_stride)
        skip_his = int(skip * self.args.stride_factor)
        if random.random() < 0.15:
            skip_his = 0

        fps_ratio = max(1, int(self.args.latent_original_fps / self.args.fps))
        frame_now = frame_ids[0] // fps_ratio
        rgb_id: List[int] = []

        for i in range(self.args.num_history, 0, -1):
            rgb_id.append(int(frame_now - i * skip_his))
        rgb_id.append(frame_now)

        for i in range(1, self.args.num_frames):
            rgb_id.append(int(frame_now + i * skip))

        rgb_id = np.array(rgb_id)
        rgb_id = np.clip(rgb_id, 0, max_frame_idx).tolist()
        rgb_id = [int(frame_id) for frame_id in rgb_id]
        state_id = np.array(rgb_id) * down_sample

        return rgb_id, state_id
    
    def __getitem__(self, index, return_frame_ids=False, dataset_id=None):
        # first sample the dataset id, than sample the data from the dataset
        if dataset_id is None:
            if (
                self.mode == "val"
                and self.balanced_validation_sampling
                and len(self.samples_all) > 1
            ):
                dataset_id = int(index % len(self.samples_all))
                index = int(np.random.randint(len(self.samples_all[dataset_id])))
            else:
                dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
        
        # get samples from the dataset
        samples = self.samples_all[dataset_id]
        # get statistics for the dataset
        state_p01, state_p99 = self.norm_all[dataset_id]

        # get sample from the dataset
        index = index % len(samples)
        sample = samples[index]

        # get annotation
        dataset_path = self.dataset_path_all[dataset_id]
        dataset_dir = dataset_path[index]
        
        # get frame ids and annotation file
        frame_ids = sample['frame_ids']
        ann_file = self._resolve_annotation_file(dataset_dir, sample)
        with open(ann_file, "r") as f:
            label = json.load(f)

        if self.args.real_data:
            state_sequence = label.get(
                "action.cartesian_pose",
                label.get("observation.state.cartesian_position", []),
            )
        else:
            # Keep legacy behavior for sim data while remaining robust if key is missing.
            state_sequence = label.get(
                "observation.state.joint_position",
                label.get("observation.state.cartesian_position", []),
            )
        rgb_id, state_id = self._build_frame_ids(frame_ids, len(state_sequence) - 1)

        # prepare data
        data: Dict[str, torch.Tensor] = {}
        ## text instructions
        data['text'] = label['texts'][0]
        
        ## get the input latent data and the concatenation conditioning data
        ## we can also get hand weights here if the datasets has that information
        ## or apply interpolation on the segmentation video here if the datasets has that information
        compressed_height = self.args.height // self.args.vae_compression_rate
        compressed_width = self.args.width // self.args.vae_compression_rate

        use_vis_seg_actions_for_controlnet = bool(
            getattr(self.args, "use_vis_seg_actions_for_controlnet", False)
        )
        input_latent_datatype = self._input_latent_datatype(label)

        if self.args.num_views == 1:
            if self.args.real_data:
                cond_cam_id = 1 if self.args.only_wrist_view else 0
            else:
                cond_cam_id = 0 if self.args.only_wrist_view is False else 2
            
            ## get the latent main datatype: Can be latent_segmentation or latent_rgb
            latent_videos_cond,_ = self._get_obs(
                label, input_latent_datatype, rgb_id, cond_cam_id, video_dir=dataset_dir
            )
            data["latent_videos"] = latent_videos_cond.float()
            if label.get("latent_segmentation_videos"):
                latent_segmentation_cond,_ = self._get_obs(
                    label,
                    "latent_segmentation_videos",
                    rgb_id,
                    cond_cam_id,
                    video_dir=dataset_dir,
                )
                data["latent_segmentation_videos"] = latent_segmentation_cond.float()

            ## get the hand mask for the action encoder
            if self.args.use_hand_mask:
                data["hand_mask"] = self._get_hand_mask(
                    label, rgb_id, cond_cam_id, pre_encode=True, video_dir=dataset_dir
                ).float()
            
            ## get the segmentation video for the action encoder
            _need_rgb_seg_for_controlnet = (
                self.args.use_controlnet_conditioning
                and not getattr(self.args, "use_vae_roundtrip_for_controlnet", False)
            )
            if "dino_visual" in self.args.action_encoder or _need_rgb_seg_for_controlnet:
                seg_video = self._load_segmentation_video_and_normalize(
                    label, rgb_id, cond_cam_id, video_dir=dataset_dir
                ).float()
                data["segmentation_videos"] = seg_video
            
            ## nearest neighbour downsample the segmentation video to the latent size
            if self.args.concatenate_latent == "interpolated_downsampled_segmentation":
                seg_video = self._load_segmentation_video_and_normalize(
                    label, rgb_id, cond_cam_id, video_dir=dataset_dir
                ).float()
                seg_video = F.interpolate(
                    seg_video,
                    size=(compressed_height, compressed_width),
                    mode=self.args.downsample_method_for_segmentation,
                )
                data["interpolated_downsampled_segmentation"] = seg_video

            if use_vis_seg_actions_for_controlnet:
                vis_seg_actions = self._load_vis_seg_actions_if_available(
                    label, rgb_id, cond_cam_id, video_dir=dataset_dir
                )
                if vis_seg_actions is not None:
                    data["vis_seg_actions"] = vis_seg_actions.float()
            
        elif self.args.num_views == 2:
            cond_cam_id1 = 0

            if self.args.real_data:
                cond_cam_id2 = 1
            else:
                cond_cam_id2 = 2
            
            ## get the input latent data and the concatenation conditioning data
            latent_videos_cond1,_ = self._get_obs(
                label, input_latent_datatype, rgb_id, cond_cam_id1, video_dir=dataset_dir
            )
            latent_videos_cond2,_ = self._get_obs(
                label, input_latent_datatype, rgb_id, cond_cam_id2, video_dir=dataset_dir
            )

            ## stack the latent videos
            latent_videos = torch.zeros(
                (
                    self.args.num_frames + self.args.num_history,
                    4,
                    2 * compressed_height,
                    compressed_width,
                ),
                dtype=self.args.dtype,
            )
            latent_videos[:, :, 0:compressed_height, :] = latent_videos_cond1
            latent_videos[:, :, compressed_height:, :] = latent_videos_cond2
            data["latent_videos"] = latent_videos.float()

            if label.get("latent_segmentation_videos"):
                ## get the input latent data and the concatenation conditioning data
                latent_segmentation_cond1, _ = self._get_obs(
                    label,
                    "latent_segmentation_videos",
                    rgb_id,
                    cond_cam_id1,
                    video_dir=dataset_dir,
                )
                latent_segmentation_cond2, _ = self._get_obs(
                    label,
                    "latent_segmentation_videos",
                    rgb_id,
                    cond_cam_id2,
                    video_dir=dataset_dir,
                )

                ## stack the latent segmentation videos
                latent_segmentations = torch.zeros(
                    (
                        self.args.num_frames + self.args.num_history,
                        4,
                        2 * compressed_height,
                        compressed_width,
                    ),
                    dtype=self.args.dtype,
                )
                latent_segmentations[:, :, 0:compressed_height, :] = latent_segmentation_cond1
                latent_segmentations[:, :, compressed_height:, :] = latent_segmentation_cond2
                data["latent_segmentation_videos"] = latent_segmentations.float()

            ## get the hand mask for the action encoder
            if self.args.use_hand_mask:
                latent_seg1 = self._get_hand_mask(
                    label, rgb_id, cond_cam_id1, pre_encode=True, video_dir=dataset_dir
                )
                latent_seg2 = self._get_hand_mask(
                    label, rgb_id, cond_cam_id2, pre_encode=True, video_dir=dataset_dir
                )
                latent_seg = torch.zeros(
                    (
                        self.args.num_frames + self.args.num_history,
                        4,
                        2 * compressed_height,
                        compressed_width,
                    ),
                    dtype=self.args.dtype,
                )
                latent_seg[:, :, 0:compressed_height, :] = latent_seg1
                latent_seg[:, :, compressed_height:, :] = latent_seg2
                data["hand_mask"] = latent_seg.float()

            ## get the segmentation video for the action encoder
            if "dino_visual" in self.args.action_encoder:
                seg_action_cond1 = self._load_segmentation_video_and_normalize(
                    label, rgb_id, cond_cam_id1, video_dir=dataset_dir
                ).float() # (T, 3, H, W)
                seg_action_cond2 = self._load_segmentation_video_and_normalize(
                    label, rgb_id, cond_cam_id2, video_dir=dataset_dir
                ).float() # (T, 3, H, W)
                data["segmentation_videos"] = torch.cat((seg_action_cond1, seg_action_cond2), dim=1)
            
            if self.args.use_controlnet_conditioning and not getattr(self.args, "use_vae_roundtrip_for_controlnet", False):
                seg_action_cond1 = self._load_segmentation_video_and_normalize(
                    label, rgb_id, cond_cam_id1, video_dir=dataset_dir
                ).float() # (T, 3, H, W)
                seg_action_cond2 = self._load_segmentation_video_and_normalize(
                    label, rgb_id, cond_cam_id2, video_dir=dataset_dir
                ).float() # (T, 3, H, W)
                data["segmentation_videos"] = torch.cat((seg_action_cond1, seg_action_cond2), dim=2) # (T, 3, 2*H, W)

            ## nearest neighbour downsample the segmentation video to the latent size
            if self.args.concatenate_latent == "interpolated_downsampled_segmentation":
                seg_action_cond1 = self._load_segmentation_video_and_normalize(label, rgb_id, cond_cam_id1, video_dir=dataset_dir).float() # (T, 3, H, W)
                seg_action_cond2 = self._load_segmentation_video_and_normalize(label, rgb_id, cond_cam_id2, video_dir=dataset_dir).float() # (T, 3, H, W)
                seg_action_cond1 = F.interpolate(
                    seg_action_cond1,
                    size=(compressed_height, compressed_width),
                    mode=self.args.downsample_method_for_segmentation,
                ) # (T, 3, H, W)
                seg_action_cond2 = F.interpolate(
                    seg_action_cond2,
                    size=(compressed_height, compressed_width),
                    mode=self.args.downsample_method_for_segmentation,
                ) # (T, 3, H, W)
                data["interpolated_downsampled_segmentation"] = torch.cat((seg_action_cond1, seg_action_cond2), dim=2) # (T, 3, 2*H, W)

            if use_vis_seg_actions_for_controlnet:
                vis_seg_action_cond1 = self._load_vis_seg_actions_if_available(
                    label, rgb_id, cond_cam_id1, video_dir=dataset_dir
                )
                vis_seg_action_cond2 = self._load_vis_seg_actions_if_available(
                    label, rgb_id, cond_cam_id2, video_dir=dataset_dir
                )
                if vis_seg_action_cond1 is not None and vis_seg_action_cond2 is not None:
                    data["vis_seg_actions"] = torch.cat(
                        (vis_seg_action_cond1.float(), vis_seg_action_cond2.float()),
                        dim=2,
                    ) # (T, C, 2*H, W)

        elif self.args.num_views == 3:
            cond_cam_id1 = 0
            cond_cam_id2 = 1
            cond_cam_id3 = 2

            ## get the input latent data and the concatenation conditioning data
            latent_videos_cond1,_ = self._get_obs(label, input_latent_datatype, rgb_id, cond_cam_id1, video_dir=dataset_dir)
            latent_videos_cond2,_ = self._get_obs(label, input_latent_datatype, rgb_id, cond_cam_id2, video_dir=dataset_dir)
            latent_videos_cond3,_ = self._get_obs(label, input_latent_datatype, rgb_id, cond_cam_id3, video_dir=dataset_dir)

            ## stack the latent videos
            latent_videos = torch.zeros(
                (
                    self.args.num_frames + self.args.num_history,
                    4,
                    3 * compressed_height,
                    compressed_width,
                ),
                dtype=self.args.dtype,
            )
            latent_videos[:, :, 0:compressed_height, :] = latent_videos_cond1
            latent_videos[:, :, compressed_height:2*compressed_height, :] = latent_videos_cond2
            latent_videos[:, :, 2*compressed_height:, :] = latent_videos_cond3
            data["latent_videos"] = latent_videos.float()

            if label.get("latent_segmentation_videos"):
                ## get the input latent data and the concatenation conditioning data
                latent_segmentation_cond1,_ = self._get_obs(
                    label,
                    "latent_segmentation_videos",
                    rgb_id,
                    cond_cam_id1,
                    video_dir=dataset_dir,
                )
                latent_segmentation_cond2,_ = self._get_obs(
                    label,
                    "latent_segmentation_videos",
                    rgb_id,
                    cond_cam_id2,
                    video_dir=dataset_dir,
                )
                latent_segmentation_cond3,_ = self._get_obs(
                    label,
                    "latent_segmentation_videos",
                    rgb_id,
                    cond_cam_id3,
                    video_dir=dataset_dir,
                )

                ## stack the latent segmentation videos
                latent_segmentations = torch.zeros(
                    (
                        self.args.num_frames + self.args.num_history,
                        4,
                        3 * compressed_height,
                        compressed_width,
                    ),
                    dtype=self.args.dtype,
                )
                latent_segmentations[:, :, 0:compressed_height, :] = latent_segmentation_cond1
                latent_segmentations[:, :, compressed_height:2*compressed_height, :] = latent_segmentation_cond2
                latent_segmentations[:, :, 2*compressed_height:, :] = latent_segmentation_cond3
                data["latent_segmentation_videos"] = latent_segmentations.float()

            ## get the hand mask for the action encoder
            if self.args.use_hand_mask:
                latent_seg1 = self._get_hand_mask(label, rgb_id, cond_cam_id1, pre_encode=True, video_dir=dataset_dir)
                latent_seg2 = self._get_hand_mask(label, rgb_id, cond_cam_id2, pre_encode=True, video_dir=dataset_dir)
                latent_seg3 = self._get_hand_mask(label, rgb_id, cond_cam_id3, pre_encode=True, video_dir=dataset_dir)
                latent_seg = torch.zeros(
                    (
                        self.args.num_frames + self.args.num_history,
                        4,
                        3 * compressed_height,
                        compressed_width,
                    ),
                    dtype=self.args.dtype,
                )
                latent_seg[:, :, 0:compressed_height, :] = latent_seg1
                latent_seg[:, :, compressed_height:2*compressed_height, :] = latent_seg2
                latent_seg[:, :, 2*compressed_height:, :] = latent_seg3
                data["hand_mask"] = latent_seg.float()

            ## get the segmentation video for the action encoder
            _need_rgb_seg_for_controlnet = (
                self.args.use_controlnet_conditioning
                and not getattr(self.args, "use_vae_roundtrip_for_controlnet", False)
            )
            if "dino_visual" in self.args.action_encoder or _need_rgb_seg_for_controlnet:
                seg_action_cond1 = self._load_segmentation_video_and_normalize(label, rgb_id, cond_cam_id1, video_dir=dataset_dir).float()
                seg_action_cond2 = self._load_segmentation_video_and_normalize(label, rgb_id, cond_cam_id2, video_dir=dataset_dir).float()
                seg_action_cond3 = self._load_segmentation_video_and_normalize(label, rgb_id, cond_cam_id3, video_dir=dataset_dir).float()
                data['segmentation_videos'] = torch.concat((seg_action_cond1, seg_action_cond2, seg_action_cond3), dim=1) # (T, 9, H, W)

            if _need_rgb_seg_for_controlnet:
                seg_action_cond1 = self._load_segmentation_video_and_normalize(label, rgb_id, cond_cam_id1, video_dir=dataset_dir).float() # (T, 3, H, W)
                seg_action_cond2 = self._load_segmentation_video_and_normalize(label, rgb_id, cond_cam_id2, video_dir=dataset_dir).float() # (T, 3, H, W)
                seg_action_cond3 = self._load_segmentation_video_and_normalize(label, rgb_id, cond_cam_id3, video_dir=dataset_dir).float() # (T, 3, H, W)
                data['segmentation_videos'] = torch.concat((seg_action_cond1, seg_action_cond2, seg_action_cond3), dim=2) # (T, 3, 3*H, W)

            ## nearest neighbour downsample the segmentation video to the latent size
            if self.args.concatenate_latent == "interpolated_downsampled_segmentation":
                seg_action_cond1 = self._load_segmentation_video_and_normalize(label, rgb_id, cond_cam_id1, video_dir=dataset_dir).float()
                seg_action_cond2 = self._load_segmentation_video_and_normalize(label, rgb_id, cond_cam_id2, video_dir=dataset_dir).float()
                seg_action_cond3 = self._load_segmentation_video_and_normalize(label, rgb_id, cond_cam_id3, video_dir=dataset_dir).float()

                seg_action_cond1 = F.interpolate(
                    seg_action_cond1,
                    size=(compressed_height, compressed_width),
                    mode=self.args.downsample_method_for_segmentation,
                )
                seg_action_cond2 = F.interpolate(
                    seg_action_cond2,
                    size=(compressed_height, compressed_width),
                    mode=self.args.downsample_method_for_segmentation,
                )
                seg_action_cond3 = F.interpolate(
                    seg_action_cond3,
                    size=(compressed_height, compressed_width),
                    mode=self.args.downsample_method_for_segmentation,
                )
                data['interpolated_downsampled_segmentation'] = torch.concat((seg_action_cond1, seg_action_cond2, seg_action_cond3), dim=2)

            if use_vis_seg_actions_for_controlnet:
                vis_seg_action_cond1 = self._load_vis_seg_actions_if_available(
                    label, rgb_id, cond_cam_id1, video_dir=dataset_dir
                )
                vis_seg_action_cond2 = self._load_vis_seg_actions_if_available(
                    label, rgb_id, cond_cam_id2, video_dir=dataset_dir
                )
                vis_seg_action_cond3 = self._load_vis_seg_actions_if_available(
                    label, rgb_id, cond_cam_id3, video_dir=dataset_dir
                )
                if (
                    vis_seg_action_cond1 is not None
                    and vis_seg_action_cond2 is not None
                    and vis_seg_action_cond3 is not None
                ):
                    data["vis_seg_actions"] = torch.concat(
                        (
                            vis_seg_action_cond1.float(),
                            vis_seg_action_cond2.float(),
                            vis_seg_action_cond3.float(),
                        ),
                        dim=2,
                    ) # (T, C, 3*H, W)

        ## prepare action cond data
        ## get the cartesian pose and gripper action
        # cartesian_pose = self._annotation_array(
        #     label,
        #     ("action.cartesian_pose", "action.cartesian_position"),
        # )[state_id]
        cartesian_pose = self._annotation_array(
            label,
            ("observation.state.cartesian_position",),
        )[state_id]
        ee_pose_dims = int(cartesian_pose.shape[-1])
        raw_full_action_seq = self._annotation_array(
            label,
            ("action.hand_joint_position", "action.hand_joints"),
        )
        swap_abd_with_mcp = bool(getattr(self.args, "swap_abd_with_mcp", False))
        full_action_seq = self._maybe_swap_abd_with_mcp(
            raw_full_action_seq,
            swap_abd_with_mcp,
        )
        full_action_seq = self._shift_action_sequence(full_action_seq)
        gripper_action = full_action_seq[state_id]

        action = self._normalize_action_from_bounds(
            cartesian_pose, gripper_action, state_p01, state_p99, ee_pose_dims
        )
        data['action'] = torch.tensor(action).float()

        for model_key, norm_all in self.model_norm_all.items():
            model_swap_abd_with_mcp = bool(
                getattr(self.args, f"{model_key}_swap_abd_with_mcp", swap_abd_with_mcp)
            )
            model_full_action_seq = self._maybe_swap_abd_with_mcp(
                raw_full_action_seq,
                model_swap_abd_with_mcp,
            )
            model_full_action_seq = self._shift_action_sequence(model_full_action_seq)
            model_gripper_action = model_full_action_seq[state_id]
            model_state_p01, model_state_p99 = norm_all[dataset_id]
            model_action = self._normalize_action_from_bounds(
                cartesian_pose,
                model_gripper_action,
                model_state_p01,
                model_state_p99,
                ee_pose_dims,
            )
            data[f"action_{model_key}"] = torch.tensor(model_action).float()

        ## return the frame ids if needed
        if return_frame_ids:
            return data, frame_ids

        ## return the data
        return data
