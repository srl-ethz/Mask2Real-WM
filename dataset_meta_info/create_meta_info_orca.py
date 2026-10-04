import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
import random
import imageio
from decord import VideoReader, cpu
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
# from finetune.constants import LOG_LEVEL, LOG_NAME
import numpy as np
from scipy.spatial.transform import Rotation as R  


SEGMENTATION_COLOR_MAP = {
    'red': np.array([255, 0, 0], dtype=np.float32),
    'green': np.array([0, 255, 0], dtype=np.float32),
    'gred': np.array([0, 255, 0], dtype=np.float32),
    'blue': np.array([0, 0, 255], dtype=np.float32),
    'object': np.array([255, 0, 0], dtype=np.float32),
    'hand': np.array([0, 255, 0], dtype=np.float32),
}


def _annotation_array(ann: Dict[str, Any], keys: Tuple[str, ...], ann_file: str) -> np.ndarray:
    for key in keys:
        if key in ann:
            return np.array(ann[key])
    raise KeyError(f'{ann_file} is missing required annotation field. Tried keys: {keys}')


def _shift_hand_actions(hand_actions: np.ndarray, action_down_sample: int) -> np.ndarray:
    shift_steps = max(1, int(action_down_sample))
    num_actions = hand_actions.shape[0]
    effective_shift = min(shift_steps, num_actions)
    if effective_shift == num_actions:
        return np.zeros_like(hand_actions)

    pad_shape = (effective_shift, *hand_actions.shape[1:])
    return np.concatenate(
        (
            np.zeros(pad_shape, dtype=hand_actions.dtype),
            hand_actions[:-effective_shift],
        ),
        axis=0,
    )


def build_training_action(ann: Dict[str, Any], ann_file: str, action_down_sample: int) -> np.ndarray:
    cartesian_pose = _annotation_array(
        ann,
        ('action.cartesian_pose', 'action.cartesian_position', 'action.cartesian_relative_pose'),
        ann_file,
    )
    hand_actions = _annotation_array(
        ann,
        ('action.hand_joint_position', 'action.hand_joints'),
        ann_file,
    )

    if cartesian_pose.ndim != 2:
        raise ValueError(f'{ann_file} cartesian action must be 2D, got {cartesian_pose.shape}')
    if hand_actions.ndim != 2:
        raise ValueError(f'{ann_file} action.hand_joint_position must be 2D, got {hand_actions.shape}')
    if cartesian_pose.shape[0] != hand_actions.shape[0]:
        raise ValueError(
            f'{ann_file} action length mismatch: '
            f'cartesian={cartesian_pose.shape[0]}, hand={hand_actions.shape[0]}'
        )

    shifted_hand_actions = _shift_hand_actions(hand_actions, action_down_sample)
    return np.concatenate((cartesian_pose, shifted_hand_actions), axis=-1)




def _parse_segmentation_colors(color_names: str) -> List[np.ndarray]:
    colors = []
    for raw_name in color_names.split(','):
        name = raw_name.strip().lower()
        if not name:
            continue
        if name not in SEGMENTATION_COLOR_MAP:
            raise ValueError(
                f'Unknown segmentation color "{raw_name}". '
                f'Known colors: {sorted(SEGMENTATION_COLOR_MAP)}'
            )
        colors.append(SEGMENTATION_COLOR_MAP[name])
    if len(colors) == 0:
        raise ValueError('At least one segmentation color is required.')
    return colors


def _resolve_segmentation_video_path(
    data_root: str,
    ann: Dict[str, Any],
    ann_file: str,
    segmentation_view_idx: int,
    missing_view_policy: str,
) -> Optional[str]:
    segmentation_videos = ann.get('segmentation_videos')
    if not segmentation_videos:
        raise KeyError(f'{ann_file} is missing segmentation_videos.')

    if segmentation_view_idx < len(segmentation_videos):
        view_idx = segmentation_view_idx
    elif missing_view_policy == 'last':
        view_idx = len(segmentation_videos) - 1
        print(
            f'warning: {ann_file} has {len(segmentation_videos)} segmentation views; '
            f'using last available view index {view_idx} instead of requested '
            f'{segmentation_view_idx}.',
            flush=True,
        )
    elif missing_view_policy == 'skip':
        print(
            f'warning: skipping {ann_file}; requested segmentation view '
            f'{segmentation_view_idx}, available views={len(segmentation_videos)}.',
            flush=True,
        )
        return None
    else:
        raise IndexError(
            f'{ann_file} requested segmentation view {segmentation_view_idx}, '
            f'but only {len(segmentation_videos)} views are available.'
        )

    video_path = segmentation_videos[view_idx].get('video_path')
    if video_path is None:
        raise KeyError(f'{ann_file} segmentation_videos[{view_idx}] is missing video_path.')
    return str(Path(data_root) / video_path)


def _frames_with_required_segmentation_colors(
    video_path: str,
    max_frame: int,
    color_names: str,
    color_threshold: float,
    min_pixels: int,
) -> set:
    colors = _parse_segmentation_colors(color_names)
    reader = VideoReader(video_path, ctx=cpu(0))
    num_frames = min(len(reader), int(max_frame))
    if num_frames <= 0:
        return set()

    frames = reader.get_batch(list(range(num_frames))).asnumpy().astype(np.float32)
    present_masks = []
    for color in colors:
        color_mask = np.linalg.norm(frames - color, axis=-1) <= color_threshold
        present_masks.append(color_mask.reshape(num_frames, -1).sum(axis=1) >= int(min_pixels))

    keep = np.logical_and.reduce(present_masks)
    return set(np.flatnonzero(keep).astype(int).tolist())

def load_and_process_ann_file(
    data_root,
    ann_file,
    sequence_interval=1,
    start_interval=4,
    sequence_length=8,
    samples_per_traj=None,
    action_down_sample=1,
    segmentation_start_filter=False,
    segmentation_view_idx=2,
    segmentation_color_names='red,green',
    segmentation_color_threshold=15.0,
    segmentation_min_pixels=1,
    segmentation_missing_view='error',
):
    samples = []
    try:
        with open(f'{data_root}/{ann_file}', "r") as f:
            ann = json.load(f)
    except Exception:
        print(f'skip {ann_file}', flush=True)
        return samples

    action = build_training_action(ann, ann_file, action_down_sample)
    n_frames = min(int(ann['video_length']), action.shape[0])
    if n_frames < int(ann['video_length']):
        print(
            f'warning: {ann_file} video_length={ann["video_length"]} exceeds '
            f'action_length={action.shape[0]}; truncating sample generation.',
            flush=True,
        )
    traj_len = int(sequence_length*sequence_interval)
    end_idx = n_frames - int(traj_len*0.5)
    if end_idx < 1:
        end_idx = 1

    if samples_per_traj is not None:
        n = min(samples_per_traj, end_idx)
        start_frames = np.linspace(0, end_idx - 1, n, dtype=int).tolist()
        seen = set()
        start_frames = [x for x in start_frames if not (x in seen or seen.add(x))]
    else:
        start_frames = list(range(0, end_idx, start_interval))

    if segmentation_start_filter:
        segmentation_video_path = _resolve_segmentation_video_path(
            data_root=data_root,
            ann=ann,
            ann_file=ann_file,
            segmentation_view_idx=segmentation_view_idx,
            missing_view_policy=segmentation_missing_view,
        )
        if segmentation_video_path is None:
            return samples
        valid_start_frames = _frames_with_required_segmentation_colors(
            video_path=segmentation_video_path,
            max_frame=n_frames,
            color_names=segmentation_color_names,
            color_threshold=segmentation_color_threshold,
            min_pixels=segmentation_min_pixels,
        )
        start_frames = [idx for idx in start_frames if idx in valid_start_frames]

    for start_frame in start_frames:
        idx = start_frame
        sample = dict()
        sample['ann_file'] = ann_file
        sample['episode_id'] = ann['episode_id']
        sample['frame_ids'] = [idx]
        sample['states'] = action[idx:idx+1]
        samples.append(sample)
    return samples

def init_anns(dataset_root, data_dir):
    final_path = f'{dataset_root}/{data_dir}'
    ann_files = [os.path.join(data_dir, f) for f in os.listdir(final_path) if f.endswith('.json')]
    return ann_files


def _normalize_motion_type(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def load_ann_summary(
    data_root: str,
    ann_file: str,
    motion_type_key: str = 'data_motion_type',
) -> Optional[Dict[str, Any]]:
    try:
        with open(f'{data_root}/{ann_file}', "r") as f:
            ann = json.load(f)
    except Exception:
        print(f'skip {ann_file}', flush=True)
        return None

    if 'video_length' not in ann:
        print(f'skip {ann_file}: missing video_length', flush=True)
        return None

    return {
        'ann_file': ann_file,
        'episode_id': ann.get('episode_id', Path(ann_file).stem),
        'video_length': int(ann['video_length']),
        'motion_type': _normalize_motion_type(ann[motion_type_key]) if motion_type_key in ann else None,
    }


def _split_into_length_bins(ann_infos: List[Dict[str, Any]], num_bins: int) -> List[List[Dict[str, Any]]]:
    if len(ann_infos) == 0:
        return []

    num_bins = max(1, min(int(num_bins), len(ann_infos)))
    sorted_infos = sorted(ann_infos, key=lambda x: x['video_length'])
    return [list(chunk) for chunk in np.array_split(sorted_infos, num_bins) if len(chunk) > 0]


def _select_length_diverse_infos(
    ann_infos: List[Dict[str, Any]],
    num_to_select: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if num_to_select <= 0:
        return []
    if num_to_select >= len(ann_infos):
        return list(ann_infos)

    selected = []
    selected_files = set()
    for bin_infos in _split_into_length_bins(ann_infos, num_to_select):
        candidates = list(bin_infos)
        rng.shuffle(candidates)
        target_length = sum(x['video_length'] for x in candidates) / len(candidates)
        best = min(candidates, key=lambda x: abs(x['video_length'] - target_length))
        selected.append(best)
        selected_files.add(best['ann_file'])

    if len(selected) < num_to_select:
        remaining = [x for x in ann_infos if x['ann_file'] not in selected_files]
        rng.shuffle(remaining)
        selected.extend(remaining[:num_to_select - len(selected)])

    return selected


def _split_ann_files_by_motion_type(
    ann_infos: List[Dict[str, Any]],
    val_frame_ratio: float,
    seed: int,
    min_val_demos: int,
) -> Tuple[set, bool]:
    motion_type_presence = [x['motion_type'] is not None for x in ann_infos]
    if not any(motion_type_presence):
        print(
            'warning: no annotation files contain data_motion_type; '
            'falling back to trajectory-length validation split.',
            flush=True,
        )
        return set(), False

    if not all(motion_type_presence):
        missing_files = [x['ann_file'] for x in ann_infos if x['motion_type'] is None]
        preview = ', '.join(missing_files[:5])
        if len(missing_files) > 5:
            preview += f', ... ({len(missing_files)} total)'
        raise ValueError(f'Missing data_motion_type in some annotation files: {preview}')

    rng = random.Random(seed)
    infos_by_type = defaultdict(list)
    for ann_info in ann_infos:
        infos_by_type[ann_info['motion_type']].append(ann_info)

    motion_types = sorted(infos_by_type)
    max_equal_val_per_type = min(len(infos) for infos in infos_by_type.values())
    if max_equal_val_per_type == 0:
        return set(), True

    target_val_demos = int(round(len(ann_infos) * val_frame_ratio))
    target_val_demos = max(1, target_val_demos)
    target_val_demos = max(target_val_demos, int(min_val_demos))
    target_val_per_type = max(1, int(np.ceil(target_val_demos / len(motion_types))))
    val_per_type = min(max_equal_val_per_type, target_val_per_type)

    val_infos = []
    for motion_type in motion_types:
        val_infos.extend(
            _select_length_diverse_infos(
                infos_by_type[motion_type],
                val_per_type,
                rng,
            )
        )

    val_ann_files = {x['ann_file'] for x in val_infos}
    if len(val_ann_files) == len(ann_infos) and len(ann_infos) > 1 and val_per_type > 1:
        val_per_type -= 1
        val_infos = []
        for motion_type in motion_types:
            val_infos.extend(
                _select_length_diverse_infos(
                    infos_by_type[motion_type],
                    val_per_type,
                    rng,
                )
            )
        val_ann_files = {x['ann_file'] for x in val_infos}

    val_counts = Counter(x['motion_type'] for x in ann_infos if x['ann_file'] in val_ann_files)
    train_counts = Counter(x['motion_type'] for x in ann_infos if x['ann_file'] not in val_ann_files)
    print(
        'motion-type validation split: '
        f'val_per_type={val_per_type}, '
        f'train_motion_counts={dict(sorted(train_counts.items()))}, '
        f'val_motion_counts={dict(sorted(val_counts.items()))}',
        flush=True,
    )

    return val_ann_files, True


def split_ann_files_by_frame_ratio(
    data_root: str,
    ann_files: List[str],
    val_frame_ratio: float,
    seed: int,
    min_val_demos: int = 5,
    length_bins: int = 3,
    motion_type_key: str = 'data_motion_type',
) -> Tuple[List[str], List[str]]:
    if not 0.0 <= val_frame_ratio < 1.0:
        raise ValueError('--val_frame_ratio must be between 0 (inclusive) and 1 (exclusive).')

    ann_infos = []
    for ann_file in ann_files:
        ann_info = load_ann_summary(data_root, ann_file, motion_type_key=motion_type_key)
        if ann_info is not None:
            ann_infos.append(ann_info)

    if len(ann_infos) == 0:
        raise ValueError('No valid annotation files found for splitting.')

    if val_frame_ratio == 0.0:
        return [x['ann_file'] for x in ann_infos], []

    rng = random.Random(seed)
    total_frames = sum(x['video_length'] for x in ann_infos)
    target_val_frames = max(1, int(round(total_frames * val_frame_ratio)))
    max_val_demos = max(1, len(ann_infos) - 1) if len(ann_infos) > 1 else 1
    min_val_demos = min(max(1, int(min_val_demos)), max_val_demos)

    val_ann_files, used_motion_type_split = _split_ann_files_by_motion_type(
        ann_infos=ann_infos,
        val_frame_ratio=val_frame_ratio,
        seed=seed,
        min_val_demos=min_val_demos,
    )
    bins = _split_into_length_bins(ann_infos, length_bins)

    # Pick validation demos inside each length bin so validation covers short, medium,
    # and long trajectories while targeting frame ratio rather than demo count.
    if not used_motion_type_split:
        for bin_infos in bins:
            bin_target = sum(x['video_length'] for x in bin_infos) * val_frame_ratio
            if bin_target <= 0:
                continue

            candidates = list(bin_infos)
            rng.shuffle(candidates)
            bin_val_frames = 0
            while candidates and bin_val_frames < bin_target and len(val_ann_files) < max_val_demos:
                best = min(
                    candidates,
                    key=lambda x: abs((bin_val_frames + x['video_length']) - bin_target),
                )
                val_ann_files.add(best['ann_file'])
                bin_val_frames += best['video_length']
                candidates.remove(best)

    def current_val_frames() -> int:
        return sum(x['video_length'] for x in ann_infos if x['ann_file'] in val_ann_files)

    if not used_motion_type_split:
        remaining = [x for x in ann_infos if x['ann_file'] not in val_ann_files]
        rng.shuffle(remaining)
        while len(val_ann_files) < min_val_demos and remaining and len(val_ann_files) < max_val_demos:
            val_frames = current_val_frames()
            best = min(
                remaining,
                key=lambda x: abs((val_frames + x['video_length']) - target_val_frames),
            )
            val_ann_files.add(best['ann_file'])
            remaining.remove(best)

        remaining = [x for x in ann_infos if x['ann_file'] not in val_ann_files]
        while current_val_frames() < target_val_frames and remaining and len(val_ann_files) < max_val_demos:
            val_frames = current_val_frames()
            best = min(
                remaining,
                key=lambda x: abs((val_frames + x['video_length']) - target_val_frames),
            )
            val_ann_files.add(best['ann_file'])
            remaining.remove(best)

    if not used_motion_type_split and len(val_ann_files) == len(ann_infos) and len(ann_infos) > 1:
        largest_val = max(
            (x for x in ann_infos if x['ann_file'] in val_ann_files),
            key=lambda x: x['video_length'],
        )
        val_ann_files.remove(largest_val['ann_file'])

    train_ann_files = [x['ann_file'] for x in ann_infos if x['ann_file'] not in val_ann_files]
    val_ann_files = [x['ann_file'] for x in ann_infos if x['ann_file'] in val_ann_files]

    train_frames = sum(x['video_length'] for x in ann_infos if x['ann_file'] in train_ann_files)
    val_frames = sum(x['video_length'] for x in ann_infos if x['ann_file'] in val_ann_files)
    print(
        'demo-frame split: '
        f'train_demos={len(train_ann_files)}, val_demos={len(val_ann_files)}, '
        f'train_frames={train_frames}, val_frames={val_frames}, '
        f'val_frame_ratio={val_frames / max(1, total_frames):.4f}, '
        f'target_val_frame_ratio={val_frame_ratio:.4f}',
        flush=True,
    )

    return train_ann_files, val_ann_files



def parse_val_episode_groups(groups_json: Optional[str]) -> Optional[Dict[str, List[int]]]:
    if groups_json is None:
        return None
    groups = json.loads(groups_json)
    if not isinstance(groups, dict) or len(groups) == 0:
        raise ValueError('--val_episode_groups_json must be a non-empty JSON object.')

    parsed = {}
    for category, episode_ids in groups.items():
        if not isinstance(episode_ids, list) or len(episode_ids) == 0:
            raise ValueError(f'Validation group {category!r} must map to a non-empty list.')
        parsed[str(category)] = [int(episode_id) for episode_id in episode_ids]
    return parsed


def ann_files_by_episode_id(data_root: str, ann_files: List[str]) -> Dict[int, str]:
    by_episode_id = {}
    for ann_file in ann_files:
        with open(f'{data_root}/{ann_file}', 'r') as f:
            ann = json.load(f)
        episode_id = int(ann.get('episode_id', Path(ann_file).stem))
        if episode_id in by_episode_id:
            raise ValueError(
                f'Duplicate episode_id={episode_id}: '
                f'{by_episode_id[episode_id]} and {ann_file}'
            )
        by_episode_id[episode_id] = ann_file
    return by_episode_id


def sample_evenly_by_episode_and_frame(
    samples: List[Dict[str, Any]],
    num_samples: int,
) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError('--val_samples_per_group must be positive when custom validation groups are used.')
    ordered = sorted(samples, key=lambda x: (int(x['episode_id']), int(x['frame_ids'][0])))
    if len(ordered) <= num_samples:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, num_samples, dtype=int).tolist()
    seen = set()
    indices = [idx for idx in indices if not (idx in seen or seen.add(idx))]
    return [ordered[idx] for idx in indices]


def init_sequences(
    data_root,
    ann_files,
    sequence_interval,
    start_interval,
    sequence_length,
    samples_per_traj=None,
    action_down_sample=1,
    segmentation_start_filter=False,
    segmentation_view_idx=2,
    segmentation_color_names='red,green',
    segmentation_color_threshold=15.0,
    segmentation_min_pixels=1,
    segmentation_missing_view='error',
):
    samples = []
    with ThreadPoolExecutor(32) as executor:
        future_to_ann_file = {
            executor.submit(
                load_and_process_ann_file,
                data_root,
                ann_file,
                sequence_interval,
                start_interval,
                sequence_length,
                samples_per_traj,
                action_down_sample,
                segmentation_start_filter,
                segmentation_view_idx,
                segmentation_color_names,
                segmentation_color_threshold,
                segmentation_min_pixels,
                segmentation_missing_view,
            ): ann_file
            for ann_file in ann_files
        }
        for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
            samples.extend(future.result())
    return samples


def compute_state_stat(samples_all):
    state_all = []
    for sample in samples_all:
        state = np.array(sample['states'])
        if state.ndim == 1:
            state = state[None, :]
        state_all.append(state)

    if len(state_all) == 0:
        raise ValueError('Cannot compute action statistics from an empty sample set.')

    state_all = np.concatenate(state_all, axis=0)
    print('action_shape:', state_all.shape, flush=True)
    state_all = state_all.reshape(-1, state_all.shape[-1])
    state_01 = np.percentile(state_all, 1, axis=0)
    state_99 = np.percentile(state_all, 99, axis=0)
    print('state_01:', state_01, flush=True)
    print('state_99:', state_99, flush=True)
    return {
        'state_01': state_01.tolist(),
        'state_99': state_99.tolist(),
    }


def _episode_json_name(episode_id: Any) -> str:
    episode_name = str(episode_id)
    if not episode_name.endswith('.json'):
        episode_name = f'{episode_name}.json'
    return episode_name


def resolve_sample_ann_file(data_root: str, sample: Dict[str, Any], split_name: str, single_ann_dir: str) -> str:
    sample_ann_file = sample.get('ann_file')
    if sample_ann_file is not None:
        sample_ann_path = Path(sample_ann_file)
        if sample_ann_path.is_absolute() and sample_ann_path.exists():
            return str(sample_ann_path)
        candidate = Path(data_root) / sample_ann_file
        if candidate.exists():
            return str(candidate)

    episode_name = _episode_json_name(sample['episode_id'])
    candidates = [
        Path(data_root) / single_ann_dir / episode_name,
        Path(data_root) / 'annotation' / split_name / episode_name,
        Path(data_root) / 'annotation' / 'validation' / episode_name,
        Path(data_root) / 'annotation' / 'val' / episode_name,
        Path(data_root) / 'annotation' / 'train' / episode_name,
        Path(data_root) / 'annotation' / episode_name,
    ]
    for candidate in dict.fromkeys(candidates):
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f'Could not resolve annotation for episode_id={sample["episode_id"]}. '
        f'Tried: {[str(x) for x in dict.fromkeys(candidates)]}'
    )


def compute_state_stat_from_sample_files(
    data_root: str,
    samples_all: List[Dict[str, Any]],
    split_name: str,
    single_ann_dir: str,
    action_down_sample: int,
) -> Dict[str, List[float]]:
    action_cache: Dict[str, np.ndarray] = {}
    stat_samples = []
    for sample in tqdm(samples_all, desc=f'loading {split_name} actions'):
        ann_file = resolve_sample_ann_file(data_root, sample, split_name, single_ann_dir)
        if ann_file not in action_cache:
            with open(ann_file, 'r') as f:
                ann = json.load(f)
            action_cache[ann_file] = build_training_action(ann, ann_file, action_down_sample)

        frame_ids = np.array(sample['frame_ids'], dtype=np.int64)
        action = action_cache[ann_file]
        if np.any(frame_ids < 0) or np.any(frame_ids >= action.shape[0]):
            raise IndexError(
                f'{ann_file} has action length {action.shape[0]}, '
                f'but sample frame_ids={frame_ids.tolist()}'
            )
        stat_samples.append({'states': action[frame_ids]})

    return compute_state_stat(stat_samples)

def strip_states_and_shuffle(samples_all, rng):
    samples_without_states = []
    for sample in samples_all:
        sample_new = dict(sample)
        sample_new.pop('states', None)
        samples_without_states.append(sample_new)
    rng.shuffle(samples_without_states)
    return samples_without_states


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--orca_output_path', type=str, default='dataset_example/orca_dataset')
    # dataset_name
    parser.add_argument('--sequence_length', type=int, default=None, help="The sequence length should correspond to history + future predictions.")
    parser.add_argument('--dataset_name', type=str, default='orca_dataset')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional meta-info output directory. Defaults to dataset_meta_info/<parent>/<dataset_name>.')
    parser.add_argument('--stats_only', action='store_true', help='Only recompute stat.json from existing train/val sample files; do not rewrite sample files.')
    parser.add_argument('--action_down_sample', type=int, default=1, help='Matches dataset.down_sample used by dataset_orca.py when shifting hand actions.')
    parser.add_argument(
        '--split_mode',
        type=str,
        choices=['auto', 'pre_split', 'from_single_dir'],
        default='auto',
        help='auto: use train/val folders if they exist, otherwise split one annotation folder into train/val.',
    )
    parser.add_argument(
        '--single_ann_dir',
        type=str,
        default='annotation',
        help='Annotation directory (relative to --orca_output_path) for from_single_dir mode.',
    )
    parser.add_argument(
        '--split_strategy',
        type=str,
        choices=['demo_frame_balanced', 'sample'],
        default='demo_frame_balanced',
        help=(
            'Split strategy for from_single_dir mode. demo_frame_balanced keeps whole demos '
            'in one split and targets validation frame ratio across length bins; sample preserves '
            'the legacy sample-window split.'
        ),
    )
    parser.add_argument('--train_ratio', type=float, default=0.0, help='Train split ratio for sample split mode, or for deriving validation ratio when --val_frame_ratio is unset.')
    parser.add_argument('--val_frame_ratio', type=float, default=None, help='Target validation frame ratio for demo_frame_balanced mode. Defaults to 0.05 when train_ratio is 0, otherwise 1 - train_ratio.')
    parser.add_argument('--min_val_demos', type=int, default=5, help='Minimum number of validation demos for demo_frame_balanced mode, capped to leave at least one train demo.')
    parser.add_argument('--length_bins', type=int, default=3, help='Number of trajectory-length bins used to cover short/medium/long demos in validation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting/shuffling.')
    parser.add_argument('--samples_per_traj', type=int, default=None, help='Fixed number of samples per trajectory (evenly spaced). If not set, uses start_interval.')
    parser.add_argument('--val_episode_groups_json', type=str, default=None, help='JSON object mapping validation category names to episode-id lists. Listed episodes are excluded from train, and validation is sampled per group.')
    parser.add_argument('--val_samples_per_group', type=int, default=None, help='Number of validation samples to keep from each --val_episode_groups_json category.')
    parser.add_argument('--segmentation_start_filter', action='store_true', help='Keep only start frames whose selected segmentation view contains all requested colors.')
    parser.add_argument('--segmentation_view_idx', type=int, default=2, help='Zero-based segmentation view index used by --segmentation_start_filter. Default 2 means the third view.')
    parser.add_argument('--segmentation_color_names', type=str, default='red,green', help='Comma-separated color names required in the selected segmentation frame.')
    parser.add_argument('--segmentation_color_threshold', type=float, default=15.0, help='RGB Euclidean distance threshold for segmentation color matching.')
    parser.add_argument('--segmentation_min_pixels', type=int, default=1, help='Minimum number of pixels required for each requested color.')
    parser.add_argument(
        '--segmentation_missing_view',
        type=str,
        choices=['error', 'last', 'skip'],
        default='error',
        help='What to do when --segmentation_view_idx is not available in an annotation.',
    )
    args = parser.parse_args()
    
    if args.split_mode == 'auto':
        train_ann_dir = Path(args.orca_output_path) / 'annotation' / 'train'
        val_ann_dir = Path(args.orca_output_path) / 'annotation' / 'val'
        use_pre_split = train_ann_dir.exists() and val_ann_dir.exists()
    else:
        use_pre_split = args.split_mode == 'pre_split'

    if not 0.0 <= args.train_ratio < 1.0:
        raise ValueError('--train_ratio must be between 0 (inclusive) and 1 (exclusive).')
    if args.val_frame_ratio is not None and not 0.0 <= args.val_frame_ratio < 1.0:
        raise ValueError('--val_frame_ratio must be between 0 (inclusive) and 1 (exclusive).')
    if args.action_down_sample < 1:
        raise ValueError('--action_down_sample must be >= 1.')
    if args.segmentation_min_pixels < 1:
        raise ValueError('--segmentation_min_pixels must be >= 1.')
    if args.segmentation_view_idx < 0:
        raise ValueError('--segmentation_view_idx must be >= 0.')
    if args.val_episode_groups_json is not None and args.val_samples_per_group is None:
        raise ValueError('--val_samples_per_group is required with --val_episode_groups_json.')
    if args.val_samples_per_group is not None and args.val_samples_per_group < 1:
        raise ValueError('--val_samples_per_group must be >= 1.')
    if not args.stats_only and args.sequence_length is None:
        raise ValueError('--sequence_length is required unless --stats_only is set.')

    data_root = args.orca_output_path
    dataset_name = args.dataset_name
    sequence_interval = 1
    start_interval = 1
    rng = random.Random(args.seed)

    second_last_folder = os.path.basename(os.path.dirname(args.orca_output_path))
    output_dir = args.output_dir or f'dataset_meta_info/{second_last_folder}/{dataset_name}'

    split_samples = {}
    split_ann_counts = {}

    try:
        if args.stats_only:
            split_sample_paths = {
                'train': Path(output_dir) / 'train_sample.json',
                'val': Path(output_dir) / 'val_sample.json',
            }
            loaded_samples = {}
            for split_name, sample_path in split_sample_paths.items():
                if sample_path.exists():
                    with open(sample_path, 'r') as f:
                        loaded_samples[split_name] = json.load(f)
                else:
                    loaded_samples[split_name] = []

            stat_split = 'train' if len(loaded_samples['train']) > 0 else 'val'
            if len(loaded_samples[stat_split]) == 0:
                raise ValueError(f'No existing train/val samples found in {output_dir}.')

            print(
                f'stats_only: recomputing stat.json from {stat_split}_sample.json '
                f'using action.cartesian_pose + shifted action.hand_joint_position',
                flush=True,
            )
            stat = compute_state_stat_from_sample_files(
                data_root=data_root,
                samples_all=loaded_samples[stat_split],
                split_name=stat_split,
                single_ann_dir=args.single_ann_dir,
                action_down_sample=args.action_down_sample,
            )
            os.makedirs(output_dir, exist_ok=True)
            with open(f'{output_dir}/stat.json', 'w') as f:
                json.dump(stat, f)
            print(f'wrote {output_dir}/stat.json', flush=True)
            raise SystemExit(0)

        if use_pre_split:
            for data_type in ['train', 'val']:
                ann_dir = f'annotation/{data_type}'
                ann_files = init_anns(data_root, ann_dir)
                samples = init_sequences(
                    data_root,
                    ann_files,
                    sequence_interval,
                    start_interval,
                    args.sequence_length,
                    args.samples_per_traj,
                    args.action_down_sample,
                    args.segmentation_start_filter,
                    args.segmentation_view_idx,
                    args.segmentation_color_names,
                    args.segmentation_color_threshold,
                    args.segmentation_min_pixels,
                    args.segmentation_missing_view,
                )
                print(f'{data_root} {data_type}: {len(samples)} samples', flush=True)
                split_samples[data_type] = samples
                split_ann_counts[data_type] = len(ann_files)
        else:
            ann_files = init_anns(data_root, args.single_ann_dir)
            if len(ann_files) == 0:
                raise ValueError('No annotation files found in the provided annotation directory.')

            val_episode_groups = parse_val_episode_groups(args.val_episode_groups_json)
            if val_episode_groups is not None:
                ann_by_episode_id = ann_files_by_episode_id(data_root, ann_files)
                val_episode_ids = {
                    episode_id
                    for episode_ids in val_episode_groups.values()
                    for episode_id in episode_ids
                }
                missing_episode_ids = sorted(val_episode_ids - set(ann_by_episode_id))
                if missing_episode_ids:
                    raise ValueError(f'Validation episodes not found in annotations: {missing_episode_ids}')

                train_ann_files = [
                    ann_file
                    for episode_id, ann_file in sorted(ann_by_episode_id.items())
                    if episode_id not in val_episode_ids
                ]
                split_samples['train'] = init_sequences(
                    data_root,
                    train_ann_files,
                    sequence_interval,
                    start_interval,
                    args.sequence_length,
                    args.samples_per_traj,
                    args.action_down_sample,
                    args.segmentation_start_filter,
                    args.segmentation_view_idx,
                    args.segmentation_color_names,
                    args.segmentation_color_threshold,
                    args.segmentation_min_pixels,
                    args.segmentation_missing_view,
                )

                val_samples = []
                for category, episode_ids in val_episode_groups.items():
                    group_ann_files = [ann_by_episode_id[episode_id] for episode_id in episode_ids]
                    group_samples_all = init_sequences(
                        data_root,
                        group_ann_files,
                        sequence_interval,
                        start_interval,
                        args.sequence_length,
                        args.samples_per_traj,
                        args.action_down_sample,
                        args.segmentation_start_filter,
                        args.segmentation_view_idx,
                        args.segmentation_color_names,
                        args.segmentation_color_threshold,
                        args.segmentation_min_pixels,
                        args.segmentation_missing_view,
                    )
                    group_samples = sample_evenly_by_episode_and_frame(
                        group_samples_all,
                        args.val_samples_per_group,
                    )
                    if len(group_samples) < args.val_samples_per_group:
                        print(
                            f'warning: validation group {category!r} requested '
                            f'{args.val_samples_per_group} samples but only '
                            f'{len(group_samples)} were available.',
                            flush=True,
                        )
                    print(
                        f'custom validation group {category}: '
                        f'episodes={episode_ids}, candidates={len(group_samples_all)}, '
                        f'selected={len(group_samples)}',
                        flush=True,
                    )
                    val_samples.extend(group_samples)

                split_samples['val'] = val_samples
                split_ann_counts['train'] = len(train_ann_files)
                split_ann_counts['val'] = len(val_episode_ids)
            elif args.split_strategy == 'demo_frame_balanced':
                val_frame_ratio = (
                    args.val_frame_ratio
                    if args.val_frame_ratio is not None
                    else (1.0 - args.train_ratio if args.train_ratio > 0.0 else 0.05)
                )
                train_ann_files, val_ann_files = split_ann_files_by_frame_ratio(
                    data_root=data_root,
                    ann_files=ann_files,
                    val_frame_ratio=val_frame_ratio,
                    seed=args.seed,
                    min_val_demos=args.min_val_demos,
                    length_bins=args.length_bins,
                    motion_type_key='data_motion_type',
                )
                split_samples['train'] = init_sequences(
                    data_root,
                    train_ann_files,
                    sequence_interval,
                    start_interval,
                    args.sequence_length,
                    args.samples_per_traj,
                    args.action_down_sample,
                    args.segmentation_start_filter,
                    args.segmentation_view_idx,
                    args.segmentation_color_names,
                    args.segmentation_color_threshold,
                    args.segmentation_min_pixels,
                    args.segmentation_missing_view,
                )
                split_samples['val'] = init_sequences(
                    data_root,
                    val_ann_files,
                    sequence_interval,
                    start_interval,
                    args.sequence_length,
                    args.samples_per_traj,
                    args.action_down_sample,
                    args.segmentation_start_filter,
                    args.segmentation_view_idx,
                    args.segmentation_color_names,
                    args.segmentation_color_threshold,
                    args.segmentation_min_pixels,
                    args.segmentation_missing_view,
                )
                split_ann_counts['train'] = len(train_ann_files)
                split_ann_counts['val'] = len(val_ann_files)
            else:
                all_samples = init_sequences(
                    data_root,
                    ann_files,
                    sequence_interval,
                    start_interval,
                    args.sequence_length,
                    args.samples_per_traj,
                    args.action_down_sample,
                    args.segmentation_start_filter,
                    args.segmentation_view_idx,
                    args.segmentation_color_names,
                    args.segmentation_color_threshold,
                    args.segmentation_min_pixels,
                    args.segmentation_missing_view,
                )
                print(f'{data_root} all: {len(all_samples)} samples', flush=True)

                if len(all_samples) == 0:
                    raise ValueError('No samples found in the provided annotation directory.')

                rng.shuffle(all_samples)
                split_idx = int(len(all_samples) * args.train_ratio)
                if len(all_samples) > 1 and 0.0 < args.train_ratio < 1.0:
                    split_idx = max(1, min(split_idx, len(all_samples) - 1))

                split_samples['train'] = all_samples[:split_idx]
                split_samples['val'] = all_samples[split_idx:]
                split_ann_counts['train'] = len(set(sample['episode_id'] for sample in split_samples['train']))
                split_ann_counts['val'] = len(set(sample['episode_id'] for sample in split_samples['val']))

            print(
                f'split from {args.single_ann_dir}: '
                f'train={len(split_samples["train"])}, val={len(split_samples["val"])}, '
                f'train_traj={split_ann_counts["train"]}, val_traj={split_ann_counts["val"]}',
                flush=True,
            )

        # Compute statistics from training split (or fall back to val if train is empty).
        print("########################### action ###########################")
        stat_source = split_samples['train'] if len(split_samples.get('train', [])) > 0 else split_samples['val']
        stat = compute_state_stat(stat_source)

        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/stat.json', 'w') as f:
            json.dump(stat, f)

        for data_type in ['train', 'val']:
            data_samples = strip_states_and_shuffle(split_samples.get(data_type, []), rng)
            print('step_num', data_type, len(data_samples), flush=True)
            print('traj_num', data_type, split_ann_counts.get(data_type, 0), flush=True)
            with open(f'{output_dir}/{data_type}_sample.json', 'w') as f:
                json.dump(data_samples, f, indent=4)
    except Exception as e:
        print(f"An error occurred while processing dataset: {e}", flush=True)
