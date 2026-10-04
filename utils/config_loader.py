"""
Configuration loading utilities with support for variable interpolation.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
from typing import Any, Dict, Optional, Tuple, List
from omegaconf import OmegaConf, DictConfig
from config import wm_orca_args
import re


def load_experiment_config(config_path: str, base_args: Optional[wm_orca_args] = None) -> wm_orca_args:
    """
    Load experiment configuration from YAML with variable interpolation support.
    Automatically detects and merges with base config if specified in the experiment config.
    
    Args:
        config_path: Path to experiment YAML config
        base_args: Base config dataclass (default: wm_orca_args())
    
    Returns:
        Updated config dataclass with interpolated values
    """
    if base_args is None:
        base_args = wm_orca_args()
    
    # Load YAML config with OmegaConf for interpolation support
    omega_config = OmegaConf.load(config_path)
    
    # Check if experiment specifies a base config
    if 'experiment' in omega_config and 'base_config' in omega_config.experiment:
        base_config_path = omega_config.experiment.base_config
        # Resolve relative path
        if not os.path.isabs(base_config_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            base_config_path = os.path.normpath(os.path.join(config_dir, base_config_path))
        
        if os.path.exists(base_config_path):
            print(f"📂 Loading base config: {base_config_path}")
            base_omega = OmegaConf.load(base_config_path)
            # Merge: base config first, then experiment config (experiment overrides)
            omega_config = OmegaConf.merge(base_omega, omega_config)
        else:
            # Silently continuing would build the model from dataclass defaults instead.
            raise FileNotFoundError(
                f"Base config '{omega_config.experiment.base_config}' referenced by {config_path} "
                f"not found (resolved to {base_config_path})"
            )
    
    # Resolve all interpolations (${...} references)
    OmegaConf.resolve(omega_config)
    
    # Convert to plain dict
    exp_config = OmegaConf.to_container(omega_config, resolve=True)
    
    # Flatten nested config structure
    flat_config = _flatten_config(exp_config)
    
    # Update base_args with values from YAML
    for key, value in flat_config.items():
        if hasattr(base_args, key):
            setattr(base_args, key, value)
        else:
            print(f"Warning: Config key '{key}' not found in base config, skipping...")
    
    # Compute derived parameters
    _compute_derived_params(base_args)
    
    return base_args


def _flatten_config(config: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary structure.
    
    Args:
        config: Nested configuration dictionary
        parent_key: Parent key for recursion
        sep: Separator for nested keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and not _is_special_dict(k):
            items.extend(_flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((k, v))
    return dict(items)


def _is_special_dict(key: str) -> bool:
    """Check if a key represents a special dictionary that shouldn't be flattened."""
    special_keys = ['gripper_max_dict', 'model_paths', 'exclude_episode_ids_by_dataset']
    return key in special_keys


def _compute_derived_params(args: wm_orca_args):
    """Compute derived parameters based on config values."""
    # Compute down_sample from fps
    if hasattr(args, 'original_fps') and hasattr(args, 'fps'):
        args.down_sample = int(args.original_fps / args.fps)
    
    # Ensure dataset_cfgs matches dataset_names if not explicitly set
    if hasattr(args, 'dataset_names') and not hasattr(args, 'dataset_cfgs'):
        args.dataset_cfgs = args.dataset_names
    
    # Update output paths based on tag
    if hasattr(args, 'tag'):
        if not args.output_dir or args.output_dir == f"model_ckpt/{args.tag}":
            args.output_dir = f"model_ckpt/{args.tag}"
    
    # Convert dtype string to torch dtype
    if hasattr(args, 'dtype') and isinstance(args.dtype, str):
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }
        args.dtype = dtype_map.get(args.dtype.lower(), torch.bfloat16)


def save_config_to_yaml(args: wm_orca_args, save_path: str):
    """
    Save current config to YAML for reproducibility.
    
    Args:
        args: Configuration dataclass
        save_path: Path to save YAML file
    """
    config_dict = {}
    
    # Iterate through all instance attributes (including class variables)
    for attr_name in dir(args):
        # Skip private/magic methods and callables
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
                value = dtype_str_map.get(value, 'bfloat16')
            
            config_dict[attr_name] = value
        except Exception:
            # Skip attributes that can't be accessed or serialized
            continue
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Config saved to {save_path}")


def list_available_experiments(experiments_dir: str = "experiments") -> list:
    """
    List all available experiment configs.
    
    Args:
        experiments_dir: Directory containing experiment configs
    
    Returns:
        List of experiment metadata dictionaries
    """
    if not os.path.exists(experiments_dir):
        print(f"Experiments directory not found: {experiments_dir}")
        return []
    
    configs = []
    for root, dirs, files in os.walk(experiments_dir):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                config_path = os.path.join(root, file)
                try:
                    omega_config = OmegaConf.load(config_path)
                    exp_info = {
                        'file': os.path.relpath(config_path, experiments_dir),
                        'full_path': config_path,
                        'name': file.replace('.yaml', '').replace('.yml', ''),
                        'description': 'N/A'
                    }
                    
                    # Try to extract experiment metadata
                    if 'experiment' in omega_config:
                        exp_info['name'] = omega_config.experiment.get('name', exp_info['name'])
                        exp_info['description'] = omega_config.experiment.get('description', 'N/A')
                    
                    configs.append(exp_info)
                except Exception as e:
                    print(f"Warning: Failed to load {config_path}: {e}")
    
    return configs


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dictionaries
    
    Returns:
        Merged configuration dictionary
    """
    merged = OmegaConf.create({})
    for config in configs:
        if isinstance(config, (dict, DictConfig)):
            merged = OmegaConf.merge(merged, config)
    return merged


def find_experiment_config_by_name(experiments_dir: str, experiment_name: str) -> Optional[str]:
    """
    Recursively search all *.yaml/*.yml under experiments_dir (including nested subfolders) and return:
    - an exact match on `experiment.experiment_name` if found
    - otherwise the best match (highest similarity) across:
      `experiment.experiment_name`, `training.tag`, and the YAML filename stem.

    This mirrors how experiment YAMLs are structured in this repo (see `experiments/mask2real/`).
    """
    if not os.path.exists(experiments_dir):
        return None
    if OmegaConf is None:
        raise ImportError("OmegaConf is required to scan experiment YAMLs. Please install omegaconf.")

    import difflib

    target = str(experiment_name).strip().lower()
    best_path: Optional[str] = None
    best_score: float = -1.0

    for root, _, files in os.walk(experiments_dir):
        for f in files:
            if not (f.endswith(".yaml") or f.endswith(".yml")):
                continue
            path = os.path.join(root, f)
            try:
                cfg = OmegaConf.load(path)
                if "experiment" not in cfg:
                    continue
                exp = cfg.experiment
                exp_name = str(exp.get("experiment_name", "")).strip()
                train_tag = ""
                try:
                    if "training" in cfg:
                        train_tag = str(cfg.training.get("tag", "")).strip()
                except Exception:
                    train_tag = ""

                # Exact match first
                if exp_name and exp_name.strip().lower() == target:
                    return path

                # Otherwise score and keep best match
                candidates = [
                    exp_name,
                    train_tag,
                    os.path.splitext(os.path.basename(f))[0],
                ]
                for cand in candidates:
                    cand_norm = str(cand).strip().lower()
                    if not cand_norm:
                        continue
                    score = difflib.SequenceMatcher(a=target, b=cand_norm).ratio()
                    # small bonuses for containment (helps tags that are prefixes/suffixes)
                    if target in cand_norm or cand_norm in target:
                        score += 0.15
                    if score > best_score:
                        best_score = score
                        best_path = path
            except Exception:
                # Skip unreadable configs
                continue

    # Require a minimal similarity so we don't return an unrelated config
    if best_path is not None and best_score >= 0.45:
        return best_path
    return None


def load_args_from_experiment_name(
    experiments_dir: str,
    experiment_name: str,
    base_args: Optional[wm_orca_args] = None,
) -> Tuple[wm_orca_args, str]:
    """
    Find experiment YAML by `experiment.experiment_name` and load it into wm_orca_args
    using the same loader as training (`utils.config_loader.load_experiment_config`).
    Returns (args, config_path).
    """
    config_path = find_experiment_config_by_name(experiments_dir, experiment_name)
    if config_path is None:
        raise FileNotFoundError(
            f"Could not find an experiment YAML with experiment.experiment_name='{experiment_name}' under '{experiments_dir}'."
        )
    args = load_experiment_config(config_path, base_args or wm_orca_args())
    return args, config_path


def resolve_ckpt_tag_and_args(
    name_or_tag: str,
    model_ckpt_root: str,
    dataset_root: str,
    experiments_dir: Optional[str] = None,
) -> Tuple[str, wm_orca_args, Optional[str]]:
    """
    Resolve a user-provided identifier into:
    - `tag`: the folder name under model_ckpt_root
    - `args`: fully-populated wm_orca_args (from YAML if found, else heuristic defaults)
    - `config_path`: the YAML path if resolved via experiments_dir, else None

    Resolution order:
    - If model_ckpt_root/name_or_tag exists as a directory, treat it as training.tag
    - Else if experiments_dir provided, treat name_or_tag as experiment.experiment_name and load YAML,
      then use args.tag (training.tag) to locate the checkpoint folder.
    """
    config_path = None

    args, config_path = load_args_from_experiment_name(experiments_dir, name_or_tag, wm_orca_args())
    tag = getattr(args, "tag", None) or name_or_tag
    # In YAMLs, training.tag should map to args.tag
    ckpt_dir2 = os.path.join(model_ckpt_root, tag)
    if not os.path.isdir(ckpt_dir2):
        raise FileNotFoundError(
            f"Resolved experiment_name='{name_or_tag}' to tag='{tag}', but checkpoint folder not found: '{ckpt_dir2}'."
        )
    return tag, args, config_path


def select_checkpoints(ckpt_dir: str, max_ckpts: int = 3, max_iteration: int = None) -> List[str]:
    # Expect files like checkpoint-<step>.pt
    pts = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("checkpoint-") and f.endswith(".pt"):
            m = re.search(r"checkpoint-(\d+)\.pt$", f)
            if m:
                step = int(m.group(1))
                pts.append((step, os.path.join(ckpt_dir, f)))
    if not pts:
        return []
    pts.sort(key=lambda x: x[0])
    
    if max_iteration is not None:
        pts = [p for p in pts if p[0] <= max_iteration]
    if len(pts) <= max_ckpts:
        return [p for _, p in pts]
    # pick early, mid, late
    idxs = [0, len(pts)//2, len(pts)-1]
    sel = [pts[i][1] for i in idxs[-max_ckpts:]]
    # deduplicate in edge cases
    return list(dict.fromkeys(sel).keys()) if isinstance(sel, dict) else sel


def make_args_for_tag(
    tag: str,
    dataset_root: str,
    dataset_meta_info_path: str = "dataset_meta_info",
) -> wm_orca_args:
    args = wm_orca_args()
    # Heuristic: dataset name often equals the tag prefix up to fps or descriptors
    # If a dataset with the exact tag exists, use that; else try to find a dataset folder that matches a prefix.
    args.dataset_root_path = dataset_root
    datasets_available = set(os.listdir(dataset_root)) if os.path.exists(dataset_root) else set()
    if tag in datasets_available:
        args.dataset_names = tag
    else:
        # try to find a best match (longest common prefix)
        best_match = None
        best_len = -1
        for d in datasets_available:
            common = os.path.commonprefix([tag, d])
            if len(common) > best_len:
                best_len = len(common)
                best_match = d
        if best_match:
            args.dataset_names = best_match
    args.dataset_meta_info_path = dataset_meta_info_path
    args.tag = tag
    args.output_dir = f"model_ckpt/{tag}"
    args.wandb_project_name = "wm_benchmark"
    # Default to 1 view, 256x256 unless specific experiments changed that
    return args


if __name__ == "__main__":
    # Example usage
    config = load_experiment_config("experiments/mask2real/wm2.yaml")
    assert config != wm_orca_args()

    # list all experiments
    experiments = list_available_experiments()
    for exp in experiments:
        print(f"Experiment: {exp['name']}, Description: {exp['description']}, File: {exp['file']}")

    # save config to yaml
    save_config_to_yaml(config, "output/saved_config.yaml")