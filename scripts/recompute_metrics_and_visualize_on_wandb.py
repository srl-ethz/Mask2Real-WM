#!/usr/bin/env python3
"""
Metric aggregator: fetches WandB inference runs, recomputes LPIPS-VGG,
DINO similarity, and I3D-FVD from downloaded video artifacts, then uploads
all per-video metrics (+ videos) to a new WandB run in --dest_project.

Usage
-----
  python scripts/recompute_metrics_and_visualize_on_wandb.py \\
      --project  real_world_big_data_inference_OOD \\
      --entity   <wandb_entity> \\
      --group    my_experiment_group \\
      [--category_json dataset_meta_info/.../val_sample_filtered_categorized.json] \\
      [--dest_project  inference_metrics_ood] \\
      [--prefix_filter   wm2_ar] \\
      [--exclude_prefixes wm1_seg seg] \\
      [--run_id  id1] \\
      [--run_ids  id1 id2] \\
      [--rank_by  psnr]        # psnr | ssim | lpips | mse_img | mse_lat
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

METRICS: List[Tuple[str, str, bool]] = [
    # (column_key, display_label, higher_is_better)
    ("psnr",         "PSNR",             True),
    ("ssim",         "SSIM",             True),
    ("lpips",        "LPIPS (logged)",   False),
    ("lpips_vgg",    "LPIPS-VGG",        False),
    ("mse_img",      "MSE (image)",      False),
    ("mse_lat",      "MSE (latent)",     False),
    ("dino_sim",     "DINO Similarity",  True),
    ("fvd_feat_sim", "FVD Feature Sim",  True),
]

_VIDEO_COMPUTED_METRICS = {"dino_sim", "lpips_vgg", "fvd_feat_sim"}

_HASH_RE = re.compile(r"-[A-Za-z0-9_\-]{6}$")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch inference runs, compute extra metrics, upload to WandB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project",  required=True, help="Source WandB project name.")
    p.add_argument("--entity",   default=None,  help="WandB entity (username/team).")
    p.add_argument(
        "--group",
        default=None,
        help="WandB run group tag to filter by. Optional when --run_id is set.",
    )
    p.add_argument(
        "--category_json",
        default=None,
        help="Optional category JSON (category -> sample ids) for per-category breakdowns.",
    )
    p.add_argument("--fvd_csv", default=None,
                   help="Also write the per-run FVD values (one row per run and prefix) to this CSV.")
    p.add_argument("--dest_project", default="inference_metrics_ood",
                   help="Destination WandB project for metric upload.")
    p.add_argument("--dest_run_name", default=None,
                   help="Name for the destination WandB run (defaults to source group/run ID).")
    p.add_argument("--prefix_filter", nargs="*", default=None,
                   help="Only include table artifacts from these model prefixes.")
    p.add_argument("--exclude_prefixes", nargs="*", default=["wm1_seg"],
                   help="Exclude table artifacts whose prefix matches any of these.")
    p.add_argument("--run_id", default=None,
                   help="Fetch one exact source run by WandB run ID; --group is optional.")
    p.add_argument("--run_ids", nargs="*", default=None,
                   help="Restrict to specific run IDs (default: all in the group).")
    p.add_argument("--rank_by", default="lpips",
                   choices=["psnr", "ssim", "lpips", "lpips_vgg",
                            "mse_img", "mse_lat", "dino_sim", "fvd_feat_sim"],
                   help="Primary metric used to order runs in rankings.")
    # ── LPIPS-VGG ────────────────────────────────────────────────────────────
    p.add_argument("--skip_lpips_vgg", action="store_true",
                   help="Skip LPIPS-VGG recomputation from video frames.")
    # ── FVD ──────────────────────────────────────────────────────────────────
    p.add_argument("--skip_fvd", action="store_true",
                   help="Skip I3D feature similarity and dataset-level FVD computation.")
    p.add_argument("--only_fvd", action="store_true",
                   help="Only recompute/upload FVD-derived metrics; skip image metrics, LPIPS-VGG, and DINO.")
    p.add_argument("--fvd_n_frames", type=int, default=16,
                   help="Number of frames sampled per clip for FVD (I3D).")
    # ── DINO similarity ──────────────────────────────────────────────────────
    p.add_argument("--skip_dino", action="store_true",
                   help="Skip DINO similarity computation.")
    p.add_argument("--dino_model", default="dinov2_vits14",
                   choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
                   help="DINOv2 model variant to use.")
    p.add_argument("--dino_batch_size", type=int, default=16,
                   help="Frame batch size for DINO inference.")
    p.add_argument("--dino_device", default=None,
                   help="Device for DINO/FVD/LPIPS ('cuda', 'cpu'). Auto-detected if omitted.")
    p.add_argument("--num_views", type=int, default=2,
                   help="Number of horizontal views per frame half (set 2 for 480=240+240).")
    # ── Auto best-run selection ──────────────────────────────────────────────
    p.add_argument("--auto_select_best", action="store_true",
                   help="Keep only the best run_variant per (prefix, SVD-group) pair.")
    p.add_argument("--select_prefixes", nargs="*",
                   default=["wm2_ar", "baseline_wm_ar"],
                   help="Prefixes to apply auto-selection to (with --auto_select_best).")
    p.add_argument("--svd_keyword", default="svd_pretrained_weights",
                   help="Substring identifying SVD-pretrained runs.")
    p.add_argument("--exclude_svd", action="store_true",
                   help="Only consider runs WITHOUT svd_keyword when auto-selecting best.")
    p.add_argument("--only_svd", action="store_true",
                   help="Only consider runs WITH svd_keyword when auto-selecting best.")
    return p.parse_args()


def requested_run_ids(args: argparse.Namespace) -> List[str]:
    run_ids: List[str] = []
    if args.run_id:
        run_ids.append(args.run_id)
    if args.run_ids:
        run_ids.extend(args.run_ids)

    deduped: List[str] = []
    seen: set = set()
    for run_id in run_ids:
        if run_id in seen:
            continue
        seen.add(run_id)
        deduped.append(run_id)
    return deduped


def get_run_group(run: Any) -> Optional[str]:
    group = getattr(run, "group", None)
    return str(group) if group else None


def infer_source_group_label(args: argparse.Namespace, runs: List[Any]) -> str:
    if args.group:
        return args.group

    groups = sorted({g for g in (get_run_group(run) for run in runs) if g})
    if len(groups) == 1:
        return groups[0]

    run_ids = requested_run_ids(args) or [str(run.id) for run in runs]
    if len(run_ids) == 1:
        return f"run_id:{run_ids[0]}"
    return "run_ids:" + ",".join(run_ids)


def fetch_source_runs(api: Any, entity_project: str, args: argparse.Namespace) -> List[Any]:
    run_ids = requested_run_ids(args)

    if args.run_id or (run_ids and not args.group):
        print(f"[fetch] Run ID(s): {run_ids}")
        runs = []
        for run_id in run_ids:
            try:
                runs.append(api.run(f"{entity_project}/{run_id}"))
            except Exception as exc:
                print(f"[ERROR] Could not fetch run_id={run_id!r}: {exc}")
                raise SystemExit(1) from exc

        if args.group:
            mismatched = [
                (run.id, get_run_group(run))
                for run in runs
                if get_run_group(run) != args.group
            ]
            if mismatched:
                print("[fetch] Warning: fetched run_id(s) outside requested --group:")
                for run_id, group in mismatched:
                    print(f"  {run_id}: group={group!r}")
        return runs

    if not args.group:
        print("[ERROR] Provide --group, or --run_id/--run_ids for an exact fetch.")
        raise SystemExit(1)

    try:
        runs = list(api.runs(entity_project, filters={"group": args.group}))
    except Exception as exc:
        print(f"[ERROR] Could not fetch runs: {exc}")
        raise SystemExit(1) from exc

    if run_ids:
        runs = [run for run in runs if run.id in run_ids]
        missing = sorted(set(run_ids) - {run.id for run in runs})
        print(f"[fetch] Restricted to {len(runs)} run(s) by --run_ids")
        if missing:
            print(f"[fetch] Missing run ID(s) in group: {', '.join(missing)}")

    return runs


# ──────────────────────────────────────────────────────────────────────────────
# Category loading
# ──────────────────────────────────────────────────────────────────────────────

def load_sample_to_category(path: str) -> Tuple[Dict[int, str], List[str]]:
    with open(path) as f:
        data = json.load(f)

    cat_ids: Dict[str, List[int]] = data
    sample_to_cat: Dict[int, str] = {}
    for cat_name, ids in cat_ids.items():
        for sid in ids:
            sample_to_cat[int(sid)] = str(cat_name)

    category_order = sorted(cat_ids.keys())
    print(
        f"[category] {len(sample_to_cat)} sample→category entries "
        f"across {len(category_order)} categories"
    )
    return sample_to_cat, category_order


# ──────────────────────────────────────────────────────────────────────────────
# WandB artifact helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strip_version(name: str) -> str:
    return name.split(":")[0]

def _strip_hash(name: str) -> str:
    return _HASH_RE.sub("", name)

def _parse_safe_key(art_name: str, run_id: str) -> Optional[str]:
    name = _strip_hash(_strip_version(art_name))
    prefix = f"run-{run_id}-"
    return name[len(prefix):] if name.startswith(prefix) else None

def _classify_safe_key(safe_key: str) -> Optional[Tuple[str, str, str]]:
    m = re.match(r"^(.*)batch_(\d+)_samples$", safe_key)
    if m:
        return ("batch", m.group(1), f"batch_{m.group(2)}_samples")
    m2 = re.match(r"^(.*)all_videos$", safe_key)
    if m2:
        return ("all_videos", m2.group(1), "all_videos")
    if safe_key == "videos_by_psnr":
        return ("videos_by_psnr", "", "videos_by_psnr")
    return None

def get_table_artifacts(
    run,
    prefix_filter: Optional[str],
    exclude_prefixes: Optional[List[str]],
) -> List[Tuple[Any, str, str]]:
    all_videos_prefixes: set = set()
    entries: List[Tuple[Any, str, str, str]] = []

    for art in run.logged_artifacts():
        if art.type != "run_table":
            continue
        safe_key = _parse_safe_key(art.name, run.id)
        if safe_key is None:
            continue
        result = _classify_safe_key(safe_key)
        if result is None:
            continue
        kind, prefix, table_file = result
        if kind == "all_videos":
            all_videos_prefixes.add(prefix)
        entries.append((art, kind, prefix, table_file))

    filtered = [
        (a, k, p, tf) for a, k, p, tf in entries
        if not (k == "batch" and p in all_videos_prefixes)
    ]

    if prefix_filter:
        allowed = {x.rstrip("_") for x in prefix_filter}
        filtered = [(a, k, p, tf) for a, k, p, tf in filtered if p.rstrip("_") in allowed]
    if exclude_prefixes:
        excl = {x.rstrip("_") for x in exclude_prefixes}
        filtered = [(a, k, p, tf) for a, k, p, tf in filtered if p.rstrip("_") not in excl]

    return [(art, prefix, table_file) for art, _, prefix, table_file in filtered]


def load_table(artifact, table_file_stem: str) -> Optional[wandb.Table]:
    try:
        artifact_dir = Path(artifact.download())
    except Exception as e:
        print(f"     [!] download failed: {e}")
        return None

    json_path = artifact_dir / f"{table_file_stem}.table.json"
    if not json_path.exists():
        candidates = sorted(artifact_dir.rglob("*.table.json"))
        if not candidates:
            print(f"     [!] no .table.json in {artifact_dir}")
            return None
        json_path = candidates[0]

    try:
        with open(json_path) as f:
            raw = json.load(f)
        return wandb.Table.from_json(raw, artifact)
    except Exception as e:
        print(f"     [!] failed to parse table JSON: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Row extraction
# ──────────────────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None

def _normalize_sample_id(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    return None

def extract_rows(
    table: wandb.Table,
    run_name: str,
    run_id: str,
    prefix: str,
    sample_to_cat: Dict[int, str],
) -> List[Dict[str, Any]]:
    cols = [c.lower() for c in table.columns]
    rows = []
    for data_row in table.data:
        r = dict(zip(cols, data_row))

        def _get(*keys):
            for k in keys:
                if k in r:
                    return _safe_float(r[k])
            return None

        sid_raw = r.get("id", r.get("sample_id"))
        sid_int = _normalize_sample_id(sid_raw)
        category = (
            sample_to_cat.get(sid_int, "uncategorized")
            if sid_int is not None else "uncategorized"
        )

        video_obj = r.get("video")
        video_path: Optional[str] = None
        if video_obj is not None:
            candidate = getattr(video_obj, "_path", None) or (
                video_obj if isinstance(video_obj, str) else None
            )
            if candidate and os.path.isfile(str(candidate)):
                video_path = str(candidate)

        rows.append({
            "run_name":   run_name,
            "run_id":     run_id,
            "prefix":     prefix.rstrip("_"),
            "sample_id":  sid_int if sid_int is not None else sid_raw,
            "category":   category,
            "video_path": video_path,
            "psnr":       _get("psnr"),
            "ssim":       _get("ssim"),
            "lpips":      _get("lpips"),
            "mse_img":    _get("mse_img", "mse"),
            "mse_lat":    _get("mse_lat"),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Run variant label
# ──────────────────────────────────────────────────────────────────────────────

def make_run_variant(run_name: str, prefix: str) -> str:
    p = prefix if prefix else "default"
    return f"{run_name} [{p}]"


# ──────────────────────────────────────────────────────────────────────────────
# Rankings
# ──────────────────────────────────────────────────────────────────────────────

def _active_metric_specs(df: pd.DataFrame) -> List[Tuple[str, str, bool]]:
    specs: List[Tuple[str, str, bool]] = list(METRICS)
    base_label = {k: label for k, label, _ in METRICS}
    base_higher = {k: higher for k, _, higher in METRICS}

    for col in sorted(df.columns):
        m = re.match(r"^(psnr|ssim|mse_img|lpips_vgg|dino_sim|fvd_feat_sim)_v(\d{2})$", col)
        if not m:
            continue
        base_key = m.group(1)
        view_idx = int(m.group(2))
        label = f"{base_label[base_key]} [view {view_idx}]"
        specs.append((col, label, base_higher[base_key]))

    return specs


def compute_rankings(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    rankings: Dict[str, pd.DataFrame] = {}
    for key, _, higher in _active_metric_specs(df):
        sub = df[df[key].notna()]
        if sub.empty:
            continue
        agg = (
            sub.groupby("run_variant")[key]
            .agg(n_samples="count", mean="mean", std="std", median="median")
            .reset_index()
            .sort_values("mean", ascending=not higher)
            .reset_index(drop=True)
        )
        agg.index += 1
        agg.index.name = "rank"
        rankings[key] = agg
    return rankings


def print_rankings(rankings: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 72)
    for key, label, higher in _active_metric_specs(pd.DataFrame(columns=list(rankings.keys()))):
        if key not in rankings:
            continue
        direction = "↑ higher is better" if higher else "↓ lower is better"
        print(f"\n  {label}  ({direction})")
        print(rankings[key].to_string())
    print("=" * 72)


def _split_views(frame_bgr: np.ndarray, num_views: int) -> List[np.ndarray]:
    if num_views <= 1:
        return [frame_bgr]
    _, w = frame_bgr.shape[:2]
    if w % num_views != 0:
        return [frame_bgr]
    view_w = w // num_views
    return [frame_bgr[:, i * view_w:(i + 1) * view_w] for i in range(num_views)]


def _video_img_metrics(
    video_path: str,
    num_views: int,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    frames = _read_frames(video_path)
    if not frames:
        return [None] * num_views, [None] * num_views, [None] * num_views

    psnr_by_view: List[List[float]] = [[] for _ in range(num_views)]
    ssim_by_view: List[List[float]] = [[] for _ in range(num_views)]
    mse_by_view: List[List[float]] = [[] for _ in range(num_views)]

    for f in frames:
        mid = f.shape[0] // 2
        gt_views = _split_views(f[:mid], num_views)
        gen_views = _split_views(f[mid:], num_views)
        n_pairs = min(len(gt_views), len(gen_views), num_views)
        if n_pairs == 0:
            continue

        for i in range(n_pairs):
            gt_view = gt_views[i]
            gen_view = gen_views[i]
            mse_val = float(np.mean((gt_view.astype(np.float32) - gen_view.astype(np.float32)) ** 2)) / (127.5 ** 2)
            mse_by_view[i].append(mse_val)
            psnr_by_view[i].append(float(peak_signal_noise_ratio(gt_view, gen_view, data_range=255.0)))
            ssim_by_view[i].append(float(structural_similarity(
                gt_view, gen_view, channel_axis=2, data_range=255.0
            )))

    def _finalize(values: List[List[float]]) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for arr in values:
            out.append(float(np.mean(arr)) if arr else None)
        return out

    return _finalize(psnr_by_view), _finalize(ssim_by_view), _finalize(mse_by_view)


def compute_img_metrics_columns(df: pd.DataFrame, num_views: int) -> pd.DataFrame:
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: F401
    except ImportError:
        print("[img-metrics] 'scikit-image' package not installed — run: pip install scikit-image")
        df = df.copy()
        df["psnr"] = None
        df["ssim"] = None
        df["mse_img"] = None
        return df

    psnr_scores_by_view: List[List[Optional[float]]] = [[] for _ in range(num_views)]
    ssim_scores_by_view: List[List[Optional[float]]] = [[] for _ in range(num_views)]
    mse_scores_by_view: List[List[Optional[float]]] = [[] for _ in range(num_views)]
    paths = df["video_path"].tolist()
    n_total, n_ok = len(paths), 0

    for i, path in enumerate(paths):
        print(f"  [img-metrics] {i + 1}/{n_total} …", end="\r", flush=True)
        if path is None or not os.path.isfile(str(path)):
            for vi in range(num_views):
                psnr_scores_by_view[vi].append(None)
                ssim_scores_by_view[vi].append(None)
                mse_scores_by_view[vi].append(None)
            continue
        try:
            psnr_vals, ssim_vals, mse_vals = _video_img_metrics(str(path), num_views)
            if any(v is not None for v in psnr_vals):
                n_ok += 1
        except Exception as exc:
            print(f"\n  [img-metrics] error on {path}: {exc}")
            psnr_vals = [None] * num_views
            ssim_vals = [None] * num_views
            mse_vals = [None] * num_views

        for vi in range(num_views):
            psnr_scores_by_view[vi].append(psnr_vals[vi])
            ssim_scores_by_view[vi].append(ssim_vals[vi])
            mse_scores_by_view[vi].append(mse_vals[vi])

    print(f"  [img-metrics] done — computed {n_ok}/{n_total} rows            ")
    df = df.copy()
    for vi in range(num_views):
        suffix = f"_v{vi:02d}"
        df[f"psnr{suffix}"] = psnr_scores_by_view[vi]
        df[f"ssim{suffix}"] = ssim_scores_by_view[vi]
        df[f"mse_img{suffix}"] = mse_scores_by_view[vi]

    # Keep legacy columns for compatibility: view 0 values.
    df["psnr"] = psnr_scores_by_view[0] if num_views > 0 else [None] * len(df)
    df["ssim"] = ssim_scores_by_view[0] if num_views > 0 else [None] * len(df)
    df["mse_img"] = mse_scores_by_view[0] if num_views > 0 else [None] * len(df)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# LPIPS-VGG  (recomputed from raw video frames)
# ──────────────────────────────────────────────────────────────────────────────

def _frame_to_lpips_tensor(frame_bgr: np.ndarray, device: str) -> Any:
    import torch
    rgb = frame_bgr[..., ::-1].astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def _video_lpips_vgg(
    video_path: str,
    loss_fn: Any,
    device: str,
    num_views: int,
) -> List[Optional[float]]:
    frames = _read_frames(video_path)
    if not frames:
        return [None] * num_views

    scores_by_view: List[List[float]] = [[] for _ in range(num_views)]
    import torch
    for f in frames:
        mid = f.shape[0] // 2
        gt_views = _split_views(f[:mid], num_views)
        gen_views = _split_views(f[mid:], num_views)
        n_pairs = min(len(gt_views), len(gen_views), num_views)
        for vi in range(n_pairs):
            gt_t = _frame_to_lpips_tensor(gt_views[vi], device)
            gen_t = _frame_to_lpips_tensor(gen_views[vi], device)
            with torch.no_grad():
                scores_by_view[vi].append(float(loss_fn(gt_t, gen_t).item()))

    out: List[Optional[float]] = []
    for arr in scores_by_view:
        out.append(float(np.mean(arr)) if arr else None)
    return out


def compute_lpips_vgg_column(
    df: pd.DataFrame,
    device: Optional[str],
    num_views: int,
) -> pd.DataFrame:
    import torch
    try:
        import lpips as lpips_lib
    except ImportError:
        print("[lpips-vgg] 'lpips' package not installed — run: pip install lpips")
        df = df.copy()
        df["lpips_vgg"] = None
        return df

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lpips-vgg] loading VGG model on {device} …")
    loss_fn = lpips_lib.LPIPS(net="vgg").to(device)
    loss_fn.eval()

    scores_by_view: List[List[Optional[float]]] = [[] for _ in range(num_views)]
    paths = df["video_path"].tolist()
    n_total, n_ok = len(paths), 0

    for i, path in enumerate(paths):
        print(f"  [lpips-vgg] {i + 1}/{n_total} …", end="\r", flush=True)
        if path is None or not os.path.isfile(str(path)):
            for vi in range(num_views):
                scores_by_view[vi].append(None)
            continue
        try:
            sims = _video_lpips_vgg(str(path), loss_fn, device, num_views)
            if any(v is not None for v in sims):
                n_ok += 1
        except Exception as exc:
            print(f"\n  [lpips-vgg] error on {path}: {exc}")
            sims = [None] * num_views
        for vi in range(num_views):
            scores_by_view[vi].append(sims[vi])

    print(f"  [lpips-vgg] done — computed {n_ok}/{n_total} scores            ")
    df = df.copy()
    for vi in range(num_views):
        df[f"lpips_vgg_v{vi:02d}"] = scores_by_view[vi]
    df["lpips_vgg"] = scores_by_view[0] if num_views > 0 else [None] * len(df)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FVD  (per-video I3D feature similarity + dataset-level Fréchet distance)
# ──────────────────────────────────────────────────────────────────────────────

def _load_i3d_model(device: str) -> Any:
    import torch

    print(f"[fvd] loading PyTorchVideo I3D-R50 on {device} …")
    try:
        from pytorchvideo.models.hub import i3d_r50
        model = i3d_r50(pretrained=True)
    except Exception:
        try:
            model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
        except Exception as hub_exc:
            raise RuntimeError(
                "Could not load PyTorchVideo I3D-R50. Install pytorchvideo, "
                "or ensure torch.hub can access/cache facebookresearch/pytorchvideo."
            ) from hub_exc

    # The final block is a classification head. Removing the projection gives
    # pooled I3D embeddings for Fréchet statistics instead of class logits.
    head = model.blocks[-1]
    if hasattr(head, "proj"):
        head.proj = torch.nn.Identity()
    if hasattr(head, "activation"):
        head.activation = torch.nn.Identity()

    model.eval().to(device)
    return model


def _sample_frames_uniform(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    if not frames:
        return []
    if len(frames) <= n:
        return frames + [frames[-1]] * (n - len(frames))
    idx = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in idx]


def _clip_to_tensor(frames_bgr: List[np.ndarray], device: str, n_frames: int) -> Any:
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    sampled = _sample_frames_uniform(frames_bgr, n_frames)
    tensors = []
    for f in sampled:
        pil = Image.fromarray(f[..., ::-1].astype(np.uint8))
        pil = TF.resize(pil, [256, 256])
        pil = TF.center_crop(pil, [224, 224])
        t = TF.to_tensor(pil)
        t = TF.normalize(t, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        tensors.append(t)
    return torch.stack(tensors).permute(1, 0, 2, 3).unsqueeze(0).float().to(device)


def _flatten_video_feature(feat: Any) -> np.ndarray:
    feat_np = feat.detach().cpu().float().numpy()
    return feat_np.reshape(feat_np.shape[0], -1).squeeze(0)


def _frechet_distance(mu1: np.ndarray, sig1: np.ndarray,
                      mu2: np.ndarray, sig2: np.ndarray) -> float:
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    cov = sqrtm(sig1 @ sig2)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * cov))


def compute_fvd_columns(
    df: pd.DataFrame,
    device: Optional[str],
    n_frames: int,
    num_views: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_i3d_model(device)

    gt_feats_all: Dict[str, Dict[int, List[np.ndarray]]] = {}
    gen_feats_all: Dict[str, Dict[int, List[np.ndarray]]] = {}
    feat_sim_scores_by_view: List[List[Optional[float]]] = [[] for _ in range(num_views)]
    paths = df["video_path"].tolist()
    run_variants = df["run_variant"].tolist()
    n_total, n_ok = len(paths), 0

    for i, (path, rv) in enumerate(zip(paths, run_variants)):
        print(f"  [fvd] {i + 1}/{n_total} …", end="\r", flush=True)
        if path is None or not os.path.isfile(str(path)):
            for vi in range(num_views):
                feat_sim_scores_by_view[vi].append(None)
            continue
        try:
            frames = _read_frames(str(path))
            if not frames:
                for vi in range(num_views):
                    feat_sim_scores_by_view[vi].append(None)
                continue

            gt_by_view: List[List[np.ndarray]] = [[] for _ in range(num_views)]
            gen_by_view: List[List[np.ndarray]] = [[] for _ in range(num_views)]
            for f in frames:
                mid = f.shape[0] // 2
                gt_views = _split_views(f[:mid], num_views)
                gen_views = _split_views(f[mid:], num_views)
                n_pairs = min(len(gt_views), len(gen_views), num_views)
                for vi in range(n_pairs):
                    gt_by_view[vi].append(gt_views[vi])
                    gen_by_view[vi].append(gen_views[vi])

            had_any = False
            sim_vals = [None] * num_views
            for vi in range(num_views):
                if not gt_by_view[vi] or not gen_by_view[vi]:
                    continue
                gt_clip = _clip_to_tensor(gt_by_view[vi], device, n_frames)
                gen_clip = _clip_to_tensor(gen_by_view[vi], device, n_frames)
                with torch.no_grad():
                    gt_feat = _flatten_video_feature(model(gt_clip))
                    gen_feat = _flatten_video_feature(model(gen_clip))
                gt_n = gt_feat / (np.linalg.norm(gt_feat) + 1e-8)
                gen_n = gen_feat / (np.linalg.norm(gen_feat) + 1e-8)
                sim_vals[vi] = float((gt_n * gen_n).sum())
                gt_feats_all.setdefault(rv, {}).setdefault(vi, []).append(gt_feat)
                gen_feats_all.setdefault(rv, {}).setdefault(vi, []).append(gen_feat)
                had_any = True

            if had_any:
                n_ok += 1
            for vi in range(num_views):
                feat_sim_scores_by_view[vi].append(sim_vals[vi])

        except Exception as exc:
            print(f"\n  [fvd] error on {path}: {exc}")
            for vi in range(num_views):
                feat_sim_scores_by_view[vi].append(None)

    print(f"  [fvd] done — computed {n_ok}/{n_total} feature similarities            ")

    df = df.copy()
    for vi in range(num_views):
        df[f"fvd_feat_sim_v{vi:02d}"] = feat_sim_scores_by_view[vi]
    df["fvd_feat_sim"] = feat_sim_scores_by_view[0] if num_views > 0 else [None] * len(df)

    fvd_rows: List[Dict[str, Any]] = []
    for rv in df["run_variant"].unique():
        row: Dict[str, Any] = {"run_variant": rv}
        for vi in range(num_views):
            gt_list = gt_feats_all.get(rv, {}).get(vi, [])
            gen_list = gen_feats_all.get(rv, {}).get(vi, [])
            fvd_col = f"fvd_v{vi:02d}"
            n_col = f"n_videos_v{vi:02d}"
            if len(gt_list) < 2:
                row[fvd_col] = None
                row[n_col] = len(gt_list)
                continue
            gt_mat = np.stack(gt_list)
            gen_mat = np.stack(gen_list)
            mu1, sig1 = gt_mat.mean(0), np.cov(gt_mat, rowvar=False)
            mu2, sig2 = gen_mat.mean(0), np.cov(gen_mat, rowvar=False)
            try:
                fvd_val = _frechet_distance(mu1, sig1, mu2, sig2)
            except Exception as exc:
                print(f"  [fvd] Fréchet distance failed for {rv} view={vi}: {exc}")
                fvd_val = None
            row[fvd_col] = fvd_val
            row[n_col] = len(gt_list)

        # Legacy columns for compatibility (view 0)
        row["fvd"] = row.get("fvd_v00")
        row["n_videos"] = row.get("n_videos_v00")
        fvd_rows.append(row)

    fvd_df = pd.DataFrame(fvd_rows).sort_values("fvd")
    return df, fvd_df


# ──────────────────────────────────────────────────────────────────────────────
# DINO similarity
# ──────────────────────────────────────────────────────────────────────────────

def _load_dino_model(model_name: str, device: Optional[str]) -> Tuple[Any, Any, str]:
    import torch
    import torchvision.transforms as T

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dino] loading {model_name} on {device} …")
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    model.eval().to(device)

    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform, device


def _read_frames(video_path: str) -> List[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _dino_features(
    model: Any,
    transform: Any,
    frames_bgr: List[np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch
    from PIL import Image

    tensors = [
        transform(Image.fromarray(f[..., ::-1].astype(np.uint8)))
        for f in frames_bgr
    ]
    feats: List[np.ndarray] = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        with torch.no_grad():
            feats.append(model(batch).cpu().float().numpy())
    return np.concatenate(feats, axis=0)


def _video_dino_sim(
    video_path: str,
    model: Any,
    transform: Any,
    device: str,
    batch_size: int,
    num_views: int,
) -> List[Optional[float]]:
    frames = _read_frames(video_path)
    if not frames:
        return [None] * num_views

    gt_by_view: List[List[np.ndarray]] = [[] for _ in range(num_views)]
    gen_by_view: List[List[np.ndarray]] = [[] for _ in range(num_views)]
    for f in frames:
        mid = f.shape[0] // 2
        gt_views = _split_views(f[:mid], num_views)
        gen_views = _split_views(f[mid:], num_views)
        n_pairs = min(len(gt_views), len(gen_views), num_views)
        for vi in range(n_pairs):
            gt_by_view[vi].append(gt_views[vi])
            gen_by_view[vi].append(gen_views[vi])

    out: List[Optional[float]] = [None] * num_views
    for vi in range(num_views):
        gt_frames = gt_by_view[vi]
        gen_frames = gen_by_view[vi]
        if not gt_frames or not gen_frames:
            continue

        gt_feats = _dino_features(model, transform, gt_frames, device, batch_size)
        gen_feats = _dino_features(model, transform, gen_frames, device, batch_size)
        gt_n = gt_feats / (np.linalg.norm(gt_feats, axis=1, keepdims=True) + 1e-8)
        gen_n = gen_feats / (np.linalg.norm(gen_feats, axis=1, keepdims=True) + 1e-8)
        out[vi] = float((gt_n * gen_n).sum(axis=1).mean())
    return out


def compute_dino_column(
    df: pd.DataFrame,
    model_name: str,
    device: Optional[str],
    batch_size: int,
    num_views: int,
) -> pd.DataFrame:
    model, transform, device = _load_dino_model(model_name, device)

    scores_by_view: List[List[Optional[float]]] = [[] for _ in range(num_views)]
    paths = df["video_path"].tolist()
    n_total, n_ok = len(paths), 0

    for i, path in enumerate(paths):
        print(f"  [dino] {i + 1}/{n_total} …", end="\r", flush=True)
        if path is None or not os.path.isfile(str(path)):
            for vi in range(num_views):
                scores_by_view[vi].append(None)
            continue
        try:
            sims = _video_dino_sim(str(path), model, transform, device, batch_size, num_views)
            if any(v is not None for v in sims):
                n_ok += 1
        except Exception as exc:
            print(f"\n  [dino] error on {path}: {exc}")
            sims = [None] * num_views
        for vi in range(num_views):
            scores_by_view[vi].append(sims[vi])

    print(f"  [dino] done — computed {n_ok}/{n_total} scores            ")
    df = df.copy()
    for vi in range(num_views):
        df[f"dino_sim_v{vi:02d}"] = scores_by_view[vi]
    df["dino_sim"] = scores_by_view[0] if num_views > 0 else [None] * len(df)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Auto best-run selection
# ──────────────────────────────────────────────────────────────────────────────

def select_best_run_variants(
    df: pd.DataFrame,
    prefixes: List[str],
    svd_keyword: str,
    exclude_svd: bool = False,
    only_svd: bool = False,
) -> List[str]:
    selected: List[str] = []

    if exclude_svd:
        svd_groups = (False,)
    elif only_svd:
        svd_groups = (True,)
    else:
        svd_groups = (True, False)

    for prefix in prefixes:
        prefix_df = df[df["prefix"] == prefix]
        if prefix_df.empty:
            print(f"  [auto-select] prefix={prefix!r} – no data, skipping")
            continue

        for has_svd in svd_groups:
            svd_label = "with SVD" if has_svd else "without SVD"
            if has_svd:
                mask = prefix_df["run_name"].str.contains(svd_keyword, na=False)
            else:
                mask = ~prefix_df["run_name"].str.contains(svd_keyword, na=False)

            sub = prefix_df[mask]
            if sub.empty:
                print(f"  [auto-select] prefix={prefix!r} {svd_label} – no candidates, skipping")
                continue

            candidates = list(sub["run_variant"].unique())
            if len(candidates) == 1:
                selected.append(candidates[0])
                print(f"  [auto-select] prefix={prefix!r} {svd_label} → {candidates[0]}  (only candidate)")
                continue

            avgs: Dict[str, Dict[str, Optional[float]]] = {}
            for rv in candidates:
                rv_data = sub[sub["run_variant"] == rv]
                avgs[rv] = {}
                for key, _, _ in METRICS:
                    v = rv_data[key].mean()
                    avgs[rv][key] = None if (v is None or np.isnan(v)) else float(v)

            rank_sum: Dict[str, int] = {rv: 0 for rv in candidates}
            for key, _, higher in METRICS:
                valid = {rv: avgs[rv][key] for rv in candidates if avgs[rv].get(key) is not None}
                if not valid:
                    continue
                sorted_rvs = sorted(valid, key=lambda rv: valid[rv], reverse=higher)  # type: ignore[arg-type]
                for rank, rv in enumerate(sorted_rvs, 1):
                    rank_sum[rv] += rank

            best = min(rank_sum, key=rank_sum.__getitem__)
            selected.append(best)
            print(f"  [auto-select] prefix={prefix!r} {svd_label} → {best}")
            print(f"               (composite ranks: { {rv: rank_sum[rv] for rv in candidates} })")

    return selected


# ──────────────────────────────────────────────────────────────────────────────
# WandB upload
# ──────────────────────────────────────────────────────────────────────────────

def upload_to_wandb(
    df: pd.DataFrame,
    run_order: List[str],
    rankings: Dict[str, pd.DataFrame],
    fvd_run_df: Optional[pd.DataFrame],
    source_group: str,
    args: argparse.Namespace,
) -> None:
    entity = args.entity
    dest_project = args.dest_project
    run_name = args.dest_run_name or source_group

    print(f"\n[upload] Initialising WandB run in project '{dest_project}' …")
    dest_run = wandb.init(
        project=dest_project,
        entity=entity,
        name=run_name,
        job_type="metric_aggregation",
        config={
            "source_project":  args.project,
            "source_group":    source_group,
            "source_run_ids":  sorted(str(run_id) for run_id in df["run_id"].dropna().unique()),
            "rank_by":         args.rank_by,
            "num_views":       args.num_views,
            "n_run_variants":  len(run_order),
            "n_samples_total": len(df),
            "only_fvd":        bool(args.only_fvd),
            "fvd_model":       "pytorchvideo/i3d_r50",
            # Store full variant names so the index→name mapping is queryable
            "run_variants": {f"rv_{i:02d}": rv for i, rv in enumerate(run_order)},
        },
    )

    # ── Per-run-variant table (with videos) ───────────────────────────────────
    # Keys are short ("metrics/rv_00", "metrics/rv_01", …) to stay under
    # WandB's 128-char artifact name limit.  Full run_variant is the first column.
    if args.only_fvd:
        fixed_metric_cols = ["fvd_feat_sim"]
    else:
        fixed_metric_cols = ["psnr", "ssim", "lpips", "lpips_vgg", "mse_img", "mse_lat", "dino_sim", "fvd_feat_sim"]
    per_view_pattern = (
        r"^fvd_feat_sim_v\d{2}$"
        if args.only_fvd
        else r"^(psnr|ssim|mse_img|lpips_vgg|dino_sim|fvd_feat_sim)_v\d{2}$"
    )
    per_view_cols = sorted(c for c in df.columns if re.match(per_view_pattern, c))
    table_metric_cols = [c for c in fixed_metric_cols if c in df.columns] + per_view_cols
    table_columns = ["run_variant", "sample_id", "category"] + table_metric_cols + ["video"]

    log_payload: Dict[str, Any] = {}

    print("\n[upload] Index → run_variant mapping:")
    for i, rv in enumerate(run_order):
        rv_df = df[df["run_variant"] == rv].copy()
        table = wandb.Table(columns=table_columns)

        n_videos = 0
        for _, row in rv_df.iterrows():
            video_obj = None
            vpath = row.get("video_path")
            if vpath and os.path.isfile(str(vpath)):
                try:
                    video_obj = wandb.Video(str(vpath), fps=5, format="mp4")
                    n_videos += 1
                except Exception as exc:
                    print(f"  [upload] wandb.Video failed for {vpath}: {exc}")

            def _val(col: str, _row: Any = row) -> Any:
                v = _row.get(col)
                return None if (v is not None and isinstance(v, float) and np.isnan(v)) else v

            values = [rv, _val("sample_id"), _val("category")]
            for col in table_metric_cols:
                values.append(_val(col))
            values.append(video_obj)
            table.add_data(*values)

        key = f"metrics/rv_{i:02d}"
        log_payload[key] = table
        print(f"  rv_{i:02d} → {rv!r}  ({len(rv_df)} rows, {n_videos} videos)")

    # ── Rankings tables ───────────────────────────────────────────────────────
    for key, rank_df in rankings.items():
        log_payload[f"rankings/{key}"] = wandb.Table(dataframe=rank_df.reset_index())

    # ── Dataset-level FVD ─────────────────────────────────────────────────────
    if fvd_run_df is not None and not fvd_run_df.empty:
        log_payload["fvd_per_run"] = wandb.Table(dataframe=fvd_run_df.reset_index(drop=True))

    # ── Per-category breakdown table  (run_variant × category → mean metrics) ─
    metric_cols = (
        ["fvd_feat_sim"] if args.only_fvd and "fvd_feat_sim" in df.columns
        else [k for k, _, _ in METRICS if k in df.columns]
    ) + per_view_cols
    cat_cols = ["run_variant", "category"] + metric_cols
    cat_summary = (
        df[cat_cols]
        .groupby(["run_variant", "category"])[metric_cols]
        .agg(["mean", "count"])
        .round(5)
    )
    cat_summary.columns = ["_".join(c) for c in cat_summary.columns]
    cat_table = wandb.Table(dataframe=cat_summary.reset_index())
    log_payload["category_breakdown"] = cat_table

    # ── Summary scalars (mean per metric, keyed by rv index) ─────────────────
    summary: Dict[str, float] = {}
    for i, rv in enumerate(run_order):
        rv_df = df[df["run_variant"] == rv]
        sk = f"rv_{i:02d}"
        for key in metric_cols:
            if key not in rv_df.columns:
                continue
            val = rv_df[key].mean()
            if not np.isnan(val):
                summary[f"{sk}/{key}"] = float(val)

    dest_run.log(log_payload)
    dest_run.summary.update(summary)
    dest_run.finish()

    print(f"[upload] Done → {dest_run.url}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.num_views < 1:
        print("[ERROR] --num_views must be >= 1")
        raise SystemExit(1)
    if args.only_fvd and args.skip_fvd:
        print("[ERROR] --only_fvd cannot be combined with --skip_fvd")
        raise SystemExit(1)
    if args.only_fvd:
        args.skip_lpips_vgg = True
        args.skip_dino = True
        if args.rank_by != "fvd_feat_sim":
            print(f"[only-fvd] overriding --rank_by {args.rank_by!r} -> 'fvd_feat_sim'")
            args.rank_by = "fvd_feat_sim"

    # ── Categories ────────────────────────────────────────────────────────────
    sample_to_cat = load_sample_to_category(args.category_json)[0] if args.category_json else {}

    # ── WandB source ─────────────────────────────────────────────────────────
    api = wandb.Api()
    entity = args.entity or api.default_entity
    entity_project = f"{entity}/{args.project}" if entity else args.project

    print(f"\n[fetch] Project : {entity_project}")
    if args.group:
        print(f"[fetch] Group   : {args.group!r}")
    runs = fetch_source_runs(api, entity_project, args)

    if not runs:
        print("[fetch] No runs found. Exiting.")
        return

    source_group = infer_source_group_label(args, runs)
    print(f"[fetch] Source label: {source_group!r}")
    print(f"[fetch] {len(runs)} run(s) found")

    # ── Collect rows ──────────────────────────────────────────────────────────
    all_rows: List[Dict] = []

    for run in runs:
        print(f"\n[run] {run.id}  '{run.name}'  state={run.state}")
        table_entries = get_table_artifacts(run, args.prefix_filter, args.exclude_prefixes)
        if not table_entries:
            print("  → no recognised table artifacts, skipping")
            continue

        for art, prefix, table_file in table_entries:
            print(f"  → loading  prefix={prefix!r:<20}  file={table_file}.table.json")
            table = load_table(art, table_file)
            if table is None:
                continue
            rows = extract_rows(table, run.name, run.id, prefix, sample_to_cat)
            if not rows:
                continue
            all_rows.extend(rows)
            print(f"     {len(rows)} rows")

    if not all_rows:
        print("\n[report] No data collected. Exiting.")
        return

    # ── DataFrame ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df["run_variant"] = df.apply(
        lambda r: make_run_variant(r["run_name"], r["prefix"]), axis=1
    )

    # Deduplicate: keep row with most non-null metrics per (run_variant, sample_id)
    metric_keys = [k for k, _, _ in METRICS if k not in _VIDEO_COMPUTED_METRICS]
    df["_completeness"] = df[metric_keys].notna().sum(axis=1)
    df = (
        df.sort_values("_completeness", ascending=False)
          .drop_duplicates(subset=["run_variant", "sample_id"])
          .drop(columns=["_completeness"])
          .reset_index(drop=True)
    )

    # ── Recompute image-space metrics (per-view) ──────────────────────────────
    n_with_video = df["video_path"].notna().sum()
    if args.only_fvd:
        print("\n[img-metrics] skipped (--only_fvd)")
        for col in ["psnr", "ssim", "lpips", "lpips_vgg", "mse_img", "mse_lat", "dino_sim"]:
            if col in df.columns:
                df[col] = None
    else:
        print(
            f"\n[img-metrics] recomputing PSNR/SSIM/MSE(image) "
            f"for {n_with_video} videos (num_views={args.num_views}) …"
        )
        if n_with_video == 0:
            print("[img-metrics] no video paths found — setting psnr/ssim/mse_img to None")
            df["psnr"] = None
            df["ssim"] = None
            df["mse_img"] = None
        else:
            df = compute_img_metrics_columns(df, args.num_views)
            parts = []
            for vi in range(args.num_views):
                parts.append(
                    f"v{vi:02d}: "
                    f"psnr={df[f'psnr_v{vi:02d}'].notna().sum()}/{len(df)}, "
                    f"ssim={df[f'ssim_v{vi:02d}'].notna().sum()}/{len(df)}, "
                    f"mse_img={df[f'mse_img_v{vi:02d}'].notna().sum()}/{len(df)}"
                )
            print("[img-metrics] available: " + " | ".join(parts))

    # ── LPIPS-VGG ─────────────────────────────────────────────────────────────
    if not args.skip_lpips_vgg:
        n_with_video = df["video_path"].notna().sum()
        print(f"\n[lpips-vgg] computing for {n_with_video} videos …")
        if n_with_video == 0:
            print("[lpips-vgg] no video paths found — skipping")
            df["lpips_vgg"] = None
        else:
            df = compute_lpips_vgg_column(df, args.dino_device, args.num_views)
            parts = [
                f"v{vi:02d}={df[f'lpips_vgg_v{vi:02d}'].notna().sum()}/{len(df)}"
                for vi in range(args.num_views)
                if f"lpips_vgg_v{vi:02d}" in df.columns
            ]
            print(f"[lpips-vgg] available per view: {' | '.join(parts)}")
    else:
        df["lpips_vgg"] = None
        print("\n[lpips-vgg] skipped (--skip_lpips_vgg)")

    # ── FVD ───────────────────────────────────────────────────────────────────
    fvd_run_df: Optional[pd.DataFrame] = None
    if not args.skip_fvd:
        n_with_video = df["video_path"].notna().sum()
        print(f"\n[fvd] computing I3D features for {n_with_video} videos …")
        if n_with_video == 0:
            print("[fvd] no video paths found — skipping")
            df["fvd_feat_sim"] = None
        else:
            df, fvd_run_df = compute_fvd_columns(
                df, args.dino_device, args.fvd_n_frames, args.num_views
            )
            parts = [
                f"v{vi:02d}={df[f'fvd_feat_sim_v{vi:02d}'].notna().sum()}/{len(df)}"
                for vi in range(args.num_views)
                if f"fvd_feat_sim_v{vi:02d}" in df.columns
            ]
            print(f"[fvd] fvd_feat_sim available per view: {' | '.join(parts)}")
            print(fvd_run_df.to_string(index=False))
            if args.fvd_csv:
                run_ids = df.drop_duplicates("run_variant").set_index("run_variant")["run_id"]
                out = fvd_run_df.copy()
                out.insert(0, "run_id", out["run_variant"].map(run_ids))
                os.makedirs(os.path.dirname(os.path.abspath(args.fvd_csv)), exist_ok=True)
                out.to_csv(args.fvd_csv, index=False)
                print(f"[fvd] wrote {args.fvd_csv}")
    else:
        df["fvd_feat_sim"] = None
        print("\n[fvd] skipped (--skip_fvd)")

    # ── DINO similarity ───────────────────────────────────────────────────────
    if not args.skip_dino:
        n_with_video = df["video_path"].notna().sum()
        print(f"\n[dino] computing DINO similarity for {n_with_video} videos …")
        if n_with_video == 0:
            print("[dino] no video paths found — skipping")
            df["dino_sim"] = None
        else:
            df = compute_dino_column(
                df, args.dino_model, args.dino_device, args.dino_batch_size, args.num_views
            )
            parts = [
                f"v{vi:02d}={df[f'dino_sim_v{vi:02d}'].notna().sum()}/{len(df)}"
                for vi in range(args.num_views)
                if f"dino_sim_v{vi:02d}" in df.columns
            ]
            print(f"[dino] available per view: {' | '.join(parts)}")
    else:
        df["dino_sim"] = None
        print("\n[dino] skipped (--skip_dino)")

    # ── Run order (best-first by rank_by metric) ──────────────────────────────
    rank_higher = next(h for k, _, h in METRICS if k == args.rank_by)
    run_avgs = (
        df[df[args.rank_by].notna()]
        .groupby("run_variant")[args.rank_by]
        .mean()
        .sort_values(ascending=not rank_higher)
    )
    run_order = run_avgs.index.tolist()
    extras = sorted(rv for rv in df["run_variant"].unique() if rv not in run_order)
    run_order.extend(extras)

    print(f"\n[report] {len(df)} total rows across {len(run_order)} run variant(s)")
    print(f"[report] Run order (by {args.rank_by}, best-first):")
    for i, rv in enumerate(run_order, 1):
        print(f"  {i}. {rv}")

    # ── Auto best-run selection ───────────────────────────────────────────────
    if args.auto_select_best:
        print(f"\n[auto-select] Selecting best run per (prefix, SVD-group) …")
        selected = select_best_run_variants(
            df, args.select_prefixes, args.svd_keyword,
            exclude_svd=args.exclude_svd, only_svd=args.only_svd,
        )
        if selected:
            df = df[df["run_variant"].isin(selected)].reset_index(drop=True)
            run_order = [rv for rv in run_order if rv in selected]
            print(f"[auto-select] Kept {len(run_order)} run variant(s)")
            for rv in run_order:
                print(f"  • {rv}")
        else:
            print("[auto-select] No variants selected — using all.")

    # ── Rankings ──────────────────────────────────────────────────────────────
    rankings = compute_rankings(df)
    print_rankings(rankings)

    # ── Upload to WandB ───────────────────────────────────────────────────────
    upload_to_wandb(df, run_order, rankings, fvd_run_df, source_group, args)


if __name__ == "__main__":
    main()
