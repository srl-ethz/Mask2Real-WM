"""Frechet Inception Distance (FID) of generated vs. ground-truth frames for the fidelity tables.

Each evaluation run logs one comparison video per sample (top half ground truth, bottom half
generated, camera views side by side). For every dataset in the runs spec, the ground-truth
frames of the first listed model's run form the reference pool shared by all models, and 8
frames per (video, view) are sampled uniformly from each run. Per (dataset, model) this reports

  fid                          pooled over both camera views (torch-fidelity)
  fid_view{0,1}                per camera view, from Inception-v3 pool features
  fid_view{0,1}_bootstrap_sd   SD of the per-view FID over --n_boot bootstrap resamples of the
                               frame features; one RandomState(--seed) stream visited in
                               dataset, model, view order

Frames and features are cached under --work_dir; existing frame folders are reused.

Example:
    python scripts/paper/compute_fid.py --entity <wandb_entity> \
        --runs scripts/paper/fidelity_runs.json --work_dir outputs/fid \
        --out_csv results/fidelity/fid.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import torch
import torch_fidelity
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recompute_metrics_and_visualize_on_wandb import (  # noqa: E402
    _read_frames, _sample_frames_uniform, _split_views, extract_rows, get_table_artifacts, load_table,
)

NUM_VIEWS = 2
VIEW_RE = re.compile(r"_view(\d+)_")


def run_video_paths(api, entity_project: str, run_id: str):
    run = api.run(f"{entity_project}/{run_id}")
    rows = []
    for art, prefix, table_file in get_table_artifacts(run, None, ["wm1_seg"]):
        table = load_table(art, table_file)
        if table is not None:
            rows.extend(extract_rows(table, run.name, run.id, prefix, {}))
    return [r.get("video_path") for r in rows]


def dump_frames(video_paths, out_dir: str, half: str, frames_per_view: int) -> int:
    """Write frames_per_view frames per (video, view) of the ground-truth or generated half."""
    os.makedirs(out_dir, exist_ok=True)
    n_written = 0
    for video_idx, path in enumerate(video_paths):
        if path is None or not os.path.isfile(str(path)):
            continue
        frames = _read_frames(str(path))
        if not frames:
            continue
        by_view = [[] for _ in range(NUM_VIEWS)]
        for frame in frames:
            mid = frame.shape[0] // 2
            gt_views = _split_views(frame[:mid], NUM_VIEWS)
            gen_views = _split_views(frame[mid:], NUM_VIEWS)
            for view in range(min(len(gt_views), len(gen_views), NUM_VIEWS)):
                by_view[view].append(gt_views[view] if half == "gt" else gen_views[view])
        for view in range(NUM_VIEWS):
            if not by_view[view]:
                continue
            for frame_idx, frame_bgr in enumerate(_sample_frames_uniform(by_view[view], frames_per_view)):
                cv2.imwrite(f"{out_dir}/v{video_idx:04d}_view{view}_f{frame_idx:02d}.png", frame_bgr)
                n_written += 1
    return n_written


def view_features(frame_dir: str, view: int, cache_path: str, extractor, device: str) -> np.ndarray:
    if os.path.exists(cache_path):
        return np.load(cache_path)
    files = sorted(
        f for f in glob.glob(f"{frame_dir}/*.png")
        if VIEW_RE.search(f) and int(VIEW_RE.search(f).group(1)) == view
    )
    feats = []
    with torch.no_grad():
        for i in range(0, len(files), 64):
            imgs = [cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in files[i:i + 64]]
            batch = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).to(torch.uint8).to(device)
            (feat,) = extractor(batch)
            feats.append(feat.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, feats)
    return feats


def fid_from_features(gt_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    s_gt = fid_features_to_statistics(torch.from_numpy(gt_feats))
    s_gen = fid_features_to_statistics(torch.from_numpy(gen_feats))
    return float(fid_statistics_to_metric(s_gen, s_gt, False)["frechet_inception_distance"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entity", default=None, help="W&B entity; only needed for frames not cached yet")
    parser.add_argument("--project", default=None, help="W&B project (default: the one in the runs spec)")
    parser.add_argument("--runs", default="scripts/paper/fidelity_runs.json")
    parser.add_argument("--work_dir", default="outputs/fid")
    parser.add_argument("--out_csv", default="results/fidelity/fid.csv")
    parser.add_argument("--frames_per_view", type=int, default=8)
    parser.add_argument("--n_boot", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.runs) as f:
        spec = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    api = None

    # 1) frames: shared ground-truth pool per dataset, generated frames per model
    for dataset, dataset_spec in spec["datasets"].items():
        for i, (model, run_id) in enumerate(dataset_spec["runs"].items()):
            targets = [("gen", f"{args.work_dir}/frames/{dataset}/gen_{model}")]
            if i == 0:
                targets.insert(0, ("gt", f"{args.work_dir}/frames/{dataset}/gt"))
            missing = [(half, d) for half, d in targets if not glob.glob(f"{d}/*.png")]
            if not missing:
                continue
            if api is None:
                import wandb
                api = wandb.Api()
            paths = run_video_paths(api, f"{args.entity}/{args.project or spec['wandb_project']}", run_id)
            for half, frame_dir in missing:
                n = dump_frames(paths, frame_dir, half, args.frames_per_view)
                print(f"{dataset} / {model}: wrote {n} {half} frames -> {frame_dir}")

    # 2) FID
    extractor = FeatureExtractorInceptionV3("inception", ["2048"]).to(device).eval()
    rng = np.random.RandomState(args.seed)
    rows = []
    for dataset, dataset_spec in spec["datasets"].items():
        gt_dir = f"{args.work_dir}/frames/{dataset}/gt"
        for model, run_id in dataset_spec["runs"].items():
            gen_dir = f"{args.work_dir}/frames/{dataset}/gen_{model}"
            pooled = torch_fidelity.calculate_metrics(
                input1=gen_dir, input2=gt_dir, cuda=device == "cuda", fid=True,
                samples_find_deep=False, batch_size=64, verbose=False,
            )["frechet_inception_distance"]
            row = {"dataset": dataset, "model": model, "run_id": run_id, "fid": float(pooled)}
            for view in range(NUM_VIEWS):
                gt_feats = view_features(gt_dir, view, f"{args.work_dir}/features/{dataset}_gt_view{view}.npy", extractor, device)
                gen_feats = view_features(gen_dir, view, f"{args.work_dir}/features/{dataset}_{model}_view{view}.npy", extractor, device)
                boot = [
                    fid_from_features(gt_feats[rng.randint(0, len(gt_feats), size=len(gt_feats))],
                                      gen_feats[rng.randint(0, len(gen_feats), size=len(gen_feats))])
                    for _ in range(args.n_boot)
                ]
                row[f"fid_view{view}"] = fid_from_features(gt_feats, gen_feats)
                row[f"fid_view{view}_bootstrap_sd"] = float(np.std(boot, ddof=1))
                row[f"n_frames_view{view}"] = len(gen_feats)
            print(f"{dataset} / {model}: FID {row['fid']:.4f} | view0 {row['fid_view0']:.4f} "
                  f"+- {row['fid_view0_bootstrap_sd']:.4f} | view1 {row['fid_view1']:.4f} "
                  f"+- {row['fid_view1_bootstrap_sd']:.4f}", flush=True)
            rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
