"""Export the per-sample fidelity metrics of the evaluation runs from W&B to CSV.

For every (dataset, model) run in the runs spec this downloads the two tables that
scripts/inference_wm1_to_wm2.py logs, <prefix>/per_sample_view_metrics (one row per sample and
camera view) and <prefix>/metrics_over_ar_steps (one row per sample, view and autoregressive
step), and writes them to <out_dir>/per_sample/<dataset>/<model>.csv and
<out_dir>/ar_steps/<dataset>/<model>.csv. build_fidelity_tables.py reads only these CSVs.

Example:
    python scripts/paper/export_fidelity_metrics.py --entity <wandb_entity> \
        --runs scripts/paper/fidelity_runs.json --out_dir results/fidelity
"""

import argparse
import json
import os
import tempfile

import pandas as pd
import wandb


def fetch_table(run, key: str, download_dir: str) -> pd.DataFrame:
    ref = dict(run.summary)[key]
    run.file(ref["path"]).download(root=download_dir, replace=True)
    with open(os.path.join(download_dir, ref["path"])) as f:
        data = json.load(f)
    return pd.DataFrame(data["data"], columns=data["columns"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entity", required=True, help="W&B entity that owns the runs")
    parser.add_argument("--project", default=None, help="W&B project (default: the one in the runs spec)")
    parser.add_argument("--runs", default="scripts/paper/fidelity_runs.json")
    parser.add_argument("--out_dir", default="results/fidelity")
    args = parser.parse_args()

    with open(args.runs) as f:
        spec = json.load(f)
    project = args.project or spec["wandb_project"]
    api = wandb.Api()

    with tempfile.TemporaryDirectory() as download_dir:
        for dataset, dataset_spec in spec["datasets"].items():
            for model, run_id in dataset_spec["runs"].items():
                prefix = spec["models"][model]["prefix"]
                run = api.run(f"{args.entity}/{project}/{run_id}")
                print(f"{dataset} / {model}: run {run_id} ({run.name})")
                for key, subdir in (("per_sample_view_metrics", "per_sample"), ("metrics_over_ar_steps", "ar_steps")):
                    table = fetch_table(run, f"{prefix}/{key}", download_dir)
                    out_path = os.path.join(args.out_dir, subdir, dataset, f"{model}.csv")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    # Media columns (video/image references) are W&B-internal paths; keep metrics only.
                    table = table[[c for c in table.columns if not isinstance(table[c].iloc[0], dict)]]
                    table.to_csv(out_path, index=False)
                    print(f"  {len(table)} rows -> {out_path}")


if __name__ == "__main__":
    main()
