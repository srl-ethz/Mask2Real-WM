"""Build the video-fidelity tables and error-accumulation plots from the exported CSVs.

Reads, under --results_dir (see export_fidelity_metrics.py, compute_fid.py and
recompute_metrics_and_visualize_on_wandb.py --fvd_csv):
  per_sample/<dataset>/<model>.csv   one row per sample and camera view
  ar_steps/<dataset>/<model>.csv     one row per sample, view and autoregressive step
  fvd.csv, fid.csv                   dataset-level Frechet distances per (dataset, model);
                                     fvd.csv may also be the per-run file written by --fvd_csv
and writes summary.csv (mean and std over samples and views), tables.tex, fidelity_tables.md
and error_accumulation_{main,supplementary}.{pdf,png}.

Example:
    python scripts/paper/build_fidelity_tables.py --results_dir results/fidelity
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

METRICS = ["psnr", "ssim", "lpips", "mae_img", "mse_img", "mae_lat", "mse_lat"]
HIGHER_BETTER = {"psnr", "ssim"}
METRIC_TEX = {
    "psnr": r"PSNR $\uparrow$", "ssim": r"SSIM $\uparrow$", "lpips": r"LPIPS $\downarrow$",
    "mae_img": r"MAE$_{\text{img}}$ $\downarrow$", "mse_img": r"MSE$_{\text{img}}$ $\downarrow$",
    "mae_lat": r"MAE$_{\text{lat}}$ $\downarrow$", "mse_lat": r"MSE$_{\text{lat}}$ $\downarrow$",
}
METRIC_MD = {"psnr": "PSNR↑", "ssim": "SSIM↑", "lpips": "LPIPS↓", "mae_img": "MAE_img↓",
             "mse_img": "MSE_img↓", "mae_lat": "MAE_lat↓", "mse_lat": "MSE_lat↓"}
PLOT_TITLE = {"psnr": "PSNR (dB) $\\uparrow$", "ssim": "SSIM $\\uparrow$", "lpips": "LPIPS $\\downarrow$",
              "mae_img": "MAE (img) $\\downarrow$", "mse_img": "MSE (img) $\\downarrow$"}
# Display order and plot colors of the paper's models.
MODEL_ORDER = ["mono_r", "mono_sr_58000", "cascade_s", "cascade_sr", "cascade_r"]
MODEL_COLOR = {"mono_r": "#7b4fb3", "mono_sr_58000": "#1a1a1a", "cascade_s": "#3b74c9",
               "cascade_sr": "#3aa15c", "cascade_r": "#d1495b"}


def decimals(metric: str) -> int:
    return 2 if metric == "psnr" else 3


def load(results_dir: str, spec: dict):
    rows, ar = [], {}
    for dataset, dataset_spec in spec["datasets"].items():
        for model in dataset_spec["runs"]:
            per_sample = pd.read_csv(os.path.join(results_dir, "per_sample", dataset, f"{model}.csv"))
            for metric in METRICS:
                rows.append({"dataset": dataset, "model": model, "metric": metric,
                             "mean": per_sample[metric].mean(), "std": per_sample[metric].std(ddof=1),
                             "n": len(per_sample)})
            steps = pd.read_csv(os.path.join(results_dir, "ar_steps", dataset, f"{model}.csv"))
            ar[(dataset, model)] = steps.groupby("ar_step_idx")[METRICS].agg(["mean", "std"])
    summary = pd.DataFrame(rows)
    fvd = load_fvd(os.path.join(results_dir, "fvd.csv"), spec)
    fid = pd.read_csv(os.path.join(results_dir, "fid.csv")).set_index(["dataset", "model"])
    return summary, ar, fvd, fid


def load_fvd(path: str, spec: dict) -> pd.DataFrame:
    """Read fvd.csv, either already keyed by (dataset, model) or as written per run by
    recompute_metrics_and_visualize_on_wandb.py --fvd_csv (run_id, fvd_v00, fvd_v01, ...)."""
    fvd = pd.read_csv(path)
    if "dataset" not in fvd.columns:
        by_run = {run_id: (dataset, model) for dataset, dataset_spec in spec["datasets"].items()
                  for model, run_id in dataset_spec["runs"].items()}
        missing = sorted(set(by_run) - set(fvd["run_id"]))
        if missing:
            raise ValueError(f"{path} has no FVD for runs {missing}")
        fvd = fvd[fvd["run_id"].isin(by_run)].copy()
        fvd["dataset"] = fvd["run_id"].map(lambda run_id: by_run[run_id][0])
        fvd["model"] = fvd["run_id"].map(lambda run_id: by_run[run_id][1])
        fvd = fvd.rename(columns={"fvd_v00": "fvd_view0", "fvd_v01": "fvd_view1"})
    return fvd.set_index(["dataset", "model"])


def best_models(values: dict, higher_better: bool) -> set:
    if len(values) < 2:
        return set()
    target = max(values.values()) if higher_better else min(values.values())
    return {m for m, v in values.items() if v == target}


def build_tables(spec, summary, fvd, fid):
    tex = [r"% Requires: \usepackage{booktabs} \usepackage{graphicx}"]
    md = ["# Video fidelity", "",
          "Mean ± std over samples and camera views. FVD (I3D-R50, 16-frame clips, camera view 0) and FID "
          "(Inception-v3, 8 frames per video and view, pooled over both views) are dataset-level Fréchet "
          "distances. Bold: best value in the column.", ""]
    for dataset, dataset_spec in spec["datasets"].items():
        models = [m for m in MODEL_ORDER if m in dataset_spec["runs"]]
        label = dataset_spec["label"]
        cell = summary[summary.dataset == dataset].set_index(["model", "metric"])
        best = {metric: best_models({m: cell.loc[(m, metric), "mean"] for m in models}, metric in HIGHER_BETTER)
                for metric in METRICS}
        best["fvd"] = best_models({m: fvd.loc[(dataset, m), "fvd_view0"] for m in models}, False)
        best["fid"] = best_models({m: fid.loc[(dataset, m), "fid"] for m in models}, False)

        lines = [r"\begin{table}[t]", r"\centering",
                 f"\\caption{{{label} ({dataset_spec['num_samples']} samples). Mean $\\pm$ std over samples and "
                 "camera views; FVD (I3D, view 0) and FID (Inception-v3) are dataset-level Fr\\'echet distances.}",
                 f"\\label{{tab:fidelity_{dataset}}}", r"\resizebox{\textwidth}{!}{%",
                 r"\begin{tabular}{l" + "c" * (len(METRICS) + 2) + "}", r"\toprule",
                 " & ".join(["Model"] + [METRIC_TEX[m] for m in METRICS] + [r"FVD $\downarrow$", r"FID $\downarrow$"]) + r" \\",
                 r"\midrule"]
        md += [f"## {label} ({dataset_spec['num_samples']} samples)", "",
               "| Model | " + " | ".join(METRIC_MD[m] for m in METRICS) + " | FVD↓ | FID↓ |",
               "|---" * (len(METRICS) + 3) + "|"]
        for model in models:
            name = spec["models"][model]["label"]
            tex_cells, md_cells = [], []
            for metric in METRICS:
                mean, std = cell.loc[(model, metric), "mean"], cell.loc[(model, metric), "std"]
                body = f"{mean:.{decimals(metric)}f} \\pm {std:.{decimals(metric)}f}"
                text = f"{mean:.{decimals(metric)}f}±{std:.{decimals(metric)}f}"
                bold = model in best[metric]
                tex_cells.append(f"$\\mathbf{{{body}}}$" if bold else f"${body}$")
                md_cells.append(f"**{text}**" if bold else text)
            for key, value in (("fvd", fvd.loc[(dataset, model), "fvd_view0"]), ("fid", fid.loc[(dataset, model), "fid"])):
                bold = model in best[key]
                tex_cells.append(f"$\\mathbf{{{value:.2f}}}$" if bold else f"${value:.2f}$")
                md_cells.append(f"**{value:.2f}**" if bold else f"{value:.2f}")
            lines.append(f"{name} & " + " & ".join(tex_cells) + r" \\")
            md.append(f"| {name} | " + " | ".join(md_cells) + " |")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        tex.append("\n".join(lines))
        md.append("")

    md += ["# FVD and FID per camera view", "",
           "View 0: side camera, view 1: wrist camera. FID ± is the SD over 20 bootstrap resamples of the "
           "frame features.", ""]
    for dataset, dataset_spec in spec["datasets"].items():
        md += [f"## {dataset_spec['label']}", "", "| Model | FVD v0 | FVD v1 | FID v0 | FID v1 |",
               "|---|---:|---:|---:|---:|"]
        for model in [m for m in MODEL_ORDER if m in dataset_spec["runs"]]:
            v, f = fvd.loc[(dataset, model)], fid.loc[(dataset, model)]
            md.append(f"| {spec['models'][model]['label']} | {v.fvd_view0:.3f} | {v.fvd_view1:.3f} | "
                      f"{f.fid_view0:.3f} ± {f.fid_view0_bootstrap_sd:.3f} | "
                      f"{f.fid_view1:.3f} ± {f.fid_view1_bootstrap_sd:.3f} |")
        md.append("")
    return "\n\n".join(tex) + "\n", "\n".join(md)


def plot_error_accumulation(spec, ar, metrics, out_base, title):
    datasets = list(spec["datasets"])
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(3.2 * len(datasets), 2.4 * len(metrics)), squeeze=False)
    for row, metric in enumerate(metrics):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            for model in [m for m in MODEL_ORDER if m in spec["datasets"][dataset]["runs"]]:
                stats = ar[(dataset, model)]
                t = (stats.index.values + 1) * 1.0  # 5 frames per step at 5 fps: 1 s per step
                mean, std = stats[(metric, "mean")], stats[(metric, "std")]
                ax.plot(t, mean, color=MODEL_COLOR[model], label=spec["models"][model]["label"], linewidth=1.6)
                ax.fill_between(t, mean - std, mean + std, color=MODEL_COLOR[model], alpha=0.15, linewidth=0)
            if row == 0:
                ax.set_title(spec["datasets"][dataset]["label"], fontsize=9)
            if col == 0:
                ax.set_ylabel(PLOT_TITLE[metric])
            if row == len(metrics) - 1:
                ax.set_xlabel("Rollout time (s)")
            ax.grid(alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODEL_ORDER), bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(title, y=1.01, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results_dir", default="results/fidelity")
    parser.add_argument("--runs", default="scripts/paper/fidelity_runs.json")
    parser.add_argument("--out_dir", default=None, help="Output folder (default: --results_dir)")
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(args.runs) as f:
        spec = json.load(f)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    summary, ar, fvd, fid = load(args.results_dir, spec)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    tex, md = build_tables(spec, summary, fvd, fid)
    with open(os.path.join(out_dir, "tables.tex"), "w") as f:
        f.write(tex)
    with open(os.path.join(out_dir, "fidelity_tables.md"), "w") as f:
        f.write(md)
    plot_error_accumulation(spec, ar, ["psnr", "ssim", "lpips"], os.path.join(out_dir, "error_accumulation_main"),
                            "Rollout error accumulation -- perceptual metrics")
    plot_error_accumulation(spec, ar, ["mae_img", "mse_img"], os.path.join(out_dir, "error_accumulation_supplementary"),
                            "Rollout error accumulation -- pixel-space error")
    print(f"wrote summary.csv, tables.tex, fidelity_tables.md and plots to {out_dir}")


if __name__ == "__main__":
    main()
