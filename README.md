# Mask2Real-WM — Project Website

Project page for **"Mask2Real-WM: Controllable Dexterous World Models via Segmentation Masks as a Sim-to-Real Bridge"** (preprint, 2026).

A two-stage action-conditioned world model for dexterous manipulation that decouples
**dynamics** (segmentation-mask prediction, pretrained on >50 h of simulation) from
**rendering** (RGB, trained on <2.5 h of real data), using segmentation space as a
sim-to-real bridge for 23-DoF control.

🌐 **Live site:** <https://srl-ethz.github.io/Mask2Real-WM/>

This repository has two unrelated histories: the branch `master` is the website (this folder), and the
branch `main` is the code release. Work on the website only on `master`, and never merge `main` into it.

## Contents

```
index.html          # Project page (abstract, method, results, citation)
paper.pdf           # Full paper
demo/               # Self-contained interactive DOF-controllability demo (Three.js)
  index.html        #   3D ORCA hand viewer + per-DOF prediction videos
  scene.json        #   Mesh transforms / DOF mapping
  meshes/           #   STL meshes for the hand + arena
  videos/           #   Per-DOF prediction videos for each model & sample
assets/
  img/              # Figures (teaser, method, results, qualitative)
  video/            # Long-horizon rollout comparison videos
.nojekyll           # Tells GitHub Pages to serve files as-is (don't run Jekyll)
```

## The interactive demo

Embedded in the main page (and openable full-screen at `demo/index.html`), the demo lets
you click any of the 23 degrees of freedom on a 3D model of the ORCA hand and watch the
world model's prediction when that single action component is perturbed by a sinusoid.
You can compare four models — **WM + LoRA (ours)**, **WM Mid-train**, **WM Real-Only**,
and the monolithic **Baseline** — across multiple evaluation samples.

## Publishing to GitHub Pages

GitHub Pages serves the `master` branch of `srl-ethz/Mask2Real-WM` at the URL above.

1. Clone only the website branch:
   ```bash
   git clone --single-branch --branch master git@github.com:srl-ethz/Mask2Real-WM.git mask2real-website
   ```
2. Edit, commit, and push to `master`. A push publishes immediately.
3. In the repo: **Settings → Pages → Build and deployment → Source: Deploy from a branch**,
   pick `master` / `root`, and save.

> Everything is static — no build step. The `.nojekyll` file ensures the `demo/` folder
> and all assets are served verbatim.

## Notes

- The 3D demo loads Three.js from a CDN, so it needs an internet connection to render the hand.
- Total size is ~95 MB (mostly demo meshes and per-DOF videos); all files are well under
  GitHub's 100 MB per-file limit.
