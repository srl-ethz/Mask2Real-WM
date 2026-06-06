# Mask2Real-WM — Project Website

Project page for **"Mask2Real-WM: Segmentation Masks as a Sim-to-Real Bridge for Controllable Dexterous World Models"** (CoRL 2025).

A two-stage action-conditioned world model for dexterous manipulation that decouples
**dynamics** (segmentation-mask prediction, pretrained on >50 h of simulation) from
**rendering** (RGB, trained on <2.5 h of real data), using segmentation space as a
sim-to-real bridge for 23-DoF control.

🌐 **Live site:** `https://<your-username>.github.io/<repo-name>/`

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

1. Create a new, empty GitHub repository.
2. From inside this folder:
   ```bash
   git init
   git add .
   git commit -m "Add Mask2Real-WM project website"
   git branch -M main
   git remote add origin git@github.com:<your-username>/<repo-name>.git
   git push -u origin main
   ```
3. In the repo: **Settings → Pages → Build and deployment → Source: Deploy from a branch**,
   pick `main` / `root`, and save. The site goes live at the URL above in a minute or two.

> Everything is static — no build step. The `.nojekyll` file ensures the `demo/` folder
> and all assets are served verbatim.

## Notes

- Update the **Code** link in `index.html` (`id="code-link"`) once the code repo is public.
- The 3D demo loads Three.js from a CDN, so it needs an internet connection to render the hand.
- Total size is ~95 MB (mostly demo meshes and per-DOF videos); all files are well under
  GitHub's 100 MB per-file limit.
