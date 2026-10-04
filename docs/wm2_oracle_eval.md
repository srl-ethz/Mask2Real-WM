# WM2 with ground-truth masks

This evaluation measures WM2 on its own. Instead of the masks predicted by WM1, WM2 receives
the ground-truth segmentation of the out-of-distribution dataset (produced with SAM 3 by the
[labeling pipeline](https://github.com/riccardofeingold/sam3_based_labeling_pipeline)) and
predicts the RGB video. It answers how well WM2 turns a correct mask into video, which
separates the errors of WM2 from those of WM1 in the cascade.

## Running it

`scripts/inference_wm_one_stage.py` evaluates a single model on the validation split that its
config points to. The configs in
[`experiments/mask2real/wm2_oracle_sam3/`](../experiments/mask2real/wm2_oracle_sam3/) are
copies of the WM2 training config in which only the dataset block differs: one per category
(150 validation samples each) and `ood_all.yaml` for the first 100 samples of the combined
split.

```bash
for category in no_cube cube duck rotated_arena; do
  python scripts/inference_wm_one_stage.py \
    --model_ckpt_path checkpoints/wm2/model.safetensors \
    --config_path experiments/mask2real/wm2_oracle_sam3/ood_${category}.yaml \
    --mode val --output_path outputs/wm2_oracle/${category} \
    --wandb_project_name mask2real-wm-wm2-oracle --wandb_run_name wm2_oracle_${category}
done
```

Run the categories one after the other; every process loads the checkpoint into main memory
first. Each run predicts 5 future frames from 5 history frames in one shot (not
autoregressively) with 20 denoising steps, logs PSNR, SSIM, LPIPS, MSE and MAE to W&B and saves
`pred_videos.pt` and `gt_videos.pt` in its output folder. Actions are normalized with the
statistics of the evaluated split, which the configs select; the script has no option to
override them.

The LPIPS values below were computed from the saved tensors:

```bash
python scripts/paper/recompute_lpips.py \
  --run no_cube=outputs/wm2_oracle/no_cube --run cube=outputs/wm2_oracle/cube \
  --run duck=outputs/wm2_oracle/duck --run rotated_arena=outputs/wm2_oracle/rotated_arena
```

## Results

WM2 checkpoint at step 70,000, 150 validation samples and two camera views per category:

| Category | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Cube | 21.14 | 0.708 | 0.134 |
| No Cube | 20.49 | 0.690 | 0.163 |
| Duck | 20.38 | 0.685 | 0.179 |
| Rotated Arena | 19.79 | 0.636 | 0.190 |
| Mean of the four | 20.45 | 0.680 | 0.167 |

The first 100 samples of the combined split (`ood_all.yaml`) give PSNR 20.38, SSIM 0.677 and
LPIPS 0.169.
