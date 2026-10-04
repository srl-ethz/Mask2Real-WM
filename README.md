# Mask2Real-WM: Controllable Dexterous World Models via Segmentation Masks as a Sim-to-Real Bridge

Riccardo O. Feingold, Davide Liconti, Chenyu Yang, Robert K. Katzschmann<br>
Soft Robotics Lab, ETH Zurich

[Paper](https://arxiv.org/abs/2607.04546) ·
[Project page](https://srl-ethz.github.io/Mask2Real-WM/) ·
[Checkpoints](https://huggingface.co/riccardofeingold/Mask2Real-WM)

Mask2Real-WM is an action-conditioned video world model for a dexterous robot hand (an ORCA
hand on a Franka arm, seen from a side camera and a wrist camera). Instead of predicting RGB
video directly from actions, it splits the prediction into two stages:

- **WM1** predicts a future *segmentation-mask* video (hand and object) from the past frames
  and the 23-dimensional action sequence.
- **WM2** renders the future *RGB* video, conditioned on that mask video through a ControlNet
  branch and on the actions.

WM1 can therefore be pretrained on simulated masks and finetuned on real data. The repository
also contains the monolithic baselines that predict RGB directly, and everything used to
evaluate the models: video fidelity, controllability (LLM judge and human A/B study against
simulated ground truth), a sine-sweep rating study and the hand-mask IoU of WM1.

The code is a fork of [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World) and all models
start from the [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
weights.

## Models

| Model | Stage | Training | Config | Checkpoint folder |
|---|---|---|---|---|
| Cascade-R | WM1 | real data only | [`wm1_cascade_r.yaml`](experiments/mask2real/wm1_cascade_r.yaml) | `wm1_cascade_r` (step 15,000) |
| Cascade-S | WM1 | simulation only | [`wm1_cascade_s.yaml`](experiments/mask2real/wm1_cascade_s.yaml) | `wm1_cascade_s` (step 55,000) |
| Cascade-SR | WM1 | simulation, then LoRA on real data | [`wm1_cascade_sr.yaml`](experiments/mask2real/wm1_cascade_sr.yaml) | `wm1_cascade_sr` (LoRA step 45,000) |
| all cascades | WM2 | real data, ControlNet + UNet LoRA | [`wm2.yaml`](experiments/mask2real/wm2.yaml) | `wm2` (step 70,000) |
| Mono-R | single | real data only | [`mono_r.yaml`](experiments/mask2real/mono_r.yaml) | `mono_r` (step 22,000) |
| Mono-SR | single | simulation, then LoRA on real data | [`mono_sr.yaml`](experiments/mask2real/mono_sr.yaml) | `mono_sr_57500`, `mono_sr_58000` |
| Mono-SR, stage 1 | single | simulation only | [`mono_s_base.yaml`](experiments/mask2real/mono_s_base.yaml) | `mono_s_base` (step 55,000) |

The three cascades share one WM2 and differ only in WM1. Two Mono-SR checkpoints are released
because the evaluations used two: step 57,500 for the controllability evaluation and step
58,000 for the video-fidelity tables and the sine-sweep study. `mono_r` is the checkpoint of
the controllability evaluation and the sine-sweep study.

All models work on two camera views (0: side camera, 1: wrist camera) at 135x240 and 5 fps,
take 5 history frames and predict 5 future frames per step; longer videos are rolled out
autoregressively. The 23 action dimensions are the end-effector pose (x, y, z, roll, pitch,
yaw) and 17 hand joints (wrist, four thumb joints and three joints for each other finger).

## Getting started

### 1. Install

Tested with Python 3.11, PyTorch 2.7.1 and CUDA 12.6 on Linux. `ffmpeg` has to be installed;
the scripts read and write videos through it.

```bash
git clone --single-branch --branch main https://github.com/srl-ethz/Mask2Real-WM.git
cd Mask2Real-WM
scripts/setup_env.sh            # creates the conda environment "mask2real-wm" and checks the imports
conda activate mask2real-wm
python -m pytest                # unit tests, no GPU or data needed
```

Or install the pinned dependencies into an environment of your own with
`pip install -r requirements.txt` (`requirements-dev.txt` adds pytest).

The models are built from the Stable Video Diffusion and CLIP weights on the Hugging Face Hub,
which are downloaded on first use. If the download of Stable Video Diffusion is refused, accept
its license on the [model page](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
and log in with `hf auth login`.

Optional settings (Hugging Face cache, W&B mode, the API key for the LLM judge) go into a
`.env` file in the repository root; see [`.env.example`](.env.example).

### 2. Download the checkpoints

The checkpoints are hosted on the Hugging Face Hub at
[riccardofeingold/Mask2Real-WM](https://huggingface.co/riccardofeingold/Mask2Real-WM), about
72 GB for all eight:

```bash
hf download riccardofeingold/Mask2Real-WM --local-dir checkpoints
```

To fetch a single model, add for example `--include "mono_sr_58000/*"`. Each folder
`checkpoints/<name>/` holds

- `model.safetensors`: the full model in one file. The LoRA-stage checkpoints already include
  their base model and adapters.
- `stat.json`: the action statistics this model was trained with. Always pass it to the model
  it belongs to (`--wm1_dataset_stat_path`, `--wm2_dataset_stat_path`,
  `--baseline_wm_dataset_stat_path`); several models were trained with the simulation
  statistics, not with those of the evaluation dataset.
- `resolved_config.yaml`: the fully resolved training configuration, for reference.

The checkpoints are derivatives of Stable Video Diffusion and are released under the Stability
AI Community License, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### 3. Get the data

The datasets of the paper are not public yet. The scripts expect them under `datasets/` in the
repository root:

```
datasets/
  2026-03-14T13-34-49/large_real_dataset_5fps_135_240/      real-world training and validation data
  2026-04-05T20-09-04/processed_OOD_combined_all4_5fps/     out-of-distribution evaluation data
```

The sample lists and normalization statistics that belong to them are already part of this
repository under [`dataset_meta_info/`](dataset_meta_info/). The out-of-distribution set has
four categories (`no_cube`, `cube`, `duck`, `rotated_arena`), each with its own validation
list `processed_OOD_combined_all4_5fps_val_<category>`.

A dataset consists of one folder with the episodes and one folder with its meta-information:

```
datasets/<stamp>/<name>/
  annotation/<episode>.json                        actions and file paths of one episode
  videos/<episode>/<view>.mp4                      RGB video at 135x240 and 5 fps
  segmentation_videos/<episode>/<view>.mp4         mask video: hand green, object blue, background black
  latent_videos/<episode>/<view>.pt                SVD-VAE latents of the RGB video
  latent_segmentation_videos/<episode>/<view>.pt   SVD-VAE latents of the mask video
dataset_meta_info/<stamp>/<name>/
  train_sample.json, val_sample.json               one {"episode_id", "frame_ids"} entry per start frame
  stat.json                                        1st and 99th percentile of every action dimension
```

From an annotation file the loader reads the end-effector pose
(`observation.state.cartesian_position`, 6 values per frame) and the hand joint commands
(`action.hand_joint_position`, 17 values per frame), which together form the action, and the
paths of the four kinds of videos.

To prepare recordings of your own, `scripts/prepare_dataset.sh <converted_dataset_dir>
<output_root>` resamples and resizes the videos, encodes them with the SVD VAE and writes the
sample lists and `stat.json`. The input folder needs `annotation/<episode>.json` and, per
episode, the RGB and segmentation videos of every view.

### 4. Run a model

`scripts/inference_wm1_to_wm2.py` rolls out a monolithic model or a cascade (WM1, then WM2) on
dataset samples and logs PSNR, SSIM, LPIPS and the videos to Weights & Biases. With
`WANDB_MODE=offline` the run stays in `wandb/` and can be uploaded later with `wandb sync`.

A monolithic model on 24 validation samples of the real-world set (about 20 minutes on an RTX
3090):

```bash
WANDB_MODE=offline python scripts/inference_wm1_to_wm2.py \
  --baseline_wm_model_ckpt_path checkpoints/mono_sr_58000/model.safetensors \
  --baseline_wm_config_path experiments/mask2real/mono_sr.yaml \
  --baseline_wm_dataset_stat_path checkpoints/mono_sr_58000/stat.json \
  --dataset_root_path datasets/2026-03-14T13-34-49 \
  --dataset_names large_real_dataset_5fps_135_240 \
  --dataset_meta_info_path dataset_meta_info/2026-03-14T13-34-49 \
  --mode val --inference_mode autoregressive --ar_num_steps 10 --seed 0 \
  --num_of_samples_for_inference 24 --validation_batch_size 24 \
  --baseline_wm_num_inference_steps 50
```

It ends with the averages over the samples; on an RTX 3090 this command gives PSNR 19.12, SSIM
0.635 and LPIPS 0.172.

A cascade takes a WM1 and the WM2 instead of the `--baseline_wm_*` arguments:

```bash
WANDB_MODE=offline python scripts/inference_wm1_to_wm2.py \
  --wm1_model_ckpt_path checkpoints/wm1_cascade_sr/model.safetensors \
  --wm1_config_path experiments/mask2real/wm1_cascade_sr.yaml \
  --wm1_dataset_stat_path checkpoints/wm1_cascade_sr/stat.json \
  --wm2_model_ckpt_path checkpoints/wm2/model.safetensors \
  --wm2_config_path experiments/mask2real/wm2.yaml \
  --wm2_dataset_stat_path checkpoints/wm2/stat.json \
  --dataset_root_path datasets/2026-03-14T13-34-49 \
  --dataset_names large_real_dataset_5fps_135_240 \
  --dataset_meta_info_path dataset_meta_info/2026-03-14T13-34-49 \
  --mode val --inference_mode autoregressive --ar_num_steps 10 --seed 0 \
  --num_of_samples_for_inference 24 \
  --wm1_num_inference_steps 50 --wm2_num_inference_steps 50
```

Useful options:

- `--debug` runs a few samples only.
- `--ar_num_steps` sets the length of the rollout: every step adds 5 frames (1 s).
- For an out-of-distribution category, point the `--dataset_*` arguments at the OOD set and
  add `--dataset_meta_info_name processed_OOD_combined_all4_5fps_val_<category>`.

GPU memory: a monolithic model runs on a 24 GB GPU. A cascade needs more than 24 GB in this
script, which keeps WM1 and WM2 on the GPU at the same time. The controllability and
sine-sweep scripts can hold one of the two at a time (`--sequential_wm1_wm2_loading`); the
controllability rollouts run on a 24 GB GPU this way, and both scripts save their videos as
local `.mp4` files (see [Evaluation](#evaluation)).

## Training

All stages use the same script; the config selects the model.

```bash
# WM1 on real data only (Cascade-R)
accelerate launch scripts/train_wm.py --config experiments/mask2real/wm1_cascade_r.yaml

# WM2, shared by all cascades (trained with the simulation action statistics)
accelerate launch scripts/train_wm.py --config experiments/mask2real/wm2.yaml \
  --dataset_stat_path dataset_meta_info/2026-05-19T03-28-53/SIM_PRETRAINING_DATA_ALL_lerobot_5fps/stat.json

# WM1 of Cascade-SR: LoRA finetuning of checkpoints/wm1_cascade_s on real data
accelerate launch scripts/train_wm.py --config experiments/mask2real/wm1_cascade_sr.yaml \
  --dataset_stat_path dataset_meta_info/2026-05-19T03-28-53/SIM_PRETRAINING_DATA_ALL_lerobot_5fps/stat.json

# Monolithic baselines: real data only, and LoRA finetuning of checkpoints/mono_s_base
accelerate launch scripts/train_wm.py --config experiments/mask2real/mono_r.yaml
accelerate launch scripts/train_wm.py --config experiments/mask2real/mono_sr.yaml
```

Checkpoints are written to `model_ckpt/<tag>/checkpoint-<step>.pt`. The two simulation stages
(`wm1_cascade_s.yaml`, `mono_s_base.yaml`) need the simulation datasets, which are not part of
the release; their checkpoints are, so the real-data stages above can be rerun.

Configs live in [`experiments/`](experiments/) and inherit from a base config through
`experiment.base_config`. [`utils/config_loader.py`](utils/config_loader.py) merges the two
and maps the result onto the `wm_orca_args` dataclass in [`config.py`](config.py), which holds
the default of every field. A key that is not a field of `wm_orca_args` is skipped with a
warning. To train on a dataset of your own, copy one of the configs and change its `dataset`
block and `training.tag`.

## Evaluation

| Evaluation | Entry point | Guide |
|---|---|---|
| Video fidelity (PSNR, SSIM, LPIPS, FVD, FID) | [`scripts/paper/run_fidelity_eval.sh`](scripts/paper/run_fidelity_eval.sh) | [docs/reproducing_paper_results.md](docs/reproducing_paper_results.md) |
| Controllability (LLM judge, human A/B study) | [`scripts/paper/run_controllability_rollouts.sh`](scripts/paper/run_controllability_rollouts.sh) | [docs/controllability_eval.md](docs/controllability_eval.md) |
| WM1 hand-mask IoU | [`scripts/compute_wm1_controllability_seg_iou.py`](scripts/compute_wm1_controllability_seg_iou.py) | [docs/reproducing_paper_results.md](docs/reproducing_paper_results.md) |
| Sine-sweep rating study | [`scripts/paper/run_sine_sweeps.sh`](scripts/paper/run_sine_sweeps.sh) | [docs/reproducing_paper_results.md](docs/reproducing_paper_results.md) |
| WM2 with ground-truth masks | [`scripts/inference_wm_one_stage.py`](scripts/inference_wm_one_stage.py) | [docs/wm2_oracle_eval.md](docs/wm2_oracle_eval.md) |

The launch scripts in `scripts/paper/` run with the settings of the paper; the fidelity and
controllability scripts take `DEBUG=1` for a quick test. Every Python script documents its
arguments with `--help`.

## Repository layout

```
config.py                  default values of every configuration field
experiments/               base configs and the configs of the released models
models/                    world model, pipelines and the UNet with action conditioning
dataset/                   dataset loader
dataset_example/           latent extraction for new recordings
dataset_meta_info/         sample lists and normalization statistics
scripts/                   training, inference and evaluation
scripts/paper/             launch scripts and table builders for the paper's evaluations
docs/                      evaluation guides
tests/                     unit tests
```

## License and acknowledgements

The code is released under the MIT License ([LICENSE](LICENSE)). It builds on
[Ctrl-World](https://github.com/Robert-gyj/Ctrl-World) (Guo et al.,
[arXiv:2510.10125](https://arxiv.org/abs/2510.10125)) and contains two files adapted from
[diffusers](https://github.com/huggingface/diffusers); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The released checkpoints are derived from
Stable Video Diffusion and are licensed under the Stability AI Community License.

Powered by Stability AI.

## Citation

```bibtex
@article{feingold2026mask2realwm,
  title   = {Mask2Real-WM: Controllable Dexterous World Models via Segmentation Masks as a Sim-to-Real Bridge},
  author  = {Feingold, Riccardo O. and Liconti, Davide and Yang, Chenyu and Katzschmann, Robert K.},
  journal = {arXiv preprint arXiv:2607.04546},
  year    = {2026}
}
```
