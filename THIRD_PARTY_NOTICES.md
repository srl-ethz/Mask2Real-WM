# Third-party notices

This repository is released under the MIT License (see [LICENSE](LICENSE)). It builds on the
following work.

## Ctrl-World

The code is a fork of [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World)
(Guo, Shi, Chen, Finn: *Ctrl-World: A Controllable Generative World Model for Robot
Manipulation*, [arXiv:2510.10125](https://arxiv.org/abs/2510.10125)), released under the MIT
License, Copyright (c) 2025 Tsinghua University. That copyright line is kept in
[LICENSE](LICENSE).

## diffusers

Two files are adapted from [diffusers](https://github.com/huggingface/diffusers)
(Copyright The HuggingFace Team, Apache License 2.0) and have been modified:

- `models/pipeline_stable_video_diffusion.py`
- `models/unet_spatio_temporal_condition.py`

A copy of the Apache License 2.0 is in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

## Stable Video Diffusion

All models are initialized from the
[Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
image-to-video weights by Stability AI. The weights are not part of this repository; the code
downloads them from the Hugging Face Hub, where you have to accept their license first.

The released Mask2Real-WM checkpoints are derivative works of those weights. They are
therefore distributed under the Stability AI Community License, not under the MIT License of
this code. The license text and the required notice are distributed together with the
checkpoints.

Powered by Stability AI.

## CLIP

The models load the text encoder of
[CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) by OpenAI
([github.com/openai/CLIP](https://github.com/openai/CLIP), MIT License) from the Hugging Face
Hub at run time. The released checkpoints contain a copy of its weights.
