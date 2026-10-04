import torch
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class wm_orca_args:
    real_data: bool = False
    experiment_name: str = "default_experiment"
    experiment_description: str = "A default experiment configuration"
    base_config: str = None
    ########################### training args ##############################
    # model paths
    svd_model_path: str = "stabilityai/stable-video-diffusion-img2vid"
    clip_model_path: str = "openai/clip-vit-base-patch32"
    ckpt_path: str = None
    pi_ckpt: str = None
    # Optional initialization from official Ctrl-World pretrained weights
    load_ctrl_world_pretrained: bool = False
    ctrl_world_pretrained_path: str = "yjguo/Ctrl-World"
    ctrl_world_pretrained_strict: bool = False
    ctrl_world_pretrained_strict_require_all_keys: bool = False
    ctrl_world_pretrained_load_mode: str = "video_predictor"  # "video_predictor" or "all"
    ctrl_world_pretrained_action_copy_dims: int = 9
    ctrl_world_pretrained_init_controlnet_from_unet: bool = False

    # dataset parameters
    # raw data
    dataset_root_path: str = "dataset_example"
    # NOTE: you can combine multiple datasets by using '+' to separate them
    dataset_names: str = 'orca_dataset'
    # meta info
    dataset_meta_info_path: str = 'dataset_meta_info'
    dataset_meta_info_name: str = None
    dataset_stat_path: str = None
    wm1_dataset_stat_path: str = None
    wm2_dataset_stat_path: str = None
    baseline_wm_dataset_stat_path: str = None
    dataset_cfgs: str = dataset_names
    prob: List[float] = field(default_factory=lambda: [1.0])
    exclude_episode_ids_by_dataset: Dict[str, List[int]] = field(default_factory=dict)
    balanced_validation_sampling: bool = True
    annotation_name: str = 'annotation' #'annotation_all_skip1'
    original_fps: int = 50
    latent_original_fps: int = 10 # NOTE: this is the fps of the latent videos, not the original fps of the dataset
    num_workers: int = 4
    max_num_samples: int = 13000
    max_num_samples_for_validation: int = 100
    down_sample: int = 5 # NOTE: is the same value used to skip rgb frames in extract_latent_orca. Only the states are already downsampled during the extract_latent_orca process
    skip_step: int = 1 # defines how many past frames we skip => defines what is put into history
    use_hand_mask: bool = False
    use_only_hand_actions: bool = False
    use_only_ee_pose_actions: bool = False
    swap_abd_with_mcp: bool = False
    use_average_scalar_hand_action: bool = False

    min_stride: int = 1
    max_stride: int = 2
    stride_factor: int = 1

    encode_segmentation_with_svd: bool = False
    use_vae_roundtrip_for_controlnet: bool = False  # decode latent_segmentation_videos → pixel space before the ControlNet CNN
    use_latent_segmentation_for_controlnet: bool = False
    use_vis_seg_actions_for_controlnet: bool = False
    predicted_datatype: str = "latent_segmentation_videos" # "latent_segmentation_videos" or "latent_videos" or "segmentation_videos" or "videos"
    downsample_method_for_segmentation: str = "nearest" # "nearest" or "bilinear" or ""
    concatenate_latent: str = None # "interpolated_downsampled_segmentation" or "latent_segmentation_videos" or "latent_videos" or "segmentation_videos" or "videos"

    # action encoder parameters
    action_encoder: str = "dino_visual"
    hub_dir: str = None  # torch.hub cache for the DINOv2 action encoders; None uses the torch default
    dinov2_size: int = 224

    # compression rate of VAE
    vae_compression_rate: int = 8

    # conditional
    num_views: int = 1   # number of camera views used during training
    only_wrist_view: bool = False
    # logs parameters
    debug: bool = False
    tag: str = 'orca_dataset'
    output_dir: str = f"model_ckpt/{tag}"
    wandb_run_name: str = tag
    wandb_project_name: str = "orca_example"
    wandb_video_display_fps: int = 1 # NOTE: this is the fps of the video displayed in wandb, not the fps of the video used for training

    # training parameters
    hand_weight: float = 2.5
    learning_rate: float = 1e-5 # 5e-6
    lr_scheduler: str = "linear"  # diffusers scheduler: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
    lr_warmup_steps: int = None   # if None, defaults to 3% of max_train_steps
    lr_num_cycles: float = 1.0    # used by cosine_with_restarts/polynomial schedulers in diffusers
    lr_power: float = 1.0         # used by polynomial scheduler in diffusers
    lr_range_test: bool = False
    lr_range_test_start_lr: float = 1e-7
    lr_range_test_end_lr: float = 1.0
    lr_range_test_num_steps: int = 1000
    gradient_accumulation_steps: int = 1
    mixed_precision: str = 'fp16'
    train_batch_size: int = 4
    shuffle: bool = True
    num_train_epochs: int = 100
    max_train_steps: int = 100010
    checkpointing_steps: int = 10000
    validation_steps: int = 2500
    max_grad_norm: float = 1.0
    
    freeze_temporal_transformer_blocks: bool = False
    use_controlnet_conditioning: bool = False
    controlnet_conditioning_scale: float = 1.0
    # Conditioning dropout (active during training only). Applied when both action
    # conditioning and ControlNet conditioning are enabled.
    cond_dropout_mask_only_prob: float = 0.10
    cond_dropout_action_only_prob: float = 0.10
    cond_dropout_both_prob: float = 0.05
    use_unet_lora: bool = False
    finetune_from_checkpoint_before_lora: bool = False
    finetune_lora_ckpt_path: str = None
    unet_lora_rank: int = 8
    unet_lora_alpha: float = 8.0

    starting_global_step: int = 0
    # for val
    video_num: int = 4
    validation_batch_size: int = 64
    max_patience: int = 5 # number of epochs to wait before stopping training

    ############################ model args ##############################

    # model parameters
    # defines how much motion there is in the videos: Value between 0 to 255
    # the motion bucket id to use for the generated video. This can be used to control the motion of the generated video. Increasing the motion bucket id increases the motion of the generated video.
    motion_bucket_id: int = 127
    fps: int = 5
    guidance_scale: float = 2 #7.5 #7.5 #7.5 #3.0
    num_inference_steps: int = 50
    decode_chunk_size: int = 5
    width: int = 256
    height: int = 256
    # num history and num future predictions
    num_frames: int = 5
    num_history: int = 5
    action_dim: int = 23 # 6 for xyz+rpy EE pose + 17 for hand joint pos
    relative_pose_dims: int = 6
    text_cond: bool = False
    frame_level_cond: bool = True
    his_cond_zero: bool = False
    action_encoder_hidden_dims: List[int] = field(default_factory=lambda: [1024])
    dtype: torch.dtype = torch.bfloat16 # [torch.float32, torch.bfloat16] # during inference, we can use bfloat16 to accelerate the inference speed and save memory

    num_channels_concatenate: int = 0



    ########################### rollout args ############################
    # policy
    task_type: str = "pickplace" # choose from ['pickplace', 'towel_fold', 'wipe_table', 'tissue', 'close_laptop','tissue','drawer','stack']
    gripper_max_dict: Dict[str, float] = field(default_factory=lambda: {'replay':1.0, 'pickplace':0.75, 'towel_fold':0.95, 'wipe_table':0.95, 'tissue':0.97, 'close_laptop':0.95,'drawer':0.75,'stack':0.75,})
    ##############################################################################
    policy_type: str = 'pi05' # choose from ['pi05', 'pi0', 'pi0fast']
    action_adapter: str = 'models/action_adapter/model2_15_9.pth' # adapat action from joint vel to cartesian pose
    pred_step: int = 5 # predict 5 steps (1s) action each time
    policy_skip_step: int = 2 # horizon = (pred_step-1) * policy_skip_step
    interact_num: int = 12 # number of interactions (each interaction contains pred_step steps)

    # wm
    data_stat_path: str = 'dataset_meta_info/orca_dataset/stat.json'
    val_model_path: str = ckpt_path
    history_idx = [0,0,-12,-9,-6,-3]

    # save
    save_dir: str = 'synthetic_traj'
    # select different traj for different tasks
    def __post_init__(self):
        # Per-task gripper max
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.75)
        # Default task_name
        self.task_name = f"Rollouts_interact_pi"
        if self.task_type == "replay":
            self.task_name = "Rollouts_replay"

        # Configure per-task eval sets
        if self.task_type == "replay":
            self.val_dataset_dir = "dataset_example/orca_dataset"
            self.val_id = ["899", "18599","199",]
            self.start_idx = [8, 14, 8] * len(self.val_id)
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_replay"

        elif self.task_type == "keyboard":
            self.val_dataset_dir = "dataset_example/orca_dataset"
            self.val_id = ["1799"]
            self.start_idx = [23] * len(self.val_id)
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_keyboard"


        elif self.task_type == "pickplace":
            self.interact_num = 15
            self.val_dataset_dir = "dataset_example/orca_new_setup"
            self.val_id = ['0001','0002','0003']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = [
                "pick up the green block and place in plate",
                "pick up the green block and place in plate",
                "pick up the blue block and place in plate",]

        elif self.task_type == "towel_fold":
            self.interact_num = 15
            self.val_dataset_dir = "dataset_example/orca_new_setup"
            self.val_id =['0004','0005']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["fold the towel"] * len(self.val_id)

        elif self.task_type == "wipe_table":
            self.val_dataset_dir = "dataset_example/orca_new_setup"
            self.val_id = ['0006','0007']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = [
                "move the towel from left to right",
                "move the towel from left to right"
            ]

        elif self.task_type == "tissue":
            self.interact_num = 10
            self.val_dataset_dir = "dataset_example/orca_new_setup"
            self.val_id = ['0008','0009']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["pull one tissue out of the box"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "close_laptop":
            self.val_dataset_dir = "dataset_example/orca_new_setup"
            self.val_id = ['0010','0011']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "stack":
            self.val_dataset_dir = "dataset_example/orca_new_setup"
            self.val_id = ['0012','0013']
            self.start_idx = [5] * len(self.val_id)
            self.instruction = ["stack the blue block on the red block"] * len(self.val_id)
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
