from types import SimpleNamespace
from unittest.mock import patch

import torch

import scripts.inference_wm1_to_wm2 as inference_wm1_to_wm2


class FakeCtrlWorld:
    instances = []

    def __init__(self, args):
        self.use_unet_lora_at_init = args.use_unet_lora
        self.load_calls = []
        self.lora_calls = []
        self.eval_called = False
        FakeCtrlWorld.instances.append(self)

    def load_state_dict(self, state_dict, strict=True):
        self.load_calls.append({"state_dict": state_dict, "strict": strict})
        return [], []

    def enable_unet_lora(self, rank=8, alpha=8.0):
        self.lora_calls.append({"rank": rank, "alpha": alpha})
        return 1

    def eval(self):
        self.eval_called = True
        return self


def _args(**overrides):
    values = {
        "use_unet_lora": True,
        "finetune_from_checkpoint_before_lora": True,
        "finetune_lora_ckpt_path": "lora.pt",
        "unet_lora_rank": 16,
        "unet_lora_alpha": 16.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_inference_loader_loads_base_before_lora_and_then_loads_finetune_lora():
    FakeCtrlWorld.instances = []
    args = _args()
    base_state = {"weight": torch.tensor(1.0)}

    with patch.object(inference_wm1_to_wm2, "CtrlWorld", FakeCtrlWorld), patch.object(
        inference_wm1_to_wm2, "_load_torch_checkpoint", return_value={"model": base_state}
    ), patch.object(
        inference_wm1_to_wm2, "load_finetune_lora_checkpoint", return_value=None
    ) as load_lora:
        model = inference_wm1_to_wm2.load_ctrlworld_model_for_inference(
            args, "base.pt", "WM1"
        )

    assert args.use_unet_lora is True
    assert model.use_unet_lora_at_init is False
    assert model.load_calls == [{"state_dict": base_state, "strict": True}]
    assert model.lora_calls == [{"rank": 16, "alpha": 16.0}]
    assert model.eval_called is True
    load_lora.assert_called_once_with(model, "lora.pt", restore_optimizer=False)


class OrderedFakeCtrlWorld(FakeCtrlWorld):
    def __init__(self, args):
        super().__init__(args)
        self.events = []

    def load_state_dict(self, state_dict, strict=True):
        self.events.append("load")
        return super().load_state_dict(state_dict, strict=strict)

    def enable_unet_lora(self, rank=8, alpha=8.0):
        self.events.append("enable_lora")
        return super().enable_unet_lora(rank=rank, alpha=alpha)


def test_inference_loader_enables_lora_before_loading_self_contained_lora_checkpoint():
    OrderedFakeCtrlWorld.instances = []
    args = _args(finetune_lora_ckpt_path=None)
    full_state = {
        "unet.block.weight": torch.tensor(1.0),
        "unet.block.lora_down.weight": torch.tensor(2.0),
        "unet.block.lora_up.weight": torch.tensor(3.0),
    }

    with patch.object(inference_wm1_to_wm2, "CtrlWorld", OrderedFakeCtrlWorld), patch.object(
        inference_wm1_to_wm2, "_load_torch_checkpoint", return_value={"model": full_state}
    ), patch.object(
        inference_wm1_to_wm2, "load_finetune_lora_checkpoint", return_value=None
    ) as load_lora:
        model = inference_wm1_to_wm2.load_ctrlworld_model_for_inference(
            args, "model.safetensors", "WM1"
        )

    assert model.use_unet_lora_at_init is False
    assert model.events == ["enable_lora", "load"]
    assert model.load_calls == [{"state_dict": full_state, "strict": True}]
    assert model.lora_calls == [{"rank": 16, "alpha": 16.0}]
    load_lora.assert_called_once_with(model, None, restore_optimizer=False)
