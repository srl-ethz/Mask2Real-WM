from types import SimpleNamespace
from unittest.mock import patch

import scripts.train_wm as train_wm


class FakeCtrlWorld:
    instances = []

    def __init__(self, args):
        self.use_unet_lora_at_init = args.use_unet_lora
        self.loaded_state_dict = None
        self.load_strict = None
        self.load_calls = []
        self.lora_calls = []
        self._unet_lora_enabled = bool(args.use_unet_lora)
        FakeCtrlWorld.instances.append(self)

    def state_dict(self):
        return {
            "weight": object(),
            "unet.block.lora_down.weight": object(),
            "unet.block.lora_up.weight": object(),
        }

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = state_dict
        self.load_strict = strict
        self.load_calls.append({"state_dict": state_dict, "strict": strict})
        return [], []

    def enable_unet_lora(self, rank=8, alpha=8.0):
        self.lora_calls.append({"rank": rank, "alpha": alpha})
        self._unet_lora_enabled = True
        return 1


def _args(**overrides):
    values = {
        "ckpt_path": "checkpoint.pt",
        "use_unet_lora": True,
        "finetune_from_checkpoint_before_lora": True,
        "finetune_lora_ckpt_path": None,
        "unet_lora_rank": 16,
        "unet_lora_alpha": 16.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_finetune_checkpoint_loads_before_lora_and_ignores_optimizer():
    FakeCtrlWorld.instances = []
    checkpoint = {"model": {"weight": object()}, "optimizer": {"momentum": object()}}

    with patch.object(train_wm, "CtrlWorld", FakeCtrlWorld), patch.object(
        train_wm, "_load_torch_checkpoint", return_value=checkpoint
    ):
        model, optimizer_state = train_wm.build_model_and_maybe_load_checkpoint(_args())

    assert optimizer_state is None
    assert model.use_unet_lora_at_init is False
    assert model.loaded_state_dict is checkpoint["model"]
    assert model.load_strict is True
    assert model.lora_calls == [{"rank": 16, "alpha": 16.0}]


def test_regular_resume_keeps_existing_optimizer_restore_behavior():
    FakeCtrlWorld.instances = []
    checkpoint = {"model": {"weight": object()}, "optimizer": {"momentum": object()}}

    with patch.object(train_wm, "CtrlWorld", FakeCtrlWorld), patch.object(
        train_wm, "_load_torch_checkpoint", return_value=checkpoint
    ):
        model, optimizer_state = train_wm.build_model_and_maybe_load_checkpoint(
            _args(finetune_from_checkpoint_before_lora=False)
        )

    assert optimizer_state is checkpoint["optimizer"]
    assert model.use_unet_lora_at_init is True
    assert model.loaded_state_dict is checkpoint["model"]
    assert model.lora_calls == []


def test_finetune_lora_checkpoint_loads_after_lora_is_enabled():
    FakeCtrlWorld.instances = []
    base_checkpoint = {"model": {"weight": object()}, "optimizer": {"momentum": object()}}
    lora_checkpoint = {
        "model": {
            "unet.block.lora_down.weight": object(),
            "unet.block.lora_up.weight": object(),
        },
        "optimizer": {"lora_momentum": object()},
    }

    with patch.object(train_wm, "CtrlWorld", FakeCtrlWorld), patch.object(
        train_wm, "_load_torch_checkpoint", side_effect=[base_checkpoint, lora_checkpoint]
    ):
        model, optimizer_state = train_wm.build_model_and_maybe_load_checkpoint(
            _args(finetune_lora_ckpt_path="lora.pt")
        )

    assert optimizer_state is lora_checkpoint["optimizer"]
    assert model.use_unet_lora_at_init is False
    assert model.lora_calls == [{"rank": 16, "alpha": 16.0}]
    assert model.load_calls == [
        {"state_dict": base_checkpoint["model"], "strict": True},
        {"state_dict": lora_checkpoint["model"], "strict": False},
    ]


def test_full_finetune_lora_checkpoint_loads_strictly_and_returns_optimizer_state():
    FakeCtrlWorld.instances = []
    model = FakeCtrlWorld(_args(use_unet_lora=True))
    full_lora_state = model.state_dict()
    optimizer_state = {"full_lora_momentum": object()}

    with patch.object(
        train_wm,
        "_load_torch_checkpoint",
        return_value={"model": full_lora_state, "optimizer": optimizer_state},
    ):
        loaded_optimizer_state = train_wm.load_finetune_lora_checkpoint(
            model, "full_lora.pt"
        )

    assert loaded_optimizer_state is optimizer_state
    assert model.load_calls == [{"state_dict": full_lora_state, "strict": True}]
