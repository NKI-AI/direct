"""Tests for opt-in training options: weight EMA, cosine LR schedule, loader timeout."""

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from direct.checkpointer import load_ema_weights
from direct.config.defaults import TrainingConfig
from direct.data.lr_scheduler import WarmupCosineLR
from direct.engine import Engine


class _Engine(Engine):
    """Concrete engine with no-op abstract methods, enough to test EMA plumbing."""


_Engine.__abstractmethods__ = frozenset()


def _engine(ema_decay):
    cfg = OmegaConf.create({"training": {"ema_decay": ema_decay, "dataloader_timeout": 600}})
    return _Engine(cfg, nn.Linear(3, 2), device="cpu", forward_operator=None, backward_operator=None)


def test_training_config_defaults_keep_old_behaviour():
    cfg = TrainingConfig()
    assert cfg.lr_scheduler == "multistep"
    assert cfg.ema_decay is None
    assert cfg.dataloader_timeout == 120


def test_ema_disabled_is_noop():
    engine = _engine(None)
    engine._setup_ema()
    assert engine.ema_model is None
    before = engine.model.weight.detach().clone()
    with engine._ema_weights():
        assert torch.equal(engine.model.weight, before)


def test_ema_tracks_and_swaps_weights():
    engine = _engine(0.5)
    engine._setup_ema()
    assert engine.ema_model is not None
    engine.ema_model.update_parameters(engine.model)  # first update copies the weights
    with torch.no_grad():
        engine.model.weight.add_(1.0)
    engine.ema_model.update_parameters(engine.model)
    trained = engine.model.weight.detach().clone()
    ema = engine.ema_model.module.weight.detach().clone()
    torch.testing.assert_close(ema, trained - 0.5)  # 0.5 * old + 0.5 * new
    with engine._ema_weights():
        torch.testing.assert_close(engine.model.weight, ema)
    torch.testing.assert_close(engine.model.weight, trained)


def test_invalid_ema_decay_rejected():
    engine = _engine(1.5)
    with pytest.raises(ValueError, match="ema_decay"):
        engine._setup_ema()


def test_dataloader_timeout_from_config():
    assert _engine(None)._dataloader_timeout() == 600


def test_load_ema_weights_from_checkpoint(tmp_path):
    model = nn.Linear(3, 2)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9))
    with torch.no_grad():
        ema.module.weight.fill_(7.0)
    path = tmp_path / "model_10.pt"
    torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, path)

    fresh = nn.Linear(3, 2)
    assert load_ema_weights(fresh, path)
    assert torch.all(fresh.weight == 7.0)

    torch.save({"model": model.state_dict()}, tmp_path / "model_11.pt")
    assert not load_ema_weights(fresh, tmp_path / "model_11.pt")


def test_cosine_schedule_decays_to_zero():
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = WarmupCosineLR(optimizer, max_iters=100, warmup_factor=1 / 3, warmup_iterations=10)
    lrs = []
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs[50] < 0.6
    assert lrs[-1] < 1e-3
