"""Unit tests for the diffusion model code (CPU, tiny model, no dataset needed).

Run from the repository root:
    python -m pytest Reconstruction/tests -q
"""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Reconstruction.common.config_loader import load_params
from Reconstruction.model.checkpoint import EMA, LATEST, load_checkpoint, save_checkpoint
from Reconstruction.model.diffusion import GaussianDiffusion, cosine_alpha_bar
from Reconstruction.model.objective import compute_loss, foreground_weights
from Reconstruction.model.sampling import sample_missing
from Reconstruction.model.train import learning_rate
from Reconstruction.model.unet import ConditionalUNet

HYPER_FILE: Path = Path(__file__).resolve().parents[1] / "model" / "hyperparameters.toml"
CHANNELS: int = 13
SIZE: int = 32


@pytest.fixture(scope="module")
def hyper() -> SimpleNamespace:
    """Debug hyperparameters (tiny model)."""
    return load_params(HYPER_FILE, "debug")


@pytest.fixture(scope="module")
def model(hyper) -> ConditionalUNet:
    """Tiny U-Net on a 32x32 canvas."""
    torch.manual_seed(0)
    return ConditionalUNet(CHANNELS, SIZE, hyper.model)


def test_debug_overrides_apply_on_top_of_base(hyper):
    base = load_params(HYPER_FILE)
    assert hyper.model.BASE_CHANNELS < base.model.BASE_CHANNELS
    assert hyper.loss.FOREGROUND_WEIGHT == base.loss.FOREGROUND_WEIGHT  # untouched values are kept


def test_unet_output_shapes(model, hyper):
    x = torch.randn(2, CHANNELS, SIZE, SIZE)
    v, logits = model(x, torch.zeros_like(x), torch.tensor([0, 999]))
    assert v.shape == x.shape
    assert logits.shape == (2, len(hyper.model.TYPE_CLASSES))


def test_cosine_schedule_is_monotonic_and_bounded():
    alpha_bar = cosine_alpha_bar(1000, 0.999)
    assert alpha_bar[0] > 0.999 and alpha_bar[-1] < 1e-3
    assert torch.all(alpha_bar[1:] < alpha_bar[:-1])


def test_v_parameterization_round_trip():
    diffusion = GaussianDiffusion(1000, 0.999)
    x0 = torch.rand(3, CHANNELS, 8, 8) * 2 - 1
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 500, 999])
    x_t = diffusion.q_sample(x0, t, noise)
    recovered = diffusion.x0_from_v(x_t, t, diffusion.v_target(x0, t, noise))
    assert torch.allclose(recovered, x0, atol=1e-4)


def test_foreground_weights_cover_dilated_target():
    target = torch.zeros(1, CHANNELS, 9, 9)
    target[0, 4, 4, 4] = 1.0
    weights = foreground_weights(target, weight=8.0, dilation_px=1)
    assert weights[0, 0, 4, 4] == 9.0 and weights[0, 0, 3, 5] == 9.0
    assert weights[0, 0, 0, 0] == 1.0


def test_loss_is_finite_and_backpropagates(model, hyper):
    diffusion = GaussianDiffusion(hyper.diffusion.TIMESTEPS, hyper.diffusion.MAX_BETA)
    incomplete = (torch.rand(2, CHANNELS, SIZE, SIZE) > 0.9).float()
    target = torch.zeros_like(incomplete)
    target[:, 4, 10:14, 10:20] = 1.0
    terms = compute_loss(model, diffusion, incomplete, target, torch.tensor([1, 1]), hyper.loss)
    terms.total.backward()
    assert math.isfinite(float(terms.total.detach()))
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    model.zero_grad(set_to_none=True)


def test_sampling_returns_scores_in_unit_range(model, hyper):
    diffusion = GaussianDiffusion(hyper.diffusion.TIMESTEPS, hyper.diffusion.MAX_BETA)
    incomplete = torch.zeros(2, CHANNELS, SIZE, SIZE)
    generator = torch.Generator().manual_seed(0)
    scores = sample_missing(model.eval(), diffusion, incomplete, 3, True, generator, None)
    assert scores.shape == incomplete.shape and scores.min() >= 0 and scores.max() <= 1
    again = sample_missing(model, diffusion, incomplete, 3, True, torch.Generator().manual_seed(0), None)
    assert torch.equal(scores, again)  # same seed, same samples


def test_learning_rate_warmup_and_cosine_floor(hyper):
    cfg = hyper.train
    assert learning_rate(0, cfg) == pytest.approx(cfg.LEARNING_RATE / cfg.WARMUP_STEPS)
    assert learning_rate(cfg.WARMUP_STEPS, cfg) == pytest.approx(cfg.LEARNING_RATE)
    assert learning_rate(cfg.TOTAL_STEPS, cfg) == pytest.approx(cfg.LEARNING_RATE * cfg.MIN_LR_FRACTION)


def test_checkpoint_pruning_keeps_milestones(tmp_path, model, hyper):
    ema = EMA(model, 0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in (1, 2, 3, 4):
        save_checkpoint(tmp_path, step, 0, model, ema, optimizer, hyper, keep=2, milestone_every=2)
    assert sorted(p.name for p in tmp_path.glob("step_*.pt")) == ["step_0000003.pt", "step_0000004.pt"]
    assert sorted(p.name for p in tmp_path.glob("milestone_*.pt")) == ["milestone_step_0000002.pt", "milestone_step_0000004.pt"]
    milestone = load_checkpoint(tmp_path / "milestone_step_0000002.pt", torch.device("cpu"))
    assert "optimizer" not in milestone and "ema" in milestone  # weights only
    state = load_checkpoint(tmp_path / LATEST, torch.device("cpu"))
    assert state["step"] == 4 and "optimizer" in state
    assert state["hyperparameters"]["model"]["BASE_CHANNELS"] == hyper.model.BASE_CHANNELS


def test_denoising_snapshots_follow_the_schedule(model, hyper):
    from Reconstruction.model.denoising import denoise_with_snapshots, scheduled_snapshots
    diffusion = GaussianDiffusion(hyper.diffusion.TIMESTEPS, hyper.diffusion.MAX_BETA)
    timesteps = scheduled_snapshots([999, 500, 0], steps=5, timesteps=hyper.diffusion.TIMESTEPS)
    assert timesteps[0] == 999 and timesteps[-1] == 0
    snaps = denoise_with_snapshots(model.eval(), diffusion, torch.zeros(2, CHANNELS, SIZE, SIZE), 5, True, 0, None, timesteps)
    assert snaps.shape == (2, len(timesteps), CHANNELS, SIZE, SIZE)
    assert snaps.min() >= 0 and snaps.max() <= 1
