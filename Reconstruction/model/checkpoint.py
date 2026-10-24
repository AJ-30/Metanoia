"""EMA weights and checkpoint save / resume.

``checkpoints/`` of a run holds:

* ``step_XXXXXXX.pt``: full checkpoints (model, EMA, optimizer); only the
  newest ``KEEP_CHECKPOINTS`` are kept.
* ``milestone_step_XXXXXXX.pt``: every ``MILESTONE_EVERY_STEPS`` steps a
  weights-only copy (model + EMA, about half the size) that is never deleted,
  so earlier stages of training can always be inspected or compared.
* ``latest.pt``: copy of the newest full checkpoint. Resuming a run (same run
  name) loads it, so an interrupted SLURM job continues where it stopped.
"""

import copy
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

LATEST: str = "latest.pt"


class EMA:
    """Exponential moving average of model weights (used for sampling)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        """Copy the model as the initial average.

        Args:
            model (nn.Module): Unwrapped model.
            decay (float): Target decay (e.g. 0.9999).
        """
        self.model: nn.Module = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay: float = decay

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Blend current weights in. Early steps use a smaller decay so the
        average is not dominated by the random initialization.

        Args:
            model (nn.Module): Unwrapped model being trained.
            step (int): Current optimizer step.
        """
        decay: float = min(self.decay, (1 + step) / (10 + step))
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.lerp_(param.detach(), 1.0 - decay)
        for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def save_checkpoint(directory: Path, step: int, epoch: int, model: nn.Module, ema: EMA,
                    optimizer: torch.optim.Optimizer, hyper: SimpleNamespace, keep: int, milestone_every: int) -> Path:
    """Write a checkpoint, refresh ``latest.pt``, add a milestone if due, prune old step files.

    Args:
        directory (Path): Checkpoint directory.
        step (int): Optimizer step.
        epoch (int): Epoch.
        model (nn.Module): Unwrapped model.
        ema (EMA): EMA weights.
        optimizer (torch.optim.Optimizer): Optimizer.
        hyper (SimpleNamespace): Hyperparameters (stored for reproducibility).
        keep (int): Number of full step checkpoints to keep (>= 1).
        milestone_every (int): Steps between permanent weights-only milestones.

    Returns:
        Path: Written full checkpoint path.
    """
    assert keep >= 1, "keep at least one full checkpoint (needed to resume)"
    directory.mkdir(parents=True, exist_ok=True)
    weights: dict[str, Any] = {"step": step, "epoch": epoch, "model": model.state_dict(), "ema": ema.model.state_dict(),
                               "hyperparameters": _to_dict(hyper)}
    path: Path = directory / f"step_{step:07d}.pt"
    _atomic_save({**weights, "optimizer": optimizer.state_dict()}, path)
    shutil.copyfile(path, directory / LATEST)
    if step % milestone_every == 0:
        _atomic_save(weights, directory / f"milestone_step_{step:07d}.pt")
    for old in sorted(directory.glob("step_*.pt"))[:-keep]:
        old.unlink()
    return path


def _atomic_save(state: dict[str, Any], path: Path) -> None:
    """Save via a temp file so an interrupted write never leaves a corrupt file.

    Args:
        state (dict[str, Any]): Checkpoint content.
        path (Path): Destination.
    """
    temp: Path = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint dict onto a device.

    Args:
        path (Path): Checkpoint file.
        device (torch.device): Target device.

    Returns:
        dict[str, Any]: Checkpoint state.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def resolve_checkpoints(directory: Path, selectors: list[str]) -> list[Path]:
    """Turn checkpoint selectors into existing files (duplicates removed, order kept).

    Args:
        directory (Path): The run's checkpoints directory.
        selectors (list[str]): See ``--checkpoints``.

    Returns:
        list[Path]: Checkpoint files.

    Raises:
        FileNotFoundError: If a selector matches no file.
    """
    paths: list[Path] = []
    for selector in selectors:
        if selector == "latest":
            found = [directory / LATEST]
        elif selector == "milestones":
            found = sorted(directory.glob("milestone_step_*.pt"))
        elif selector.isdigit():
            step: str = f"{int(selector):07d}"
            found = [p for p in (directory / f"step_{step}.pt", directory / f"milestone_step_{step}.pt") if p.is_file()][:1]
        else:
            found = [Path(selector)]
        found = [p for p in found if p.is_file()]
        if not found:
            raise FileNotFoundError(f"No checkpoint matches '{selector}' in {directory}")
        paths.extend(p for p in found if p not in paths)
    return paths


def _to_dict(value: Any) -> Any:
    """Convert nested namespaces to plain dicts (for saving).

    Args:
        value (Any): Namespace, list or scalar.

    Returns:
        Any: Plain Python structure.
    """
    if isinstance(value, SimpleNamespace):
        return {key: _to_dict(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    return value
