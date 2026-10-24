"""How a prediction forms during denoising: snapshots of the x0 estimate.

At a few sampling timesteps (e.g. t = 999, 800, ..., 0) the model's current
estimate of the missing part is recorded. Early estimates show whether the
model already knows *where* something is missing; late ones show whether the
*shape* is right. Saved as

* a PNG grid: one row per case, columns
  Input | estimate at each snapshot t | Composed plan | Ground truth
  (estimates drawn thresholded, exactly as evaluation decodes them), and
* an ``.npz`` with the raw estimates (uint8 scores * 255, not thresholded)
  plus ``case_ids`` and ``timesteps``, for closer inspection.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from Reconstruction.data_prep.missing import Case
from Reconstruction.data_prep.render import render_semantic
from Reconstruction.data_prep.schema import Schema
from Reconstruction.evaluation.examples import prediction_panels
from Reconstruction.model.diffusion import GaussianDiffusion
from Reconstruction.model.sampling import sample_missing

PANEL_INCHES: float = 2.6
DPI: int = 100


def scheduled_snapshots(requested: list[int], steps: int, timesteps: int) -> list[int]:
    """Map requested timesteps to the nearest ones the DDIM schedule visits.

    Args:
        requested (list[int]): Wanted timesteps in [0, T-1].
        steps (int): DDIM steps.
        timesteps (int): Training timesteps T.

    Returns:
        list[int]: Distinct scheduled timesteps, from noisiest to cleanest.
    """
    schedule: list[int] = torch.linspace(timesteps - 1, 0, steps).round().long().tolist()
    return sorted({min(schedule, key=lambda s: abs(s - r)) for r in requested}, reverse=True)


@torch.no_grad()
def denoise_with_snapshots(model: nn.Module, diffusion: GaussianDiffusion, incomplete: torch.Tensor, steps: int,
                           clip_x0: bool, seed: int, autocast_dtype: torch.dtype | None,
                           snapshot_timesteps: list[int]) -> np.ndarray:
    """Sample a batch and keep the x0 estimate at each snapshot timestep.

    Args:
        model (nn.Module): Unwrapped model (EMA copy).
        diffusion (GaussianDiffusion): Diffusion process.
        incomplete (torch.Tensor): (N, C, H, W) inputs in {0, 1} on the model's device.
        steps (int): DDIM steps.
        clip_x0 (bool): Clip x0 estimates.
        seed (int): Noise seed (fixed -> comparable across checkpoints).
        autocast_dtype (torch.dtype | None): Mixed-precision dtype.
        snapshot_timesteps (list[int]): Scheduled timesteps to record.

    Returns:
        np.ndarray: float32 (N, S, C, H, W) scores in [0, 1], S = len(snapshot_timesteps).
    """
    captured: dict[int, np.ndarray] = {}

    def record(t: int, x0: torch.Tensor) -> None:
        if t in snapshot_timesteps:
            captured[t] = ((x0.float() + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy()

    generator = torch.Generator(device=incomplete.device).manual_seed(seed)
    sample_missing(model, diffusion, incomplete, steps, clip_x0, generator, autocast_dtype, on_step=record)
    missing: set[int] = set(snapshot_timesteps) - set(captured)
    assert not missing, f"snapshot timesteps never visited by the schedule: {sorted(missing)}"
    return np.stack([captured[t] for t in snapshot_timesteps], axis=1)


def save_denoising(cases: list[Case], snapshots: np.ndarray, timesteps: list[int], schema: Schema, threshold: float,
                   title: str, png_path: Path, npz_path: Path) -> None:
    """Write the snapshot grid (PNG) and raw estimates (NPZ).

    Args:
        cases (list[Case]): Cases (rows).
        snapshots (np.ndarray): (N, S, C, H, W) scores from ``denoise_with_snapshots``.
        timesteps (list[int]): Timestep of each snapshot (columns).
        schema (Schema): Taxonomy.
        threshold (float): Decoding threshold (as in evaluation).
        title (str): Figure title.
        png_path (Path): Grid output.
        npz_path (Path): Raw estimates output.
    """
    columns: list[str] = ["Input"] + [f"t = {t}" for t in timesteps] + ["Composed (final)", "Ground truth"]
    fig, axes = plt.subplots(len(cases), len(columns), squeeze=False,
                             figsize=(PANEL_INCHES * len(columns), PANEL_INCHES * len(cases) + 0.8))
    for row, case in enumerate(cases):
        final_panels = prediction_panels(case, snapshots[row, -1], schema, threshold)
        panels = ([render_semantic(case.incomplete, schema)]
                  + [prediction_panels(case, snapshots[row, s], schema, threshold)[1] for s in range(len(timesteps))]
                  + [final_panels[2], final_panels[3]])
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel, interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(columns[col], fontsize=10)
        axes[row, 0].set_ylabel(f"{case.info['case_id']}\n{case.info['primary_cls']} | {case.info['mode']}", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=DPI)
    plt.close(fig)
    np.savez_compressed(npz_path, snapshots=np.round(snapshots * 255).astype(np.uint8),
                        timesteps=np.array(timesteps), case_ids=np.array([c.info["case_id"] for c in cases]))
