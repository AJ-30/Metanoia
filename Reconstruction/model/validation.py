"""Validation during training on a fixed subset of the frozen val cases.

The subset (``VAL_CASES`` cases, seeded) and the sampling noise are fixed,
so numbers are comparable across validations. Metrics are the same as in
``evaluation/`` (component detection, primary detection, type/connection
accuracy, validity checks). Each validation also saves a grid of final
predictions and a denoising grid (estimate at several timesteps) for a few
fixed cases covering different component types and modes.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from Reconstruction.common.config_loader import load_params
from Reconstruction.data_prep.case_source import CaseSource
from Reconstruction.data_prep.missing import Case
from Reconstruction.evaluation.case_metrics import RECORD_FIELDS, CaseContext, prepare_case, score_samples
from Reconstruction.evaluation.examples import save_examples
from Reconstruction.evaluation.report import build_tables
from Reconstruction.model.denoising import denoise_with_snapshots, save_denoising, scheduled_snapshots
from Reconstruction.model.diffusion import GaussianDiffusion
from Reconstruction.model.sampling import sample_missing

EVAL_PARAMS_FILE: Path = Path(__file__).resolve().parents[1] / "evaluation" / "params.toml"
VAL_SPLIT: str = "val"
# Overall metrics copied into the validation log.
LOGGED_METRICS: tuple[str, ...] = ("f1@0.5", "primary_detection", "primary_iou", "type_accuracy",
                                   "connection_accuracy", "pixel_iou", "shell_exact", "rooms_enclosed", "rooms_reachable")


class Validator:
    """Holds the fixed validation subset and scores a model on it."""

    def __init__(self, processed_dir: Path, hyper: SimpleNamespace, device: torch.device, log: logging.Logger) -> None:
        """Build the subset and pre-compute prediction-independent case data.

        Args:
            processed_dir (Path): Processed dataset directory.
            hyper (SimpleNamespace): Training hyperparameters.
            device (torch.device): Device for sampling.
            log (logging.Logger): Logger.
        """
        self.eval_params: SimpleNamespace = load_params(EVAL_PARAMS_FILE)
        self.hyper: SimpleNamespace = hyper
        self.device: torch.device = device
        source = CaseSource(processed_dir, VAL_SPLIT, seed=hyper.train.SEED)
        self.schema = source.schema
        picks: list[int] = sorted(np.random.default_rng(hyper.train.SEED).choice(
            len(source), size=min(hyper.logging.VAL_CASES, len(source)), replace=False).tolist())
        cases: list[Case] = [source.case(i) for i in picks]
        self.contexts: list[CaseContext] = [prepare_case(c, self.schema, self.eval_params) for c in cases]
        self.infos: list[dict[str, Any]] = [c.info for c in cases]
        self.incomplete: torch.Tensor = torch.from_numpy(np.stack([c.incomplete for c in cases]))
        self.grid_cases: list[Case] = cases[: hyper.logging.VAL_GRID_ROWS]
        self.denoise_cases: list[Case] = _diverse_cases(cases, hyper.logging.DENOISE_GRID_ROWS)
        self.snapshot_timesteps: list[int] = scheduled_snapshots(hyper.logging.DENOISE_SNAPSHOT_TIMESTEPS,
                                                                 hyper.diffusion.SAMPLING_STEPS, hyper.diffusion.TIMESTEPS)
        log.info(f"Validation subset: {len(cases)} of {len(source)} val cases; denoising snapshots at t={self.snapshot_timesteps}")

    @torch.no_grad()
    def predict(self, model: nn.Module, diffusion: GaussianDiffusion, autocast_dtype: torch.dtype | None) -> np.ndarray:
        """Sample the missing part of every subset case with fixed noise.

        Args:
            model (nn.Module): Unwrapped model (EMA copy).
            diffusion (GaussianDiffusion): Diffusion process.
            autocast_dtype (torch.dtype | None): Mixed-precision dtype.

        Returns:
            np.ndarray: float32 (N, C, H, W) scores in [0, 1].
        """
        generator = torch.Generator(device=self.device).manual_seed(self.hyper.train.SEED)
        batch: int = self.hyper.logging.VAL_BATCH_SIZE
        outputs: list[np.ndarray] = []
        for start in range(0, len(self.incomplete), batch):
            incomplete: torch.Tensor = self.incomplete[start:start + batch].to(self.device).float()
            outputs.append(sample_missing(model, diffusion, incomplete, self.hyper.diffusion.SAMPLING_STEPS,
                                          self.hyper.diffusion.CLIP_X0, generator, autocast_dtype).cpu().numpy())
        return np.concatenate(outputs)

    def run(self, model: nn.Module, diffusion: GaussianDiffusion, autocast_dtype: torch.dtype | None,
            out_dir: Path, tag: str) -> dict[str, float]:
        """Score the model and save the example and denoising grids.

        Writes ``<tag>_examples.png`` (final predictions), ``<tag>_denoising.png``
        and ``<tag>_denoising.npz`` (estimates at the snapshot timesteps) to ``out_dir``.

        Args:
            model (nn.Module): Unwrapped model (EMA copy).
            diffusion (GaussianDiffusion): Diffusion process.
            autocast_dtype (torch.dtype | None): Mixed-precision dtype.
            out_dir (Path): Output directory.
            tag (str): File name prefix, e.g. "step_0005000".

        Returns:
            dict[str, float]: Overall metrics plus primary detection per component type.
        """
        model.eval()
        predictions: np.ndarray = self.predict(model, diffusion, autocast_dtype)
        records = pd.DataFrame([{**{f: info.get(f) for f in RECORD_FIELDS},
                                 **score_samples(ctx, pred[None], self.schema, self.eval_params)}
                                for ctx, info, pred in zip(self.contexts, self.infos, predictions)])
        tables = build_tables(records, self.eval_params.components.IOU_THRESHOLDS)
        overall = tables["overall"].iloc[0]
        metrics: dict[str, float] = {name: float(overall[name]) for name in LOGGED_METRICS if name in overall}
        # Group keys are 1-tuples, e.g. ("door",), because tables group by a list of columns.
        metrics.update({f"primary_detection/{key[0] if isinstance(key, tuple) else key}": float(value)
                        for key, value in tables["by_type"]["primary_detection"].items()})
        threshold: float = self.eval_params.decode.THRESHOLD
        rows = [(case, predictions[i], f"{case.info['case_id']}\n{case.info['primary_cls']} | {case.info['mode']}")
                for i, case in enumerate(self.grid_cases)]
        save_examples(rows, self.schema, threshold, f"{tag}: final predictions", out_dir / f"{tag}_examples.png")
        self._save_denoising(model, diffusion, autocast_dtype, out_dir, tag)
        return metrics

    def _save_denoising(self, model: nn.Module, diffusion: GaussianDiffusion, autocast_dtype: torch.dtype | None,
                        out_dir: Path, tag: str) -> None:
        """Save the denoising-snapshot grid and raw estimates for the fixed denoising cases.

        Args:
            model (nn.Module): Unwrapped model (EMA copy).
            diffusion (GaussianDiffusion): Diffusion process.
            autocast_dtype (torch.dtype | None): Mixed-precision dtype.
            out_dir (Path): Output directory.
            tag (str): File name prefix.
        """
        incomplete: torch.Tensor = torch.from_numpy(np.stack([c.incomplete for c in self.denoise_cases])).to(self.device).float()
        snapshots: np.ndarray = denoise_with_snapshots(model, diffusion, incomplete, self.hyper.diffusion.SAMPLING_STEPS,
                                                       self.hyper.diffusion.CLIP_X0, self.hyper.train.SEED,
                                                       autocast_dtype, self.snapshot_timesteps)
        save_denoising(self.denoise_cases, snapshots, self.snapshot_timesteps, self.schema, self.eval_params.decode.THRESHOLD,
                       f"{tag}: estimate of the missing part while denoising", out_dir / f"{tag}_denoising.png",
                       out_dir / f"{tag}_denoising.npz")


def _diverse_cases(cases: list[Case], count: int) -> list[Case]:
    """Pick up to ``count`` cases covering as many (type, mode) pairs as possible.

    Args:
        cases (list[Case]): Candidate cases (already a seeded random subset).
        count (int): Number to pick.

    Returns:
        list[Case]: One case per distinct (type, mode) first, then the next ones in order.
    """
    seen: set[tuple[str, str]] = set()
    first: list[Case] = []
    for case in cases:
        key: tuple[str, str] = (case.info["comp_type"], case.info["mode"])
        if key not in seen:
            seen.add(key)
            first.append(case)
    rest: list[Case] = [c for c in cases if all(c is not f for f in first)]
    return (first + rest)[:count]
