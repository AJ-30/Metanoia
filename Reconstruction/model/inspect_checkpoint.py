"""Inspect saved checkpoints of a run (also while it is still training).

For each selected checkpoint, with its EMA weights:
    * scores a fixed, seeded subset of val cases with the evaluation metrics,
    * saves the final-prediction grid and the denoising grid (estimate of the
      missing part at several timesteps) plus raw estimates,
    * appends one row to RUNS_DIR/<run>/inspect/summary.csv, so checkpoints
      can be compared side by side.

Outputs: RUNS_DIR/<run>/inspect/<checkpoint>_{examples.png, denoising.png, denoising.npz}.
The same val cases and noise are used for every checkpoint, so grids are
directly comparable.

Usage (from the repository root; GPU if available, otherwise CPU, slower):
    python -m Reconstruction.model.inspect_checkpoint --run-name v1                        # latest.pt
    python -m Reconstruction.model.inspect_checkpoint --run-name v1 --checkpoints milestones
    python -m Reconstruction.model.inspect_checkpoint --run-name v1 --checkpoints 25000 50000 latest
    sbatch Reconstruction/slurm/inspect_a40.sbatch v1 milestones latest
"""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from Reconstruction.common.config_loader import load_run_config, resolve_path
from Reconstruction.common.logging_setup import make_run_id, setup_logger
from Reconstruction.data_prep.case_source import CaseSource
from Reconstruction.model.checkpoint import resolve_checkpoints
from Reconstruction.model.diffusion import GaussianDiffusion
from Reconstruction.model.predict import load_model
from Reconstruction.model.train import CONFIG_FILE, append_csv
from Reconstruction.model.validation import VAL_SPLIT, Validator

INSPECT_SUBDIR: str = "inspect"
DEFAULT_VAL_CASES: int = 64


def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=["debug", "prod"], default="prod")
    parser.add_argument("--checkpoints", nargs="+", default=["latest"],
                        help="'latest', 'milestones' (all permanent ones), step numbers, or checkpoint paths")
    parser.add_argument("--val-cases", type=int, default=DEFAULT_VAL_CASES, help="Val cases to score per checkpoint")
    return parser.parse_args()


def main() -> None:
    """Entry point (see module docstring)."""
    args: argparse.Namespace = parse_args()
    config: SimpleNamespace = load_run_config(CONFIG_FILE, args.mode)
    log: logging.Logger = setup_logger("inspect", resolve_path(config.LOG_FILE_PATH), make_run_id(f"inspect-{args.run_name}"),
                                       config.IS_DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir: Path = resolve_path(config.RUNS_DIR) / args.run_name
    out_dir: Path = run_dir / INSPECT_SUBDIR
    processed_dir: Path = resolve_path(config.PROCESSED_ROOT) / config.DATASET
    source = CaseSource(processed_dir, VAL_SPLIT, seed=0)
    schema, size = source.schema, source.params.canvas.SIZE
    validator: Validator | None = None
    for path in resolve_checkpoints(run_dir / "checkpoints", args.checkpoints):
        model, hyper, step = load_model(path, schema.num_channels, size, device)
        if validator is None:
            # Built once with the first checkpoint's settings, so every
            # checkpoint is scored on the same cases with the same noise.
            hyper.logging.VAL_CASES = args.val_cases
            validator = Validator(processed_dir, hyper, device, log)
        dtype: torch.dtype | None = torch.bfloat16 if hyper.train.PRECISION == "bf16" and device.type == "cuda" else None
        tag: str = path.stem if path.stem != "latest" else f"latest_step_{step:07d}"
        diffusion = GaussianDiffusion(hyper.diffusion.TIMESTEPS, hyper.diffusion.MAX_BETA)
        metrics: dict[str, Any] = validator.run(model, diffusion, dtype, out_dir, tag)
        append_csv(out_dir / "summary.csv", {"checkpoint": path.name, "step": step, "val_cases": args.val_cases, **metrics})
        log.info(f"{path.name} (step {step}): " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    log.info(f"Done. Grids and summary.csv in {out_dir}")


if __name__ == "__main__":
    main()
