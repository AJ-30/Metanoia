"""FurnitureArranger version of model/predict.py (own config/hyperparameters).

Generate predictions of a trained run for the frozen cases of a split.

Writes one ``<case_id>.npz`` per case with ``samples``: uint8 (K, C, H, W),
thresholded at the evaluation decode threshold and stored as {0, 255}, to
RUNS_DIR/<run>/predictions/<dataset>_<split>/step_<step>_k<K>/, so predictions of
different checkpoints or sample counts never mix. Existing files are skipped, so
an interrupted job can simply be restarted. Evaluate afterwards with:

    python -m Reconstruction.evaluation.evaluate --dataset furniture --split test \
        --predictions Reconstruction/runs/<run>/predictions/furniture_test/step_0060000_k1 --name <run>_step60000_k1 --mode prod

Usage:
    python -m Reconstruction.furniture_model.predict --run-name v1 --split test --mode prod
    python -m Reconstruction.furniture_model.predict --run-name v1 --checkpoint 60000 --mode prod
    torchrun --standalone --nproc_per_node=2 -m Reconstruction.furniture_model.predict --run-name v1 --dataset furniture --mode prod
"""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from Reconstruction.common.config_loader import load_params, load_run_config, resolve_path, to_namespace
from Reconstruction.common.logging_setup import make_run_id, setup_logger
from Reconstruction.data_prep.case_source import CaseSource
from Reconstruction.model.checkpoint import load_checkpoint, resolve_checkpoints
from Reconstruction.model.diffusion import GaussianDiffusion
from Reconstruction.model.sampling import sample_missing
from Reconstruction.furniture_model.train import CONFIG_FILE, HYPER_FILE, setup_distributed
from Reconstruction.model.unet import ConditionalUNet
from Reconstruction.model.validation import EVAL_PARAMS_FILE

SCORE_SCALE: int = 255


def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    parser.add_argument("--dataset", default=None, help="Processed dataset (default: the training dataset)")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", default="latest",
                        help="'latest' (default), a step number (step_*.pt or milestone_step_*.pt) or a checkpoint path")
    parser.add_argument("--samples", type=int, default=None, help="Samples per case K (default: [predict] SAMPLES_PER_CASE)")
    parser.add_argument("--types", default=None,
                        help="Comma-separated component types to predict (e.g. wall,door,window); other cases are skipped")
    parser.add_argument("--max-cases", type=int, default=None, help="Only the first N cases (smoke tests)")
    return parser.parse_args()


def load_model(checkpoint: Path, num_channels: int, image_size: int, device: torch.device) -> tuple[ConditionalUNet, SimpleNamespace, int]:
    """Rebuild the model from a checkpoint's own hyperparameters and load EMA weights.

    Args:
        checkpoint (Path): Checkpoint file.
        num_channels (int): Semantic channels of the dataset.
        image_size (int): Canvas size.
        device (torch.device): Device.

    Returns:
        tuple[ConditionalUNet, SimpleNamespace, int]: Model (eval mode), its
        hyperparameters and the checkpoint step.
    """
    state: dict[str, Any] = load_checkpoint(checkpoint, device)
    hyper: SimpleNamespace = to_namespace(state["hyperparameters"])
    model = ConditionalUNet(num_channels, image_size, hyper.model).to(device)
    model.load_state_dict(state["ema"])
    return model.eval(), hyper, int(state["step"])


def predict_split(source: CaseSource, indices: list[int], model: ConditionalUNet, hyper: SimpleNamespace, samples: int,
                  batch_size: int, decode_threshold: float, out_dir: Path, device: torch.device,
                  log: logging.Logger) -> int:
    """Sample and save predictions for the given case indices.

    Args:
        source (CaseSource): Frozen cases of the split.
        indices (list[int]): Case indices handled by this process.
        model (ConditionalUNet): Model.
        hyper (SimpleNamespace): Model's hyperparameters.
        samples (int): Samples per case K.
        batch_size (int): Cases per batch.
        decode_threshold (float): Score above which a channel is "on" (as in evaluation).
        out_dir (Path): Output directory.
        device (torch.device): Device.
        log (logging.Logger): Logger.

    Returns:
        int: Number of cases written.
    """
    diffusion = GaussianDiffusion(hyper.diffusion.TIMESTEPS, hyper.diffusion.MAX_BETA)
    dtype: torch.dtype | None = torch.bfloat16 if hyper.train.PRECISION == "bf16" and device.type == "cuda" else None
    todo: list[int] = [i for i in indices if not (out_dir / f"{source.spec(i, 0).case_id}.npz").exists()]
    for start in range(0, len(todo), batch_size):
        batch: list[int] = todo[start:start + batch_size]
        cases = [source.case(i) for i in batch]
        incomplete: torch.Tensor = torch.from_numpy(np.stack([c.incomplete for c in cases])).to(device).float()
        per_sample: list[np.ndarray] = []
        for k in range(samples):
            # Seeded per (batch, sample): a full rerun reproduces the same samples
            # (a resumed run batches differently and draws different noise).
            generator = torch.Generator(device=device).manual_seed(hyper.train.SEED + 1000 * batch[0] + k)
            scores = sample_missing(model, diffusion, incomplete, hyper.diffusion.SAMPLING_STEPS, hyper.diffusion.CLIP_X0, generator, dtype)
            # Stored thresholded ({0, 255}): evaluation decodes at the same
            # threshold, and binary maps compress ~10x better than raw scores.
            binary: np.ndarray = scores.cpu().numpy() > decode_threshold
            per_sample.append(binary.astype(np.uint8) * SCORE_SCALE)
        stacked: np.ndarray = np.stack(per_sample, axis=1)  # (B, K, C, H, W)
        for case, case_samples in zip(cases, stacked):
            np.savez_compressed(out_dir / f"{case.info['case_id']}.npz", samples=case_samples)
        log.info(f"{min(start + batch_size, len(todo))}/{len(todo)} cases done")
    return len(todo)


def main() -> None:
    """Entry point (see module docstring)."""
    args: argparse.Namespace = parse_args()
    rank, world, device = setup_distributed()
    config: SimpleNamespace = load_run_config(CONFIG_FILE, args.mode)
    run_hyper: SimpleNamespace = load_params(HYPER_FILE, "debug" if args.mode == "debug" else None)
    log: logging.Logger = setup_logger(f"predict.rank{rank}", resolve_path(config.LOG_FILE_PATH),
                                       make_run_id(f"predict-{args.run_name}"), config.IS_DEBUG)
    dataset: str = args.dataset or config.DATASET
    run_dir: Path = resolve_path(config.RUNS_DIR) / args.run_name
    source = CaseSource(resolve_path(config.PROCESSED_ROOT) / dataset, args.split, seed=0)
    checkpoint: Path = resolve_checkpoints(run_dir / "checkpoints", [args.checkpoint])[0]
    model, hyper, step = load_model(checkpoint, source.schema.num_channels, source.params.canvas.SIZE, device)
    samples: int = args.samples or run_hyper.predict.SAMPLES_PER_CASE
    out_dir: Path = run_dir / config.PREDICTIONS_SUBDIR / f"{dataset}_{args.split}" / f"step_{step:07d}_k{samples}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Rank {rank}/{world}: {checkpoint.name} (step {step}), {dataset}/{args.split}, K={samples} -> {out_dir}")
    total: int = len(source) if args.max_cases is None else min(args.max_cases, len(source))
    wanted: list[int] = list(range(total))
    if args.types:
        types: set[str] = set(args.types.split(","))
        wanted = [i for i in wanted if source.spec(i, 0).comp_type in types]
    written: int = predict_split(source, wanted[rank::world], model, hyper, samples,
                                 run_hyper.predict.BATCH_SIZE, load_params(EVAL_PARAMS_FILE).decode.THRESHOLD,
                                 out_dir, device, log)
    log.info(f"Rank {rank}: wrote {written} prediction files")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
