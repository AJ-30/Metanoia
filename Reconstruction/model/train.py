"""Train the conditional diffusion model on missing-component cases.

Single GPU:   python -m Reconstruction.model.train --run-name v1 --mode prod
Multi-GPU:    torchrun --standalone --nproc_per_node=4 -m Reconstruction.model.train --run-name v1 --mode prod
On SLURM:     sbatch Reconstruction/slurm/train_dgx.sbatch v1

Everything of a run lives in RUNS_DIR/<run name>/:
    run_info.json          hyperparameters, config, git commit, world size
    train_log.csv          loss / lr / grad-norm / throughput every LOG_EVERY_STEPS
    val_log.csv            validation metrics every VALIDATE_EVERY_STEPS
    val_grids/             per validation: step_*_examples.png (final predictions),
                           step_*_denoising.png/.npz (estimates at several timesteps)
    checkpoints/           step_*.pt (+ latest.pt)

Re-running with the same run name resumes from ``checkpoints/latest.pt``.
``--max-hours`` stops cleanly (checkpoint first) before a job time limit.
"""

import argparse
import contextlib
import csv
import json
import logging
import math
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from Reconstruction.common.config_loader import load_params, load_run_config, resolve_path
from Reconstruction.common.logging_setup import make_run_id, setup_logger
from Reconstruction.data_prep.case_source import TRAIN_SPLIT
from Reconstruction.data_prep.dataset import MissingComponentDataset
from Reconstruction.model.checkpoint import EMA, LATEST, load_checkpoint, save_checkpoint
from Reconstruction.model.diffusion import GaussianDiffusion
from Reconstruction.model.objective import LossTerms, compute_loss
from Reconstruction.model.unet import ConditionalUNet
from Reconstruction.model.validation import Validator

SCRIPT_DIR: Path = Path(__file__).resolve().parent
CONFIG_FILE: Path = SCRIPT_DIR / "config.toml"
HYPER_FILE: Path = SCRIPT_DIR / "hyperparameters.toml"
# Seconds reserved before --max-hours to write the final checkpoint.
STOP_MARGIN_S: float = 15 * 60


def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True, help="Run directory name; reusing it resumes the run")
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug", help="debug = tiny model, few steps")
    parser.add_argument("--max-hours", type=float, default=None, help="Stop cleanly after this wall time")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialize DDP when launched by torchrun, else single process.

    Returns:
        tuple[int, int, torch.device]: (rank, world size, device).
    """
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    local_rank: int = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def learning_rate(step: int, cfg) -> float:
    """Linear warmup, then cosine decay to a fraction=MIN_LR_FRACTION of the peak.
    """
    if step < cfg.WARMUP_STEPS:
        return cfg.LEARNING_RATE * (step + 1) / cfg.WARMUP_STEPS
    decay_prog= min(1.0, (step - cfg.WARMUP_STEPS) / max(1, cfg.TOTAL_STEPS - cfg.WARMUP_STEPS))
    floor= cfg.MIN_LR_FRACTION
    return cfg.LEARNING_RATE * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * decay_prog)))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    new=not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def make_loader(processed_dir: Path, hyper: SimpleNamespace, rank: int, world: int) -> tuple[DataLoader, DistributedSampler | None]:
    """Training data loader (on-the-fly cases, sharded across ranks).

    Args:
        processed_dir (Path): Processed dataset directory.
        hyper (SimpleNamespace): Hyperparameters.
        rank (int): Process rank.
        world (int): World size.

    Returns:
        tuple[DataLoader, DistributedSampler | None]: Loader and its sampler (None if single process).
    """
    dataset = MissingComponentDataset(processed_dir, TRAIN_SPLIT, seed=hyper.train.SEED)
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=hyper.train.SEED, drop_last=True) if world > 1 else None
    # Workers are re-created every epoch (not persistent) so they see the
    # dataset's new epoch number and draw new missing components.
    loader = DataLoader(dataset, batch_size=hyper.train.BATCH_SIZE_PER_GPU, sampler=sampler, shuffle=sampler is None,
                        num_workers=hyper.train.NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True,
                        generator=torch.Generator().manual_seed(hyper.train.SEED))
    return loader, sampler


def type_labels(infos: list[str], classes: list[str], device: torch.device) -> torch.Tensor:
    """Missing-type class indices from the batch's JSON metadata.

    Args:
        infos (list[str]): JSON case metadata per item.
        classes (list[str]): ``[model] TYPE_CLASSES``.
        device (torch.device): Device.

    Returns:
        torch.Tensor: (B,) long tensor.
    """
    return torch.tensor([classes.index(json.loads(info)["comp_type"]) for info in infos], device=device)


class Trainer:
    """Owns the model, optimizer, EMA and the training loop of one process."""

    def __init__(self, args: argparse.Namespace, rank: int, world: int, device: torch.device) -> None:
        """Set up everything; resume from the run's latest checkpoint if present.

        Args:
            args (argparse.Namespace): Command-line arguments.
            rank (int): Process rank.
            world (int): World size.
            device (torch.device): Device.
        """
        self.config = load_run_config(CONFIG_FILE, args.mode)
        if device.type != "cuda" and not self.config.IS_DEBUG:
            raise RuntimeError("No usable CUDA device; prod training on CPU is refused (check the driver / PyTorch CUDA build)")
        self.hyper = load_params(HYPER_FILE, args.mode if args.mode == "debug" else None)
        self.rank, self.world, self.device = rank, world, device
        self.is_main: bool = rank == 0
        self.run_dir: Path = resolve_path(self.config.RUNS_DIR) / args.run_name
        self.max_seconds: float | None = args.max_hours * 3600 - STOP_MARGIN_S if args.max_hours else None
        self.started: float = time.time()
        run_id: str = make_run_id(f"train-{args.run_name}")
        self.log: logging.Logger = setup_logger(f"train.rank{rank}", resolve_path(self.config.LOG_FILE_PATH), run_id, self.config.IS_DEBUG and self.is_main)
        if not self.is_main:
            self.log.setLevel(logging.WARNING)
        torch.manual_seed(self.hyper.train.SEED + rank)

        processed_dir: Path = resolve_path(self.config.PROCESSED_ROOT) / self.config.DATASET
        self.loader, self.sampler = make_loader(processed_dir, self.hyper, rank, world)
        schema = self.loader.dataset.source.schema
        self.model = ConditionalUNet(schema.num_channels, self.loader.dataset.source.params.canvas.SIZE, self.hyper.model).to(device)
        self.ema = EMA(self.model, self.hyper.train.EMA_DECAY)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hyper.train.LEARNING_RATE, betas=tuple(self.hyper.train.ADAM_BETAS), weight_decay=self.hyper.train.WEIGHT_DECAY)
        self.diffusion = GaussianDiffusion(self.hyper.diffusion.TIMESTEPS, self.hyper.diffusion.MAX_BETA)
        self.autocast_dtype: torch.dtype | None = torch.bfloat16 if self.hyper.train.PRECISION == "bf16" else None
        self.step, self.epoch = self._resume()
        self.ddp = DistributedDataParallel(self.model, device_ids=[device.index]) if world > 1 else self.model
        self.validator: Validator | None = Validator(processed_dir, self.hyper, device, self.log) if self.is_main else None
        if self.is_main:
            self._write_run_info(args, run_id)

    def _resume(self) -> tuple[int, int]:
        """Load run's latest checkpoint if it exists and return tuple (step, epoch) to continue from.
        """
        latest: Path = self.run_dir / "checkpoints" / LATEST
        if not latest.is_file():
            return 0, 0
        state: dict[str, Any] = load_checkpoint(latest, self.device)
        self.model.load_state_dict(state["model"])
        self.ema.model.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.log.info(f"Resumed from {latest} at step {state['step']} (epoch {state['epoch']})")
        return int(state["step"]), int(state["epoch"])

    def _write_run_info(self, args: argparse.Namespace, run_id: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, capture_output=True, text=True, check=False)
        path: Path = self.run_dir / "run_info.json"
        history: list[dict[str, Any]] = json.loads(path.read_text()) if path.exists() else []
        history.append({"run_id": run_id, "mode": args.mode, "start_step": self.step, "world_size": self.world,
                        "git_commit": commit.stdout.strip() or None, "config": vars(self.config),
                        "hyperparameters": json.loads(json.dumps(self.hyper, default=vars)),
                        "parameters_millions": sum(p.numel() for p in self.model.parameters()) / 1e6})
        path.write_text(json.dumps(history, indent=2))
        self.log.info(f"Run dir {self.run_dir}; model has {history[-1]['parameters_millions']:.1f}M parameters; "
                      f"global batch {self.hyper.train.BATCH_SIZE_PER_GPU * self.world}")

    def _train_step(self, batch: dict[str, Any]) -> tuple[LossTerms, float]:
        """One optimizer step.

        Args:
            batch (dict[str, Any]): Batch from the loader.

        Returns:
            tuple[LossTerms, float]: Loss terms and the pre-clip gradient norm.
        """
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate(self.step, self.hyper.train)
        incomplete = batch["incomplete"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        labels = type_labels(batch["info"], self.hyper.model.TYPE_CLASSES, self.device)
        # without changing the optimizer batch Gradient accumulation over micro batches bounds activation memory DDP syncs only on the last one.
        micro= self.hyper.train.MICRO_BATCH_SIZE or len(target)
        chunks = list(zip(incomplete.split(micro), target.split(micro), labels.split(micro)))
        self.optimizer.zero_grad(set_to_none=True)
        parts: list[LossTerms] = []
        for index, (inc, tgt, lab) in enumerate(chunks):
            share: float = len(tgt) / len(target)
            last: bool = index == len(chunks) - 1
            with (self.ddp.no_sync() if self.world > 1 and not last else contextlib.nullcontext()):
                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype or torch.float32, enabled=self.autocast_dtype is not None):
                    part: LossTerms = compute_loss(self.ddp, self.diffusion, inc, tgt, lab, self.hyper.loss)(part.total * share).backward()
            parts.append(LossTerms(*(value.detach() * share for value in vars(part).values())))
        terms= LossTerms(*(sum(values) for values in zip(*(vars(part).values() for part in parts))))
        grad_norm= float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper.train.GRAD_CLIP_NORM))
        self.optimizer.step()
        self.ema.update(self.model, self.step)
        self.step += 1
        return terms, grad_norm

    def _reduce_mean(self, values: list[float]) -> list[float]:
        """Average values over all ranks.
        Args:
            values (list[float]): Local values.
        Returns:
            list: Global means.
        """
        tensor = torch.tensor(values, device=self.device, dtype=torch.float64)
        if self.world > 1:
            dist.all_reduce(tensor)
            tensor /= self.world
        return tensor.tolist()

    def _should_stop(self) -> bool:
        """based on if the wall-time budget is used up.
        """
        local: float = float(self.max_seconds is not None and time.time() - self.started > self.max_seconds)
        return self._reduce_mean([local])[0] > 0

    def _log_window(self, window: list[list[float]], window_start: float) -> None:
        """Write averages of last log
        """
        means: list[float] = self._reduce_mean([sum(col) / len(col) for col in zip(*window)])
        if not self.is_main:
            return
        seconds: float = time.time() - window_start
        row: dict[str, Any] = {
            "step": self.step, "epoch": self.epoch, "lr": learning_rate(self.step, self.hyper.train),
            "loss": means[0], "loss_diffusion": means[1], "loss_type": means[2], "type_accuracy": means[3],
            "grad_norm": means[4], "samples_per_s": len(window) * self.hyper.train.BATCH_SIZE_PER_GPU * self.world / seconds,
            "gpu_mem_gb": torch.cuda.max_memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else 0.0,
            "elapsed_h": (time.time() - self.started) / 3600,
        }
        append_csv(self.run_dir / "train_log.csv", row)
        self.log.info(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    def _checkpoint(self) -> None:
        """Save model checkpoint"""
        if self.is_main:
            path: Path = save_checkpoint(self.run_dir / "checkpoints", self.step, self.epoch, self.model, self.ema,
                                         self.optimizer, self.hyper, self.hyper.logging.KEEP_CHECKPOINTS,
                                         self.hyper.logging.MILESTONE_EVERY_STEPS)
            self.log.info(f"Saved {path}")
        if self.world > 1:
            dist.barrier()

    def _validate(self) -> None:
        """Validate the EMA model on rank 0; all ranks wait for it."""
        if self.is_main:
            start: float = time.time()
            metrics = self.validator.run(self.ema.model, self.diffusion, self.autocast_dtype, self.run_dir / "val_grids", f"step_{self.step:07d}")
            append_csv(self.run_dir / "val_log.csv", {"step": self.step, **metrics})
            self.log.info(f"Validation at step {self.step} ({time.time() - start:.0f}s): "+ ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        if self.world > 1:
            dist.barrier()

    def fit(self) -> None:
        """Run the training loop until TOTAL_STEPS or the wall-time budget."""
        cfg= self.hyper.logging
        self.model.train()
        window: list[list[float]] = []
        window_start = time.time()
        stop = False
        while self.step < self.hyper.train.TOTAL_STEPS and not stop:
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            self.loader.dataset.set_epoch(self.epoch)
            for batch in self.loader:
                terms, grad_norm = self._train_step(batch)
                window.append([float(terms.total.detach()), float(terms.diffusion), float(terms.type_ce), float(terms.type_accuracy), grad_norm])
                if self.step % cfg.LOG_EVERY_STEPS == 0:
                    self._log_window(window, window_start)
                    window, window_start = [], time.time()
                    stop = self._should_stop()
                if self.step % cfg.CHECKPOINT_EVERY_STEPS == 0 or stop:
                    self._checkpoint()
                if self.step % cfg.VALIDATE_EVERY_STEPS == 0:
                    self._validate()
                    self.model.train()
                if stop or self.step >= self.hyper.train.TOTAL_STEPS:
                    break
            else:
                self.epoch += 1
        if not stop:
            if self.step % cfg.CHECKPOINT_EVERY_STEPS:
                self._checkpoint()
            if self.step % cfg.VALIDATE_EVERY_STEPS:
                self._validate()
        self.log.info("Stopped for the time limit; resubmit with the same run name to continue." if stop else "Training finished.")


def main() -> None:
    args: argparse.Namespace = parse_args()
    rank, world, device = setup_distributed()
    try:
        Trainer(args, rank, world, device).fit()
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
