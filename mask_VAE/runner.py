"""
Training runner for mask_VAE.

Loads configuration and hyperparameters from TOML files, resolves the mask
channel layout from the dataset's own ``*_color_index.json`` maps, builds train
and test dataloaders over the segmentation masks, runs the training loop on the
configured device, and evaluates the model on the test split after every epoch
and once on the full test set at the end.

What is written per epoch:

- a telemetry CSV row with the losses and the foreground IoU / Dice, so the
  dashboard plots the real quality curve and not just a loss curve;
- a per-class CSV row with one IoU and one Dice value per class, so a class the
  model never learns can be named;
- a **colorized** reconstruction grid, ground truth above reconstruction, one
  row pair per mask group;
- a **colorized** generation grid sampled from the latent prior, one row per
  mask group.

The grids are what the dashboard serves, and they are colored through the
dataset's color index maps rather than left as raw class-id maps, so what the
developer sees on the server is a floor plan and not a grayscale blob.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from helpers.telemetry_writer import TelemetryWriter
from mask_VAE.dataset import MaskDataset
from mask_VAE.masking import MaskLayout, build_mask_layout, render_group_rgb
from mask_VAE.metrics import confusion_matrix, foreground_mean, overlap_from_confusion
from mask_VAE.model import MaskVAEModel, mask_vae_loss
from mask_VAE.per_class_telemetry import PerClassTelemetryWriter
from VAE.runner import resolve_device

# Project root directory so the shared `helpers/` package and the sibling `VAE/`
# package (for the reused encoder and device resolution) are importable no
# matter which directory the runner is launched from.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Directory of this file, used to resolve the TOML files next to it.
SCRIPT_DIR: Path = Path(__file__).resolve().parent

# Filename prefix of the checkpoints this runner writes.
CHECKPOINT_PREFIX: str = "mask_vae"

# Columns of the main telemetry CSV, in order.
TELEMETRY_FIELDS: list[str] = [
    "epoch",
    "train_loss",
    "bce_loss",
    "kl_loss",
    "kl_per_dim",
    "test_loss",
    "test_bce",
    "test_kl",
    "test_fg_iou",
    "test_fg_dice",
]


def set_reproducibility_seed(seed: int) -> None:
    """
    Seed every RNG this training run uses.

    Applied once, before the model and the dataloaders are built, so two runs
    with the same seed, the same config and the same device produce the same
    result.

    Args:
        seed (int): Seed value for Python's and torch's global RNGs.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_reconstruction_grid(
    model: MaskVAEModel,
    mask_batch: torch.Tensor,
    layout: MaskLayout,
    device: torch.device,
    save_path: Path,
    log: object,
) -> None:
    """
    Save a colorized ground-truth-above-reconstruction grid.

    Rows are emitted group by group, ground truth first: for each mask group one
    colored row of the real masks followed by one colored row of the model's
    reconstruction. Both are produced by an ``argmax`` over the group's channels
    and a lookup in that group's color table, so a mask is never displayed as
    raw class ids.

    The model is put back into training mode before returning, so this is safe
    to call from inside the training loop.

    Args:
        model (MaskVAEModel): Trained model, switched to eval mode internally.
        mask_batch (torch.Tensor): ``(B, num_channels, H, W)`` binary target
            masks held on CPU, used as the ground-truth rows.
        layout (MaskLayout): Channel layout defining the groups and colors.
        device (torch.device): Device the forward pass runs on.
        save_path (Path): PNG path to write.
        log (object): Project logger, used to report where the grid went.
    """
    was_training: bool = model.training
    model.eval()
    with torch.no_grad():
        reconstruction: torch.Tensor = model(mask_batch.to(device))[0].cpu()

    rows: list[torch.Tensor] = []
    for group in layout.groups:
        rows.append(render_group_rgb(mask_batch, group, torch.device("cpu")))
        rows.append(render_group_rgb(reconstruction, group, torch.device("cpu")))

    save_rgb_rows(rows, nrow=mask_batch.shape[0], save_path=save_path, log=log)
    model.train(was_training)


def save_generation_grid(
    model: MaskVAEModel,
    num_samples: int,
    latent_dim: int,
    layout: MaskLayout,
    device: torch.device,
    save_path: Path,
    log: object,
) -> None:
    """
    Save a colorized grid of layouts sampled from the latent prior.

    A fresh ``N(0, I)`` latent is drawn every epoch so the generated content
    evolves visibly instead of showing the same handful of samples forever. One
    colored row is emitted per mask group; every row comes from the *same* latent
    vector, so the floorplan row and the furniture row describe one layout.

    The model is put back into training mode before returning.

    Args:
        model (MaskVAEModel): Trained model, switched to eval mode internally.
        num_samples (int): Number of latent vectors to sample.
        latent_dim (int): Dimensionality of the latent vector.
        layout (MaskLayout): Channel layout defining the groups and colors.
        device (torch.device): Device the sampling runs on.
        save_path (Path): PNG path to write.
        log (object): Project logger, used to report where the grid went.
    """
    was_training: bool = model.training
    model.eval()
    with torch.no_grad():
        latent: torch.Tensor = torch.randn(num_samples, latent_dim, device=device)
        generated: torch.Tensor = model.decode(latent).cpu()

    rows: list[torch.Tensor] = [
        render_group_rgb(generated, group, torch.device("cpu"))
        for group in layout.groups
    ]
    save_rgb_rows(rows, nrow=num_samples, save_path=save_path, log=log)
    model.train(was_training)


def save_rgb_rows(
    rows: list[torch.Tensor],
    nrow: int,
    save_path: Path,
    log: object,
) -> None:
    """
    Stack RGB batches into one image grid and write it to disk.

    Args:
        rows (list[torch.Tensor]): ``(B, 3, H, W)`` RGB batches, each one grid
            row, all with the same batch size.
        nrow (int): Number of columns, i.e. the batch size of each row.
        save_path (Path): PNG path to write.
        log (object): Project logger, used to report where the grid went.

    Raises:
        ValueError: If ``rows`` is empty, or if a row is not a 3-channel tensor
            with the expected batch size.
    """
    if not rows:
        raise ValueError("A saved grid needs at least one row.")
    for index, row in enumerate(rows):
        if row.ndim != 4 or row.shape[1] != 3 or row.shape[0] != nrow:
            raise ValueError(
                f"Grid row {index} has shape {tuple(row.shape)}; expected "
                f"({nrow}, 3, H, W)."
            )

    grid: torch.Tensor = make_grid(
        torch.cat(rows, dim=0), nrow=nrow, padding=2, normalize=False
    )
    # normalize=False matters: the rows are already in the 0..1 range the color
    # table produced, and make_grid's default normalization rescales the whole
    # grid to its own min/max, which would shift every color away from the one
    # the dataset's color index specifies.
    transforms.ToPILImage()(grid).save(str(save_path))
    log.info(f"Saved colorized grid to {save_path}")


class MaskVAERunner:
    """
    Trains a :class:`~mask_VAE.model.MaskVAEModel` on semantic segmentation masks.

    Args:
        mode (str): Run mode selecting the ``[debug]`` or ``[prod]`` table of
            ``config.toml``. Parsed from ``--mode`` when not passed explicitly.

    Raises:
        ValueError: If ``MASK_CHANNELS`` in the hyperparameters disagrees with
            the channel count implied by ``MASK_GROUPS`` in the config, which
            would silently train a model that cannot be decoded.
    """

    def __init__(self, mode: str | None = None) -> None:
        """Load config and hyperparameters, then build the model and its loggers."""
        self.config_file: str = str(SCRIPT_DIR / "config.toml")
        self.hyperparams_file: str = str(SCRIPT_DIR / "hyperparameters.toml")

        if mode is None:
            parser = argparse.ArgumentParser(
                description="Train the mask VAE on floor-plan segmentation masks.")
            parser.add_argument(
                "--mode",
                help="Run mode (debug/prod)",
                choices=["debug", "prod"],
                default="debug",
            )
            mode = parser.parse_args().mode

        self.config: SimpleNamespace = load_config(mode, self.config_file)
        self.hyperparams: SimpleNamespace = load_hyperparams(self.hyperparams_file)
        self.log = setup_logger(self.config)

        set_reproducibility_seed(int(self.hyperparams.SEED))

        # The channel layout is the contract between the dataset, the model and
        # the colorized grids, so it is resolved once here and handed to all
        # three. The color index JSONs live at the dataset root, next to the
        # split folders.
        self.layout: MaskLayout = build_mask_layout(
            self.config,
            fallback_color_index_dir=Path(self.config.TRAIN_IMAGES_DIR).parent,
        )
        self._assert_channel_count_matches_layout()

        self.device: torch.device = resolve_device(self.config.DEVICE, self.log)
        self.log.info(
            f"Starting mask VAE training with run_id: "
            f"<red>{self.config.runtime.RUN_ID}</red> at time: "
            f"{datetime.fromtimestamp(self.config.runtime.START_TIME)} on device: "
            f"{self.device} | seed: {self.hyperparams.SEED}"
        )
        self.log.info(self.layout.describe())

        self.model: MaskVAEModel = MaskVAEModel(self.hyperparams).to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparams.LEARNING_RATE
        )
        self.telemetry_writer: TelemetryWriter = TelemetryWriter(
            fieldnames=TELEMETRY_FIELDS,
            directory=self.config.TELEMETRY_LOG_DIRECTORY,
            config=self.config,
        )
        self.per_class_writer: PerClassTelemetryWriter = PerClassTelemetryWriter(
            class_names=list(self.layout.class_names),
            directory=self.config.PER_CLASS_TELEMETRY_LOG_DIRECTORY,
            config=self.config,
        )

    def _assert_channel_count_matches_layout(self) -> None:
        """
        Fail before training if the model's channel count is not the layout's.

        The channel count appears in two files that are edited independently, so
        a mismatch is easy to introduce and expensive to notice: the model would
        train happily against a target it cannot even be built for.

        Raises:
            ValueError: If ``MASK_CHANNELS`` and the layout disagree.
        """
        configured: int = int(self.hyperparams.MASK_CHANNELS)
        if configured != self.layout.num_channels:
            raise ValueError(
                f"MASK_CHANNELS={configured} in mask_VAE/hyperparameters.toml does "
                f"not match the {self.layout.num_channels} channels implied by "
                f"MASK_GROUPS={self.config.MASK_GROUPS} in mask_VAE/config.toml. "
                "The channel count is 1 shared background channel plus one per "
                "foreground class of every selected mask group."
            )

    def _make_dataloader(
        self,
        split_dir: str,
        max_samples: int,
        shuffle: bool,
    ) -> DataLoader[MaskDataset]:
        """
        Build the dataloader for one split.

        Args:
            split_dir (str): Split folder of per-sample mask folders.
            max_samples (int): Cap on samples, negative to keep all of them.
            shuffle (bool): Whether to shuffle. The test split is left
                unshuffled so the per-epoch test metrics cover a fixed subset and
                stay comparable across epochs.

        Returns:
            DataLoader[MaskDataset]: Loader yielding
                ``(batch, num_channels, H, W)`` binary mask batches.
        """
        preload: bool = bool(self.hyperparams.PRELOAD_MASKS)
        self.log.info(
            f"Building mask dataset from {split_dir} "
            f"(groups: {self.config.MASK_GROUPS}, preload into RAM: {preload})..."
        )
        dataset: MaskDataset = MaskDataset(
            split_dir=split_dir,
            layout=self.layout,
            image_size=int(self.hyperparams.IMAGE_SIZE),
            max_samples=max_samples,
            preload=preload,
        )
        return DataLoader(
            dataset,
            batch_size=int(self.hyperparams.BATCH_SIZE),
            shuffle=shuffle,
            num_workers=int(self.hyperparams.NUM_WORKERS),
            # Only CUDA supports pinned host memory; MPS on silicon ignores it.
            pin_memory=self.device.type == "cuda",
        )

    def build_dataloaders(self) -> tuple[DataLoader[MaskDataset], DataLoader[MaskDataset]]:
        """
        Build the train and test dataloaders.

        Returns:
            tuple[DataLoader[MaskDataset], DataLoader[MaskDataset]]: The shuffled
                train loader and the unshuffled, uncapped test loader.
        """
        train_loader: DataLoader[MaskDataset] = self._make_dataloader(
            split_dir=self.config.TRAIN_IMAGES_DIR,
            max_samples=int(self.hyperparams.MAX_SAMPLES),
            shuffle=True,
        )
        test_loader: DataLoader[MaskDataset] = self._make_dataloader(
            split_dir=self.config.TEST_IMAGES_DIR,
            max_samples=-1,
            shuffle=False,
        )
        self.log.info(
            f"Train masks: {len(train_loader.dataset)} | "
            f"Test masks: {len(test_loader.dataset)}"
        )
        return train_loader, test_loader

    def training_step(
        self,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one optimizer step on a batch of binary masks.

        Args:
            batch (torch.Tensor): ``(B, num_channels, H, W)`` binary mask batch on
                CPU.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Detached total loss, BCE term, KL term and per-dimension KL.
        """
        batch = batch.to(self.device)
        reconstruction, mu, logvar = self.model(batch)
        total_loss, bce_loss, kl_loss, kl_per_dim = mask_vae_loss(
            batch,
            reconstruction,
            mu,
            logvar,
            kl_weight=float(self.hyperparams.KL_WEIGHT),
            bce_pos_weight=float(self.hyperparams.BCE_POS_WEIGHT),
        )
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return (
            total_loss.detach(),
            bce_loss.detach(),
            kl_loss.detach(),
            kl_per_dim.detach(),
        )

    def run_epoch(
        self,
        dataloader: DataLoader[MaskDataset],
    ) -> dict[str, float]:
        """
        Train for one pass over a dataloader.

        Args:
            dataloader (DataLoader[MaskDataset]): Loader to train on.

        Returns:
            dict[str, float]: ``train_loss``, ``bce_loss``, ``kl_loss`` and
                ``kl_per_dim``, each averaged over batches.

        Raises:
            ValueError: If the loader is empty, which would silently average
                over nothing.
        """
        self.model.train()
        sums: dict[str, float] = {
            "train_loss": 0.0, "bce_loss": 0.0, "kl_loss": 0.0, "kl_per_dim": 0.0,
        }
        num_batches: int = 0

        for batch in dataloader:
            losses = self.training_step(batch)
            sums["train_loss"] += float(losses[0].item())
            sums["bce_loss"] += float(losses[1].item())
            sums["kl_loss"] += float(losses[2].item())
            sums["kl_per_dim"] += float(losses[3].item())
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Cannot run an epoch over an empty dataloader.")
        return {name: total / num_batches for name, total in sums.items()}

    def evaluate_test(
        self,
        dataloader: DataLoader[MaskDataset],
        max_samples: int | None = None,
    ) -> dict[str, object]:
        """
        Evaluate the model on the test split without updating weights.

        The per-class confusion matrix is accumulated across batches and turned
        into IoU and Dice once at the end, so the reported per-class numbers
        describe the whole evaluated subset rather than an average of per-batch
        values.

        Args:
            dataloader (DataLoader[MaskDataset]): Loader to evaluate on.
            max_samples (int | None): Cap on evaluated masks, ``None`` for all.

        Returns:
            dict[str, object]: ``test_loss``, ``test_bce``, ``test_kl`` and
                ``samples`` as floats, plus ``per_class_iou`` and
                ``per_class_dice`` as lists ordered like
                :attr:`MaskLayout.class_names`, and ``fg_iou`` / ``fg_dice`` as
                the foreground-only means.

        Raises:
            ValueError: If the requested subset is empty.
        """
        self.model.eval()
        num_channels: int = self.layout.num_channels
        total_samples: int = 0
        loss_sums: dict[str, float] = {"test_loss": 0.0, "test_bce": 0.0, "test_kl": 0.0}
        confusion_total: torch.Tensor = torch.zeros(
            (num_channels, num_channels), dtype=torch.int64
        )

        with torch.no_grad():
            for batch in dataloader:
                if max_samples is not None:
                    remaining: int = max_samples - total_samples
                    if remaining <= 0:
                        break
                    # Slice to exactly the remaining request so the per-epoch
                    # metrics cover exactly max_samples masks.
                    batch = batch[:remaining]

                batch = batch.to(self.device)
                reconstruction, mu, logvar = self.model(batch)
                total_loss, bce_loss, kl_loss, _ = mask_vae_loss(
                    batch,
                    reconstruction,
                    mu,
                    logvar,
                    kl_weight=float(self.hyperparams.KL_WEIGHT),
                    bce_pos_weight=float(self.hyperparams.BCE_POS_WEIGHT),
                )
                batch_size: int = batch.shape[0]
                # The loss terms are per-image sums, so weight by image count to
                # keep a short final batch from skewing the average.
                loss_sums["test_loss"] += float(total_loss.item()) * batch_size
                loss_sums["test_bce"] += float(bce_loss.item()) * batch_size
                loss_sums["test_kl"] += float(kl_loss.item()) * batch_size
                total_samples += batch_size
                confusion_total += confusion_matrix(reconstruction.cpu(), batch.cpu(), num_channels)

        if total_samples == 0:
            raise ValueError(
                "Evaluation covered zero masks. Check TEST_LOSS_SAMPLES and the "
                "test split path."
            )

        overlap = overlap_from_confusion(confusion_total)
        per_class_iou: list[float] = [float(value) for value in overlap.iou.tolist()]
        per_class_dice: list[float] = [float(value) for value in overlap.dice.tolist()]
        return {
            "test_loss": loss_sums["test_loss"] / total_samples,
            "test_bce": loss_sums["test_bce"] / total_samples,
            "test_kl": loss_sums["test_kl"] / total_samples,
            "samples": float(total_samples),
            "per_class_iou": per_class_iou,
            "per_class_dice": per_class_dice,
            "fg_iou": foreground_mean(overlap.iou, self.layout),
            "fg_dice": foreground_mean(overlap.dice, self.layout),
        }

    def save_checkpoint(self, epoch: int) -> None:
        """
        Write a checkpoint holding the weights, the optimizer and the layout.

        The hyperparameters and the mask layout are stored alongside the weights
        so a later stage can rebuild this exact model without having to guess
        which mask groups and channel count were in effect.

        Args:
            epoch (int): Epoch number to record in the file name and payload.
        """
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        checkpoint_path: str = os.path.join(
            self.config.MODEL_SAVE_PATH,
            f"{CHECKPOINT_PREFIX}_{self.config.runtime.RUN_ID}_epoch_{epoch}.pt",
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "hyperparameters": vars(self.hyperparams),
                "mask_groups": list(self.config.MASK_GROUPS),
                "mask_channel_indices": {
                    group.name: list(group.channel_indices) for group in self.layout.groups
                },
                "mask_class_names": list(self.layout.class_names),
            },
            checkpoint_path,
        )
        self.log.info(f"Saved checkpoint to {checkpoint_path}")

    def save_grids(self, fixed_batch: torch.Tensor, epoch: int) -> None:
        """
        Write the colorized reconstruction and generation grids for an epoch.

        Args:
            fixed_batch (torch.Tensor): Fixed test batch used as the
                ground-truth half of the reconstruction grid, held on CPU.
            epoch (int): Epoch number embedded in the file names.
        """
        reconstruction_path: Path = (
            Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
            / f"reconstruction_epoch_{epoch}_{self.config.runtime.RUN_ID}.png"
        )
        save_reconstruction_grid(
            model=self.model,
            mask_batch=fixed_batch,
            layout=self.layout,
            device=self.device,
            save_path=reconstruction_path,
            log=self.log,
        )

        generation_path: Path = (
            Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
            / f"generation_epoch_{epoch}_{self.config.runtime.RUN_ID}.png"
        )
        save_generation_grid(
            model=self.model,
            num_samples=int(self.hyperparams.NUM_GEN_SAMPLES),
            latent_dim=int(self.hyperparams.LATENT_DIM),
            layout=self.layout,
            device=self.device,
            save_path=generation_path,
            log=self.log,
        )

    def train(self) -> None:
        """
        Run the full training loop and the final full-test evaluation.

        After every epoch the runner trains on the whole train split, evaluates
        a fixed test subset, writes both telemetry CSVs and both colorized
        grids, and checkpoints every ``CHECKPOINT_INTERVAL`` epochs.

        Raises:
            ValueError: If a configured grid is empty.
        """
        torch.autograd.set_detect_anomaly(bool(self.config.IS_DEBUG_TORCH))
        os.makedirs(self.config.OUTPUT_SAMPLES_DIRECTORY, exist_ok=True)

        train_loader, test_loader = self.build_dataloaders()

        # One fixed batch of test masks backs the reconstruction grid, so every
        # epoch's picture shows the same layouts and progress is comparable.
        num_recon: int = int(self.hyperparams.NUM_RECON_SAMPLES)
        fixed_batch: torch.Tensor = next(iter(test_loader))[:num_recon]
        if fixed_batch.shape[0] == 0:
            raise ValueError(
                f"NUM_RECON_SAMPLES={num_recon} but the first test batch is empty."
            )

        self.log.info(f"Torch version: {torch.__version__}")
        execution_start: float = time.time()

        for epoch in range(1, int(self.hyperparams.NUM_EPOCHS) + 1):
            epoch_start: float = time.time()
            train_metrics: dict[str, float] = self.run_epoch(train_loader)
            test_metrics: dict[str, object] = self.evaluate_test(
                test_loader, max_samples=int(self.hyperparams.TEST_LOSS_SAMPLES)
            )

            if (
                epoch % int(self.hyperparams.TELEMETRY_INTERVAL) == 0
                or epoch == int(self.hyperparams.NUM_EPOCHS)
            ):
                self.telemetry_writer.log(
                    epoch=epoch,
                    train_loss=train_metrics["train_loss"],
                    bce_loss=train_metrics["bce_loss"],
                    kl_loss=train_metrics["kl_loss"],
                    kl_per_dim=train_metrics["kl_per_dim"],
                    test_loss=test_metrics["test_loss"],
                    test_bce=test_metrics["test_bce"],
                    test_kl=test_metrics["test_kl"],
                    test_fg_iou=test_metrics["fg_iou"],
                    test_fg_dice=test_metrics["fg_dice"],
                )
                self.per_class_writer.log(
                    epoch=epoch,
                    iou=test_metrics["per_class_iou"],
                    dice=test_metrics["per_class_dice"],
                )

            self.save_grids(fixed_batch, epoch)

            if (
                epoch % int(self.hyperparams.CHECKPOINT_INTERVAL) == 0
                or epoch == int(self.hyperparams.NUM_EPOCHS)
            ):
                self.save_checkpoint(epoch)

            self.log.info(
                f"Epoch {epoch}/{self.hyperparams.NUM_EPOCHS} | "
                f"train_loss: {train_metrics['train_loss']:.4f} "
                f"(bce: {train_metrics['bce_loss']:.4f}, "
                f"kl/dim: {train_metrics['kl_per_dim']:.6f}) | "
                f"test_loss: {test_metrics['test_loss']:.4f} | "
                f"test_fg_iou: {test_metrics['fg_iou']:.4f} | "
                f"test_fg_dice: {test_metrics['fg_dice']:.4f} "
                f"(N={int(test_metrics['samples'])}) | "
                f"epoch_time: {time.time() - epoch_start:.1f}s | "
                f"total: {time.time() - execution_start:.1f}s"
            )

        final_metrics: dict[str, object] = self.evaluate_test(test_loader)
        self.save_checkpoint(int(self.hyperparams.NUM_EPOCHS))
        self.log.success(
            f"FINAL TEST RESULT on full test set "
            f"(N={int(final_metrics['samples'])}): "
            f"test_loss={final_metrics['test_loss']:.4f} | "
            f"test_bce={final_metrics['test_bce']:.4f} | "
            f"test_kl={final_metrics['test_kl']:.4f} | "
            f"test_fg_iou={final_metrics['fg_iou']:.4f} | "
            f"test_fg_dice={final_metrics['fg_dice']:.4f}"
        )
        self._log_worst_classes(final_metrics)
        self.close()

    def _log_worst_classes(self, metrics: dict[str, object]) -> None:
        """
        Report the foreground classes the model reconstructs worst.

        Averages that hide a single never-predicted class are the main way a
        multi-class mask model looks healthy while it is not, so the bottom of
        the ranking is named explicitly at the end of every run.

        Args:
            metrics (dict[str, object]): Evaluation result carrying
                ``per_class_iou`` and ``fg_iou``.
        """
        per_class_iou: list[float] = metrics["per_class_iou"]  # type: ignore[assignment]
        ranked: list[tuple[float, str]] = sorted(
            zip(
                per_class_iou,
                self.layout.class_names,
                strict=True,
            )
        )
        self.log.info(
            f"Worst 5 classes by test IoU (mean foreground IoU "
            f"{metrics['fg_iou']:.4f}): "
            + ", ".join(f"{name}={iou:.4f}" for iou, name in ranked[:5])
        )

    def close(self) -> None:
        """Flush and close the telemetry writers.

        Both writers flush after every row, so this only guarantees the file
        handles are released when :meth:`train` returns rather than at process
        exit. Called by :meth:`train`; safe to call twice.
        """
        self.per_class_writer.close()
        self.telemetry_writer.file.close()
