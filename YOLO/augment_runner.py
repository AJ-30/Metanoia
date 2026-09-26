"""Independent frozen-VAE augmentation runner (no YOLO dependency).

Builds the 5x YOLO dataset from frozen VAE latents and writes ``data.yaml``.
Launched via ``./run_augmentation.sh`` (or ``python YOLO/run_augment.py``).
Reads only ``YOLO/config.toml`` + ``YOLO/augment_hyperparameters.toml``; the
YOLO trainer never triggers this runner.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

# Project root so helpers/ + YOLO.augment import from any working directory.
SCRIPT_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger

from YOLO.augment import run_augmentation, write_data_yaml


class AugmentRunner:
    """Runner that builds the augmented dataset and its data.yaml."""

    def __init__(self, mode: str) -> None:
        """Load augmentation config/hyperparams and set up logging.

        Args:
            mode (str): Run mode ("debug" or "prod") selecting config overrides.
        """
        self.config: SimpleNamespace = load_config(mode, str(SCRIPT_DIR / "config.toml"))
        self.hyperparams: SimpleNamespace = load_hyperparams(
            str(SCRIPT_DIR / "augment_hyperparameters.toml"))
        self.log = setup_logger(self.config)
        self.log.info(
            f"Starting VAE augmentation run_id: {self.config.runtime.RUN_ID} "
            f"(factor={self.hyperparams.AUG_FACTOR}, "
            f"checkpoints={list(self.hyperparams.CHECKPOINT_EPOCHS)}, "
            f"noise_scales={list(self.hyperparams.LATENT_NOISE_SCALES)})")

    def run(self) -> dict[str, int]:
        """Generate the augmented dataset and write data.yaml.

        Returns:
            dict[str, int]: Counts {"train_images", "val_images", "test_images"}.
        """
        counts: dict[str, int] = run_augmentation(self.config, self.hyperparams, self.log)
        yaml_path: str = write_data_yaml(self.config, self.hyperparams)
        self.log.info(f"Augmentation complete: {counts}; data.yaml at {yaml_path}")
        return counts


def main() -> None:
    """Parse CLI args and launch the augmentation run."""
    parser = argparse.ArgumentParser(description="Build the 5x VAE-augmented YOLO dataset.")
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    args: argparse.Namespace = parser.parse_args()
    AugmentRunner(mode=args.mode).run()


if __name__ == "__main__":
    main()
