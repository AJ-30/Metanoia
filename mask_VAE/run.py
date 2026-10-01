"""
Entry point for mask_VAE training.

Trains a convolutional VAE on the floor-plan segmentation masks, with a
per-class binary cross-entropy reconstruction loss and a KL term on the latent.

Usage:
    python mask_VAE/run.py --mode debug|prod
"""

import sys
from pathlib import Path

# Add the repo root to sys.path so the mask_VAE package, the shared helpers/
# package and the sibling VAE/ package are importable no matter how this script
# is launched.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mask_VAE.runner import MaskVAERunner


def main() -> None:
    """Build the mask VAE runner and start training."""
    runner = MaskVAERunner()
    runner.train()


if __name__ == "__main__":
    main()
