"""Entry point for missing-component diffusion on frozen mask VAE latents.

Usage:
    python mask_diffusion/run.py --mode debug|prod
"""

import sys
from pathlib import Path

# Add the repo root to sys.path so the mask_diffusion package and the shared
# helpers/ package are importable no matter how this script is launched.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mask_diffusion.runner import MaskDiffusionRunner


def main() -> None:
    """Build the missing-component diffusion runner and start training."""
    runner = MaskDiffusionRunner()
    runner.train()


if __name__ == "__main__":
    main()