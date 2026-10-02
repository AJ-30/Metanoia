"""
Entry point for floorplan-conditioned diffusion on frozen mask VAE latents.

Usage:
    python conditional_diffusion/run.py --mode debug|prod
"""

import sys
from pathlib import Path

# Add the repo root to sys.path so the conditional_diffusion package and the
# shared helpers/ package are importable no matter how this script is launched.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from conditional_diffusion.runner import ConditionalDiffusionRunner


def main() -> None:
    """Build the conditional diffusion runner and start training."""
    runner = ConditionalDiffusionRunner()
    runner.train()


if __name__ == "__main__":
    main()
