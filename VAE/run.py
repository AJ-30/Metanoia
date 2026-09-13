"""
Entry point for VAE training.

Usage:
    python VAE/run.py --mode debug|prod
"""

import sys
from pathlib import Path

# Add the repo root to sys.path so the VAE package and the shared helpers/
# package are importable no matter how this script is launched.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from VAE.runner import VAERunner


def main() -> None:
    """Build the VAE runner and start training."""
    runner = VAERunner()
    runner.train()


if __name__ == "__main__":
    main()