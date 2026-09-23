"""Entry point for YOLO26 training (no augmentation).

Trains on the prebuilt augmented dataset; build it first if missing:

    ./run_augmentation.sh          # one-time dataset build
    python YOLO/run.py --mode debug
    python YOLO/run.py --mode prod
"""

import argparse
import sys
from pathlib import Path

# Project root so the YOLO package imports when launched as a script
# (``python YOLO/run.py`` puts YOLO/, not the root, on sys.path).
_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_ROOT_DIR: Path = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from YOLO.runner import YOLORunner


def main() -> None:
    """Parse CLI args and launch the YOLO finetune run."""
    parser = argparse.ArgumentParser(description="Finetune YOLO26 on VAE-augmented data.")
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    args: argparse.Namespace = parser.parse_args()
    YOLORunner(mode=args.mode).train()


if __name__ == "__main__":
    main()
