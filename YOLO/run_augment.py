"""Entry point for the independent augmentation pipeline.

Builds the 15x frozen-VAE YOLO dataset (no YOLO/training involved):

    python YOLO/run_augment.py --mode debug
    python YOLO/run_augment.py --mode prod
"""

import sys
from pathlib import Path

# Project root so the YOLO package imports when launched as a script
# (``python YOLO/run_augment.py`` puts YOLO/, not the root, on sys.path).
_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_ROOT_DIR: Path = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from YOLO.augment_runner import main

if __name__ == "__main__":
    main()
