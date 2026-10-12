"""Entry point for live-diffusion YOLO26 training.

Trains on overlays redrawn live from the frozen floorplan->arrangement
diffusion model (no prebuilt dataset; see :mod:`YOLO.dataset`): every
available train sample is redrawn from a fresh draw at the end of every
epoch, and fixed val images are published with ground-truth + predicted
boxes at every epoch end.

    python YOLO/run.py --mode debug
    python YOLO/run.py --mode prod
"""

import sys
from pathlib import Path

# Project root so the YOLO package imports when launched as a script
# (``python YOLO/run.py`` puts YOLO/, not the root, on sys.path).
_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_ROOT_DIR: Path = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from YOLO.runner import main

if __name__ == "__main__":
    main()
