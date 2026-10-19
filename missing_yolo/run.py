"""Entry point for missing-component detector training.

One run trains a detector for ONE missing component (``tv``, ``showcase`` or
``dining_set``) on images the frozen ``mask_diffusion`` generator produces, with
the dataset's own ``labels.txt`` boxes as the only detection target (see
:mod:`missing_yolo.dataset`).

    python missing_yolo/run.py --mode debug
    python missing_yolo/run.py --mode prod --component showcase
"""

import sys
from pathlib import Path

# Project root so the missing_yolo package imports when launched as a script
# (``python missing_yolo/run.py`` puts missing_yolo/, not the root, on sys.path).
_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_ROOT_DIR: Path = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from missing_yolo.runner import main

if __name__ == "__main__":
    main()