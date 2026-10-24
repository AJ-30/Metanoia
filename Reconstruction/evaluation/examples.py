"""Qualitative prediction grids: one row per case.

Panels: Input | Prediction (added parts over the faded input) | Composed plan |
Ground truth (missing part highlighted over the faded complete plan).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from Reconstruction.data_prep.missing import Case
from Reconstruction.data_prep.render import FADE_ALPHA, render_semantic
from Reconstruction.data_prep.schema import Schema
from Reconstruction.evaluation.labels import LabelMaps, added, compose, labels_to_semantic, semantic_to_labels

TITLES: tuple[str, ...] = ("Input (incomplete)", "Prediction (added)", "Composed plan", "Ground truth (missing part)")
PANEL_INCHES: float = 3.2
DPI: int = 100


def _highlight(background: np.ndarray, overlay: np.ndarray, schema: Schema) -> np.ndarray:
    """Overlay's pixels in full colour over a faded background plan.

    Args:
        background (np.ndarray): (C, H, W) plan drawn faded.
        overlay (np.ndarray): (C, H, W) parts drawn in full colour.
        schema (Schema): Taxonomy and colours.

    Returns:
        np.ndarray: float (H, W, 3) image.
    """
    white: np.ndarray = np.array(to_rgb(schema.colors["background"]))
    image: np.ndarray = FADE_ALPHA * render_semantic(background, schema) + (1 - FADE_ALPHA) * white
    shown: np.ndarray = overlay.any(axis=0)
    image[shown] = render_semantic(overlay, schema)[shown]
    return image


def prediction_panels(case: Case, prediction: np.ndarray, schema: Schema, threshold: float) -> list[np.ndarray]:
    """The four panels of one case (see ``TITLES``).

    Args:
        case (Case): Case.
        prediction (np.ndarray): (C, H, W) predicted missing part (scores).
        schema (Schema): Taxonomy.
        threshold (float): Decoding threshold.

    Returns:
        list[np.ndarray]: Four float (H, W, 3) images.
    """
    base: LabelMaps = semantic_to_labels(case.incomplete, schema, 0.5)
    composed: LabelMaps = compose(base, semantic_to_labels(prediction, schema, threshold))
    return [render_semantic(case.incomplete, schema),
            _highlight(case.incomplete, labels_to_semantic(added(base, composed), schema), schema),
            render_semantic(labels_to_semantic(composed, schema), schema),
            _highlight(case.full, case.target, schema)]


def save_examples(rows: list[tuple[Case, np.ndarray, str]], schema: Schema, threshold: float, title: str, path: Path) -> None:
    """Save a grid of prediction examples.

    Args:
        rows (list[tuple[Case, np.ndarray, str]]): (case, prediction (C, H, W), row label).
        schema (Schema): Taxonomy.
        threshold (float): Decoding threshold.
        title (str): Figure title.
        path (Path): Output PNG.
    """
    fig, axes = plt.subplots(len(rows), len(TITLES), squeeze=False,
                             figsize=(PANEL_INCHES * len(TITLES), PANEL_INCHES * len(rows) + 1.2))
    for r, (case, prediction, label) in enumerate(rows):
        for c, panel in enumerate(prediction_panels(case, prediction, schema, threshold)):
            axes[r, c].imshow(panel, interpolation="nearest")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(TITLES[c], fontsize=11)
        axes[r, 0].set_ylabel(label, fontsize=7)
    fig.legend(handles=[Patch(color=schema.colors[n], label=n) for n in schema.channels],
               loc="lower center", ncol=schema.num_channels, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
