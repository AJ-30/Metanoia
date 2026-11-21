"""Colour rendering of semantic maps and case comparison panels.

Kept separate from the CLI so later stages (model sampling, evaluation) can
render predictions with exactly the same colours and panel layout.
"""

import numpy as np
from matplotlib.colors import to_rgb

from Reconstruction.data_prep.missing import Case
from Reconstruction.data_prep.schema import Schema

# Blend factor used to fade the complete plan behind the highlighted target.
FADE_ALPHA: float = 0.3
# Opacity of the erased-mask tint in the mask panel.
MASK_ALPHA: float = 0.45

PANEL_TITLES: tuple[str, ...] = ("Before (complete)", "After removal (model input)", "Missing component", "Erased region")


def render_semantic(semantic: np.ndarray, schema: Schema) -> np.ndarray:
    """RGB image of a semantic map: rooms first, then structure on top.

    Args:
        semantic (np.ndarray): (C, H, W) map with values in {0, 1}.
        schema (Schema): Taxonomy and colours.

    Returns:
        np.ndarray: float (H, W, 3) in [0, 1].
    """
    image: np.ndarray = np.empty(semantic.shape[1:] + (3,))
    image[:] = to_rgb(schema.colors["background"])
    # Structure classes are drawn last and in taxonomy order (openings over walls).
    for name in schema.room_classes + schema.structure_classes:
        image[semantic[schema.channel_index(name)] > 0] = to_rgb(schema.colors[name])
    return image


def case_panels(case: Case, schema: Schema) -> list[np.ndarray]:
    """The four comparison panels of a case (see ``PANEL_TITLES``).

    Args:
        case (Case): Generated case.
        schema (Schema): Taxonomy and colours.

    Returns:
        list[np.ndarray]: Four float (H, W, 3) images.
    """
    background: np.ndarray = np.array(to_rgb(schema.colors["background"]))
    before: np.ndarray = render_semantic(case.full, schema)
    after: np.ndarray = render_semantic(case.incomplete, schema)

    # Missing component in full colour over a faded copy of the complete plan.
    missing: np.ndarray = FADE_ALPHA * before + (1 - FADE_ALPHA) * background
    target_pixels: np.ndarray = case.target.any(axis=0)
    missing[target_pixels] = render_semantic(case.target, schema)[target_pixels]

    erased: np.ndarray = after.copy()
    tint: np.ndarray = np.array(to_rgb(schema.colors["mask_outline"]))
    erased[case.mask > 0] = (1 - MASK_ALPHA) * erased[case.mask > 0] + MASK_ALPHA * tint
    return [before, after, missing, erased]
