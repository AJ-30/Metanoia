"""Split a raster wall mask into individual wall-segment instances.

ResPlan stores walls as one union whose pieces already stop at door/window
openings, but a single piece can still contain T/X junctions. To remove
"one wall" we need finer instances, so each piece is cut at the junctions of
its skeleton:

1. Skeletonize the (upscaled) wall mask to get 1-px centre lines.
2. Junction pixels = skeleton pixels with 3+ skeleton neighbours.
3. Erase a small disk around every junction; the remaining skeleton breaks
   into branches. Branches shorter than a threshold are spurs (artefacts of
   wall thickness at corners/ends) and are discarded.
4. Every wall pixel joins the nearest surviving branch *of its own connected
   wall piece* (Euclidean distance transform), giving one label per segment.
5. Labels are sampled down to the canvas resolution.

L-shaped corners have no junction, so an L-shaped wall run stays one segment.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import disk, skeletonize

# 8-connectivity for labelling thin skeleton branches.
_EIGHT_CONNECTED: np.ndarray = np.ones((3, 3), dtype=bool)


def _skeleton_branches(mask_up: np.ndarray, junction_radius_up: int, min_length_up: int) -> np.ndarray:
    """Cut the skeleton at junctions and keep branches long enough to be walls.

    Args:
        mask_up (np.ndarray): Bool wall mask at upscaled resolution, (H, W).
        junction_radius_up (int): Disk radius (upscaled px) erased at junctions.
        min_length_up (int): Minimum branch length in skeleton pixels.

    Returns:
        np.ndarray: int32 (H, W) map of branch ids (0 = none), ids 1..K.
    """
    skeleton: np.ndarray = skeletonize(mask_up)
    # Count 8-neighbours of each skeleton pixel (the kernel includes the
    # pixel itself, hence the -1).
    neighbours: np.ndarray = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    junctions: np.ndarray = skeleton & (neighbours >= 3)
    cut: np.ndarray = skeleton & ~ndimage.binary_dilation(junctions, structure=disk(junction_radius_up))
    branches, count = ndimage.label(cut, structure=_EIGHT_CONNECTED)
    if count == 0:
        return np.zeros(mask_up.shape, dtype=np.int32)
    lengths: np.ndarray = np.bincount(branches.ravel(), minlength=count + 1)
    keep: np.ndarray = lengths >= min_length_up
    keep[0] = False
    # Relabel surviving branches consecutively; dropped branches become 0.
    lut: np.ndarray = np.zeros(count + 1, dtype=np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lut[branches]


def _assign_pixels_to_branches(mask_up: np.ndarray, branches: np.ndarray) -> np.ndarray:
    """Label every wall pixel with the nearest branch of its own wall piece.

    Pieces without any surviving branch (short stubs) become one segment.

    Args:
        mask_up (np.ndarray): Bool wall mask, (H, W).
        branches (np.ndarray): int32 branch ids from ``_skeleton_branches``.

    Returns:
        np.ndarray: int32 (H, W) segment labels (0 = not wall).
    """
    pieces, piece_count = ndimage.label(mask_up, structure=_EIGHT_CONNECTED)
    labels: np.ndarray = np.zeros(mask_up.shape, dtype=np.int32)
    next_label: int = int(branches.max()) + 1
    for piece_id, piece_slice in enumerate(ndimage.find_objects(pieces), start=1):
        in_piece: np.ndarray = pieces[piece_slice] == piece_id
        seeds: np.ndarray = np.where(in_piece, branches[piece_slice], 0)
        if not seeds.any():
            labels[piece_slice][in_piece] = next_label
            next_label += 1
            continue
        # Nearest seed pixel for every pixel of the crop; restricted to seeds
        # of this piece so labels never jump across a gap to another piece.
        _, (rows, cols) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
        labels[piece_slice][in_piece] = seeds[rows, cols][in_piece]
    assert piece_count == 0 or labels[mask_up].min() > 0, "every wall pixel must receive a segment label"
    return labels


def _downsample_labels(labels_up: np.ndarray, mask_canvas: np.ndarray, upscale: int) -> np.ndarray:
    """Transfer upscaled segment labels onto the canvas-resolution wall mask.

    Each canvas pixel takes the label at the centre of its upscaled block, or
    the nearest labelled pixel when the two rasterizations disagree slightly.

    Args:
        labels_up (np.ndarray): int32 labels at upscaled resolution.
        mask_canvas (np.ndarray): Bool wall mask at canvas resolution.
        upscale (int): Upscale factor between the two resolutions.

    Returns:
        np.ndarray: int32 labels at canvas resolution (0 = not wall).
    """
    _, (rows, cols) = ndimage.distance_transform_edt(labels_up == 0, return_indices=True)
    centres: np.ndarray = np.arange(mask_canvas.shape[0]) * upscale + upscale // 2
    nearest: np.ndarray = labels_up[rows[np.ix_(centres, centres)], cols[np.ix_(centres, centres)]]
    return np.where(mask_canvas, nearest, 0).astype(np.int32)


def split_walls(mask_up: np.ndarray, mask_canvas: np.ndarray, upscale: int,
                junction_radius_px: int, min_segment_length_px: int) -> np.ndarray:
    """Split walls into segment instances.

    Args:
        mask_up (np.ndarray): Bool wall mask rasterized at ``upscale`` x canvas size.
        mask_canvas (np.ndarray): Bool wall mask at canvas size (square).
        upscale (int): Integer ratio between the two masks (>= 1).
        junction_radius_px (int): Junction clearing radius in canvas px.
        min_segment_length_px (int): Minimum branch length in canvas px.

    Returns:
        np.ndarray: int32 canvas labels, 0 = no wall, 1..K = wall segments
        (ids may have gaps; the caller relabels).
    """
    assert upscale >= 1, "upscale must be >= 1"
    assert mask_up.shape == (mask_canvas.shape[0] * upscale, mask_canvas.shape[1] * upscale), "mask sizes disagree with upscale"
    if not mask_up.any():
        return np.zeros(mask_canvas.shape, dtype=np.int32)
    branches: np.ndarray = _skeleton_branches(mask_up, junction_radius_px * upscale, min_segment_length_px * upscale)
    labels_up: np.ndarray = _assign_pixels_to_branches(mask_up, branches)
    return _downsample_labels(labels_up, mask_canvas, upscale)
