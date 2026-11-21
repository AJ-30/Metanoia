"""Near-duplicate detection between the train split and val/test splits.

ResPlan reports that some listings were republished, so a val/test plan can
have a near-copy in train. Such plans are excluded from evaluation cases to
prevent leakage (the train split itself is left untouched).

Similarity of two plans = fraction of *occupied* thumbnail pixels (non-empty
in either plan) whose class labels agree. Plans are compared in the fixed
canvas framing produced by rasterization; rotated/flipped copies are not
searched for.
"""

import numpy as np

# The fine comparison runs only on candidates whose coarse score is within
# this margin of the threshold (coarse thumbnails are 4x smaller per side).
_COARSE_MARGIN: float = 0.15
_COARSE_STEP: int = 4


def _agreement(query: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Label agreement of one flattened thumbnail against many.

    Args:
        query (np.ndarray): uint8 (P,) flattened thumbnail.
        pool (np.ndarray): uint8 (N, P) flattened thumbnails.

    Returns:
        np.ndarray: float (N,) agreement in [0, 1].
    """
    occupied: np.ndarray = (pool > 0) | (query > 0)
    agree: np.ndarray = (pool == query) & occupied
    return agree.sum(axis=1) / np.maximum(occupied.sum(axis=1), 1)


def find_near_duplicates(train_thumbs: np.ndarray, eval_thumbs: np.ndarray, threshold: float) -> list[dict[str, float | int]]:
    """Find eval plans whose best train match reaches ``threshold``.

    Args:
        train_thumbs (np.ndarray): uint8 (N_train, S, S) label thumbnails.
        eval_thumbs (np.ndarray): uint8 (N_eval, S, S) label thumbnails.
        threshold (float): Agreement at or above which a plan is a duplicate.

    Returns:
        list[dict[str, float | int]]: One entry per duplicate with keys
        ``eval_index``, ``train_index`` and ``agreement``.
    """
    coarse_train: np.ndarray = train_thumbs[:, ::_COARSE_STEP, ::_COARSE_STEP].reshape(len(train_thumbs), -1)
    fine_train: np.ndarray = train_thumbs.reshape(len(train_thumbs), -1)
    duplicates: list[dict[str, float | int]] = []
    for eval_index, thumb in enumerate(eval_thumbs):
        coarse: np.ndarray = _agreement(thumb[::_COARSE_STEP, ::_COARSE_STEP].ravel(), coarse_train)
        candidates: np.ndarray = np.nonzero(coarse >= threshold - _COARSE_MARGIN)[0]
        if candidates.size == 0:
            continue
        fine: np.ndarray = _agreement(thumb.ravel(), fine_train[candidates])
        best: int = int(np.argmax(fine))
        if fine[best] >= threshold:
            duplicates.append({"eval_index": eval_index, "train_index": int(candidates[best]), "agreement": float(fine[best])})
    return duplicates
