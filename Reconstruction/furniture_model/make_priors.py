"""Measure per-class furniture shape priors on the TRAIN split.

Used by the post-processing and the placement-agnostic evaluation to decide
what a plausible piece looks like (area, side lengths). Written to
``furniture_priors.json`` next to this file.

Usage (from the repository root):
    python -m Reconstruction.furniture_model.make_priors [--plans 1500]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from Reconstruction.common.config_loader import load_run_config, resolve_path
from Reconstruction.data_prep.case_source import CaseSource

HERE: Path = Path(__file__).resolve().parent
PRIORS_FILE: Path = HERE / "furniture_priors.json"
QUANTILES: tuple[float, float] = (0.02, 0.98)
EIGHT: np.ndarray = np.ones((3, 3), bool)


def rect_sides(mask: np.ndarray) -> tuple[float, float, float]:
    """Min-area rectangle of a mask.

    Args:
        mask (np.ndarray): bool (H, W) single component.

    Returns:
        tuple[float, float, float]: (long side, short side, rectangularity =
        area / rectangle area).
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points: np.ndarray = np.concatenate(contours)
    (_, _), (w, h), _ = cv2.minAreaRect(points)
    long_side, short_side = max(w, h), min(w, h)
    return long_side, short_side, float(mask.sum()) / max(long_side * short_side, 1.0)


def main() -> None:
    """Compute and save the priors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=int, default=1500)
    parser.add_argument("--mode", choices=["debug", "prod"], default="prod")
    args = parser.parse_args()
    config = load_run_config(HERE / "config.toml", args.mode)
    source = CaseSource(resolve_path(config.PROCESSED_ROOT) / "furniture", "train", seed=0)
    schema = source.schema
    furniture: list[str] = [c for c in schema.structure_classes if schema.component_type[c] == "furniture"]
    stats: dict[str, dict[str, list[float]]] = {c: {"area": [], "long": [], "short": []} for c in furniture}
    for index in np.linspace(0, len(source) - 1, min(args.plans, len(source))).astype(int):
        full: np.ndarray = source.case(int(index)).full
        for cls in furniture:
            pieces, count = ndimage.label(full[schema.structure_classes.index(cls)] > 0, structure=EIGHT)
            for piece in range(1, count + 1):
                mask = pieces == piece
                if mask.sum() < 4:
                    continue
                long_side, short_side, rectangularity = rect_sides(mask)
                # Touching same-class pieces merge into non-rectangular blobs: skip them.
                if rectangularity < 0.9:
                    continue
                stats[cls]["area"].append(float(mask.sum()))
                stats[cls]["long"].append(long_side)
                stats[cls]["short"].append(short_side)
    priors: dict[str, dict[str, float]] = {}
    for cls, values in stats.items():
        if not values["area"]:
            continue
        priors[cls] = {"count": len(values["area"])}
        for key, series in values.items():
            low, high = np.quantile(series, QUANTILES)
            priors[cls].update({f"{key}_min": float(low), f"{key}_max": float(high), f"{key}_median": float(np.median(series))})
    PRIORS_FILE.write_text(json.dumps(priors, indent=2))
    for cls, row in priors.items():
        print(f"{cls:13s} n={row['count']:5d} area {row['area_min']:.0f}-{row['area_max']:.0f} "
              f"long {row['long_min']:.0f}-{row['long_max']:.0f} short {row['short_min']:.0f}-{row['short_max']:.0f}")


if __name__ == "__main__":
    main()
