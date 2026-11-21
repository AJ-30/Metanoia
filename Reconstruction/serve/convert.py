"""Convert ResPlan plans into files the web demo accepts.

ResPlan ships as a pickle, which the public demo never opens. Run this on your
own copy of the release to write one upload file per plan:

* ``<id>.json``: the plan in ResPlan layout with WKT geometries (vector upload);
* ``<id>.png``:  the rasterized plan in the project palette (raster upload);
* ``<id>.npy``:  its (C, H, W) multi-hot semantic map (array upload).

Usage (from the repository root):
    python -m Reconstruction.serve.convert --ids 1 59 69 --out Reconstruction/outputs/serve_uploads
    python -m Reconstruction.serve.convert --split test --count 5 --out Reconstruction/outputs/serve_uploads
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from shapely.geometry.base import BaseGeometry

from Reconstruction.common.config_loader import load_params, load_run_config, resolve_path
from Reconstruction.data_prep.render import render_semantic
from Reconstruction.data_prep.resplan_source import extract_plan_geometry, load_resplan_plans, load_split_ids
from Reconstruction.data_prep.rasterize import rasterize_plan
from Reconstruction.data_prep.schema import Schema

DATA_PREP_DIR: Path = Path(__file__).resolve().parents[1] / "data_prep"


def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", type=int, nargs="*", default=[], help="ResPlan plan ids to export")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None, help="Export from this split")
    parser.add_argument("--count", type=int, default=5, help="Plans taken from --split")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    return parser.parse_args()


def plan_to_json(plan: dict[str, Any]) -> dict[str, Any]:
    """ResPlan plan dict -> JSON-serializable dict (geometries as WKT).

    Args:
        plan (dict[str, Any]): Raw ResPlan plan.

    Returns:
        dict[str, Any]: Same keys; Shapely geometries as WKT strings, graphs dropped.
    """
    out: dict[str, Any] = {}
    for key, value in plan.items():
        if isinstance(value, BaseGeometry):
            if not value.is_empty:
                out[key] = value.wkt
        elif isinstance(value, (int, float, str)) and not isinstance(value, bool):
            out[key] = value
    return out


def main() -> None:
    """Write the upload files."""
    args: argparse.Namespace = parse_args()
    config = load_run_config(DATA_PREP_DIR / "config.toml", "prod")
    params = load_params(DATA_PREP_DIR / "params.toml")
    schema: Schema = Schema.from_params(params)
    plans: list[dict[str, Any]] = load_resplan_plans(resolve_path(config.RESPLAN_PICKLE_PATH))
    by_id: dict[int, dict[str, Any]] = {int(p["id"]): p for p in plans}
    ids: list[int] = list(args.ids)
    if args.split:
        ids += load_split_ids(resolve_path(config.RESPLAN_SPLIT_PATH), False)[args.split][:args.count]
    if not ids:
        raise SystemExit("give --ids and/or --split")
    args.out.mkdir(parents=True, exist_ok=True)
    class_map: dict[str, str] = dict(vars(params.taxonomy.resplan_class_map))
    for plan_id in ids:
        plan: dict[str, Any] = by_id[plan_id]
        (args.out / f"{plan_id}.json").write_text(json.dumps(plan_to_json(plan)))
        raster = rasterize_plan(extract_plan_geometry(plan, class_map), schema, params)
        semantic: np.ndarray = np.zeros((schema.num_channels,) + raster.room.shape, np.uint8)
        for inst in raster.instances:
            layer: np.ndarray = raster.structure if inst["layer"] == "structure" else raster.room
            semantic[schema.channel_index(inst["cls"])][layer == inst["id"]] = 1
        np.save(args.out / f"{plan_id}.npy", semantic)
        rgb: np.ndarray = np.round(render_semantic(semantic, schema) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(args.out / f"{plan_id}.png")
        print(f"plan {plan_id}: {args.out / f'{plan_id}'}.{{json,png,npy}} ({raster.metres_per_px:.4f} m/px)")


if __name__ == "__main__":
    main()
