"""Inference engine of the web demo: uploads -> plan -> diffusion -> panels.

Everything model-related reuses the package as it is used for evaluation:
plans are rasterized by ``data_prep``, cases are built by the frozen
``MissingComponentGenerator``, sampling is ``model.sampling.sample_missing``,
K samples are fused and cleaned by ``postprocess``, predictions are composed
with the input by ``evaluation.labels`` and scored by
``evaluation.case_metrics``. So a request produces exactly what
``predict.py`` + ``postprocess/run.py`` + ``evaluate.py`` would.

Accepted uploads (pickles are never opened; see ``convert.py``):

* Plan JSON in ResPlan layout: one object whose keys are ResPlan categories
  (``wall``, ``door``, ``front_door``, ``window``, ``living``, ``bedroom``, ...)
  with geometries as WKT strings, GeoJSON geometries / features, or lists of
  coordinate rings. Optional ``id``, ``area`` (m^2) + ``inner`` (footprint)
  give the metric scale, as in ResPlan; ``metres_per_unit`` overrides it.
* Raster image (PNG / JPEG / WebP): a plan drawn in the project palette
  (``params.toml [colors]``, as rendered by ``data_prep/render.py``).
* Array (``.npy`` / ``.npz``): a (C, H, W) or (H, W, C) multi-hot semantic map
  in the project's channel order.
* FurnitureArranger profiles also take the dataset's own layout: an SVG (read only
  through its embedded ``<metadata>`` JSON, never parsed as XML) or a JSON with
  ``floorplan`` (WKT per category) and ``arrangement.furniture``.

The engine serves one *profile* (a trained model with its dataset parameters).
The ResPlan profile is the default and is configured by ``config.toml``; further
profiles come from ``profiles.toml`` (see ``app.py``).
"""

import base64
import binascii
import io
import json
import math
import secrets
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from matplotlib.colors import to_rgb
from PIL import Image
from scipy import ndimage
from shapely import wkt
from shapely.errors import ShapelyError
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from Reconstruction.common.config_loader import load_params, resolve_path, to_namespace
from Reconstruction.data_prep.case_source import load_generator
from Reconstruction.data_prep.furniture_source import layout_to_geometry, parse_layout
from Reconstruction.data_prep.missing import EASY, HARD, Case, CaseSpec
from Reconstruction.data_prep.plan_store import PlanStore
from Reconstruction.data_prep.rasterize import _finalize_layer, rasterize_plan
from Reconstruction.data_prep.render import FADE_ALPHA, MASK_ALPHA, render_semantic
from Reconstruction.data_prep.resplan_source import extract_plan_geometry
from Reconstruction.data_prep.schema import LAYERS, ROOM_LAYER, STRUCTURE_LAYER, Schema
from Reconstruction.data_prep.wall_split import split_walls
from Reconstruction.evaluation.case_metrics import prepare_case, score_samples
from Reconstruction.evaluation.components import extract_components
from Reconstruction.evaluation.labels import EMPTY, LabelMaps, added, compose, labels_to_semantic, semantic_to_labels
from Reconstruction.model.checkpoint import resolve_checkpoints
from Reconstruction.model.denoising import scheduled_snapshots
from Reconstruction.model.diffusion import GaussianDiffusion
from Reconstruction.model.sampling import sample_missing
from Reconstruction.model.unet import ConditionalUNet
from Reconstruction.postprocess.run import postprocess_case

SERVE_DIR: Path = Path(__file__).resolve().parent
PACKAGE_DIR: Path = SERVE_DIR.parent
PARAMS_FILE: Path = SERVE_DIR / "params.toml"
EVAL_PARAMS_FILE: Path = PACKAGE_DIR / "evaluation" / "params.toml"
POSTPROCESS_PARAMS_FILE: Path = PACKAGE_DIR / "postprocess" / "params.toml"
SCORE_SCALE: int = 255
# Extra keys a plan JSON may carry besides the class geometries.
_PLAN_META_KEYS: tuple[str, ...] = ("id", "area", "inner", "metres_per_unit")
# ResPlan's own key typo (see resplan_source._TYPO_KEYS).
_KEY_ALIASES: dict[str, str] = {"balacony": "balcony"}
# Legend order of the page: shell classes first, then furniture, then rooms.
_LEGEND_FIRST: tuple[str, ...] = ("wall", "door", "window", "railing", "column")


class UploadError(ValueError):
    """An upload or request the demo cannot process; the message is shown to the user."""


@dataclass
class Plan:
    """A parsed plan in the dataset's store format.

    Attributes:
        layers (dict[str, np.ndarray]): uint8 (H, W) instance ids per layer.
        record (dict[str, Any]): ``plan_id``, ``metres_per_px`` and ``instances``
            (as in ``plans.jsonl``).
        source (str): What was uploaded ("plan-json", "image", "array", "example").
        notes (list[str]): Messages about assumptions made while parsing.
    """

    layers: dict[str, np.ndarray]
    record: dict[str, Any]
    source: str
    notes: list[str]


# ----------------------------------------------------------------------------
# Uploads -> Plan
# ----------------------------------------------------------------------------
def _to_geometry(value: Any) -> BaseGeometry:
    """Parse one category's geometry from WKT, GeoJSON or coordinate rings.

    Args:
        value (Any): WKT string, GeoJSON geometry / Feature / FeatureCollection,
            or a list of rings ([[x, y], ...]) or polygons ([[ring], [hole], ...]).

    Returns:
        BaseGeometry: Valid (Multi)Polygon geometry.

    Raises:
        UploadError: If the value cannot be read as polygons.
    """
    try:
        if isinstance(value, str):
            geom: BaseGeometry = wkt.loads(value)
        elif isinstance(value, dict) and value.get("type") == "FeatureCollection":
            geom = unary_union([shape(f["geometry"]) for f in value.get("features", [])])
        elif isinstance(value, dict) and value.get("type") == "Feature":
            geom = shape(value["geometry"])
        elif isinstance(value, dict):
            geom = shape(value)
        elif isinstance(value, list):
            polygons: list[Polygon] = []
            for item in value:
                # A ring is a list of [x, y] points; a polygon is a list of rings (shell first).
                is_ring: bool = isinstance(item[0][0], (int, float))
                polygons.append(Polygon(item) if is_ring else Polygon(item[0], item[1:]))
            geom = MultiPolygon(polygons) if len(polygons) != 1 else polygons[0]
        else:
            raise TypeError(type(value).__name__)
    except Exception as error:  # shapely raises many types on malformed input
        raise UploadError(f"could not read a geometry ({error})") from None
    return geom if geom.is_valid else geom.buffer(0)


def _count_vertices(geom: BaseGeometry) -> int:
    """Number of coordinates in a geometry (upload size guard).

    Args:
        geom (BaseGeometry): Geometry.

    Returns:
        int: Vertex count.
    """
    if hasattr(geom, "geoms"):
        return sum(_count_vertices(part) for part in geom.geoms)
    if isinstance(geom, Polygon):
        return len(geom.exterior.coords) + sum(len(ring.coords) for ring in geom.interiors)
    return len(getattr(geom, "coords", []))


def plan_from_json(raw: bytes, schema: Schema, params: SimpleNamespace, cfg: SimpleNamespace) -> Plan:
    """Rasterize a ResPlan-style plan JSON exactly like the dataset build.

    Args:
        raw (bytes): Uploaded JSON.
        schema (Schema): Taxonomy.
        params (SimpleNamespace): Dataset build parameters.
        cfg (SimpleNamespace): ``params.toml [upload]``.

    Returns:
        Plan: Parsed plan.

    Raises:
        UploadError: On malformed JSON, unknown content or oversized geometry.
    """
    try:
        data: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UploadError(f"not valid JSON ({error})") from None
    notes: list[str] = []
    if isinstance(data, list):
        if not data or not isinstance(data[0], dict):
            raise UploadError("the JSON list holds no plan objects")
        if len(data) > 1:
            notes.append(f"the file holds {len(data)} plans; using the first")
        data = data[0]
    if not isinstance(data, dict):
        raise UploadError("expected a JSON object with ResPlan category keys")
    class_map: dict[str, str] = dict(vars(params.taxonomy.resplan_class_map))
    plan: dict[str, Any] = {"id": int(data.get("id") or 0)}
    vertices: int = 0
    for key, value in data.items():
        key = _KEY_ALIASES.get(key, key)
        if key in class_map or key == "inner":
            plan[key] = _to_geometry(value)
            vertices += _count_vertices(plan[key])
    if vertices > cfg.MAX_VERTICES:
        raise UploadError(f"the plan has {vertices} vertices (limit {cfg.MAX_VERTICES})")
    if not any(key in plan for key in class_map):
        raise UploadError(f"no plan categories found; expected keys such as {', '.join(sorted(class_map)[:8])}")
    if "wall" not in plan or plan["wall"].is_empty:
        raise UploadError("the plan has no 'wall' geometry")
    ignored: list[str] = sorted(k for k in data if _KEY_ALIASES.get(k, k) not in class_map and k not in _PLAN_META_KEYS)
    if ignored:
        notes.append(f"ignored keys (context only, as in the dataset): {', '.join(ignored[:10])}")
    if data.get("area") is not None:
        plan["area"] = float(data["area"])
    geometry = extract_plan_geometry(plan, class_map)
    if data.get("metres_per_unit") is not None:
        geometry.metres_per_unit = float(data["metres_per_unit"])
    try:
        raster = rasterize_plan(geometry, schema, params)
    except ValueError as error:
        raise UploadError(str(error)) from None
    scale: float = raster.metres_per_px
    if not math.isfinite(scale):
        scale = cfg.DEFAULT_METRES_PER_PX
        notes.append(f"no 'area' + 'inner' or 'metres_per_unit' given; assuming {scale} m/px for distances")
    record: dict[str, Any] = {"plan_id": raster.plan_id, "metres_per_px": scale, "instances": raster.instances}
    return Plan({STRUCTURE_LAYER: raster.structure, ROOM_LAYER: raster.room}, record, "plan-json", notes)


def plan_from_furniture(raw: bytes, schema: Schema, params: SimpleNamespace, cfg: SimpleNamespace) -> Plan:
    """Rasterize a FurnitureArranger layout (SVG or JSON) exactly like the dataset build.

    Furniture is drawn only if the profile's taxonomy has furniture classes (the
    shell-only profile has none, so the furniture list is ignored).

    Args:
        raw (bytes): Uploaded SVG (with its ``<metadata>`` block) or layout JSON.
        schema (Schema): Taxonomy.
        params (SimpleNamespace): Dataset build parameters (``params_furniture*.toml`` copy).
        cfg (SimpleNamespace): ``params.toml [upload]``.

    Returns:
        Plan: Parsed plan.

    Raises:
        UploadError: On a malformed layout, oversized geometry or a plan that does not fit the canvas.
    """
    try:
        layout: dict[str, Any] = parse_layout(raw.decode("utf-8"))
    except UnicodeDecodeError:
        raise UploadError("the file is not UTF-8 text") from None
    except ValueError as error:
        raise UploadError(str(error)) from None
    try:
        geometry = layout_to_geometry(0, layout, params)
        vertices: int = _count_vertices(geometry.wall) + sum(_count_vertices(g) for _, g in geometry.instances)
        if vertices > cfg.MAX_VERTICES:
            raise UploadError(f"the plan has {vertices} vertices (limit {cfg.MAX_VERTICES})")
        raster = rasterize_plan(geometry, schema, params)
    except (ValueError, KeyError, TypeError, ShapelyError) as error:
        if isinstance(error, UploadError):
            raise
        raise UploadError(f"the layout could not be read ({error})") from None
    notes: list[str] = []
    if layout["furniture"] and not any(schema.component_type[c] == "furniture" for c in schema.channels):
        notes.append("this model works on the building shell only: the furniture in the layout is not drawn")
    record: dict[str, Any] = {"plan_id": raster.plan_id, "metres_per_px": raster.metres_per_px, "instances": raster.instances}
    return Plan({STRUCTURE_LAYER: raster.structure, ROOM_LAYER: raster.room}, record, "layout", notes)


def _fit_canvas(image: Image.Image, size: int) -> tuple[Image.Image, list[str]]:
    """Pad an image to a square on white and resize it to the canvas (nearest).

    Args:
        image (Image.Image): RGB image.
        size (int): Canvas size.

    Returns:
        tuple[Image.Image, list[str]]: Canvas-sized image and notes.
    """
    notes: list[str] = []
    width, height = image.size
    if width != height:
        side: int = max(width, height)
        square: Image.Image = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(image, ((side - width) // 2, (side - height) // 2))
        image = square
        notes.append(f"padded the {width}x{height} image to a square")
    if image.size != (size, size):
        notes.append(f"resized {image.size[0]}x{image.size[1]} to {size}x{size} (nearest); thin openings may be lost")
        image = image.resize((size, size), Image.NEAREST)
    return image, notes


def labels_from_image(raw: bytes, schema: Schema, size: int, cfg: SimpleNamespace) -> tuple[LabelMaps, list[str]]:
    """Read a palette-coloured plan drawing into label maps.

    Each pixel takes the nearest of the class colours or the background colour;
    pixels far from all of them are background. Rendered plans draw structure
    over rooms, so rooms are grown one pixel under their walls, as in the data.

    Args:
        raw (bytes): Image file.
        schema (Schema): Taxonomy and colours.
        size (int): Canvas size.
        cfg (SimpleNamespace): ``params.toml [upload]``.

    Returns:
        tuple[LabelMaps, list[str]]: Label maps and notes.

    Raises:
        UploadError: If the image cannot be decoded or is too large.
    """
    Image.MAX_IMAGE_PIXELS = cfg.MAX_IMAGE_PIXELS
    try:
        image: Image.Image = Image.open(io.BytesIO(raw))
        image.load()
    except (Image.DecompressionBombError, OSError, ValueError) as error:
        raise UploadError(f"could not read the image ({error})") from None
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        white: Image.Image = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image)
    image, notes = _fit_canvas(image.convert("RGB"), size)
    pixels: np.ndarray = np.asarray(image, np.float32).reshape(-1, 3)
    names: list[str] = list(schema.structure_classes) + list(schema.room_classes) + ["background"]
    palette: np.ndarray = np.array([to_rgb(schema.colors[n]) for n in names], np.float32) * 255.0
    distance: np.ndarray = np.linalg.norm(pixels[:, None, :] - palette[None], axis=2)
    nearest: np.ndarray = distance.argmin(axis=1)
    nearest[distance.min(axis=1) > cfg.COLOUR_TOLERANCE] = len(names) - 1
    n_struct: int = len(schema.structure_classes)
    nearest = nearest.reshape(size, size)
    structure: np.ndarray = np.where(nearest < n_struct, nearest, EMPTY).astype(np.int8)
    room: np.ndarray = np.where((nearest >= n_struct) & (nearest < len(names) - 1), nearest - n_struct, EMPTY).astype(np.int8)
    grown: np.ndarray = ndimage.grey_dilation(room.astype(np.int16) + 1, size=3) - 1
    room = np.where((room == EMPTY) & (structure != EMPTY), grown, room).astype(np.int8)
    if not (structure == schema.structure_classes.index("wall")).any():
        raise UploadError("no wall-coloured pixels found; draw the plan in the palette shown on the page")
    return LabelMaps(structure, room), notes


def labels_from_array(raw: bytes, schema: Schema, size: int) -> tuple[LabelMaps, list[str]]:
    """Read a multi-hot semantic map (.npy / .npz, never pickled) into label maps.

    Args:
        raw (bytes): File content.
        schema (Schema): Taxonomy.
        size (int): Canvas size.

    Returns:
        tuple[LabelMaps, list[str]]: Label maps and notes.

    Raises:
        UploadError: If the array is unreadable or has the wrong shape.
    """
    notes: list[str] = []
    try:
        if raw[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 64 * schema.num_channels * size * size:
                    raise UploadError("the .npz expands to more data than a plan can hold")
            with np.load(io.BytesIO(raw), allow_pickle=False) as bundle:
                key: str = "semantic" if "semantic" in bundle.files else bundle.files[0]
                array: np.ndarray = bundle[key]
                notes.append(f"using array '{key}' from the .npz")
        else:
            array = np.load(io.BytesIO(raw), allow_pickle=False)
    except UploadError:
        raise
    except Exception as error:  # numpy raises several types on malformed files
        raise UploadError(f"could not read the array ({error})") from None
    channels: int = schema.num_channels
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == channels and array.shape[0] != channels:
        array = np.moveaxis(array, -1, 0)
    if array.ndim != 3 or array.shape != (channels, size, size):
        raise UploadError(f"expected a ({channels}, {size}, {size}) semantic map, got shape {tuple(array.shape)}")
    scores: np.ndarray = array.astype(np.float32)
    if scores.max() > 1.0:
        scores /= 255.0
    return semantic_to_labels(scores, schema, 0.5), notes


def plan_from_labels(labels: LabelMaps, schema: Schema, params: SimpleNamespace, metres_per_px: float,
                     source: str, notes: list[str]) -> Plan:
    """Turn label maps into instance layers the way the dataset defines instances.

    Walls are split into segments at junctions (``wall_split``); every other
    class becomes one instance per connected piece.

    Args:
        labels (LabelMaps): Structure / room class labels.
        schema (Schema): Taxonomy.
        params (SimpleNamespace): Dataset build parameters.
        metres_per_px (float): Plan scale.
        source (str): Upload kind.
        notes (list[str]): Notes collected so far.

    Returns:
        Plan: Parsed plan.

    Raises:
        UploadError: If a layer has more instances than the store format allows.
    """
    layers: dict[str, np.ndarray] = {}
    instances: list[dict[str, Any]] = []
    for layer, label_map, classes in ((STRUCTURE_LAYER, labels.structure, schema.structure_classes),
                                      (ROOM_LAYER, labels.room, schema.room_classes)):
        provisional: np.ndarray = np.zeros(label_map.shape, np.int32)
        names: dict[int, str] = {}
        for index, cls in enumerate(classes):
            mask: np.ndarray = label_map == index
            if not mask.any():
                continue
            if cls == "wall":
                pieces: np.ndarray = split_walls(mask, mask, 1, params.wall_split.JUNCTION_CLEAR_RADIUS_PX,
                                                 params.wall_split.MIN_SEGMENT_LENGTH_PX)
            else:
                pieces, _ = ndimage.label(mask)
            offset: int = max(names, default=0)
            for piece in np.unique(pieces[pieces > 0]):
                provisional[pieces == piece] = offset + int(piece)
                names[offset + int(piece)] = cls
        try:
            layers[layer], meta = _finalize_layer(provisional, names, {}, layer, params.canvas.MIN_INSTANCE_AREA_PX)
        except ValueError as error:
            raise UploadError(str(error)) from None
        instances.extend(meta)
    record: dict[str, Any] = {"plan_id": 0, "metres_per_px": metres_per_px, "instances": instances}
    return Plan(layers, record, source, notes)


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def to_png(image: np.ndarray, upscale: int) -> str:
    """Encode a float RGB image in [0, 1] as a PNG data URI.

    Args:
        image (np.ndarray): (H, W, 3) floats.
        upscale (int): Nearest-neighbour upscale factor.

    Returns:
        str: ``data:image/png;base64,...``.
    """
    pixels: np.ndarray = np.clip(np.round(image * 255), 0, 255).astype(np.uint8)
    if upscale > 1:
        pixels = pixels.repeat(upscale, axis=0).repeat(upscale, axis=1)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def overlay(base: np.ndarray, extra: LabelMaps, schema: Schema) -> np.ndarray:
    """Added labels in full colour over a faded rendering of a plan.

    Args:
        base (np.ndarray): (C, H, W) semantic map shown faded.
        extra (LabelMaps): Labels drawn on top (-1 = nothing).
        schema (Schema): Taxonomy and colours.

    Returns:
        np.ndarray: float (H, W, 3).
    """
    background: np.ndarray = np.array(to_rgb(schema.colors["background"]))
    image: np.ndarray = FADE_ALPHA * render_semantic(base, schema) + (1 - FADE_ALPHA) * background
    on: np.ndarray = (extra.structure != EMPTY) | (extra.room != EMPTY)
    image[on] = render_semantic(labels_to_semantic(extra, schema), schema)[on]
    return image


def tint(image: np.ndarray, mask: np.ndarray, schema: Schema) -> np.ndarray:
    """Tint a region of an image with the mask colour (erased region).

    Args:
        image (np.ndarray): float (H, W, 3).
        mask (np.ndarray): (H, W) region.
        schema (Schema): Colours.

    Returns:
        np.ndarray: float (H, W, 3).
    """
    out: np.ndarray = image.copy()
    region: np.ndarray = mask > 0
    out[region] = (1 - MASK_ALPHA) * out[region] + MASK_ALPHA * np.array(to_rgb(schema.colors["mask_outline"]))
    return out


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
class Engine:
    """Loaded model, taxonomy and parsed-plan cache; one instance per process.

    Not thread-safe for inference: the caller serializes ``predict`` calls.
    """

    def __init__(self, config: SimpleNamespace) -> None:
        """Load the dataset parameters, the example store and the checkpoint.

        Args:
            config (SimpleNamespace): ``serve/config.toml`` for the chosen mode. A profile
                from ``profiles.toml`` is selected by overriding ``DATASET``, ``EXAMPLE_SPLIT``
                and ``CHECKPOINT_FILE`` and setting ``PROFILE`` (see ``app.profile_config``);
                without ``PROFILE`` the engine serves ResPlan exactly as before.

        Raises:
            RuntimeError: If no CUDA device is available.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("the demo serves the diffusion model on a GPU; no CUDA device found")
        self.device: torch.device = torch.device("cuda")
        self.cfg: SimpleNamespace = load_params(PARAMS_FILE)
        self.eval_params: SimpleNamespace = load_params(EVAL_PARAMS_FILE)
        self.post_params: SimpleNamespace = load_params(POSTPROCESS_PARAMS_FILE)
        self.threshold: float = self.eval_params.decode.THRESHOLD
        processed: Path = resolve_path(config.PROCESSED_ROOT) / config.DATASET
        self.params, self.generator = load_generator(processed)
        self.schema: Schema = self.generator.schema
        self.size: int = self.params.canvas.SIZE
        self.examples = PlanStore(processed / config.EXAMPLE_SPLIT)
        profile: dict[str, Any] = dict(getattr(config, "PROFILE", None) or {})
        self.profile_name: str = str(profile.get("name", "resplan"))
        self.input_kind: str = str(profile.get("input", "resplan"))
        self.dataset_label: str = str(profile.get("dataset_label", "ResPlan"))
        self.description: str | None = profile.get("description")
        self.removable_types: list[str] | None = profile.get("removable_types")
        self.protected_types: list[str] = list(profile.get("protected_types", []))
        self.default_mpp: float = float(self.cfg.upload.DEFAULT_METRES_PER_PX)
        if self.input_kind == "furniture":
            # Fixed scale: one plan unit is one pixel, so the dataset's metre value applies to every plan.
            self.default_mpp = float(self.params.furniture.METRES_PER_UNIT)
        checkpoint_file = getattr(config, "CHECKPOINT_FILE", None)
        if checkpoint_file:
            checkpoint: Path = resolve_checkpoints(resolve_path(str(config.CHECKPOINT_FILE)).parent,
                                                   [str(resolve_path(str(config.CHECKPOINT_FILE)))])[0]
        else:
            checkpoint = resolve_checkpoints(resolve_path(config.RUNS_DIR) / config.RUN_NAME / "checkpoints",
                                             [str(config.CHECKPOINT)])[0]
        # weights_only: a public server never unpickles arbitrary objects.
        state: dict[str, Any] = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.hyper: SimpleNamespace = to_namespace(state["hyperparameters"])
        self.model = ConditionalUNet(self.schema.num_channels, self.size, self.hyper.model).to(self.device)
        self.model.load_state_dict(state["ema"])
        self.model.eval()
        self.step: int = int(state["step"])
        self.checkpoint_name: str = checkpoint.name
        self.diffusion = GaussianDiffusion(self.hyper.diffusion.TIMESTEPS, self.hyper.diffusion.MAX_BETA)
        self.dtype: torch.dtype | None = torch.bfloat16 if self.hyper.train.PRECISION == "bf16" else None
        self.pipelines: dict[str, list[str]] = {k: list(v) for k, v in vars(self.post_params.pipelines).items()}
        self._plans: OrderedDict[str, tuple[float, Plan]] = OrderedDict()
        # Completed plans of recent predictions, kept so the furniture step can place furniture into them.
        self._composed: OrderedDict[str, LabelMaps] = OrderedDict()

    # ---------------------------------------------------------------- plans
    def info(self) -> dict[str, Any]:
        """Static facts for the page: taxonomy, colours, limits, model.

        Returns:
            dict[str, Any]: JSON-serializable info.
        """
        cfg = self.cfg.inference
        legend: list[str] = ([c for c in _LEGEND_FIRST if c in self.schema.structure_classes]
                             + [c for c in self.schema.structure_classes if c not in _LEGEND_FIRST]
                             + list(self.schema.room_classes))
        return {
            "profile": {"name": self.profile_name, "dataset_label": self.dataset_label, "description": self.description,
                        "input": self.input_kind, "removable_types": self.removable_types,
                        "default_metres_per_px": self.default_mpp},
            "legend": legend,
            "structure_classes": list(self.schema.structure_classes), "room_classes": list(self.schema.room_classes),
            "colors": dict(self.schema.colors), "component_type": dict(self.schema.component_type),
            "canvas": self.size, "examples": len(self.examples), "pipelines": list(self.pipelines),
            "model": {"checkpoint": self.checkpoint_name, "step": self.step,
                      "parameters_m": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 2)},
            "limits": {"max_bytes": self.cfg.upload.MAX_BYTES, "max_samples": cfg.MAX_SAMPLES,
                       "min_steps": cfg.MIN_STEPS, "max_steps": cfg.MAX_STEPS, "default_steps": cfg.DEFAULT_STEPS,
                       "margins": list(self.params.missing.MARGINS_PX),
                       "hard_types": list(self.params.missing.HARD_MODE_TYPES)},
        }

    def parse_upload(self, raw: bytes, filename: str, metres_per_px: float | None) -> Plan:
        """Detect the upload format and parse it into a plan.

        Args:
            raw (bytes): File content.
            filename (str): Original file name (only used for format hints).
            metres_per_px (float | None): Scale for raster uploads.

        Returns:
            Plan: Parsed plan.

        Raises:
            UploadError: On unsupported or malformed uploads.
        """
        name: str = filename.lower()
        if name.endswith((".pkl", ".pickle")) or raw[:2] in (b"\x80\x04", b"\x80\x05"):
            raise UploadError("pickles are not accepted; convert ResPlan plans to JSON with "
                              "`python -m Reconstruction.serve.convert` first")
        scale: float = metres_per_px if metres_per_px and metres_per_px > 0 else self.default_mpp
        stripped: bytes = raw.lstrip()[:1]
        if self.input_kind == "furniture":
            head: bytes = raw.lstrip()[:5]
            if head[:4] == b"<svg" or head == b"<?xml" or name.endswith(".svg") or b'"floorplan"' in raw:
                return plan_from_furniture(raw, self.schema, self.params, self.cfg.upload)
        if stripped in (b"{", b"[") or name.endswith((".json", ".geojson")):
            return plan_from_json(raw, self.schema, self.params, self.cfg.upload)
        if raw[:6] == b"\x93NUMPY" or raw[:4] == b"PK\x03\x04" or name.endswith((".npy", ".npz")):
            labels, notes = labels_from_array(raw, self.schema, self.size)
            kind: str = "array"
        else:
            labels, notes = labels_from_image(raw, self.schema, self.size, self.cfg.upload)
            kind = "image"
        if not metres_per_px:
            notes.append(f"no scale given; assuming {scale} m/px for distances")
        return plan_from_labels(labels, self.schema, self.params, scale, kind, notes)

    def example(self, index: int) -> Plan:
        """A plan of the example split (frozen test plans), as stored.

        Args:
            index (int): Plan index.

        Returns:
            Plan: The plan.

        Raises:
            UploadError: If the index is out of range.
        """
        if not 0 <= index < len(self.examples):
            raise UploadError(f"example index must be in [0, {len(self.examples) - 1}]")
        layers: dict[str, np.ndarray] = self.examples.layers(index)
        record: dict[str, Any] = self.examples.record(index)
        return Plan(layers, record, "example", [f"{self.dataset_label} test plan {record['plan_id']} (index {index})"])

    def remember(self, plan: Plan) -> str:
        """Cache a parsed plan and return its token.

        Args:
            plan (Plan): Parsed plan.

        Returns:
            str: Opaque token.
        """
        now: float = time.time()
        ttl: float = self.cfg.cache.PLAN_TTL_S
        for token in [t for t, (stamp, _) in self._plans.items() if now - stamp > ttl]:
            del self._plans[token]
        while len(self._plans) >= self.cfg.cache.MAX_PLANS:
            self._plans.popitem(last=False)
        token = secrets.token_urlsafe(12)
        self._plans[token] = (now, plan)
        return token

    def recall(self, token: str) -> Plan:
        """Fetch a cached plan.

        Args:
            token (str): Token from ``remember``.

        Returns:
            Plan: The plan.

        Raises:
            UploadError: If the token is unknown or expired.
        """
        if token not in self._plans:
            raise UploadError("this plan has expired; upload it again")
        stamp, plan = self._plans[token]
        self._plans[token] = (time.time(), plan)
        return plan

    def remember_composed(self, composed: LabelMaps) -> str:
        """Cache a completed plan for the furniture step and return its token.

        Args:
            composed (LabelMaps): The input plan plus what the model added.

        Returns:
            str: Opaque token for ``recall_composed``.
        """
        while len(self._composed) >= self.cfg.cache.MAX_PLANS:
            self._composed.popitem(last=False)
        token: str = secrets.token_urlsafe(12)
        self._composed[token] = composed
        return token

    def recall_composed(self, token: str) -> LabelMaps:
        """Fetch a completed plan cached by ``remember_composed``.

        Args:
            token (str): Token from the prediction result (``composed_token``).

        Returns:
            LabelMaps: The completed plan.

        Raises:
            UploadError: If the token is unknown or expired.
        """
        if token not in self._composed:
            raise UploadError("this completed plan has expired; run the reconstruction again")
        return self._composed[token]

    def semantic(self, plan: Plan) -> np.ndarray:
        """Multi-hot (C, H, W) map of a plan (as ``MissingComponentGenerator._semantic``).

        Args:
            plan (Plan): Plan.

        Returns:
            np.ndarray: uint8 (C, H, W).
        """
        semantic: np.ndarray = np.zeros((self.schema.num_channels, self.size, self.size), np.uint8)
        for inst in plan.record["instances"]:
            semantic[self.schema.channel_index(inst["cls"])][plan.layers[inst["layer"]] == inst["id"]] = 1
        return semantic

    def describe(self, token: str, plan: Plan) -> dict[str, Any]:
        """Preview of a parsed plan for the page (render, instances, hit-test layers).

        Args:
            token (str): Plan token.
            plan (Plan): Plan.

        Returns:
            dict[str, Any]: JSON-serializable description.
        """
        scale: float = plan.record["metres_per_px"]
        instances = [{"layer": i["layer"], "id": i["id"], "cls": i["cls"],
                      "type": self.schema.component_type[i["cls"]], "area_px": i["area_px"],
                      "area_m2": round(i["area_px"] * scale * scale, 2)} for i in plan.record["instances"]]
        return {
            "token": token, "source": plan.source, "notes": plan.notes, "metres_per_px": scale,
            "image": to_png(render_semantic(self.semantic(plan), self.schema), self.cfg.render.UPSCALE),
            "instances": instances,
            "hit": {layer: base64.b64encode(plan.layers[layer].astype(np.uint8).tobytes()).decode("ascii") for layer in LAYERS},
        }

    # ---------------------------------------------------------------- inference
    def _sample(self, incomplete: np.ndarray, samples: int, steps: int, seed: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Draw K samples of the missing part and record x0 snapshots of sample 0.

        Args:
            incomplete (np.ndarray): uint8 (C, H, W) model input.
            samples (int): K.
            steps (int): DDIM steps.
            seed (int): Noise seed.

        Returns:
            tuple[np.ndarray, dict[int, np.ndarray]]: uint8 (K, C, H, W) {0, 255}
            samples and {timestep: (C, H, W) scores in [0, 1]} snapshots.
        """
        batch: torch.Tensor = torch.from_numpy(np.repeat(incomplete[None], samples, axis=0)).to(self.device).float()
        wanted: list[int] = scheduled_snapshots(list(self.cfg.inference.SNAPSHOT_TIMESTEPS), steps, self.hyper.diffusion.TIMESTEPS)
        snapshots: dict[int, np.ndarray] = {}

        def record(t: int, x0: torch.Tensor) -> None:
            if t in wanted:
                snapshots[t] = ((x0[0].float() + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        scores: torch.Tensor = sample_missing(self.model, self.diffusion, batch, steps, self.hyper.diffusion.CLIP_X0,
                                              generator, self.dtype, on_step=record)
        binary: np.ndarray = (scores.cpu().numpy() > self.threshold).astype(np.uint8) * SCORE_SCALE
        return binary, snapshots

    def predict(self, plan: Plan, request: SimpleNamespace) -> dict[str, Any]:
        """Run one request: optionally remove a component, sample, fuse, compose, score.

        Args:
            plan (Plan): Parsed plan.
            request (SimpleNamespace): mode ("reconstruct" | "remove" | "erase"),
                layer, instance_id, removal ("easy" | "hard"), margin_px (remove);
                mask, keep_inside (erase); samples, steps, pipeline, seed.

        Returns:
            dict[str, Any]: Panels (PNG data URIs), added components, metrics, timing.

        Raises:
            UploadError: On an invalid request.
        """
        start: float = time.time()
        cfg = self.cfg.inference
        if not 1 <= request.samples <= cfg.MAX_SAMPLES:
            raise UploadError(f"samples must be in [1, {cfg.MAX_SAMPLES}]")
        if not cfg.MIN_STEPS <= request.steps <= cfg.MAX_STEPS:
            raise UploadError(f"steps must be in [{cfg.MIN_STEPS}, {cfg.MAX_STEPS}]")
        if request.pipeline not in self.pipelines:
            raise UploadError(f"unknown pipeline '{request.pipeline}'")
        upscale: int = self.cfg.render.UPSCALE
        case = None
        if request.mode == "remove":
            case = self._build_case(plan, request)
            incomplete: np.ndarray = case.incomplete
        elif request.mode == "erase":
            case = self._erase_case(plan, request.mask)
            incomplete = case.incomplete
        elif request.mode == "reconstruct":
            incomplete = self.semantic(plan)
        else:
            raise UploadError("mode must be 'reconstruct', 'remove' or 'erase'")

        samples, snapshots = self._sample(incomplete, request.samples, request.steps, request.seed)
        fused: np.ndarray = postprocess_case(samples, incomplete, self.schema, self.post_params,
                                             self.pipelines[request.pipeline], self.threshold)
        if request.mode == "erase" and request.keep_inside:
            # An eraser edit should only change what was erased; a thin rim lets walls join up.
            zone: np.ndarray = ndimage.binary_dilation(case.mask > 0, iterations=self.cfg.erase.KEEP_RIM_PX)
            fused[:, :, ~zone] = 0
        prediction: np.ndarray = fused[0].astype(np.float32) / SCORE_SCALE
        base: LabelMaps = semantic_to_labels(incomplete, self.schema, self.threshold)
        composed: LabelMaps = compose(base, semantic_to_labels(prediction, self.schema, self.threshold))
        extra: LabelMaps = added(base, composed)
        scale: float = plan.record["metres_per_px"]
        components = [{"cls": c.cls, "type": c.comp_type, "area_px": c.area,
                       "area_m2": round(c.area * scale * scale, 2),
                       "centroid": [round(c.centroid[0], 1), round(c.centroid[1], 1)]}
                      for c in extract_components(extra, self.schema, self.eval_params.components.MIN_AREA_PX)]
        components.sort(key=lambda c: -c["area_px"])

        input_image: np.ndarray = render_semantic(incomplete, self.schema)
        panels: dict[str, str] = {
            "input": to_png(tint(input_image, case.mask, self.schema) if case is not None else input_image, upscale),
            "added": to_png(overlay(incomplete, extra, self.schema), upscale),
            "composed": to_png(render_semantic(labels_to_semantic(composed, self.schema), self.schema), upscale),
        }
        strip: list[dict[str, Any]] = []
        for t in sorted(snapshots, reverse=True):
            estimate: LabelMaps = added(base, compose(base, semantic_to_labels(snapshots[t], self.schema, self.threshold)))
            strip.append({"t": t, "image": to_png(overlay(incomplete, estimate, self.schema), 1)})
        result: dict[str, Any] = {"mode": request.mode, "panels": panels, "denoising": strip, "components": components,
                                  "samples": request.samples, "steps": request.steps, "pipeline": request.pipeline,
                                  "composed_token": self.remember_composed(composed)}
        if case is not None:
            truth: LabelMaps = added(base, semantic_to_labels(case.full, self.schema, self.threshold))
            panels["truth"] = to_png(overlay(incomplete, truth, self.schema), upscale)
            panels["full"] = to_png(render_semantic(case.full, self.schema), upscale)
            context = prepare_case(case, self.schema, self.eval_params)
            metrics: dict[str, Any] = score_samples(context, prediction[None], self.schema, self.eval_params)
            result["case"] = {k: case.info.get(k) for k in ("comp_type", "primary_cls", "mode", "margin_px", "removed",
                                                             "target_area_px", "size_bin", "location")}
            result["metrics"] = {k: _plain(v) for k, v in metrics.items()}
        result["seconds"] = round(time.time() - start, 2)
        return result

    def _erase_case(self, plan: Plan, mask_b64: str | None) -> Case:
        """Clear everything under an eraser mask; the plan as loaded is the ground truth.

        Args:
            plan (Plan): Plan.
            mask_b64 (str | None): Base64 of H*W bytes, non-zero = erased.

        Returns:
            Case: Case without a primary component (component and pixel metrics still apply).

        Raises:
            UploadError: If the mask is malformed, empty, too large or erases nothing.
        """
        try:
            raw: bytes = base64.b64decode(mask_b64 or "", validate=True)
        except (binascii.Error, ValueError):
            raise UploadError("the eraser mask could not be read") from None
        if len(raw) != self.size * self.size:
            raise UploadError(f"the eraser mask must cover {self.size} x {self.size} pixels")
        mask: np.ndarray = np.frombuffer(raw, np.uint8).reshape(self.size, self.size) > 0
        if not mask.any():
            raise UploadError("erase part of the plan first")
        if mask.mean() > self.cfg.erase.MAX_FRACTION:
            raise UploadError(f"erase at most {int(self.cfg.erase.MAX_FRACTION * 100)}% of the canvas; the model needs context")
        full: np.ndarray = self.semantic(plan)
        incomplete: np.ndarray = full.copy()
        incomplete[:, mask] = 0
        # Context the model was never trained to redraw (furniture) is not erased.
        protected: list[int] = [self.schema.channel_index(c) for c in self.schema.channels
                                if self.schema.component_type[c] in self.protected_types]
        if protected:
            incomplete[protected] = full[protected]
        target: np.ndarray = full & (1 - incomplete)
        if not target.any():
            raise UploadError("the erased area holds nothing of the plan; erase over walls, openings or rooms")
        info: dict[str, Any] = {"case_id": "erase", "comp_type": None, "primary_cls": None, "mode": "erase", "margin_px": 0,
                                "metres_per_px": plan.record["metres_per_px"], "removed": [],
                                "target_area_px": int(target.any(axis=0).sum())}
        return Case(full, incomplete, target, mask.astype(np.uint8), np.zeros(mask.shape, np.uint8), info)

    def _build_case(self, plan: Plan, request: SimpleNamespace):
        """Remove the chosen instance with the frozen generator (easy / hard, margin).

        Args:
            plan (Plan): Plan.
            request (SimpleNamespace): Request (layer, instance_id, removal, margin_px).

        Returns:
            Case: Generated case.

        Raises:
            UploadError: If the instance or the removal settings are invalid.
        """
        match = [i for i in plan.record["instances"] if i["layer"] == request.layer and i["id"] == request.instance_id]
        if not match:
            raise UploadError("choose a component of the plan to remove")
        comp_type: str = self.schema.component_type[match[0]["cls"]]
        if self.removable_types is not None and comp_type not in self.removable_types:
            raise UploadError(f"this model restores {', '.join(self.removable_types)} only; "
                              f"a {match[0]['cls'].replace('_', ' ')} stays in the plan as context")
        if request.removal not in (EASY, HARD):
            raise UploadError("removal must be 'easy' or 'hard'")
        if request.removal == HARD and comp_type not in self.params.missing.HARD_MODE_TYPES:
            raise UploadError(f"hard removal is defined for {', '.join(self.params.missing.HARD_MODE_TYPES)} only")
        if request.margin_px not in self.params.missing.MARGINS_PX:
            raise UploadError(f"margin must be one of {list(self.params.missing.MARGINS_PX)} px")
        spec = CaseSpec("upload", 0, int(plan.record["plan_id"]), comp_type, request.layer, request.instance_id,
                        request.removal, request.margin_px)
        return self.generator.build_case({k: v.copy() for k, v in plan.layers.items()}, plan.record, spec)


def _plain(value: Any) -> Any:
    """JSON-safe scalar (NaN -> None, numpy -> Python).

    Args:
        value (Any): Metric value.

    Returns:
        Any: Plain value.
    """
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else round(float(value), 4)
    if isinstance(value, frozenset):
        return sorted(value)
    return value

