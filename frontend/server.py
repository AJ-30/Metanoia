"""Flask front end for the missing-component generator.

    python frontend/server.py [--mode debug|prod] [--host 127.0.0.1] [--port 8087]

The first upload builds the context (three frozen mask VAEs, the diffusion
denoiser, the detector); later ones reuse it. Every request's artifacts live in
their own folder under the configured upload/output directories, so two uploads
in a row never mix up their panels.

``/inpaint`` is a separate editor and pipeline: it loads all three component
contexts, runs them one at a time on the edited furniture mask, and merges their
generated masks into one layout.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
from typing import Any

from PIL import Image

# MPS fallback for ops Ultralytics needs (torchvision::nms has no MPS kernel).
# Read when the operator dispatches, so it must be set before torch is imported:
# hence above every project import, matching the evaluator's entry point.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402 - must follow the MPS fallback switch above

# The repo root, so ``frontend.*`` and the other stages' packages import when this
# file is launched as a script (``python frontend/server.py``) from anywhere.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flask import (  # noqa: E402
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
)

from frontend.inpainting import (  # noqa: E402
    erasable_channels_from_palette,
    load_furniture_palette,
    validate_inpaint_edit,
)
from frontend.pipeline import (  # noqa: E402
    stage_inpaint,
    stage_upload,
    validate_and_load_mask,
)
from frontend.settings import (  # noqa: E402
    available_components,
    default_component,
    get_context,
    load_stage_config,
)
from helpers.hyperparams import load_hyperparams  # noqa: E402

# Upload field name -> staged filename, per mask role.
UPLOAD_MASKS: dict[str, str] = {
    "floorplan": "floorplan.png",
    "furniture": "furniture.png",
}
# Optional YOLO labels uploaded beside the masks, and the name they are staged
# under. The labels are text, not an image, so they are kept out of
# UPLOAD_MASKS and handled separately.
GROUND_TRUTH_FIELD: str = "ground_truth"
GROUND_TRUTH_LABELS_FILENAME: str = "ground_truth_labels.txt"
# The inpainting form sends the browser's edited class ids as a base64 string of
# one byte per pixel, not as a PNG: a canvas PNG is RGBA and the upload validator
# correctly refuses it as a class-id mask. The raw bytes keep the ids exact.
INPAINT_EDITED_FIELD: str = "edited_ids"
INPAINT_PREVIEW_DIR: str = "inpaint_preview"


def format_metric(value: float | int | None, digits: int = 3) -> str:
    """Format an optional metric for the result page.

    Args:
        value (float | int | None): Metric value, or ``None`` when undefined.
        digits (int): Decimal places to show.

    Returns:
        str: The formatted number, or ``"n/a"`` when the metric is undefined,
            which is different from a real zero.
    """
    if value is None:
        return "n/a"
    return f"{float(value):.{int(digits)}f}"


def create_app(mode: str = "debug") -> Flask:
    """Create the Flask application.

    Args:
        mode (str): Runtime configuration mode (``debug`` or ``prod``), selecting
            the matching table of ``frontend/config.toml``.

    Returns:
        Flask: The configured application with the upload, inpainting, result
            and image routes.
    """
    script_dir: Path = Path(__file__).resolve().parent
    hyperparams: SimpleNamespace = load_hyperparams(
        str(script_dir / "hyperparameters.toml")
    )
    # Only paths and the default selection are read here. Building a full
    # context would load a VAE stack and a detector before the first upload; the
    # selected component's context is built lazily and cached per component.
    stage_config: SimpleNamespace = load_stage_config(mode)
    components: list[str] = available_components()
    default_selection: str = default_component(stage_config)
    if default_selection not in components:
        raise KeyError(
            f"Default component '{default_selection}' is not declared in "
            f"components.toml; choose one of {components} or fix "
            "MISSING_COMPONENT_CLASS_NAME in frontend/config.toml."
        )
    # Anchored at the repository root by frontend/settings, so both senders get an
    # absolute directory: Flask prefixes RELATIVE send_from_directory paths with
    # the app's root path, which is this package's folder.
    upload_root: Path = Path(str(stage_config.UPLOAD_DIRECTORY))
    output_root: Path = Path(str(stage_config.OUTPUT_DIRECTORY))
    # The inpainting canvas needs the furniture palette and the class ids it may
    # erase. Both come from the same colour index the encoders use, so the page
    # cannot invent a colour or an erasable class that the model does not know.
    palette_path: Path = (
        Path(str(stage_config.MASK_COLOR_INDEX_DIR))
        / str(stage_config.FURNITURE_COLOR_INDEX_FILE)
    )
    palette: list[dict[str, Any]] = load_furniture_palette(palette_path)
    if not palette:
        raise ValueError(f"Furniture colour index is empty: {palette_path}")
    palette_num_classes: int = max(
        int(entry["id"]) for entry in palette
    ) + 1
    erasable_channels: set[int] = erasable_channels_from_palette(
        palette, components
    )

    app: Flask = Flask(__name__, template_folder=str(script_dir / "templates"))
    app.add_template_filter(format_metric, "metric")
    # Cap on one request body: both masks, so the configured megabytes.
    app.config["MAX_CONTENT_LENGTH"] = int(hyperparams.MAX_UPLOAD_MB) * 1024 * 1024
    max_label_bytes: int = (
        int(hyperparams.MAX_GROUND_TRUTH_LABEL_MB) * 1024 * 1024
    )

    @app.get("/")
    def index() -> str:
        """Serve the upload form.

        Returns:
            str: Rendered form page.
        """
        # Keep an explicitly selected component across a failed upload, but never
        # render a key the registry does not declare.
        requested: str = str(request.args.get("component", ""))
        selected: str = (
            requested if requested in components else default_selection
        )
        return render_template(
            "index.html",
            component=selected,
            components=components,
            selected_component=selected,
            max_draws=int(hyperparams.MAX_DRAWS),
            default_draws=int(hyperparams.DEFAULT_DRAWS),
            ensemble_threshold=float(
                hyperparams.ENSEMBLE_PROBABILITY_THRESHOLD
            ),
            device=str(stage_config.DEVICE),
            error=request.args.get("error"),
        )

    @app.post("/upload")
    def upload() -> Any:
        """Run one upload end to end and render the result page.

        Returns:
            Any: Rendered result page on success, or a redirect to the form
                carrying the failure message.
        """
        started: float = time.time()
        request_id: str = (
            f"upload_{time.strftime('%Y%m%d_%H%M%S')}_{int(started * 1000) % 100000}"
        )
        # The form's component selection decides both models. An empty value
        # (an older page) falls back to the configured default.
        selected_component: str = (
            str(request.form.get("component", "")) or default_selection
        )
        if selected_component not in components:
            return redirect(
                f"/?error={quote(f'Unknown missing component {selected_component!r}.')}"
                f"&component={quote(default_selection)}"
            )
        staged_dir: Path = upload_root / request_id
        staged_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: dict[str, Path] = {}
        for role, filename in UPLOAD_MASKS.items():
            uploaded: Any = request.files.get(role)
            if uploaded is None or not uploaded.filename:
                if role == "floorplan":
                    return redirect(
                        f"/?error={quote('A floor-plan mask is required.')}"
                        f"&component={quote(selected_component)}"
                    )
                continue
            destination: Path = staged_dir / filename
            uploaded.save(str(destination))
            saved_paths[role] = destination

        # Optional ground-truth labels: a text file, validated by the pipeline
        # before the diffusion chain starts. Absent means "no comparison".
        ground_truth_path: Path | None = None
        uploaded_labels: Any = request.files.get(GROUND_TRUTH_FIELD)
        if uploaded_labels is not None and uploaded_labels.filename:
            destination = staged_dir / GROUND_TRUTH_LABELS_FILENAME
            uploaded_labels.save(str(destination))
            if destination.stat().st_size > max_label_bytes:
                shutil.rmtree(staged_dir, ignore_errors=True)
                limit_mb: int = int(hyperparams.MAX_GROUND_TRUTH_LABEL_MB)
                message: str = (
                    "The ground-truth labels file is larger than "
                    f"{limit_mb} MB."
                )
                return redirect(
                    f"/?error={quote(message)}"
                    f"&component={quote(selected_component)}"
                )
            ground_truth_path = destination

        # The form's draw count, clamped by the pipeline against MAX_DRAWS.
        requested_draws: int | None = None
        if request.form.get("draws"):
            try:
                requested_draws = int(request.form["draws"])
            except ValueError:
                requested_draws = None

        try:
            # Lazy per component: first use pays the model load, later uses reuse
            # the cached context. get_context validates the registry key too.
            context: Any = get_context(mode, selected_component)
            payload: dict[str, Any] = stage_upload(
                context,
                request_id,
                floorplan_path=saved_paths["floorplan"],
                furniture_path=saved_paths.get("furniture"),
                draws=requested_draws,
                ground_truth_path=ground_truth_path,
            )
        except (ValueError, FileNotFoundError) as exc:
            # A mask or label file the model cannot read, or a missing selected
            # model, is a user error shown on the form instead of a stack trace.
            # Anything else is a real 500.
            shutil.rmtree(staged_dir, ignore_errors=True)
            return redirect(
                f"/?error={quote(str(exc))}"
                f"&component={quote(selected_component)}"
            )

        _prune_old_runs(output_root, int(hyperparams.MAX_CACHED_RUNS))
        _prune_old_runs(upload_root, int(hyperparams.MAX_CACHED_RUNS))
        return render_template(
            "result.html",
            request_id=request_id,
            panels=payload["panels"],
            detections=payload["detections"],
            component=payload["component"],
            component_label=payload["component_label"],
            draws=payload["draws"],
            ensemble_threshold=payload["ensemble_probability_threshold"],
            generated_pixels=payload["generated_pixels"],
            elapsed_seconds=payload["elapsed_seconds"],
            ground_truth_uploaded=payload["ground_truth_uploaded"],
            ground_truth_boxes=payload["ground_truth_boxes"],
            seed=int(hyperparams.SEED),
            conf=float(hyperparams.YOLO_CONF),
            match_iou=float(hyperparams.MATCH_IOU_THRESHOLD),
            image_size=int(context.sampler.image_size),
            checkpoint=str(context.config.MASK_DIFFUSION_CHECKPOINT_PATH),
            weights=str(context.config.YOLO_WEIGHTS_PATH),
        )

    @app.get("/inpaint")
    def inpaint_page() -> str:
        """Serve the inpainting editor.

        Returns:
            str: Rendered page with the canvas, brush controls and palette.
        """
        return render_template(
            "inpaint.html",
            components=components,
            canvas_size=int(hyperparams.INPAINT_CANVAS_SIZE),
            palette=palette,
            erasable_channels=sorted(erasable_channels),
            brush_min=int(hyperparams.INPAINT_BRUSH_MIN),
            brush_max=int(hyperparams.INPAINT_BRUSH_MAX),
            brush_default=int(hyperparams.INPAINT_BRUSH_DEFAULT),
            erase_overlay_alpha=float(
                hyperparams.INPAINT_ERASE_OVERLAY_ALPHA
            ),
            undo_limit=int(hyperparams.INPAINT_UNDO_LIMIT),
            max_draws=int(hyperparams.MAX_DRAWS),
            default_draws=int(hyperparams.DEFAULT_DRAWS),
            ensemble_threshold=float(
                hyperparams.ENSEMBLE_PROBABILITY_THRESHOLD
            ),
            device=str(stage_config.DEVICE),
            error=request.args.get("error"),
        )

    @app.post("/inpaint/preview")
    def inpaint_preview() -> Any:
        """Normalize an uploaded furniture mask for the browser canvas.

        The canvas must edit class ids, not the RGB values a browser decodes
        from a palette PNG. This route validates the upload with the same rules
        as the generation path, resizes it to the model's square domain and
        returns a mode-L PNG whose red channel is the class id.

        Returns:
            Any: The normalized PNG, or a 400 with a user-facing message.
        """
        uploaded: Any = request.files.get("furniture")
        if uploaded is None or not uploaded.filename:
            return "A furniture mask is required for inpainting.", 400

        preview_dir: Path = upload_root / INPAINT_PREVIEW_DIR
        preview_dir.mkdir(parents=True, exist_ok=True)
        temporary: Path = preview_dir / f"preview_{time.time_ns()}.png"
        uploaded.save(str(temporary))
        try:
            ids: torch.Tensor = validate_and_load_mask(
                temporary,
                palette_num_classes,
                int(hyperparams.INPAINT_CANVAS_SIZE),
                "furniture",
            )
        except (ValueError, FileNotFoundError) as exc:
            temporary.unlink(missing_ok=True)
            return str(exc), 400
        temporary.unlink(missing_ok=True)

        buffer: io.BytesIO = io.BytesIO()
        Image.fromarray(ids.to(torch.uint8).numpy(), mode="L").save(
            buffer, format="PNG"
        )
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")

    @app.post("/inpaint")
    def inpaint_run() -> Any:
        """Run the three component models on one edited mask.

        Returns:
            Any: Rendered merged result page on success, or a redirect to the
                editor carrying the failure message.
        """
        started: float = time.time()
        request_id: str = (
            f"inpaint_{time.strftime('%Y%m%d_%H%M%S')}_"
            f"{int(started * 1000) % 100000}"
        )
        staged_dir: Path = upload_root / request_id
        staged_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, Path] = {}
        for role, filename in UPLOAD_MASKS.items():
            uploaded: Any = request.files.get(role)
            if uploaded is None or not uploaded.filename:
                shutil.rmtree(staged_dir, ignore_errors=True)
                message: str = (
                    "A floor-plan mask and a furniture mask are both required "
                    "for inpainting."
                )
                return redirect(f"/inpaint?error={quote(message)}")
            destination: Path = staged_dir / filename
            uploaded.save(str(destination))
            saved_paths[role] = destination

        try:
            edited_ids: torch.Tensor = _decode_edited_ids(
                str(request.form.get(INPAINT_EDITED_FIELD, "")),
                int(hyperparams.INPAINT_CANVAS_SIZE),
            )
        except ValueError as exc:
            shutil.rmtree(staged_dir, ignore_errors=True)
            return redirect(f"/inpaint?error={quote(str(exc))}")

        requested_draws: int | None = None
        if request.form.get("draws"):
            try:
                requested_draws = int(request.form["draws"])
            except ValueError:
                requested_draws = None

        try:
            # Load the first context and validate the uploads before paying to
            # load the other two models. A bad mask should fail after one model
            # load, not after three.
            first_context: Any = get_context(mode, components[0])
            first_sampler: Any = first_context.sampler
            validate_and_load_mask(
                saved_paths["floorplan"],
                first_sampler.floorplan_layout.num_channels,
                first_sampler.image_size,
                "floorplan",
            )
            original_ids: torch.Tensor = validate_and_load_mask(
                saved_paths["furniture"],
                first_sampler.target_layout.num_channels,
                first_sampler.image_size,
                "furniture",
            )
            expected_shape: tuple[int, int] = (
                first_sampler.image_size,
                first_sampler.image_size,
            )
            if tuple(edited_ids.shape) != expected_shape:
                raise ValueError(
                    "The edited furniture mask has shape "
                    f"{tuple(edited_ids.shape)}, but this model expects "
                    f"{expected_shape}."
                )
            validate_inpaint_edit(
                original_ids, edited_ids, erasable_channels
            )

            # Every component is loaded and run one at a time. Contexts are
            # cached per component, so a second request reuses whatever is
            # already in memory.
            contexts: dict[str, Any] = {components[0]: first_context}
            for component_name in components[1:]:
                contexts[component_name] = get_context(mode, component_name)
            payload: dict[str, Any] = stage_inpaint(
                contexts,
                request_id,
                floorplan_path=saved_paths["floorplan"],
                original_furniture_path=saved_paths["furniture"],
                edited_ids=edited_ids,
                draws=requested_draws,
            )
        except (ValueError, FileNotFoundError) as exc:
            shutil.rmtree(staged_dir, ignore_errors=True)
            shutil.rmtree(output_root / request_id, ignore_errors=True)
            return redirect(f"/inpaint?error={quote(str(exc))}")

        _prune_old_runs(output_root, int(hyperparams.MAX_CACHED_RUNS))
        _prune_old_runs(upload_root, int(hyperparams.MAX_CACHED_RUNS))
        return render_template(
            "inpaint_result.html",
            request_id=request_id,
            panels=payload["panels"],
            components=payload["components"],
            merged_found=payload["merged_found"],
            draws=payload["draws"],
            ensemble_threshold=payload["ensemble_probability_threshold"],
            elapsed_seconds=payload["elapsed_seconds"],
            image_size=payload["image_size"],
            protect_original=payload["protect_original"],
            seed=int(hyperparams.SEED),
            conf=float(hyperparams.YOLO_CONF),
        )

    @app.get("/outputs/<request_id>/<path:filename>")
    def outputs(request_id: str, filename: str) -> Any:
        """Serve one request's panel image.

        Args:
            request_id (str): Request folder under the output directory.
            filename (str): Panel filename inside it.

        Returns:
            Any: The image, or 404 when it does not exist.
        """
        directory: Path = output_root / Path(request_id).name
        if not directory.is_dir():
            abort(404)
        return send_from_directory(str(directory), filename)

    @app.get("/uploads/<request_id>/<path:filename>")
    def uploads(request_id: str, filename: str) -> Any:
        """Serve one request's staged mask, so the page can show the input.

        Args:
            request_id (str): Request folder under the upload directory.
            filename (str): Mask filename inside it.

        Returns:
            Any: The image, or 404 when it does not exist.
        """
        directory: Path = upload_root / Path(request_id).name
        if not directory.is_dir():
            abort(404)
        return send_from_directory(str(directory), filename)

    # Step 0 of the combined pipeline: complete a plan with a missing wall, door or window, then hand it to the pages above.
    from frontend.completion import register_completion  # noqa: E402

    register_completion(app, upload_root, mode)

    return app


def _decode_edited_ids(encoded: str, image_size: int) -> torch.Tensor:
    """Decode the editor's raw class-id bytes into a square int64 tensor.

    Args:
        encoded (str): Base64 string sent by the browser. It contains exactly
            ``image_size * image_size`` bytes, one class id per pixel.
        image_size (int): Square side length of the canvas.

    Returns:
        torch.Tensor: int64 ``(image_size, image_size)`` class-id tensor.

    Raises:
        ValueError: If the string is empty, not valid base64, or has the wrong
            number of bytes.
    """
    if not encoded:
        raise ValueError(
            "The edited furniture mask is missing; draw on the canvas before "
            "submitting."
        )
    try:
        raw: bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            "The edited furniture mask is not valid base64 data."
        ) from exc
    expected: int = int(image_size) * int(image_size)
    if len(raw) != expected:
        raise ValueError(
            "The edited furniture mask has "
            f"{len(raw)} byte(s), but the {int(image_size)}x{int(image_size)} "
            f"canvas needs {expected}."
        )
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
        int(image_size), int(image_size)
    ).to(torch.int64)


def _prune_old_runs(directory: Path, keep: int) -> None:
    """Delete the oldest request folders beyond the configured cache size.

    Args:
        directory (Path): Upload or output directory holding request folders.
        keep (int): How many newest folders to keep; negative keeps every folder.
    """
    if keep < 0 or not directory.is_dir():
        return
    folders: list[Path] = sorted(
        (path for path in directory.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    for folder in folders[: max(0, len(folders) - keep)]:
        shutil.rmtree(folder, ignore_errors=True)


def main() -> int:
    """Run the front end from the command line.

    Returns:
        int: Process exit code, ``0`` on clean shutdown.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Upload a floor-plan mask and see the generated missing component "
            "with the detector's bounding boxes."
        )
    )
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8087)
    args: argparse.Namespace = parser.parse_args()
    app: Flask = create_app(args.mode)
    print(f"Missing-component UI on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
