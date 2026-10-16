"""Flask front end for the missing-component generator.

    python frontend/server.py [--mode debug|prod] [--host 127.0.0.1] [--port 8087]

The first upload builds the context (three frozen mask VAEs, the diffusion
denoiser, the detector); later ones reuse it. Every request's artifacts live in
their own folder under the configured upload/output directories, so two uploads
in a row never mix up their panels.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
from typing import Any

# MPS fallback for ops Ultralytics needs (torchvision::nms has no MPS kernel).
# Read when the operator dispatches, so it must be set before torch is imported:
# hence above every project import, matching the evaluator's entry point.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# The repo root, so ``frontend.*`` and the other stages' packages import when this
# file is launched as a script (``python frontend/server.py``) from anywhere.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, abort, redirect, render_template, request, send_from_directory  # noqa: E402

from frontend.pipeline import stage_upload  # noqa: E402
from frontend.settings import get_context  # noqa: E402
from helpers.hyperparams import load_hyperparams  # noqa: E402

# Upload field name -> staged filename, per mask role.
UPLOAD_MASKS: dict[str, str] = {
    "floorplan": "floorplan.png",
    "furniture": "furniture.png",
}


def create_app(mode: str = "debug") -> Flask:
    """Create the Flask application.

    Args:
        mode (str): Runtime configuration mode (``debug`` or ``prod``), selecting
            the matching table of ``frontend/config.toml``.

    Returns:
        Flask: The configured application with the upload, result and image
            routes.
    """
    script_dir: Path = Path(__file__).resolve().parent
    hyperparams: SimpleNamespace = load_hyperparams(
        str(script_dir / "hyperparameters.toml")
    )
    context: Any = get_context(mode)
    # Anchored at the repository root by frontend/settings, so both senders get an
    # absolute directory: Flask prefixes RELATIVE send_from_directory paths with
    # the app's root path, which is this package's folder.
    upload_root: Path = Path(str(context.config.UPLOAD_DIRECTORY))
    output_root: Path = Path(str(context.config.OUTPUT_DIRECTORY))

    app: Flask = Flask(__name__, template_folder=str(script_dir / "templates"))
    # Cap on one request body: both masks, so the configured megabytes.
    app.config["MAX_CONTENT_LENGTH"] = int(hyperparams.MAX_UPLOAD_MB) * 1024 * 1024

    @app.get("/")
    def index() -> str:
        """Serve the upload form.

        Returns:
            str: Rendered form page.
        """
        return render_template(
            "index.html",
            component=context.ontology.name,
            component_label=context.class_names[context.ontology.yolo_class_id],
            image_size=context.sampler.image_size,
            max_draws=int(hyperparams.MAX_DRAWS),
            device=str(context.config.DEVICE),
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
        staged_dir: Path = upload_root / request_id
        staged_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: dict[str, Path] = {}
        for role, filename in UPLOAD_MASKS.items():
            uploaded: Any = request.files.get(role)
            if uploaded is None or not uploaded.filename:
                if role == "floorplan":
                    return redirect(f"/?error={quote('A floor-plan mask is required.')}")
                continue
            destination: Path = staged_dir / filename
            uploaded.save(str(destination))
            saved_paths[role] = destination

        # The form's draw count, clamped by the pipeline against MAX_DRAWS.
        requested_draws: int | None = None
        if request.form.get("draws"):
            try:
                requested_draws = int(request.form["draws"])
            except ValueError:
                requested_draws = None

        try:
            payload: dict[str, Any] = stage_upload(
                context,
                request_id,
                saved_paths["floorplan"],
                saved_paths.get("furniture"),
                requested_draws,
            )
        except (ValueError, FileNotFoundError) as exc:
            # A mask the model cannot read is a user error, so it is shown on the
            # form instead of a stack trace. Anything else is a real 500.
            shutil.rmtree(staged_dir, ignore_errors=True)
            return redirect(f"/?error={quote(str(exc))}")

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
            generated_pixels=payload["generated_pixels"],
            elapsed_seconds=payload["elapsed_seconds"],
            seed=int(hyperparams.SEED),
            conf=float(hyperparams.YOLO_CONF),
            checkpoint=str(context.config.MASK_DIFFUSION_CHECKPOINT_PATH),
            weights=str(context.config.YOLO_WEIGHTS_PATH),
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

    return app


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
