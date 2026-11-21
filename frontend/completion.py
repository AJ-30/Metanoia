"""Step 0 of the combined pipeline: complete a plan with a missing wall, door or window, then hand it to the furniture pages.

The wall/door/window model (``Reconstruction``, a FastAPI app) runs in this process: its API is mounted under ``/api`` and
its page is served at ``/complete`` with one addition, a "Next: furniture" step (``complete/handoff_step.js``). The
completed plan is converted (``handoff.py``) to the two class-id masks the furniture pages read and stored under
``<uploads>/handoff_<id>/``; the upload page (``/?plan=<id>``) and the inpainting editor (``/inpaint?plan=<id>``) fetch
them from ``/handoff/<id>/<name>`` and fill their file inputs. The furniture flow itself is unchanged.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from PIL import Image

from frontend.handoff import load_settings, to_furniture_masks

LOG = logging.getLogger("frontend.completion")
HERE: Path = Path(__file__).resolve().parent
COMPLETE_DIR: Path = HERE / "complete"
HANDOFF_PREFIX: str = "handoff_"
HANDOFF_FILES: tuple[str, str] = ("floorplan_segmentation.png", "furniture_segmentation.png")
ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")


def load_plan_model(mode: str = "prod") -> Any:
    """Import the wall/door/window demo app and set it up the way its ``main()`` does (everything but starting uvicorn).

    Args:
        mode (str): ``debug`` or ``prod`` table of ``Reconstruction/serve/config.toml``.

    Returns:
        Any: The ``Reconstruction.serve.app`` module, with its profiles discovered and the default model loaded.
    """
    from Reconstruction.common.config_loader import load_run_config, resolve_path
    from Reconstruction.common.logging_setup import make_run_id, setup_logger
    from Reconstruction.serve import app as plan_app

    config = load_run_config(plan_app.CONFIG_FILE, mode)
    plan_app.STATE.log = setup_logger("reconstruction", resolve_path(config.LOG_FILE_PATH), make_run_id("reconstruction"), config.IS_DEBUG)
    plan_app.STATE.config = config
    plan_app.STATE.profiles = plan_app.discover_profiles(config, plan_app.STATE.log)
    plan_app.engine_for(plan_app.DEFAULT_PROFILE)       # a bad checkpoint fails at start-up, not at the first plan
    return plan_app


class ApiMount:
    """WSGI middleware: ``/api/*`` goes to the plan model's ASGI app, everything else to Flask."""

    def __init__(self, flask_wsgi: Any, asgi_app: Any) -> None:
        """Wrap both apps.

        Args:
            flask_wsgi (Any): The Flask app's WSGI callable.
            asgi_app (Any): The FastAPI app.
        """
        from a2wsgi import ASGIMiddleware

        self.flask_wsgi, self.api = flask_wsgi, ASGIMiddleware(asgi_app)

    def __call__(self, environ: dict, start_response: Any) -> Any:
        """Route one WSGI request by its path."""
        if environ.get("PATH_INFO", "").startswith("/api/"):
            return self.api(environ, start_response)
        return self.flask_wsgi(environ, start_response)


def complete_page_html() -> str:
    """The plan model's page with the hand-off step added before ``</body>``.

    Returns:
        str: HTML of ``Reconstruction/serve/static/index.html`` plus a link back to the furniture pages.
    """
    from Reconstruction.serve.engine import SERVE_DIR

    html: str = (SERVE_DIR / "static" / "index.html").read_text()
    extra: str = '<script src="/complete/handoff_step.js"></script>\n'
    return html.replace("<body>", '<body>\n<p style="margin:8px 16px"><a href="/">&larr; Furniture demo</a></p>', 1).replace("</body>", extra + "</body>", 1)


def register_completion(app: Flask, upload_root: Path, mode: str = "prod") -> None:
    """Add the completion page, the ``/api`` mount and the hand-off routes to the Flask app.

    Args:
        app (Flask): The furniture demo.
        upload_root (Path): Folder of the demo's uploads; hand-off masks go to ``handoff_<id>/`` inside it.
        mode (str): ``debug`` or ``prod``.
    """
    try:
        plan_app = load_plan_model(mode)
    except (FileNotFoundError, ModuleNotFoundError) as error:
        # The furniture pages do not depend on step 0: serve them, and say on /complete why the step is unavailable.
        LOG.error("Plan completion (step 0) is unavailable, the furniture pages still work: %s", error)
        reason: str = f"Plan completion is unavailable on this server: {error}"

        @app.get("/complete")
        def complete_unavailable() -> Response:
            """Explain why step 0 is not running."""
            return Response(reason, status=503, mimetype="text/plain")

        return
    settings = load_settings()
    app.wsgi_app = ApiMount(app.wsgi_app, plan_app.app)
    page: str = complete_page_html()

    @app.get("/complete")
    def complete_page() -> Response:
        """The plan model's page: load or draw a plan with something missing and complete it."""
        return Response(page, mimetype="text/html")

    @app.get("/complete/handoff_step.js")
    def handoff_script() -> Any:
        """The script that adds the hand-off step to that page."""
        return send_from_directory(str(COMPLETE_DIR), "handoff_step.js", mimetype="text/javascript")

    @app.post("/handoff")
    def handoff() -> Any:
        """Convert a completed plan to the furniture pages' two masks and store them under a short id."""
        body = request.get_json(silent=True) or {}
        profile, token = str(body.get("profile", "")), str(body.get("composed_token", ""))
        try:
            engine = plan_app.engine_for(profile)
            labels = engine.recall_composed(token)
        except Exception as error:  # an unknown profile or an expired token is the caller's to fix
            return jsonify({"detail": str(error)}), 400
        floorplan, furniture = to_furniture_masks(labels, engine.schema, settings)
        if furniture is None:                                   # the shell-only model carries no furniture
            furniture = np.zeros_like(floorplan)
        plan_id = f"{int(time.time() * 1000) & 0xFFFFFFFFFFFF:012x}"
        folder = upload_root / f"{HANDOFF_PREFIX}{plan_id}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, ids in zip(HANDOFF_FILES, (floorplan, furniture)):
            Image.fromarray(np.asarray(ids, np.uint8), mode="L").save(folder / name)
        return jsonify({"id": plan_id})

    @app.get("/handoff/<plan_id>/<name>")
    def handoff_file(plan_id: str, name: str) -> Any:
        """One of the two stored masks, fetched by the upload and inpainting pages to fill their file inputs."""
        if not ID_PATTERN.match(plan_id) or name not in HANDOFF_FILES:
            abort(404)
        folder = upload_root / f"{HANDOFF_PREFIX}{plan_id}"
        if not folder.is_dir():
            abort(404)
        return send_from_directory(str(folder), name, mimetype="image/png")
