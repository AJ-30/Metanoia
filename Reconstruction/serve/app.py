"""Web demo: upload a floor plan, get the missing component reconstructed.

Serves ``static/index.html`` and a small JSON API on HOST:PORT (localhost).

    GET  /api/profiles        the selectable models (ResPlan, plus those of profiles.toml whose files exist)
    GET  /api/info            taxonomy, colours, limits, model
    POST /api/plan            multipart ``file`` (+ optional ``metres_per_px``) -> parsed plan
    GET  /api/example?index=  a frozen test plan -> parsed plan
    POST /api/predict         JSON request (complete / remove a component / erase) -> panels, components, metrics

``info``, ``plan``, ``example`` and ``predict`` take an optional ``profile`` (query, form field or JSON
field); it defaults to ``resplan``, so clients written before profiles existed keep working. A profile's
model loads on first use.

The GPU runs one request at a time; at most ``[inference] MAX_WAITING``
requests queue behind it, later ones get HTTP 503.

Usage (from the repository root):
    python -m Reconstruction.serve.app --mode prod
"""

import argparse
import asyncio
import copy
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from Reconstruction.common.config_loader import load_run_config, read_toml, resolve_path
from Reconstruction.common.logging_setup import make_run_id, setup_logger
from Reconstruction.serve.engine import SERVE_DIR, Engine, UploadError

CONFIG_FILE: Path = SERVE_DIR / "config.toml"
PROFILES_FILE: Path = SERVE_DIR / "profiles.toml"
INDEX_FILE: Path = SERVE_DIR / "static" / "index.html"
DEFAULT_PROFILE: str = "resplan"


class PredictRequest(BaseModel):
    """Body of ``POST /api/predict``."""

    profile: str = Field(default=DEFAULT_PROFILE, max_length=40)
    token: str = Field(max_length=64)
    mode: Literal["reconstruct", "remove", "erase"] = "reconstruct"
    layer: Literal["structure", "room"] | None = None
    instance_id: int | None = Field(default=None, ge=1, le=255)
    removal: Literal["easy", "hard"] = "easy"
    margin_px: int = Field(default=0, ge=0, le=64)
    samples: int = Field(default=1, ge=1, le=16)
    steps: int = Field(default=50, ge=1, le=1000)
    pipeline: str = Field(default="none", max_length=32)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    # Eraser: base64 of canvas-sized bytes (non-zero = erased) and whether to keep edits inside it.
    mask: str | None = Field(default=None, max_length=100_000)
    keep_inside: bool = True


app = FastAPI(title="Floor-plan reconstruction", docs_url=None, redoc_url=None, openapi_url=None)
STATE = SimpleNamespace(config=None, profiles={}, engines={}, log=None, gpu=asyncio.Lock(), waiting=0,
                        load_lock=threading.Lock())


def profile_config(config: SimpleNamespace, name: str, spec: dict) -> SimpleNamespace:
    """The engine configuration of one ``profiles.toml`` profile.

    Args:
        config (SimpleNamespace): ``config.toml`` of the chosen mode (ResPlan settings).
        name (str): Profile name.
        spec (dict): The profile's table.

    Returns:
        SimpleNamespace: Copy of ``config`` pointing at the profile's dataset, examples and checkpoint.
    """
    derived: SimpleNamespace = copy.copy(config)
    derived.DATASET = spec["dataset"]
    derived.EXAMPLE_SPLIT = spec["example_split"]
    derived.CHECKPOINT_FILE = spec["checkpoint"]
    derived.PROFILE = {**spec, "name": name, "dataset_label": spec.get("label", name).split(":")[0]}
    return derived


def discover_profiles(config: SimpleNamespace, log: logging.Logger) -> dict:
    """The profiles the demo can serve: ResPlan plus every complete profile of ``profiles.toml``.

    A profile with a missing file is skipped with a warning; it never affects ResPlan.

    Args:
        config (SimpleNamespace): ``config.toml`` of the chosen mode.
        log (logging.Logger): Logger.

    Returns:
        dict: name -> {"label", "description", "config"}; ``resplan`` first.
    """
    profiles: dict = {DEFAULT_PROFILE: {"label": "ResPlan apartments: walls, doors, windows, rooms and stairs",
                                        "description": None, "config": config}}
    if not PROFILES_FILE.is_file():
        return profiles
    for name, spec in read_toml(PROFILES_FILE).items():
        if name == DEFAULT_PROFILE or not isinstance(spec, dict):
            continue
        processed = resolve_path(config.PROCESSED_ROOT) / spec["dataset"]
        needed = [resolve_path(spec["checkpoint"]), processed / "params.toml", processed / "size_bins.json",
                  *(processed / spec["example_split"] / f for f in ("plans.jsonl", "structure.npy", "room.npy"))]
        missing = [str(p) for p in needed if not p.is_file()]
        if missing:
            log.warning(f"profile '{name}' skipped; missing {missing[0]}" + (f" (+{len(missing) - 1} more)" if len(missing) > 1 else ""))
            continue
        profiles[name] = {"label": spec["label"], "description": spec.get("description"),
                          "config": profile_config(config, name, spec)}
    return profiles


def engine_for(profile: str) -> Engine:
    """The engine of a profile, loading its model on first use.

    Args:
        profile (str): Profile name.

    Returns:
        Engine: The loaded engine.

    Raises:
        UploadError: For an unknown profile.
    """
    if profile not in STATE.profiles:
        raise UploadError(f"unknown model '{profile}'")
    with STATE.load_lock:
        if profile not in STATE.engines:
            STATE.engines[profile] = Engine(STATE.profiles[profile]["config"])
            engine = STATE.engines[profile]
            STATE.log.info(f"Loaded profile {profile}: {engine.checkpoint_name} (step {engine.step})")
        return STATE.engines[profile]


@app.exception_handler(UploadError)
async def upload_error(_, error: UploadError) -> JSONResponse:
    """Report invalid uploads / requests as HTTP 400 with a readable message."""
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.get("/")
async def index() -> FileResponse:
    """The single-page interface."""
    return FileResponse(INDEX_FILE, media_type="text/html")


@app.get("/healthz")
async def health() -> dict:
    """Liveness probe."""
    return {"ok": True, "waiting": STATE.waiting, "busy": STATE.gpu.locked()}


@app.get("/api/profiles")
async def profiles() -> dict:
    """The selectable models; the first one is the default."""
    return {"default": DEFAULT_PROFILE,
            "profiles": [{"name": n, "label": p["label"], "description": p["description"]} for n, p in STATE.profiles.items()]}


@app.get("/api/info")
async def info(profile: str = DEFAULT_PROFILE) -> dict:
    """Taxonomy, colours, limits and model facts of a profile."""
    engine = await run_in_threadpool(engine_for, profile)
    return engine.info()


@app.post("/api/plan")
async def upload_plan(file: UploadFile = File(...), metres_per_px: float | None = Form(default=None),
                      profile: str = Form(default=DEFAULT_PROFILE)) -> dict:
    """Parse an uploaded plan (JSON, SVG, image or array) and cache it."""
    engine = await run_in_threadpool(engine_for, profile)
    limit: int = engine.cfg.upload.MAX_BYTES
    raw: bytes = await file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(status_code=413, detail=f"files up to {limit // 1_000_000} MB are accepted")
    if not raw:
        raise UploadError("the file is empty")
    plan = await run_in_threadpool(engine.parse_upload, raw, file.filename or "", metres_per_px)
    STATE.log.info(f"parsed {plan.source} upload '{file.filename}' ({len(raw)} bytes, {len(plan.record['instances'])} instances)")
    return engine.describe(engine.remember(plan), plan)


@app.get("/api/example")
async def example(index: int = 0, profile: str = DEFAULT_PROFILE) -> dict:
    """A frozen test plan, parsed exactly as stored."""
    engine = await run_in_threadpool(engine_for, profile)
    plan = engine.example(index)
    return engine.describe(engine.remember(plan), plan)


@app.post("/api/predict")
async def predict(request: PredictRequest) -> dict:
    """Reconstruct the missing part of a cached plan on the GPU."""
    engine = await run_in_threadpool(engine_for, request.profile)
    plan = engine.recall(request.token)
    if STATE.waiting >= engine.cfg.inference.MAX_WAITING:
        raise HTTPException(status_code=503, detail="the GPU is busy with other requests; try again in a minute")
    STATE.waiting += 1
    try:
        async with STATE.gpu:
            result: dict = await run_in_threadpool(engine.predict, plan, SimpleNamespace(**request.model_dump()))
    finally:
        STATE.waiting -= 1
    STATE.log.info(f"[{request.profile}] {request.mode} K={request.samples} steps={request.steps} pipeline={request.pipeline} "
                   f"-> {len(result['components'])} components in {result['seconds']} s")
    return result


def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    return parser.parse_args()


def main() -> None:
    """Load the model and serve."""
    args: argparse.Namespace = parse_args()
    config: SimpleNamespace = load_run_config(CONFIG_FILE, args.mode)
    log: logging.Logger = setup_logger("serve", resolve_path(config.LOG_FILE_PATH), make_run_id("serve"), config.IS_DEBUG)
    STATE.log = log
    STATE.config = config
    STATE.profiles = discover_profiles(config, log)
    # The default (ResPlan) model loads now so a bad checkpoint fails at start-up; the others load on first use.
    engine_for(DEFAULT_PROFILE)
    log.info(f"Profiles: {', '.join(STATE.profiles)}; serving on {config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info" if config.IS_DEBUG else "warning",
                proxy_headers=True, forwarded_allow_ips="127.0.0.1")


if __name__ == "__main__":
    main()
