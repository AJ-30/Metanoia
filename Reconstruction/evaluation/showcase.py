"""Experiment sweeps and per-case showcase material for the results write-up.

Two sub-commands, both on the frozen test cases and one checkpoint:

* ``sweep``: scores the first N cases under several sampler settings
  (DDIM steps, K samples fused by vote, post-processing pipeline) and writes
  ``sweep.json``.
* ``cases``: for chosen case indices, draws the input, the x0 estimate at
  each denoising snapshot, the composed plan and the ground truth as PNGs and
  writes their metrics to ``cases.json``.

Usage (from the repository root):
    python -m Reconstruction.evaluation.showcase sweep --run-name run2-a40 --checkpoint 150000 --max-cases 200
    python -m Reconstruction.evaluation.showcase cases --run-name run2-a40 --checkpoint 150000 --indices 3 17 42
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

from Reconstruction.common.config_loader import REPO_ROOT, load_params, to_namespace
from Reconstruction.data_prep.case_source import CaseSource
from Reconstruction.data_prep.missing import Case
from Reconstruction.data_prep.render import render_semantic
from Reconstruction.evaluation.case_metrics import prepare_case, score_samples
from Reconstruction.evaluation.examples import prediction_panels
from Reconstruction.model.checkpoint import resolve_checkpoints
from Reconstruction.model.denoising import scheduled_snapshots
from Reconstruction.model.diffusion import GaussianDiffusion
from Reconstruction.model.sampling import sample_missing
from Reconstruction.model.unet import ConditionalUNet
from Reconstruction.postprocess.run import postprocess_case

PACKAGE_DIR: Path = REPO_ROOT / "Reconstruction"
SCORE_SCALE: int = 255
SNAPSHOT_TIMESTEPS: list[int] = [999, 800, 600, 400, 200, 0]
OUT_DIR: Path = PACKAGE_DIR / "outputs" / "experiments"
# (name, DDIM steps, K samples, post-processing pipeline)
SWEEP: list[tuple[str, int, int, str]] = [
    ("ddim10_k1", 10, 1, "none"),
    ("ddim25_k1", 25, 1, "none"),
    ("ddim50_k1", 50, 1, "none"),
    ("ddim50_k1_full", 50, 1, "full"),
    ("ddim50_k4_vote", 50, 4, "none"),
    ("ddim50_k4_vote_full", 50, 4, "full"),
]


def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sweep", "cases"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--dataset", default="resplan")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-cases", type=int, default=200, help="sweep: first N cases")
    parser.add_argument("--indices", type=int, nargs="*", default=[], help="cases: case indices to draw")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class Runner:
    """Loaded model, taxonomy and parameter files for repeated sampling."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Load the checkpoint (EMA weights) and the case source.

        Args:
            args (argparse.Namespace): Command-line arguments.
        """
        self.device = torch.device("cuda")
        processed: Path = REPO_ROOT / "datasets" / "processed" / args.dataset
        self.source = CaseSource(processed, args.split, seed=0)
        self.schema = self.source.schema
        self.eval_params: SimpleNamespace = load_params(PACKAGE_DIR / "evaluation" / "params.toml")
        self.post_params: SimpleNamespace = load_params(PACKAGE_DIR / "postprocess" / "params.toml")
        self.pipelines: dict[str, list[str]] = {k: list(v) for k, v in vars(self.post_params.pipelines).items()}
        self.threshold: float = self.eval_params.decode.THRESHOLD
        path: Path = resolve_checkpoints(PACKAGE_DIR / "runs" / args.run_name / "checkpoints", [str(args.checkpoint)])[0]
        state: dict[str, Any] = torch.load(path, map_location=self.device, weights_only=True)
        self.hyper = to_namespace(state["hyperparameters"])
        self.step: int = int(state["step"])
        self.model = ConditionalUNet(self.schema.num_channels, self.source.params.canvas.SIZE, self.hyper.model).to(self.device)
        self.model.load_state_dict(state["ema"])
        self.model.eval()
        self.diffusion = GaussianDiffusion(self.hyper.diffusion.TIMESTEPS, self.hyper.diffusion.MAX_BETA)
        self.dtype = torch.bfloat16 if self.hyper.train.PRECISION == "bf16" else None

    def sample(self, case: Case, samples: int, steps: int, seed: int,
               snapshots: list[int] | None = None) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Draw K samples of the missing part, optionally recording x0 snapshots of sample 0.

        Args:
            case (Case): Case.
            samples (int): K.
            steps (int): DDIM steps.
            seed (int): Noise seed.
            snapshots (list[int] | None): Timesteps to record.

        Returns:
            tuple[np.ndarray, dict[int, np.ndarray]]: Scores (K, C, H, W) in [0, 1] and {t: (C, H, W)} snapshots.
        """
        batch = torch.from_numpy(np.repeat(case.incomplete[None], samples, axis=0)).to(self.device).float()
        wanted: list[int] = scheduled_snapshots(snapshots, steps, self.hyper.diffusion.TIMESTEPS) if snapshots else []
        captured: dict[int, np.ndarray] = {}

        def record(t: int, x0: torch.Tensor) -> None:
            if t in wanted:
                captured[t] = ((x0[0].float() + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            scores = sample_missing(self.model, self.diffusion, batch, steps, self.hyper.diffusion.CLIP_X0,
                                    generator, self.dtype, on_step=record if wanted else None)
        return scores.cpu().numpy(), captured

    def fuse(self, scores: np.ndarray, case: Case, pipeline: str) -> np.ndarray:
        """Vote and post-process K samples into one prediction.

        Args:
            scores (np.ndarray): (K, C, H, W) scores.
            case (Case): Case.
            pipeline (str): Post-processing pipeline name.

        Returns:
            np.ndarray: (C, H, W) float prediction in [0, 1].
        """
        binary: np.ndarray = (scores > self.threshold).astype(np.uint8) * SCORE_SCALE
        fused: np.ndarray = postprocess_case(binary, case.incomplete, self.schema, self.post_params,
                                             self.pipelines[pipeline], self.threshold)
        return fused[0].astype(np.float32) / SCORE_SCALE


def plain(value: Any) -> Any:
    """JSON-safe value (NaN -> None, numpy -> Python).

    Args:
        value (Any): Metric value.

    Returns:
        Any: Plain value.
    """
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def run_sweep(runner: Runner, args: argparse.Namespace) -> None:
    """Score the first N cases under every sampler setting in ``SWEEP``.

    Args:
        runner (Runner): Loaded model.
        args (argparse.Namespace): Command-line arguments.
    """
    count: int = min(args.max_cases, len(runner.source))
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name, *_ in SWEEP}
    seconds: dict[str, float] = {name: 0.0 for name, *_ in SWEEP}
    for index in range(count):
        case: Case = runner.source.case(index)
        context = prepare_case(case, runner.schema, runner.eval_params)
        cache: dict[tuple[int, int], np.ndarray] = {}
        for name, steps, samples, pipeline in SWEEP:
            start: float = time.time()
            if (steps, samples) not in cache:
                cache[(steps, samples)] = runner.sample(case, samples, steps, args.seed + index)[0]
            prediction: np.ndarray = runner.fuse(cache[(steps, samples)], case, pipeline)
            metrics = score_samples(context, prediction[None], runner.schema, runner.eval_params)
            rows[name].append({"case_id": case.info["case_id"], "comp_type": case.info["comp_type"],
                               "mode": case.info["mode"], **plain(metrics)})
            seconds[name] += time.time() - start
        if (index + 1) % 20 == 0:
            print(f"{index + 1}/{count} cases", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"run": args.run_name, "step": runner.step, "cases": count,
           "settings": {n: {"steps": s, "k": k, "pipeline": p} for n, s, k, p in SWEEP},
           "seconds": seconds, "rows": rows}
    (OUT_DIR / "sweep.json").write_text(json.dumps(out))


def png(image: np.ndarray, scale: int, path: Path) -> None:
    """Save a float (H, W, 3) image as an upscaled PNG.

    Args:
        image (np.ndarray): Image in [0, 1].
        scale (int): Integer nearest-neighbour upscale.
        path (Path): Output file.
    """
    pixels = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    pixels.resize((pixels.width * scale, pixels.height * scale), Image.NEAREST).save(path)


def run_cases(runner: Runner, args: argparse.Namespace) -> None:
    """Draw input, denoising snapshots, composed plan and truth for chosen cases.

    Args:
        runner (Runner): Loaded model.
        args (argparse.Namespace): Command-line arguments.
    """
    folder: Path = OUT_DIR / "cases"
    folder.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []
    for index in args.indices:
        case: Case = runner.source.case(index)
        scores, snaps = runner.sample(case, 1, 50, args.seed + index, SNAPSHOT_TIMESTEPS)
        prediction: np.ndarray = runner.fuse(scores, case, "none")
        context = prepare_case(case, runner.schema, runner.eval_params)
        metrics = plain(score_samples(context, prediction[None], runner.schema, runner.eval_params))
        panels = prediction_panels(case, prediction, runner.schema, runner.threshold)
        cid: str = case.info["case_id"]
        png(panels[0], 2, folder / f"{cid}_input.png")
        png(panels[1], 2, folder / f"{cid}_added.png")
        png(panels[2], 2, folder / f"{cid}_composed.png")
        png(panels[3], 2, folder / f"{cid}_truth.png")
        png(render_semantic(case.full, runner.schema), 2, folder / f"{cid}_full.png")
        steps_info: list[dict[str, Any]] = []
        for t in sorted(snaps, reverse=True):
            png(prediction_panels(case, snaps[t], runner.schema, runner.threshold)[1], 2, folder / f"{cid}_t{t}.png")
            steps_info.append({"t": t, "file": f"{cid}_t{t}.png"})
        report.append({"index": index, "case_id": cid, "info": plain({k: case.info.get(k) for k in (
            "comp_type", "primary_cls", "mode", "margin_px", "target_area_px", "size_bin")}), "metrics": metrics,
            "snapshots": steps_info})
    (OUT_DIR / "cases.json").write_text(json.dumps(report))


def main() -> None:
    """Entry point."""
    args = parse_args()
    runner = Runner(args)
    run_sweep(runner, args) if args.command == "sweep" else run_cases(runner, args)


if __name__ == "__main__":
    main()
