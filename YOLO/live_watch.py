"""Read-only sidecar publishing live training snapshots to the dashboard.

While ``./train_YOLO.sh`` runs in another process, this daemon polls the live
dataset and the Ultralytics run directory (both read-only) and publishes one
side-by-side PNG per snapshot stem per epoch into ``YOLO/outputs/``:

    live_epoch_<E>_<stem>.png  =  live overlay  |  overlay + GT (green) + pred (red) boxes

The left half is the current diffusion-draw overlay for that stem (fresh
pixels every epoch); the right half adds the ground-truth boxes plus the
boxes predicted by the latest ``best.pt``. The dashboard's Live panel picks
these up with no training-process involvement.

Each tick also appends new Ultralytics ``results.csv`` rows to
``<run>_live.csv`` in the telemetry directory: the running trainer only
mirrors at end-of-training, which would leave the dashboard empty for hours.
A separate ``_live`` file (never the trainer's own CSV) keeps its eventual
sweep duplicate-free.

Safety contract (the trainer must never be disturbed):

  * Only READS ``LIVE_DATASET_ROOT`` (overlays, labels, manifest) and the
    Ultralytics weights (copied to a temp file before loading, so a
    mid-write ``best.pt`` can never corrupt a load — a failed load just
    publishes overlay-only snapshots this tick).
  * Only WRITES ``YOLO/outputs/`` (served by the dashboard server).
  * Never imports the training loop; a tick failure is logged and the daemon
    keeps polling. A missing dataset root fails fast (misconfiguration).

Snapshot files are the publish state: a tick skips stems whose file already
exists, so restarts resume without duplicates. Epochs come from
``live_manifest.csv`` (``seed = BASE_SEED + epoch`` rows appended per regen).

Usage:
    python YOLO/live_watch.py --mode debug            # poll forever
    python YOLO/live_watch.py --mode debug --once     # single tick (smoke test)
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

# Project root so ``helpers/`` and ``YOLO.runner`` import from any cwd.
SCRIPT_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from YOLO.runner import (
    RESULTS_CSV_MAP,
    TELEMETRY_FIELDS,
    draw_boxes_on_image,
    read_results_csv,
    resolve_train_device,
)

# Manifest columns appended by LiveDiffusionPipeline.regenerate_train_epoch.
MANIFEST_EPOCH_COLUMN: str = "epoch"

# Caption drawn on the right half while no weights exist yet.
WAITING_FOR_WEIGHTS_TEXT: str = "waiting for best.pt"


def read_manifest_max_epoch(manifest_path: Path) -> int | None:
    """Read the highest published epoch from the live manifest.

    The trainer appends rows per regen and may hold the file open; a torn
    trailing line is skipped rather than failing the tick.

    Args:
        manifest_path (Path): Path of ``live_manifest.csv``.

    Returns:
        int | None: Highest epoch number seen, or None when the manifest is
            missing/empty (training has not regenerated yet).
    """
    if not manifest_path.is_file():
        return None
    max_epoch: int | None = None
    try:
        with open(manifest_path, newline="", encoding="utf-8") as manifest_file:
            for row in csv.DictReader(manifest_file):
                try:
                    epoch: int = int(row[MANIFEST_EPOCH_COLUMN])
                except (KeyError, TypeError, ValueError):
                    continue  # torn line mid-append; next tick sees it whole
                if max_epoch is None or epoch > max_epoch:
                    max_epoch = epoch
    except OSError:
        return None
    return max_epoch


def list_train_stems(train_img_dir: Path) -> list[str]:
    """List sorted train overlay stems currently on disk.

    Args:
        train_img_dir (Path): Live ``images/train`` directory.

    Returns:
        list[str]: Sorted PNG stems (stable across regens of one run).
    """
    if not train_img_dir.is_dir():
        return []
    return sorted(path.stem for path in train_img_dir.glob("*.png"))


def snapshot_path(out_dir: Path, epoch: int, stem: str) -> Path:
    """Return the snapshot file for one stem/epoch (its existence = published).

    The ``live_epoch_<E>_`` prefix is what the dashboard's Live panel matches,
    and the embedded epoch is what its dropdown labels.

    Args:
        out_dir (Path): Dashboard outputs directory.
        epoch (int): Manifest epoch number.
        stem (str): Train image stem.

    Returns:
        Path: Snapshot destination path.
    """
    return out_dir / f"live_epoch_{epoch}_{stem}.png"


def read_gt_lines(label_path: Path) -> list[str]:
    """Read GT YOLO label lines, tolerating a mid-regen partial file.

    Args:
        label_path (Path): Path of the stem's ``.txt`` label file.

    Returns:
        list[str]: Raw GT lines (may be empty when the file is mid-write).
    """
    try:
        return label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


# Telemetry columns of the watcher's live CSV (same order as the trainer's,
# plus the writer's timestamp/runtime prefix) so the dashboard plots both
# identically. Written to "<run>_live.csv": a separate file from the trainer's
# "<run>.csv" so its end-of-run sweep can never duplicate these rows.
LIVE_TELEMETRY_SUFFIX: str = "_live.csv"
LIVE_TELEMETRY_HEADER: list[str] = ["timestamp", "runtime"] + TELEMETRY_FIELDS


def find_latest_results(outputs_dir: Path) -> tuple[str, Path] | None:
    """Locate the newest Ultralytics results.csv and its training run id.

    Args:
        outputs_dir (Path): Dashboard outputs directory holding
            ``ultralytics_<run>/train/results.csv`` per training run.

    Returns:
        tuple[str, Path] | None: (run id, results.csv path) for the newest
            run, or None when no run has written one yet.
    """
    candidates: list[Path] = sorted(
        outputs_dir.glob("ultralytics_*/train/results.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return None
    latest: Path = candidates[-1]
    return latest.parent.parent.name.removeprefix("ultralytics_"), latest


def read_live_max_epoch(live_csv: Path) -> int:
    """Read the highest epoch already mirrored into the live CSV.

    Args:
        live_csv (Path): Watcher's ``<run>_live.csv`` path.

    Returns:
        int: Highest mirrored epoch, or 0 when the file is missing/empty.
    """
    if not live_csv.is_file():
        return 0
    max_epoch: int = 0
    try:
        with open(live_csv, newline="", encoding="utf-8") as live_file:
            for row in csv.DictReader(live_file):
                try:
                    epoch: int = int(float(row["epoch"]))
                except (KeyError, TypeError, ValueError):
                    continue
                max_epoch = max(max_epoch, epoch)
    except OSError:
        return 0
    return max_epoch


def run_start_time(telemetry_path: Path) -> float:
    """Recover a training run's start time from its (trainer-owned) CSV file.

    Args:
        telemetry_path (Path): Trainer's ``<run>.csv`` path, created at init.

    Returns:
        float: Epoch timestamp of the run start (birthtime, mtime fallback).
    """
    stat_result: os.stat_result = telemetry_path.stat()
    return float(getattr(stat_result, "st_birthtime", stat_result.st_mtime))


def find_latest_weights(outputs_dir: Path) -> Path | None:
    """Locate the newest Ultralytics ``best.pt`` under the outputs directory.

    Args:
        outputs_dir (Path): Dashboard outputs directory holding
            ``ultralytics_<run>/train/weights/best.pt`` per training run.

    Returns:
        Path | None: Newest ``best.pt`` by mtime, or None when no training
            run has checkpointed yet.
    """
    candidates: list[Path] = sorted(
        outputs_dir.glob("ultralytics_*/train/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_predictor_copy(
    weights_path: Path,
    log: SimpleNamespace,
) -> object | None:
    """Load a YOLO predictor from a temp COPY of the weights (soft failure).

    Copying first means a ``best.pt`` rewritten mid-copy by the trainer can
    never corrupt the load — the worst case is a failed tick that retries on
    the next poll. The ``ultralytics`` import stays lazy so this module
    imports without the YOLO dependency.

    Args:
        weights_path (Path): Live ``best.pt`` written by the trainer.
        log (SimpleNamespace): Configured logger for soft-failure notes.

    Returns:
        object | None: Loaded Ultralytics YOLO model, or None when weights
            are missing/unreadable (caller publishes overlay-only snapshots).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log.warning("ultralytics not installed; publishing overlay-only snapshots.")
        return None
    tmp_copy: str = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_copy = tmp.name
        shutil.copy2(str(weights_path), tmp_copy)
        model: object = YOLO(tmp_copy)
        return model
    except (OSError, RuntimeError, ValueError) as exc:
        # Mid-write weights or a corrupt copy: overlay-only this tick.
        log.warning(f"Could not load weights copy ({exc}); overlay-only snapshots.")
        return None
    finally:
        if tmp_copy:
            try:
                Path(tmp_copy).unlink(missing_ok=True)
            except OSError:
                pass


def predict_boxes(
    model: object,
    img_path: Path,
    conf: float,
    device: str,
    max_boxes: int,
) -> list[tuple[float, float, float, float, int, float]]:
    """Run inference on one overlay image, returning pixel-space boxes.

    Args:
        model (object): Loaded Ultralytics YOLO model (``predict`` API).
        img_path (Path): Overlay PNG to predict on.
        conf (float): Confidence threshold for predictions.
        device (str): Inference device string ("mps", "cuda" or "cpu").
        max_boxes (int): Cap on returned boxes, highest confidence first, so an
            undertrained model cannot bury the snapshot under hundreds of boxes.

    Returns:
        list[tuple[float, float, float, float, int, float]]: Predicted
            (x1, y1, x2, y2, cls, conf) boxes in absolute pixels, at most
            ``max_boxes``, sorted by descending confidence.
    """
    results = model.predict(source=str(img_path), conf=conf, device=device, verbose=False)  # type: ignore[attr-defined]
    pred_boxes: list[tuple[float, float, float, float, int, float]] = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for (x1, y1, x2, y2), cls_id, conf_val in zip(
            boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
        ):
            pred_boxes.append(
                (float(x1), float(y1), float(x2), float(y2), int(cls_id), float(conf_val))
            )
    # Highest confidence first, then cap: an early model emits its whole NMS
    # output and the snapshot stays readable.
    pred_boxes.sort(key=lambda box: box[5], reverse=True)
    return pred_boxes[:max(1, max_boxes)]


def caption_waiting(img: Image.Image) -> Image.Image:
    """Stamp a waiting caption on an overlay copy (no weights yet).

    Args:
        img (Image.Image): RGB overlay image (copied before drawing).

    Returns:
        Image.Image: Annotated copy of the input image.
    """
    img = img.copy()
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(img)
    draw.text((8, 8), WAITING_FOR_WEIGHTS_TEXT, fill=(255, 255, 0))
    return img


def compose_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    """Concatenate two RGB images horizontally into one snapshot.

    Args:
        left (Image.Image): Left half (live overlay).
        right (Image.Image): Right half (overlay + boxes).

    Returns:
        Image.Image: Combined RGB image of double width.
    """
    combined: Image.Image = Image.new("RGB", (left.width + right.width, left.height))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width, 0))
    return combined


class LiveWatcher:
    """Polling sidecar publishing live snapshots without touching training."""

    def __init__(self, mode: str) -> None:
        """Load config/hyperparams and set up logging (read-only use).

        Args:
            mode (str): Run mode ("debug" or "prod") selecting config overrides.
        """
        self.config: SimpleNamespace = load_config(mode, str(SCRIPT_DIR / "config.toml"))
        self.hyperparams: SimpleNamespace = load_hyperparams(
            str(SCRIPT_DIR / "hyperparameters.toml")
        )
        self.log = setup_logger(self.config)
        self.live_root: Path = Path(self.config.LIVE_DATASET_ROOT)
        self.train_img_dir: Path = self.live_root / "images" / "train"
        self.train_lbl_dir: Path = self.live_root / "labels" / "train"
        self.manifest_path: Path = Path(self.config.LIVE_MANIFEST_PATH)
        self.out_dir: Path = Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.num_stems: int = max(1, int(self.hyperparams.LIVE_WATCH_STEMS))
        self.device_str: str = resolve_train_device(self.config.DEVICE, self.log)
        if not self.live_root.is_dir():
            raise FileNotFoundError(
                f"Live dataset root missing: {self.live_root}. "
                "Start ./train_YOLO.sh first; the watcher only reads it."
            )
        self.log.info(
            f"Live watcher watching {self.live_root} "
            f"(stems={self.num_stems}, device={self.device_str})"
        )

    def snapshot_stems(self) -> list[str]:
        """Return the fixed snapshot stems for this run (sorted, capped).

        The same stems are published every epoch so the dashboard shows one
        layout's pixels changing plus its boxes improving.

        Returns:
            list[str]: Snapshot stems, possibly empty before epoch 0 lands.
        """
        return list_train_stems(self.train_img_dir)[: self.num_stems]

    def publish_epoch(
        self,
        epoch: int,
        model: object | None,
    ) -> int:
        """Publish missing snapshots for one manifest epoch.

        Files that already exist are skipped, so restarts never duplicate.

        Args:
            epoch (int): Manifest epoch number to publish.
            model (object | None): Loaded predictor, or None for
                overlay-only snapshots (no weights yet / unloadable copy).

        Returns:
            int: Number of snapshots newly published.
        """
        class_names: list[str] = list(self.hyperparams.CLASS_NAMES)
        # Watcher-only confidence and box cap (training itself is unaffected).
        # Both exist to keep the snapshot readable while a model converges.
        conf: float = float(self.hyperparams.LIVE_WATCH_CONF)
        max_boxes: int = int(self.hyperparams.LIVE_WATCH_MAX_BOXES)
        published: int = 0
        for stem in self.snapshot_stems():
            dest: Path = snapshot_path(self.out_dir, epoch, stem)
            if dest.is_file():
                continue
            img_path: Path = self.train_img_dir / f"{stem}.png"
            if not img_path.is_file():
                continue
            overlay: Image.Image = Image.open(img_path).convert("RGB")
            gt_lines: list[str] = read_gt_lines(self.train_lbl_dir / f"{stem}.txt")
            if model is None:
                right: Image.Image = caption_waiting(overlay)
            else:
                pred_boxes = predict_boxes(model, img_path, conf, self.device_str, max_boxes)
                right = draw_boxes_on_image(overlay, gt_lines, pred_boxes, class_names)
            compose_side_by_side(overlay, right).save(str(dest))
            published += 1
        if published:
            self.log.info(f"Published {published} live snapshots for epoch {epoch}.")
        return published

    def mirror_live_telemetry(self) -> int:
        """Append new Ultralytics results.csv rows to the "<run>_live.csv".

        The running trainer only mirrors at end-of-training, which leaves the
        dashboard empty for hours; this sidecar mirror keeps curves live.
        Rows already present (by epoch) are skipped, so restarts never
        duplicate. A torn trailing line in the live-written results.csv is
        skipped and retried next tick.

        Returns:
            int: Rows appended this tick.
        """
        found: tuple[str, Path] | None = find_latest_results(self.out_dir)
        if found is None:
            return 0
        run_id, results_path = found
        base_csv: Path = Path(self.config.TELEMETRY_LOG_DIRECTORY) / f"{run_id}.csv"
        if not base_csv.is_file():
            return 0
        live_csv: Path = Path(self.config.TELEMETRY_LOG_DIRECTORY) / (
            f"{run_id}{LIVE_TELEMETRY_SUFFIX}"
        )
        start_time: float = run_start_time(base_csv)
        max_epoch: int = read_live_max_epoch(live_csv)
        rows: list[dict[str, str]] = read_results_csv(str(results_path))
        fresh: list[dict[str, float]] = []
        for row in rows:
            metrics: dict[str, float] = {}
            try:
                for ours, candidates in RESULTS_CSV_MAP:
                    for candidate in candidates:
                        if candidate in row and row[candidate] not in ("", None):
                            metrics[ours] = float(row[candidate])
                            break
                    else:
                        raise KeyError(
                            f"none of {candidates} present in results.csv row"
                        )
            except (KeyError, ValueError) as exc:
                self.log.warning(f"Skipping malformed results.csv row: {exc}")
                continue
            if int(metrics["epoch"]) <= max_epoch:
                continue
            fresh.append(metrics)
            max_epoch = int(metrics["epoch"])
        if not fresh:
            return 0
        write_header: bool = not live_csv.is_file()
        with open(live_csv, "a", newline="", encoding="utf-8") as live_file:
            writer = csv.DictWriter(live_file, fieldnames=LIVE_TELEMETRY_HEADER)
            if write_header:
                writer.writeheader()
            now: float = time.time()
            for metrics in fresh:
                writer.writerow({
                    "timestamp": now,
                    "runtime": now - start_time,
                    **{key: metrics[key] for key in TELEMETRY_FIELDS},
                })
        self.log.info(f"Mirrored {len(fresh)} telemetry rows to {live_csv.name}.")
        return len(fresh)

    def tick(self) -> int:
        """Run one poll tick: mirror telemetry, publish current-epoch snapshots.

        Only the CURRENT max epoch is published — never backfilled. Past
        epochs' pixels are already overwritten on disk, so a snapshot labeled
        ``live_epoch_<E>`` must always show that epoch's actual pixels.
        Existing files are skipped, so restarts resume without duplicates.

        Loads the newest ``best.pt`` copy once per tick (cheap for YOLO26n)
        so boxes track improving weights; falls back to overlay-only when
        weights are missing or unloadable.

        Returns:
            int: Snapshots published this tick.
        """
        self.mirror_live_telemetry()
        max_epoch: int | None = read_manifest_max_epoch(self.manifest_path)
        if max_epoch is None:
            self.log.info("No manifest epochs yet; waiting for the first regen.")
            return 0
        weights: Path | None = find_latest_weights(self.out_dir)
        model: object | None = (
            load_predictor_copy(weights, self.log) if weights is not None else None
        )
        if weights is None:
            self.log.info("No best.pt yet; publishing overlay-only snapshots.")
        return self.publish_epoch(max_epoch, model)

    def run(self, once: bool, interval_sec: int) -> None:
        """Poll forever (or once) publishing live snapshots.

        Args:
            once (bool): Single tick then return (smoke test).
            interval_sec (int): Seconds between ticks in daemon mode.
        """
        while True:
            try:
                self.tick()
            except (OSError, RuntimeError, ValueError) as exc:
                # Daemon policy: log transient tick failures and keep polling.
                # Fail-fast paths (missing dataset root) raise in __init__.
                self.log.warning(f"Live tick failed ({exc}); retrying next poll.")
            if once:
                return
            time.sleep(max(1, interval_sec))


def main() -> None:
    """Parse CLI args and launch the live watcher."""
    parser = argparse.ArgumentParser(description="Publish live YOLO training snapshots.")
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    parser.add_argument("--once", action="store_true", help="Single tick then exit.")
    args: argparse.Namespace = parser.parse_args()
    hyperparams: SimpleNamespace = load_hyperparams(
        str(SCRIPT_DIR / "hyperparameters.toml")
    )
    LiveWatcher(mode=args.mode).run(
        once=args.once, interval_sec=int(hyperparams.LIVE_WATCH_INTERVAL_SEC)
    )


if __name__ == "__main__":
    main()
