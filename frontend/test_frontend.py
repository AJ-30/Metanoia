"""Unit tests for the frontend's pure logic.

Arrange-Act-Assert throughout, with no context, detector or GPU: only the parts of
the stage that can be judged without a model. Covers the failure modes that matter
for an interface, because a wrong message or a crashed upload page is what a user
sees instead of the result:

1. An upload that is not a class-id mask (RGB photo, palette photo, garbage
   bytes) must produce a message about the upload, never a server path or a 500.
2. An id outside the layout's range must be refused rather than clamped onto a
   neighbouring class.
3. A missing furniture upload must become a blank layout, not a crash.
4. The draw count must be clamped, so one request cannot launch a 1000-step chain
   a thousand times.
5. Old request folders must be pruned so the outputs directory cannot grow
   without bound.
6. The staged masks must be re-saved under the exact filenames the frozen
   encoders read.

Run with: ``./venv/bin/python frontend/test_frontend.py``
(or ``pytest frontend/test_frontend.py`` when pytest is installed).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Repo root so ``frontend``/``mask_diffusion`` import when run as a script.
_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from frontend.pipeline import (  # noqa: E402
    _clamp_draws,
    build_blank_mask,
    save_upload_as,
    validate_and_load_mask,
)
from frontend.server import _prune_old_runs, UPLOAD_MASKS  # noqa: E402

# Class counts of the two layouts the encoders were trained with: 18 floorplan
# classes and 11 furniture classes, background included.
FLOORPLAN_CLASSES: int = 18
FURNITURE_CLASSES: int = 11
IMAGE_SIZE: int = 32


def _write_png(path: Path, array: np.ndarray, mode: str) -> None:
    """Write one test image.

    Args:
        path (Path): Destination PNG.
        array (np.ndarray): Pixel data.
        mode (str): PIL image mode to save in.
    """
    Image.fromarray(array, mode=mode).save(str(path))


def test_validate_and_load_mask_resizes_and_keeps_ids() -> None:
    """Any input size becomes the encoder's square, ids untouched."""
    mask = np.zeros((16, 48), dtype=np.uint8)
    mask[4:8, 4:8] = 3
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.png"
        _write_png(path, mask, "L")
        ids = validate_and_load_mask(path, FLOORPLAN_CLASSES, IMAGE_SIZE, "floorplan")
    assert ids.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert ids.dtype == torch.int64
    # The inner square survives the nearest-neighbour resize.
    assert int(ids.max()) == 3


def test_validate_and_load_mask_accepts_palette_mode() -> None:
    """A palette-encoded class-id mask is as valid as a mode-L one."""
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:4, 0:4] = 5
    # The identity grey ramp keeps palette index == class id, which is what a
    # P-mode class-id mask means: converting to L gives the ids back unchanged.
    palette = [channel for index in range(256) for channel in (index, index, index)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.png"
        image = Image.new("P", mask.shape[::-1])
        image.putpalette(palette)
        image.putdata([int(value) for value in mask.reshape(-1)])
        image.save(str(path))
        ids = validate_and_load_mask(path, FLOORPLAN_CLASSES, IMAGE_SIZE, "floorplan")
    assert int(ids.max()) == 5


def test_validate_and_load_mask_rejects_rgb_photo() -> None:
    """A colour photo must be refused with a message about the mask, not a path."""
    photo = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype("uint8")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "photo.png"
        _write_png(path, photo, "RGB")
        try:
            validate_and_load_mask(path, FLOORPLAN_CLASSES, IMAGE_SIZE, "floorplan")
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("an RGB upload was accepted")
    assert "class-id mask" in message
    assert "floorplan" in message
    assert str(path) not in message  # never quote a server-side path


def test_validate_and_load_mask_rejects_unreadable_file() -> None:
    """Garbage bytes must become a message about the upload."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "garbage.png"
        path.write_bytes(b"definitely not a png")
        try:
            validate_and_load_mask(path, FLOORPLAN_CLASSES, IMAGE_SIZE, "furniture")
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("garbage bytes were accepted")
    assert "not a readable PNG" in message
    assert "furniture" in message


def test_validate_and_load_mask_rejects_out_of_range_id() -> None:
    """An id above the layout's range must be refused, never clamped."""
    mask = np.full((16, 16), 250, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.png"
        _write_png(path, mask, "L")
        try:
            validate_and_load_mask(path, FLOORPLAN_CLASSES, IMAGE_SIZE, "floorplan")
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("an out-of-range class id was accepted")
    assert "250" in message and "18" in message


def test_build_blank_mask_is_all_background() -> None:
    """The empty layout the encoders read is every pixel background."""
    blank = build_blank_mask((IMAGE_SIZE, IMAGE_SIZE))
    assert blank.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert blank.dtype == torch.int64
    assert int(blank.max()) == 0
    assert int(blank.sum()) == 0


def test_save_upload_as_writes_the_expected_filename() -> None:
    """The staged mask is written where the encoders expect to read it."""
    ids = build_blank_mask((8, 8))
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / "sample" / "floorplan_segmentation.png"
        save_upload_as(ids, destination)
        assert destination.is_file()
        with Image.open(destination) as image:
            assert image.mode == "L"
            assert np.array(image).max() == 0


def test_clamp_draws_limits_one_request_cost() -> None:
    """The draw count is clamped into the allowed band on both ends."""
    assert _clamp_draws(0, 4) == 1
    assert _clamp_draws(-7, 4) == 1
    assert _clamp_draws(2, 4) == 2
    assert _clamp_draws(99, 4) == 4
    assert _clamp_draws(4, 1) == 1


def test_prune_old_runs_keeps_the_newest_folders() -> None:
    """Old request folders are removed so the outputs directory cannot grow."""
    import os
    import time as time_module

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "outputs"
        for index in range(4):
            folder = root / f"upload_{index}"
            folder.mkdir(parents=True)
            # Age the folders in creation order, oldest first.
            stamp: float = time_module.time() - (4 - index) * 60
            os.utime(folder, (stamp, stamp))
        _prune_old_runs(root, 2)
        remaining = sorted(path.name for path in root.iterdir())
        assert remaining == ["upload_2", "upload_3"]
        # Negative keeps every folder, and a missing directory is not an error.
        _prune_old_runs(root, -1)
        assert len(list(root.iterdir())) == 2
        _prune_old_runs(root / "absent", 2)


def test_upload_mask_roles_name_each_field_once() -> None:
    """Each form field maps to exactly one staged filename."""
    assert set(UPLOAD_MASKS) == {"floorplan", "furniture"}
    assert len(set(UPLOAD_MASKS.values())) == len(UPLOAD_MASKS)


def _run_all_tests() -> int:
    """Run every ``test_*`` function in this module, reporting failures.

    Returns:
        int: Number of failed tests, ``0`` when all pass.
    """
    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]
    failures: list[str] = []
    for name, function in tests:
        try:
            function()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}")
        else:
            print(f"pass {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed.")
    return len(failures)


if __name__ == "__main__":
    raise SystemExit(1 if _run_all_tests() else 0)
